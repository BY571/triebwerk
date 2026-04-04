/**
 * CUDA kernels for Jetson LLM inference engine.
 *
 * Optimized for Jetson Orin sm_87 (Ampere, 32 tensor cores).
 * Single-batch decode (batch_size=1), which means GEMV not GEMM.
 *
 * Key optimization: fuse operations to reduce kernel launch overhead.
 * The profiler showed 1400 kernel launches per token with bitsandbytes;
 * we target ~10-15 kernel launches per token.
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>
#include "model.h"

// Shared-memory reduction arrays are sized [4] = max 4 warps (128 threads).
// Kernels that use these must be launched with <= 128 threads per block.
// Compile-time guard: if any kernel is launched with more threads, this
// static_assert will fire. The value 4 matches THREADS_PER_BLOCK/32 max.
#define MAX_WARPS_PER_BLOCK 4

// ============================================================================
// NF4 Dequantization Lookup Table (standard bitsandbytes NF4 values)
// ============================================================================

__constant__ float NF4_LOOKUP[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

// ============================================================================
// Kernel 1: Fused NF4 Dequant + GEMV (Matrix-Vector multiply)
// ============================================================================
// Replaces 196 separate bitsandbytes kernel calls per token.
// For decode: input is (1, in_dim), weight is (out_dim, in_dim) in NF4.
// Output: (1, out_dim) in fp16.

// ============================================================================
// Kernel 2: RMSNorm
// ============================================================================
// output[i] = (input[i] / rms) * weight[i]
// where rms = sqrt(mean(input^2) + eps)

__global__ void rms_norm_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ output,
    int dim,
    float eps
) {
    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[i]);
        sum_sq += val * val;
    }

    // Block reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane_id == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / dim + eps);

    // Normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[i]) * rms;
        output[i] = __float2half(val * __half2float(weight[i]));
    }
}

// Fused: copy input → residual AND RMSNorm(input) → norm_out in one kernel.
// Replaces cudaMemcpyAsync + launch_rms_norm (2 ops → 1 kernel).
__global__ void copy_rms_norm_kernel(
    const half* __restrict__ input,    // read once: hidden state
    const half* __restrict__ weight,   // norm weights
    half* __restrict__ residual,       // output: copy of input (for residual add later)
    half* __restrict__ norm_out,       // output: RMSNorm(input)
    int dim, float eps
) {
    // Pass 1: copy input → residual AND compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        half val_h = input[i];
        residual[i] = val_h;  // copy
        float val = __half2float(val_h);
        sum_sq += val * val;
    }

    // Reduction (same as rms_norm_kernel)
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane_id == 0) shared[0] = sum_sq;
    }
    __syncthreads();
    float rms = rsqrtf(shared[0] / dim + eps);

    // Pass 2: normalize and scale
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(input[i]) * rms;
        norm_out[i] = __float2half(val * __half2float(weight[i]));
    }
}

// ============================================================================
// Kernel 2b: Fused QKNorm (RMSNorm applied per-head to Q and K)
// ============================================================================
// Replaces 24 separate rms_norm_kernel launches per layer with ONE kernel.
// Each block handles one head (Q or K).

__global__ void qk_norm_kernel(
    half* __restrict__ q,           // (Q_DIM,) = (NUM_HEADS * HEAD_DIM,)
    half* __restrict__ k,           // (KV_DIM,) = (NUM_KV_HEADS * HEAD_DIM,)
    const half* __restrict__ q_weight, // (HEAD_DIM,) shared across all Q heads
    const half* __restrict__ k_weight, // (HEAD_DIM,) shared across all K heads
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps
) {
    int block_id = blockIdx.x;
    bool is_q = block_id < num_q_heads;
    int head_idx = is_q ? block_id : (block_id - num_q_heads);

    half* data = is_q ? (q + head_idx * head_dim) : (k + head_idx * head_dim);
    const half* weight = is_q ? q_weight : k_weight;

    // RMSNorm: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = __half2float(data[i]);
        sum_sq += val * val;
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum_sq;
    __syncthreads();

    if (warp_id == 0) {
        sum_sq = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        if (lane_id == 0) shared[0] = sum_sq;
    }
    __syncthreads();

    float rms = rsqrtf(shared[0] / head_dim + eps);

    // Normalize and scale
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float val = __half2float(data[i]) * rms;
        data[i] = __float2half(val * __half2float(weight[i]));
    }
}

// ============================================================================
// Kernel 3: RoPE (Rotary Position Embedding)
// ============================================================================
// Apply rotation to Q and K vectors in-place.

__global__ void rope_kernel(
    half* __restrict__ q,       // (Q_DIM,) = (NUM_HEADS * HEAD_DIM,)
    half* __restrict__ k,       // (KV_DIM,) = (NUM_KV_HEADS * HEAD_DIM,)
    const half* __restrict__ cos_table,  // (HEAD_DIM/2,) for this position
    const half* __restrict__ sin_table,  // (HEAD_DIM/2,) for this position
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_head = head_dim / 2;
    int total_q = num_q_heads * half_head;
    int total_kv = num_kv_heads * half_head;

    // Process Q
    if (idx < total_q) {
        int head = idx / half_head;
        int d = idx % half_head;
        int base = head * head_dim;

        float q0 = __half2float(q[base + d]);
        float q1 = __half2float(q[base + d + half_head]);
        float c = __half2float(cos_table[d]);
        float s = __half2float(sin_table[d]);

        q[base + d] = __float2half(q0 * c - q1 * s);
        q[base + d + half_head] = __float2half(q1 * c + q0 * s);
    }

    // Process K
    if (idx < total_kv) {
        int head = idx / half_head;
        int d = idx % half_head;
        int base = head * head_dim;

        float k0 = __half2float(k[base + d]);
        float k1 = __half2float(k[base + d + half_head]);
        float c = __half2float(cos_table[d]);
        float s = __half2float(sin_table[d]);

        k[base + d] = __float2half(k0 * c - k1 * s);
        k[base + d + half_head] = __float2half(k1 * c + k0 * s);
    }
}

// ============================================================================
// Kernel 4: GQA Attention (decode, single token)
// ============================================================================
// For position `pos`, compute attention over KV cache [0..pos].
// GQA: each Q head group shares one KV head.

__global__ void gqa_attention_decode_kernel(
    const half* __restrict__ q,         // (Q_DIM,) current Q
    const half* __restrict__ k_cache,   // (max_seq, KV_DIM) full K cache
    const half* __restrict__ v_cache,   // (max_seq, KV_DIM) full V cache
    half* __restrict__ output,          // (Q_DIM,) attention output
    float* __restrict__ attn_scratch,   // (NUM_HEADS, max_seq) scratch for scores
    const int* __restrict__ d_pos,      // device-side position pointer
    int pos_unused,                     // kept for non-graph path
    int max_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int head = blockIdx.x; // one block per Q head
    if (head >= num_q_heads) return;

    int pos = (d_pos != nullptr) ? *d_pos : pos_unused;

    int kv_head = head / (num_q_heads / num_kv_heads); // GQA mapping
    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute attention scores: q @ k^T for this head
    float* scores = attn_scratch + head * max_seq_len;
    const half* q_head = q + head * head_dim;

    // Score computation: each thread handles some positions
    float max_score = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const half* k_p = k_cache + p * (num_kv_heads * head_dim) + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += __half2float(q_head[d]) * __half2float(k_p[d]);
        }
        dot *= scale;
        scores[p] = dot;
        if (dot > max_score) max_score = dot;
    }

    // Block-level max reduction for softmax stability
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));

    __shared__ float shared_max[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_max[warp_id] = max_score;
    __syncthreads();
    if (warp_id == 0) {
        max_score = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane_id] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
        if (lane_id == 0) shared_max[0] = max_score;
    }
    __syncthreads();
    max_score = shared_max[0];

    // Softmax: exp and sum
    float sum_exp = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        float val = expf(scores[p] - max_score);
        scores[p] = val;
        sum_exp += val;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);

    __shared__ float shared_sum[32];
    if (lane_id == 0) shared_sum[warp_id] = sum_exp;
    __syncthreads();
    if (warp_id == 0) {
        sum_exp = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, offset);
        if (lane_id == 0) shared_sum[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / (shared_sum[0] + 1e-8f);

    // Normalize scores
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        scores[p] *= inv_sum;
    }
    __syncthreads();

    // Weighted sum of V with half2 vectorized loads (2 elements per load)
    half* out_head = output + head * head_dim;
    int half_dim = head_dim / 2;
    for (int d2 = threadIdx.x; d2 < half_dim; d2 += blockDim.x) {
        float val0 = 0.0f, val1 = 0.0f;
        for (int p = 0; p <= pos; p++) {
            half2 v2 = reinterpret_cast<const half2*>(
                &v_cache[p * (num_kv_heads * head_dim) + kv_head * head_dim])[d2];
            float s = scores[p];
            val0 += s * __half2float(v2.x);
            val1 += s * __half2float(v2.y);
        }
        reinterpret_cast<half2*>(out_head)[d2] = __halves2half2(
            __float2half(val0), __float2half(val1));
    }
}

// ============================================================================
// Kernel 5: Fused SiLU-Gate-Mul
// ============================================================================
// output[i] = silu(gate[i]) * up[i]
// where silu(x) = x * sigmoid(x)

__global__ void silu_gate_mul_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ output,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        float silu_g = g / (1.0f + expf(-g)); // silu = x * sigmoid(x)
        output[idx] = __float2half(silu_g * u);
    }
}

// ============================================================================
// Kernel 6: Embedding Lookup
// ============================================================================

__global__ void embedding_lookup_kernel(
    const half* __restrict__ embed_table,   // (vocab_size, hidden)
    int token_id,
    half* __restrict__ output,              // (hidden,)
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = embed_table[(size_t)token_id * hidden_dim + idx];
    }
}

// ============================================================================
// Kernel 7: Residual Add
// ============================================================================

__global__ void residual_add_kernel(
    half* __restrict__ output,      // output += residual
    const half* __restrict__ residual,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        float o = __half2float(output[idx]);
        float r = __half2float(residual[idx]);
        output[idx] = __float2half(o + r);
    }
}

// ============================================================================
// Kernel 8a: FP16 GEMV (weight @ input -> output, all fp16)
// ============================================================================
// Used for attention projections and dequantized MLP weights.
// One block per output element, threads cooperate on the dot product.

__global__ void fp16_gemv_kernel(
    const half* __restrict__ weight,    // (out_dim, in_dim) row-major
    const half* __restrict__ input,     // (in_dim,)
    half* __restrict__ output,          // (out_dim,)
    int in_dim,
    int out_dim
) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    float sum = 0.0f;
    const half* row = weight + (size_t)out_idx * in_dim;

    for (int j = threadIdx.x; j < in_dim; j += blockDim.x) {
        sum += __half2float(row[j]) * __half2float(input[j]);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) {
            output[out_idx] = __float2half(sum);
        }
    }
}

// ============================================================================
// Kernel 8b: GEMV for lm_head (fp16 weight @ fp16 input -> fp32 logits)
// ============================================================================
// lm_head is tied to embedding, so weights are fp16 (not NF4)

__global__ void fp16_gemv_logits_kernel(
    const half* __restrict__ weight,    // (vocab_size, hidden)
    const half* __restrict__ input,     // (hidden,)
    float* __restrict__ logits,         // (vocab_size,) float32
    int hidden_dim,
    int vocab_size
) {
    int out_idx = blockIdx.x;
    if (out_idx >= vocab_size) return;

    float sum = 0.0f;
    for (int j = threadIdx.x; j < hidden_dim; j += blockDim.x) {
        sum += __half2float(weight[(size_t)out_idx * hidden_dim + j]) *
               __half2float(input[j]);
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        if (lane_id == 0) {
            logits[out_idx] = sum;
        }
    }
}

// ============================================================================
// Kernel 9: GPU-side argmax for greedy/low-temp sampling
// ============================================================================
// Finds the index of the maximum logit value entirely on GPU.
// Returns result in a single int on device memory.

__global__ void argmax_kernel(
    const float* __restrict__ logits,
    int* __restrict__ result,
    int vocab_size
) {
    // Block reduction to find max
    __shared__ float shared_val[256];
    __shared__ int shared_idx[256];

    float max_val = -1e30f;
    int max_idx = 0;

    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float v = logits[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }

    shared_val[threadIdx.x] = max_val;
    shared_idx[threadIdx.x] = max_idx;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_val[threadIdx.x + s] > shared_val[threadIdx.x]) {
                shared_val[threadIdx.x] = shared_val[threadIdx.x + s];
                shared_idx[threadIdx.x] = shared_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *result = shared_idx[0];
    }
}

// ============================================================================
// Kernel 9b: GPU-side multinomial sampling (temperature + softmax + sample)
// ============================================================================
// Single kernel: applies temperature, finds max (for stable softmax),
// computes softmax, then does cumulative sum sampling with a random threshold.
// All on GPU, returns a single token index.
//
// Approach: one block with 256 threads.
// Step 1: temperature scale + find max (parallel reduction)
// Step 2: exp(x - max) + sum (parallel reduction)
// Step 3: normalize (divide by sum)
// Step 4: cumulative sum scan + sample (sequential, but on GPU)

__global__ void gpu_sample_kernel(
    float* __restrict__ logits,    // (vocab_size,) modified in-place
    int* __restrict__ result,       // output: sampled token id
    int vocab_size,
    float temperature,
    float random_val,              // uniform random in [0, 1)
    float top_p
) {
    __shared__ float s_max;
    __shared__ float s_sum;

    // Step 1: temperature + find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        logits[i] /= temperature;
        if (logits[i] > local_max) local_max = logits[i];
    }

    // Reduce max
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));

    __shared__ float shared_max[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        local_max = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_max[lane_id] : -1e30f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
        if (lane_id == 0) s_max = local_max;
    }
    __syncthreads();

    // Step 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        float val = expf(logits[i] - s_max);
        logits[i] = val;  // store unnormalized prob
        local_sum += val;
    }

    // Reduce sum
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float shared_sum[32];
    if (lane_id == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        local_sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        if (lane_id == 0) s_sum = local_sum;
    }
    __syncthreads();

    // Step 3: normalize to probabilities
    float inv_sum = 1.0f / s_sum;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // Step 4: Top-p nucleus sampling using ALL threads for parallel max-find
    if (top_p >= 1.0f) {
        // No top-p, thread 0 does simple cumulative scan
        if (threadIdx.x == 0) {
            float cum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                cum += logits[i];
                if (cum >= random_val) { *result = i; return; }
            }
            *result = vocab_size - 1;
        }
    } else {
        // All threads participate in finding nucleus tokens
        // Iteratively: all threads find max in parallel, thread 0 accumulates
        __shared__ float s_nucleus_mass;
        if (threadIdx.x == 0) s_nucleus_mass = 0.0f;
        __syncthreads();

        while (true) {
            // Parallel max reduction across all threads
            float my_max = 0.0f;
            int my_idx = 0;
            for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
                if (logits[i] > my_max) { my_max = logits[i]; my_idx = i; }
            }

            // Warp reduction for max
            for (int off = warpSize/2; off > 0; off /= 2) {
                float other_max = __shfl_down_sync(0xFFFFFFFF, my_max, off);
                int other_idx = __shfl_down_sync(0xFFFFFFFF, my_idx, off);
                if (other_max > my_max) { my_max = other_max; my_idx = other_idx; }
            }
            if (lane_id == 0) { shared_max[warp_id] = my_max; shared_sum[warp_id] = __int_as_float(my_idx); }
            __syncthreads();
            if (warp_id == 0) {
                my_max = (lane_id < (blockDim.x+31)/32) ? shared_max[lane_id] : 0.0f;
                my_idx = (lane_id < (blockDim.x+31)/32) ? __float_as_int(shared_sum[lane_id]) : 0;
                for (int off = warpSize/2; off > 0; off /= 2) {
                    float o = __shfl_down_sync(0xFFFFFFFF, my_max, off);
                    int oi = __shfl_down_sync(0xFFFFFFFF, my_idx, off);
                    if (o > my_max) { my_max = o; my_idx = oi; }
                }
                if (lane_id == 0) {
                    if (my_max <= 0.0f || s_nucleus_mass >= top_p) {
                        shared_max[0] = -1.0f;  // signal done
                    } else {
                        s_nucleus_mass += my_max;
                        logits[my_idx] = -my_max;  // negate = in nucleus
                        shared_max[0] = my_max;
                    }
                }
            }
            __syncthreads();
            if (shared_max[0] < 0.0f) break;  // done building nucleus
        }

        // Thread 0 samples from nucleus
        if (threadIdx.x == 0) {
            float threshold = random_val * s_nucleus_mass;
            float cum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                if (logits[i] < 0.0f) {
                    cum += -logits[i];
                    if (cum >= threshold) { *result = i; return; }
                }
            }
            *result = vocab_size - 1;
        }
    }
}

// ============================================================================
// Host-side launcher functions
// ============================================================================

extern "C" {


void launch_rms_norm(
    const half* input,
    const half* weight,
    half* output,
    int dim,
    float eps,
    cudaStream_t stream
) {
    rms_norm_kernel<<<1, 256, 0, stream>>>(input, weight, output, dim, eps);
}

void launch_copy_rms_norm(
    const half* input, const half* weight,
    half* residual, half* norm_out,
    int dim, float eps, cudaStream_t stream
) {
    copy_rms_norm_kernel<<<1, 256, 0, stream>>>(input, weight, residual, norm_out, dim, eps);
}

void launch_qk_norm(
    half* q, half* k,
    const half* q_weight, const half* k_weight,
    int num_q_heads, int num_kv_heads, int head_dim,
    float eps, cudaStream_t stream
) {
    int total_heads = num_q_heads + num_kv_heads;
    qk_norm_kernel<<<total_heads, 128, 0, stream>>>(
        q, k, q_weight, k_weight,
        num_q_heads, num_kv_heads, head_dim, eps
    );
}

void launch_rope(
    half* q, half* k,
    const half* cos_table, const half* sin_table,
    int pos, int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream
) {
    int n = num_heads * head_dim / 2;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rope_kernel<<<blocks, threads, 0, stream>>>(
        q, k,
        cos_table + pos * (head_dim / 2),
        sin_table + pos * (head_dim / 2),
        num_heads, num_kv_heads, head_dim
    );
}

void launch_gqa_attention(
    const half* q,
    const half* k_cache, const half* v_cache,
    half* output,
    float* attn_scratch,
    int pos, int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream
) {
    // One block per Q head, 128 threads per block
    gqa_attention_decode_kernel<<<num_heads, 128, 0, stream>>>(
        q, k_cache, v_cache, output, attn_scratch,
        nullptr, pos, max_seq_len,
        num_heads, num_kv_heads, head_dim
    );
}

void launch_silu_gate_mul(
    const half* gate, const half* up, half* output,
    int dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    silu_gate_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, output, dim);
}

void launch_embedding(
    const half* embed_table, int token_id,
    half* output, int hidden_dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        embed_table, token_id, output, hidden_dim
    );
}

void launch_residual_add(
    half* output, const half* residual,
    int dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (dim + threads - 1) / threads;
    residual_add_kernel<<<blocks, threads, 0, stream>>>(output, residual, dim);
}

void launch_fp16_gemv(
    const half* weight,
    const half* input,
    half* output,
    int out_dim,
    int in_dim,
    cudaStream_t stream
) {
    fp16_gemv_kernel<<<out_dim, 128, 0, stream>>>(
        weight, input, output, in_dim, out_dim
    );
}

void launch_gpu_sample(
    float* logits,    // modified in-place!
    int* result,
    int vocab_size,
    float temperature,
    float random_val,
    float top_p,
    cudaStream_t stream
) {
    gpu_sample_kernel<<<1, 256, 0, stream>>>(
        logits, result, vocab_size, temperature, random_val, top_p
    );
}

// ============================================================================
// BATCHED kernels for generate_batch (G sequences in parallel)
// All activation buffers are (dim, G) column-major.
// ============================================================================

// Batched embedding: tokens[G] -> hidden (hidden_size, G)
__global__ void embed_batch_kernel(
    half* hidden, const half* embed_table, const int* tokens, int G, int hidden_size
) {
    int g = blockIdx.x;
    if (g >= G) return;
    int token = tokens[g];
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x)
        hidden[d + g * hidden_size] = embed_table[token * hidden_size + d];
}

// Batched RMSNorm: (dim, G) -> (dim, G), weight is (dim,)
__global__ void rms_norm_batch_kernel(
    half* out, const half* in, const half* weight, int dim, int G, float eps,
    bool biased  // Qwen3.5: output *= (1 + weight) instead of *= weight
) {
    int g = blockIdx.x;
    if (g >= G) return;
    const half* x = in + g * dim;
    half* y = out + g * dim;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = __half2float(x[i]);
        sum_sq += v * v;
    }
    // Block reduction
    for (int off = warpSize/2; off > 0; off /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float s[32];
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    if (lid == 0) s[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lid < (blockDim.x+31)/32) ? s[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
        if (lid == 0) s[0] = sum_sq;
    }
    __syncthreads();
    float rms = rsqrtf(s[0] / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float w = __half2float(weight[i]);
        if (biased) w += 1.0f;  // Qwen3.5 biased RMSNorm: (1 + weight) * normalized
        y[i] = __float2half(__half2float(x[i]) * rms * w);
    }
}

// Batched copy: src (dim, G) -> dst (dim, G)
__global__ void copy_batch_kernel(half* dst, const half* src, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) dst[idx] = src[idx];
}

// Batched residual add: out (dim, G) += residual (dim, G)
__global__ void residual_add_batch_kernel(half* out, const half* res, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float a = __half2float(out[idx]);
        float b = __half2float(res[idx]);
        out[idx] = __float2half(a + b);
    }
}

// Batched QKNorm: apply per-head RMSNorm to Q (q_dim, G) and K (kv_dim, G)
__global__ void qk_norm_batch_kernel(
    half* q, half* k, const half* q_w, const half* k_w,
    int num_q, int num_kv, int head_dim, int G, float eps,
    int q_dim, int kv_dim, bool biased
) {
    int block_id = blockIdx.x;  // over total_heads * G
    int g = block_id / (num_q + num_kv);
    int head_in_block = block_id % (num_q + num_kv);
    if (g >= G) return;

    bool is_q = head_in_block < num_q;
    int head_idx = is_q ? head_in_block : (head_in_block - num_q);
    int dim_total = is_q ? q_dim : kv_dim;
    half* data = is_q ? (q + g * dim_total + head_idx * head_dim)
                      : (k + g * dim_total + head_idx * head_dim);
    const half* w = is_q ? q_w : k_w;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float v = __half2float(data[i]);
        sum_sq += v * v;
    }
    // Full cross-warp reduction (was broken: only captured warp 0)
    for (int off = warpSize/2; off > 0; off /= 2)
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
    __shared__ float s_reduce[32];
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    if (lid == 0) s_reduce[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lid < (blockDim.x + warpSize - 1) / warpSize) ? s_reduce[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, off);
        if (lid == 0) s_reduce[0] = sum_sq;
    }
    __syncthreads();
    float rms = rsqrtf(s_reduce[0] / head_dim + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float wv = __half2float(w[i]);
        if (biased) wv += 1.0f;
        data[i] = __float2half(__half2float(data[i]) * rms * wv);
    }
}

// Batched RoPE: apply to Q (q_dim, G) and K (kv_dim, G) with positions[G]
// Supports partial rotary: only first rope_dim dimensions get rotary encoding.
__global__ void rope_batch_kernel(
    half* q, half* k, const half* cos_table, const half* sin_table,
    const int* positions, int max_seq_len, int G,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim,
    int rope_dim  // rotary dimension (= head_dim for full RoPE, < head_dim for partial)
) {
    int g = blockIdx.x;
    if (g >= G) return;
    int pos = positions[g];
    int half_rope = rope_dim / 2;

    // Apply to all Q heads (only first rope_dim dims, rest untouched)
    for (int h = 0; h < num_heads; h++) {
        half* qh = q + g * q_dim + h * head_dim;
        for (int d = threadIdx.x; d < half_rope; d += blockDim.x) {
            float c = __half2float(cos_table[pos * half_rope + d]);
            float s_val = __half2float(sin_table[pos * half_rope + d]);
            float q0 = __half2float(qh[d]);
            float q1 = __half2float(qh[d + half_rope]);
            qh[d] = __float2half(q0 * c - q1 * s_val);
            qh[d + half_rope] = __float2half(q0 * s_val + q1 * c);
        }
    }
    // Apply to all KV heads
    for (int h = 0; h < num_kv_heads; h++) {
        half* kh = k + g * kv_dim + h * head_dim;
        for (int d = threadIdx.x; d < half_rope; d += blockDim.x) {
            float c = __half2float(cos_table[pos * half_rope + d]);
            float s_val = __half2float(sin_table[pos * half_rope + d]);
            float k0 = __half2float(kh[d]);
            float k1 = __half2float(kh[d + half_rope]);
            kh[d] = __float2half(k0 * c - k1 * s_val);
            kh[d + half_rope] = __float2half(k0 * s_val + k1 * c);
        }
    }
}

// Batched KV cache write: K (kv_dim, G), V (kv_dim, G) -> cache at positions[G]
__global__ void kv_cache_write_batch_kernel(
    half* cache_k, half* cache_v, const half* k, const half* v,
    const int* positions, int max_seq_len, int G, int kv_dim
) {
    int g = blockIdx.x;
    if (g >= G) return;
    int pos = positions[g];
    for (int d = threadIdx.x; d < kv_dim; d += blockDim.x) {
        int src = d + g * kv_dim;
        int dst = (g * max_seq_len + pos) * kv_dim + d;
        cache_k[dst] = k[src];
        cache_v[dst] = v[src];
    }
}

// Batched GQA attention: Q (q_dim, G) x KV_cache -> attn_out (q_dim, G)
// One block per (g, head) pair
__global__ void gqa_attention_batch_kernel(
    half* out, const half* q, const half* cache_k, const half* cache_v,
    float* attn_scratch, const int* positions,
    int max_seq_len, int G,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim
) {
    int g = blockIdx.x;
    int head = blockIdx.y;
    if (g >= G || head >= num_heads) return;

    int gqa_groups = num_heads / num_kv_heads;
    int pos = positions[g];
    int kv_head = head / gqa_groups;
    float scale = rsqrtf((float)head_dim);

    const half* qh = q + g * q_dim + head * head_dim;
    float* scores = attn_scratch + (g * num_heads + head) * max_seq_len;

    // Score computation
    float max_score = -1e30f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        const half* kp = cache_k + (g * max_seq_len + p) * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += __half2float(qh[d]) * __half2float(kp[d]);
        dot *= scale;
        scores[p] = dot;
        max_score = fmaxf(max_score, dot);
    }

    // Reduce max
    for (int off = warpSize/2; off > 0; off /= 2)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
    __shared__ float s_max[32];
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    if (lid == 0) s_max[wid] = max_score;
    __syncthreads();
    if (wid == 0) {
        max_score = (lid < (blockDim.x+31)/32) ? s_max[lid] : -1e30f;
        for (int off = warpSize/2; off > 0; off /= 2)
            max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
        if (lid == 0) s_max[0] = max_score;
    }
    __syncthreads();
    max_score = s_max[0];

    // Softmax
    float sum_exp = 0.0f;
    for (int p = threadIdx.x; p <= pos; p += blockDim.x) {
        float v = expf(scores[p] - max_score);
        scores[p] = v;
        sum_exp += v;
    }
    for (int off = warpSize/2; off > 0; off /= 2)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
    __shared__ float s_sum[32];
    if (lid == 0) s_sum[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        sum_exp = (lid < (blockDim.x+31)/32) ? s_sum[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
        if (lid == 0) s_sum[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_sum[0] + 1e-8f);
    for (int p = threadIdx.x; p <= pos; p += blockDim.x)
        scores[p] *= inv_sum;
    __syncthreads();

    // Weighted sum of V with half2 vectorized loads
    half* oh = out + g * q_dim + head * head_dim;
    int half_dim = head_dim / 2;
    for (int d2 = threadIdx.x; d2 < half_dim; d2 += blockDim.x) {
        float val0 = 0.0f, val1 = 0.0f;
        for (int p = 0; p <= pos; p++) {
            half2 v2 = reinterpret_cast<const half2*>(
                &cache_v[(g * max_seq_len + p) * kv_dim + kv_head * head_dim])[d2];
            float s = scores[p];
            val0 += s * __half2float(v2.x);
            val1 += s * __half2float(v2.y);
        }
        reinterpret_cast<half2*>(oh)[d2] = __halves2half2(
            __float2half(val0), __float2half(val1));
    }
}

// Causal self-attention for chunked prefill (single sequence, T query tokens)
// Grid: dim3(num_heads, T) -- one block per (head, query_position)
// All queries attend to seq 0's KV cache. Causal: query tq attends to 0..tq.
__global__ void gqa_prefill_attention_kernel(
    half* out, const half* q, const half* cache_k, const half* cache_v,
    float* attn_scratch,
    int T, int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim
) {
    int head = blockIdx.x;
    int tq = blockIdx.y;  // query position
    if (head >= num_heads || tq >= T) return;

    int gqa_groups = num_heads / num_kv_heads;
    int kv_head = head / gqa_groups;
    float scale = rsqrtf((float)head_dim);

    // Query at position tq (column-major: q_buf is (q_dim, T))
    const half* qh = q + tq * q_dim + head * head_dim;
    // Scratch: one row per (tq, head), length tq+1
    float* scores = attn_scratch + (tq * num_heads + head) * T;

    // Score: dot product with cached K at positions 0..tq (causal mask)
    // KV cache for seq 0: cache_k[p * kv_dim + kv_head * head_dim]
    float max_score = -1e30f;
    for (int p = threadIdx.x; p <= tq; p += blockDim.x) {
        const half* kp = cache_k + p * kv_dim + kv_head * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += __half2float(qh[d]) * __half2float(kp[d]);
        dot *= scale;
        scores[p] = dot;
        max_score = fmaxf(max_score, dot);
    }

    // Reduce max (same pattern as decode attention)
    for (int off = warpSize/2; off > 0; off /= 2)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
    __shared__ float s_max[32];
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    if (lid == 0) s_max[wid] = max_score;
    __syncthreads();
    if (wid == 0) {
        max_score = (lid < (blockDim.x+31)/32) ? s_max[lid] : -1e30f;
        for (int off = warpSize/2; off > 0; off /= 2)
            max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
        if (lid == 0) s_max[0] = max_score;
    }
    __syncthreads();
    max_score = s_max[0];

    // Softmax
    float sum_exp = 0.0f;
    for (int p = threadIdx.x; p <= tq; p += blockDim.x) {
        float v = expf(scores[p] - max_score);
        scores[p] = v;
        sum_exp += v;
    }
    for (int off = warpSize/2; off > 0; off /= 2)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
    __shared__ float s_sum[32];
    if (lid == 0) s_sum[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        sum_exp = (lid < (blockDim.x+31)/32) ? s_sum[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
        if (lid == 0) s_sum[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_sum[0] + 1e-8f);
    for (int p = threadIdx.x; p <= tq; p += blockDim.x)
        scores[p] *= inv_sum;
    __syncthreads();

    // Weighted V sum (column-major output: out is (q_dim, T))
    half* oh = out + tq * q_dim + head * head_dim;
    int half_dim = head_dim / 2;
    for (int d2 = threadIdx.x; d2 < half_dim; d2 += blockDim.x) {
        float val0 = 0.0f, val1 = 0.0f;
        for (int p = 0; p <= tq; p++) {
            half2 v2 = reinterpret_cast<const half2*>(
                &cache_v[p * kv_dim + kv_head * head_dim])[d2];
            float s = scores[p];
            val0 += s * __half2float(v2.x);
            val1 += s * __half2float(v2.y);
        }
        reinterpret_cast<half2*>(oh)[d2] = __halves2half2(
            __float2half(val0), __float2half(val1));
    }
}

// Batched SiLU gate mul: gate (dim, G) = SiLU(gate) * up
__global__ void silu_mul_batch_kernel(half* gate, const half* up, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        gate[idx] = __float2half((g / (1.0f + expf(-g))) * u);
    }
}

// Batched argmax sampling: logits (VOCAB_SIZE, G) -> tokens[G]
__global__ void argmax_batch_kernel(const float* logits, int* tokens, int vocab, int G) {
    int g = blockIdx.x;
    if (g >= G) return;
    const float* col = logits + g * vocab;
    float best = -1e30f;
    int best_idx = 0;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (col[i] > best) { best = col[i]; best_idx = i; }
    }
    // Warp reduce
    for (int off = warpSize/2; off > 0; off /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, best, off);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, off);
        if (other > best) { best = other; best_idx = other_idx; }
    }
    __shared__ float s_best[32];
    __shared__ int s_idx[32];
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    if (lid == 0) { s_best[wid] = best; s_idx[wid] = best_idx; }
    __syncthreads();
    if (wid == 0) {
        best = (lid < (blockDim.x+31)/32) ? s_best[lid] : -1e30f;
        best_idx = (lid < (blockDim.x+31)/32) ? s_idx[lid] : 0;
        for (int off = warpSize/2; off > 0; off /= 2) {
            float other = __shfl_down_sync(0xFFFFFFFF, best, off);
            int other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, off);
            if (other > best) { best = other; best_idx = other_idx; }
        }
        if (lid == 0) tokens[g] = best_idx;
    }
}

// Batched temperature + top-p sampling: one block per sequence
__global__ void sample_batch_kernel(
    float* __restrict__ logits,   // (VOCAB_SIZE, G) column-major, modified in-place
    int* __restrict__ tokens,     // (G,) output token ids
    const float* const* __restrict__ randoms_ptr, // pointer to (G,) randoms (indirection for CUDA graphs)
    int vocab, int G,
    float temperature, float top_p
) {
    int g = blockIdx.x;
    if (g >= G) return;
    const float* randoms = *randoms_ptr;  // dereference outer pointer
    float* col = logits + (size_t)g * vocab;
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;
    float my_random = randoms[g];

    // Step 1: temperature + find max
    float local_max = -1e30f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        col[i] /= temperature;
        if (col[i] > local_max) local_max = col[i];
    }
    for (int off = warpSize/2; off > 0; off /= 2)
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));
    __shared__ float s_max[32];
    if (lid == 0) s_max[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lid < n_warps) ? s_max[lid] : -1e30f;
        for (int off = warpSize/2; off > 0; off /= 2)
            local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, off));
        if (lid == 0) s_max[0] = local_max;
    }
    __syncthreads();
    float max_val = s_max[0];

    // Step 2: exp + sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        float v = expf(col[i] - max_val);
        col[i] = v;
        local_sum += v;
    }
    for (int off = warpSize/2; off > 0; off /= 2)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);
    __shared__ float s_sum[32];
    if (lid == 0) s_sum[wid] = local_sum;
    __syncthreads();
    if (wid == 0) {
        local_sum = (lid < n_warps) ? s_sum[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);
        if (lid == 0) s_sum[0] = local_sum;
    }
    __syncthreads();

    // Step 3: normalize
    float inv_sum = 1.0f / s_sum[0];
    for (int i = threadIdx.x; i < vocab; i += blockDim.x)
        col[i] *= inv_sum;
    __syncthreads();

    // Step 4: Top-p nucleus sampling (parallel max-find with all threads)
    if (top_p >= 1.0f) {
        if (threadIdx.x == 0) {
            float cum = 0.0f;
            for (int i = 0; i < vocab; i++) {
                cum += col[i];
                if (cum >= my_random) { tokens[g] = i; return; }
            }
            tokens[g] = vocab - 1;
        }
    } else {
        __shared__ float s_nuc_mass;
        if (threadIdx.x == 0) s_nuc_mass = 0.0f;
        __syncthreads();

        while (true) {
            float my_max = 0.0f; int my_idx = 0;
            for (int i = threadIdx.x; i < vocab; i += blockDim.x)
                if (col[i] > my_max) { my_max = col[i]; my_idx = i; }
            for (int off = warpSize/2; off > 0; off /= 2) {
                float o = __shfl_down_sync(0xFFFFFFFF, my_max, off);
                int oi = __shfl_down_sync(0xFFFFFFFF, my_idx, off);
                if (o > my_max) { my_max = o; my_idx = oi; }
            }
            if (lid == 0) { s_max[wid] = my_max; s_sum[wid] = __int_as_float(my_idx); }
            __syncthreads();
            if (wid == 0) {
                my_max = (lid < n_warps) ? s_max[lid] : 0.0f;
                my_idx = (lid < n_warps) ? __float_as_int(s_sum[lid]) : 0;
                for (int off = warpSize/2; off > 0; off /= 2) {
                    float o = __shfl_down_sync(0xFFFFFFFF, my_max, off);
                    int oi = __shfl_down_sync(0xFFFFFFFF, my_idx, off);
                    if (o > my_max) { my_max = o; my_idx = oi; }
                }
                if (lid == 0) {
                    if (my_max <= 0.0f || s_nuc_mass >= top_p) {
                        s_max[0] = -1.0f;
                    } else {
                        s_nuc_mass += my_max;
                        col[my_idx] = -my_max;
                        s_max[0] = my_max;
                    }
                }
            }
            __syncthreads();
            if (s_max[0] < 0.0f) break;
        }

        if (threadIdx.x == 0) {
            float threshold = my_random * s_nuc_mass;
            float cum = 0.0f;
            for (int i = 0; i < vocab; i++) {
                if (col[i] < 0.0f) {
                    cum += -col[i];
                    if (cum >= threshold) { tokens[g] = i; return; }
                }
            }
            tokens[g] = vocab - 1;
        }
    }
}

void launch_sample_batch(float* logits, int* tokens, const float* const* randoms_ptr,
                         int vocab, int G, float temperature, float top_p, cudaStream_t s) {
    sample_batch_kernel<<<G, 256, 0, s>>>(logits, tokens, randoms_ptr, vocab, G, temperature, top_p);
}

// Launch wrappers for batch kernels
void launch_embed_batch(half* h, const half* et, const int* tok, int G, int hidden_size, cudaStream_t s) {
    embed_batch_kernel<<<G, 256, 0, s>>>(h, et, tok, G, hidden_size);
}
void launch_rms_norm_batch(half* out, const half* in, const half* w, int dim, int G, float eps, cudaStream_t s, bool biased = false) {
    rms_norm_batch_kernel<<<G, 256, 0, s>>>(out, in, w, dim, G, eps, biased);
}
void launch_copy_batch(half* dst, const half* src, int total, cudaStream_t s) {
    copy_batch_kernel<<<(total+255)/256, 256, 0, s>>>(dst, src, total);
}
void launch_residual_add_batch(half* out, const half* res, int total, cudaStream_t s) {
    residual_add_batch_kernel<<<(total+255)/256, 256, 0, s>>>(out, res, total);
}
void launch_qk_norm_batch(half* q, half* k, const half* qw, const half* kw, int nq, int nkv, int hd, int G, float eps, int q_dim, int kv_dim, cudaStream_t s, bool biased = false) {
    qk_norm_batch_kernel<<<G * (nq + nkv), hd, 0, s>>>(q, k, qw, kw, nq, nkv, hd, G, eps, q_dim, kv_dim, biased);
}
void launch_rope_batch(half* q, half* k, const half* ct, const half* st, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, int rope_dim, cudaStream_t s) {
    rope_batch_kernel<<<G, 256, 0, s>>>(q, k, ct, st, pos, msl, G, num_heads, num_kv_heads, head_dim, q_dim, kv_dim, rope_dim);
}
void launch_kv_cache_write_batch(half* ck, half* cv, const half* k, const half* v, const int* pos, int msl, int G, int kv_dim, cudaStream_t s) {
    kv_cache_write_batch_kernel<<<G, 256, 0, s>>>(ck, cv, k, v, pos, msl, G, kv_dim);
}
void launch_gqa_attention_batch(half* out, const half* q, const half* ck, const half* cv, float* as, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s) {
    dim3 grid(G, num_heads);
    gqa_attention_batch_kernel<<<grid, head_dim, 0, s>>>(out, q, ck, cv, as, pos, msl, G, num_heads, num_kv_heads, head_dim, q_dim, kv_dim);
}

void launch_gqa_prefill_attention(half* out, const half* q, const half* ck, const half* cv, float* as, int T, int msl, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s) {
    dim3 grid(num_heads, T);
    gqa_prefill_attention_kernel<<<grid, head_dim, 0, s>>>(out, q, ck, cv, as, T, msl, num_heads, num_kv_heads, head_dim, q_dim, kv_dim);
}
void launch_silu_mul_batch(half* gate, const half* up, int total, cudaStream_t s) {
    silu_mul_batch_kernel<<<(total+255)/256, 256, 0, s>>>(gate, up, total);
}
void launch_argmax_batch(const float* logits, int* tokens, int vocab, int G, cudaStream_t s) {
    argmax_batch_kernel<<<G, 256, 0, s>>>(logits, tokens, vocab, G);
}

// Increment all positions by 1 (avoids host-device sync per token)
__global__ void increment_positions_kernel(int* positions, int G) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < G) positions[g]++;
}

void launch_increment_positions(int* positions, int G, cudaStream_t s) {
    increment_positions_kernel<<<(G + 255) / 256, 256, 0, s>>>(positions, G);
}

// Convert fp16 array to fp32 (for NF4 LM head → fp32 logits)
__global__ void fp16_to_fp32_kernel(const half* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __half2float(input[idx]);
}

void launch_fp16_to_fp32(const half* input, float* output, int n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    fp16_to_fp32_kernel<<<blocks, 256, 0, stream>>>(input, output, n);
}

void launch_argmax(
    const float* logits,
    int* result,
    int vocab_size,
    cudaStream_t stream
) {
    argmax_kernel<<<1, 256, 0, stream>>>(logits, result, vocab_size);
}

void launch_lm_head(
    const half* weight, const half* input,
    float* logits, int hidden_dim, int vocab_size,
    cudaStream_t stream
) {
    fp16_gemv_logits_kernel<<<vocab_size, 128, 0, stream>>>(
        weight, input, logits, hidden_dim, vocab_size
    );
}

// ============================================================================
// TurboQuant: 2-bit KV Cache Quantization (Zandieh et al. 2025)
// ============================================================================
//
// Algorithm: randomly rotate K/V vectors, then quantize each coordinate to
// 2 bits using Lloyd-Max centroids precomputed for the resulting Beta/Gaussian
// distribution. Inner products are preserved because softmax is scale-invariant
// (K side) and V weighted sums are rotated back after accumulation.
//
// Memory: 2 bits/value → 8x reduction vs fp16 (32 bytes/head vs 256 bytes/head)
// Quality: near-lossless at 3.5 bits, marginal degradation at 2 bits (LongBench)

// Lloyd-Max centroids for N(0, 1/d) with d=128
// σ = 1/√128 ≈ 0.08839

// 2-bit: 4 centroids at ±0.4528σ, ±1.5104σ
__constant__ float TURBO_CENTROIDS_2[4] = {
    -0.13351f, -0.04003f, 0.04003f, 0.13351f
};
__constant__ float TURBO_BOUNDS_2[3] = {
    -0.08677f, 0.0f, 0.08677f
};

// 4-bit: 16 centroids (standard Lloyd-Max for N(0,1) scaled by σ)
__constant__ float TURBO_CENTROIDS_4[16] = {
    -0.24155f, -0.18293f, -0.14304f, -0.11099f,
    -0.08330f, -0.05806f, -0.03432f, -0.01135f,
     0.01135f,  0.03432f,  0.05806f,  0.08330f,
     0.11099f,  0.14304f,  0.18293f,  0.24155f
};
__constant__ float TURBO_BOUNDS_4[15] = {
    -0.21224f, -0.16299f, -0.12701f, -0.09715f,
    -0.07068f, -0.04619f, -0.02284f,  0.00000f,
     0.02284f,  0.04619f,  0.07068f,  0.09715f,
     0.12701f,  0.16299f,  0.21224f
};

// Helper: quantize a rotated value to b-bit index using precomputed boundaries
__device__ __forceinline__ int turbo_quantize(float val, int bits) {
    if (bits == 4) {
        // Binary search for 16 centroids (4 comparisons)
        int idx;
        if (val < TURBO_BOUNDS_4[7]) {
            if (val < TURBO_BOUNDS_4[3]) {
                if (val < TURBO_BOUNDS_4[1]) idx = val < TURBO_BOUNDS_4[0] ? 0 : 1;
                else idx = val < TURBO_BOUNDS_4[2] ? 2 : 3;
            } else {
                if (val < TURBO_BOUNDS_4[5]) idx = val < TURBO_BOUNDS_4[4] ? 4 : 5;
                else idx = val < TURBO_BOUNDS_4[6] ? 6 : 7;
            }
        } else {
            if (val < TURBO_BOUNDS_4[11]) {
                if (val < TURBO_BOUNDS_4[9]) idx = val < TURBO_BOUNDS_4[8] ? 8 : 9;
                else idx = val < TURBO_BOUNDS_4[10] ? 10 : 11;
            } else {
                if (val < TURBO_BOUNDS_4[13]) idx = val < TURBO_BOUNDS_4[12] ? 12 : 13;
                else idx = val < TURBO_BOUNDS_4[14] ? 14 : 15;
            }
        }
        return idx;
    } else {
        return (val >= TURBO_BOUNDS_2[0]) + (val >= TURBO_BOUNDS_2[1]) + (val >= TURBO_BOUNDS_2[2]);
    }
}

// Helper: look up centroid value from index
__device__ __forceinline__ float turbo_centroid(int idx, int bits) {
    return (bits == 4) ? TURBO_CENTROIDS_4[idx] : TURBO_CENTROIDS_2[idx];
}

// Quantize + pack: rotate K or V head by Π, quantize, pack.
// Grid: (num_tokens, num_kv_heads), blockDim: 128 (= head_dim)
__global__ void turbo_kv_quantize_kernel(
    uint8_t* __restrict__ cache_q,
    half* __restrict__ cache_norms,
    const half* __restrict__ kv_buf,
    const half* __restrict__ rotation,
    const int* __restrict__ positions,
    int max_seq_len, int num_tokens,
    int num_kv_heads, int head_dim, int kv_dim,
    int bits  // 2 or 4
) {
    int tok = blockIdx.x;
    int h = blockIdx.y;
    if (tok >= num_tokens || h >= num_kv_heads) return;

    int d = threadIdx.x;
    if (d >= head_dim) return;

    int pos = positions[tok];
    int vals_per_byte = (bits == 2) ? 4 : 2;
    int bytes_per_head = head_dim / vals_per_byte;

    // Load and compute norm
    __shared__ float head_vec[128];
    float val = __half2float(kv_buf[tok * kv_dim + h * head_dim + d]);
    head_vec[d] = val;

    float norm_sq = val * val;
    for (int off = warpSize / 2; off > 0; off /= 2)
        norm_sq += __shfl_down_sync(0xFFFFFFFF, norm_sq, off);
    __shared__ float s_partial[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_partial[wid] = norm_sq;
    __syncthreads();
    if (wid == 0) {
        norm_sq = (lid < (head_dim + 31) / 32) ? s_partial[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            norm_sq += __shfl_down_sync(0xFFFFFFFF, norm_sq, off);
        if (lid == 0) s_partial[0] = norm_sq;
    }
    __syncthreads();

    float inv_norm = rsqrtf(s_partial[0] + 1e-12f);
    head_vec[d] *= inv_norm;
    __syncthreads();

    // Rotate + quantize
    const half* rot_row = rotation + d * head_dim;
    float rotated = 0.0f;
    for (int k = 0; k < head_dim; k++)
        rotated += __half2float(rot_row[k]) * head_vec[k];

    int idx = turbo_quantize(rotated, bits);

    __shared__ uint8_t quant_idx[128];
    quant_idx[d] = (uint8_t)idx;
    __syncthreads();

    // Pack
    int kv_dim_packed = kv_dim / vals_per_byte;
    if (d < bytes_per_head) {
        uint8_t packed;
        if (bits == 2) {
            int base = d * 4;
            packed = quant_idx[base]
                   | (quant_idx[base + 1] << 2)
                   | (quant_idx[base + 2] << 4)
                   | (quant_idx[base + 3] << 6);
        } else {
            int base = d * 2;
            packed = quant_idx[base] | (quant_idx[base + 1] << 4);
        }
        cache_q[pos * kv_dim_packed + h * bytes_per_head + d] = packed;
    }

    if (d == 0) {
        float norm = sqrtf(s_partial[0] + 1e-12f);
        cache_norms[pos * num_kv_heads + h] = __float2half(norm);
    }
}

// Batched variant
__global__ void turbo_kv_quantize_batch_kernel(
    uint8_t* __restrict__ cache_q,
    half* __restrict__ cache_norms,
    const half* __restrict__ kv_buf,
    const half* __restrict__ rotation,
    const int* __restrict__ positions,
    int max_seq_len, int G,
    int num_kv_heads, int head_dim, int kv_dim,
    int bits
) {
    int block_id = blockIdx.x;
    int g = block_id / num_kv_heads;
    int h = block_id % num_kv_heads;
    if (g >= G) return;

    int d = threadIdx.x;
    if (d >= head_dim) return;

    int pos = positions[g];
    int vals_per_byte = (bits == 2) ? 4 : 2;
    int bytes_per_head = head_dim / vals_per_byte;
    int kv_dim_packed = kv_dim / vals_per_byte;

    __shared__ float head_vec[128];
    float val = __half2float(kv_buf[g * kv_dim + h * head_dim + d]);
    head_vec[d] = val;

    float norm_sq = val * val;
    for (int off = warpSize / 2; off > 0; off /= 2)
        norm_sq += __shfl_down_sync(0xFFFFFFFF, norm_sq, off);
    __shared__ float s_partial[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_partial[wid] = norm_sq;
    __syncthreads();
    if (wid == 0) {
        norm_sq = (lid < (head_dim + 31) / 32) ? s_partial[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            norm_sq += __shfl_down_sync(0xFFFFFFFF, norm_sq, off);
        if (lid == 0) s_partial[0] = norm_sq;
    }
    __syncthreads();

    float inv_norm = rsqrtf(s_partial[0] + 1e-12f);
    head_vec[d] *= inv_norm;
    __syncthreads();

    const half* rot_row = rotation + d * head_dim;
    float rotated = 0.0f;
    for (int k = 0; k < head_dim; k++)
        rotated += __half2float(rot_row[k]) * head_vec[k];

    int idx = turbo_quantize(rotated, bits);

    __shared__ uint8_t quant_idx[128];
    quant_idx[d] = (uint8_t)idx;
    __syncthreads();

    if (d < bytes_per_head) {
        uint8_t packed;
        if (bits == 2) {
            int base = d * 4;
            packed = quant_idx[base] | (quant_idx[base+1]<<2) | (quant_idx[base+2]<<4) | (quant_idx[base+3]<<6);
        } else {
            int base = d * 2;
            packed = quant_idx[base] | (quant_idx[base+1] << 4);
        }
        cache_q[(g * max_seq_len + pos) * kv_dim_packed + h * bytes_per_head + d] = packed;
    }

    if (d == 0) {
        float norm = sqrtf(s_partial[0] + 1e-12f);
        cache_norms[(g * max_seq_len + pos) * num_kv_heads + h] = __float2half(norm);
    }
}

// Unified TurboQuant attention kernel (batch decode).
// Supports both 2-bit and 4-bit via the `bits` parameter.
// Grid: (G, num_heads), blockDim: 128 (= head_dim)
__global__ void turbo_gqa_attention_batch_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const uint8_t* __restrict__ cache_k_q,
    const uint8_t* __restrict__ cache_v_q,
    const half* __restrict__ k_norms,
    const half* __restrict__ v_norms,
    const half* __restrict__ rotation,
    float* __restrict__ attn_scratch,
    const int* __restrict__ positions,
    int max_seq_len, int G,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim,
    int bits
) {
    int g = blockIdx.x;
    int head = blockIdx.y;
    if (g >= G || head >= num_heads) return;

    int d = threadIdx.x;
    if (d >= head_dim) return;

    int gqa_groups = num_heads / num_kv_heads;
    int pos = positions[g];
    int kv_head = head / gqa_groups;
    float scale = rsqrtf((float)head_dim);
    int vals_per_byte = (bits == 2) ? 4 : 2;
    int bytes_per_head = head_dim / vals_per_byte;
    int kv_dim_packed = kv_dim / vals_per_byte;

    // --- Phase 1: Rotate Q ---
    __shared__ float q_rot[128];
    {
        __shared__ float q_raw[128];
        q_raw[d] = __half2float(q[g * q_dim + head * head_dim + d]);
        __syncthreads();
        const half* rot_row = rotation + d * head_dim;
        float r = 0.0f;
        for (int k = 0; k < head_dim; k++)
            r += __half2float(rot_row[k]) * q_raw[k];
        q_rot[d] = r;
    }
    __syncthreads();

    // --- Phase 2: Score computation ---
    float* scores = attn_scratch + (g * num_heads + head) * max_seq_len;
    float max_score = -1e30f;

    for (int p = d; p <= pos; p += head_dim) {
        float k_norm = __half2float(k_norms[(g * max_seq_len + p) * num_kv_heads + kv_head]);
        int k_off = (g * max_seq_len + p) * kv_dim_packed + kv_head * bytes_per_head;
        float dot = 0.0f;
        if (bits == 4) {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t packed = cache_k_q[k_off + b];
                dot += q_rot[b*2+0] * TURBO_CENTROIDS_4[packed & 0xF];
                dot += q_rot[b*2+1] * TURBO_CENTROIDS_4[(packed >> 4) & 0xF];
            }
        } else {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t packed = cache_k_q[k_off + b];
                dot += q_rot[b*4+0] * TURBO_CENTROIDS_2[(packed   ) & 3];
                dot += q_rot[b*4+1] * TURBO_CENTROIDS_2[(packed>>2) & 3];
                dot += q_rot[b*4+2] * TURBO_CENTROIDS_2[(packed>>4) & 3];
                dot += q_rot[b*4+3] * TURBO_CENTROIDS_2[(packed>>6) & 3];
            }
        }
        dot *= k_norm * scale;
        scores[p] = dot;
        max_score = fmaxf(max_score, dot);
    }

    // Reduce max + softmax (same pattern for all bit widths)
    for (int off = warpSize/2; off > 0; off /= 2)
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
    __shared__ float s_max[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_max[wid] = max_score;
    __syncthreads();
    if (wid == 0) {
        max_score = (lid < (head_dim+31)/32) ? s_max[lid] : -1e30f;
        for (int off = warpSize/2; off > 0; off /= 2)
            max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, off));
        if (lid == 0) s_max[0] = max_score;
    }
    __syncthreads();
    max_score = s_max[0];

    float sum_exp = 0.0f;
    for (int p = d; p <= pos; p += head_dim) {
        float v = expf(scores[p] - max_score);
        scores[p] = v;
        sum_exp += v;
    }
    for (int off = warpSize/2; off > 0; off /= 2)
        sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
    __shared__ float s_sum[4];
    if (lid == 0) s_sum[wid] = sum_exp;
    __syncthreads();
    if (wid == 0) {
        sum_exp = (lid < (head_dim+31)/32) ? s_sum[lid] : 0.0f;
        for (int off = warpSize/2; off > 0; off /= 2)
            sum_exp += __shfl_down_sync(0xFFFFFFFF, sum_exp, off);
        if (lid == 0) s_sum[0] = sum_exp;
    }
    __syncthreads();
    float inv_sum = 1.0f / (s_sum[0] + 1e-8f);
    for (int p = d; p <= pos; p += head_dim)
        scores[p] *= inv_sum;
    __syncthreads();

    // --- Phase 3: V weighted sum ---
    int sub, byte_in_head;
    if (bits == 4) { sub = d % 2; byte_in_head = d / 2; }
    else { sub = d % 4; byte_in_head = d / 4; }
    int bit_shift = (bits == 4) ? sub * 4 : sub * 2;
    int mask = (bits == 4) ? 0xF : 3;
    float v_acc = 0.0f;

    for (int p = 0; p <= pos; p++) {
        float s = scores[p];
        float vn = __half2float(v_norms[(g * max_seq_len + p) * num_kv_heads + kv_head]);
        int v_off = (g * max_seq_len + p) * kv_dim_packed + kv_head * bytes_per_head + byte_in_head;
        uint8_t packed = cache_v_q[v_off];
        int idx = (packed >> bit_shift) & mask;
        v_acc += s * vn * turbo_centroid(idx, bits);
    }

    // --- Phase 4: Rotate V back ---
    __shared__ float v_acc_shared[128];
    v_acc_shared[d] = v_acc;
    __syncthreads();
    float out_val = 0.0f;
    for (int k = 0; k < head_dim; k++)
        out_val += __half2float(rotation[k * head_dim + d]) * v_acc_shared[k];
    out[g * q_dim + head * head_dim + d] = __float2half(out_val);
}

// Single-sequence decode attention (same structure, no g offset)
__global__ void turbo_gqa_attention_kernel(
    const half* __restrict__ q,
    const uint8_t* __restrict__ cache_k_q,
    const uint8_t* __restrict__ cache_v_q,
    const half* __restrict__ k_norms,
    const half* __restrict__ v_norms,
    const half* __restrict__ rotation,
    half* __restrict__ output,
    float* __restrict__ attn_scratch,
    int pos, int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim,
    int bits
) {
    int head = blockIdx.x;
    if (head >= num_heads) return;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int gqa_groups = num_heads / num_kv_heads;
    int kv_head = head / gqa_groups;
    float scale = rsqrtf((float)head_dim);
    int kv_dim = num_kv_heads * head_dim;
    int vals_per_byte = (bits == 2) ? 4 : 2;
    int bytes_per_head = head_dim / vals_per_byte;
    int kv_dim_packed = kv_dim / vals_per_byte;

    __shared__ float q_rot[128];
    { __shared__ float q_raw[128];
      q_raw[d] = __half2float(q[head * head_dim + d]);
      __syncthreads();
      float r = 0.0f;
      const half* rot_row = rotation + d * head_dim;
      for (int k = 0; k < head_dim; k++) r += __half2float(rot_row[k]) * q_raw[k];
      q_rot[d] = r;
    }
    __syncthreads();

    float* scores = attn_scratch + head * max_seq_len;
    float max_score = -1e30f;
    for (int p = d; p <= pos; p += head_dim) {
        float kn = __half2float(k_norms[p * num_kv_heads + kv_head]);
        int k_off = p * kv_dim_packed + kv_head * bytes_per_head;
        float dot = 0.0f;
        if (bits == 4) {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t pk = cache_k_q[k_off+b];
                dot += q_rot[b*2+0]*TURBO_CENTROIDS_4[pk & 0xF];
                dot += q_rot[b*2+1]*TURBO_CENTROIDS_4[(pk>>4) & 0xF];
            }
        } else {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t pk = cache_k_q[k_off+b];
                dot += q_rot[b*4+0]*TURBO_CENTROIDS_2[(pk   )&3];
                dot += q_rot[b*4+1]*TURBO_CENTROIDS_2[(pk>>2)&3];
                dot += q_rot[b*4+2]*TURBO_CENTROIDS_2[(pk>>4)&3];
                dot += q_rot[b*4+3]*TURBO_CENTROIDS_2[(pk>>6)&3];
            }
        }
        dot *= kn * scale; scores[p] = dot;
        max_score = fmaxf(max_score, dot);
    }

    for (int off=warpSize/2;off>0;off/=2) max_score=fmaxf(max_score,__shfl_down_sync(0xFFFFFFFF,max_score,off));
    __shared__ float s_max[4]; int wid=d/warpSize, lid=d%warpSize;
    if(lid==0) s_max[wid]=max_score; __syncthreads();
    if(wid==0){max_score=(lid<(head_dim+31)/32)?s_max[lid]:-1e30f;
      for(int off=warpSize/2;off>0;off/=2) max_score=fmaxf(max_score,__shfl_down_sync(0xFFFFFFFF,max_score,off));
      if(lid==0)s_max[0]=max_score;} __syncthreads(); max_score=s_max[0];

    float sum_exp=0; for(int p=d;p<=pos;p+=head_dim){float v=expf(scores[p]-max_score);scores[p]=v;sum_exp+=v;}
    for(int off=warpSize/2;off>0;off/=2) sum_exp+=__shfl_down_sync(0xFFFFFFFF,sum_exp,off);
    __shared__ float s_sum[4]; if(lid==0)s_sum[wid]=sum_exp; __syncthreads();
    if(wid==0){sum_exp=(lid<(head_dim+31)/32)?s_sum[lid]:0;
      for(int off=warpSize/2;off>0;off/=2) sum_exp+=__shfl_down_sync(0xFFFFFFFF,sum_exp,off);
      if(lid==0)s_sum[0]=sum_exp;} __syncthreads();
    float inv_sum=1.0f/(s_sum[0]+1e-8f);
    for(int p=d;p<=pos;p+=head_dim) scores[p]*=inv_sum; __syncthreads();

    int sub, byte_in_head;
    if(bits==4){sub=d%2;byte_in_head=d/2;} else{sub=d%4;byte_in_head=d/4;}
    int bit_shift=(bits==4)?sub*4:sub*2; int mask=(bits==4)?0xF:3;
    float v_acc=0;
    for(int p=0;p<=pos;p++){
        float s=scores[p]; float vn=__half2float(v_norms[p*num_kv_heads+kv_head]);
        uint8_t pk=cache_v_q[p*kv_dim_packed+kv_head*bytes_per_head+byte_in_head];
        v_acc+=s*vn*turbo_centroid((pk>>bit_shift)&mask,bits);
    }

    __shared__ float v_acc_shared[128]; v_acc_shared[d]=v_acc; __syncthreads();
    float out_val=0; for(int k=0;k<head_dim;k++) out_val+=__half2float(rotation[k*head_dim+d])*v_acc_shared[k];
    output[head*head_dim+d]=__float2half(out_val);
}

// Prefill attention with TurboQuant (causal).
// Grid: (num_heads, T), blockDim: 128
__global__ void turbo_gqa_prefill_attention_kernel(
    half* __restrict__ out,
    const half* __restrict__ q,
    const uint8_t* __restrict__ cache_k_q,
    const uint8_t* __restrict__ cache_v_q,
    const half* __restrict__ k_norms,
    const half* __restrict__ v_norms,
    const half* __restrict__ rotation,
    float* __restrict__ attn_scratch,
    int T, int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim,
    int bits
) {
    int head = blockIdx.x; int tq = blockIdx.y;
    if (head >= num_heads || tq >= T) return;
    int d = threadIdx.x; if (d >= head_dim) return;

    int gqa_groups = num_heads / num_kv_heads;
    int kv_head = head / gqa_groups;
    float scale = rsqrtf((float)head_dim);
    int vals_per_byte = (bits == 2) ? 4 : 2;
    int bytes_per_head = head_dim / vals_per_byte;
    int kv_dim_packed = kv_dim / vals_per_byte;

    __shared__ float q_rot[128];
    { __shared__ float q_raw[128];
      q_raw[d] = __half2float(q[tq * q_dim + head * head_dim + d]);
      __syncthreads();
      float r = 0.0f; const half* rot_row = rotation + d * head_dim;
      for (int k = 0; k < head_dim; k++) r += __half2float(rot_row[k]) * q_raw[k];
      q_rot[d] = r;
    } __syncthreads();

    float* scores = attn_scratch + (tq * num_heads + head) * T;
    float max_score = -1e30f;
    for (int p = d; p <= tq; p += head_dim) {
        float kn = __half2float(k_norms[p * num_kv_heads + kv_head]);
        int k_off = p * kv_dim_packed + kv_head * bytes_per_head;
        float dot = 0.0f;
        if (bits == 4) {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t pk = cache_k_q[k_off+b];
                dot += q_rot[b*2+0]*TURBO_CENTROIDS_4[pk & 0xF];
                dot += q_rot[b*2+1]*TURBO_CENTROIDS_4[(pk>>4) & 0xF];
            }
        } else {
            for (int b = 0; b < bytes_per_head; b++) {
                uint8_t pk = cache_k_q[k_off+b];
                dot += q_rot[b*4+0]*TURBO_CENTROIDS_2[(pk   )&3];
                dot += q_rot[b*4+1]*TURBO_CENTROIDS_2[(pk>>2)&3];
                dot += q_rot[b*4+2]*TURBO_CENTROIDS_2[(pk>>4)&3];
                dot += q_rot[b*4+3]*TURBO_CENTROIDS_2[(pk>>6)&3];
            }
        }
        dot *= kn * scale; scores[p] = dot; max_score = fmaxf(max_score, dot);
    }

    for(int off=warpSize/2;off>0;off/=2) max_score=fmaxf(max_score,__shfl_down_sync(0xFFFFFFFF,max_score,off));
    __shared__ float s_max[4]; int wid=d/warpSize, lid=d%warpSize;
    if(lid==0) s_max[wid]=max_score; __syncthreads();
    if(wid==0){max_score=(lid<(head_dim+31)/32)?s_max[lid]:-1e30f;
      for(int off=warpSize/2;off>0;off/=2) max_score=fmaxf(max_score,__shfl_down_sync(0xFFFFFFFF,max_score,off));
      if(lid==0)s_max[0]=max_score;} __syncthreads(); max_score=s_max[0];

    float sum_exp=0; for(int p=d;p<=tq;p+=head_dim){float v=expf(scores[p]-max_score);scores[p]=v;sum_exp+=v;}
    for(int off=warpSize/2;off>0;off/=2) sum_exp+=__shfl_down_sync(0xFFFFFFFF,sum_exp,off);
    __shared__ float s_sum[4]; if(lid==0)s_sum[wid]=sum_exp; __syncthreads();
    if(wid==0){sum_exp=(lid<(head_dim+31)/32)?s_sum[lid]:0;
      for(int off=warpSize/2;off>0;off/=2) sum_exp+=__shfl_down_sync(0xFFFFFFFF,sum_exp,off);
      if(lid==0)s_sum[0]=sum_exp;} __syncthreads();
    float inv_sum=1.0f/(s_sum[0]+1e-8f);
    for(int p=d;p<=tq;p+=head_dim) scores[p]*=inv_sum; __syncthreads();

    int sub, byte_in_head;
    if(bits==4){sub=d%2;byte_in_head=d/2;} else{sub=d%4;byte_in_head=d/4;}
    int bit_shift=(bits==4)?sub*4:sub*2; int mask=(bits==4)?0xF:3;
    float v_acc=0;
    for(int p=0;p<=tq;p++){
        float s=scores[p]; float vn=__half2float(v_norms[p*num_kv_heads+kv_head]);
        uint8_t pk=cache_v_q[p*kv_dim_packed+kv_head*bytes_per_head+byte_in_head];
        v_acc+=s*vn*turbo_centroid((pk>>bit_shift)&mask,bits);
    }

    __shared__ float v_acc_shared[128]; v_acc_shared[d]=v_acc; __syncthreads();
    float out_val=0; for(int k=0;k<head_dim;k++) out_val+=__half2float(rotation[k*head_dim+d])*v_acc_shared[k];
    out[tq*q_dim+head*head_dim+d]=__float2half(out_val);
}

// Launch wrappers for TurboQuant kernels
void launch_turbo_kv_quantize(
    uint8_t* cache_q, half* cache_norms,
    const half* kv_buf, const half* rotation, const int* positions,
    int max_seq_len, int num_tokens, int num_kv_heads, int head_dim, int kv_dim,
    int bits, cudaStream_t s
) {
    dim3 grid(num_tokens, num_kv_heads);
    turbo_kv_quantize_kernel<<<grid, head_dim, 0, s>>>(
        cache_q, cache_norms, kv_buf, rotation, positions,
        max_seq_len, num_tokens, num_kv_heads, head_dim, kv_dim, bits);
}

void launch_turbo_kv_quantize_batch(
    uint8_t* cache_q, half* cache_norms,
    const half* kv_buf, const half* rotation, const int* positions,
    int max_seq_len, int G, int num_kv_heads, int head_dim, int kv_dim,
    int bits, cudaStream_t s
) {
    turbo_kv_quantize_batch_kernel<<<G * num_kv_heads, head_dim, 0, s>>>(
        cache_q, cache_norms, kv_buf, rotation, positions,
        max_seq_len, G, num_kv_heads, head_dim, kv_dim, bits);
}

void launch_turbo_gqa_attention_batch(
    half* out, const half* q,
    const uint8_t* ck_q, const uint8_t* cv_q,
    const half* k_norms, const half* v_norms,
    const half* rotation,
    float* as, const int* pos,
    int msl, int G, int num_heads, int num_kv_heads, int head_dim,
    int q_dim, int kv_dim, int bits, cudaStream_t s
) {
    dim3 grid(G, num_heads);
    turbo_gqa_attention_batch_kernel<<<grid, head_dim, 0, s>>>(
        out, q, ck_q, cv_q, k_norms, v_norms, rotation, as, pos,
        msl, G, num_heads, num_kv_heads, head_dim, q_dim, kv_dim, bits);
}

void launch_turbo_gqa_attention(
    const half* q,
    const uint8_t* ck_q, const uint8_t* cv_q,
    const half* k_norms, const half* v_norms,
    const half* rotation,
    half* output, float* as,
    int pos, int msl,
    int num_heads, int num_kv_heads, int head_dim, int bits, cudaStream_t s
) {
    turbo_gqa_attention_kernel<<<num_heads, head_dim, 0, s>>>(
        q, ck_q, cv_q, k_norms, v_norms, rotation, output, as,
        pos, msl, num_heads, num_kv_heads, head_dim, bits);
}

void launch_turbo_gqa_prefill_attention(
    half* out, const half* q,
    const uint8_t* ck_q, const uint8_t* cv_q,
    const half* k_norms, const half* v_norms,
    const half* rotation,
    float* as, int T, int msl,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim,
    int bits, cudaStream_t s
) {
    dim3 grid(num_heads, T);
    turbo_gqa_prefill_attention_kernel<<<grid, head_dim, 0, s>>>(
        out, q, ck_q, cv_q, k_norms, v_norms, rotation, as,
        T, msl, num_heads, num_kv_heads, head_dim, q_dim, kv_dim, bits);
}

// ============================================================================
// SSM: Gated Delta Rule (Qwen3.5 hybrid architecture)
// ============================================================================
//
// Implements the decode step for the Gated Delta Rule SSM:
//   dt = softplus(a + dt_bias)
//   decay = exp(-exp(A_log) * dt)
//   beta = sigmoid(b)
//   kv_recall = state @ k          (what state remembers for this key)
//   delta = beta * (v - kv_recall)  (selective update)
//   state = decay * state + outer(k, delta)
//   output = state @ q
//
// State shape: (num_v_heads, k_head_dim, v_head_dim) per sequence

// --- Causal conv1d (decode: single token) ---
// Each channel does: output = silu(sum(weight[0..3] * [conv_state[0..2], input]))
// Then shifts conv_state left and pushes input.
// Grid: ceil(total_channels * G / 256), Block: 256
__global__ void ssm_conv1d_decode_kernel(
    float* __restrict__ output,      // (conv_dim, G) col-major — fp32 for SSM precision
    float* __restrict__ conv_state,  // (G, conv_dim, kernel-1) row-major — fp32
    const half* __restrict__ input,  // (conv_dim, G) col-major — fp16 from GEMM
    const half* __restrict__ weight, // (conv_dim, kernel) row-major
    const half* __restrict__ bias,   // (conv_dim,) or nullptr
    int conv_dim, int G, int kernel_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = conv_dim * G;
    if (idx >= total) return;

    int g = idx / conv_dim;  // batch (col-major: g is slow index)
    int c = idx % conv_dim;  // channel (col-major: c is fast index)
    int hist = kernel_size - 1;  // 3 for kernel=4

    // Gather: [conv_state[g,c,0], conv_state[g,c,1], conv_state[g,c,2], input[c,g]]
    float acc = 0.0f;
    for (int k = 0; k < hist; k++) {
        float s = conv_state[(g * conv_dim + c) * hist + k];  // fp32 state
        float w = __half2float(weight[c * kernel_size + k]);
        acc += s * w;
    }
    // Col-major: element (c, g) at c + g * conv_dim
    float in_val = __half2float(input[c + g * conv_dim]);
    acc += in_val * __half2float(weight[c * kernel_size + hist]);
    if (bias) acc += __half2float(bias[c]);

    // SiLU activation
    float silu_val = acc / (1.0f + expf(-acc));

    output[c + g * conv_dim] = silu_val;  // fp32, col-major

    // Shift conv state left, push input (fp32)
    for (int k = 0; k < hist - 1; k++) {
        conv_state[(g * conv_dim + c) * hist + k] =
            conv_state[(g * conv_dim + c) * hist + k + 1];
    }
    conv_state[(g * conv_dim + c) * hist + (hist - 1)] = in_val;  // store raw input (fp32)
}

// --- Compute decay from A_log, dt_bias, and per-token 'a' input ---
// decay[h] = exp(-exp(A_log[h]) * softplus(a[h] + dt_bias[h]))
// beta[h] = sigmoid(b[h])
// Grid: ceil(num_v_heads * G / 256), Block: 256
__global__ void ssm_compute_dt_decay_kernel(
    float* __restrict__ decay,       // (num_v_heads, G) col-major
    float* __restrict__ beta,        // (num_v_heads, G) col-major
    const half* __restrict__ a_in,   // (num_v_heads, G) col-major
    const half* __restrict__ b_in,   // (num_v_heads, G) col-major
    const half* __restrict__ A_log,  // (num_v_heads,)
    const half* __restrict__ dt_bias,// (num_v_heads,)
    int num_v_heads, int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_v_heads * G) return;

    // Col-major: element (h, g) at h + g * num_v_heads
    int g = idx / num_v_heads;
    int h = idx % num_v_heads;

    float a_val = __half2float(a_in[h + g * num_v_heads]);
    float b_val = __half2float(b_in[h + g * num_v_heads]);
    float A = expf(__half2float(A_log[h]));
    float dt = __half2float(dt_bias[h]) + a_val;
    dt = logf(1.0f + expf(dt));  // softplus
    float d = expf(-A * dt);     // decay

    decay[h + g * num_v_heads] = d;
    beta[h + g * num_v_heads] = 1.0f / (1.0f + expf(-b_val));  // sigmoid
}

// --- SSM state update + output (Gated Delta Rule, batched decode) ---
// Optimized: fp16 state (halves 128KB→64KB memory traffic per head per step),
// shared memory for k/q (eliminates 32K redundant global reads per head),
// __ldg for read-only pass, __fmaf_rn for fused multiply-add.
//
// Grid: (num_v_heads * G), Block: v_head_dim (128 for Qwen3.5)
// Shared memory: 2 * k_head_dim * sizeof(float) = 1KB
__global__
__launch_bounds__(128, 4)
void ssm_gated_delta_rule_kernel(
    float* __restrict__ state,    // (G, num_v_heads, k_head_dim, v_head_dim) — fp32
    float* __restrict__ y_out,    // (value_dim, G) col-major — fp32
    const float* __restrict__ q,  // (key_dim, G) col-major — fp32
    const float* __restrict__ k,  // (key_dim, G) col-major — fp32
    const float* __restrict__ v,  // (value_dim, G) col-major — fp32
    const float* __restrict__ decay, // (num_v_heads, G)
    const float* __restrict__ beta,  // (num_v_heads, G)
    int num_v_heads, int num_k_heads,
    int k_head_dim, int v_head_dim, int G
) {
    int vh = blockIdx.x / G;
    int g  = blockIdx.x % G;
    int vd = threadIdx.x;

    // Load k[] and q[] into shared memory — all 128 threads share the same vectors
    extern __shared__ float smem[];
    float* k_s = smem;
    float* q_s = smem + k_head_dim;

    int kv_ratio = num_v_heads / num_k_heads;
    int kh = vh / kv_ratio;
    int key_dim = num_k_heads * k_head_dim;
    int value_dim = num_v_heads * v_head_dim;
    int base_kq = kh * k_head_dim + g * key_dim;

    if (vd < k_head_dim) {
        k_s[vd] = k[base_kq + vd];
        q_s[vd] = q[base_kq + vd];
    }
    __syncthreads();

    if (vd >= v_head_dim) return;

    float d = decay[vh + g * num_v_heads];
    float b = beta[vh + g * num_v_heads];
    int state_off = (g * num_v_heads + vh) * k_head_dim * v_head_dim;
    float* sp = state + state_off;

    float v_val = v[(vh * v_head_dim + vd) + g * value_dim];

    // Pass 1: kv_recall = sum_kd(state[kd,vd] * k[kd])  — read-only, use __ldg
    float kv_recall = 0.0f;
    for (int kd = 0; kd < k_head_dim; kd++) {
        float s = __ldg(sp + kd * v_head_dim + vd);
        kv_recall = __fmaf_rn(s, k_s[kd], kv_recall);
    }

    float delta = b * (v_val - kv_recall);

    // Pass 2: state update + output  — read+write state, accumulate y
    float y_val = 0.0f;
    for (int kd = 0; kd < k_head_dim; kd++) {
        float s_old = sp[kd * v_head_dim + vd];
        float s_new = __fmaf_rn(d, s_old, k_s[kd] * delta);
        sp[kd * v_head_dim + vd] = s_new;
        y_val = __fmaf_rn(s_new, q_s[kd], y_val);
    }

    y_val *= rsqrtf((float)k_head_dim);
    y_out[(vh * v_head_dim + vd) + g * value_dim] = y_val;
}

// --- Gated RMSNorm: y = RMSNorm(y) * SiLU(z) ---
// One block per (head, batch). Block: v_head_dim threads.
// Grid: (num_v_heads * G), Block: v_head_dim
__global__ void ssm_gated_rmsnorm_kernel(
    half* __restrict__ y_out,       // (value_dim, G) output — fp16
    const float* __restrict__ y_in, // (value_dim, G) input — fp32 from delta rule
    const half* __restrict__ z,     // (value_dim, G) — gate
    const half* __restrict__ weight, // (v_head_dim,) — shared across heads
    int num_v_heads, int v_head_dim, int G, float eps
) {
    int block_id = blockIdx.x;
    int vh = block_id / G;
    int g = block_id % G;
    if (vh >= num_v_heads) return;

    int d = threadIdx.x;
    if (d >= v_head_dim) return;

    // Col-major: element (dim, g) at dim + g * value_dim
    int value_dim = num_v_heads * v_head_dim;
    int idx = (vh * v_head_dim + d) + g * value_dim;
    float y_val = y_in[idx];

    // Compute RMS (reduce across v_head_dim)
    float sq = y_val * y_val;
    // Warp reduction for RMS
    for (int off = warpSize / 2; off > 0; off /= 2)
        sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
    __shared__ float s_partial[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_partial[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        sq = (lid < (v_head_dim + 31) / 32) ? s_partial[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
        if (lid == 0) s_partial[0] = sq;
    }
    __syncthreads();

    float rms = rsqrtf(s_partial[0] / v_head_dim + eps);
    float w = __half2float(weight[d]);
    float normalized = y_val * rms * w;

    // Gate with SiLU(z) — z is also col-major from GEMM
    float z_val = __half2float(z[idx]);
    float silu_z = z_val / (1.0f + expf(-z_val));

    y_out[idx] = __float2half(normalized * silu_z);
}

// --- Gated RMSNorm (col-major) for prefill ---
// Same as above but uses col-major layout: element (d, t) at d + t * value_dim
// Grid: (num_v_heads * T), Block: v_head_dim
__global__ void ssm_gated_rmsnorm_colmajor_kernel(
    half* __restrict__ y_out,       // (value_dim, T) col-major — fp16
    const float* __restrict__ y_in, // (value_dim, T) col-major — fp32
    const half* __restrict__ z,     // (value_dim, T) col-major — gate
    const half* __restrict__ weight, // (v_head_dim,) — shared across heads
    int num_v_heads, int v_head_dim, int T, float eps
) {
    int block_id = blockIdx.x;
    int vh = block_id / T;
    int t = block_id % T;
    if (vh >= num_v_heads) return;

    int d = threadIdx.x;
    if (d >= v_head_dim) return;

    int value_dim = num_v_heads * v_head_dim;
    // Col-major: element (dim, token) at dim + token * value_dim
    int idx = vh * v_head_dim + d + t * value_dim;
    float y_val = y_in[idx];

    // Compute RMS (reduce across v_head_dim)
    float sq = y_val * y_val;
    for (int off = warpSize / 2; off > 0; off /= 2)
        sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
    __shared__ float s_partial[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_partial[wid] = sq;
    __syncthreads();
    if (wid == 0) {
        sq = (lid < (v_head_dim + 31) / 32) ? s_partial[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            sq += __shfl_down_sync(0xFFFFFFFF, sq, off);
        if (lid == 0) s_partial[0] = sq;
    }
    __syncthreads();

    float rms = rsqrtf(s_partial[0] / v_head_dim + eps);
    float w = __half2float(weight[d]);
    float normalized = y_val * rms * w;

    float z_val = __half2float(z[idx]);
    float silu_z = z_val / (1.0f + expf(-z_val));

    y_out[idx] = __float2half(normalized * silu_z);
}

// --- L2-normalize Q and K per head (required by Gated Delta Rule) ---
// Grid: (num_k_heads * G), Block: k_head_dim
__global__ void ssm_l2norm_qk_kernel(
    float* __restrict__ q,  // (key_dim, G) col-major — fp32, normalized in-place
    float* __restrict__ k,  // (key_dim, G) col-major — fp32
    int num_k_heads, int k_head_dim, int G
) {
    int block_id = blockIdx.x;
    int h = block_id / G;
    int g = block_id % G;
    if (h >= num_k_heads) return;

    int d = threadIdx.x;
    if (d >= k_head_dim) return;

    // Col-major: element (dim, g) at dim + g * key_dim
    int key_dim = num_k_heads * k_head_dim;
    int q_idx = (h * k_head_dim + d) + g * key_dim;
    int k_idx = q_idx;

    float qv = q[q_idx];  // fp32
    float kv = k[k_idx];  // fp32

    // Reduce for Q norm
    float q_sq = qv * qv;
    for (int off = warpSize / 2; off > 0; off /= 2)
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, off);
    __shared__ float s_qnorm, s_knorm;
    __shared__ float s_partial_q[4], s_partial_k[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) s_partial_q[wid] = q_sq;
    __syncthreads();
    if (wid == 0) {
        q_sq = (lid < (k_head_dim + 31) / 32) ? s_partial_q[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, off);
        if (lid == 0) s_qnorm = q_sq;
    }
    __syncthreads();

    // Reduce for K norm
    float k_sq = kv * kv;
    for (int off = warpSize / 2; off > 0; off /= 2)
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, off);
    if (lid == 0) s_partial_k[wid] = k_sq;
    __syncthreads();
    if (wid == 0) {
        k_sq = (lid < (k_head_dim + 31) / 32) ? s_partial_k[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2)
            k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, off);
        if (lid == 0) s_knorm = k_sq;
    }
    __syncthreads();

    float q_inv = (s_qnorm > 1e-6f) ? rsqrtf(s_qnorm) : 0.0f;
    float k_inv = (s_knorm > 1e-6f) ? rsqrtf(s_knorm) : 0.0f;

    q[q_idx] = qv * q_inv;  // fp32
    k[k_idx] = kv * k_inv;  // fp32
}

// --- Expand K heads to V heads (GQA-like repeat for SSM) ---
// When num_v_heads > num_k_heads, repeat each K head (num_v_heads/num_k_heads) times.
// Grid: ceil(key_dim * G / 256), Block: 256
// Output: (value_dim_keys, G) where value_dim_keys = num_v_heads * k_head_dim
__global__ void ssm_expand_kv_heads_kernel(
    half* __restrict__ out,     // (num_v_heads * k_head_dim, G)
    const half* __restrict__ in, // (num_k_heads * k_head_dim, G)
    int num_k_heads, int num_v_heads, int k_head_dim, int G
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_v_heads * k_head_dim * G;
    if (idx >= total) return;

    int g = idx % G;
    int rem = idx / G;
    int vh = rem / k_head_dim;
    int d = rem % k_head_dim;

    int ratio = num_v_heads / num_k_heads;
    int kh = vh / ratio;

    out[idx] = in[(kh * k_head_dim + d) * G + g];
}

// ============================================================================
// SSM Launch Wrappers
// ============================================================================

void launch_ssm_conv1d_decode(
    float* output, float* conv_state, const half* input,
    const half* weight, const half* bias,
    int conv_dim, int G, int kernel_size, cudaStream_t s
) {
    int total = conv_dim * G;
    int block = 256;
    int grid = (total + block - 1) / block;
    ssm_conv1d_decode_kernel<<<grid, block, 0, s>>>(
        output, conv_state, input, weight, bias, conv_dim, G, kernel_size);
}

void launch_ssm_compute_dt_decay(
    float* decay, float* beta,
    const half* a_in, const half* b_in,
    const half* A_log, const half* dt_bias,
    int num_v_heads, int G, cudaStream_t s
) {
    int total = num_v_heads * G;
    int block = 256;
    int grid = (total + block - 1) / block;
    ssm_compute_dt_decay_kernel<<<grid, block, 0, s>>>(
        decay, beta, a_in, b_in, A_log, dt_bias, num_v_heads, G);
}

void launch_ssm_gated_delta_rule(
    float* state, float* y_out,
    const float* q, const float* k, const float* v,
    const float* decay, const float* beta,
    int num_v_heads, int num_k_heads,
    int k_head_dim, int v_head_dim, int G, cudaStream_t s
) {
    int grid = num_v_heads * G;
    int smem = 2 * k_head_dim * sizeof(float);  // shared k[] + q[]
    ssm_gated_delta_rule_kernel<<<grid, v_head_dim, smem, s>>>(
        state, y_out, q, k, v, decay, beta,
        num_v_heads, num_k_heads, k_head_dim, v_head_dim, G);
}

void launch_ssm_gated_rmsnorm(
    half* y_out, const float* y_in, const half* z, const half* weight,
    int num_v_heads, int v_head_dim, int G, float eps, cudaStream_t s
) {
    int grid = num_v_heads * G;
    ssm_gated_rmsnorm_kernel<<<grid, v_head_dim, 0, s>>>(
        y_out, y_in, z, weight, num_v_heads, v_head_dim, G, eps);
}

void launch_ssm_gated_rmsnorm_colmajor(
    half* y_out, const float* y_in, const half* z, const half* weight,
    int num_v_heads, int v_head_dim, int T, float eps, cudaStream_t s
) {
    int grid = num_v_heads * T;
    ssm_gated_rmsnorm_colmajor_kernel<<<grid, v_head_dim, 0, s>>>(
        y_out, y_in, z, weight, num_v_heads, v_head_dim, T, eps);
}

void launch_ssm_l2norm_qk(
    float* q, float* k, int num_k_heads, int k_head_dim, int G, cudaStream_t s
) {
    int grid = num_k_heads * G;
    ssm_l2norm_qk_kernel<<<grid, k_head_dim, 0, s>>>(
        q, k, num_k_heads, k_head_dim, G);
}

void launch_ssm_expand_kv_heads(
    half* out, const half* in,
    int num_k_heads, int num_v_heads, int k_head_dim, int G, cudaStream_t s
) {
    int total = num_v_heads * k_head_dim * G;
    int block = 256;
    int grid = (total + block - 1) / block;
    ssm_expand_kv_heads_kernel<<<grid, block, 0, s>>>(
        out, in, num_k_heads, num_v_heads, k_head_dim, G);
}

// ============================================================================
// Chunked Prefill Kernels for Gated Delta Net
// ============================================================================

// --- Causal Conv1d for prefill (all T tokens in parallel) ---
// Each thread processes one channel, loops over T tokens.
// Produces fp32 output with SiLU activation.
// Also stores last (K-1) tokens as conv_state for future decode.
// Grid: ceil(conv_dim / 256), Block: 256
__global__ void ssm_causal_conv1d_prefill_kernel(
    float* __restrict__ output,      // (conv_dim, T) — fp32, same layout as input
    float* __restrict__ conv_state,  // (1, conv_dim, K-1) — fp32, for future decode
    const half* __restrict__ input,  // (conv_dim, T) — fp16 from GEMM
    const half* __restrict__ weight, // (conv_dim, K)
    const half* __restrict__ bias,   // (conv_dim,) or nullptr
    int conv_dim, int T, int K       // K = kernel_size (typically 4)
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;

    // Load weights for this channel
    float w[8];  // max kernel size
    for (int k = 0; k < K; k++)
        w[k] = __half2float(weight[c * K + k]);
    float b = bias ? __half2float(bias[c]) : 0.0f;

    // Process each token with causal padding (zeros for t < 0)
    // PyTorch conv1d convention: w[K-1] applies to current input, w[0] to oldest
    // output[t] = sum_{k=0}^{K-1} w[k] * input[t + k - (K-1)]
    // Layout: col-major (conv_dim, T) → element (c, t) at c + t * conv_dim
    for (int t = 0; t < T; t++) {
        float acc = b;
        for (int k = 0; k < K; k++) {
            int src_t = t + k - (K - 1);  // causal: oldest first
            float x = (src_t >= 0) ? __half2float(input[c + src_t * conv_dim]) : 0.0f;
            acc += x * w[k];
        }
        // SiLU activation
        output[c + t * conv_dim] = acc / (1.0f + expf(-acc));
    }

    // Store last (K-1) tokens as conv state (raw values, no activation)
    int hist = K - 1;
    for (int k = 0; k < hist; k++) {
        int t = T - hist + k;
        float x = (t >= 0) ? __half2float(input[c + t * conv_dim]) : 0.0f;
        conv_state[c * hist + k] = x;  // (1, conv_dim, K-1) → c * hist + k for seq 0
    }
}

// --- Compute raw gate g (NOT exp(g)) and beta for chunked prefill ---
// g[h,t] = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
// beta[h,t] = sigmoid(b[h,t])
// Grid: ceil(num_v_heads * T / 256), Block: 256
__global__ void ssm_compute_g_beta_kernel(
    float* __restrict__ g_out,       // (num_v_heads, T) — raw gate values (negative)
    float* __restrict__ beta_out,    // (num_v_heads, T)
    const half* __restrict__ a_in,   // (num_v_heads, T)
    const half* __restrict__ b_in,   // (num_v_heads, T)
    const half* __restrict__ A_log,  // (num_v_heads,)
    const half* __restrict__ dt_bias,// (num_v_heads,)
    int num_v_heads, int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_v_heads * T) return;

    int h = idx / T;
    int t = idx % T;

    // Input from cuBLAS is col-major: element (h, t) at h + t * num_v_heads
    float a_val = __half2float(a_in[h + t * num_v_heads]);
    float b_val = __half2float(b_in[h + t * num_v_heads]);
    float A = expf(__half2float(A_log[h]));
    float dt = __half2float(dt_bias[h]) + a_val;
    dt = logf(1.0f + expf(dt));  // softplus
    float g = -A * dt;           // negative gate value

    // Output in (H, T) row-major for internal use by rearrange/delta kernels
    g_out[h * T + t] = g;
    beta_out[h * T + t] = 1.0f / (1.0f + expf(-b_val));  // sigmoid
}

// --- Rearrange QKV from (dim, T) col-major to (H, T_padded, D) contiguous ---
// Also applies L2 normalization to Q and K, and scales Q by 1/sqrt(D).
// Computes k_beta = K * beta and v_beta = V * beta.
// Grid: (num_heads, T_padded), Block: head_dim
__global__ void ssm_chunk_rearrange_kernel(
    float* __restrict__ Q_out,       // (H, T_padded, D) — L2-normed, scaled by 1/sqrt(D)
    float* __restrict__ K_out,       // (H, T_padded, D) — L2-normed
    float* __restrict__ V_out,       // (H, T_padded, D)
    float* __restrict__ K_beta_out,  // (H, T_padded, D) — K * beta
    float* __restrict__ V_beta_out,  // (H, T_padded, D) — V * beta
    const float* __restrict__ qkv,   // (conv_dim, T) col-major — fp32 from conv1d
    const float* __restrict__ beta,  // (H, T) row-major — from g_beta kernel
    int num_heads, int head_dim, int T, int T_padded,
    int key_dim, int value_dim, int conv_dim
) {
    int h = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    if (h >= num_heads || d >= head_dim) return;

    int out_idx = (h * T_padded + t) * head_dim + d;

    if (t >= T) {
        // Padding: zero everything
        Q_out[out_idx] = 0.0f;
        K_out[out_idx] = 0.0f;
        V_out[out_idx] = 0.0f;
        K_beta_out[out_idx] = 0.0f;
        V_beta_out[out_idx] = 0.0f;
        return;
    }

    // Read Q, K, V from col-major (conv_dim, T) layout
    // element (channel, token) at: channel + token * conv_dim
    int q_ch = h * head_dim + d;
    int k_ch = key_dim + h * head_dim + d;
    int v_ch = 2 * key_dim + h * head_dim + d;
    float q_val = qkv[q_ch + t * conv_dim];
    float k_val = qkv[k_ch + t * conv_dim];
    float v_val = qkv[v_ch + t * conv_dim];

    // L2 norm reduction for Q and K (across head_dim)
    float q_sq = q_val * q_val;
    float k_sq = k_val * k_val;

    // Warp reduction
    for (int off = warpSize / 2; off > 0; off /= 2) {
        q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, off);
        k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, off);
    }

    // Cross-warp reduction via shared memory
    __shared__ float s_q_partial[4], s_k_partial[4];
    int wid = d / warpSize, lid = d % warpSize;
    if (lid == 0) { s_q_partial[wid] = q_sq; s_k_partial[wid] = k_sq; }
    __syncthreads();
    if (wid == 0) {
        q_sq = (lid < (head_dim + 31) / 32) ? s_q_partial[lid] : 0.0f;
        k_sq = (lid < (head_dim + 31) / 32) ? s_k_partial[lid] : 0.0f;
        for (int off = warpSize / 2; off > 0; off /= 2) {
            q_sq += __shfl_down_sync(0xFFFFFFFF, q_sq, off);
            k_sq += __shfl_down_sync(0xFFFFFFFF, k_sq, off);
        }
        if (lid == 0) { s_q_partial[0] = q_sq; s_k_partial[0] = k_sq; }
    }
    __syncthreads();

    float q_inv = (s_q_partial[0] > 1e-12f) ? rsqrtf(s_q_partial[0]) : 0.0f;
    float k_inv = (s_k_partial[0] > 1e-12f) ? rsqrtf(s_k_partial[0]) : 0.0f;

    q_val *= q_inv;
    k_val *= k_inv;

    float scale = rsqrtf((float)head_dim);
    float b = beta[h * T + t];

    Q_out[out_idx] = q_val * scale;
    K_out[out_idx] = k_val;
    V_out[out_idx] = v_val;
    K_beta_out[out_idx] = k_val * b;
    V_beta_out[out_idx] = v_val * b;
}

// --- Chunked Delta Rule: core computation for one chunk ---
// Processes one chunk for one head. Called in a loop over chunks from C++.
// Computes:
//   1. Decay mask from gate values
//   2. Correction matrix (k_beta @ K^T * decay_mask) with sequential fixup
//   3. V_corrected = (correction + I) @ v_beta
//   4. Intra-chunk attention
//   5. Inter-chunk state query
//   6. Combined output
//   7. State update
//
// Grid: (num_heads), Block: (head_dim) — 128 threads per head
// Each block processes one head for the given chunk.
__global__ void ssm_chunked_delta_rule_kernel(
    float* __restrict__ state,       // (H, D_k, D_v) — read/write recurrent state
    float* __restrict__ output,      // (H, T_padded, D_v) — write output for this chunk
    const float* __restrict__ Q,     // (H, T_padded, D_k) — L2-normed, scaled
    const float* __restrict__ K,     // (H, T_padded, D_k) — L2-normed
    const float* __restrict__ V,     // (H, T_padded, D_v)
    const float* __restrict__ K_beta,// (H, T_padded, D_k) — K * beta
    const float* __restrict__ V_beta,// (H, T_padded, D_v) — V * beta
    const float* __restrict__ g,     // (H, T) — raw gate values (negative)
    float* __restrict__ workspace,   // (H, cs*cs + cs*D + cs*D + cs*D + cs) scratch
    int num_heads, int head_dim, int T, int T_padded,
    int chunk_idx, int chunk_size, int T_actual  // T_actual = real seq len before padding
) {
    int h = blockIdx.x;
    if (h >= num_heads) return;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int cs = chunk_size;
    int chunk_start = chunk_idx * cs;

    // Workspace layout per head:
    // [0..cs*cs): correction matrix (cs x cs)
    // [cs*cs..cs*cs+cs*D): V_corrected (cs x D)
    // [cs*cs+cs*D..cs*cs+2*cs*D): k_cumdecay (cs x D)
    // [cs*cs+2*cs*D..cs*cs+3*cs*D): V_new (cs x D)
    // [cs*cs+3*cs*D..cs*cs+3*cs*D+cs): g_cumsum (cs)
    int ws_per_head = cs * cs + 3 * cs * head_dim + cs;
    float* ws = workspace + h * ws_per_head;
    float* correction = ws;                              // [cs, cs]
    float* V_corr     = ws + cs * cs;                    // [cs, D]
    float* k_cumdecay = ws + cs * cs + cs * head_dim;    // [cs, D]
    float* V_new      = ws + cs * cs + 2 * cs * head_dim;// [cs, D]
    float* g_cumsum   = ws + cs * cs + 3 * cs * head_dim;// [cs]

    // Offsets into the (H, T_padded, D) arrays for this head's chunk
    int base = h * T_padded * head_dim + chunk_start * head_dim;

    // ====================================================================
    // Step 1: Compute g_cumsum within this chunk
    // Thread 0 does this sequentially (it's only cs=64 values)
    // ====================================================================
    if (d == 0) {
        float cumsum = 0.0f;
        for (int i = 0; i < cs; i++) {
            int t = chunk_start + i;
            float gv = (t < T_actual) ? g[h * T + t] : 0.0f;
            cumsum += gv;
            g_cumsum[i] = cumsum;
        }
    }
    __syncthreads();

    // ====================================================================
    // Step 2: Compute correction matrix
    // correction[i][j] = -(K_beta[i] . K[j]) * exp(g_cumsum[i] - g_cumsum[j])
    // for i >= j, else 0
    // Then sequential fixup, then add identity
    //
    // Each thread d computes partial dot products, then we reduce.
    // We process one (i,j) pair at a time with all threads contributing.
    // ====================================================================

    // Compute strictly lower triangular: correction[i][j] for j < i only.
    // Diagonal and upper triangle are zero (HF masks with triu(diagonal=0)).
    for (int i = 0; i < cs; i++) {
        for (int j = 0; j < i; j++) {  // strictly j < i
            // Dot product: K_beta[chunk_start+i] . K[chunk_start+j]
            float kb_val = K_beta[base + i * head_dim + d];
            float k_val  = K[base + j * head_dim + d];
            float partial = kb_val * k_val;

            // Warp reduction
            for (int off = warpSize / 2; off > 0; off /= 2)
                partial += __shfl_down_sync(0xFFFFFFFF, partial, off);

            // Cross-warp reduction
            __shared__ float s_reduce[4];
            int wid = d / warpSize, lid = d % warpSize;
            if (lid == 0) s_reduce[wid] = partial;
            __syncthreads();
            if (wid == 0) {
                partial = (lid < (head_dim + 31) / 32) ? s_reduce[lid] : 0.0f;
                for (int off = warpSize / 2; off > 0; off /= 2)
                    partial += __shfl_down_sync(0xFFFFFFFF, partial, off);
            }

            if (d == 0) {
                float decay = expf(g_cumsum[i] - g_cumsum[j]);
                correction[i * cs + j] = -partial * decay;
            }
            __syncthreads();
        }
        // Zero diagonal and upper triangle
        if (d == 0) {
            for (int j = i; j < cs; j++)  // includes diagonal
                correction[i * cs + j] = 0.0f;
        }
    }
    __syncthreads();

    // ====================================================================
    // Step 3: Sequential fixup (thread 0 only, cs iterations)
    // for i in 1..cs-1:
    //   for j in 0..i-1:
    //     correction[i][j] += sum_m(correction[i][m] * correction[m][j]) for m in 0..i-1
    // Then add identity
    // ====================================================================
    if (d == 0) {
        for (int i = 1; i < cs; i++) {
            for (int j = 0; j < i; j++) {
                float fix = 0.0f;
                for (int m = j; m < i; m++) {
                    fix += correction[i * cs + m] * correction[m * cs + j];
                }
                correction[i * cs + j] += fix;
            }
        }
        // Add identity
        for (int i = 0; i < cs; i++) {
            correction[i * cs + i] += 1.0f;
        }
    }
    __syncthreads();

    // ====================================================================
    // Step 4: V_corrected = correction @ V_beta  (cs x cs) @ (cs x D) → (cs x D)
    // Each thread d computes one column of the result
    // ====================================================================
    for (int i = 0; i < cs; i++) {
        float sum = 0.0f;
        for (int j = 0; j <= i; j++) {  // correction is lower triangular + identity
            sum += correction[i * cs + j] * V_beta[base + j * head_dim + d];
        }
        V_corr[i * head_dim + d] = sum;
    }
    __syncthreads();

    // ====================================================================
    // Step 5: k_cumdecay = correction @ (K_beta * exp(g_cumsum))
    // Each thread d computes one column
    // ====================================================================
    for (int i = 0; i < cs; i++) {
        float sum = 0.0f;
        for (int j = 0; j <= i; j++) {
            float kb_scaled = K_beta[base + j * head_dim + d] * expf(g_cumsum[j]);
            sum += correction[i * cs + j] * kb_scaled;
        }
        k_cumdecay[i * head_dim + d] = sum;
    }
    __syncthreads();

    // ====================================================================
    // Step 6: Inter-chunk state query: v_prime = k_cumdecay @ S
    // V_new = V_corrected - v_prime
    // Each thread computes v_prime[i][d] = sum_kd(k_cumdecay[i][kd] * S[kd][d])
    // This requires reading along the K dimension (all threads read different S columns)
    // ====================================================================
    int state_base = h * head_dim * head_dim;
    for (int i = 0; i < cs; i++) {
        float v_prime = 0.0f;
        for (int kd = 0; kd < head_dim; kd++) {
            v_prime += k_cumdecay[i * head_dim + kd] * state[state_base + kd * head_dim + d];
        }
        V_new[i * head_dim + d] = V_corr[i * head_dim + d] - v_prime;
    }
    __syncthreads();

    // ====================================================================
    // Step 7: Intra-chunk causal attention: Q @ K^T * decay_mask (strictly causal)
    // Then: output[i] = inter[i] + sum_j(intra[i][j] * V_new[j])
    //
    // inter[i][d] = (Q[i] * exp(g_cumsum[i])) @ S[:, d]
    //             = sum_kd(Q[i][kd] * exp(g_cumsum[i]) * S[kd][d])
    //
    // intra[i][j] = (Q[i] . K[j]) * exp(g_cumsum[i] - g_cumsum[j])  for j < i
    //             = (Q[i] . K[i]) * 1.0  for j == i
    // (strictly lower triangular + diagonal for causal mask)
    // The upper triangle is zero (can't attend to future tokens)
    //
    // To avoid storing the full cs x cs attention matrix, we compute
    // output[i] incrementally for each thread d:
    //   output[i][d] = inter[i][d] + sum_{j<=i}(intra[i][j] * V_new[j][d])
    //
    // But intra[i][j] requires a dot product across head_dim... which needs reduction.
    // This is the most complex part. Let's compute it row-by-row.
    // ====================================================================

    for (int i = 0; i < cs; i++) {
        // Inter-chunk contribution: Q[i] * exp(g_cumsum[i]) @ S
        float inter = 0.0f;
        float q_decay = Q[base + i * head_dim + d] * expf(g_cumsum[i]);
        // Wait — Q is (H, T_padded, D) and we need Q[h, chunk_start+i, :] @ S[h, :, d]
        // But thread d only has Q[h, chunk_start+i, d] — we need the full Q vector for this row.
        // Actually, inter[i][d] = sum_kd(Q[i][kd] * exp(g_cumsum[i]) * S[kd][d])
        // Thread d computes the d-th output dimension.
        // It needs Q[i][kd] for all kd, and S[kd][d] for all kd.
        // S[kd][d] can be loaded in a loop over kd.
        // Q[i][kd] needs shared memory or global loads.

        for (int kd = 0; kd < head_dim; kd++) {
            float q_kd = Q[base + i * head_dim + kd] * expf(g_cumsum[i]);
            inter += q_kd * state[state_base + kd * head_dim + d];
        }

        // Intra-chunk attention: need Q[i] . K[j] for each j <= i
        // This is a dot product that requires reduction across d.
        // We compute the dot products one at a time, reducing across threads.
        float intra_sum = 0.0f;
        for (int j = 0; j <= i; j++) {
            // Compute Q[i] . K[j] via reduction
            float dot = Q[base + i * head_dim + d] * K[base + j * head_dim + d];

            // Warp reduction for dot product
            for (int off = warpSize / 2; off > 0; off /= 2)
                dot += __shfl_down_sync(0xFFFFFFFF, dot, off);

            __shared__ float s_dot[4];
            int wid = d / warpSize, lid = d % warpSize;
            if (lid == 0) s_dot[wid] = dot;
            __syncthreads();
            if (wid == 0) {
                dot = (lid < (head_dim + 31) / 32) ? s_dot[lid] : 0.0f;
                for (int off = warpSize / 2; off > 0; off /= 2)
                    dot += __shfl_down_sync(0xFFFFFFFF, dot, off);
                if (lid == 0) s_dot[0] = dot;
            }
            __syncthreads();

            float attn_ij = s_dot[0];
            // Apply causal decay mask (upper triangle already excluded by j <= i)
            if (j < i) {
                attn_ij *= expf(g_cumsum[i] - g_cumsum[j]);
            }
            // j == i: decay factor is exp(0) = 1, already correct

            intra_sum += attn_ij * V_new[j * head_dim + d];
        }

        int t = chunk_start + i;
        if (t < T_actual) {
            output[h * T_padded * head_dim + t * head_dim + d] = inter + intra_sum;
        }
    }
    __syncthreads();

    // ====================================================================
    // Step 8: Update recurrent state
    // S = S * exp(g_cumsum[-1]) + K_decay^T @ V_new
    // where K_decay[j][d] = K[j][d] * exp(g_cumsum[-1] - g_cumsum[j])
    //
    // Thread d handles column d of S.
    // S[kd][d] = exp(g_cumsum[-1]) * S[kd][d]
    //          + sum_j(K[j][kd] * exp(g_cumsum[-1] - g_cumsum[j]) * V_new[j][d])
    // ====================================================================
    float g_end = g_cumsum[cs - 1];
    float decay_total = expf(g_end);

    for (int kd = 0; kd < head_dim; kd++) {
        float s_val = decay_total * state[state_base + kd * head_dim + d];

        float update = 0.0f;
        for (int j = 0; j < cs && (chunk_start + j) < T_actual; j++) {
            float k_decay = K[base + j * head_dim + kd] * expf(g_end - g_cumsum[j]);
            update += k_decay * V_new[j * head_dim + d];
        }

        state[state_base + kd * head_dim + d] = s_val + update;
    }
}

// --- Rearrange output back from (H, T, D) to (value_dim, T) col-major ---
// Grid: ceil(value_dim * T / 256), Block: 256
__global__ void ssm_chunk_output_rearrange_kernel(
    float* __restrict__ y_out,        // (value_dim, T) col-major — fp32
    const float* __restrict__ output, // (H, T_padded, D) — fp32
    int num_heads, int head_dim, int T, int T_padded
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_heads * head_dim * T;
    if (idx >= total) return;

    int t = idx % T;
    int rem = idx / T;
    int h = rem / head_dim;
    int d = rem % head_dim;

    // Write col-major: element (dim, token) at dim + token * value_dim
    int value_dim = num_heads * head_dim;
    y_out[(h * head_dim + d) + t * value_dim] = output[h * T_padded * head_dim + t * head_dim + d];
}


// ============================================================================
// Chunked Prefill Launch Wrappers
// ============================================================================

void launch_ssm_causal_conv1d_prefill(
    float* output, float* conv_state, const half* input,
    const half* weight, const half* bias,
    int conv_dim, int T, int kernel_size, cudaStream_t s
) {
    int block = 256;
    int grid = (conv_dim + block - 1) / block;
    ssm_causal_conv1d_prefill_kernel<<<grid, block, 0, s>>>(
        output, conv_state, input, weight, bias, conv_dim, T, kernel_size);
}

void launch_ssm_compute_g_beta(
    float* g_out, float* beta_out,
    const half* a_in, const half* b_in,
    const half* A_log, const half* dt_bias,
    int num_v_heads, int T, cudaStream_t s
) {
    int total = num_v_heads * T;
    int block = 256;
    int grid = (total + block - 1) / block;
    ssm_compute_g_beta_kernel<<<grid, block, 0, s>>>(
        g_out, beta_out, a_in, b_in, A_log, dt_bias, num_v_heads, T);
}

void launch_ssm_chunk_rearrange(
    float* Q_out, float* K_out, float* V_out,
    float* K_beta_out, float* V_beta_out,
    const float* qkv, const float* beta,
    int num_heads, int head_dim, int T, int T_padded,
    int key_dim, int value_dim, int conv_dim, cudaStream_t s
) {
    dim3 grid(num_heads, T_padded);
    int block = head_dim;
    ssm_chunk_rearrange_kernel<<<grid, block, 0, s>>>(
        Q_out, K_out, V_out, K_beta_out, V_beta_out,
        qkv, beta, num_heads, head_dim, T, T_padded, key_dim, value_dim, conv_dim);
}

void launch_ssm_chunked_delta_rule(
    float* state, float* output,
    const float* Q, const float* K, const float* V,
    const float* K_beta, const float* V_beta,
    const float* g, float* workspace,
    int num_heads, int head_dim, int T, int T_padded,
    int chunk_idx, int chunk_size, int T_actual, cudaStream_t s
) {
    int block = head_dim;  // 128 threads per block
    int grid = num_heads;
    ssm_chunked_delta_rule_kernel<<<grid, block, 0, s>>>(
        state, output, Q, K, V, K_beta, V_beta, g, workspace,
        num_heads, head_dim, T, T_padded, chunk_idx, chunk_size, T_actual);
}

// --- Sigmoid gate: data *= sigmoid(gate), both (dim, G) col-major fp16 ---
__global__ void sigmoid_gate_batch_kernel(
    half* __restrict__ data, const half* __restrict__ gate, int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float d = __half2float(data[i]);
    float g = __half2float(gate[i]);
    data[i] = __float2half(d / (1.0f + expf(-g)));
}

void launch_sigmoid_gate_batch(half* data, const half* gate, int dim, int G, cudaStream_t s) {
    int total = dim * G;
    sigmoid_gate_batch_kernel<<<(total + 255) / 256, 256, 0, s>>>(data, gate, total);
}

void launch_ssm_chunk_output_rearrange(
    float* y_out, const float* output,
    int num_heads, int head_dim, int T, int T_padded, cudaStream_t s
) {
    int total = num_heads * head_dim * T;
    int block = 256;
    int grid = (total + block - 1) / block;
    ssm_chunk_output_rearrange_kernel<<<grid, block, 0, s>>>(
        y_out, output, num_heads, head_dim, T, T_padded);
}

} // extern "C"
