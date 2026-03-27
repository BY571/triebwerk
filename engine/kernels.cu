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
//
// Each thread block computes one output element.
// Threads in the block cooperatively dequantize and dot-product.

__global__ void nf4_gemv_kernel(
    const uint8_t* __restrict__ weight_data,    // NF4 packed (2 values per byte)
    const float* __restrict__ absmax,            // per-block scales (float32, pre-dequantized)
    const float* __restrict__ quant_map,         // NF4 lookup (16 entries)
    const half* __restrict__ input,              // (in_dim,) fp16
    half* __restrict__ output,                   // (out_dim,) fp16
    int in_dim,
    int out_dim,
    int block_size                               // quantization block size (64)
) {
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    // Each output element = dot(weight_row[out_idx], input)
    // Flat layout: element [out_idx, j] is at flat index out_idx * in_dim + j
    // Packed: flat_idx / 2 = byte, hi nibble first, lo nibble second

    int total_per_row = in_dim;
    int row_start = out_idx * total_per_row;
    int num_blocks_per_row = total_per_row / block_size;

    float sum = 0.0f;

    // Each thread handles some blocks of the dot product
    for (int b = threadIdx.x; b < num_blocks_per_row; b += blockDim.x) {
        int flat_block_idx = out_idx * num_blocks_per_row + b;
        float scale = absmax[flat_block_idx];
        int col_start = b * block_size;

        float local_sum = 0.0f;
        for (int j = col_start; j < col_start + block_size; j += 2) {
            int flat_idx = row_start + j;
            int byte_idx = flat_idx / 2;
            uint8_t packed = weight_data[byte_idx];
            // BNB nibble order: hi first, lo second
            uint8_t first_nib = (packed >> 4) & 0x0F;
            uint8_t second_nib = packed & 0x0F;

            float w0 = quant_map[first_nib] * scale;
            float w1 = quant_map[second_nib] * scale;
            float x0 = __half2float(input[j]);
            float x1 = __half2float(input[j + 1]);

            local_sum += w0 * x0 + w1 * x1;
        }
        sum += local_sum;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Block-level reduction (first warp collects)
    __shared__ float shared_sum[32]; // max 32 warps
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        shared_sum[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (lane_id == 0) {
            output[out_idx] = __float2half(sum);
        }
    }
}

// With LoRA: output = (base_weight @ input) + scale * (B @ (A @ input))
__global__ void nf4_gemv_lora_kernel(
    const uint8_t* __restrict__ weight_data,
    const half* __restrict__ absmax,
    const half* __restrict__ input,
    const half* __restrict__ lora_A,    // (rank, in_dim)
    const half* __restrict__ lora_B,    // (out_dim, rank)
    half* __restrict__ output,
    half* __restrict__ lora_scratch,    // (rank,) intermediate
    int in_dim,
    int out_dim,
    int block_size,
    int rank,
    float lora_scale
) {
    // First compute base: same as nf4_gemv_kernel
    int out_idx = blockIdx.x;
    if (out_idx >= out_dim) return;

    const uint8_t* row = weight_data + (size_t)out_idx * (in_dim / 2);
    int num_blocks = (in_dim + block_size - 1) / block_size;

    float sum = 0.0f;
    for (int b = threadIdx.x; b < num_blocks; b += blockDim.x) {
        float scale = __half2float(absmax[out_idx * num_blocks + b]);
        int start = b * block_size;
        int end = min(start + block_size, in_dim);

        float local_sum = 0.0f;
        for (int j = start; j < end; j += 2) {
            int byte_idx = j / 2;
            uint8_t packed = row[byte_idx];
            float w0 = NF4_LOOKUP[packed & 0x0F] * scale;
            float w1 = NF4_LOOKUP[(packed >> 4) & 0x0F] * scale;
            local_sum += w0 * __half2float(input[j]);
            if (j + 1 < end) local_sum += w1 * __half2float(input[j + 1]);
        }
        sum += local_sum;
    }

    // Reduce sum (same as above)
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    __shared__ float shared_sum[32];
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;
    if (lane_id == 0) shared_sum[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x + warpSize - 1) / warpSize) ? shared_sum[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

        if (lane_id == 0) {
            // Add LoRA contribution: scale * B[out_idx, :] @ lora_scratch
            // lora_scratch was pre-computed as A @ input
            float lora_sum = 0.0f;
            for (int r = 0; r < rank; r++) {
                lora_sum += __half2float(lora_B[out_idx * rank + r]) *
                            __half2float(lora_scratch[r]);
            }
            output[out_idx] = __float2half(sum + lora_scale * lora_sum);
        }
    }
}

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

// Fused: copy + RMSNorm + q8 quantization in one kernel (saves 1 launch + 2KB round-trip)
__global__ void copy_rms_norm_q8_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    half* __restrict__ residual,
    half* __restrict__ norm_out,
    int8_t* __restrict__ q8_data,     // output: quantized norm_out
    float* __restrict__ q8_scales,    // output: per-block scale
    float* __restrict__ q8_sums,      // output: per-block sum
    int dim, float eps
) {
    // Pass 1: copy + sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        half val_h = input[i];
        residual[i] = val_h;
        float val = __half2float(val_h);
        sum_sq += val * val;
    }

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

    // Pass 2: normalize, write fp16, AND quantize to int8
    // Process in 64-element blocks for q8 quantization
    const int block_size = 64;
    int n_blocks = dim / block_size;

    for (int blk = 0; blk < n_blocks; blk++) {
        int base = blk * block_size;

        // Find max abs in this q8 block (all threads participate)
        float my_max = 0.0f;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float val = __half2float(input[base + i]) * rms * __half2float(weight[base + i]);
            my_max = fmaxf(my_max, fabsf(val));
        }
        // Quick warp reduce for max (single block, all threads)
        for (int off = warpSize / 2; off > 0; off /= 2)
            my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, off));
        if (lane_id == 0) shared[warp_id] = my_max;
        __syncthreads();
        if (warp_id == 0) {
            my_max = (lane_id < (blockDim.x + 31) / 32) ? shared[lane_id] : 0.0f;
            for (int off = warpSize / 2; off > 0; off /= 2)
                my_max = fmaxf(my_max, __shfl_down_sync(0xFFFFFFFF, my_max, off));
            if (lane_id == 0) shared[0] = my_max;
        }
        __syncthreads();

        float scale = shared[0] / 127.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        float inv_scale = 1.0f / scale;

        // Normalize, write fp16, quantize to int8, compute sum
        int my_sum = 0;
        for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
            float val = __half2float(input[base + i]) * rms * __half2float(weight[base + i]);
            norm_out[base + i] = __float2half(val);
            int q = __float2int_rn(val * inv_scale);
            q = max(-127, min(127, q));
            q8_data[base + i] = (int8_t)q;
            my_sum += q;
        }

        // Reduce sum for this block
        for (int off = warpSize / 2; off > 0; off /= 2)
            my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, off);
        if (lane_id == 0) shared[warp_id] = (float)my_sum;
        __syncthreads();
        if (warp_id == 0) {
            float s = (lane_id < (blockDim.x + 31) / 32) ? shared[lane_id] : 0.0f;
            for (int off = warpSize / 2; off > 0; off /= 2)
                s += __shfl_down_sync(0xFFFFFFFF, s, off);
            if (lane_id == 0) {
                q8_scales[blk] = scale;
                q8_sums[blk] = s;
            }
        }
        __syncthreads();
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

// RoPE that reads position from device memory (for CUDA graph)
__global__ void rope_device_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    const half* __restrict__ cos_table_base,  // full table: (max_seq, HEAD_DIM/2)
    const half* __restrict__ sin_table_base,
    const int* __restrict__ d_pos,            // device-side position
    int num_q_heads,
    int num_kv_heads,
    int head_dim
) {
    int pos = *d_pos;
    int half_head = head_dim / 2;
    const half* cos_table = cos_table_base + pos * half_head;
    const half* sin_table = sin_table_base + pos * half_head;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = num_q_heads * half_head;

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

    int total_kv = num_kv_heads * half_head;
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

// Version that reads token_id from device memory (for CUDA graph capture)
__global__ void embedding_lookup_device_kernel(
    const half* __restrict__ embed_table,
    const int* __restrict__ d_token_id,     // device-side token id
    half* __restrict__ output,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_dim) {
        output[idx] = embed_table[(size_t)(*d_token_id) * hidden_dim + idx];
    }
}

// ============================================================================
// Kernel 6b: KV cache write (reads position from device memory for CUDA graph)
// ============================================================================

__global__ void kv_cache_write_kernel(
    half* __restrict__ k_cache,     // (max_seq, KV_DIM)
    half* __restrict__ v_cache,     // (max_seq, KV_DIM)
    const half* __restrict__ k_new, // (KV_DIM,)
    const half* __restrict__ v_new, // (KV_DIM,)
    const int* __restrict__ d_pos,
    int kv_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = *d_pos;
    if (idx < kv_dim) {
        k_cache[pos * kv_dim + idx] = k_new[idx];
        v_cache[pos * kv_dim + idx] = v_new[idx];
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

    // Step 4: cumulative sum sampling (thread 0 only, sequential but on GPU)
    if (threadIdx.x == 0) {
        float threshold = random_val;

        if (top_p < 1.0f) {
            // Simple top-p: scan from highest probability
            // For speed, we just do a linear scan (vocab is large but on GPU cache)
            threshold *= top_p;  // scale threshold to top_p region
        }

        float cum = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cum += logits[i];
            if (cum >= threshold) {
                *result = i;
                return;
            }
        }
        *result = vocab_size - 1;
    }
}

// ============================================================================
// Host-side launcher functions
// ============================================================================

extern "C" {

void launch_nf4_gemv(
    const uint8_t* packed,
    const float* absmax,
    const float* quant_map,
    const half* input,
    half* output,
    int out_dim,
    int in_dim,
    int block_size,
    cudaStream_t stream
) {
    nf4_gemv_kernel<<<out_dim, 128, 0, stream>>>(
        packed, absmax, quant_map, input, output,
        in_dim, out_dim, block_size
    );
}

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

void launch_copy_rms_norm_q8(
    const half* input, const half* weight,
    half* residual, half* norm_out,
    int8_t* q8_data, float* q8_scales, float* q8_sums,
    int dim, float eps, cudaStream_t stream
) {
    copy_rms_norm_q8_kernel<<<1, 256, 0, stream>>>(
        input, weight, residual, norm_out, q8_data, q8_scales, q8_sums, dim, eps);
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

void launch_embedding_device(
    const half* embed_table, const int* d_token_id,
    half* output, int hidden_dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;
    embedding_lookup_device_kernel<<<blocks, threads, 0, stream>>>(
        embed_table, d_token_id, output, hidden_dim
    );
}

void launch_rope_device(
    half* q, half* k,
    const half* cos_table_base, const half* sin_table_base,
    const int* d_pos,
    int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream
) {
    int n = num_heads * head_dim / 2;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    rope_device_kernel<<<blocks, threads, 0, stream>>>(
        q, k, cos_table_base, sin_table_base, d_pos,
        num_heads, num_kv_heads, head_dim
    );
}

void launch_gqa_attention_device(
    const half* q, const half* k_cache, const half* v_cache,
    half* output, float* attn_scratch, const int* d_pos,
    int max_seq_len,
    int num_heads, int num_kv_heads, int head_dim,
    cudaStream_t stream
) {
    gqa_attention_decode_kernel<<<num_heads, 128, 0, stream>>>(
        q, k_cache, v_cache, output, attn_scratch, d_pos, 0,
        max_seq_len, num_heads, num_kv_heads, head_dim
    );
}

void launch_kv_cache_write(
    half* k_cache, half* v_cache,
    const half* k_new, const half* v_new,
    const int* d_pos, int kv_dim, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (kv_dim + threads - 1) / threads;
    kv_cache_write_kernel<<<blocks, threads, 0, stream>>>(
        k_cache, v_cache, k_new, v_new, d_pos, kv_dim
    );
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
    half* out, const half* in, const half* weight, int dim, int G, float eps
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
    for (int i = threadIdx.x; i < dim; i += blockDim.x)
        y[i] = __float2half(__half2float(x[i]) * rms * __half2float(weight[i]));
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
    int q_dim, int kv_dim
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
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        data[i] = __float2half(__half2float(data[i]) * rms * __half2float(w[i]));
}

// Batched RoPE: apply to Q (q_dim, G) and K (kv_dim, G) with positions[G]
__global__ void rope_batch_kernel(
    half* q, half* k, const half* cos_table, const half* sin_table,
    const int* positions, int max_seq_len, int G,
    int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim
) {
    int g = blockIdx.x;
    if (g >= G) return;
    int pos = positions[g];
    int half_dim = head_dim / 2;

    // Apply to all Q heads
    for (int h = 0; h < num_heads; h++) {
        half* qh = q + g * q_dim + h * head_dim;
        for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
            float c = __half2float(cos_table[pos * half_dim + d]);
            float s_val = __half2float(sin_table[pos * half_dim + d]);
            float q0 = __half2float(qh[d]);
            float q1 = __half2float(qh[d + half_dim]);
            qh[d] = __float2half(q0 * c - q1 * s_val);
            qh[d + half_dim] = __float2half(q0 * s_val + q1 * c);
        }
    }
    // Apply to all KV heads
    for (int h = 0; h < num_kv_heads; h++) {
        half* kh = k + g * kv_dim + h * head_dim;
        for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
            float c = __half2float(cos_table[pos * half_dim + d]);
            float s_val = __half2float(sin_table[pos * half_dim + d]);
            float k0 = __half2float(kh[d]);
            float k1 = __half2float(kh[d + half_dim]);
            kh[d] = __float2half(k0 * c - k1 * s_val);
            kh[d + half_dim] = __float2half(k0 * s_val + k1 * c);
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
    const float* __restrict__ randoms, // (G,) uniform random values [0,1)
    int vocab, int G,
    float temperature, float top_p
) {
    int g = blockIdx.x;
    if (g >= G) return;
    float* col = logits + (size_t)g * vocab;
    int wid = threadIdx.x / warpSize, lid = threadIdx.x % warpSize;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

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

    // Step 4: cumulative sum sampling (thread 0 only)
    if (threadIdx.x == 0) {
        float threshold = randoms[g];
        if (top_p < 1.0f) threshold *= top_p;
        float cum = 0.0f;
        for (int i = 0; i < vocab; i++) {
            cum += col[i];
            if (cum >= threshold) { tokens[g] = i; return; }
        }
        tokens[g] = vocab - 1;
    }
}

void launch_sample_batch(float* logits, int* tokens, const float* randoms,
                         int vocab, int G, float temperature, float top_p, cudaStream_t s) {
    sample_batch_kernel<<<G, 256, 0, s>>>(logits, tokens, randoms, vocab, G, temperature, top_p);
}

// Launch wrappers for batch kernels
void launch_embed_batch(half* h, const half* et, const int* tok, int G, int hidden_size, cudaStream_t s) {
    embed_batch_kernel<<<G, 256, 0, s>>>(h, et, tok, G, hidden_size);
}
void launch_rms_norm_batch(half* out, const half* in, const half* w, int dim, int G, float eps, cudaStream_t s) {
    rms_norm_batch_kernel<<<G, 256, 0, s>>>(out, in, w, dim, G, eps);
}
void launch_copy_batch(half* dst, const half* src, int total, cudaStream_t s) {
    copy_batch_kernel<<<(total+255)/256, 256, 0, s>>>(dst, src, total);
}
void launch_residual_add_batch(half* out, const half* res, int total, cudaStream_t s) {
    residual_add_batch_kernel<<<(total+255)/256, 256, 0, s>>>(out, res, total);
}
void launch_qk_norm_batch(half* q, half* k, const half* qw, const half* kw, int nq, int nkv, int hd, int G, float eps, int q_dim, int kv_dim, cudaStream_t s) {
    qk_norm_batch_kernel<<<G * (nq + nkv), 128, 0, s>>>(q, k, qw, kw, nq, nkv, hd, G, eps, q_dim, kv_dim);
}
void launch_rope_batch(half* q, half* k, const half* ct, const half* st, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s) {
    rope_batch_kernel<<<G, 256, 0, s>>>(q, k, ct, st, pos, msl, G, num_heads, num_kv_heads, head_dim, q_dim, kv_dim);
}
void launch_kv_cache_write_batch(half* ck, half* cv, const half* k, const half* v, const int* pos, int msl, int G, int kv_dim, cudaStream_t s) {
    kv_cache_write_batch_kernel<<<G, 256, 0, s>>>(ck, cv, k, v, pos, msl, G, kv_dim);
}
void launch_gqa_attention_batch(half* out, const half* q, const half* ck, const half* cv, float* as, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s) {
    dim3 grid(G, num_heads);
    gqa_attention_batch_kernel<<<grid, 128, 0, s>>>(out, q, ck, cv, as, pos, msl, G, num_heads, num_kv_heads, head_dim, q_dim, kv_dim);
}
void launch_silu_mul_batch(half* gate, const half* up, int total, cudaStream_t s) {
    silu_mul_batch_kernel<<<(total+255)/256, 256, 0, s>>>(gate, up, total);
}
void launch_argmax_batch(const float* logits, int* tokens, int vocab, int G, cudaStream_t s) {
    argmax_batch_kernel<<<G, 256, 0, s>>>(logits, tokens, vocab, G);
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

} // extern "C"
