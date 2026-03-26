/**
 * C++ inference engine — the core decode loop.
 *
 * One function call from Python generates an entire completion.
 * No Python per token, no kernel launch overhead from PyTorch dispatch.
 */

#include "model.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <iostream>

// cuBLAS handle (created once, reused)
static cublasHandle_t cublas_handle = nullptr;

static void ensure_cublas() {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
        cublasSetStream(cublas_handle, 0);  // force default stream
    }
}

// cuBLAS fp16 GEMV with optional LoRA: y = W @ x + scale * B @ (A @ x)
static void cublas_hgemv_lora(const half* weight, const half* input, half* output,
                                int out_dim, int in_dim,
                                const LoRAAdapter* lora, half* lora_scratch) {
    ensure_cublas();
    __half alpha_h = __float2half(1.0f);
    __half beta_zero = __float2half(0.0f);

    // Base: y = W @ x
    cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 out_dim, 1, in_dim, &alpha_h,
                 weight, CUDA_R_16F, in_dim,
                 input, CUDA_R_16F, in_dim,
                 &beta_zero, output, CUDA_R_16F, out_dim,
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (lora && lora->A && lora->B) {
        // Step 1: scratch = A @ x  (rank, in_dim) @ (in_dim, 1) = (rank, 1)
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     lora->rank, 1, lora->in_features, &alpha_h,
                     lora->A, CUDA_R_16F, lora->in_features,
                     input, CUDA_R_16F, lora->in_features,
                     &beta_zero, lora_scratch, CUDA_R_16F, lora->rank,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        // Step 2: y += scale * B @ scratch  (out_dim, rank) @ (rank, 1)
        __half scale_h = __float2half(lora->scale);
        __half one_h = __float2half(1.0f);
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     lora->out_features, 1, lora->rank, &scale_h,
                     lora->B, CUDA_R_16F, lora->rank,
                     lora_scratch, CUDA_R_16F, lora->rank,
                     &one_h, output, CUDA_R_16F, lora->out_features,
                     CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

// cuBLAS fp16 GEMV: y = weight * x
// Use cublasGemmEx: treat x as (in_dim, 1) matrix, W as (out_dim, in_dim) row-major
// y(out_dim,1) = W(out_dim,in_dim) @ x(in_dim,1)
// In cublas col-major: y = W^T(in_dim,out_dim) transposed @ x
static void cublas_hgemv(const half* weight, const half* input, half* output,
                          int out_dim, int in_dim) {
    ensure_cublas();
    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);
    // W is (out_dim, in_dim) row-major = (in_dim, out_dim) col-major
    // y = op(W) @ x where op(W) = W^T in col-major = W in row-major
    // cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_dim, 1, in_dim,
    //              alpha, W, in_dim, x, in_dim, beta, y, out_dim)
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_T, CUBLAS_OP_N,
                 out_dim, 1, in_dim,  // M, N, K
                 &alpha,
                 weight, CUDA_R_16F, in_dim,   // A (in_dim x out_dim col-major), lda=in_dim
                 input, CUDA_R_16F, in_dim,    // B (in_dim x 1), ldb=in_dim
                 &beta,
                 output, CUDA_R_16F, out_dim,  // C (out_dim x 1), ldc=out_dim
                 CUDA_R_16F,                   // compute type
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// Helper to avoid void** casts everywhere
template<typename T>
inline cudaError_t cudaMallocTyped(T** ptr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}

// Forward declarations for kernel launchers (defined in kernels.cu)
extern "C" {
    void launch_fp16_gemv(const half* weight, const half* input, half* output, int out_dim, int in_dim, cudaStream_t stream);
    void launch_nf4_gemv(const uint8_t* packed, const float* absmax, const float* quant_map, const half* input, half* output, int out_dim, int in_dim, int block_size, cudaStream_t stream);
    void launch_rms_norm(const half* input, const half* weight, half* output, int dim, float eps, cudaStream_t stream);
    void launch_qk_norm(half* q, half* k, const half* q_weight, const half* k_weight, int num_q_heads, int num_kv_heads, int head_dim, float eps, cudaStream_t stream);
    void launch_rope(half* q, half* k, const half* cos_table, const half* sin_table, int pos, int max_seq_len, cudaStream_t stream);
    void launch_gqa_attention(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, int pos, int max_seq_len, cudaStream_t stream);
    void launch_silu_gate_mul(const half* gate, const half* up, half* output, int dim, cudaStream_t stream);
    void launch_embedding(const half* embed_table, int token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_residual_add(half* output, const half* residual, int dim, cudaStream_t stream);
    void launch_lm_head(const half* weight, const half* input, float* logits, int hidden_dim, int vocab_size, cudaStream_t stream);
    void launch_argmax(const float* logits, int* result, int vocab_size, cudaStream_t stream);
    void launch_gpu_sample(float* logits, int* result, int vocab_size, float temperature, float random_val, float top_p, cudaStream_t stream);
    void launch_embedding_device(const half* embed_table, const int* d_token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_rope_device(half* q, half* k, const half* cos_table_base, const half* sin_table_base, const int* d_pos, cudaStream_t stream);
    void launch_gqa_attention_device(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, const int* d_pos, int max_seq_len, cudaStream_t stream);
    void launch_kv_cache_write(half* k_cache, half* v_cache, const half* k_new, const half* v_new, const int* d_pos, int kv_dim, cudaStream_t stream);
}

using namespace qwen3;

// ============================================================================
// Constructor / Destructor
// ============================================================================

InferenceEngine::InferenceEngine(int max_seq_len) {
    state_.max_seq_len = max_seq_len;
    state_.current_pos = 0;

    // Allocate KV caches
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMallocTyped(&state_.kv_cache[i].key, max_seq_len * KV_DIM * sizeof(half));
        cudaMallocTyped(&state_.kv_cache[i].value, max_seq_len * KV_DIM * sizeof(half));
    }

    // Allocate scratch buffers
    cudaMallocTyped(&state_.hidden, HIDDEN_SIZE * sizeof(half));
    cudaMallocTyped(&state_.residual, HIDDEN_SIZE * sizeof(half));
    cudaMallocTyped(&state_.q_buf, Q_DIM * sizeof(half));
    cudaMallocTyped(&state_.k_buf, KV_DIM * sizeof(half));
    cudaMallocTyped(&state_.v_buf, KV_DIM * sizeof(half));
    cudaMallocTyped(&state_.attn_out, Q_DIM * sizeof(half));
    cudaMallocTyped(&state_.gate_buf, INTERMEDIATE_SIZE * sizeof(half));
    cudaMallocTyped(&state_.up_buf, INTERMEDIATE_SIZE * sizeof(half));
    cudaMallocTyped(&state_.ffn_out, HIDDEN_SIZE * sizeof(half));
    cudaMallocTyped(&state_.logits, VOCAB_SIZE * sizeof(float));
    cudaMallocTyped(&state_.attn_scores, NUM_HEADS * max_seq_len * sizeof(float));

    // Allocate and precompute RoPE tables
    cudaMallocTyped(&state_.rope_cos, max_seq_len * (HEAD_DIM / 2) * sizeof(half));
    cudaMallocTyped(&state_.rope_sin, max_seq_len * (HEAD_DIM / 2) * sizeof(half));
    precompute_rope();

    // GPU-side sampling
    cudaMallocTyped(&state_.sample_result, sizeof(int));

    // LoRA scratch
    cudaMallocTyped(&state_.lora_scratch, 64 * sizeof(half)); // max rank 64

    // CUDA graph control buffers (device-side, updated before graph replay)
    cudaMallocTyped(&state_.d_token_id, sizeof(int));
    cudaMallocTyped(&state_.d_pos, sizeof(int));

    // Zero-init weights
    memset(&weights_, 0, sizeof(weights_));
}

InferenceEngine::~InferenceEngine() {
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaFree(state_.kv_cache[i].key);
        cudaFree(state_.kv_cache[i].value);
    }
    cudaFree(state_.hidden);
    cudaFree(state_.residual);
    cudaFree(state_.q_buf);
    cudaFree(state_.k_buf);
    cudaFree(state_.v_buf);
    cudaFree(state_.attn_out);
    cudaFree(state_.gate_buf);
    cudaFree(state_.up_buf);
    cudaFree(state_.ffn_out);
    cudaFree(state_.logits);
    cudaFree(state_.attn_scores);
    cudaFree(state_.rope_cos);
    cudaFree(state_.rope_sin);
    cudaFree(state_.sample_result);
}

// ============================================================================
// RoPE precomputation
// ============================================================================

void InferenceEngine::precompute_rope() {
    int half_dim = HEAD_DIM / 2;
    std::vector<half> cos_h(state_.max_seq_len * half_dim);
    std::vector<half> sin_h(state_.max_seq_len * half_dim);

    for (int pos = 0; pos < state_.max_seq_len; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * d) / HEAD_DIM);
            float angle = pos * freq;
            cos_h[pos * half_dim + d] = __float2half(cosf(angle));
            sin_h[pos * half_dim + d] = __float2half(sinf(angle));
        }
    }

    cudaMemcpy(state_.rope_cos, cos_h.data(),
               state_.max_seq_len * half_dim * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(state_.rope_sin, sin_h.data(),
               state_.max_seq_len * half_dim * sizeof(half), cudaMemcpyHostToDevice);
}

// ============================================================================
// Reset
// ============================================================================

void InferenceEngine::reset() {
    state_.current_pos = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMemset(state_.kv_cache[i].key, 0, state_.max_seq_len * KV_DIM * sizeof(half));
        cudaMemset(state_.kv_cache[i].value, 0, state_.max_seq_len * KV_DIM * sizeof(half));
    }
}

// ============================================================================
// Forward pass through one transformer layer
// ============================================================================

void InferenceEngine::forward_layer(int layer_idx) {
    auto& layer = weights_.layers[layer_idx];
    auto& kv = state_.kv_cache[layer_idx];
    cudaStream_t stream = 0; // default stream

    // 1. Input LayerNorm
    launch_rms_norm(state_.hidden, layer.input_layernorm, state_.residual,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);
    // After norm: residual = norm(hidden), hidden = original (for residual add later)
    // Wait, we need to save hidden for residual. Let's swap:
    // Actually: residual should store the original hidden for the add later.
    // Let me restructure: norm writes to a temp, we use hidden as residual.

    // Save hidden as residual
    cudaMemcpyAsync(state_.residual, state_.hidden, HIDDEN_SIZE * sizeof(half),
                     cudaMemcpyDeviceToDevice, stream);

    // Norm into hidden (reuse buffer)
    half* norm_out = state_.ffn_out; // temporary borrow
    launch_rms_norm(state_.residual, layer.input_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // 2. QKV projections (fp16 or NF4 depending on layer)
    if (layer.attn_is_nf4) {
        auto& q = layer.q_proj_nf4; auto& k = layer.k_proj_nf4; auto& v = layer.v_proj_nf4;
        launch_nf4_gemv(q.data, q.absmax, q.quant_map, norm_out, state_.q_buf, q.out_dim, q.in_dim, q.block_size, stream);
        cudaStreamSynchronize(stream);  // DEBUG: sync after NF4 GEMV
        launch_nf4_gemv(k.data, k.absmax, k.quant_map, norm_out, state_.k_buf, k.out_dim, k.in_dim, k.block_size, stream);
        cudaStreamSynchronize(stream);
        launch_nf4_gemv(v.data, v.absmax, v.quant_map, norm_out, state_.v_buf, v.out_dim, v.in_dim, v.block_size, stream);
        cudaStreamSynchronize(stream);
    } else {
        cublas_hgemv_lora(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, layer.lora_q, state_.lora_scratch);
        cublas_hgemv_lora(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, layer.lora_k, state_.lora_scratch);
        cublas_hgemv_lora(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, layer.lora_v, state_.lora_scratch);
    }

    // 2b. Fused QKNorm (one kernel for all 24 heads instead of 24 separate launches)
    launch_qk_norm(state_.q_buf, state_.k_buf, layer.q_norm, layer.k_norm,
                   NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_NORM_EPS, stream);

    // 3. RoPE
    launch_rope(state_.q_buf, state_.k_buf,
                state_.rope_cos, state_.rope_sin,
                state_.current_pos, state_.max_seq_len, stream);

    // 4. Store K, V into cache at current position
    cudaMemcpyAsync(kv.key + state_.current_pos * KV_DIM, state_.k_buf,
                     KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kv.value + state_.current_pos * KV_DIM, state_.v_buf,
                     KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    // 5. GQA Attention
    launch_gqa_attention(state_.q_buf, kv.key, kv.value, state_.attn_out,
                          state_.attn_scores, state_.current_pos, state_.max_seq_len, stream);

    // 6. Output projection (fp16 or NF4)
    if (layer.attn_is_nf4) {
        auto& o = layer.o_proj_nf4;
        launch_nf4_gemv(o.data, o.absmax, o.quant_map, state_.attn_out, state_.hidden, o.out_dim, o.in_dim, o.block_size, stream);
    } else {
        cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
    }

    // 7. Residual add (hidden += residual)
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // 8. Post-attention LayerNorm
    cudaMemcpyAsync(state_.residual, state_.hidden, HIDDEN_SIZE * sizeof(half),
                     cudaMemcpyDeviceToDevice, stream);
    launch_rms_norm(state_.residual, layer.post_attn_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // 9. FFN: gate_proj and up_proj
    if (layer.mlp_is_nf4) {
        auto& g = layer.gate_proj_nf4;
        auto& u = layer.up_proj_nf4;
        auto& d = layer.down_proj_nf4;
        launch_nf4_gemv(g.data, g.absmax, g.quant_map, norm_out, state_.gate_buf, g.out_dim, g.in_dim, g.block_size, stream);
        launch_nf4_gemv(u.data, u.absmax, u.quant_map, norm_out, state_.up_buf, u.out_dim, u.in_dim, u.block_size, stream);
        launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
        launch_nf4_gemv(d.data, d.absmax, d.quant_map, state_.gate_buf, state_.hidden, d.out_dim, d.in_dim, d.block_size, stream);
    } else {
        // Fuse gate + up into single GEMM if weights are contiguous in memory
        // gate_proj is (INTERMEDIATE, HIDDEN), up_proj is (INTERMEDIATE, HIDDEN)
        // If gate and up are contiguous, we can do one GEMM of (2*INTERMEDIATE, HIDDEN) @ (HIDDEN, 1)
        // Output goes to gate_buf which must be 2*INTERMEDIATE large
        // For now, still 2 calls (safe). TODO: concat weights at load time.
        cublas_hgemv_lora(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_gate, state_.lora_scratch);
        cublas_hgemv_lora(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_up, state_.lora_scratch);
        launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
        cublas_hgemv_lora(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, layer.lora_down, state_.lora_scratch);
    }

    // 12. Residual add
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);
}

// ============================================================================
// Decode: process one token
// ============================================================================

void InferenceEngine::decode(int token_id) {
    cudaStream_t stream = 0;

    // Embedding lookup
    launch_embedding(weights_.embed_tokens, token_id, state_.hidden, HIDDEN_SIZE, stream);

    // Forward through all layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        forward_layer(i);
    }

    // Final LayerNorm
    half* norm_out = state_.ffn_out; // borrow
    launch_rms_norm(state_.hidden, weights_.final_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // LM head (tied to embedding) -> fp32 logits via cuBLAS mixed precision
    {
        ensure_cublas();
        float alpha = 1.0f, beta = 0.0f;
        // embed_tokens is (VOCAB_SIZE, HIDDEN_SIZE) row-major
        // We want logits[i] = embed[i,:] @ norm_out for all i
        // = embed @ norm_out where embed is (VOCAB, HIDDEN) and norm_out is (HIDDEN, 1)
        // cuBLAS col-major: embed^T is (HIDDEN, VOCAB), so CUBLAS_OP_T
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE,
                     &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta,
                     state_.logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F,  // compute in fp32
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    state_.current_pos++;
}

// ============================================================================
// Graph-captured forward layer (uses device-side position)
// ============================================================================

void InferenceEngine::forward_layer_graph(int layer_idx, cudaStream_t stream) {
    auto& layer = weights_.layers[layer_idx];
    auto& kv = state_.kv_cache[layer_idx];

    // Save hidden as residual
    cudaMemcpyAsync(state_.residual, state_.hidden, HIDDEN_SIZE * sizeof(half),
                     cudaMemcpyDeviceToDevice, stream);
    half* norm_out = state_.ffn_out;
    launch_rms_norm(state_.residual, layer.input_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // QKV projections (fp16 or NF4)
    if (layer.attn_is_nf4) {
        auto& q = layer.q_proj_nf4; auto& k = layer.k_proj_nf4; auto& v = layer.v_proj_nf4;
        launch_nf4_gemv(q.data, q.absmax, q.quant_map, norm_out, state_.q_buf, q.out_dim, q.in_dim, q.block_size, stream);
        launch_nf4_gemv(k.data, k.absmax, k.quant_map, norm_out, state_.k_buf, k.out_dim, k.in_dim, k.block_size, stream);
        launch_nf4_gemv(v.data, v.absmax, v.quant_map, norm_out, state_.v_buf, v.out_dim, v.in_dim, v.block_size, stream);
    } else {
        cublas_hgemv_lora(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, layer.lora_q, state_.lora_scratch);
        cublas_hgemv_lora(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, layer.lora_k, state_.lora_scratch);
        cublas_hgemv_lora(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, layer.lora_v, state_.lora_scratch);
    }

    // Fused QKNorm
    launch_qk_norm(state_.q_buf, state_.k_buf, layer.q_norm, layer.k_norm,
                   NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_NORM_EPS, stream);

    // RoPE (reads position from device memory)
    launch_rope_device(state_.q_buf, state_.k_buf,
                       state_.rope_cos, state_.rope_sin, state_.d_pos, stream);

    // KV cache write (reads position from device memory)
    launch_kv_cache_write(kv.key, kv.value, state_.k_buf, state_.v_buf,
                           state_.d_pos, KV_DIM, stream);

    // GQA Attention (reads position from device memory)
    launch_gqa_attention_device(state_.q_buf, kv.key, kv.value, state_.attn_out,
                                 state_.attn_scores, state_.d_pos,
                                 state_.max_seq_len, stream);

    // Output projection (fp16 or NF4)
    if (layer.attn_is_nf4) {
        auto& o = layer.o_proj_nf4;
        launch_nf4_gemv(o.data, o.absmax, o.quant_map, state_.attn_out, state_.hidden, o.out_dim, o.in_dim, o.block_size, stream);
    } else {
        cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
    }

    // Residual add
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // Post-attention norm + FFN
    cudaMemcpyAsync(state_.residual, state_.hidden, HIDDEN_SIZE * sizeof(half),
                     cudaMemcpyDeviceToDevice, stream);
    launch_rms_norm(state_.residual, layer.post_attn_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    if (layer.mlp_is_nf4) {
        auto& g = layer.gate_proj_nf4;
        auto& u = layer.up_proj_nf4;
        auto& d = layer.down_proj_nf4;
        launch_nf4_gemv(g.data, g.absmax, g.quant_map, norm_out, state_.gate_buf, g.out_dim, g.in_dim, g.block_size, stream);
        launch_nf4_gemv(u.data, u.absmax, u.quant_map, norm_out, state_.up_buf, u.out_dim, u.in_dim, u.block_size, stream);
        launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
        launch_nf4_gemv(d.data, d.absmax, d.quant_map, state_.gate_buf, state_.hidden, d.out_dim, d.in_dim, d.block_size, stream);
    } else {
        cublas_hgemv_lora(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_gate, state_.lora_scratch);
        cublas_hgemv_lora(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_up, state_.lora_scratch);
        launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
        cublas_hgemv_lora(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, layer.lora_down, state_.lora_scratch);
    }

    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);
}

// ============================================================================
// CUDA Graph capture and replay
// ============================================================================

void InferenceEngine::enable_cuda_graph() {
    if (graph_captured_) return;

    // Create a dedicated stream for graph capture
    cudaStreamCreate(&graph_stream_);

    // Set cuBLAS to use our stream
    cublasSetStream(cublas_handle, graph_stream_);

    // Warm up: run one full decode on the graph stream
    int warmup_token = 0;
    cudaMemcpy(state_.d_token_id, &warmup_token, sizeof(int), cudaMemcpyHostToDevice);
    int warmup_pos = state_.current_pos;
    cudaMemcpy(state_.d_pos, &warmup_pos, sizeof(int), cudaMemcpyHostToDevice);

    // Embedding (device-side token id)
    launch_embedding_device(weights_.embed_tokens, state_.d_token_id,
                             state_.hidden, HIDDEN_SIZE, graph_stream_);
    for (int i = 0; i < NUM_LAYERS; i++) {
        forward_layer_graph(i, graph_stream_);
    }
    half* norm_out = state_.ffn_out;
    launch_rms_norm(state_.hidden, weights_.final_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, graph_stream_);
    // lm_head
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta, state_.logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaStreamSynchronize(graph_stream_);

    // Now capture
    cudaStreamBeginCapture(graph_stream_, cudaStreamCaptureModeGlobal);

    launch_embedding_device(weights_.embed_tokens, state_.d_token_id,
                             state_.hidden, HIDDEN_SIZE, graph_stream_);
    for (int i = 0; i < NUM_LAYERS; i++) {
        forward_layer_graph(i, graph_stream_);
    }
    launch_rms_norm(state_.hidden, weights_.final_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, graph_stream_);
    {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta, state_.logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    cudaStreamEndCapture(graph_stream_, &cuda_graph_);
    cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0);

    // Restore cuBLAS to default stream
    cublasSetStream(cublas_handle, 0);

    graph_captured_ = true;
    use_cuda_graph_ = true;
    std::cout << "  CUDA graph captured for decode step" << std::endl;
}

// ============================================================================
// Graph-accelerated decode
// ============================================================================

void InferenceEngine::decode_graph(int token_id) {
    if (!graph_captured_) {
        enable_cuda_graph();
    }

    // Update device-side control values
    cudaMemcpy(state_.d_token_id, &token_id, sizeof(int), cudaMemcpyHostToDevice);
    int pos = state_.current_pos;
    cudaMemcpy(state_.d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice);

    // Replay the captured graph
    cudaGraphLaunch(cuda_graph_exec_, graph_stream_);
    cudaStreamSynchronize(graph_stream_);

    state_.current_pos++;
}

// ============================================================================
// GPU-side greedy sampling (no CPU copy)
// ============================================================================

// ============================================================================
// LoRA weight sync from PyTorch
// ============================================================================

void InferenceEngine::update_lora_weight(
    int layer_idx, const char* proj_name,
    const half* A_data, int A_rows, int A_cols,
    const half* B_data, int B_rows, int B_cols,
    float scale
) {
    if (layer_idx < 0 || layer_idx >= NUM_LAYERS) return;
    auto& layer = weights_.layers[layer_idx];

    // Find target adapter
    LoRAAdapter** target = nullptr;
    std::string proj(proj_name);
    if (proj == "q_proj") target = &layer.lora_q;
    else if (proj == "k_proj") target = &layer.lora_k;
    else if (proj == "v_proj") target = &layer.lora_v;
    else if (proj == "o_proj") target = &layer.lora_o;
    else if (proj == "gate_proj") target = &layer.lora_gate;
    else if (proj == "up_proj") target = &layer.lora_up;
    else if (proj == "down_proj") target = &layer.lora_down;
    if (!target) return;

    // Create or update adapter
    if (!*target) {
        *target = new LoRAAdapter();
        cudaMallocTyped(&(*target)->A, A_rows * A_cols * sizeof(half));
        cudaMallocTyped(&(*target)->B, B_rows * B_cols * sizeof(half));
    }

    // Copy weights to GPU
    cudaMemcpy((*target)->A, A_data, A_rows * A_cols * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy((*target)->B, B_data, B_rows * B_cols * sizeof(half), cudaMemcpyHostToDevice);
    (*target)->rank = A_rows;
    (*target)->in_features = A_cols;
    (*target)->out_features = B_rows;
    (*target)->scale = scale;
}

int InferenceEngine::sample_greedy_gpu() {
    launch_argmax(state_.logits, state_.sample_result, VOCAB_SIZE, 0);
    int result;
    cudaMemcpy(&result, state_.sample_result, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

int InferenceEngine::sample_gpu(float temperature, float top_p) {
    if (temperature < 0.01f) {
        return sample_greedy_gpu();
    }
    // Generate random number on CPU (fast enough, 1 random per token)
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    // GPU kernel: temperature + softmax + cumulative sample
    // NOTE: this modifies logits in-place (they become probabilities)
    launch_gpu_sample(state_.logits, state_.sample_result, VOCAB_SIZE,
                      temperature, r, top_p, 0);
    int result;
    cudaMemcpy(&result, state_.sample_result, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================================
// Prefill: process multiple tokens
// ============================================================================

void InferenceEngine::prefill(const int* token_ids, int n_tokens) {
    // Simple implementation: process one token at a time
    // TODO: optimize with batch prefill kernel
    for (int i = 0; i < n_tokens; i++) {
        decode(token_ids[i]);
    }
}

// ============================================================================
// Sampling
// ============================================================================

int InferenceEngine::sample(float temperature, float top_p) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error before sample: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    std::vector<float> logits_host(VOCAB_SIZE);
    err = cudaMemcpy(logits_host.data(), state_.logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error in sample: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    // Temperature
    if (temperature != 1.0f) {
        for (auto& l : logits_host) l /= temperature;
    }

    // Softmax
    float max_val = *std::max_element(logits_host.begin(), logits_host.end());
    float sum = 0.0f;
    for (auto& l : logits_host) {
        l = expf(l - max_val);
        sum += l;
    }
    for (auto& l : logits_host) l /= sum;

    // Top-p (nucleus) sampling
    if (top_p < 1.0f) {
        // Create index array and sort by probability (descending)
        std::vector<int> indices(VOCAB_SIZE);
        for (int i = 0; i < VOCAB_SIZE; i++) indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + std::min(1000, VOCAB_SIZE),
                          indices.end(),
                          [&](int a, int b) { return logits_host[a] > logits_host[b]; });

        // Find nucleus (top-p cumulative probability)
        float cumsum = 0.0f;
        int nucleus_end = 0;
        for (int i = 0; i < std::min(1000, VOCAB_SIZE); i++) {
            cumsum += logits_host[indices[i]];
            nucleus_end = i + 1;
            if (cumsum >= top_p) break;
        }

        // Sample from nucleus
        static std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, cumsum);
        float r = dist(rng);

        float running = 0.0f;
        for (int i = 0; i < nucleus_end; i++) {
            running += logits_host[indices[i]];
            if (running >= r) return indices[i];
        }
        return indices[nucleus_end - 1];
    }

    // Multinomial sampling (no top-p)
    static std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float running = 0.0f;
    for (int i = 0; i < VOCAB_SIZE; i++) {
        running += logits_host[i];
        if (running >= r) return i;
    }
    return VOCAB_SIZE - 1;
}

// ============================================================================
// Full generation
// ============================================================================

std::vector<int> InferenceEngine::generate(
    const std::vector<int>& prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    int eos_token_id,
    const std::vector<int>& stop_token_ids
) {
    reset();

    // Prefill
    prefill(prompt.data(), prompt.size());

    // Sample first token from prefill logits (always GPU)
    int token = sample_gpu(temperature, top_p);
    std::vector<int> output = {token};

    // CUDA graph currently only works with fp16 weights (NF4+cuBLAS stream mismatch)
    // TODO: fix graph capture for mixed NF4/cuBLAS layers
    // if (!graph_captured_ && prompt.size() > 0) {
    //     enable_cuda_graph();
    // }

    // Decode loop (CUDA graph replay, no Python, all GPU sampling)
    for (int i = 0; i < max_new_tokens - 1; i++) {
        if (token == eos_token_id) break;

        // Check stop tokens
        bool should_stop = false;
        int stop_len = stop_token_ids.size();
        if (stop_len > 0 && (int)output.size() >= stop_len) {
            should_stop = true;
            for (int j = 0; j < stop_len; j++) {
                if (output[output.size() - stop_len + j] != stop_token_ids[j]) {
                    should_stop = false;
                    break;
                }
            }
        }
        if (should_stop) break;

        if (use_cuda_graph_) {
            decode_graph(token);
        } else {
            decode(token);
        }
        token = sample_gpu(temperature, top_p);
        output.push_back(token);
    }

    return output;
}
