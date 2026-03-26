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
    void launch_nf4_gemv_fast(const uint8_t* packed, const float* absmax, const half* input, half* output, int out_dim, int in_dim, int block_size, cudaStream_t stream);
    void launch_q4l_gemv(const uint8_t* packed, const float* scales, const half* input, half* output, int out_dim, int in_dim, int block_size, cudaStream_t stream);
    void launch_nf4_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_nf4_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_quantize_input_q8(const half* input, int8_t* q8_data, float* q8_scales, float* q8_sums, int dim, cudaStream_t stream);
    void launch_q4l_dp4a_gemv(const uint8_t* w, const float* w_scales, const int8_t* q8, const float* q8_sc, const float* q8_sm, half* y, int out_dim, int in_dim, cudaStream_t stream);
    void launch_dequant_q4l(half* out, const uint8_t* data, const float* scales, int out_dim, int in_dim, cudaStream_t stream);

    // Batched kernel launchers
    void launch_embed_batch(half* h, const half* et, const int* tok, int G, cudaStream_t s);
    void launch_rms_norm_batch(half* out, const half* in, const half* w, int dim, int G, float eps, cudaStream_t s);
    void launch_copy_batch(half* dst, const half* src, int total, cudaStream_t s);
    void launch_residual_add_batch(half* out, const half* res, int total, cudaStream_t s);
    void launch_qk_norm_batch(half* q, half* k, const half* qw, const half* kw, int nq, int nkv, int hd, int G, float eps, cudaStream_t s);
    void launch_rope_batch(half* q, half* k, const half* ct, const half* st, const int* pos, int msl, int G, cudaStream_t s);
    void launch_kv_cache_write_batch(half* ck, half* cv, const half* k, const half* v, const int* pos, int msl, int G, cudaStream_t s);
    void launch_gqa_attention_batch(half* out, const half* q, const half* ck, const half* cv, float* as, const int* pos, int msl, int G, cudaStream_t s);
    void launch_silu_mul_batch(half* gate, const half* up, int total, cudaStream_t s);
    void launch_argmax_batch(const float* logits, int* tokens, int vocab, int G, cudaStream_t s);
    void launch_rms_norm(const half* input, const half* weight, half* output, int dim, float eps, cudaStream_t stream);
    void launch_copy_rms_norm(const half* input, const half* weight, half* residual, half* norm_out, int dim, float eps, cudaStream_t stream);
    void launch_qk_norm(half* q, half* k, const half* q_weight, const half* k_weight, int num_q_heads, int num_kv_heads, int head_dim, float eps, cudaStream_t stream);
    void launch_rope(half* q, half* k, const half* cos_table, const half* sin_table, int pos, int max_seq_len, cudaStream_t stream);
    void launch_gqa_attention(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, int pos, int max_seq_len, cudaStream_t stream);
    void launch_silu_gate_mul(const half* gate, const half* up, half* output, int dim, cudaStream_t stream);
    void launch_embedding(const half* embed_table, int token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_residual_add(half* output, const half* residual, int dim, cudaStream_t stream);
    void launch_lm_head(const half* weight, const half* input, float* logits, int hidden_dim, int vocab_size, cudaStream_t stream);
    void launch_fp16_to_fp32(const half* input, float* output, int n, cudaStream_t stream);
    void launch_argmax(const float* logits, int* result, int vocab_size, cudaStream_t stream);
    void launch_gpu_sample(float* logits, int* result, int vocab_size, float temperature, float random_val, float top_p, cudaStream_t stream);
    void launch_embedding_device(const half* embed_table, const int* d_token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_rope_device(half* q, half* k, const half* cos_table_base, const half* sin_table_base, const int* d_pos, cudaStream_t stream);
    void launch_gqa_attention_device(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, const int* d_pos, int max_seq_len, cudaStream_t stream);
    void launch_kv_cache_write(half* k_cache, half* v_cache, const half* k_new, const half* v_new, const int* d_pos, int kv_dim, cudaStream_t stream);
}

using namespace qwen3;

// Dispatch macros: use dp4a for Q4L, fallback to fp32 FMA for NF4
#define GEMV_4BIT(w, in, out, stream) \
    do { if (weights_.is_q4l) launch_q4l_dp4a_gemv((w).data, (w).absmax, state_.q8_data, state_.q8_scales, state_.q8_sums, (out), (w).out_dim, (w).in_dim, (stream)); \
         else launch_nf4_gemv_fast((w).data, (w).absmax, (in), (out), (w).out_dim, (w).in_dim, (w).block_size, (stream)); } while(0)
#define FUSED2_4BIT(aw, ay, bw, by, x, id, stream) \
    do { if (weights_.is_q4l) launch_q4l_fused_2((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (x), (id), (stream)); \
         else launch_nf4_fused_2((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (x), (id), (stream)); } while(0)
#define FUSED3_4BIT(aw, ay, bw, by, cw, cy, x, id, stream) \
    do { if (weights_.is_q4l) launch_q4l_fused_3((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (cw).data, (cw).absmax, (cy), (cw).out_dim, (x), (id), (stream)); \
         else launch_nf4_fused_3((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (cw).data, (cw).absmax, (cy), (cw).out_dim, (x), (id), (stream)); } while(0)

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

    // NF4 LM head temp buffer (allocated upfront, ~304KB)
    cudaMallocTyped(&state_.lm_head_fp16_buf, VOCAB_SIZE * sizeof(half));

    // dp4a input quantization buffers
    int max_q8_dim = INTERMEDIATE_SIZE; // largest input dimension
    int max_q8_blocks = max_q8_dim / 64;
    cudaMallocTyped(&state_.q8_data, max_q8_dim * sizeof(int8_t));
    cudaMallocTyped(&state_.q8_scales, max_q8_blocks * sizeof(float));
    cudaMallocTyped(&state_.q8_sums, max_q8_blocks * sizeof(float));

    // LoRA scratch
    cudaMallocTyped(&state_.lora_scratch, 64 * sizeof(half)); // max rank 64

    // CUDA graph control buffers (device-side, updated before graph replay)
    cudaMallocTyped(&state_.d_token_id, sizeof(int));
    cudaMallocTyped(&state_.d_pos, sizeof(int));

    // Pinned host memory for async graph control updates
    cudaHostAlloc(&h_token_id_pinned_, sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_pos_pinned_, sizeof(int), cudaHostAllocDefault);

    // Zero-init weights
    memset(&weights_, 0, sizeof(weights_));
    batch_ = nullptr;

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

    // 1. Input LayerNorm: copy hidden → residual AND normalize → norm_out (1 kernel)
    half* norm_out = state_.ffn_out;
    launch_copy_rms_norm(state_.hidden, layer.input_layernorm,
                         state_.residual, norm_out,
                         HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // Quantize input to int8 for dp4a (shared by QKV projections)
    if (weights_.is_q4l) {
        launch_quantize_input_q8(norm_out, state_.q8_data, state_.q8_scales,
                                 state_.q8_sums, HIDDEN_SIZE, stream);
    }

    // 2. QKV projections
    if (!layer.q_proj_fp16 && !layer.k_proj_fp16 && !layer.v_proj_fp16 && !weights_.is_q4l) {
        // Fused NF4 path (3→1 launch)
        auto& q = layer.q_proj_nf4; auto& k = layer.k_proj_nf4; auto& v = layer.v_proj_nf4;
        FUSED3_4BIT(q, state_.q_buf, k, state_.k_buf, v, state_.v_buf, norm_out, q.in_dim, stream);
    } else {
        if (layer.q_proj_fp16) cublas_hgemv_lora(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, layer.lora_q, state_.lora_scratch);
        else { auto& w = layer.q_proj_nf4; GEMV_4BIT(w, norm_out, state_.q_buf, stream); }
        if (layer.k_proj_fp16) cublas_hgemv_lora(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, layer.lora_k, state_.lora_scratch);
        else { auto& w = layer.k_proj_nf4; GEMV_4BIT(w, norm_out, state_.k_buf, stream); }
        if (layer.v_proj_fp16) cublas_hgemv_lora(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, layer.lora_v, state_.lora_scratch);
        else { auto& w = layer.v_proj_nf4; GEMV_4BIT(w, norm_out, state_.v_buf, stream); }
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

    // 6. Output projection (input: attn_out, dim=Q_DIM)
    if (!layer.o_proj_fp16 && weights_.is_q4l) {
        launch_quantize_input_q8(state_.attn_out, state_.q8_data, state_.q8_scales,
                                 state_.q8_sums, Q_DIM, stream);
    }
    if (layer.o_proj_fp16) cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
    else { auto& w = layer.o_proj_nf4; GEMV_4BIT(w, state_.attn_out, state_.hidden, stream); }

    // 7. Residual add (hidden += residual)
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // 8. Post-attention LayerNorm: copy hidden → residual AND normalize → norm_out
    launch_copy_rms_norm(state_.hidden, layer.post_attn_layernorm,
                         state_.residual, norm_out,
                         HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // Quantize FFN input for dp4a (shared by gate+up+down)
    if (weights_.is_q4l) {
        launch_quantize_input_q8(norm_out, state_.q8_data, state_.q8_scales,
                                 state_.q8_sums, HIDDEN_SIZE, stream);
    }

    // 9. FFN: fused gate+up when both quantized (NF4 fused, Q4L individual dp4a)
    if (!layer.gate_proj_fp16 && !layer.up_proj_fp16 && !weights_.is_q4l) {
        auto& g = layer.gate_proj_nf4; auto& u = layer.up_proj_nf4;
        FUSED2_4BIT(g, state_.gate_buf, u, state_.up_buf, norm_out, g.in_dim, stream);
    } else {
        if (layer.gate_proj_fp16) cublas_hgemv_lora(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_gate, state_.lora_scratch);
        else { auto& w = layer.gate_proj_nf4; GEMV_4BIT(w, norm_out, state_.gate_buf, stream); }
        if (layer.up_proj_fp16) cublas_hgemv_lora(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_up, state_.lora_scratch);
        else { auto& w = layer.up_proj_nf4; GEMV_4BIT(w, norm_out, state_.up_buf, stream); }
    }
    launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
    // Quantize SiLU output for down projection dp4a (dim=INTERMEDIATE_SIZE)
    if (!layer.down_proj_fp16 && weights_.is_q4l) {
        launch_quantize_input_q8(state_.gate_buf, state_.q8_data, state_.q8_scales,
                                 state_.q8_sums, INTERMEDIATE_SIZE, stream);
    }
    if (layer.down_proj_fp16) cublas_hgemv_lora(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, layer.lora_down, state_.lora_scratch);
    else { auto& w = layer.down_proj_nf4; GEMV_4BIT(w, state_.gate_buf, state_.hidden, stream); }

    // 12. Residual add
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);
}

// ============================================================================
// Decode: process one token
// ============================================================================

void InferenceEngine::decode(int token_id) {
    // Use CUDA graph if captured (eliminates kernel launch overhead)
    if (use_cuda_graph_ && graph_captured_) {
        decode_graph(token_id);
        return;
    }

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

    // LM head -> fp32 logits (always cuBLAS fp16: 3.1ms vs 6.5ms for NF4)
    {
        ensure_cublas();
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE,
                     &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta,
                     state_.logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    state_.current_pos++;
}

// ============================================================================
// Profiled decode: measure each operation category with CUDA events
// ============================================================================

std::vector<std::pair<std::string, float>> InferenceEngine::profile_decode(int token_id) {
    cudaStream_t stream = 0;
    std::vector<std::pair<std::string, float>> timings;

    auto timed = [&](const char* name, auto fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        timings.push_back({name, ms * 1000.0f}); // microseconds
        cudaEventDestroy(start); cudaEventDestroy(stop);
    };

    timed("embedding", [&]{ launch_embedding(weights_.embed_tokens, token_id, state_.hidden, HIDDEN_SIZE, stream); });

    float total_nf4_gemv = 0, total_attention = 0, total_other = 0;

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& layer = weights_.layers[i];
        auto& kv = state_.kv_cache[i];
        half* norm_out = state_.ffn_out;

        // Norm + residual save
        timed("norm+copy", [&]{
            launch_copy_rms_norm(state_.hidden, layer.input_layernorm,
                                state_.residual, norm_out, HIDDEN_SIZE, RMS_NORM_EPS, stream);
        });

        // QKV
        timed("qkv_gemv", [&]{
            if (!layer.q_proj_fp16 && !layer.k_proj_fp16 && !layer.v_proj_fp16) {
                auto& q = layer.q_proj_nf4; auto& k = layer.k_proj_nf4; auto& v = layer.v_proj_nf4;
                launch_nf4_fused_3(q.data, q.absmax, state_.q_buf, q.out_dim, k.data, k.absmax, state_.k_buf, k.out_dim, v.data, v.absmax, state_.v_buf, v.out_dim, norm_out, q.in_dim, stream);
            } else {
                if (layer.q_proj_fp16) cublas_hgemv_lora(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, layer.lora_q, state_.lora_scratch);
                else { auto& w = layer.q_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.q_buf, w.out_dim, w.in_dim, w.block_size, stream); }
                if (layer.k_proj_fp16) cublas_hgemv_lora(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, layer.lora_k, state_.lora_scratch);
                else { auto& w = layer.k_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.k_buf, w.out_dim, w.in_dim, w.block_size, stream); }
                if (layer.v_proj_fp16) cublas_hgemv_lora(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, layer.lora_v, state_.lora_scratch);
                else { auto& w = layer.v_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.v_buf, w.out_dim, w.in_dim, w.block_size, stream); }
            }
        });

        // QKNorm + RoPE + KV cache + Attention
        timed("attn_ops", [&]{
            launch_qk_norm(state_.q_buf, state_.k_buf, layer.q_norm, layer.k_norm, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_NORM_EPS, stream);
            launch_rope(state_.q_buf, state_.k_buf, state_.rope_cos, state_.rope_sin, state_.current_pos, state_.max_seq_len, stream);
            cudaMemcpyAsync(kv.key + state_.current_pos * KV_DIM, state_.k_buf, KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(kv.value + state_.current_pos * KV_DIM, state_.v_buf, KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            launch_gqa_attention(state_.q_buf, kv.key, kv.value, state_.attn_out, state_.attn_scores, state_.current_pos, state_.max_seq_len, stream);
        });

        // O proj
        timed("o_gemv", [&]{
            if (layer.o_proj_fp16) cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
            else { auto& w = layer.o_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, state_.attn_out, state_.hidden, w.out_dim, w.in_dim, w.block_size, stream); }
        });

        // Residual + norm
        timed("res+norm2", [&]{
            launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);
            launch_copy_rms_norm(state_.hidden, layer.post_attn_layernorm,
                                state_.residual, norm_out, HIDDEN_SIZE, RMS_NORM_EPS, stream);
        });

        // Gate+Up
        timed("gate_up_gemv", [&]{
            if (!layer.gate_proj_fp16 && !layer.up_proj_fp16) {
                auto& g = layer.gate_proj_nf4; auto& u = layer.up_proj_nf4;
                launch_nf4_fused_2(g.data, g.absmax, state_.gate_buf, g.out_dim, u.data, u.absmax, state_.up_buf, u.out_dim, norm_out, g.in_dim, stream);
            } else {
                if (layer.gate_proj_fp16) cublas_hgemv_lora(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_gate, state_.lora_scratch);
                else { auto& w = layer.gate_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.gate_buf, w.out_dim, w.in_dim, w.block_size, stream); }
                if (layer.up_proj_fp16) cublas_hgemv_lora(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_up, state_.lora_scratch);
                else { auto& w = layer.up_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.up_buf, w.out_dim, w.in_dim, w.block_size, stream); }
            }
        });

        // SiLU + down
        timed("silu+down", [&]{
            launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
            if (layer.down_proj_fp16) cublas_hgemv_lora(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, layer.lora_down, state_.lora_scratch);
            else { auto& w = layer.down_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, state_.gate_buf, state_.hidden, w.out_dim, w.in_dim, w.block_size, stream); }
            launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);
        });
    }

    // Final norm + LM head
    half* norm_out = state_.ffn_out;
    timed("final_norm", [&]{ launch_rms_norm(state_.hidden, weights_.final_layernorm, norm_out, HIDDEN_SIZE, RMS_NORM_EPS, stream); });

    timed("lm_head", [&]{
        ensure_cublas();
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, VOCAB_SIZE, 1, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE, norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta, state_.logits, CUDA_R_32F, VOCAB_SIZE, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    });

    state_.current_pos++;

    // Aggregate per-layer timings
    float agg_norm = 0, agg_qkv = 0, agg_attn = 0, agg_o = 0, agg_res_norm = 0, agg_gate_up = 0, agg_silu_down = 0;
    for (auto& [name, us] : timings) {
        if (name == "norm+copy") agg_norm += us;
        else if (name == "qkv_gemv") agg_qkv += us;
        else if (name == "attn_ops") agg_attn += us;
        else if (name == "o_gemv") agg_o += us;
        else if (name == "res+norm2") agg_res_norm += us;
        else if (name == "gate_up_gemv") agg_gate_up += us;
        else if (name == "silu+down") agg_silu_down += us;
    }

    std::vector<std::pair<std::string, float>> summary;
    summary.push_back({"embedding", timings[0].second});
    summary.push_back({"norm+copy (28L)", agg_norm});
    summary.push_back({"QKV GEMV (28L)", agg_qkv});
    summary.push_back({"attn_ops (28L)", agg_attn});
    summary.push_back({"O GEMV (28L)", agg_o});
    summary.push_back({"res+norm2 (28L)", agg_res_norm});
    summary.push_back({"gate+up GEMV (28L)", agg_gate_up});
    summary.push_back({"silu+down (28L)", agg_silu_down});
    summary.push_back({"final_norm", timings[timings.size()-2].second});
    summary.push_back({"lm_head", timings[timings.size()-1].second});
    return summary;
}

// ============================================================================
// Graph-captured forward layer (uses device-side position)
// ============================================================================

void InferenceEngine::forward_layer_graph(int layer_idx, cudaStream_t stream) {
    auto& layer = weights_.layers[layer_idx];
    auto& kv = state_.kv_cache[layer_idx];

    // Fused: copy hidden → residual AND RMSNorm → norm_out
    half* norm_out = state_.ffn_out;
    launch_copy_rms_norm(state_.hidden, layer.input_layernorm,
                         state_.residual, norm_out, HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // QKV projections — fused when all 3 are NF4
    if (!layer.q_proj_fp16 && !layer.k_proj_fp16 && !layer.v_proj_fp16) {
        auto& q = layer.q_proj_nf4; auto& k = layer.k_proj_nf4; auto& v = layer.v_proj_nf4;
        launch_nf4_fused_3(
            q.data, q.absmax, state_.q_buf, q.out_dim,
            k.data, k.absmax, state_.k_buf, k.out_dim,
            v.data, v.absmax, state_.v_buf, v.out_dim,
            norm_out, q.in_dim, stream);
    } else {
        if (layer.q_proj_fp16) cublas_hgemv_lora(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, layer.lora_q, state_.lora_scratch);
        else { auto& w = layer.q_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.q_buf, w.out_dim, w.in_dim, w.block_size, stream); }
        if (layer.k_proj_fp16) cublas_hgemv_lora(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, layer.lora_k, state_.lora_scratch);
        else { auto& w = layer.k_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.k_buf, w.out_dim, w.in_dim, w.block_size, stream); }
        if (layer.v_proj_fp16) cublas_hgemv_lora(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, layer.lora_v, state_.lora_scratch);
        else { auto& w = layer.v_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.v_buf, w.out_dim, w.in_dim, w.block_size, stream); }
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
    if (layer.o_proj_fp16) cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
    else { auto& w = layer.o_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, state_.attn_out, state_.hidden, w.out_dim, w.in_dim, w.block_size, stream); }

    // Residual add
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // Post-attention: copy hidden → residual AND RMSNorm → norm_out
    launch_copy_rms_norm(state_.hidden, layer.post_attn_layernorm,
                         state_.residual, norm_out, HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // Gate + Up — fused when both NF4
    if (!layer.gate_proj_fp16 && !layer.up_proj_fp16) {
        auto& g = layer.gate_proj_nf4; auto& u = layer.up_proj_nf4;
        launch_nf4_fused_2(
            g.data, g.absmax, state_.gate_buf, g.out_dim,
            u.data, u.absmax, state_.up_buf, u.out_dim,
            norm_out, g.in_dim, stream);
    } else {
        if (layer.gate_proj_fp16) cublas_hgemv_lora(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_gate, state_.lora_scratch);
        else { auto& w = layer.gate_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.gate_buf, w.out_dim, w.in_dim, w.block_size, stream); }
        if (layer.up_proj_fp16) cublas_hgemv_lora(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, layer.lora_up, state_.lora_scratch);
        else { auto& w = layer.up_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, norm_out, state_.up_buf, w.out_dim, w.in_dim, w.block_size, stream); }
    }
    launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf, INTERMEDIATE_SIZE, stream);
    if (layer.down_proj_fp16) cublas_hgemv_lora(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, layer.lora_down, state_.lora_scratch);
    else { auto& w = layer.down_proj_nf4; launch_nf4_gemv_fast(w.data, w.absmax, state_.gate_buf, state_.hidden, w.out_dim, w.in_dim, w.block_size, stream); }

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

    // LM head: always cuBLAS fp16 (2x faster than NF4 GEMV for this large matrix)
    auto lm_head_on_stream = [&](cudaStream_t s) {
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     norm_out, CUDA_R_16F, HIDDEN_SIZE,
                     &beta, state_.logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };

    lm_head_on_stream(graph_stream_);
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
    lm_head_on_stream(graph_stream_);

    cudaStreamEndCapture(graph_stream_, &cuda_graph_);
    cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0);

    // Keep cuBLAS on graph_stream_ for replay (LM head + fp16 projections)
    // This ensures cuBLAS operations are part of the graph replay
    // (cuBLAS stream is NOT restored to 0 — all ops use graph_stream_ when graph is active)

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

    // Async update device-side control values via pinned memory
    // (pinned memory ensures cudaMemcpyAsync is truly async)
    *h_token_id_pinned_ = token_id;
    *h_pos_pinned_ = state_.current_pos;
    cudaMemcpyAsync(state_.d_token_id, h_token_id_pinned_, sizeof(int),
                    cudaMemcpyHostToDevice, graph_stream_);
    cudaMemcpyAsync(state_.d_pos, h_pos_pinned_, sizeof(int),
                    cudaMemcpyHostToDevice, graph_stream_);

    // Replay the captured graph (all on graph_stream_, fully pipelined)
    cudaGraphLaunch(cuda_graph_exec_, graph_stream_);
    // NO sync here — let GPU run while CPU prepares next token
    // Sync happens in sample_gpu() when reading the result

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
    // Use graph_stream_ if CUDA graph is active, otherwise default stream
    cudaStream_t s = use_cuda_graph_ ? graph_stream_ : 0;
    launch_argmax(state_.logits, state_.sample_result, VOCAB_SIZE, s);
    // Sync the specific stream and copy result
    if (s) cudaStreamSynchronize(s);
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

    // Use graph_stream_ if CUDA graph is active, otherwise default stream
    cudaStream_t s = use_cuda_graph_ ? graph_stream_ : 0;

    // GPU kernel: temperature + softmax + cumulative sample
    // NOTE: this modifies logits in-place (they become probabilities)
    launch_gpu_sample(state_.logits, state_.sample_result, VOCAB_SIZE,
                      temperature, r, top_p, s);
    // Sync the specific stream and copy result
    if (s) cudaStreamSynchronize(s);
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

    // CUDA graph disabled for NF4 on Jetson Orin: graph replay is 14% slower
    // than the non-graph path because sample→decode serialization prevents
    // CPU pipelining. The non-graph path benefits from 0.5ms of CPU kernel
    // submission overlapping with GPU execution.
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

// ============================================================================
// Batched generation: G sequences in parallel (GEMM, tensor cores)
// ============================================================================

void InferenceEngine::alloc_batch(int G, int max_seq_len) {
    if (batch_ && batch_->G >= G && batch_->max_seq_len >= max_seq_len) return;
    if (batch_) { /* TODO: free old */ }
    batch_ = new BatchState();
    batch_->G = G;
    batch_->max_seq_len = max_seq_len;

    cudaMalloc(&batch_->hidden, HIDDEN_SIZE * G * sizeof(half));
    cudaMalloc(&batch_->residual, HIDDEN_SIZE * G * sizeof(half));
    cudaMalloc(&batch_->norm_buf, HIDDEN_SIZE * G * sizeof(half));
    cudaMalloc(&batch_->q_buf, Q_DIM * G * sizeof(half));
    cudaMalloc(&batch_->k_buf, KV_DIM * G * sizeof(half));
    cudaMalloc(&batch_->v_buf, KV_DIM * G * sizeof(half));
    cudaMalloc(&batch_->attn_out, Q_DIM * G * sizeof(half));
    cudaMalloc(&batch_->gate_buf, INTERMEDIATE_SIZE * G * sizeof(half));
    cudaMalloc(&batch_->up_buf, INTERMEDIATE_SIZE * G * sizeof(half));
    cudaMalloc(&batch_->logits, VOCAB_SIZE * G * sizeof(float));
    cudaMalloc(&batch_->attn_scores, G * NUM_HEADS * max_seq_len * sizeof(float));
    // Dequant scratch: largest projection
    cudaMalloc(&batch_->dequant_scratch, INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(half));
    // KV caches
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMalloc(&batch_->kv_keys[i], (size_t)G * max_seq_len * KV_DIM * sizeof(half));
        cudaMalloc(&batch_->kv_values[i], (size_t)G * max_seq_len * KV_DIM * sizeof(half));
    }
    // Per-sequence state
    batch_->h_positions = new int[G]();
    cudaMalloc(&batch_->d_positions, G * sizeof(int));
    batch_->h_tokens = new int[G]();
    cudaMalloc(&batch_->d_tokens, G * sizeof(int));
    batch_->h_finished = new bool[G]();

    std::cout << "  Batch allocated: G=" << G << " max_seq=" << max_seq_len
              << " KV=" << (G * max_seq_len * KV_DIM * 2 * NUM_LAYERS * 2 / 1e6) << "MB" << std::endl;
}

void InferenceEngine::batch_gemm(half* out, const half* weight, const half* in,
                                  int M, int N, int K, cudaStream_t stream) {
    ensure_cublas();
    // weight is (M, K) row-major = (K, M) col-major
    // in is (K, N) col-major (already correct)
    // out is (M, N) col-major
    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 M, N, K, &alpha,
                 weight, CUDA_R_16F, K,   // A^T: (K,M) -> (M,K)
                 in, CUDA_R_16F, K,       // B: (K,N)
                 &beta, out, CUDA_R_16F, M,  // C: (M,N)
                 CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void InferenceEngine::batch_gemm_q4l(half* out, const NF4Weight& w, const half* in,
                                      int N, cudaStream_t stream) {
    // Dequantize Q4L to fp16 scratch, then cuBLAS GEMM
    launch_dequant_q4l(batch_->dequant_scratch, w.data, w.absmax,
                       w.out_dim, w.in_dim, stream);
    batch_gemm(out, batch_->dequant_scratch, in, w.out_dim, N, w.in_dim, stream);
}

void InferenceEngine::forward_layer_batch(int layer_idx, int G, cudaStream_t stream) {
    auto& L = weights_.layers[layer_idx];
    auto* B = batch_;

    // 1. Copy hidden -> residual, RMSNorm -> norm_buf
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.input_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // 2. QKV projections (GEMM)
    auto project = [&](half* out, half* fp16w, NF4Weight& nf4w, int out_dim, int in_dim) {
        if (fp16w) batch_gemm(out, fp16w, B->norm_buf, out_dim, G, in_dim, stream);
        else batch_gemm_q4l(out, nf4w, B->norm_buf, G, stream);
    };
    project(B->q_buf, L.q_proj_fp16, L.q_proj_nf4, Q_DIM, HIDDEN_SIZE);
    project(B->k_buf, L.k_proj_fp16, L.k_proj_nf4, KV_DIM, HIDDEN_SIZE);
    project(B->v_buf, L.v_proj_fp16, L.v_proj_nf4, KV_DIM, HIDDEN_SIZE);

    // 3. QKNorm + RoPE
    launch_qk_norm_batch(B->q_buf, B->k_buf, L.q_norm, L.k_norm,
                         NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, G, RMS_NORM_EPS, stream);
    launch_rope_batch(B->q_buf, B->k_buf, state_.rope_cos, state_.rope_sin,
                      B->d_positions, B->max_seq_len, G, stream);

    // 4. KV cache write
    launch_kv_cache_write_batch(B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                 B->k_buf, B->v_buf, B->d_positions, B->max_seq_len, G, stream);

    // 5. GQA attention
    launch_gqa_attention_batch(B->attn_out, B->q_buf,
                                B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                B->attn_scores, B->d_positions, B->max_seq_len, G, stream);

    // 6. Output projection
    project(B->hidden, L.o_proj_fp16, L.o_proj_nf4, HIDDEN_SIZE, Q_DIM);

    // 7. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);

    // 8. Post-attention norm
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // 9. FFN: gate, up, SiLU, down
    project(B->gate_buf, L.gate_proj_fp16, L.gate_proj_nf4, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    project(B->up_buf, L.up_proj_fp16, L.up_proj_nf4, INTERMEDIATE_SIZE, HIDDEN_SIZE);
    launch_silu_mul_batch(B->gate_buf, B->up_buf, INTERMEDIATE_SIZE * G, stream);
    project(B->hidden, L.down_proj_fp16, L.down_proj_nf4, HIDDEN_SIZE, INTERMEDIATE_SIZE);

    // 10. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);
}

void InferenceEngine::decode_batch(int G) {
    cudaStream_t stream = 0;
    auto* B = batch_;

    // Embedding
    launch_embed_batch(B->hidden, weights_.embed_tokens, B->d_tokens, G, stream);

    // Forward layers
    for (int i = 0; i < NUM_LAYERS; i++)
        forward_layer_batch(i, G, stream);

    // Final norm
    launch_rms_norm_batch(B->norm_buf, B->hidden, weights_.final_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // LM head: (VOCAB_SIZE, HIDDEN_SIZE) @ (HIDDEN_SIZE, G) -> (VOCAB_SIZE, G) in fp32
    {
        ensure_cublas();
        float alpha = 1.0f, beta = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, G, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     B->norm_buf, CUDA_R_16F, HIDDEN_SIZE,
                     &beta, B->logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

std::vector<std::vector<int>> InferenceEngine::generate_batch(
    const std::vector<std::vector<int>>& prompts,
    int max_new_tokens, float temperature, float top_p, int eos_token_id
) {
    int G = prompts.size();
    int max_prompt_len = 0;
    for (auto& p : prompts) max_prompt_len = std::max(max_prompt_len, (int)p.size());
    int total_max_len = max_prompt_len + max_new_tokens;

    alloc_batch(G, total_max_len);
    auto* B = batch_;

    // Reset
    for (int i = 0; i < G; i++) { B->h_positions[i] = 0; B->h_finished[i] = false; }
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMemset(B->kv_keys[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
        cudaMemset(B->kv_values[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
    }

    std::vector<std::vector<int>> outputs(G);

    // Phase 1: Prefill (token by token, all G sequences in parallel)
    for (int t = 0; t < max_prompt_len; t++) {
        for (int g = 0; g < G; g++)
            B->h_tokens[g] = (t < (int)prompts[g].size()) ? prompts[g][t] : 0;
        cudaMemcpy(B->d_tokens, B->h_tokens, G * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);

        decode_batch(G);

        for (int g = 0; g < G; g++)
            if (t < (int)prompts[g].size()) B->h_positions[g]++;
    }

    // Phase 2: Decode (generate new tokens)
    for (int step = 0; step < max_new_tokens; step++) {
        // Sample from logits
        launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, 0);
        cudaMemcpy(B->h_tokens, B->d_tokens, G * sizeof(int), cudaMemcpyDeviceToHost);

        // Check stopping
        bool all_done = true;
        for (int g = 0; g < G; g++) {
            if (!B->h_finished[g]) {
                outputs[g].push_back(B->h_tokens[g]);
                if (B->h_tokens[g] == eos_token_id) B->h_finished[g] = true;
                else all_done = false;
            }
        }
        if (all_done) break;

        // Advance positions
        for (int g = 0; g < G; g++)
            if (!B->h_finished[g]) B->h_positions[g]++;
        cudaMemcpy(B->d_tokens, B->h_tokens, G * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);

        decode_batch(G);
    }

    return outputs;
}
