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
#include <stdexcept>

// CUDA error checking macro — converts silent corruption into actionable errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error [%s:%d]: status=%d\n", \
                __FILE__, __LINE__, (int)status); \
        throw std::runtime_error("cuBLAS error"); \
    } \
} while(0)

// cuBLAS handle (created once, reused)
static cublasHandle_t cublas_handle = nullptr;

static void* cublas_workspace = nullptr;
static const size_t CUBLAS_WORKSPACE_SIZE = 4 * 1024 * 1024; // 4MB

static cudaStream_t cublas_current_stream = 0;

static void ensure_cublas(cudaStream_t stream = 0) {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
        cudaMalloc(&cublas_workspace, CUBLAS_WORKSPACE_SIZE);
        cublasSetWorkspace(cublas_handle, cublas_workspace, CUBLAS_WORKSPACE_SIZE);
    }
    if (stream != cublas_current_stream) {
        cublasSetStream(cublas_handle, stream);
        cublas_current_stream = stream;
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

// Helper to avoid void** casts everywhere
template<typename T>
inline void cudaMallocChecked(T** ptr, size_t size) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(ptr), size));
}

// Forward declarations for kernel launchers (defined in kernels.cu)
extern "C" {
    void launch_fp16_gemv(const half* weight, const half* input, half* output, int out_dim, int in_dim, cudaStream_t stream);
    void launch_nf4_gemv_fast(const uint8_t* packed, const float* absmax, const half* input, half* output, int out_dim, int in_dim, int block_size, cudaStream_t stream);
    void launch_nf4_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_nf4_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_quantize_input_q8(const half* input, int8_t* q8_data, float* q8_scales, float* q8_sums, int dim, cudaStream_t stream);
    void launch_q4l_dp4a_gemv(const uint8_t* w, const float* w_scales, const int8_t* q8, const float* q8_sc, const float* q8_sm, half* y, int out_dim, int in_dim, cudaStream_t stream);
    void launch_dequant_q4l(half* out, const uint8_t* data, const float* scales, int out_dim, int in_dim, cudaStream_t stream);

    // Batched kernel launchers
    void launch_embed_batch(half* h, const half* et, const int* tok, int G, int hidden_size, cudaStream_t s);
    void launch_rms_norm_batch(half* out, const half* in, const half* w, int dim, int G, float eps, cudaStream_t s);
    void launch_copy_batch(half* dst, const half* src, int total, cudaStream_t s);
    void launch_residual_add_batch(half* out, const half* res, int total, cudaStream_t s);
    void launch_qk_norm_batch(half* q, half* k, const half* qw, const half* kw, int nq, int nkv, int hd, int G, float eps, int q_dim, int kv_dim, cudaStream_t s);
    void launch_rope_batch(half* q, half* k, const half* ct, const half* st, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s);
    void launch_kv_cache_write_batch(half* ck, half* cv, const half* k, const half* v, const int* pos, int msl, int G, int kv_dim, cudaStream_t s);
    void launch_gqa_attention_batch(half* out, const half* q, const half* ck, const half* cv, float* as, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s);
    void launch_silu_mul_batch(half* gate, const half* up, int total, cudaStream_t s);
    void launch_argmax_batch(const float* logits, int* tokens, int vocab, int G, cudaStream_t s);
    void launch_sample_batch(float* logits, int* tokens, const float* randoms, int vocab, int G, float temperature, float top_p, cudaStream_t s);
    void launch_rms_norm(const half* input, const half* weight, half* output, int dim, float eps, cudaStream_t stream);
    void launch_copy_rms_norm(const half* input, const half* weight, half* residual, half* norm_out, int dim, float eps, cudaStream_t stream);
    void launch_qk_norm(half* q, half* k, const half* q_weight, const half* k_weight, int num_q_heads, int num_kv_heads, int head_dim, float eps, cudaStream_t stream);
    void launch_rope(half* q, half* k, const half* cos_table, const half* sin_table, int pos, int max_seq_len, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream);
    void launch_gqa_attention(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, int pos, int max_seq_len, int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream);
    void launch_silu_gate_mul(const half* gate, const half* up, half* output, int dim, cudaStream_t stream);
    void launch_embedding(const half* embed_table, int token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_residual_add(half* output, const half* residual, int dim, cudaStream_t stream);
    void launch_fp16_to_fp32(const half* input, float* output, int n, cudaStream_t stream);
    void launch_argmax(const float* logits, int* result, int vocab_size, cudaStream_t stream);
    void launch_gpu_sample(float* logits, int* result, int vocab_size, float temperature, float random_val, float top_p, cudaStream_t stream);
}

// JSON config parser (simple, no external dependency)
#include <fstream>
#include <sstream>

ModelConfig ModelConfig::from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "WARNING: config not found at " << path << ", using Qwen3-0.6B defaults" << std::endl;
        return qwen3_0_6b();
    }
    std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    // Minimal JSON parser for flat int/float fields
    auto get_int = [&](const char* key) -> int {
        std::string needle = std::string("\"") + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) return -1;
        pos = json.find(':', pos);
        return std::stoi(json.substr(pos + 1));
    };
    auto get_float = [&](const char* key) -> float {
        std::string needle = std::string("\"") + key + "\"";
        auto pos = json.find(needle);
        if (pos == std::string::npos) return -1.0f;
        pos = json.find(':', pos);
        return std::stof(json.substr(pos + 1));
    };

    ModelConfig c;
    c.hidden_size = get_int("hidden_size");
    c.intermediate_size = get_int("intermediate_size");
    c.num_layers = get_int("num_hidden_layers");
    c.num_heads = get_int("num_attention_heads");
    c.num_kv_heads = get_int("num_key_value_heads");
    c.head_dim = get_int("head_dim");
    c.vocab_size = get_int("vocab_size");
    c.rms_norm_eps = get_float("rms_norm_eps");
    c.rope_theta = get_float("rope_theta");

    // Fallback for head_dim if not in config
    if (c.head_dim <= 0 && c.hidden_size > 0 && c.num_heads > 0)
        c.head_dim = c.hidden_size / c.num_heads;

    std::cout << "  Config: " << c.hidden_size << "h, " << c.intermediate_size << "i, "
              << c.num_layers << "L, " << c.num_heads << "Qh, " << c.num_kv_heads << "KVh, "
              << c.head_dim << "hd, " << c.vocab_size << "V" << std::endl;
    return c;
}

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

// Runtime config aliases — these macros replace the old compile-time qwen3:: namespace.
// They expand to config_ member access, so they work in all InferenceEngine methods.
#define HIDDEN_SIZE config_.hidden_size
#define INTERMEDIATE_SIZE config_.intermediate_size
#define NUM_LAYERS config_.num_layers
#define NUM_HEADS config_.num_heads
#define NUM_KV_HEADS config_.num_kv_heads
#define HEAD_DIM config_.head_dim
#define VOCAB_SIZE config_.vocab_size
#define Q_DIM config_.q_dim()
#define KV_DIM config_.kv_dim()
#define RMS_NORM_EPS config_.rms_norm_eps
#define ROPE_THETA config_.rope_theta
#define GQA_GROUPS config_.gqa_groups()

// ============================================================================
// Constructor / Destructor
// ============================================================================

InferenceEngine::InferenceEngine(int max_seq_len) {
    // Config and buffers are allocated in load_weights() when we know the model dims.
    // Here we just store max_seq_len and zero-init pointers.
    config_ = ModelConfig::qwen3_0_6b(); // default, overwritten by load_weights
    state_ = {};
    state_.max_seq_len = max_seq_len;
    state_.current_pos = 0;
    batch_ = nullptr;
    cudaStreamCreate(&engine_stream_);
}

void InferenceEngine::allocate_buffers() {
    int max_seq_len = state_.max_seq_len;

    // Allocate KV caches
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMallocChecked(&state_.kv_cache[i].key, max_seq_len * KV_DIM * sizeof(half));
        cudaMallocChecked(&state_.kv_cache[i].value, max_seq_len * KV_DIM * sizeof(half));
    }

    // Allocate scratch buffers
    cudaMallocChecked(&state_.hidden, HIDDEN_SIZE * sizeof(half));
    cudaMallocChecked(&state_.residual, HIDDEN_SIZE * sizeof(half));
    cudaMallocChecked(&state_.q_buf, Q_DIM * sizeof(half));
    cudaMallocChecked(&state_.k_buf, KV_DIM * sizeof(half));
    cudaMallocChecked(&state_.v_buf, KV_DIM * sizeof(half));
    cudaMallocChecked(&state_.attn_out, Q_DIM * sizeof(half));
    cudaMallocChecked(&state_.gate_buf, INTERMEDIATE_SIZE * sizeof(half));
    cudaMallocChecked(&state_.up_buf, INTERMEDIATE_SIZE * sizeof(half));
    cudaMallocChecked(&state_.ffn_out, HIDDEN_SIZE * sizeof(half));
    cudaMallocChecked(&state_.logits, VOCAB_SIZE * sizeof(float));
    cudaMallocChecked(&state_.attn_scores, NUM_HEADS * max_seq_len * sizeof(float));

    // Allocate and precompute RoPE tables
    cudaMallocChecked(&state_.rope_cos, max_seq_len * (HEAD_DIM / 2) * sizeof(half));
    cudaMallocChecked(&state_.rope_sin, max_seq_len * (HEAD_DIM / 2) * sizeof(half));
    precompute_rope();

    // GPU-side sampling
    cudaMallocChecked(&state_.sample_result, sizeof(int));

    // dp4a input quantization buffers
    int max_q8_dim = INTERMEDIATE_SIZE; // largest input dimension
    int max_q8_blocks = max_q8_dim / 64;
    cudaMallocChecked(&state_.q8_data, max_q8_dim * sizeof(int8_t));
    cudaMallocChecked(&state_.q8_scales, max_q8_blocks * sizeof(float));
    cudaMallocChecked(&state_.q8_sums, max_q8_blocks * sizeof(float));

    // LoRA scratch
    cudaMallocChecked(&state_.lora_scratch, 64 * sizeof(half)); // max rank 64

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
    cudaFree(state_.q8_data);
    cudaFree(state_.q8_scales);
    cudaFree(state_.q8_sums);
    cudaFree(state_.lora_scratch);

    // Free batch state
    if (batch_) {
        if (batch_->d_all_randoms) cudaFree(batch_->d_all_randoms);
        delete[] batch_->h_positions; delete[] batch_->h_tokens;
        delete[] batch_->h_finished; delete[] batch_->h_randoms;
        delete batch_;
    }
    if (batch_arena_.base) cudaFree(batch_arena_.base);
    if (decode_graph_exec_) cudaGraphExecDestroy(decode_graph_exec_);
    if (decode_graph_) cudaGraphDestroy(decode_graph_);
    if (engine_stream_) cudaStreamDestroy(engine_stream_);
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

void InferenceEngine::share_embedding(void* external_embed_ptr) {
    if (weights_.embed_tokens && !embed_is_external_) {
        cudaFree(weights_.embed_tokens);
    }
    weights_.embed_tokens = (half*)external_embed_ptr;
    embed_is_external_ = true;
    std::cout << "  Embedding shared from PyTorch (saved ~311MB)" << std::endl;
}

void InferenceEngine::reset() {
    state_.current_pos = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaMemset(state_.kv_cache[i].key, 0, state_.max_seq_len * KV_DIM * sizeof(half));
        cudaMemset(state_.kv_cache[i].value, 0, state_.max_seq_len * KV_DIM * sizeof(half));
    }
}

// ============================================================================
// Sleep/wake: free GPU buffers during training, re-allocate for generation
// ============================================================================

void InferenceEngine::sleep() {
    // Free batch state (GPU memory is one arena block)
    if (batch_) {
        delete[] batch_->h_positions; delete[] batch_->h_tokens;
        delete[] batch_->h_finished; delete[] batch_->h_randoms;
        delete batch_;
        batch_ = nullptr;
    }
    if (batch_arena_.base) {
        cudaFree(batch_arena_.base);
        batch_arena_.base = nullptr;
        batch_arena_.capacity = 0;
        batch_arena_.offset = 0;
    }

    // Free single-sequence KV caches
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaFree(state_.kv_cache[i].key); state_.kv_cache[i].key = nullptr;
        cudaFree(state_.kv_cache[i].value); state_.kv_cache[i].value = nullptr;
    }

    // Free single-sequence scratch buffers
    cudaFree(state_.hidden); state_.hidden = nullptr;
    cudaFree(state_.residual); state_.residual = nullptr;
    cudaFree(state_.q_buf); state_.q_buf = nullptr;
    cudaFree(state_.k_buf); state_.k_buf = nullptr;
    cudaFree(state_.v_buf); state_.v_buf = nullptr;
    cudaFree(state_.attn_out); state_.attn_out = nullptr;
    cudaFree(state_.gate_buf); state_.gate_buf = nullptr;
    cudaFree(state_.up_buf); state_.up_buf = nullptr;
    cudaFree(state_.ffn_out); state_.ffn_out = nullptr;
    cudaFree(state_.logits); state_.logits = nullptr;
    cudaFree(state_.attn_scores); state_.attn_scores = nullptr;
    cudaFree(state_.sample_result); state_.sample_result = nullptr;
    cudaFree(state_.q8_data); state_.q8_data = nullptr;
    cudaFree(state_.q8_scales); state_.q8_scales = nullptr;
    cudaFree(state_.q8_sums); state_.q8_sums = nullptr;
    cudaFree(state_.lora_scratch); state_.lora_scratch = nullptr;
    // Keep rope_cos/rope_sin (small, reused across wake cycles)

    // Free cached fp16 weights
    if (weights_cached_) {
        auto free_cache = [](NF4Weight& w) {
            if (w.fp16_cache) { cudaFree(w.fp16_cache); w.fp16_cache = nullptr; }
        };
        for (int i = 0; i < NUM_LAYERS; i++) {
            auto& L = weights_.layers[i];
            free_cache(L.q_proj_nf4); free_cache(L.k_proj_nf4);
            free_cache(L.v_proj_nf4); free_cache(L.o_proj_nf4);
            free_cache(L.gate_proj_nf4); free_cache(L.up_proj_nf4);
            free_cache(L.down_proj_nf4);
        }
        weights_cached_ = false;
    }

    cudaDeviceSynchronize();
}

void InferenceEngine::wake() {
    // Re-allocate single-sequence buffers (if they were freed)
    if (!state_.hidden) {
        int max_seq_len = state_.max_seq_len;
        for (int i = 0; i < NUM_LAYERS; i++) {
            cudaMallocChecked(&state_.kv_cache[i].key, max_seq_len * KV_DIM * sizeof(half));
            cudaMallocChecked(&state_.kv_cache[i].value, max_seq_len * KV_DIM * sizeof(half));
        }
        cudaMallocChecked(&state_.hidden, HIDDEN_SIZE * sizeof(half));
        cudaMallocChecked(&state_.residual, HIDDEN_SIZE * sizeof(half));
        cudaMallocChecked(&state_.q_buf, Q_DIM * sizeof(half));
        cudaMallocChecked(&state_.k_buf, KV_DIM * sizeof(half));
        cudaMallocChecked(&state_.v_buf, KV_DIM * sizeof(half));
        cudaMallocChecked(&state_.attn_out, Q_DIM * sizeof(half));
        cudaMallocChecked(&state_.gate_buf, INTERMEDIATE_SIZE * sizeof(half));
        cudaMallocChecked(&state_.up_buf, INTERMEDIATE_SIZE * sizeof(half));
        cudaMallocChecked(&state_.ffn_out, HIDDEN_SIZE * sizeof(half));
        cudaMallocChecked(&state_.logits, VOCAB_SIZE * sizeof(float));
        cudaMallocChecked(&state_.attn_scores, NUM_HEADS * max_seq_len * sizeof(float));
        cudaMallocChecked(&state_.sample_result, sizeof(int));
        int max_q8_dim = INTERMEDIATE_SIZE;
        cudaMallocChecked(&state_.q8_data, max_q8_dim * sizeof(int8_t));
        cudaMallocChecked(&state_.q8_scales, (max_q8_dim / 64) * sizeof(float));
        cudaMallocChecked(&state_.q8_sums, (max_q8_dim / 64) * sizeof(float));
        cudaMallocChecked(&state_.lora_scratch, 64 * sizeof(half));
    }
    // Batch state is re-allocated lazily by alloc_batch() on next generate_batch() call
}

// ============================================================================
// Forward pass through one transformer layer
// ============================================================================

void InferenceEngine::forward_layer(int layer_idx) {
    auto& layer = weights_.layers[layer_idx];
    auto& kv = state_.kv_cache[layer_idx];
    cudaStream_t stream = 0; // default stream

    // 1. Input LayerNorm
    half* norm_out = state_.ffn_out;
    launch_copy_rms_norm(state_.hidden, layer.input_layernorm,
                         state_.residual, norm_out,
                         HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // Quantize input to int8 for dp4a
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

    // 2b. Fused QKNorm
    launch_qk_norm(state_.q_buf, state_.k_buf, layer.q_norm, layer.k_norm,
                   NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, RMS_NORM_EPS, stream);

    // 3. RoPE
    launch_rope(state_.q_buf, state_.k_buf,
                state_.rope_cos, state_.rope_sin,
                state_.current_pos, state_.max_seq_len,
                NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, stream);

    // 4. Store K, V into cache at current position
    cudaMemcpyAsync(kv.key + state_.current_pos * KV_DIM, state_.k_buf,
                     KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(kv.value + state_.current_pos * KV_DIM, state_.v_buf,
                     KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);

    // 5. GQA Attention
    launch_gqa_attention(state_.q_buf, kv.key, kv.value, state_.attn_out,
                          state_.attn_scores, state_.current_pos, state_.max_seq_len,
                          NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, stream);

    // 6. Output projection (input: attn_out, dim=Q_DIM)
    if (!layer.o_proj_fp16 && weights_.is_q4l) {
        launch_quantize_input_q8(state_.attn_out, state_.q8_data, state_.q8_scales,
                                 state_.q8_sums, Q_DIM, stream);
    }
    if (layer.o_proj_fp16) cublas_hgemv_lora(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, layer.lora_o, state_.lora_scratch);
    else { auto& w = layer.o_proj_nf4; GEMV_4BIT(w, state_.attn_out, state_.hidden, stream); }

    // 7. Residual add (hidden += residual)
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // 8. Post-attention LayerNorm
    launch_copy_rms_norm(state_.hidden, layer.post_attn_layernorm,
                         state_.residual, norm_out,
                         HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // Quantize FFN input for dp4a
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
    cudaStream_t stream = 0;

    // Embedding lookup
    launch_embedding(weights_.embed_tokens, token_id, state_.hidden, HIDDEN_SIZE, stream);

    for (int i = 0; i < NUM_LAYERS; i++) {
        forward_layer(i);
    }

    // Final LayerNorm
    half* norm_out = state_.ffn_out;
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
        cudaMallocChecked(&(*target)->A, A_rows * A_cols * sizeof(half));
        cudaMallocChecked(&(*target)->B, B_rows * B_cols * sizeof(half));
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

    // Decode loop (all GPU sampling, no Python per token)
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

        decode(token);
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

    // Free old batch (host only -- GPU memory is in the arena)
    if (batch_) {
        delete[] batch_->h_positions; delete[] batch_->h_tokens;
        delete[] batch_->h_finished; delete[] batch_->h_randoms;
        delete batch_;
    }

    // Calculate total GPU memory needed
    size_t need = 0;
    auto add = [&](size_t bytes) { need = (need + 255) & ~(size_t)255; need += bytes; };
    add(HIDDEN_SIZE * G * sizeof(half));          // hidden
    add(HIDDEN_SIZE * G * sizeof(half));          // residual
    add(HIDDEN_SIZE * G * sizeof(half));          // norm_buf
    add(Q_DIM * G * sizeof(half));                // q_buf
    add(KV_DIM * G * sizeof(half));               // k_buf
    add(KV_DIM * G * sizeof(half));               // v_buf
    add(Q_DIM * G * sizeof(half));                // attn_out
    add(INTERMEDIATE_SIZE * G * sizeof(half));     // gate_buf
    add(INTERMEDIATE_SIZE * G * sizeof(half));     // up_buf
    add(VOCAB_SIZE * G * sizeof(float));           // logits
    add(G * NUM_HEADS * max_seq_len * sizeof(float)); // attn_scores
    if (!weights_cached_)
        add((size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(half)); // dequant_scratch
    for (int i = 0; i < NUM_LAYERS; i++) {
        add((size_t)G * max_seq_len * KV_DIM * sizeof(half));  // kv_keys
        add((size_t)G * max_seq_len * KV_DIM * sizeof(half));  // kv_values
    }
    add(G * sizeof(int));   // d_positions
    add(G * sizeof(int));   // d_tokens
    add(G * sizeof(float)); // d_randoms
    add(64 * G * sizeof(half)); // lora_scratch

    // (Re)allocate arena if needed (invalidates CUDA graph)
    if (need > batch_arena_.capacity) {
        if (batch_arena_.base) cudaFree(batch_arena_.base);
        CUDA_CHECK(cudaMalloc(&batch_arena_.base, need));
        batch_arena_.capacity = need;
        if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
        if (decode_graph_) { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }
        graph_G_ = 0;
    }
    batch_arena_.reset();

    // Suballocate all buffers from the arena (zero per-step cudaMalloc calls)
    batch_ = new BatchState();
    batch_->G = G;
    batch_->max_seq_len = max_seq_len;
    batch_->d_all_randoms = nullptr;
    batch_->all_randoms_size = 0;

    batch_->hidden    = (half*)batch_arena_.alloc(HIDDEN_SIZE * G * sizeof(half));
    batch_->residual  = (half*)batch_arena_.alloc(HIDDEN_SIZE * G * sizeof(half));
    batch_->norm_buf  = (half*)batch_arena_.alloc(HIDDEN_SIZE * G * sizeof(half));
    batch_->q_buf     = (half*)batch_arena_.alloc(Q_DIM * G * sizeof(half));
    batch_->k_buf     = (half*)batch_arena_.alloc(KV_DIM * G * sizeof(half));
    batch_->v_buf     = (half*)batch_arena_.alloc(KV_DIM * G * sizeof(half));
    batch_->attn_out  = (half*)batch_arena_.alloc(Q_DIM * G * sizeof(half));
    batch_->gate_buf  = (half*)batch_arena_.alloc(INTERMEDIATE_SIZE * G * sizeof(half));
    batch_->up_buf    = (half*)batch_arena_.alloc(INTERMEDIATE_SIZE * G * sizeof(half));
    batch_->logits    = (float*)batch_arena_.alloc(VOCAB_SIZE * G * sizeof(float));
    batch_->attn_scores = (float*)batch_arena_.alloc(G * NUM_HEADS * max_seq_len * sizeof(float));
    if (!weights_cached_) {
        batch_->dequant_scratch = (half*)batch_arena_.alloc((size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(half));
    } else {
        batch_->dequant_scratch = nullptr;
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        batch_->kv_keys[i]   = (half*)batch_arena_.alloc((size_t)G * max_seq_len * KV_DIM * sizeof(half));
        batch_->kv_values[i] = (half*)batch_arena_.alloc((size_t)G * max_seq_len * KV_DIM * sizeof(half));
    }
    batch_->d_positions  = (int*)batch_arena_.alloc(G * sizeof(int));
    batch_->d_tokens     = (int*)batch_arena_.alloc(G * sizeof(int));
    batch_->d_randoms    = (float*)batch_arena_.alloc(G * sizeof(float));
    batch_->lora_scratch = (half*)batch_arena_.alloc(64 * G * sizeof(half));

    // Host allocations
    batch_->h_positions = new int[G]();
    batch_->h_tokens = new int[G]();
    batch_->h_finished = new bool[G]();
    batch_->h_randoms = new float[G]();

    std::cout << "  Batch arena: " << batch_arena_.used() / 1e6 << "MB (1 cudaMalloc, G="
              << G << " seq=" << max_seq_len << ")" << std::endl;
}

void InferenceEngine::batch_gemm(half* out, const half* weight, const half* in,
                                  int M, int N, int K, cudaStream_t stream) {
    ensure_cublas();
    // weight is (M, K) row-major = (K, M) col-major for cuBLAS
    // in is (K, N) col-major, out is (M, N) col-major
    // Use fp32 accumulation for precision (fp16 loses accuracy for K=1024+)
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 M, N, K, &alpha,
                 weight, CUDA_R_16F, K,
                 in, CUDA_R_16F, K,
                 &beta, out, CUDA_R_16F, M,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void InferenceEngine::cache_weights() {
    if (weights_cached_ || !weights_.is_q4l) return;
    size_t total_bytes = 0;
    auto cache_one = [&](NF4Weight& w) {
        if (!w.data || w.fp16_cache) return;
        size_t sz = (size_t)w.out_dim * w.in_dim * sizeof(half);
        CUDA_CHECK(cudaMalloc(&w.fp16_cache, sz));
        launch_dequant_q4l(w.fp16_cache, w.data, w.absmax, w.out_dim, w.in_dim, 0);
        total_bytes += sz;
    };
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& L = weights_.layers[i];
        cache_one(L.q_proj_nf4); cache_one(L.k_proj_nf4);
        cache_one(L.v_proj_nf4); cache_one(L.o_proj_nf4);
        cache_one(L.gate_proj_nf4); cache_one(L.up_proj_nf4);
        cache_one(L.down_proj_nf4);
    }
    cudaDeviceSynchronize();
    weights_cached_ = true;
    std::cout << "  Cached " << total_bytes / 1e6 << "MB dequanted weights" << std::endl;
}

void InferenceEngine::batch_gemm_q4l(half* out, const NF4Weight& w, const half* in,
                                      int N, cudaStream_t stream) {
    if (w.fp16_cache) {
        // Use pre-dequanted fp16 weights (no per-step dequant)
        batch_gemm(out, w.fp16_cache, in, w.out_dim, N, w.in_dim, stream);
    } else {
        // Fallback: dequant to scratch each step
        launch_dequant_q4l(batch_->dequant_scratch, w.data, w.absmax,
                           w.out_dim, w.in_dim, stream);
        batch_gemm(out, (const half*)batch_->dequant_scratch, in,
                   w.out_dim, N, w.in_dim, stream);
    }
}

void InferenceEngine::forward_layer_batch(int layer_idx, int G, cudaStream_t stream) {
    auto& L = weights_.layers[layer_idx];
    auto* B = batch_;

    // Batched projection: base GEMM + optional LoRA
    auto project = [&](half* out, half* fp16w, NF4Weight& nf4w, const half* input,
                       int out_dim, int in_dim, const LoRAAdapter* lora) {
        if (fp16w) batch_gemm(out, fp16w, input, out_dim, G, in_dim, stream);
        else batch_gemm_q4l(out, nf4w, input, G, stream);
        // LoRA: out += scale * B @ (A @ input)
        if (lora && lora->A && lora->B) {
            ensure_cublas(stream);
            // scratch = A @ input: (rank, in_dim) @ (in_dim, G) -> (rank, G)
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         lora->rank, G, lora->in_features, &alpha,
                         lora->A, CUDA_R_16F, lora->in_features,
                         input, CUDA_R_16F, lora->in_features,
                         &beta, B->lora_scratch, CUDA_R_16F, lora->rank,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            // out += scale * B @ scratch: (out_dim, rank) @ (rank, G) -> (out_dim, G)
            float one = 1.0f;
            float scale = lora->scale;
            cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         lora->out_features, G, lora->rank, &scale,
                         lora->B, CUDA_R_16F, lora->rank,
                         B->lora_scratch, CUDA_R_16F, lora->rank,
                         &one, out, CUDA_R_16F, lora->out_features,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    };

    // 1. Copy hidden -> residual, RMSNorm -> norm_buf
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.input_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // 2. QKV projections
    project(B->q_buf, L.q_proj_fp16, L.q_proj_nf4, B->norm_buf, Q_DIM, HIDDEN_SIZE, L.lora_q);
    project(B->k_buf, L.k_proj_fp16, L.k_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE, L.lora_k);
    project(B->v_buf, L.v_proj_fp16, L.v_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE, L.lora_v);

    // 3. QKNorm + RoPE
    launch_qk_norm_batch(B->q_buf, B->k_buf, L.q_norm, L.k_norm,
                         NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, G, RMS_NORM_EPS,
                         Q_DIM, KV_DIM, stream);
    launch_rope_batch(B->q_buf, B->k_buf, state_.rope_cos, state_.rope_sin,
                      B->d_positions, B->max_seq_len, G,
                      NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM, stream);

    // 4. KV cache write
    launch_kv_cache_write_batch(B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                 B->k_buf, B->v_buf, B->d_positions, B->max_seq_len, G,
                                 KV_DIM, stream);

    // 5. GQA attention
    launch_gqa_attention_batch(B->attn_out, B->q_buf,
                                B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                B->attn_scores, B->d_positions, B->max_seq_len, G,
                                NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM, stream);

    // 6. Output projection
    project(B->hidden, L.o_proj_fp16, L.o_proj_nf4, B->attn_out, HIDDEN_SIZE, Q_DIM, L.lora_o);

    // 7. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);

    // 8. Post-attention norm
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // 9. FFN
    project(B->gate_buf, L.gate_proj_fp16, L.gate_proj_nf4, B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_gate);
    project(B->up_buf, L.up_proj_fp16, L.up_proj_nf4, B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_up);
    launch_silu_mul_batch(B->gate_buf, B->up_buf, INTERMEDIATE_SIZE * G, stream);
    project(B->hidden, L.down_proj_fp16, L.down_proj_nf4, B->gate_buf, HIDDEN_SIZE, INTERMEDIATE_SIZE, L.lora_down);

    // 10. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);
}

void InferenceEngine::decode_batch(int G, cudaStream_t stream) {
    auto* B = batch_;

    // Embedding
    launch_embed_batch(B->hidden, weights_.embed_tokens, B->d_tokens, G, HIDDEN_SIZE, stream);

    // Forward layers
    for (int i = 0; i < NUM_LAYERS; i++)
        forward_layer_batch(i, G, stream);

    // Final norm
    launch_rms_norm_batch(B->norm_buf, B->hidden, weights_.final_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream);

    // LM head: (VOCAB_SIZE, HIDDEN_SIZE) @ (HIDDEN_SIZE, G) -> (VOCAB_SIZE, G) in fp32
    {
        ensure_cublas(stream);
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
    int max_new_tokens, float temperature, float top_p, int eos_token_id,
    const std::vector<int>& stop_token_ids
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

    // Pre-generate ALL random values for decode phase (enables CUDA graph)
    // Upload once to GPU, then index by step during decode.
    int n_randoms = max_new_tokens * G;
    if (temperature >= 0.01f) {
        if (!B->d_all_randoms || B->all_randoms_size < n_randoms) {
            if (B->d_all_randoms) cudaFree(B->d_all_randoms);
            CUDA_CHECK(cudaMalloc(&B->d_all_randoms, n_randoms * sizeof(float)));
            B->all_randoms_size = n_randoms;
        }
        std::vector<float> h_randoms(n_randoms);
        static std::mt19937 batch_rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (int i = 0; i < n_randoms; i++) h_randoms[i] = dist(batch_rng);
        cudaMemcpy(B->d_all_randoms, h_randoms.data(), n_randoms * sizeof(float), cudaMemcpyHostToDevice);
    }

    // CUDA graph: capture decode_batch + sampling on engine_stream_.
    // The graph reads d_tokens, d_positions (updated via cudaMemcpy before launch)
    // and d_all_randoms[step * G] (pre-uploaded, step offset via pointer arithmetic).
    // Since sampling reads a DIFFERENT slice of d_all_randoms each step, and CUDA
    // graphs bake kernel args, we CAN'T put sampling in the graph directly.
    // Instead: graph captures ONLY decode_batch. Sampling runs eagerly after.
    bool use_graph = false;
    cudaStream_t gen_stream = engine_stream_;  // use dedicated stream for decode

    if (G > 1 && max_new_tokens >= 50 && !(graph_G_ == G && decode_graph_exec_)) {
        if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
        if (decode_graph_) { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }

        // Warmup cuBLAS plans on engine_stream_
        ensure_cublas(gen_stream);
        decode_batch(G, gen_stream);
        cudaStreamSynchronize(gen_stream);

        // Capture decode_batch only (sampling stays eager)
        cudaError_t err = cudaStreamBeginCapture(gen_stream, cudaStreamCaptureModeRelaxed);
        if (err == cudaSuccess) {
            decode_batch(G, gen_stream);
            cudaGraph_t graph = nullptr;
            err = cudaStreamEndCapture(gen_stream, &graph);
            if (err == cudaSuccess && graph) {
                err = cudaGraphInstantiate(&decode_graph_exec_, graph, 0);
                if (err == cudaSuccess) {
                    decode_graph_ = graph;
                    graph_G_ = G;
                    use_graph = true;
                    std::cout << "  CUDA graph captured (G=" << G << ")" << std::endl;
                } else { cudaGraphDestroy(graph); }
            }
            if (!use_graph) { cudaGetLastError(); }
        }

        // Re-prefill (warmup decode corrupted KV cache)
        for (int i = 0; i < NUM_LAYERS; i++) {
            cudaMemset(B->kv_keys[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
            cudaMemset(B->kv_values[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
        }
        for (int g = 0; g < G; g++) B->h_positions[g] = 0;
        for (int t = 0; t < max_prompt_len; t++) {
            for (int g = 0; g < G; g++)
                B->h_tokens[g] = (t < (int)prompts[g].size()) ? prompts[g][t] : 0;
            cudaMemcpy(B->d_tokens, B->h_tokens, G * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);
            decode_batch(G, gen_stream);
            cudaStreamSynchronize(gen_stream);
            for (int g = 0; g < G; g++)
                if (t < (int)prompts[g].size()) B->h_positions[g]++;
        }
    } else if (graph_G_ == G && decode_graph_exec_) {
        use_graph = true;
    }

    // Phase 2: Decode (generate new tokens)
    for (int step = 0; step < max_new_tokens; step++) {
        // Forward pass (graph or eager, on engine_stream_)
        if (use_graph) {
            cudaGraphLaunch(decode_graph_exec_, gen_stream);
        } else {
            decode_batch(G, gen_stream);
        }

        // Sampling (always eager, on engine_stream_ after decode completes)
        if (temperature < 0.01f) {
            launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, gen_stream);
        } else {
            // Point to this step's pre-generated randoms
            launch_sample_batch(B->logits, B->d_tokens,
                                B->d_all_randoms + step * G,
                                VOCAB_SIZE, G, temperature, top_p, gen_stream);
        }

        // Wait for engine_stream_ to finish, then read tokens
        // Use event sync (lighter than stream sync) to synchronize with default stream
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, gen_stream);
        cudaStreamWaitEvent(0, ev);  // default stream waits for engine_stream_
        cudaEventDestroy(ev);
        cudaMemcpy(B->h_tokens, B->d_tokens, G * sizeof(int), cudaMemcpyDeviceToHost);

        // Check stopping (eos + stop sequences)
        bool all_done = true;
        int stop_len = stop_token_ids.size();
        for (int g = 0; g < G; g++) {
            if (!B->h_finished[g]) {
                outputs[g].push_back(B->h_tokens[g]);
                if (B->h_tokens[g] == eos_token_id) {
                    B->h_finished[g] = true;
                } else if (stop_len > 0 && (int)outputs[g].size() >= stop_len) {
                    bool match = true;
                    for (int j = 0; j < stop_len; j++) {
                        if (outputs[g][outputs[g].size() - stop_len + j] != stop_token_ids[j]) {
                            match = false; break;
                        }
                    }
                    if (match) B->h_finished[g] = true;
                }
                if (!B->h_finished[g]) all_done = false;
            }
        }
        if (all_done) break;

        // Update tokens + positions on engine_stream_ for next decode step
        cudaMemcpyAsync(B->d_tokens, B->h_tokens, G * sizeof(int), cudaMemcpyHostToDevice, gen_stream);
        cudaMemcpyAsync(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice, gen_stream);

        // Advance positions
        for (int g = 0; g < G; g++)
            if (!B->h_finished[g]) B->h_positions[g]++;
    }

    // Restore cuBLAS to stream 0 for PyTorch compatibility
    ensure_cublas(0);

    return outputs;
}
