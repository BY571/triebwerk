/**
 * C++ inference engine — the core decode loop.
 *
 * One function call from Python generates an entire completion.
 * No Python per token, no kernel launch overhead from PyTorch dispatch.
 */

#include "model.h"
#include "gguf_loader.h"
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
static bool cublas_capture_mode = false;  // skip cublasSetStream during graph capture

static void ensure_cublas(cudaStream_t stream = 0) {
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
        cudaMalloc(&cublas_workspace, CUBLAS_WORKSPACE_SIZE);
        cublasSetWorkspace(cublas_handle, cublas_workspace, CUBLAS_WORKSPACE_SIZE);
    }
    if (!cublas_capture_mode && stream != cublas_current_stream) {
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
    void launch_nf4_dp4a_gemv(const uint8_t* w, const float* absmax, const int8_t* q8, const float* q8_sc, const float* q8_sm, half* y, int out_dim, int in_dim, cudaStream_t stream);
    void launch_nf4_batch_gemv(const uint8_t* w, const float* absmax, const half* input, half* output, int out_dim, int in_dim, int G, cudaStream_t stream);
    void launch_nf4_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_2(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_nf4_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_q4l_fused_3(const uint8_t* a_w, const float* a_abs, half* a_out, int a_dim, const uint8_t* b_w, const float* b_abs, half* b_out, int b_dim, const uint8_t* c_w, const float* c_abs, half* c_out, int c_dim, const half* input, int in_dim, cudaStream_t stream);
    void launch_quantize_input_q8(const half* input, int8_t* q8_data, float* q8_scales, float* q8_sums, int dim, cudaStream_t stream);
    void launch_q4l_dp4a_gemv(const uint8_t* w, const float* w_scales, const int8_t* q8, const float* q8_sc, const float* q8_sm, half* y, int out_dim, int in_dim, cudaStream_t stream);
    void launch_dequant_q4l(half* out, const uint8_t* data, const float* scales, int out_dim, int in_dim, cudaStream_t stream);
    void launch_q4l_batch_gemm(const uint8_t* w, const float* w_scales, const half* input, half* output, int out_dim, int in_dim, int G, cudaStream_t stream);
    void launch_nf4_batch_gemv(const uint8_t* w, const float* absmax, const half* input, half* output, int out_dim, int in_dim, int G, cudaStream_t stream);

    // Batched kernel launchers
    void launch_embed_batch(half* h, const half* et, const int* tok, int G, int hidden_size, cudaStream_t s);
    void launch_rms_norm_batch(half* out, const half* in, const half* w, int dim, int G, float eps, cudaStream_t s, bool biased = false);
    void launch_copy_batch(half* dst, const half* src, int total, cudaStream_t s);
    void launch_residual_add_batch(half* out, const half* res, int total, cudaStream_t s);
    void launch_qk_norm_batch(half* q, half* k, const half* qw, const half* kw, int nq, int nkv, int hd, int G, float eps, int q_dim, int kv_dim, cudaStream_t s, bool biased = false);
    void launch_rope_batch(half* q, half* k, const half* ct, const half* st, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, int rope_dim, cudaStream_t s);
    void launch_kv_cache_write_batch(half* ck, half* cv, const half* k, const half* v, const int* pos, int msl, int G, int kv_dim, cudaStream_t s);
    void launch_gqa_attention_batch(half* out, const half* q, const half* ck, const half* cv, float* as, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s);
    void launch_gqa_prefill_attention(half* out, const half* q, const half* ck, const half* cv, float* as, int T, int msl, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, cudaStream_t s);
    void launch_silu_mul_batch(half* gate, const half* up, int total, cudaStream_t s);
    void launch_argmax_batch(const float* logits, int* tokens, int vocab, int G, cudaStream_t s);
    void launch_increment_positions(int* positions, int G, cudaStream_t s);
    void launch_sample_batch(float* logits, int* tokens, const float* const* randoms_ptr, int vocab, int G, float temperature, float top_p, cudaStream_t s);
    // TurboQuant kernel launchers
    void launch_turbo_kv_quantize(uint8_t* cache_q, half* cache_norms, const half* kv_buf, const half* rotation, const int* positions, int max_seq_len, int num_tokens, int num_kv_heads, int head_dim, int kv_dim, int bits, cudaStream_t s);
    void launch_turbo_kv_quantize_batch(uint8_t* cache_q, half* cache_norms, const half* kv_buf, const half* rotation, const int* positions, int max_seq_len, int G, int num_kv_heads, int head_dim, int kv_dim, int bits, cudaStream_t s);
    void launch_turbo_gqa_attention_batch(half* out, const half* q, const uint8_t* ck_q, const uint8_t* cv_q, const half* k_norms, const half* v_norms, const half* rotation, float* as, const int* pos, int msl, int G, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, int bits, cudaStream_t s);
    void launch_turbo_gqa_attention(const half* q, const uint8_t* ck_q, const uint8_t* cv_q, const half* k_norms, const half* v_norms, const half* rotation, half* output, float* as, int pos, int msl, int num_heads, int num_kv_heads, int head_dim, int bits, cudaStream_t s);
    void launch_turbo_gqa_prefill_attention(half* out, const half* q, const uint8_t* ck_q, const uint8_t* cv_q, const half* k_norms, const half* v_norms, const half* rotation, float* as, int T, int msl, int num_heads, int num_kv_heads, int head_dim, int q_dim, int kv_dim, int bits, cudaStream_t s);

    // SSM (Gated Delta Rule) kernel launchers
    void launch_ssm_conv1d_decode(float* output, float* conv_state, const half* input, const half* weight, const half* bias, int conv_dim, int G, int kernel_size, cudaStream_t s);
    void launch_ssm_compute_dt_decay(float* decay, float* beta, const half* a_in, const half* b_in, const half* A_log, const half* dt_bias, int num_v_heads, int G, cudaStream_t s);
    void launch_ssm_gated_delta_rule(float* state, float* y_out, const float* q, const float* k, const float* v, const float* decay, const float* beta, int num_v_heads, int num_k_heads, int k_head_dim, int v_head_dim, int G, cudaStream_t s);
    void launch_ssm_gated_rmsnorm(half* y_out, const float* y_in, const half* z, const half* weight, int num_v_heads, int v_head_dim, int G, float eps, cudaStream_t s);
    void launch_ssm_gated_rmsnorm_colmajor(half* y_out, const float* y_in, const half* z, const half* weight, int num_v_heads, int v_head_dim, int T, float eps, cudaStream_t s);
    void launch_ssm_l2norm_qk(float* q, float* k, int num_k_heads, int k_head_dim, int G, cudaStream_t s);
    void launch_ssm_expand_kv_heads(half* out, const half* in, int num_k_heads, int num_v_heads, int k_head_dim, int G, cudaStream_t s);

    // SSM chunked prefill kernel launchers
    void launch_ssm_causal_conv1d_prefill(float* output, float* conv_state, const half* input, const half* weight, const half* bias, int conv_dim, int T, int kernel_size, cudaStream_t s);
    void launch_ssm_compute_g_beta(float* g_out, float* beta_out, const half* a_in, const half* b_in, const half* A_log, const half* dt_bias, int num_v_heads, int T, cudaStream_t s);
    void launch_ssm_chunk_rearrange(float* Q_out, float* K_out, float* V_out, float* K_beta_out, float* V_beta_out, const float* qkv, const float* beta, int num_heads, int head_dim, int T, int T_padded, int key_dim, int value_dim, int conv_dim, cudaStream_t s);
    void launch_ssm_chunked_delta_rule(float* state, float* output, const float* Q, const float* K, const float* V, const float* K_beta, const float* V_beta, const float* g, float* workspace, int num_heads, int head_dim, int T, int T_padded, int chunk_idx, int chunk_size, int T_actual, cudaStream_t s);
    void launch_ssm_chunk_output_rearrange(float* y_out, const float* output, int num_heads, int head_dim, int T, int T_padded, cudaStream_t s);

    void launch_sigmoid_gate_batch(half* data, const half* gate, int dim, int G, cudaStream_t s);

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

    // Partial rotary: rope_dim = rotary dimensions (default = head_dim)
    c.rope_dim = get_int("rope_dim");
    if (c.rope_dim <= 0) c.rope_dim = c.head_dim;

    // Gated attention: q_proj outputs 2x (query + gate), Qwen3.5 style
    c.gated_attn = (get_int("gated_attn") > 0);

    // Hybrid SSM fields (default 0 = pure transformer)
    c.ssm_num_k_heads = std::max(0, get_int("ssm_num_k_heads"));
    c.ssm_num_v_heads = std::max(0, get_int("ssm_num_v_heads"));
    c.ssm_k_head_dim = std::max(0, get_int("ssm_k_head_dim"));
    c.ssm_v_head_dim = std::max(0, get_int("ssm_v_head_dim"));
    c.ssm_conv_kernel = std::max(0, get_int("ssm_conv_kernel"));
    if (c.ssm_conv_kernel == 0) c.ssm_conv_kernel = 4; // default

    // Parse layer_types array: [0,1,1,0,...] (0=attention, 1=SSM)
    // Default: all attention
    for (int i = 0; i < c.num_layers; i++) c.layer_type[i] = LAYER_ATTENTION;

    if (c.ssm_num_v_heads > 0) {
        // Find "layer_types" array in JSON
        auto pos = json.find("\"layer_types\"");
        if (pos != std::string::npos) {
            auto start = json.find('[', pos);
            auto end = json.find(']', start);
            if (start != std::string::npos && end != std::string::npos) {
                std::string arr = json.substr(start + 1, end - start - 1);
                std::istringstream ss(arr);
                std::string tok;
                int idx = 0;
                while (std::getline(ss, tok, ',') && idx < c.num_layers) {
                    int val = std::stoi(tok);
                    c.layer_type[idx++] = (val == 1) ? LAYER_SSM : LAYER_ATTENTION;
                }
            }
        }
    }

    std::cout << "  Config: " << c.hidden_size << "h, " << c.intermediate_size << "i, "
              << c.num_layers << "L, " << c.num_heads << "Qh, " << c.num_kv_heads << "KVh, "
              << c.head_dim << "hd, " << c.vocab_size << "V" << std::endl;
    if (c.is_hybrid()) {
        std::cout << "  Hybrid: " << c.num_attn_layers() << " attn + "
                  << c.num_ssm_layers() << " SSM layers (k_heads="
                  << c.ssm_num_k_heads << ", v_heads=" << c.ssm_num_v_heads << ")" << std::endl;
    }
    return c;
}

// Dispatch macros: per-weight format detection (quant_map != null → NF4 dp4a, null → Q4L dp4a)
// Both use dp4a int8 dot products. NF4 adds a 16-entry shared memory lookup.
#define GEMV_4BIT(w, in, out, stream) \
    do { if ((w).quant_map) launch_nf4_dp4a_gemv((w).data, (w).absmax, state_.q8_data, state_.q8_scales, state_.q8_sums, (out), (w).out_dim, (w).in_dim, (stream)); \
         else launch_q4l_dp4a_gemv((w).data, (w).absmax, state_.q8_data, state_.q8_scales, state_.q8_sums, (out), (w).out_dim, (w).in_dim, (stream)); } while(0)
#define FUSED2_4BIT(aw, ay, bw, by, x, id, stream) \
    do { if ((aw).quant_map) launch_nf4_fused_2((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (x), (id), (stream)); \
         else launch_q4l_fused_2((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (x), (id), (stream)); } while(0)
#define FUSED3_4BIT(aw, ay, bw, by, cw, cy, x, id, stream) \
    do { if ((aw).quant_map) launch_nf4_fused_3((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (cw).data, (cw).absmax, (cy), (cw).out_dim, (x), (id), (stream)); \
         else launch_q4l_fused_3((aw).data, (aw).absmax, (ay), (aw).out_dim, (bw).data, (bw).absmax, (by), (bw).out_dim, (cw).data, (cw).absmax, (cy), (cw).out_dim, (x), (id), (stream)); } while(0)

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
#define Q_PROJ_DIM config_.q_proj_dim()
#define KV_DIM config_.kv_dim()
#define RMS_NORM_EPS config_.rms_norm_eps
#define ROPE_THETA config_.rope_theta
#define GQA_GROUPS config_.gqa_groups()
// TurboQuant: packed KV dimension (bytes per position per all heads)
// 2-bit: 4 values per byte → kv_dim/4. 4-bit: 2 values per byte → kv_dim/2.
#define KV_DIM_PACKED (kv_bits_ == 4 ? KV_DIM / 2 : KV_DIM / 4)

// ============================================================================
// Constructor / Destructor
// ============================================================================

InferenceEngine::InferenceEngine(int max_seq_len, int kv_bits) {
    // Config and buffers are allocated in load_weights() when we know the model dims.
    // Here we just store max_seq_len and zero-init pointers.
    config_ = ModelConfig::qwen3_0_6b(); // default, overwritten by load_weights
    state_ = {};
    state_.max_seq_len = max_seq_len;
    state_.current_pos = 0;
    kv_bits_ = kv_bits;
    batch_ = nullptr;
    cudaStreamCreate(&engine_stream_);
    if (kv_bits_ != 0 && kv_bits_ != 2 && kv_bits_ != 4) {
        std::cerr << "WARNING: kv_bits=" << kv_bits_ << " not supported, using 0 (fp16)" << std::endl;
        kv_bits_ = 0;
    }
    if (kv_bits_ > 0)
        std::cout << "  TurboQuant: " << kv_bits_ << "-bit KV cache enabled ("
                  << (16 / kv_bits_) << "x memory reduction)" << std::endl;
}

void InferenceEngine::allocate_buffers() {
    int max_seq_len = state_.max_seq_len;

    // Allocate KV caches (fp16 when kv_bits_==0; quantized caches allocated later)
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (kv_bits_ == 0) {
            cudaMallocChecked(&state_.kv_cache[i].key, max_seq_len * KV_DIM * sizeof(half));
            cudaMallocChecked(&state_.kv_cache[i].value, max_seq_len * KV_DIM * sizeof(half));
        } else {
            state_.kv_cache[i].key = nullptr;
            state_.kv_cache[i].value = nullptr;
        }
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
    cudaMallocChecked(&state_.rope_cos, max_seq_len * (config_.rope_dim / 2) * sizeof(half));
    cudaMallocChecked(&state_.rope_sin, max_seq_len * (config_.rope_dim / 2) * sizeof(half));
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

    // TurboQuant: generate rotation matrix and allocate quantized KV caches
    state_.turbo_rotation = nullptr;
    if (kv_bits_ > 0) {
        // Generate random orthogonal matrix via Gram-Schmidt on CPU, upload to GPU
        int d = HEAD_DIM;
        std::mt19937 gen(42); // fixed seed for reproducibility
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> mat(d * d);
        for (int i = 0; i < d * d; i++) mat[i] = dist(gen);
        // Gram-Schmidt orthogonalization
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < i; j++) {
                float dot = 0;
                for (int k = 0; k < d; k++) dot += mat[i * d + k] * mat[j * d + k];
                for (int k = 0; k < d; k++) mat[i * d + k] -= dot * mat[j * d + k];
            }
            float norm = 0;
            for (int k = 0; k < d; k++) norm += mat[i * d + k] * mat[i * d + k];
            norm = 1.0f / sqrtf(norm);
            for (int k = 0; k < d; k++) mat[i * d + k] *= norm;
        }
        // Upload as fp16
        cudaMallocChecked(&state_.turbo_rotation, d * d * sizeof(half));
        std::vector<half> fp16_mat(d * d);
        for (int i = 0; i < d * d; i++) fp16_mat[i] = __float2half(mat[i]);
        CUDA_CHECK(cudaMemcpy(state_.turbo_rotation, fp16_mat.data(),
                              d * d * sizeof(half), cudaMemcpyHostToDevice));
        std::cout << "  TurboQuant rotation matrix: " << d << "x" << d
                  << " (" << d * d * 2 / 1024 << " KB)" << std::endl;

        // Allocate quantized single-sequence KV caches
        for (int i = 0; i < NUM_LAYERS; i++) {
            cudaMallocChecked(&state_.kv_cache[i].key_quant,
                              max_seq_len * (KV_DIM_PACKED) * sizeof(uint8_t));
            cudaMallocChecked(&state_.kv_cache[i].value_quant,
                              max_seq_len * (KV_DIM_PACKED) * sizeof(uint8_t));
            cudaMallocChecked(&state_.kv_cache[i].key_norms,
                              max_seq_len * NUM_KV_HEADS * sizeof(half));
            cudaMallocChecked(&state_.kv_cache[i].value_norms,
                              max_seq_len * NUM_KV_HEADS * sizeof(half));
        }
    }

    // Zero-init weights
    memset(&weights_, 0, sizeof(weights_));
}

InferenceEngine::~InferenceEngine() {
    // Free LoRA adapters (GPU memory + host struct)
    auto free_lora = [](LoRAAdapter*& lora) {
        if (lora) {
            if (lora->A) cudaFree(lora->A);
            if (lora->B) cudaFree(lora->B);
            delete lora;
            lora = nullptr;
        }
    };
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& L = weights_.layers[i];
        free_lora(L.lora_q); free_lora(L.lora_k); free_lora(L.lora_v);
        free_lora(L.lora_o); free_lora(L.lora_gate); free_lora(L.lora_up);
        free_lora(L.lora_down); free_lora(L.lora_ssm_qkv);
        free_lora(L.lora_ssm_z); free_lora(L.lora_ssm_out);

        // Free fp16 weights
        if (!embed_is_external_) {
            if (L.q_proj_fp16) cudaFree(L.q_proj_fp16);
            if (L.k_proj_fp16) cudaFree(L.k_proj_fp16);
            if (L.v_proj_fp16) cudaFree(L.v_proj_fp16);
            if (L.o_proj_fp16) cudaFree(L.o_proj_fp16);
        }
        if (L.gate_proj_fp16) cudaFree(L.gate_proj_fp16);
        if (L.up_proj_fp16) cudaFree(L.up_proj_fp16);
        if (L.down_proj_fp16) cudaFree(L.down_proj_fp16);
        if (L.input_layernorm) cudaFree(L.input_layernorm);
        if (L.post_attn_layernorm) cudaFree(L.post_attn_layernorm);
        if (L.q_norm) cudaFree(L.q_norm);
        if (L.k_norm) cudaFree(L.k_norm);

        // Free NF4/Q4L weights
        if (L.q_proj_nf4.data) cudaFree(L.q_proj_nf4.data);
        if (L.q_proj_nf4.absmax) cudaFree(L.q_proj_nf4.absmax);
        if (L.q_proj_nf4.quant_map) cudaFree(L.q_proj_nf4.quant_map);
        if (L.q_proj_nf4.fp16_cache) cudaFree(L.q_proj_nf4.fp16_cache);
        if (L.k_proj_nf4.data) cudaFree(L.k_proj_nf4.data);
        if (L.k_proj_nf4.absmax) cudaFree(L.k_proj_nf4.absmax);
        if (L.k_proj_nf4.quant_map) cudaFree(L.k_proj_nf4.quant_map);
        if (L.k_proj_nf4.fp16_cache) cudaFree(L.k_proj_nf4.fp16_cache);
        if (L.v_proj_nf4.data) cudaFree(L.v_proj_nf4.data);
        if (L.v_proj_nf4.absmax) cudaFree(L.v_proj_nf4.absmax);
        if (L.v_proj_nf4.quant_map) cudaFree(L.v_proj_nf4.quant_map);
        if (L.v_proj_nf4.fp16_cache) cudaFree(L.v_proj_nf4.fp16_cache);
        if (L.o_proj_nf4.data) cudaFree(L.o_proj_nf4.data);
        if (L.o_proj_nf4.absmax) cudaFree(L.o_proj_nf4.absmax);
        if (L.o_proj_nf4.quant_map) cudaFree(L.o_proj_nf4.quant_map);
        if (L.o_proj_nf4.fp16_cache) cudaFree(L.o_proj_nf4.fp16_cache);
        if (L.gate_proj_nf4.data) cudaFree(L.gate_proj_nf4.data);
        if (L.gate_proj_nf4.absmax) cudaFree(L.gate_proj_nf4.absmax);
        if (L.gate_proj_nf4.quant_map) cudaFree(L.gate_proj_nf4.quant_map);
        if (L.gate_proj_nf4.fp16_cache) cudaFree(L.gate_proj_nf4.fp16_cache);
        if (L.up_proj_nf4.data) cudaFree(L.up_proj_nf4.data);
        if (L.up_proj_nf4.absmax) cudaFree(L.up_proj_nf4.absmax);
        if (L.up_proj_nf4.quant_map) cudaFree(L.up_proj_nf4.quant_map);
        if (L.up_proj_nf4.fp16_cache) cudaFree(L.up_proj_nf4.fp16_cache);
        if (L.down_proj_nf4.data) cudaFree(L.down_proj_nf4.data);
        if (L.down_proj_nf4.absmax) cudaFree(L.down_proj_nf4.absmax);
        if (L.down_proj_nf4.quant_map) cudaFree(L.down_proj_nf4.quant_map);
        if (L.down_proj_nf4.fp16_cache) cudaFree(L.down_proj_nf4.fp16_cache);

        // Free SSM weights
        if (L.ssm_in_proj_qkv_fp16) cudaFree(L.ssm_in_proj_qkv_fp16);
        if (L.ssm_in_proj_qkv_nf4.data) cudaFree(L.ssm_in_proj_qkv_nf4.data);
        if (L.ssm_in_proj_qkv_nf4.absmax) cudaFree(L.ssm_in_proj_qkv_nf4.absmax);
        if (L.ssm_in_proj_qkv_nf4.quant_map) cudaFree(L.ssm_in_proj_qkv_nf4.quant_map);
        if (L.ssm_in_proj_qkv_nf4.fp16_cache) cudaFree(L.ssm_in_proj_qkv_nf4.fp16_cache);
        if (L.ssm_in_proj_z_fp16) cudaFree(L.ssm_in_proj_z_fp16);
        if (L.ssm_in_proj_z_nf4.data) cudaFree(L.ssm_in_proj_z_nf4.data);
        if (L.ssm_in_proj_z_nf4.absmax) cudaFree(L.ssm_in_proj_z_nf4.absmax);
        if (L.ssm_in_proj_z_nf4.quant_map) cudaFree(L.ssm_in_proj_z_nf4.quant_map);
        if (L.ssm_in_proj_z_nf4.fp16_cache) cudaFree(L.ssm_in_proj_z_nf4.fp16_cache);
        if (L.ssm_out_proj_fp16) cudaFree(L.ssm_out_proj_fp16);
        if (L.ssm_out_proj_nf4.data) cudaFree(L.ssm_out_proj_nf4.data);
        if (L.ssm_out_proj_nf4.absmax) cudaFree(L.ssm_out_proj_nf4.absmax);
        if (L.ssm_out_proj_nf4.quant_map) cudaFree(L.ssm_out_proj_nf4.quant_map);
        if (L.ssm_out_proj_nf4.fp16_cache) cudaFree(L.ssm_out_proj_nf4.fp16_cache);
        if (L.ssm_in_proj_a_fp16) cudaFree(L.ssm_in_proj_a_fp16);
        if (L.ssm_in_proj_b_fp16) cudaFree(L.ssm_in_proj_b_fp16);
        if (L.ssm_conv1d_weight) cudaFree(L.ssm_conv1d_weight);
        if (L.ssm_conv1d_bias) cudaFree(L.ssm_conv1d_bias);
        if (L.ssm_A_log) cudaFree(L.ssm_A_log);
        if (L.ssm_dt_bias) cudaFree(L.ssm_dt_bias);
        if (L.ssm_norm_weight) cudaFree(L.ssm_norm_weight);
    }

    // Free embedding and final norm (only if not externally shared)
    if (weights_.embed_tokens && !embed_is_external_) cudaFree(weights_.embed_tokens);
    if (weights_.final_layernorm) cudaFree(weights_.final_layernorm);

    // Free KV caches and state buffers
    for (int i = 0; i < NUM_LAYERS; i++) {
        cudaFree(state_.kv_cache[i].key);
        cudaFree(state_.kv_cache[i].value);
        cudaFree(state_.kv_cache[i].key_quant);
        cudaFree(state_.kv_cache[i].value_quant);
        cudaFree(state_.kv_cache[i].key_norms);
        cudaFree(state_.kv_cache[i].value_norms);
    }
    cudaFree(state_.turbo_rotation);
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
        if (batch_->d_randoms_ptr) cudaFree(batch_->d_randoms_ptr);
        if (batch_->d_token_history) cudaFree(batch_->d_token_history);
        delete[] batch_->h_token_history;
        delete[] batch_->h_positions; delete[] batch_->h_tokens;
        delete[] batch_->h_finished; delete[] batch_->h_randoms;
        delete batch_;
        batch_ = nullptr;
    }
    if (batch_arena_.base && !embed_is_external_) cudaFree(batch_arena_.base);
    if (decode_graph_exec_) cudaGraphExecDestroy(decode_graph_exec_);
    if (decode_graph_) cudaGraphDestroy(decode_graph_);
    if (engine_stream_) cudaStreamDestroy(engine_stream_);
    if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
    if (cublas_workspace) { cudaFree(cublas_workspace); cublas_workspace = nullptr; }
}

// ============================================================================
// RoPE precomputation
// ============================================================================

void InferenceEngine::precompute_rope() {
    // Partial RoPE: only rope_dim dimensions get rotary encoding
    int rope_dim = config_.rope_dim;  // e.g., 64 for Qwen3.5 (25% of head_dim=256)
    int half_rope = rope_dim / 2;
    std::vector<half> cos_h(state_.max_seq_len * half_rope);
    std::vector<half> sin_h(state_.max_seq_len * half_rope);

    for (int pos = 0; pos < state_.max_seq_len; pos++) {
        for (int d = 0; d < half_rope; d++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * d) / rope_dim);
            float angle = pos * freq;
            cos_h[pos * half_rope + d] = __float2half(cosf(angle));
            sin_h[pos * half_rope + d] = __float2half(sinf(angle));
        }
    }

    cudaMemcpy(state_.rope_cos, cos_h.data(),
               state_.max_seq_len * half_rope * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(state_.rope_sin, sin_h.data(),
               state_.max_seq_len * half_rope * sizeof(half), cudaMemcpyHostToDevice);
}

// ============================================================================
// Reset
// ============================================================================

// ============================================================================
// GGUF weight loading — single quantized model file, no conversion step
// ============================================================================
void InferenceEngine::load_weights_gguf(const std::string& gguf_path) {
    GGUFFile gguf;
    if (!gguf.open(gguf_path)) {
        fprintf(stderr, "Failed to open GGUF: %s\n", gguf_path.c_str());
        return;
    }

    // Extract model config from GGUF metadata
    // Try both "qwen35" and "qwen3" prefixes (model-dependent)
    std::string arch = gguf.get_string("general.architecture", "qwen3");
    auto gi = [&](const std::string& key, int def) {
        int v = gguf.get_int(arch + "." + key, -1);
        return v >= 0 ? v : def;
    };
    auto gf = [&](const std::string& key, float def) {
        float v = gguf.get_float(arch + "." + key, -1.0f);
        return v >= 0 ? v : def;
    };

    config_.hidden_size = gi("embedding_length", 1024);
    config_.num_layers = gi("block_count", 24);
    config_.num_heads = gi("attention.head_count", 8);
    config_.num_kv_heads = gi("attention.head_count_kv", 2);
    config_.head_dim = gi("attention.key_length", config_.hidden_size / config_.num_heads);
    config_.intermediate_size = gi("feed_forward_length", 3584);
    config_.vocab_size = 248320;  // not always in metadata; derive from token_embd shape
    auto* emb_t = gguf.find_tensor("token_embd.weight");
    if (emb_t && emb_t->n_dims >= 2) config_.vocab_size = emb_t->dims[1];
    config_.rms_norm_eps = gf("attention.layer_norm_rms_epsilon", 1e-6f);
    config_.rope_theta = gf("rope.freq_base", 10000000.0f);
    config_.rope_dim = gi("rope.dimension_count", config_.head_dim);

    // SSM config
    config_.ssm_conv_kernel = gi("ssm.conv_kernel", 4);
    int ssm_state = gi("ssm.state_size", 0);
    int ssm_groups = gi("ssm.group_count", 0);
    config_.ssm_k_head_dim = ssm_state > 0 ? ssm_state : 128;
    config_.ssm_v_head_dim = config_.ssm_k_head_dim;
    config_.ssm_num_k_heads = ssm_groups > 0 ? ssm_groups : 16;
    config_.ssm_num_v_heads = config_.ssm_num_k_heads;

    // Detect layer types from tensor names
    for (int i = 0; i < config_.num_layers; i++) {
        std::string ssm_name = "blk." + std::to_string(i) + ".ssm_a";
        if (gguf.find_tensor(ssm_name)) {
            config_.layer_type[i] = LAYER_SSM;
        } else {
            config_.layer_type[i] = LAYER_ATTENTION;
        }
    }

    // Detect gated attention from attn_gate tensor
    config_.gated_attn = (gguf.find_tensor("blk.0.attn_gate.weight") != nullptr);

    // Detect full_attention_interval for hybrid models
    int attn_interval = gi("full_attention_interval", 0);
    if (attn_interval > 0) {
        // Override layer types: every attn_interval-th layer is attention
        for (int i = 0; i < config_.num_layers; i++) {
            config_.layer_type[i] = ((i + 1) % attn_interval == 0) ? LAYER_ATTENTION : LAYER_SSM;
        }
    }

    fprintf(stderr, "  GGUF config: %dh, %di, %dL, %dQh/%dKVh, %dhd, %dV\n",
            config_.hidden_size, config_.intermediate_size, config_.num_layers,
            config_.num_heads, config_.num_kv_heads, config_.head_dim, config_.vocab_size);
    if (config_.is_hybrid())
        fprintf(stderr, "  Hybrid: SSM k_heads=%d, v_heads=%d\n",
                config_.ssm_num_k_heads, config_.ssm_num_v_heads);
    if (config_.gated_attn)
        fprintf(stderr, "  Gated attention (Q_PROJ_DIM=%d)\n", config_.q_proj_dim());

    // Allocate buffers now that config is known
    allocate_buffers();

    // Helper: load small tensor (norms, biases) as fp16
    auto load_fp16 = [&](const std::string& name) -> half* {
        auto* t = gguf.find_tensor(name);
        if (!t) return nullptr;
        half* ptr;
        cudaMalloc(&ptr, t->n_elements * sizeof(half));
        gguf.load_tensor_fp16(name, ptr);
        return ptr;
    };

    // Helper: load projection as Q4L (quantized, dp4a-ready)
    auto load_q4l = [&](const std::string& name, int out_dim, int in_dim) -> NF4Weight {
        NF4Weight w = {};
        auto* t = gguf.find_tensor(name);
        if (!t) return w;
        w.out_dim = out_dim;
        w.in_dim = in_dim;
        w.block_size = 64;
        w.n_blocks = ((size_t)out_dim * in_dim + 63) / 64;
        size_t data_bytes = (size_t)w.n_blocks * 64 / 2;
        size_t scale_bytes = (size_t)w.n_blocks * sizeof(float);
        cudaMalloc(&w.data, data_bytes);
        cudaMalloc(&w.absmax, scale_bytes);
        w.quant_map = nullptr;  // Q4L has no lookup table
        w.fp16_cache = nullptr;
        gguf.load_tensor_q4l(name, w.data, w.absmax, (uint64_t)out_dim * in_dim);
        return w;
    };

    // Helper: load and transpose embedding (GGUF [hidden, vocab] → engine [vocab, hidden])
    auto load_embed = [&](const std::string& name) -> half* {
        auto* t = gguf.find_tensor(name);
        if (!t || t->n_dims < 2) return load_fp16(name);
        int rows = t->dims[1], cols = t->dims[0];
        half* tmp;
        cudaMalloc(&tmp, t->n_elements * sizeof(half));
        gguf.load_tensor_fp16(name, tmp);
        std::vector<half> h_src(t->n_elements), h_dst(t->n_elements);
        cudaMemcpy(h_src.data(), tmp, t->n_elements * sizeof(half), cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                h_dst[r * cols + c] = h_src[c * rows + r];
        cudaMemcpy(tmp, h_dst.data(), t->n_elements * sizeof(half), cudaMemcpyHostToDevice);
        return tmp;
    };

    weights_.embed_tokens = load_embed("token_embd.weight");
    weights_.final_layernorm = load_fp16("output_norm.weight");
    weights_.is_q4l = true;  // all projections are Q4L

    int SSM_CONV_DIM = config_.ssm_conv_dim();
    int SSM_VALUE_DIM = config_.ssm_value_dim();
    int SSM_NUM_V_HEADS = config_.ssm_num_v_heads;

    int n_attn = 0, n_ssm = 0;
    for (int i = 0; i < config_.num_layers; i++) {
        auto& L = weights_.layers[i];
        L = {};
        std::string p = "blk." + std::to_string(i) + ".";

        // Norms (small, always fp16)
        L.input_layernorm = load_fp16(p + "attn_norm.weight");
        L.post_attn_layernorm = load_fp16(p + "post_attention_norm.weight");
        if (!L.post_attn_layernorm) L.post_attn_layernorm = load_fp16(p + "ffn_norm.weight");

        // MLP projections (Q4L quantized)
        L.gate_proj_nf4 = load_q4l(p + "ffn_gate.weight", INTERMEDIATE_SIZE, HIDDEN_SIZE);
        L.up_proj_nf4 = load_q4l(p + "ffn_up.weight", INTERMEDIATE_SIZE, HIDDEN_SIZE);
        L.down_proj_nf4 = load_q4l(p + "ffn_down.weight", HIDDEN_SIZE, INTERMEDIATE_SIZE);
        L.mlp_is_q4l = true;

        if (config_.layer_type[i] == LAYER_ATTENTION) {
            // Attention projections (Q4L quantized)
            L.q_proj_nf4 = load_q4l(p + "attn_q.weight", Q_PROJ_DIM, HIDDEN_SIZE);
            L.k_proj_nf4 = load_q4l(p + "attn_k.weight", KV_DIM, HIDDEN_SIZE);
            L.v_proj_nf4 = load_q4l(p + "attn_v.weight", KV_DIM, HIDDEN_SIZE);
            L.o_proj_nf4 = load_q4l(p + "attn_output.weight", HIDDEN_SIZE, Q_DIM);
            L.attn_is_q4l = true;
            // QK norms (fp16)
            L.q_norm = load_fp16(p + "attn_q_norm.weight");
            L.k_norm = load_fp16(p + "attn_k_norm.weight");
            n_attn++;
        } else {
            // SSM projections (Q4L for large, fp16 for small)
            L.ssm_in_proj_qkv_nf4 = load_q4l(p + "attn_qkv.weight", SSM_CONV_DIM, HIDDEN_SIZE);
            L.ssm_in_proj_z_nf4 = load_q4l(p + "attn_gate.weight", SSM_VALUE_DIM, HIDDEN_SIZE);
            L.ssm_out_proj_nf4 = load_q4l(p + "ssm_out.weight", HIDDEN_SIZE, SSM_VALUE_DIM);
            L.ssm_is_q4l = true;
            // Small SSM params (fp16)
            L.ssm_in_proj_a_fp16 = load_fp16(p + "ssm_alpha.weight");
            L.ssm_in_proj_b_fp16 = load_fp16(p + "ssm_beta.weight");
            L.ssm_conv1d_weight = load_fp16(p + "ssm_conv1d.weight");
            L.ssm_conv1d_bias = nullptr;
            L.ssm_A_log = load_fp16(p + "ssm_a");
            L.ssm_dt_bias = load_fp16(p + "ssm_dt.bias");
            L.ssm_norm_weight = load_fp16(p + "ssm_norm.weight");
            n_ssm++;
        }

        L.lora_q = L.lora_k = L.lora_v = L.lora_o = nullptr;
        L.lora_gate = L.lora_up = L.lora_down = nullptr;
        L.lora_ssm_qkv = L.lora_ssm_z = L.lora_ssm_out = nullptr;
    }

    fprintf(stderr, "  Loaded %d attention + %d SSM layers from GGUF (Q4L quantized)\n", n_attn, n_ssm);
}

void InferenceEngine::load_config(const std::string& config_path) {
    config_ = ModelConfig::from_json(config_path);
    if (!state_.hidden) {
        // First time: allocate all GPU buffers
        allocate_buffers();
    }

    // Zero all weight pointers (will be set via share_weight/share_weight_nf4)
    weights_.embed_tokens = nullptr;
    weights_.final_layernorm = nullptr;
    weights_.is_q4l = true;
    for (int i = 0; i < config_.num_layers; i++) {
        auto& L = weights_.layers[i];
        // Zero only the weight DATA pointers, keep struct metadata
        L.q_proj_fp16 = L.k_proj_fp16 = L.v_proj_fp16 = L.o_proj_fp16 = nullptr;
        L.gate_proj_fp16 = L.up_proj_fp16 = L.down_proj_fp16 = nullptr;
        L.input_layernorm = L.post_attn_layernorm = nullptr;
        L.q_norm = L.k_norm = nullptr;
        L.q_proj_nf4 = L.k_proj_nf4 = L.v_proj_nf4 = L.o_proj_nf4 = {};
        L.gate_proj_nf4 = L.up_proj_nf4 = L.down_proj_nf4 = {};
        L.attn_is_nf4 = L.attn_is_q4l = false;
        L.mlp_is_nf4 = L.mlp_is_q4l = false;
        L.ssm_in_proj_qkv_fp16 = L.ssm_in_proj_z_fp16 = nullptr;
        L.ssm_out_proj_fp16 = nullptr;
        L.ssm_in_proj_a_fp16 = L.ssm_in_proj_b_fp16 = nullptr;
        L.ssm_conv1d_weight = L.ssm_conv1d_bias = nullptr;
        L.ssm_A_log = L.ssm_dt_bias = L.ssm_norm_weight = nullptr;
        L.ssm_in_proj_qkv_nf4 = L.ssm_in_proj_z_nf4 = L.ssm_out_proj_nf4 = {};
        L.ssm_is_q4l = false;
        L.lora_q = L.lora_k = L.lora_v = L.lora_o = nullptr;
        L.lora_gate = L.lora_up = L.lora_down = nullptr;
        L.lora_ssm_qkv = L.lora_ssm_z = L.lora_ssm_out = nullptr;
    }
    fprintf(stderr, "  Config loaded: %dh, %dL, %dQh/%dKVh, %dhd\n",
            config_.hidden_size, config_.num_layers, config_.num_heads,
            config_.num_kv_heads, config_.head_dim);
}

void InferenceEngine::share_weight(int layer, const char* name, half* ptr) {
    std::string n(name);

    // Model-level weights (layer index ignored)
    if (n == "final_layernorm") { weights_.final_layernorm = ptr; return; }

    if (layer < 0 || layer >= config_.num_layers) return;
    auto& L = weights_.layers[layer];

    // Attention projections
    if (n == "q_proj") { L.q_proj_fp16 = ptr; }
    else if (n == "k_proj") { L.k_proj_fp16 = ptr; }
    else if (n == "v_proj") { L.v_proj_fp16 = ptr; }
    else if (n == "o_proj") { L.o_proj_fp16 = ptr; }
    // MLP projections
    else if (n == "gate_proj") { L.gate_proj_fp16 = ptr; }
    else if (n == "up_proj") { L.up_proj_fp16 = ptr; }
    else if (n == "down_proj") { L.down_proj_fp16 = ptr; }
    // Norms
    else if (n == "input_layernorm") { L.input_layernorm = ptr; }
    else if (n == "post_attention_layernorm") { L.post_attn_layernorm = ptr; }
    else if (n == "q_norm") { L.q_norm = ptr; }
    else if (n == "k_norm") { L.k_norm = ptr; }
    // SSM projections
    else if (n == "ssm_in_proj_qkv") { L.ssm_in_proj_qkv_fp16 = ptr; }
    else if (n == "ssm_in_proj_z") { L.ssm_in_proj_z_fp16 = ptr; }
    else if (n == "ssm_in_proj_a") { L.ssm_in_proj_a_fp16 = ptr; }
    else if (n == "ssm_in_proj_b") { L.ssm_in_proj_b_fp16 = ptr; }
    else if (n == "ssm_out_proj") { L.ssm_out_proj_fp16 = ptr; }
    else if (n == "ssm_conv1d_weight") { L.ssm_conv1d_weight = ptr; }
    else if (n == "ssm_conv1d_bias") { L.ssm_conv1d_bias = ptr; }
    else if (n == "ssm_norm") { L.ssm_norm_weight = ptr; }
    else if (n == "ssm_A_log") { L.ssm_A_log = ptr; }
    else if (n == "ssm_dt_bias") { L.ssm_dt_bias = ptr; }
}

void InferenceEngine::share_weight_nf4(int layer, const char* name,
                                        uint8_t* data, float* absmax, float* quant_map,
                                        int out_dim, int in_dim) {
    if (layer < 0 || layer >= config_.num_layers) return;
    auto& L = weights_.layers[layer];
    std::string n(name);

    auto make_nf4 = [&](NF4Weight& old) -> NF4Weight {
        // Free old weight data if it was internally allocated
        if (old.data) cudaFree(old.data);
        if (old.absmax) cudaFree(old.absmax);
        if (old.quant_map) cudaFree(old.quant_map);
        if (old.fp16_cache) cudaFree(old.fp16_cache);
        NF4Weight w = {};
        w.data = data;
        w.absmax = absmax;
        w.quant_map = quant_map;
        w.fp16_cache = nullptr;
        w.out_dim = out_dim;
        w.in_dim = in_dim;
        w.block_size = 64;
        w.n_blocks = ((size_t)out_dim * in_dim + 63) / 64;
        return w;
    };

    // Detect format: quant_map=null → Q4L, quant_map!=null → NF4
    bool is_q4l = (quant_map == nullptr);

    // Attention projections
    if (n == "q_proj") { L.q_proj_nf4 = make_nf4(L.q_proj_nf4); if (is_q4l) L.attn_is_q4l = true; else L.attn_is_nf4 = true; }
    else if (n == "k_proj") { L.k_proj_nf4 = make_nf4(L.k_proj_nf4); }
    else if (n == "v_proj") { L.v_proj_nf4 = make_nf4(L.v_proj_nf4); }
    else if (n == "o_proj") { L.o_proj_nf4 = make_nf4(L.o_proj_nf4); }
    // MLP
    else if (n == "gate_proj") { L.gate_proj_nf4 = make_nf4(L.gate_proj_nf4); if (is_q4l) L.mlp_is_q4l = true; else L.mlp_is_nf4 = true; }
    else if (n == "up_proj") { L.up_proj_nf4 = make_nf4(L.up_proj_nf4); }
    else if (n == "down_proj") { L.down_proj_nf4 = make_nf4(L.down_proj_nf4); }
    // SSM
    else if (n == "ssm_in_proj_qkv") { L.ssm_in_proj_qkv_nf4 = make_nf4(L.ssm_in_proj_qkv_nf4); if (is_q4l) L.ssm_is_q4l = true; }
    else if (n == "ssm_in_proj_z") { L.ssm_in_proj_z_nf4 = make_nf4(L.ssm_in_proj_z_nf4); }
    else if (n == "ssm_out_proj") { L.ssm_out_proj_nf4 = make_nf4(L.ssm_out_proj_nf4); }

    if (is_q4l) weights_.is_q4l = true;
}

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
        if (config_.layer_type[i] == LAYER_SSM) {
            // SSM layers don't use KV cache — their state is in batch_->ssm_state
            continue;
        }
        if (kv_bits_ == 0) {
            if (state_.kv_cache[i].key)
                cudaMemset(state_.kv_cache[i].key, 0, state_.max_seq_len * KV_DIM * sizeof(half));
            if (state_.kv_cache[i].value)
                cudaMemset(state_.kv_cache[i].value, 0, state_.max_seq_len * KV_DIM * sizeof(half));
        } else {
            if (state_.kv_cache[i].key_quant)
                cudaMemset(state_.kv_cache[i].key_quant, 0, state_.max_seq_len * (KV_DIM_PACKED));
            if (state_.kv_cache[i].value_quant)
                cudaMemset(state_.kv_cache[i].value_quant, 0, state_.max_seq_len * (KV_DIM_PACKED));
            if (state_.kv_cache[i].key_norms)
                cudaMemset(state_.kv_cache[i].key_norms, 0, state_.max_seq_len * NUM_KV_HEADS * sizeof(half));
            if (state_.kv_cache[i].value_norms)
                cudaMemset(state_.kv_cache[i].value_norms, 0, state_.max_seq_len * NUM_KV_HEADS * sizeof(half));
        }
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
        cudaFree(state_.kv_cache[i].key_quant); state_.kv_cache[i].key_quant = nullptr;
        cudaFree(state_.kv_cache[i].value_quant); state_.kv_cache[i].value_quant = nullptr;
        cudaFree(state_.kv_cache[i].key_norms); state_.kv_cache[i].key_norms = nullptr;
        cudaFree(state_.kv_cache[i].value_norms); state_.kv_cache[i].value_norms = nullptr;
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
            if (kv_bits_ == 0) {
                cudaMallocChecked(&state_.kv_cache[i].key, max_seq_len * KV_DIM * sizeof(half));
                cudaMallocChecked(&state_.kv_cache[i].value, max_seq_len * KV_DIM * sizeof(half));
            } else {
                cudaMallocChecked(&state_.kv_cache[i].key_quant,
                                  max_seq_len * (KV_DIM_PACKED) * sizeof(uint8_t));
                cudaMallocChecked(&state_.kv_cache[i].value_quant,
                                  max_seq_len * (KV_DIM_PACKED) * sizeof(uint8_t));
                cudaMallocChecked(&state_.kv_cache[i].key_norms,
                                  max_seq_len * NUM_KV_HEADS * sizeof(half));
                cudaMallocChecked(&state_.kv_cache[i].value_norms,
                                  max_seq_len * NUM_KV_HEADS * sizeof(half));
            }
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
    if (kv_bits_ == 0) {
        cudaMemcpyAsync(kv.key + state_.current_pos * KV_DIM, state_.k_buf,
                         KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(kv.value + state_.current_pos * KV_DIM, state_.v_buf,
                         KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    } else {
        // TurboQuant: rotate + quantize K and V to 2-bit
        // Use a 1-element position array on the stack (single-sequence decode)
        int h_pos = state_.current_pos;
        int* d_pos_tmp;
        cudaMallocChecked(&d_pos_tmp, sizeof(int));
        cudaMemcpyAsync(d_pos_tmp, &h_pos, sizeof(int), cudaMemcpyHostToDevice, stream);
        launch_turbo_kv_quantize(kv.key_quant, kv.key_norms,
                                 state_.k_buf, state_.turbo_rotation, d_pos_tmp,
                                 state_.max_seq_len, 1,
                                 NUM_KV_HEADS, HEAD_DIM, KV_DIM,
                                 kv_bits_, stream);
        launch_turbo_kv_quantize(kv.value_quant, kv.value_norms,
                                 state_.v_buf, state_.turbo_rotation, d_pos_tmp,
                                 state_.max_seq_len, 1,
                                 NUM_KV_HEADS, HEAD_DIM, KV_DIM,
                                 kv_bits_, stream);
        cudaFree(d_pos_tmp);
    }

    // 5. GQA Attention
    if (kv_bits_ == 0) {
        launch_gqa_attention(state_.q_buf, kv.key, kv.value, state_.attn_out,
                              state_.attn_scores, state_.current_pos, state_.max_seq_len,
                              NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, stream);
    } else {
        launch_turbo_gqa_attention(state_.q_buf,
                                    kv.key_quant, kv.value_quant,
                                    kv.key_norms, kv.value_norms,
                                    state_.turbo_rotation,
                                    state_.attn_out, state_.attn_scores,
                                    state_.current_pos, state_.max_seq_len,
                                    NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, kv_bits_, stream);
    }

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
    // SSM (Gated Delta Rule) projections
    else if (proj == "ssm_qkv") target = &layer.lora_ssm_qkv;
    else if (proj == "ssm_z") target = &layer.lora_ssm_z;
    else if (proj == "ssm_out") target = &layer.lora_ssm_out;
    if (!target) return;

    // Create or update adapter
    if (!*target) {
        *target = new LoRAAdapter();
        cudaMallocChecked(&(*target)->A, A_rows * A_cols * sizeof(half));
        cudaMallocChecked(&(*target)->B, B_rows * B_cols * sizeof(half));
    } else {
        // Resize if dimensions changed
        LoRAAdapter* old = *target;
        if (old->rank != A_rows || old->in_features != A_cols || old->out_features != B_rows) {
            cudaFree(old->A);
            cudaFree(old->B);
            cudaMallocChecked(&old->A, A_rows * A_cols * sizeof(half));
            cudaMallocChecked(&old->B, B_rows * B_cols * sizeof(half));
        }
    }

    // Copy weights (auto-detect host vs device source)
    cudaPointerAttributes attr;
    cudaPointerGetAttributes(&attr, A_data);
    auto kind = (attr.type == cudaMemoryTypeDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy((*target)->A, A_data, A_rows * A_cols * sizeof(half), kind);
    cudaMemcpy((*target)->B, B_data, B_rows * B_cols * sizeof(half), kind);
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
        // Free LoRA adapters from old batch state
        delete[] batch_->h_positions; delete[] batch_->h_tokens;
        delete[] batch_->h_finished; delete[] batch_->h_randoms;
        delete[] batch_->h_token_history;
        if (batch_->d_token_history) cudaFree(batch_->d_token_history);
        delete batch_;
        batch_ = nullptr;
    }

    // Activation buffers sized for max(G, prefill_tokens) to support chunked prefill.
    // During prefill: process T prompt tokens in one GEMM (N=T).
    // During decode: process G sequences in parallel (N=G).
    // Prompt is at most max_seq_len tokens, but typically < 256.
    int N = std::max(G, std::min(max_seq_len, 256));

    // Calculate total GPU memory needed
    size_t need = 0;
    auto add = [&](size_t bytes) { need = (need + 255) & ~(size_t)255; need += bytes; };
    add(HIDDEN_SIZE * N * sizeof(half));          // hidden
    add(HIDDEN_SIZE * N * sizeof(half));          // residual
    add(HIDDEN_SIZE * N * sizeof(half));          // norm_buf
    add(Q_PROJ_DIM * N * sizeof(half));            // q_buf (2x for gated attn)
    add(KV_DIM * N * sizeof(half));               // k_buf
    add(KV_DIM * N * sizeof(half));               // v_buf
    add(Q_DIM * N * sizeof(half));                // attn_out
    add(INTERMEDIATE_SIZE * N * sizeof(half));     // gate_buf
    add(INTERMEDIATE_SIZE * N * sizeof(half));     // up_buf
    add(VOCAB_SIZE * G * sizeof(float));           // logits (decode only, stays at G)
    add(std::max((size_t)G, (size_t)N) * NUM_HEADS * max_seq_len * sizeof(float)); // attn_scores
    if (!weights_cached_) {
        // Size for largest weight matrix: max of MLP (intermediate×hidden) and SSM (conv_dim×hidden)
        size_t max_proj = std::max((size_t)INTERMEDIATE_SIZE, (size_t)config_.ssm_conv_dim());
        add(max_proj * HIDDEN_SIZE * sizeof(half)); // dequant_scratch
    }
    if (weights_.is_q4l) {
        int max_in = std::max(HIDDEN_SIZE, INTERMEDIATE_SIZE);
        add(max_in * sizeof(int8_t));           // q8_data
        add((max_in / 64) * sizeof(float));     // q8_scales
        add((max_in / 64) * sizeof(float));     // q8_sums
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (config_.layer_type[i] == LAYER_SSM) {
            // SSM state: (G, num_v_heads, k_head_dim, v_head_dim) per layer — fp32
            add((size_t)G * config_.ssm_num_v_heads * config_.ssm_k_head_dim
                * config_.ssm_v_head_dim * sizeof(float));
            // Conv state: (G, conv_dim, conv_kernel-1)
            add((size_t)G * config_.ssm_conv_dim() * (config_.ssm_conv_kernel - 1) * sizeof(float));
        } else if (kv_bits_ == 0) {
            add((size_t)G * max_seq_len * KV_DIM * sizeof(half));  // kv_keys
            add((size_t)G * max_seq_len * KV_DIM * sizeof(half));  // kv_values
        } else {
            add((size_t)G * max_seq_len * (KV_DIM_PACKED));           // kv_keys_q (2-bit packed)
            add((size_t)G * max_seq_len * (KV_DIM_PACKED));           // kv_values_q
            add((size_t)G * max_seq_len * NUM_KV_HEADS * sizeof(half)); // kv_keys_norms
            add((size_t)G * max_seq_len * NUM_KV_HEADS * sizeof(half)); // kv_values_norms
        }
    }
    // SSM scratch buffers (shared across layers, sized for largest)
    if (config_.is_hybrid()) {
        int cd = config_.ssm_conv_dim();
        int vd = config_.ssm_value_dim();
        int nv = config_.ssm_num_v_heads;
        add((size_t)cd * N * sizeof(half));              // ssm_qkv_buf (fp16 GEMM output)
        add((size_t)cd * N * sizeof(float));             // ssm_qkv_fp32 (conv1d output)
        add((size_t)vd * N * sizeof(half));              // ssm_z_buf
        add((size_t)vd * N * sizeof(float));             // ssm_y_fp32 (delta rule output)
        add((size_t)vd * N * sizeof(half));              // ssm_y_buf (after rmsnorm, for GEMM)
        add((size_t)2 * nv * N * sizeof(float));         // ssm_dt_buf (decay + beta) — N not G for prefill
        add((size_t)nv * N * sizeof(half));              // ssm_a_buf
        add((size_t)nv * N * sizeof(half));              // ssm_b_buf

        // Chunked prefill workspace
        int hd = config_.ssm_k_head_dim;
        int cs = 64;  // chunk size
        int T_padded = ((N + cs - 1) / cs) * cs;
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_Q
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_K
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_V
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_K_beta
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_V_beta
        add((size_t)nv * N * sizeof(float));               // ssm_chunk_g
        add((size_t)nv * N * sizeof(float));               // ssm_chunk_beta
        add((size_t)nv * T_padded * hd * sizeof(float));  // ssm_chunk_output
        int ws_per_head = cs * cs + 3 * cs * hd + cs;
        add((size_t)nv * ws_per_head * sizeof(float));     // ssm_chunk_workspace
    }
    add(N * sizeof(int));   // d_positions (max(G, T))
    add(N * sizeof(int));   // d_tokens (max(G, T))
    add(G * sizeof(float)); // d_randoms
    add(64 * N * sizeof(half)); // lora_scratch (N = max(G, prompt_len) for prefill LoRA)

    // (Re)allocate arena if needed
    if (need > batch_arena_.capacity) {
        // Only cudaMalloc if no external arena was set
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
    batch_->d_randoms_ptr = nullptr;
    batch_->d_token_history = nullptr;
    batch_->h_token_history = nullptr;
    batch_->token_history_size = 0;

    batch_->hidden    = (half*)batch_arena_.alloc(HIDDEN_SIZE * N * sizeof(half));
    batch_->residual  = (half*)batch_arena_.alloc(HIDDEN_SIZE * N * sizeof(half));
    batch_->norm_buf  = (half*)batch_arena_.alloc(HIDDEN_SIZE * N * sizeof(half));
    batch_->q_buf     = (half*)batch_arena_.alloc(Q_PROJ_DIM * N * sizeof(half));
    batch_->k_buf     = (half*)batch_arena_.alloc(KV_DIM * N * sizeof(half));
    batch_->v_buf     = (half*)batch_arena_.alloc(KV_DIM * N * sizeof(half));
    batch_->attn_out  = (half*)batch_arena_.alloc(Q_DIM * N * sizeof(half));
    batch_->gate_buf  = (half*)batch_arena_.alloc(INTERMEDIATE_SIZE * N * sizeof(half));
    batch_->up_buf    = (half*)batch_arena_.alloc(INTERMEDIATE_SIZE * N * sizeof(half));
    batch_->logits    = (float*)batch_arena_.alloc(VOCAB_SIZE * G * sizeof(float));
    batch_->attn_scores = (float*)batch_arena_.alloc(std::max((size_t)G, (size_t)N) * NUM_HEADS * max_seq_len * sizeof(float));
    if (!weights_cached_) {
        size_t max_proj = std::max((size_t)INTERMEDIATE_SIZE, (size_t)config_.ssm_conv_dim());
        batch_->dequant_scratch = (half*)batch_arena_.alloc(max_proj * HIDDEN_SIZE * sizeof(half));
    } else {
        batch_->dequant_scratch = nullptr;
    }
    // dp4a q8 buffers for batched GEMV (decode path, reused per-sequence)
    if (weights_.is_q4l) {
        int max_in = std::max(HIDDEN_SIZE, INTERMEDIATE_SIZE);
        batch_->q8_data = (int8_t*)batch_arena_.alloc(max_in * sizeof(int8_t));
        batch_->q8_scales = (float*)batch_arena_.alloc((max_in / 64) * sizeof(float));
        batch_->q8_sums = (float*)batch_arena_.alloc((max_in / 64) * sizeof(float));
    } else {
        batch_->q8_data = nullptr;
        batch_->q8_scales = nullptr;
        batch_->q8_sums = nullptr;
    }
    for (int i = 0; i < NUM_LAYERS; i++) {
        // Initialize all pointers to null
        batch_->kv_keys[i] = batch_->kv_values[i] = nullptr;
        batch_->kv_keys_q[i] = batch_->kv_values_q[i] = nullptr;
        batch_->kv_keys_norms[i] = batch_->kv_values_norms[i] = nullptr;
        batch_->ssm_state[i] = nullptr;
        batch_->ssm_conv_state[i] = nullptr;

        if (config_.layer_type[i] == LAYER_SSM) {
            batch_->ssm_state[i] = (float*)batch_arena_.alloc(
                (size_t)G * config_.ssm_num_v_heads * config_.ssm_k_head_dim
                * config_.ssm_v_head_dim * sizeof(float));  // fp16 state
            batch_->ssm_conv_state[i] = (float*)batch_arena_.alloc(
                (size_t)G * config_.ssm_conv_dim() * (config_.ssm_conv_kernel - 1) * sizeof(float));
        } else if (kv_bits_ == 0) {
            batch_->kv_keys[i]   = (half*)batch_arena_.alloc((size_t)G * max_seq_len * KV_DIM * sizeof(half));
            batch_->kv_values[i] = (half*)batch_arena_.alloc((size_t)G * max_seq_len * KV_DIM * sizeof(half));
        } else {
            batch_->kv_keys_q[i]   = (uint8_t*)batch_arena_.alloc((size_t)G * max_seq_len * (KV_DIM_PACKED));
            batch_->kv_values_q[i] = (uint8_t*)batch_arena_.alloc((size_t)G * max_seq_len * (KV_DIM_PACKED));
            batch_->kv_keys_norms[i]   = (half*)batch_arena_.alloc((size_t)G * max_seq_len * NUM_KV_HEADS * sizeof(half));
            batch_->kv_values_norms[i] = (half*)batch_arena_.alloc((size_t)G * max_seq_len * NUM_KV_HEADS * sizeof(half));
        }
    }
    // SSM scratch buffers (shared across all SSM layers)
    if (config_.is_hybrid()) {
        int cd = config_.ssm_conv_dim();
        int vd = config_.ssm_value_dim();
        int nv = config_.ssm_num_v_heads;
        batch_->ssm_qkv_buf = (half*)batch_arena_.alloc((size_t)cd * N * sizeof(half));
        batch_->ssm_qkv_fp32 = (float*)batch_arena_.alloc((size_t)cd * N * sizeof(float));
        batch_->ssm_z_buf   = (half*)batch_arena_.alloc((size_t)vd * N * sizeof(half));
        batch_->ssm_y_fp32  = (float*)batch_arena_.alloc((size_t)vd * N * sizeof(float));
        batch_->ssm_y_buf   = (half*)batch_arena_.alloc((size_t)vd * N * sizeof(half));
        batch_->ssm_dt_buf  = (float*)batch_arena_.alloc((size_t)2 * nv * N * sizeof(float)); // decay + beta — N for prefill
        batch_->ssm_a_buf   = (half*)batch_arena_.alloc((size_t)nv * N * sizeof(half));
        batch_->ssm_b_buf   = (half*)batch_arena_.alloc((size_t)nv * N * sizeof(half));

        // Chunked prefill workspace
        int hd = config_.ssm_k_head_dim;
        int cs = 64;
        int T_padded = ((N + cs - 1) / cs) * cs;
        batch_->ssm_chunk_Q       = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        batch_->ssm_chunk_K       = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        batch_->ssm_chunk_V       = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        batch_->ssm_chunk_K_beta  = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        batch_->ssm_chunk_V_beta  = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        batch_->ssm_chunk_g       = (float*)batch_arena_.alloc((size_t)nv * N * sizeof(float));
        batch_->ssm_chunk_beta    = (float*)batch_arena_.alloc((size_t)nv * N * sizeof(float));
        batch_->ssm_chunk_output  = (float*)batch_arena_.alloc((size_t)nv * T_padded * hd * sizeof(float));
        int ws_per_head = cs * cs + 3 * cs * hd + cs;
        batch_->ssm_chunk_workspace = (float*)batch_arena_.alloc((size_t)nv * ws_per_head * sizeof(float));
    } else {
        batch_->ssm_qkv_buf = batch_->ssm_z_buf = batch_->ssm_y_buf = nullptr;
        batch_->ssm_dt_buf = nullptr;
        batch_->ssm_a_buf = batch_->ssm_b_buf = nullptr;
        batch_->ssm_chunk_Q = batch_->ssm_chunk_K = batch_->ssm_chunk_V = nullptr;
        batch_->ssm_chunk_K_beta = batch_->ssm_chunk_V_beta = nullptr;
        batch_->ssm_chunk_g = batch_->ssm_chunk_beta = nullptr;
        batch_->ssm_chunk_output = batch_->ssm_chunk_workspace = nullptr;
    }

    batch_->d_positions  = (int*)batch_arena_.alloc(N * sizeof(int));
    batch_->d_tokens     = (int*)batch_arena_.alloc(N * sizeof(int));
    batch_->d_randoms    = (float*)batch_arena_.alloc(G * sizeof(float));
    batch_->lora_scratch = (half*)batch_arena_.alloc(64 * N * sizeof(half));

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
    ensure_cublas(stream);
    // weight is (M, K) row-major = (K, M) col-major for cuBLAS
    // in is (K, N) col-major, out is (M, N) col-major
    // Use fp32 accumulation for precision (fp16 loses accuracy for K=1024+)
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t stat = cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 M, N, K, &alpha,
                 weight, CUDA_R_16F, K,
                 in, CUDA_R_16F, K,
                 &beta, out, CUDA_R_16F, M,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[batch_gemm] cuBLAS error %d: M=%d N=%d K=%d weight=%p in=%p out=%p stream=%p\n",
                (int)stat, M, N, K, (void*)weight, (void*)in, (void*)out, (void*)stream);
    }
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
        // Attention projections
        cache_one(L.q_proj_nf4); cache_one(L.k_proj_nf4);
        cache_one(L.v_proj_nf4); cache_one(L.o_proj_nf4);
        // MLP projections
        cache_one(L.gate_proj_nf4); cache_one(L.up_proj_nf4);
        cache_one(L.down_proj_nf4);
        // SSM projections (hybrid models)
        cache_one(L.ssm_in_proj_qkv_nf4);
        cache_one(L.ssm_in_proj_z_nf4);
        cache_one(L.ssm_out_proj_nf4);
    }
    cudaDeviceSynchronize();
    weights_cached_ = true;
    std::cout << "  Cached " << total_bytes / 1e6 << "MB dequanted weights" << std::endl;
}

void InferenceEngine::set_arena(void* ptr, size_t size) {
    // Use externally-allocated memory (from PyTorch) as the batch arena.
    // This eliminates cudaMalloc/PyTorch allocator contention.
    if (batch_arena_.base && batch_arena_.capacity > 0) {
        // Free internally-allocated arena before replacing
        cudaFree(batch_arena_.base);
    }
    batch_arena_.base = (char*)ptr;
    batch_arena_.capacity = size;
    batch_arena_.offset = 0;
    std::cout << "  Arena set to external memory (" << size / 1e6 << "MB)" << std::endl;
}

void InferenceEngine::batch_gemm_q4l(half* out, const NF4Weight& w, const half* in,
                                      int N, cudaStream_t stream) {
    if (w.fp16_cache) {
        // Use pre-dequanted fp16 weights with cuBLAS tensor cores
        batch_gemm(out, w.fp16_cache, in, w.out_dim, N, w.in_dim, stream);
    } else if (N <= 8 && w.quant_map) {
        // NF4 batched W4A16: read weights once, multiply against N fp16 inputs
        launch_nf4_batch_gemv(w.data, w.absmax, in, out,
                              w.out_dim, w.in_dim, N, stream);
    } else if (N <= 8 && batch_->q8_data) {
        // Q4L format (no lookup table): N independent dp4a GEMVs
        for (int g = 0; g < N; g++) {
            const half* in_g = in + (size_t)g * w.in_dim;
            half* out_g = out + (size_t)g * w.out_dim;
            launch_quantize_input_q8(in_g, batch_->q8_data, batch_->q8_scales,
                                     batch_->q8_sums, w.in_dim, stream);
            launch_q4l_dp4a_gemv(w.data, w.absmax, batch_->q8_data,
                                 batch_->q8_scales, batch_->q8_sums,
                                 out_g, w.out_dim, w.in_dim, stream);
        }
    } else if (w.quant_map) {
        // NF4 large batch (prefill): chunks of 8
        for (int start = 0; start < N; start += 8) {
            int chunk = std::min(8, N - start);
            launch_nf4_batch_gemv(w.data, w.absmax,
                                  in + (size_t)start * w.in_dim,
                                  out + (size_t)start * w.out_dim,
                                  w.out_dim, w.in_dim, chunk, stream);
        }
    } else {
        // Q4L large batch (prefill): dequant to scratch then cuBLAS
        launch_dequant_q4l(batch_->dequant_scratch, w.data, w.absmax,
                           w.out_dim, w.in_dim, stream);
        batch_gemm(out, (const half*)batch_->dequant_scratch, in,
                   w.out_dim, N, w.in_dim, stream);
    }
}

// ============================================================================
// SSM (Gated Delta Rule) batch forward — one decode step for G sequences
// ============================================================================
void InferenceEngine::forward_layer_ssm_batch(int layer_idx, int G, cudaStream_t stream) {
    auto& L = weights_.layers[layer_idx];
    auto* B = batch_;
    ensure_cublas(stream);

    int SSM_CONV_DIM = config_.ssm_conv_dim();
    int SSM_KEY_DIM = config_.ssm_key_dim();
    int SSM_VALUE_DIM = config_.ssm_value_dim();
    int SSM_NUM_K_HEADS = config_.ssm_num_k_heads;
    int SSM_NUM_V_HEADS = config_.ssm_num_v_heads;
    int SSM_K_HEAD_DIM = config_.ssm_k_head_dim;
    int SSM_V_HEAD_DIM = config_.ssm_v_head_dim;

    // Batched projection with optional LoRA (same pattern as attention forward)
    auto project = [&](half* out, half* fp16w, NF4Weight& nf4w, const half* input,
                       int out_dim, int in_dim, const LoRAAdapter* lora = nullptr) {
        if (fp16w) batch_gemm(out, fp16w, input, out_dim, G, in_dim, stream);
        else batch_gemm_q4l(out, nf4w, input, G, stream);
        if (lora && lora->A && lora->B) {
            ensure_cublas(stream);
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         lora->rank, G, lora->in_features, &alpha,
                         lora->A, CUDA_R_16F, lora->in_features,
                         input, CUDA_R_16F, lora->in_features,
                         &beta, B->lora_scratch, CUDA_R_16F, lora->rank,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
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

    // SSM_CHECK: no-op in release (debug sync kills perf: 60K syncs/step)
    #define SSM_CHECK(msg) ((void)0)

    // 1. RMSNorm
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    SSM_CHECK("copy");
    launch_rms_norm_batch(B->norm_buf, B->residual, L.input_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream, config_.is_hybrid());
    SSM_CHECK("rmsnorm");

    // 2. Projections: QKV, Z, A, B
    project(B->ssm_qkv_buf, L.ssm_in_proj_qkv_fp16, L.ssm_in_proj_qkv_nf4,
            B->norm_buf, SSM_CONV_DIM, HIDDEN_SIZE, L.lora_ssm_qkv);
    SSM_CHECK("qkv_proj");
    project(B->ssm_z_buf, L.ssm_in_proj_z_fp16, L.ssm_in_proj_z_nf4,
            B->norm_buf, SSM_VALUE_DIM, HIDDEN_SIZE, L.lora_ssm_z);
    SSM_CHECK("z_proj");
    // Small projections: always fp16 GEMV
    batch_gemm(B->ssm_a_buf, L.ssm_in_proj_a_fp16, B->norm_buf,
               SSM_NUM_V_HEADS, G, HIDDEN_SIZE, stream);
    SSM_CHECK("a_proj");
    batch_gemm(B->ssm_b_buf, L.ssm_in_proj_b_fp16, B->norm_buf,
               SSM_NUM_V_HEADS, G, HIDDEN_SIZE, stream);
    SSM_CHECK("b_proj");


    // 3. Causal conv1d on QKV: reads fp16 input, outputs fp32
    launch_ssm_conv1d_decode(B->ssm_qkv_fp32, B->ssm_conv_state[layer_idx],
                              B->ssm_qkv_buf,  // fp16 input from GEMM
                              L.ssm_conv1d_weight, L.ssm_conv1d_bias,
                              SSM_CONV_DIM, G, config_.ssm_conv_kernel, stream);
    SSM_CHECK("conv1d");

    // 4. Compute dt (decay) and beta from A, B inputs
    float* ssm_decay = B->ssm_dt_buf;
    float* ssm_beta = B->ssm_dt_buf + SSM_NUM_V_HEADS * G;
    launch_ssm_compute_dt_decay(ssm_decay, ssm_beta,
                                 B->ssm_a_buf, B->ssm_b_buf,
                                 L.ssm_A_log, L.ssm_dt_bias,
                                 SSM_NUM_V_HEADS, G, stream);
    SSM_CHECK("dt_decay");

    // 5. Split QKV into Q, K, V (fp32 pointers into ssm_qkv_fp32)
    float* ssm_q = B->ssm_qkv_fp32;                           // (key_dim, G)
    float* ssm_k = B->ssm_qkv_fp32 + SSM_KEY_DIM * G;         // (key_dim, G)
    float* ssm_v = B->ssm_qkv_fp32 + 2 * SSM_KEY_DIM * G;     // (value_dim, G)

    // 6. L2-normalize Q and K per head (fp32)
    launch_ssm_l2norm_qk(ssm_q, ssm_k, SSM_NUM_K_HEADS, SSM_K_HEAD_DIM, G, stream);

    // 7. Expand K heads to match V heads if needed (GQA-like)
    // Q and K have num_k_heads, but SSM state has num_v_heads.
    // The kernel handles this internally via kv_ratio = num_v_heads / num_k_heads.

    SSM_CHECK("l2norm");


    // 8. Gated Delta Rule: state update + output (fp32 for precision)
    launch_ssm_gated_delta_rule(B->ssm_state[layer_idx], B->ssm_y_fp32,
                                 ssm_q, ssm_k, ssm_v,
                                 ssm_decay, ssm_beta,
                                 SSM_NUM_V_HEADS, SSM_NUM_K_HEADS,
                                 SSM_K_HEAD_DIM, SSM_V_HEAD_DIM, G, stream);
    SSM_CHECK("gated_delta_rule");

    // 9. Gated RMSNorm: y_buf = RMSNorm(y_fp32) * SiLU(z) → fp16
    launch_ssm_gated_rmsnorm(B->ssm_y_buf, B->ssm_y_fp32, B->ssm_z_buf, L.ssm_norm_weight,
                              SSM_NUM_V_HEADS, SSM_V_HEAD_DIM, G, RMS_NORM_EPS, stream);
    SSM_CHECK("gated_rmsnorm");


    // 10. Output projection: hidden = out_proj @ y_buf
    project(B->hidden, L.ssm_out_proj_fp16, L.ssm_out_proj_nf4,
            B->ssm_y_buf, HIDDEN_SIZE, SSM_VALUE_DIM, L.lora_ssm_out);
    SSM_CHECK("out_proj");

    // 11. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);

    // 12. Post-SSM LayerNorm + MLP (same as attention layers)
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream, config_.is_hybrid());

    project(B->gate_buf, L.gate_proj_fp16, L.gate_proj_nf4,
            B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_gate);
    project(B->up_buf, L.up_proj_fp16, L.up_proj_nf4,
            B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_up);
    launch_silu_mul_batch(B->gate_buf, B->up_buf, INTERMEDIATE_SIZE * G, stream);
    project(B->hidden, L.down_proj_fp16, L.down_proj_nf4,
            B->gate_buf, HIDDEN_SIZE, INTERMEDIATE_SIZE, L.lora_down);

    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);
}

// ============================================================================
// Chunked SSM prefill: process all T tokens through one SSM layer using
// the parallel (chunked) Gated Delta Rule formulation.
// This avoids sequential error accumulation from token-by-token recurrence.
// ============================================================================
void InferenceEngine::forward_layer_ssm_prefill(int layer_idx, int T, cudaStream_t stream) {
    auto& L = weights_.layers[layer_idx];
    auto* B = batch_;
    ensure_cublas(stream);

    int SSM_CONV_DIM = config_.ssm_conv_dim();
    int SSM_KEY_DIM = config_.ssm_key_dim();
    int SSM_VALUE_DIM = config_.ssm_value_dim();
    int SSM_NUM_HEADS = config_.ssm_num_v_heads;  // 16
    int SSM_HEAD_DIM = config_.ssm_k_head_dim;     // 128

    int CHUNK_SIZE = 64;
    int T_padded = ((T + CHUNK_SIZE - 1) / CHUNK_SIZE) * CHUNK_SIZE;
    int num_chunks = T_padded / CHUNK_SIZE;

    // Batched projection with optional LoRA
    auto project = [&](half* out, half* fp16w, NF4Weight& nf4w, const half* input,
                       int out_dim, int in_dim, const LoRAAdapter* lora = nullptr) {
        if (fp16w) batch_gemm(out, fp16w, input, out_dim, T, in_dim, stream);
        else batch_gemm_q4l(out, nf4w, input, T, stream);
        if (lora && lora->A && lora->B) {
            ensure_cublas(stream);
            float alpha = 1.0f, beta = 0.0f;
            cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         lora->rank, T, lora->in_features, &alpha,
                         lora->A, CUDA_R_16F, lora->in_features,
                         input, CUDA_R_16F, lora->in_features,
                         &beta, B->lora_scratch, CUDA_R_16F, lora->rank,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            float one = 1.0f;
            float scale = lora->scale;
            cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                         lora->out_features, T, lora->rank, &scale,
                         lora->B, CUDA_R_16F, lora->rank,
                         B->lora_scratch, CUDA_R_16F, lora->rank,
                         &one, out, CUDA_R_16F, lora->out_features,
                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    };

    // 1. RMSNorm (all T tokens)
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * T, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.input_layernorm,
                          HIDDEN_SIZE, T, RMS_NORM_EPS, stream, config_.is_hybrid());

    // 2. Projections: QKV, Z, A, B (all T tokens at once via batched GEMM)
    project(B->ssm_qkv_buf, L.ssm_in_proj_qkv_fp16, L.ssm_in_proj_qkv_nf4,
            B->norm_buf, SSM_CONV_DIM, HIDDEN_SIZE, L.lora_ssm_qkv);
    project(B->ssm_z_buf, L.ssm_in_proj_z_fp16, L.ssm_in_proj_z_nf4,
            B->norm_buf, SSM_VALUE_DIM, HIDDEN_SIZE, L.lora_ssm_z);
    batch_gemm(B->ssm_a_buf, L.ssm_in_proj_a_fp16, B->norm_buf,
               SSM_NUM_HEADS, T, HIDDEN_SIZE, stream);
    batch_gemm(B->ssm_b_buf, L.ssm_in_proj_b_fp16, B->norm_buf,
               SSM_NUM_HEADS, T, HIDDEN_SIZE, stream);

    // 3. Causal conv1d for all T tokens (parallel prefill kernel)
    // Outputs fp32 QKV, also sets conv_state for future decode
    launch_ssm_causal_conv1d_prefill(B->ssm_qkv_fp32, B->ssm_conv_state[layer_idx],
                                      B->ssm_qkv_buf, L.ssm_conv1d_weight, L.ssm_conv1d_bias,
                                      SSM_CONV_DIM, T, config_.ssm_conv_kernel, stream);

    // 4. Compute raw gate g and beta for all T tokens
    launch_ssm_compute_g_beta(B->ssm_chunk_g, B->ssm_chunk_beta,
                               B->ssm_a_buf, B->ssm_b_buf,
                               L.ssm_A_log, L.ssm_dt_bias,
                               SSM_NUM_HEADS, T, stream);

    // 5. Rearrange QKV from (dim, T) to (H, T_padded, D), L2-normalize Q/K,
    //    scale Q by 1/sqrt(D), compute K*beta and V*beta
    launch_ssm_chunk_rearrange(B->ssm_chunk_Q, B->ssm_chunk_K, B->ssm_chunk_V,
                                B->ssm_chunk_K_beta, B->ssm_chunk_V_beta,
                                B->ssm_qkv_fp32, B->ssm_chunk_beta,
                                SSM_NUM_HEADS, SSM_HEAD_DIM, T, T_padded,
                                SSM_KEY_DIM, SSM_VALUE_DIM, SSM_CONV_DIM, stream);

    // 6. Zero the recurrent state for seq 0 before prefill
    size_t state_size = (size_t)SSM_NUM_HEADS * SSM_HEAD_DIM * SSM_HEAD_DIM * sizeof(float);
    cudaMemsetAsync(B->ssm_state[layer_idx], 0, state_size, stream);

    // 7. Process chunks sequentially through the chunked delta rule
    for (int c = 0; c < num_chunks; c++) {
        launch_ssm_chunked_delta_rule(
            B->ssm_state[layer_idx], B->ssm_chunk_output,
            B->ssm_chunk_Q, B->ssm_chunk_K, B->ssm_chunk_V,
            B->ssm_chunk_K_beta, B->ssm_chunk_V_beta,
            B->ssm_chunk_g, B->ssm_chunk_workspace,
            SSM_NUM_HEADS, SSM_HEAD_DIM, T, T_padded,
            c, CHUNK_SIZE, T, stream);
    }

    // 8. Rearrange output from (H, T_padded, D) back to (value_dim, T) col-major
    launch_ssm_chunk_output_rearrange(B->ssm_y_fp32, B->ssm_chunk_output,
                                       SSM_NUM_HEADS, SSM_HEAD_DIM, T, T_padded, stream);

    // 9. Gated RMSNorm (col-major): y = RMSNorm(y) * SiLU(z) → fp16
    // y_fp32 and z_buf are col-major from rearrange kernel / GEMM output
    launch_ssm_gated_rmsnorm_colmajor(B->ssm_y_buf, B->ssm_y_fp32, B->ssm_z_buf, L.ssm_norm_weight,
                                       SSM_NUM_HEADS, SSM_HEAD_DIM, T, RMS_NORM_EPS, stream);

    // 10. Output projection
    project(B->hidden, L.ssm_out_proj_fp16, L.ssm_out_proj_nf4,
            B->ssm_y_buf, HIDDEN_SIZE, SSM_VALUE_DIM, L.lora_ssm_out);

    // 11. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * T, stream);

    // 12. Post-SSM LayerNorm + MLP
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * T, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                          HIDDEN_SIZE, T, RMS_NORM_EPS, stream, config_.is_hybrid());

    project(B->gate_buf, L.gate_proj_fp16, L.gate_proj_nf4,
            B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_gate);
    project(B->up_buf, L.up_proj_fp16, L.up_proj_nf4,
            B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, L.lora_up);
    launch_silu_mul_batch(B->gate_buf, B->up_buf, INTERMEDIATE_SIZE * T, stream);
    project(B->hidden, L.down_proj_fp16, L.down_proj_nf4,
            B->gate_buf, HIDDEN_SIZE, INTERMEDIATE_SIZE, L.lora_down);

    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * T, stream);
}

void InferenceEngine::forward_layer_batch(int layer_idx, int G, cudaStream_t stream) {
    // Dispatch SSM layers to separate forward path
    if (config_.layer_type[layer_idx] == LAYER_SSM) {
        forward_layer_ssm_batch(layer_idx, G, stream);
        return;
    }

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
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream, config_.is_hybrid());

    // 2. QKV projections
    // For gated attention: q_buf holds [query | gate], each Q_DIM wide
    project(B->q_buf, L.q_proj_fp16, L.q_proj_nf4, B->norm_buf, Q_PROJ_DIM, HIDDEN_SIZE, L.lora_q);
    project(B->k_buf, L.k_proj_fp16, L.k_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE, L.lora_k);
    project(B->v_buf, L.v_proj_fp16, L.v_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE, L.lora_v);

    // For gated attention: split q_buf into query (first half) and gate (second half)
    // In col-major (Q_PROJ_DIM, G): query is rows [0, Q_DIM), gate is rows [Q_DIM, Q_PROJ_DIM)
    // The QK norm and RoPE only operate on the query part (first Q_DIM rows)
    // Gate stays in q_buf starting at offset Q_DIM (col-major: rows Q_DIM..Q_PROJ_DIM-1)

    // 3. QKNorm + RoPE (on query portion only)
    launch_qk_norm_batch(B->q_buf, B->k_buf, L.q_norm, L.k_norm,
                         NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, G, RMS_NORM_EPS,
                         Q_PROJ_DIM, KV_DIM, stream, config_.is_hybrid());
    launch_rope_batch(B->q_buf, B->k_buf, state_.rope_cos, state_.rope_sin,
                      B->d_positions, B->max_seq_len, G,
                      NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_PROJ_DIM, KV_DIM, config_.rope_dim, stream);

    // 4. KV cache write
    if (kv_bits_ == 0) {
        launch_kv_cache_write_batch(B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                     B->k_buf, B->v_buf, B->d_positions, B->max_seq_len, G,
                                     KV_DIM, stream);
    } else {
        launch_turbo_kv_quantize_batch(B->kv_keys_q[layer_idx], B->kv_keys_norms[layer_idx],
                                        B->k_buf, state_.turbo_rotation, B->d_positions,
                                        B->max_seq_len, G, NUM_KV_HEADS, HEAD_DIM, KV_DIM, kv_bits_, stream);
        launch_turbo_kv_quantize_batch(B->kv_values_q[layer_idx], B->kv_values_norms[layer_idx],
                                        B->v_buf, state_.turbo_rotation, B->d_positions,
                                        B->max_seq_len, G, NUM_KV_HEADS, HEAD_DIM, KV_DIM, kv_bits_, stream);
    }

    // 5. GQA attention
    if (kv_bits_ == 0) {
        launch_gqa_attention_batch(B->attn_out, B->q_buf,
                                    B->kv_keys[layer_idx], B->kv_values[layer_idx],
                                    B->attn_scores, B->d_positions, B->max_seq_len, G,
                                    NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM, stream);
    } else {
        launch_turbo_gqa_attention_batch(B->attn_out, B->q_buf,
                                          B->kv_keys_q[layer_idx], B->kv_values_q[layer_idx],
                                          B->kv_keys_norms[layer_idx], B->kv_values_norms[layer_idx],
                                          state_.turbo_rotation,
                                          B->attn_scores, B->d_positions, B->max_seq_len, G,
                                          NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_DIM, KV_DIM, kv_bits_, stream);
    }

    // 6. Gated attention: attn_out *= sigmoid(gate)
    // Gate is in q_buf rows [Q_DIM..Q_PROJ_DIM), col-major layout
    if (config_.gated_attn) {
        launch_sigmoid_gate_batch(B->attn_out, B->q_buf + Q_DIM, Q_DIM, G, stream);
    }

    // 7. Output projection
    project(B->hidden, L.o_proj_fp16, L.o_proj_nf4, B->attn_out, HIDDEN_SIZE, Q_DIM, L.lora_o);

    // 8. Residual add
    launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * G, stream);

    // 9. Post-attention norm
    launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * G, stream);
    launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream, config_.is_hybrid());

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
                          HIDDEN_SIZE, G, RMS_NORM_EPS, stream, config_.is_hybrid());

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

// ============================================================================
// Chunked prefill: process T prompt tokens in one forward pass per layer
// All G sequences share the same prompt (GRPO), so we prefill 1 sequence
// and broadcast the KV cache to all G sequences.
// ============================================================================

void InferenceEngine::prefill_chunked(int T, int G, cudaStream_t stream) {
    auto* B = batch_;
    ensure_cublas(stream);

    // Embed all T tokens at once: (HIDDEN_SIZE, T)
    launch_embed_batch(B->hidden, weights_.embed_tokens, B->d_tokens, T, HIDDEN_SIZE, stream);

    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        auto& L = weights_.layers[layer];

        // SSM layers: use chunked parallel delta rule (avoids sequential error accumulation)
        if (config_.layer_type[layer] == LAYER_SSM) {
            forward_layer_ssm_prefill(layer, T, stream);
            continue;  // skip the attention/MLP code below
        }

        // 1. Copy + RMSNorm (treat T as batch dim)
        launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * T, stream);
        launch_rms_norm_batch(B->norm_buf, B->residual, L.input_layernorm,
                              HIDDEN_SIZE, T, RMS_NORM_EPS, stream, config_.is_hybrid());

        // 2. QKV projections: GEMM with N=T
        auto project = [&](half* out, half* fp16w, NF4Weight& nf4w, const half* input,
                           int out_dim, int in_dim) {
            if (fp16w) batch_gemm(out, fp16w, input, out_dim, T, in_dim, stream);
            else batch_gemm_q4l(out, nf4w, input, T, stream);
            // No LoRA during prefill (LoRA not synced yet at this point)
        };
        project(B->q_buf, L.q_proj_fp16, L.q_proj_nf4, B->norm_buf, Q_PROJ_DIM, HIDDEN_SIZE);
        project(B->k_buf, L.k_proj_fp16, L.k_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE);
        project(B->v_buf, L.v_proj_fp16, L.v_proj_nf4, B->norm_buf, KV_DIM, HIDDEN_SIZE);

        // 3. QKNorm + RoPE
        // For gated attention: q_buf layout is [query(Q_DIM) | gate(Q_DIM)] per token in col-major
        // QK norm and RoPE operate on query only. The q_dim stride is Q_PROJ_DIM (so gate is skipped).
        // But head_idx * head_dim addresses correctly since weights are reordered to [all_query | all_gate].
        launch_qk_norm_batch(B->q_buf, B->k_buf, L.q_norm, L.k_norm,
                             NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, T, RMS_NORM_EPS,
                             Q_PROJ_DIM, KV_DIM, stream, config_.is_hybrid());
        launch_rope_batch(B->q_buf, B->k_buf, state_.rope_cos, state_.rope_sin,
                          B->d_positions, B->max_seq_len, T,
                          NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_PROJ_DIM, KV_DIM, config_.rope_dim, stream);

        // 4. Write K,V to cache for sequence 0
        if (kv_bits_ == 0) {
            // k_buf is (KV_DIM, T) col-major -> cache is (max_seq_len, KV_DIM) row-major
            // k_buf[t] = col t = k_buf + t * KV_DIM, cache[t] = cache + t * KV_DIM -> same layout
            cudaMemcpyAsync(B->kv_keys[layer], B->k_buf,
                            T * KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(B->kv_values[layer], B->v_buf,
                            T * KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        } else {
            // TurboQuant: rotate + quantize all T tokens at once
            launch_turbo_kv_quantize(B->kv_keys_q[layer], B->kv_keys_norms[layer],
                                     B->k_buf, state_.turbo_rotation, B->d_positions,
                                     B->max_seq_len, T,
                                     NUM_KV_HEADS, HEAD_DIM, KV_DIM,
                                     kv_bits_, stream);
            launch_turbo_kv_quantize(B->kv_values_q[layer], B->kv_values_norms[layer],
                                     B->v_buf, state_.turbo_rotation, B->d_positions,
                                     B->max_seq_len, T,
                                     NUM_KV_HEADS, HEAD_DIM, KV_DIM,
                                     kv_bits_, stream);
        }

        // 5. Causal self-attention (Q stride = Q_PROJ_DIM so query part is correctly accessed)
        if (kv_bits_ == 0) {
            launch_gqa_prefill_attention(B->attn_out, B->q_buf,
                                          B->kv_keys[layer], B->kv_values[layer],
                                          B->attn_scores, T, B->max_seq_len,
                                          NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_PROJ_DIM, KV_DIM, stream);
        } else {
            launch_turbo_gqa_prefill_attention(B->attn_out, B->q_buf,
                                                B->kv_keys_q[layer], B->kv_values_q[layer],
                                                B->kv_keys_norms[layer], B->kv_values_norms[layer],
                                                state_.turbo_rotation,
                                                B->attn_scores, T, B->max_seq_len,
                                                NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, Q_PROJ_DIM, KV_DIM, kv_bits_, stream);
        }

        // 5b. Gated attention: attn_out *= sigmoid(gate)
        if (config_.gated_attn) {
            launch_sigmoid_gate_batch(B->attn_out, B->q_buf + Q_DIM, Q_DIM, T, stream);
        }

        // 6. Output projection + residual
        project(B->hidden, L.o_proj_fp16, L.o_proj_nf4, B->attn_out, HIDDEN_SIZE, Q_DIM);
        launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * T, stream);

        // 7. Post-attention norm
        launch_copy_batch(B->residual, B->hidden, HIDDEN_SIZE * T, stream);
        launch_rms_norm_batch(B->norm_buf, B->residual, L.post_attn_layernorm,
                              HIDDEN_SIZE, T, RMS_NORM_EPS, stream, config_.is_hybrid());

        // 8. FFN
        project(B->gate_buf, L.gate_proj_fp16, L.gate_proj_nf4, B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        project(B->up_buf, L.up_proj_fp16, L.up_proj_nf4, B->norm_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE);
        launch_silu_mul_batch(B->gate_buf, B->up_buf, INTERMEDIATE_SIZE * T, stream);
        project(B->hidden, L.down_proj_fp16, L.down_proj_nf4, B->gate_buf, HIDDEN_SIZE, INTERMEDIATE_SIZE);

        // 9. Residual add
        launch_residual_add_batch(B->hidden, B->residual, HIDDEN_SIZE * T, stream);

    }

    // Broadcast KV cache (attention layers) and SSM state (SSM layers) from seq 0 to seqs 1..G-1
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        for (int g = 1; g < G; g++) {
            if (config_.layer_type[layer] == LAYER_SSM) {
                // Broadcast SSM recurrent state
                size_t state_per_seq = (size_t)config_.ssm_num_v_heads * config_.ssm_k_head_dim
                                     * config_.ssm_v_head_dim * sizeof(half);
                size_t conv_per_seq = (size_t)config_.ssm_conv_dim() * (config_.ssm_conv_kernel - 1) * sizeof(float);
                cudaMemcpyAsync((char*)B->ssm_state[layer] + g * state_per_seq,
                                B->ssm_state[layer],
                                state_per_seq, cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync((char*)B->ssm_conv_state[layer] + g * conv_per_seq,
                                B->ssm_conv_state[layer],
                                conv_per_seq, cudaMemcpyDeviceToDevice, stream);
                continue;
            }
            if (kv_bits_ == 0) {
                cudaMemcpyAsync(B->kv_keys[layer] + (size_t)g * B->max_seq_len * KV_DIM,
                                B->kv_keys[layer],
                                T * KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(B->kv_values[layer] + (size_t)g * B->max_seq_len * KV_DIM,
                                B->kv_values[layer],
                                T * KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            } else {
                // Broadcast quantized KV cache
                size_t q_stride = (size_t)B->max_seq_len * (KV_DIM_PACKED);
                size_t n_stride = (size_t)B->max_seq_len * NUM_KV_HEADS;
                cudaMemcpyAsync(B->kv_keys_q[layer] + g * q_stride,
                                B->kv_keys_q[layer],
                                T * (KV_DIM_PACKED), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(B->kv_values_q[layer] + g * q_stride,
                                B->kv_values_q[layer],
                                T * (KV_DIM_PACKED), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(B->kv_keys_norms[layer] + g * n_stride,
                                B->kv_keys_norms[layer],
                                T * NUM_KV_HEADS * sizeof(half), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(B->kv_values_norms[layer] + g * n_stride,
                                B->kv_values_norms[layer],
                                T * NUM_KV_HEADS * sizeof(half), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    // For hybrid models: compute logits for the last token directly from prefill output.
    // This avoids re-processing through decode_batch which would corrupt SSM state
    // (the delta rule updates are not idempotent like KV cache writes).
    if (config_.is_hybrid()) {
        // Extract last token's hidden state: hidden[:, T-1] in col-major
        // Copy to position 0 of norm_buf for the final norm + LM head
        // hidden is (HIDDEN_SIZE, T) col-major: last token at offset (T-1) * HIDDEN_SIZE
        half* last_hidden = B->hidden + (T - 1) * HIDDEN_SIZE;

        // Final norm on the last token only
        launch_rms_norm_batch(B->norm_buf, last_hidden, weights_.final_layernorm,
                              HIDDEN_SIZE, 1, RMS_NORM_EPS, stream, config_.is_hybrid());

        // LM head: (VOCAB_SIZE, HIDDEN_SIZE) @ (HIDDEN_SIZE, 1) -> (VOCAB_SIZE, 1)
        ensure_cublas(stream);
        float alpha = 1.0f, beta_val = 0.0f;
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     VOCAB_SIZE, 1, HIDDEN_SIZE, &alpha,
                     weights_.embed_tokens, CUDA_R_16F, HIDDEN_SIZE,
                     B->norm_buf, CUDA_R_16F, HIDDEN_SIZE,
                     &beta_val, B->logits, CUDA_R_32F, VOCAB_SIZE,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        // Logits are now in B->logits for G=1 (broadcast to all G later if needed)
        // For G > 1, broadcast logits (same prompt → same logits for first token)
        for (int g = 1; g < G; g++) {
            cudaMemcpyAsync(B->logits + g * VOCAB_SIZE, B->logits,
                            VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
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
        if (config_.layer_type[i] == LAYER_SSM) {
            // Zero SSM state and conv state
            if (B->ssm_state[i]) {
                size_t state_sz = (size_t)G * config_.ssm_num_v_heads * config_.ssm_k_head_dim
                                * config_.ssm_v_head_dim * sizeof(half);
                cudaMemset(B->ssm_state[i], 0, state_sz);
            }
            if (B->ssm_conv_state[i]) {
                size_t conv_sz = (size_t)G * config_.ssm_conv_dim() * (config_.ssm_conv_kernel - 1) * sizeof(float);
                cudaMemset(B->ssm_conv_state[i], 0, conv_sz);
            }
        } else if (kv_bits_ == 0) {
            cudaMemset(B->kv_keys[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
            cudaMemset(B->kv_values[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
        } else {
            cudaMemset(B->kv_keys_q[i], 0, (size_t)G * total_max_len * (KV_DIM_PACKED));
            cudaMemset(B->kv_values_q[i], 0, (size_t)G * total_max_len * (KV_DIM_PACKED));
            cudaMemset(B->kv_keys_norms[i], 0, (size_t)G * total_max_len * NUM_KV_HEADS * sizeof(half));
            cudaMemset(B->kv_values_norms[i], 0, (size_t)G * total_max_len * NUM_KV_HEADS * sizeof(half));
        }
    }

    std::vector<std::vector<int>> outputs(G);

    // Phase 1: Chunked prefill -- all T prompt tokens in one forward pass.
    // All G sequences share the same prompt (GRPO), so prefill 1 sequence
    // and broadcast the KV cache.
    cudaStream_t gen_stream = engine_stream_;
    {
        // Upload all prompt tokens and positions [0..T-1] at once
        std::vector<int> h_tokens(max_prompt_len);
        std::vector<int> h_positions(max_prompt_len);
        for (int t = 0; t < max_prompt_len; t++) {
            h_tokens[t] = (t < (int)prompts[0].size()) ? prompts[0][t] : 0;
            h_positions[t] = t;
        }
        cudaMemcpyAsync(B->d_tokens, h_tokens.data(), max_prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, gen_stream);
        cudaMemcpyAsync(B->d_positions, h_positions.data(), max_prompt_len * sizeof(int),
                        cudaMemcpyHostToDevice, gen_stream);

        // Run chunked prefill (one forward pass, fills KV cache for all G seqs)
        prefill_chunked(max_prompt_len, G, gen_stream);
        cudaStreamSynchronize(gen_stream);

        if (config_.is_hybrid()) {
            // For hybrid models: prefill already computed logits for the last token.
            // Skip decode_batch to avoid corrupting SSM state (delta rule is not idempotent).
            for (int g = 0; g < G; g++) B->h_positions[g] = max_prompt_len;
        } else {
            // For attention-only models: re-process last token to get logits
            for (int g = 0; g < G; g++) {
                B->h_tokens[g] = prompts[0].back();
                B->h_positions[g] = max_prompt_len - 1;
            }
            cudaMemcpy(B->d_tokens, B->h_tokens, G * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);
            decode_batch(G, gen_stream);
            cudaStreamSynchronize(gen_stream);
            for (int g = 0; g < G; g++) B->h_positions[g] = max_prompt_len;
        }
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

    bool use_graph = false;

    // Allocate token history + randoms indirection for amortized stop checking
    const int CHECK_INTERVAL = 8;
    int hist_size = max_new_tokens * G;
    if (!B->d_token_history || B->token_history_size < hist_size) {
        if (B->d_token_history) cudaFree(B->d_token_history);
        CUDA_CHECK(cudaMalloc(&B->d_token_history, hist_size * sizeof(int)));
        delete[] B->h_token_history;
        B->h_token_history = new int[hist_size]();
        B->token_history_size = hist_size;
    }
    if (!B->d_randoms_ptr) {
        CUDA_CHECK(cudaMalloc(&B->d_randoms_ptr, sizeof(float*)));
    }

    // Ensure d_positions is up to date before decode loop
    cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);

    // CUDA graph captures: decode_batch + sampling (with pointer indirection for randoms).
    // The graph reads *d_randoms_ptr which is updated via 8-byte cudaMemcpyAsync per step.
    // CUDA graph not compatible with SSM layers (they use cudaMemset during decode)
    if (!config_.is_hybrid() && G > 1 && max_new_tokens >= 50 && !(graph_G_ == G && decode_graph_exec_)) {
        if (decode_graph_exec_) { cudaGraphExecDestroy(decode_graph_exec_); decode_graph_exec_ = nullptr; }
        if (decode_graph_) { cudaGraphDestroy(decode_graph_); decode_graph_ = nullptr; }

        // Warmup cuBLAS + sampling
        ensure_cublas(gen_stream);
        float* init_ptr = B->d_all_randoms;
        cudaMemcpy(B->d_randoms_ptr, &init_ptr, sizeof(float*), cudaMemcpyHostToDevice);
        decode_batch(G, gen_stream);
        if (temperature >= 0.01f)
            launch_sample_batch(B->logits, B->d_tokens, B->d_randoms_ptr, VOCAB_SIZE, G, temperature, top_p, gen_stream);
        else
            launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, gen_stream);
        cudaStreamSynchronize(gen_stream);

        // Capture decode + sampling + position increment
        // Set cublas_capture_mode to prevent cublasSetStream during capture
        cublas_capture_mode = true;
        cudaError_t err = cudaStreamBeginCapture(gen_stream, cudaStreamCaptureModeRelaxed);
        if (err == cudaSuccess) {
            decode_batch(G, gen_stream);
            if (temperature >= 0.01f)
                launch_sample_batch(B->logits, B->d_tokens, B->d_randoms_ptr, VOCAB_SIZE, G, temperature, top_p, gen_stream);
            else
                launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, gen_stream);
            launch_increment_positions(B->d_positions, G, gen_stream);

            cudaGraph_t graph = nullptr;
            err = cudaStreamEndCapture(gen_stream, &graph);
            if (err == cudaSuccess && graph) {
                err = cudaGraphInstantiate(&decode_graph_exec_, graph, 0);
                if (err == cudaSuccess) {
                    decode_graph_ = graph;
                    graph_G_ = G;
                    use_graph = true;
                    std::cout << "  CUDA graph captured with sampling (G=" << G << ")" << std::endl;
                } else { cudaGraphDestroy(graph); }
            }
            if (!use_graph) {
                std::cerr << "  CUDA graph capture FAILED (err=" << (int)err << ")" << std::endl;
                cudaGetLastError();
            }
        }
        cublas_capture_mode = false;

        // Re-prefill (warmup corrupted KV/SSM state) using chunked prefill
        for (int i = 0; i < NUM_LAYERS; i++) {
            if (config_.layer_type[i] == LAYER_SSM) {
                if (B->ssm_state[i]) cudaMemset(B->ssm_state[i], 0,
                    (size_t)G * config_.ssm_num_v_heads * config_.ssm_k_head_dim * config_.ssm_v_head_dim * sizeof(float));
                if (B->ssm_conv_state[i]) cudaMemset(B->ssm_conv_state[i], 0,
                    (size_t)G * config_.ssm_conv_dim() * (config_.ssm_conv_kernel - 1) * sizeof(half));
            } else if (kv_bits_ == 0) {
                cudaMemset(B->kv_keys[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
                cudaMemset(B->kv_values[i], 0, (size_t)G * total_max_len * KV_DIM * sizeof(half));
            } else {
                cudaMemset(B->kv_keys_q[i], 0, (size_t)G * total_max_len * (KV_DIM_PACKED));
                cudaMemset(B->kv_values_q[i], 0, (size_t)G * total_max_len * (KV_DIM_PACKED));
                cudaMemset(B->kv_keys_norms[i], 0, (size_t)G * total_max_len * NUM_KV_HEADS * sizeof(half));
                cudaMemset(B->kv_values_norms[i], 0, (size_t)G * total_max_len * NUM_KV_HEADS * sizeof(half));
            }
        }
        {
            std::vector<int> h_tok(max_prompt_len), h_pos(max_prompt_len);
            for (int t = 0; t < max_prompt_len; t++) {
                h_tok[t] = (t < (int)prompts[0].size()) ? prompts[0][t] : 0;
                h_pos[t] = t;
            }
            cudaMemcpyAsync(B->d_tokens, h_tok.data(), max_prompt_len * sizeof(int), cudaMemcpyHostToDevice, gen_stream);
            cudaMemcpyAsync(B->d_positions, h_pos.data(), max_prompt_len * sizeof(int), cudaMemcpyHostToDevice, gen_stream);
            prefill_chunked(max_prompt_len, G, gen_stream);
            cudaStreamSynchronize(gen_stream);
            for (int g = 0; g < G; g++) B->h_positions[g] = max_prompt_len;
        }
        cudaStreamSynchronize(gen_stream);
        cudaMemcpy(B->d_positions, B->h_positions, G * sizeof(int), cudaMemcpyHostToDevice);
    } else if (graph_G_ == G && decode_graph_exec_) {
        use_graph = true;
    }

    // For hybrid models: sample the first token from prefill logits before the decode loop.
    // The prefill already computed logits — sample now and set d_tokens for step 0.
    int first_step = 0;
    if (config_.is_hybrid()) {
        if (temperature < 0.01f) {
            launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, gen_stream);
        } else {
            float* step_randoms = B->d_all_randoms;
            cudaMemcpyAsync(B->d_randoms_ptr, &step_randoms, sizeof(float*),
                            cudaMemcpyHostToDevice, gen_stream);
            launch_sample_batch(B->logits, B->d_tokens, B->d_randoms_ptr,
                                VOCAB_SIZE, G, temperature, top_p, gen_stream);
        }
        // Store first token in history and advance positions
        cudaMemcpyAsync(B->d_token_history, B->d_tokens,
                        G * sizeof(int), cudaMemcpyDeviceToDevice, gen_stream);
        launch_increment_positions(B->d_positions, G, gen_stream);
        first_step = 1;  // decode loop starts from step 1
    }

    // Phase 2: Decode with amortized stop checking
    int last_checked = -1;
    for (int step = first_step; step < max_new_tokens; step++) {
        if (use_graph) {
            // Update randoms indirection pointer (8 bytes, no sync needed)
            if (temperature >= 0.01f) {
                float* step_randoms = B->d_all_randoms + step * G;
                cudaMemcpyAsync(B->d_randoms_ptr, &step_randoms, sizeof(float*),
                                cudaMemcpyHostToDevice, gen_stream);
            }
            cudaGraphLaunch(decode_graph_exec_, gen_stream);
        } else {
            decode_batch(G, gen_stream);
            if (temperature < 0.01f) {
                launch_argmax_batch(B->logits, B->d_tokens, VOCAB_SIZE, G, gen_stream);
            } else {
                float* step_randoms = B->d_all_randoms + step * G;
                cudaMemcpyAsync(B->d_randoms_ptr, &step_randoms, sizeof(float*),
                                cudaMemcpyHostToDevice, gen_stream);
                launch_sample_batch(B->logits, B->d_tokens, B->d_randoms_ptr,
                                    VOCAB_SIZE, G, temperature, top_p, gen_stream);
            }
            launch_increment_positions(B->d_positions, G, gen_stream);
        }

        // Store token in history (D2D, no sync)
        cudaMemcpyAsync(B->d_token_history + step * G, B->d_tokens,
                        G * sizeof(int), cudaMemcpyDeviceToDevice, gen_stream);

        // Check stops every CHECK_INTERVAL steps
        bool should_check = ((step + 1) % CHECK_INTERVAL == 0) || (step + 1 == max_new_tokens);
        if (!should_check) continue;

        cudaStreamSynchronize(gen_stream);
        cudaMemcpy(B->h_positions, B->d_positions, G * sizeof(int), cudaMemcpyDeviceToHost);

        int check_start = last_checked + 1;
        int check_count = (step + 1 - check_start) * G;
        cudaMemcpy(B->h_token_history + check_start * G,
                   B->d_token_history + check_start * G,
                   check_count * sizeof(int), cudaMemcpyDeviceToHost);

        bool all_done = true;
        int stop_len = stop_token_ids.size();
        for (int s = check_start; s <= step; s++) {
            for (int g = 0; g < G; g++) {
                if (B->h_finished[g]) continue;
                int tok = B->h_token_history[s * G + g];
                outputs[g].push_back(tok);
                if (tok == eos_token_id) {
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
        last_checked = step;
        if (all_done) break;
    }

    // Collect any unchecked tokens (e.g., hybrid first_step=1 with max_new_tokens=1)
    if (first_step > 0 && last_checked < 0) {
        // Pre-loop tokens were stored but never checked/collected
        cudaStreamSynchronize(gen_stream);
        int unchecked = std::min(first_step, max_new_tokens);
        cudaMemcpy(B->h_token_history, B->d_token_history,
                   unchecked * G * sizeof(int), cudaMemcpyDeviceToHost);
        for (int s = 0; s < unchecked; s++) {
            for (int g = 0; g < G; g++) {
                if (!B->h_finished[g]) {
                    int tok = B->h_token_history[s * G + g];
                    outputs[g].push_back(tok);
                    if (tok == eos_token_id) B->h_finished[g] = true;
                }
            }
        }
    }

    // Restore cuBLAS to stream 0 for PyTorch compatibility
    ensure_cublas(0);

    return outputs;
}
