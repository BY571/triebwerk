#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>

// Maximum number of transformer layers (covers all Qwen3 variants)
constexpr int MAX_LAYERS = 128;

// Runtime model configuration — loaded from config.json alongside weights.
// Replaces the old compile-time qwen3:: namespace.
struct ModelConfig {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;          // Q heads
    int num_kv_heads;       // KV heads (GQA)
    int head_dim;
    int vocab_size;
    float rms_norm_eps;
    float rope_theta;

    // Derived dimensions
    int q_dim() const { return num_heads * head_dim; }
    int kv_dim() const { return num_kv_heads * head_dim; }
    int gqa_groups() const { return num_heads / num_kv_heads; }

    // Built-in presets
    static ModelConfig qwen3_0_6b() {
        return {1024, 3072, 28, 16, 8, 128, 151936, 1e-6f, 1000000.0f};
    }
    static ModelConfig qwen3_1_7b() {
        return {2048, 6144, 28, 16, 8, 128, 151936, 1e-6f, 1000000.0f};
    }
    static ModelConfig qwen3_4b() {
        return {2560, 9216, 36, 32, 8, 128, 151936, 1e-6f, 1000000.0f};
    }
    static ModelConfig qwen3_8b() {
        return {4096, 12288, 36, 32, 8, 128, 151936, 1e-6f, 1000000.0f};
    }

    // Load from JSON file (written by convert_weights.py)
    static ModelConfig from_json(const std::string& path);
};

// 4-bit quantized weight: used for both NF4 (non-linear, lookup table) and Q4L
// (linear, (nibble - 8) * scale) formats. The format is determined by the
// is_q4l flags on TransformerLayerWeights and ModelWeights, not by this struct.
// For NF4: absmax stores per-block absmax values, dequant uses quant_map lookup.
// For Q4L: absmax stores per-block scale factors, dequant is linear arithmetic.
struct NF4Weight {
    uint8_t* data;          // packed NF4 values (2 per byte), on GPU
    float* absmax;          // per-block scale factors (float32, pre-dequantized), on GPU
    float* quant_map;       // NF4 dequant lookup table (16 entries), on GPU
    half* fp16_cache;       // pre-dequanted fp16 weights (nullptr = not cached)
    int out_dim;            // output dimension (rows)
    int in_dim;             // input dimension (cols)
    int block_size;         // typically 64
    int n_blocks;           // out_dim * in_dim / block_size

    int total_params() const { return out_dim * in_dim; }
    size_t data_bytes() const { return (size_t)total_params() / 2; }
};

// LoRA adapter weights for one linear layer
struct LoRAAdapter {
    half* A;    // (rank, in_features) - stored row-major
    half* B;    // (out_features, rank) - stored row-major
    int rank;
    int in_features;
    int out_features;
    float scale; // alpha / rank
};

// One transformer layer's weights
struct TransformerLayerWeights {
    // Attention: fp16 for layer 0, NF4 for layers 1-27 in unsloth model
    // Only one of fp16 or nf4 is non-null per projection
    half* q_proj_fp16;      // (Q_DIM, HIDDEN) if fp16
    half* k_proj_fp16;      // (KV_DIM, HIDDEN) if fp16
    half* v_proj_fp16;      // (KV_DIM, HIDDEN) if fp16
    half* o_proj_fp16;      // (HIDDEN, Q_DIM) if fp16

    NF4Weight q_proj_nf4;   // if NF4
    NF4Weight k_proj_nf4;   // if NF4
    NF4Weight v_proj_nf4;   // if NF4
    NF4Weight o_proj_nf4;   // if NF4
    bool attn_is_nf4 = false;
    bool attn_is_q4l = false;  // Q4 linear (no lookup table)

    // MLP: either fp16 (dequantized) or NF4 (native quantized)
    // Only one of these is non-null per layer
    half* gate_proj_fp16;   // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* up_proj_fp16;     // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* down_proj_fp16;   // (HIDDEN, INTERMEDIATE) if fp16 mode

    NF4Weight gate_proj_nf4;  // if NF4 mode
    NF4Weight up_proj_nf4;    // if NF4 mode
    NF4Weight down_proj_nf4;  // if NF4 mode
    bool mlp_is_nf4 = false;
    bool mlp_is_q4l = false;

    // Norms (fp16, small)
    half* input_layernorm;  // (HIDDEN,)
    half* post_attn_layernorm; // (HIDDEN,)

    // QKNorm (Qwen3 specific — RMSNorm applied to Q and K after projection)
    half* q_norm;           // (HEAD_DIM,) = (128,)
    half* k_norm;           // (HEAD_DIM,) = (128,)

    // Optional LoRA adapters (nullptr if not used)
    LoRAAdapter* lora_q;
    LoRAAdapter* lora_k;
    LoRAAdapter* lora_v;
    LoRAAdapter* lora_o;
    LoRAAdapter* lora_gate;
    LoRAAdapter* lora_up;
    LoRAAdapter* lora_down;
};

// KV cache for one layer
struct KVCache {
    half* key;   // (max_seq_len, KV_DIM)
    half* value; // (max_seq_len, KV_DIM)
};

// Full model weights
struct ModelWeights {
    // Embedding (fp16, used for both token lookup and LM head via cuBLAS)
    half* embed_tokens;     // (VOCAB_SIZE, HIDDEN)

    // Transformer layers (dynamic count, up to MAX_LAYERS)
    TransformerLayerWeights layers[MAX_LAYERS];

    // Final norm
    half* final_layernorm;  // (HIDDEN,)

    // Model-wide quantization format (set during loading)
    bool is_q4l = false;  // true if weights are Q4 Linear (not NF4)
};

// Inference state (pre-allocated buffers)
struct InferenceState {
    // KV caches for all layers (dynamic count, up to MAX_LAYERS)
    KVCache kv_cache[MAX_LAYERS];
    int max_seq_len;
    int current_pos;    // current position in KV cache

    // Scratch buffers (reused across layers)
    half* hidden;       // (HIDDEN,) current hidden state
    half* residual;     // (HIDDEN,) residual connection
    half* q_buf;        // (Q_DIM,)
    half* k_buf;        // (KV_DIM,)
    half* v_buf;        // (KV_DIM,)
    half* attn_out;     // (Q_DIM,)
    half* gate_buf;     // (INTERMEDIATE,)
    half* up_buf;       // (INTERMEDIATE,)
    half* ffn_out;      // (HIDDEN,)
    float* logits;      // (VOCAB_SIZE,) float32 for numerical stability in sampling

    // Attention scratch
    float* attn_scores; // (NUM_HEADS, max_seq_len) float32 for softmax stability

    // RoPE precomputed cos/sin
    half* rope_cos;     // (max_seq_len, HEAD_DIM/2)
    half* rope_sin;     // (max_seq_len, HEAD_DIM/2)

    // GPU-side sampling result
    int* sample_result; // single int on GPU

    // dp4a input quantization buffers (for Q4L dp4a GEMV)
    int8_t* q8_data;        // (INTERMEDIATE_SIZE,) quantized input
    float* q8_scales;       // (INTERMEDIATE_SIZE/64,) per-block scale
    float* q8_sums;         // (INTERMEDIATE_SIZE/64,) per-block sum of q8 values

    // LoRA scratch buffer (for A @ x intermediate, max rank = 64)
    half* lora_scratch; // (max_lora_rank,)
};

// Simple GPU arena: one cudaMalloc, bump-pointer suballocation.
// Prevents fragmentation with PyTorch's caching allocator.
struct GpuArena {
    char* base = nullptr;
    size_t capacity = 0;
    size_t offset = 0;

    // Allocate from arena (256-byte aligned for cuBLAS)
    void* alloc(size_t bytes) {
        offset = (offset + 255) & ~(size_t)255;
        void* ptr = base + offset;
        offset += bytes;
        return ptr;
    }

    void reset() { offset = 0; }
    size_t used() const { return offset; }
};

// Batched generation state (G sequences in parallel)
struct BatchState {
    int G;              // batch size (number of sequences)
    int max_seq_len;

    // Activation buffers: (dim, G) column-major for cuBLAS
    half* hidden;       // (HIDDEN_SIZE, G)
    half* residual;     // (HIDDEN_SIZE, G)
    half* norm_buf;     // (HIDDEN_SIZE, G) temp for norm output
    half* q_buf;        // (Q_DIM, G)
    half* k_buf;        // (KV_DIM, G)
    half* v_buf;        // (KV_DIM, G)
    half* attn_out;     // (Q_DIM, G)
    half* gate_buf;     // (INTERMEDIATE_SIZE, G)
    half* up_buf;       // (INTERMEDIATE_SIZE, G)
    float* logits;      // (VOCAB_SIZE, G) fp32

    // Attention scratch
    float* attn_scores; // (G * NUM_HEADS * max_seq_len)

    // Batched KV cache: (G * max_seq_len * KV_DIM) per layer
    half* kv_keys[MAX_LAYERS];
    half* kv_values[MAX_LAYERS];

    // Q4L dequant scratch (largest projection = INTERMEDIATE_SIZE * HIDDEN_SIZE)
    half* dequant_scratch;

    // LoRA scratch for batched A @ x intermediate (max_rank * G)
    half* lora_scratch;

    // Per-sequence state
    int* h_positions;   // host (G,)
    int* d_positions;   // device (G,)
    int* h_tokens;      // host (G,) sampled tokens
    int* d_tokens;      // device (G,)
    bool* h_finished;   // host (G,)
    float* h_randoms;   // host (G,) for stochastic sampling
    float* d_randoms;   // device (G,)
};

// Top-level engine
class InferenceEngine {
public:
    InferenceEngine(int max_seq_len = 1024);
    ~InferenceEngine();

    // Load weights from safetensors directory (HF format)
    void load_weights(const std::string& model_dir);

    // Load LoRA adapter from file
    void load_lora(const std::string& lora_dir, float scale = 1.0f);

    // Update a single LoRA adapter from raw fp16 data (for live sync from PyTorch)
    void update_lora_weight(int layer_idx, const char* proj_name,
                            const half* A_data, int A_rows, int A_cols,
                            const half* B_data, int B_rows, int B_cols,
                            float scale);

    // Reset KV cache for new generation
    void reset();

    // Share embedding from PyTorch (avoids 311MB duplicate on Jetson)
    // Pass the raw GPU pointer from model.embed_tokens.weight.data_ptr()
    void share_embedding(void* external_embed_ptr);
    bool embed_is_external_ = false;

    // Prefill: process multiple tokens at once
    // Returns logits for the last token
    void prefill(const int* token_ids, int n_tokens);

    // Decode: process one token, returns logits
    void decode(int token_id);

    // Get logits (after prefill or decode)
    float* get_logits() const { return state_.logits; }

    // Fast greedy sampling on GPU (only copies 4 bytes instead of 600KB)
    int sample_greedy_gpu();

    // GPU sampling with temperature + top-p (only copies 4 bytes)
    int sample_gpu(float temperature = 1.0f, float top_p = 0.9f);

    // Full generation: prefill + decode loop
    std::vector<int> generate(const std::vector<int>& prompt,
                               int max_new_tokens = 512,
                               float temperature = 1.0f,
                               float top_p = 0.9f,
                               int eos_token_id = -1,
                               const std::vector<int>& stop_token_ids = {});

    // Batched generation (G sequences in parallel, GEMM with tensor cores)
    std::vector<std::vector<int>> generate_batch(
        const std::vector<std::vector<int>>& prompts,
        int max_new_tokens = 512,
        float temperature = 1.0f,
        float top_p = 0.9f,
        int eos_token_id = -1,
        const std::vector<int>& stop_token_ids = {});

    // Pre-dequant Q4L weights to fp16 for fast batched GEMM
    void cache_weights();

    // Sleep/wake: free GPU buffers during training, re-allocate for generation
    void sleep();
    void wake();

    const ModelConfig& config() const { return config_; }

private:
    bool weights_cached_ = false;
    GpuArena batch_arena_;  // single allocation for all batch GPU buffers

    // Dedicated stream for engine (avoids conflicts with PyTorch's default stream)
    cudaStream_t engine_stream_ = nullptr;

    // CUDA graph for batched decode (captured on first decode, replayed thereafter)
    cudaGraph_t decode_graph_ = nullptr;
    cudaGraphExec_t decode_graph_exec_ = nullptr;
    int graph_G_ = 0;  // G used when graph was captured
    ModelConfig config_;
    ModelWeights weights_;
    InferenceState state_;

    // Forward pass for one token through one layer
    void forward_layer(int layer_idx);

    // Allocate GPU buffers (called from load_weights after config is known)
    void allocate_buffers();

    // Precompute RoPE cos/sin tables
    void precompute_rope();

    // Batch generation internals
    BatchState* batch_;  // allocated on first generate_batch call
    void alloc_batch(int G, int max_seq_len);
    void decode_batch(int G);
    void forward_layer_batch(int layer_idx, int G, cudaStream_t stream);

    // Batched GEMM: (M,K) @ (K,N) -> (M,N) where N=G
    void batch_gemm(half* out, const half* weight, const half* in,
                    int M, int N, int K, cudaStream_t stream);
    // Q4L batch: dequant to scratch then cuBLAS GEMM
    void batch_gemm_q4l(half* out, const NF4Weight& w, const half* in,
                        int N, cudaStream_t stream);

};
