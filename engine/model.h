#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <string>

// Qwen3-0.6B architecture constants
namespace qwen3 {
    constexpr int HIDDEN_SIZE = 1024;
    constexpr int INTERMEDIATE_SIZE = 3072;
    constexpr int NUM_LAYERS = 28;
    constexpr int NUM_HEADS = 16;       // Q heads
    constexpr int NUM_KV_HEADS = 8;     // KV heads (GQA ratio = 2)
    constexpr int HEAD_DIM = 128;
    constexpr int VOCAB_SIZE = 151936;
    constexpr int Q_DIM = NUM_HEADS * HEAD_DIM;     // 2048
    constexpr int KV_DIM = NUM_KV_HEADS * HEAD_DIM; // 1024
    constexpr float RMS_NORM_EPS = 1e-6f;
    constexpr float ROPE_THETA = 1000000.0f;
    constexpr int GQA_GROUPS = NUM_HEADS / NUM_KV_HEADS; // 2
}

// NF4 quantized weight: stored as uint8 (2 values per byte) + quantization metadata
struct NF4Weight {
    uint8_t* data;          // packed NF4 values (2 per byte), on GPU
    float* absmax;          // per-block scale factors (float32, pre-dequantized), on GPU
    float* quant_map;       // NF4 dequant lookup table (16 entries), on GPU
    int out_dim;            // output dimension (rows)
    int in_dim;             // input dimension (cols)
    int block_size;         // typically 64
    int n_blocks;           // out_dim * in_dim / block_size

    int total_params() const { return out_dim * in_dim; }
    size_t data_bytes() const { return (size_t)total_params() / 2; }
};

// FP16 weight (for embedding, norms, LoRA)
struct FP16Weight {
    half* data;
    int rows;
    int cols;
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

    // MLP: either fp16 (dequantized) or NF4 (native quantized)
    // Only one of these is non-null per layer
    half* gate_proj_fp16;   // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* up_proj_fp16;     // (INTERMEDIATE, HIDDEN) if fp16 mode
    half* down_proj_fp16;   // (HIDDEN, INTERMEDIATE) if fp16 mode

    NF4Weight gate_proj_nf4;  // if NF4 mode
    NF4Weight up_proj_nf4;    // if NF4 mode
    NF4Weight down_proj_nf4;  // if NF4 mode
    bool mlp_is_nf4 = false;

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
    // Embedding (fp16, shared with lm_head since tie_word_embeddings=True)
    half* embed_tokens;     // (VOCAB_SIZE, HIDDEN)

    // Transformer layers
    TransformerLayerWeights layers[qwen3::NUM_LAYERS];

    // Final norm
    half* final_layernorm;  // (HIDDEN,)
};

// Inference state (pre-allocated buffers)
struct InferenceState {
    // KV caches for all layers
    KVCache kv_cache[qwen3::NUM_LAYERS];
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

    // LoRA scratch buffer (for A @ x intermediate, max rank = 64)
    half* lora_scratch; // (max_lora_rank,)

    // CUDA graph: device-side control values (updated before graph replay)
    int* d_token_id;    // current token id (for embedding lookup)
    int* d_pos;         // current position (for RoPE, attention, KV cache)
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

    // Prefill: process multiple tokens at once
    // Returns logits for the last token
    void prefill(const int* token_ids, int n_tokens);

    // Decode: process one token, returns logits
    void decode(int token_id);

    // Get logits (after prefill or decode)
    float* get_logits() const { return state_.logits; }

    // Sample from logits (CPU-side, full top-p support)
    int sample(float temperature = 1.0f, float top_p = 0.9f);

    // Fast greedy sampling on GPU (only copies 4 bytes instead of 600KB)
    int sample_greedy_gpu();

    // Fast GPU sampling with temperature + top-p (only copies 4 bytes)
    int sample_gpu(float temperature = 1.0f, float top_p = 0.9f);

    // Full generation: prefill + decode loop
    std::vector<int> generate(const std::vector<int>& prompt,
                               int max_new_tokens = 512,
                               float temperature = 1.0f,
                               float top_p = 0.9f,
                               int eos_token_id = -1,
                               const std::vector<int>& stop_token_ids = {});

    // CUDA graph for fast decode replay
    bool use_cuda_graph_ = false;
    void enable_cuda_graph();  // capture after first decode

private:
    ModelWeights weights_;
    InferenceState state_;

    // CUDA graph state
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    bool graph_captured_ = false;
    cudaStream_t graph_stream_ = nullptr;

    // Forward pass for one token through one layer
    void forward_layer(int layer_idx);

    // Graph-captured forward (uses device-side position)
    void forward_layer_graph(int layer_idx, cudaStream_t stream);

    // Graph-accelerated decode (capture on first call, replay after)
    void decode_graph(int token_id);

    // Precompute RoPE cos/sin tables
    void precompute_rope();

    // Apply RoPE to Q and K
    void apply_rope(half* q, half* k, int pos);

    // GQA attention
    void attention(int layer_idx, int pos);

    // FFN: gate * silu(up) then down
    void ffn(int layer_idx);
};
