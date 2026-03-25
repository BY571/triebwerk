/**
 * C++ inference engine — the core decode loop.
 *
 * One function call from Python generates an entire completion.
 * No Python per token, no kernel launch overhead from PyTorch dispatch.
 */

#include "model.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>

// Helper to avoid void** casts everywhere
template<typename T>
inline cudaError_t cudaMallocTyped(T** ptr, size_t size) {
    return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}

// Forward declarations for kernel launchers (defined in kernels.cu)
extern "C" {
    void launch_fp16_gemv(const half* weight, const half* input, half* output, int out_dim, int in_dim, cudaStream_t stream);
    void launch_rms_norm(const half* input, const half* weight, half* output, int dim, float eps, cudaStream_t stream);
    void launch_rope(half* q, half* k, const half* cos_table, const half* sin_table, int pos, int max_seq_len, cudaStream_t stream);
    void launch_gqa_attention(const half* q, const half* k_cache, const half* v_cache, half* output, float* attn_scratch, int pos, int max_seq_len, cudaStream_t stream);
    void launch_silu_gate_mul(const half* gate, const half* up, half* output, int dim, cudaStream_t stream);
    void launch_embedding(const half* embed_table, int token_id, half* output, int hidden_dim, cudaStream_t stream);
    void launch_residual_add(half* output, const half* residual, int dim, cudaStream_t stream);
    void launch_lm_head(const half* weight, const half* input, float* logits, int hidden_dim, int vocab_size, cudaStream_t stream);
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

    // 2. QKV projections (fp16 GEMV — attention weights not quantized)
    launch_fp16_gemv(layer.q_proj_fp16, norm_out, state_.q_buf, Q_DIM, HIDDEN_SIZE, stream);
    launch_fp16_gemv(layer.k_proj_fp16, norm_out, state_.k_buf, KV_DIM, HIDDEN_SIZE, stream);
    launch_fp16_gemv(layer.v_proj_fp16, norm_out, state_.v_buf, KV_DIM, HIDDEN_SIZE, stream);

    // 2b. QKNorm (Qwen3: RMSNorm applied per-head to Q and K after projection)
    // Q: apply norm to each of NUM_HEADS heads (each HEAD_DIM)
    for (int h = 0; h < NUM_HEADS; h++) {
        launch_rms_norm(state_.q_buf + h * HEAD_DIM, layer.q_norm,
                        state_.q_buf + h * HEAD_DIM, HEAD_DIM, RMS_NORM_EPS, stream);
    }
    // K: apply norm to each of NUM_KV_HEADS heads
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        launch_rms_norm(state_.k_buf + h * HEAD_DIM, layer.k_norm,
                        state_.k_buf + h * HEAD_DIM, HEAD_DIM, RMS_NORM_EPS, stream);
    }

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

    // 6. Output projection (fp16)
    launch_fp16_gemv(layer.o_proj_fp16, state_.attn_out, state_.hidden, HIDDEN_SIZE, Q_DIM, stream);

    // 7. Residual add (hidden += residual)
    launch_residual_add(state_.hidden, state_.residual, HIDDEN_SIZE, stream);

    // 8. Post-attention LayerNorm
    cudaMemcpyAsync(state_.residual, state_.hidden, HIDDEN_SIZE * sizeof(half),
                     cudaMemcpyDeviceToDevice, stream);
    launch_rms_norm(state_.residual, layer.post_attn_layernorm, norm_out,
                    HIDDEN_SIZE, RMS_NORM_EPS, stream);

    // 9. FFN: gate_proj and up_proj (fp16, dequantized from NF4 at load time)
    launch_fp16_gemv(layer.gate_proj_fp16, norm_out, state_.gate_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, stream);
    launch_fp16_gemv(layer.up_proj_fp16, norm_out, state_.up_buf, INTERMEDIATE_SIZE, HIDDEN_SIZE, stream);

    // 10. SiLU gate * up
    launch_silu_gate_mul(state_.gate_buf, state_.up_buf, state_.gate_buf,
                          INTERMEDIATE_SIZE, stream);

    // 11. Down projection
    launch_fp16_gemv(layer.down_proj_fp16, state_.gate_buf, state_.hidden, HIDDEN_SIZE, INTERMEDIATE_SIZE, stream);

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

    // LM head (tied to embedding)
    launch_lm_head(weights_.embed_tokens, norm_out, state_.logits,
                   HIDDEN_SIZE, VOCAB_SIZE, stream);

    state_.current_pos++;
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
    // Copy logits to host for sampling
    std::vector<float> logits_host(VOCAB_SIZE);
    cudaMemcpy(logits_host.data(), state_.logits, VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

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
        // Sort indices by probability (descending)
        std::vector<std::pair<float, int>> sorted_probs;
        sorted_probs.reserve(VOCAB_SIZE);
        for (int i = 0; i < VOCAB_SIZE; i++) {
            if (logits_host[i] > 1e-8f) {
                sorted_probs.push_back({logits_host[i], i});
            }
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float cumsum = 0.0f;
        std::vector<std::pair<float, int>> nucleus;
        for (auto& [prob, idx] : sorted_probs) {
            nucleus.push_back({prob, idx});
            cumsum += prob;
            if (cumsum >= top_p) break;
        }

        // Renormalize
        float nucleus_sum = 0.0f;
        for (auto& [prob, _] : nucleus) nucleus_sum += prob;

        // Sample from nucleus
        static std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, nucleus_sum);
        float r = dist(rng);

        float running = 0.0f;
        for (auto& [prob, idx] : nucleus) {
            running += prob;
            if (running >= r) return idx;
        }
        return nucleus.back().second;
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

    // Sample first token from prefill logits
    int token = sample(temperature, top_p);
    std::vector<int> output = {token};

    // Decode loop (pure C++, no Python)
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
        token = sample(temperature, top_p);
        output.push_back(token);
    }

    return output;
}
