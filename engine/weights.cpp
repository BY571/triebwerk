/**
 * Load model weights from the flat binary format produced by convert_weights.py.
 *
 * Format:
 *   8 bytes: header_size (uint64 LE)
 *   header_size bytes: JSON index
 *   padding to 4096 boundary
 *   tensor data: all fp16, contiguous
 */

#include "model.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <sstream>

using namespace qwen3;

// Minimal JSON value parser (just strings, ints, arrays of ints)
struct TensorInfo {
    int64_t offset;
    std::vector<int64_t> shape;
    int64_t nbytes;
};

// Simple JSON parser for our specific format
static std::unordered_map<std::string, TensorInfo> parse_index(const char* json, size_t len) {
    std::unordered_map<std::string, TensorInfo> result;
    std::string s(json, len);

    // Find each tensor entry: "name": {"offset": N, "shape": [...], "nbytes": N}
    size_t pos = 0;
    while (pos < s.size()) {
        // Find tensor name
        size_t q1 = s.find('"', pos);
        if (q1 == std::string::npos) break;
        size_t q2 = s.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        std::string name = s.substr(q1 + 1, q2 - q1 - 1);

        // Find offset value
        size_t off_pos = s.find("\"offset\"", q2);
        if (off_pos == std::string::npos) break;
        size_t colon = s.find(':', off_pos + 8);
        int64_t offset = 0;
        sscanf(s.c_str() + colon + 1, " %ld", &offset);

        // Find shape array
        size_t shape_pos = s.find("\"shape\"", q2);
        size_t bracket_open = s.find('[', shape_pos);
        size_t bracket_close = s.find(']', bracket_open);
        std::string shape_str = s.substr(bracket_open + 1, bracket_close - bracket_open - 1);
        std::vector<int64_t> shape;
        std::istringstream ss(shape_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            shape.push_back(std::stol(token));
        }

        // Find nbytes
        size_t nb_pos = s.find("\"nbytes\"", q2);
        size_t nb_colon = s.find(':', nb_pos + 8);
        int64_t nbytes = 0;
        sscanf(s.c_str() + nb_colon + 1, " %ld", &nbytes);

        result[name] = {offset, shape, nbytes};
        pos = bracket_close + 1;
    }
    return result;
}

// Allocate GPU memory and copy a tensor from the binary data
static half* load_fp16_tensor(const char* data_start, const TensorInfo& info) {
    half* gpu_ptr;
    cudaMalloc(reinterpret_cast<void**>(&gpu_ptr), info.nbytes);
    cudaMemcpy(gpu_ptr, data_start + info.offset, info.nbytes, cudaMemcpyHostToDevice);
    return gpu_ptr;
}

void InferenceEngine::load_weights(const std::string& path) {
    std::cout << "Loading weights from: " << path << std::endl;

    // Read entire file
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open weights file: " + path);
    }
    size_t file_size = f.tellg();
    f.seekg(0);

    // Read header size
    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    // Read header
    std::vector<char> header(header_size);
    f.read(header.data(), header_size);

    // Parse index
    auto index = parse_index(header.data(), header_size);
    std::cout << "  Found " << index.size() << " tensors" << std::endl;

    // Calculate data start (after header + padding to 4096)
    size_t padded_header = ((header_size + 8 + 4095) / 4096) * 4096;

    // Read all tensor data into host memory
    std::vector<char> data(file_size - padded_header);
    f.seekg(padded_header);
    f.read(data.data(), data.size());
    f.close();

    const char* data_ptr = data.data();

    // Helper to load a tensor by name
    auto load = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) {
            std::cerr << "  WARNING: tensor not found: " << name << std::endl;
            return nullptr;
        }
        return load_fp16_tensor(data_ptr, it->second);
    };

    // Embedding
    weights_.embed_tokens = load("embed_tokens.weight");
    std::cout << "  embed_tokens loaded" << std::endl;

    // Final norm
    weights_.final_layernorm = load("norm.weight");

    // Layers
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& layer = weights_.layers[i];
        std::string prefix = "layers." + std::to_string(i) + ".";

        // Attention weights (fp16 in this model)
        layer.q_proj_fp16 = load(prefix + "self_attn.q_proj.weight");
        layer.k_proj_fp16 = load(prefix + "self_attn.k_proj.weight");
        layer.v_proj_fp16 = load(prefix + "self_attn.v_proj.weight");
        layer.o_proj_fp16 = load(prefix + "self_attn.o_proj.weight");

        // MLP weights (dequantized to fp16 by converter)
        layer.gate_proj_fp16 = load(prefix + "mlp.gate_proj.weight");
        layer.up_proj_fp16 = load(prefix + "mlp.up_proj.weight");
        layer.down_proj_fp16 = load(prefix + "mlp.down_proj.weight");

        // Norms
        layer.input_layernorm = load(prefix + "input_layernorm.weight");
        layer.post_attn_layernorm = load(prefix + "post_attention_layernorm.weight");

        // QKNorm (Qwen3 specific)
        layer.q_norm = load(prefix + "self_attn.q_norm.weight");
        layer.k_norm = load(prefix + "self_attn.k_norm.weight");

        // LoRA starts as nullptr
        layer.lora_q = nullptr;
        layer.lora_k = nullptr;
        layer.lora_v = nullptr;
        layer.lora_o = nullptr;
        layer.lora_gate = nullptr;
        layer.lora_up = nullptr;
        layer.lora_down = nullptr;

        if (i == 0 || i == NUM_LAYERS - 1) {
            std::cout << "  layer " << i << " loaded" << std::endl;
        }
    }

    std::cout << "  All " << NUM_LAYERS << " layers loaded" << std::endl;
}

void InferenceEngine::load_lora(const std::string& lora_path, float scale) {
    std::cout << "Loading LoRA from: " << lora_path << std::endl;

    // Read LoRA weights file (same binary format)
    std::ifstream f(lora_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open LoRA file: " + lora_path);
    }
    size_t file_size = f.tellg();
    f.seekg(0);

    uint64_t header_size;
    f.read(reinterpret_cast<char*>(&header_size), 8);

    std::vector<char> header(header_size);
    f.read(header.data(), header_size);

    auto index = parse_index(header.data(), header_size);

    size_t padded_header = ((header_size + 8 + 4095) / 4096) * 4096;
    std::vector<char> data(file_size - padded_header);
    f.seekg(padded_header);
    f.read(data.data(), data.size());
    f.close();

    const char* data_ptr = data.data();

    auto load = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) return nullptr;
        return load_fp16_tensor(data_ptr, it->second);
    };

    auto get_dim = [&](const std::string& name, int dim) -> int {
        auto it = index.find(name);
        if (it == index.end()) return 0;
        return (int)it->second.shape[dim];
    };

    // Load LoRA for each layer
    int loaded = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& layer = weights_.layers[i];
        std::string prefix = "base_model.model.model.layers." + std::to_string(i) + ".";

        struct { const char* proj; LoRAAdapter** adapter; } targets[] = {
            {"self_attn.q_proj", &layer.lora_q},
            {"self_attn.k_proj", &layer.lora_k},
            {"self_attn.v_proj", &layer.lora_v},
            {"self_attn.o_proj", &layer.lora_o},
            {"mlp.gate_proj",   &layer.lora_gate},
            {"mlp.up_proj",     &layer.lora_up},
            {"mlp.down_proj",   &layer.lora_down},
        };

        for (auto& [proj, adapter_ptr] : targets) {
            std::string a_name = prefix + proj + ".lora_A.weight";
            std::string b_name = prefix + proj + ".lora_B.weight";

            half* a_data = load(a_name);
            half* b_data = load(b_name);

            if (a_data && b_data) {
                auto* adapter = new LoRAAdapter();
                adapter->A = a_data;
                adapter->B = b_data;
                adapter->rank = get_dim(a_name, 0);
                adapter->in_features = get_dim(a_name, 1);
                adapter->out_features = get_dim(b_name, 0);
                adapter->scale = scale;
                *adapter_ptr = adapter;
                loaded++;
            }
        }
    }
    std::cout << "  Loaded " << loaded << " LoRA adapters (scale=" << scale << ")" << std::endl;
}
