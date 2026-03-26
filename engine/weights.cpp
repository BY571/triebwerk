/**
 * Load weights from .bin + .idx files produced by convert_weights.py.
 *
 * ALL weights are fp16 (converter handles NF4 dequantization in Python).
 * Index format: "name offset nbytes dtype shape" per line.
 */

#include "model.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstring>

using namespace qwen3;

struct TensorInfo {
    size_t offset;
    size_t nbytes;
    std::string dtype;
    std::vector<int> shape;
};

static std::unordered_map<std::string, TensorInfo> load_index(const std::string& path) {
    std::unordered_map<std::string, TensorInfo> idx;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string name, dtype, shape_str;
        size_t offset, nbytes;
        ss >> name >> offset >> nbytes >> dtype >> shape_str;
        std::vector<int> shape;
        std::istringstream ss2(shape_str);
        std::string dim;
        while (std::getline(ss2, dim, ',')) shape.push_back(std::stoi(dim));
        idx[name] = {offset, nbytes, dtype, shape};
    }
    return idx;
}

void InferenceEngine::load_weights(const std::string& prefix) {
    auto index = load_index(prefix + ".idx");
    std::ifstream f(prefix + ".bin", std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> data(sz);
    f.read(data.data(), sz); f.close();
    const char* base = data.data();

    std::cout << "Loading " << index.size() << " tensors (" << sz/1e6 << "MB)" << std::endl;

    auto load = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) { std::cerr << "  MISS: " << name << std::endl; return nullptr; }
        void* p; cudaMalloc(&p, it->second.nbytes);
        cudaMemcpy(p, base + it->second.offset, it->second.nbytes, cudaMemcpyHostToDevice);
        return (half*)p;
    };

    weights_.embed_tokens = load("embed_tokens.weight");
    weights_.final_layernorm = load("norm.weight");

    // NF4 loader (used for both attention and MLP)
    auto load_nf4 = [&](const std::string& prefix, int out_dim, int in_dim) -> NF4Weight {
                NF4Weight w = {};
                w.out_dim = out_dim;
                w.in_dim = in_dim;
                w.block_size = 64;
                w.n_blocks = out_dim * in_dim / 64;

                auto it_d = index.find(prefix + ".weight");
                auto it_a = index.find(prefix + ".weight.absmax");
                auto it_q = index.find(prefix + ".weight.quant_map");

                if (it_d != index.end()) {
                    void* p; cudaMalloc(&p, it_d->second.nbytes);
                    cudaMemcpy(p, base + it_d->second.offset, it_d->second.nbytes, cudaMemcpyHostToDevice);
                    w.data = (uint8_t*)p;
                }
                if (it_a != index.end()) {
                    void* p; cudaMalloc(&p, it_a->second.nbytes);
                    cudaMemcpy(p, base + it_a->second.offset, it_a->second.nbytes, cudaMemcpyHostToDevice);
                    w.absmax = (float*)p;
                }
                if (it_q != index.end()) {
                    void* p; cudaMalloc(&p, it_q->second.nbytes);
                    cudaMemcpy(p, base + it_q->second.offset, it_q->second.nbytes, cudaMemcpyHostToDevice);
                    w.quant_map = (float*)p;
                }
        return w;
    };

    // Helper: check if a weight is NF4 (uint8) or fp16
    auto is_nf4 = [&](const std::string& name) -> bool {
        auto it = index.find(name);
        return it != index.end() && it->second.dtype == "uint8";
    };

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& L = weights_.layers[i];
        std::string p = "layers." + std::to_string(i) + ".";

        // Attention: check if NF4 or fp16
        std::string qw = p + "self_attn.q_proj.weight";
        if (is_nf4(qw)) {
            L.attn_is_nf4 = true;
            L.q_proj_fp16 = L.k_proj_fp16 = L.v_proj_fp16 = L.o_proj_fp16 = nullptr;
            L.q_proj_nf4 = load_nf4(p + "self_attn.q_proj", Q_DIM, HIDDEN_SIZE);
            L.k_proj_nf4 = load_nf4(p + "self_attn.k_proj", KV_DIM, HIDDEN_SIZE);
            L.v_proj_nf4 = load_nf4(p + "self_attn.v_proj", KV_DIM, HIDDEN_SIZE);
            L.o_proj_nf4 = load_nf4(p + "self_attn.o_proj", HIDDEN_SIZE, Q_DIM);
        } else {
            L.attn_is_nf4 = false;
            L.q_proj_fp16 = load(qw);
            L.k_proj_fp16 = load(p + "self_attn.k_proj.weight");
            L.v_proj_fp16 = load(p + "self_attn.v_proj.weight");
            L.o_proj_fp16 = load(p + "self_attn.o_proj.weight");
        }

        // MLP: check if NF4 or fp16
        std::string gw = p + "mlp.gate_proj.weight";
        if (is_nf4(gw)) {
            L.mlp_is_nf4 = true;
            L.gate_proj_fp16 = L.up_proj_fp16 = L.down_proj_fp16 = nullptr;
            L.gate_proj_nf4 = load_nf4(p + "mlp.gate_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE);
            L.up_proj_nf4 = load_nf4(p + "mlp.up_proj", INTERMEDIATE_SIZE, HIDDEN_SIZE);
            L.down_proj_nf4 = load_nf4(p + "mlp.down_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE);
        } else {
            L.mlp_is_nf4 = false;
            L.gate_proj_fp16 = load(gw);
            L.up_proj_fp16 = load(p + "mlp.up_proj.weight");
            L.down_proj_fp16 = load(p + "mlp.down_proj.weight");
        }
        L.input_layernorm = load(p + "input_layernorm.weight");
        L.post_attn_layernorm = load(p + "post_attention_layernorm.weight");
        L.q_norm = load(p + "self_attn.q_norm.weight");
        L.k_norm = load(p + "self_attn.k_norm.weight");
        L.lora_q = L.lora_k = L.lora_v = L.lora_o = nullptr;
        L.lora_gate = L.lora_up = L.lora_down = nullptr;
    }
    std::cout << "  All " << NUM_LAYERS << " layers loaded" << std::endl;
}

void InferenceEngine::load_lora(const std::string& prefix, float scale) {
    auto index = load_index(prefix + ".idx");
    std::ifstream f(prefix + ".bin", std::ios::binary | std::ios::ate);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<char> data(sz);
    f.read(data.data(), sz); f.close();
    const char* base = data.data();

    auto load = [&](const std::string& name) -> half* {
        auto it = index.find(name);
        if (it == index.end()) return nullptr;
        void* p; cudaMalloc(&p, it->second.nbytes);
        cudaMemcpy(p, base + it->second.offset, it->second.nbytes, cudaMemcpyHostToDevice);
        return (half*)p;
    };
    auto dim = [&](const std::string& name, int d) -> int {
        auto it = index.find(name);
        if (it == index.end() || d >= (int)it->second.shape.size()) return 0;
        return it->second.shape[d];
    };

    int n = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& L = weights_.layers[i];
        std::string p = "base_model.model.model.layers." + std::to_string(i) + ".";
        struct { const char* proj; LoRAAdapter** ptr; } tgts[] = {
            {"self_attn.q_proj", &L.lora_q}, {"self_attn.k_proj", &L.lora_k},
            {"self_attn.v_proj", &L.lora_v}, {"self_attn.o_proj", &L.lora_o},
            {"mlp.gate_proj", &L.lora_gate}, {"mlp.up_proj", &L.lora_up},
            {"mlp.down_proj", &L.lora_down},
        };
        for (auto& [proj, ptr] : tgts) {
            std::string an = p + proj + ".lora_A.weight";
            std::string bn = p + proj + ".lora_B.weight";
            half* a = load(an); half* b = load(bn);
            if (a && b) {
                auto* ad = new LoRAAdapter{a, b, dim(an,0), dim(an,1), dim(bn,0), scale};
                *ptr = ad; n++;
            }
        }
    }
    std::cout << "  Loaded " << n << " LoRA adapters" << std::endl;
}
