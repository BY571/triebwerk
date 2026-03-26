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

// Runtime config aliases (same as engine.cpp — macros reference config_ member)
#define HIDDEN_SIZE config_.hidden_size
#define INTERMEDIATE_SIZE config_.intermediate_size
#define NUM_LAYERS config_.num_layers
#define NUM_HEADS config_.num_heads
#define NUM_KV_HEADS config_.num_kv_heads
#define HEAD_DIM config_.head_dim
#define VOCAB_SIZE config_.vocab_size
#define Q_DIM config_.q_dim()
#define KV_DIM config_.kv_dim()

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
    // Load model config (try config.json next to weights, fall back to default)
    config_ = ModelConfig::from_json(prefix + ".config.json");
    allocate_buffers();

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

    // Helper: checked cudaMalloc
    auto checked_malloc = [](void** ptr, size_t size, const char* name) {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err != cudaSuccess) {
            std::cerr << "  CUDA MALLOC FAILED: " << name << " (" << size << " bytes): "
                      << cudaGetErrorString(err) << std::endl;
            *ptr = nullptr;
        }
    };

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
                    void* p;
                    checked_malloc(&p, it_d->second.nbytes, (prefix + ".data").c_str());
                    if (p) cudaMemcpy(p, base + it_d->second.offset, it_d->second.nbytes, cudaMemcpyHostToDevice);
                    w.data = (uint8_t*)p;
                }
                if (it_a != index.end()) {
                    void* p;
                    checked_malloc(&p, it_a->second.nbytes, (prefix + ".absmax").c_str());
                    if (p) cudaMemcpy(p, base + it_a->second.offset, it_a->second.nbytes, cudaMemcpyHostToDevice);
                    w.absmax = (float*)p;
                }
                if (it_q != index.end()) {
                    void* p;
                    checked_malloc(&p, it_q->second.nbytes, (prefix + ".qmap").c_str());
                    if (p) cudaMemcpy(p, base + it_q->second.offset, it_q->second.nbytes, cudaMemcpyHostToDevice);
                    w.quant_map = (float*)p;
                }
        if (!w.data || !w.absmax || !w.quant_map) {
            std::cerr << "  NF4 MISSING: " << prefix
                      << " data=" << (void*)w.data
                      << " absmax=" << (void*)w.absmax
                      << " qmap=" << (void*)w.quant_map << std::endl;
        }
        return w;
    };

    // Helper: check quantization format
    auto is_nf4 = [&](const std::string& name) -> bool {
        auto it = index.find(name);
        if (it == index.end() || it->second.dtype != "uint8") return false;
        // NF4 has quant_map, Q4L does not
        return index.find(name.substr(0, name.size()-7) + ".weight.quant_map") != index.end();
    };
    auto is_q4l = [&](const std::string& name) -> bool {
        auto it = index.find(name);
        if (it == index.end() || it->second.dtype != "uint8") return false;
        // Q4L has absmax but NO quant_map
        std::string base = name.substr(0, name.size()-7);
        return index.find(base + ".weight.absmax") != index.end()
            && index.find(base + ".weight.quant_map") == index.end();
    };

    for (int i = 0; i < NUM_LAYERS; i++) {
        auto& L = weights_.layers[i];
        std::string p = "layers." + std::to_string(i) + ".";

        // Attention: check EACH projection individually
        L.attn_is_nf4 = false;
        struct { const char* name; half** fp16; NF4Weight* nf4; int out_dim; int in_dim; } attn_projs[] = {
            {"self_attn.q_proj", &L.q_proj_fp16, &L.q_proj_nf4, Q_DIM, HIDDEN_SIZE},
            {"self_attn.k_proj", &L.k_proj_fp16, &L.k_proj_nf4, KV_DIM, HIDDEN_SIZE},
            {"self_attn.v_proj", &L.v_proj_fp16, &L.v_proj_nf4, KV_DIM, HIDDEN_SIZE},
            {"self_attn.o_proj", &L.o_proj_fp16, &L.o_proj_nf4, HIDDEN_SIZE, Q_DIM},
        };
        for (auto& ap : attn_projs) {
            std::string wname = p + ap.name + ".weight";
            if (is_q4l(wname)) {
                *ap.nf4 = load_nf4(p + ap.name, ap.out_dim, ap.in_dim);
                *ap.fp16 = nullptr;
                L.attn_is_q4l = true;
            } else if (is_nf4(wname)) {
                *ap.nf4 = load_nf4(p + ap.name, ap.out_dim, ap.in_dim);
                *ap.fp16 = nullptr;
                L.attn_is_nf4 = true;
            } else {
                *ap.fp16 = load(wname);
            }
        }

        // MLP: check EACH weight individually
        L.mlp_is_nf4 = false;
        L.mlp_is_q4l = false;

        // Helper: load one MLP projection (Q4L > NF4 > fp16 priority)
        auto load_mlp_proj = [&](const char* proj, half** fp16, NF4Weight* nf4,
                                  int out_d, int in_d, bool& set_nf4, bool& set_q4l) {
            std::string wname = p + std::string("mlp.") + proj + ".weight";
            std::string prefix = p + std::string("mlp.") + proj;
            if (is_q4l(wname)) {
                *nf4 = load_nf4(prefix, out_d, in_d);
                *fp16 = nullptr;
                set_q4l = true;
            } else if (is_nf4(wname)) {
                *nf4 = load_nf4(prefix, out_d, in_d);
                *fp16 = nullptr;
                set_nf4 = true;
            } else {
                *fp16 = load(wname);
            }
        };

        load_mlp_proj("gate_proj", &L.gate_proj_fp16, &L.gate_proj_nf4,
                       INTERMEDIATE_SIZE, HIDDEN_SIZE, L.mlp_is_nf4, L.mlp_is_q4l);
        load_mlp_proj("up_proj", &L.up_proj_fp16, &L.up_proj_nf4,
                       INTERMEDIATE_SIZE, HIDDEN_SIZE, L.mlp_is_nf4, L.mlp_is_q4l);
        load_mlp_proj("down_proj", &L.down_proj_fp16, &L.down_proj_nf4,
                       HIDDEN_SIZE, INTERMEDIATE_SIZE, L.mlp_is_nf4, L.mlp_is_q4l);
        L.input_layernorm = load(p + "input_layernorm.weight");
        L.post_attn_layernorm = load(p + "post_attention_layernorm.weight");
        L.q_norm = load(p + "self_attn.q_norm.weight");
        L.k_norm = load(p + "self_attn.k_norm.weight");
        L.lora_q = L.lora_k = L.lora_v = L.lora_o = nullptr;
        L.lora_gate = L.lora_up = L.lora_down = nullptr;
    }
    // Detect model-wide Q4L format
    weights_.is_q4l = weights_.layers[1].attn_is_q4l || weights_.layers[1].mlp_is_q4l;
    if (weights_.is_q4l) std::cout << "  Weights: Q4 Linear (no lookup table)" << std::endl;
    std::cout << "  All " << NUM_LAYERS << " layers loaded" << std::endl;

    // Load NF4 LM head if available (quantized copy of embed_tokens for fast GEMV)
    if (index.find("lm_head_nf4.weight") != index.end()) {
        weights_.lm_head_nf4 = load_nf4("lm_head_nf4", VOCAB_SIZE, HIDDEN_SIZE);
        weights_.has_nf4_lm_head = (weights_.lm_head_nf4.data != nullptr);
        if (weights_.has_nf4_lm_head) {
            std::cout << "  NF4 LM head loaded (saves "
                      << (VOCAB_SIZE * HIDDEN_SIZE * 2 - VOCAB_SIZE * HIDDEN_SIZE / 2) / 1e6
                      << "MB)" << std::endl;
        }
    }
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
