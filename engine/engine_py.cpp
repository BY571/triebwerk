/**
 * Python bindings for the Jetson LLM inference engine.
 *
 * Usage from Python:
 *   import jetson_engine
 *   engine = jetson_engine.Engine(max_seq_len=1024)
 *   engine.load_weights("/path/to/model")
 *   engine.load_lora("/path/to/lora", scale=1.0)
 *   tokens = engine.generate(prompt_ids, max_tokens=512, temperature=1.0)
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "model.h"

namespace py = pybind11;

PYBIND11_MODULE(jetson_engine, m) {
    m.doc() = "Jetson LLM Inference Engine — fast generation for small transformers";

    py::class_<InferenceEngine>(m, "Engine")
        .def(py::init<int>(), py::arg("max_seq_len") = 1024,
             "Create inference engine with pre-allocated KV cache")

        .def("load_weights", &InferenceEngine::load_weights,
             py::arg("model_dir"),
             "Load model weights from safetensors directory")

        .def("load_lora", &InferenceEngine::load_lora,
             py::arg("lora_dir"), py::arg("scale") = 1.0f,
             "Load LoRA adapter weights")

        .def("reset", &InferenceEngine::reset,
             "Reset KV cache for new generation")

        .def("share_embedding", [](InferenceEngine& self, size_t ptr) {
                 self.share_embedding(reinterpret_cast<void*>(ptr));
             },
             py::arg("data_ptr"),
             "Share embedding from PyTorch tensor (pass tensor.data_ptr())")

        .def("generate", &InferenceEngine::generate,
             py::arg("prompt"),
             py::arg("max_new_tokens") = 512,
             py::arg("temperature") = 1.0f,
             py::arg("top_p") = 0.9f,
             py::arg("eos_token_id") = -1,
             py::arg("stop_token_ids") = std::vector<int>{},
             "Generate tokens from prompt. Returns list of generated token IDs.")

        .def("sample", &InferenceEngine::sample,
             py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
             "Sample from current logits (CPU-side, full top-p)")

        .def("sample_greedy_gpu", &InferenceEngine::sample_greedy_gpu,
             "Greedy sample on GPU (4 bytes copy instead of 600KB)")

        .def("sample_gpu", &InferenceEngine::sample_gpu,
             py::arg("temperature") = 1.0f, py::arg("top_p") = 0.9f,
             "GPU sampling with temperature + top-p (4 bytes copy)")

        .def("update_lora", [](InferenceEngine& self,
                                int layer_idx, const std::string& proj_name,
                                py::buffer A_buf, py::buffer B_buf, float scale) {
                 auto A_info = A_buf.request();
                 auto B_info = B_buf.request();
                 self.update_lora_weight(
                     layer_idx, proj_name.c_str(),
                     static_cast<const half*>(A_info.ptr), A_info.shape[0], A_info.shape[1],
                     static_cast<const half*>(B_info.ptr), B_info.shape[0], B_info.shape[1],
                     scale);
             },
             py::arg("layer_idx"), py::arg("proj_name"),
             py::arg("A"), py::arg("B"), py::arg("scale") = 1.0f,
             "Update a single LoRA adapter (pass numpy fp16 arrays)")

        .def("enable_cuda_graph", &InferenceEngine::enable_cuda_graph,
             "Capture CUDA graph for decode step (call after first prefill)")

        .def("decode_token", &InferenceEngine::decode,
             py::arg("token_id"),
             "Process one token through the model")

        .def("prefill", [](InferenceEngine& self, const std::vector<int>& tokens) {
                self.prefill(tokens.data(), tokens.size());
             },
             py::arg("token_ids"),
             "Process prompt tokens (prefill phase)")

        .def("generate_batch", &InferenceEngine::generate_batch,
             py::arg("prompts"),
             py::arg("max_new_tokens") = 512,
             py::arg("temperature") = 1.0f,
             py::arg("top_p") = 0.9f,
             py::arg("eos_token_id") = -1,
             "Generate from G prompts in parallel (GEMM, tensor cores).")

        .def("profile_decode", [](InferenceEngine& self, int token_id) {
                 auto result = self.profile_decode(token_id);
                 py::dict d;
                 for (auto& [name, us] : result) d[py::cast(name)] = us;
                 return d;
             },
             py::arg("token_id"),
             "Profile one decode step, returns dict of {operation: time_us}")

        .def("model_config", [](InferenceEngine& self) {
                 const auto& c = self.config();
                 py::dict d;
                 d["hidden_size"] = c.hidden_size;
                 d["intermediate_size"] = c.intermediate_size;
                 d["num_layers"] = c.num_layers;
                 d["num_heads"] = c.num_heads;
                 d["num_kv_heads"] = c.num_kv_heads;
                 d["head_dim"] = c.head_dim;
                 d["vocab_size"] = c.vocab_size;
                 return d;
             },
             "Get model configuration (loaded from config.json alongside weights)");
}
