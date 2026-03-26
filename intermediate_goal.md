# Intermediate Goal: Validate GRPO Training with C++ Engine

## What we're proving

End-to-end GRPO training on Jetson Orin Nano 8GB using our custom C++ inference engine for generation. If the model learns (reward improves over training steps), the approach is validated and we proceed to make it a general library.

## Scope

### What this is
- Fast GRPO adapter fine-tuning for **Qwen models** on NVIDIA GPUs
- Independent C++/CUDA inference engine (no TRL/HuggingFace at runtime)
- LoRA adapters exported in **HuggingFace PEFT-compatible format** (adapter_config.json + adapter_model.safetensors)
- Users train with our engine, deploy with HuggingFace/vLLM/llama.cpp

### What this is NOT (yet)
- Not a general multi-architecture engine (LLaMA, Gemma, Mistral come later)
- Not multi-GPU / distributed training
- Not full fine-tuning (LoRA/QLoRA only for now)

## Supported models (Phase 1)
- Qwen3-0.6B (validated)
- Qwen3-1.7B, Qwen3-4B, Qwen3-8B (same architecture, different constants)

## Supported hardware
- Any NVIDIA GPU with sm_61+ (Pascal and later)
- Validated: Jetson Orin Nano 8GB
- Target: RTX 3090, RTX 4090, A100, H100

## Architecture

```
Training loop (Python):
  for each step:
    1. C++ engine: generate G completions per prompt (batched GEMM, dp4a)
    2. Reward functions: score completions (Python)
    3. PyTorch: compute log-probs via forward pass
    4. GRPO loss: clipped surrogate objective, advantage normalization
    5. PyTorch: backward + optimizer step
    6. LoRA sync: push updated weights to C++ engine
```

### Key components
- `engine/` -- C++/CUDA inference engine (Q4L dp4a single + batched GEMM)
- `convert_weights.py` -- Qwen model to Q4L format
- `engine_rollout.py` -- Wraps C++ engine for GRPO rollout
- `lora_sync.py` -- Bidirectional LoRA weight sync (PyTorch <-> C++ engine)
- `train.py` -- GRPO training script (own implementation, no TRL dependency)

### Output format
LoRA adapters saved as:
```
checkpoint/
  adapter_config.json    # HuggingFace PEFT format
  adapter_model.safetensors  # standard safetensors
```

Loadable with:
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "checkpoint/")
```

## Benchmarks (target)

### Speed comparison vs TRL+vLLM (Qwen3-0.6B, G=4, 512 max tokens)

| Setup | Device | Expected tok/s | Step time |
|---|---|---|---|
| TRL + vLLM | Jetson Orin 8GB | not feasible (OOM) | -- |
| TRL + HF generate | Jetson Orin 8GB | ~7 tok/s | ~135s/step |
| **Ours (C++ engine)** | **Jetson Orin 8GB** | **~83 tok/s** | **~30s/step** |
| TRL + vLLM | RTX 4090 24GB | ~200 tok/s | ~15s/step |
| **Ours (C++ engine)** | **RTX 4090 24GB** | **TBD** | **TBD** |

### Quality comparison
- Train both setups on GSM8K for 300 steps
- Compare: final reward, eval accuracy, training curves
- The model should learn equivalently (same GRPO algorithm, just faster generation)

## Current status

### Working
- [x] C++ engine: 50.1 tok/s single (dp4a), 83 tok/s batch G=4
- [x] Q4L weight format with dp4a-friendly packing
- [x] Batched generation with temperature + top-p sampling
- [x] LoRA weight sync (PyTorch -> C++ engine)
- [x] TRL GRPOTrainer integration via rollout_func
- [x] Weight converter (bitsandbytes NF4 -> Q4L)
- [x] Correct output (batch matches single after QK norm fix)

### Next: validation experiment
- [ ] Run 300-step GRPO training on Jetson with C++ engine
- [ ] Verify reward improves over training
- [ ] Compare against TRL baseline (same hyperparameters)
- [ ] Export LoRA adapter, verify it loads in HuggingFace

### Then: generalize
- [ ] Runtime model config (read from JSON, not compile-time constants)
- [ ] Support Qwen3-1.7B, 4B, 8B
- [ ] Own GRPO implementation (remove TRL dependency)
- [ ] Proper top-p sampling (sort by probability)
- [ ] HuggingFace PEFT export
- [ ] pip-installable package with pre-built CUDA wheels
- [ ] README, docs, benchmarks
