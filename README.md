# Triebwerk

Fast LoRA fine-tuning engine for LLMs. Custom C++/CUDA inference engine with CUDA graphs, dp4a 4-bit quantization, and zero-overhead batched generation. PyTorch handles gradients, Triebwerk handles speed.

## Performance

| Setup | Device | Step time | tok/s |
|---|---|---|---|
| TRL + HF generate | RTX 4060 | 68.3s | ~15 |
| TRL + HF generate | Jetson Orin 8GB | 130.6s | ~7 |
| vLLM | RTX 4060 | 5.2s | ~375 |
| **Triebwerk** | **RTX 4060** | **5.8s** | **~300** |
| **Triebwerk** | **Jetson Orin 8GB** | **16.5s** | **~104** |

Output: standard HuggingFace PEFT LoRA adapters. Load with `PeftModel.from_pretrained()`, deploy anywhere.

## Quick start

```python
from triebwerk import GRPOTrainer

def my_reward(completions, answer, **kwargs):
    return [2.0 if ans in text else -1.0
            for text, ans in zip(completions, answer)]

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[my_reward],
)
trainer.train(dataset, max_steps=300)
```

## Training algorithms

```python
from triebwerk import GRPOTrainer, DGTrainer

# GRPO / DAPO (default) -- clipped surrogate with reference model
trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[my_reward],
    loss_type="dapo",  # or "grpo"
)

# Delightful Policy Gradient (https://arxiv.org/abs/2603.14608)
# No reference model, simpler, one hyperparameter
# Includes Kondo gate (https://arxiv.org/abs/2603.20526): auto-skips backward for low-value samples
trainer = DGTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[my_reward],
    eta=1.0,
)
```

## Examples

```bash
# Countdown numbers game (recommended -- clearest learning signal for small models)
PYTHONPATH=engine/build_local python3 examples/countdown.py --max-steps 300

# Letter counting (continuous reward, good for debugging)
PYTHONPATH=engine/build_local python3 examples/letter_counting.py --max-steps 300

# GSM8K math (granular rewards: format + integer + correctness)
PYTHONPATH=engine/build_local python3 examples/gsm8k.py --max-steps 300

# Dry run (no C++ engine needed, uses HF generate)
python3 examples/countdown.py --dry-run --max-steps 5
```

## What makes it fast

- **Chunked prefill**: all prompt tokens in one forward pass per layer (weights read once, not T times)
- **CUDA graphs**: decode + sampling captured as a single graph, ~1us launch overhead per token
- **Pointer-indirection sampling**: device-side `float**` lets the graph read different random values each step
- **Q4L 4-bit with dp4a**: integer dot product, 4 MACs per instruction on Jetson
- **TurboQuant KV cache**: 4x-8x KV cache compression via random rotation + Lloyd-Max quantization ([paper](https://arxiv.org/abs/2504.19874))
- **Batched generation**: all G completions in parallel via cuBLAS GEMM with tensor cores
- **Arena allocator**: single cudaMalloc for all batch buffers, zero fragmentation with PyTorch
- **Amortized stop checking**: sync every 8 tokens instead of every token, GPU-side position increment
- **GPU-GPU LoRA sync**: direct device-to-device memcpy, no CPU roundtrip

## Architecture

```
Training step:
  1. Triebwerk generates G completions (chunked prefill + CUDA graph decode, ~300 tok/s)
  2. Reward functions score completions (Python)
  3. PyTorch forward pass computes log-probs (batched, one call for all G)
  4. Loss: GRPO clipped surrogate or DG sigmoid gate
  5. PyTorch backward + optimizer step
  6. LoRA weights synced to engine via GPU-GPU memcpy (~0.02ms)
```

## Setup

```bash
# Build the engine (one-time)
cd engine && mkdir -p build_local && cd build_local
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  # 87 for Jetson, 89 for RTX 4060
make -j$(nproc)

# Convert weights (one-time)
python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights_q4l --mode q4l

# Train
PYTHONPATH=engine/build_local python3 train.py --max-steps 300

# Or use the Delightful PG algorithm (no reference model)
PYTHONPATH=engine/build_local python3 train.py --max-steps 300 --loss-type dg
```

## TurboQuant: compressed KV cache

Compresses the KV cache using random rotation + optimal scalar quantization ([Zandieh et al. 2025](https://arxiv.org/abs/2504.19874)). The rotation decorrelates coordinates so each can be quantized independently with Lloyd-Max centroids. Only 32 KB of overhead for the shared rotation matrix.

| Mode | KV bits | Memory reduction | Quality (0.6B) |
|---|---|---|---|
| `kv_bits=0` | 16 (fp16) | 1x (baseline) | Perfect |
| `kv_bits=4` | 4 | **4x** | Good -- coherent, slight accuracy loss |
| `kv_bits=2` | 2 | **8x** | Needs 8B+ models (too aggressive for 0.6B) |

### Max context length (Qwen3-0.6B, G=4 GRPO training)

| Device | fp16 | TurboQuant 4-bit | Gain |
|---|---|---|---|
| Jetson Orin 8GB | ~7K tokens | ~24K tokens | 3.4x |
| RTX 4060 8GB | ~7K tokens | ~24K tokens | 3.4x |
| RTX 4090 24GB | ~20K tokens | ~65K tokens | 3.3x |

```bash
# Enable 4-bit KV cache (recommended for Qwen3-0.6B)
PYTHONPATH=engine/build_local python3 train.py --max-steps 300 --kv-bits 4
```

## Supported models

All Qwen3 variants (runtime config, no recompilation):

| Model | Validated | Notes |
|---|---|---|
| Qwen3-0.6B | Yes | Fits on 8GB |
| Qwen3-1.7B | Config ready | Needs 16GB+ |
| Qwen3-4B | Config ready | Needs 16GB+ |
| Qwen3-8B | Config ready | Needs 24GB+ |

## Supported hardware

- **Validated**: RTX 4060 Laptop 8GB (sm_89), Jetson Orin Nano 8GB (sm_87)
- **Target**: Any NVIDIA GPU with sm_61+ (Pascal and later)

