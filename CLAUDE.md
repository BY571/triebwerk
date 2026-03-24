# CLAUDE.md

## Project overview

Optimize LLM fine-tuning (SFT, GRPO) for NVIDIA Jetson Orin edge devices with 8-16GB unified memory. General-purpose, not tied to any specific application.

## Structure

```
Dockerfile              # Jetson Docker image with full training stack
jetson_compat.py        # Patches for Jetson-specific issues (bf16 AMP, dtype)
train_gsm8k.py          # Baseline: GRPO on GSM8K with Qwen3-0.6B
```

## Running

Everything runs inside Docker on the Jetson:

```bash
docker build --network=host -t jetson-llm-train .
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  jetson-llm-train python3 train_gsm8k.py --max-steps 10
```

## Key constraints

- **No vLLM**: ARM architecture, not supported. HF generate only.
- **No bf16**: Jetson Orin AMP doesn't support bf16. Use fp16 everywhere.
- **8GB unified memory**: CPU and GPU share RAM. Every byte counts.
- **Torch must be from Jetson AI Lab index**: `pypi.jetson-ai-lab.io/jp6/cu126`. Pip packages that depend on torch will overwrite it with x86 version. Always use `--no-deps`.

## Jetson-specific patches (jetson_compat.py)

- `patch_amp_for_jetson()`: Monkey-patches AMP grad scaler to cast bf16 grads to fp32
- `cast_model_to_fp16()`: Casts all bf16 params/buffers to fp16
- `load_model_for_jetson()`: Loads 4-bit model with fp16 compute dtype
