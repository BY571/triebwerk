```
   ████████╗██████╗ ██╗███████╗██████╗ ██╗    ██╗███████╗██████╗ ██╗  ██╗
   ╚══██╔══╝██╔══██╗██║██╔════╝██╔══██╗██║    ██║██╔════╝██╔══██╗██║ ██╔╝
      ██║   ██████╔╝██║█████╗  ██████╔╝██║ █╗ ██║█████╗  ██████╔╝█████╔╝
      ██║   ██╔══██╗██║██╔══╝  ██╔══██╗██║███╗██║██╔══╝  ██╔══██╗██╔═██╗
      ██║   ██║  ██║██║███████╗██████╔╝╚███╔███╔╝███████╗██║  ██║██║  ██╗
      ╚═╝   ╚═╝  ╚═╝╚═╝╚══════╝╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝

   ==============================================================
                            TRIEBWERK
              High-Performance RL Training for LLMs
   --------------------------------------------------------------
    Version:  0.1.0
    Device:   NVIDIA GeForce RTX 4060 Laptop GPU (8.2 GB)
    Engine:   C++ dp4a
    Gear:     Fast
    Model:    Qwen/Qwen3-0.6B
    KV Cache: fp16 (281 MB for 612 tok)
    Context:  612/13108 (5%)
    Steps:    300 (G=4, 512 tok)
   ==============================================================

    Memory:  [███████████████░░░░░░░░░░░░░░░░░░░░░░░░░] 3.2/8.2 GB (39%)
             ██ Weights 0.6G  ██ FP16-Cache 0.8G  ██ PyTorch 1.4G
             ██ Arena+KV 0.5G  ░░ Free 5.0G

   [1/300] loss=-0.67  reward=-0.88  tok/s=300  gen=5.2s  step=5.8s
   [2/300] loss=-0.42  reward= 0.25  tok/s=284  gen=3.9s  step=4.6s
   [3/300] loss=-0.50  reward=-0.62  tok/s=253  gen=5.2s  step=5.7s
   ...
```

RL fine-tuning engine for LLMs. C++/CUDA inference engine with CUDA graphs and 4-bit quantization. Matches vLLM on desktop GPUs, runs on Jetson Orin where vLLM can't. Outputs standard HuggingFace PEFT LoRA adapters.

## Algorithms

| Algorithm | Paper | Description | Reference model |
|---|---|---|---|
| **GRPO** | [Shao et al. 2024](https://arxiv.org/abs/2402.03300) | Group Relative Policy Optimization with clipped surrogate | Yes |
| **DAPO** | [Yu et al. 2025](https://arxiv.org/abs/2503.14476) | Dynamic sampling GRPO, asymmetric clipping | Yes |
| **DG** | [Osband 2025](https://arxiv.org/abs/2603.14608) | Delightful Policy Gradient, sigmoid delight gate | No |
| **Kondo** | [Osband 2025](https://arxiv.org/abs/2603.20526) | Skip backward for low-value samples (built into DG) | No |

## Performance

| Setup | Device | Step time | tok/s |
|---|---|---|---|
| TRL + HF generate | RTX 4060 | 68.3s | ~15 |
| TRL + HF generate | Jetson Orin 8GB | 130.6s | ~7 |
| vLLM | RTX 4060 | 5.2s | ~375 |
| **Triebwerk** | **RTX 4060** | **5.8s** | **~300** |
| **Triebwerk** | **Jetson Orin 8GB** | **16.5s** | **~104** |

## Setup

<details>
<summary><b>Desktop GPU (RTX 30xx/40xx/50xx)</b></summary>

```bash
git clone https://github.com/BY571/triebwerk && cd triebwerk
uv sync

# Build C++ engine (one-time)
cd engine && mkdir -p build_local && cd build_local
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  # 89 for RTX 40xx, 86 for RTX 30xx
make -j$(nproc) && cd ../..

# Convert weights (one-time)
python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights_q4l --mode q4l

# Train
PYTHONPATH=engine/build_local uv run python3 train.py --max-steps 300

# Dry run (no engine build needed)
uv run python3 train.py --max-steps 5 --dry-run
```
</details>

<details>
<summary><b>Jetson Orin (Docker)</b></summary>

Jetson needs Docker because bitsandbytes requires a specific CUDA/ARM build.

```bash
# Build Docker image (one-time, ~15 min)
docker build --network=host -t triebwerk .

# Convert weights + build engine (one-time)
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace -v ~/.cache/huggingface:/root/.cache/huggingface \
  triebwerk bash -c '
    python3 engine/convert_weights.py --model Qwen/Qwen3-0.6B --output engine/weights_q4l --mode q4l &&
    cd engine && mkdir -p build_docker && cd build_docker &&
    cmake .. -DCMAKE_CUDA_ARCHITECTURES=87 && make -j4'

# Train
docker run --runtime nvidia --network=host \
  -v $(pwd):/workspace -v ~/.cache/huggingface:/root/.cache/huggingface \
  triebwerk bash -c \
  'ENGINE_BUILD=engine/build_docker PYTHONPATH=engine/build_docker python3 train.py --max-steps 300'
```

> **Tip:** disable desktop GUI (`sudo systemctl set-default multi-user.target`) to free ~800MB RAM.
</details>

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

```python
from triebwerk import DGTrainer

# Delightful Policy Gradient -- no reference model, includes Kondo gate
trainer = DGTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=[my_reward],
)
trainer.train(dataset, max_steps=300)
```

## Examples

```bash
PYTHONPATH=engine/build_local uv run python3 examples/countdown.py --max-steps 300       # Countdown numbers
PYTHONPATH=engine/build_local uv run python3 examples/letter_counting.py --max-steps 300  # Letter counting
PYTHONPATH=engine/build_local uv run python3 examples/gsm8k.py --max-steps 300            # GSM8K math
uv run python3 examples/countdown.py --dry-run --max-steps 5                              # Dry run (no engine)
```


## TurboQuant: compressed KV cache

Compresses the KV cache using random rotation + optimal scalar quantization ([Zandieh et al. 2025](https://arxiv.org/abs/2504.19874)). The rotation decorrelates coordinates so each can be quantized independently with Lloyd-Max centroids. Only 32 KB of overhead for the shared rotation matrix.

| Mode | KV bits | Memory reduction | Quality (0.6B) |
|---|---|---|---|
| `kv_bits=0` | 16 (fp16) | 1x (baseline) | Perfect |
| `kv_bits=4` | 4 | **4x** | Good -- coherent, slight accuracy loss |
| `kv_bits=2` | 2 | **8x** | Needs 8B+ models (too aggressive for 0.6B) |

### Jetson Orin 8 GB memory breakdown (measured, Qwen3-0.6B GRPO training)

```
Total unified memory:              7,990 MB
- CUDA runtime + OS:              -2,012 MB
- Engine weights (Q4L + fp16):      -937 MB
- PyTorch model (4-bit BnB):      -1,572 MB
- LoRA + AdamW optimizer:             -2 MB
                                  ---------
= Available for batch arena:       3,467 MB
```

### Max context length (measured, G=4)

| Prompt | Completion | Total | fp16 KV | TQ4 KV | Fits 8 GB? |
|--------|------------|-------|---------|--------|------------|
| 200 | 824 | 1,024 | 448 MB | 116 MB | Both: Yes |
| 200 | 3,896 | 4,096 | 1,792 MB | 462 MB | Both: Yes |
| 200 | 6,968 | **7,168** | **3,406 MB** | 879 MB | **fp16: limit** |
| 200 | 7,992 | 8,192 | 3,893 MB | 1,005 MB | fp16: OOM |
| 200 | 16,184 | 16,384 | -- | 2,008 MB | TQ4: Yes |
| 200 | 25,400 | **25,600** | -- | **3,447 MB** | **TQ4: limit** |

**TQ4: 3.6x longer context** (7,168 -> 25,600 tokens on Jetson Orin 8 GB).

### Usage

```bash
# Enable 4-bit KV cache (recommended for Qwen3-0.6B)
PYTHONPATH=engine/build_local python3 train.py --max-steps 300 --kv-bits 4
```

## Supported models & hardware

The engine reads model dimensions from a JSON config at runtime, no recompilation needed. We're actively extending support to more models and GPUs.

| Model | Status | VRAM |
|---|---|---|
| Qwen3-0.6B | Validated | 8GB+ |
| Qwen3-1.7B | Config ready | 16GB+ |
| Qwen3-4B | Config ready | 16GB+ |
| Qwen3-8B | Config ready | 24GB+ |

| GPU | Status |
|---|---|
| RTX 4060 Laptop 8GB (sm_89) | Validated |
| Jetson Orin Nano 8GB (sm_87) | Validated |
| Any NVIDIA GPU sm_61+ (Pascal and later) | Target |

Want to see a specific model or GPU supported? [Open an issue](https://github.com/BY571/triebwerk/issues).

