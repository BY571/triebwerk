"""Shared benchmark configuration — used by BOTH TRL and our engine.

This ensures identical hyperparameters for a fair speed comparison.
Neither script defines its own training params — they all come from here.
"""

# Model
MODEL = "Qwen/Qwen3-0.6B"
LORA_RANK = 16
LORA_ALPHA = 16
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# GRPO
NUM_GENERATIONS = 4       # G: completions per prompt
MAX_COMPLETION_TOKENS = 512
LEARNING_RATE = 5e-6
BETA = 0.0                # no KL penalty (DAPO)
TEMPERATURE = 1.0         # generation temperature
TOP_P = 0.9
LOSS_TYPE = "dapo"
SCALE_REWARDS = False
MASK_TRUNCATED = True
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.1
EPSILON = 0.2             # PPO clip range

# Training
BENCHMARK_STEPS = 20      # enough to measure average step time
GRADIENT_CHECKPOINTING = True

# Quantization
LOAD_IN_4BIT = True
BNB_COMPUTE_DTYPE = "float16"
BNB_QUANT_TYPE = "nf4"
BNB_DOUBLE_QUANT = True

# System prompt (same for both)
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""
