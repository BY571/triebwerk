"""GRPO training on GSM8K with Qwen3-0.6B — Jetson Orin baseline.

Adapted from the unsloth GRPO tutorial:
https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

Tutorial defaults (cloud with vLLM):
  model=Qwen2.5-3B, use_vllm=True, num_generations=8, lora_rank=32,
  learning_rate=5e-6, max_steps=300, max_seq_length=1024

Jetson adaptations:
  - No vLLM (ARM, not supported) -> HF generate
  - Smaller model (0.6B vs 3B) to fit 8GB
  - fp16 compute dtype (Jetson doesn't support bf16 AMP)
  - generation_batch_size omitted (must be multiple of batch size)
  - Monkey-patched AMP for bf16 gradient handling

Usage:
    # Full baseline (300 steps, ~12h on Jetson Orin 8GB)
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      jetson-llm-train python3 train_gsm8k.py

    # Quick smoke test (10 steps)
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      jetson-llm-train python3 train_gsm8k.py --max-steps 10
"""
import argparse
import json
import os
import re
import time
from datetime import datetime

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

# ── Jetson patches (must be called before training) ──
patch_amp_for_jetson()

# ── System prompt and format (from unsloth tutorial) ──
SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ── Reward functions (from unsloth tutorial) ──
def correctness_reward(completions, answer, **kwargs):
    """Binary reward: +2 if extracted answer matches ground truth, -1 otherwise."""
    rewards = []
    for completion, ans in zip(completions, answer):
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        extracted = extract_xml_answer(text)
        rewards.append(2.0 if extracted == ans else -1.0)
    return rewards


def format_reward(completions, **kwargs):
    """Reward correct XML format: <reasoning>...</reasoning><answer>...</answer>."""
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            text = completion[-1]["content"] if completion else ""
        else:
            text = str(completion)
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        if has_reasoning and has_answer:
            rewards.append(1.0)
        elif has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ── Dataset ──
def get_gsm8k_dataset(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return data


def save_run_config(args, grpo_config, model_info, output_path):
    """Save all hyperparameters and system info for reproducibility."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "script": "train_gsm8k.py",
        "tutorial_source": "https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo",
        "model": {
            "name": args.model,
            "quantization": "4-bit NF4",
            "compute_dtype": "float16",
            "double_quant": True,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_rank,
            "max_seq_length": args.max_seq_length,
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
            "trainable_params": model_info["trainable"],
            "total_params": model_info["total"],
        },
        "grpo": {
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "per_device_train_batch_size": args.num_generations,
            "max_completion_length": args.max_completion_tokens,
            "learning_rate": args.lr,
            "beta": 0.0,
            "temperature": 1.0,
            "loss_type": "dapo",
            "scale_rewards": False,
            "mask_truncated_completions": False,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "gradient_checkpointing": True,
            "fp16": True,
            "bf16": False,
        },
        "reward_functions": ["format_reward", "correctness_reward"],
        "dataset": "openai/gsm8k",
        "jetson": {
            "amp_patch": "bf16_grads_to_fp32",
            "vllm": False,
            "note": "No vLLM on ARM. HF generate fallback.",
        },
        "tutorial_defaults_diff": {
            "use_vllm": "tutorial=True, jetson=False (ARM)",
            "num_generations": f"tutorial=8, jetson={args.num_generations} (memory)",
            "max_completion_tokens": f"tutorial=1024, jetson={args.max_completion_tokens}",
            "max_seq_length": f"tutorial=1024, jetson={args.max_seq_length}",
            "model": f"tutorial=Qwen2.5-3B, jetson={args.model} (8GB)",
            "lora_rank": f"tutorial=32, jetson={args.lora_rank}",
        },
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Run config saved to: {output_path}")


class MetricsLogger:
    """Collect per-step metrics from TRL trainer logs."""

    def __init__(self, output_path):
        self.output_path = output_path
        self.metrics = []
        self.step_times = []
        self.t_last = None

    def on_log(self, step, logs):
        now = time.time()
        if self.t_last is not None:
            self.step_times.append(now - self.t_last)
        self.t_last = now
        entry = {"step": step, "wall_time": now, **logs}
        self.metrics.append(entry)

    def save(self, total_time):
        summary = {
            "total_time_s": total_time,
            "total_steps": len(self.metrics),
            "avg_step_time_s": sum(self.step_times) / len(self.step_times) if self.step_times else 0,
            "steps": self.metrics,
        }
        with open(self.output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Metrics saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Base model (quantized on-the-fly with bnb 4-bit)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Training steps (unsloth tutorial recommends 300 minimum)")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Completions per prompt (tutorial uses 8, reduced for 8GB)")
    parser.add_argument("--max-completion-tokens", type=int, default=1024,
                        help="Max new tokens per completion (matches tutorial max_seq_length)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max total sequence length (prompt+completion). Limits KV cache allocation.")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate (matches tutorial)")
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (matches tutorial default of 32)")
    parser.add_argument("--output-dir", default="./checkpoints",
                        help="Checkpoint directory")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training — GSM8K Baseline (Jetson Orin)")
    print(f"Run: {run_id}")
    print("=" * 60)

    # ── Load model ──
    print(f"\nLoading {args.model} (4-bit NF4, fp16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=args.max_seq_length,  # limit KV cache allocation
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_seq_length
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = cast_model_to_fp16(model)

    # ── Add LoRA ──
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={args.lora_rank}, trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── Dataset ──
    print("\nLoading GSM8K dataset...")
    dataset = get_gsm8k_dataset()
    print(f"  {len(dataset)} training examples")

    # ── Save run config ──
    save_run_config(args, None, {"trainable": trainable, "total": total},
                    f"{run_dir}/config.json")

    # ── GRPO config ──
    G = args.num_generations
    grpo_config = GRPOConfig(
        output_dir=run_dir,
        run_name=f"gsm8k-jetson-{run_id}",
        max_steps=args.max_steps,
        num_generations=G,
        per_device_train_batch_size=G,
        max_completion_length=args.max_completion_tokens,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        beta=0.0,
        temperature=1.0,
        loss_type="dapo",
        scale_rewards=False,
        mask_truncated_completions=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        log_completions=True,
        save_steps=100,
        save_strategy="steps",
        report_to="none",
    )

    print(f"\nConfig:")
    print(f"  steps={args.max_steps}, G={G}, max_tokens={args.max_completion_tokens}, max_seq={args.max_seq_length}")
    print(f"  LR={args.lr}, beta=0.0, loss=dapo, scale_rewards=False")
    print(f"  lora_rank={args.lora_rank}, gradient_checkpointing=True")
    print(f"  fp16=True, bf16=False (Jetson compat)")
    print("=" * 60)

    # ── Train ──
    t0 = time.time()
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Hook into trainer logging to capture metrics
    metrics_logger = MetricsLogger(f"{run_dir}/metrics.json")
    original_log = trainer.log

    def patched_log(logs):
        metrics_logger.on_log(trainer.state.global_step, logs)
        return original_log(logs)

    trainer.log = patched_log

    trainer.train()
    elapsed = time.time() - t0

    # ── Save ──
    lora_path = f"{run_dir}/final_lora"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    metrics_logger.save(elapsed)

    # Save final summary
    summary = {
        "run_id": run_id,
        "total_time_s": elapsed,
        "total_time_h": elapsed / 3600,
        "steps": args.max_steps,
        "s_per_step": elapsed / args.max_steps,
        "model": args.model,
        "lora_path": lora_path,
    }
    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Time: {elapsed/3600:.1f}h ({elapsed/args.max_steps:.1f}s/step)")
    print(f"  LoRA: {lora_path}")
    print(f"  Config: {run_dir}/config.json")
    print(f"  Metrics: {run_dir}/metrics.json")
    print(f"  Summary: {run_dir}/summary.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
