"""GRPO training on GSM8K with Qwen3-0.6B — Jetson Orin baseline.

Adapted from the unsloth GRPO tutorial:
https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo

Key differences from cloud training:
- No vLLM (not available on ARM/Jetson)
- fp16 instead of bf16 (Jetson AMP compat)
- generation_batch_size for memory management
- Monkey-patched AMP for bf16 gradient handling

Usage:
    docker run --runtime nvidia --network=host \
      -v $(pwd):/workspace \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      jetson-llm-train python3 train_gsm8k.py

    # Quick test (10 steps)
    docker run --runtime nvidia --network=host \
      -v $(pwd):/workspace \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      jetson-llm-train python3 train_gsm8k.py --max-steps 10
"""
import argparse
import re
import time

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

# ── Jetson patches (must be called before training) ──
patch_amp_for_jetson()

# ── System prompt and format ──
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


# ── Reward functions ──
def correctness_reward(completions, answer, **kwargs):
    """Binary reward: +2 if answer matches, -1 otherwise."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Base model (quantized on-the-fly)")
    parser.add_argument("--max-steps", type=int, default=300, help="Training steps (300 minimum for results)")
    parser.add_argument("--num-generations", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max-completion-tokens", type=int, default=512, help="Max tokens per completion")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--output-dir", default="./checkpoints", help="Checkpoint directory")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training — GSM8K Baseline (Jetson Orin)")
    print("=" * 60)

    # ── Load model ──
    print(f"\nLoading {args.model} (4-bit, fp16 compute)...")
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
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
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

    # ── GRPO config ──
    G = args.num_generations
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name="gsm8k-jetson",
        max_steps=args.max_steps,
        num_generations=G,
        per_device_train_batch_size=G,
        max_completion_length=args.max_completion_tokens,
        learning_rate=args.lr,
        beta=0.0,
        temperature=1.0,
        loss_type="dapo",
        scale_rewards=False,
        mask_truncated_completions=False,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=100,
        save_strategy="steps",
        report_to="none",
    )

    print(f"\nTraining: {args.max_steps} steps, G={G}, max_tokens={args.max_completion_tokens}")
    print(f"  LR={args.lr}, beta=0.0, loss=dapo")
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
    trainer.train()
    elapsed = time.time() - t0

    # ── Save ──
    model.save_pretrained(f"{args.output_dir}/final_lora")
    tokenizer.save_pretrained(f"{args.output_dir}/final_lora")
    print(f"\nTraining complete in {elapsed/3600:.1f}h ({elapsed/args.max_steps:.1f}s/step)")
    print(f"LoRA saved to: {args.output_dir}/final_lora")


if __name__ == "__main__":
    main()
