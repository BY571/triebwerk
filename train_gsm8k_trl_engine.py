"""GRPO training using TRL's GRPOTrainer + C++ engine for generation.

Best of both worlds:
- C++ engine for fast generation (57 tok/s)
- TRL's correct GRPO loss (clipped surrogate, advantage normalization, DAPO)

The engine generates completions, PyTorch computes log-probs,
and TRL handles the entire training loop with proper GRPO math.

Usage:
    docker run --runtime nvidia --network=host \\
      -v $(pwd):/workspace \\
      -v ~/.cache/huggingface:/root/.cache/huggingface \\
      -e CUDA_HOME=/usr/local/cuda-12.6 \\
      -e PATH=... \\
      grpo-jetson bash -c 'PYTHONPATH=engine/build python3 train_gsm8k_trl_engine.py --max-steps 300'
"""
import argparse
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, "engine/build")
import jetson_engine

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16
from lora_sync import LoRASyncer
from engine_rollout import EngineRollout

patch_amp_for_jetson()

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


def correctness_reward(completions, answer, **kwargs):
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
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", default="./checkpoints_trl_engine")
    args = parser.parse_args()

    print("=" * 60)
    print("GRPO Training — TRL + C++ Engine")
    print("  TRL's GRPO loss (clipped surrogate, DAPO)")
    print("  C++ engine generation (57 tok/s)")
    print("=" * 60)

    # ── C++ Engine ──
    print("\nLoading C++ engine...")
    engine = jetson_engine.Engine(1024)
    engine.load_weights("engine/weights")

    # ── PyTorch model ──
    print(f"\nLoading PyTorch model ({args.model})...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cast_model_to_fp16(model)

    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # ── LoRA syncer ──
    syncer = LoRASyncer(model, engine, lora_alpha=args.lora_rank, lora_rank=args.lora_rank)

    # ── Engine rollout function ──
    rollout = EngineRollout(
        engine, model, tokenizer,
        max_completion_tokens=args.max_completion_tokens,
        temperature=1.0, top_p=0.9,
        lora_syncer=syncer,
    )

    # ── Dataset ──
    print("\nLoading GSM8K...")
    dataset = get_gsm8k_dataset()

    # ── GRPOConfig (same as baseline for fair comparison) ──
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name="gsm8k-trl-engine",
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.num_generations,
        max_completion_length=args.max_completion_tokens,
        learning_rate=args.lr,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        beta=0.0,
        temperature=1.0,
        loss_type="dapo",
        scale_rewards=False,
        mask_truncated_completions=True,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=1,
        save_steps=100,
        save_strategy="steps",
        report_to="none",
    )

    print(f"\nTraining: {args.max_steps} steps, G={args.num_generations}")
    print(f"  Loss: TRL GRPO (DAPO, clipped surrogate)")
    print(f"  Generation: C++ engine (57 tok/s)")
    print("=" * 60)

    # ── Train with TRL + engine rollout ──
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        rollout_func=rollout,
    )
    trainer.train()

    # Save
    model.save_pretrained(f"{args.output_dir}/final_lora")
    tokenizer.save_pretrained(f"{args.output_dir}/final_lora")
    print(f"\nDone! LoRA saved to {args.output_dir}/final_lora")


if __name__ == "__main__":
    main()
