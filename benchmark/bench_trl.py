"""TRL GRPOTrainer benchmark — the baseline to beat.

Uses HuggingFace generate (no C++ engine). Identical hyperparameters
to bench_ours.py via shared config.py.

Usage:
    python3 benchmark/bench_trl.py
"""
import json
import os
import re
import sys
import time

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.dirname(__file__))
from config import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()


def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def correctness_reward(completions, answer, **kwargs):
    rewards = []
    for completion, ans in zip(completions, answer):
        text = completion[-1]["content"] if isinstance(completion, list) and completion else str(completion)
        rewards.append(2.0 if extract_xml_answer(text) == ans else -1.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[-1]["content"] if isinstance(completion, list) and completion else str(completion)
        has_r = bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL))
        has_a = bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        rewards.append(1.0 if has_r and has_a else (0.5 if has_a else 0.0))
    return rewards


def extract_hash_answer(text):
    return text.split("####")[1].strip() if "####" in text else None


def main():
    print("=" * 60)
    print("BENCHMARK: TRL GRPOTrainer (HF generate)")
    print(f"  Model: {MODEL}, G={NUM_GENERATIONS}, tokens={MAX_COMPLETION_TOKENS}")
    print(f"  Steps: {BENCHMARK_STEPS}")
    print("=" * 60)

    # Load model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=LOAD_IN_4BIT,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type=BNB_QUANT_TYPE,
        bnb_4bit_use_double_quant=BNB_DOUBLE_QUANT,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cast_model_to_fp16(model)

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # Dataset
    dataset = load_dataset("openai/gsm8k", "main")["train"]
    dataset = dataset.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })

    # GRPOConfig — all TRL optimizations enabled for fairest comparison
    grpo_config = GRPOConfig(
        output_dir="/tmp/bench_trl",
        run_name="benchmark-trl",
        max_steps=BENCHMARK_STEPS,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_TOKENS,
        learning_rate=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        beta=BETA,
        temperature=TEMPERATURE,
        loss_type=LOSS_TYPE,
        scale_rewards=SCALE_REWARDS,
        mask_truncated_completions=MASK_TRUNCATED,
        fp16=True, bf16=False,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        # TRL speed optimizations (fair: same algorithm, just faster execution)
        num_iterations=1,            # standard GRPO: generate once, train once
        log_completions=False,       # skip string processing overhead
        disable_dropout=True,        # fewer ops per forward pass
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )

    # Train
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, correctness_reward],
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    avg_step = elapsed / BENCHMARK_STEPS
    print(f"\n{'='*60}")
    print(f"TRL RESULT: {BENCHMARK_STEPS} steps in {elapsed:.0f}s ({avg_step:.1f}s/step)")
    print(f"{'='*60}")

    # Save result
    result = {
        "engine": "trl",
        "steps": BENCHMARK_STEPS,
        "total_time_s": elapsed,
        "avg_step_time_s": avg_step,
        "config": {
            "model": MODEL, "G": NUM_GENERATIONS,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "lora_rank": LORA_RANK, "loss": LOSS_TYPE,
        },
    }
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/trl.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
