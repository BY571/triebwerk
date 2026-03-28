"""vLLM generation + PyTorch GRPO training benchmark.

Uses vLLM (the industry standard) for generation and our same PyTorch
GRPO training loop for gradients. This is the fair comparison: identical
training, only the generation backend differs.

Usage:
    python3 benchmark/bench_vllm.py
"""
import json
import math
import os
import re
import sys
import time
from datetime import datetime

# Must set before vLLM imports
os.environ["VLLM_PLUGINS"] = ""
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

sys.path.insert(0, os.path.dirname(__file__))
from config import *

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16
from train import (compute_advantages, compute_token_logprobs, grpo_step,
                   extract_hash_answer, format_reward, correctness_reward,
                   get_gsm8k_dataset)

patch_amp_for_jetson()


def main():
    print("=" * 60)
    print("BENCHMARK: vLLM generation + PyTorch GRPO")
    print(f"  Model: {MODEL}, G={NUM_GENERATIONS}, tokens={MAX_COMPLETION_TOKENS}")
    print(f"  Steps: {BENCHMARK_STEPS}")
    print("=" * 60)

    # ── Load PyTorch model for training ──
    print(f"\nLoading {MODEL} for training (4-bit NF4, fp16)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, quantization_config=bnb_config,
        device_map="auto", torch_dtype=torch.float16,
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
    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={LORA_RANK}, trainable: {trainable:,} / {total:,}")

    # ── Load vLLM engine for generation ──
    print(f"\nLoading vLLM engine...")
    # Free GPU memory before loading vLLM
    torch.cuda.empty_cache()
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.4,
        max_model_len=2048,
        dtype="half",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
    )
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_COMPLETION_TOKENS,
        stop=["</answer>"],
    )

    # ── Setup ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE, weight_decay=0.0,
    )
    scaler = torch.amp.GradScaler("cuda")
    device = next(model.parameters()).device

    dataset = get_gsm8k_dataset()
    dataset_iter = iter(dataset)

    warmup_steps = int(BENCHMARK_STEPS * WARMUP_RATIO)

    # Config namespace for grpo_step
    import argparse
    config = argparse.Namespace(
        epsilon=EPSILON, epsilon_high=EPSILON * 2,
        loss_type=LOSS_TYPE, max_grad_norm=MAX_GRAD_NORM,
        mask_truncated=MASK_TRUNCATED,
    )

    all_metrics = []
    t_start = time.time()

    print(f"\nTraining: {BENCHMARK_STEPS} steps, G={NUM_GENERATIONS}")
    print("=" * 60)

    for step in range(1, BENCHMARK_STEPS + 1):
        t_step = time.time()

        # LR schedule
        if step <= warmup_steps:
            lr = LEARNING_RATE * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(BENCHMARK_STEPS - warmup_steps, 1)
            lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample prompt
        try:
            sample = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            sample = next(dataset_iter)

        prompt = sample["prompt"]
        answer = sample["answer"]

        # 1. Generate with vLLM
        t_gen = time.time()
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer(text).input_ids
        eos_id = tokenizer.eos_token_id

        outputs = llm.generate(
            [text] * NUM_GENERATIONS,
            sampling_params,
            use_tqdm=False,
        )

        completions = []
        for out in outputs:
            comp_ids = list(out.outputs[0].token_ids)
            truncated = (
                len(comp_ids) >= MAX_COMPLETION_TOKENS
                and (len(comp_ids) == 0 or comp_ids[-1] != eos_id)
            )
            completions.append({
                "prompt_ids": prompt_ids,
                "completion_ids": comp_ids,
                "truncated": truncated,
            })
        t_gen = time.time() - t_gen

        # 2. Score
        comp_texts = [
            tokenizer.decode(c["completion_ids"], skip_special_tokens=True)
            for c in completions
        ]
        expanded_answers = [answer] * NUM_GENERATIONS

        rewards = torch.zeros(NUM_GENERATIONS)
        for func in [format_reward, correctness_reward]:
            scores = func(comp_texts, answer=expanded_answers)
            rewards += torch.tensor(scores, dtype=torch.float32)

        # 3. Advantages
        advantages = compute_advantages(rewards, NUM_GENERATIONS)

        # 4. Reference log-probs
        old_logprobs = []
        with torch.no_grad():
            for c in completions:
                lp = compute_token_logprobs(
                    model, c["prompt_ids"], c["completion_ids"], device
                ).detach()
                old_logprobs.append(lp)

        # 5. Build samples
        samples = []
        for i, c in enumerate(completions):
            mask_weight = 0.0 if (MASK_TRUNCATED and c["truncated"]) else 1.0
            samples.append({
                "prompt_ids": c["prompt_ids"],
                "completion_ids": c["completion_ids"],
                "old_logprobs": old_logprobs[i],
                "advantage": advantages[i].item(),
                "mask_weight": mask_weight,
            })

        # 6. GRPO step
        loss = grpo_step(model, optimizer, scaler, samples, config)

        # 7. Metrics
        total_tokens = sum(len(c["completion_ids"]) for c in completions)
        step_time = time.time() - t_step

        metrics = {
            "step": step, "loss": loss,
            "mean_reward": rewards.mean().item(),
            "gen_time": t_gen,
            "gen_tokens": total_tokens,
            "gen_tok_per_s": total_tokens / t_gen if t_gen > 0 else 0,
            "step_time": step_time,
        }
        all_metrics.append(metrics)

        elapsed = time.time() - t_start
        print(
            f"[{step:>4}/{BENCHMARK_STEPS}] "
            f"loss={loss:>7.4f}  "
            f"reward={rewards.mean().item():>5.2f}  "
            f"tok/s={metrics['gen_tok_per_s']:>5.0f}  "
            f"gen={t_gen:>4.1f}s  "
            f"step={step_time:>4.1f}s  "
            f"[{elapsed/60:.0f}m]"
        )

    total_time = time.time() - t_start
    avg_step = total_time / BENCHMARK_STEPS

    print(f"\n{'='*60}")
    print(f"vLLM RESULT: {BENCHMARK_STEPS} steps in {total_time:.0f}s ({avg_step:.1f}s/step)")
    print(f"{'='*60}")

    result = {
        "engine": "vllm",
        "steps": BENCHMARK_STEPS,
        "total_time_s": total_time,
        "avg_step_time_s": avg_step,
        "config": {
            "model": MODEL, "G": NUM_GENERATIONS,
            "max_tokens": MAX_COMPLETION_TOKENS,
            "lora_rank": LORA_RANK, "loss": LOSS_TYPE,
        },
    }
    os.makedirs("benchmark/results", exist_ok=True)
    with open("benchmark/results/vllm.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
