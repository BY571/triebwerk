"""Standalone GRPO training with C++ engine — no TRL dependency.

Implements GRPO (Group Relative Policy Optimization) directly in PyTorch.
Uses the C++ engine for fast generation and PyTorch for loss + gradients.

Supports two loss variants:
  - "grpo": PPO-style clipped surrogate (min of clipped and unclipped)
  - "dapo": Direct clipping without min, skips degenerate groups

Usage (on Jetson with C++ engine):
    PYTHONPATH=engine/build2 python3 train.py --max-steps 300

Usage (dry run on any machine, uses HF generate):
    python3 train.py --max-steps 5 --dry-run
"""
import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16

patch_amp_for_jetson()

# ── Prompt / reward (same as TRL baseline for fair comparison) ──

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
        extracted = extract_xml_answer(completion)
        rewards.append(2.0 if extracted == ans else -1.0)
    return rewards


def format_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        has_reasoning = bool(re.search(r"<reasoning>.*?</reasoning>", completion, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
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


# ── GRPO Core ──

def compute_advantages(rewards, num_generations):
    """Per-group advantage normalization.

    Groups rewards by prompt (G completions each), normalizes within group.
    Returns zero advantages for groups with zero variance (DAPO dynamic sampling).
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)

    # Zero-variance groups get zero advantage (no gradient signal)
    advantages = torch.where(
        std > 1e-8,
        (grouped - mean) / std,
        torch.zeros_like(grouped),
    )
    return advantages.view(-1)


def compute_token_logprobs(model, prompt_ids, completion_ids, device):
    """Forward pass to get per-token log-probs for the completion.

    Returns (completion_len,) tensor of log-probs.
    """
    if len(completion_ids) == 0:
        return torch.zeros(0, device=device)

    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([full_ids], device=device)

    # Use same AMP context as grpo_step to avoid ratio bias from precision mismatch
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_tensor)
        logits = outputs.logits[0, :-1, :]
    targets = input_tensor[0, 1:]

    # fp32 softmax for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Keep only completion portion
    comp_lp = token_lp[len(prompt_ids) - 1:]

    del outputs, logits, log_probs, token_lp, input_tensor
    return comp_lp


def grpo_step(model, optimizer, scaler, samples, config):
    """One GRPO gradient step.

    Args:
        model: PEFT model (LoRA on 4-bit base)
        optimizer: AdamW
        scaler: GradScaler for fp16 AMP
        samples: list of dicts with keys:
            prompt_ids, completion_ids, old_logprobs, advantage, mask_weight
        config: namespace with epsilon, epsilon_high, loss_type, max_grad_norm

    Returns:
        mean loss (float)
    """
    device = next(model.parameters()).device
    optimizer.zero_grad()

    # Pre-filter valid samples for correct loss normalization
    valid_samples = [s for s in samples
                     if s["mask_weight"] != 0.0
                     and len(s["completion_ids"]) > 0
                     and s["advantage"] != 0.0]
    n_valid = len(valid_samples)
    if n_valid == 0:
        optimizer.zero_grad()
        return 0.0

    total_loss_val = 0.0

    for s in valid_samples:
        p_ids = s["prompt_ids"]
        c_ids = s["completion_ids"]
        old_lp = s["old_logprobs"]
        adv = s["advantage"]

        full_ids = p_ids + c_ids
        input_tensor = torch.tensor([full_ids], device=device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(input_tensor)
            logits = outputs.logits[0, :-1, :]
            targets = input_tensor[0, 1:]

            new_lp = F.log_softmax(logits.float(), dim=-1)
            new_lp = new_lp.gather(1, targets.unsqueeze(1)).squeeze(1)
            new_comp_lp = new_lp[len(p_ids) - 1:]

            ratio = torch.exp(new_comp_lp - old_lp)

            if config.loss_type == "grpo":
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon_high) * adv
                per_token_loss = -torch.min(surr1, surr2)
            else:  # dapo
                clipped_ratio = torch.clamp(ratio, 1.0 - config.epsilon, 1.0 + config.epsilon_high)
                per_token_loss = -clipped_ratio * adv

            sample_loss = per_token_loss.mean()

        # Skip NaN/Inf losses (can happen with fp16 overflow)
        loss_val = sample_loss.item()
        if math.isnan(loss_val) or math.isinf(loss_val):
            del outputs, logits, new_lp, input_tensor, sample_loss
            del ratio, per_token_loss, new_comp_lp
            continue

        # Normalize by n_valid (not n_samples) for consistent gradients
        scaler.scale(sample_loss / n_valid).backward()
        total_loss_val += loss_val

        # Free intermediates + flush CUDA allocator
        # empty_cache() IS needed on Jetson unified memory to prevent fragmentation
        # that causes OOM after ~8 steps. On discrete GPUs this can be removed.
        del outputs, logits, new_lp, input_tensor, sample_loss
        del ratio, per_token_loss, new_comp_lp
        if config.loss_type == "grpo":
            del surr1, surr2
        else:
            del clipped_ratio
        torch.cuda.empty_cache()

    # Gradient clipping + optimizer step
    if n_valid > 0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        if math.isnan(grad_norm.item()) or math.isinf(grad_norm.item()):
            print(f"  WARNING: NaN/Inf gradient norm, skipping step")
            optimizer.zero_grad()
            return float('nan')
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.zero_grad()

    return total_loss_val / max(n_valid, 1)


# ── Generation ──

def generate_with_engine(engine, tokenizer, prompt, num_generations,
                         max_tokens, temperature, top_p, stop_ids):
    """Generate G completions using C++ engine. Sequential (dp4a, memory-safe)."""
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    prompt_ids = tokenizer(text).input_ids
    eos_id = tokenizer.eos_token_id
    completions = []

    for _ in range(num_generations):
        engine.reset()
        comp_ids = engine.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
            stop_token_ids=stop_ids,
        )
        truncated = (
            len(comp_ids) >= max_tokens
            and (len(comp_ids) == 0 or comp_ids[-1] != eos_id)
        )
        completions.append({
            "prompt_ids": prompt_ids,
            "completion_ids": comp_ids,
            "truncated": truncated,
        })

    return completions


def generate_with_hf(model, tokenizer, prompt, num_generations,
                     max_tokens, temperature, top_p, stop_ids):
    """Generate G completions using HuggingFace generate (for dry-run)."""
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    eos_id = tokenizer.eos_token_id
    completions = []

    for _ in range(num_generations):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=eos_id,
            )
        comp_ids = out[0][len(prompt_ids):].tolist()
        truncated = (
            len(comp_ids) >= max_tokens
            and (len(comp_ids) == 0 or comp_ids[-1] != eos_id)
        )
        completions.append({
            "prompt_ids": prompt_ids,
            "completion_ids": comp_ids,
            "truncated": truncated,
        })

    return completions


# ── Training Loop ──

def train(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{args.output_dir}/{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    print("=" * 60)
    print("GRPO Training — Standalone (no TRL)")
    print(f"  Loss: {args.loss_type}, epsilon={args.epsilon}")
    print(f"  Model: {args.model}")
    print(f"  Engine: {'C++ engine' if not args.dry_run else 'HF generate (dry-run)'}")
    print(f"  Run: {run_id}")
    print("=" * 60)

    # ── Load model ──
    print(f"\nLoading {args.model} (4-bit NF4, fp16)...")
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
    cast_model_to_fp16(model)

    # ── LoRA ──
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA rank={args.lora_rank}, trainable: {trainable:,} / {total:,}")

    # ── C++ Engine (if not dry-run) ──
    engine = None
    syncer = None
    if not args.dry_run:
        sys.path.insert(0, os.environ.get("ENGINE_BUILD", "engine/build2"))
        import jetson_engine
        engine = jetson_engine.Engine(1024)
        weights_path = os.environ.get("ENGINE_WEIGHTS", "engine/weights_q4l")
        print(f"\nLoading C++ engine weights from {weights_path}...")
        engine.load_weights(weights_path)

        # Share PyTorch's embedding with C++ engine (saves 311MB on Jetson)
        embed_weight = model.base_model.model.model.embed_tokens.weight
        if embed_weight.dtype == torch.float16 and embed_weight.is_cuda:
            engine.share_embedding(embed_weight.data_ptr())
        else:
            print(f"  Warning: cannot share embedding (dtype={embed_weight.dtype}, cuda={embed_weight.is_cuda})")

        from lora_sync import LoRASyncer
        syncer = LoRASyncer(model, engine,
                            lora_alpha=args.lora_rank, lora_rank=args.lora_rank)

    # ── Optimizer + scaler ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    device = next(model.parameters()).device

    # ── Dataset & rewards ──
    if hasattr(args, "_dataset") and args._dataset is not None:
        dataset = args._dataset
        reward_funcs = args._reward_funcs
        stop_texts = args._stop_texts or []
    else:
        print("\nLoading GSM8K...")
        dataset = get_gsm8k_dataset()
        reward_funcs = [format_reward, correctness_reward]
        stop_texts = ["</answer>"]
    dataset_iter = iter(dataset)

    stop_ids = []
    for text in stop_texts:
        stop_ids.extend(tokenizer.encode(text, add_special_tokens=False))
    warmup_steps = int(args.max_steps * args.warmup_ratio)

    # ── Save config ──
    config_dict = {k: v for k, v in vars(args).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool, type(None)))}
    config_dict["run_id"] = run_id
    config_dict["trainable_params"] = trainable
    config_dict["total_params"] = total
    with open(f"{run_dir}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # ── Train ──
    all_metrics = []
    t_start = time.time()

    print(f"\nTraining: {args.max_steps} steps, G={args.num_generations}")
    print("=" * 60)

    for step in range(1, args.max_steps + 1):
        t_step = time.time()

        # LR schedule: linear warmup + cosine decay
        if step <= warmup_steps:
            lr = args.lr * step / max(warmup_steps, 1)
        else:
            progress = (step - warmup_steps) / max(args.max_steps - warmup_steps, 1)
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample a prompt
        try:
            sample = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            sample = next(dataset_iter)

        prompt = sample["prompt"]
        answer = sample["answer"]

        # 1. Sync LoRA to engine
        if syncer is not None:
            syncer.sync()

        # 2. Generate G completions
        t_gen = time.time()
        if engine is not None:
            completions = generate_with_engine(
                engine, tokenizer, prompt, args.num_generations,
                args.max_completion_tokens, args.temperature, args.top_p, stop_ids,
            )
        else:
            completions = generate_with_hf(
                model, tokenizer, prompt, args.num_generations,
                args.max_completion_tokens, args.temperature, args.top_p, stop_ids,
            )
        t_gen = time.time() - t_gen

        # 3. Decode and score
        comp_texts = [
            tokenizer.decode(c["completion_ids"], skip_special_tokens=True)
            for c in completions
        ]
        expanded_answers = [answer] * args.num_generations

        rewards = torch.zeros(args.num_generations)
        for func in reward_funcs:
            scores = func(comp_texts, answer=expanded_answers)
            rewards += torch.tensor(scores, dtype=torch.float32)

        # 4. Compute advantages
        advantages = compute_advantages(rewards, args.num_generations)

        # 5. Compute reference log-probs (old policy, no grad, one at a time)
        old_logprobs = []
        with torch.no_grad():
            for c in completions:
                lp = compute_token_logprobs(
                    model, c["prompt_ids"], c["completion_ids"], device
                ).detach()
                old_logprobs.append(lp)
                # No .cpu() on Jetson unified memory (CPU=GPU RAM, copy wastes memory)
                # No empty_cache() — allocator reuses blocks for next forward pass

        # 6. Build samples for the gradient step
        samples = []
        for i, c in enumerate(completions):
            mask_weight = 0.0 if (args.mask_truncated and c["truncated"]) else 1.0
            samples.append({
                "prompt_ids": c["prompt_ids"],
                "completion_ids": c["completion_ids"],
                "old_logprobs": old_logprobs[i],
                "advantage": advantages[i].item(),
                "mask_weight": mask_weight,
            })

        # 7. GRPO gradient step
        loss = grpo_step(model, optimizer, scaler, samples, args)

        # 8. Metrics
        total_tokens = sum(len(c["completion_ids"]) for c in completions)
        step_time = time.time() - t_step

        metrics = {
            "step": step,
            "loss": loss,
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "min_reward": rewards.min().item(),
            "reward_std": rewards.std().item(),
            "gen_time": t_gen,
            "gen_tokens": total_tokens,
            "gen_tok_per_s": total_tokens / t_gen if t_gen > 0 else 0,
            "step_time": step_time,
            "lr": lr,
            "n_truncated": sum(1 for c in completions if c["truncated"]),
        }
        all_metrics.append(metrics)

        if step % args.logging_steps == 0:
            elapsed = time.time() - t_start
            print(
                f"[{step:>4}/{args.max_steps}] "
                f"loss={loss:>7.4f}  "
                f"reward={rewards.mean().item():>5.2f}  "
                f"tok/s={metrics['gen_tok_per_s']:>5.0f}  "
                f"gen={t_gen:>4.1f}s  "
                f"step={step_time:>4.1f}s  "
                f"lr={lr:.1e}  "
                f"[{elapsed/60:.0f}m]"
            )

        # Save checkpoint
        if step % args.save_steps == 0:
            ckpt_path = f"{run_dir}/step_{step}_lora"
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            with open(f"{run_dir}/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=2)
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final save ──
    total_time = time.time() - t_start
    final_path = f"{run_dir}/final_lora"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    summary = {
        "run_id": run_id,
        "total_time_s": total_time,
        "total_time_h": total_time / 3600,
        "steps": args.max_steps,
        "s_per_step": total_time / args.max_steps,
        "final_mean_reward": all_metrics[-1]["mean_reward"] if all_metrics else 0,
    }
    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Time: {total_time/3600:.1f}h ({total_time/args.max_steps:.1f}s/step)")
    print(f"  Final reward: {summary['final_mean_reward']:.2f}")
    print(f"  LoRA: {final_path}")
    print(f"  Metrics: {run_dir}/metrics.json")
    print(f"{'=' * 60}")


def grpo_train(
    dataset,
    reward_funcs,
    model="Qwen/Qwen3-0.6B",
    max_steps=300,
    num_generations=4,
    max_completion_tokens=512,
    lr=5e-6,
    lora_rank=16,
    loss_type="dapo",
    output_dir="./checkpoints_engine",
    dry_run=False,
    stop_texts=None,
    **kwargs,
):
    """High-level GRPO training API.

    Args:
        dataset: HuggingFace dataset with "prompt" and "answer" columns.
            "prompt" is either a string or list of message dicts.
            "answer" is passed to reward functions as a keyword arg.
        reward_funcs: List of callables (completions, **kwargs) -> list[float].
            Each receives decoded completion strings and any extra columns.
        model: HuggingFace model name (loaded with 4-bit quantization).
        max_steps: Number of GRPO training steps.
        num_generations: Completions per prompt (G).
        max_completion_tokens: Max tokens per completion.
        lr: Learning rate.
        lora_rank: LoRA adapter rank.
        loss_type: "dapo" or "grpo".
        output_dir: Where to save checkpoints.
        dry_run: Use HF generate instead of C++ engine.
        stop_texts: List of strings to stop generation (e.g. ["</answer>"]).
        **kwargs: Additional args (epsilon, temperature, top_p, etc.)

    Returns:
        Path to final LoRA adapter directory.
    """
    args = argparse.Namespace(
        model=model,
        max_steps=max_steps,
        num_generations=num_generations,
        max_completion_tokens=max_completion_tokens,
        lr=lr,
        lora_rank=lora_rank,
        loss_type=loss_type,
        output_dir=output_dir,
        dry_run=dry_run,
        epsilon=kwargs.get("epsilon", 0.2),
        epsilon_high=kwargs.get("epsilon_high", kwargs.get("epsilon", 0.2)),
        temperature=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 0.9),
        max_grad_norm=kwargs.get("max_grad_norm", 1.0),
        warmup_ratio=kwargs.get("warmup_ratio", 0.1),
        gradient_checkpointing=kwargs.get("gradient_checkpointing", True),
        mask_truncated=kwargs.get("mask_truncated", True),
        logging_steps=kwargs.get("logging_steps", 1),
        save_steps=kwargs.get("save_steps", 100),
    )

    # Inject dataset and reward_funcs so train() can use them
    args._dataset = dataset
    args._reward_funcs = reward_funcs
    args._stop_texts = stop_texts

    train(args)


def main():
    parser = argparse.ArgumentParser(description="GRPO training (standalone, no TRL)")
    # Model
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                        action="store_false")
    # Training
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    # GRPO
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=None)
    parser.add_argument("--loss-type", choices=["grpo", "dapo"], default="dapo")
    parser.add_argument("--mask-truncated", action="store_true", default=True)
    parser.add_argument("--no-mask-truncated", dest="mask_truncated", action="store_false")
    # Output
    parser.add_argument("--output-dir", default="./checkpoints_engine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    # Dry run (no C++ engine, uses HF generate)
    parser.add_argument("--dry-run", action="store_true",
                        help="Use HF generate instead of C++ engine (for testing)")
    args = parser.parse_args()

    if args.epsilon_high is None:
        args.epsilon_high = args.epsilon

    train(args)


if __name__ == "__main__":
    main()
