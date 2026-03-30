"""Triebwerk CLI entry point.

Standalone GRPO training with C++ engine -- no TRL dependency.

Implements GRPO (Group Relative Policy Optimization) directly in PyTorch.
Uses the C++ engine for fast generation and PyTorch for loss + gradients.

Supports three loss variants:
  - "grpo": PPO-style clipped surrogate (min of clipped and unclipped)
  - "dapo": Direct clipping without min, skips degenerate groups
  - "dg":   Delightful Policy Gradient + Kondo gate (no reference model)

Usage (on Jetson with C++ engine):
    PYTHONPATH=engine/build2 python3 train.py --max-steps 300

Usage (dry run on any machine, uses HF generate):
    python3 train.py --max-steps 5 --dry-run
"""
import argparse

import torch

from triebwerk import GRPOTrainer, DGTrainer


# ── Backward compatibility ──
# Keep `from train import grpo_train` working for existing examples

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
    """High-level GRPO training API (backward-compatible wrapper).

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
        loss_type: "dapo", "grpo", or "dg".
        output_dir: Where to save checkpoints.
        dry_run: Use HF generate instead of C++ engine.
        stop_texts: List of strings to stop generation (e.g. ["</answer>"]).
        **kwargs: Additional args (epsilon, temperature, top_p, etc.)

    Returns:
        Path to final LoRA adapter directory.
    """
    TrainerClass = DGTrainer if loss_type == "dg" else GRPOTrainer

    trainer_kwargs = dict(
        model=model,
        reward_funcs=reward_funcs,
        lora_rank=lora_rank,
        output_dir=output_dir,
        dry_run=dry_run,
        lr=lr,
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

    if loss_type == "dg":
        trainer_kwargs["eta"] = kwargs.get("dg_eta", 1.0)
    else:
        trainer_kwargs["loss_type"] = loss_type

    trainer = TrainerClass(**trainer_kwargs)
    return trainer.train(
        dataset=dataset,
        max_steps=max_steps,
        num_generations=num_generations,
        max_completion_tokens=max_completion_tokens,
        stop_texts=stop_texts,
    )


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
    parser.add_argument("--loss-type", choices=["grpo", "dapo", "dg"], default="dapo")
    parser.add_argument("--dg-eta", type=float, default=1.0,
                        help="DG temperature (only used with --loss-type dg, default 1.0)")
    parser.add_argument("--mask-truncated", action="store_true", default=True)
    parser.add_argument("--no-mask-truncated", dest="mask_truncated", action="store_false")
    # Output
    parser.add_argument("--output-dir", default="./checkpoints_engine")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=100)
    # Engine options
    parser.add_argument("--dry-run", action="store_true",
                        help="Use HF generate instead of C++ engine (for testing)")
    parser.add_argument("--kv-bits", type=int, default=0,
                        help="KV cache quantization: 0=fp16, 2=TurboQuant 2-bit (8x compression)")
    args = parser.parse_args()

    if args.epsilon_high is None:
        args.epsilon_high = args.epsilon

    TrainerClass = DGTrainer if args.loss_type == "dg" else GRPOTrainer

    trainer_kwargs = dict(
        model=args.model,
        lora_rank=args.lora_rank,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        lr=args.lr,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        temperature=args.temperature,
        top_p=args.top_p,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        gradient_checkpointing=args.gradient_checkpointing,
        mask_truncated=args.mask_truncated,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        kv_bits=args.kv_bits,
    )

    if args.loss_type == "dg":
        trainer_kwargs["eta"] = args.dg_eta
    else:
        trainer_kwargs["loss_type"] = args.loss_type

    trainer = TrainerClass(**trainer_kwargs)
    trainer.train(
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_tokens=args.max_completion_tokens,
    )


if __name__ == "__main__":
    main()
