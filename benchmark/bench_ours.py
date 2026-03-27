"""Our C++ engine benchmark — the challenger.

Uses custom GRPO trainer with dp4a C++ engine for generation.
Identical hyperparameters to bench_trl.py via shared config.py.

Usage:
    PYTHONPATH=engine/build2 python3 benchmark/bench_ours.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from config import *

# Import the custom trainer's grpo_train API
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from train import train


def main():
    print("=" * 60)
    print("BENCHMARK: C++ Engine (dp4a Q4L)")
    print(f"  Model: {MODEL}, G={NUM_GENERATIONS}, tokens={MAX_COMPLETION_TOKENS}")
    print(f"  Steps: {BENCHMARK_STEPS}")
    print("=" * 60)

    # Build args namespace matching train.py expectations
    args = argparse.Namespace(
        model=MODEL,
        lora_rank=LORA_RANK,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        max_steps=BENCHMARK_STEPS,
        lr=LEARNING_RATE,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        num_generations=NUM_GENERATIONS,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        epsilon=EPSILON,
        epsilon_high=EPSILON * 2,  # DAPO uses wider upper clip
        loss_type=LOSS_TYPE,
        mask_truncated=MASK_TRUNCATED,
        output_dir="/tmp/bench_ours",
        logging_steps=1,
        save_steps=9999,  # don't save during benchmark
        dry_run=False,
        _dataset=None,
        _reward_funcs=None,
        _stop_texts=None,
    )

    t0 = time.time()
    train(args)
    elapsed = time.time() - t0

    avg_step = elapsed / BENCHMARK_STEPS
    print(f"\n{'='*60}")
    print(f"OUR ENGINE RESULT: {BENCHMARK_STEPS} steps in {elapsed:.0f}s ({avg_step:.1f}s/step)")
    print(f"{'='*60}")

    result = {
        "engine": "ours_dp4a",
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
    with open("benchmark/results/ours.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
