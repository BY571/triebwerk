"""Train with a custom reward function.

Shows how to define your own reward and dataset for any task.
This example trains the model to write concise, well-structured answers.

Dry run (any machine):
    python3 examples/custom_reward.py --dry-run --max-steps 5

On Jetson:
    PYTHONPATH=engine/build2 python3 examples/custom_reward.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from triebwerk import GRPOTrainer


# ── Step 1: Define your reward function ──

def conciseness_reward(completions, **kwargs):
    """Reward shorter completions (prefer concise answers)."""
    rewards = []
    for text in completions:
        words = len(text.split())
        if words < 20:
            rewards.append(1.0)    # very concise
        elif words < 50:
            rewards.append(0.5)    # reasonable
        elif words < 100:
            rewards.append(0.0)    # too long
        else:
            rewards.append(-0.5)   # way too long
    return rewards


def quality_reward(completions, answer, **kwargs):
    """Reward completions that contain the expected answer."""
    rewards = []
    for text, ans in zip(completions, answer):
        if ans.lower() in text.lower():
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards


# ── Step 2: Build your dataset ──
# Needs "prompt" (str or message list) and "answer" (passed to reward funcs)

questions = [
    {"prompt": "What is the capital of France?", "answer": "Paris"},
    {"prompt": "What is 2 + 2?", "answer": "4"},
    {"prompt": "What color is the sky?", "answer": "blue"},
    {"prompt": "What is the largest planet?", "answer": "Jupiter"},
    {"prompt": "Who wrote Romeo and Juliet?", "answer": "Shakespeare"},
    {"prompt": "What is the boiling point of water in Celsius?", "answer": "100"},
    {"prompt": "How many days in a week?", "answer": "7"},
    {"prompt": "What is the chemical symbol for gold?", "answer": "Au"},
]

dataset = Dataset.from_list(questions)


# ── Step 3: Train ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=[quality_reward, conciseness_reward],
        loss_type="dapo",
        dry_run=args.dry_run,
    )
    trainer.train(
        dataset=dataset,
        max_steps=args.max_steps,
        num_generations=4,
        max_completion_tokens=128,
    )
