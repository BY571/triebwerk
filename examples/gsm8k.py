"""Train Qwen3-0.6B to solve GSM8K math problems with GRPO.

Uses granular reward functions (format, integer check, correctness) that
provide learning signal even when the model can't solve the math yet.
Based on Will Bieniek's GRPO recipe and SimpleRL-Zoo.

Usage:
    PYTHONPATH=engine/build_local python3 examples/gsm8k.py
    PYTHONPATH=engine/build_local python3 examples/gsm8k.py --max-steps 500
    python3 examples/gsm8k.py --dry-run --max-steps 5
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from triebwerk import GRPOTrainer

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


def get_gsm8k():
    data = load_dataset("openai/gsm8k", "main")["train"]
    return data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x["question"]},
        ],
        "answer": x["answer"].split("####")[1].strip(),
    })


# ── Granular reward functions ──
# Multiple small rewards > one big binary reward. The model learns
# format first (easy), then integer extraction, then correctness.

def xmlcount_reward(completions, **kwargs):
    """Granular XML tag reward (0-0.5). Each correct tag placement = 0.125."""
    rewards = []
    for text in completions:
        score = 0.0
        if text.count("<reasoning>") == 1: score += 0.125
        if text.count("</reasoning>") == 1: score += 0.125
        if text.count("<answer>") == 1: score += 0.125
        if text.count("</answer>") == 1: score += 0.125
        # Penalize trailing content after </answer>
        parts = text.split("</answer>")
        if len(parts) > 1:
            trailing = parts[-1].strip()
            score -= len(trailing) * 0.001
        rewards.append(max(0.0, score))
    return rewards


def format_reward(completions, **kwargs):
    """Full structure reward (0 or 0.5). Requires both tags in correct order."""
    rewards = []
    for text in completions:
        if re.search(r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>", text, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def integer_reward(completions, **kwargs):
    """Is the extracted answer a valid integer? (0 or 0.5)."""
    rewards = []
    for text in completions:
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            answer_text = match.group(1).strip()
            # Remove commas, dollar signs, etc.
            cleaned = re.sub(r'[,$%\s]', '', answer_text)
            try:
                int(cleaned)
                rewards.append(0.5)
            except ValueError:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards


def correctness_reward(completions, answer, **kwargs):
    """Exact match with ground truth (0 or 2.0)."""
    rewards = []
    for text, target in zip(completions, answer):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if match:
            extracted = re.sub(r'[,$%\s]', '', match.group(1).strip())
            target_clean = re.sub(r'[,$%\s]', '', target.strip())
            rewards.append(2.0 if extracted == target_clean else 0.0)
        else:
            rewards.append(0.0)
    return rewards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=[xmlcount_reward, format_reward, integer_reward, correctness_reward],
        loss_type="dapo",
        dry_run=args.dry_run,
    )
    trainer.train(
        dataset=get_gsm8k(),
        max_steps=args.max_steps,
        num_generations=4,
        max_completion_tokens=512,
        stop_texts=["</answer>"],
    )
