"""Train Qwen3-0.6B on the Countdown Numbers Game with GRPO.

Given 3-4 numbers and a target, construct an arithmetic expression that
equals the target. Each number used at most once. This is the recommended
task for validating GRPO on small models (0.5-1B).

Based on: github.com/philschmid/deep-learning-pytorch-huggingface

Usage:
    PYTHONPATH=engine/build_local python3 examples/countdown.py
    PYTHONPATH=engine/build_local python3 examples/countdown.py --max-steps 500
    python3 examples/countdown.py --dry-run --max-steps 5
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from triebwerk import GRPOTrainer

SYSTEM_PROMPT = """You are given a list of numbers and a target number.
Your task is to find an arithmetic expression using the given numbers that equals the target.
Each number can be used at most once. You can use +, -, *, and /.

Respond in the following format:
<think>
...
</think>
<answer>
EXPRESSION = TARGET
</answer>"""


def get_countdown():
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4")["train"]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Using the numbers {x['nums']}, create an expression that equals {x['target']}."},
        ],
        "answer": str(x["target"]),
        "nums": x["nums"],
        "target": x["target"],
    })
    return data


# ── Reward functions ──

def format_reward(completions, **kwargs):
    """Reward correct <think>/<answer> structure (0.0 to 1.0)."""
    rewards = []
    for text in completions:
        score = 0.0
        if "<think>" in text: score += 0.125
        if "</think>" in text: score += 0.125
        if "<answer>" in text: score += 0.125
        if "</answer>" in text: score += 0.125
        # Full structure bonus
        if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", text, re.DOTALL):
            score += 0.5
        rewards.append(score)
    return rewards


def equation_reward(completions, nums, target, **kwargs):
    """Verify the arithmetic equation is correct (0.0 or 2.0).

    Checks: (1) expression evaluates to target, (2) only uses given numbers,
    (3) each number used at most once.
    """
    rewards = []
    for text, available_nums, tgt in zip(completions, nums, target):
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        answer_text = match.group(1).strip()
        # Extract the expression (before '=' if present)
        expr = answer_text.split("=")[0].strip() if "=" in answer_text else answer_text

        try:
            # Evaluate the expression
            # Only allow digits, operators, spaces, and parentheses
            if not re.match(r'^[\d\s\+\-\*/\(\)\.]+$', expr):
                rewards.append(0.0)
                continue

            result = eval(expr)
            if abs(result - int(tgt)) < 1e-6:
                # Check that only available numbers are used
                used_nums = [int(n) for n in re.findall(r'\d+', expr)]
                avail = list(available_nums) if not isinstance(available_nums, list) else available_nums
                avail = [int(n) for n in avail]
                valid = True
                avail_copy = avail.copy()
                for n in used_nums:
                    if n in avail_copy:
                        avail_copy.remove(n)
                    else:
                        valid = False
                        break
                rewards.append(2.0 if valid else 0.5)
            else:
                # Partial credit: correct format but wrong answer
                rewards.append(0.0)
        except Exception:
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
        reward_funcs=[format_reward, equation_reward],
        loss_type="dapo",
        dry_run=args.dry_run,
    )
    trainer.train(
        dataset=get_countdown(),
        max_steps=args.max_steps,
        num_generations=4,
        max_completion_tokens=512,
        stop_texts=["</answer>"],
    )
