"""Train Qwen3-0.6B to solve GSM8K math problems with GRPO.

This is the standard benchmark: the model learns to output structured
reasoning in <reasoning>...</reasoning><answer>...</answer> format.

On Jetson (with C++ engine):
    PYTHONPATH=engine/build2 python3 examples/gsm8k.py

Dry run (any machine, HF generate):
    python3 examples/gsm8k.py --dry-run --max-steps 5
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from train import grpo_train

# ── Dataset ──

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


# ── Reward functions ──

def correctness_reward(completions, answer, **kwargs):
    """2.0 if extracted answer matches, -1.0 otherwise."""
    rewards = []
    for completion, ans in zip(completions, answer):
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        extracted = match.group(1).strip() if match else ""
        rewards.append(2.0 if extracted == ans else -1.0)
    return rewards


def format_reward(completions, **kwargs):
    """1.0 for correct XML format, 0.5 for answer only, 0.0 otherwise."""
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


# ── Train ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    grpo_train(
        dataset=get_gsm8k(),
        reward_funcs=[format_reward, correctness_reward],
        model="Qwen/Qwen3-0.6B",
        max_steps=args.max_steps,
        num_generations=4,
        max_completion_tokens=512,
        stop_texts=["</answer>"],
        dry_run=args.dry_run,
    )
