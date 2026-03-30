"""Train Qwen3-0.6B on letter counting with GRPO.

"How many times does the letter 'r' appear in 'strawberry'?"

This is a deceptively hard task for LLMs due to tokenization (the model
never "sees" individual characters). GRPO teaches the model to reason
step-by-step through each character. Uses a continuous reward signal
(closer = more reward) for richer gradients than binary correctness.

Based on: github.com/oumi-ai/oumi GRPO letter counting notebook

Usage:
    PYTHONPATH=engine/build_local python3 examples/letter_counting.py
    python3 examples/letter_counting.py --dry-run --max-steps 5
"""
import re
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from triebwerk import GRPOTrainer

SYSTEM_PROMPT = """Count the exact number of times a specific letter appears in a word.
Think step by step, going through each character of the word.

Respond in the following format:
<think>
Let me go through each character...
</think>
<answer>
NUMBER
</answer>"""

# Common words that are tricky for tokenization
WORDS = [
    "strawberry", "banana", "mississippi", "accommodation", "occurrence",
    "committee", "successful", "necessary", "embarrassment", "millennium",
    "possession", "recommendation", "accidentally", "environment",
    "professional", "communication", "intelligence", "approximately",
    "responsibility", "understanding", "encyclopedia", "extraordinary",
    "hippopotamus", "onomatopoeia", "supercalifragilistic", "abracadabra",
    "programming", "algorithm", "tensorflow", "transformer", "attention",
    "raspberry", "blueberry", "blackberry", "pineapple", "watermelon",
    "appreciate", "broccoli", "cappuccino", "zucchini", "mozzarella",
    "butterscotch", "bookkeeper", "ratatouille", "thoroughbred",
]


def generate_dataset(n=5000, seed=42):
    """Generate letter counting problems."""
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        word = rng.choice(WORDS)
        # Pick a letter that appears at least once
        letters_in_word = list(set(word.lower()))
        letter = rng.choice(letters_in_word)
        count = word.lower().count(letter)
        samples.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"How many times does the letter '{letter}' appear in the word '{word}'?"},
            ],
            "answer": str(count),
            "word": word,
            "letter": letter,
        })
    return Dataset.from_list(samples)


# ── Reward functions ──

def format_reward(completions, **kwargs):
    """Reward correct <think>/<answer> structure."""
    rewards = []
    for text in completions:
        score = 0.0
        if "<think>" in text: score += 0.125
        if "</think>" in text: score += 0.125
        if "<answer>" in text: score += 0.125
        if "</answer>" in text: score += 0.125
        if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", text, re.DOTALL):
            score += 0.5
        rewards.append(score)
    return rewards


def counting_reward(completions, answer, **kwargs):
    """Continuous reward: closer to correct count = higher reward.

    Exact match: 2.0, off by 1: 0.5, off by 2: -0.5, worse: -1.0.
    Unparseable: -1.5.
    """
    rewards = []
    for text, target in zip(completions, answer):
        match = re.search(r"<answer>\s*(\d+)\s*</answer>", text, re.DOTALL)
        if not match:
            rewards.append(-1.5)
            continue

        predicted = int(match.group(1))
        correct = int(target)
        delta = abs(predicted - correct)

        if delta == 0:
            rewards.append(2.0)
        elif delta == 1:
            rewards.append(0.5)
        elif delta == 2:
            rewards.append(-0.5)
        else:
            rewards.append(-1.0)

    return rewards


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-0.6B",
        reward_funcs=[format_reward, counting_reward],
        loss_type="dapo",
        dry_run=args.dry_run,
    )
    trainer.train(
        dataset=generate_dataset(),
        max_steps=args.max_steps,
        num_generations=4,
        max_completion_tokens=512,
        stop_texts=["</answer>"],
    )
