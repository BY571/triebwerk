"""Built-in reward functions and dataset helpers."""

import re
from datasets import load_dataset


# ── XML extraction helpers ──

def extract_xml_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def extract_hash_answer(text):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ── Reward functions (GSM8K defaults) ──

SYSTEM_PROMPT = """Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""


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


# ── Dataset loaders ──

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
