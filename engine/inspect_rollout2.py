"""Deeper inspection of rollout_func return format."""
from trl.trainer.grpo_trainer import GRPOTrainer
import inspect

# Find where rollout_func is called
src = inspect.getsource(GRPOTrainer._generate_and_score_completions)

# Print sections about rollout_func
lines = src.split("\n")
for i, line in enumerate(lines):
    if "rollout" in line.lower():
        for j in range(max(0, i-1), min(len(lines), i+15)):
            print(f"{j:4d}: {lines[j]}")
        print("---")
