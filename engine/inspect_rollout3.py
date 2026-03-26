"""Find rollout_func call and return format."""
from trl.trainer.grpo_trainer import GRPOTrainer
import inspect

src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
lines = src.split("\n")

# Find "rollout_func" assignment or call
for i, line in enumerate(lines):
    if "self.rollout_func" in line or "_rollout_func" in line or "rollout_result" in line:
        for j in range(max(0, i-2), min(len(lines), i+20)):
            print(f"{j:4d}: {lines[j]}")
        print("===")
