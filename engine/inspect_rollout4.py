"""Get the exact rollout_func return format."""
import inspect
from trl.trainer.grpo_trainer import GRPOTrainer
full_src = inspect.getsource(GRPOTrainer)
lines = full_src.split("\n")

# Print lines 1420-1480 (around rollout_func call)
for i in range(1420, min(1490, len(lines))):
    print(f"{i}: {lines[i]}")
