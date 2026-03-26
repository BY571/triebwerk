"""Inspect TRL's rollout_func interface."""
from trl.trainer.grpo_trainer import GRPOTrainer
import inspect

# Check rollout_func type
sig = inspect.signature(GRPOTrainer.__init__)
for name, p in sig.parameters.items():
    if name == "rollout_func":
        print(f"rollout_func: annotation={p.annotation}, default={p.default}")

# Look at how rollout_func is called in the trainer
src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
for i, line in enumerate(src.split("\n")):
    if "rollout_func" in line or "self._rollout" in line:
        # Print context around the line
        lines = src.split("\n")
        for j in range(max(0, i-2), min(len(lines), i+5)):
            print(f"  {j}: {lines[j]}")
        print()

# Check what RolloutFunc type is
print("\n=== RolloutFunc type ===")
try:
    from trl.trainer.grpo_trainer import RolloutFunc
    print(f"RolloutFunc: {RolloutFunc}")
    print(f"Signature: {inspect.signature(RolloutFunc)}")
except ImportError:
    # Check the type alias
    import trl.trainer.grpo_trainer as mod
    for name in dir(mod):
        if "ollout" in name:
            print(f"  {name}: {getattr(mod, name)}")
