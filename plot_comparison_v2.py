"""Head-to-head: baseline (TRL) vs CUDA graph v2 (fixed GRPO)."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("baseline_metrics_raw.json") as f:
    baseline = json.load(f)

with open("fast_v2_metrics.json") as f:
    fast_raw = json.load(f)
fast = {
    "reward": [s["mean_reward"] for s in fast_raw["steps"]],
    "format": [s["mean_format"] for s in fast_raw["steps"]],
    "correctness": [s["mean_correctness"] for s in fast_raw["steps"]],
    "step_time": [s["total_time"] for s in fast_raw["steps"]],
}

steps = np.arange(1, 301)
window = 20

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode="valid")

pad = window - 1

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Baseline (TRL, 11.3h) vs CUDA Graph v2 (3.7h) -- Fixed GRPO",
             fontsize=14, fontweight="bold")

# Combined reward
ax = axes[0, 0]
ax.plot(steps[pad:], smooth(baseline["reward"], window), color="#e67e22", linewidth=1.5, label="Baseline (MA20)")
ax.plot(steps[pad:], smooth(fast["reward"], window), color="#2980b9", linewidth=1.5, label="CUDA Graph v2 (MA20)")
ax.scatter(steps, baseline["reward"], color="#e67e22", alpha=0.05, s=5)
ax.scatter(steps, fast["reward"], color="#2980b9", alpha=0.05, s=5)
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Combined Reward")
ax.set_title("Combined Reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# Format reward
ax = axes[0, 1]
ax.plot(steps[pad:], smooth(baseline["format"], window), color="#e67e22", linewidth=1.5, label="Baseline")
ax.plot(steps[pad:], smooth(fast["format"], window), color="#2980b9", linewidth=1.5, label="CUDA Graph v2")
ax.set_ylabel("Format Reward")
ax.set_title("Format Reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# Correctness reward
ax = axes[1, 0]
ax.plot(steps[pad:], smooth(baseline["correctness"], window), color="#e67e22", linewidth=1.5, label="Baseline")
ax.plot(steps[pad:], smooth(fast["correctness"], window), color="#2980b9", linewidth=1.5, label="CUDA Graph v2")
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Correctness Reward")
ax.set_xlabel("Step")
ax.set_title("Correctness Reward")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

# Step time
ax = axes[1, 1]
ax.plot(steps[pad:], smooth(baseline["step_time"], window), color="#e67e22", linewidth=1.5,
        label=f"Baseline: {np.mean(baseline['step_time']):.0f}s avg")
ax.plot(steps[pad:], smooth(fast["step_time"], window), color="#2980b9", linewidth=1.5,
        label=f"CUDA Graph: {np.mean(fast['step_time']):.0f}s avg")
ax.set_ylabel("Step Time (s)")
ax.set_xlabel("Step")
ax.set_title("Step Time (lower is better)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.15)

stats = (
    f"Baseline: 11.3h, 135.6s/step, reward +0.44 (last 50)  |  "
    f"CUDA Graph v2: 3.7h, 44.1s/step, reward +0.56 (last 50)  |  "
    f"3.1x faster, better reward"
)
fig.text(0.5, -0.01, stats, ha="center", fontsize=9, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))

plt.tight_layout()
plt.savefig("plot_comparison_v2.png", dpi=150, bbox_inches="tight")
print("Saved: plot_comparison_v2.png")
