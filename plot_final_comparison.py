"""Final comparison: Baseline (TRL) vs CUDA Graph vs C++ Engine — all 300 steps."""
import json
import numpy as np
import matplotlib.pyplot as plt

# Load all metrics
with open("baseline_metrics_raw.json") as f:
    baseline = json.load(f)
with open("fast_v2_metrics.json") as f:
    cuda_graph = json.load(f)
with open("engine_metrics.json") as f:
    engine_raw = json.load(f)

engine = {
    "reward": [s["mean_reward"] for s in engine_raw["steps"]],
    "format": [s["mean_format"] for s in engine_raw["steps"]],
    "correctness": [s["mean_correctness"] for s in engine_raw["steps"]],
    "step_time": [s["total_time"] for s in engine_raw["steps"]],
}
cg = {
    "reward": [s["mean_reward"] for s in cuda_graph["steps"]],
    "format": [s["mean_format"] for s in cuda_graph["steps"]],
    "correctness": [s["mean_correctness"] for s in cuda_graph["steps"]],
    "step_time": [s["total_time"] for s in cuda_graph["steps"]],
}

steps = np.arange(1, 301)
window = 20
pad = window - 1

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode="valid")

colors = {"baseline": "#e67e22", "cuda_graph": "#2ecc71", "engine": "#2980b9"}

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("GSM8K GRPO — Three Approaches on Jetson Orin 8GB\n"
             "Baseline (TRL, 11.3h) vs CUDA Graph (3.5h) vs C++ Engine (3.5h)",
             fontsize=13, fontweight="bold")

# Combined reward
ax = axes[0, 0]
ax.plot(steps[pad:], smooth(baseline["reward"], window), color=colors["baseline"], linewidth=1.5, label="Baseline (TRL)")
ax.plot(steps[pad:], smooth(cg["reward"], window), color=colors["cuda_graph"], linewidth=1.5, label="CUDA Graph")
ax.plot(steps[pad:], smooth(engine["reward"], window), color=colors["engine"], linewidth=1.5, label="C++ Engine")
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Combined Reward")
ax.set_title("Combined Reward (higher = better)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# Correctness
ax = axes[0, 1]
ax.plot(steps[pad:], smooth(baseline["correctness"], window), color=colors["baseline"], linewidth=1.5, label="Baseline")
ax.plot(steps[pad:], smooth(cg["correctness"], window), color=colors["cuda_graph"], linewidth=1.5, label="CUDA Graph")
ax.plot(steps[pad:], smooth(engine["correctness"], window), color=colors["engine"], linewidth=1.5, label="C++ Engine")
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Correctness Reward")
ax.set_title("Correctness (math answers)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# Format
ax = axes[1, 0]
ax.plot(steps[pad:], smooth(baseline["format"], window), color=colors["baseline"], linewidth=1.5, label="Baseline")
ax.plot(steps[pad:], smooth(cg["format"], window), color=colors["cuda_graph"], linewidth=1.5, label="CUDA Graph")
ax.plot(steps[pad:], smooth(engine["format"], window), color=colors["engine"], linewidth=1.5, label="C++ Engine")
ax.set_ylabel("Format Reward")
ax.set_xlabel("Step")
ax.set_title("Format (XML structure)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

# Step time
ax = axes[1, 1]
ax.plot(steps[pad:], smooth(baseline["step_time"], window), color=colors["baseline"], linewidth=1.5,
        label=f"Baseline: {np.mean(baseline['step_time']):.0f}s/step")
ax.plot(steps[pad:], smooth(cg["step_time"], window), color=colors["cuda_graph"], linewidth=1.5,
        label=f"CUDA Graph: {np.mean(cg['step_time']):.0f}s/step")
ax.plot(steps[pad:], smooth(engine["step_time"], window), color=colors["engine"], linewidth=1.5,
        label=f"C++ Engine: {np.mean(engine['step_time']):.0f}s/step")
ax.set_ylabel("Step Time (s)")
ax.set_xlabel("Step")
ax.set_title("Step Time (lower = faster)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.15)

stats = (
    f"Baseline (TRL):  11.3h, 135.6s/step, reward +0.44  |  "
    f"CUDA Graph:  3.5h, 44.1s/step, reward +0.56  |  "
    f"C++ Engine:  3.5h, 41.5s/step, reward +0.22"
)
fig.text(0.5, -0.01, stats, ha="center", fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))

plt.tight_layout()
plt.savefig("plot_final_comparison.png", dpi=150, bbox_inches="tight")
print("Saved: plot_final_comparison.png")
