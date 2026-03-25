"""Plot baseline training metrics from GSM8K run on Jetson Orin."""
import json
import numpy as np
import matplotlib.pyplot as plt

with open("baseline_metrics_raw.json") as f:
    m = json.load(f)

steps = np.arange(1, len(m["reward"]) + 1)
window = 20

def smooth(x, w):
    return np.convolve(x, np.ones(w)/w, mode="valid")

pad = window - 1
reward = np.array(m["reward"])
fmt = np.array(m["format"])
cor = np.array(m["correctness"])
clipped = np.array(m["clipped"])
length = np.array(m["length"])
step_time = np.array(m["step_time"])

TITLE_BASE = "GSM8K GRPO Baseline — Qwen3-0.6B 4-bit, Jetson Orin 8GB"
STATS = (f"300 steps | 11.3h | {np.mean(step_time):.1f}s/step avg | "
         f"G=4, max_tokens=512, LoRA r=16, 4-bit NF4")


# ── Combined plot ──
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.subplots_adjust(hspace=0.08)

ax = axes[0]
ax.plot(steps[pad:], smooth(reward, window), color="#2980b9", linewidth=1.5, label=f"Combined (MA{window})")
ax.plot(steps[pad:], smooth(fmt, window), color="#2ecc71", linewidth=1.2, alpha=0.8, label=f"Format (MA{window})")
ax.plot(steps[pad:], smooth(cor, window), color="#e74c3c", linewidth=1.2, alpha=0.8, label=f"Correctness (MA{window})")
ax.scatter(steps, reward, color="#2980b9", alpha=0.1, s=8, zorder=1)
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Reward", fontsize=10)
ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax.set_title(f"{TITLE_BASE} (300 steps, 11.3h)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)

ax2 = axes[1]
ax2_r = ax2.twinx()
ax2.plot(steps[pad:], smooth(clipped, window), color="#e67e22", linewidth=1.2, label=f"Clipped ratio (MA{window})")
ax2_r.plot(steps[pad:], smooth(length, window), color="#9b59b6", linewidth=1.2, alpha=0.7, label=f"Mean length (MA{window})")
ax2.set_ylabel("Clipped ratio", fontsize=10, color="#e67e22")
ax2_r.set_ylabel("Mean completion length", fontsize=10, color="#9b59b6")
ax2.set_ylim(-0.05, 1.05)
ax2_r.set_ylim(200, 550)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_r.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.15)

ax3 = axes[2]
ax3.plot(steps, step_time, color="#34495e", linewidth=0.8, alpha=0.5)
ax3.plot(steps[pad:], smooth(step_time, window), color="#34495e", linewidth=1.5, label=f"Step time (MA{window})")
ax3.axhline(np.mean(step_time), color="#e74c3c", linewidth=1, linestyle="--", alpha=0.5,
            label=f"Mean: {np.mean(step_time):.1f}s/step")
ax3.set_ylabel("Step time (s)", fontsize=10)
ax3.set_xlabel("Step", fontsize=10)
ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax3.grid(True, alpha=0.15)

fig.text(0.5, -0.01, STATS, ha="center", fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))
plt.savefig("plot_baseline_combined.png", dpi=150, bbox_inches="tight")
print("Saved: plot_baseline_combined.png")
plt.close()


# ── Individual plot: Rewards ──
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps[pad:], smooth(reward, window), color="#2980b9", linewidth=2, label=f"Combined (MA{window})")
ax.plot(steps[pad:], smooth(fmt, window), color="#2ecc71", linewidth=1.5, alpha=0.8, label=f"Format (MA{window})")
ax.plot(steps[pad:], smooth(cor, window), color="#e74c3c", linewidth=1.5, alpha=0.8, label=f"Correctness (MA{window})")
ax.scatter(steps, reward, color="#2980b9", alpha=0.08, s=10, zorder=1)
ax.axhline(0, color="#333", linewidth=0.5, linestyle="--", alpha=0.3)
ax.set_ylabel("Reward", fontsize=11)
ax.set_xlabel("Step", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_title(f"{TITLE_BASE}\nReward Trajectory", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)
fig.text(0.5, -0.02, STATS, ha="center", fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))
plt.savefig("plot_baseline_rewards.png", dpi=150, bbox_inches="tight")
print("Saved: plot_baseline_rewards.png")
plt.close()


# ── Individual plot: Completions (clipped + length) ──
fig, ax = plt.subplots(figsize=(12, 5))
ax_r = ax.twinx()
l1, = ax.plot(steps[pad:], smooth(clipped, window), color="#e67e22", linewidth=2, label=f"Clipped ratio (MA{window})")
l2, = ax_r.plot(steps[pad:], smooth(length, window), color="#9b59b6", linewidth=2, label=f"Mean length (MA{window})")
ax.scatter(steps, clipped, color="#e67e22", alpha=0.08, s=10, zorder=1)
ax.set_ylabel("Clipped ratio", fontsize=11, color="#e67e22")
ax_r.set_ylabel("Mean completion length (tokens)", fontsize=11, color="#9b59b6")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylim(-0.05, 1.05)
ax_r.set_ylim(200, 550)
ax.axhline(0.5, color="#e67e22", linewidth=0.5, linestyle=":", alpha=0.3)
ax.legend([l1, l2], [l1.get_label(), l2.get_label()], fontsize=9, framealpha=0.9, loc="upper right")
ax.set_title(f"{TITLE_BASE}\nCompletion Length & Truncation", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)
fig.text(0.5, -0.02, STATS, ha="center", fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))
plt.savefig("plot_baseline_completions.png", dpi=150, bbox_inches="tight")
print("Saved: plot_baseline_completions.png")
plt.close()


# ── Individual plot: Step time ──
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(steps, step_time, color="#34495e", linewidth=0.8, alpha=0.4)
ax.plot(steps[pad:], smooth(step_time, window), color="#34495e", linewidth=2, label=f"Step time (MA{window})")
ax.axhline(np.mean(step_time), color="#e74c3c", linewidth=1.5, linestyle="--", alpha=0.6,
           label=f"Mean: {np.mean(step_time):.1f}s/step")
ax.fill_between(steps, step_time, alpha=0.05, color="#34495e")
ax.set_ylabel("Step time (seconds)", fontsize=11)
ax.set_xlabel("Step", fontsize=11)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_title(f"{TITLE_BASE}\nStep Time (target for optimization)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.15)
fig.text(0.5, -0.02, STATS, ha="center", fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f8f8", edgecolor="#ccc"))
plt.savefig("plot_baseline_steptime.png", dpi=150, bbox_inches="tight")
print("Saved: plot_baseline_steptime.png")
plt.close()
