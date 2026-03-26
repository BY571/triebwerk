"""Side-by-side benchmark: our GRPO trainer vs TRL baseline.

Runs both setups with identical hyperparameters on the same hardware,
collects metrics, and generates a comparison report with plots.

Usage:
    # Full benchmark (300 steps each, ~2h on RTX 4060)
    python3 benchmark/compare_trl.py --steps 300

    # Quick sanity check (10 steps each)
    python3 benchmark/compare_trl.py --steps 10

    # Engine-only (skip TRL baseline, just run ours)
    python3 benchmark/compare_trl.py --steps 300 --engine-only

    # Analyze existing results (no training, just plot)
    python3 benchmark/compare_trl.py --analyze-only --results-dir benchmark/results/20260326

    # On Jetson with C++ engine:
    PYTHONPATH=engine/build2 python3 benchmark/compare_trl.py --steps 300

    # On laptop with local build:
    PYTHONPATH=engine/build_local python3 benchmark/compare_trl.py --steps 300
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import torch
import numpy as np


def get_device_info():
    """Collect hardware info for the report."""
    info = {"timestamp": datetime.now().isoformat()}

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["gpu_memory_mb"] = getattr(props, "total_memory", getattr(props, "total_mem", 0)) // (1024 * 1024)
        info["cuda_version"] = torch.version.cuda
        info["compute_capability"] = ".".join(str(x) for x in torch.cuda.get_device_capability(0))
    else:
        info["gpu_name"] = "CPU only"

    info["torch_version"] = torch.__version__
    info["python_version"] = sys.version.split()[0]

    return info


def run_engine_benchmark(steps, output_dir, dry_run=False):
    """Run our standalone GRPO trainer."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Engine trainer (train.py)")
    print("=" * 60)

    cmd = [
        sys.executable, "train.py",
        "--max-steps", str(steps),
        "--num-generations", "4",
        "--max-completion-tokens", "512",
        "--lr", "5e-6",
        "--lora-rank", "16",
        "--loss-type", "dapo",
        "--logging-steps", "1",
        "--save-steps", str(max(steps, 100)),  # save only at end
        "--output-dir", output_dir,
    ]
    if dry_run:
        cmd.append("--dry-run")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False, env=os.environ)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: Engine trainer failed (exit code {result.returncode})")
        return None

    # Find the run directory (most recent subdirectory)
    runs = sorted(os.listdir(output_dir))
    if not runs:
        return None
    run_dir = os.path.join(output_dir, runs[-1])

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path) as f:
        metrics = json.load(f)

    return {
        "name": "engine",
        "label": "Ours (C++ engine)" if not dry_run else "Ours (HF generate, dry-run)",
        "wall_time": elapsed,
        "metrics": metrics,
        "run_dir": run_dir,
    }


def run_trl_benchmark(steps, output_dir):
    """Run TRL GRPOTrainer baseline."""
    print("\n" + "=" * 60)
    print("BENCHMARK: TRL baseline (train_gsm8k.py)")
    print("=" * 60)

    cmd = [
        sys.executable, "train_gsm8k.py",
        "--max-steps", str(steps),
        "--num-generations", "4",
        "--max-completion-tokens", "512",
        "--lr", "5e-6",
        "--lora-rank", "16",
        "--logging-steps", "1",
        "--save-steps", str(max(steps, 100)),
        "--output-dir", output_dir,
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"ERROR: TRL baseline failed (exit code {result.returncode})")
        return None

    runs = sorted(os.listdir(output_dir))
    if not runs:
        return None
    run_dir = os.path.join(output_dir, runs[-1])

    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path) as f:
        raw = json.load(f)

    # TRL metrics format: {"total_time_s": ..., "steps": [...]}
    # Each step has trainer log keys like "loss", "reward/...", etc.
    if isinstance(raw, dict) and "steps" in raw:
        metrics = raw["steps"]
    else:
        metrics = raw

    return {
        "name": "trl",
        "label": "TRL + HF generate",
        "wall_time": elapsed,
        "metrics": metrics,
        "run_dir": run_dir,
    }


def extract_metrics(result):
    """Extract comparable metrics from either engine or TRL results."""
    metrics = result["metrics"]
    if not metrics:
        return {}

    # Engine format: list of dicts with step, loss, mean_reward, gen_tok_per_s, step_time, gen_time
    # TRL format: list of dicts with step, loss, reward/format_reward, reward/correctness_reward, etc.

    steps = []
    rewards = []
    step_times = []
    tok_per_s = []
    gen_times = []

    for m in metrics:
        step = m.get("step", 0)
        steps.append(step)

        # Reward: engine uses mean_reward, TRL uses reward/* keys
        if "mean_reward" in m:
            rewards.append(m["mean_reward"])
        elif "reward" in m:
            rewards.append(m["reward"])
        else:
            # Sum TRL reward keys
            r = sum(v for k, v in m.items() if k.startswith("reward/") and isinstance(v, (int, float)))
            rewards.append(r if r != 0 else None)

        # Step time
        if "step_time" in m:
            step_times.append(m["step_time"])
        elif "wall_time" in m and len(steps) >= 2:
            # Compute from wall_time differences
            pass

        # Generation speed
        if "gen_tok_per_s" in m:
            tok_per_s.append(m["gen_tok_per_s"])

        if "gen_time" in m:
            gen_times.append(m["gen_time"])

    return {
        "steps": steps,
        "rewards": [r for r in rewards if r is not None],
        "step_times": step_times,
        "tok_per_s": tok_per_s,
        "gen_times": gen_times,
    }


def compute_summary(result, extracted):
    """Compute summary statistics."""
    s = {
        "name": result["label"],
        "total_wall_time_s": result["wall_time"],
        "total_steps": len(extracted["steps"]),
    }

    if extracted["step_times"]:
        s["avg_step_time_s"] = np.mean(extracted["step_times"])
        s["median_step_time_s"] = np.median(extracted["step_times"])
        s["p95_step_time_s"] = np.percentile(extracted["step_times"], 95)

    if extracted["tok_per_s"]:
        s["avg_tok_per_s"] = np.mean(extracted["tok_per_s"])
        s["median_tok_per_s"] = np.median(extracted["tok_per_s"])

    if extracted["gen_times"]:
        s["avg_gen_time_s"] = np.mean(extracted["gen_times"])

    if extracted["rewards"]:
        s["final_reward"] = extracted["rewards"][-1]
        # Rolling average of last 10%
        window = max(1, len(extracted["rewards"]) // 10)
        s["final_reward_avg"] = np.mean(extracted["rewards"][-window:])
        s["max_reward"] = max(extracted["rewards"])

    # Peak GPU memory
    if torch.cuda.is_available():
        s["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated() // (1024 * 1024)

    return s


def generate_report(results, device_info, output_dir):
    """Generate markdown comparison report with summary table."""
    lines = ["# GRPO Benchmark Report\n"]
    lines.append(f"**Date**: {device_info['timestamp'][:10]}\n")
    lines.append(f"**GPU**: {device_info.get('gpu_name', 'N/A')} "
                 f"({device_info.get('gpu_memory_mb', '?')} MB)\n")
    lines.append(f"**CUDA**: {device_info.get('cuda_version', 'N/A')}, "
                 f"**PyTorch**: {device_info.get('torch_version', 'N/A')}\n")

    # Summary table
    summaries = []
    for r in results:
        ext = extract_metrics(r)
        s = compute_summary(r, ext)
        summaries.append(s)

    lines.append("\n## Summary\n")
    lines.append("| Metric | " + " | ".join(s["name"] for s in summaries) + " |")
    lines.append("|---|" + "|".join(["---"] * len(summaries)) + "|")

    def row(label, key, fmt=".1f"):
        vals = []
        for s in summaries:
            v = s.get(key)
            vals.append(f"{v:{fmt}}" if v is not None else "N/A")
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    row("Total wall time (s)", "total_wall_time_s", ".0f")
    row("Steps", "total_steps", ".0f")
    row("Avg step time (s)", "avg_step_time_s")
    row("Median step time (s)", "median_step_time_s")
    row("Avg tok/s (generation)", "avg_tok_per_s", ".0f")
    row("Avg generation time (s)", "avg_gen_time_s")
    row("Final reward (last 10% avg)", "final_reward_avg", ".2f")
    row("Max reward seen", "max_reward", ".2f")

    # Speedup
    if len(summaries) == 2:
        lines.append("\n## Speedup\n")
        s0, s1 = summaries[0], summaries[1]  # engine, trl
        if s0.get("avg_step_time_s") and s1.get("avg_step_time_s"):
            speedup = s1["avg_step_time_s"] / s0["avg_step_time_s"]
            lines.append(f"- **Step time**: {speedup:.1f}x faster")
        if s0.get("avg_tok_per_s") and s1.get("avg_tok_per_s"):
            speedup = s0["avg_tok_per_s"] / s1["avg_tok_per_s"]
            lines.append(f"- **Generation**: {speedup:.1f}x faster")

    # Config
    lines.append("\n## Hyperparameters (identical for both)\n")
    lines.append("```")
    lines.append("model: Qwen/Qwen3-0.6B (4-bit NF4)")
    lines.append("lora_rank: 16")
    lines.append("num_generations: 4")
    lines.append("max_completion_tokens: 512")
    lines.append("learning_rate: 5e-6")
    lines.append("loss_type: dapo")
    lines.append("warmup_ratio: 0.1")
    lines.append("gradient_checkpointing: true")
    lines.append("```")

    report = "\n".join(lines)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")
    return report


def generate_plots(results, output_dir):
    """Generate comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    extracted = {r["name"]: extract_metrics(r) for r in results}
    colors = {"engine": "#2196F3", "trl": "#FF5722"}
    labels = {r["name"]: r["label"] for r in results}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GRPO Training: Engine vs TRL Baseline", fontsize=14, fontweight="bold")

    # 1. Reward curve
    ax = axes[0, 0]
    for name, ext in extracted.items():
        if ext["rewards"]:
            steps = ext["steps"][:len(ext["rewards"])]
            # Smooth with rolling average
            window = max(1, len(ext["rewards"]) // 20)
            smoothed = np.convolve(ext["rewards"], np.ones(window)/window, mode="valid")
            ax.plot(steps[:len(smoothed)], smoothed, color=colors.get(name, "gray"),
                    label=labels[name], linewidth=2)
            # Raw as transparent
            ax.plot(steps, ext["rewards"], color=colors.get(name, "gray"), alpha=0.15)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Step time
    ax = axes[0, 1]
    for name, ext in extracted.items():
        if ext["step_times"]:
            ax.plot(ext["steps"][:len(ext["step_times"])], ext["step_times"],
                    color=colors.get(name, "gray"), label=labels[name], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (s)")
    ax.set_title("Step Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Generation tok/s
    ax = axes[1, 0]
    for name, ext in extracted.items():
        if ext["tok_per_s"]:
            ax.plot(ext["steps"][:len(ext["tok_per_s"])], ext["tok_per_s"],
                    color=colors.get(name, "gray"), label=labels[name], linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens/sec")
    ax.set_title("Generation Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary bar chart
    ax = axes[1, 1]
    summaries = []
    for r in results:
        ext = extract_metrics(r)
        summaries.append(compute_summary(r, ext))

    bar_metrics = ["avg_step_time_s", "avg_tok_per_s"]
    bar_labels = ["Avg step time (s)", "Avg tok/s"]
    x = np.arange(len(bar_labels))
    width = 0.35

    for i, s in enumerate(summaries):
        vals = [s.get(m, 0) or 0 for m in bar_metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=s["name"],
                      color=colors.get(results[i]["name"], "gray"))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_title("Performance Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "comparison.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


def analyze_existing(results_dir):
    """Load and analyze existing benchmark results."""
    results = []
    for name in ["engine", "trl"]:
        # Find the metrics file
        subdir = os.path.join(results_dir, name)
        if not os.path.isdir(subdir):
            continue
        runs = sorted(os.listdir(subdir))
        if not runs:
            continue
        run_dir = os.path.join(subdir, runs[-1])
        metrics_path = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        with open(metrics_path) as f:
            raw = json.load(f)

        if isinstance(raw, dict) and "steps" in raw:
            metrics = raw["steps"]
            wall_time = raw.get("total_time_s", 0)
        else:
            metrics = raw
            wall_time = sum(m.get("step_time", 0) for m in metrics)

        label = "Ours (C++ engine)" if name == "engine" else "TRL + HF generate"
        results.append({
            "name": name,
            "label": label,
            "wall_time": wall_time,
            "metrics": metrics,
            "run_dir": run_dir,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="GRPO benchmark: engine vs TRL")
    parser.add_argument("--steps", type=int, default=300,
                        help="Training steps for each setup")
    parser.add_argument("--engine-only", action="store_true",
                        help="Only run engine trainer (skip TRL)")
    parser.add_argument("--trl-only", action="store_true",
                        help="Only run TRL baseline (skip engine)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Engine uses HF generate instead of C++ engine")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only analyze existing results, no training")
    parser.add_argument("--results-dir", default=None,
                        help="Directory with existing results (for --analyze-only)")
    parser.add_argument("--output-dir", default="benchmark/results",
                        help="Output directory for results")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)

    if args.analyze_only:
        if not args.results_dir:
            # Use most recent
            base = args.output_dir
            if os.path.isdir(base):
                runs = sorted(os.listdir(base))
                if runs:
                    args.results_dir = os.path.join(base, runs[-1])
        if not args.results_dir or not os.path.isdir(args.results_dir):
            print("No results found. Run a benchmark first.")
            return
        output_dir = args.results_dir
        results = analyze_existing(output_dir)
        if not results:
            print("No metrics found in", output_dir)
            return
    else:
        os.makedirs(output_dir, exist_ok=True)
        results = []

        # Run engine benchmark
        if not args.trl_only:
            engine_dir = os.path.join(output_dir, "engine")
            r = run_engine_benchmark(args.steps, engine_dir, dry_run=args.dry_run)
            if r:
                results.append(r)

        # Run TRL baseline
        if not args.engine_only:
            trl_dir = os.path.join(output_dir, "trl")
            r = run_trl_benchmark(args.steps, trl_dir)
            if r:
                results.append(r)

    if not results:
        print("No results to analyze.")
        return

    # Collect device info and generate report
    device_info = get_device_info()
    with open(os.path.join(output_dir, "device_info.json"), "w") as f:
        json.dump(device_info, f, indent=2)

    report = generate_report(results, device_info, output_dir)
    print("\n" + report)

    generate_plots(results, output_dir)

    # Save raw summaries
    summaries = []
    for r in results:
        ext = extract_metrics(r)
        summaries.append(compute_summary(r, ext))
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
