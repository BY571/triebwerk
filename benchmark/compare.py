"""Compare benchmark results — reads JSON files and prints table."""
import json
import os

results_dir = os.path.join(os.path.dirname(__file__), "results")

entries = [
    ("hf_generate.json", "HF generate"),
    ("trl.json", "TRL (HF gen)"),
    ("vllm.json", "vLLM"),
    ("ours.json", "Ours (C++)"),
]

results = {}
for filename, label in entries:
    path = os.path.join(results_dir, filename)
    if os.path.exists(path):
        results[label] = json.load(open(path))

if len(results) < 2:
    print(f"Need at least 2 results. Found: {list(results.keys())}")
    print("Run bench_hf.py, bench_trl.py, and/or bench_ours.py first.")
    exit(1)

# Print header
first = next(iter(results.values()))
print("=" * 70)
print("GRPO Training Benchmark Results")
print(f"Model: {first['config']['model']}")
print(f"Config: G={first['config']['G']}, tokens={first['config']['max_tokens']}, "
      f"LoRA rank={first['config']['lora_rank']}, loss={first['config']['loss']}")
print("=" * 70)

labels = list(results.keys())
col_w = max(14, max(len(l) for l in labels) + 2)
print(f"{'':20s}" + "".join(f"{l:>{col_w}s}" for l in labels))
print(f"{'-'*20}" + f"{'-'*col_w}" * len(labels))

print(f"{'Steps':20s}" + "".join(f"{results[l]['steps']:>{col_w}d}" for l in labels))
print(f"{'Total time':20s}" + "".join(f"{results[l]['total_time_s']:>{col_w-1}.0f}s" for l in labels))
print(f"{'Avg step time':20s}" + "".join(f"{results[l]['avg_step_time_s']:>{col_w-1}.1f}s" for l in labels))
print(f"{'300-step estimate':20s}" + "".join(
    f"{results[l]['avg_step_time_s']*300/3600:>{col_w-1}.1f}h" for l in labels))

slowest = max(r["avg_step_time_s"] for r in results.values())
print(f"{'Speedup':20s}" + "".join(
    f"{slowest / results[l]['avg_step_time_s']:>{col_w-1}.1f}x" for l in labels))

print("=" * 70)

fastest_label = min(results, key=lambda l: results[l]["avg_step_time_s"])
for label in labels:
    if label != fastest_label:
        speedup = results[label]["avg_step_time_s"] / results[fastest_label]["avg_step_time_s"]
        print(f"{fastest_label} is {speedup:.1f}x faster than {label}")
