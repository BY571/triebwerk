"""Benchmark NF4 weights vs fp16 weights."""
import sys, os, time, torch
sys.path.insert(0, "engine/build")
import jetson_engine
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]

for weights_path, label in [("engine/weights", "fp16 (1.2GB)"), ("engine/weights_nf4", "NF4 (568MB)")]:
    idx_path = weights_path + ".idx"
    if not os.path.exists(idx_path):
        print(f"{label}: SKIP (no index file)")
        continue

    engine = jetson_engine.Engine(1024)
    engine.load_weights(weights_path)

    # Warmup
    engine.decode_token(0)
    engine.sample(1.0, 1.0)
    engine.reset()

    # Generate
    t0 = time.perf_counter()
    tokens = engine.generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=-1)
    elapsed = time.perf_counter() - t0

    tps = len(tokens) / elapsed
    text = tok.decode(tokens)[:80]
    print(f"{label}: {tps:.1f} tok/s ({elapsed/len(tokens)*1000:.1f} ms/tok)")
    print(f"  Text: {text}")

    del engine
    torch.cuda.empty_cache()
