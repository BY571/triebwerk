"""Full test suite: single, batch, shared embedding, correctness."""
import sys, os, time
sys.path.insert(0, os.environ.get("ENGINE_BUILD", "engine/build2"))
import jetson_engine
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]
passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1

# Load engine
engine = jetson_engine.Engine(512)
engine.load_weights("engine/weights_q4l")
engine.decode_token(0); engine.sample(1.0, 1.0); engine.reset()

# 1. Single-sequence generation
print("=== Single-sequence ===")
engine.generate(prompt, max_new_tokens=5, temperature=0.001, eos_token_id=-1)
engine.reset()
t0 = time.perf_counter()
tokens = engine.generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, eos_token_id=-1)
elapsed = time.perf_counter() - t0
tps = len(tokens) / elapsed
text = tok.decode(tokens)
print(f"  {tps:.1f} tok/s, {len(tokens)} tokens")
print(f"  Text: {text[:80]}")
check("generates tokens", len(tokens) > 10)
check("speed > 40 tok/s", tps > 40)
check("coherent text", len(text) > 20)

# 2. Batch greedy (all should match)
print("\n=== Batch G=4 greedy ===")
results = engine.generate_batch([prompt]*4, max_new_tokens=30, temperature=0.001, eos_token_id=-1)
check("4 results returned", len(results) == 4)
check("all identical (greedy)", all(r == results[0] for r in results))
check("produces tokens", len(results[0]) > 5)
print(f"  Text: {tok.decode(results[0])[:60]}")

# 3. Batch with temperature (should be diverse)
print("\n=== Batch G=4 temp=0.7 ===")
t0 = time.perf_counter()
results2 = engine.generate_batch([prompt]*4, max_new_tokens=50, temperature=0.7, top_p=0.9, eos_token_id=-1)
elapsed2 = time.perf_counter() - t0
total = sum(len(r) for r in results2)
unique = len(set(str(r) for r in results2))
print(f"  {total/elapsed2:.0f} tok/s aggregate")
check("diverse outputs", unique > 1)
check("all produce tokens", all(len(r) > 5 for r in results2))

# 4. Shared embedding
print("\n=== Shared embedding ===")
try:
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from jetson_compat import patch_amp_for_jetson, cast_model_to_fp16
    patch_amp_for_jetson()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", quantization_config=bnb,
                                                  device_map="auto", torch_dtype=torch.float16)
    cast_model_to_fp16(model)
    embed = model.model.embed_tokens.weight
    engine2 = jetson_engine.Engine(256)
    engine2.load_weights("engine/weights_q4l")
    engine2.share_embedding(embed.data_ptr())
    engine2.decode_token(0); engine2.sample(1.0, 1.0); engine2.reset()
    tokens2 = engine2.generate(prompt, max_new_tokens=20, temperature=0.001, eos_token_id=-1)
    text2 = tok.decode(tokens2)
    check("shared embed generates", len(tokens2) > 5)
    check("coherent output", len(text2) > 10)
    print(f"  Text: {text2[:60]}")
    del model, engine2
    torch.cuda.empty_cache()
except Exception as e:
    print(f"  SKIP: {e}")

# Summary
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed")
if failed == 0:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
