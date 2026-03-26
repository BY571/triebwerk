"""Debug the NF4 async crash by isolating which operation fails."""
import sys, torch, time
sys.path.insert(0, "engine/build")
import jetson_engine

engine = jetson_engine.Engine(1024)
engine.load_weights("engine/weights_nf4")

# Test: decode tokens one at a time with sync after each
print("Testing decode with sync after each...")
for i in range(20):
    engine.decode_token(i)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  decode {i}: sync error: {e}")
        break
    print(f"  decode {i}: OK")

# Test: sample with CPU
print("\nTesting CPU sample...")
tok = engine.sample(0.001, 1.0)
print(f"  CPU sample: {tok}")

# Test: sample with GPU
print("\nTesting GPU sample...")
tok2 = engine.sample_gpu(0.7, 0.9)
print(f"  GPU sample: {tok2}")

# Test: decode the sampled token and sample again
print("\nTesting decode(sampled) + sample...")
engine.decode_token(tok)
try:
    torch.cuda.synchronize()
    print("  decode(tok): sync OK")
except Exception as e:
    print(f"  decode(tok): sync error: {e}")

tok3 = engine.sample(0.001, 1.0)
print(f"  sample after decode(tok): {tok3}")

# Full loop test
print("\nFull loop test (decode + CPU sample)...")
engine.reset()
prompt = [3838, 374, 220, 17, 488, 220, 17, 30]
for t in prompt:
    engine.decode_token(t)
torch.cuda.synchronize()
print("  Prefill 8 tokens: OK")

for i in range(50):
    tok = engine.sample(0.001, 1.0)
    engine.decode_token(tok)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  Step {i}: CRASH after decode({tok}) at pos {8+i+1}: {e}")
        break
    if i % 10 == 0:
        print(f"  Step {i}: token={tok} OK")
else:
    print("  50 steps: ALL OK!")
