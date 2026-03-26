"""Convert HuggingFace model weights to flat binary for the C++ engine.

Two modes:
  --mode fp16: dequantize everything to fp16 (1.2GB, for correctness testing)
  --mode nf4:  keep MLP as NF4, attention as fp16 (saves ~900MB)

NF4 mode saves per MLP layer:
  .weight          (uint8 packed, 2 values per byte)
  .absmax          (float32, dequantized from double-quant)
  .quant_map       (float32, 16 entries)
  Block size = 64, shape = (out_dim, in_dim)

Usage:
    python3 engine/convert_weights.py --model unsloth/Qwen3-0.6B-unsloth-bnb-4bit --output engine/weights --mode nf4
"""
import argparse
import json
import os

import torch
import numpy as np
import safetensors.torch as st
from huggingface_hub import snapshot_download


NF4_TABLE = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=np.float32)


def quantize_fp16_to_q4l(weight_fp16, block_size=64):
    """Quantize fp16 weight to linear Q4 format with dp4a-friendly packing.
    Dequant at inference: (nibble - 8) * scale.

    Packing (Q4_0-style for dp4a compatibility):
      For each group of 8 elements [e0..e7]:
        byte[0] = e0 | (e4 << 4)
        byte[1] = e1 | (e5 << 4)
        byte[2] = e2 | (e6 << 4)
        byte[3] = e3 | (e7 << 4)

      dp4a extraction:
        (packed >> 0) & 0x0F0F0F0F = [e0, e1, e2, e3]  (low nibbles)
        (packed >> 4) & 0x0F0F0F0F = [e4, e5, e6, e7]  (high nibbles)
      Both sequential, pairing correctly with q8[0:3] and q8[4:7].

    Returns (packed_uint8, scales_float32)."""
    w = weight_fp16.astype(np.float32).flatten()
    n = len(w)
    assert n % block_size == 0
    n_blocks = n // block_size

    blocks = w.reshape(n_blocks, block_size)
    maxabs = np.max(np.abs(blocks), axis=1)
    scales = (maxabs / 7.5).astype(np.float32)
    scales = np.maximum(scales, 1e-10)

    normalized = blocks / scales[:, None]
    nibbles = np.clip(np.round(normalized + 8.0), 0, 15).astype(np.uint8)

    # dp4a-friendly packing: for each group of 8 elements, interleave
    # first-4 into low nibbles and second-4 into high nibbles
    flat = nibbles.flatten()
    n_groups = n // 8
    groups = flat.reshape(n_groups, 8)
    # groups[:,0:4] = first 4 elements (go to low nibbles)
    # groups[:,4:8] = next 4 elements (go to high nibbles)
    packed = np.zeros(n // 2, dtype=np.uint8)
    for i in range(4):
        packed[i::4] = groups[:, i] | (groups[:, i + 4] << 4)

    return packed, scales


def quantize_fp16_to_nf4(weight_fp16, block_size=64):
    """Quantize fp16 weight matrix to NF4 format.
    Returns (packed_uint8, absmax_float32, quant_map_float32)."""
    w = weight_fp16.astype(np.float32).flatten()
    n = len(w)
    assert n % block_size == 0, f"Weight size {n} not divisible by block_size {block_size}"
    n_blocks = n // block_size

    # Compute per-block absmax
    blocks = w.reshape(n_blocks, block_size)
    absmax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    absmax = np.maximum(absmax, 1e-10)  # avoid division by zero

    # Normalize and find nearest NF4 level
    normalized = blocks / absmax[:, None]  # range [-1, 1]
    # For each element, find the NF4 index that minimizes |normalized - NF4_TABLE[idx]|
    # Broadcast: (n_blocks, block_size, 1) vs (1, 1, 16)
    diffs = np.abs(normalized[:, :, None] - NF4_TABLE[None, None, :])
    indices = np.argmin(diffs, axis=2).astype(np.uint8)  # (n_blocks, block_size)

    # Pack pairs into bytes: hi nibble first
    flat_indices = indices.flatten()
    n_bytes = n // 2
    packed = np.zeros(n_bytes, dtype=np.uint8)
    packed = (flat_indices[0::2] << 4) | flat_indices[1::2]

    return packed, absmax, NF4_TABLE.copy()


def convert_nf4(args):
    """Save NF4 MLP weights as-is, dequant absmax to float32, attention as fp16."""
    print(f"Converting {args.model} (NF4 mode)...")
    model_dir = snapshot_download(args.model)

    tensors = {}
    for fname in os.listdir(model_dir):
        if fname.endswith(".safetensors"):
            for name, t in st.load_file(os.path.join(model_dir, fname)).items():
                clean = name[6:] if name.startswith("model.") else name
                tensors[clean] = t

    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name in sorted(tensors.keys()):
            t = tensors[name]

            # Skip quant_state (we extract offset from it)
            if "quant_state" in name:
                continue
            # Skip nested quant map (we dequant absmax ourselves)
            if "nested_quant_map" in name or "nested_absmax" in name:
                continue

            # For absmax: dequantize from uint8 to float32 using nested quant info
            if name.endswith(".weight.absmax"):
                base = name.replace(".weight.absmax", ".weight")
                nested_absmax = tensors.get(base + ".nested_absmax")
                nested_qmap = tensors.get(base + ".nested_quant_map")
                qs_raw = tensors.get(base + ".quant_state.bitsandbytes__nf4")

                nested_offset = 0.0
                if qs_raw is not None:
                    qs_json = bytes(qs_raw.numpy().tolist()).decode("utf-8")
                    nested_offset = json.loads(qs_json).get("nested_offset", 0.0)

                absmax_u8 = t.numpy()
                na = nested_absmax.numpy() if nested_absmax is not None else np.zeros(1)
                nq = nested_qmap.numpy() if nested_qmap is not None else np.zeros(256)

                # Dequantize: val = nq[absmax_u8[i]] * na[i // 256] + offset
                n_blocks = len(absmax_u8)
                absmax_f32 = np.zeros(n_blocks, dtype=np.float32)
                for i in range(n_blocks):
                    group = i // 256
                    absmax_f32[i] = nq[absmax_u8[i]] * na[group] + nested_offset

                data = absmax_f32.tobytes()
                dtype = "float32"
                shape = f"{n_blocks}"
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # For NF4 packed weights: save as uint8
            if t.dtype == torch.uint8 and t.numel() > 10000:
                data = t.numpy().flatten().tobytes()
                dtype = "uint8"
                shape = ",".join(str(s) for s in t.shape)
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # For quant_map (16 float32 entries): save as-is
            if "quant_map" in name:
                data = t.numpy().tobytes()
                dtype = "float32"
                shape = str(t.shape[0])
                lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
                f.write(data)
                offset += len(data)
                continue

            # Everything else: convert to fp16
            t_fp16 = t.to(torch.float16) if t.dtype == torch.bfloat16 else t
            if t_fp16.dtype == torch.float16:
                data = t_fp16.contiguous().cpu().numpy().tobytes()
                dtype = "float16"
            else:
                data = t.contiguous().cpu().numpy().tobytes()
                dtype = str(t.dtype).replace("torch.", "")

            shape = ",".join(str(s) for s in t.shape)
            lines.append(f"{name} {offset} {len(data)} {dtype} {shape}")
            f.write(data)
            offset += len(data)

    # Quantize embed_tokens to NF4 for LM head (saves 220MB, same speed)
    embed_key = "embed_tokens.weight"
    if embed_key in tensors:
        embed = tensors[embed_key]
        embed_fp16 = embed.to(torch.float16) if embed.dtype == torch.bfloat16 else embed
        embed_np = embed_fp16.cpu().numpy()
        print(f"  Quantizing LM head ({embed_np.shape}) to NF4...")
        packed, absmax_f32, qmap = quantize_fp16_to_nf4(embed_np, block_size=64)

        with open(bin_path, "ab") as f:
            for lm_name, lm_data, lm_dtype, lm_shape in [
                ("lm_head_nf4.weight", packed.tobytes(), "uint8",
                 f"{embed_np.shape[0]},{embed_np.shape[1]//2}"),
                ("lm_head_nf4.weight.absmax", absmax_f32.tobytes(), "float32",
                 f"{len(absmax_f32)}"),
                ("lm_head_nf4.weight.quant_map", qmap.tobytes(), "float32", "16"),
            ]:
                lines.append(f"{lm_name} {offset} {len(lm_data)} {lm_dtype} {lm_shape}")
                f.write(lm_data)
                offset += len(lm_data)
        print(f"  LM head NF4: {packed.nbytes/1e6:.1f}MB packed + {absmax_f32.nbytes/1e3:.0f}KB absmax")

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"Saved {len(lines)} tensors ({total_mb:.1f}MB)")


def convert_fp16(args):
    """Dequantize everything to fp16 using bitsandbytes."""
    print(f"Converting {args.model} (fp16 mode)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import bitsandbytes as bnb

    if "bnb-4bit" in args.model:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config,
            device_map="cuda", torch_dtype=torch.float16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float16, device_map="cpu"
        )

    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []

    with open(bin_path, "wb") as f:
        for name, param in model.named_parameters():
            clean = name[6:] if name.startswith("model.") else name
            if hasattr(param, "quant_state"):
                data = bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).to(torch.float16).contiguous().cpu()
            else:
                data = param.data.to(torch.float16).contiguous().cpu()

            raw = data.numpy().tobytes()
            shape = ",".join(str(s) for s in data.shape)
            lines.append(f"{clean} {offset} {len(raw)} float16 {shape}")
            f.write(raw)
            offset += len(raw)

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"Saved {len(lines)} tensors ({total_mb:.1f}MB)")


def convert_q4l(args):
    """Dequantize NF4 to fp16, then re-quantize to linear Q4.
    This eliminates the NF4 lookup table — dequant is just (nibble-8)*scale."""
    print(f"Converting {args.model} (Q4L mode: NF4→fp16→Q4Linear)...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import bitsandbytes as bnb

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config,
        device_map="cuda", torch_dtype=torch.float16
    )

    bin_path = args.output + ".bin"
    idx_path = args.output + ".idx"
    offset = 0
    lines = []
    n_q4l = 0

    with open(bin_path, "wb") as f:
        for name, param in model.named_parameters():
            clean = name[6:] if name.startswith("model.") else name

            if hasattr(param, "quant_state"):
                # Dequantize NF4 → fp16 → re-quantize to Q4L
                fp16_data = bnb.functional.dequantize_4bit(
                    param.data, param.quant_state
                ).to(torch.float16).contiguous().cpu().numpy()

                packed, q4l_scales = quantize_fp16_to_q4l(fp16_data, block_size=64)

                # Save packed weights as uint8
                data = packed.tobytes()
                shape = ",".join(str(s) for s in fp16_data.shape)
                lines.append(f"{clean} {offset} {len(data)} uint8 {shape}")
                f.write(data)
                offset += len(data)

                # Save scales as float32 (with .absmax suffix for compatibility)
                sdata = q4l_scales.tobytes()
                lines.append(f"{clean}.absmax {offset} {len(sdata)} float32 {len(q4l_scales)}")
                f.write(sdata)
                offset += len(sdata)

                # Marker: no quant_map means Q4L format (not NF4)
                n_q4l += 1
            else:
                # Keep fp16 (embeddings, norms, etc.)
                data = param.data.to(torch.float16).contiguous().cpu().numpy().tobytes()
                shape = ",".join(str(s) for s in param.shape)
                lines.append(f"{clean} {offset} {len(data)} float16 {shape}")
                f.write(data)
                offset += len(data)

    with open(idx_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    total_mb = os.path.getsize(bin_path) / 1e6
    print(f"Saved {len(lines)} tensors ({total_mb:.1f}MB, {n_q4l} Q4L projections)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit")
    parser.add_argument("--output", default="engine/weights")
    parser.add_argument("--mode", choices=["fp16", "nf4", "q4l"], default="fp16")
    args = parser.parse_args()

    if args.mode == "nf4":
        convert_nf4(args)
    elif args.mode == "q4l":
        convert_q4l(args)
    else:
        convert_fp16(args)


if __name__ == "__main__":
    main()
