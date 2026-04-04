"""C++ engine loading and initialization."""

import os
import sys


def load_engine(num_generations, max_completion_tokens, temperature, top_p,
                kv_bits=0):
    """Load C++ engine and weights.

    Supports weight formats via ENGINE_WEIGHTS env var:
      - path/to/model.gguf  → GGUF quantized (converted to Q4L on load)
      - path/to/weights_q4l → Q4L binary (.bin + .idx + .config.json)
      - "shared"            → no file, weights shared from PyTorch at runtime

    Returns (engine, model_config).
    """
    engine_build = os.environ.get("ENGINE_BUILD", "engine/build2")
    if engine_build not in sys.path:
        sys.path.insert(0, engine_build)
    import jetson_engine

    engine = jetson_engine.Engine(1024, kv_bits=kv_bits)
    weights_path = os.environ.get("ENGINE_WEIGHTS", "engine/weights_q4l")

    if weights_path.endswith(".gguf"):
        print(f"\nLoading C++ engine weights from GGUF: {weights_path}...")
        engine.load_weights_gguf(weights_path)
    else:
        print(f"\nLoading C++ engine weights from {weights_path}...")
        engine.load_weights(weights_path)

    cfg = engine.model_config()
    real_kv_dim = cfg['num_kv_heads'] * cfg['head_dim']
    print(f"  Engine: {cfg['num_layers']}L, {cfg['hidden_size']}h, "
          f"{cfg['num_heads']}Qh/{cfg['num_kv_heads']}KVh, KV_dim={real_kv_dim}")
    return engine, cfg


def share_bnb_weights(engine, model):
    """Share bnb NF4 weights directly from PyTorch to the C++ engine.

    Override engine's Q4L projections with bnb's NF4 data pointers.
    The engine already has norms/embedding from load_weights() — we just
    replace the large projection weights with bnb's quantized bytes.

    Returns list of tensor references (caller must keep to prevent GC).
    """
    import torch
    import numpy as np

    base_model = model.base_model.model.model  # unwrap PEFT
    _refs = []

    # Share embedding (engine frees its own copy)
    embed = base_model.embed_tokens.weight
    if embed.is_cuda:
        if embed.dtype != torch.float16:
            embed = embed.data.to(torch.float16).contiguous()
            _refs.append(embed)
        engine.share_embedding(embed.data_ptr())

    nf4_code_gpu = None

    hf_config = model.config
    if hasattr(hf_config, 'text_config'):
        hf_config = hf_config.text_config
    n_layers = hf_config.num_hidden_layers
    n_shared_nf4 = 0
    n_shared_fp16 = 0

    # Map: HF module suffix → engine projection name
    nf4_proj_map = {
        "self_attn.q_proj": "q_proj",
        "self_attn.k_proj": "k_proj",
        "self_attn.v_proj": "v_proj",
        "self_attn.o_proj": "o_proj",
        "mlp.gate_proj": "gate_proj",
        "mlp.up_proj": "up_proj",
        "mlp.down_proj": "down_proj",
        "linear_attn.in_proj_qkv": "ssm_in_proj_qkv",
        "linear_attn.in_proj_z": "ssm_in_proj_z",
        "linear_attn.out_proj": "ssm_out_proj",
    }
    fp16_map = {
        "self_attn.q_norm": "q_norm",
        "self_attn.k_norm": "k_norm",
        "input_layernorm": "input_layernorm",
        "post_attention_layernorm": "post_attn_layernorm",
        "linear_attn.in_proj_a": "ssm_in_proj_a",
        "linear_attn.in_proj_b": "ssm_in_proj_b",
        "linear_attn.conv1d": "ssm_conv1d_weight",
        "linear_attn.norm": "ssm_norm",
    }
    # Non-weight params (no .weight suffix in param name)
    param_map = {
        "linear_attn.A_log": "ssm_A_log",
        "linear_attn.dt_bias": "ssm_dt_bias",
    }

    # Iterate over base model params (handles PEFT's base_layer wrapping)
    for param_name, param in base_model.named_parameters():
        if "lora_" in param_name or "embed_tokens" in param_name:
            continue

        # Parse layer index
        if not param_name.startswith("layers."):
            continue
        parts = param_name.split(".", 2)
        if len(parts) < 3:
            continue
        layer_idx = int(parts[1])
        suffix = parts[2]
        # Strip PEFT's base_layer wrapper
        suffix_clean = suffix.replace(".base_layer.", ".")

        # Check NF4 quantized projections
        if suffix_clean.endswith(".weight") and hasattr(param, "quant_state"):
            mod_name = suffix_clean[:-len(".weight")]
            eng_name = nf4_proj_map.get(mod_name)
            if eng_name is None:
                continue

            qs = param.quant_state
            out_dim, in_dim = qs.shape

            # Dequant double-quantized absmax: uint8 → float32 (vectorized)
            s2 = qs.state2
            absmax_u8 = qs.absmax.cpu().numpy().astype(np.int64)
            s2_code = s2.code.cpu().numpy().astype(np.float32)
            s2_absmax = s2.absmax.cpu().numpy().astype(np.float32)
            offset = float(qs.offset) if hasattr(qs, 'offset') else 0.0

            # Vectorized: lookup + broadcast multiply + offset
            vals = s2_code[absmax_u8]
            groups = np.arange(len(absmax_u8)) // s2.blocksize
            absmax_f32 = vals * s2_absmax[groups] + offset

            # Upload float32 absmax to GPU
            absmax_gpu = torch.from_numpy(absmax_f32).cuda()
            _refs.append(absmax_gpu)

            # Upload NF4 code table (once, shared)
            if nf4_code_gpu is None:
                nf4_code = qs.code.to(torch.float32).cuda().contiguous()
                nf4_code_gpu = nf4_code
                _refs.append(nf4_code_gpu)

            # Unpack bnb NF4 nibbles to a flat array, optionally reorder rows,
            # then repack into dp4a-interleaved format.
            raw = param.data.cpu().numpy().flatten()
            n_elem = out_dim * in_dim

            # Unpack: hi nibble = elem[2i], lo nibble = elem[2i+1]
            nibbles = np.empty(n_elem, dtype=np.uint8)
            nibbles[0::2] = (raw >> 4) & 0x0F
            nibbles[1::2] = raw & 0x0F

            # Gated attention q_proj: reorder rows from interleaved to split layout
            # bnb: [q_h0(hd), gate_h0(hd), q_h1(hd), gate_h1(hd), ...]
            # engine: [q_h0, q_h1, ..., gate_h0, gate_h1, ...]
            if eng_name == "q_proj" and hasattr(hf_config, "model_type") and \
               "qwen3_5" in getattr(hf_config, "model_type", "").lower():
                nh = hf_config.num_attention_heads
                hd = out_dim // (nh * 2)  # head_dim
                nib_2d = nibbles.reshape(out_dim, in_dim)
                nib_4d = nib_2d.reshape(nh, 2, hd, in_dim)
                query = nib_4d[:, 0, :, :].reshape(-1, in_dim)
                gate = nib_4d[:, 1, :, :].reshape(-1, in_dim)
                nibbles = np.concatenate([query, gate], axis=0).flatten()
                # Also reorder absmax to match
                bpr = in_dim // 64  # blocks per row
                abs_2d = absmax_f32.reshape(out_dim, bpr)
                abs_4d = abs_2d.reshape(nh, 2, hd, bpr)
                abs_q = abs_4d[:, 0, :, :].reshape(-1, bpr)
                abs_g = abs_4d[:, 1, :, :].reshape(-1, bpr)
                absmax_f32 = np.concatenate([abs_q, abs_g], axis=0).flatten()
                absmax_gpu = torch.from_numpy(absmax_f32).cuda()
                _refs.append(absmax_gpu)

            # Repack into dp4a order: groups of 8
            grp = nibbles.reshape(-1, 8)
            packed = np.empty(n_elem // 2, dtype=np.uint8)
            for j in range(4):
                packed[j::4] = grp[:, j] | (grp[:, j + 4] << 4)

            repacked = torch.from_numpy(packed).cuda()
            _refs.append(repacked)

            engine.share_weight_nf4(
                layer_idx, eng_name,
                repacked.data_ptr(),
                absmax_gpu.data_ptr(),
                nf4_code_gpu.data_ptr(),
                out_dim, in_dim,
            )
            n_shared_nf4 += 1
            continue

        # Check fp16 small weights (norms, small SSM projections)
        if suffix_clean.endswith(".weight"):
            mod_name = suffix_clean[:-len(".weight")]
            eng_name = fp16_map.get(mod_name)
            if eng_name is None:
                continue
            w = param.data.to(torch.float16).contiguous()
            if w.dim() == 3:
                w = w.squeeze(1)  # conv1d: (C, 1, K) → (C, K)
            if not w.is_cuda:
                w = w.cuda()
            _refs.append(w)
            engine.share_weight(layer_idx, eng_name, w.data_ptr())
            n_shared_fp16 += 1
            continue

        # Non-weight params (A_log, dt_bias)
        eng_name = param_map.get(suffix_clean)
        if eng_name:
            w = param.data.to(torch.float16).contiguous()
            if not w.is_cuda:
                w = w.cuda()
            _refs.append(w)
            engine.share_weight(layer_idx, eng_name, w.data_ptr())
            n_shared_fp16 += 1

    torch.cuda.synchronize()
    print(f"  Shared {n_shared_nf4} NF4 projections (zero-copy) + {n_shared_fp16} fp16 params")
    return _refs


def _write_and_load_config(engine, hf_config):
    """Write temporary config.json from HF config and load into engine."""
    import json
    import tempfile

    config = {
        "hidden_size": hf_config.hidden_size,
        "intermediate_size": hf_config.intermediate_size,
        "num_hidden_layers": hf_config.num_hidden_layers,
        "num_attention_heads": hf_config.num_attention_heads,
        "num_key_value_heads": hf_config.num_key_value_heads,
        "head_dim": getattr(hf_config, "head_dim",
                            hf_config.hidden_size // hf_config.num_attention_heads),
        "vocab_size": hf_config.vocab_size,
        "rms_norm_eps": hf_config.rms_norm_eps,
        "rope_theta": getattr(hf_config, "rope_theta", 10000.0),
    }

    prf = getattr(hf_config, "partial_rotary_factor", None)
    if prf is None:
        rp = getattr(hf_config, "rope_parameters", None)
        if rp and isinstance(rp, dict):
            prf = rp.get("partial_rotary_factor")
            if "rope_theta" in rp:
                config["rope_theta"] = rp["rope_theta"]
    if prf is not None and prf < 1.0:
        config["rope_dim"] = int(config["head_dim"] * prf)

    model_type = getattr(hf_config, "model_type", "")
    if "qwen3_5" in model_type.lower() or "qwen3.5" in model_type.lower():
        config["gated_attn"] = 1

    if hasattr(hf_config, "linear_num_key_heads"):
        config["ssm_num_k_heads"] = hf_config.linear_num_key_heads
        config["ssm_num_v_heads"] = hf_config.linear_num_value_heads
        config["ssm_k_head_dim"] = hf_config.linear_key_head_dim
        config["ssm_v_head_dim"] = hf_config.linear_value_head_dim
        config["ssm_conv_kernel"] = getattr(hf_config, "linear_conv_kernel_dim", 4)
        if hasattr(hf_config, "layer_types"):
            config["layer_types"] = [
                0 if t == "full_attention" else 1 for t in hf_config.layer_types
            ]

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        tmp_path = f.name

    engine.load_config(tmp_path)
    os.unlink(tmp_path)
