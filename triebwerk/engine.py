"""C++ engine loading and initialization."""

import os
import sys


def load_engine(num_generations, max_completion_tokens, temperature, top_p,
                kv_bits=0):
    """Load C++ engine and weights.

    Returns (engine, model_config) or raises ImportError if engine not built.
    Engine is loaded BEFORE PyTorch so it gets clean VRAM with no fragmentation.
    """
    sys.path.insert(0, os.environ.get("ENGINE_BUILD", "engine/build2"))
    import jetson_engine

    engine = jetson_engine.Engine(1024, kv_bits=kv_bits)
    weights_path = os.environ.get("ENGINE_WEIGHTS", "engine/weights_q4l")
    print(f"\nLoading C++ engine weights from {weights_path}...")
    engine.load_weights(weights_path)
    engine.cache_weights()

    # Print actual model config from engine
    cfg = engine.model_config()
    real_kv_dim = cfg['num_kv_heads'] * cfg['head_dim']
    print(f"  Engine: {cfg['num_layers']}L, {cfg['hidden_size']}h, "
          f"{cfg['num_heads']}Qh/{cfg['num_kv_heads']}KVh, KV_dim={real_kv_dim}")

    return engine, cfg


def warmup_engine(engine, model, num_generations, max_completion_tokens,
                  temperature, top_p):
    """Share embedding, init LoRA syncer, pre-allocate arena + capture CUDA graph.

    Must be called AFTER model loading but BEFORE training loop.
    Returns LoRASyncer instance.
    """
    from triebwerk.lora_sync import LoRASyncer

    # Share embedding (engine uses PyTorch's tensor, saves 311MB)
    embed_weight = model.base_model.model.model.embed_tokens.weight
    import torch
    if embed_weight.dtype == torch.float16 and embed_weight.is_cuda:
        engine.share_embedding(embed_weight.data_ptr())

    # Dummy generation to pre-allocate arena + capture CUDA graph
    # (AFTER share_embedding so pointers are final)
    dummy_prompt = list(range(200))
    engine.generate_batch(
        [dummy_prompt] * num_generations,
        max_new_tokens=max_completion_tokens,
        temperature=temperature, top_p=top_p,
        eos_token_id=dummy_prompt[0],
    )
    print(f"  Arena pre-allocated + CUDA graph captured")

    return None  # Syncer created separately by caller
