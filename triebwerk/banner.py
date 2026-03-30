"""ASCII banner and memory map display."""

import os
import torch
from triebwerk import VERSION


def print_banner(args, run_id):
    gpu_name = "CPU"
    gpu_mem = ""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_mem = f" ({mem_gb:.1f} GB)"

    is_integrated = torch.cuda.get_device_properties(0).is_integrated if torch.cuda.is_available() else False
    if args.dry_run:
        gang = "Leerlauf (dry-run)"
    elif is_integrated:
        gang = "Sparsam"
    else:
        gang = "Schnell"

    engine_mode = "C++ dp4a" if not args.dry_run else "HF generate"
    kv_bits = getattr(args, 'kv_bits', 0)
    kv_mode = "TurboQuant 2-bit" if kv_bits == 2 else "fp16"

    # Estimate KV cache capacity from model config
    # Auto-detect from model name, or use engine config if available
    model_configs = {
        "Qwen/Qwen3-0.6B": (28, 1024),   # (num_layers, kv_dim)
        "Qwen/Qwen3-1.7B": (28, 2048),
        "Qwen/Qwen3-4B":   (36, 2560),
        "Qwen/Qwen3-8B":   (36, 4096),
    }
    num_layers, kv_dim = model_configs.get(args.model, (28, 1024))

    # Override with engine config if passed
    if hasattr(args, '_model_config') and args._model_config:
        cfg = args._model_config
        num_layers = cfg.get('num_layers', num_layers)
        kv_dim = cfg.get('num_kv_heads', 8) * cfg.get('head_dim', 128)

    G = args.num_generations
    max_tokens = args.max_completion_tokens
    prompt_est = 100

    if kv_bits == 2:
        kv_bytes_per_token_per_layer = G * (kv_dim * 2 * 0.25 + 16)
    else:
        kv_bytes_per_token_per_layer = G * kv_dim * 2 * 2

    total_context = prompt_est + max_tokens
    kv_total_mb = total_context * num_layers * kv_bytes_per_token_per_layer / 1e6

    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0]
        kv_budget = max(0, free_mem - 2e9)
        max_context = int(kv_budget / (num_layers * kv_bytes_per_token_per_layer))
        max_context = min(max_context, 32768)
    else:
        max_context = 1024

    context_pct = min(100, total_context / max_context * 100) if max_context > 0 else 100
    if context_pct > 90:
        ctx_color = "\033[1;31m"  # red
    elif context_pct > 70:
        ctx_color = "\033[1;33m"  # yellow
    else:
        ctx_color = "\033[1;32m"  # green

    print("""
   \033[1;36m\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557    \u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2557 \u2588\u2588\u2557  \u2588\u2588\u2557\033[0m
   \033[1;36m\u255a\u2550\u2550\u2588\u2588\u2554\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551    \u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u2550\u2550\u255d\u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551 \u2588\u2588\u2554\u255d\033[0m
   \033[1;36m   \u2588\u2588\u2551   \u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2551 \u2588\u2557 \u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2557  \u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2554\u255d \033[0m
   \033[1;36m   \u2588\u2588\u2551   \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2588\u2557\u2588\u2588\u2551\u2588\u2588\u2554\u2550\u2550\u255d  \u2588\u2588\u2554\u2550\u2550\u2588\u2588\u2557\u2588\u2588\u2554\u2550\u2588\u2588\u2557 \033[0m
   \033[1;36m   \u2588\u2588\u2551   \u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2588\u2588\u2588\u2588\u2554\u255d\u255a\u2588\u2588\u2588\u2554\u2588\u2588\u2588\u2554\u255d\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2557\u2588\u2588\u2551  \u2588\u2588\u2551\u2588\u2588\u2551  \u2588\u2588\u2557\033[0m
   \033[1;36m   \u255a\u2550\u255d   \u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u2550\u2550\u2550\u2550\u255d  \u255a\u2550\u2550\u255d\u255a\u2550\u2550\u255d \u255a\u2550\u2550\u2550\u2550\u2550\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\u255a\u2550\u255d  \u255a\u2550\u255d\033[0m
""")
    w = 62
    print(f"   \033[90m{'=' * w}\033[0m")
    print(f"   \033[1m{'TRIEBWERK':^{w}}\033[0m")
    print(f"   \033[90m{'Hochleistung GRPO Training':^{w}}\033[0m")
    print(f"   \033[90m{'-' * w}\033[0m")
    print(f"   \033[1m Version:\033[0m  {VERSION}")
    print(f"   \033[1m Maschine:\033[0m {gpu_name}{gpu_mem}")
    print(f"   \033[1m Antrieb:\033[0m  {engine_mode}")
    print(f"   \033[1m Gang:\033[0m     {gang}")
    print(f"   \033[1m Modell:\033[0m   {args.model}")
    print(f"   \033[1m KV-Cache:\033[0m {kv_mode} ({kv_total_mb:.0f} MB for {total_context} tok)")
    print(f"   \033[1m Kontext:\033[0m  {ctx_color}{total_context}/{max_context} ({context_pct:.0f}%)\033[0m")
    print(f"   \033[1m Schritte:\033[0m {args.max_steps} (G={G}, {max_tokens} tok)")
    print(f"   \033[1m Lauf:\033[0m     {run_id}")
    print(f"   \033[90m{'=' * w}\033[0m")


def print_memory_map(engine=None, weights_path=None):
    """Print ASCII VRAM breakdown after all initialization is complete."""
    if not torch.cuda.is_available():
        return

    total_mem = torch.cuda.get_device_properties(0).total_memory
    free_mem = torch.cuda.mem_get_info()[0]
    used_mem = total_mem - free_mem
    total_gb = total_mem / 1e9

    # PyTorch tracks its own allocations; engine uses raw cudaMalloc (not tracked by PyTorch).
    # Total used = PyTorch reserved + engine cudaMalloc + CUDA context overhead.
    pt_reserved = torch.cuda.memory_reserved()

    # Engine allocations (cudaMalloc, not visible to PyTorch)
    engine_bytes = max(0, used_mem - pt_reserved)

    # Break down engine: weights + fp16 cache + arena
    weight_bytes = 0
    cache_bytes = 0
    if weights_path and os.path.exists(weights_path + ".bin"):
        weight_bytes = os.path.getsize(weights_path + ".bin")
    if engine and hasattr(engine, 'model_config'):
        cfg = engine.model_config()
        n_layers = cfg.get('num_layers', 28)
        hidden = cfg.get('hidden_size', 1024)
        inter = cfg.get('intermediate_size', 3072)
        kv_dim = cfg.get('num_kv_heads', 8) * cfg.get('head_dim', 128)
        per_layer = (hidden*hidden*2 + kv_dim*hidden*2 + inter*hidden*2 + hidden*inter) * 2
        cache_bytes = n_layers * per_layer
    arena_bytes = max(0, engine_bytes - weight_bytes - cache_bytes)

    segments = [
        ("Gewichte",   weight_bytes, "\033[36m"),   # cyan
        ("FP16-Cache", cache_bytes,  "\033[34m"),   # blue
        ("PyTorch",    pt_reserved,  "\033[33m"),   # yellow
        ("Arena+KV",   arena_bytes,  "\033[35m"),   # magenta
    ]
    free_bytes = free_mem

    # Render bar
    bar_width = 40
    bar = ""
    legend = []
    for name, nbytes, color in segments:
        if nbytes <= 0:
            continue
        chars = max(1, round(nbytes / total_mem * bar_width))
        blocks = '\u2588' * chars
        bar += f"{color}{blocks}\033[0m"
        gb = nbytes / 1e9
        legend.append(f"{color}\u2588\u2588\033[0m {name} {gb:.1f}G")

    free_chars = bar_width - len(bar.replace('\033[0m', '').replace('\033[36m', '').replace('\033[34m', '').replace('\033[33m', '').replace('\033[35m', ''))
    # Simpler: count actual block chars
    block_count = bar.count('\u2588')
    free_chars = max(0, bar_width - block_count)
    free_blocks = '\u2591' * free_chars
    bar += f"\033[90m{free_blocks}\033[0m"
    legend.append(f"\033[90m\u2591\u2591\033[0m Frei {free_bytes/1e9:.1f}G")

    pct = used_mem / total_mem * 100
    if pct > 90:
        pct_color = "\033[1;31m"
    elif pct > 70:
        pct_color = "\033[1;33m"
    else:
        pct_color = "\033[1;32m"

    print(f"\n   \033[1m Speicher:\033[0m [{bar}] {pct_color}{used_mem/1e9:.1f}/{total_gb:.1f} GB ({pct:.0f}%)\033[0m")
    # Print legend in rows of 3
    for i in range(0, len(legend), 3):
        row = "  ".join(legend[i:i+3])
        print(f"             {row}")
