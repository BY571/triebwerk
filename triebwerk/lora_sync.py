"""Sync LoRA weights from PyTorch PEFT model to C++ engine.

Called after each training step to keep generation model in sync.
Uses cached parameter references for speed (~2ms instead of ~2s).
"""
import torch
import numpy as np


class LoRASyncer:
    """Cached LoRA weight syncer. Build the param map once, sync fast every step."""

    def __init__(self, model, engine, lora_alpha=16, lora_rank=16):
        self.engine = engine
        self.scale = lora_alpha / lora_rank

        # Build map: (layer_idx, proj_name) -> (A_param, B_param)
        self.param_map = {}
        proj_map = {
            "self_attn.q_proj": "q_proj",
            "self_attn.k_proj": "k_proj",
            "self_attn.v_proj": "v_proj",
            "self_attn.o_proj": "o_proj",
            "mlp.gate_proj": "gate_proj",
            "mlp.up_proj": "up_proj",
            "mlp.down_proj": "down_proj",
        }

        params = dict(model.named_parameters())
        for name, param in params.items():
            if "lora_A" not in name:
                continue
            parts = name.split(".")
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue

            proj_key = None
            for full_proj, short_proj in proj_map.items():
                if full_proj in name:
                    proj_key = short_proj
                    break
            if proj_key is None:
                continue

            b_name = name.replace("lora_A", "lora_B")
            b_param = params.get(b_name)
            if b_param is None:
                continue

            self.param_map[(layer_idx, proj_key)] = (param, b_param)

        print(f"  LoRA syncer: {len(self.param_map)} adapters cached")

    def sync(self):
        """Push current LoRA weights to engine via GPU-GPU memcpy (~0.02ms for 196 adapters)."""
        use_gpu = hasattr(self.engine, 'update_lora_gpu')
        for (layer_idx, proj_key), (a_param, b_param) in self.param_map.items():
            a = a_param.data.to(torch.float16).contiguous()
            b = b_param.data.to(torch.float16).contiguous()
            if use_gpu and a.is_cuda and b.is_cuda:
                self.engine.update_lora_gpu(
                    layer_idx, proj_key,
                    a.data_ptr(), a.shape[0], a.shape[1],
                    b.data_ptr(), b.shape[0], b.shape[1],
                    self.scale,
                )
            else:
                # Fallback: CPU roundtrip (needed if params not on GPU)
                self.engine.update_lora(
                    layer_idx, proj_key,
                    a.cpu().numpy(), b.cpu().numpy(), self.scale,
                )
