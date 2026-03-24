"""Jetson Orin compatibility patches for LLM training.

Apply these before training to work around Jetson-specific issues:
- bf16 not supported in AMP grad scaler
- bitsandbytes bnb-4bit models default to bf16 compute dtype

Usage:
    from jetson_compat import patch_amp_for_jetson, load_model_for_jetson
"""
import torch


def patch_amp_for_jetson():
    """Monkey-patch AMP grad scaler to handle bf16 gradients on Jetson.

    The Jetson Orin's torch doesn't implement
    _amp_foreach_non_finite_check_and_unscale_ for bf16.
    This patch casts bf16 grads to fp32 before the unscale op.
    """
    _orig = torch._amp_foreach_non_finite_check_and_unscale_

    def _patched(grads, found_inf, inv_scale):
        casted = [g.float() if g.dtype == torch.bfloat16 else g for g in grads]
        return _orig(casted, found_inf, inv_scale)

    torch._amp_foreach_non_finite_check_and_unscale_ = _patched
    print("[jetson_compat] Patched AMP unscale for bf16 -> fp32")


def cast_model_to_fp16(model):
    """Cast all bf16 parameters and buffers to fp16.

    Jetson Orin doesn't support bf16 in AMP. The unsloth bnb-4bit models
    have bf16 as the default compute dtype, and the LoRA weights from
    bf16-trained checkpoints are also bf16.
    """
    model.config.torch_dtype = torch.float16

    count = 0
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
            count += 1
    for name, buf in model.named_buffers():
        if buf.dtype == torch.bfloat16:
            buf.data = buf.data.to(torch.float16)
            count += 1

    if count:
        print(f"[jetson_compat] Cast {count} bf16 tensors to fp16")
    return model


def load_model_for_jetson(model_name, max_seq_length=4096, lora_rank=16):
    """Load a quantized model optimized for Jetson Orin.

    Uses fp16 compute dtype instead of bf16, which is the key difference
    from standard unsloth/cloud loading.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # NOT bf16 (Jetson compat)
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = cast_model_to_fp16(model)

    # Speed optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return model, tokenizer
