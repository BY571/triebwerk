"""Generation backends: C++ engine and HuggingFace fallback."""

import torch


def generate_with_engine(engine, tokenizer, prompt, num_generations,
                         max_tokens, temperature, top_p, stop_ids):
    """Generate G completions using C++ engine (batched GEMM with tensor cores)."""
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    prompt_ids = tokenizer(text).input_ids
    eos_id = tokenizer.eos_token_id

    # Batched generation: all G completions in parallel
    batch_results = engine.generate_batch(
        [prompt_ids] * num_generations,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        stop_token_ids=stop_ids,
    )

    completions = []
    for comp_ids in batch_results:
        truncated = (
            len(comp_ids) >= max_tokens
            and (len(comp_ids) == 0 or comp_ids[-1] != eos_id)
        )
        completions.append({
            "prompt_ids": prompt_ids,
            "completion_ids": comp_ids,
            "truncated": truncated,
        })

    return completions


def generate_with_hf(model, tokenizer, prompt, num_generations,
                     max_tokens, temperature, top_p, stop_ids):
    """Generate G completions using HuggingFace generate (for dry-run)."""
    if isinstance(prompt, list):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
    else:
        text = prompt

    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids[0].tolist()
    eos_id = tokenizer.eos_token_id
    completions = []

    for _ in range(num_generations):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=eos_id,
            )
        comp_ids = out[0][len(prompt_ids):].tolist()
        truncated = (
            len(comp_ids) >= max_tokens
            and (len(comp_ids) == 0 or comp_ids[-1] != eos_id)
        )
        completions.append({
            "prompt_ids": prompt_ids,
            "completion_ids": comp_ids,
            "truncated": truncated,
        })

    return completions
