"""Core GRPO utilities: advantage computation and log-prob computation."""

import torch
import torch.nn.functional as F


def compute_advantages(rewards, num_generations):
    """Per-group advantage normalization.

    Groups rewards by prompt (G completions each), normalizes within group.
    Returns zero advantages for groups with zero variance (DAPO dynamic sampling).
    """
    grouped = rewards.view(-1, num_generations)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)

    # Zero-variance groups get zero advantage (no gradient signal)
    advantages = torch.where(
        std > 1e-8,
        (grouped - mean) / std,
        torch.zeros_like(grouped),
    )
    return advantages.view(-1)


def compute_token_logprobs(model, prompt_ids, completion_ids, device):
    """Forward pass to get per-token log-probs for the completion.

    Returns (completion_len,) tensor of log-probs.
    """
    if len(completion_ids) == 0:
        return torch.zeros(0, device=device)

    full_ids = prompt_ids + completion_ids
    input_tensor = torch.tensor([full_ids], device=device)

    # Use same AMP context as grpo_step to avoid ratio bias from precision mismatch
    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_tensor)
        logits = outputs.logits[0, :-1, :]
    targets = input_tensor[0, 1:]

    # fp32 softmax for numerical stability
    log_probs = F.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Keep only completion portion
    comp_lp = token_lp[len(prompt_ids) - 1:]

    del outputs, logits, log_probs, token_lp, input_tensor
    return comp_lp


def compute_batch_token_logprobs(model, completions, device):
    """Batched forward pass for all G completions at once.

    Much faster than G sequential calls: one GEMM instead of G GEMVs.
    Returns list of (completion_len_i,) tensors.
    """
    # Build padded batch (left-pad so completion tokens align at the right)
    seqs = [c["prompt_ids"] + c["completion_ids"] for c in completions]
    max_len = max(len(s) for s in seqs)
    prompt_lens = [len(c["prompt_ids"]) for c in completions]
    comp_lens = [len(c["completion_ids"]) for c in completions]

    # Pad sequences and create attention mask
    pad_id = 0
    padded = []
    masks = []
    for s in seqs:
        pad_len = max_len - len(s)
        padded.append([pad_id] * pad_len + s)
        masks.append([0] * pad_len + [1] * len(s))

    input_ids = torch.tensor(padded, device=device)
    attention_mask = torch.tensor(masks, device=device)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(input_ids, attention_mask=attention_mask)

    # Compute log-probs per-sequence to avoid OOM on full vocab softmax
    # (4 x 600 x 151936 x 4 bytes = 1.4GB in fp32)
    result = []
    for i in range(len(completions)):
        seq_logits = outputs.logits[i, :-1, :]  # (seq_len-1, vocab)
        seq_targets = input_ids[i, 1:]
        # fp32 softmax for stability, but only one sequence at a time
        lp = F.log_softmax(seq_logits.float(), dim=-1)
        tok_lp = lp.gather(1, seq_targets.unsqueeze(1)).squeeze(1)
        # Extract completion portion
        pad_len = max_len - len(seqs[i])
        start = pad_len + prompt_lens[i] - 1
        end = start + comp_lens[i]
        result.append(tok_lp[start:end].detach())
        del seq_logits, lp, tok_lp

    del outputs, input_ids, attention_mask
    return result
