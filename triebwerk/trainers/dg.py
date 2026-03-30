"""Delightful Policy Gradient trainer with Kondo gate."""

import math

import torch
import torch.nn.functional as F

from triebwerk.trainers.base import BaseTrainer


class DGTrainer(BaseTrainer):
    """Delightful Policy Gradient trainer (Osband 2025).

    Uses gate = sigmoid(advantage * surprisal / eta), no reference model needed.
    Kondo gate: skips backward for low-delight samples (lambda=0 adaptive).

    Args:
        eta: DG temperature (default 1.0).
        **kwargs: Passed to BaseTrainer.
    """

    def __init__(self, eta=1.0, **kwargs):
        self.dg_eta = eta
        super().__init__(loss_type="dg", **kwargs)

    @property
    def needs_reference_logprobs(self):
        return False

    def compute_loss(self, model, optimizer, scaler, samples):
        """One DG gradient step with Kondo gate.

        Positive delight = breakthrough (rare correct action), worth training on.
        Negative delight = blunder (rare wrong action), skip to save compute.
        """
        device = next(model.parameters()).device
        optimizer.zero_grad()

        # Pre-filter valid samples
        valid_samples = [s for s in samples
                         if s["mask_weight"] != 0.0
                         and len(s["completion_ids"]) > 0
                         and s["advantage"] != 0.0]
        if len(valid_samples) == 0:
            return 0.0

        total_tokens = sum(len(s["completion_ids"]) for s in valid_samples)
        if total_tokens == 0:
            return 0.0

        total_loss_val = 0.0

        for s in valid_samples:
            p_ids = s["prompt_ids"]
            c_ids = s["completion_ids"]
            adv = s["advantage"]

            full_ids = p_ids + c_ids
            input_tensor = torch.tensor([full_ids], device=device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_tensor)
                logits = outputs.logits[0, :-1, :]
                targets = input_tensor[0, 1:]

                new_lp = F.log_softmax(logits.float(), dim=-1)
                new_lp = new_lp.gather(1, targets.unsqueeze(1)).squeeze(1)
                new_comp_lp = new_lp[len(p_ids) - 1:]

                # Delightful Policy Gradient + Kondo Gate
                surprisal = -new_comp_lp.detach()
                delight = adv * surprisal
                gate = torch.sigmoid(delight / self.dg_eta)
                mean_delight = delight.mean().item()

                # Kondo gate: skip backward for low-delight samples
                if mean_delight <= 0:
                    del outputs, logits, new_lp, input_tensor, new_comp_lp
                    continue

                per_token_loss = -(gate.detach() * adv * new_comp_lp)
                sample_loss = per_token_loss.sum() / total_tokens

            loss_val = sample_loss.item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                scaler.scale(sample_loss).backward()
                total_loss_val += per_token_loss.sum().item()

            del outputs, logits, new_lp, input_tensor, sample_loss, per_token_loss, new_comp_lp

        if self.empty_cache:
            torch.cuda.empty_cache()

        # Gradient clipping + optimizer step
        if total_loss_val == 0.0:
            return 0.0
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )
        if math.isnan(grad_norm.item()) or math.isinf(grad_norm.item()):
            print(f"  WARNING: NaN/Inf gradient norm, skipping step")
            optimizer.zero_grad()
            scaler.update()
            return float('nan')
        scaler.step(optimizer)
        scaler.update()

        return total_loss_val / total_tokens
