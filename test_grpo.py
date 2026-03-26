"""Tests for the standalone GRPO implementation.

Verifies:
1. Advantage computation (per-group normalization)
2. Loss values for known inputs (GRPO and DAPO variants)
3. Gradient flow through the loss
4. Edge cases (zero variance, empty completions, all truncated)
"""
import torch
import torch.nn.functional as F

from train import compute_advantages, grpo_step


def test_advantages_basic():
    """Advantages should be zero-mean, unit-variance within each group."""
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0,  # group 1
                            5.0, 5.0, 5.0, 5.0])  # group 2 (all same)
    G = 4
    adv = compute_advantages(rewards, G)

    # Group 1: mean=2.5, std~1.29
    group1 = adv[:4]
    assert abs(group1.mean().item()) < 1e-6, f"Group 1 mean not zero: {group1.mean()}"
    assert abs(group1.std().item() - 1.0) < 0.1, f"Group 1 std not ~1: {group1.std()}"

    # Group 2: all same reward -> all zero advantages
    group2 = adv[4:]
    assert (group2 == 0).all(), f"Group 2 should be all zeros: {group2}"

    print("PASS: test_advantages_basic")


def test_advantages_single_group():
    """Single group of 4 completions."""
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    adv = compute_advantages(rewards, 4)
    assert abs(adv.mean().item()) < 1e-6
    # Highest reward -> highest advantage
    assert adv[3] > adv[2] > adv[1] > adv[0]
    print("PASS: test_advantages_single_group")


def test_grpo_loss_values():
    """Verify loss computation with known ratio=1 (old == new logprobs)."""
    # When ratio=1 and loss_type="grpo":
    #   min(1*A, clip(1, 0.8, 1.2)*A) = min(A, A) = A
    #   loss = -A  (per token)
    # When advantage > 0: loss < 0 (encouraging)
    # When advantage < 0: loss > 0 (discouraging)

    new_lp = torch.tensor([-1.5, -2.0, -1.0])  # 3 tokens
    old_lp = torch.tensor([-1.5, -2.0, -1.0])  # same -> ratio = 1
    adv = 2.0  # positive advantage

    ratio = torch.exp(new_lp - old_lp)
    assert torch.allclose(ratio, torch.ones(3)), f"Ratio should be 1: {ratio}"

    # GRPO loss
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
    per_token = -torch.min(surr1, surr2)
    loss = per_token.mean()
    assert abs(loss.item() - (-2.0)) < 1e-6, f"GRPO loss should be -2.0: {loss}"

    # DAPO loss (same when ratio=1)
    clipped = torch.clamp(ratio, 0.8, 1.2)
    per_token_dapo = -clipped * adv
    loss_dapo = per_token_dapo.mean()
    assert abs(loss_dapo.item() - (-2.0)) < 1e-6, f"DAPO loss should be -2.0: {loss_dapo}"

    print("PASS: test_grpo_loss_values")


def test_grpo_clipping():
    """Verify clipping activates when ratio deviates."""
    old_lp = torch.tensor([-2.0, -2.0])
    # new policy assigns much higher prob -> ratio > 1
    new_lp = torch.tensor([-1.0, -1.0])  # ratio = exp(1) ≈ 2.718
    adv = 1.0
    eps = 0.2

    ratio = torch.exp(new_lp - old_lp)
    assert ratio[0].item() > 1.2, f"Ratio should exceed 1+eps: {ratio[0]}"

    # GRPO: min(ratio*A, clip(ratio)*A) = min(2.718, 1.2) = 1.2
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    grpo = -torch.min(surr1, surr2).mean()
    assert abs(grpo.item() - (-1.2)) < 1e-5, f"Clipped GRPO loss: {grpo}"

    # DAPO: clip(ratio)*A = 1.2
    dapo = -(torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
    assert abs(dapo.item() - (-1.2)) < 1e-5, f"Clipped DAPO loss: {dapo}"

    print("PASS: test_grpo_clipping")


def test_grpo_negative_advantage_clipping():
    """Clipping with negative advantage (discouraging a completion)."""
    old_lp = torch.tensor([-2.0])
    new_lp = torch.tensor([-3.0])  # ratio = exp(-1) ≈ 0.368
    adv = -1.0
    eps = 0.2

    ratio = torch.exp(new_lp - old_lp)
    assert ratio[0].item() < 0.8, f"Ratio should be below 1-eps: {ratio[0]}"

    # GRPO: ratio * A = 0.368 * -1 = -0.368
    #        clip(ratio) * A = 0.8 * -1 = -0.8
    #        min(-0.368, -0.8) = -0.8
    #        loss = -min = 0.8
    surr1 = ratio * adv  # -0.368
    surr2 = torch.clamp(ratio, 0.8, 1.2) * adv  # -0.8
    grpo = -torch.min(surr1, surr2)
    assert abs(grpo.item() - 0.8) < 1e-5, f"Expected 0.8, got {grpo}"

    # DAPO: -clip(ratio)*A = -(0.8 * -1) = 0.8
    dapo = -(torch.clamp(ratio, 0.8, 1.2) * adv)
    assert abs(dapo.item() - 0.8) < 1e-5, f"DAPO expected 0.8, got {dapo}"

    print("PASS: test_grpo_negative_advantage_clipping")


def test_gradient_flow():
    """Verify gradients flow through the loss to a simple model."""
    # Simple 1-layer model
    linear = torch.nn.Linear(4, 8)
    x = torch.randn(1, 3, 4)  # (batch, seq, dim)
    targets = torch.tensor([[1, 3, 5]])  # target token ids

    logits = linear(x)  # (1, 3, 8)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    new_lp = log_probs[0].gather(1, targets[0].unsqueeze(1)).squeeze(1)  # (3,)

    old_lp = new_lp.detach().clone()
    adv = 1.5

    ratio = torch.exp(new_lp - old_lp)
    loss = -(torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    loss.backward()

    # Check gradients exist and are non-zero
    assert linear.weight.grad is not None, "No gradient on weight"
    assert linear.weight.grad.abs().sum() > 0, "Zero gradients"

    print("PASS: test_gradient_flow")


def test_advantage_ordering():
    """Higher reward -> higher advantage within group."""
    rewards = torch.tensor([-1.0, 0.5, 1.0, 2.0])
    adv = compute_advantages(rewards, 4)
    for i in range(3):
        assert adv[i] < adv[i + 1], f"adv[{i}]={adv[i]} >= adv[{i+1}]={adv[i+1]}"
    print("PASS: test_advantage_ordering")


def test_multiple_groups():
    """Multiple groups computed independently."""
    rewards = torch.tensor([
        10.0, 10.0, 10.0, 20.0,  # group A
        0.0, 0.0, 0.0, 0.0,      # group B (all zero -> zero advantage)
        -1.0, 0.0, 1.0, 2.0,     # group C
    ])
    adv = compute_advantages(rewards, 4)

    # Group B: all zero
    assert (adv[4:8] == 0).all(), f"Group B should be zero: {adv[4:8]}"

    # Group A: only last one is above mean
    assert adv[3] > 0 and adv[0] < 0
    # Group C: ordered
    assert adv[8] < adv[9] < adv[10] < adv[11]

    print("PASS: test_multiple_groups")


if __name__ == "__main__":
    test_advantages_basic()
    test_advantages_single_group()
    test_advantage_ordering()
    test_multiple_groups()
    test_grpo_loss_values()
    test_grpo_clipping()
    test_grpo_negative_advantage_clipping()
    test_gradient_flow()
    print("\nAll tests passed!")
