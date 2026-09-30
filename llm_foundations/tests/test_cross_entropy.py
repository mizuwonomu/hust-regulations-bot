"""Kiểm tra hợp đồng toán học của stable cross-entropy."""

import math

import torch
import torch.nn.functional as F

from llm_foundations.cross_entropy import cross_entropy_loss

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_equal_logits_have_log_two_loss():
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64, device=_DEVICE)
    targets = torch.tensor([1], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    expected = torch.tensor([math.log(2.0)], dtype=torch.float64, device=_DEVICE)

    assert losses.shape == torch.Size([1])
    assert losses.device == logits.device
    torch.testing.assert_close(losses, expected, rtol=1e-12, atol=1e-12)


def test_very_unlikely_target_has_finite_large_loss():
    logits = torch.tensor([[0.0, -1000.0]], dtype=torch.float64, device=_DEVICE)
    targets = torch.tensor([1], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    expected = torch.tensor([1000.0], dtype=torch.float64, device=_DEVICE)
    oracle = F.cross_entropy(logits, targets, reduction="none")

    assert losses.shape == torch.Size([1])
    assert losses.device == logits.device
    assert torch.isfinite(losses).all()
    torch.testing.assert_close(losses, expected, rtol=1e-12, atol=1e-10)
    torch.testing.assert_close(losses, oracle, rtol=1e-12, atol=1e-10)


def test_single_class_has_zero_loss():
    logits = torch.tensor([[5.0], [-7.0], [0.0]], dtype=torch.float64, device=_DEVICE)
    targets = torch.tensor([0, 0, 0], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    expected = torch.zeros(3, dtype=torch.float64, device=_DEVICE)

    assert losses.shape == torch.Size([3])
    assert losses.device == logits.device
    torch.testing.assert_close(losses, expected, rtol=1e-12, atol=1e-12)


def test_rectangular_batch_selects_each_rows_target():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 3, 2], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    oracle = F.cross_entropy(logits, targets, reduction="none")

    assert losses.shape == torch.Size([3])
    assert losses.device == logits.device
    torch.testing.assert_close(losses, oracle, rtol=1e-12, atol=1e-12)


def test_changing_one_row_does_not_change_other_losses():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    changed_logits = logits.clone()
    changed_logits[1] = torch.tensor([-900.0, 800.0, 1200.0, -300.0], dtype=torch.float64, device=_DEVICE)
    changed_losses = cross_entropy_loss(changed_logits, targets, reduction="none")

    torch.testing.assert_close(changed_losses[[0, 2]], losses[[0, 2]], rtol=1e-12, atol=1e-12)


def test_separate_B_by_1_row_shifts_do_not_change_losses():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=_DEVICE)
    row_offsets = torch.tensor([[1000.0], [-1000.0], [17.5]], dtype=torch.float64, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    shifted_losses = cross_entropy_loss(logits + row_offsets, targets, reduction="none")

    assert losses.shape == torch.Size([3])
    assert shifted_losses.shape == torch.Size([3])
    torch.testing.assert_close(shifted_losses, losses, rtol=1e-12, atol=1e-12)


def test_reductions_match_per_sample_losses():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=_DEVICE)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    summed = cross_entropy_loss(logits, targets, reduction="sum")
    mean = cross_entropy_loss(logits, targets, reduction="mean")
    oracle_none = F.cross_entropy(logits, targets, reduction="none")
    oracle_sum = F.cross_entropy(logits, targets, reduction="sum")
    oracle_mean = F.cross_entropy(logits, targets, reduction="mean")

    assert losses.shape == torch.Size([3])
    assert summed.shape == torch.Size([])
    assert mean.shape == torch.Size([])
    assert losses.device == logits.device
    assert summed.device == logits.device
    assert mean.device == logits.device
    torch.testing.assert_close(summed, losses.sum(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mean, losses.mean(), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(losses, oracle_none, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(summed, oracle_sum, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(mean, oracle_mean, rtol=1e-12, atol=1e-12)


def test_repeating_batch_doubles_sum_and_preserves_mean():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=_DEVICE)
    duplicated_logits = logits.repeat((2, 1))
    duplicated_targets = targets.repeat(2)

    losses = cross_entropy_loss(logits, targets, reduction="none")
    summed = cross_entropy_loss(logits, targets, reduction="sum")
    mean = cross_entropy_loss(logits, targets, reduction="mean")
    duplicated_losses = cross_entropy_loss(duplicated_logits, duplicated_targets, reduction="none")
    duplicated_sum = cross_entropy_loss(duplicated_logits, duplicated_targets, reduction="sum")
    duplicated_mean = cross_entropy_loss(duplicated_logits, duplicated_targets, reduction="mean")

    torch.testing.assert_close(duplicated_losses, torch.cat((losses, losses)), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(duplicated_sum, 2 * summed, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(duplicated_mean, mean, rtol=1e-12, atol=1e-12)


def test_does_not_mutate_logits_or_targets():
    logits = torch.tensor(
        [[3.0, 1.0, 0.0, -2.0], [-1.0, 0.0, 4.0, 2.0], [5.0, 3.0, 2.0, 1.0]],
        dtype=torch.float64,
        device=_DEVICE,
    )
    targets = torch.tensor([0, 2, 1], dtype=torch.long, device=_DEVICE)
    original_logits = logits.clone()
    original_targets = targets.clone()

    losses = cross_entropy_loss(logits, targets, reduction="none")

    assert torch.equal(logits, original_logits)
    assert torch.equal(targets, original_targets)
    assert losses.device == logits.device
