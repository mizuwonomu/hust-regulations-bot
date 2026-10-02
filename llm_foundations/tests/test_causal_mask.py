"""Verify the causal mask contract and its masked-score interpretation."""

import pytest
import torch

from llm_foundations.causal_mask import build_causal_mask

_TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_mask(mask: torch.Tensor, seq_len: int, expected: torch.Tensor) -> None:
    assert mask.shape == (seq_len, seq_len)
    assert mask.dtype == torch.bool
    assert mask.device == expected.device
    assert torch.equal(mask, expected)


def test_length_one_keeps_the_self_key():
    expected = torch.tensor([[False]], dtype=torch.bool, device=_TEST_DEVICE)

    mask = build_causal_mask(1, device=_TEST_DEVICE)

    _assert_mask(mask, 1, expected)


def test_length_two_blocks_only_the_future_key():
    expected = torch.tensor(
        [[False, True], [False, False]],
        dtype=torch.bool,
        device=_TEST_DEVICE,
    )

    mask = build_causal_mask(2, device=_TEST_DEVICE)

    _assert_mask(mask, 2, expected)


def test_length_four_matches_the_contract_example():
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
        device=_TEST_DEVICE,
    )

    mask = build_causal_mask(4, device=_TEST_DEVICE)

    _assert_mask(mask, 4, expected)


@pytest.mark.parametrize("seq_len", [1, 2, 4, 7])
def test_string_device_matches_each_future_position(seq_len: int):
    mask = build_causal_mask(seq_len, device=str(_TEST_DEVICE))
    expected = torch.tensor(
        [[column_index > row_index for column_index in range(seq_len)] for row_index in range(seq_len)],
        dtype=torch.bool,
        device=mask.device,
    )

    _assert_mask(mask, seq_len, expected)
    expected_blocked_per_row = torch.tensor(
        [seq_len - row_index - 1 for row_index in range(seq_len)],
        device=mask.device,
    )
    assert torch.equal(mask.sum(dim=1), expected_blocked_per_row)
    assert int(mask.sum().item()) == seq_len * (seq_len - 1) // 2


def test_device_object_keeps_every_row_with_at_least_one_visible_key():
    seq_len = 4

    mask = build_causal_mask(seq_len, device=_TEST_DEVICE)

    assert mask.dtype == torch.bool
    assert mask.shape == (seq_len, seq_len)
    allocated_device = torch.empty((), device=_TEST_DEVICE).device
    assert mask.device == allocated_device
    assert (~mask).any(dim=1).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_masked_fill_blocks_future_scores_without_mutating_input(dtype: torch.dtype):
    scores = torch.tensor(
        [[-7, 2, 3, 4], [5, -6, 7, 8], [9, 10, -5, 12], [13, 14, 15, -4]],
        dtype=dtype,
        device=_TEST_DEVICE,
    )
    original = scores.clone()
    expected = torch.tensor(
        [
            [-7, float("-inf"), float("-inf"), float("-inf")],
            [5, -6, float("-inf"), float("-inf")],
            [9, 10, -5, float("-inf")],
            [13, 14, 15, -4],
        ],
        dtype=dtype,
        device=scores.device,
    )
    mask = build_causal_mask(4, device=scores.device)

    masked_scores = scores.masked_fill(mask, float("-inf"))

    assert masked_scores.shape == scores.shape
    assert masked_scores.dtype == dtype
    assert masked_scores.device == scores.device
    assert torch.equal(masked_scores, expected)
    assert torch.equal(scores, original)
