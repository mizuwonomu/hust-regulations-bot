"""Kiểm tra hợp đồng toán học của stable softmax."""

import pytest
import torch

from llm_foundations.softmax import stable_softmax

_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float64: {"rtol": 1e-12, "atol": 1e-12},
}


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def test_uniform_logits_produce_uniform_probabilities():
    logits = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    expected = torch.full((1, 3), 1 / 3, dtype=torch.float64)

    _assert_close(stable_softmax(logits), expected)


def test_known_logits_match_expected_probabilities():
    logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64)
    expected = torch.tensor(
        [[0.6652409557748218, 0.24472847105479764, 0.09003057317038046]],
        dtype=torch.float64,
    )

    _assert_close(stable_softmax(logits), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rectangular_batch_matches_torch_oracle(dtype: torch.dtype):
    logits = torch.tensor(
        [[1.25, -0.5, 3.0], [-2.0, 0.75, 1.5]],
        dtype=dtype,
    )

    result = stable_softmax(logits)

    assert result.shape == (2, 3)
    _assert_close(result, torch.softmax(logits, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_each_row_is_an_independent_probability_distribution(dtype: torch.dtype):
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [3.5, -1.0, 0.5, 2.0], [-2.0, 4.0, 1.5, 0.0]],
        dtype=dtype,
    )

    result = stable_softmax(logits)

    assert torch.isfinite(result).all()
    assert (result >= 0).all()
    assert (result <= 1).all()
    _assert_close(result.sum(dim=-1), torch.ones(3, dtype=dtype))

    sorted_indices = torch.argsort(logits, dim=-1)
    sorted_probabilities = result.gather(dim=-1, index=sorted_indices)
    assert (sorted_probabilities[:, 1:] >= sorted_probabilities[:, :-1]).all()

    perturbed_logits = logits.clone()
    perturbed_logits[1] = torch.tensor([1000.0, -1000.0, 0.0, 500.0], dtype=dtype)
    perturbed_result = stable_softmax(perturbed_logits)
    _assert_close(perturbed_result[[0, 2]], result[[0, 2]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_separate_B_by_1_row_shifts_do_not_change_probabilities(dtype: torch.dtype):
    logits = torch.tensor(
        [[0.5, 2.0, -1.0, 3.0], [4.0, -2.0, 1.0, 0.0], [-3.0, 0.5, 2.5, 1.0]],
        dtype=dtype,
    )
    row_offsets = torch.tensor([[1000.0], [-1000.0], [17.5]], dtype=dtype)

    _assert_close(stable_softmax(logits + row_offsets), stable_softmax(logits))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_positive_and_negative_logits_remain_finite(dtype: torch.dtype):
    logits = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -999.0, -998.0]], dtype=dtype)

    result = stable_softmax(logits)

    assert torch.isfinite(result).all()
    _assert_close(result, torch.softmax(logits, dim=-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_vocabulary_item_has_probability_one(dtype: torch.dtype):
    logits = torch.tensor([[5.0], [-7.0], [0.0]], dtype=dtype)

    result = stable_softmax(logits)

    assert result.shape == logits.shape
    assert result.dtype == dtype
    _assert_close(result, torch.ones_like(logits))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_preserves_tensor_metadata_and_does_not_mutate_input(dtype: torch.dtype):
    logits = torch.tensor(
        [[0.25, -1.0, 2.5, 3.0], [4.0, 0.0, -2.0, 1.5], [-1.0, 1.0, 2.0, 0.5]],
        dtype=dtype,
    )
    original = logits.clone()

    result = stable_softmax(logits)

    assert result.shape == logits.shape
    assert result.dtype == logits.dtype
    assert result.device == logits.device
    assert torch.equal(logits, original)
