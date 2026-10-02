"""Verify causal scaled attention and its stable-softmax dependency."""

import math

import pytest
import torch

from llm_foundations.scaled_attention import causal_attention
from llm_foundations.softmax import stable_softmax

_TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float64: {"rtol": 1e-12, "atol": 1e-12},
}


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _tensors(
    q_values: list[list[float]],
    k_values: list[list[float]],
    v_values: list[list[float]],
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(q_values, dtype=dtype, device=_TEST_DEVICE),
        torch.tensor(k_values, dtype=dtype, device=_TEST_DEVICE),
        torch.tensor(v_values, dtype=dtype, device=_TEST_DEVICE),
    )


def _fixture_u(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _tensors(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 2, 3], [-1, 0, 2], [3, 1, 0], [0, 2, 1]],
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype,
    )


def _fixture_n(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _tensors(
        [[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]],
        [[0, 0, 0, 0], [math.log(2), 0, 0, 0], [math.log(4), 0, 0, 0]],
        [[3, 0], [0, 6], [7, 7]],
        dtype,
    )


def _fixture_l(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _tensors(
        [[2, 0, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0]],
        [[1000, 0, 0, 0], [1001, 0, 0, 0], [1002, 0, 0, 0]],
        [[1, -1], [2, 0], [0, 3]],
        dtype,
    )


def _fixture_multifeature(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _tensors(
        [[1, 1, 1, 1], [1, 2, 3, 4], [2, -1, 1, 3]],
        [[-1, 2, 0, 1], [2, 1, -1, 0], [1, -2, 2, 1]],
        [[1, -1], [2, 3], [-2, 4]],
        dtype,
    )


def _manual_multifeature_scores(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[1, 1, 1], [3.5, 0.5, 3.5], [-0.5, 1, 4.5]],
        dtype=dtype,
        device=device,
    )


def _prefix_scores_oracle(
    scores: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_len = scores.shape[0]
    weights = torch.zeros((seq_len, seq_len), dtype=scores.dtype, device=scores.device)
    output = torch.empty((seq_len, v.shape[1]), dtype=v.dtype, device=v.device)

    for query_index in range(seq_len):
        prefix_weights = torch.softmax(scores[query_index, : query_index + 1], dim=-1)
        weights[query_index, : query_index + 1] = prefix_weights
        output[query_index] = prefix_weights @ v[: query_index + 1]

    return output, weights


def _prefix_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q @ k.transpose(0, 1) / math.sqrt(q.shape[1])
    return _prefix_scores_oracle(scores, v)


def _assert_attention_contract(
    output: torch.Tensor,
    weights: torch.Tensor,
    q: torch.Tensor,
    v: torch.Tensor,
) -> None:
    seq_len = q.shape[0]
    assert output.shape == (seq_len, v.shape[1])
    assert weights.shape == (seq_len, seq_len)
    assert output.dtype == q.dtype == weights.dtype
    assert output.device == q.device == weights.device
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    _assert_close(weights.sum(dim=-1), torch.ones(seq_len, dtype=q.dtype, device=q.device))

    row_indices = torch.arange(seq_len, device=q.device).unsqueeze(1)
    column_indices = torch.arange(seq_len, device=q.device).unsqueeze(0)
    future_positions = column_indices > row_indices
    assert torch.equal(weights[future_positions], torch.zeros_like(weights[future_positions]))
    _assert_close(
        weights[0],
        torch.tensor([1.0] + [0.0] * (seq_len - 1), dtype=q.dtype, device=q.device),
    )
    _assert_close(output[0], v[0])


def _assert_matches_prefix_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, weights = causal_attention(q, k, v)
    expected_output, expected_weights = _prefix_oracle(q, k, v)

    _assert_attention_contract(output, weights, q, v)
    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)
    return output, weights


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stable_softmax_accepts_negative_infinity_with_a_finite_value_per_row(dtype: torch.dtype):
    logits = torch.tensor(
        [[-2, float("-inf"), float("-inf")], [-1, -3, float("-inf")], [1000, 1001, 1002]],
        dtype=dtype,
        device=_TEST_DEVICE,
    )
    original = logits.clone()
    expected = torch.softmax(logits, dim=-1)

    actual = stable_softmax(logits)

    assert actual.shape == logits.shape
    assert actual.dtype == dtype
    assert actual.device == logits.device
    assert torch.isfinite(actual).all()
    _assert_close(actual, expected)
    _assert_close(actual.sum(dim=-1), torch.ones(3, dtype=dtype, device=logits.device))
    assert torch.equal(actual[0, 1:], torch.zeros(2, dtype=dtype, device=logits.device))
    assert torch.equal(actual[1, 2:], torch.zeros(1, dtype=dtype, device=logits.device))
    _assert_close(actual[0], torch.tensor([1, 0, 0], dtype=dtype, device=logits.device))
    assert torch.equal(logits, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_equal_scores_produce_uniform_prefix_weights_and_means(dtype: torch.dtype):
    q, k, v = _fixture_u(dtype)
    expected_weights = torch.tensor(
        [[1, 0, 0, 0], [1 / 2, 1 / 2, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0], [1 / 4] * 4],
        dtype=dtype,
        device=q.device,
    )
    expected_output = torch.tensor(
        [[1, 2], [2, 3], [3, 4], [4, 5]],
        dtype=dtype,
        device=q.device,
    )

    output, weights = _assert_matches_prefix_oracle(q, k, v)

    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unequal_scores_use_sqrt_key_dimension_and_mix_value_features(dtype: torch.dtype):
    q, k, v = _fixture_n(dtype)
    expected_weights = torch.tensor(
        [[1, 0, 0], [1 / 3, 2 / 3, 0], [1 / 7, 2 / 7, 4 / 7]],
        dtype=dtype,
        device=q.device,
    )
    expected_output = torch.tensor(
        [[3, 0], [1, 4], [31 / 7, 40 / 7]],
        dtype=dtype,
        device=q.device,
    )

    output, weights = _assert_matches_prefix_oracle(q, k, v)

    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multifeature_fixture_matches_hand_calculated_score_oracle(dtype: torch.dtype):
    q, k, v = _fixture_multifeature(dtype)
    expected_scores = _manual_multifeature_scores(dtype, q.device)
    expected_output, expected_weights = _prefix_scores_oracle(expected_scores, v)

    output, weights = causal_attention(q, k, v)

    _assert_attention_contract(output, weights, q, v)
    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)


@pytest.mark.parametrize(
    ("feature_index", "score_delta_for_key_zero"),
    [
        (1, [0.5, 1.0, -0.5]),
        (2, [0.5, 1.5, 0.5]),
        (3, [0.5, 2.0, 1.5]),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_perturbing_each_nonleading_key_feature_changes_query_one(
    feature_index: int,
    score_delta_for_key_zero: list[float],
    dtype: torch.dtype,
):
    q, k, v = _fixture_multifeature(dtype)
    baseline_output, baseline_weights = causal_attention(q, k, v)
    changed_k = k.clone()
    changed_k[0, feature_index] += 1

    changed_output, changed_weights = causal_attention(q, changed_k, v)
    expected_scores = _manual_multifeature_scores(dtype, q.device)
    expected_scores[:, 0] += torch.tensor(
        score_delta_for_key_zero,
        dtype=dtype,
        device=q.device,
    )
    expected_output, expected_weights = _prefix_scores_oracle(expected_scores, v)

    _assert_attention_contract(changed_output, changed_weights, q, v)
    assert not torch.allclose(changed_weights[1], baseline_weights[1])
    assert not torch.allclose(changed_output[1], baseline_output[1])
    _assert_close(changed_weights[1], expected_weights[1])
    _assert_close(changed_output[1], expected_output[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_single_position_returns_its_value_and_unit_weight(dtype: torch.dtype):
    q, k, v = _tensors([[2, -1, 3]], [[1, 4, -2]], [[5, -7]], dtype)

    output, weights = _assert_matches_prefix_oracle(q, k, v)

    assert output.shape == (1, 2)
    assert weights.shape == (1, 1)
    _assert_close(output, torch.tensor([[5, -7]], dtype=dtype, device=q.device))
    _assert_close(weights, torch.ones((1, 1), dtype=dtype, device=q.device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_changing_a_future_value_preserves_weights_and_earlier_outputs(dtype: torch.dtype):
    q, k, v = _fixture_n(dtype)
    changed_v = v.clone()
    changed_v[2] = torch.tensor([-11, 13], dtype=dtype, device=v.device)

    output, weights = _assert_matches_prefix_oracle(q, k, v)
    changed_output, changed_weights = _assert_matches_prefix_oracle(q, k, changed_v)

    assert torch.equal(weights, changed_weights)
    _assert_close(output[:2], changed_output[:2])
    expected_changed_output, _ = _prefix_oracle(q, k, changed_v)
    _assert_close(changed_output[2], expected_changed_output[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_changing_a_future_key_preserves_prefix_rows(dtype: torch.dtype):
    q, k, v = _fixture_n(dtype)
    changed_k = k.clone()
    changed_k[2] = torch.tensor([3, 1, -2, 0], dtype=dtype, device=k.device)

    output, weights = _assert_matches_prefix_oracle(q, k, v)
    changed_output, changed_weights = _assert_matches_prefix_oracle(q, changed_k, v)

    assert torch.equal(weights[:2], changed_weights[:2])
    _assert_close(output[:2], changed_output[:2])
    expected_output, expected_weights = _prefix_oracle(q, changed_k, v)
    _assert_close(changed_weights[2], expected_weights[2])
    _assert_close(changed_output[2], expected_output[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_changing_one_query_does_not_couple_other_query_rows(dtype: torch.dtype):
    q, k, v = _fixture_n(dtype)
    changed_q = q.clone()
    changed_q[1] = torch.tensor([-1, 2, 0, 1], dtype=dtype, device=q.device)

    output, weights = _assert_matches_prefix_oracle(q, k, v)
    changed_output, changed_weights = _assert_matches_prefix_oracle(changed_q, k, v)

    assert torch.equal(weights[[0, 2]], changed_weights[[0, 2]])
    _assert_close(output[[0, 2]], changed_output[[0, 2]])
    expected_output, expected_weights = _prefix_oracle(changed_q, k, v)
    _assert_close(changed_weights[1], expected_weights[1])
    _assert_close(changed_output[1], expected_output[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_scores_remain_finite_and_match_prefix_oracle(dtype: torch.dtype):
    q, k, v = _fixture_l(dtype)

    output, weights = _assert_matches_prefix_oracle(q, k, v)

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()


@pytest.mark.parametrize("fixture", [_fixture_u, _fixture_n], ids=["uniform", "unequal"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inputs_are_unchanged_and_outputs_preserve_metadata(fixture, dtype: torch.dtype):
    q, k, v = fixture(dtype)
    originals = (q.clone(), k.clone(), v.clone())

    output, weights = causal_attention(q, k, v)

    assert torch.equal(q, originals[0])
    assert torch.equal(k, originals[1])
    assert torch.equal(v, originals[2])
    assert output.shape == (q.shape[0], v.shape[1])
    assert weights.shape == (q.shape[0], q.shape[0])
    assert output.dtype == weights.dtype == q.dtype
    assert output.device == weights.device == q.device
    expected_output, expected_weights = _prefix_oracle(q, k, v)
    _assert_close(output, expected_output)
    _assert_close(weights, expected_weights)
