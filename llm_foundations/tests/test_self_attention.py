"""Verify the learned projections and causal behavior of single-head self-attention."""

import math

import pytest
import torch
from torch import nn

from llm_foundations.self_attention import CausalSelfAttention

_TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float64: {"rtol": 1e-12, "atol": 1e-12},
}


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _tensor(values, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _make_module(
    d_model: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype,
) -> CausalSelfAttention:
    return CausalSelfAttention(d_model, d_k, d_v).to(device=_TEST_DEVICE, dtype=dtype)


def _parameters(module: CausalSelfAttention) -> dict[str, nn.Parameter]:
    return {
        "w_q": module.w_q,
        "w_k": module.w_k,
        "w_v": module.w_v,
    }


def _copy_weights(
    module: CausalSelfAttention,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
) -> None:
    with torch.no_grad():
        module.w_q.copy_(w_q)
        module.w_k.copy_(w_k)
        module.w_v.copy_(w_v)


def _fixture(dtype: torch.dtype):
    x = _tensor(
        [[1, 0, 2, -1], [0, 1, -1, 2], [2, -1, 0, 1]],
        dtype,
        _TEST_DEVICE,
    )
    w_q = _tensor(
        [[0.25, 0], [0, 0.25], [0.25, 0.25], [-0.25, 0.25]],
        dtype,
        _TEST_DEVICE,
    )
    w_k = _tensor(
        [[0, 0.5], [0.5, -0.5], [-0.5, 1], [0.5, 0]],
        dtype,
        _TEST_DEVICE,
    )
    w_v = _tensor(
        [[1, 0, -1], [0, 2, 1], [0.5, -1, 0], [-1, 0.5, 2]],
        dtype,
        _TEST_DEVICE,
    )
    return x, w_q, w_k, w_v


def _project(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    rows = []
    for position in range(x.shape[0]):
        row = []
        for output_feature in range(weight.shape[1]):
            row.append(
                sum(
                    float(x[position, input_feature].item())
                    * float(weight[input_feature, output_feature].item())
                    for input_feature in range(x.shape[1])
                )
            )
        rows.append(row)
    return _tensor(rows, x.dtype, x.device)


def _scaled_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    score_rows = []
    for query_index in range(q.shape[0]):
        score_row = []
        for key_index in range(k.shape[0]):
            dot_product = sum(
                float(q[query_index, feature].item())
                * float(k[key_index, feature].item())
                for feature in range(q.shape[1])
            )
            score_row.append(dot_product / math.sqrt(q.shape[1]))
        score_rows.append(score_row)
    return _tensor(score_rows, q.dtype, q.device)


def _reference(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
):
    q = _project(x, w_q)
    k = _project(x, w_k)
    v = _project(x, w_v)
    seq_len = x.shape[0]
    scores = _scaled_scores(q, k)
    weights = torch.zeros((seq_len, seq_len), dtype=x.dtype, device=x.device)
    output = torch.empty((seq_len, v.shape[1]), dtype=x.dtype, device=x.device)
    for query_index in range(seq_len):
        prefix_weights = torch.softmax(scores[query_index, : query_index + 1], dim=0)
        weights[query_index, : query_index + 1] = prefix_weights
        output[query_index] = prefix_weights @ v[: query_index + 1]
    return output, weights, q, k, v


def _run(module: CausalSelfAttention, x: torch.Tensor):
    result = module(x)
    assert isinstance(result, tuple)
    assert len(result) == 2
    return result


def _assert_forward_matches_reference(
    module: CausalSelfAttention,
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
):
    output, weights = _run(module, x)
    expected_output, expected_weights, q, _, v = _reference(x, w_q, w_k, w_v)
    seq_len = x.shape[0]
    assert output.shape == (seq_len, v.shape[1])
    assert weights.shape == (seq_len, seq_len)
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    _assert_close(weights.sum(dim=-1), torch.ones(seq_len, dtype=x.dtype, device=x.device))
    row_indices = torch.arange(seq_len, device=x.device).unsqueeze(1)
    column_indices = torch.arange(seq_len, device=x.device).unsqueeze(0)
    future_positions = column_indices > row_indices
    assert torch.equal(weights[future_positions], torch.zeros_like(weights[future_positions]))
    _assert_close(output, expected_output)
    _assert_close(weights, expected_weights)
    _assert_close(output[0], v[0])
    return output, weights, q, v


@pytest.mark.parametrize(
    ("d_model", "d_k", "d_v"),
    [(6, 3, 5), (4, 2, 3), (1, 1, 1)],
)
def test_sa1_construction_registers_exactly_three_initialized_parameters(
    d_model: int,
    d_k: int,
    d_v: int,
):
    module = CausalSelfAttention(d_model, d_k, d_v)

    assert isinstance(module, nn.Module)
    registered = dict(module.named_parameters())
    parameters = _parameters(module)
    expected_shapes = {
        "w_q": (d_model, d_k),
        "w_k": (d_model, d_k),
        "w_v": (d_model, d_v),
    }
    assert set(registered) == set(expected_shapes)
    assert len(list(module.parameters())) == 3
    assert len({id(parameter) for parameter in parameters.values()}) == 3
    for name, shape in expected_shapes.items():
        assert isinstance(parameters[name], nn.Parameter)
        assert registered[name] is parameters[name]
        assert parameters[name].shape == shape
        assert parameters[name].requires_grad
        assert torch.isfinite(parameters[name]).all()
    assert sum(parameter.numel() for parameter in registered.values()) == (
        2 * d_model * d_k + d_model * d_v
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa2_fixed_multifeature_fixture_matches_independent_attention_reference(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)

    output, weights, q, v = _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    expected_q = _tensor([[1, 0.25], [-0.75, 0.5], [0.25, 0]], dtype, x.device)
    expected_k = _tensor([[-1.5, 2.5], [2, -1.5], [0, 1.5]], dtype, x.device)
    expected_v = _tensor([[3, -2.5, -3], [-2.5, 4, 5], [1, -1.5, -1]], dtype, x.device)
    _assert_close(q, expected_q)
    expected_scores = _tensor(
        [[-0.875, 1.625, 0.375], [2.375, -2.25, 0.75], [-0.375, 0.5, 0]],
        dtype,
        x.device,
    ) / math.sqrt(2)
    _assert_close(_scaled_scores(q, _project(x, w_k)), expected_scores)
    _assert_close(_project(x, w_k), expected_k)
    _assert_close(v, expected_v)
    _assert_close(output[0], _tensor([3, -2.5, -3], dtype, x.device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa3_one_instance_preserves_parameters_across_sequence_lengths(dtype: torch.dtype):
    _, w_q, w_k, w_v = _fixture(dtype)
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)
    registered_before = dict(module.named_parameters())
    identities_before = {name: id(parameter) for name, parameter in registered_before.items()}
    values_before = {name: parameter.detach().clone() for name, parameter in registered_before.items()}
    shapes_before = {name: parameter.shape for name, parameter in registered_before.items()}
    element_count = sum(parameter.numel() for parameter in registered_before.values())

    inputs = [
        _tensor(
            [[(((3 * t + 2 * c) % 11) - 5) / 5 for c in range(4)] for t in range(seq_len)],
            dtype,
            _TEST_DEVICE,
        )
        for seq_len in (4, 10)
    ]
    outputs = []
    weights = []
    for x in (inputs[0], inputs[1], inputs[0]):
        output, attention_weights, _, _ = _assert_forward_matches_reference(
            module, x, w_q, w_k, w_v
        )
        outputs.append(output)
        weights.append(attention_weights)
        registered_after = dict(module.named_parameters())
        assert list(registered_after) == list(registered_before)
        assert {name: id(parameter) for name, parameter in registered_after.items()} == identities_before
        assert {name: parameter.shape for name, parameter in registered_after.items()} == shapes_before
        assert sum(parameter.numel() for parameter in registered_after.values()) == element_count
        for name, parameter in registered_after.items():
            assert torch.equal(parameter, values_before[name])

    assert [output.shape for output in outputs] == [(4, 3), (10, 3), (4, 3)]
    assert [weight.shape for weight in weights] == [(4, 4), (10, 10), (4, 4)]
    _assert_close(outputs[0], outputs[2])
    _assert_close(weights[0], weights[2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa4_future_input_changes_preserve_prefix_and_match_truncated_call(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    changed_x = x.clone()
    changed_x[2] += _tensor([5, -4, 3, -2], dtype, x.device)
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)

    output, weights, _, _ = _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    changed_output, changed_weights, _, _ = _assert_forward_matches_reference(
        module, changed_x, w_q, w_k, w_v
    )
    prefix_output, prefix_weights, _, _ = _assert_forward_matches_reference(
        module, x[:2], w_q, w_k, w_v
    )

    _assert_close(output[:2], changed_output[:2])
    _assert_close(weights[:2, :], changed_weights[:2, :])
    _assert_close(output[:2], prefix_output)
    _assert_close(weights[:2, :2], prefix_weights)
    assert torch.equal(weights[:2, 2:], torch.zeros_like(weights[:2, 2:]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa5_changed_input_recomputes_projections_without_mutating_parameters(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    changed_x = x.clone()
    changed_x[1, 2] += 0.75
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)
    parameter_ids = {name: id(parameter) for name, parameter in _parameters(module).items()}
    parameter_values = {name: parameter.detach().clone() for name, parameter in _parameters(module).items()}
    expected_output, expected_weights, *_ = _reference(x, w_q, w_k, w_v)
    changed_expected_output, changed_expected_weights, *_ = _reference(changed_x, w_q, w_k, w_v)
    assert not torch.allclose(expected_output, changed_expected_output)
    assert not torch.allclose(expected_weights, changed_expected_weights)

    output, weights, _, _ = _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    changed_output, changed_weights, _, _ = _assert_forward_matches_reference(
        module, changed_x, w_q, w_k, w_v
    )

    assert not torch.allclose(output, changed_output)
    assert not torch.allclose(weights, changed_weights)
    for name, parameter in _parameters(module).items():
        assert id(parameter) == parameter_ids[name]
        assert torch.equal(parameter, parameter_values[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa6_forward_uses_current_query_key_and_value_parameters(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    module = _make_module(4, 2, 3, dtype)
    parameters = _parameters(module)
    _copy_weights(module, w_q, w_k, w_v)
    identities = {name: id(parameter) for name, parameter in parameters.items()}
    baseline_matrices = {"w_q": w_q.clone(), "w_k": w_k.clone(), "w_v": w_v.clone()}
    baseline_output, baseline_weights, *_ = _reference(x, w_q, w_k, w_v)
    variants = [
        ("w_q", (2, 1), 0.25),
        ("w_k", (3, 1), 0.5),
        ("w_v", (0, 2), 0.75),
    ]

    for changed_name, index, delta in variants:
        _copy_weights(module, w_q, w_k, w_v)
        values_before = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        expected_matrices = {name: matrix.clone() for name, matrix in baseline_matrices.items()}
        expected_matrices[changed_name][index] += delta

        with torch.no_grad():
            parameters[changed_name][index].add_(delta)

        for name, parameter in parameters.items():
            assert id(parameter) == identities[name]
            if name == changed_name:
                assert torch.equal(parameter, expected_matrices[name])
            else:
                assert torch.equal(parameter, values_before[name])

        expected_output, expected_weights, *_ = _reference(
            x,
            expected_matrices["w_q"],
            expected_matrices["w_k"],
            expected_matrices["w_v"],
        )
        if changed_name in {"w_q", "w_k"}:
            assert not torch.allclose(expected_weights, baseline_weights)
        else:
            assert torch.equal(expected_weights, baseline_weights)
            assert not torch.allclose(expected_output, baseline_output)

        output, weights = _run(module, x)
        _assert_close(output, expected_output)
        _assert_close(weights, expected_weights)
        if changed_name in {"w_q", "w_k"}:
            assert not torch.allclose(weights, baseline_weights)
        else:
            _assert_close(weights, baseline_weights)
            assert not torch.allclose(output, baseline_output)

        for name, parameter in parameters.items():
            assert id(parameter) == identities[name]
            assert torch.equal(parameter, expected_matrices[name])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa7_single_token_and_unit_dimensions_keep_all_axes(dtype: torch.dtype):
    x = _tensor([[2, -1, 3]], dtype, _TEST_DEVICE)
    w_q = _tensor([[1], [2], [-1]], dtype, _TEST_DEVICE)
    w_k = _tensor([[0], [1], [1]], dtype, _TEST_DEVICE)
    w_v = _tensor([[1, 0], [0, 2], [-1, 1]], dtype, _TEST_DEVICE)
    module = _make_module(3, 1, 2, dtype)
    _copy_weights(module, w_q, w_k, w_v)
    output, weights, _, _ = _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    assert output.shape == (1, 2)
    assert weights.shape == (1, 1)
    _assert_close(output, _tensor([[-1, 1]], dtype, x.device))
    _assert_close(weights, _tensor([[1]], dtype, x.device))

    x_unit = _tensor([[2]], dtype, _TEST_DEVICE)
    w_q_unit = _tensor([[0.5]], dtype, _TEST_DEVICE)
    w_k_unit = _tensor([[-1]], dtype, _TEST_DEVICE)
    w_v_unit = _tensor([[3]], dtype, _TEST_DEVICE)
    unit_module = _make_module(1, 1, 1, dtype)
    _copy_weights(unit_module, w_q_unit, w_k_unit, w_v_unit)
    unit_output, unit_weights, _, _ = _assert_forward_matches_reference(
        unit_module, x_unit, w_q_unit, w_k_unit, w_v_unit
    )
    assert unit_output.shape == (1, 1)
    assert unit_weights.shape == (1, 1)
    _assert_close(unit_output, _tensor([[6]], dtype, x_unit.device))
    _assert_close(unit_weights, _tensor([[1]], dtype, x_unit.device))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa8_zero_query_projection_gives_uniform_causal_prefix_averages(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    w_q.zero_()
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)

    output, weights, _, v = _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    expected_weights = _tensor(
        [[1, 0, 0], [0.5, 0.5, 0], [1 / 3, 1 / 3, 1 / 3]], dtype, x.device
    )
    expected_output = _tensor([[3, -2.5, -3], [0.25, 0.75, 1], [0.5, 0, 1 / 3]], dtype, x.device)
    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)
    _assert_close(output[1], v[:2].mean(dim=0))
    _assert_close(output[2], v.mean(dim=0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sa9_input_and_strided_view_backing_storage_are_unchanged(dtype: torch.dtype):
    x, w_q, w_k, w_v = _fixture(dtype)
    module = _make_module(4, 2, 3, dtype)
    _copy_weights(module, w_q, w_k, w_v)
    x_before = x.clone()
    _assert_forward_matches_reference(module, x, w_q, w_k, w_v)
    assert torch.equal(x, x_before)

    backing = torch.zeros((3, 8), dtype=dtype, device=_TEST_DEVICE)
    strided_x = backing[:, ::2]
    strided_x.copy_(x)
    backing_before = backing.clone()
    view_before = strided_x.clone()
    assert strided_x.shape == (3, 4)
    assert strided_x.stride(1) == 2

    output, weights, *_ = _assert_forward_matches_reference(module, strided_x, w_q, w_k, w_v)
    expected_output, expected_weights, *_ = _reference(x, w_q, w_k, w_v)
    _assert_close(output, expected_output)
    _assert_close(weights, expected_weights)
    assert torch.equal(strided_x, view_before)
    assert torch.equal(backing, backing_before)
