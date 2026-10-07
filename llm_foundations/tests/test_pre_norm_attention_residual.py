"""Kiểm tra shape, giá trị causal và autograd của pre-norm attention residual."""

import inspect
import math
import typing

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm_foundations.layer_norm import LayerNorm
from llm_foundations.multi_head_attention import MultiHeadCausalSelfAttention
from llm_foundations.pre_norm_attention_residual import PreNormAttentionResidual
from llm_foundations.self_attention import CausalSelfAttention


_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPES = (torch.float32, torch.float64)
_TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-8, "atol": 1e-9},
}
_D_MODEL = 5
_NUM_HEADS = 2
_D_K = 2
_D_V = 3
_EPS = 0.25
_X = [
    [1, 2, 4, 8, -1],
    [-3, 0, 1, 6, 2],
    [9, 7, 2, -2, 1],
    [2, -1, 0, 3, -4],
]
_GAMMA = [1.2, -0.7, 2.0, 0.3, 1.1]
_BETA = [0.25, -0.75, 1.5, -2.0, 0.4]
_G_Y = [
    [0.3, -1.2, 2.1, 0.7, -0.4],
    [1.4, 0.2, -0.9, 2.3, 0.6],
    [-0.6, 1.7, 0.5, -1.1, 1.3],
    [0.8, -0.5, 1.2, 0.4, -1.7],
]


def _tensor(values, dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    return torch.tensor(values, dtype=dtype, device=device)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _assert_finite(tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all().item()


def _fixture_parameters(
    dtype: torch.dtype,
    device: torch.device = _TEST_DEVICE,
    d_model: int = _D_MODEL,
    num_heads: int = _NUM_HEADS,
    d_k: int = _D_K,
    d_v: int = _D_V,
):
    values = {
        "normalized_input.gamma": _tensor(_GAMMA[:d_model], dtype, device),
        "normalized_input.beta": _tensor(_BETA[:d_model], dtype, device),
    }
    for head_index in range(num_heads):
        values[f"multi_head_attention.heads.{head_index}.w_q"] = _tensor(
            [
                [
                    0.08 * ((i + 2 * j + 3 * head_index) % 7 - 3)
                    + 0.02 * (head_index + 1)
                    for j in range(d_k)
                ]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
        values[f"multi_head_attention.heads.{head_index}.w_k"] = _tensor(
            [
                [
                    0.07 * ((2 * i + j + 2 * head_index) % 9 - 4)
                    - 0.01 * (head_index + 1)
                    for j in range(d_k)
                ]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
        values[f"multi_head_attention.heads.{head_index}.w_v"] = _tensor(
            [
                [
                    0.09 * ((i + 3 * j + head_index) % 7 - 3)
                    + 0.03 * (j + 1)
                    for j in range(d_v)
                ]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
    values["multi_head_attention.w_o"] = _tensor(
        [
            [0.08 * ((2 * row + j) % 9 - 4) + 0.02 * (row + 1) for j in range(d_model)]
            for row in range(num_heads * d_v)
        ],
        dtype,
        device,
    )
    return values


def _f2_parameters(dtype: torch.dtype):
    return {
        "normalized_input.gamma": _tensor([2, -1, 0.5], dtype),
        "normalized_input.beta": _tensor([0.25, -0.5, 1], dtype),
        "multi_head_attention.heads.0.w_q": _tensor([[0, 0], [0, 0], [0, 0]], dtype),
        "multi_head_attention.heads.0.w_k": _tensor([[0, 0], [0, 0], [0, 0]], dtype),
        "multi_head_attention.heads.0.w_v": _tensor([[1, 0], [0, 1], [1, -1]], dtype),
        "multi_head_attention.w_o": _tensor([[0.5, 1, -0.5], [1, -0.5, 0.25]], dtype),
    }


def _make_candidate(
    dtype: torch.dtype,
    d_model: int = _D_MODEL,
    num_heads: int = _NUM_HEADS,
    d_k: int = _D_K,
    d_v: int = _D_V,
    eps: float | None = _EPS,
):
    constructor = {
        "d_model": d_model,
        "num_heads": num_heads,
        "d_k": d_k,
        "d_v": d_v,
    }
    if eps is not None:
        constructor["eps"] = eps
    return PreNormAttentionResidual(**constructor).to(device=_TEST_DEVICE, dtype=dtype)


def _load_parameters(candidate: nn.Module, values) -> None:
    parameters = dict(candidate.named_parameters())
    assert set(parameters) == set(values)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])


def _clone_parameters(values, requires_grad: bool = False):
    return {
        name: value.detach().clone().requires_grad_(requires_grad)
        for name, value in values.items()
    }


def _attention_reference(z: torch.Tensor, parameters):
    num_heads = sum(name.endswith(".w_q") for name in parameters)
    w_o = parameters["multi_head_attention.w_o"]
    head_outputs = []
    head_weights = []
    for head_index in range(num_heads):
        prefix = f"multi_head_attention.heads.{head_index}"
        w_q = parameters[f"{prefix}.w_q"]
        w_k = parameters[f"{prefix}.w_k"]
        w_v = parameters[f"{prefix}.w_v"]
        q = z @ w_q
        k = z @ w_k
        v = z @ w_v
        output_rows = []
        probability_rows = []
        for query_index in range(z.shape[0]):
            scores = (q[query_index : query_index + 1] @ k[: query_index + 1].T).squeeze(0)
            probabilities = torch.softmax(scores / math.sqrt(w_q.shape[1]), dim=0)
            future_zeros = torch.zeros(
                z.shape[0] - query_index - 1,
                dtype=z.dtype,
                device=z.device,
            )
            probability_rows.append(torch.cat((probabilities, future_zeros)))
            output_rows.append(probabilities @ v[: query_index + 1])
        head_outputs.append(torch.stack(output_rows))
        head_weights.append(torch.stack(probability_rows))

    projected_heads = []
    d_v = head_outputs[0].shape[1]
    for head_index, head_output in enumerate(head_outputs):
        output_projection = w_o[head_index * d_v : (head_index + 1) * d_v, :]
        projected_heads.append(head_output @ output_projection)
    attention_output = torch.stack(projected_heads).sum(dim=0)
    return attention_output, torch.stack(head_weights), head_outputs


def _reference(x: torch.Tensor, parameters, eps: float):
    d_model = x.shape[1]
    normalized = F.layer_norm(
        x,
        (d_model,),
        parameters["normalized_input.gamma"],
        parameters["normalized_input.beta"],
        eps,
    )
    attention_output, weights, head_outputs = _attention_reference(normalized, parameters)
    return x + attention_output, weights, attention_output, normalized, head_outputs


def _run(candidate: nn.Module, x: torch.Tensor):
    result = candidate(x)
    assert isinstance(result, tuple), "forward must return the (y, weights) tuple"
    assert len(result) == 2, "forward must return exactly two tensors"
    assert all(isinstance(value, torch.Tensor) for value in result)
    return result


def _assert_matches_reference(candidate: nn.Module, x: torch.Tensor, parameters, eps: float):
    actual_y, actual_weights = _run(candidate, x)
    expected_y, expected_weights, _, _, _ = _reference(x, parameters, eps)
    assert actual_y.shape == (x.shape[0], parameters["normalized_input.gamma"].numel())
    assert actual_weights.shape == (
        sum(name.endswith(".w_q") for name in parameters),
        x.shape[0],
        x.shape[0],
    )
    assert actual_y.dtype == x.dtype and actual_weights.dtype == x.dtype
    assert actual_y.device == x.device and actual_weights.device == x.device
    _assert_finite(actual_y)
    _assert_finite(actual_weights)
    assert (actual_weights >= 0).all().item()
    num_heads = actual_weights.shape[0]
    _assert_close(
        actual_weights.sum(dim=-1),
        torch.ones((num_heads, x.shape[0]), dtype=x.dtype, device=x.device),
    )
    positions = torch.arange(x.shape[0], device=x.device)
    future = positions.unsqueeze(0) > positions.unsqueeze(1)
    assert torch.equal(actual_weights[:, future], torch.zeros_like(actual_weights[:, future]))
    _assert_close(actual_y, expected_y)
    _assert_close(actual_weights, expected_weights)
    return actual_y, actual_weights


def _parameter_snapshot(module: nn.Module):
    return {
        name: {
            "identity": id(parameter),
            "storage": parameter.untyped_storage().data_ptr(),
            "value": parameter.detach().clone(),
        }
        for name, parameter in module.named_parameters()
    }


def _assert_snapshot(module: nn.Module, snapshot, changed: str | None = None) -> None:
    current = dict(module.named_parameters())
    assert set(current) == set(snapshot)
    for name, parameter in current.items():
        assert id(parameter) == snapshot[name]["identity"]
        assert parameter.untyped_storage().data_ptr() == snapshot[name]["storage"]
        if name != changed:
            assert torch.equal(parameter, snapshot[name]["value"])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn1_public_api_registration_parameter_shapes_and_configurable_widths(dtype):
    assert issubclass(PreNormAttentionResidual, nn.Module)
    init_signature = inspect.signature(PreNormAttentionResidual.__init__)
    init_hints = typing.get_type_hints(PreNormAttentionResidual.__init__)
    assert list(init_signature.parameters) == ["self", "d_model", "num_heads", "d_k", "d_v", "eps"]
    assert [init_hints[name] for name in ("d_model", "num_heads", "d_k", "d_v")] == [int, int, int, int]
    assert init_hints["eps"] is float
    assert init_signature.parameters["eps"].default == 1e-5
    assert init_hints["return"] is type(None)

    forward_signature = inspect.signature(PreNormAttentionResidual.forward)
    forward_hints = typing.get_type_hints(PreNormAttentionResidual.forward)
    assert list(forward_signature.parameters) == ["self", "x"]
    assert forward_hints["x"] is torch.Tensor
    return_hint = forward_hints["return"]
    assert typing.get_origin(return_hint) is tuple
    assert typing.get_args(return_hint) == (torch.Tensor, torch.Tensor)

    default = PreNormAttentionResidual(d_model=5, num_heads=2, d_k=2, d_v=3).to(
        device=_TEST_DEVICE,
        dtype=dtype,
    )
    explicit = PreNormAttentionResidual(
        d_model=5,
        num_heads=2,
        d_k=2,
        d_v=3,
        eps=0.25,
    ).to(device=_TEST_DEVICE, dtype=dtype)
    assert hasattr(default, "normalized_input"), "missing core child normalized_input"
    assert hasattr(default, "multi_head_attention"), "missing core child multi_head_attention"
    assert isinstance(default.normalized_input, LayerNorm), (
        "normalized_input must use llm_foundations.layer_norm.LayerNorm"
    )
    assert isinstance(default.multi_head_attention, MultiHeadCausalSelfAttention), (
        "multi_head_attention must use llm_foundations.multi_head_attention.MultiHeadCausalSelfAttention"
    )
    assert hasattr(default.normalized_input, "gamma"), "missing parameter normalized_input.gamma"
    assert hasattr(default.normalized_input, "beta"), "missing parameter normalized_input.beta"
    assert hasattr(default.multi_head_attention, "heads"), "missing child multi_head_attention.heads"
    assert hasattr(default.multi_head_attention, "w_o"), "missing parameter multi_head_attention.w_o"
    assert default.normalized_input.eps == 1e-5
    assert explicit.normalized_input.eps == 0.25
    assert set(dict(default.named_children())) == {"normalized_input", "multi_head_attention"}
    assert isinstance(default.multi_head_attention.heads, nn.ModuleList)
    assert len(default.multi_head_attention.heads) == 2
    assert all(isinstance(head, CausalSelfAttention) for head in default.multi_head_attention.heads)
    assert default.multi_head_attention.heads[0] is not default.multi_head_attention.heads[1]
    for head_index, head in enumerate(default.multi_head_attention.heads):
        for parameter_name in ("w_q", "w_k", "w_v"):
            assert hasattr(head, parameter_name), (
                f"missing public parameter multi_head_attention.heads.{head_index}.{parameter_name}"
            )
    assert torch.equal(default.normalized_input.gamma, torch.ones(5, dtype=dtype, device=_TEST_DEVICE))
    assert torch.equal(default.normalized_input.beta, torch.zeros(5, dtype=dtype, device=_TEST_DEVICE))

    expected_shapes = {
        "normalized_input.gamma": (5,),
        "normalized_input.beta": (5,),
        "multi_head_attention.w_o": (6, 5),
    }
    for head_index in range(2):
        expected_shapes.update(
            {
                f"multi_head_attention.heads.{head_index}.w_q": (5, 2),
                f"multi_head_attention.heads.{head_index}.w_k": (5, 2),
                f"multi_head_attention.heads.{head_index}.w_v": (5, 3),
            }
        )
    parameters = dict(default.named_parameters())
    assert set(parameters) == set(expected_shapes)
    assert set(default.state_dict()) == set(expected_shapes)
    assert sum(parameter.numel() for parameter in parameters.values()) == 110
    for name, parameter in parameters.items():
        assert isinstance(parameter, nn.Parameter)
        assert parameter.requires_grad
        assert tuple(parameter.shape) == expected_shapes[name]
    parameter_storages = {
        parameter.untyped_storage().data_ptr()
        for parameter in parameters.values()
    }
    assert len(parameter_storages) == len(parameters)
    for first_index in range(2):
        for second_index in range(first_index + 1, 2):
            for parameter_name in ("w_q", "w_k", "w_v"):
                first = getattr(default.multi_head_attention.heads[first_index], parameter_name)
                second = getattr(default.multi_head_attention.heads[second_index], parameter_name)
                assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()

    values = _fixture_parameters(dtype)
    _load_parameters(default, values)
    x = _tensor(_X, dtype)
    _assert_matches_reference(default, x, values, 1e-5)

    alt = _make_candidate(dtype, d_model=3, num_heads=4, d_k=2, d_v=1)
    alt_values = _fixture_parameters(dtype, d_model=3, num_heads=4, d_k=2, d_v=1)
    _load_parameters(alt, alt_values)
    alt_x = _tensor([[1, -2, 4], [3, 0, -1]], dtype)
    _assert_matches_reference(alt, alt_x, alt_values, _EPS)


def test_pn1_import_and_public_class_are_available():
    assert PreNormAttentionResidual.__name__ == "PreNormAttentionResidual"




@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn2_hand_computed_uniform_attention_and_original_skip(dtype):
    x = _tensor([[2, 4, 6], [1, 3, 5]], dtype)
    candidate = _make_candidate(dtype, d_model=3, num_heads=1, d_k=2, d_v=2, eps=1 / 3)
    values = _f2_parameters(dtype)
    _load_parameters(candidate, values)

    z = [0.25 - 4 / math.sqrt(3), -0.5, 1 + 1 / math.sqrt(3)]
    v = [z[0] + z[2], z[1] - z[2]]
    attention = [
        0.5 * v[0] + v[1],
        v[0] - 0.5 * v[1],
        -0.5 * v[0] + 0.25 * v[1],
    ]
    expected_y = _tensor(
        [
            [2 + attention[0], 4 + attention[1], 6 + attention[2]],
            [1 + attention[0], 3 + attention[1], 5 + attention[2]],
        ],
        dtype,
    )
    expected_weights = _tensor([[[1, 0], [0.5, 0.5]]], dtype)
    actual_y, actual_weights = _run(candidate, x)
    _assert_close(actual_y, expected_y)
    _assert_close(actual_weights, expected_weights)
    _assert_close(actual_y[0] - actual_y[1], x[0] - x[1])
    oracle_y, oracle_weights, _, _, _ = _reference(x, values, 1 / 3)
    _assert_close(actual_y, oracle_y)
    _assert_close(actual_weights, oracle_weights)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn3_full_forward_equivalence_causal_probabilities_and_epsilon(dtype):
    x = _tensor(_X, dtype)
    values = _fixture_parameters(dtype)
    default = _make_candidate(dtype, eps=None)
    _load_parameters(default, values)
    _assert_matches_reference(default, x, values, 1e-5)

    custom = _make_candidate(dtype, eps=_EPS)
    _load_parameters(custom, values)
    _assert_matches_reference(custom, x, values, _EPS)
    default_y, _ = _run(default, x)
    custom_y, _ = _run(custom, x)
    assert (default_y - custom_y).abs().max().item() > 1e-3

    _, weights = _run(custom, x)
    assert torch.equal(weights[:, 0, 0], torch.ones_like(weights[:, 0, 0]))
    assert not torch.allclose(weights[0], weights[1], **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn4_ordering_alternatives_and_child_routing(dtype):
    x = _tensor(_X, dtype)
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    accepted, _, accepted_attention, z, _ = _reference(x, values, _EPS)
    assert torch.linalg.vector_norm(accepted_attention).item() > 0
    raw_attention, _, _ = _attention_reference(x, values)
    post_ln = F.layer_norm(
        x + raw_attention,
        (x.shape[1],),
        values["normalized_input.gamma"],
        values["normalized_input.beta"],
        _EPS,
    )
    normalized_attention, _, _ = _attention_reference(z, values)
    normalized_skip = z + normalized_attention
    unnormalized_attention = x + raw_attention
    final_ln = F.layer_norm(
        accepted,
        (x.shape[1],),
        values["normalized_input.gamma"],
        values["normalized_input.beta"],
        _EPS,
    )
    for alternative in (post_ln, normalized_skip, unnormalized_attention, final_ln):
        assert (accepted - alternative).abs().max().item() > 1e-3
    actual_y, _ = _run(candidate, x)
    _assert_close(actual_y, accepted)

    observations = {"norm_inputs": [], "attention_inputs": [], "head_inputs": [], "attention_outputs": []}
    handles = []
    try:
        handles.append(
            candidate.normalized_input.register_forward_pre_hook(
                lambda _module, args: observations["norm_inputs"].append(args[0].detach().clone())
            )
        )
        handles.append(
            candidate.multi_head_attention.register_forward_pre_hook(
                lambda _module, args: observations["attention_inputs"].append(args[0].detach().clone())
            )
        )
        handles.append(
            candidate.multi_head_attention.register_forward_hook(
                lambda _module, _args, output: observations["attention_outputs"].append(output)
            )
        )
        for head in candidate.multi_head_attention.heads:
            handles.append(
                head.register_forward_pre_hook(
                    lambda _module, args: observations["head_inputs"].append(args[0].detach().clone())
                )
            )
        routed_y, routed_weights = _run(candidate, x)
    finally:
        for handle in handles:
            handle.remove()
    _assert_close(routed_y, accepted)
    assert len(observations["norm_inputs"]) == 1
    assert len(observations["attention_inputs"]) == 1
    assert len(observations["attention_outputs"]) == 1
    assert len(observations["head_inputs"]) == _NUM_HEADS
    _assert_close(observations["norm_inputs"][0], x)
    _assert_close(observations["attention_inputs"][0], z)
    for head_input in observations["head_inputs"]:
        _assert_close(head_input, z)
    _assert_close(routed_weights, observations["attention_outputs"][0][1])
    _assert_close(routed_weights, _reference(x, values, _EPS)[1])
    _assert_finite(accepted_attention)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn5_zero_output_projection_is_identity_with_direct_input_gradient(dtype):
    x = _tensor(_X, dtype).detach().requires_grad_(True)
    values = _fixture_parameters(dtype)
    with torch.no_grad():
        values["multi_head_attention.w_o"].zero_()
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    with torch.no_grad():
        candidate.multi_head_attention.w_o.zero_()
    candidate.zero_grad(set_to_none=True)
    parameters = dict(candidate.named_parameters())
    for parameter in parameters.values():
        parameter.grad = None

    gy = _tensor(_G_Y, dtype)
    y, weights = _run(candidate, x)
    expected_y, expected_weights, _, normalized, _ = _reference(x.detach(), values, _EPS)
    assert torch.equal(y, x.detach())
    assert not torch.equal(normalized, x.detach())
    _assert_close(weights, expected_weights)
    _assert_close(y, expected_y)
    (y * gy).sum().backward()
    _assert_close(x.grad, gy)

    ref_x = x.detach().clone().requires_grad_(True)
    ref_parameters = _clone_parameters(values, requires_grad=True)
    ref_y, _, _, _, _ = _reference(ref_x, ref_parameters, _EPS)
    (ref_y * gy).sum().backward()
    for name, parameter in parameters.items():
        assert parameter.grad is not None
        _assert_finite(parameter.grad)
        _assert_close(parameter.grad, ref_parameters[name].grad)
    for name in (
        "normalized_input.gamma",
        "normalized_input.beta",
        "multi_head_attention.heads.0.w_q",
        "multi_head_attention.heads.0.w_k",
        "multi_head_attention.heads.0.w_v",
        "multi_head_attention.heads.1.w_q",
        "multi_head_attention.heads.1.w_k",
        "multi_head_attention.heads.1.w_v",
    ):
        assert torch.equal(ref_parameters[name].grad, torch.zeros_like(ref_parameters[name].grad))
    assert torch.linalg.vector_norm(ref_parameters["multi_head_attention.w_o"].grad).item() > 0


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn6_causal_prefix_survives_prefix_calls_future_changes_and_append(dtype):
    x = _tensor(_X, dtype)
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    full_y, full_weights = _assert_matches_reference(candidate, x, values, _EPS)
    for prefix_length in (1, 2, 3):
        prefix_y, prefix_weights = _assert_matches_reference(
            candidate,
            x[:prefix_length],
            values,
            _EPS,
        )
        _assert_close(full_y[:prefix_length], prefix_y)
        _assert_close(
            full_weights[:, :prefix_length, :prefix_length],
            prefix_weights,
        )

    changed = x.clone()
    changed[2:] = _tensor([[6, -2, 1, 0, 4], [-1, 5, 2, -3, 0]], dtype)
    changed_y, changed_weights = _assert_matches_reference(candidate, changed, values, _EPS)
    _assert_close(full_y[:2], changed_y[:2])
    _assert_close(full_weights[:, :2, :2], changed_weights[:, :2, :2])
    assert not torch.allclose(full_y[2:], changed_y[2:], **_TOLERANCES[dtype])

    appended = torch.cat((x, _tensor([[3, 0, -2, 1, 5]], dtype)), dim=0)
    appended_y, appended_weights = _assert_matches_reference(candidate, appended, values, _EPS)
    _assert_close(full_y, appended_y[: x.shape[0]])
    _assert_close(full_weights, appended_weights[:, : x.shape[0], : x.shape[0]])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn7_singleton_constant_rows_single_head_and_unit_width(dtype):
    f1 = _tensor(_X, dtype)
    values = _fixture_parameters(dtype)
    singleton_candidate = _make_candidate(dtype)
    _load_parameters(singleton_candidate, values)
    singleton_x = f1[:1].detach().requires_grad_(True)
    singleton_y, singleton_weights = _assert_matches_reference(
        singleton_candidate,
        singleton_x,
        values,
        _EPS,
    )
    assert singleton_weights.shape == (_NUM_HEADS, 1, 1)
    assert torch.equal(singleton_weights, torch.ones_like(singleton_weights))
    _assert_finite(singleton_y)
    singleton_coefficients = _tensor([[0.4, -0.7, 1.1, 0.3, -1.4]], dtype)
    (singleton_y * singleton_coefficients).sum().backward()
    singleton_ref_x = singleton_x.detach().clone().requires_grad_(True)
    singleton_ref_parameters = _clone_parameters(values, requires_grad=True)
    singleton_ref_y, _, _, _, _ = _reference(singleton_ref_x, singleton_ref_parameters, _EPS)
    (singleton_ref_y * singleton_coefficients).sum().backward()
    _assert_close(singleton_x.grad, singleton_ref_x.grad)
    for name, parameter in dict(singleton_candidate.named_parameters()).items():
        expected_gradient = singleton_ref_parameters[name].grad
        if name.endswith(".w_q") or name.endswith(".w_k"):
            assert expected_gradient is not None
            assert torch.equal(expected_gradient, torch.zeros_like(expected_gradient))
            if parameter.grad is not None:
                _assert_finite(parameter.grad)
                assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        else:
            assert parameter.grad is not None
            assert expected_gradient is not None
            _assert_finite(parameter.grad)
            _assert_close(parameter.grad, expected_gradient)

    one_head = _make_candidate(dtype, num_heads=1)
    one_head_values = _fixture_parameters(dtype, num_heads=1)
    _load_parameters(one_head, one_head_values)
    _, one_head_weights = _assert_matches_reference(one_head, f1, one_head_values, _EPS)
    assert one_head_weights.shape == (1, f1.shape[0], f1.shape[0])

    constant_x = _tensor([[5, 5, 5, 5, 5], [-2, -2, -2, -2, -2], [0, 0, 0, 0, 0]], dtype)
    constant_candidate = _make_candidate(dtype)
    constant_values = _fixture_parameters(dtype)
    _load_parameters(constant_candidate, constant_values)
    constant_y, _ = _assert_matches_reference(constant_candidate, constant_x, constant_values, _EPS)
    _, _, constant_attention, normalized, _ = _reference(constant_x, constant_values, _EPS)
    _assert_close(normalized, _tensor([_BETA, _BETA, _BETA], dtype))
    assert torch.linalg.vector_norm(constant_attention).item() > 0
    assert not torch.equal(constant_y, constant_x)

    zero_beta_candidate = _make_candidate(dtype)
    zero_beta_values = _fixture_parameters(dtype)
    zero_beta_values["normalized_input.beta"].zero_()
    _load_parameters(zero_beta_candidate, zero_beta_values)
    zero_beta_y, _ = _assert_matches_reference(
        zero_beta_candidate,
        constant_x,
        zero_beta_values,
        _EPS,
    )
    assert torch.equal(zero_beta_y, constant_x)

    unit_x = _tensor([[-7], [0], [12]], dtype).detach().requires_grad_(True)
    unit_candidate = _make_candidate(dtype, d_model=1, num_heads=2, d_k=2, d_v=3, eps=_EPS)
    unit_values = _fixture_parameters(dtype, d_model=1, num_heads=2, d_k=2, d_v=3)
    unit_values["normalized_input.gamma"].copy_(_tensor([3], dtype))
    unit_values["normalized_input.beta"].copy_(_tensor([0.4], dtype))
    _load_parameters(unit_candidate, unit_values)
    unit_y, unit_weights = _assert_matches_reference(unit_candidate, unit_x, unit_values, _EPS)
    gy = _tensor([[0.3], [-0.8], [1.2]], dtype)
    (unit_y * gy).sum().backward()
    ref_x = unit_x.detach().clone().requires_grad_(True)
    ref_values = _clone_parameters(unit_values, requires_grad=True)
    ref_y, _, _, _, _ = _reference(ref_x, ref_values, _EPS)
    (ref_y * gy).sum().backward()
    unit_parameters = dict(unit_candidate.named_parameters())
    for name, parameter in unit_parameters.items():
        assert parameter.grad is not None
        _assert_finite(parameter.grad)
        _assert_close(parameter.grad, ref_values[name].grad)
    assert torch.equal(ref_values["normalized_input.gamma"].grad, torch.zeros_like(ref_values["normalized_input.gamma"].grad))
    assert unit_x.grad.dtype == gy.dtype == dtype
    assert ref_x.grad.dtype == unit_x.grad.dtype
    torch.testing.assert_close(unit_x.grad, gy, rtol=0, atol=1e-6)
    torch.testing.assert_close(unit_x.grad, ref_x.grad, rtol=0, atol=1e-6)
    _assert_finite(unit_weights)

    combined_x = _tensor([[-7]], dtype)
    combined_candidate = _make_candidate(dtype, d_model=1, num_heads=1, d_k=2, d_v=3, eps=_EPS)
    combined_values = _fixture_parameters(dtype, d_model=1, num_heads=1, d_k=2, d_v=3)
    _load_parameters(combined_candidate, combined_values)
    _, combined_weights = _assert_matches_reference(combined_candidate, combined_x, combined_values, _EPS)
    assert combined_weights.shape == (1, 1, 1)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn8_forward_preserves_inputs_parameters_grad_buffers_and_strided_storage(dtype):
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    x = _tensor(_X, dtype).detach().requires_grad_(True)
    input_before = x.detach().clone()
    parameter_snapshot = _parameter_snapshot(candidate)
    assert x.grad is None
    assert all(parameter.grad is None for parameter in candidate.parameters())
    y, weights = _run(candidate, x)
    _assert_close(y, _reference(x.detach(), values, _EPS)[0])
    _assert_close(weights, _reference(x.detach(), values, _EPS)[1])
    assert torch.equal(x.detach(), input_before)
    assert x.grad is None
    _assert_snapshot(candidate, parameter_snapshot)
    assert all(parameter.grad is None for parameter in candidate.parameters())

    backing = torch.full((4, 10), -37, dtype=dtype, device=_TEST_DEVICE)
    strided_x = backing[:, ::2]
    strided_x.copy_(x)
    backing_before = backing.clone()
    assert not strided_x.is_contiguous()
    strided_y, strided_weights = _run(candidate, strided_x)
    expected_y, expected_weights, _, _, _ = _reference(strided_x, values, _EPS)
    _assert_close(strided_y, expected_y)
    _assert_close(strided_weights, expected_weights)
    assert torch.equal(backing, backing_before)
    _assert_snapshot(candidate, parameter_snapshot)
    grad_backing = torch.full((4, 10), -37, dtype=dtype, device=_TEST_DEVICE)
    grad_view = grad_backing[:, ::2]
    grad_view.copy_(x.detach())
    grad_backing_before = grad_backing.clone()
    strided_leaf = grad_view.detach().requires_grad_(True)
    assert not strided_leaf.is_contiguous()
    candidate.zero_grad(set_to_none=True)
    strided_y, _ = _run(candidate, strided_leaf)
    gy = _tensor(_G_Y, dtype)
    (strided_y * gy).sum().backward()

    ref_backing = grad_backing.detach().clone()
    ref_leaf = ref_backing[:, ::2].detach().requires_grad_(True)
    ref_parameters = _clone_parameters(values, requires_grad=True)
    ref_y, _, _, _, _ = _reference(ref_leaf, ref_parameters, _EPS)
    (ref_y * gy).sum().backward()
    _assert_close(strided_leaf.grad, ref_leaf.grad)
    _assert_finite(strided_leaf.grad)
    for name, parameter in candidate.named_parameters():
        assert parameter.grad is not None
        _assert_finite(parameter.grad)
        _assert_close(parameter.grad, ref_parameters[name].grad)
    assert torch.equal(grad_backing, grad_backing_before)
    assert torch.equal(ref_backing, grad_backing_before)
    _assert_snapshot(candidate, parameter_snapshot)

    buffered_x = _tensor(_X, dtype).detach().requires_grad_(True)
    with torch.no_grad():
        buffered_x.grad = torch.full_like(buffered_x, 0.7)
        for parameter_index, parameter in enumerate(candidate.parameters(), start=1):
            parameter.grad = torch.full_like(parameter, 0.1 * parameter_index)
    x_grad_before = buffered_x.grad.clone()
    grad_snapshot = {name: parameter.grad.clone() for name, parameter in candidate.named_parameters()}
    _run(candidate, buffered_x)
    assert torch.equal(buffered_x.grad, x_grad_before)
    for name, parameter in candidate.named_parameters():
        assert torch.equal(parameter.grad, grad_snapshot[name])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn9_non_degenerate_autograd_matches_every_reference_gradient(dtype):
    x = _tensor(_X, dtype).detach().requires_grad_(True)
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    candidate.zero_grad(set_to_none=True)
    gy = _tensor(_G_Y, dtype)
    y, weights = _run(candidate, x)
    loss = (y * gy).sum()
    loss.backward()
    _assert_finite(y)
    _assert_finite(weights)
    _assert_finite(loss)

    candidate_parameters = dict(candidate.named_parameters())
    reference_x = x.detach().clone().requires_grad_(True)
    reference_parameters = _clone_parameters(values, requires_grad=True)
    expected_y, expected_weights, _, _, _ = _reference(reference_x, reference_parameters, _EPS)
    reference_loss = (expected_y * gy).sum()
    reference_loss.backward()
    _assert_close(y.detach(), expected_y.detach())
    _assert_close(weights.detach(), expected_weights.detach())
    _assert_close(loss.detach(), reference_loss.detach())
    _assert_close(x.grad, reference_x.grad)
    for name, parameter in candidate_parameters.items():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
        assert parameter.grad.dtype == dtype and parameter.grad.device == _TEST_DEVICE
        _assert_finite(reference_parameters[name].grad)
        _assert_close(parameter.grad, reference_parameters[name].grad)
        assert torch.linalg.vector_norm(reference_parameters[name].grad).item() > 0

    transformed_x = x.detach().clone().requires_grad_(True)
    transformed_parameters = _clone_parameters(values, requires_grad=True)
    _, _, transformed_attention, _, _ = _reference(transformed_x, transformed_parameters, _EPS)
    (transformed_attention * gy).sum().backward()
    transformed_gradient = transformed_x.grad
    assert torch.linalg.vector_norm(transformed_gradient).item() > 0
    assert not torch.allclose(x.grad, gy, **_TOLERANCES[dtype])
    assert not torch.allclose(x.grad, transformed_gradient, **_TOLERANCES[dtype])
    _assert_close(x.grad, gy + transformed_gradient)
    assert torch.equal(x.detach(), _tensor(_X, dtype))
    for name, parameter in candidate_parameters.items():
        _assert_close(parameter.detach(), values[name])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn10_returned_probability_graph_matches_query_key_reference(dtype):
    x = _tensor(_X, dtype).detach().requires_grad_(True)
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    parameters = dict(candidate.named_parameters())
    for parameter in parameters.values():
        parameter.grad = None
    _, weights = _run(candidate, x)
    coefficients = _tensor(
        [
            [
                [0.13 * (head + 1) + 0.17 * (query + 1) * (key + 1) - 0.11 * (key + 1) ** 2
                 for key in range(x.shape[0])]
                for query in range(x.shape[0])
            ]
            for head in range(_NUM_HEADS)
        ],
        dtype,
    )
    loss = (weights * coefficients).sum()
    loss.backward()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_parameters = _clone_parameters(values, requires_grad=True)
    _, ref_weights, _, _, _ = _reference(ref_x, ref_parameters, _EPS)
    ref_loss = (ref_weights * coefficients).sum()
    ref_loss.backward()
    _assert_close(loss.detach(), ref_loss.detach())
    _assert_close(x.grad, ref_x.grad)
    _assert_finite(x.grad)
    assert torch.linalg.vector_norm(ref_x.grad).item() > 0
    for name in ("normalized_input.gamma", "normalized_input.beta"):
        _assert_close(parameters[name].grad, ref_parameters[name].grad)
        assert parameters[name].grad is not None
        _assert_finite(parameters[name].grad)
        assert torch.linalg.vector_norm(ref_parameters[name].grad).item() > 0
    for head_index in range(_NUM_HEADS):
        for projection in ("w_q", "w_k"):
            name = f"multi_head_attention.heads.{head_index}.{projection}"
            _assert_close(parameters[name].grad, ref_parameters[name].grad)
            assert parameters[name].grad is not None
            _assert_finite(parameters[name].grad)
            assert torch.linalg.vector_norm(ref_parameters[name].grad).item() > 0
        name = f"multi_head_attention.heads.{head_index}.w_v"
        for unused_gradient in (parameters[name].grad, ref_parameters[name].grad):
            if unused_gradient is not None:
                assert unused_gradient.shape == parameters[name].shape
                _assert_finite(unused_gradient)
                assert torch.equal(unused_gradient, torch.zeros_like(unused_gradient))
    for name in ("multi_head_attention.w_o",):
        for unused_gradient in (parameters[name].grad, ref_parameters[name].grad):
            if unused_gradient is not None:
                assert unused_gradient.shape == parameters[name].shape
                _assert_finite(unused_gradient)
                assert torch.equal(unused_gradient, torch.zeros_like(unused_gradient))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn11_recomputation_mode_parameter_identity_and_instance_independence(dtype):
    x = _tensor(_X, dtype)
    values = _fixture_parameters(dtype)
    candidate = _make_candidate(dtype)
    _load_parameters(candidate, values)
    state_snapshot = _parameter_snapshot(candidate)
    original_modules = {
        name: id(module)
        for name, module in candidate.named_modules()
    }
    first_y, first_weights = _assert_matches_reference(candidate, x, values, _EPS)
    for sequence in (x[:1], _tensor(_X + [[3, 0, -2, 1, 5]], dtype)):
        _assert_matches_reference(candidate, sequence, values, _EPS)
    _assert_snapshot(candidate, state_snapshot)
    assert {name: id(module) for name, module in candidate.named_modules()} == original_modules
    repeated_y, repeated_weights = _assert_matches_reference(candidate, x, values, _EPS)
    _assert_close(first_y, repeated_y)
    _assert_close(first_weights, repeated_weights)

    candidate.train()
    train_y, train_weights = _run(candidate, x)
    candidate.eval()
    eval_y, eval_weights = _run(candidate, x)
    _assert_close(train_y, eval_y)
    _assert_close(train_weights, eval_weights)

    changed_values = _clone_parameters(values)
    perturbation = torch.linspace(
        -0.04,
        0.05,
        candidate.multi_head_attention.w_o.numel(),
        dtype=dtype,
        device=_TEST_DEVICE,
    ).reshape_as(candidate.multi_head_attention.w_o)
    with torch.no_grad():
        candidate.multi_head_attention.w_o.add_(perturbation)
    changed_values["multi_head_attention.w_o"].add_(perturbation)
    changed_y, changed_weights = _assert_matches_reference(candidate, x, changed_values, _EPS)
    assert not torch.allclose(first_y, changed_y, **_TOLERANCES[dtype])
    _assert_close(first_weights, changed_weights)
    assert {name: id(module) for name, module in candidate.named_modules()} == original_modules
    _assert_snapshot(candidate, state_snapshot, changed="multi_head_attention.w_o")

    independent = _make_candidate(dtype)
    _load_parameters(independent, values)
    original_y, _ = _run(candidate, x)
    original_parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in candidate.named_parameters()
    }
    for name, parameter in candidate.named_parameters():
        other = dict(independent.named_parameters())[name]
        assert parameter.untyped_storage().data_ptr() != other.untyped_storage().data_ptr()
    with torch.no_grad():
        independent.normalized_input.beta.add_(0.2)
        independent.multi_head_attention.heads[0].w_q.add_(0.1)
        independent.multi_head_attention.w_o.mul_(0.7)
    assert all(
        torch.equal(parameter, original_parameters_before[name])
        for name, parameter in candidate.named_parameters()
    )
    _assert_close(_run(candidate, x)[0], original_y)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_pn12_in_memory_state_dict_round_trip_is_independent(dtype):
    values = _fixture_parameters(dtype)
    source = _make_candidate(dtype)
    _load_parameters(source, values)
    source_snapshot = _parameter_snapshot(source)
    saved_state = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}

    destination = _make_candidate(dtype)
    load_result = destination.load_state_dict(saved_state, strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert destination is not source
    assert destination.normalized_input.eps == source.normalized_input.eps
    assert set(destination.state_dict()) == set(saved_state)
    for name, tensor in destination.state_dict().items():
        assert torch.equal(tensor, saved_state[name])
    source_y, source_weights = _run(source, _tensor(_X, dtype))
    destination_y, destination_weights = _run(destination, _tensor(_X, dtype))
    _assert_close(source_y, destination_y)
    _assert_close(source_weights, destination_weights)
    _assert_close(source_y, _reference(_tensor(_X, dtype), values, _EPS)[0])
    _assert_snapshot(source, source_snapshot)

    with torch.no_grad():
        destination.normalized_input.beta.add_(0.3)
    assert torch.equal(source.normalized_input.beta, saved_state["normalized_input.beta"])
    assert torch.equal(saved_state["normalized_input.beta"], values["normalized_input.beta"])
    assert not torch.equal(destination.normalized_input.beta, source.normalized_input.beta)
