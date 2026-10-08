"""Verify the pre-norm decoder block contract with independent numerical oracles."""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm_foundations.decoder_block import PreNormDecoderBlock
from llm_foundations.ffn import PositionWiseFFN
from llm_foundations.layer_norm import LayerNorm
from llm_foundations.pre_norm_attention_residual import PreNormAttentionResidual


_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPES = (torch.float32, torch.float64)
_FORWARD_TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-9, "atol": 1e-10},
}
_GRADIENT_TOLERANCES = {"rtol": 1e-7, "atol": 1e-9}
_FINITE_DIFFERENCE_TOLERANCES = {"rel_tol": 1e-4, "abs_tol": 1e-6}
_SCALAR_TOLERANCES = {"rel_tol": 1e-10, "abs_tol": 1e-11}
_MAIN = {"d_model": 5, "num_heads": 2, "d_k": 3, "d_v": 2, "d_ff": 7}
_EPS = 3e-4
_MAIN_X = [
    [0.2, -0.7, 1.1, 0.4, -0.3],
    [1.2, 0.1, -0.4, 0.8, -0.9],
    [-0.5, 0.6, 0.3, -1.0, 0.7],
    [0.9, -0.2, -0.8, 0.5, 0.1],
]
_MAIN_G = [
    [0.13 * (((2 * t + 3 * j) % 7) - 3) + 0.02 * (j + 1) for j in range(5)]
    for t in range(4)
]
_GAMMA1 = [0.8, 1.1, 0.6, 1.3, 0.9]
_BETA1 = [0.10, -0.15, 0.05, 0.20, -0.10]
_GAMMA2 = [1.2, 0.7, 1.4, 0.9, 0.5]
_BETA2 = [-0.20, 0.12, 0.08, -0.05, 0.18]


@pytest.fixture(scope="module", autouse=True)
def _disable_reduced_precision():
    allow_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    with torch.autocast(device_type=_TEST_DEVICE.type, enabled=False):
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(matmul_precision)
            torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
            torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32


def _tensor(values, dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    return torch.tensor(values, dtype=dtype, device=device)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, gradient=False) -> None:
    tolerances = _GRADIENT_TOLERANCES if gradient else _FORWARD_TOLERANCES[actual.dtype]
    torch.testing.assert_close(actual, expected, **tolerances)


def _assert_finite(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all().item()


def _dimensions(**overrides):
    return {**_MAIN, **overrides}


def _new_block(dtype=torch.float64, eps=_EPS, **dimensions):
    return PreNormDecoderBlock(**_dimensions(**dimensions), eps=eps).to(
        device=_TEST_DEVICE,
        dtype=dtype,
    )


def _fixture_values(dimensions, dtype, device=_TEST_DEVICE):
    d_model = dimensions["d_model"]
    num_heads = dimensions["num_heads"]
    d_k = dimensions["d_k"]
    d_v = dimensions["d_v"]
    d_ff = dimensions["d_ff"]
    values = {
        "attention_residual.normalized_input.gamma": _tensor(_GAMMA1[:d_model], dtype, device),
        "attention_residual.normalized_input.beta": _tensor(_BETA1[:d_model], dtype, device),
        "ffn_norm.gamma": _tensor(_GAMMA2[:d_model], dtype, device),
        "ffn_norm.beta": _tensor(_BETA2[:d_model], dtype, device),
    }
    for h in range(num_heads):
        prefix = f"attention_residual.multi_head_attention.heads.{h}"
        values[f"{prefix}.w_q"] = _tensor(
            [
                [0.07 * (((i + 2 * j + 3 * h) % 7) - 3) + 0.02 * (h + 1) for j in range(d_k)]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
        values[f"{prefix}.w_k"] = _tensor(
            [
                [0.06 * (((2 * i + j + 2 * h) % 9) - 4) - 0.01 * (h + 1) for j in range(d_k)]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
        values[f"{prefix}.w_v"] = _tensor(
            [
                [0.09 * (((i + 3 * j + h) % 7) - 3) + 0.03 * (j + 1) for j in range(d_v)]
                for i in range(d_model)
            ],
            dtype,
            device,
        )
    values["attention_residual.multi_head_attention.w_o"] = _tensor(
        [
            [0.08 * (((2 * r + j) % 7) - 3) + 0.015 * (r + 1) for j in range(d_model)]
            for r in range(num_heads * d_v)
        ],
        dtype,
        device,
    )
    values["ffn.w_1"] = _tensor(
        [
            [0.08 * (((i + 2 * a) % 9) - 4) + 0.01 * (a + 1) for a in range(d_ff)]
            for i in range(d_model)
        ],
        dtype,
        device,
    )
    values["ffn.b_1"] = _tensor([0.04 * ((a % 5) - 2) for a in range(d_ff)], dtype, device)
    values["ffn.w_2"] = _tensor(
        [
            [0.07 * (((3 * a + j) % 11) - 5) - 0.015 * (j + 1) for j in range(d_model)]
            for a in range(d_ff)
        ],
        dtype,
        device,
    )
    values["ffn.b_2"] = _tensor([0.03 * (j - 2) for j in range(d_model)], dtype, device)
    return values


def _fixture_input(dimensions, dtype, tokens=4):
    d_model = dimensions["d_model"]
    if dimensions == _MAIN and tokens == 4:
        return _tensor(_MAIN_X, dtype)
    return _tensor(
        [
            [0.17 * (3 * t - 2 * j) + 0.11 * (((t + 2 * j) % 4) - 1) for j in range(d_model)]
            for t in range(tokens)
        ],
        dtype,
    )


def _upstream(x: torch.Tensor):
    if tuple(x.shape) == (4, 5):
        return _tensor(_MAIN_G, x.dtype, x.device)
    return _tensor(
        [
            [0.13 * (((2 * t + 3 * j) % 7) - 3) + 0.02 * (j + 1) for j in range(x.shape[1])]
            for t in range(x.shape[0])
        ],
        x.dtype,
        x.device,
    )


def _load_fixture(module: nn.Module, values) -> None:
    parameters = dict(module.named_parameters())
    assert parameters.keys() == values.keys()
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])


def _clone_parameters(module: nn.Module, *, requires_grad=False):
    return {
        name: parameter.detach().clone().requires_grad_(requires_grad)
        for name, parameter in module.named_parameters()
    }


def _reference(x: torch.Tensor, parameters, eps: float):
    d_model = x.shape[1]
    normalized1 = F.layer_norm(
        x,
        (d_model,),
        parameters["attention_residual.normalized_input.gamma"],
        parameters["attention_residual.normalized_input.beta"],
        eps,
    )
    w_o = parameters["attention_residual.multi_head_attention.w_o"]
    head_outputs = []
    head_probabilities = []
    num_heads = sum(name.endswith(".w_q") for name in parameters)
    for h in range(num_heads):
        prefix = f"attention_residual.multi_head_attention.heads.{h}"
        q = normalized1 @ parameters[f"{prefix}.w_q"]
        k = normalized1 @ parameters[f"{prefix}.w_k"]
        v = normalized1 @ parameters[f"{prefix}.w_v"]
        outputs = []
        probabilities_by_query = []
        for t in range(x.shape[0]):
            visible_scores = q[t] @ k[: t + 1].T
            visible_probabilities = torch.softmax(
                visible_scores / math.sqrt(q.shape[1]),
                dim=0,
            )
            future_probabilities = x.new_zeros(x.shape[0] - t - 1)
            probabilities_by_query.append(torch.cat((visible_probabilities, future_probabilities)))
            outputs.append(visible_probabilities @ v[: t + 1])
        head_outputs.append(torch.stack(outputs))
        head_probabilities.append(torch.stack(probabilities_by_query))
    projected_heads = []
    d_v = head_outputs[0].shape[1]
    for h, head_output in enumerate(head_outputs):
        projection = w_o[h * d_v : (h + 1) * d_v, :]
        projected_heads.append(head_output @ projection)
    attention_update = torch.stack(projected_heads).sum(dim=0)
    probabilities = torch.stack(head_probabilities)
    h_residual = x + attention_update
    normalized2 = F.layer_norm(
        h_residual,
        (d_model,),
        parameters["ffn_norm.gamma"],
        parameters["ffn_norm.beta"],
        eps,
    )
    hidden = F.linear(normalized2, parameters["ffn.w_1"].T, parameters["ffn.b_1"])
    activated = F.gelu(hidden, approximate="none")
    ffn_update = F.linear(activated, parameters["ffn.w_2"].T, parameters["ffn.b_2"])
    output = h_residual + ffn_update
    return {
        "y": output,
        "p": probabilities,
        "h": h_residual,
        "z2": normalized2,
        "ffn_update": ffn_update,
        "attention_update": attention_update,
    }


def _run(module: nn.Module, x: torch.Tensor):
    result = module(x)
    assert isinstance(result, tuple), "forward must return the (Y, P) tuple"
    assert len(result) == 2, "forward must return exactly two tensors"
    y, probabilities = result
    assert isinstance(y, torch.Tensor) and isinstance(probabilities, torch.Tensor)
    assert y.shape == x.shape
    assert probabilities.ndim == 3 and probabilities.shape[1:] == (x.shape[0], x.shape[0])
    assert y.dtype == x.dtype and probabilities.dtype == x.dtype
    assert y.device == x.device and probabilities.device == x.device
    return y, probabilities


def _assert_matches_reference(module: nn.Module, x: torch.Tensor, eps=_EPS):
    actual_y, actual_p = _run(module, x)
    expected = _reference(x, _clone_parameters(module), eps)
    _assert_close(actual_y, expected["y"])
    _assert_close(actual_p, expected["p"])
    _assert_finite(actual_y)
    _assert_finite(actual_p)
    return expected


def _set_parameter(module: nn.Module, name: str, value: torch.Tensor) -> None:
    with torch.no_grad():
        dict(module.named_parameters())[name].copy_(value)


def _gradient_comparison(module: nn.Module, x: torch.Tensor, upstream: torch.Tensor):
    candidate_x = x.detach().clone().requires_grad_(True)
    candidate_parameters = dict(module.named_parameters())
    actual_y, _ = _run(module, candidate_x)
    actual_grads = torch.autograd.grad(
        torch.sum(actual_y * upstream),
        (candidate_x, *candidate_parameters.values()),
    )
    reference_x = x.detach().clone().requires_grad_(True)
    reference_parameters = _clone_parameters(module, requires_grad=True)
    expected = _reference(reference_x, reference_parameters, _EPS)
    reference_grads = torch.autograd.grad(
        torch.sum(expected["y"] * upstream),
        (reference_x, *reference_parameters.values()),
    )
    assert len(actual_grads) == len(reference_grads)
    for actual, wanted in zip(actual_grads, reference_grads):
        assert actual is not None and wanted is not None
        _assert_finite(actual)
        _assert_finite(wanted)
        _assert_close(actual, wanted, gradient=True)
    return actual_grads, reference_grads


def test_db1_public_api_children_and_constructor_call_forms():
    assert issubclass(PreNormDecoderBlock, nn.Module)
    signature = inspect.signature(PreNormDecoderBlock.__init__)
    assert list(signature.parameters) == ["self", "d_model", "num_heads", "d_k", "d_v", "d_ff", "eps"]
    assert signature.parameters["eps"].default == 1e-5
    positional = PreNormDecoderBlock(5, 2, 3, 2, 7)
    keyword = PreNormDecoderBlock(d_model=5, num_heads=2, d_k=3, d_v=2, d_ff=7, eps=3e-4)
    assert isinstance(positional.attention_residual, PreNormAttentionResidual)
    assert isinstance(positional.ffn_norm, LayerNorm)
    assert isinstance(positional.ffn, PositionWiseFFN)
    assert positional.attention_residual.normalized_input.eps == 1e-5
    assert positional.ffn_norm.eps == 1e-5
    assert keyword.attention_residual.normalized_input.eps == 3e-4
    assert keyword.ffn_norm.eps == 3e-4


def test_db1_forward_returns_features_and_attention_probabilities():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64)
    y, probabilities = _run(module, x)
    assert y.shape == (4, 5)
    assert probabilities.shape == (2, 4, 4)


def test_db2_child_configuration_and_epsilon_affect_nearly_constant_rows():
    dimensions = _dimensions(d_model=3, num_heads=2, d_k=2, d_v=1, d_ff=4)
    values = _fixture_values(dimensions, torch.float64)
    x = _tensor([[1.0, 1.0001, 0.9999], [1.0, 1.0001, 0.9999]], torch.float64)
    outputs = []
    for eps in (1e-5, 0.1):
        module = _new_block(torch.float64, eps=eps, **dimensions)
        _load_fixture(module, values)
        _set_parameter(
            module,
            "attention_residual.multi_head_attention.w_o",
            torch.zeros_like(module.attention_residual.multi_head_attention.w_o),
        )
        assert isinstance(module.attention_residual, PreNormAttentionResidual)
        assert isinstance(module.ffn_norm, LayerNorm)
        assert isinstance(module.ffn, PositionWiseFFN)
        assert module.attention_residual.d_model == dimensions["d_model"]
        assert module.attention_residual.num_heads == dimensions["num_heads"]
        assert module.attention_residual.d_k == dimensions["d_k"]
        assert module.attention_residual.d_v == dimensions["d_v"]
        assert module.attention_residual.normalized_input.eps == eps
        assert module.ffn_norm.eps == eps
        assert module.ffn.d_model == dimensions["d_model"]
        assert module.ffn.d_ff == dimensions["d_ff"]
        _assert_matches_reference(module, x, eps)
        outputs.append(_run(module, x)[0])
    assert not torch.allclose(outputs[0], outputs[1], rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_db3_rectangular_full_forward_matches_independent_reference(dtype):
    module = _new_block(dtype)
    _load_fixture(module, _fixture_values(_MAIN, dtype))
    x = _fixture_input(_MAIN, dtype)
    expected = _assert_matches_reference(module, x)
    assert not torch.allclose(expected["h"], x, **_FORWARD_TOLERANCES[dtype])
    assert torch.any(expected["ffn_update"] < 0).item()
    assert torch.any(expected["ffn_update"] > 0).item()
    assert not torch.allclose(expected["ffn_update"], torch.zeros_like(expected["ffn_update"]))
    visible = expected["p"][:, 1, :2]
    assert torch.any(torch.abs(visible[:, 0] - visible[:, 1]) > 1e-8).item()


def test_db4_existing_children_are_called_once_and_route_the_expected_values():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64)
    observed = {"attention": [], "norm": [], "ffn": []}
    handles = [
        module.attention_residual.register_forward_hook(
            lambda _module, inputs, output: observed["attention"].append((inputs[0], output))
        ),
        module.ffn_norm.register_forward_hook(
            lambda _module, inputs, output: observed["norm"].append((inputs[0], output))
        ),
        module.ffn.register_forward_hook(
            lambda _module, inputs, output: observed["ffn"].append((inputs[0], output))
        ),
    ]
    try:
        y, probabilities = _run(module, x)
    finally:
        for handle in handles:
            handle.remove()
    assert [len(observed[key]) for key in ("attention", "norm", "ffn")] == [1, 1, 1]
    h, child_probabilities = observed["attention"][0][1]
    norm_input, normalized = observed["norm"][0]
    ffn_input, update = observed["ffn"][0]
    _assert_close(norm_input, h)
    _assert_close(ffn_input, normalized)
    _assert_close(y, h + update)
    _assert_close(probabilities, child_probabilities)


def test_db5_norm_parameters_are_independent_and_affect_only_their_branch():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    first = module.attention_residual.normalized_input
    second = module.ffn_norm
    assert first is not second
    for name in ("gamma", "beta"):
        left = getattr(first, name)
        right = getattr(second, name)
        assert left is not right
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        assert f"attention_residual.normalized_input.{name}" in dict(module.named_parameters())
        assert f"ffn_norm.{name}" in dict(module.named_parameters())
        assert f"attention_residual.normalized_input.{name}" in module.state_dict()
        assert f"ffn_norm.{name}" in module.state_dict()
    x = _fixture_input(_MAIN, torch.float64)
    baseline_y, baseline_p = _run(module, x)
    _set_parameter(module, "ffn_norm.beta", second.beta.detach() + 0.2)
    changed_y, changed_p = _run(module, x)
    _assert_close(changed_p, baseline_p)
    assert not torch.allclose(changed_y, baseline_y, **_FORWARD_TOLERANCES[torch.float64])
    _assert_matches_reference(module, x)
    before_second = {name: value.detach().clone() for name, value in second.named_parameters()}
    _set_parameter(module, "attention_residual.normalized_input.gamma", first.gamma.detach() + 0.1)
    first_changed_y, first_changed_p = _run(module, x)
    _assert_matches_reference(module, x)
    assert not torch.allclose(first_changed_p, changed_p, **_FORWARD_TOLERANCES[torch.float64])
    assert not torch.allclose(first_changed_y, changed_y, **_FORWARD_TOLERANCES[torch.float64])
    assert all(torch.equal(getattr(second, name), before_second[name]) for name in before_second)


def test_db6_zero_final_ffn_projection_leaves_attention_residual_h():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    _set_parameter(module, "ffn.w_2", torch.zeros_like(module.ffn.w_2))
    _set_parameter(module, "ffn.b_2", torch.zeros_like(module.ffn.b_2))
    x = _fixture_input(_MAIN, torch.float64)
    expected = _reference(x, _clone_parameters(module), _EPS)
    y, probabilities = _run(module, x)
    _assert_close(y, expected["h"])
    assert not torch.allclose(expected["h"], x)
    _assert_close(probabilities, expected["p"])


def test_db7_zero_final_weight_preserves_nonzero_bias_update():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    _set_parameter(module, "ffn.w_2", torch.zeros_like(module.ffn.w_2))
    bias = _tensor([-0.4, 0.2, -0.1, 0.3, 0.15], torch.float64)
    _set_parameter(module, "ffn.b_2", bias)
    x = _fixture_input(_MAIN, torch.float64)
    expected = _reference(x, _clone_parameters(module), _EPS)
    y, _ = _run(module, x)
    _assert_close(y, expected["h"] + bias)
    assert not torch.allclose(y, expected["h"])


def test_db8_both_updates_zero_preserve_direct_input_gradient_and_all_gradients():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    _set_parameter(
        module,
        "attention_residual.multi_head_attention.w_o",
        torch.zeros_like(module.attention_residual.multi_head_attention.w_o),
    )
    _set_parameter(module, "ffn.w_2", torch.zeros_like(module.ffn.w_2))
    _set_parameter(module, "ffn.b_2", torch.zeros_like(module.ffn.b_2))
    x = _fixture_input(_MAIN, torch.float64)
    upstream = _upstream(x)
    actual_y, actual_p = _run(module, x)
    torch.testing.assert_close(actual_y, x, rtol=0, atol=0)
    _assert_close(actual_p, _reference(x, _clone_parameters(module), _EPS)["p"])
    actual_grads, _ = _gradient_comparison(module, x, upstream)
    _assert_close(actual_grads[0], upstream, gradient=True)
    assert all(gradient is not None for gradient in actual_grads)


def test_db9_zero_attention_projection_keeps_the_ffn_branch_active():
    module = _new_block()
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    _set_parameter(
        module,
        "attention_residual.multi_head_attention.w_o",
        torch.zeros_like(module.attention_residual.multi_head_attention.w_o),
    )
    x = _fixture_input(_MAIN, torch.float64)
    expected = _reference(x, _clone_parameters(module), _EPS)
    y, _ = _run(module, x)
    _assert_close(expected["h"], x)
    _assert_close(y, x + expected["ffn_update"])
    assert not torch.allclose(expected["ffn_update"], torch.zeros_like(expected["ffn_update"]))


@pytest.mark.parametrize(
    "dimensions,tokens",
    [
        (_dimensions(num_heads=1, d_k=2, d_v=3, d_ff=6), 3),
        (_dimensions(d_model=3, num_heads=4, d_k=2, d_v=1, d_ff=5), 4),
        (_dimensions(d_ff=1), 2),
        (_dimensions(d_ff=5), 2),
        (_dimensions(d_ff=7), 2),
        (_dimensions(d_model=1, num_heads=2, d_k=2, d_v=1, d_ff=3), 3),
        (_dimensions(d_model=3, num_heads=4, d_k=2, d_v=1, d_ff=5), 1),
    ],
)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_db10_dimension_families_match_independent_reference(dimensions, tokens, dtype):
    module = _new_block(dtype, **dimensions)
    _load_fixture(module, _fixture_values(dimensions, dtype))
    x = _fixture_input(dimensions, dtype, tokens=tokens)
    expected = _assert_matches_reference(module, x)
    if tokens == 1:
        _assert_close(expected["p"][:, 0, 0], torch.ones(dimensions["num_heads"], dtype=dtype, device=_TEST_DEVICE))
    if dimensions["d_model"] == 1:
        _assert_close(expected["z2"], module.ffn_norm.beta.expand_as(expected["z2"]))


def test_db10_one_module_reuses_parameters_across_sequence_lengths():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    identities = {name: id(parameter) for name, parameter in module.named_parameters()}
    values = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    for tokens in (1, 2, 6):
        x = _fixture_input(_MAIN, torch.float64, tokens=tokens)
        expected = _reference(x, _clone_parameters(module), _EPS)
        actual_y, actual_p = _run(module, x)
        _assert_close(actual_y, expected["y"])
        _assert_close(actual_p, expected["p"])
        assert {name: id(parameter) for name, parameter in module.named_parameters()} == identities
        assert all(torch.equal(dict(module.named_parameters())[name], value) for name, value in values.items())


def test_db11_constant_rows_follow_epsilon_affine_shift_and_exact_gelu():
    dimensions = _dimensions(d_model=3, num_heads=2, d_k=2, d_v=1, d_ff=4)
    module = _new_block(torch.float64, **dimensions)
    _load_fixture(module, _fixture_values(dimensions, torch.float64))
    _set_parameter(
        module,
        "attention_residual.multi_head_attention.w_o",
        torch.zeros_like(module.attention_residual.multi_head_attention.w_o),
    )
    x = _tensor([[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0], [3.0, 3.0, 3.0]], torch.float64)
    captured = []
    handle = module.ffn_norm.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        y, _ = _run(module, x)
    finally:
        handle.remove()
    expected = _reference(x, _clone_parameters(module), _EPS)
    _assert_close(captured[0], module.ffn_norm.beta.expand_as(captured[0]))
    _assert_close(y, expected["y"])
    assert not torch.allclose(expected["ffn_update"], torch.zeros_like(expected["ffn_update"]))


def test_db11_gelu_probe_matches_scalar_erf_and_zero_local_derivative():
    dimensions = _dimensions(d_model=1, num_heads=1, d_k=1, d_v=1, d_ff=1)
    module = _new_block(torch.float64, **dimensions)
    _load_fixture(module, _fixture_values(dimensions, torch.float64))
    _set_parameter(
        module,
        "attention_residual.multi_head_attention.w_o",
        torch.zeros_like(module.attention_residual.multi_head_attention.w_o),
    )
    _set_parameter(module, "ffn_norm.gamma", torch.ones_like(module.ffn_norm.gamma))
    _set_parameter(module, "ffn.w_1", torch.ones_like(module.ffn.w_1))
    _set_parameter(module, "ffn.b_1", torch.zeros_like(module.ffn.b_1))
    _set_parameter(module, "ffn.w_2", torch.ones_like(module.ffn.w_2))
    _set_parameter(module, "ffn.b_2", torch.zeros_like(module.ffn.b_2))
    x = _tensor([[0.25]], torch.float64)
    for u in (-3.0, -1.0, 0.0, 1.0, 3.0):
        _set_parameter(module, "ffn_norm.beta", torch.tensor([u], dtype=torch.float64, device=_TEST_DEVICE))
        y, _ = _run(module, x)
        expected = x.item() + 0.5 * u * (1.0 + math.erf(u / math.sqrt(2.0)))
        assert math.isclose(y.item(), expected, **_SCALAR_TOLERANCES)
    _set_parameter(module, "ffn_norm.beta", torch.zeros_like(module.ffn_norm.beta))
    y, _ = _run(module, x)
    beta_gradient = torch.autograd.grad(y.sum(), module.ffn_norm.beta)[0]
    assert math.isclose(beta_gradient.item(), 0.5, rel_tol=0, abs_tol=1e-12)
    _set_parameter(module, "ffn.b_2", torch.full_like(module.ffn.b_2, -2.0))
    negative_y, _ = _run(module, x)
    assert negative_y.item() < 0
    assert math.isclose(negative_y.item(), x.item() - 2.0, **_SCALAR_TOLERANCES)


def test_db12_probabilities_and_causal_prefixes_are_invariant_to_future_tokens():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64)
    full_y, full_p = _run(module, x)
    _assert_matches_reference(module, x)
    assert torch.isfinite(full_p).all().item()
    assert (full_p >= 0).all().item()
    _assert_close(full_p.sum(dim=-1), torch.ones_like(full_p.sum(dim=-1)))
    for query in range(x.shape[0]):
        assert torch.equal(full_p[:, query, query + 1 :], torch.zeros_like(full_p[:, query, query + 1 :]))
    for prefix_length in (1, 2, 3):
        changed = x.clone()
        perturbation = _tensor([0.4, -0.6, 0.9, -1.1, 0.2], x.dtype, x.device)
        changed[prefix_length:] = changed[prefix_length:] + perturbation
        normalized = F.layer_norm(
            x,
            (x.shape[1],),
            module.attention_residual.normalized_input.gamma,
            module.attention_residual.normalized_input.beta,
            _EPS,
        )
        normalized_changed = F.layer_norm(
            changed,
            (x.shape[1],),
            module.attention_residual.normalized_input.gamma,
            module.attention_residual.normalized_input.beta,
            _EPS,
        )
        assert not torch.allclose(
            normalized[prefix_length:],
            normalized_changed[prefix_length:],
            **_FORWARD_TOLERANCES[x.dtype],
        )
        changed_y, changed_p = _run(module, changed)
        prefix_x = x[:prefix_length]
        prefix_y, prefix_p = _run(module, prefix_x)
        _assert_close(changed_y[:prefix_length], full_y[:prefix_length])
        _assert_close(changed_p[:, :prefix_length, :prefix_length], full_p[:, :prefix_length, :prefix_length])
        _assert_close(prefix_y, full_y[:prefix_length])
        _assert_close(prefix_p, full_p[:, :prefix_length, :prefix_length])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_db13_noncontiguous_input_preserves_values_parameters_gradients_and_placement(dtype):
    module = _new_block(dtype)
    _load_fixture(module, _fixture_values(_MAIN, dtype))
    x_values = _fixture_input(_MAIN, dtype)
    storage = torch.empty((x_values.shape[0], 2 * x_values.shape[1]), dtype=dtype, device=_TEST_DEVICE)
    x = storage[:, ::2]
    x.copy_(x_values)
    assert not x.is_contiguous()
    before_x = x.detach().clone()
    parameters = dict(module.named_parameters())
    parameter_values = {name: value.detach().clone() for name, value in parameters.items()}
    for index, parameter in enumerate(parameters.values()):
        parameter.grad = torch.full_like(parameter, 0.01 * (index + 1))
    gradient_values = {name: value.grad.detach().clone() for name, value in parameters.items()}
    expected = _reference(x, _clone_parameters(module), _EPS)
    y, probabilities = _run(module, x)
    _assert_close(y, expected["y"])
    _assert_close(probabilities, expected["p"])
    contiguous_y, contiguous_p = _run(module, x.contiguous())
    _assert_close(y, contiguous_y)
    _assert_close(probabilities, contiguous_p)
    gradient_input = x.detach().requires_grad_(True)
    assert not gradient_input.is_contiguous()
    gradient_y, _ = _run(module, gradient_input)
    input_gradient = torch.autograd.grad(
        torch.sum(gradient_y * _upstream(gradient_input)),
        gradient_input,
    )[0]
    assert input_gradient.shape == gradient_input.shape
    assert input_gradient.dtype == dtype
    assert input_gradient.device == _TEST_DEVICE
    _assert_finite(input_gradient)
    assert torch.equal(x, before_x)
    for name, parameter in parameters.items():
        assert parameter.dtype == dtype and parameter.device == _TEST_DEVICE
        assert torch.equal(parameter, parameter_values[name])
        assert torch.equal(parameter.grad, gradient_values[name])
    assert y.dtype == probabilities.dtype == dtype
    assert y.device == probabilities.device == x.device


def test_db14_all_parameter_and_input_gradients_match_independent_reference():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64)
    for upstream in (_upstream(x), -0.37 * _upstream(x) + 0.19):
        actual, expected = _gradient_comparison(module, x, upstream)
        assert len(actual) == 1 + len(dict(module.named_parameters()))
        assert len(expected) == len(actual)


def test_db15_probability_objective_gradients_reach_only_attention_probability_roles():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64).requires_grad_(True)
    _, actual_p = _run(module, x)
    coefficient = torch.tensor(
        [
            [
                [0.11 * (((h + 2 * t + 3 * k) % 7) - 3) if k <= t else 0.0 for k in range(4)]
                for t in range(4)
            ]
            for h in range(2)
        ],
        dtype=torch.float64,
        device=_TEST_DEVICE,
    )
    actual_parameters = dict(module.named_parameters())
    selected_names = [
        "attention_residual.normalized_input.gamma",
        "attention_residual.normalized_input.beta",
        *[
            f"attention_residual.multi_head_attention.heads.{h}.{projection}"
            for h in range(2)
            for projection in ("w_q", "w_k")
        ],
    ]
    actual_grads = torch.autograd.grad(
        torch.sum(actual_p * coefficient),
        (x, *(actual_parameters[name] for name in selected_names)),
    )
    reference_x = x.detach().clone().requires_grad_(True)
    reference_parameters = _clone_parameters(module, requires_grad=True)
    expected = _reference(reference_x, reference_parameters, _EPS)
    reference_grads = torch.autograd.grad(
        torch.sum(expected["p"] * coefficient),
        (reference_x, *(reference_parameters[name] for name in selected_names)),
    )
    assert not torch.allclose(expected["p"][:, 1, :2], expected["p"][:, 1, :2].flip(-1))
    for actual, wanted in zip(actual_grads, reference_grads):
        _assert_finite(actual)
        _assert_close(actual, wanted, gradient=True)


def test_db16_causal_output_derivatives_and_central_finite_differences():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    x = _fixture_input(_MAIN, torch.float64)
    upstream = _upstream(x)
    for t in (0, 1, 2):
        candidate_x = x.detach().clone().requires_grad_(True)
        y, _ = _run(module, candidate_x)
        actual = torch.autograd.grad(torch.sum(y[t] * upstream[t]), candidate_x)[0]
        reference_x = x.detach().clone().requires_grad_(True)
        reference = _reference(reference_x, _clone_parameters(module, requires_grad=True), _EPS)
        expected = torch.autograd.grad(
            torch.sum(reference["y"][t] * upstream[t]),
            reference_x,
        )[0]
        _assert_close(actual, expected, gradient=True)
        assert torch.equal(actual[t + 1 :], torch.zeros_like(actual[t + 1 :]))
    candidate_x = x.detach().clone().requires_grad_(True)
    y, _ = _run(module, candidate_x)
    parameters = dict(module.named_parameters())
    analytic = torch.autograd.grad(
        torch.sum(y * upstream),
        (candidate_x, *parameters.values()),
    )
    step = 1e-6

    plus_x = x.detach().clone()
    minus_x = x.detach().clone()
    plus_x[2, 3] += step
    minus_x[2, 3] -= step
    plus_value = torch.sum(_run(module, plus_x)[0] * upstream).item()
    minus_value = torch.sum(_run(module, minus_x)[0] * upstream).item()
    finite_difference = (plus_value - minus_value) / (2 * step)
    assert math.isclose(
        finite_difference,
        analytic[0][2, 3].item(),
        **_FINITE_DIFFERENCE_TOLERANCES,
    )
    for (name, parameter), gradient in zip(parameters.items(), analytic[1:]):
        flat_index = parameter.numel() // 2
        index = tuple(int(value.item()) for value in torch.unravel_index(torch.tensor(flat_index), parameter.shape))
        original = parameter[index].detach().clone()
        try:
            with torch.no_grad():
                parameter[index] = original + step
            plus_value = torch.sum(_run(module, x)[0] * upstream).item()
            with torch.no_grad():
                parameter[index] = original - step
            minus_value = torch.sum(_run(module, x)[0] * upstream).item()
        finally:
            with torch.no_grad():
                parameter[index] = original
        finite_difference = (plus_value - minus_value) / (2 * step)
        assert math.isclose(
            finite_difference,
            gradient[index].item(),
            **_FINITE_DIFFERENCE_TOLERANCES,
        ), name


def test_db17_registered_state_counts_identity_determinism_and_zero_grad():
    module = _new_block(torch.float64)
    _load_fixture(module, _fixture_values(_MAIN, torch.float64))
    named_parameters = dict(module.named_parameters())
    expected_parameter_count = 3 * _MAIN["num_heads"] + 9
    expected_scalar_count = (
        4 * _MAIN["d_model"]
        + _MAIN["num_heads"]
        * (2 * _MAIN["d_model"] * _MAIN["d_k"] + _MAIN["d_model"] * _MAIN["d_v"])
        + _MAIN["num_heads"] * _MAIN["d_v"] * _MAIN["d_model"]
        + 2 * _MAIN["d_model"] * _MAIN["d_ff"]
        + _MAIN["d_ff"]
        + _MAIN["d_model"]
    )
    assert len(named_parameters) == expected_parameter_count
    assert sum(parameter.numel() for parameter in named_parameters.values()) == expected_scalar_count
    assert all(parameter.requires_grad for parameter in named_parameters.values())
    assert len({id(parameter) for parameter in named_parameters.values()}) == expected_parameter_count
    assert len({parameter.untyped_storage().data_ptr() for parameter in named_parameters.values()}) == expected_parameter_count
    assert set(module.state_dict()) == set(named_parameters)
    head_parameters = [
        parameter
        for head in module.attention_residual.multi_head_attention.heads
        for parameter in head.parameters()
    ]
    assert len({id(parameter) for parameter in head_parameters}) == len(head_parameters)
    parameter_ids = {name: id(parameter) for name, parameter in named_parameters.items()}
    parameter_values = {name: parameter.detach().clone() for name, parameter in named_parameters.items()}
    x = _fixture_input(_MAIN, torch.float64)
    baseline_y, baseline_p = _run(module, x)
    repeated_y, repeated_p = _run(module, x)
    _assert_close(repeated_y, baseline_y)
    _assert_close(repeated_p, baseline_p)
    for tokens in (1, 2, 6):
        _run(module, _fixture_input(_MAIN, torch.float64, tokens=tokens))
    assert {name: id(parameter) for name, parameter in module.named_parameters()} == parameter_ids
    assert all(torch.equal(dict(module.named_parameters())[name], value) for name, value in parameter_values.items())
    loss = _run(module, x)[0].square().sum()
    loss.backward()
    assert all(parameter.grad is not None for parameter in module.parameters())
    module.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in module.parameters())


def test_db18_state_round_trip_dtype_and_module_lifecycle():
    source = _new_block(torch.float32)
    _load_fixture(source, _fixture_values(_MAIN, torch.float32))
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    destination = _new_block(torch.float32)
    result = destination.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert all(torch.equal(destination.state_dict()[name], value) for name, value in state.items())
    x32 = _fixture_input(_MAIN, torch.float32)
    source_y, source_p = _run(source, x32)
    destination_y, destination_p = _run(destination, x32)
    _assert_close(destination_y, source_y)
    _assert_close(destination_p, source_p)
    destination.to(dtype=torch.float64)
    y64, p64 = _run(destination, x32.to(dtype=torch.float64))
    assert y64.dtype == p64.dtype == torch.float64
    destination.to(dtype=torch.float32)
    y_back, p_back = _run(destination, x32)
    assert y_back.dtype == p_back.dtype == torch.float32
    _assert_close(y_back, source_y)
    _assert_close(p_back, source_p)
    destination.train()
    assert all(child.training for child in destination.modules())
    train_y, train_p = _run(destination, x32)
    destination.eval()
    assert all(not child.training for child in destination.modules())
    eval_y, eval_p = _run(destination, x32)
    _assert_close(train_y, eval_y)
    _assert_close(train_p, eval_p)
