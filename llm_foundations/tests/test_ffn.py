"""Verify the exact GELU activation and position-wise FFN contract."""

import math

import pytest
import torch

from llm_foundations.ffn import PositionWiseFFN, gelu


_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPES = (torch.float32, torch.float64)
_FORWARD_TOLERANCES = {
    torch.float32: {"atol": 1e-5, "rtol": 1e-5},
    torch.float64: {"atol": 1e-11, "rtol": 1e-10},
}
_GRADIENT_TOLERANCES = {
    torch.float32: {"atol": 3e-5, "rtol": 3e-5},
    torch.float64: {"atol": 1e-10, "rtol": 1e-9},
}
_FINITE_DIFFERENCE_TOLERANCES = {"atol": 2e-7, "rtol": 2e-5}
_SQRT_2 = math.sqrt(2.0)
_SQRT_2_PI = math.sqrt(2.0 * math.pi)
_X_A = [[1, 2], [-1, 1], [0, -2], [2, -1]]
_W1_A = [[1, -1, 0.5], [2, 0.5, -1]]
_B1_A = [-1, 0.25, 0.5]
_W2_A = [[1, -2], [0.5, 1], [-1, 0.25]]
_B2_A = [0.25, -0.5]
_G_A = [[1, -2], [0.5, 3], [-1, 0.25], [2, -0.5]]
_Y_A = [
    [4.483366859674, -8.389733862127],
    [1.248603491676, 1.140232662005],
    [-2.319462527085, -0.048848688596],
    [-2.406883622354, 0.410924403560],
]
_GELU_VALUES = [
    (-3, -0.004049694095),
    (-2, -0.045500263896),
    (-1, -0.158655253931),
    (-0.5, -0.154268769363),
    (0, 0),
    (0.5, 0.345731230637),
    (1, 0.841344746069),
    (2, 1.954499736104),
    (3, 2.995950305905),
]
_GELU_DERIVATIVES = [
    (-3, -0.011945647204),
    (-2, -0.085231801078),
    (-1, -0.083315470588),
    (-0.5, 0.132504875344),
    (0, 0.5),
    (0.5, 0.867495124656),
    (1, 1.083315470588),
    (2, 1.085231801078),
    (3, 1.011945647204),
]


@pytest.fixture(scope="module", autouse=True)
def _disable_reduced_precision():
    allow_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type=_TEST_DEVICE.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32


def _tensor(values, dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    return torch.tensor(values, dtype=dtype, device=device)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, *, gradient=False) -> None:
    tolerances = _GRADIENT_TOLERANCES if gradient else _FORWARD_TOLERANCES
    torch.testing.assert_close(actual, expected, **tolerances[actual.dtype])


def _assert_finite(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all().item()


def _assert_output_placement(output: torch.Tensor, x: torch.Tensor) -> None:
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def _new_module(d_model: int = 2, d_ff: int = 3, dtype: torch.dtype = torch.float64):
    return PositionWiseFFN(d_model=d_model, d_ff=d_ff).to(
        device=_TEST_DEVICE,
        dtype=dtype,
    )


def _fixture_values(dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    return {
        "w_1": _tensor(_W1_A, dtype, device),
        "b_1": _tensor(_B1_A, dtype, device),
        "w_2": _tensor(_W2_A, dtype, device),
        "b_2": _tensor(_B2_A, dtype, device),
    }


def _load_fixture(module: PositionWiseFFN) -> None:
    values = _fixture_values(module.w_1.dtype, module.w_1.device)
    parameters = dict(module.named_parameters())
    assert parameters.keys() == values.keys()
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(values[name])


def _fixture_module(dtype: torch.dtype = torch.float64) -> PositionWiseFFN:
    module = _new_module(dtype=dtype)
    _load_fixture(module)
    return module


def _fixture_input(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return _tensor(_X_A, dtype)


def _fixture_objective_weights(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return _tensor(_G_A, dtype)


def _reference_parameters(module: PositionWiseFFN, *, requires_grad=False):
    return {
        name: parameter.detach().clone().requires_grad_(requires_grad)
        for name, parameter in module.named_parameters()
    }


def _exact_gelu_reference(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (1 + torch.erf(value / _SQRT_2))


def _reference(x: torch.Tensor, parameters) -> torch.Tensor:
    hidden = x @ parameters["w_1"] + parameters["b_1"]
    activated = _exact_gelu_reference(hidden)
    return activated @ parameters["w_2"] + parameters["b_2"]


def _reference_for(module: PositionWiseFFN, x: torch.Tensor) -> torch.Tensor:
    return _reference(x, _reference_parameters(module))


def _indexed_fixture(d_model: int, d_ff: int, dtype: torch.dtype, tokens: int = 5):
    x = _tensor(
        [[((3 * t + 2 * i) % 7 - 3) / 4 for i in range(d_model)] for t in range(tokens)],
        dtype,
    )
    w_1 = _tensor(
        [[((2 * i + 3 * h) % 9 - 4) / 5 for h in range(d_ff)] for i in range(d_model)],
        dtype,
    )
    b_1 = _tensor([(h - 1) / 8 for h in range(d_ff)], dtype)
    w_2 = _tensor(
        [[((3 * h + 2 * j) % 7 - 3) / 4 for j in range(d_model)] for h in range(d_ff)],
        dtype,
    )
    b_2 = _tensor([(1 - j) / 6 for j in range(d_model)], dtype)
    return x, {"w_1": w_1, "b_1": b_1, "w_2": w_2, "b_2": b_2}


def _set_parameter(module: PositionWiseFFN, name: str, value: torch.Tensor) -> None:
    with torch.no_grad():
        dict(module.named_parameters())[name].copy_(value)


def _analytic_gradients(x: torch.Tensor, parameters, upstream: torch.Tensor):
    w_1 = parameters["w_1"]
    b_1 = parameters["b_1"]
    w_2 = parameters["w_2"]
    hidden = x @ w_1 + b_1
    activated = _exact_gelu_reference(hidden)
    cdf = 0.5 * (1 + torch.erf(hidden / _SQRT_2))
    density = torch.exp(-0.5 * hidden.square()) / _SQRT_2_PI
    hidden_gradient = (upstream @ w_2.T) * (cdf + hidden * density)
    return {
        "x": hidden_gradient @ w_1.T,
        "w_1": x.T @ hidden_gradient,
        "b_1": hidden_gradient.sum(dim=0),
        "w_2": activated.T @ upstream,
        "b_2": upstream.sum(dim=0),
    }


def _module_and_reference_gradients(
    module: PositionWiseFFN,
    x: torch.Tensor,
    upstream: torch.Tensor,
):
    module.zero_grad(set_to_none=True)
    core_x = x.detach().clone().requires_grad_(True)
    (module(core_x) * upstream).sum().backward()
    core_gradients = {"x": core_x.grad.detach().clone()}
    core_gradients.update(
        {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}
    )

    reference_x = x.detach().clone().requires_grad_(True)
    reference_parameters = _reference_parameters(module, requires_grad=True)
    (_reference(reference_x, reference_parameters) * upstream).sum().backward()
    reference_gradients = {"x": reference_x.grad.detach().clone()}
    reference_gradients.update(
        {name: parameter.grad.detach().clone() for name, parameter in reference_parameters.items()}
    )
    return core_gradients, reference_gradients


def _assert_gradient_groups(actual, expected) -> None:
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        actual_value = actual[name]
        assert actual_value is not None
        assert actual_value.shape == expected_value.shape
        assert actual_value.dtype == expected_value.dtype
        assert actual_value.device == expected_value.device
        _assert_finite(actual_value)
        _assert_close(actual_value, expected_value, gradient=True)


def _parameter_snapshot(module: PositionWiseFFN):
    return {
        name: {
            "identity": id(parameter),
            "storage": parameter.untyped_storage().data_ptr(),
            "value": parameter.detach().clone(),
        }
        for name, parameter in module.named_parameters()
    }


def _assert_parameter_snapshot(module: PositionWiseFFN, snapshot) -> None:
    current = dict(module.named_parameters())
    assert current.keys() == snapshot.keys()
    for name, parameter in current.items():
        previous = snapshot[name]
        assert id(parameter) == previous["identity"]
        assert parameter.untyped_storage().data_ptr() == previous["storage"]
        assert torch.equal(parameter, previous["value"])


def _scalar_gelu(value: float) -> float:
    return 0.5 * value * (1 + math.erf(value / _SQRT_2))


def _scalar_gelu_derivative(value: float) -> float:
    cdf = 0.5 * (1 + math.erf(value / _SQRT_2))
    density = math.exp(-0.5 * value * value) / _SQRT_2_PI
    return cdf + value * density


def test_ffn01_exact_gelu_mixed_sign_values_float64_and_float32():
    values = [value for value, _ in _GELU_VALUES]
    x64 = _tensor(values, torch.float64)
    expected64 = _tensor([value for _, value in _GELU_VALUES], torch.float64)
    output64 = gelu(x64)
    _assert_output_placement(output64, x64)
    _assert_close(output64, expected64)

    x32 = _tensor(values, torch.float32)
    output32 = gelu(x32)
    _assert_output_placement(output32, x32)
    _assert_close(output32, _exact_gelu_reference(x32))


def test_ffn02_tight_exact_erf_gelu_rejects_tanh_approximation():
    values = [-3, -2, -1, 1, 2, 3]
    x = _tensor(values, torch.float64)
    expected = _tensor([_scalar_gelu(value) for value in values], torch.float64)
    _assert_close(gelu(x), expected)


def test_ffn03_gelu_pair_identity_and_position_independence():
    positive = [0.25, 0.75, 1.5, 3.0]
    values = [item for value in positive for item in (value, -value)]
    x = _tensor([values, values, list(reversed(values))], torch.float64)
    output = gelu(x)
    for index, value in enumerate(positive):
        assert math.isclose((output[0, 2 * index] - output[0, 2 * index + 1]).item(), value)
    assert torch.equal(output[0], output[1])
    permutation = torch.tensor([2, 0, 1], device=_TEST_DEVICE)
    assert torch.equal(gelu(x.index_select(0, permutation)), output.index_select(0, permutation))


def test_ffn04_exact_gelu_input_gradient_matches_analytic_derivative():
    values = [value for value, _ in _GELU_DERIVATIVES]
    coefficients = [1, -2, 0.5, 3, -1, 2, 0.25, -0.75, 1.5]
    x = _tensor(values, torch.float64).requires_grad_(True)
    upstream = _tensor(coefficients, torch.float64)
    (gelu(x) * upstream).sum().backward()
    expected = _tensor(
        [scale * _scalar_gelu_derivative(value) for value, scale in zip(values, coefficients)],
        torch.float64,
    )
    _assert_gradient_groups({"gradient": x.grad}, {"gradient": expected})


def test_ffn05_gelu_zero_has_zero_value_and_half_local_derivative():
    x = _tensor([0.0], torch.float64).requires_grad_(True)
    output = gelu(x)
    (2 * output).sum().backward()
    assert output.item() == 0
    assert x.grad.item() == 1


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ffn06_gelu_finite_negative_and_positive_tails(dtype):
    values = [-20, -10, -8, -5, 5, 8, 10, 20]
    x = _tensor(values, dtype).requires_grad_(True)
    output = gelu(x)
    output.sum().backward()
    _assert_finite(output)
    _assert_finite(x.grad)
    _assert_close(output, _exact_gelu_reference(x))
    expected_gradient = 0.5 * (1 + torch.erf(x.detach() / _SQRT_2)) + x.detach() * torch.exp(
        -0.5 * x.detach().square()
    ) / _SQRT_2_PI
    _assert_close(x.grad, expected_gradient, gradient=True)
    if dtype == torch.float64:
        assert math.isclose(output[3].item(), -1.433257859e-6, abs_tol=1e-12, rel_tol=1e-8)


def test_ffn07_rectangular_affine_fixture_matches_hand_values_and_exact_oracle():
    module = _fixture_module(torch.float64)
    x = _fixture_input(torch.float64)
    expected = _tensor(_Y_A, torch.float64)
    output = module(x)
    _assert_output_placement(output, x)
    _assert_close(output, expected)
    _assert_close(output, _reference_for(module, x))

    module32 = _fixture_module(torch.float32)
    x32 = _fixture_input(torch.float32)
    _assert_close(module32(x32), _reference_for(module32, x32))


def test_ffn08_nonleading_input_feature_changes_selected_output_row():
    module = _fixture_module()
    x = _fixture_input()
    changed_x = x.clone()
    changed_x[1, 1] += 0.5
    actual = module(changed_x)
    expected = _reference_for(module, changed_x)
    _assert_close(actual, expected)
    assert not torch.allclose(actual[1], module(x)[1], **_FORWARD_TOLERANCES[x.dtype])


def test_ffn09_nonleading_weight_entries_affect_independently_verified_output():
    module = _fixture_module()
    x = _fixture_input()
    parameters = dict(module.named_parameters())
    for name, index, delta in (("w_1", (1, 2), 0.75), ("w_2", (2, 1), -0.5)):
        baseline = parameters[name][index].detach().clone()
        baseline_output = module(x).detach().clone()
        try:
            with torch.no_grad():
                parameters[name][index].add_(delta)
            actual = module(x)
            _assert_close(actual, _reference_for(module, x))
            assert not torch.allclose(actual, baseline_output, **_FORWARD_TOLERANCES[x.dtype])
        finally:
            with torch.no_grad():
                parameters[name][index].copy_(baseline)


def test_ffn10_zero_input_uses_first_bias_and_broadcasts_across_tokens():
    module = _fixture_module()
    x = torch.zeros((4, 2), dtype=torch.float64, device=_TEST_DEVICE)
    output = module(x)
    _assert_close(output, _reference_for(module, x))
    assert torch.equal(output, output[:1].expand_as(output))


def test_ffn11_second_bias_and_projection_are_used_without_residual():
    module = _fixture_module()
    x = _fixture_input()
    original_w2 = module.w_2.detach().clone()
    original_b2 = module.b_2.detach().clone()
    with torch.no_grad():
        module.w_2.zero_()
    output = module(x)
    expected_bias = original_b2.expand_as(output)
    _assert_close(output, expected_bias)

    with torch.no_grad():
        module.w_2.copy_(original_w2)
        module.b_2.zero_()
    output_without_second_bias = module(x)
    _assert_close(output_without_second_bias, _reference_for(module, x))
    assert not torch.allclose(output_without_second_bias, x, **_FORWARD_TOLERANCES[x.dtype])


def test_ffn12_final_affine_outputs_remain_signed():
    module = _fixture_module()
    output = module(_fixture_input())
    assert output[2, 0].item() < 0
    assert output[2, 1].item() < 0
    assert output[3, 0].item() < 0
    _assert_close(output, _tensor(_Y_A, torch.float64))


@pytest.mark.parametrize("tokens,d_model,d_ff", [(1, 2, 3), (3, 1, 4), (5, 3, 7)])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_ffn13_rectangular_shapes_match_indexed_independent_reference(tokens, d_model, d_ff, dtype):
    x, values = _indexed_fixture(d_model, d_ff, dtype, tokens=tokens)
    module = _new_module(d_model, d_ff, dtype)
    for name, value in values.items():
        _set_parameter(module, name, value)
    _assert_close(module(x), _reference(x, values))


@pytest.mark.parametrize("tokens,d_model,d_ff", [(3, 4, 1), (2, 3, 3), (4, 5, 2)])
def test_ffn14_singleton_equal_and_bottleneck_hidden_widths_are_supported(tokens, d_model, d_ff):
    x, values = _indexed_fixture(d_model, d_ff, torch.float64, tokens=tokens)
    module = _new_module(d_model, d_ff)
    for name, value in values.items():
        _set_parameter(module, name, value)
    _assert_close(module(x), _reference(x, values))


def test_ffn15_activation_makes_scaling_input_nonlinear():
    module = _fixture_module()
    x = _fixture_input()
    output = module(x)
    scaled_output = module(2 * x)
    _assert_close(output, _reference_for(module, x))
    _assert_close(scaled_output, _reference_for(module, 2 * x))
    assert not torch.allclose(scaled_output, 2 * output, **_FORWARD_TOLERANCES[x.dtype])


def test_ffn16_changing_one_token_does_not_change_other_outputs():
    module = _fixture_module()
    x = _fixture_input()
    baseline = module(x)
    changed_x = x.clone()
    changed_x[2] += _tensor([0.5, -1], x.dtype)
    actual = module(changed_x)
    _assert_close(actual, _reference_for(module, changed_x))
    _assert_close(actual[[0, 1, 3]], baseline[[0, 1, 3]])
    assert not torch.allclose(actual[2], baseline[2], **_FORWARD_TOLERANCES[x.dtype])


def test_ffn17_permuting_tokens_permutates_outputs_identically():
    module = _fixture_module()
    x = _fixture_input()
    permutation = torch.tensor([2, 0, 3, 1], device=_TEST_DEVICE)
    baseline = module(x)
    permuted = module(x.index_select(0, permutation))
    _assert_close(permuted, baseline.index_select(0, permutation))


def test_ffn18_slicing_and_duplicate_rows_preserve_positionwise_outputs():
    module = _fixture_module()
    x = _fixture_input()
    whole = module(x)
    sliced = torch.cat((module(x[:1]), module(x[1:3]), module(x[3:])), dim=0)
    _assert_close(whole, sliced)
    duplicated = module(torch.cat((x, x[:1]), dim=0))
    _assert_close(duplicated[-1], duplicated[0])


def test_ffn19_reuses_fixed_registered_parameters_across_sequence_lengths():
    module = _fixture_module()
    snapshot = _parameter_snapshot(module)
    state_shapes = {name: tuple(value.shape) for name, value in module.state_dict().items()}
    for length in (1, 4, 7):
        x, _ = _indexed_fixture(2, 3, torch.float64, tokens=length)
        _assert_close(module(x), _reference_for(module, x))
        assert {name: tuple(value.shape) for name, value in module.state_dict().items()} == state_shapes
        _assert_parameter_snapshot(module, snapshot)


def test_ffn20_forward_and_backward_do_not_mutate_inputs_or_parameters():
    module = _fixture_module()
    x = _fixture_input().requires_grad_(True)
    input_before = x.detach().clone()
    parameter_before = {name: value.detach().clone() for name, value in module.named_parameters()}
    (module(x) * _fixture_objective_weights()).sum().backward()
    assert torch.equal(x.detach(), input_before)
    for name, parameter in module.named_parameters():
        assert torch.equal(parameter, parameter_before[name])


def test_ffn21_noncontiguous_transposed_input_matches_contiguous_reference():
    module = _fixture_module()
    x = _tensor([[1, -1, 0, 2], [2, 1, -2, -1]], torch.float64).T
    assert not x.is_contiguous()
    output = module(x)
    _assert_close(output, module(x.contiguous()))
    _assert_close(output, _reference_for(module, x))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ffn22_output_and_registered_state_preserve_dtype_and_device(dtype):
    module = _fixture_module(dtype)
    x = _fixture_input(dtype)
    output = module(x)
    _assert_output_placement(output, x)
    for parameter in module.parameters():
        assert parameter.dtype == dtype
        assert parameter.device == _TEST_DEVICE


def test_ffn23_autograd_matches_independent_exact_erf_reference_in_both_dtypes():
    for dtype in _DTYPES:
        module = _fixture_module(dtype)
        x = _fixture_input(dtype)
        upstream = _fixture_objective_weights(dtype)
        actual, expected = _module_and_reference_gradients(module, x, upstream)
        _assert_gradient_groups(actual, expected)


def test_ffn24_all_parameter_and_input_gradients_match_analytic_chain_rule():
    for dtype in _DTYPES:
        module = _fixture_module(dtype)
        x = _fixture_input(dtype)
        upstream = _fixture_objective_weights(dtype)
        actual, _ = _module_and_reference_gradients(module, x, upstream)
        analytic = _analytic_gradients(x, _reference_parameters(module), upstream)
        _assert_gradient_groups(actual, analytic)


def test_ffn25_zero_preactivation_has_half_derivative_and_expected_gradients():
    module = _fixture_module()
    x = _tensor([[1, -2], [3, 1]], torch.float64).requires_grad_(True)
    upstream = _tensor([[1, -1], [2, 0.5]], torch.float64)
    with torch.no_grad():
        module.w_1.zero_()
        module.b_1.zero_()
        module.w_2.copy_(_tensor([[1, 2], [-1, 0.5], [0.25, -2]], torch.float64))
        module.b_2.copy_(_tensor([0.75, -0.5], torch.float64))
    output = module(x)
    _assert_close(output, _tensor([[0.75, -0.5], [0.75, -0.5]], torch.float64))
    (output * upstream).sum().backward()
    expected = {
        "x": torch.zeros((2, 2), dtype=torch.float64, device=_TEST_DEVICE),
        "w_1": _tensor([[4, -3.375, 0.375], [2.5, 0.625, -2.5]], torch.float64),
        "b_1": _tensor([1, -1.625, 0.875], torch.float64),
        "w_2": torch.zeros((3, 2), dtype=torch.float64, device=_TEST_DEVICE),
        "b_2": _tensor([3, -0.5], torch.float64),
    }
    actual = {"x": x.grad}
    actual.update({name: parameter.grad for name, parameter in module.named_parameters()})
    for name, expected_value in expected.items():
        torch.testing.assert_close(actual[name], expected_value, atol=1e-12, rtol=0)


def test_ffn26_central_finite_differences_check_every_input_and_parameter_element():
    module = _fixture_module(torch.float64)
    x = _fixture_input(torch.float64).requires_grad_(True)
    upstream = _fixture_objective_weights(torch.float64)
    (module(x) * upstream).sum().backward()
    analytic = {"x": x.grad.detach().clone()}
    analytic.update(
        {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}
    )
    coordinates = [("x", x)] + list(module.named_parameters())
    step = 1e-6
    for name, tensor in coordinates:
        for index in torch.cartesian_prod(
            *[torch.arange(size, device=_TEST_DEVICE) for size in tensor.shape]
        ).reshape(-1, tensor.ndim):
            location = tuple(int(value.item()) for value in index)
            original = tensor[location].detach().clone()
            try:
                with torch.no_grad():
                    tensor[location] = original + step
                    with torch.no_grad():
                        plus = (module(x) * upstream).sum().item()
                    tensor[location] = original - step
                    with torch.no_grad():
                        minus = (module(x) * upstream).sum().item()
            finally:
                with torch.no_grad():
                    tensor[location] = original
            numerical = (plus - minus) / (2 * step)
            torch.testing.assert_close(
                torch.tensor(numerical, dtype=torch.float64, device=_TEST_DEVICE),
                analytic[name][location],
                **_FINITE_DIFFERENCE_TOLERANCES,
            )


def test_ffn27_single_output_row_loss_has_no_other_input_row_gradient():
    module = _fixture_module()
    x = _fixture_input()
    coefficients = _tensor([1.5, -0.75], torch.float64)
    module.zero_grad(set_to_none=True)
    core_x = x.detach().clone().requires_grad_(True)
    (module(core_x)[1] * coefficients).sum().backward()
    actual = {"x": core_x.grad.detach().clone()}
    actual.update(
        {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}
    )

    reference_x = x.detach().clone().requires_grad_(True)
    reference_parameters = _reference_parameters(module, requires_grad=True)
    (_reference(reference_x, reference_parameters)[1] * coefficients).sum().backward()
    expected = {"x": reference_x.grad.detach().clone()}
    expected.update(
        {name: parameter.grad.detach().clone() for name, parameter in reference_parameters.items()}
    )
    _assert_gradient_groups(actual, expected)
    assert torch.equal(actual["x"][[0, 2, 3]], torch.zeros_like(actual["x"][[0, 2, 3]]))

def test_ffn28_duplicated_sequence_doubles_shared_parameter_gradients():
    module = _fixture_module()
    x = _fixture_input()
    baseline_x = x.detach().clone().requires_grad_(True)
    module(baseline_x).sum().backward()
    baseline_parameters = {
        name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()
    }
    original_x_gradient = baseline_x.grad.detach().clone()

    module.zero_grad(set_to_none=True)
    duplicated_x = torch.cat((x, x), dim=0).requires_grad_(True)
    module(duplicated_x).sum().backward()
    _assert_close(duplicated_x.grad[: x.shape[0]], original_x_gradient, gradient=True)
    _assert_close(duplicated_x.grad[x.shape[0] :], original_x_gradient, gradient=True)
    for name, parameter in module.named_parameters():
        _assert_close(parameter.grad, 2 * baseline_parameters[name], gradient=True)
def test_ffn29_four_affine_roles_are_registered_trainable_and_persistent():
    module = _new_module(2, 3, torch.float64)
    parameters = dict(module.named_parameters())
    expected_shapes = {
        "w_1": (2, 3),
        "b_1": (3,),
        "w_2": (3, 2),
        "b_2": (2,),
    }
    assert parameters.keys() == expected_shapes.keys()
    for name, parameter in parameters.items():
        assert isinstance(parameter, torch.nn.Parameter)
        assert parameter.requires_grad
        assert tuple(parameter.shape) == expected_shapes[name]
    assert sum(parameter.numel() for parameter in parameters.values()) == 2 * 2 * 3 + 3 + 2
    assert set(module.state_dict()) == set(expected_shapes)


def test_ffn30_calls_keep_parameter_identity_and_observe_in_place_updates():
    module = _fixture_module()
    x = _fixture_input()
    snapshot = _parameter_snapshot(module)
    first = module(x)
    second = module(x)
    assert torch.equal(first, second)
    _assert_parameter_snapshot(module, snapshot)

    original_bias = module.b_2.detach().clone()
    try:
        with torch.no_grad():
            module.b_2[0].add_(0.75)
        changed = module(x)
        _assert_close(changed, _reference_for(module, x))
        assert not torch.allclose(changed, first, **_FORWARD_TOLERANCES[x.dtype])
    finally:
        with torch.no_grad():
            module.b_2.copy_(original_bias)
    _assert_parameter_snapshot(module, snapshot)


def test_ffn31_state_dict_round_trip_preserves_fixed_input_output():
    module = _fixture_module()
    x = _fixture_input()
    state = {name: value.detach().clone() for name, value in module.state_dict().items()}
    restored = _new_module()
    restored.load_state_dict(state)
    _assert_close(restored(x), _reference_for(module, x))
    _assert_close(restored(x), module(x))


def test_ffn32_train_and_eval_modes_preserve_exact_ffn_behavior():
    module = _fixture_module()
    x = _fixture_input()
    module.train()
    training_output = module(x)
    module.eval()
    evaluation_output = module(x)
    _assert_close(training_output, _reference_for(module, x))
    _assert_close(evaluation_output, _reference_for(module, x))
    _assert_close(training_output, evaluation_output)


def test_ffn33_public_gelu_helper_preserves_shape_values_input_and_gradients():
    x = _tensor([[0, -1, 2], [0.5, -3, 1]], torch.float64).requires_grad_(True)
    before = x.detach().clone()
    output = gelu(x)
    _assert_output_placement(output, x)
    _assert_close(output, _exact_gelu_reference(x))
    assert torch.equal(x.detach(), before)
    upstream = _tensor([[1, -2, 0.5], [3, -1, 0.25]], torch.float64)
    (output * upstream).sum().backward()
    expected_gradient = upstream * (
        0.5 * (1 + torch.erf(x.detach() / _SQRT_2))
        + x.detach() * torch.exp(-0.5 * x.detach().square()) / _SQRT_2_PI
    )
    _assert_close(x.grad, expected_gradient, gradient=True)
