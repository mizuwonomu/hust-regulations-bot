"""Kiểm tra độc lập hợp đồng forward, lifecycle và autograd của LayerNorm."""

import math

import pytest
import torch
from torch import nn

from llm_foundations.layer_norm import LayerNorm


_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPES = (torch.float32, torch.float64)
_TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-8, "atol": 1e-9},
}
_LN4_X = [[1, 2, 4, 8], [-3, 0, 1, 6], [9, 7, 2, -2]]
_LN4_GAMMA = [2, -1, 0.5, 0]
_LN4_BETA = [0.25, -0.75, 1.5, -2]
_LN4_EPS = 0.25


def _tensor(values, dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    return torch.tensor(values, dtype=dtype, device=device)


def _module_pair(d_model: int, eps: float, dtype: torch.dtype):
    candidate = LayerNorm(d_model=d_model, eps=eps).to(device=_TEST_DEVICE, dtype=dtype)
    reference = nn.LayerNorm(
        d_model,
        eps=eps,
        elementwise_affine=True,
    ).to(device=_TEST_DEVICE, dtype=dtype)
    return candidate, reference


def _set_affine(candidate, reference, gamma, beta, dtype: torch.dtype) -> None:
    gamma_values = _tensor(gamma, dtype)
    beta_values = _tensor(beta, dtype)
    with torch.no_grad():
        candidate.gamma.copy_(gamma_values)
        candidate.beta.copy_(beta_values)
        reference.weight.copy_(gamma_values)
        reference.bias.copy_(beta_values)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _assert_finite(value: torch.Tensor) -> None:
    assert torch.isfinite(value).all().item()


def _assert_output_placement(output: torch.Tensor, x: torch.Tensor) -> None:
    assert isinstance(output, torch.Tensor)
    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert output.device == x.device


def _assert_gradient(actual, expected: torch.Tensor) -> None:
    assert actual is not None
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    _assert_finite(actual)
    _assert_close(actual, expected)


def _parameter_snapshot(module):
    return {
        name: {
            "identity": id(parameter),
            "storage": parameter.untyped_storage().data_ptr(),
            "value": parameter.detach().clone(),
        }
        for name, parameter in module.named_parameters()
    }


def _assert_parameter_snapshot(module, snapshot) -> None:
    current = dict(module.named_parameters())
    assert current.keys() == snapshot.keys()
    for name, parameter in current.items():
        previous = snapshot[name]
        assert id(parameter) == previous["identity"]
        assert parameter.untyped_storage().data_ptr() == previous["storage"]
        assert torch.equal(parameter, previous["value"])


def _ln4_pair(dtype: torch.dtype):
    candidate, reference = _module_pair(4, _LN4_EPS, dtype)
    _set_affine(candidate, reference, _LN4_GAMMA, _LN4_BETA, dtype)
    return candidate, reference


@pytest.mark.parametrize("dtype", _DTYPES)
def test_public_api_initialization_registration_and_output_placement(dtype):
    default = LayerNorm(d_model=4).to(device=_TEST_DEVICE, dtype=dtype)
    custom = LayerNorm(d_model=4, eps=0.25).to(device=_TEST_DEVICE, dtype=dtype)

    assert isinstance(default, nn.Module)
    assert default.d_model == 4
    assert default.eps == 1e-5
    assert custom.d_model == 4
    assert custom.eps == 0.25

    for module in (default, custom):
        assert isinstance(module.gamma, nn.Parameter)
        assert isinstance(module.beta, nn.Parameter)
        assert module.gamma.requires_grad
        assert module.beta.requires_grad
        assert module.gamma.shape == (4,)
        assert module.beta.shape == (4,)
        assert torch.equal(module.gamma, torch.ones(4, dtype=dtype, device=_TEST_DEVICE))
        assert torch.equal(module.beta, torch.zeros(4, dtype=dtype, device=_TEST_DEVICE))
        assert module.gamma is not module.beta
        assert module.gamma.untyped_storage().data_ptr() != module.beta.untyped_storage().data_ptr()
        assert set(dict(module.named_parameters())) == {"gamma", "beta"}
        assert sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad) == 8
        assert set(module.state_dict()) == {"gamma", "beta"}

    x = _tensor([[1, 2, 4, 8], [-3, 0, 1, 6], [9, 7, 2, -2]], dtype)
    output = default(x)
    _assert_output_placement(output, x)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_hand_computed_population_variance_and_default_epsilon(dtype):
    x = _tensor([[2, 4, 6]], dtype)
    expected_default = _tensor(
        [[-2 / math.sqrt(8 / 3 + 1e-5), 0, 2 / math.sqrt(8 / 3 + 1e-5)]],
        dtype,
    )
    default = LayerNorm(d_model=3).to(device=_TEST_DEVICE, dtype=dtype)
    default_output = default(x)
    _assert_finite(default_output)
    _assert_close(default_output, expected_default)

    custom = LayerNorm(d_model=3, eps=1 / 3).to(device=_TEST_DEVICE, dtype=dtype)
    expected_custom = _tensor([[-2 / math.sqrt(3), 0, 2 / math.sqrt(3)]], dtype)
    custom_output = custom(x)
    _assert_finite(custom_output)
    _assert_close(custom_output, expected_custom)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_constant_rows_and_singleton_features(dtype):
    x = _tensor([[5, 5, 5], [-2, -2, -2], [0, 0, 0]], dtype)
    candidate, reference = _module_pair(3, 1e-5, dtype)
    output = candidate(x)
    _assert_finite(output)
    assert torch.equal(output, torch.zeros_like(x))

    gamma = [2, -1, 0.5]
    beta = [0.25, -0.75, 1.5]
    _set_affine(candidate, reference, gamma, beta, dtype)
    affine_output = candidate(x)
    expected = _tensor([beta, beta, beta], dtype)
    _assert_finite(affine_output)
    _assert_close(affine_output, expected)

    singleton_x = _tensor([[-7], [0], [12]], dtype)
    singleton, singleton_reference = _module_pair(1, 1e-5, dtype)
    _set_affine(singleton, singleton_reference, [3], [-0.4], dtype)
    singleton_output = singleton(singleton_x)
    _assert_finite(singleton_output)
    _assert_close(singleton_output, _tensor([[-0.4], [-0.4], [-0.4]], dtype))

    one_token_one_feature_x = _tensor([[1]], dtype)
    one_token_one_feature = singleton(one_token_one_feature_x)
    _assert_finite(one_token_one_feature)
    _assert_close(one_token_one_feature, _tensor([[-0.4]], dtype))
    _assert_output_placement(one_token_one_feature, one_token_one_feature_x)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_rectangular_multi_token_affine_matches_independent_oracle(dtype):
    candidate, reference = _ln4_pair(dtype)
    x = _tensor(_LN4_X, dtype)
    reference_x = x.clone()

    output = candidate(x)
    expected = reference(reference_x)
    _assert_finite(output)
    _assert_finite(expected)
    _assert_output_placement(output, x)
    _assert_close(output, expected)
    _assert_close(output[:, 3], _tensor([_LN4_BETA[3]] * 3, dtype))

    single_candidate = candidate(x[1:2].clone())
    single_reference = reference(reference_x[1:2].clone())
    _assert_close(single_candidate, single_reference)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_token_independence_permutation_translation_and_single_row_calls(dtype):
    candidate, reference = _ln4_pair(dtype)
    x = _tensor(_LN4_X, dtype)
    baseline = candidate(x)

    changed_x = x.clone()
    changed_x[1] = _tensor([20, -5, 3, 9], dtype)
    changed_output = candidate(changed_x)
    changed_reference = reference(changed_x.clone())
    _assert_close(changed_output[[0, 2]], baseline[[0, 2]])
    assert not torch.allclose(changed_output[1], baseline[1], **_TOLERANCES[dtype])
    _assert_close(changed_output, changed_reference)

    permutation = torch.tensor([2, 0, 1], device=_TEST_DEVICE)
    permuted_output = candidate(x.index_select(0, permutation))
    _assert_close(permuted_output, baseline.index_select(0, permutation))

    offsets = _tensor([[10], [-3], [5]], dtype)
    translated_output = candidate(x + offsets)
    _assert_close(translated_output, baseline)

    separate_rows = torch.cat([candidate(x[index : index + 1]) for index in range(x.shape[0])])
    _assert_close(baseline, separate_rows)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_epsilon_inside_sqrt_and_near_constant_row(dtype):
    candidate, reference = _module_pair(2, 3.0, dtype)
    x = _tensor([[0, 2]], dtype)
    output = candidate(x)
    expected = _tensor([[-0.5, 0.5]], dtype)
    _assert_finite(output)
    _assert_close(output, expected)
    _assert_close(output, reference(x.clone()))

    candidate, reference = _module_pair(2, 0.0001, dtype)
    near_constant = _tensor([[-0.001, 0.001]], dtype)
    near_output = candidate(near_constant)
    reference_output = reference(near_constant.clone())
    hand_expected = _tensor(
        [[-0.001 / math.sqrt(0.000001 + 0.0001), 0.001 / math.sqrt(0.000001 + 0.0001)]],
        dtype,
    )
    _assert_finite(near_output)
    _assert_close(near_output, hand_expected)
    _assert_close(near_output, reference_output)
    expected_population_variance = 0.000001 / 0.000101
    actual_population_variance = near_output.var(dim=1, unbiased=False)
    _assert_close(actual_population_variance, _tensor([expected_population_variance], dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_parameter_lifecycle_repeatability_and_module_modes(dtype):
    candidate, reference = _ln4_pair(dtype)
    snapshot = _parameter_snapshot(candidate)
    names = set(dict(candidate.named_parameters()))
    trainable_elements = sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad)
    state_keys = set(candidate.state_dict())
    inputs = [
        _tensor(_LN4_X, dtype),
        _tensor([[-3, 0, 1, 6]], dtype),
        _tensor([[1, 2, 4, 8], [-3, 0, 1, 6], [9, 7, 2, -2], [0, 3, -2, 5], [4, -1, 6, 2]], dtype),
    ]

    outputs = []
    for x in inputs:
        output = candidate(x)
        expected = reference(x.clone())
        _assert_finite(output)
        _assert_close(output, expected)
        outputs.append(output)
        _assert_parameter_snapshot(candidate, snapshot)
        assert set(dict(candidate.named_parameters())) == names
        assert sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad) == trainable_elements
        assert set(candidate.state_dict()) == state_keys

    repeated = candidate(inputs[0])
    _assert_close(repeated, outputs[0])
    _assert_parameter_snapshot(candidate, snapshot)
    assert set(candidate.state_dict()) == state_keys
    candidate.eval()
    evaluation_output = candidate(inputs[0])
    _assert_close(evaluation_output, outputs[0])
    _assert_parameter_snapshot(candidate, snapshot)
    assert set(candidate.state_dict()) == state_keys
    candidate.train()
    training_output = candidate(inputs[0])
    _assert_close(training_output, evaluation_output)
    _assert_parameter_snapshot(candidate, snapshot)
    assert set(candidate.state_dict()) == state_keys

    independent = LayerNorm(d_model=4, eps=_LN4_EPS).to(device=_TEST_DEVICE, dtype=dtype)
    with torch.no_grad():
        independent.gamma.add_(1)
        independent.beta.sub_(1)
    _assert_parameter_snapshot(candidate, snapshot)
    assert set(candidate.state_dict()) == state_keys
    assert not torch.equal(candidate.gamma, independent.gamma)
    assert not torch.equal(candidate.beta, independent.beta)
    assert candidate.gamma.untyped_storage().data_ptr() != independent.gamma.untyped_storage().data_ptr()
    assert candidate.beta.untyped_storage().data_ptr() != independent.beta.untyped_storage().data_ptr()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_forward_preserves_inputs_parameters_and_existing_or_empty_grad_buffers(dtype):
    candidate, reference = _ln4_pair(dtype)
    x = _tensor(_LN4_X, dtype).requires_grad_()
    x_before = x.detach().clone()
    gamma_before = candidate.gamma.detach().clone()
    beta_before = candidate.beta.detach().clone()
    x.grad = _tensor([[0.2, -0.1, 0.4, 0.3], [-0.3, 0.5, 0.1, -0.2], [0.7, 0.2, -0.4, 0.6]], dtype)
    candidate.gamma.grad = _tensor([0.1, -0.2, 0.3, -0.4], dtype)
    candidate.beta.grad = _tensor([-0.5, 0.6, -0.7, 0.8], dtype)
    gradient_snapshot = {
        "x": x.grad.clone(),
        "gamma": candidate.gamma.grad.clone(),
        "beta": candidate.beta.grad.clone(),
    }

    output = candidate(x)
    expected = reference(x.detach().clone())
    _assert_finite(output)
    _assert_close(output, expected)
    assert torch.equal(x, x_before)
    assert torch.equal(candidate.gamma, gamma_before)
    assert torch.equal(candidate.beta, beta_before)
    assert torch.equal(x.grad, gradient_snapshot["x"])
    assert torch.equal(candidate.gamma.grad, gradient_snapshot["gamma"])
    assert torch.equal(candidate.beta.grad, gradient_snapshot["beta"])

    fresh, _ = _ln4_pair(dtype)
    fresh_x = _tensor(_LN4_X, dtype).requires_grad_()
    assert fresh_x.grad is None
    assert fresh.gamma.grad is None
    assert fresh.beta.grad is None
    fresh(fresh_x)
    assert fresh_x.grad is None
    assert fresh.gamma.grad is None
    assert fresh.beta.grad is None

    noncontiguous_candidate, noncontiguous_reference = _ln4_pair(dtype)
    backing = _tensor(
        [[1, 91, 2, 92, 4, 93, 8, 94], [-3, 91, 0, 92, 1, 93, 6, 94], [9, 91, 7, 92, 2, 93, -2, 94]],
        dtype,
    )
    backing_before = backing.clone()
    view = backing[:, ::2]
    assert view.shape == (3, 4)
    assert not view.is_contiguous()
    view_before = view.clone()
    gamma_before = noncontiguous_candidate.gamma.detach().clone()
    beta_before = noncontiguous_candidate.beta.detach().clone()
    noncontiguous_output = noncontiguous_candidate(view)
    noncontiguous_expected = noncontiguous_reference(view.clone())
    _assert_finite(noncontiguous_output)
    _assert_close(noncontiguous_output, noncontiguous_expected)
    assert torch.equal(view, view_before)
    assert torch.equal(backing, backing_before)
    assert torch.equal(noncontiguous_candidate.gamma, gamma_before)
    assert torch.equal(noncontiguous_candidate.beta, beta_before)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_non_degenerate_autograd_matches_independent_oracle(dtype):
    candidate, reference = _ln4_pair(dtype)
    gamma = [1.2, -0.7, 2, 0.3]
    beta = [0.25, -0.75, 1.5, -2]
    _set_affine(candidate, reference, gamma, beta, dtype)
    candidate_x = _tensor(_LN4_X, dtype).requires_grad_()
    reference_x = _tensor(_LN4_X, dtype).requires_grad_()
    coefficients = _tensor(
        [[0.3, -1.2, 2.1, 0.7], [1.4, 0.2, -0.9, 2.3], [-0.6, 1.7, 0.5, -1.1]],
        dtype,
    )
    candidate_values = {
        "x": candidate_x.detach().clone(),
        "gamma": candidate.gamma.detach().clone(),
        "beta": candidate.beta.detach().clone(),
    }
    reference_values = {
        "x": reference_x.detach().clone(),
        "gamma": reference.weight.detach().clone(),
        "beta": reference.bias.detach().clone(),
    }

    candidate_output = candidate(candidate_x)
    reference_output = reference(reference_x)
    candidate_objective = (candidate_output * coefficients).sum()
    reference_objective = (reference_output * coefficients).sum()
    candidate_objective.backward()
    reference_objective.backward()

    _assert_finite(candidate_output)
    _assert_finite(candidate_objective)
    _assert_finite(candidate_x.grad)
    _assert_finite(candidate.gamma.grad)
    _assert_finite(candidate.beta.grad)
    _assert_close(candidate_output, reference_output)
    _assert_close(candidate_objective, reference_objective)
    _assert_gradient(candidate_x.grad, reference_x.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    assert not torch.equal(reference_x.grad, torch.zeros_like(reference_x.grad))
    assert not torch.equal(reference.weight.grad, torch.zeros_like(reference.weight.grad))
    _assert_close(candidate.beta.grad, _tensor([1.1, 0.7, 1.7, 1.9], dtype))
    assert torch.equal(candidate_x, candidate_values["x"])
    assert torch.equal(candidate.gamma, candidate_values["gamma"])
    assert torch.equal(candidate.beta, candidate_values["beta"])
    assert torch.equal(reference_x, reference_values["x"])
    assert torch.equal(reference.weight, reference_values["gamma"])
    assert torch.equal(reference.bias, reference_values["beta"])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_constant_and_singleton_autograd_match_independent_oracle(dtype):
    candidate, reference = _module_pair(3, 0.25, dtype)
    gamma = [2, -1, 0.5]
    beta = [0.25, -0.75, 1.5]
    _set_affine(candidate, reference, gamma, beta, dtype)
    candidate_x = _tensor([[5, 5, 5], [-2, -2, -2], [0, 0, 0]], dtype).requires_grad_()
    reference_x = _tensor([[5, 5, 5], [-2, -2, -2], [0, 0, 0]], dtype).requires_grad_()
    coefficients = _tensor([[1, -2, 0.5], [0.3, 1.2, -0.7], [-1, 0.4, 2]], dtype)

    candidate_output = candidate(candidate_x)
    reference_output = reference(reference_x)
    (candidate_output * coefficients).sum().backward()
    (reference_output * coefficients).sum().backward()
    _assert_close(candidate_output, reference_output)
    _assert_gradient(candidate_x.grad, reference_x.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    _assert_close(candidate.gamma.grad, torch.zeros_like(candidate.gamma.grad))
    assert not torch.equal(candidate_x.grad, torch.zeros_like(candidate_x.grad))

    singleton_candidate, singleton_reference = _module_pair(1, 0.25, dtype)
    _set_affine(singleton_candidate, singleton_reference, [3], [-0.4], dtype)
    singleton_x = _tensor([[-7], [0], [12]], dtype).requires_grad_()
    singleton_reference_x = _tensor([[-7], [0], [12]], dtype).requires_grad_()
    singleton_coefficients = _tensor([[0.2], [-1.1], [2.4]], dtype)
    singleton_output = singleton_candidate(singleton_x)
    singleton_reference_output = singleton_reference(singleton_reference_x)
    (singleton_output * singleton_coefficients).sum().backward()
    (singleton_reference_output * singleton_coefficients).sum().backward()
    _assert_close(singleton_output, singleton_reference_output)
    _assert_gradient(singleton_x.grad, singleton_reference_x.grad)
    _assert_gradient(singleton_candidate.gamma.grad, singleton_reference.weight.grad)
    _assert_gradient(singleton_candidate.beta.grad, singleton_reference.bias.grad)
    _assert_close(singleton_x.grad, torch.zeros_like(singleton_x.grad))
    _assert_close(singleton_candidate.gamma.grad, torch.zeros_like(singleton_candidate.gamma.grad))
    _assert_close(singleton_candidate.beta.grad, _tensor([1.5], dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_registered_parameter_state_round_trip_and_storage_independence(dtype):
    source, source_reference = _ln4_pair(dtype)
    saved_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    destination = LayerNorm(d_model=4, eps=_LN4_EPS).to(device=_TEST_DEVICE, dtype=dtype)
    destination.load_state_dict(saved_state)

    assert list(destination.state_dict()) == ["gamma", "beta"]
    assert torch.equal(destination.gamma, saved_state["gamma"])
    assert torch.equal(destination.beta, saved_state["beta"])
    assert destination.gamma.untyped_storage().data_ptr() != source.gamma.untyped_storage().data_ptr()
    assert destination.beta.untyped_storage().data_ptr() != source.beta.untyped_storage().data_ptr()

    x = _tensor(_LN4_X, dtype)
    source_output = source(x.clone())
    destination_output = destination(x.clone())
    _assert_close(source_output, destination_output)
    _assert_close(source_output, source_reference(x.clone()))

    with torch.no_grad():
        destination.gamma.add_(1)
        destination.beta.sub_(1)
    assert torch.equal(source.gamma, saved_state["gamma"])
    assert torch.equal(source.beta, saved_state["beta"])
    assert torch.equal(saved_state["gamma"], _tensor(_LN4_GAMMA, dtype))
    assert torch.equal(saved_state["beta"], _tensor(_LN4_BETA, dtype))

@pytest.mark.parametrize("dtype", _DTYPES)
def test_square_sequence_and_feature_dimensions_keep_rows_independent(dtype):
    candidate, reference = _module_pair(3, 0.25, dtype)
    gamma = [0.8, -1.3, 2]
    beta = [0.1, -0.4, 0.7]
    _set_affine(candidate, reference, gamma, beta, dtype)
    x = _tensor([[1, 2, 4], [-3, 0, 6], [9, 7, -2]], dtype)

    baseline = candidate(x)
    _assert_close(baseline, reference(x.clone()))

    changed_x = x.clone()
    changed_x[1] = _tensor([5, -2, 8], dtype)
    changed_output = candidate(changed_x)
    _assert_close(changed_output, reference(changed_x.clone()))
    _assert_close(changed_output[[0, 2]], baseline[[0, 2]])
    assert not torch.allclose(changed_output[1], baseline[1], **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_affine_updates_after_forward_apply_on_the_next_call(dtype):
    candidate, reference = _module_pair(4, _LN4_EPS, dtype)
    x = _tensor(_LN4_X, dtype)
    first_output = candidate(x)
    first_reference = reference(x.clone())
    _assert_close(first_output, first_reference)

    gamma = [1.5, 0.5, -0.75, 2]
    beta = [0.3, -1.1, 0.6, 2.2]
    parameter_identity = {
        name: (id(parameter), parameter.untyped_storage().data_ptr())
        for name, parameter in candidate.named_parameters()
    }
    _set_affine(candidate, reference, gamma, beta, dtype)
    updated_output = candidate(x)
    updated_reference = reference(x.clone())

    _assert_finite(updated_output)
    _assert_close(updated_output, updated_reference)
    assert not torch.allclose(updated_output, first_output, **_TOLERANCES[dtype])
    for name, parameter in candidate.named_parameters():
        assert (id(parameter), parameter.untyped_storage().data_ptr()) == parameter_identity[name]


@pytest.mark.parametrize("dtype", _DTYPES)
def test_noncontiguous_view_autograd_matches_oracle_on_backing_leaf(dtype):
    candidate, reference = _module_pair(4, _LN4_EPS, dtype)
    gamma = [1.2, -0.7, 2, 0.3]
    beta = [0.25, -0.75, 1.5, -2]
    _set_affine(candidate, reference, gamma, beta, dtype)
    backing_values = [
        [1, 91, 2, 92, 4, 93, 8, 94],
        [-3, 91, 0, 92, 1, 93, 6, 94],
        [9, 91, 7, 92, 2, 93, -2, 94],
    ]
    candidate_backing = _tensor(backing_values, dtype).requires_grad_()
    reference_backing = _tensor(backing_values, dtype).requires_grad_()
    candidate_x = candidate_backing[:, ::2]
    reference_x = reference_backing[:, ::2]
    assert not candidate_x.is_contiguous()
    coefficients = _tensor(
        [[0.3, -1.2, 2.1, 0.7], [1.4, 0.2, -0.9, 2.3], [-0.6, 1.7, 0.5, -1.1]],
        dtype,
    )
    candidate_backing_before = candidate_backing.detach().clone()
    reference_backing_before = reference_backing.detach().clone()

    candidate_output = candidate(candidate_x)
    reference_output = reference(reference_x)
    _assert_finite(candidate_output)
    _assert_close(candidate_output, reference_output)
    (candidate_output * coefficients).sum().backward()
    (reference_output * coefficients).sum().backward()

    _assert_gradient(candidate_backing.grad, reference_backing.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    _assert_close(candidate_backing.grad[:, 1::2], torch.zeros_like(candidate_backing.grad[:, 1::2]))
    assert torch.equal(candidate_backing, candidate_backing_before)
    assert torch.equal(reference_backing, reference_backing_before)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_multiple_forward_backward_calls_preserve_graphs_and_accumulate_gradients(dtype):
    candidate, reference = _ln4_pair(dtype)
    candidate_x1 = _tensor(_LN4_X, dtype).requires_grad_()
    reference_x1 = _tensor(_LN4_X, dtype).requires_grad_()
    second_values = [[0, 3, -2, 5], [4, -1, 6, 2], [-4, 2, 1, 3]]
    candidate_x2 = _tensor(second_values, dtype).requires_grad_()
    reference_x2 = _tensor(second_values, dtype).requires_grad_()
    coefficients1 = _tensor(
        [[0.3, -1.2, 2.1, 0.7], [1.4, 0.2, -0.9, 2.3], [-0.6, 1.7, 0.5, -1.1]],
        dtype,
    )
    coefficients2 = _tensor(
        [[1, -0.5, 0.2, 1.5], [-0.3, 1.2, 0.7, -0.8], [2, 0.4, -1.1, 0.6]],
        dtype,
    )

    candidate_output1 = candidate(candidate_x1)
    reference_output1 = reference(reference_x1)
    candidate_output2 = candidate(candidate_x2)
    reference_output2 = reference(reference_x2)
    _assert_close(candidate_output1, reference_output1)
    _assert_close(candidate_output2, reference_output2)

    (candidate_output1 * coefficients1).sum().backward()
    (reference_output1 * coefficients1).sum().backward()
    _assert_gradient(candidate_x1.grad, reference_x1.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    assert candidate_x2.grad is None
    assert reference_x2.grad is None
    first_backward_grads = {
        "x1": candidate_x1.grad.clone(),
        "gamma": candidate.gamma.grad.clone(),
        "beta": candidate.beta.grad.clone(),
    }

    probe_x = _tensor([[-2, 4, 1, 5]], dtype)
    probe_output = candidate(probe_x)
    probe_reference = reference(probe_x.clone())
    _assert_close(probe_output, probe_reference)
    assert torch.equal(candidate_x1.grad, first_backward_grads["x1"])
    assert torch.equal(candidate.gamma.grad, first_backward_grads["gamma"])
    assert torch.equal(candidate.beta.grad, first_backward_grads["beta"])

    (candidate_output2 * coefficients2).sum().backward()
    (reference_output2 * coefficients2).sum().backward()
    _assert_gradient(candidate_x1.grad, reference_x1.grad)
    _assert_gradient(candidate_x2.grad, reference_x2.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    assert not torch.equal(candidate.beta.grad, first_backward_grads["beta"])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_near_constant_autograd_matches_oracle_with_small_epsilon(dtype):
    candidate, reference = _module_pair(2, 1e-8, dtype)
    gamma = [1.3, -0.4]
    beta = [0.2, -0.7]
    _set_affine(candidate, reference, gamma, beta, dtype)
    values = [[-0.001, 0.001], [-0.002, 0.002], [0.003, -0.003]]
    candidate_x = _tensor(values, dtype).requires_grad_()
    reference_x = _tensor(values, dtype).requires_grad_()
    coefficients = _tensor([[0.3, -1.1], [1.7, 0.2], [-0.6, 2.1]], dtype)

    candidate_output = candidate(candidate_x)
    reference_output = reference(reference_x)
    candidate_objective = (candidate_output * coefficients).sum()
    reference_objective = (reference_output * coefficients).sum()
    candidate_objective.backward()
    reference_objective.backward()

    _assert_finite(candidate_output)
    _assert_finite(candidate_objective)
    _assert_gradient(candidate_x.grad, reference_x.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    _assert_close(candidate_output, reference_output)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_single_token_single_feature_autograd_matches_oracle(dtype):
    candidate, reference = _module_pair(1, 0.25, dtype)
    _set_affine(candidate, reference, [2.5], [-0.3], dtype)
    candidate_x = _tensor([[7.5]], dtype).requires_grad_()
    reference_x = _tensor([[7.5]], dtype).requires_grad_()
    coefficient = _tensor([[0.7]], dtype)

    candidate_output = candidate(candidate_x)
    reference_output = reference(reference_x)
    _assert_close(candidate_output, reference_output)
    (candidate_output * coefficient).sum().backward()
    (reference_output * coefficient).sum().backward()

    _assert_gradient(candidate_x.grad, reference_x.grad)
    _assert_gradient(candidate.gamma.grad, reference.weight.grad)
    _assert_gradient(candidate.beta.grad, reference.bias.grad)
    _assert_close(candidate_x.grad, torch.zeros_like(candidate_x.grad))
    _assert_close(candidate.gamma.grad, torch.zeros_like(candidate.gamma.grad))
    _assert_close(candidate.beta.grad, _tensor([0.7], dtype))
