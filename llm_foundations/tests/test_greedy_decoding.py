"""Verify greedy token selection against independent IDs and input invariants."""

import inspect

import pytest
import torch

from llm_foundations.decoding import greedy_token, sampling


_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DEVICE_REASON = "CUDA is available" if torch.cuda.is_available() else "CUDA is unavailable"
print(f"Decoding tests selected device: {_DEVICE} ({_DEVICE_REASON})")


def _tensor(values, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_DEVICE)


def _assert_scalar_id(token: torch.Tensor, logits: torch.Tensor, expected_ids: set[int]) -> None:
    assert isinstance(token, torch.Tensor)
    assert token.ndim == 0
    assert token.dtype == torch.long
    assert token.device == logits.device
    token_id = int(token.item())
    assert 0 <= token_id < logits.numel()
    assert token_id in expected_ids


def _assert_rejected_call(selector) -> None:
    try:
        selector()
    except RuntimeError:
        raise
    except Exception:
        return
    pytest.fail("selector returned a token for an invalid call")


def _assert_same_values(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def _rng_states() -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )
    return cpu_state, cuda_states


def _assert_rng_states_equal(
    actual: tuple[torch.Tensor, list[torch.Tensor] | None],
    expected: tuple[torch.Tensor, list[torch.Tensor] | None],
) -> None:
    assert torch.equal(actual[0], expected[0])
    assert (actual[1] is None) == (expected[1] is None)
    if actual[1] is not None and expected[1] is not None:
        assert len(actual[1]) == len(expected[1])
        assert all(torch.equal(left, right) for left, right in zip(actual[1], expected[1]))


def test_api01_binds_to_the_real_selector_names_and_signatures():
    greedy_signature = inspect.signature(greedy_token)
    sampling_signature = inspect.signature(sampling)

    assert tuple(greedy_signature.parameters) == ("logits",)
    assert tuple(sampling_signature.parameters) == (
        "logits",
        "temperature",
        "top_k",
        "top_p",
        "generator",
    )
    assert sampling_signature.parameters["temperature"].default == 1.0
    assert sampling_signature.parameters["top_k"].default is None
    assert sampling_signature.parameters["top_p"].default is None
    assert sampling_signature.parameters["generator"].default is None
    assert all(
        sampling_signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("temperature", "top_k", "top_p", "generator")
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_g01_unique_positive_maximum_returns_its_original_id(dtype: torch.dtype):
    logits = _tensor([-3.0, 2.0, -1.0, 6.0, 0.0], dtype=dtype)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {3})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_g02_signed_logits_are_not_interpreted_as_probabilities(dtype: torch.dtype):
    logits = _tensor([-8.0, -3.0, -6.0, -4.0], dtype=dtype)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {1})


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_g03_singleton_vocabulary_returns_scalar_id_zero(dtype: torch.dtype):
    logits = _tensor([-12.5], dtype=dtype)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {0})


def test_g04_tied_maxima_return_any_member_of_the_argmax_set():
    tied = _tensor([4.0, -2.0, 4.0, 1.0])
    all_tied = _tensor([1.0, 1.0, 1.0])

    tied_token = greedy_token(tied)
    all_tied_token = greedy_token(all_tied)

    _assert_scalar_id(tied_token, tied, {0, 2})
    _assert_scalar_id(all_tied_token, all_tied, {0, 1, 2})


def test_g05_common_offset_and_positive_scaling_preserve_unique_argmax():
    values = [0.5, -2.0, 1.25, -0.75]
    fixtures = (
        _tensor(values),
        _tensor([value + 16.0 for value in values]),
        _tensor([value * 3.0 for value in values]),
    )

    for logits in fixtures:
        _assert_scalar_id(greedy_token(logits), logits, {2})


def test_g06_unique_argmax_is_rng_independent_and_does_not_consume_rng():
    fixtures = (
        (_tensor([9.0, -1.0, 2.0, 0.0, -5.0, 4.0]), 0),
        (_tensor([-4.0, 2.0, 0.0, -1.0, 3.0, 1.0, 8.0]), 6),
    )
    initial_states = _rng_states()

    try:
        for logits, expected_id in fixtures:
            results = []
            for seed in (101, 202):
                torch.manual_seed(seed)
                before = _rng_states()

                token = greedy_token(logits)

                after = _rng_states()
                _assert_rng_states_equal(after, before)
                _assert_scalar_id(token, logits, {expected_id})
                results.append(int(token.item()))
            assert results == [expected_id, expected_id]
    finally:
        torch.random.set_rng_state(initial_states[0])
        if initial_states[1] is not None:
            torch.cuda.set_rng_state_all(initial_states[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_g07_noncontiguous_view_keeps_original_position_and_base_unchanged(
    dtype: torch.dtype,
):
    base = _tensor([1.0, 99.0, -7.0, 98.0, 3.0, 97.0, -2.0, 96.0, 8.0, 95.0], dtype=dtype)
    logits = base[::2]
    base_before = base.clone()
    logits_metadata = (logits.shape, logits.stride(), logits.storage_offset(), logits.dtype, logits.device)
    base_metadata = (base.shape, base.stride(), base.storage_offset(), base.dtype, base.device)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {4})
    assert torch.equal(base, base_before)
    assert (logits.shape, logits.stride(), logits.storage_offset(), logits.dtype, logits.device) == logits_metadata
    assert (base.shape, base.stride(), base.storage_offset(), base.dtype, base.device) == base_metadata


def test_g08_preserves_input_values_and_prefilled_gradient():
    logits = _tensor([2.0, -4.0, 7.0, 1.0, -3.0], dtype=torch.float64).requires_grad_()
    logits.grad = _tensor([0.5, -1.0, 2.0, 3.0, -4.0], dtype=torch.float64)
    values_before = logits.detach().clone()
    gradient_before = logits.grad.clone()
    metadata_before = (logits.shape, logits.stride(), logits.dtype, logits.device, logits.requires_grad)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {2})
    assert torch.equal(logits.detach(), values_before)
    assert torch.equal(logits.grad, gradient_before)
    assert (logits.shape, logits.stride(), logits.dtype, logits.device, logits.requires_grad) == metadata_before


@pytest.mark.parametrize(
    ("values", "expected_id"),
    [
        ([10000.0, 9996.0, -10000.0], 0),
        ([-10000.0, -9997.0, -9999.0, -10001.0], 1),
    ],
    ids=["large-positive", "large-negative"],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_g09_large_finite_logits_preserve_argmax(values, expected_id, dtype):
    logits = _tensor(values, dtype=dtype)

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, {expected_id})


@pytest.mark.parametrize(
    "kind",
    ["list", "scalar", "rank-two", "empty", "integer", "boolean"],
)
def test_v01_greedy_rejects_unsupported_input_families(kind: str):
    invalid = {
        "list": [1.0, -2.0, 3.0],
        "scalar": _tensor(2.5),
        "rank-two": _tensor([[1.0, -1.0], [0.0, 3.0]]),
        "empty": torch.empty((0,), dtype=torch.float64, device=_DEVICE),
        "integer": torch.tensor([1, -2, 3], dtype=torch.int64, device=_DEVICE),
        "boolean": torch.tensor([True, False, True], dtype=torch.bool, device=_DEVICE),
    }[kind]
    before = invalid.clone() if isinstance(invalid, torch.Tensor) else None

    _assert_rejected_call(lambda: greedy_token(invalid))

    if before is not None:
        _assert_same_values(invalid, before)

@pytest.mark.parametrize(
    ("values", "expected_ids"),
    [
        ([-float("inf"), 3.0, -1.0], {1}),
        ([2.0, -float("inf"), 1.0, -3.0], {0}),
    ],
    ids=["masked-first", "masked-interior"],
)
def test_v02_greedy_accepts_negative_infinity_when_a_finite_logit_remains(
    values, expected_ids
):
    logits = _tensor(values)
    before = logits.clone()

    token = greedy_token(logits)

    _assert_scalar_id(token, logits, expected_ids)
    _assert_same_values(logits, before)


@pytest.mark.parametrize(
    "values",
    [
        [0.5, float("nan"), 1.0],
        [0.5, float("inf"), -2.0],
        [-float("inf"), -float("inf"), -float("inf")],
    ],
    ids=["nan", "positive-infinity", "all-masked"],
)
def test_v02_greedy_rejects_unsupported_nonfinite_logits(values):
    logits = _tensor(values)
    before = logits.clone()

    _assert_rejected_call(lambda: greedy_token(logits))

    _assert_same_values(logits, before)
