"""Verify categorical selection, filtering, RNG behavior and original token IDs."""

import math

import pytest
import torch

from llm_foundations.decoding import greedy_token, sampling


_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DEVICE_REASON = "CUDA is available" if torch.cuda.is_available() else "CUDA is unavailable"
print(f"Decoding tests selected device: {_DEVICE} ({_DEVICE_REASON})")

_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-7},
    torch.float64: {"rtol": 1e-10, "atol": 1e-12},
}


def _tensor(values, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_DEVICE)


def _scalar_softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [value / temperature for value in logits]
    maximum = max(value for value in scaled if math.isfinite(value))
    exponentials = [math.exp(value - maximum) if value != -math.inf else 0.0 for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    return [value / total for value in values]


def _top_k_oracle(probabilities: list[float], k: int) -> tuple[list[int], list[float]]:
    ids = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))[:k]
    return ids, _normalize([probabilities[index] for index in ids])


def _top_p_oracle(
    probabilities: list[float], p: float
) -> tuple[list[int], list[float]]:
    ids = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    retained = []
    cumulative = 0.0
    for token_id in ids:
        retained.append(token_id)
        cumulative += probabilities[token_id]
        if cumulative >= p:
            break
    return retained, _normalize([probabilities[index] for index in retained])


def _assert_scalar_id(
    token: torch.Tensor, logits: torch.Tensor, expected_ids: set[int]
) -> None:
    assert isinstance(token, torch.Tensor)
    assert token.ndim == 0
    assert token.dtype == torch.long
    assert token.device == logits.device
    token_id = int(token.item())
    assert 0 <= token_id < logits.numel()
    assert token_id in expected_ids


@pytest.fixture
def multinomial_spy(monkeypatch):
    original_multinomial = torch.multinomial
    calls = []

    def spy(input, num_samples, *args, **kwargs):
        call = {
            "weights": None,
            "sampled_positions": None,
        }
        calls.append(call)
        call["weights"] = input.detach().clone()
        result = original_multinomial(input, num_samples, *args, **kwargs)
        call["sampled_positions"] = result.detach().clone()
        return result

    monkeypatch.setattr(torch, "multinomial", spy)
    return calls


def _sample_and_check(
    logits: torch.Tensor,
    calls,
    expected_weights: list[float],
    expected_ids: list[int] | None,
    *,
    expected_support: set[int] | None = None,
    strict_tail: bool = False,
    zero_weight_positions: set[int] | None = None,
    **kwargs,
) -> torch.Tensor:
    call_count = len(calls)

    token = sampling(logits, **kwargs)

    assert len(calls) == call_count + 1
    call = calls[-1]
    weights = call["weights"]
    sampled_positions = call["sampled_positions"]
    if expected_support is None:
        expected_support = set(range(logits.numel()))
    _assert_scalar_id(token, logits, expected_support)
    assert weights.ndim == 1
    assert sampled_positions.shape == (1,)
    assert sampled_positions.dtype == torch.long
    assert sampled_positions.device == logits.device
    assert weights.device == logits.device
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    assert weights.sum().item() > 0
    if zero_weight_positions:
        masked_positions = torch.tensor(
            sorted(zero_weight_positions), dtype=torch.long, device=weights.device
        )
        assert torch.equal(weights[masked_positions], torch.zeros_like(weights[masked_positions]))
    normalized = weights / weights.sum()
    expected = torch.tensor(expected_weights, dtype=logits.dtype, device=logits.device)
    tolerances = dict(_TOLERANCES[logits.dtype])
    if strict_tail:
        tolerances["atol"] = 0.0
    torch.testing.assert_close(normalized, expected, **tolerances)
    if expected_ids is not None:
        position = int(sampled_positions[0].item())
        assert 0 <= position < len(expected_ids)
        assert int(token.item()) == expected_ids[position]
    return token


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


def _assert_rejected(logits: torch.Tensor, calls, **kwargs) -> None:
    before = logits.clone()

    _assert_rejected_call(lambda: sampling(logits, **kwargs))

    _assert_same_values(logits, before)
    assert calls == []


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s01_unfiltered_sampling_uses_tau_one_softmax(dtype: torch.dtype, multinomial_spy):
    q = [0.15, 0.05, 0.50, 0.30]
    logits = _tensor([math.log(value) for value in q], dtype=dtype)

    _sample_and_check(logits, multinomial_spy, q, [0, 1, 2, 3])


def test_s02_singleton_unfiltered_top_k_and_top_p_each_select_zero(multinomial_spy):
    logits = _tensor([7.25])

    for kwargs in ({}, {"top_k": 1}, {"top_p": 0.2}):
        token = _sample_and_check(logits, multinomial_spy, [1.0], [0], **kwargs)
        assert int(token.item()) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s03_temperature_half_sharpens_distribution_without_filtering(
    dtype: torch.dtype, multinomial_spy
):
    values = [-1.5, 0.0, 2.0, -0.5, 1.0]
    logits = _tensor(values, dtype=dtype)
    expected = _scalar_softmax(values, temperature=0.5)

    _sample_and_check(logits, multinomial_spy, expected, list(range(len(values))), temperature=0.5)
    assert max(expected) > max(_scalar_softmax(values))
    assert all(value > 0 for value in expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s04_temperature_two_spreads_distribution_without_filtering(
    dtype: torch.dtype, multinomial_spy
):
    values = [0.25, -2.0, 1.5, 0.0, -0.75, 1.0]
    logits = _tensor(values, dtype=dtype)
    expected = _scalar_softmax(values, temperature=2.0)

    _sample_and_check(logits, multinomial_spy, expected, list(range(len(values))), temperature=2.0)
    assert max(expected) < max(_scalar_softmax(values))
    assert all(value > 0 for value in expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s05_top_k_two_filters_and_renormalizes_tau_one(
    dtype: torch.dtype, multinomial_spy
):
    values = [2.0, 1.0, 0.0]
    logits = _tensor(values, dtype=dtype)
    ids, expected = _top_k_oracle(_scalar_softmax(values), 2)

    _sample_and_check(logits, multinomial_spy, expected, ids, top_k=2)


def test_s06_top_k_weights_use_temperature_before_filtering(multinomial_spy):
    values = [2.0, 1.0, 0.0]
    logits = _tensor(values)
    ids, expected = _top_k_oracle(_scalar_softmax(values, temperature=2.0), 2)

    _sample_and_check(logits, multinomial_spy, expected, ids, top_k=2, temperature=2.0)


def test_s07_top_k_one_agrees_with_greedy_on_original_id(multinomial_spy):
    logits = _tensor([-4.0, 1.0, -2.0, 6.0, 0.0])
    greedy_id = int(greedy_token(logits).item())

    token = _sample_and_check(logits, multinomial_spy, [1.0], [3], top_k=1)

    assert greedy_id == 3
    assert int(token.item()) == greedy_id


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s08_top_p_retains_crossing_token_and_renormalizes_original_ids(
    dtype: torch.dtype, multinomial_spy
):
    q = [0.10, 0.25, 0.05, 0.60]
    logits = _tensor([math.log(value) for value in q], dtype=dtype)
    ids, expected = _top_p_oracle(q, 0.8)

    _sample_and_check(logits, multinomial_spy, expected, ids, top_p=0.8)
    assert set(ids) == {1, 3}


def test_s09_top_p_keeps_first_token_when_it_crosses_threshold(multinomial_spy):
    q = [0.10, 0.25, 0.05, 0.60]
    logits = _tensor([math.log(value) for value in q])
    ids, expected = _top_p_oracle(q, 0.5)

    token = _sample_and_check(logits, multinomial_spy, expected, ids, top_p=0.5)

    assert ids == [3]
    assert int(token.item()) == 3


def test_s10_top_p_one_preserves_every_positive_float64_tail(multinomial_spy):
    q = [0.40, 0.35, 0.24999999, 0.00000001]
    logits = _tensor([math.log(value) for value in q])
    ids, expected = _top_p_oracle(q, 1.0)

    _sample_and_check(
        logits,
        multinomial_spy,
        expected,
        ids,
        top_p=1.0,
        strict_tail=True,
    )
    assert ids == [0, 1, 2, 3]
    assert all(value > 0 for value in expected)


def test_s11_top_p_uses_temperature_adjusted_probabilities(multinomial_spy):
    q = [0.55, 0.10, 0.30, 0.05]
    logits = _tensor([math.log(value) for value in q])
    probabilities = _scalar_softmax([math.log(value) for value in q], temperature=2.0)
    ids, expected = _top_p_oracle(probabilities, 0.8)

    _sample_and_check(
        logits,
        multinomial_spy,
        expected,
        ids,
        top_p=0.8,
        temperature=2.0,
    )
    assert set(ids) == {0, 1, 2}


def test_s12_top_k_tie_retains_exactly_k_equal_candidates(multinomial_spy):
    logits = _tensor([0.0, 0.0, 0.0, 0.0, 0.0])

    token = _sample_and_check(
        logits,
        multinomial_spy,
        [0.5, 0.5],
        expected_ids=None,
        expected_support=set(range(5)),
        top_k=2,
    )

    assert int(token.item()) in range(5)


def test_s13_top_p_uniform_mass_obeys_three_exact_cutoffs(multinomial_spy):
    logits = _tensor([0.0, 0.0, 0.0, 0.0])

    for p, candidate_count in ((0.5, 2), (0.8, 4), (0.75, 3)):
        token = _sample_and_check(
            logits,
            multinomial_spy,
            [1.0 / candidate_count] * candidate_count,
            expected_ids=None,
            expected_support={0, 1, 2, 3},
            top_p=p,
        )
        assert int(token.item()) in range(4)


@pytest.mark.parametrize(
    ("argument", "value", "values"),
    [
        ("temperature", True, [0.0, 1.0, -1.0]),
        ("temperature", "1.0", [2.0, 0.0, -2.0, 1.0]),
        ("top_k", True, [3.0, 1.0, 0.0, -2.0, 2.0]),
        ("top_k", 1.5, [0.0, -1.0, 3.0]),
        ("top_p", True, [1.0, 0.0, -1.0, 2.0]),
        ("top_p", "0.5", [3.0, 2.0, 1.0, 0.0, -1.0]),
    ],
    ids=["bool-temperature", "string-temperature", "bool-top-k", "fractional-top-k", "bool-top-p", "string-top-p"],
)
def test_s14_rejects_wrong_filter_argument_types(argument, value, values, multinomial_spy):
    logits = _tensor(values)

    _assert_rejected(logits, multinomial_spy, **{argument: value})


@pytest.mark.parametrize(
    ("temperature", "values"),
    [
        (0.0, [1.0, -1.0]),
        (-0.5, [0.0, 2.0, -2.0]),
        (float("nan"), [3.0, 0.0, -1.0, 2.0]),
        (float("inf"), [1.0, -2.0, 4.0, 0.0, -3.0]),
        (-float("inf"), [0.0, 3.0, -1.0, 2.0, -2.0, 1.0]),
    ],
    ids=["zero", "negative", "nan", "positive-infinity", "negative-infinity"],
)
def test_s15_rejects_invalid_temperature_domains(temperature, values, multinomial_spy):
    _assert_rejected(_tensor(values), multinomial_spy, temperature=temperature)


@pytest.mark.parametrize(
    ("top_k", "values"),
    [
        (0, [2.0, 1.0, 0.0]),
        (-1, [0.0, 4.0, -2.0, 1.0]),
        (5, [3.0, -1.0, 2.0, 0.0]),
    ],
    ids=["zero", "negative", "larger-than-vocab"],
)
def test_s16_rejects_invalid_top_k_domains(top_k, values, multinomial_spy):
    _assert_rejected(_tensor(values), multinomial_spy, top_k=top_k)


@pytest.mark.parametrize(
    ("top_p", "values"),
    [
        (0.0, [1.0, -1.0]),
        (-0.2, [0.0, 2.0, -2.0]),
        (1.1, [3.0, 0.0, -1.0, 2.0]),
        (float("nan"), [1.0, -2.0, 4.0, 0.0, -3.0]),
        (float("inf"), [0.0, 3.0, -1.0, 2.0, -2.0, 1.0]),
    ],
    ids=["zero", "negative", "greater-than-one", "nan", "infinity"],
)
def test_s17_rejects_invalid_top_p_domains(top_p, values, multinomial_spy):
    _assert_rejected(_tensor(values), multinomial_spy, top_p=top_p)


def test_s18_rejects_simultaneous_filters_without_sampling(multinomial_spy):
    logits = _tensor([1.0, 4.0, -2.0, 3.0, 0.0, 2.0])

    _assert_rejected(logits, multinomial_spy, top_k=2, top_p=0.8)


def test_s19_rejects_wrong_generator_type_without_mutation(multinomial_spy):
    logits = _tensor([0.5, -1.0, 2.0, 1.0])

    _assert_rejected(logits, multinomial_spy, generator=object())


def test_s19_rejects_cpu_generator_for_cuda_logits_without_consuming_state(
    multinomial_spy,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; device-mismatch variant is unexercised")

    logits = _tensor([0.0, 2.0, -1.0, 1.0])
    cpu_generator = torch.Generator(device="cpu").manual_seed(2718)
    state_before = cpu_generator.get_state().clone()

    _assert_rejected(logits, multinomial_spy, generator=cpu_generator)

    assert torch.equal(cpu_generator.get_state(), state_before)


def test_s20_same_seed_generators_replay_sequences_and_advance_independently():
    q = [0.20, 0.45, 0.10, 0.25]
    logits = _tensor([math.log(value) for value in q])
    first = torch.Generator(device=_DEVICE).manual_seed(1729)
    second = torch.Generator(device=_DEVICE).manual_seed(1729)
    first_before = first.get_state().clone()
    second_before = second.get_state().clone()

    first_sequence = [int(sampling(logits, generator=first).item()) for _ in range(32)]
    second_sequence = [int(sampling(logits, generator=second).item()) for _ in range(32)]

    assert first_sequence == second_sequence
    assert torch.equal(first.get_state(), second.get_state())
    assert all(token_id in range(len(q)) for token_id in first_sequence)
    assert not torch.equal(first.get_state(), first_before)
    assert not torch.equal(second.get_state(), second_before)


def test_s21_generator_restore_replays_and_leaves_global_rng_untouched():
    logits = _tensor([-0.25, 1.75, -1.0, 0.5, 0.25])
    generator = torch.Generator(device=_DEVICE).manual_seed(31415)
    initial_generator_state = generator.get_state().clone()
    global_cpu_before = torch.random.get_rng_state().clone()
    global_cuda_before = (
        [state.clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else None
    )

    first = [int(sampling(logits, generator=generator).item()) for _ in range(8)]
    first_final_state = generator.get_state().clone()
    generator.set_state(initial_generator_state)
    second = [int(sampling(logits, generator=generator).item()) for _ in range(8)]
    second_final_state = generator.get_state().clone()

    assert first == second
    assert torch.equal(first_final_state, second_final_state)
    assert not torch.equal(first_final_state, initial_generator_state)
    assert torch.equal(torch.random.get_rng_state(), global_cpu_before)
    if global_cuda_before is not None:
        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(torch.cuda.get_rng_state_all(), global_cuda_before)
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s22_negative_infinity_mask_and_noncontiguous_view_preserve_state(
    dtype: torch.dtype, multinomial_spy
):
    base = _tensor(
        [-math.inf, 91.0, -1.0, 92.0, 2.0, 93.0, -3.0, 94.0, 0.0, 95.0],
        dtype=dtype,
    ).requires_grad_()
    logits = base[::2]
    logits.retain_grad()
    base.grad = _tensor([0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0], dtype=dtype)
    logits.grad = _tensor([0.25, -0.5, 1.5, -2.0, 3.0], dtype=dtype)
    base_before = base.detach().clone()
    logits_before = logits.detach().clone()
    base_grad_before = base.grad.clone()
    logits_grad_before = logits.grad.clone()
    base_metadata = (
        base.shape,
        base.stride(),
        base.storage_offset(),
        base.dtype,
        base.device,
        base.requires_grad,
    )
    logits_metadata = (
        logits.shape,
        logits.stride(),
        logits.storage_offset(),
        logits.dtype,
        logits.device,
        logits.requires_grad,
    )
    finite_probabilities = _scalar_softmax([-math.inf, -1.0, 2.0, -3.0, 0.0])

    token = _sample_and_check(
        logits,
        multinomial_spy,
        finite_probabilities,
        [0, 1, 2, 3, 4],
        expected_support={1, 2, 3, 4},
        zero_weight_positions={0},
    )

    assert int(token.item()) != 0
    assert torch.equal(base.detach(), base_before)
    assert torch.equal(logits.detach(), logits_before)
    assert torch.equal(base.grad, base_grad_before)
    assert torch.equal(logits.grad, logits_grad_before)
    assert (
        base.shape,
        base.stride(),
        base.storage_offset(),
        base.dtype,
        base.device,
        base.requires_grad,
    ) == base_metadata
    assert (
        logits.shape,
        logits.stride(),
        logits.storage_offset(),
        logits.dtype,
        logits.device,
        logits.requires_grad,
    ) == logits_metadata


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([10000.0, 9999.0, 9997.0], torch.float32),
        ([10000.0, 9999.0, 9997.0], torch.float64),
        ([-10000.0, -9998.0, -9999.0, -10002.0], torch.float32),
        ([-10000.0, -9998.0, -9999.0, -10002.0], torch.float64),
    ],
    ids=["positive-f32", "positive-f64", "negative-f32", "negative-f64"],
)
def test_s23_extreme_finite_logits_match_stable_scalar_oracle(values, dtype, multinomial_spy):
    logits = _tensor(values, dtype=dtype)
    expected = _scalar_softmax(values)

    _sample_and_check(logits, multinomial_spy, expected, list(range(len(values))))


def test_s24_fixed_frequency_budget_matches_categorical_probabilities():
    q = [0.12, 0.23, 0.41, 0.24]
    logits = _tensor([math.log(value) for value in q])
    generator = torch.Generator(device=_DEVICE).manual_seed(1729)
    draw_count = 8192
    counts = [0] * len(q)

    for _ in range(draw_count):
        token_id = int(sampling(logits, generator=generator).item())
        assert token_id in range(len(q))
        counts[token_id] += 1

    epsilon = math.sqrt(math.log(2 * len(q) / 1e-6) / (2 * draw_count))
    observed = [count / draw_count for count in counts]
    assert all(abs(actual - expected) <= epsilon for actual, expected in zip(observed, q))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_s25_top_k_equal_to_vocab_size_matches_full_softmax(
    dtype: torch.dtype, multinomial_spy
):
    values = [3.0, -2.0, 0.0, 1.0, -1.0, 2.0]
    logits = _tensor(values, dtype=dtype)
    ids, expected = _top_k_oracle(_scalar_softmax(values), len(values))

    _sample_and_check(
        logits,
        multinomial_spy,
        expected,
        ids,
        top_k=len(values),
    )


def test_s26_top_p_maps_sorted_position_to_original_vocabulary_id(multinomial_spy):
    q = [0.03, 0.02, 0.90, 0.05]
    logits = _tensor([math.log(value) for value in q])
    ids, expected = _top_p_oracle(q, 0.8)

    token = _sample_and_check(logits, multinomial_spy, expected, ids, top_p=0.8)

    assert ids == [2]
    assert int(token.item()) == 2


def test_s27_small_positive_top_p_keeps_the_highest_candidate(multinomial_spy):
    q = [0.20, 0.45, 0.35]
    logits = _tensor([math.log(value) for value in q])
    ids, expected = _top_p_oracle(q, 0.1)

    token = _sample_and_check(logits, multinomial_spy, expected, ids, top_p=0.1)

    assert ids == [1]
    assert int(token.item()) == 1


@pytest.mark.parametrize(
    "kind",
    ["list", "scalar", "rank-two", "empty", "integer", "boolean"],
)
def test_v01_sampling_rejects_unsupported_input_families(kind: str, multinomial_spy):
    invalid = {
        "list": [1.0, -2.0, 3.0, 0.5],
        "scalar": _tensor(-3.5),
        "rank-two": _tensor([[2.0, -1.0], [0.5, 3.0]]),
        "empty": torch.empty((0,), dtype=torch.float64, device=_DEVICE),
        "integer": torch.tensor([2, -1, 3, 0], dtype=torch.int64, device=_DEVICE),
        "boolean": torch.tensor([False, True, False, True], dtype=torch.bool, device=_DEVICE),
    }[kind]
    before = invalid.clone() if isinstance(invalid, torch.Tensor) else None

    _assert_rejected_call(lambda: sampling(invalid))

    if before is not None:
        _assert_same_values(invalid, before)
    assert multinomial_spy == []


@pytest.mark.parametrize(
    "values",
    [
        [float("nan"), 2.0, -1.0],
        [float("inf"), -2.0, 1.0, 0.0],
        [-math.inf, -math.inf, -math.inf, -math.inf, -math.inf],
    ],
    ids=["nan", "positive-infinity", "all-masked"],
)
def test_v02_sampling_rejects_unsupported_nonfinite_logits(values, multinomial_spy):
    _assert_rejected(_tensor(values), multinomial_spy)


def test_v02_sampling_accepts_negative_infinity_mask_with_finite_candidates(
    multinomial_spy,
):
    values = [-math.inf, 3.0, -1.0, 0.0]
    logits = _tensor(values)
    expected = _scalar_softmax(values)

    token = _sample_and_check(
        logits,
        multinomial_spy,
        expected,
        list(range(len(values))),
        expected_support={1, 2, 3},
        zero_weight_positions={0},
    )

    assert int(token.item()) != 0
