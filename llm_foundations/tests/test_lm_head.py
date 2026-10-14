"""Verify the learner's LM Head projection with independent value and gradient oracles."""

import inspect
import math

import pytest
import torch
from torch import nn

from llm_foundations.lm_head import LMHead


_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DEVICE_REASON = "CUDA is available" if torch.cuda.is_available() else "CUDA is unavailable"
print(f"LM Head tests selected device: {_DEVICE} ({_DEVICE_REASON})")

_RECTANGULAR_H = [
    [1.0, 2.0, -1.0],
    [-2.0, 0.5, 3.0],
]
_RECTANGULAR_W = [
    [1.0, 0.0, -2.0, 0.5],
    [0.0, 3.0, 1.0, -1.0],
    [2.0, -1.0, 0.0, 4.0],
]
_RECTANGULAR_Z = [
    [-1.0, 7.0, 0.0, -5.5],
    [4.0, -1.5, 4.5, 10.5],
]
_RECTANGULAR_U = [
    [1.0, -2.0, 0.5, 3.0],
    [-1.0, 0.25, 2.0, -0.5],
]
_RECTANGULAR_DH = [
    [1.5, -8.5, 16.0],
    [-5.25, 3.25, -4.25],
]
_RECTANGULAR_DW = [
    [3.0, -2.5, -3.5, 4.0],
    [1.5, -3.875, 2.0, 5.75],
    [-4.0, 2.75, 5.5, -4.5],
]


def _head(
    d_model: int,
    vocab_size: int,
    dtype: torch.dtype = torch.float64,
) -> LMHead:
    return LMHead(d_model=d_model, vocab_size=vocab_size).to(
        device=_DEVICE,
        dtype=dtype,
    )


def _tensor(values, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_DEVICE)


def _set_weight(head: LMHead, values) -> None:
    with torch.no_grad():
        head.w_lmhead.copy_(
            torch.tensor(
                values,
                dtype=head.w_lmhead.dtype,
                device=head.w_lmhead.device,
            )
        )


def _fixture(dtype: torch.dtype = torch.float64) -> tuple[LMHead, torch.Tensor]:
    head = _head(d_model=3, vocab_size=4, dtype=dtype)
    _set_weight(head, _RECTANGULAR_W)
    hidden_states = _tensor(_RECTANGULAR_H, dtype=dtype)
    return head, hidden_states


def _scalar_projection(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    hidden_values = hidden_states.detach().cpu().tolist()
    weight_values = weight.detach().cpu().tolist()
    expected = [
        [
            sum(hidden_values[t][d] * weight_values[d][v] for d in range(len(weight_values)))
            for v in range(len(weight_values[0]))
        ]
        for t in range(len(hidden_values))
    ]
    return torch.tensor(expected, dtype=hidden_states.dtype, device=hidden_states.device).reshape(
        hidden_states.shape[0], weight.shape[1]
    )


def _scalar_gradients(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_values = hidden_states.detach().cpu().tolist()
    weight_values = weight.detach().cpu().tolist()
    upstream_values = upstream.detach().cpu().tolist()
    time_steps = len(hidden_values)
    d_model = len(weight_values)
    vocab_size = len(weight_values[0])
    grad_hidden = [
        [
            sum(upstream_values[t][v] * weight_values[d][v] for v in range(vocab_size))
            for d in range(d_model)
        ]
        for t in range(time_steps)
    ]
    grad_weight = [
        [
            sum(hidden_values[t][d] * upstream_values[t][v] for t in range(time_steps))
            for v in range(vocab_size)
        ]
        for d in range(d_model)
    ]
    return (
        torch.tensor(grad_hidden, dtype=hidden_states.dtype, device=hidden_states.device).reshape(
            time_steps, d_model
        ),
        torch.tensor(grad_weight, dtype=weight.dtype, device=weight.device).reshape(
            d_model, vocab_size
        ),
    )


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    return 1e-10, 1e-10


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    rtol, atol = _tolerances(actual.dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)




def test_lh01_binds_directly_to_the_actual_public_api():
    assert issubclass(LMHead, nn.Module)
    assert tuple(inspect.signature(LMHead.__init__).parameters) == (
        "self",
        "d_model",
        "vocab_size",
    )
    assert tuple(inspect.signature(LMHead.forward).parameters) == (
        "self",
        "hidden_states",
    )



def test_lh02_registers_one_persistent_trainable_projection_parameter():
    head = _head(d_model=3, vocab_size=4)
    named_parameters = dict(head.named_parameters())
    state = head.state_dict()

    assert set(named_parameters) == {"w_lmhead"}
    assert set(state) == {"w_lmhead"}
    assert head.w_lmhead is named_parameters["w_lmhead"]
    assert head.w_lmhead.shape == (3, 4)
    assert head.w_lmhead.requires_grad
    assert head.w_lmhead.dtype == torch.float64
    assert head.w_lmhead.device == _DEVICE


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh03_hand_fixture_returns_raw_rank_two_logits(dtype):
    head = _head(d_model=2, vocab_size=3, dtype=dtype)
    _set_weight(head, [[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    hidden_states = _tensor([[1.0, 2.0]], dtype=dtype)

    logits = head(hidden_states)
    assert isinstance(logits, torch.Tensor)

    assert logits.shape == (1, 3)
    assert logits.dtype == dtype
    assert logits.device == _DEVICE
    _assert_close(logits, _tensor([[1.0, 2.0, 1.0]], dtype=dtype))
    assert bool((logits > 1.0).any())
    assert logits.is_floating_point()




@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
@pytest.mark.parametrize(
    "time_steps,d_model,vocab_size",
    [(3, 2, 5), (4, 5, 2), (5, 3, 7)],
    ids=["T3-D2-V5", "T4-D5-V2", "T5-D3-V7"],
)
def test_lh05_bounded_rectangular_sweep_matches_scalar_oracle(
    dtype: torch.dtype,
    time_steps: int,
    d_model: int,
    vocab_size: int,
):
    generator = torch.Generator(device="cpu").manual_seed(211 + time_steps + d_model + vocab_size)
    hidden_values = torch.rand((time_steps, d_model), generator=generator, dtype=torch.float64) * 2 - 1
    weight_values = torch.rand((d_model, vocab_size), generator=generator, dtype=torch.float64) * 2 - 1
    hidden_states = hidden_values.to(device=_DEVICE, dtype=dtype)
    head = _head(d_model=d_model, vocab_size=vocab_size, dtype=dtype)
    with torch.no_grad():
        head.w_lmhead.copy_(weight_values.to(device=_DEVICE, dtype=dtype))
    expected = _scalar_projection(hidden_states, head.w_lmhead)

    logits = head(hidden_states)

    assert logits.shape == (time_steps, vocab_size)
    _assert_close(logits, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh06_singleton_axes_are_preserved(dtype: torch.dtype):
    head = _head(d_model=1, vocab_size=1, dtype=dtype)
    _set_weight(head, [[-2.5]])
    hidden_states = _tensor([[3.0]], dtype=dtype)

    logits = head(hidden_states)

    assert logits.shape == (1, 1)
    _assert_close(logits, _tensor([[-7.5]], dtype=dtype))


@pytest.mark.parametrize("scenario", ["zero-hidden", "zero-weight"], ids=["zero-hidden", "zero-weight"])
def test_lh07_zero_hidden_or_weight_returns_zero_logits(scenario: str):
    head, hidden_states = _fixture(dtype=torch.float32)
    if scenario == "zero-hidden":
        hidden_states = torch.zeros_like(hidden_states)
    else:
        with torch.no_grad():
            head.w_lmhead.zero_()

    logits = head(hidden_states)

    assert logits.shape == (2, 4)
    assert torch.equal(logits, torch.zeros_like(logits))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh08_identity_input_exposes_each_weight_row(dtype: torch.dtype):
    head, _ = _fixture(dtype=dtype)
    identity = torch.eye(3, dtype=dtype, device=_DEVICE)

    logits = head(identity)

    _assert_close(logits, head.w_lmhead.detach())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh08_permuted_weight_columns_keep_vocabulary_order(dtype: torch.dtype):
    head, _ = _fixture(dtype=dtype)
    identity = torch.eye(3, dtype=dtype, device=_DEVICE)
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long, device=_DEVICE)
    expected = torch.tensor(_RECTANGULAR_W, dtype=dtype, device=_DEVICE)[:, permutation]
    with torch.no_grad():
        head.w_lmhead.copy_(expected)

    logits = head(identity)

    _assert_close(logits, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh09_rows_follow_sequence_order_and_do_not_mix(dtype: torch.dtype):
    head, _ = _fixture(dtype=dtype)
    hidden_states = _tensor(
        [
            [4.0, -3.0, 2.0],
            [1.0, 2.0, -1.0],
            [-2.0, 0.5, 3.0],
            [1.0, 2.0, -1.0],
            [1.0, 2.0, -0.25],
        ],
        dtype=dtype,
    )
    expected = _scalar_projection(hidden_states, head.w_lmhead)

    logits = head(hidden_states)

    assert logits.shape == (5, 4)
    _assert_close(logits, expected)
    _assert_close(logits[1], logits[3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh10_lh11_reuse_weight_across_lengths_and_observe_edits(dtype: torch.dtype):
    head, hidden_states = _fixture(dtype=dtype)
    weight_identity = id(head.w_lmhead)
    parameter_count = len(list(head.parameters()))
    original_weight = head.w_lmhead.detach().clone()
    expected_before = _scalar_projection(hidden_states, original_weight)

    logits_before = head(hidden_states)

    assert logits_before.shape == (2, 4)
    _assert_close(logits_before, expected_before)

    hidden_states[1, 0] += 1.5
    with torch.no_grad():
        head.w_lmhead[1, 2] += 2.0
    expected_after = _scalar_projection(hidden_states, head.w_lmhead)

    logits_after = head(hidden_states)

    assert logits_after.shape == (2, 4)
    _assert_close(logits_after, expected_after)
    _assert_close(logits_after - logits_before, expected_after - expected_before)
    assert id(head.w_lmhead) == weight_identity
    assert len(list(head.parameters())) == parameter_count == 1

    for time_steps in (5, 1):
        hidden_states = _tensor(
            [[float(t + 1), float(2 - t), float(t - 2)] for t in range(time_steps)],
            dtype=dtype,
        )
        expected = _scalar_projection(hidden_states, head.w_lmhead)
        logits = head(hidden_states)

        assert logits.shape == (time_steps, 4)
        _assert_close(logits, expected)
        assert id(head.w_lmhead) == weight_identity
        assert len(list(head.parameters())) == parameter_count == 1
        assert logits.dtype == head.w_lmhead.dtype
        assert logits.device == head.w_lmhead.device


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh12_forward_preserves_storage_and_accumulates_existing_gradients(dtype: torch.dtype):
    head = _head(d_model=3, vocab_size=4, dtype=dtype)
    _set_weight(head, _RECTANGULAR_W)
    backing = _tensor(
        [[1.0, 90.0, 2.0, 91.0, -1.0, 92.0], [-2.0, 93.0, 0.5, 94.0, 3.0, 95.0]],
        dtype=dtype,
    )
    hidden_states = backing[:, ::2].requires_grad_()
    upstream = _tensor(_RECTANGULAR_U, dtype=dtype)
    original_backing = backing.clone()
    original_hidden = hidden_states.clone()
    original_weight = head.w_lmhead.detach().clone()
    identity = id(head.w_lmhead)
    hidden_states.grad = torch.full_like(hidden_states, 7.0)
    head.w_lmhead.grad = torch.full_like(head.w_lmhead, -3.0)
    original_hidden_grad = hidden_states.grad.clone()
    original_weight_grad = head.w_lmhead.grad.clone()
    expected_hidden, expected_weight = _scalar_gradients(hidden_states, head.w_lmhead, upstream)

    logits = head(hidden_states)

    _assert_close(logits, _scalar_projection(hidden_states, head.w_lmhead))
    assert torch.equal(backing, original_backing)
    assert torch.equal(hidden_states, original_hidden)
    assert torch.equal(head.w_lmhead, original_weight)
    assert id(head.w_lmhead) == identity
    assert torch.equal(hidden_states.grad, original_hidden_grad)
    assert torch.equal(head.w_lmhead.grad, original_weight_grad)

    if dtype == torch.float64:
        (logits * upstream).sum().backward()
        _assert_close(hidden_states.grad, original_hidden_grad + expected_hidden)
        _assert_close(head.w_lmhead.grad, original_weight_grad + expected_weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh13_transposed_dense_view_matches_scalar_oracle(dtype: torch.dtype):
    head, _ = _fixture(dtype=dtype)
    backing = _tensor([[1.0, -2.0], [2.0, 0.5], [-1.0, 3.0]], dtype=dtype)
    hidden_states = backing.T
    original_backing = backing.clone()

    logits = head(hidden_states)

    assert hidden_states.shape == (2, 3)
    assert not hidden_states.is_contiguous()
    _assert_close(logits, _scalar_projection(hidden_states, head.w_lmhead))
    assert torch.equal(backing, original_backing)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh16_rectangular_logits_gradients_and_placement(dtype: torch.dtype):
    head, hidden_states = _fixture(dtype=dtype)
    hidden_states.requires_grad_()
    upstream = _tensor(_RECTANGULAR_U, dtype=dtype)
    logits = head(hidden_states)
    (logits * upstream).sum().backward()
    expected_hidden, expected_weight = _scalar_gradients(hidden_states, head.w_lmhead, upstream)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 4)
    _assert_close(logits, _tensor(_RECTANGULAR_Z, dtype=dtype))
    assert logits.device == _DEVICE
    assert logits.dtype == dtype
    assert head.w_lmhead.device == _DEVICE
    assert head.w_lmhead.dtype == dtype
    assert hidden_states.grad.device == _DEVICE
    assert head.w_lmhead.grad.device == _DEVICE
    assert hidden_states.grad.dtype == dtype
    assert head.w_lmhead.grad.dtype == dtype
    _assert_close(hidden_states.grad, expected_hidden)
    _assert_close(head.w_lmhead.grad, expected_weight)
    if dtype == torch.float64:
        assert hidden_states.grad.shape == hidden_states.shape
        assert head.w_lmhead.grad.shape == head.w_lmhead.shape
        _assert_close(hidden_states.grad, _tensor(_RECTANGULAR_DH))
        _assert_close(head.w_lmhead.grad, _tensor(_RECTANGULAR_DW))
        assert bool(torch.isfinite(hidden_states.grad).all())
        assert bool(torch.isfinite(head.w_lmhead.grad).all())


@pytest.mark.parametrize(
    "scenario",
    ["zero-hidden", "zero-weight", "zero-upstream"],
    ids=["zero-hidden", "zero-weight", "zero-upstream"],
)
def test_lh17_degenerate_inputs_have_formula_gradients(scenario: str):
    head, hidden_states = _fixture()
    hidden_states.requires_grad_()
    upstream = _tensor(_RECTANGULAR_U)

    if scenario == "zero-hidden":
        hidden_states = torch.zeros_like(hidden_states, requires_grad=True)
    elif scenario == "zero-weight":
        with torch.no_grad():
            head.w_lmhead.zero_()
    else:
        upstream = torch.zeros_like(upstream)

    logits = head(hidden_states)
    if scenario in {"zero-hidden", "zero-weight"}:
        assert torch.equal(logits, torch.zeros_like(logits))
    (logits * upstream).sum().backward()
    expected_hidden, expected_weight = _scalar_gradients(hidden_states, head.w_lmhead, upstream)

    assert hidden_states.grad is not None
    assert head.w_lmhead.grad is not None
    _assert_close(hidden_states.grad, expected_hidden)
    _assert_close(head.w_lmhead.grad, expected_weight)
    if scenario == "zero-hidden":
        assert torch.equal(head.w_lmhead.grad, torch.zeros_like(head.w_lmhead))
        assert bool((hidden_states.grad != 0).any())
    elif scenario == "zero-weight":
        assert torch.equal(hidden_states.grad, torch.zeros_like(hidden_states))
        assert bool((head.w_lmhead.grad != 0).any())
    else:
        assert torch.equal(hidden_states.grad, torch.zeros_like(hidden_states))
        assert torch.equal(head.w_lmhead.grad, torch.zeros_like(head.w_lmhead))


def test_lh19_state_dict_round_trip_preserves_projection_and_isolation():
    head, hidden_states = _fixture()
    original_weight = head.w_lmhead.detach().clone()
    cloned_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
    restored = _head(d_model=3, vocab_size=4)
    result = restored.load_state_dict(cloned_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert set(restored.state_dict()) == {"w_lmhead"}
    assert restored.w_lmhead.shape == (3, 4)
    expected_logits = _scalar_projection(hidden_states, original_weight)

    logits = restored(hidden_states)

    _assert_close(logits, expected_logits)
    with torch.no_grad():
        restored.w_lmhead[0, 0] += 1.0
    assert not torch.equal(restored.w_lmhead, original_weight)
    assert torch.equal(head.w_lmhead, original_weight)


@pytest.mark.parametrize(
    "d_model,vocab_size",
    [(1, 1), (2, 3), (5, 2)],
    ids=["unit-dimensions", "rectangular", "narrow-vocabulary"],
)
def test_lh20_constructs_positive_integer_dimensions(d_model: int, vocab_size: int):
    head = _head(d_model=d_model, vocab_size=vocab_size)
    assert head.w_lmhead.shape == (d_model, vocab_size)


def test_lh21_empty_sequence_preserves_output_and_gradient_shapes():
    head = _head(d_model=3, vocab_size=4)
    _set_weight(head, _RECTANGULAR_W)
    hidden_states = torch.empty((0, 3), dtype=torch.float64, device=_DEVICE, requires_grad=True)

    logits = head(hidden_states)
    logits.sum().backward()

    assert logits.shape == (0, 4)
    assert hidden_states.grad.shape == (0, 3)
    assert head.w_lmhead.grad.shape == (3, 4)
    assert torch.equal(head.w_lmhead.grad, torch.zeros_like(head.w_lmhead))


@pytest.mark.parametrize("shape", [(), (0,)], ids=["scalar", "empty-rank1"])
def test_lh22_incompatible_non_rank_two_inputs_raise_native_matmul_error(shape: tuple[int, ...]):
    head = _head(d_model=3, vocab_size=4)
    hidden_states = torch.empty(shape, dtype=torch.float64, device=_DEVICE)

    with pytest.raises(RuntimeError):
        head(hidden_states)


@pytest.mark.parametrize(
    "shape",
    [(2, 2), (2, 4), (2, 0), (0, 4), (0, 0)],
    ids=["short-width", "long-width", "zero-width", "empty-long-width", "empty-zero-width"],
)
def test_lh23_incompatible_feature_widths_raise_native_matmul_error(shape: tuple[int, int]):
    head = _head(d_model=3, vocab_size=4)
    hidden_states = torch.empty(shape, dtype=torch.float64, device=_DEVICE)

    with pytest.raises(RuntimeError):
        head(hidden_states)


@pytest.mark.parametrize(
    "hidden_states,error_type",
    [
        ([[1.0, 2.0, 3.0]], TypeError),
        ([[[1.0, 2.0, 3.0]]], TypeError),
        (None, TypeError),
        (1.0, TypeError),
        ({"hidden_states": [[1.0, 2.0, 3.0]]}, TypeError),
        (torch.tensor([[1, 2, 3]], dtype=torch.int64), RuntimeError),
        (torch.tensor([[True, False, True]], dtype=torch.bool), RuntimeError),
    ],
    ids=["list", "nested-list", "none", "scalar", "dict", "integer-tensor", "bool-tensor"],
)
def test_lh24_unsupported_operands_surface_their_native_type_errors(hidden_states, error_type):
    head = _head(d_model=3, vocab_size=4)
    if isinstance(hidden_states, torch.Tensor):
        hidden_states = hidden_states.to(device=_DEVICE)

    with pytest.raises(error_type):
        head(hidden_states)


@pytest.mark.parametrize(
    "input_dtype,weight_dtype",
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
    ids=["float32-input-float64-weight", "float64-input-float32-weight"],
)
def test_lh25_rejects_dtype_mismatch_without_mutation(
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
):
    head = _head(d_model=3, vocab_size=4, dtype=weight_dtype)
    _set_weight(head, _RECTANGULAR_W)
    hidden_states = _tensor(_RECTANGULAR_H, dtype=input_dtype)
    original_hidden = hidden_states.clone()
    original_weight = head.w_lmhead.detach().clone()
    head.w_lmhead.grad = torch.full_like(head.w_lmhead, 4.0)
    original_grad = head.w_lmhead.grad.clone()

    with pytest.raises(RuntimeError):
        head(hidden_states)

    assert torch.equal(hidden_states, original_hidden)
    assert torch.equal(head.w_lmhead, original_weight)
    assert torch.equal(head.w_lmhead.grad, original_grad)


@pytest.mark.parametrize(
    "direction",
    ["cpu-head-cuda-input", "cuda-head-cpu-input"],
    ids=["cpu-head-cuda-input", "cuda-head-cpu-input"],
)
def test_lh26_device_mismatch_surfaces_native_error_without_migration(direction: str):
    if not torch.cuda.is_available():
        print(f"LH26 {direction} not exercised: CUDA is unavailable")
        return

    if direction == "cpu-head-cuda-input":
        head = LMHead(d_model=3, vocab_size=4)
        hidden_states = torch.ones((2, 3), dtype=head.w_lmhead.dtype, device="cuda:0")
    else:
        head = _head(d_model=3, vocab_size=4, dtype=torch.float32)
        hidden_states = torch.ones((2, 3), dtype=head.w_lmhead.dtype, device="cpu")
    original_weight = head.w_lmhead.detach().clone()
    original_device = head.w_lmhead.device

    with pytest.raises(RuntimeError):
        head(hidden_states)

    assert head.w_lmhead.device == original_device
    assert torch.equal(head.w_lmhead, original_weight)


@pytest.mark.parametrize("d_model,vocab_size", [(2, 3), (3, 4), (5, 2)], ids=["2x3", "3x4", "5x2"])
def test_lh27_xavier_initialization_is_registered_finite_and_within_its_bound(
    d_model: int,
    vocab_size: int,
):
    head = _head(d_model=d_model, vocab_size=vocab_size)
    weight = head.w_lmhead
    bound = math.sqrt(6.0 / (d_model + vocab_size))

    assert weight.shape == (d_model, vocab_size)
    assert weight.requires_grad
    assert list(dict(head.named_parameters())) == ["w_lmhead"]
    assert bool(torch.isfinite(weight).all())
    assert bool((weight.abs() <= bound).all())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_lh28_finite_cancellation_and_zero_rows_match_scalar_sums(dtype: torch.dtype):
    head = _head(d_model=3, vocab_size=2, dtype=dtype)
    _set_weight(head, [[1.0, 2.0], [1.0, -1.0], [1.0, 0.5]])
    hidden_states = _tensor([[1024.0, -1024.0, 0.25], [0.0, 0.0, 0.0]], dtype=dtype)
    expected = _scalar_projection(hidden_states, head.w_lmhead)
    logits = head(hidden_states)

    assert bool(torch.isfinite(logits).all())
    torch.testing.assert_close(logits[1], torch.zeros(2, dtype=dtype, device=_DEVICE))
    _assert_close(logits, expected)
