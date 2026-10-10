"""Verify learned token-plus-position input construction with independent oracles."""

import inspect

import pytest
import torch
from torch import nn

from llm_foundations.embeddings.input_embedding import InputEmbedding


_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Input embedding tests selected device: {_TEST_DEVICE}")

_TOKEN_VALUES = [
    [0, 1, 2],
    [10, 11, 12],
    [20, 21, 22],
    [30, 31, 32],
    [40, 41, 42],
]
_POSITION_VALUES = [
    [0.25, -0.5, 1],
    [1.5, 0, -2],
    [-1, 2, 0.5],
    [3, -1, -0.25],
    [-2, 0.75, 4],
    [0.5, 1.25, -3],
]
_CANONICAL_IDS = [3, 1, 3, 0]
_CANONICAL_OUTPUT = [
    [30.25, 30.5, 33],
    [11.5, 11, 10],
    [29, 33, 32.5],
    [3, 0, 1.75],
]
_UPSTREAM = [
    [1, 2, -1],
    [-2, 0.5, 3],
    [4, -3, 2],
    [0.25, 0, -0.5],
]
_TOKEN_GRADIENT = [
    [0.25, 0, -0.5],
    [-2, 0.5, 3],
    [0, 0, 0],
    [5, -1, 1],
    [0, 0, 0],
]
_POSITION_GRADIENT = [
    [1, 2, -1],
    [-2, 0.5, 3],
    [4, -3, 2],
    [0.25, 0, -0.5],
    [0, 0, 0],
    [0, 0, 0],
]


def _new_module(
    vocab_size: int = 5,
    max_seq_len: int = 6,
    d_model: int = 3,
    dtype: torch.dtype = torch.float64,
) -> InputEmbedding:
    return InputEmbedding(vocab_size, max_seq_len, d_model).to(
        device=_TEST_DEVICE,
        dtype=dtype,
    )


def _set_tables(
    module: InputEmbedding,
    token_values,
    position_values,
) -> None:
    with torch.no_grad():
        module.token_embed.copy_(
            torch.tensor(
                token_values,
                dtype=module.token_embed.dtype,
                device=module.token_embed.device,
            )
        )
        module.pos_embed.copy_(
            torch.tensor(
                position_values,
                dtype=module.pos_embed.dtype,
                device=module.pos_embed.device,
            )
        )


def _fixture_module(dtype: torch.dtype = torch.float64) -> InputEmbedding:
    module = _new_module(dtype=dtype)
    _set_tables(module, _TOKEN_VALUES, _POSITION_VALUES)
    return module


def _ids(values, dtype: torch.dtype = torch.long) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_TEST_DEVICE)


def _values(values, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=_TEST_DEVICE)



def _weighted_objective(
    module: InputEmbedding,
    token_ids: torch.Tensor,
    upstream: torch.Tensor,
) -> torch.Tensor:
    return (module(token_ids) * upstream).sum()


def _expected_gradients(dtype: torch.dtype = torch.float64):
    return (
        _values(_TOKEN_GRADIENT, dtype),
        _values(_POSITION_GRADIENT, dtype),
    )


def test_ie01_ie02_api_binding_and_registered_tables():
    assert issubclass(InputEmbedding, nn.Module)
    assert callable(InputEmbedding.forward)

    constructor_parameters = inspect.signature(InputEmbedding.__init__).parameters
    assert tuple(constructor_parameters) == (
        "self",
        "vocab_size",
        "max_seq_len",
        "d_model",
    )
    assert all(
        parameter.default is inspect.Parameter.empty
        for name, parameter in constructor_parameters.items()
        if name != "self"
    )
    assert tuple(inspect.signature(InputEmbedding.forward).parameters) == (
        "self",
        "token_ids",
    )

    module = _fixture_module()
    named_parameters = dict(module.named_parameters())
    state = module.state_dict()
    assert set(named_parameters) == {"token_embed", "pos_embed"}
    assert set(state) == {"token_embed", "pos_embed"}
    assert module.token_embed is named_parameters["token_embed"]
    assert module.pos_embed is named_parameters["pos_embed"]
    assert module.token_embed.shape == (5, 3)
    assert module.pos_embed.shape == (6, 3)
    assert module.token_embed.requires_grad
    assert module.pos_embed.requires_grad
    assert module.token_embed.untyped_storage().data_ptr() != (
        module.pos_embed.untyped_storage().data_ptr()
    )

    output = module(_ids(_CANONICAL_IDS))
    assert isinstance(output, torch.Tensor)
    assert output.shape == (4, 3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ie04_ie07_ie11_values_boundaries_and_placement(dtype):
    module = _fixture_module(dtype=dtype)
    expected_canonical = _values(_CANONICAL_OUTPUT, dtype=dtype)
    expected_endpoints = _values(
        [
            [0.25, 0.5, 3],
            [41.5, 41, 40],
        ],
        dtype=dtype,
    )

    canonical = module(_ids(_CANONICAL_IDS))
    endpoints = module(_ids([0, 4]))

    assert torch.equal(canonical, expected_canonical)
    assert torch.equal(endpoints, expected_endpoints)
    assert canonical.shape == (4, 3)
    assert endpoints.shape == (2, 3)
    assert canonical.dtype == dtype
    assert canonical.device == _TEST_DEVICE
    assert module.token_embed.dtype == dtype
    assert module.pos_embed.dtype == dtype
    assert module.token_embed.device == _TEST_DEVICE
    assert module.pos_embed.device == _TEST_DEVICE
    assert module.token_embed.shape == (5, 3)
    assert module.pos_embed.shape == (6, 3)

def test_ie05_repeated_tokens_use_distinct_position_rows():
    output = _fixture_module()(_ids([3, 3]))
    expected = _values(
        [
            [30.25, 30.5, 33],
            [31.5, 31, 30],
        ]
    )

    assert torch.equal(output, expected)


def test_ie06_large_token_id_selects_position_zero():
    token_values = [[0, 0, 0] for _ in range(18)]
    token_values[17] = [7, 8, 9]
    position_values = [[0, 0, 0], [0, 0, 0]]
    position_values[0] = [0.25, -0.5, 1]
    module = _new_module(vocab_size=18, max_seq_len=2, d_model=3)
    _set_tables(module, token_values, position_values)

    output = module(_ids([17]))

    assert torch.equal(output, _values([[7.25, 7.5, 10]]))


def test_ie08_singleton_vocabulary_width_and_sequence_are_preserved():
    singleton_vocab = _new_module(vocab_size=1, max_seq_len=2, d_model=3)
    _set_tables(
        singleton_vocab,
        [[2, -1, 4]],
        [[0.25, 0.5, -2], [-1, 2, 0]],
    )
    vocab_output = singleton_vocab(_ids([0, 0]))

    singleton_width = _new_module(vocab_size=2, max_seq_len=2, d_model=1)
    _set_tables(singleton_width, [[10], [20]], [[0.5], [-1]])
    width_output = singleton_width(_ids([1, 0]))

    sequence_output = _fixture_module()(_ids([4]))

    assert vocab_output.shape == (2, 3)
    assert torch.equal(vocab_output, _values([[2.25, -0.5, 2], [1, 1, 4]]))
    assert width_output.shape == (2, 1)
    assert torch.equal(width_output, _values([[20.5], [9]]))
    assert sequence_output.shape == (1, 3)
    assert torch.equal(sequence_output, _values([[40.25, 40.5, 43]]))


def test_ie09_full_position_capacity_is_usable():
    output = _fixture_module()(_ids([0, 0, 0, 0, 0, 0]))
    expected = _values(
        [
            [0.25, 0.5, 3],
            [1.5, 1, 0],
            [-1, 3, 2.5],
            [3, 0, 1.75],
            [-2, 1.75, 6],
            [0.5, 2.25, -1],
        ]
    )

    assert output.shape == (6, 3)
    assert torch.equal(output, expected)


def test_ie10_forward_preserves_state_rng_and_restarts_positions():
    module = _fixture_module()
    token_ids = _ids(_CANONICAL_IDS)
    ids_before = token_ids.clone()
    parameters = (module.token_embed, module.pos_embed)
    parameter_values = tuple(parameter.detach().clone() for parameter in parameters)
    state_keys = tuple(module.state_dict())
    cpu_rng_state = torch.random.get_rng_state().clone()
    cuda_rng_state = (
        torch.cuda.get_rng_state(_TEST_DEVICE).clone()
        if _TEST_DEVICE.type == "cuda"
        else None
    )

    first = module(token_ids)
    assert module.token_embed is parameters[0]
    assert module.pos_embed is parameters[1]
    second = module(token_ids)
    assert module.token_embed is parameters[0]
    assert module.pos_embed is parameters[1]
    shorter = module(token_ids[:2])
    assert module.token_embed is parameters[0]
    assert module.pos_embed is parameters[1]

    assert torch.equal(first, second)
    assert torch.equal(shorter, _values(_CANONICAL_OUTPUT[:2]))
    assert torch.equal(token_ids, ids_before)
    assert tuple(module.state_dict()) == state_keys
    assert torch.equal(module.token_embed, parameter_values[0])
    assert torch.equal(module.pos_embed, parameter_values[1])
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    if cuda_rng_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(_TEST_DEVICE), cuda_rng_state)


def test_ie12_each_embedding_table_contributes_to_output():
    zero_tokens = _new_module()
    _set_tables(zero_tokens, [[0, 0, 0] for _ in range(5)], _POSITION_VALUES)
    token_branch_output = zero_tokens(_ids(_CANONICAL_IDS))

    zero_positions = _new_module()
    _set_tables(zero_positions, _TOKEN_VALUES, [[0, 0, 0] for _ in range(6)])
    position_branch_output = zero_positions(_ids(_CANONICAL_IDS))

    assert torch.equal(
        token_branch_output,
        _values(_POSITION_VALUES[:4]),
    )
    assert torch.equal(
        position_branch_output,
        _values([[30, 31, 32], [10, 11, 12], [30, 31, 32], [0, 1, 2]]),
    )


def test_ie13_table_perturbations_affect_only_selected_rows_and_features():
    token_module = _fixture_module()
    token_original = token_module.token_embed[3, 1].detach().clone()
    token_before = token_module(_ids(_CANONICAL_IDS)).detach().clone()
    with torch.no_grad():
        token_module.token_embed[3, 1].add_(2)
    token_after = token_module(_ids(_CANONICAL_IDS)).detach().clone()
    with torch.no_grad():
        token_module.token_embed[3, 1].copy_(token_original)
    token_delta = _values(
        [
            [0, 2, 0],
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ]
    )

    position_module = _fixture_module()
    position_original = position_module.pos_embed[2, 2].detach().clone()
    position_before = position_module(_ids(_CANONICAL_IDS)).detach().clone()
    with torch.no_grad():
        position_module.pos_embed[2, 2].add_(4)
    position_after = position_module(_ids(_CANONICAL_IDS)).detach().clone()
    with torch.no_grad():
        position_module.pos_embed[2, 2].copy_(position_original)
    position_delta = _values(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 4],
            [0, 0, 0],
        ]
    )

    assert torch.equal(token_after - token_before, token_delta)
    assert torch.equal(position_after - position_before, position_delta)
    assert torch.equal(token_module.token_embed[3, 1], token_original)
    assert torch.equal(position_module.pos_embed[2, 2], position_original)


def test_ie14_swapping_tokens_keeps_positions_attached_to_sequence_slots():
    module = _fixture_module()
    original = module(_ids([3, 1, 3, 0]))
    swapped = module(_ids([1, 3, 3, 0]))
    expected_swapped = _values(
        [
            [10.25, 10.5, 13],
            [31.5, 31, 30],
            [29, 33, 32.5],
            [3, 0, 1.75],
        ]
    )

    assert torch.equal(swapped, expected_swapped)
    assert not torch.equal(swapped, original[[1, 0, 2, 3]])


def test_ie15_ie16_ie18_full_analytical_gradients_and_no_parameter_update():
    module = _fixture_module()
    token_ids = _ids(_CANONICAL_IDS)
    upstream = _values(_UPSTREAM)
    expected_token_gradient, expected_position_gradient = _expected_gradients()
    parameters = (module.token_embed, module.pos_embed)
    parameter_values = tuple(parameter.detach().clone() for parameter in parameters)
    module.zero_grad(set_to_none=True)

    _weighted_objective(module, token_ids, upstream).backward()

    assert module.token_embed is parameters[0]
    assert module.pos_embed is parameters[1]
    assert module.token_embed.grad is not None
    assert module.pos_embed.grad is not None
    assert torch.isfinite(module.token_embed.grad).all().item()
    assert torch.isfinite(module.pos_embed.grad).all().item()
    assert torch.equal(module.token_embed.grad, expected_token_gradient)
    assert torch.equal(module.pos_embed.grad, expected_position_gradient)
    assert torch.equal(module.token_embed, parameter_values[0])
    assert torch.equal(module.pos_embed, parameter_values[1])


def test_ie17_connected_zero_objective_produces_zero_table_gradients():
    module = _fixture_module()
    module.zero_grad(set_to_none=True)
    output = module(_ids(_CANONICAL_IDS))
    zero_upstream = torch.zeros_like(output)

    (output * zero_upstream).sum().backward()

    assert module.token_embed.grad is not None
    assert module.pos_embed.grad is not None
    assert torch.equal(module.token_embed.grad, torch.zeros_like(module.token_embed))
    assert torch.equal(module.pos_embed.grad, torch.zeros_like(module.pos_embed))


def test_ie19_gradients_accumulate_then_reset_to_one_analytical_copy():
    module = _fixture_module()
    token_ids = _ids(_CANONICAL_IDS)
    upstream = _values(_UPSTREAM)
    expected_token_gradient, expected_position_gradient = _expected_gradients()

    _weighted_objective(module, token_ids, upstream).backward()
    _weighted_objective(module, token_ids, upstream).backward()

    assert torch.equal(module.token_embed.grad, 2 * expected_token_gradient)
    assert torch.equal(module.pos_embed.grad, 2 * expected_position_gradient)

    module.zero_grad(set_to_none=True)
    _weighted_objective(module, token_ids, upstream).backward()

    assert torch.equal(module.token_embed.grad, expected_token_gradient)
    assert torch.equal(module.pos_embed.grad, expected_position_gradient)


def test_ie20_float64_finite_differences_match_analytical_and_autograd_gradients():
    module = _fixture_module(dtype=torch.float64)
    token_ids = _ids(_CANONICAL_IDS)
    upstream = _values(_UPSTREAM)
    module.zero_grad(set_to_none=True)
    _weighted_objective(module, token_ids, upstream).backward()
    probes = [
        (module.token_embed, (3, 1), -1.0),
        (module.pos_embed, (2, 2), 2.0),
        (module.token_embed, (2, 0), 0.0),
        (module.pos_embed, (5, 0), 0.0),
    ]

    for parameter, index, expected_gradient in probes:
        analytical_gradient = (
            module.token_embed.grad[index]
            if parameter is module.token_embed
            else module.pos_embed.grad[index]
        )
        assert analytical_gradient.item() == expected_gradient
        original = parameter[index].detach().clone()
        step = 1e-5
        try:
            with torch.no_grad():
                parameter[index].copy_(original + step)
                plus = _weighted_objective(module, token_ids, upstream).item()
                parameter[index].copy_(original - step)
                minus = _weighted_objective(module, token_ids, upstream).item()
        finally:
            with torch.no_grad():
                parameter[index].copy_(original)
        numerical_gradient = (plus - minus) / (2 * step)
        torch.testing.assert_close(
            _values(numerical_gradient),
            _values(expected_gradient),
            rtol=1e-6,
            atol=1e-8,
        )
        torch.testing.assert_close(
            _values(numerical_gradient),
            analytical_gradient.detach(),
            rtol=1e-6,
            atol=1e-8,
        )


def test_ie21_state_dict_round_trip_and_train_eval_are_independent():
    original = _fixture_module()
    saved_state = {name: value.detach().clone() for name, value in original.state_dict().items()}
    restored = _new_module()
    restored.load_state_dict(saved_state)
    token_ids = _ids(_CANONICAL_IDS)
    expected = _values(_CANONICAL_OUTPUT)

    original.train()
    restored.train()
    original_training_output = original(token_ids)
    restored_training_output = restored(token_ids)
    original.eval()
    restored.eval()
    original_eval_output = original(token_ids)
    restored_eval_output = restored(token_ids)

    assert set(saved_state) == {"token_embed", "pos_embed"}
    assert torch.equal(restored.token_embed, saved_state["token_embed"])
    assert torch.equal(restored.pos_embed, saved_state["pos_embed"])
    assert torch.equal(original_training_output, expected)
    assert torch.equal(restored_training_output, expected)
    assert torch.equal(original_eval_output, original_training_output)
    assert torch.equal(restored_eval_output, restored_training_output)
    assert original.token_embed.untyped_storage().data_ptr() != (
        restored.token_embed.untyped_storage().data_ptr()
    )
    assert original.pos_embed.untyped_storage().data_ptr() != (
        restored.pos_embed.untyped_storage().data_ptr()
    )

    original_token_values = original.token_embed.detach().clone()
    with torch.no_grad():
        restored.token_embed[3, 0].add_(1)
    assert torch.equal(original.token_embed, original_token_values)
    assert not torch.equal(restored.token_embed, original_token_values)


def test_ie28_noncontiguous_id_view_preserves_values_and_backing_storage():
    module = _fixture_module()
    backing_ids = _ids([3, 99, 1, 99, 3, 99, 0, 99])
    ids_before = backing_ids.clone()
    token_ids = backing_ids[::2]
    view_before = token_ids.clone()
    assert not token_ids.is_contiguous()

    output = module(token_ids)

    assert torch.equal(output, _values(_CANONICAL_OUTPUT))
    assert torch.equal(token_ids, view_before)
    assert torch.equal(backing_ids, ids_before)


def test_ie29_repeated_token_gradient_contributions_cancel():
    module = _fixture_module()
    token_ids = _ids([3, 3])
    upstream = _values([[1, 2, -1], [-1, -2, 1]])
    expected_token_gradient = _values(
        [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    )
    expected_position_gradient = _values(
        [
            [1, 2, -1],
            [-1, -2, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )
    module.zero_grad(set_to_none=True)

    _weighted_objective(module, token_ids, upstream).backward()

    assert module.token_embed.grad is not None
    assert module.pos_embed.grad is not None
    assert torch.equal(module.token_embed.grad, expected_token_gradient)
    assert torch.equal(module.pos_embed.grad, expected_position_gradient)
