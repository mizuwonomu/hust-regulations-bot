"""Kiểm tra composition multi-head với KV cache tường minh."""

import math

import pytest
import torch
from torch import nn

from llm_foundations.cached_self_attention import CachedCausalSelfAttention
from llm_foundations.kv_cache.multi_head_attention import CachedMultiHeadAttention


TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected test device: {TEST_DEVICE}")


def _module(
    d_model: int,
    num_heads: int,
    d_k: int,
    d_v: int,
    dtype: torch.dtype = torch.float32,
) -> CachedMultiHeadAttention:
    return CachedMultiHeadAttention(d_model, num_heads, d_k, d_v).to(
        device=TEST_DEVICE, dtype=dtype
    )


def _tensor(values, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(values, device=TEST_DEVICE, dtype=dtype)


def _assert_device(value) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device == TEST_DEVICE
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_device(item)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.device == TEST_DEVICE
    assert expected.device == TEST_DEVICE
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    if actual.dtype == torch.float64:
        rtol, atol = 1e-10, 1e-12
    else:
        rtol, atol = 1e-5, 1e-6
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def _assert_cache_close(actual, expected) -> None:
    assert isinstance(actual, list)
    assert len(actual) == len(expected)
    for actual_pair, expected_pair in zip(actual, expected):
        assert isinstance(actual_pair, tuple)
        assert len(actual_pair) == 2
        for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
            _assert_close(actual_tensor, expected_tensor)


def _clone_cache(cache):
    return [
        (key.detach().clone(), value.detach().clone())
        for key, value in cache
    ]


def _clone_cache_pair(pair):
    return pair[0].detach().clone(), pair[1].detach().clone()


def _clone_result(result):
    return (
        result[0].detach().clone(),
        result[1].detach().clone(),
        _clone_cache(result[2]),
    )


def _set_distinct_weights(module: CachedMultiHeadAttention) -> None:
    with torch.no_grad():
        for head_index, cached_head in enumerate(module.cached_heads):
            parameters = (
                cached_head.attention_head.w_q,
                cached_head.attention_head.w_k,
                cached_head.attention_head.w_v,
            )
            for parameter_index, parameter in enumerate(parameters):
                values = torch.arange(
                    parameter.numel(), device=TEST_DEVICE, dtype=parameter.dtype
                ).reshape_as(parameter)
                values = (values.remainder(7) - 3) * 0.11
                values = values + (head_index + 1) * 0.037 + parameter_index * 0.019
                parameter.copy_(values)
        projection_values = torch.arange(
            module.w_o.numel(), device=TEST_DEVICE, dtype=module.w_o.dtype
        ).reshape_as(module.w_o)
        module.w_o.copy_((projection_values.remainder(9) - 4) * 0.07 + 0.03)


def _copy_real_weights(module: CachedMultiHeadAttention, weights) -> None:
    with torch.no_grad():
        for cached_head, head_weights in zip(module.cached_heads, weights):
            for parameter, values in zip(
                (
                    cached_head.attention_head.w_q,
                    cached_head.attention_head.w_k,
                    cached_head.attention_head.w_v,
                ),
                head_weights,
            ):
                parameter.copy_(_tensor(values, dtype=parameter.dtype))
        module.w_o.copy_(_tensor(weights[-1], dtype=module.w_o.dtype))


def _snapshot_parameters(module: CachedMultiHeadAttention):
    return {
        name: (id(parameter), parameter.detach().clone())
        for name, parameter in module.named_parameters()
    }


def _assert_parameter_snapshot(module: CachedMultiHeadAttention, snapshot) -> None:
    current = dict(module.named_parameters())
    assert current.keys() == snapshot.keys()
    for name, parameter in current.items():
        identity, value = snapshot[name]
        assert id(parameter) == identity
        assert torch.equal(parameter, value)


def _cache_values_from_snapshot(module, sequence: torch.Tensor, snapshot):
    return [
        (
            sequence @ snapshot[f"cached_heads.{head_index}.attention_head.w_k"][1],
            sequence @ snapshot[f"cached_heads.{head_index}.attention_head.w_v"][1],
        )
        for head_index in range(len(module.cached_heads))
    ]


def test_cma01_public_api_keeps_singleton_axes():
    module = _module(d_model=3, num_heads=1, d_k=2, d_v=1)
    x = _tensor([[0.2, -0.1, 0.3]])
    x_new = _tensor([[-0.4, 0.5, 0.1]])

    with torch.no_grad():
        prefill_result = module.prefill(x)
        prefill_snapshot = _clone_result(prefill_result)
        assert isinstance(prefill_result, tuple) and len(prefill_result) == 3
        assert isinstance(prefill_result[2], list) and len(prefill_result[2]) == 1
        assert isinstance(prefill_result[2][0], tuple) and len(prefill_result[2][0]) == 2
        decode_result = module.decode(x_new, prefill_result[2])
        decode_snapshot = _clone_result(decode_result)

    assert isinstance(decode_result, tuple) and len(decode_result) == 3
    assert isinstance(decode_result[2], list) and len(decode_result[2]) == 1
    assert isinstance(decode_result[2][0], tuple) and len(decode_result[2][0]) == 2
    assert prefill_snapshot[0].shape == (1, 3)
    assert prefill_snapshot[1].shape == (1, 1, 1)
    assert prefill_snapshot[2][0][0].shape == (1, 2)
    assert prefill_snapshot[2][0][1].shape == (1, 1)
    assert decode_snapshot[0].shape == (1, 3)
    assert decode_snapshot[1].shape == (1, 1, 2)
    assert decode_snapshot[2][0][0].shape == (2, 2)
    assert decode_snapshot[2][0][1].shape == (2, 1)
    _assert_device(prefill_snapshot)
    _assert_device(decode_snapshot)


def test_cma02_registers_independent_heads_and_rectangular_projection():
    module = _module(d_model=5, num_heads=2, d_k=2, d_v=3)

    assert len(module.cached_heads) == 2
    assert module.cached_heads[0] is not module.cached_heads[1]
    registered_parameters = dict(module.named_parameters())
    head_parameters = []
    for head_index, cached_head in enumerate(module.cached_heads):
        assert isinstance(cached_head, CachedCausalSelfAttention)
        parameters = (
            cached_head.attention_head.w_q,
            cached_head.attention_head.w_k,
            cached_head.attention_head.w_v,
        )
        head_parameters.extend(parameters)
        for parameter_name, parameter, shape in zip(
            ("w_q", "w_k", "w_v"), parameters, ((5, 2), (5, 2), (5, 3))
        ):
            assert parameter.shape == shape
            registered_name = (
                f"cached_heads.{head_index}.attention_head.{parameter_name}"
            )
            assert registered_name in registered_parameters
            assert registered_parameters[registered_name] is parameter
    assert len({id(parameter) for parameter in head_parameters}) == 6
    assert isinstance(module.w_o, nn.Parameter)
    assert module.w_o.shape == (6, 5)
    assert "w_o" in registered_parameters
    assert registered_parameters["w_o"] is module.w_o


def test_cma03_prefill_dispatches_same_input_and_preserves_head_cache_order():
    module = _module(d_model=4, num_heads=2, d_k=2, d_v=3)
    _set_distinct_weights(module)
    x = _tensor([[0.2, -0.3, 0.1, 0.4], [-0.5, 0.6, 0.3, -0.2]])
    calls = [[] for _ in module.cached_heads]

    for head_index, cached_head in enumerate(module.cached_heads):
        original = cached_head.prefill

        def record_prefill(input_x, original=original, head_index=head_index):
            result = original(input_x)
            calls[head_index].append((input_x, result))
            return result

        cached_head.prefill = record_prefill

    with torch.no_grad():
        result = module.prefill(x)

    assert [len(head_calls) for head_calls in calls] == [1, 1]
    captured_caches = []
    for head_index, head_calls in enumerate(calls):
        called_input, child_result = head_calls[0]
        assert called_input is x
        assert child_result[0].shape == (2, 3)
        assert child_result[1].shape == (2, 2)
        captured_caches.append(child_result[2])
        _assert_close(result[2][head_index][0], child_result[2][0])
        _assert_close(result[2][head_index][1], child_result[2][1])
    _assert_cache_close(result[2], captured_caches)
    assert not torch.equal(captured_caches[0][0], captured_caches[1][0])
    assert not torch.equal(captured_caches[0][1], captured_caches[1][1])
    _assert_device(result)


def test_cma04_decode_routes_each_cache_and_continues_one_row_calls():
    module = _module(d_model=4, num_heads=2, d_k=2, d_v=2)
    _set_distinct_weights(module)
    prefix = _tensor([[0.1, -0.2, 0.3, 0.4], [-0.3, 0.5, 0.2, -0.1]])
    x_first = _tensor([[0.6, -0.4, 0.1, 0.2]])
    x_second = _tensor([[-0.2, 0.3, -0.5, 0.7]])
    calls = [[] for _ in module.cached_heads]

    with torch.no_grad():
        initial_caches = module.prefill(prefix)[2]
        initial_cache_snapshot = _clone_cache(initial_caches)

    for head_index, cached_head in enumerate(module.cached_heads):
        original_decode = cached_head.decode

        def forbidden_prefill(*args, **kwargs):
            pytest.fail("decode khong duoc goi prefill")

        def record_decode(input_x, cache, original=original_decode, head_index=head_index):
            input_snapshot = input_x.detach().clone()
            incoming_snapshot = _clone_cache_pair(cache)
            result = original(input_x, cache)
            returned_snapshot = _clone_cache_pair(result[2])
            calls[head_index].append(
                (
                    input_x,
                    input_snapshot,
                    cache,
                    incoming_snapshot,
                    returned_snapshot,
                )
            )
            return result

        cached_head.prefill = forbidden_prefill
        cached_head.decode = record_decode

    with torch.no_grad():
        first_incoming_refs = tuple(initial_caches)
        first_result = module.decode(x_first, initial_caches)
        first_result_snapshot = _clone_result(first_result)
        second_incoming_refs = tuple(first_result[2])
        second_result = module.decode(x_second, first_result[2])
        second_result_snapshot = _clone_result(second_result)

    assert [len(head_calls) for head_calls in calls] == [2, 2]
    assert not torch.equal(initial_cache_snapshot[0][0], initial_cache_snapshot[1][0])
    assert not torch.equal(initial_cache_snapshot[0][1], initial_cache_snapshot[1][1])
    first_result_snapshots = []
    second_result_snapshots = []
    for head_index, head_calls in enumerate(calls):
        first_input, first_input_snapshot, first_cache, first_incoming, first_returned = head_calls[0]
        second_input, second_input_snapshot, second_cache, second_incoming, second_returned = head_calls[1]
        assert first_input is x_first
        assert second_input is x_second
        _assert_close(first_input_snapshot, x_first)
        _assert_close(second_input_snapshot, x_second)
        assert first_cache is first_incoming_refs[head_index]
        assert second_cache is second_incoming_refs[head_index]
        _assert_cache_close([first_incoming], [initial_cache_snapshot[head_index]])
        _assert_cache_close([second_incoming], [first_returned])
        _assert_cache_close([first_result_snapshot[2][head_index]], [first_returned])
        _assert_cache_close([second_result_snapshot[2][head_index]], [second_returned])
        assert first_returned[0].shape == (3, 2)
        assert first_returned[1].shape == (3, 2)
        assert second_returned[0].shape == (4, 2)
        assert second_returned[1].shape == (4, 2)
        first_result_snapshots.append(first_returned)
        second_result_snapshots.append(second_returned)
    _assert_device(first_result_snapshot)
    _assert_device(second_result_snapshot)
    _assert_device((first_result_snapshots, second_result_snapshots))


def test_cma05_concatenates_head_features_then_projects_outputs():
    module = _module(d_model=3, num_heads=2, d_k=1, d_v=2)
    x = _tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    x_new = _tensor([[-0.1, 0.3, 0.2]])
    prefill_outputs = (
        _tensor([[1.0, 2.0], [3.0, 4.0]]),
        _tensor([[5.0, 6.0], [7.0, 8.0]]),
    )
    decode_outputs = (_tensor([[-1.0, 3.0]]), _tensor([[2.0, -2.0]]))
    prefill_weights = (
        _tensor([[1.0, 0.0], [0.2, 0.8]]),
        _tensor([[1.0, 0.0], [0.6, 0.4]]),
    )
    decode_weights = (_tensor([[0.2, 0.3, 0.5]]),) * 2
    controlled_prefill_caches = (
        (_tensor([[31.0], [32.0]]), _tensor([[41.0, 42.0], [43.0, 44.0]])),
        (_tensor([[51.0], [52.0]]), _tensor([[61.0, 62.0], [63.0, 64.0]])),
    )
    controlled_decode_caches = (
        (_tensor([[31.0], [32.0], [33.0]]), _tensor([[41.0, 42.0], [43.0, 44.0], [45.0, 46.0]])),
        (_tensor([[51.0], [52.0], [53.0]]), _tensor([[61.0, 62.0], [63.0, 64.0], [65.0, 66.0]])),
    )
    with torch.no_grad():
        module.w_o.copy_(
            _tensor([[1.0, 0.0, -1.0], [0.0, 2.0, 1.0], [-1.0, 1.0, 0.0], [2.0, -1.0, 1.0]])
        )

    for head_index, cached_head in enumerate(module.cached_heads):
        def controlled_prefill(
            input_x,
            head_index=head_index,
        ):
            return (
                prefill_outputs[head_index],
                prefill_weights[head_index],
                controlled_prefill_caches[head_index],
            )

        def controlled_decode(
            input_x,
            cache,
            head_index=head_index,
        ):
            return (
                decode_outputs[head_index],
                decode_weights[head_index],
                controlled_decode_caches[head_index],
            )

        cached_head.prefill = controlled_prefill
        cached_head.decode = controlled_decode

    with torch.no_grad():
        prefill_result = module.prefill(x)
        decode_result = module.decode(x_new, prefill_result[2])

    _assert_close(prefill_result[0], _tensor([[8.0, 3.0, 7.0], [12.0, 7.0, 9.0]]))
    _assert_close(decode_result[0], _tensor([[-7.0, 10.0, 2.0]]))
    _assert_device(prefill_result)
    _assert_device(decode_result)


def test_cma06_stacks_attention_weights_by_head_on_leading_axis():
    module = _module(d_model=3, num_heads=2, d_k=1, d_v=1)
    x = _tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]])
    x_new = _tensor([[0.4, -0.2, 0.1]])
    prefill_matrices = (
        _tensor([[1.0, 0.0], [0.25, 0.75]]),
        _tensor([[1.0, 0.0], [0.6, 0.4]]),
    )
    decode_matrices = (
        _tensor([[0.2, 0.3, 0.5]]),
        _tensor([[0.6, 0.1, 0.3]]),
    )

    for head_index, cached_head in enumerate(module.cached_heads):
        def controlled_prefill(input_x, head_index=head_index):
            return (
                _tensor([[0.1], [0.2]]),
                prefill_matrices[head_index],
                (_tensor([[1.0], [2.0]]), _tensor([[3.0], [4.0]])),
            )

        def controlled_decode(input_x, cache, head_index=head_index):
            return (
                _tensor([[0.3]]),
                decode_matrices[head_index],
                (_tensor([[1.0], [2.0], [3.0]]), _tensor([[3.0], [4.0], [5.0]])),
            )

        cached_head.prefill = controlled_prefill
        cached_head.decode = controlled_decode

    with torch.no_grad():
        prefill_result = module.prefill(x)
        decode_result = module.decode(x_new, prefill_result[2])

    _assert_close(prefill_result[1], torch.stack(prefill_matrices, dim=0))
    _assert_close(decode_result[1], torch.stack(decode_matrices, dim=0))
    assert prefill_result[1].shape == (2, 2, 2)
    assert decode_result[1].shape == (2, 1, 3)
    _assert_device(prefill_result)
    _assert_device(decode_result)


def _full_causal_reference(module, sequence, weights_snapshot):
    head_outputs = []
    attention_matrices = []
    projected_caches = []
    causal_mask = torch.triu(
        torch.ones(
            sequence.shape[0], sequence.shape[0],
            dtype=torch.bool,
            device=TEST_DEVICE,
        ),
        diagonal=1,
    )
    for head_index in range(len(module.cached_heads)):
        w_q, w_k, w_v = weights_snapshot[head_index]
        queries = sequence @ w_q
        keys = sequence @ w_k
        values = sequence @ w_v
        scores = (queries @ keys.transpose(0, 1)) / math.sqrt(w_q.shape[1])
        scores = scores.masked_fill(causal_mask, -torch.inf)
        attention = torch.softmax(scores, dim=-1)
        head_outputs.append(attention @ values)
        attention_matrices.append(attention)
        projected_caches.append((keys, values))
    concatenated = torch.cat(head_outputs, dim=-1)
    full_output = concatenated @ weights_snapshot[-1]
    return full_output, torch.stack(attention_matrices, dim=0), projected_caches


def test_cma07_real_cached_execution_matches_independent_full_causal_oracle():
    module = _module(d_model=5, num_heads=2, d_k=2, d_v=3)
    weights = [
        (
            [[0.20, -0.10], [0.30, 0.40], [-0.20, 0.50], [0.10, 0.25], [0.40, -0.30]],
            [[-0.10, 0.30], [0.20, -0.40], [0.50, 0.10], [-0.30, 0.20], [0.25, 0.15]],
            [[0.15, -0.20, 0.30], [0.25, 0.10, -0.35], [-0.40, 0.20, 0.05], [0.30, -0.15, 0.25], [-0.10, 0.35, 0.20]],
        ),
        (
            [[-0.15, 0.25], [0.45, -0.20], [0.10, 0.35], [-0.30, 0.15], [0.20, 0.40]],
            [[0.35, -0.20], [-0.25, 0.10], [0.15, 0.45], [0.40, -0.30], [-0.10, 0.20]],
            [[-0.25, 0.30, 0.10], [0.20, -0.35, 0.15], [0.40, 0.05, -0.20], [-0.10, 0.25, 0.35], [0.30, -0.15, -0.05]],
        ),
        [
            [0.20, -0.10, 0.30, 0.05, -0.20],
            [-0.15, 0.25, 0.10, -0.30, 0.20],
            [0.35, 0.05, -0.25, 0.15, 0.10],
            [-0.20, 0.30, 0.05, 0.25, -0.10],
            [0.10, -0.35, 0.20, 0.05, 0.30],
            [0.25, 0.15, -0.10, -0.20, 0.35],
        ],
    ]
    _copy_real_weights(module, weights)
    prefix = _tensor([[0.4, -0.2, 0.1, 0.3, -0.5], [-0.1, 0.6, -0.4, 0.2, 0.3]])
    x_new = _tensor([[0.25, -0.35, 0.45, -0.15, 0.05]])
    sequence = torch.cat((prefix, x_new), dim=0)
    parameter_snapshot = [
        tuple(
            parameter.detach().clone()
            for parameter in (
                cached_head.attention_head.w_q,
                cached_head.attention_head.w_k,
                cached_head.attention_head.w_v,
            )
        )
        for cached_head in module.cached_heads
    ]
    parameter_snapshot.append(module.w_o.detach().clone())

    with torch.no_grad():
        expected_output, expected_weights, expected_caches = _full_causal_reference(
            module, sequence, parameter_snapshot
        )
        prefill_result = module.prefill(prefix)
        prefill_snapshot = _clone_result(prefill_result)
        decode_result = module.decode(x_new, prefill_result[2])
        decode_snapshot = _clone_result(decode_result)

    _assert_close(prefill_snapshot[0], expected_output[:2])
    _assert_close(prefill_snapshot[1], expected_weights[:, :2, :2])
    _assert_cache_close(prefill_snapshot[2], [
        (keys[:2], values[:2]) for keys, values in expected_caches
    ])
    _assert_close(decode_snapshot[0], expected_output[-1:])
    _assert_close(decode_snapshot[1], expected_weights[:, -1:, :])
    _assert_cache_close(decode_snapshot[2], expected_caches)
    _assert_device(prefill_snapshot)
    _assert_device(decode_snapshot)


def test_cma08_explicit_caches_isolate_sequences_and_preserve_parameters():
    module = _module(d_model=4, num_heads=2, d_k=2, d_v=2)
    _set_distinct_weights(module)
    prefix_a = _tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.6, -0.2, 0.1]])
    prefix_b = _tensor([[-0.3, 0.2, 0.1, -0.4], [0.5, -0.6, 0.3, 0.2]])
    a_one = _tensor([[0.7, -0.2, 0.1, -0.3]])
    b_one = _tensor([[-0.1, 0.4, -0.5, 0.6]])
    a_two = _tensor([[-0.4, 0.3, 0.5, -0.2]])
    parameter_snapshot = _snapshot_parameters(module)
    head_identities = [id(cached_head) for cached_head in module.cached_heads]

    with torch.no_grad():
        prefill_a = module.prefill(prefix_a)
        caches_a = prefill_a[2]
        caches_a_snapshot = _clone_result(prefill_a)
        prefill_b = module.prefill(prefix_b)
        caches_b = prefill_b[2]
        caches_b_snapshot = _clone_result(prefill_b)
        decoded_a_one_result = module.decode(a_one, caches_a)
        decoded_a_one = decoded_a_one_result[2]
        decoded_a_one_snapshot = _clone_result(decoded_a_one_result)
        decoded_b_one_result = module.decode(b_one, caches_b)
        decoded_b_one = decoded_b_one_result[2]
        decoded_b_one_snapshot = _clone_result(decoded_b_one_result)
        decoded_a_two_result = module.decode(a_two, decoded_a_one)
        decoded_a_two_snapshot = _clone_result(decoded_a_two_result)

        expected_a_one = _cache_values_from_snapshot(
            module, torch.cat((prefix_a, a_one), dim=0), parameter_snapshot
        )
        expected_b_one = _cache_values_from_snapshot(
            module, torch.cat((prefix_b, b_one), dim=0), parameter_snapshot
        )
        expected_a_two = _cache_values_from_snapshot(
            module, torch.cat((prefix_a, a_one, a_two), dim=0), parameter_snapshot
        )

    _assert_cache_close(
        caches_a_snapshot[2], _cache_values_from_snapshot(module, prefix_a, parameter_snapshot)
    )
    _assert_cache_close(
        caches_b_snapshot[2], _cache_values_from_snapshot(module, prefix_b, parameter_snapshot)
    )
    _assert_cache_close(decoded_a_one_snapshot[2], expected_a_one)
    _assert_cache_close(decoded_b_one_snapshot[2], expected_b_one)
    _assert_cache_close(decoded_a_two_snapshot[2], expected_a_two)
    assert [pair[0].shape[0] for pair in decoded_a_one_snapshot[2]] == [3, 3]
    assert [pair[0].shape[0] for pair in decoded_b_one_snapshot[2]] == [3, 3]
    assert [pair[0].shape[0] for pair in decoded_a_two_snapshot[2]] == [4, 4]
    assert [id(cached_head) for cached_head in module.cached_heads] == head_identities
    _assert_parameter_snapshot(module, parameter_snapshot)
    _assert_device((
        caches_a_snapshot, caches_b_snapshot, decoded_a_one_snapshot,
        decoded_b_one_snapshot, decoded_a_two_snapshot,
    ))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
def test_cma09_preserves_dtype_and_selected_device(dtype):
    module = _module(d_model=4, num_heads=2, d_k=2, d_v=3, dtype=dtype)
    prefix = _tensor([[0.2, -0.1, 0.4, 0.3], [-0.5, 0.6, -0.2, 0.1]], dtype=dtype)
    x_new = _tensor([[0.7, -0.2, 0.1, -0.3]], dtype=dtype)

    with torch.no_grad():
        prefill_result = module.prefill(prefix)
        decode_result = module.decode(x_new, prefill_result[2])

    for parameter in module.parameters():
        assert parameter.dtype == dtype
        assert parameter.device == TEST_DEVICE
    for result in (prefill_result, decode_result):
        _assert_device(result)
        tensors = []

        def collect(value):
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    collect(item)

        collect(result)
        assert tensors
        for tensor in tensors:
            assert tensor.dtype == dtype
            assert tensor.device == TEST_DEVICE
            assert torch.isfinite(tensor).all()
