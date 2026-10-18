"""Kiểm tra prefill, decode và tái sử dụng KV cache của attention một head."""

import math

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from llm_foundations.cached_self_attention import CachedCausalSelfAttention

_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected test device: {_TEST_DEVICE}")
_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float64: {"rtol": 1e-10, "atol": 1e-12},
}


def _tensor(values, dtype=torch.float32):
    """Tạo tensor trên device và dtype đã chọn."""
    return torch.tensor(values, dtype=dtype, device=_TEST_DEVICE)


def _parameters(head):
    """Trả về ba projection parameter theo API hiện có."""
    attention_head = head.attention_head
    return {
        "w_q": attention_head.w_q,
        "w_k": attention_head.w_k,
        "w_v": attention_head.w_v,
    }


def _make_head(d_model, d_k, d_v, w_q, w_k, w_v, dtype=torch.float32):
    """Tạo head mới và chép các projection weight đã biết."""
    head = CachedCausalSelfAttention(d_model, d_k, d_v).to(
        device=_TEST_DEVICE, dtype=dtype
    )
    with torch.no_grad():
        parameters = _parameters(head)
        parameters["w_q"].copy_(w_q)
        parameters["w_k"].copy_(w_k)
        parameters["w_v"].copy_(w_v)
    return head


def _snapshot_parameters(head):
    """Lưu identity và giá trị của các projection parameter."""
    return {
        name: (id(parameter), parameter.detach().clone())
        for name, parameter in _parameters(head).items()
    }


def _assert_parameters_unchanged(head, snapshot):
    """Xác nhận lời gọi không sửa hoặc thay projection parameter."""
    for name, parameter in _parameters(head).items():
        identity, value = snapshot[name]
        assert id(parameter) == identity
        assert torch.equal(parameter, value)


def _prefill(head, x):
    """Chạy prefill trong vùng no-grad do test sở hữu."""
    with torch.no_grad():
        return head.prefill(x)


def _decode(head, x_new, cache):
    """Chạy decode trong vùng no-grad do test sở hữu."""
    with torch.no_grad():
        return head.decode(x_new, cache)


def _reference(x, w_q, w_k, w_v):
    """Tính attention độc lập theo từng visible prefix."""
    with torch.no_grad():
        q = x @ w_q
        k = x @ w_k
        v = x @ w_v
        length = x.shape[0]
        weights = torch.zeros((length, length), dtype=x.dtype, device=x.device)
        output = torch.empty((length, v.shape[1]), dtype=x.dtype, device=x.device)
        for query_index in range(length):
            visible_k = k[: query_index + 1]
            scores = q[query_index : query_index + 1] @ visible_k.transpose(0, 1)
            scores = scores / math.sqrt(q.shape[1])
            visible_weights = torch.softmax(scores, dim=-1)
            weights[query_index, : query_index + 1] = visible_weights[0]
            output[query_index : query_index + 1] = visible_weights @ v[: query_index + 1]
    return output, weights, k, v, q


def _assert_close(actual, expected):
    """Kiểm tra hữu hạn và độ gần theo dtype của tensor."""
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _assert_device_dtype(tensors, dtype):
    """Kiểm tra device và dtype của mọi tensor trả về."""
    for tensor in tensors:
        assert tensor.device == _TEST_DEVICE
        assert tensor.dtype == dtype


class _ProjectionOperandObserver(TorchDispatchMode):
    """Quan sát operand của matrix operation dùng projection parameter."""

    def __init__(self, parameters):
        super().__init__()
        self.parameters = parameters
        self.projection_inputs = {name: [] for name in parameters}
        self.other_matrix_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.ops.aten.mm.default, torch.ops.aten.matmul.default):
            left, right = args[:2]
        elif func is torch.ops.aten.addmm.default:
            left, right = args[1:3]
        else:
            return func(*args, **kwargs)

        projection_name = next(
            (name for name, parameter in self.parameters.items() if right is parameter),
            None,
        )
        if projection_name is None:
            self.other_matrix_shapes.append((tuple(left.shape), tuple(right.shape)))
        else:
            with _disable_current_modes():
                self.projection_inputs[projection_name].append(left.detach().clone())
        return func(*args, **kwargs)


def test_ca01_public_returns_keep_rectangular_axes_and_selected_device():
    """Kiểm tra cấu trúc return, shape, dtype và device của hai API."""
    x = _tensor([[1, 0, 2, -1], [0, 1, -1, 2], [2, -1, 0, 1]])
    x_new = _tensor([[0.5, -1, 1.5, 0.25]])
    w_q = _tensor([[0.25, -0.5], [0.5, 0.25], [-0.25, 0.5], [0.75, -0.25]])
    w_k = _tensor([[-0.5, 0.25], [0.25, 0.75], [0.5, -0.25], [-0.25, 0.5]])
    w_v = _tensor(
        [[0.5, 0, -0.25, 0.75, 0.25], [0, 0.5, 0.25, -0.5, 1],
         [-0.25, 0.25, 0.5, 0.5, -0.5], [0.75, -0.25, 0, 0.25, 0.5]]
    )
    head = _make_head(4, 2, 5, w_q, w_k, w_v)
    parameters = _parameters(head)
    expected_shapes = {"w_q": (4, 2), "w_k": (4, 2), "w_v": (4, 5)}
    assert set(dict(head.named_parameters())) == {
        "attention_head.w_q", "attention_head.w_k", "attention_head.w_v"
    }
    for name, parameter in parameters.items():
        assert tuple(parameter.shape) == expected_shapes[name]

    prefill_result = _prefill(head, x)
    assert isinstance(prefill_result, tuple) and len(prefill_result) == 3
    output, weights, cache = prefill_result
    assert isinstance(cache, tuple) and len(cache) == 2
    key_cache, value_cache = cache
    assert tuple(output.shape) == (3, 5)
    assert tuple(weights.shape) == (3, 3)
    assert tuple(key_cache.shape) == (3, 2)
    assert tuple(value_cache.shape) == (3, 5)
    _assert_device_dtype((output, weights, key_cache, value_cache), torch.float32)

    decode_result = _decode(head, x_new, cache)
    assert isinstance(decode_result, tuple) and len(decode_result) == 3
    decode_output, decode_weights, updated_cache = decode_result
    assert isinstance(updated_cache, tuple) and len(updated_cache) == 2
    assert tuple(decode_output.shape) == (1, 5)
    assert tuple(decode_weights.shape) == (1, 4)
    assert tuple(updated_cache[0].shape) == (4, 2)
    assert tuple(updated_cache[1].shape) == (4, 5)
    _assert_device_dtype(
        (decode_output, decode_weights, updated_cache[0], updated_cache[1]),
        torch.float32,
    )


def test_ca02_prefill_uses_only_visible_keys_and_normalizes_rows():
    """Kiểm tra causal visibility bằng attention có score bằng không."""
    x = _tensor([[2, 0], [0, 4], [6, 8]])
    w_q = _tensor([[0, 0, 0], [0, 0, 0]])
    w_k = _tensor([[0, 0, 0], [0, 0, 0]])
    w_v = _tensor([[1, 0], [0, 1]])
    head = _make_head(2, 3, 2, w_q, w_k, w_v)

    output, weights, _ = _prefill(head, x)
    expected_weights = _tensor([[1, 0, 0], [0.5, 0.5, 0], [1 / 3, 1 / 3, 1 / 3]])
    expected_output = _tensor([[2, 0], [1, 2], [8 / 3, 4]])
    _assert_close(weights, expected_weights)
    _assert_close(output, expected_output)
    assert torch.equal(weights[0, 1:], torch.zeros_like(weights[0, 1:]))
    assert torch.equal(weights[1, 2:], torch.zeros_like(weights[1, 2:]))
    _assert_close(weights.sum(dim=-1), _tensor([1, 1, 1]))


def test_ca03_prefill_returns_projected_cache_without_mutation():
    """Kiểm tra K/V cache chính xác và parameter được giữ nguyên."""
    x = _tensor([[1, 2, 0], [0, -1, 3]])
    w_q = _tensor([[0.5, 0.25], [-0.25, 0.5], [0.75, -0.5]])
    w_k = _tensor([[1, 0], [0, 2], [1, -1]])
    w_v = _tensor([[2], [-1], [3]])
    head = _make_head(3, 2, 1, w_q, w_k, w_v)
    x_before = x.clone()
    parameter_snapshot = _snapshot_parameters(head)

    _, _, (key_cache, value_cache) = _prefill(head, x)
    assert torch.equal(key_cache, _tensor([[1, 4], [3, -5]]))
    assert torch.equal(value_cache, _tensor([[0], [10]]))
    assert torch.equal(x, x_before)
    _assert_parameters_unchanged(head, parameter_snapshot)


def test_ca04_decode_appends_only_new_projection_and_preserves_inputs():
    """Kiểm tra append K/V đúng thứ tự và không sửa dữ liệu caller."""
    prefix = _tensor([[1, 0, 2], [-1, 2, 1]])
    x_new = _tensor([[2, -1, 3]])
    w_q = _tensor([[0.25, 0.5], [-0.5, 0.25], [0.75, -0.25]])
    w_k = _tensor([[1, 0], [0, 1], [1, -1]])
    w_v = _tensor([[1], [2], [-1]])
    head = _make_head(3, 2, 1, w_q, w_k, w_v)
    _, _, prefix_cache = _prefill(head, prefix)
    prefix_cache_before = tuple(tensor.clone() for tensor in prefix_cache)
    x_new_before = x_new.clone()
    parameter_snapshot = _snapshot_parameters(head)

    _, _, (key_cache, value_cache) = _decode(head, x_new, prefix_cache)
    assert torch.equal(key_cache, _tensor([[3, -2], [0, 1], [5, -4]]))
    assert torch.equal(value_cache, _tensor([[-1], [2], [-3]]))
    assert torch.equal(key_cache[:2], prefix_cache_before[0])
    assert torch.equal(value_cache[:2], prefix_cache_before[1])
    assert torch.equal(prefix_cache[0], prefix_cache_before[0])
    assert torch.equal(prefix_cache[1], prefix_cache_before[1])
    assert torch.equal(x_new, x_new_before)
    _assert_parameters_unchanged(head, parameter_snapshot)


def test_ca05_decode_projects_only_the_new_hidden_row():
    """Quan sát operand thật của projection và tách khỏi matrix attention."""
    prefix = _tensor(
        [[((3 * t + 2 * c) % 11 - 5) / 5 for c in range(5)] for t in range(4)]
    )
    x_new = _tensor([[(2 * c + 1) % 7 / 4 - 0.75 for c in range(5)]])
    w_q = _tensor([[(i + 2 * c + 1) / 13 for c in range(2)] for i in range(5)])
    w_k = _tensor([[(2 * i - c + 3) / 17 for c in range(2)] for i in range(5)])
    w_v = _tensor([[(3 * i + c + 2) / 19 for c in range(3)] for i in range(5)])
    head = _make_head(5, 2, 3, w_q, w_k, w_v)
    _, _, cache = _prefill(head, prefix)
    observer = _ProjectionOperandObserver(_parameters(head))

    with torch.no_grad(), observer:
        head.decode(x_new, cache)

    for name, operands in observer.projection_inputs.items():
        assert operands, f"Không quan sát được projection {name}"
        assert all(tuple(operand.shape) == (1, 5) for operand in operands)
        assert all(torch.equal(operand, x_new) for operand in operands)
    assert observer.other_matrix_shapes
    assert ((1, 2), (2, 5)) in observer.other_matrix_shapes
    assert ((1, 5), (5, 3)) in observer.other_matrix_shapes


def test_ca06_decode_attends_to_every_old_position_and_current_position():
    """Kiểm tra query mới thấy toàn bộ cache và chính vị trí hiện tại."""
    prefix = _tensor([[2, 0], [0, 4]])
    x_new = _tensor([[6, 8]])
    w_q = torch.zeros((2, 3), dtype=torch.float32, device=_TEST_DEVICE)
    w_k = torch.zeros((2, 3), dtype=torch.float32, device=_TEST_DEVICE)
    w_v = _tensor([[1, 0], [0, 1]])
    head = _make_head(2, 3, 2, w_q, w_k, w_v)
    _, _, cache = _prefill(head, prefix)

    output, weights, _ = _decode(head, x_new, cache)
    _assert_close(weights, _tensor([[1 / 3, 1 / 3, 1 / 3]]))
    _assert_close(output, _tensor([[8 / 3, 4]]))


def test_ca07_prefill_and_decode_match_unequal_score_reference():
    """So sánh cả hai đường đi với reference visible-prefix độc lập."""
    prefix = _tensor(
        [[1, 0.5, -0.5, 2], [-1, 1, 0.25, 0.5], [0.5, -0.75, 1, -1]]
    )
    x_new = _tensor([[0.25, 1.5, -0.5, 0.75]])
    w_q = _tensor(
        [[0.5, -0.25], [0.25, 0.5], [-0.5, 0.25], [0.25, 0.25]]
    )
    w_k = _tensor(
        [[0.25, 0.5], [-0.5, 0.25], [0.5, -0.25], [0.25, -0.5]]
    )
    w_v = _tensor(
        [[0.5, 0.25, -0.25], [-0.25, 0.5, 0.25], [0.25, -0.5, 0.5],
         [0.5, 0.25, 0.25]]
    )
    head = _make_head(4, 2, 3, w_q, w_k, w_v)
    expected_prefix = _reference(prefix, w_q, w_k, w_v)
    output, weights, cache = _prefill(head, prefix)
    _assert_close(output, expected_prefix[0])
    _assert_close(weights, expected_prefix[1])
    assert torch.all(weights >= 0)

    sequence = torch.cat((prefix, x_new), dim=0)
    expected_full = _reference(sequence, w_q, w_k, w_v)
    final_scores = expected_full[4][-1:] @ expected_full[2].transpose(0, 1)
    final_scores = final_scores / math.sqrt(w_q.shape[1])
    assert not torch.allclose(
        final_scores, torch.full_like(final_scores, final_scores[0, 0])
    )
    assert torch.all(expected_full[4][-1].abs() > 0)
    assert torch.all(
        (expected_full[4][-1:] * expected_full[2]).abs().sum(dim=0) > 0
    )

    decode_output, decode_weights, _ = _decode(head, x_new, cache)
    _assert_close(decode_output, expected_full[0][-1:])
    _assert_close(decode_weights, expected_full[1][-1:])
    assert torch.isfinite(decode_weights).all()
    assert torch.all(decode_weights >= 0)


def test_ca08_returned_cache_composes_across_two_decode_calls():
    """Kiểm tra cache trả về dùng được trực tiếp ở decode kế tiếp."""
    prefix = _tensor([[1, 0, 1], [0, 2, -1]])
    first_new = _tensor([[1, -1, 0.5]])
    second_new = _tensor([[-0.5, 1, 2]])
    w_q = _tensor([[0.5, 0.25], [-0.25, 0.5], [0.25, -0.5]])
    w_k = _tensor([[0.25, -0.5], [0.5, 0.25], [-0.25, 0.5]])
    w_v = _tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    head = _make_head(3, 2, 3, w_q, w_k, w_v)
    _, _, initial_cache = _prefill(head, prefix)

    _, _, first_cache = _decode(head, first_new, initial_cache)
    first_cache_before = tuple(tensor.clone() for tensor in first_cache)
    second_output, second_weights, final_cache = _decode(head, second_new, first_cache)
    sequence = torch.cat((prefix, first_new, second_new), dim=0)
    expected = _reference(sequence, w_q, w_k, w_v)
    assert tuple(final_cache[0].shape) == (4, 2)
    assert tuple(final_cache[1].shape) == (4, 3)
    assert torch.equal(first_cache[0], first_cache_before[0])
    assert torch.equal(first_cache[1], first_cache_before[1])
    assert torch.equal(final_cache[0][:3], first_cache_before[0])
    assert torch.equal(final_cache[1][:3], first_cache_before[1])
    _assert_close(final_cache[0], expected[2])
    _assert_close(final_cache[1], expected[3])
    _assert_close(second_output, expected[0][-1:])
    _assert_close(second_weights, expected[1][-1:])


def test_ca09_interleaved_prefixes_keep_explicit_caches_independent():
    """Kiểm tra hai prefix trên cùng head không trộn lịch sử."""
    prefix_a = _tensor([[1, 0], [0, 1]])
    prefix_b = _tensor([[2, -1]])
    x_new = _tensor([[1, 2]])
    w_q = _tensor([[0.5, 0.25], [-0.25, 0.5]])
    w_k = _tensor([[0.25, -0.5], [0.5, 0.25]])
    w_v = _tensor([[1], [2]])
    head = _make_head(2, 2, 1, w_q, w_k, w_v)
    _, _, cache_a = _prefill(head, prefix_a)
    _, _, cache_b = _prefill(head, prefix_b)
    cache_b_before = tuple(tensor.clone() for tensor in cache_b)

    output, weights, updated_a = _decode(head, x_new, cache_a)
    expected = _reference(torch.cat((prefix_a, x_new), dim=0), w_q, w_k, w_v)
    _assert_close(output, expected[0][-1:])
    _assert_close(weights, expected[1][-1:])
    assert tuple(updated_a[0].shape) == (3, 2)
    assert tuple(updated_a[1].shape) == (3, 1)
    assert torch.equal(cache_b[0], cache_b_before[0])
    assert torch.equal(cache_b[1], cache_b_before[1])


def test_ca10_float64_decode_preserves_dtype_device_and_reference_values():
    """Kiểm tra đường đi float64 với độ chính xác chặt."""
    dtype = torch.float64
    prefix = _tensor([[0.5, -1], [1.5, 0.25]], dtype=dtype)
    x_new = _tensor([[-0.25, 0.75]], dtype=dtype)
    w_q = _tensor([[0.5], [-0.25]], dtype=dtype)
    w_k = _tensor([[-0.25], [0.5]], dtype=dtype)
    w_v = _tensor([[1, 0.25], [-0.5, 1]], dtype=dtype)
    head = _make_head(2, 1, 2, w_q, w_k, w_v, dtype=dtype)
    prefill_output, prefill_weights, cache = _prefill(head, prefix)
    _assert_device_dtype(
        (prefill_output, prefill_weights, cache[0], cache[1]), dtype
    )
    output, weights, updated_cache = _decode(head, x_new, cache)
    expected = _reference(torch.cat((prefix, x_new), dim=0), w_q, w_k, w_v)

    _assert_device_dtype(
        (output, weights, updated_cache[0], updated_cache[1]), dtype
    )
    _assert_close(output, expected[0][-1:])
    _assert_close(weights, expected[1][-1:])
    _assert_close(updated_cache[0], expected[2])
    _assert_close(updated_cache[1], expected[3])
