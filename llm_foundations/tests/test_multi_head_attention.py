"""Kiểm tra đăng ký multi-head, giá trị causal, trạng thái tham số và autograd."""

import math

import pytest
import torch
from torch import nn

from llm_foundations.multi_head_attention import MultiHeadCausalSelfAttention
from llm_foundations.self_attention import CausalSelfAttention

_TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_DTYPES = (torch.float32, torch.float64)
_TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float64: {"rtol": 1e-9, "atol": 1e-11},
}


def _tensor(values, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _main_fixture(dtype: torch.dtype, device: torch.device = _TEST_DEVICE):
    x = _tensor(
        [[1, 0, -1, 2, 1], [0, 2, 1, -1, 1], [-1, 1, 2, 0, -2], [2, -1, 0, 1, 0]],
        dtype,
        device,
    )
    head_weights = [
        (
            _tensor(
                [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.1], [0.1, 0.2], [-0.1, 0.3]],
                dtype,
                device,
            ),
            _tensor(
                [[-0.1, 0.3], [0.2, -0.2], [0.4, 0.1], [-0.3, 0.2], [0.1, -0.1]],
                dtype,
                device,
            ),
            _tensor(
                [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2], [-0.3, 0.2, 0.1], [0.4, 0.1, -0.1], [-0.2, -0.3, 0.2]],
                dtype,
                device,
            ),
        ),
        (
            _tensor(
                [[-0.3, 0.2], [0.1, -0.4], [0.2, 0.3], [-0.1, 0.1], [0.4, -0.2]],
                dtype,
                device,
            ),
            _tensor(
                [[0.2, 0.1], [-0.3, 0.2], [0.1, -0.4], [0.3, -0.1], [-0.2, 0.3]],
                dtype,
                device,
            ),
            _tensor(
                [[-0.1, 0.3, 0.2], [0.2, -0.2, 0.4], [0.3, 0.1, -0.3], [-0.2, 0.4, 0.1], [0.1, -0.1, -0.2]],
                dtype,
                device,
            ),
        ),
    ]
    w_o = _tensor(
        [
            [0.2, -0.1, 0.3, 0.1, -0.2],
            [-0.3, 0.4, 0.1, -0.2, 0.3],
            [0.1, 0.2, -0.4, 0.3, 0.1],
            [0.4, -0.2, 0.2, -0.1, 0.3],
            [-0.2, 0.1, 0.3, 0.4, -0.1],
            [0.3, -0.3, 0.1, 0.2, 0.4],
        ],
        dtype,
        device,
    )
    return x, head_weights, w_o


def _make_module(dtype: torch.dtype, d_model: int = 5, num_heads: int = 2, d_k: int = 2, d_v: int = 3):
    return MultiHeadCausalSelfAttention(d_model, num_heads, d_k, d_v).to(
        device=_TEST_DEVICE,
        dtype=dtype,
    )


def _load_fixture(module, head_weights, w_o: torch.Tensor) -> None:
    with torch.no_grad():
        for head, (w_q, w_k, w_v) in zip(module.heads, head_weights):
            head.w_q.copy_(w_q)
            head.w_k.copy_(w_k)
            head.w_v.copy_(w_v)
        module.w_o.copy_(w_o)


def _named_parameters(module) -> dict[str, nn.Parameter]:
    return dict(module.named_parameters())


def _snapshot_parameters(module):
    return {
        name: {
            "identity": id(parameter),
            "storage": parameter.untyped_storage().data_ptr(),
            "shape": tuple(parameter.shape),
            "dtype": parameter.dtype,
            "device": parameter.device,
            "value": parameter.detach().clone(),
        }
        for name, parameter in module.named_parameters()
    }


def _assert_parameter_snapshot(module, snapshot) -> None:
    current = dict(module.named_parameters())
    assert list(current) == list(snapshot)
    for name, parameter in current.items():
        previous = snapshot[name]
        assert id(parameter) == previous["identity"]
        assert parameter.untyped_storage().data_ptr() == previous["storage"]
        assert tuple(parameter.shape) == previous["shape"]
        assert parameter.dtype == previous["dtype"]
        assert parameter.device == previous["device"]
        assert torch.equal(parameter, previous["value"])


def _head_reference(x: torch.Tensor, w_q: torch.Tensor, w_k: torch.Tensor, w_v: torch.Tensor):
    """Tính attention nhân quả từ các tiền tố nhìn thấy, không dùng helper tạo mask của module."""
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v
    seq_len = x.shape[0]
    scale = math.sqrt(w_q.shape[1])
    output_rows = []
    weight_rows = []
    for query_index in range(seq_len):
        visible_scores = (q[query_index : query_index + 1] @ k[: query_index + 1].T).squeeze(0)
        probabilities = torch.softmax(visible_scores / scale, dim=0)
        future_zeros = torch.zeros(seq_len - query_index - 1, dtype=x.dtype, device=x.device)
        weight_rows.append(torch.cat((probabilities, future_zeros)))
        output_rows.append(probabilities @ v[: query_index + 1])
    return torch.stack(output_rows), torch.stack(weight_rows)


def _indexed_projection(head_outputs, w_o: torch.Tensor, block_order=None) -> torch.Tensor:
    """Chiếu đặc trưng các head theo định nghĩa chỉ số toán học tường minh."""
    num_heads = len(head_outputs)
    d_v = head_outputs[0].shape[1]
    seq_len = head_outputs[0].shape[0]
    d_model = w_o.shape[1]
    if block_order is None:
        block_order = tuple(range(num_heads))
    output_rows = []
    for token_index in range(seq_len):
        output_features = []
        for output_feature in range(d_model):
            terms = [
                head_outputs[block_order[head_index]][token_index, value_feature]
                * w_o[head_index * d_v + value_feature, output_feature]
                for head_index in range(num_heads)
                for value_feature in range(d_v)
            ]
            output_features.append(torch.stack(terms).sum())
        output_rows.append(torch.stack(output_features))
    return torch.stack(output_rows)


def _multi_head_reference(x: torch.Tensor, head_weights, w_o: torch.Tensor):
    per_head = [_head_reference(x, *weights) for weights in head_weights]
    head_outputs = [result[0] for result in per_head]
    expected_output = _indexed_projection(head_outputs, w_o)
    expected_weights = torch.stack([result[1] for result in per_head], dim=0)
    return expected_output, expected_weights, head_outputs


def _run(module, x: torch.Tensor):
    result = module(x)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(value, torch.Tensor) for value in result)
    return result


def _assert_matches_reference(module, x, head_weights, w_o):
    output, weights = _run(module, x)
    expected_output, expected_weights, _ = _multi_head_reference(x, head_weights, w_o)
    num_heads = len(head_weights)
    assert output.shape == (x.shape[0], w_o.shape[1])
    assert weights.shape == (num_heads, x.shape[0], x.shape[0])
    assert output.dtype == x.dtype and weights.dtype == x.dtype
    assert output.device == x.device and weights.device == x.device
    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    _assert_close(weights.sum(dim=-1), torch.ones((num_heads, x.shape[0]), dtype=x.dtype, device=x.device))
    positions = torch.arange(x.shape[0], device=x.device)
    future = positions.unsqueeze(0) > positions.unsqueeze(1)
    assert torch.equal(weights[:, future], torch.zeros_like(weights[:, future]))
    _assert_close(output, expected_output)
    _assert_close(weights, expected_weights)
    return output, weights


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma1_registers_independent_heads_and_output_projection(dtype: torch.dtype):
    module = MultiHeadCausalSelfAttention(5, 2, 2, 3)
    assert isinstance(module, nn.Module)
    assert isinstance(module.heads, nn.ModuleList)
    assert len(module.heads) == 2
    assert all(isinstance(head, CausalSelfAttention) for head in module.heads)
    assert id(module.heads[0]) != id(module.heads[1])

    parameters = _named_parameters(module)
    expected_names = {
        "heads.0.w_q",
        "heads.0.w_k",
        "heads.0.w_v",
        "heads.1.w_q",
        "heads.1.w_k",
        "heads.1.w_v",
        "w_o",
    }
    assert set(parameters) == expected_names
    assert len(parameters) == 3 * 2 + 1
    assert len({id(parameter) for parameter in parameters.values()}) == len(parameters)
    assert len({parameter.untyped_storage().data_ptr() for parameter in parameters.values()}) == len(parameters)
    expected_shapes = {
        "heads.0.w_q": (5, 2),
        "heads.0.w_k": (5, 2),
        "heads.0.w_v": (5, 3),
        "heads.1.w_q": (5, 2),
        "heads.1.w_k": (5, 2),
        "heads.1.w_v": (5, 3),
        "w_o": (6, 5),
    }
    for name, shape in expected_shapes.items():
        assert isinstance(parameters[name], nn.Parameter)
        assert parameters[name].requires_grad
        assert tuple(parameters[name].shape) == shape
    assert sum(parameter.numel() for parameter in parameters.values()) == 100

    module.to(device=_TEST_DEVICE, dtype=dtype)
    for parameter in module.parameters():
        assert parameter.device == _TEST_DEVICE
        assert parameter.dtype == dtype
    assert module.state_dict().keys() == parameters.keys()

    head1_before = {name: parameter.detach().clone() for name, parameter in parameters.items() if name.startswith("heads.1.")}
    w_o_before = parameters["w_o"].detach().clone()
    with torch.no_grad():
        module.heads[0].w_v[0, 0].add_(1)
    for name, parameter in _named_parameters(module).items():
        if name.startswith("heads.1."):
            assert torch.equal(parameter, head1_before[name])
    assert torch.equal(module.w_o, w_o_before)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma2_rectangular_fixture_matches_independent_values_and_head_order(dtype: torch.dtype):
    x, head_weights, w_o = _main_fixture(dtype)
    module = _make_module(dtype)
    _load_fixture(module, head_weights, w_o)
    output, weights = _assert_matches_reference(module, x, head_weights, w_o)
    assert output.shape == (4, 5)
    assert weights.shape == (2, 4, 4)
    _assert_close(weights[0], _multi_head_reference(x, head_weights, w_o)[1][0])
    _assert_close(weights[1], _multi_head_reference(x, head_weights, w_o)[1][1])

    expected_output, expected_weights, head_outputs = _multi_head_reference(x, head_weights, w_o)
    reversed_output = _indexed_projection(head_outputs, w_o, block_order=(1, 0))
    assert not torch.allclose(expected_output, reversed_output, **_TOLERANCES[dtype])
    assert not torch.allclose(expected_weights[0, 1:], expected_weights[1, 1:], **_TOLERANCES[dtype])

    for head_index, (w_q, w_k, w_v) in enumerate(head_weights):
        q, k, v = x @ w_q, x @ w_k, x @ w_v
        for feature in range(q.shape[1]):
            score_part = q[:, feature : feature + 1] @ k[:, feature : feature + 1].T
            assert torch.count_nonzero(score_part).item() > 0
        assert all(torch.count_nonzero(v[:, feature]).item() > 0 for feature in range(v.shape[1]))
        assert head_index in (0, 1)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma3_each_head_is_causal_and_future_rows_preserve_prefix(dtype: torch.dtype):
    x, head_weights, w_o = _main_fixture(dtype)
    module = _make_module(dtype)
    _load_fixture(module, head_weights, w_o)
    output, weights = _assert_matches_reference(module, x, head_weights, w_o)

    changed_x = x.clone()
    changed_x[2] += _tensor([3, -2, 1, 4, -3], dtype, x.device)
    changed_x[3] += _tensor([-1, 2, 5, -4, 3], dtype, x.device)
    changed_output, changed_weights = _assert_matches_reference(module, changed_x, head_weights, w_o)
    _assert_close(output[:2], changed_output[:2])
    _assert_close(weights[:, :2, :], changed_weights[:, :2, :])
    assert torch.equal(weights[:, :2, 2:], torch.zeros_like(weights[:, :2, 2:]))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma4_repeated_calls_and_new_sequence_length_preserve_parameters(dtype: torch.dtype):
    x, head_weights, w_o = _main_fixture(dtype)
    module = _make_module(dtype)
    _load_fixture(module, head_weights, w_o)
    snapshot = _snapshot_parameters(module)
    x_seven = _tensor(
        [[((3 * t + 2 * c) % 13 - 6) / 5 for c in range(5)] for t in range(7)],
        dtype,
        _TEST_DEVICE,
    )

    first = _assert_matches_reference(module, x, head_weights, w_o)
    _assert_parameter_snapshot(module, snapshot)
    repeated = _assert_matches_reference(module, x, head_weights, w_o)
    _assert_close(first[0], repeated[0])
    _assert_close(first[1], repeated[1])
    _assert_parameter_snapshot(module, snapshot)
    longer = _assert_matches_reference(module, x_seven, head_weights, w_o)
    assert longer[0].shape == (7, 5)
    assert longer[1].shape == (2, 7, 7)
    assert sum(parameter.numel() for parameter in module.parameters()) == 100
    _assert_parameter_snapshot(module, snapshot)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma5_current_input_and_parameters_are_used(dtype: torch.dtype):
    x, base_head_weights, base_w_o = _main_fixture(dtype)
    module = _make_module(dtype)
    _load_fixture(module, base_head_weights, base_w_o)
    base_output, base_weights, _ = _multi_head_reference(x, base_head_weights, base_w_o)

    changed_x = x.clone()
    changed_x[2, 1] += 0.8
    changed_input_reference = _multi_head_reference(changed_x, base_head_weights, base_w_o)
    assert not torch.allclose(base_output, changed_input_reference[0], **_TOLERANCES[dtype])
    assert not torch.allclose(base_weights, changed_input_reference[1], **_TOLERANCES[dtype])
    changed_input_output, changed_input_weights = _assert_matches_reference(
        module, changed_x, base_head_weights, base_w_o
    )
    assert not torch.allclose(changed_input_output, base_output, **_TOLERANCES[dtype])
    assert not torch.allclose(changed_input_weights, base_weights, **_TOLERANCES[dtype])

    # Khôi phục baseline trước mỗi lần can thiệp vào tham số
    _load_fixture(module, base_head_weights, base_w_o)
    baseline_output, baseline_weights = _assert_matches_reference(module, x, base_head_weights, base_w_o)
    baseline_snapshot = _snapshot_parameters(module)

    changed_w_o = base_w_o.clone()
    changed_w_o[0, 0] += 0.6
    changed_w_o_reference = _multi_head_reference(x, base_head_weights, changed_w_o)
    assert torch.count_nonzero(_multi_head_reference(x, base_head_weights, base_w_o)[2][0][:, 0]).item() > 0
    assert not torch.allclose(baseline_output, changed_w_o_reference[0], **_TOLERANCES[dtype])
    with torch.no_grad():
        module.w_o[0, 0].add_(0.6)
    changed_w_o_output, changed_w_o_weights = _assert_matches_reference(
        module, x, base_head_weights, changed_w_o
    )
    assert not torch.allclose(changed_w_o_output, baseline_output, **_TOLERANCES[dtype])
    _assert_close(changed_w_o_weights, baseline_weights)
    for name, parameter in module.named_parameters():
        if name != "w_o":
            previous = baseline_snapshot[name]
            assert id(parameter) == previous["identity"]
            assert torch.equal(parameter, previous["value"])

    for parameter_name, index, delta in (
        ("w_v", (0, 0), 0.7),
        ("w_q", (2, 1), 0.4),
        ("w_k", (3, 1), 0.5),
    ):
        _load_fixture(module, base_head_weights, base_w_o)
        baseline_output, baseline_weights = _assert_matches_reference(module, x, base_head_weights, base_w_o)
        modified_heads = [(w_q.clone(), w_k.clone(), w_v.clone()) for w_q, w_k, w_v in base_head_weights]
        target_matrix = {"w_q": 0, "w_k": 1, "w_v": 2}[parameter_name]
        modified_heads[0][target_matrix][index] += delta
        modified_reference = _multi_head_reference(x, modified_heads, base_w_o)
        if parameter_name in {"w_q", "w_k"}:
            assert not torch.allclose(
                modified_reference[1][0], baseline_weights[0], **_TOLERANCES[dtype]
            )
            _assert_close(modified_reference[1][1], baseline_weights[1])
        else:
            _assert_close(modified_reference[1], baseline_weights)
            assert not torch.allclose(
                modified_reference[0], baseline_output, **_TOLERANCES[dtype]
            )

        parameter = getattr(module.heads[0], parameter_name)
        with torch.no_grad():
            parameter[index].add_(delta)
        actual_output, actual_weights = _assert_matches_reference(module, x, modified_heads, base_w_o)
        if parameter_name in {"w_q", "w_k"}:
            assert not torch.allclose(actual_weights[0], baseline_weights[0], **_TOLERANCES[dtype])
            _assert_close(actual_weights[1], baseline_weights[1])
        else:
            _assert_close(actual_weights, baseline_weights)
            assert not torch.allclose(actual_output, baseline_output, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma6_single_head_and_singleton_dimensions(dtype: torch.dtype):
    x, head_weights, _ = _main_fixture(dtype)
    one_head_w_o = _tensor(
        [[0.2, -0.1, 0.3, 0.1, -0.2], [-0.3, 0.4, 0.1, -0.2, 0.3], [0.1, 0.2, -0.4, 0.3, 0.1]],
        dtype,
        x.device,
    )
    one_head = _make_module(dtype, d_model=5, num_heads=1, d_k=2, d_v=3)
    _load_fixture(one_head, head_weights[:1], one_head_w_o)
    one_output, one_weights = _assert_matches_reference(one_head, x, head_weights[:1], one_head_w_o)
    assert one_output.shape == (4, 5)
    assert one_weights.shape == (1, 4, 4)

    x_single = _tensor([[1, 2, -1]], dtype, _TEST_DEVICE)
    w_q = torch.zeros((3, 2), dtype=dtype, device=_TEST_DEVICE)
    w_k = torch.ones((3, 2), dtype=dtype, device=_TEST_DEVICE)
    w_v0 = _tensor([[1, 0], [0, 2], [1, -1]], dtype, _TEST_DEVICE)
    w_v1 = _tensor([[-1, 1], [2, 0], [0, 1]], dtype, _TEST_DEVICE)
    w_o_single = _tensor(
        [[1, 0, -1], [2, 1, 0], [-1, 3, 2], [0, -2, 1]], dtype, _TEST_DEVICE
    )
    singleton_heads = [(w_q, w_k, w_v0), (w_q.clone(), w_k.clone(), w_v1)]
    singleton = _make_module(dtype, d_model=3, num_heads=2, d_k=2, d_v=2)
    _load_fixture(singleton, singleton_heads, w_o_single)
    singleton_output, singleton_weights = _assert_matches_reference(
        singleton, x_single, singleton_heads, w_o_single
    )
    assert singleton_output.shape == (1, 3)
    assert singleton_weights.shape == (2, 1, 1)
    _assert_close(singleton_weights, torch.ones((2, 1, 1), dtype=dtype, device=_TEST_DEVICE))
    _assert_close(singleton_output, _tensor([[7, 14, 6]], dtype, _TEST_DEVICE))

    x_unit = _tensor([[2]], dtype, _TEST_DEVICE)
    w_q_unit = _tensor([[0.5]], dtype, _TEST_DEVICE)
    w_k_unit = _tensor([[-1]], dtype, _TEST_DEVICE)
    w_v_unit = _tensor([[3]], dtype, _TEST_DEVICE)
    w_o_unit = _tensor([[2]], dtype, _TEST_DEVICE)
    unit_heads = [(w_q_unit, w_k_unit, w_v_unit)]
    unit = _make_module(dtype, d_model=1, num_heads=1, d_k=1, d_v=1)
    _load_fixture(unit, unit_heads, w_o_unit)
    unit_output, unit_weights = _assert_matches_reference(unit, x_unit, unit_heads, w_o_unit)
    assert unit_output.shape == (1, 1)
    assert unit_weights.shape == (1, 1, 1)
    _assert_close(unit_output, _tensor([[12]], dtype, _TEST_DEVICE))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma7_zero_query_and_zero_output_projection(dtype: torch.dtype):
    x, base_head_weights, w_o = _main_fixture(dtype)
    zero_query_heads = [(torch.zeros_like(w_q), w_k, w_v) for w_q, w_k, w_v in base_head_weights]
    module = _make_module(dtype)
    _load_fixture(module, zero_query_heads, w_o)
    output, weights = _assert_matches_reference(module, x, zero_query_heads, w_o)
    for query_index in range(x.shape[0]):
        expected_prefix = torch.full(
            (query_index + 1,),
            1 / (query_index + 1),
            dtype=dtype,
            device=x.device,
        )
        for head_index, (_, _, w_v) in enumerate(zero_query_heads):
            _assert_close(weights[head_index, query_index, : query_index + 1], expected_prefix)
            expected_value = expected_prefix @ (x[: query_index + 1] @ w_v)
            actual_head_output = _head_reference(x, *zero_query_heads[head_index])[0][query_index]
            _assert_close(actual_head_output, expected_value)

    zero_w_o = torch.zeros_like(w_o)
    zero_projection_module = _make_module(dtype)
    _load_fixture(zero_projection_module, base_head_weights, zero_w_o)
    zero_output, zero_weights = _assert_matches_reference(
        zero_projection_module, x, base_head_weights, zero_w_o
    )
    assert torch.equal(zero_output, torch.zeros_like(zero_output))
    _assert_close(zero_weights, _multi_head_reference(x, base_head_weights, w_o)[1])


@pytest.mark.parametrize("dtype", _DTYPES)
def test_ma8_strided_input_preserves_backing_storage_and_parameters(dtype: torch.dtype):
    x, head_weights, w_o = _main_fixture(dtype)
    module = _make_module(dtype)
    _load_fixture(module, head_weights, w_o)
    backing = torch.zeros((4, 10), dtype=dtype, device=_TEST_DEVICE)
    strided_x = backing[:, ::2]
    strided_x.copy_(x)
    assert strided_x.shape == (4, 5)
    assert strided_x.stride(1) == 2
    backing_before = backing.clone()
    input_before = strided_x.clone()
    parameter_snapshot = _snapshot_parameters(module)

    _assert_matches_reference(module, strided_x, head_weights, w_o)
    assert torch.equal(strided_x, input_before)
    assert torch.equal(backing, backing_before)
    _assert_parameter_snapshot(module, parameter_snapshot)


@pytest.mark.parametrize("dtype", [torch.float64])
def test_ma9_autograd_matches_independent_reference_for_input_and_all_parameters(dtype: torch.dtype):
    x, fixture_heads, fixture_w_o = _main_fixture(dtype)
    x = x.detach().clone().requires_grad_(True)
    module = _make_module(dtype)
    _load_fixture(module, fixture_heads, fixture_w_o)
    module.zero_grad(set_to_none=True)

    objective_weights = _tensor(
        [[0.2, -0.4, 0.1, 0.3, -0.2], [-0.1, 0.3, -0.2, 0.4, 0.1], [0.4, 0.1, -0.3, -0.2, 0.2], [-0.2, 0.2, 0.4, 0.1, -0.3]],
        dtype,
        x.device,
    )
    output, _ = _run(module, x)
    (output * objective_weights).sum().backward()

    candidate_parameters = _named_parameters(module)
    candidate_gradients = {name: parameter.grad for name, parameter in candidate_parameters.items()}
    assert x.grad is not None and x.grad.shape == x.shape and torch.isfinite(x.grad).all()
    for name, parameter in candidate_parameters.items():
        assert parameter.grad is not None
        assert parameter.grad.shape == parameter.shape
        assert torch.isfinite(parameter.grad).all()

    ref_x = x.detach().clone().requires_grad_(True)
    ref_heads = [
        tuple(value.detach().clone().requires_grad_(True) for value in weights)
        for weights in fixture_heads
    ]
    ref_w_o = fixture_w_o.detach().clone().requires_grad_(True)
    reference_output, _, _ = _multi_head_reference(ref_x, ref_heads, ref_w_o)
    reference_loss = (reference_output * objective_weights).sum()
    reference_parameters = {
        f"heads.{head_index}.{name}": parameter
        for head_index, (w_q, w_k, w_v) in enumerate(ref_heads)
        for name, parameter in (("w_q", w_q), ("w_k", w_k), ("w_v", w_v))
    }
    reference_parameters["w_o"] = ref_w_o
    reference_loss.backward()

    _assert_close(x.grad, ref_x.grad)
    for name, parameter in candidate_parameters.items():
        _assert_close(candidate_gradients[name], reference_parameters[name].grad)
        assert torch.linalg.vector_norm(reference_parameters[name].grad).item() > 0

    per_head_input_gradients = []
    for head_index, (w_q, w_k, w_v) in enumerate(fixture_heads):
        isolated_x = x.detach().clone().requires_grad_(True)
        isolated_parameters = tuple(value.detach().clone().requires_grad_(True) for value in (w_q, w_k, w_v))
        head_output, _ = _head_reference(isolated_x, *isolated_parameters)
        start = head_index * head_output.shape[1]
        contribution = head_output @ fixture_w_o[start : start + head_output.shape[1]]
        (contribution * objective_weights).sum().backward()
        per_head_input_gradients.append(isolated_x.grad)
        assert torch.linalg.vector_norm(isolated_x.grad).item() > 0
    summed_head_gradient = torch.stack(per_head_input_gradients).sum(dim=0)
    _assert_close(ref_x.grad, summed_head_gradient)
