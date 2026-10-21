"""Kiểm tra composition và luồng KV cache của decoder block pre-norm."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from llm_foundations.decoder_block import PreNormDecoderBlock
from llm_foundations.ffn import PositionWiseFFN
from llm_foundations.kv_cache.decoder_block import CachedPreNormDecoderBlock
from llm_foundations.kv_cache.multi_head_attention import CachedMultiHeadAttention
from llm_foundations.layer_norm import LayerNorm
from llm_foundations.multi_head_attention import MultiHeadCausalSelfAttention


TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected test device: {TEST_DEVICE}")
_DIMS = {"d_model": 5, "num_heads": 2, "d_k": 3, "d_v": 2, "d_ff": 7}
_FLOW_DIMS = {"d_model": 3, "num_heads": 2, "d_k": 2, "d_v": 1, "d_ff": 5}
_EPS = 3e-4
_TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-9, "atol": 1e-10},
}
_MAIN_X = [
    [0.2, -0.7, 1.1, 0.4, -0.3],
    [1.2, 0.1, -0.4, 0.8, -0.9],
    [-0.5, 0.6, 0.3, -1.0, 0.7],
    [0.9, -0.2, -0.8, 0.5, 0.1],
]
_GAMMA1 = [0.8, 1.1, 0.6, 1.3, 0.9]
_BETA1 = [0.10, -0.15, 0.05, 0.20, -0.10]
_GAMMA2 = [1.2, 0.7, 1.4, 0.9, 0.5]
_BETA2 = [-0.20, 0.12, 0.08, -0.05, 0.18]


@pytest.fixture(autouse=True)
def _disable_reduced_precision():
    allow_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    allow_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type=TEST_DEVICE.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(matmul_precision)
        torch.backends.cuda.matmul.allow_tf32 = allow_matmul_tf32
        torch.backends.cudnn.allow_tf32 = allow_cudnn_tf32


def _dimensions(**overrides):
    return {**_DIMS, **overrides}


def _tensor(values, dtype=torch.float32):
    return torch.tensor(values, dtype=dtype, device=TEST_DEVICE)


def _new_block(dtype=torch.float32, eps=_EPS, **dimensions):
    return CachedPreNormDecoderBlock(**_dimensions(**dimensions), eps=eps).to(
        device=TEST_DEVICE,
        dtype=dtype,
    )


def _assert_close(actual, expected):
    assert isinstance(actual, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == TEST_DEVICE
    assert expected.device == TEST_DEVICE
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(expected).all().item()
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _diagnostic_cache(length, dimensions, dtype, offset=0):
    result = []
    for head_index in range(dimensions["num_heads"]):
        key = torch.arange(length * dimensions["d_k"], dtype=dtype, device=TEST_DEVICE).reshape(
            length, dimensions["d_k"]
        )
        value = torch.arange(length * dimensions["d_v"], dtype=dtype, device=TEST_DEVICE).reshape(
            length, dimensions["d_v"]
        )
        result.append(
            (
                key + offset + 20 * head_index + 1,
                value + offset + 30 * head_index + 11,
            )
        )
    return result


def _assert_cache_close(actual, expected):
    assert isinstance(actual, list)
    assert isinstance(expected, list)
    assert len(actual) == len(expected)
    for actual_pair, expected_pair in zip(actual, expected):
        assert isinstance(actual_pair, tuple) and len(actual_pair) == 2
        assert isinstance(expected_pair, tuple) and len(expected_pair) == 2
        for actual_tensor, expected_tensor in zip(actual_pair, expected_pair):
            _assert_close(actual_tensor, expected_tensor)


def _assert_cache_shapes(caches, dimensions, length, dtype):
    assert isinstance(caches, list)
    assert len(caches) == dimensions["num_heads"]
    for pair in caches:
        assert isinstance(pair, tuple) and len(pair) == 2
        key, value = pair
        assert key.shape == (length, dimensions["d_k"])
        assert value.shape == (length, dimensions["d_v"])
        for tensor in (key, value):
            assert tensor.dtype == dtype
            assert tensor.device == TEST_DEVICE


def _layer_norm_reference(x, norm):
    return F.layer_norm(
        x,
        (x.shape[-1],),
        norm.gamma,
        norm.beta,
        norm.eps,
    )


class AttentionDispatchSpy(nn.Module):
    """Ghi nhận dispatch cached-attention và trả payload kiểm soát được."""

    def __init__(self, prefill_result=None, decode_result=None):
        super().__init__()
        self.prefill_result = prefill_result
        self.decode_result = decode_result
        self.prefill_inputs = []
        self.decode_inputs = []
        self.decode_caches = []

    def prefill(self, x):
        self.prefill_inputs.append(x.detach().clone())
        return self.prefill_result

    def decode(self, x_new, caches):
        self.decode_inputs.append(x_new.detach().clone())
        self.decode_caches.append(
            [(key.detach().clone(), value.detach().clone()) for key, value in caches]
        )
        return self.decode_result


def _flow_input(path, dtype):
    if path == "prefill":
        return _tensor([[0.2, -0.5, 1.0], [-0.7, 0.9, 0.1]], dtype)
    return _tensor([[1.1, -0.3, 0.4]], dtype)


def _flow_update(path, dtype):
    values = [[0.31, -0.21, 0.17], [-0.14, 0.27, 0.38]]
    return _tensor(values[:2] if path == "prefill" else values[:1], dtype)


def _prefill_diagnostic_weights(dtype):
    return _tensor(
        [
            [[1.0, 0.0], [0.25, 0.75]],
            [[1.0, 0.0], [0.70, 0.30]],
        ],
        dtype,
    )


def _decode_diagnostic_weights(dtype):
    return _tensor([[[0.10, 0.25, 0.65]], [[0.55, 0.30, 0.15]]], dtype)


def _snapshot_cache(caches):
    return [(key.detach().clone(), value.detach().clone()) for key, value in caches]


def _set_norm_values(block):
    d_model = block.normalize_input.gamma.numel()
    with torch.no_grad():
        block.normalize_input.gamma.copy_(_tensor(_GAMMA1[:d_model], block.normalize_input.gamma.dtype))
        block.normalize_input.beta.copy_(_tensor(_BETA1[:d_model], block.normalize_input.beta.dtype))
        block.ffn_norm.gamma.copy_(_tensor(_GAMMA2[:d_model], block.ffn_norm.gamma.dtype))
        block.ffn_norm.beta.copy_(_tensor(_BETA2[:d_model], block.ffn_norm.beta.dtype))


def _flow_spy(path, x, update):
    dimensions = _dimensions(**_FLOW_DIMS)
    if path == "prefill":
        weights = torch.full(
            (dimensions["num_heads"], x.shape[0], x.shape[0]),
            0.25,
            dtype=x.dtype,
            device=TEST_DEVICE,
        )
        caches = _diagnostic_cache(x.shape[0], dimensions, x.dtype, offset=7)
        return AttentionDispatchSpy((update, weights, caches), None), None
    incoming = _diagnostic_cache(2, dimensions, x.dtype, offset=3)
    weights = torch.full(
        (dimensions["num_heads"], 1, 3),
        0.2,
        dtype=x.dtype,
        device=TEST_DEVICE,
    )
    caches = _diagnostic_cache(3, dimensions, x.dtype, offset=13)
    return AttentionDispatchSpy(None, (update, weights, caches)), incoming


def _matrix_fixture(parameter, role):
    indices = torch.arange(parameter.numel(), device=TEST_DEVICE)
    values = 0.07 * (((3 * indices + role) % 11) - 5).to(parameter.dtype) + 0.01 * role
    return values.reshape_as(parameter)


def _bias_fixture(parameter, role):
    indices = torch.arange(parameter.numel(), device=TEST_DEVICE)
    return 0.03 * (((indices + role) % 5) - 2).to(parameter.dtype)


def _set_numerical_fixture(block):
    _set_norm_values(block)
    with torch.no_grad():
        role = 1
        for cached_head in block.cached_mha.cached_heads:
            for parameter in (
                cached_head.attention_head.w_q,
                cached_head.attention_head.w_k,
                cached_head.attention_head.w_v,
            ):
                parameter.copy_(_matrix_fixture(parameter, role))
                role += 1
        for parameter in (block.cached_mha.w_o, block.ffn.w_1, block.ffn.w_2):
            parameter.copy_(_matrix_fixture(parameter, role))
            role += 1
        for parameter in (block.ffn.b_1, block.ffn.b_2):
            parameter.copy_(_bias_fixture(parameter, role))
            role += 1


def _semantic_parameter_pairs(cached, uncached):
    pairs = [
        ("ln1.gamma", cached.normalize_input.gamma, uncached.attention_residual.normalized_input.gamma),
        ("ln1.beta", cached.normalize_input.beta, uncached.attention_residual.normalized_input.beta),
        ("ln2.gamma", cached.ffn_norm.gamma, uncached.ffn_norm.gamma),
        ("ln2.beta", cached.ffn_norm.beta, uncached.ffn_norm.beta),
        ("attention.w_o", cached.cached_mha.w_o, uncached.attention_residual.multi_head_attention.w_o),
        ("ffn.w_1", cached.ffn.w_1, uncached.ffn.w_1),
        ("ffn.b_1", cached.ffn.b_1, uncached.ffn.b_1),
        ("ffn.w_2", cached.ffn.w_2, uncached.ffn.w_2),
        ("ffn.b_2", cached.ffn.b_2, uncached.ffn.b_2),
    ]
    for head_index, (cached_head, uncached_head) in enumerate(
        zip(cached.cached_mha.cached_heads, uncached.attention_residual.multi_head_attention.heads)
    ):
        for projection in ("w_q", "w_k", "w_v"):
            pairs.append(
                (
                    f"head.{head_index}.{projection}",
                    getattr(cached_head.attention_head, projection),
                    getattr(uncached_head, projection),
                )
            )
    return pairs


def _copy_semantic_parameters(cached, uncached):
    pairs = _semantic_parameter_pairs(cached, uncached)
    expected_roles = {
        "ln1.gamma",
        "ln1.beta",
        "ln2.gamma",
        "ln2.beta",
        "attention.w_o",
        "ffn.w_1",
        "ffn.b_1",
        "ffn.w_2",
        "ffn.b_2",
        *(f"head.{head}.{projection}" for head in range(_DIMS["num_heads"]) for projection in ("w_q", "w_k", "w_v")),
    }
    assert {role for role, _, _ in pairs} == expected_roles
    assert cached.normalize_input.eps == uncached.attention_residual.normalized_input.eps
    assert cached.ffn_norm.eps == uncached.ffn_norm.eps
    with torch.no_grad():
        for _, source, destination in pairs:
            assert source is not destination
            destination.copy_(source.detach().clone())
            assert torch.equal(source, destination)
    return pairs


def _aligned_blocks(dtype):
    cached = _new_block(dtype=dtype, eps=_EPS).eval()
    uncached = PreNormDecoderBlock(**_DIMS, eps=_EPS).to(
        device=TEST_DEVICE,
        dtype=dtype,
    ).eval()
    _set_numerical_fixture(cached)
    _copy_semantic_parameters(cached, uncached)
    return cached, uncached


def _projected_kv(block, normalized):
    return [
        (
            normalized @ cached_head.attention_head.w_k,
            normalized @ cached_head.attention_head.w_v,
        )
        for cached_head in block.cached_mha.cached_heads
    ]


def _assert_tensor_tree(value, dtype):
    if isinstance(value, torch.Tensor):
        assert value.dtype == dtype
        assert value.device == TEST_DEVICE
        assert torch.isfinite(value).all().item()
        return
    assert isinstance(value, (tuple, list))
    for child in value:
        _assert_tensor_tree(child, dtype)


def test_CDB01_prefill_decode_shapes_and_cache_containers():
    block = _new_block().eval()
    x = _tensor(_MAIN_X[:3])
    x_new = _tensor([_MAIN_X[3]])

    with torch.no_grad():
        prefill_result = block.prefill(x)
        assert isinstance(prefill_result, tuple) and len(prefill_result) == 3
        assert prefill_result[0].shape == (3, 5)
        assert prefill_result[1].shape == (2, 3, 3)
        _assert_cache_shapes(prefill_result[2], _DIMS, 3, x.dtype)
        decode_result = block.decode(x_new, prefill_result[2])

    assert isinstance(decode_result, tuple) and len(decode_result) == 3
    assert decode_result[0].shape == (1, 5)
    assert decode_result[1].shape == (2, 1, 4)
    _assert_cache_shapes(decode_result[2], _DIMS, 4, x.dtype)
    assert _DIMS["num_heads"] * _DIMS["d_v"] != _DIMS["d_model"]


def test_CDB02_registered_composition_eps_and_persistent_parameters():
    for eps in (1e-5, _EPS):
        if eps == 1e-5:
            block = CachedPreNormDecoderBlock(**_DIMS).to(
                device=TEST_DEVICE,
                dtype=torch.float32,
            ).eval()
        else:
            block = CachedPreNormDecoderBlock(**_DIMS, eps=eps).to(
                device=TEST_DEVICE,
                dtype=torch.float32,
            ).eval()
        assert isinstance(block.normalize_input, LayerNorm)
        assert isinstance(block.ffn_norm, LayerNorm)
        assert block.normalize_input is not block.ffn_norm
        assert block.normalize_input.eps == eps
        assert block.ffn_norm.eps == eps
        assert isinstance(block.cached_mha, CachedMultiHeadAttention)
        assert isinstance(block.ffn, PositionWiseFFN)
        assert len(block.cached_mha.cached_heads) == _DIMS["num_heads"]
        assert block.ffn.d_model == _DIMS["d_model"]
        assert block.ffn.d_ff == _DIMS["d_ff"]
        assert block.cached_mha.w_o.shape == (_DIMS["num_heads"] * _DIMS["d_v"], _DIMS["d_model"])
        assert block.normalize_input.gamma is not block.ffn_norm.gamma
        assert block.normalize_input.beta is not block.ffn_norm.beta

        registered_modules = tuple(block.named_modules())
        registered_parameters = tuple(block.named_parameters())
        for child in (block.normalize_input, block.cached_mha, block.ffn_norm, block.ffn):
            assert any(module is child for _, module in registered_modules)
        for cached_head in block.cached_mha.cached_heads:
            assert any(module is cached_head for _, module in registered_modules)
            projections = (
                (cached_head.attention_head.w_q, (_DIMS["d_model"], _DIMS["d_k"])),
                (cached_head.attention_head.w_k, (_DIMS["d_model"], _DIMS["d_k"])),
                (cached_head.attention_head.w_v, (_DIMS["d_model"], _DIMS["d_v"])),
            )
            for parameter, shape in projections:
                assert parameter.shape == shape
                assert any(candidate is parameter for _, candidate in registered_parameters)
        for parameter in (
            block.normalize_input.gamma,
            block.normalize_input.beta,
            block.ffn_norm.gamma,
            block.ffn_norm.beta,
            block.cached_mha.w_o,
            block.ffn.w_1,
            block.ffn.b_1,
            block.ffn.w_2,
            block.ffn.b_2,
        ):
            assert any(candidate is parameter for _, candidate in registered_parameters)

        module_ids = tuple((name, id(module)) for name, module in block.named_modules())
        parameter_snapshot = {
            name: (id(parameter), parameter.detach().clone())
            for name, parameter in block.named_parameters()
        }
        x = _tensor(_MAIN_X[:2])
        x_new = _tensor([_MAIN_X[2]])
        with torch.no_grad():
            prefill_result = block.prefill(x)
            block.decode(x_new, prefill_result[2])
        assert tuple((name, id(module)) for name, module in block.named_modules()) == module_ids
        current_parameters = dict(block.named_parameters())
        assert current_parameters.keys() == parameter_snapshot.keys()
        for name, parameter in current_parameters.items():
            identity, value = parameter_snapshot[name]
            assert id(parameter) == identity
            assert torch.equal(parameter, value)


def test_CDB03_prefill_dispatches_once_with_normalized_prefix_and_preserves_returns():
    dimensions = _FLOW_DIMS
    block = _new_block(**dimensions).eval()
    _set_norm_values(block)
    x = _flow_input("prefill", torch.float32)
    update = _flow_update("prefill", x.dtype)
    weights = _prefill_diagnostic_weights(x.dtype)
    caches = _diagnostic_cache(2, dimensions, x.dtype, offset=17)
    spy = AttentionDispatchSpy((update, weights, caches), None)
    block.cached_mha = spy
    expected_weights = weights.detach().clone()
    expected_caches = _snapshot_cache(caches)

    with torch.no_grad():
        result = block.prefill(x)
        expected_normalized = _layer_norm_reference(x, block.normalize_input)

    assert len(spy.prefill_inputs) == 1
    assert spy.decode_inputs == []
    assert torch.all(expected_weights >= 0).item()
    assert torch.equal(expected_weights[:, 0, 1], torch.zeros_like(expected_weights[:, 0, 1]))
    _assert_close(
        expected_weights.sum(dim=-1),
        torch.ones((2, 2), dtype=x.dtype, device=TEST_DEVICE),
    )
    assert not torch.equal(expected_weights[0, 1], expected_weights[1, 1])
    assert not torch.equal(expected_weights[0, 1], expected_weights[0, 1].flip(0))
    assert not torch.equal(expected_weights[0, 0], expected_weights[0, 1])
    _assert_close(spy.prefill_inputs[0], expected_normalized)
    assert result[0].shape == x.shape
    _assert_close(result[1], expected_weights)
    _assert_cache_close(result[2], expected_caches)


def test_CDB04_decode_dispatches_once_with_normalized_row_and_ordered_caches(monkeypatch):
    dimensions = _FLOW_DIMS
    block = _new_block(**dimensions).eval()
    _set_norm_values(block)
    x_new = _flow_input("decode", torch.float32)
    incoming = _diagnostic_cache(2, dimensions, x_new.dtype, offset=5)
    update = _flow_update("decode", x_new.dtype)
    weights = _decode_diagnostic_weights(x_new.dtype)
    returned_caches = _diagnostic_cache(3, dimensions, x_new.dtype, offset=29)
    spy = AttentionDispatchSpy(None, (update, weights, returned_caches))
    block.cached_mha = spy
    expected_weights = weights.detach().clone()
    expected_incoming = _snapshot_cache(incoming)
    expected_returned_caches = _snapshot_cache(returned_caches)
    norm_inputs = []
    norm_hook = block.normalize_input.register_forward_pre_hook(
        lambda module, args: norm_inputs.append(args[0].detach().clone())
    )

    def forbidden_uncached_attention(*args, **kwargs):
        pytest.fail("decode khong duoc goi uncached attention")

    monkeypatch.setattr(MultiHeadCausalSelfAttention, "forward", forbidden_uncached_attention)
    try:
        with torch.no_grad():
            result = block.decode(x_new, incoming)
            expected_normalized = _layer_norm_reference(x_new, block.normalize_input)
    finally:
        norm_hook.remove()

    assert len(norm_inputs) == 1
    assert norm_inputs[0].shape == (1, dimensions["d_model"])
    _assert_close(norm_inputs[0], x_new)
    assert spy.prefill_inputs == []
    assert len(spy.decode_inputs) == 1
    _assert_close(spy.decode_inputs[0], expected_normalized)
    _assert_cache_close(spy.decode_caches[0], expected_incoming)
    assert torch.all(expected_weights > 0).item()
    _assert_close(
        expected_weights.sum(dim=-1),
        torch.ones((2, 1), dtype=x_new.dtype, device=TEST_DEVICE),
    )
    assert not torch.equal(expected_weights[0], expected_weights[1])
    assert not torch.equal(expected_weights[0, 0], expected_weights[0, 0].flip(0))
    assert result[0].shape == x_new.shape
    _assert_close(result[1], expected_weights)
    _assert_cache_close(result[2], expected_returned_caches)


@pytest.mark.parametrize("path", ("prefill", "decode"))
def test_CDB05_first_residual_uses_original_input(path):
    dimensions = _FLOW_DIMS
    block = _new_block(**dimensions).eval()
    _set_norm_values(block)
    x = _flow_input(path, torch.float32)
    update = _flow_update(path, x.dtype)
    spy, incoming = _flow_spy(path, x, update)
    block.cached_mha = spy
    ffn_norm_inputs = []
    hook = block.ffn_norm.register_forward_pre_hook(
        lambda module, args: ffn_norm_inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        if path == "prefill":
            block.prefill(x)
        else:
            block.decode(x, incoming)
    hook.remove()

    with torch.no_grad():
        expected_residual = x + update
        normalized_residual = _layer_norm_reference(x, block.normalize_input) + update
    assert len(ffn_norm_inputs) == 1
    assert not torch.allclose(expected_residual, normalized_residual, **_TOLERANCES[x.dtype])
    _assert_close(ffn_norm_inputs[0], expected_residual)


@pytest.mark.parametrize("path", ("prefill", "decode"))
def test_CDB06_ffn_uses_second_norm_exact_gelu_and_residual(path):
    dimensions = _FLOW_DIMS
    block = _new_block(**dimensions).eval()
    _set_norm_values(block)
    x = _flow_input(path, torch.float32)
    update = _flow_update(path, x.dtype)
    spy, incoming = _flow_spy(path, x, update)
    block.cached_mha = spy
    with torch.no_grad():
        block.ffn.w_1.copy_(_matrix_fixture(block.ffn.w_1, 1))
        block.ffn.w_2.copy_(_matrix_fixture(block.ffn.w_2, 2))
        block.ffn.b_1.copy_(_tensor([0.08, -0.12, 0.05, 0.09, -0.07], block.ffn.b_1.dtype))
        block.ffn.b_2.copy_(_tensor([0.04, -0.08, 0.11], block.ffn.b_2.dtype))
    ffn_inputs = []
    hook = block.ffn.register_forward_pre_hook(
        lambda module, args: ffn_inputs.append(args[0].detach().clone())
    )
    with torch.no_grad():
        if path == "prefill":
            result = block.prefill(x)
        else:
            result = block.decode(x, incoming)
    hook.remove()

    with torch.no_grad():
        attention_residual = x + update
        normalized = _layer_norm_reference(attention_residual, block.ffn_norm)
        hidden = F.linear(normalized, block.ffn.w_1.T, block.ffn.b_1)
        activated = F.gelu(hidden, approximate="none")
        expected_ffn = F.linear(activated, block.ffn.w_2.T, block.ffn.b_2)
        expected_output = attention_residual + expected_ffn
    assert len(ffn_inputs) == 1
    assert torch.count_nonzero(block.ffn.b_1).item() > 0
    assert torch.count_nonzero(block.ffn.b_2).item() > 0
    assert not torch.allclose(attention_residual, normalized, **_TOLERANCES[x.dtype])
    assert not torch.allclose(attention_residual, x, **_TOLERANCES[x.dtype])
    _assert_close(ffn_inputs[0], normalized)
    _assert_close(result[0], expected_output)


def test_CDB07_cached_prefill_and_decode_match_semantically_aligned_uncached_block():
    cached, uncached = _aligned_blocks(torch.float64)
    x = _tensor(_MAIN_X, torch.float64)

    with torch.no_grad():
        reference_output, reference_weights = uncached(x)
        prefill = cached.prefill(x[:3])
        expected_prefill_cache = _projected_kv(
            cached,
            _layer_norm_reference(x[:3], cached.normalize_input),
        )
        _assert_close(prefill[0], reference_output[:3])
        _assert_close(prefill[1], reference_weights[:, :3, :3])
        _assert_cache_close(prefill[2], expected_prefill_cache)
        prefill_cache_snapshot = _snapshot_cache(prefill[2])
        decode = cached.decode(x[3:4], prefill[2])
        expected_full_cache = _projected_kv(
            cached,
            _layer_norm_reference(x, cached.normalize_input),
        )

    _assert_close(decode[0], reference_output[3:4])
    _assert_close(decode[1], reference_weights[:, 3:4, :])
    _assert_cache_close(decode[2], expected_full_cache)
    for head_index, ((old_key, old_value), (new_key, new_value)) in enumerate(
        zip(prefill_cache_snapshot, decode[2])
    ):
        _assert_close(new_key[:3], old_key)
        _assert_close(new_value[:3], old_value)
        _assert_close(new_key[3:4], expected_full_cache[head_index][0][3:4])
        _assert_close(new_value[3:4], expected_full_cache[head_index][1][3:4])


def test_CDB08_caller_cache_continues_sequence_after_unrelated_prefill():
    cached, uncached = _aligned_blocks(torch.float32)
    sequence_a = _tensor(_MAIN_X, torch.float32)
    unrelated_b = _tensor([[4.0, -3.0, 2.0, -1.0, 0.5]], torch.float32)

    with torch.no_grad():
        reference_prefix_output, reference_prefix_weights = uncached(sequence_a[:2])
        prefill_a = cached.prefill(sequence_a[:2])
        expected_prefix_cache = _projected_kv(
            cached,
            _layer_norm_reference(sequence_a[:2], cached.normalize_input),
        )
        _assert_close(prefill_a[0], reference_prefix_output)
        _assert_close(prefill_a[1], reference_prefix_weights)
        _assert_cache_close(prefill_a[2], expected_prefix_cache)
        prefix_cache_snapshot = _snapshot_cache(prefill_a[2])

        unrelated_result = cached.prefill(unrelated_b)
        assert unrelated_result[0].shape == unrelated_b.shape
        assert unrelated_result[1].shape == (2, 1, 1)
        _assert_cache_shapes(unrelated_result[2], _DIMS, 1, unrelated_b.dtype)

        first_decode = cached.decode(sequence_a[2:3], prefill_a[2])
        reference_first_output, reference_first_weights = uncached(sequence_a[:3])
        expected_first_cache = _projected_kv(
            cached,
            _layer_norm_reference(sequence_a[:3], cached.normalize_input),
        )
        _assert_close(first_decode[0], reference_first_output[2:3])
        _assert_close(first_decode[1], reference_first_weights[:, 2:3, :])
        _assert_cache_close(first_decode[2], expected_first_cache)
        _assert_cache_shapes(first_decode[2], _DIMS, 3, sequence_a.dtype)
        first_cache_snapshot = _snapshot_cache(first_decode[2])
        for (prefix_key, prefix_value), (first_key, first_value) in zip(
            prefix_cache_snapshot,
            first_decode[2],
        ):
            _assert_close(first_key[:2], prefix_key)
            _assert_close(first_value[:2], prefix_value)

        second_decode = cached.decode(sequence_a[3:4], first_decode[2])
        reference_second_output, reference_second_weights = uncached(sequence_a)
        expected_second_cache = _projected_kv(
            cached,
            _layer_norm_reference(sequence_a, cached.normalize_input),
        )
        _assert_close(second_decode[0], reference_second_output[3:4])
        _assert_close(second_decode[1], reference_second_weights[:, 3:4, :])
        _assert_cache_close(second_decode[2], expected_second_cache)
        _assert_cache_shapes(second_decode[2], _DIMS, 4, sequence_a.dtype)
        for (first_key, first_value), (second_key, second_value) in zip(
            first_cache_snapshot,
            second_decode[2],
        ):
            _assert_close(second_key[:3], first_key)
            _assert_close(second_value[:3], first_value)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_CDB09_singleton_prefill_decode_preserve_dtype_device_and_finiteness(dtype):
    block = _new_block(dtype=dtype).eval()
    x = _tensor([[0.25, -0.4, 0.8, 0.1, -0.6]], dtype)
    x_new = _tensor([[-0.2, 0.7, -0.1, 0.5, 0.3]], dtype)

    with torch.no_grad():
        prefill = block.prefill(x)
        assert isinstance(prefill, tuple) and len(prefill) == 3
        assert prefill[0].shape == (1, _DIMS["d_model"])
        assert prefill[1].shape == (_DIMS["num_heads"], 1, 1)
        _assert_cache_shapes(prefill[2], _DIMS, 1, dtype)
        _assert_tensor_tree(prefill, dtype)
        decode = block.decode(x_new, prefill[2])

    assert isinstance(decode, tuple) and len(decode) == 3
    assert decode[0].shape == (1, _DIMS["d_model"])
    assert decode[1].shape == (_DIMS["num_heads"], 1, 2)
    _assert_cache_shapes(decode[2], _DIMS, 2, dtype)
    _assert_tensor_tree(decode, dtype)
