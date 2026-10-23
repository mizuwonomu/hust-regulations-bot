"""Kiểm thử ghép nối và suy luận prefill/decode của decoder-only LM có KV cache."""

import pytest
import torch
from torch import nn

from llm_foundations.decoder_only_lm import DecoderOnlyLM
from llm_foundations.kv_cache.decoder_only_lm import CachedDecoderOnlyLM


TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected test device: {TEST_DEVICE}")
_EPS = 3e-4
_DIMS = {
    "vocab_size": 11,
    "max_seq_len": 7,
    "d_model": 5,
    "num_layers": 2,
    "num_heads": 2,
    "d_k": 3,
    "d_v": 2,
    "d_ff": 7,
}
_TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-5},
    torch.float64: {"rtol": 1e-9, "atol": 1e-10},
}


@pytest.fixture(autouse=True)
def _disable_reduced_precision():
    """Tắt phép nhân giảm chính xác trong fixture và khôi phục cấu hình sau đó."""
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


def _tensor(values, dtype=torch.float32):
    """Tạo tensor trên device kiểm thử với dtype được chỉ định."""
    return torch.tensor(values, dtype=dtype, device=TEST_DEVICE)


def _new_model(dtype=torch.float32, eps=_EPS, **overrides):
    """Tạo model mới trên device kiểm thử với kích thước fixture."""
    dimensions = {**_DIMS, **overrides}
    return CachedDecoderOnlyLM(**dimensions, eps=eps).to(
        device=TEST_DEVICE,
        dtype=dtype,
    ).eval()


def _assert_close(actual, expected):
    """Kiểm tra shape, dtype, device, tính hữu hạn và sai số số học."""
    assert isinstance(actual, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert expected.dtype == actual.dtype
    assert actual.device == TEST_DEVICE
    assert expected.device == TEST_DEVICE
    assert torch.isfinite(actual).all().item()
    assert torch.isfinite(expected).all().item()
    torch.testing.assert_close(actual, expected, **_TOLERANCES[actual.dtype])


def _clone_tree(value):
    """Sao chép đệ quy tensor, tuple và list để cố định oracle trước lời gọi."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_tree(child) for child in value)
    if isinstance(value, list):
        return [_clone_tree(child) for child in value]
    return value


def _assert_tree_equal(actual, expected):
    """So sánh chính xác cấu trúc và giá trị cây cache đã chụp."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.device == TEST_DEVICE
        assert expected.device == TEST_DEVICE
        assert torch.equal(actual, expected)
        return
    assert type(actual) is type(expected)
    assert len(actual) == len(expected)
    for actual_child, expected_child in zip(actual, expected):
        _assert_tree_equal(actual_child, expected_child)


def _assert_tree_properties(value, dtype):
    """Kiểm tra dtype, device và tính hữu hạn trên mọi tensor lồng nhau."""
    if isinstance(value, torch.Tensor):
        assert value.dtype == dtype
        assert value.device == TEST_DEVICE
        assert torch.isfinite(value).all().item()
        return
    assert isinstance(value, (tuple, list))
    for child in value:
        _assert_tree_properties(child, dtype)


def _cache_fixture(length, dimensions, dtype, base):
    """Tạo cache phân biệt theo layer, head và vị trí trên device kiểm thử."""
    layers = []
    for layer_index in range(dimensions["num_layers"]):
        heads = []
        for head_index in range(dimensions["num_heads"]):
            key = torch.arange(
                length * dimensions["d_k"], dtype=dtype, device=TEST_DEVICE
            ).reshape(length, dimensions["d_k"])
            value = torch.arange(
                length * dimensions["d_v"], dtype=dtype, device=TEST_DEVICE
            ).reshape(length, dimensions["d_v"])
            offset = base + 31 * layer_index + 13 * head_index
            heads.append((key + offset + 1, value + offset + 7))
        layers.append(heads)
    return layers


def _assert_model_cache_shapes(caches, dimensions, length, dtype):
    """Kiểm tra số layer/head, shape K/V, dtype và device của model cache."""
    assert isinstance(caches, list)
    assert len(caches) == dimensions["num_layers"]
    for layer in caches:
        assert isinstance(layer, list)
        assert len(layer) == dimensions["num_heads"]
        for pair in layer:
            assert isinstance(pair, tuple) and len(pair) == 2
            key, value = pair
            assert key.shape == (length, dimensions["d_k"])
            assert value.shape == (length, dimensions["d_v"])
            for tensor in (key, value):
                assert tensor.dtype == dtype
                assert tensor.device == TEST_DEVICE
                assert torch.isfinite(tensor).all().item()


def _install_output_observers(monkeypatch, model, events, bias):
    """Theo dõi thứ tự final norm/head và chụp hidden states đầu vào."""
    norm_inputs = []
    head_inputs = []

    def observe_norm(hidden):
        events.append("final_norm")
        norm_inputs.append(hidden.detach().clone())
        return hidden + bias

    def observe_head(module, arguments):
        events.append("lm_head")
        head_inputs.append(arguments[0].detach().clone())

    monkeypatch.setattr(model.final_norm, "forward", observe_norm)
    handle = model.lm_head.register_forward_pre_hook(observe_head)
    return norm_inputs, head_inputs, handle


def _parameter_values(parameter, role, name):
    """Sinh giá trị hữu hạn, bị chặn và phân biệt cho một parameter."""
    indices = torch.arange(
        parameter.numel(), dtype=parameter.dtype, device=TEST_DEVICE
    ).reshape_as(parameter)
    if name.endswith(".gamma"):
        return 0.82 + 0.035 * ((indices + role) % 9)
    if name.endswith(".beta"):
        return 0.06 * (((indices + role) % 7) - 3) + 0.004 * role
    return 0.035 * (((indices * 5 + 3 * role) % 19) - 9) + 0.002 * role


def _set_deterministic_parameters(model):
    """Gán fixture xác định cho mọi parameter của model cached."""
    with torch.no_grad():
        for role, (name, parameter) in enumerate(model.named_parameters(), start=1):
            parameter.copy_(_parameter_values(parameter, role, name))


def _copy_cached_parameters(source, destination):
    """Sao chép theo tên giữa hai model cached độc lập có cùng cấu trúc."""
    source_parameters = dict(source.named_parameters())
    destination_parameters = dict(destination.named_parameters())
    assert source_parameters.keys() == destination_parameters.keys()
    with torch.no_grad():
        for name, source_parameter in source_parameters.items():
            destination_parameter = destination_parameters[name]
            assert source_parameter is not destination_parameter
            assert source_parameter.shape == destination_parameter.shape
            destination_parameter.copy_(source_parameter.detach().clone())
            assert torch.equal(source_parameter, destination_parameter)


def _semantic_model_pairs(cached, uncached):
    """Liệt kê cặp parameter tương ứng theo vai trò giữa hai kiến trúc."""
    pairs = [
        ("embedding.token_embed", cached.input_embedding.token_embed, uncached.input_embedding.token_embed),
        ("embedding.pos_embed", cached.input_embedding.pos_embed, uncached.input_embedding.pos_embed),
    ]
    for layer_index, (cached_block, uncached_block) in enumerate(
        zip(cached.blocks, uncached.blocks)
    ):
        prefix = f"block.{layer_index}"
        cached_norm = cached_block.normalize_input
        uncached_norm = uncached_block.attention_residual.normalized_input
        pairs.extend(
            [
                (f"{prefix}.ln1.gamma", cached_norm.gamma, uncached_norm.gamma),
                (f"{prefix}.ln1.beta", cached_norm.beta, uncached_norm.beta),
                (f"{prefix}.ln2.gamma", cached_block.ffn_norm.gamma, uncached_block.ffn_norm.gamma),
                (f"{prefix}.ln2.beta", cached_block.ffn_norm.beta, uncached_block.ffn_norm.beta),
                (
                    f"{prefix}.attention.w_o",
                    cached_block.cached_mha.w_o,
                    uncached_block.attention_residual.multi_head_attention.w_o,
                ),
                (f"{prefix}.ffn.w_1", cached_block.ffn.w_1, uncached_block.ffn.w_1),
                (f"{prefix}.ffn.b_1", cached_block.ffn.b_1, uncached_block.ffn.b_1),
                (f"{prefix}.ffn.w_2", cached_block.ffn.w_2, uncached_block.ffn.w_2),
                (f"{prefix}.ffn.b_2", cached_block.ffn.b_2, uncached_block.ffn.b_2),
            ]
        )
        cached_heads = cached_block.cached_mha.cached_heads
        uncached_heads = uncached_block.attention_residual.multi_head_attention.heads
        assert len(cached_heads) == len(uncached_heads)
        for head_index, (cached_head, uncached_head) in enumerate(
            zip(cached_heads, uncached_heads)
        ):
            for projection in ("w_q", "w_k", "w_v"):
                pairs.append(
                    (
                        f"{prefix}.head.{head_index}.{projection}",
                        getattr(cached_head.attention_head, projection),
                        getattr(uncached_head, projection),
                    )
                )
    pairs.extend(
        [
            ("final_norm.gamma", cached.final_norm.gamma, uncached.final_norm.gamma),
            ("final_norm.beta", cached.final_norm.beta, uncached.final_norm.beta),
            ("lm_head.w_lmhead", cached.lm_head.w_lmhead, uncached.lm_head.w_lmhead),
        ]
    )
    return pairs


def _copy_semantic_model_parameters(cached, uncached):
    """Sao chép parameters theo vai trò và xác nhận shape, epsilon, giá trị."""
    pairs = _semantic_model_pairs(cached, uncached)
    expected_roles = {"embedding.token_embed", "embedding.pos_embed", "final_norm.gamma", "final_norm.beta", "lm_head.w_lmhead"}
    for layer_index in range(_DIMS["num_layers"]):
        prefix = f"block.{layer_index}"
        expected_roles.update(
            {
                f"{prefix}.ln1.gamma",
                f"{prefix}.ln1.beta",
                f"{prefix}.ln2.gamma",
                f"{prefix}.ln2.beta",
                f"{prefix}.attention.w_o",
                f"{prefix}.ffn.w_1",
                f"{prefix}.ffn.b_1",
                f"{prefix}.ffn.w_2",
                f"{prefix}.ffn.b_2",
            }
        )
        expected_roles.update(
            f"{prefix}.head.{head_index}.{projection}"
            for head_index in range(_DIMS["num_heads"])
            for projection in ("w_q", "w_k", "w_v")
        )
    assert {role for role, _, _ in pairs} == expected_roles
    for cached_block, uncached_block in zip(cached.blocks, uncached.blocks):
        assert cached_block.normalize_input.eps == uncached_block.attention_residual.normalized_input.eps
        assert cached_block.ffn_norm.eps == uncached_block.ffn_norm.eps
    assert cached.final_norm.eps == uncached.final_norm.eps
    with torch.no_grad():
        for _, source, destination in pairs:
            assert source is not destination
            assert source.shape == destination.shape
            destination.copy_(source.detach().clone())
            assert torch.equal(source, destination)


def test_CDLM01_registered_independent_components_and_persistent_parameters():
    model = _new_model(eps=_EPS)
    dimensions = dict(_DIMS)
    registered_modules = tuple(model.named_modules())
    registered_parameters = tuple(model.named_parameters())
    _set_deterministic_parameters(model)
    assert isinstance(model.input_embedding, nn.Module)
    assert isinstance(model.blocks, nn.ModuleList)
    assert isinstance(model.final_norm, nn.Module)
    assert isinstance(model.lm_head, nn.Module)
    for module in (model.input_embedding, model.blocks, model.final_norm, model.lm_head):
        assert any(candidate is module for _, candidate in registered_modules)
    assert len(model.blocks) == 2
    assert model.blocks[0] is not model.blocks[1]
    first_ids = {id(parameter) for parameter in model.blocks[0].parameters()}
    second_ids = {id(parameter) for parameter in model.blocks[1].parameters()}
    assert first_ids.isdisjoint(second_ids)
    for block in model.blocks:
        assert block.normalize_input.eps == _EPS
        assert block.ffn_norm.eps == _EPS
        for parameter in block.parameters():
            assert any(candidate is parameter for _, candidate in registered_parameters)
    assert model.final_norm.eps == _EPS
    assert model.lm_head.w_lmhead is not model.input_embedding.token_embed
    assert model.lm_head.w_lmhead.shape == (dimensions["d_model"], dimensions["vocab_size"])
    registered_parameter_names = {name for name, _ in registered_parameters}
    assert {
        "input_embedding.token_embed",
        "input_embedding.pos_embed",
        "final_norm.gamma",
        "final_norm.beta",
        "lm_head.w_lmhead",
    } <= registered_parameter_names

    token_ids = torch.tensor([1, 4], dtype=torch.long, device=TEST_DEVICE)
    new_token = torch.tensor([7], dtype=torch.long, device=TEST_DEVICE)
    module_ids = tuple((name, id(module)) for name, module in model.named_modules())
    parameter_snapshot = {
        name: (id(parameter), parameter.detach().clone())
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        prefill_logits, prefill_cache = model.prefill(token_ids)
    assert prefill_logits.shape == (2, dimensions["vocab_size"])
    _assert_tree_properties(prefill_logits, prefill_logits.dtype)
    _assert_model_cache_shapes(prefill_cache, dimensions, 2, prefill_logits.dtype)
    for name, parameter in model.named_parameters():
        identity, value = parameter_snapshot[name]
        assert id(parameter) == identity
        assert torch.equal(parameter, value)
    with torch.no_grad():
        decode_logits, decode_cache = model.decode(new_token, prefill_cache)
    assert decode_logits.shape == (1, dimensions["vocab_size"])
    _assert_model_cache_shapes(decode_cache, dimensions, 3, decode_logits.dtype)
    assert tuple((name, id(module)) for name, module in model.named_modules()) == module_ids
    current_parameters = dict(model.named_parameters())
    assert current_parameters.keys() == parameter_snapshot.keys()
    for name, parameter in current_parameters.items():
        identity, value = parameter_snapshot[name]
        assert id(parameter) == identity
        assert torch.equal(parameter, value)


def test_CDLM02_prefill_propagates_hidden_states_and_collects_layer_caches(monkeypatch):
    model = _new_model(num_layers=2, num_heads=2)
    positions = _tensor(
        [
            [-1.0, 0.2, 0.5, -0.3, 0.8],
            [2.0, -0.4, 0.1, 0.7, -0.9],
            [0.3, 1.1, -0.6, 0.4, 0.2],
        ]
    )
    with torch.no_grad():
        model.input_embedding.token_embed.zero_()
        model.input_embedding.pos_embed[:3].copy_(positions)
        model.lm_head.w_lmhead.zero_()
        model.lm_head.w_lmhead[0, 0] = 3.0
        model.lm_head.w_lmhead[0, 1] = -3.0
    ids = torch.tensor([1, 2], dtype=torch.long, device=TEST_DEVICE)
    dimensions = {**_DIMS, "num_layers": 2, "num_heads": 2}
    calls = []
    layer_dimensions = {**dimensions, "num_layers": 1}
    first_cache = _cache_fixture(2, layer_dimensions, torch.float32, 100)[0]
    second_cache = _cache_fixture(2, layer_dimensions, torch.float32, 500)[0]
    cache_snapshots = [_clone_tree(first_cache), _clone_tree(second_cache)]
    first_offset = _tensor([0.5, -0.2, 0.4, 0.1, -0.3])
    second_offset = _tensor([0.3, 0.6, -0.5, 0.2, 0.1])
    transforms = (
        lambda hidden: hidden * 2.0 + first_offset,
        lambda hidden: hidden * 2.0 + second_offset,
    )
    spy_caches = (first_cache, second_cache)
    for layer_index, block in enumerate(model.blocks):
        def prefill_spy(hidden, index=layer_index):
            calls.append((index, hidden.detach().clone()))
            output = transforms[index](hidden)
            weights = torch.full(
                (dimensions["num_heads"], hidden.shape[0], hidden.shape[0]),
                1.0 / hidden.shape[0],
                dtype=hidden.dtype,
                device=TEST_DEVICE,
            )
            return output, weights, spy_caches[index]

        monkeypatch.setattr(block, "prefill", prefill_spy)
    events = []
    norm_bias = _tensor([2.0, 0.0, 0.0, 0.0, 0.0])
    norm_inputs, head_inputs, head_handle = _install_output_observers(
        monkeypatch, model, events, norm_bias
    )
    head_weight = model.lm_head.w_lmhead.detach().clone()
    expected_embedding = positions[:2].detach().clone()
    expected_after_first = transforms[0](expected_embedding).detach().clone()
    expected_after_second = transforms[1](expected_after_first).detach().clone()
    expected_normed = expected_after_second + norm_bias
    expected_logits = expected_normed @ head_weight
    try:
        with torch.no_grad():
            logits, caches = model.prefill(ids)
    finally:
        head_handle.remove()

    assert [index for index, _ in calls] == [0, 1]
    _assert_close(calls[0][1], expected_embedding)
    _assert_close(calls[1][1], expected_after_first)
    assert events == ["final_norm", "lm_head"]
    assert len(norm_inputs) == len(head_inputs) == 1
    _assert_close(norm_inputs[0], expected_after_second)
    _assert_close(head_inputs[0], expected_normed)
    _assert_close(logits, expected_logits)
    assert logits.shape == (2, dimensions["vocab_size"])
    assert (logits < 0).any().item()
    assert (logits > 1).any().item()
    _assert_tree_equal(caches, cache_snapshots)
    _assert_model_cache_shapes(caches, dimensions, 2, torch.float32)


def test_CDLM03_decode_routes_each_layer_cache_and_collects_updated_state(monkeypatch):
    model = _new_model(num_layers=2, num_heads=2, dtype=torch.float64)
    dimensions = {**_DIMS, "num_layers": 2, "num_heads": 2}
    incoming = _cache_fixture(3, dimensions, torch.float64, 20)
    incoming_snapshot = _clone_tree(incoming)
    updated = [
        _cache_fixture(4, dimensions, torch.float64, 900 + 300 * layer_index)[layer_index]
        for layer_index in range(dimensions["num_layers"])
    ]
    updated_snapshot = _clone_tree(updated)
    updated_payloads = _clone_tree(updated_snapshot)
    decode_inputs = []
    decode_cache_arguments = []
    decode_events = []
    first_shift = _tensor([0.4, -0.2, 0.1, 0.6, -0.5], torch.float64)
    second_shift = _tensor([-0.3, 0.7, -0.1, 0.2, 0.5], torch.float64)

    for layer_index, block in enumerate(model.blocks):
        def decode_spy(hidden, caches, index=layer_index):
            decode_events.append(index)
            decode_inputs.append(hidden.detach().clone())
            decode_cache_arguments.append(_clone_tree(caches))
            output = hidden * (2.0 if index == 0 else -0.5)
            output = output + (first_shift if index == 0 else second_shift)
            weights = torch.ones(
                (dimensions["num_heads"], 1, 4),
                dtype=hidden.dtype,
                device=TEST_DEVICE,
            ) / 4
            return output, weights, updated_payloads[index]

        monkeypatch.setattr(block, "decode", decode_spy)

        def forbidden_prefill(*args, **kwargs):
            pytest.fail("decode không được chạy lại prefill")

        monkeypatch.setattr(block, "prefill", forbidden_prefill)

    events = []
    norm_bias = _tensor([0.1, -0.2, 0.3, 0.4, -0.5], torch.float64)
    norm_inputs, head_inputs, head_handle = _install_output_observers(
        monkeypatch, model, events, norm_bias
    )
    position_zero = _tensor([-0.5, 0.3, 0.4, -0.2, 0.7], torch.float64)
    position_three = _tensor([0.9, -0.7, 0.1, 0.6, -0.4], torch.float64)
    with torch.no_grad():
        model.input_embedding.token_embed[6].copy_(_tensor([0.2, -0.4, 0.8, 0.1, -0.3], torch.float64))
        model.input_embedding.pos_embed[0].copy_(position_zero)
        model.input_embedding.pos_embed[3].copy_(position_three)
    token = torch.tensor([6], dtype=torch.long, device=TEST_DEVICE)
    expected_token = model.input_embedding.token_embed[6].detach().clone()
    expected_position_zero = model.input_embedding.pos_embed[0].detach().clone()
    expected_position_three = model.input_embedding.pos_embed[3].detach().clone()
    assert not torch.equal(expected_position_zero, expected_position_three)
    expected_first_input = (expected_token + expected_position_three).reshape(1, -1)
    expected_second_input = expected_first_input * 2.0 + first_shift
    expected_final = expected_second_input * -0.5 + second_shift
    expected_normed = expected_final + norm_bias
    head_weight = model.lm_head.w_lmhead.detach().clone()
    expected_logits = expected_normed @ head_weight
    try:
        with torch.no_grad():
            logits, returned = model.decode(token, incoming)
    finally:
        head_handle.remove()

    assert decode_events == [0, 1]
    assert len(decode_inputs) == len(decode_cache_arguments) == 2
    failures = []
    observations = (
        ("layer 0 cache", lambda: _assert_tree_equal(decode_cache_arguments[0], incoming_snapshot[0])),
        ("layer 1 cache", lambda: _assert_tree_equal(decode_cache_arguments[1], incoming_snapshot[1])),
        ("layer 0 input uses P[3]", lambda: _assert_close(decode_inputs[0], expected_first_input)),
        ("layer 1 receives previous hidden", lambda: _assert_close(decode_inputs[1], expected_second_input)),
        ("final norm input", lambda: _assert_close(norm_inputs[0], expected_final)),
        ("LM head input", lambda: _assert_close(head_inputs[0], expected_normed)),
        ("raw logits", lambda: _assert_close(logits, expected_logits)),
        ("returned cache values", lambda: _assert_tree_equal(returned, updated_snapshot)),
    )
    for label, observation in observations:
        try:
            observation()
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert events == ["final_norm", "lm_head"]
    assert logits.shape == (1, dimensions["vocab_size"])
    _assert_model_cache_shapes(returned, dimensions, 4, torch.float64)
    assert not failures, "CDLM03 mismatches:\n" + "\n".join(failures)


def test_CDLM04_decode_uses_consecutive_absolute_position_offsets():
    dimensions = {**_DIMS, "num_layers": 1, "max_seq_len": 6}
    model = _new_model(dtype=torch.float64, **dimensions)
    positions = _tensor(
        [
            [0.1, -0.4, 0.8, 0.2, -0.7],
            [0.5, 0.2, -0.3, 0.9, -0.1],
            [-0.8, 0.3, 1.1, -0.2, 0.6],
            [0.4, -0.9, 0.2, 0.7, -0.3],
            [1.0, 0.1, -0.6, 0.3, -0.2],
            [-0.5, 0.8, 0.4, -0.7, 0.2],
        ],
        torch.float64,
    )
    with torch.no_grad():
        model.input_embedding.pos_embed.copy_(positions)
        model.input_embedding.token_embed.copy_(
            _tensor(
                (
                    [
                        [0.1, 0.3, -0.2, 0.5, -0.4],
                        [0.7, -0.1, 0.2, -0.5, 0.4],
                        [-0.3, 0.8, 0.6, -0.2, 0.1],
                        [0.5, -0.6, 0.1, 0.3, 0.9],
                        [-0.7, 0.2, 0.4, 0.8, -0.1],
                    ] * 2
                )
                + [[0.2, 0.6, -0.5, 0.1, -0.3]],
                torch.float64,
            )
        )
    token_values = model.input_embedding.token_embed.detach().clone()
    position_values = model.input_embedding.pos_embed.detach().clone()
    observed_inputs = []
    hook = model.blocks[0].normalize_input.register_forward_pre_hook(
        lambda module, arguments: observed_inputs.append(arguments[0].detach().clone())
    )
    prefix = torch.tensor([1, 2], dtype=torch.long, device=TEST_DEVICE)
    first_new = torch.tensor([3], dtype=torch.long, device=TEST_DEVICE)
    second_new = torch.tensor([4], dtype=torch.long, device=TEST_DEVICE)
    try:
        with torch.no_grad():
            _, prefix_cache = model.prefill(prefix)
            prefix_snapshot = _clone_tree(prefix_cache)
            _, first_cache = model.decode(first_new, prefix_cache)
            first_snapshot = _clone_tree(first_cache)
            _, second_cache = model.decode(second_new, first_cache)
    finally:
        hook.remove()

    expected_prefill = token_values[prefix] + position_values[:2]
    expected_first = (token_values[first_new] + position_values[2:3])
    expected_second = (token_values[second_new] + position_values[3:4])
    assert len(observed_inputs) == 3
    _assert_close(observed_inputs[0], expected_prefill)
    _assert_close(observed_inputs[1], expected_first)
    _assert_close(observed_inputs[2], expected_second)
    assert not torch.allclose(
        positions[3] - positions[2],
        (positions[3] - positions[2])[0].expand_as(positions[3]),
    )
    _assert_model_cache_shapes(prefix_snapshot, dimensions, 2, torch.float64)
    _assert_model_cache_shapes(first_snapshot, dimensions, 3, torch.float64)
    _assert_model_cache_shapes(second_cache, dimensions, 4, torch.float64)


def test_CDLM05_cached_stack_matches_semantically_aligned_uncached_model():
    dimensions = {**_DIMS, "max_seq_len": 7}
    cached = CachedDecoderOnlyLM(**dimensions, eps=_EPS).to(
        device=TEST_DEVICE, dtype=torch.float64
    ).eval()
    uncached = DecoderOnlyLM(**dimensions, eps=_EPS).to(
        device=TEST_DEVICE, dtype=torch.float64
    ).eval()
    _set_deterministic_parameters(cached)
    copied_roles = _semantic_model_pairs(cached, uncached)
    _copy_semantic_model_parameters(cached, uncached)
    assert copied_roles

    prefix = torch.tensor([2, 5, 1], dtype=torch.long, device=TEST_DEVICE)
    first_new = torch.tensor([8], dtype=torch.long, device=TEST_DEVICE)
    second_new = torch.tensor([3], dtype=torch.long, device=TEST_DEVICE)
    with torch.no_grad():
        expected_prefill = uncached(prefix).detach().clone()
        actual_prefill, caches = cached.prefill(prefix)
        _assert_close(actual_prefill, expected_prefill)
        _assert_model_cache_shapes(caches, dimensions, 3, torch.float64)
        caches_before_first_decode = _clone_tree(caches)

        first_prefix = torch.cat((prefix, first_new))
        expected_first = uncached(first_prefix).detach().clone()[-1:, :]
        first_logits, first_caches = cached.decode(first_new, caches)
        _assert_close(first_logits, expected_first)
        _assert_model_cache_shapes(first_caches, dimensions, 4, torch.float64)
        _assert_tree_properties(first_logits, torch.float64)
        first_caches_before_second_decode = _clone_tree(first_caches)

        second_prefix = torch.cat((first_prefix, second_new))
        expected_second = uncached(second_prefix).detach().clone()[-1:, :]
        second_logits, second_caches = cached.decode(second_new, first_caches)
        _assert_close(second_logits, expected_second)
        _assert_model_cache_shapes(second_caches, dimensions, 5, torch.float64)
        _assert_tree_properties(second_logits, torch.float64)

    _assert_model_cache_shapes(caches_before_first_decode, dimensions, 3, torch.float64)
    _assert_model_cache_shapes(first_caches_before_second_decode, dimensions, 4, torch.float64)


def test_CDLM06_interleaved_prefixes_follow_their_supplied_caches():
    dimensions = {**_DIMS, "num_layers": 1, "max_seq_len": 6}
    interleaved = CachedDecoderOnlyLM(**dimensions, eps=_EPS).to(
        device=TEST_DEVICE, dtype=torch.float64
    ).eval()
    isolated_a = CachedDecoderOnlyLM(**dimensions, eps=_EPS).to(
        device=TEST_DEVICE, dtype=torch.float64
    ).eval()
    isolated_b = CachedDecoderOnlyLM(**dimensions, eps=_EPS).to(
        device=TEST_DEVICE, dtype=torch.float64
    ).eval()
    _set_deterministic_parameters(interleaved)
    _copy_cached_parameters(interleaved, isolated_a)
    _copy_cached_parameters(interleaved, isolated_b)
    prefix_a = torch.tensor([1, 4], dtype=torch.long, device=TEST_DEVICE)
    prefix_b = torch.tensor([7, 2], dtype=torch.long, device=TEST_DEVICE)
    new_a = torch.tensor([5], dtype=torch.long, device=TEST_DEVICE)
    new_b = torch.tensor([9], dtype=torch.long, device=TEST_DEVICE)
    with torch.no_grad():
        isolated_a_prefill, isolated_a_cache = isolated_a.prefill(prefix_a)
        isolated_a_logits, isolated_a_updated = isolated_a.decode(new_a, isolated_a_cache)
        expected_a = isolated_a_logits.detach().clone()
        expected_a_cache = _clone_tree(isolated_a_updated)

        isolated_b_prefill, isolated_b_cache = isolated_b.prefill(prefix_b)
        isolated_b_logits, isolated_b_updated = isolated_b.decode(new_b, isolated_b_cache)
        expected_b = isolated_b_logits.detach().clone()
        expected_b_cache = _clone_tree(isolated_b_updated)

        actual_a_prefill, cache_a = interleaved.prefill(prefix_a)
        actual_b_prefill, cache_b = interleaved.prefill(prefix_b)
        cache_a_snapshot = _clone_tree(cache_a)
        cache_b_snapshot = _clone_tree(cache_b)
        actual_a, updated_a = interleaved.decode(new_a, cache_a_snapshot)
        actual_b, updated_b = interleaved.decode(new_b, cache_b_snapshot)

    _assert_close(actual_a_prefill, isolated_a_prefill.detach().clone())
    _assert_close(actual_b_prefill, isolated_b_prefill.detach().clone())
    assert not torch.allclose(expected_a, expected_b, **_TOLERANCES[torch.float64])
    _assert_close(actual_a, expected_a)
    _assert_close(actual_b, expected_b)
    _assert_tree_equal(updated_a, expected_a_cache)
    _assert_tree_equal(updated_b, expected_b_cache)
    _assert_model_cache_shapes(cache_a, dimensions, 2, torch.float64)
    _assert_model_cache_shapes(cache_b, dimensions, 2, torch.float64)
    _assert_model_cache_shapes(updated_a, dimensions, 3, torch.float64)
    _assert_model_cache_shapes(updated_b, dimensions, 3, torch.float64)


@pytest.mark.parametrize("scenario", ("singleton", "full", "last_decode"))
def test_CDLM07_valid_position_capacity_boundaries(scenario):
    dimensions = {**_DIMS, "max_seq_len": 5}
    model = _new_model(dtype=torch.float64, **dimensions)
    _set_deterministic_parameters(model)
    positions = model.input_embedding.pos_embed.detach().clone()
    token_values = model.input_embedding.token_embed.detach().clone()
    observed_inputs = []
    hook = model.blocks[0].normalize_input.register_forward_pre_hook(
        lambda module, arguments: observed_inputs.append(arguments[0].detach().clone())
    )
    try:
        with torch.no_grad():
            if scenario == "singleton":
                ids = torch.tensor([3], dtype=torch.long, device=TEST_DEVICE)
                logits, caches = model.prefill(ids)
                _assert_close(observed_inputs[0], token_values[ids] + positions[:1])
                assert logits.shape == (1, dimensions["vocab_size"])
                _assert_model_cache_shapes(caches, dimensions, 1, torch.float64)
            elif scenario == "full":
                ids = torch.tensor([0, 2, 4, 6, 8], dtype=torch.long, device=TEST_DEVICE)
                logits, caches = model.prefill(ids)
                assert logits.shape == (dimensions["max_seq_len"], dimensions["vocab_size"])
                _assert_model_cache_shapes(caches, dimensions, dimensions["max_seq_len"], torch.float64)
                _assert_close(observed_inputs[0][-1:], token_values[ids[-1:]] + positions[-1:])
            else:
                ids = torch.tensor([1, 2, 4, 6], dtype=torch.long, device=TEST_DEVICE)
                new_id = torch.tensor([8], dtype=torch.long, device=TEST_DEVICE)
                prefill_logits, caches = model.prefill(ids)
                prefix_snapshot = _clone_tree(caches)
                decode_logits, updated = model.decode(new_id, caches)
                assert prefill_logits.shape == (dimensions["max_seq_len"] - 1, dimensions["vocab_size"])
                assert decode_logits.shape == (1, dimensions["vocab_size"])
                _assert_model_cache_shapes(prefix_snapshot, dimensions, dimensions["max_seq_len"] - 1, torch.float64)
                _assert_model_cache_shapes(updated, dimensions, dimensions["max_seq_len"], torch.float64)
                _assert_close(observed_inputs[1], token_values[new_id] + positions[-1:])
    finally:
        hook.remove()
    assert len(observed_inputs) == (2 if scenario == "last_decode" else 1)
    _assert_tree_properties(logits if scenario != "last_decode" else decode_logits, torch.float64)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64), ids=("float32", "float64"))
def test_CDLM08_logits_and_full_cache_preserve_dtype_device_and_finiteness(dtype):
    model = _new_model(dtype=dtype)
    _set_deterministic_parameters(model)
    prefix = torch.tensor([3, 1], dtype=torch.long, device=TEST_DEVICE)
    new_id = torch.tensor([6], dtype=torch.long, device=TEST_DEVICE)
    assert prefix.dtype == torch.long and prefix.device == TEST_DEVICE
    assert new_id.dtype == torch.long and new_id.device == TEST_DEVICE
    with torch.no_grad():
        prefill_logits, caches = model.prefill(prefix)
        _assert_model_cache_shapes(caches, _DIMS, 2, dtype)
        _assert_tree_properties(prefill_logits, dtype)
        _assert_tree_properties(caches, dtype)
        decode_logits, updated = model.decode(new_id, caches)

    assert prefill_logits.shape == (2, _DIMS["vocab_size"])
    assert decode_logits.shape == (1, _DIMS["vocab_size"])
    _assert_tree_properties(decode_logits, dtype)
    _assert_model_cache_shapes(updated, _DIMS, 3, dtype)
    _assert_tree_properties(updated, dtype)
