"""Kiểm thử vòng lặp sinh token với KV cache theo hợp đồng công khai."""

from functools import partial

import torch
from torch import nn

from llm_foundations.decoding import greedy_token, sampling
from llm_foundations.decoder_only_lm import DecoderOnlyLM
from llm_foundations.generation import generate as uncached_generate
from llm_foundations.kv_cache.decoder_only_lm import CachedDecoderOnlyLM
from llm_foundations.kv_cache.generation import generation


TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Selected test device: {TEST_DEVICE}")


class _EmbeddingHolder(nn.Module):
    """Cung cấp bảng vị trí để fake model có cùng bề mặt capacity."""

    def __init__(self, capacity, vocab_size, dtype):
        super().__init__()
        self.token_embed = nn.Parameter(torch.zeros(vocab_size, 1, dtype=dtype, device=TEST_DEVICE))
        self.pos_embed = nn.Parameter(torch.zeros(capacity, 1, dtype=dtype, device=TEST_DEVICE))


class _RecordingModel(nn.Module):
    """Ghi lại prefill/decode và trả logits, cache được lập trình sẵn."""

    def __init__(self, capacity, prefill_logits, decode_logits=(), fail_decode=False, dtype=torch.float32):
        super().__init__()
        self.prefill_logits = torch.tensor(prefill_logits, dtype=dtype, device=TEST_DEVICE)
        self.vocab_size = self.prefill_logits.shape[-1]
        self.input_embedding = _EmbeddingHolder(capacity, self.vocab_size, dtype)
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=dtype, device=TEST_DEVICE))
        self.decode_logits = [
            torch.tensor(row, dtype=dtype, device=TEST_DEVICE).reshape(1, -1)
            for row in decode_logits
        ]
        assert self.prefill_logits.ndim == 2
        assert all(logits.shape == (1, self.vocab_size) for logits in self.decode_logits)
        self.fail_decode = fail_decode
        self.events = []
        self.prefill_count = 0
        self.decode_count = 0
        self.cache_inputs = []
        self.cache_returns = []
        self.observations = []
        assert all(parameter.device == TEST_DEVICE and parameter.dtype == dtype for parameter in self.parameters())

    def _validate_ids(self, token_ids, expected_length):
        assert token_ids.dtype == torch.long and token_ids.ndim == 1
        assert token_ids.device == TEST_DEVICE
        assert token_ids.numel() == expected_length
        assert token_ids.numel() == 0 or (token_ids.min() >= 0 and token_ids.max() < self.input_embedding.token_embed.shape[0])

    def forward(self, *args, **kwargs):
        raise AssertionError("Cached generation không được gọi forward")

    def _observe(self, name):
        self.observations.append(
            (name, torch.is_grad_enabled(), tuple(module.training for module in self.modules()))
        )

    def prefill(self, token_ids):
        self._validate_ids(token_ids, token_ids.numel())
        assert token_ids.numel() > 0
        assert token_ids.numel() <= self.input_embedding.pos_embed.shape[0]
        assert self.prefill_logits.shape == (token_ids.numel(), self.vocab_size)
        self.events.append(("prefill", token_ids.detach().clone()))
        self.prefill_count += 1
        self._observe("prefill")
        # Tạo cache payload độc lập với K/V feature widths khác nhau
        cache = [[(
            torch.full((token_ids.numel(), 2), 10.0, dtype=self.prefill_logits.dtype, device=TEST_DEVICE),
            torch.full((token_ids.numel(), 3), 20.0, dtype=self.prefill_logits.dtype, device=TEST_DEVICE),
        )]]
        self.cache_returns.append(_clone_tree(cache))
        return self.prefill_logits + self.anchor * 0, cache

    def decode(self, token_ids_new, caches):
        self._validate_ids(token_ids_new, 1)
        assert token_ids_new.max() < self.input_embedding.token_embed.shape[0]
        assert self.cache_returns or self.decode_count == 0
        self.events.append(("decode", token_ids_new.detach().clone()))
        self.decode_count += 1
        self._observe("decode")
        self.cache_inputs.append(_clone_tree(caches))
        if self.fail_decode:
            raise self.fail_decode
        cache_len = caches[0][0][0].shape[0] + 1
        version = self.decode_count
        updated = [[(
            torch.full((cache_len, 2), 10.0 + version, dtype=self.prefill_logits.dtype, device=TEST_DEVICE),
            torch.full((cache_len, 3), 20.0 + version, dtype=self.prefill_logits.dtype, device=TEST_DEVICE),
        )]]
        self.cache_returns.append(_clone_tree(updated))
        return self.decode_logits[self.decode_count - 1] + self.anchor * 0, updated


class _SentinelError(Exception):
    pass


def _clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return value


def _ids(values):
    return torch.tensor(values, dtype=torch.long, device=TEST_DEVICE)


def _scalar_token(model, token):
    selected = torch.tensor(token, dtype=torch.long, device=TEST_DEVICE)
    assert selected.ndim == 0 and selected.device == TEST_DEVICE
    assert 0 <= selected.item() < model.input_embedding.token_embed.shape[0]
    return selected


def _selector(model, token, records=None):
    def select(logits):
        assert logits.ndim == 1 and logits.shape[0] == model.vocab_size
        assert logits.device == TEST_DEVICE
        assert 0 <= token < model.input_embedding.token_embed.shape[0]
        model.events.append(("select", logits.detach().clone()))
        model._observe("selector")
        if records is not None:
            records.append(logits.detach().clone())
        return _scalar_token(model, token)
    return select


def _assert_ids(actual, expected):
    assert actual.dtype == torch.long
    assert actual.device == TEST_DEVICE
    assert actual.ndim == 1
    assert actual.tolist() == expected


def _assert_cache(actual, length, key_value, val_value, dtype=torch.float32):
    assert isinstance(actual, list) and len(actual) == 1
    assert isinstance(actual[0], list) and len(actual[0]) == 1
    assert isinstance(actual[0][0], tuple) and len(actual[0][0]) == 2
    key, value = actual[0][0]
    assert key.shape == (length, 2)
    assert value.shape == (length, 3)
    for tensor in (key, value):
        assert tensor.dtype == dtype and torch.is_floating_point(tensor)
        assert tensor.device == TEST_DEVICE
    assert torch.equal(key, torch.full_like(key, key_value))
    assert torch.equal(value, torch.full_like(value, val_value))


def test_CG01_zero_new_token_budget_does_no_work():
    prompt = _ids([2, 1])
    model = _RecordingModel(7, [[0, 1, 2], [1, 0, 2]])
    model._validate_ids(prompt, 2)
    selectors = []
    result = generation(model, prompt, max_new_tokens=0, selector=lambda logits: selectors.append(logits))
    _assert_ids(result, [2, 1])
    assert prompt.tolist() == [2, 1]
    assert model.events == []
    assert (model.prefill_count, model.decode_count, selectors) == (0, 0, [])


def test_CG02_full_prompt_does_no_work_even_with_positive_budget():
    prompt = _ids([2, 1])
    model = _RecordingModel(2, [[0, 1, 2], [1, 0, 2]])
    model._validate_ids(prompt, 2)
    selectors = []
    result = generation(model, prompt, max_new_tokens=3, selector=lambda logits: selectors.append(logits))
    _assert_ids(result, [2, 1])
    assert model.events == []
    assert (model.prefill_count, model.decode_count, selectors) == (0, 0, [])


def test_CG03_one_selection_appends_the_selector_scalar():
    prompt = _ids([2, 1])
    model = _RecordingModel(6, [[0, 1, 9, 2], [1, 9, 2, 3]])
    selected_id = 3
    assert model.prefill_logits[-1].argmax().item() != selected_id
    selector_calls = []
    result = generation(model, prompt, max_new_tokens=1, selector=_selector(model, selected_id, selector_calls))
    _assert_ids(result, [2, 1, selected_id])
    assert result[:2].tolist() == prompt.tolist()
    assert model.prefill_count == 1 and model.decode_count == 0
    assert len(selector_calls) == 1 and selector_calls[0].shape == (4,)
    assert [event[0] for event in model.events] == ["prefill", "select"]
    assert torch.equal(model.events[0][1], prompt)
    assert prompt.tolist() == [2, 1]


def test_CG04_threads_latest_cache_and_stops_at_new_token_budget():
    prompt = _ids([2, 1])
    model = _RecordingModel(7, [[0, 1, 2, 0, -1], [1, 0, 2, -1, -2]], [[2, 0, 1, -1, 0], [0, 1, 2, 3, -1]])
    tokens = iter([4, 0, 3])
    result = generation(model, prompt, max_new_tokens=3, selector=lambda logits: _selector(model, next(tokens))(logits))
    _assert_ids(result, [2, 1, 4, 0, 3])
    assert [event[0] for event in model.events] == ["prefill", "select", "decode", "select", "decode", "select"]
    assert torch.equal(model.events[0][1], prompt)
    assert prompt.tolist() == [2, 1]
    assert [event[1].tolist() for event in model.events if event[0] == "decode"] == [[4], [0]]
    assert len(model.cache_inputs) == 2
    _assert_cache(model.cache_inputs[0], 2, 10.0, 20.0)
    _assert_cache(model.cache_inputs[1], 3, 11.0, 21.0)
    _assert_cache(model.cache_returns[-1], 4, 12.0, 22.0)
    assert model.decode_count == 2


def test_CG05_capacity_stops_before_decoding_final_token():
    prompt = _ids([2, 1])
    model = _RecordingModel(4, [[0, 1, 2, -1, -2], [1, 0, 2, -2, -3]], [[2, 0, 1, 3, -1]])
    tokens = iter([4, 3])
    result = generation(model, prompt, max_new_tokens=5, selector=lambda logits: _selector(model, next(tokens))(logits))
    _assert_ids(result, [2, 1, 4, 3])
    assert [event[0] for event in model.events] == ["prefill", "select", "decode", "select"]
    assert torch.equal(model.events[0][1], prompt)
    assert prompt.tolist() == [2, 1]
    assert [event[1].tolist() for event in model.events if event[0] == "decode"] == [[4]]
    _assert_cache(model.cache_returns[-1], 3, 11.0, 21.0)


def test_CG06_selector_receives_raw_logits_without_dtype_or_device_change():
    prompt = _ids([1, 2])
    model = _RecordingModel(6, [[9, 8, 7, 6, 5], [-3, 7, 2, -1, 0]], [[4, -2, 1, 0, 3]], dtype=torch.float64)
    seen = []
    chosen = iter([4, 2])

    def selector(logits):
        assert logits.shape == (model.vocab_size,) and logits.device == TEST_DEVICE
        model.events.append(("select", logits.detach().clone()))
        model._observe("selector")
        seen.append(logits.detach().clone())
        return _scalar_token(model, next(chosen))

    result = generation(model, prompt, max_new_tokens=2, selector=selector)
    _assert_ids(result, [1, 2, 4, 2])
    expected = [torch.tensor(row, dtype=torch.float64, device=TEST_DEVICE) for row in ([ -3, 7, 2, -1, 0], [4, -2, 1, 0, 3])]
    assert len(seen) == 2
    for actual, wanted in zip(seen, expected):
        assert actual.shape == (5,) and actual.dtype == torch.float64 and actual.device == TEST_DEVICE
        assert torch.equal(actual, wanted)
    assert model.events[2][1].dtype == torch.long and model.events[2][1].device == TEST_DEVICE


def test_CG07_existing_greedy_default_is_used():
    prompt = _ids([1])
    model = _RecordingModel(3, [[0, -1, 2, 7, 3]])
    result = generation(model, prompt, max_new_tokens=1)
    _assert_ids(result, [1, 3])
    assert generation.__kwdefaults__["selector"] is greedy_token
    assert model.prefill_count == 1 and model.decode_count == 0


def test_CG08_bound_existing_sampling_works_with_generator():
    prompt = _ids([1])
    model = _RecordingModel(4, [[0, 1, -1, 7, 2]], [[0, 1, 6, 2, -1]])
    generator = torch.Generator(device=TEST_DEVICE).manual_seed(1729)
    selector = partial(sampling, temperature=0.8, top_k=1, generator=generator)
    result = generation(model, prompt, max_new_tokens=2, selector=selector)
    _assert_ids(result, [1, 3, 2])
    assert [event[1].tolist() for event in model.events if event[0] == "decode"] == [[3]]
    assert generator.device == TEST_DEVICE


def _mixed_training_modes(model):
    model.train()
    model.input_embedding.eval()
    model.input_embedding.pos_embed.requires_grad_(True)
    return tuple((module, module.training) for module in model.modules())


def _assert_modes(snapshot):
    assert all(module.training == was_training for module, was_training in snapshot)


def test_CG09_inference_disables_grad_and_restores_mixed_modes():
    prompt = _ids([1])
    model = _RecordingModel(5, [[0, 1, 2, -1, -2]], [[2, 1, 0, -1, -2]])
    modes = _mixed_training_modes(model)
    selector_observations = []

    def selector(logits):
        selector_observations.append((torch.is_grad_enabled(), tuple(module.training for module in model.modules()), logits.requires_grad))
        return _scalar_token(model, 2)

    with torch.enable_grad():
        assert torch.is_grad_enabled()
        generation(model, prompt, max_new_tokens=2, selector=selector)
        assert torch.is_grad_enabled()
    expected_flags = tuple(False for _ in model.modules())
    assert [(name, grad, flags) for name, grad, flags in model.observations] == [
        ("prefill", False, expected_flags),
        ("decode", False, expected_flags),
    ]
    assert selector_observations == [
        (False, expected_flags, False),
        (False, expected_flags, False),
    ]
    _assert_modes(modes)


def test_CG10_selector_exception_restores_modes_and_propagates():
    model = _RecordingModel(5, [[0, 1, 2]])
    modes = _mixed_training_modes(model)
    failure = _SentinelError("selector failed")
    selector_grad_flags = []

    def selector(logits):
        model.events.append(("select", logits.detach().clone()))
        model._observe("selector")
        selector_grad_flags.append(logits.requires_grad)
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        raise failure

    with torch.enable_grad():
        try:
            generation(model, _ids([1]), max_new_tokens=2, selector=selector)
        except _SentinelError as error:
            assert error is failure
        else:
            raise AssertionError("Selector exception was not propagated")
        assert torch.is_grad_enabled()
    assert [event[0] for event in model.events] == ["prefill", "select"]
    expected_flags = tuple(False for _ in model.modules())
    assert model.observations == [
        ("prefill", False, expected_flags),
        ("selector", False, expected_flags),
    ]
    assert selector_grad_flags == [False]
    assert model.prefill_count == 1 and model.decode_count == 0
    _assert_modes(modes)


def test_CG10_decode_exception_restores_modes_and_propagates():
    failure = _SentinelError("decode failed")
    model = _RecordingModel(5, [[0, 1, 2, -1, -2]], [[2, 1, 0, -1, -2]], fail_decode=failure)
    modes = _mixed_training_modes(model)
    selector_grad_flags = []

    def selector(logits):
        model.events.append(("select", logits.detach().clone()))
        model._observe("selector")
        selector_grad_flags.append(logits.requires_grad)
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in model.modules())
        return _scalar_token(model, 2)

    with torch.enable_grad():
        try:
            generation(model, _ids([1]), max_new_tokens=3, selector=selector)
        except _SentinelError as error:
            assert error is failure
        else:
            raise AssertionError("Decode exception was not propagated")
        assert torch.is_grad_enabled()
    assert [event[0] for event in model.events] == ["prefill", "select", "decode"]
    expected_flags = tuple(False for _ in model.modules())
    assert model.observations == [
        ("prefill", False, expected_flags),
        ("selector", False, expected_flags),
        ("decode", False, expected_flags),
    ]
    assert selector_grad_flags == [False]
    assert model.prefill_count == 1 and model.decode_count == 1
    _assert_modes(modes)


def _semantic_parameter_pairs(cached, uncached):
    pairs = [
        ("input_embedding.token_embed", cached.input_embedding.token_embed, uncached.input_embedding.token_embed),
        ("input_embedding.pos_embed", cached.input_embedding.pos_embed, uncached.input_embedding.pos_embed),
    ]
    for index, (cb, ub) in enumerate(zip(cached.blocks, uncached.blocks)):
        prefix = f"blocks.{index}"
        pairs += [
            (f"{prefix}.normalize_input.gamma", cb.normalize_input.gamma, ub.attention_residual.normalized_input.gamma),
            (f"{prefix}.normalize_input.beta", cb.normalize_input.beta, ub.attention_residual.normalized_input.beta),
            (f"{prefix}.ffn_norm.gamma", cb.ffn_norm.gamma, ub.ffn_norm.gamma),
            (f"{prefix}.ffn_norm.beta", cb.ffn_norm.beta, ub.ffn_norm.beta),
            (f"{prefix}.cached_mha.w_o", cb.cached_mha.w_o, ub.attention_residual.multi_head_attention.w_o),
            (f"{prefix}.ffn.w_1", cb.ffn.w_1, ub.ffn.w_1),
            (f"{prefix}.ffn.b_1", cb.ffn.b_1, ub.ffn.b_1),
            (f"{prefix}.ffn.w_2", cb.ffn.w_2, ub.ffn.w_2),
            (f"{prefix}.ffn.b_2", cb.ffn.b_2, ub.ffn.b_2),
        ]
        for head_index, (ch, uh) in enumerate(zip(cb.cached_mha.cached_heads, ub.attention_residual.multi_head_attention.heads)):
            for projection in ("w_q", "w_k", "w_v"):
                pairs.append((f"{prefix}.heads.{head_index}.{projection}", getattr(ch.attention_head, projection), getattr(uh, projection)))
    pairs.extend([
        ("final_norm.gamma", cached.final_norm.gamma, uncached.final_norm.gamma),
        ("final_norm.beta", cached.final_norm.beta, uncached.final_norm.beta),
        ("lm_head.w_lmhead", cached.lm_head.w_lmhead, uncached.lm_head.w_lmhead),
    ])
    return pairs


def _initialize_semantic_weights(model):
    with torch.no_grad():
        for role, (name, parameter) in enumerate(model.named_parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=torch.float64, device=TEST_DEVICE).reshape_as(parameter)
            phase = values + role
            if name.endswith(".gamma"):
                parameter.copy_(1 + 0.03 * torch.sin(phase))
            elif name.endswith(".beta"):
                parameter.copy_(0.02 * torch.sin(phase))
            else:
                parameter.copy_(0.08 * torch.sin(phase))


def _capture_selector(delegate, records):
    def select(logits):
        records.append(logits.detach().clone())
        return delegate(logits)
    return select


def test_CG11_real_cached_generation_matches_semantically_mapped_uncached_model():
    dims = dict(vocab_size=11, max_seq_len=7, d_model=5, num_layers=2, num_heads=2, d_k=3, d_v=2, d_ff=7)
    eps = 3e-4
    cached = CachedDecoderOnlyLM(**dims, eps=eps).to(device=TEST_DEVICE, dtype=torch.float64).eval()
    uncached = DecoderOnlyLM(**dims, eps=eps).to(device=TEST_DEVICE, dtype=torch.float64).eval()
    _initialize_semantic_weights(cached)
    assert cached.input_embedding.token_embed.shape == uncached.input_embedding.token_embed.shape == (dims["vocab_size"], dims["d_model"])
    assert cached.input_embedding.pos_embed.shape == uncached.input_embedding.pos_embed.shape == (dims["max_seq_len"], dims["d_model"])
    assert len(cached.blocks) == len(uncached.blocks) == dims["num_layers"]
    assert cached.lm_head.w_lmhead.shape == uncached.lm_head.w_lmhead.shape == (dims["d_model"], dims["vocab_size"])
    for cached_block, uncached_block in zip(cached.blocks, uncached.blocks):
        cached_attention = cached_block.cached_mha
        uncached_attention = uncached_block.attention_residual.multi_head_attention
        assert len(cached_attention.cached_heads) == len(uncached_attention.heads) == dims["num_heads"]
        assert cached_attention.w_o.shape == uncached_attention.w_o.shape == (dims["num_heads"] * dims["d_v"], dims["d_model"])
        for norm in (cached_block.normalize_input, cached_block.ffn_norm, uncached_block.attention_residual.normalized_input, uncached_block.ffn_norm):
            assert norm.gamma.shape == norm.beta.shape == (dims["d_model"],)
        assert cached_block.ffn.w_1.shape == uncached_block.ffn.w_1.shape == (dims["d_model"], dims["d_ff"])
        assert cached_block.ffn.b_1.shape == uncached_block.ffn.b_1.shape == (dims["d_ff"],)
        assert cached_block.ffn.w_2.shape == uncached_block.ffn.w_2.shape == (dims["d_ff"], dims["d_model"])
        assert cached_block.ffn.b_2.shape == uncached_block.ffn.b_2.shape == (dims["d_model"],)
        for cached_head, uncached_head in zip(cached_attention.cached_heads, uncached_attention.heads):
            cached_projection = cached_head.attention_head
            for head in (cached_projection, uncached_head):
                assert head.w_q.shape == head.w_k.shape == (dims["d_model"], dims["d_k"])
                assert head.w_v.shape == (dims["d_model"], dims["d_v"])
    pairs = _semantic_parameter_pairs(cached, uncached)
    cached_parameters = list(cached.parameters())
    uncached_parameters = list(uncached.parameters())
    assert len({id(source) for _, source, _ in pairs}) == len(cached_parameters)
    assert len({id(destination) for _, _, destination in pairs}) == len(uncached_parameters)
    assert {id(source) for _, source, _ in pairs} == {id(parameter) for parameter in cached_parameters}
    assert {id(destination) for _, _, destination in pairs} == {id(parameter) for parameter in uncached_parameters}
    with torch.no_grad():
        for _, source, destination in pairs:
            assert source is not destination and source.shape == destination.shape
            destination.copy_(source.detach().clone())
            assert torch.equal(source, destination)
    for cb, ub in zip(cached.blocks, uncached.blocks):
        assert cb.normalize_input.eps == ub.attention_residual.normalized_input.eps == eps
        assert cb.ffn_norm.eps == ub.ffn_norm.eps == eps
    assert cached.final_norm.eps == uncached.final_norm.eps == eps
    assert cached.final_norm.gamma.shape == cached.final_norm.beta.shape == (dims["d_model"],)
    assert uncached.final_norm.gamma.shape == uncached.final_norm.beta.shape == (dims["d_model"],)
    assert (cached.input_embedding.token_embed.shape, cached.input_embedding.pos_embed.shape) == ((11, 5), (7, 5))
    assert (uncached.input_embedding.token_embed.shape, uncached.input_embedding.pos_embed.shape) == ((11, 5), (7, 5))
    assert len(cached.blocks) == len(uncached.blocks) == 2

    prompt = _ids([1, 4])
    params_before = {id(parameter): parameter.detach().clone() for parameter in cached.parameters()}
    uncached_before = {id(parameter): parameter.detach().clone() for parameter in uncached.parameters()}
    position_shape = cached.input_embedding.pos_embed.shape
    uncached_records, cached_records = [], []
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        with torch.autocast(device_type=TEST_DEVICE.type, enabled=False):
            uncached_result = uncached_generate(uncached, prompt.clone(), max_new_tokens=3, selector=_capture_selector(greedy_token, uncached_records))
            cached_result = generation(cached, prompt.clone(), max_new_tokens=3, selector=_capture_selector(greedy_token, cached_records))
    finally:
        torch.set_float32_matmul_precision(old_precision)
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    _assert_ids(uncached_result, uncached_result.tolist())
    _assert_ids(cached_result, uncached_result.tolist())
    assert uncached_result.shape == cached_result.shape == (5,)
    assert uncached_result[:2].tolist() == cached_result[:2].tolist() == [1, 4]
    assert len(uncached_records) == len(cached_records) == 3
    for uncached_logits, cached_logits in zip(uncached_records, cached_records):
        assert uncached_logits.shape == cached_logits.shape == (11,)
        assert uncached_logits.dtype == cached_logits.dtype == torch.float64
        assert uncached_logits.device == cached_logits.device == TEST_DEVICE
        assert torch.isfinite(uncached_logits).all().item() and torch.isfinite(cached_logits).all().item()
        torch.testing.assert_close(cached_logits, uncached_logits, rtol=1e-9, atol=1e-10)
        top_two = uncached_logits.topk(2).values
        assert (top_two[0] - top_two[1]).item() > 1e-7
    for parameter in cached.parameters():
        assert torch.equal(parameter, params_before[id(parameter)])
    for parameter in uncached.parameters():
        assert torch.equal(parameter, uncached_before[id(parameter)])
    assert cached.input_embedding.pos_embed.shape == uncached.input_embedding.pos_embed.shape == position_shape
