"""Kiểm tra vòng sinh tự hồi quy và các ranh giới inference của generation."""

from functools import partial

import pytest
import torch
from torch import nn

from llm_foundations.decoder_only_lm import DecoderOnlyLM
from llm_foundations.decoding import greedy_token, sampling
from llm_foundations.generation import generate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Autoregressive generation tests selected device: {DEVICE}")


class _FakeInputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, capacity: int, device: torch.device) -> None:
        super().__init__()
        self.token_embed = nn.Parameter(torch.zeros(vocab_size, 3, device=device))
        self.pos_embed = nn.Parameter(torch.zeros(capacity, 3, device=device))


class _RecordingModel(nn.Module):
    def __init__(
        self,
        capacity: int,
        vocab_size: int = 6,
        *,
        output_rows: list[list[float]] | None = None,
    ) -> None:
        super().__init__()
        self.input_embedding = _FakeInputEmbedding(vocab_size, capacity, DEVICE)
        self.child = nn.Identity()
        self.logit_anchor = nn.Parameter(torch.ones((), device=DEVICE))
        self.register_buffer("state_marker", torch.tensor([3.0, -2.0], device=DEVICE))
        self.vocab_size = vocab_size
        self.output_rows = output_rows
        self.forward_inputs: list[torch.Tensor] = []
        self.forward_modes: list[dict[str, bool]] = []
        self.forward_grad_states: list[bool] = []
        self.output_autograd_states: list[tuple[bool, torch.Tensor | None]] = []
        self.fail_on_forward = False

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.fail_on_forward:
            raise AssertionError("model must not be called")
        self.forward_inputs.append(token_ids.detach().clone())
        self.forward_modes.append(_module_modes(self))
        self.forward_grad_states.append(torch.is_grad_enabled())
        if self.output_rows is None:
            values = torch.arange(
                token_ids.shape[0] * self.vocab_size,
                dtype=torch.float32,
                device=DEVICE,
            ).reshape(token_ids.shape[0], self.vocab_size)
            values = values + 100.0 * (len(self.forward_inputs) - 1)
        else:
            values = torch.tensor(self.output_rows, dtype=torch.float32, device=DEVICE)
            assert values.shape == (token_ids.shape[0], self.vocab_size)
        logits = values + self.logit_anchor * 0.0
        self.output_autograd_states.append((logits.requires_grad, logits.grad_fn))
        return logits


class _RecordingSelector:
    def __init__(
        self,
        model: _RecordingModel,
        selected_ids: list[int],
        *,
        error: RuntimeError | None = None,
    ) -> None:
        self.model = model
        self.selected_ids = list(selected_ids)
        self.error = error
        self.inputs: list[torch.Tensor] = []
        self.results: list[torch.Tensor] = []
        self.module_modes: list[dict[str, bool]] = []
        self.grad_states: list[bool] = []
        self.input_autograd_states: list[tuple[bool, torch.Tensor | None]] = []

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.inputs.append(logits.detach().clone())
        self.module_modes.append(_module_modes(self.model))
        self.grad_states.append(torch.is_grad_enabled())
        self.input_autograd_states.append((logits.requires_grad, logits.grad_fn))
        if self.error is not None:
            raise self.error
        index = len(self.results)
        if index >= len(self.selected_ids):
            raise AssertionError("selector called more often than expected")
        token_id = torch.tensor(self.selected_ids[index], dtype=torch.long, device=logits.device)
        self.results.append(token_id.detach().clone())
        return token_id


class _FailingSelector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        raise AssertionError("selector must not be called")


def _module_modes(model: nn.Module) -> dict[str, bool]:
    return {name: module.training for name, module in model.named_modules()}


def _make_model(
    capacity: int,
    vocab_size: int = 6,
    *,
    output_rows: list[list[float]] | None = None,
) -> _RecordingModel:
    return _RecordingModel(capacity, vocab_size, output_rows=output_rows).to(DEVICE)


def _storage_id(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _state_snapshot(model: _RecordingModel) -> dict[str, dict[str, tuple[object, ...]]]:
    snapshot: dict[str, dict[str, tuple[object, ...]]] = {"parameters": {}, "buffers": {}}
    for name, tensor in (
        ("parameters", model.named_parameters()),
        ("buffers", model.named_buffers()),
    ):
        entries = snapshot[name]
        for tensor_name, value in tensor:
            entries[tensor_name] = (
                id(value),
                _storage_id(value),
                tuple(value.shape),
                value.dtype,
                value.device,
                value.detach().clone(),
            )
    return snapshot


def _assert_state_unchanged(model: _RecordingModel, snapshot: dict[str, dict[str, tuple[object, ...]]]) -> None:
    current = _state_snapshot(model)
    for group in ("parameters", "buffers"):
        assert current[group].keys() == snapshot[group].keys()
        for name, before in snapshot[group].items():
            after = current[group][name]
            assert after[:5] == before[:5]
            assert torch.equal(after[5], before[5])


def test_ag01_selector_receives_only_the_last_raw_logits_row() -> None:
    model = _make_model(
        5,
        5,
        output_rows=[[50.0, 0.0, 0.0, 0.0, 0.0], [7.0, -2.0, 4.0, 1.0, 0.0]],
    )
    selector = _RecordingSelector(model, [2])
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    generate(model, prompt, max_new_tokens=1, selector=selector)

    assert len(selector.inputs) == 1
    assert selector.inputs[0].shape == (5,)
    assert torch.equal(selector.inputs[0], torch.tensor([7.0, -2.0, 4.0, 1.0, 0.0], device=DEVICE))


def test_ag02_appends_the_selector_original_scalar_id() -> None:
    model = _make_model(
        5,
        6,
        output_rows=[[0.0, 8.0, 0.0, 0.0, 0.0, 0.0], [9.0, 1.0, 0.0, -3.0, 2.0, -1.0]],
    )
    selector = _RecordingSelector(model, [4])
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=1, selector=selector)

    assert torch.equal(result, torch.tensor([1, 2, 4], dtype=torch.long, device=DEVICE))


def test_ag03_forwards_the_entire_growing_prefix_each_iteration() -> None:
    model = _make_model(8)
    selector = _RecordingSelector(model, [4, 0, 3])
    prompt = torch.tensor([2, 1], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=3, selector=selector)

    expected_inputs = ([2, 1], [2, 1, 4], [2, 1, 4, 0])
    assert [values.tolist() for values in model.forward_inputs] == list(expected_inputs)
    assert torch.equal(result, torch.tensor([2, 1, 4, 0, 3], dtype=torch.long, device=DEVICE))


def test_ag04_default_selector_uses_greedy_token() -> None:
    model = _make_model(4, 5, output_rows=[[-3.0, 1.0, 2.0, -1.0, 8.0]])
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=1)

    assert torch.equal(result, torch.tensor([1, 4], dtype=torch.long, device=DEVICE))
    assert len(model.forward_inputs) == 1


def test_ag05_configured_sampling_selector_compatibility() -> None:
    model = _make_model(4, 5, output_rows=[[-4.0, 0.0, 1.0, 9.0, -2.0]])
    generator = torch.Generator(device=DEVICE).manual_seed(1729)
    selector = partial(sampling, temperature=0.7, top_k=1, generator=generator)
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=1, selector=selector)

    assert torch.equal(result, torch.tensor([1, 3], dtype=torch.long, device=DEVICE))


def test_ag06_new_token_budget_stops_without_an_extra_forward() -> None:
    model = _make_model(7)
    selector = _RecordingSelector(model, [4, 3])
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=2, selector=selector)

    assert torch.equal(result, torch.tensor([1, 2, 4, 3], dtype=torch.long, device=DEVICE))
    assert [values.tolist() for values in model.forward_inputs] == [[1, 2], [1, 2, 4]]
    assert len(model.forward_inputs) == 2
    assert len(selector.inputs) == 2


def test_ag07_position_capacity_stops_with_the_prompt_included() -> None:
    model = _make_model(4)
    selector = _RecordingSelector(model, [4, 3])
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=5, selector=selector)

    assert torch.equal(result, torch.tensor([1, 2, 4, 3], dtype=torch.long, device=DEVICE))
    assert [values.tolist() for values in model.forward_inputs] == [[1, 2], [1, 2, 4]]
    assert len(model.forward_inputs) == 2
    assert len(selector.inputs) == 2


def test_ag08_zero_budget_returns_prompt_without_collaborator_calls() -> None:
    model = _make_model(5)
    model.fail_on_forward = True
    selector = _FailingSelector()
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=0, selector=selector)

    assert torch.equal(result, torch.tensor([1, 2], dtype=torch.long, device=DEVICE))
    assert model.forward_inputs == []
    assert selector.calls == 0


def test_ag09_full_capacity_prompt_returns_without_collaborator_calls() -> None:
    model = _make_model(2)
    model.fail_on_forward = True
    selector = _FailingSelector()
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=3, selector=selector)

    assert torch.equal(result, torch.tensor([1, 2], dtype=torch.long, device=DEVICE))
    assert model.forward_inputs == []
    assert selector.calls == 0


def test_ag10_output_and_collaborator_id_metadata() -> None:
    model = _make_model(4, 5)
    selector = _RecordingSelector(model, [0, 4])
    prompt = torch.tensor([2], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=2, selector=selector)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (3,)
    assert result.dtype == torch.long
    assert result.device == DEVICE
    assert all(values.dtype == torch.long and values.device == DEVICE for values in model.forward_inputs)
    assert all(token.shape == () and token.dtype == torch.long and token.device == DEVICE for token in selector.results)


@pytest.mark.parametrize(
    ("max_new_tokens", "selected_ids"),
    [(0, []), (1, [4])],
    ids=["no-generation", "append"],
)
def test_ag11_prompt_is_immutable_and_result_has_independent_storage(
    max_new_tokens: int, selected_ids: list[int]
) -> None:
    model = _make_model(5)
    selector = _RecordingSelector(model, selected_ids)
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)
    prompt_before = prompt.clone()

    result = generate(model, prompt, max_new_tokens=max_new_tokens, selector=selector)

    assert torch.equal(prompt, prompt_before)
    assert _storage_id(result) != _storage_id(prompt)
    result[0] = 5
    assert torch.equal(prompt, prompt_before)


@pytest.mark.parametrize("entry_mode", ["all-evaluation", "mixed"], ids=["all-eval", "mixed"])
def test_ag12_evaluation_mode_is_active_and_each_module_mode_is_restored(entry_mode: str) -> None:
    model = _make_model(3)
    if entry_mode == "all-evaluation":
        model.eval()
    else:
        model.train()
        model.child.eval()
    entry_modes = _module_modes(model)
    selector = _RecordingSelector(model, [2])
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)

    generate(model, prompt, max_new_tokens=1, selector=selector)

    assert model.forward_modes == [{name: False for name in entry_modes}]
    assert selector.module_modes == [{name: False for name in entry_modes}]
    assert _module_modes(model) == entry_modes


def test_ag13_collaborator_exception_restores_modes_and_autograd_state() -> None:
    model = _make_model(3)
    model.train()
    model.child.eval()
    entry_modes = _module_modes(model)
    sentinel = RuntimeError("selector sentinel")
    selector = _RecordingSelector(model, [], error=sentinel)
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)
    prompt_before = prompt.clone()

    with torch.enable_grad():
        with pytest.raises(RuntimeError) as raised:
            generate(model, prompt, max_new_tokens=1, selector=selector)
        assert raised.value is sentinel
        assert torch.is_grad_enabled()

    assert _module_modes(model) == entry_modes
    assert len(model.forward_inputs) == 1
    assert len(selector.inputs) == 1
    assert model.forward_modes == [{name: False for name in entry_modes}]
    assert selector.module_modes == [{name: False for name in entry_modes}]
    assert model.forward_grad_states == [False]
    assert selector.grad_states == [False]
    assert torch.equal(prompt, prompt_before)


@pytest.mark.parametrize("entry_grad_enabled", [True, False], ids=["grad-enabled", "grad-disabled"])
def test_ag14_calls_run_without_autograd_and_restore_entry_setting(entry_grad_enabled: bool) -> None:
    model = _make_model(3)
    selector = _RecordingSelector(model, [2])
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)

    with torch.set_grad_enabled(entry_grad_enabled):
        generate(model, prompt, max_new_tokens=1, selector=selector)
        assert torch.is_grad_enabled() is entry_grad_enabled
        assert model.forward_grad_states == [False]
        assert selector.grad_states == [False]
        assert model.output_autograd_states == [(False, None)]
        assert selector.input_autograd_states == [(False, None)]


def test_ag15_parameters_buffers_position_table_and_gradients_remain_fixed() -> None:
    model = _make_model(3, 5)
    model.logit_anchor.grad = torch.tensor(0.25, dtype=torch.float32, device=DEVICE)
    gradients_before = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }
    state_before = _state_snapshot(model)
    selector = _RecordingSelector(model, [2, 3])
    prompt = torch.tensor([1], dtype=torch.long, device=DEVICE)

    result = generate(model, prompt, max_new_tokens=2, selector=selector)

    _assert_state_unchanged(model, state_before)
    for name, parameter in model.named_parameters():
        expected = gradients_before[name]
        assert (parameter.grad is None) == (expected is None)
        if expected is not None:
            assert torch.equal(parameter.grad, expected)
    assert model.input_embedding.pos_embed.shape == (3, 3)
    assert torch.equal(result, torch.tensor([1, 2, 3], dtype=torch.long, device=DEVICE))


def test_ag16_real_decoder_only_model_composes_with_generation() -> None:
    config = {
        "vocab_size": 6,
        "max_seq_len": 4,
        "d_model": 4,
        "num_layers": 1,
        "num_heads": 1,
        "d_k": 2,
        "d_v": 2,
        "d_ff": 8,
        "eps": 1e-5,
    }
    cuda_devices = list(range(torch.cuda.device_count())) if DEVICE.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(1729)
        model = DecoderOnlyLM(**config).to(device=DEVICE, dtype=torch.float32)
    prompt = torch.tensor([1, 2], dtype=torch.long, device=DEVICE)
    forward_calls: list[None] = []
    handle = model.register_forward_hook(lambda module, inputs, output: forward_calls.append(None))
    try:
        result = generate(model, prompt, max_new_tokens=2)
    finally:
        handle.remove()

    assert result.shape == (4,)
    assert torch.equal(result[:2], torch.tensor([1, 2], dtype=torch.long, device=DEVICE))
    assert torch.all((result >= 0) & (result < 6))
    assert result.dtype == torch.long
    assert result.device == DEVICE
    assert len(forward_calls) == 2
