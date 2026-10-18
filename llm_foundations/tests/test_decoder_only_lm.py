"""Verify the decoder-only learner assembly against independent mathematical oracles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import llm_foundations.decoder_only_lm as decoder_module
from llm_foundations.decoder_block import PreNormDecoderBlock
from llm_foundations.embeddings.input_embedding import InputEmbedding
from llm_foundations.layer_norm import LayerNorm
from llm_foundations.lm_head import LMHead

IMPORTED_CORE = Path(decoder_module.__file__).resolve()
EXPECTED_CORE = (Path.cwd() / "llm_foundations" / "decoder_only_lm.py").resolve()
print(f"Imported decoder core: {IMPORTED_CORE}")
assert IMPORTED_CORE == EXPECTED_CORE, (
    f"Run from the repository root to test its learner core: expected {EXPECTED_CORE}, got {IMPORTED_CORE}"
)
DecoderOnlyLM = decoder_module.DecoderOnlyLM

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")
DEVICE_REASON = (
    "CUDA is available; selected cuda:0"
    if CUDA_AVAILABLE
    else "torch.cuda.is_available() is false; selected CPU"
)
print(f"CUDA availability: {DEVICE_REASON}; actual device: {DEVICE}")

BASE_CONFIG = {
    "vocab_size": 7,
    "max_seq_len": 5,
    "d_model": 5,
    "num_layers": 2,
    "num_heads": 2,
    "d_k": 3,
    "d_v": 2,
    "d_ff": 7,
    "eps": 1e-5,
}
@pytest.fixture(autouse=True)
def full_precision_and_restore_global_settings():
    """Disable reduced precision for each test and restore global settings."""
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.autocast(device_type=DEVICE.type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.set_float32_matmul_precision(old_precision)


def make_raw_model(config: Mapping[str, Any] | None = None) -> nn.Module:
    """Construct the learner model without device or dtype conversion."""
    values = dict(BASE_CONFIG if config is None else config)
    return DecoderOnlyLM(
        values["vocab_size"],
        values["max_seq_len"],
        values["d_model"],
        values["num_layers"],
        values["num_heads"],
        values["d_k"],
        values["d_v"],
        values["d_ff"],
        values["eps"],
    )


def make_model(
    config: Mapping[str, Any] | None = None,
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.float64,
) -> nn.Module:
    """Construct the learner model and move it to the selected execution setup."""
    return make_raw_model(config).to(device=device, dtype=dtype)


def _matrix(rows: int, cols: int, tag: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    r = torch.arange(rows, device=device, dtype=torch.int64).reshape(rows, 1)
    c = torch.arange(cols, device=device, dtype=torch.int64).reshape(1, cols)
    residues = torch.remainder(3 * r + 2 * c + 5 * tag, 11) - 5
    values = 0.04 * residues.to(dtype)
    values = values + 0.003 * (r + 1).to(dtype) - 0.002 * (c + 1).to(dtype)
    return values


def _vector(size: int, tag: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(size, device=device, dtype=torch.int64)
    residues = torch.remainder(index + 3 * tag, 13) - 6
    return 0.025 * residues.to(dtype) + 0.003 * (index + 1).to(dtype)


def _gamma(size: int, tag: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    index = torch.arange(size, device=device, dtype=torch.int64)
    return 0.8 + 0.035 * torch.remainder(index + tag, 7).to(dtype)


def make_fixture(
    config: Mapping[str, Any], *, device: torch.device = DEVICE, dtype: torch.dtype = torch.float64
) -> dict[str, Any]:
    """Create fixture tensors with a unique tag for every semantic parameter.

    Token and position use tags 1 and 2. Each layer has stride 10 + 3H and
    owns attention norm at offsets 0-1, head h projections at 2 + 3h through
    4 + 3h, output projection at 2 + 3H, FFN norm at 3 + 3H through 4 + 3H,
    and FFN weights/biases at 5 + 3H through 8 + 3H. Final norm and LM head
    use three tags after the layer allocation.
    """
    v = config["vocab_size"]
    m = config["max_seq_len"]
    d = config["d_model"]
    h = config["num_heads"]
    dk = config["d_k"]
    dv = config["d_v"]
    f = config["d_ff"]
    layer_stride = 10 + 3 * h
    layers = []
    for layer_index in range(config["num_layers"]):
        tag = 10 + layer_index * layer_stride
        heads = []
        for head_index in range(h):
            head_tag = tag + 2 + 3 * head_index
            heads.append(
                {
                    "q": _matrix(d, dk, head_tag, device=device, dtype=dtype),
                    "k": _matrix(d, dk, head_tag + 1, device=device, dtype=dtype),
                    "v": _matrix(d, dv, head_tag + 2, device=device, dtype=dtype),
                }
            )
        layers.append(
            {
                "attention_gamma": _gamma(d, tag, device=device, dtype=dtype),
                "attention_beta": _vector(d, tag + 1, device=device, dtype=dtype),
                "heads": heads,
                "w_o": _matrix(h * dv, d, tag + 2 + 3 * h, device=device, dtype=dtype),
                "ffn_gamma": _gamma(d, tag + 3 + 3 * h, device=device, dtype=dtype),
                "ffn_beta": _vector(d, tag + 4 + 3 * h, device=device, dtype=dtype),
                "w_1": _matrix(d, f, tag + 5 + 3 * h, device=device, dtype=dtype),
                "b_1": _vector(f, tag + 6 + 3 * h, device=device, dtype=dtype),
                "w_2": _matrix(f, d, tag + 7 + 3 * h, device=device, dtype=dtype),
                "b_2": _vector(d, tag + 8 + 3 * h, device=device, dtype=dtype),
            }
        )
    final_tag = 10 + config["num_layers"] * layer_stride
    return {
        "token": _matrix(v, d, 1, device=device, dtype=dtype),
        "position": _matrix(m, d, 2, device=device, dtype=dtype),
        "layers": layers,
        "final_gamma": _gamma(d, final_tag, device=device, dtype=dtype),
        "final_beta": _vector(d, final_tag + 1, device=device, dtype=dtype),
        "head": _matrix(d, v, final_tag + 2, device=device, dtype=dtype),
    }


def candidate_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    """Bind semantic parameter roles to the inspected learner module paths."""
    params: dict[str, nn.Parameter] = {
        "token": model.input_embedding.token_embed,
        "position": model.input_embedding.pos_embed,
        "final_gamma": model.final_norm.gamma,
        "final_beta": model.final_norm.beta,
        "head": model.lm_head.w_lmhead,
    }
    for layer_index, block in enumerate(model.blocks):
        prefix = f"layer{layer_index}."
        attention = block.attention_residual
        mha = attention.multi_head_attention
        params[prefix + "attention_gamma"] = attention.normalized_input.gamma
        params[prefix + "attention_beta"] = attention.normalized_input.beta
        params[prefix + "w_o"] = mha.w_o
        for head_index, head in enumerate(mha.heads):
            head_prefix = prefix + f"head{head_index}."
            params[head_prefix + "q"] = head.w_q
            params[head_prefix + "k"] = head.w_k
            params[head_prefix + "v"] = head.w_v
        params[prefix + "ffn_gamma"] = block.ffn_norm.gamma
        params[prefix + "ffn_beta"] = block.ffn_norm.beta
        params[prefix + "w_1"] = block.ffn.w_1
        params[prefix + "b_1"] = block.ffn.b_1
        params[prefix + "w_2"] = block.ffn.w_2
        params[prefix + "b_2"] = block.ffn.b_2
    return params


def load_fixture(model: nn.Module, fixture: Mapping[str, Any]) -> None:
    """Copy generated fixture values into existing learner parameters."""
    refs = candidate_parameters(model)
    values: dict[str, torch.Tensor] = {
        "token": fixture["token"],
        "position": fixture["position"],
        "final_gamma": fixture["final_gamma"],
        "final_beta": fixture["final_beta"],
        "head": fixture["head"],
    }
    for layer_index, layer in enumerate(fixture["layers"]):
        prefix = f"layer{layer_index}."
        for name in ("attention_gamma", "attention_beta", "w_o", "ffn_gamma", "ffn_beta", "w_1", "b_1", "w_2", "b_2"):
            values[prefix + name] = layer[name]
        for head_index, head in enumerate(layer["heads"]):
            for name in ("q", "k", "v"):
                values[prefix + f"head{head_index}." + name] = head[name]
    assert set(refs) == set(values)
    with torch.no_grad():
        for name, parameter in refs.items():
            parameter.copy_(values[name])


def fixture_parameter_map(fixture: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Flatten fixture tensors by the same explicit semantic binding table."""
    values: dict[str, torch.Tensor] = {
        "token": fixture["token"],
        "position": fixture["position"],
        "final_gamma": fixture["final_gamma"],
        "final_beta": fixture["final_beta"],
        "head": fixture["head"],
    }
    for layer_index, layer in enumerate(fixture["layers"]):
        prefix = f"layer{layer_index}."
        for name in ("attention_gamma", "attention_beta", "w_o", "ffn_gamma", "ffn_beta", "w_1", "b_1", "w_2", "b_2"):
            values[prefix + name] = layer[name]
        for head_index, head in enumerate(layer["heads"]):
            for name in ("q", "k", "v"):
                values[prefix + f"head{head_index}." + name] = head[name]
    return values


def _norm_scalar(row: list[float], gamma: list[float], beta: list[float], eps: float) -> list[float]:
    mean = sum(row) / len(row)
    variance = sum((value - mean) ** 2 for value in row) / len(row)
    scale = math.sqrt(variance + eps)
    return [gamma[i] * (row[i] - mean) / scale + beta[i] for i in range(len(row))]


def _linear_scalar(row: list[float], weight: list[list[float]]) -> list[float]:
    return [sum(row[i] * weight[i][j] for i in range(len(row))) for j in range(len(weight[0]))]


def scalar_reference(
    config: Mapping[str, Any], fixture: Mapping[str, Any], token_ids: list[int], *, reverse_layers: bool = False
) -> torch.Tensor:
    """Compute logits with Python scalar loops and no learner forward helpers."""
    token_table = fixture["token"].detach().cpu().tolist()
    position_table = fixture["position"].detach().cpu().tolist()
    rows = [
        [token_table[token_id][feature] + position_table[position][feature] for feature in range(config["d_model"])]
        for position, token_id in enumerate(token_ids)
    ]
    layers = list(fixture["layers"])
    if reverse_layers:
        layers.reverse()
    for layer in layers:
        updated_rows = []
        for row_index, row in enumerate(rows):
            normalized = _norm_scalar(
                row,
                layer["attention_gamma"].detach().cpu().tolist(),
                layer["attention_beta"].detach().cpu().tolist(),
                config["eps"],
            )
            head_outputs = []
            for head in layer["heads"]:
                q_rows = [_linear_scalar(_norm_scalar(r, layer["attention_gamma"].detach().cpu().tolist(), layer["attention_beta"].detach().cpu().tolist(), config["eps"]), head["q"].detach().cpu().tolist()) for r in rows]
                k_rows = [_linear_scalar(_norm_scalar(r, layer["attention_gamma"].detach().cpu().tolist(), layer["attention_beta"].detach().cpu().tolist(), config["eps"]), head["k"].detach().cpu().tolist()) for r in rows]
                v_rows = [_linear_scalar(_norm_scalar(r, layer["attention_gamma"].detach().cpu().tolist(), layer["attention_beta"].detach().cpu().tolist(), config["eps"]), head["v"].detach().cpu().tolist()) for r in rows]
                query = q_rows[row_index]
                scores = [sum(query[k] * key[k] for k in range(len(query))) / math.sqrt(len(query)) for key in k_rows[: row_index + 1]]
                max_score = max(scores)
                exponentials = [math.exp(score - max_score) for score in scores]
                denominator = sum(exponentials)
                probabilities = [value / denominator for value in exponentials]
                head_outputs.append(
                    [
                        sum(probabilities[j] * v_rows[j][feature] for j in range(len(probabilities)))
                        for feature in range(config["d_v"])
                    ]
                )
            concatenated = [head_outputs[h][feature] for h in range(config["num_heads"]) for feature in range(config["d_v"])]
            attention_update = _linear_scalar(concatenated, layer["w_o"].detach().cpu().tolist())
            residual = [row[i] + attention_update[i] for i in range(config["d_model"])]
            ffn_input = _norm_scalar(
                residual,
                layer["ffn_gamma"].detach().cpu().tolist(),
                layer["ffn_beta"].detach().cpu().tolist(),
                config["eps"],
            )
            first = _linear_scalar(ffn_input, layer["w_1"].detach().cpu().tolist())
            first = [first[i] + layer["b_1"].detach().cpu().tolist()[i] for i in range(config["d_ff"])]
            activated = [0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))) for value in first]
            second = _linear_scalar(activated, layer["w_2"].detach().cpu().tolist())
            second = [second[i] + layer["b_2"].detach().cpu().tolist()[i] for i in range(config["d_model"])]
            updated_rows.append([residual[i] + second[i] for i in range(config["d_model"])])
        rows = updated_rows
    normalized_rows = [
        _norm_scalar(
            row,
            fixture["final_gamma"].detach().cpu().tolist(),
            fixture["final_beta"].detach().cpu().tolist(),
            config["eps"],
        )
        for row in rows
    ]
    head = fixture["head"].detach().cpu().tolist()
    logits = [_linear_scalar(row, head) for row in normalized_rows]
    return torch.tensor(logits, dtype=fixture["token"].dtype, device=fixture["token"].device)


def differentiable_reference(
    config: Mapping[str, Any], fixture: Mapping[str, Any], token_ids: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build an independent primitive-PyTorch differentiable forward oracle."""
    leaves = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in fixture_parameter_map(fixture).items()
    }
    x = leaves["token"][token_ids] + leaves["position"][: token_ids.shape[0]]
    for layer_index in range(config["num_layers"]):
        prefix = f"layer{layer_index}."
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + config["eps"])
        normalized = normalized * leaves[prefix + "attention_gamma"] + leaves[prefix + "attention_beta"]
        head_outputs = []
        for head_index in range(config["num_heads"]):
            hp = prefix + f"head{head_index}."
            q = normalized @ leaves[hp + "q"]
            k = normalized @ leaves[hp + "k"]
            v = normalized @ leaves[hp + "v"]
            scores = (q @ k.transpose(0, 1)) / math.sqrt(config["d_k"])
            future = torch.triu(
                torch.ones((token_ids.shape[0], token_ids.shape[0]), dtype=torch.bool, device=token_ids.device),
                diagonal=1,
            )
            probabilities = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)
            head_outputs.append(probabilities @ v)
        attention_update = torch.cat(head_outputs, dim=-1) @ leaves[prefix + "w_o"]
        residual = x + attention_update
        mean = residual.mean(dim=-1, keepdim=True)
        variance = ((residual - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized = (residual - mean) / torch.sqrt(variance + config["eps"])
        normalized = normalized * leaves[prefix + "ffn_gamma"] + leaves[prefix + "ffn_beta"]
        first = normalized @ leaves[prefix + "w_1"] + leaves[prefix + "b_1"]
        activation = 0.5 * first * (1.0 + torch.erf(first / math.sqrt(2.0)))
        second = activation @ leaves[prefix + "w_2"] + leaves[prefix + "b_2"]
        x = residual + second
    mean = x.mean(dim=-1, keepdim=True)
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    normalized = (x - mean) / torch.sqrt(variance + config["eps"])
    normalized = normalized * leaves["final_gamma"] + leaves["final_beta"]
    return normalized @ leaves["head"], leaves


def _assert_logits_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    if dtype == torch.float64:
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def remove_hooks(handles: list[nn.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return str(tensor.device), tensor.untyped_storage().data_ptr()


def _objective_weights(shape: tuple[int, int], *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.arange(shape[0], device=device, dtype=dtype).reshape(-1, 1)
    v = torch.arange(shape[1], device=device, dtype=dtype).reshape(1, -1)
    return 0.07 * (t + 1) - 0.03 * (v + 1) + 0.01 * torch.remainder(t + 2 * v, 3)


def test_dl01_api_is_importable_module_and_returns_only_logits_tensor():
    model = make_model({**BASE_CONFIG, "num_layers": 1})
    assert isinstance(model, nn.Module)
    assert isinstance(model.input_embedding, InputEmbedding)
    assert isinstance(model.blocks, nn.ModuleList)
    assert isinstance(model.final_norm, LayerNorm)
    assert isinstance(model.lm_head, LMHead)
    result = model(torch.tensor([1], dtype=torch.long, device=DEVICE))
    assert isinstance(result, torch.Tensor)
    assert result.ndim == 2


def test_dl02_rectangular_configuration_returns_full_logits_and_existing_children():
    config = dict(BASE_CONFIG)
    model = make_model(config)
    assert isinstance(model.input_embedding, InputEmbedding)
    assert len(model.blocks) == 2
    assert all(isinstance(block, PreNormDecoderBlock) for block in model.blocks)
    assert isinstance(model.final_norm, LayerNorm)
    assert isinstance(model.lm_head, LMHead)
    logits = model(torch.tensor([4, 1, 4], dtype=torch.long, device=DEVICE))
    assert logits.shape == (3, 7)


@pytest.mark.parametrize(
    "config,ids",
    [
        ({**BASE_CONFIG, "num_layers": 1}, [2]),
        ({**BASE_CONFIG, "num_layers": 1}, [1, 2, 3, 4, 5]),
        (
            {
                "vocab_size": 1,
                "max_seq_len": 1,
                "d_model": 1,
                "num_layers": 1,
                "num_heads": 1,
                "d_k": 1,
                "d_v": 1,
                "d_ff": 1,
                "eps": 1e-5,
            },
            [0],
        ),
    ],
)
def test_dl03_singleton_and_boundary_positive_sizes(config, ids):
    model = make_model(config)
    logits = model(torch.tensor(ids, dtype=torch.long, device=DEVICE))
    assert logits.shape == (len(ids), config["vocab_size"])
    assert torch.isfinite(logits).all()


def test_dl04_noncontiguous_long_ids_match_contiguous_clone():
    model = make_model({**BASE_CONFIG, "num_layers": 1})
    fixture = make_fixture({**BASE_CONFIG, "num_layers": 1})
    load_fixture(model, fixture)
    source = torch.tensor([1, 99, 3, 99, 2, 99], dtype=torch.long, device=DEVICE)
    noncontiguous = source[::2]
    assert not noncontiguous.is_contiguous()
    contiguous = noncontiguous.clone()
    torch.testing.assert_close(model(noncontiguous), model(contiguous), rtol=0, atol=0)


def test_dl05_zero_update_hand_calculates_embedding_final_norm_and_raw_head():
    config = {
        "vocab_size": 4,
        "max_seq_len": 3,
        "d_model": 2,
        "num_layers": 1,
        "num_heads": 1,
        "d_k": 1,
        "d_v": 1,
        "d_ff": 2,
        "eps": 0.25,
    }
    model = make_model(config)
    with torch.no_grad():
        model.input_embedding.token_embed.copy_(torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, -1.0], [2.0, 1.0]], dtype=torch.float64, device=DEVICE))
        model.input_embedding.pos_embed.copy_(torch.tensor([[0.5, -0.5], [-1.0, 1.0], [0.0, 0.0]], dtype=torch.float64, device=DEVICE))
        block = model.blocks[0]
        block.attention_residual.multi_head_attention.w_o.zero_()
        block.ffn.w_1.zero_()
        block.ffn.b_1.zero_()
        block.ffn.w_2.zero_()
        block.ffn.b_2.zero_()
        model.final_norm.gamma.copy_(torch.tensor([2.0, 0.5], dtype=torch.float64, device=DEVICE))
        model.final_norm.beta.copy_(torch.tensor([0.25, -0.75], dtype=torch.float64, device=DEVICE))
        model.lm_head.w_lmhead.copy_(torch.tensor([[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 2.0, -1.0]], dtype=torch.float64, device=DEVICE))
    ids = torch.tensor([2, 0], dtype=torch.long, device=DEVICE)
    rows = ((3.5, -1.5), (0.0, 3.0))
    expected_rows = []
    for a, b in rows:
        z = ((a - b) / 2.0) / math.sqrt(((a - b) / 2.0) ** 2 + 0.25)
        expected_rows.append((2.0 * z + 0.25, -0.5 * z - 0.75))
    expected = torch.tensor(
        [[row[0], row[1], -row[0] + 2.0 * row[1], 2.0 * row[0] - row[1]] for row in expected_rows],
        dtype=torch.float64,
        device=DEVICE,
    )
    actual = model(ids)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_dl06_active_single_layer_matches_scalar_forward_oracle():
    config = {
        "vocab_size": 5,
        "max_seq_len": 3,
        "d_model": 3,
        "num_layers": 1,
        "num_heads": 2,
        "d_k": 2,
        "d_v": 1,
        "d_ff": 4,
        "eps": 3e-4,
    }
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    ids_list = [2, 1]
    expected = scalar_reference(config, fixture, ids_list)
    actual = model(torch.tensor(ids_list, dtype=torch.long, device=DEVICE))
    _assert_logits_close(actual, expected, torch.float64)


def test_dl07_active_asymmetric_two_layer_stack_matches_scalar_oracle_and_order_matters():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    ids_list = [4, 1, 4]
    expected = scalar_reference(config, fixture, ids_list)
    reversed_expected = scalar_reference(config, fixture, ids_list, reverse_layers=True)
    assert torch.linalg.vector_norm(expected - reversed_expected).item() > 1e-8
    actual = model(torch.tensor(ids_list, dtype=torch.long, device=DEVICE))
    _assert_logits_close(actual, expected, torch.float64)


def test_dl08_stack_call_order_and_stage_inputs_are_explicit():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    token_ids = torch.tensor([4, 1, 4], dtype=torch.long, device=DEVICE)
    expected_dtype = model.input_embedding.token_embed.dtype
    events: list[str] = []
    stage_inputs: dict[str, torch.Tensor] = {}
    stage_outputs: dict[str, torch.Tensor] = {}
    handles = []

    def capture_embedding_input(_module, args):
        events.append("embedding")
        stage_inputs["embedding"] = args[0].detach().clone()

    def capture_embedding_output(_module, _args, output):
        stage_outputs["embedding"] = output.detach().clone()

    handles.append(model.input_embedding.register_forward_pre_hook(capture_embedding_input))
    handles.append(model.input_embedding.register_forward_hook(capture_embedding_output))
    try:
        for index, block in enumerate(model.blocks):
            def capture_block_input(_module, args, i=index):
                events.append(f"block{i}")
                stage_inputs[f"block{i}"] = args[0].detach().clone()

            def capture_block_output(_module, _args, output, i=index):
                stage_outputs[f"block{i}"] = output[0].detach().clone()

            handles.append(block.register_forward_pre_hook(capture_block_input))
            handles.append(block.register_forward_hook(capture_block_output))

        def capture_norm_input(_module, args):
            events.append("final_norm")
            stage_inputs["final_norm"] = args[0].detach().clone()

        def capture_norm_output(_module, _args, output):
            stage_outputs["final_norm"] = output.detach().clone()

        def capture_head_input(_module, args):
            events.append("head")
            stage_inputs["head"] = args[0].detach().clone()

        handles.append(model.final_norm.register_forward_pre_hook(capture_norm_input))
        handles.append(model.final_norm.register_forward_hook(capture_norm_output))
        handles.append(model.lm_head.register_forward_pre_hook(capture_head_input))
        model(token_ids)

        assert events == ["embedding", "block0", "block1", "final_norm", "head"]
        embedding_input = stage_inputs["embedding"]
        assert isinstance(embedding_input, torch.Tensor)
        assert embedding_input.shape == (3,)
        assert embedding_input.dtype == torch.long
        assert embedding_input.device == DEVICE
        for stage in ("block0", "block1", "final_norm", "head"):
            value = stage_inputs[stage]
            assert isinstance(value, torch.Tensor), stage
            assert value.shape == (3, config["d_model"]), stage
            assert value.is_floating_point(), stage
            assert value.dtype == expected_dtype, stage
            assert value.device == DEVICE, stage
        torch.testing.assert_close(stage_outputs["embedding"], stage_inputs["block0"], rtol=0, atol=0)
        torch.testing.assert_close(stage_outputs["block0"], stage_inputs["block1"], rtol=0, atol=0)
        torch.testing.assert_close(stage_outputs["block1"], stage_inputs["final_norm"], rtol=0, atol=0)
        torch.testing.assert_close(stage_outputs["final_norm"], stage_inputs["head"], rtol=0, atol=0)
    finally:
        remove_hooks(handles)


def test_dl09_nondefault_epsilon_reaches_all_two_n_plus_one_norms():
    config = {**BASE_CONFIG, "eps": 3e-4}
    model = make_model(config)
    norms = [model.final_norm]
    for block in model.blocks:
        norms.extend([block.attention_residual.normalized_input, block.ffn_norm])
    assert len(norms) == 2 * config["num_layers"] + 1
    assert all(norm.eps == config["eps"] for norm in norms)


def test_dl10_suffix_replacement_cannot_change_earlier_logits():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    first = [1, 3, 2, 4]
    second = [1, 3, 5, 0]
    ref_first = scalar_reference(config, fixture, first)
    ref_second = scalar_reference(config, fixture, second)
    assert torch.linalg.vector_norm(ref_first[2:] - ref_second[2:]).item() > 1e-8
    ids_first = torch.tensor(first, dtype=torch.long, device=DEVICE)
    ids_second = torch.tensor(second, dtype=torch.long, device=DEVICE)
    logits_first = model(ids_first)
    logits_second = model(ids_second)
    torch.testing.assert_close(logits_first[:2], logits_second[:2], rtol=1e-9, atol=1e-10)


def test_dl11_prefix_logits_are_stable_under_sequence_extension():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    short = [2, 1]
    extended = [2, 1, 4, 3]
    reference_short = scalar_reference(config, fixture, short)
    reference_extended = scalar_reference(config, fixture, extended)
    assert torch.linalg.vector_norm(reference_short[-1] - reference_extended[-1]).item() > 1e-8
    short_logits = model(torch.tensor(short, dtype=torch.long, device=DEVICE))
    extended_logits = model(torch.tensor(extended, dtype=torch.long, device=DEVICE))
    torch.testing.assert_close(short_logits, extended_logits[: len(short)], rtol=1e-9, atol=1e-10)
    one_token = model(torch.tensor([2], dtype=torch.long, device=DEVICE))
    extended_one = model(torch.tensor([2, 4, 3], dtype=torch.long, device=DEVICE))
    torch.testing.assert_close(one_token, extended_one[:1], rtol=1e-9, atol=1e-10)


def test_dl12_all_parameter_gradients_match_independent_differentiable_oracle():
    config = {
        "vocab_size": 5,
        "max_seq_len": 4,
        "d_model": 3,
        "num_layers": 2,
        "num_heads": 2,
        "d_k": 2,
        "d_v": 1,
        "d_ff": 4,
        "eps": 3e-4,
    }
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    ids = torch.tensor([1, 3, 1], dtype=torch.long, device=DEVICE)
    weights = _objective_weights((3, config["vocab_size"]), device=DEVICE, dtype=torch.float64)
    candidate_logits = model(ids)
    reference_logits, leaves = differentiable_reference(config, fixture, ids)
    _assert_logits_close(candidate_logits, reference_logits, torch.float64)
    (candidate_logits * weights).sum().backward()
    (reference_logits * weights).sum().backward()
    refs = candidate_parameters(model)
    assert set(refs) == set(leaves)
    for name, parameter in refs.items():
        expected = leaves[name].grad
        assert parameter.grad is not None, name
        assert expected is not None, name
        assert parameter.grad.shape == expected.shape
        assert parameter.grad.device == DEVICE and parameter.grad.dtype == torch.float64
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-7, atol=1e-9, msg=name)
    assert torch.linalg.vector_norm(leaves["token"].grad[1]).item() > 1e-9
    assert torch.linalg.vector_norm(leaves["token"].grad[3]).item() > 1e-9
    assert torch.count_nonzero(leaves["token"].grad[[0, 2, 4]]) == 0
    assert torch.count_nonzero(leaves["position"].grad[3:]) == 0
    for name in ("token", "position", "final_gamma", "head"):
        assert torch.linalg.vector_norm(leaves[name].grad).item() > 1e-8, name
    for layer_index in range(config["num_layers"]):
        prefix = f"layer{layer_index}."
        layer_grad_norm = sum(
            torch.linalg.vector_norm(leaves[name].grad).item()
            for name in leaves
            if name.startswith(prefix)
        )
        assert layer_grad_norm > 1e-8, f"reference fixture insensitive to {prefix}"


def test_dl13_central_differences_confirm_nonzero_gradients_across_major_stages():
    config = {
        "vocab_size": 5,
        "max_seq_len": 4,
        "d_model": 3,
        "num_layers": 2,
        "num_heads": 2,
        "d_k": 2,
        "d_v": 1,
        "d_ff": 4,
        "eps": 3e-4,
    }
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    ids = torch.tensor([1, 3, 1], dtype=torch.long, device=DEVICE)
    weights = _objective_weights((3, config["vocab_size"]), device=DEVICE, dtype=torch.float64)
    logits = model(ids)
    (logits * weights).sum().backward()
    reference_logits, leaves = differentiable_reference(config, fixture, ids)
    (reference_logits * weights).sum().backward()
    params = candidate_parameters(model)
    groups = ["token", "position", "layer0.head0.q", "layer1.head0.q", "final_gamma", "head"]
    for name in groups:
        parameter = params[name]
        assert parameter.grad is not None
        reference_gradient = leaves[name].grad
        assert reference_gradient is not None
        flat_index = int(torch.argmax(reference_gradient.abs()).item())
        analytical = parameter.grad.reshape(-1)[flat_index].item()
        assert abs(reference_gradient.reshape(-1)[flat_index].item()) > 1e-8, name
        assert abs(analytical) > 1e-8, f"no sensitive entry in {name}"
        original = parameter.detach().reshape(-1)[flat_index].item()
        step = 1e-6
        try:
            with torch.no_grad():
                parameter.reshape(-1)[flat_index] = original + step
                plus = (model(ids) * weights).sum().item()
                parameter.reshape(-1)[flat_index] = original - step
                minus = (model(ids) * weights).sum().item()
        finally:
            with torch.no_grad():
                parameter.reshape(-1)[flat_index] = original
        finite_difference = (plus - minus) / (2.0 * step)
        assert math.isclose(analytical, finite_difference, rel_tol=1e-4, abs_tol=1e-6), (
            name,
            analytical,
            finite_difference,
        )


def test_dl14_layers_and_heads_have_distinct_objects_and_storage():
    config = dict(BASE_CONFIG)
    model = make_raw_model(config)
    assert model.blocks[0] is not model.blocks[1]
    refs = candidate_parameters(model)
    layer_parameters = [
        {
            name: parameter
            for name, parameter in refs.items()
            if name.startswith(f"layer{layer_index}.")
        }
        for layer_index in range(config["num_layers"])
    ]
    for left_index, left_parameters in enumerate(layer_parameters):
        for right_index in range(left_index + 1, len(layer_parameters)):
            right_parameters = layer_parameters[right_index]
            for left_name, left in left_parameters.items():
                for right_name, right in right_parameters.items():
                    assert left is not right, (left_name, right_name)
                    assert _storage_key(left) != _storage_key(right), (left_name, right_name)

    heads = [
        (layer_index, head_index, head)
        for layer_index, block in enumerate(model.blocks)
        for head_index, head in enumerate(block.attention_residual.multi_head_attention.heads)
    ]
    assert len({id(head) for _, _, head in heads}) == len(heads)
    head_parameters = [
        (layer_index, head_index, projection, getattr(head, parameter_name))
        for layer_index, head_index, head in heads
        for projection, parameter_name in (("q", "w_q"), ("k", "w_k"), ("v", "w_v"))
    ]
    assert len({id(parameter) for _, _, _, parameter in head_parameters}) == len(head_parameters)
    assert len({_storage_key(parameter) for _, _, _, parameter in head_parameters}) == len(head_parameters)
    left = model.blocks[0].attention_residual.multi_head_attention.heads[0].w_q
    right = model.blocks[1].attention_residual.multi_head_attention.heads[1].w_q
    right_before = right.detach().clone()
    with torch.no_grad():
        left.add_(1.0)
    torch.testing.assert_close(right, right_before, rtol=0, atol=0)


def test_dl15_all_semantic_parameters_are_registered_once_with_contract_shapes_and_count():
    config = dict(BASE_CONFIG)
    model = make_raw_model(config)
    refs = candidate_parameters(model)
    named = dict(model.named_parameters())
    state = model.state_dict()
    assert len({id(value) for value in refs.values()}) == len(refs)
    assert len(named) == len(refs)
    assert {id(value) for value in named.values()} == {id(value) for value in refs.values()}
    assert set(named) == set(state)
    assert len(model.blocks) == config["num_layers"]
    modules = dict(model.named_modules())
    module_ids = {id(module) for module in modules.values()}
    expected_module_ids = {id(model.input_embedding), id(model.blocks), id(model.final_norm), id(model.lm_head)}
    for block in model.blocks:
        expected_module_ids.update(
            {
                id(block),
                id(block.attention_residual),
                id(block.attention_residual.normalized_input),
                id(block.attention_residual.multi_head_attention),
                id(block.ffn_norm),
                id(block.ffn),
            }
        )
        expected_module_ids.update(id(head) for head in block.attention_residual.multi_head_attention.heads)
    assert expected_module_ids <= module_ids
    expected_count = (
        2 * config["vocab_size"] * config["d_model"]
        + config["max_seq_len"] * config["d_model"]
        + config["num_layers"]
        * (
            2 * config["num_heads"] * config["d_model"] * (config["d_k"] + config["d_v"])
            + 2 * config["d_model"] * config["d_ff"]
            + config["d_ff"]
            + 5 * config["d_model"]
        )
        + 2 * config["d_model"]
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == expected_count
    expected_shapes = {
        "token": (config["vocab_size"], config["d_model"]),
        "position": (config["max_seq_len"], config["d_model"]),
        "final_gamma": (config["d_model"],),
        "final_beta": (config["d_model"],),
        "head": (config["d_model"], config["vocab_size"]),
    }
    for layer_index in range(config["num_layers"]):
        prefix = f"layer{layer_index}."
        expected_shapes.update(
            {
                prefix + "attention_gamma": (config["d_model"],),
                prefix + "attention_beta": (config["d_model"],),
                prefix + "w_o": (config["num_heads"] * config["d_v"], config["d_model"]),
                prefix + "ffn_gamma": (config["d_model"],),
                prefix + "ffn_beta": (config["d_model"],),
                prefix + "w_1": (config["d_model"], config["d_ff"]),
                prefix + "b_1": (config["d_ff"],),
                prefix + "w_2": (config["d_ff"], config["d_model"]),
                prefix + "b_2": (config["d_model"],),
            }
        )
        for head_index in range(config["num_heads"]):
            head_prefix = prefix + f"head{head_index}."
            expected_shapes.update(
                {
                    head_prefix + "q": (config["d_model"], config["d_k"]),
                    head_prefix + "k": (config["d_model"], config["d_k"]),
                    head_prefix + "v": (config["d_model"], config["d_v"]),
                }
            )
    assert set(refs) == set(expected_shapes)
    assert {name: tuple(parameter.shape) for name, parameter in refs.items()} == expected_shapes


def test_dl16_forward_preserves_parameter_identity_values_and_caller_ids():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    model = make_model(config)
    load_fixture(model, fixture)
    ids = torch.tensor([2, 1, 4], dtype=torch.long, device=DEVICE)
    ids_before = ids.clone()
    refs_before = candidate_parameters(model)
    identities = {name: id(value) for name, value in refs_before.items()}
    values = {name: value.detach().clone() for name, value in refs_before.items()}
    module_identities = {name: id(module) for name, module in model.named_modules()}
    first = model(ids)
    second = model(ids)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(ids, ids_before, rtol=0, atol=0)
    refs_after = candidate_parameters(model)
    assert {name: id(value) for name, value in refs_after.items()} == identities
    assert {name: id(module) for name, module in model.named_modules()} == module_identities
    for name, value in refs_after.items():
        torch.testing.assert_close(value, values[name], rtol=0, atol=0)


def test_dl17_input_embedding_and_output_head_weights_are_not_tied():
    model = make_raw_model({**BASE_CONFIG, "num_layers": 1})
    embedding = model.input_embedding.token_embed
    head = model.lm_head.w_lmhead
    assert embedding is not head
    assert _storage_key(embedding) != _storage_key(head)
    before = embedding.detach().clone()
    with torch.no_grad():
        head[0, 0].add_(1.0)
    torch.testing.assert_close(embedding, before, rtol=0, atol=0)


def test_dl18_strict_state_dict_round_trip_restores_outputs_without_sharing_storage():
    config = dict(BASE_CONFIG)
    fixture = make_fixture(config)
    source = make_model(config)
    load_fixture(source, fixture)
    ids = torch.tensor([4, 1, 4], dtype=torch.long, device=DEVICE)
    expected_logits = source(ids)
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    target = make_model(config)
    result = target.load_state_dict(source_state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)
        assert _storage_key(value) != _storage_key(source.state_dict()[name])
    torch.testing.assert_close(target(ids), expected_logits, rtol=0, atol=0)
    source_before = {name: value.detach().clone() for name, value in source.state_dict().items()}
    with torch.no_grad():
        next(target.parameters()).add_(1.0)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("valid_eps", [1, 0.25])
def test_dl21_positive_eps_accepts_python_int_and_float(valid_eps):
    model = make_model({**BASE_CONFIG, "eps": valid_eps})
    assert model.final_norm.eps == valid_eps


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dl28_dtype_device_preservation_in_parameters_stages_logits_and_gradients(dtype):
    config = {
        "vocab_size": 5,
        "max_seq_len": 4,
        "d_model": 3,
        "num_layers": 2,
        "num_heads": 2,
        "d_k": 2,
        "d_v": 1,
        "d_ff": 4,
        "eps": 3e-4,
    }
    fixture = make_fixture(config, dtype=dtype)
    model = make_model(config, dtype=dtype)
    load_fixture(model, fixture)
    ids = torch.tensor([1, 3, 1], dtype=torch.long, device=DEVICE)
    observed: list[torch.Tensor] = []

    def collect(_module, _args, output):
        if isinstance(output, torch.Tensor):
            observed.append(output)
        elif isinstance(output, (tuple, list)):
            observed.extend(value for value in output if isinstance(value, torch.Tensor))

    handles = [module.register_forward_hook(collect) for module in model.modules()]
    try:
        logits = model(ids)
    finally:
        remove_hooks(handles)
    assert ids.dtype == torch.long and ids.device == DEVICE
    assert logits.dtype == dtype and logits.device == DEVICE
    assert observed
    assert all(value.is_floating_point() and value.dtype == dtype and value.device == DEVICE for value in observed)
    assert all(parameter.dtype == dtype and parameter.device == DEVICE for parameter in model.parameters())
    (logits * _objective_weights(tuple(logits.shape), device=DEVICE, dtype=dtype)).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(parameter.grad.dtype == dtype and parameter.grad.device == DEVICE for parameter in model.parameters())
