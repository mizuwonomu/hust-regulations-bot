"""Kiểm tra adapter profiler, capture kỹ thuật và báo cáo offline cho KV cache."""
from __future__ import annotations

import base64
import contextlib
import contextvars
import csv
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import torch
from torch import nn

from llm_foundations.experiments import kv_cache_profiling as profiling
from llm_foundations.experiments import kv_cache_profiling_report as report
from llm_foundations.experiments import profiling_regions as regions
from llm_foundations.experiments.kv_cache_benchmark import (
    BenchmarkConfig,
    ModelConfig,
    build_matched_models,
)


MODEL_CONFIG = ModelConfig(
    vocab_size=11,
    max_seq_len=8,
    d_model=8,
    num_layers=2,
    num_heads=2,
    d_k=2,
    d_v=3,
    d_ff=12,
)
PROMPT = (1, 2, 3)


class RecordingBackend:
    """Ghi nhãn và quan hệ cha-con mà không đo thời gian."""

    def __init__(self) -> None:
        self.stack: list[str] = []
        self.events: list[tuple[str, tuple[str, ...]]] = []

    @contextlib.contextmanager
    def __call__(self, name: str):
        self.events.append((name, tuple(self.stack)))
        self.stack.append(name)
        try:
            yield
        finally:
            assert self.stack.pop() == name


class FakeEvent:
    """Sự kiện tổng hợp chỉ dùng để kiểm tra exporter."""

    def __init__(self, name, *, parent=None, cpu_self=0, cpu_total=0,
                 device_self=None, device_total=None, shapes=None,
                 memory_self=None, memory_total=None, device_type="CPU",
                 category=None, activity_type=None):
        self.name = name
        self.cpu_parent = parent
        self.self_cpu_time_total = cpu_self
        self.cpu_time_total = cpu_total
        self.self_device_time_total = device_self
        self.device_time_total = device_total
        self.input_shapes = shapes or []
        self.self_cpu_memory_usage = memory_self
        self.cpu_memory_usage = memory_total
        self.self_device_memory_usage = None
        self.device_memory_usage = None
        self.device_type = device_type
        self.category = category
        self.activity_type = activity_type


class LegacyEvent:
    """Sự kiện tổng hợp dùng tên trường CUDA cũ."""

    name = "fixture_kernel"
    device_type = "CUDA"
    self_cpu_time_total = 2
    cpu_time_total = 5
    self_cuda_time_total = 3
    cuda_time_total = 7
    cpu_parent = None
    input_shapes = [[2, 2]]


class FakeHarnessModel(nn.Module):
    """Mô hình giả cho kiểm tra điều phối không đo hiệu năng."""

    def __init__(self, arm: str) -> None:
        super().__init__()
        self.arm = arm
        self.input_embedding = SimpleNamespace(pos_embed=torch.empty(MODEL_CONFIG.max_seq_len, 1))
        self.calls: list[str] = []

    def to(self, *args, **kwargs):
        return self

    @staticmethod
    def _logits(length: int) -> torch.Tensor:
        return torch.tensor([0.0, 2.0, 1.0]).repeat(length, 1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.calls.append("forward")
        return self._logits(token_ids.shape[0])

    def prefill(self, token_ids: torch.Tensor):
        self.calls.append("prefill")
        zeros = torch.zeros(token_ids.shape[0], 1)
        return self._logits(token_ids.shape[0]), [[(zeros, zeros.clone())]]

    def decode(self, token_ids: torch.Tensor, caches):
        self.calls.append("decode")
        old_k, old_v = caches[0][0]
        new_k = torch.cat((old_k, torch.zeros(1, 1)))
        new_v = torch.cat((old_v, torch.zeros(1, 1)))
        return self._logits(1), [[(new_k, new_v)]]


class FakeProfiler:
    """Profiler tổng hợp cho kiểm tra điều phối, không cung cấp số đo."""

    active = False
    fail_enter: Exception | None = None
    export_error: Exception | None = None
    trace_events: list[dict] = []
    last = None

    def __init__(self, *, activities, record_shapes, profile_memory, with_stack):
        self.activities = activities
        self.options = (record_shapes, profile_memory, with_stack)
        root = FakeEvent("generation", cpu_self=10, cpu_total=30)
        layer = FakeEvent("layer/0", parent=root, cpu_self=4, cpu_total=15)
        leaf = FakeEvent("ffn", parent=layer, cpu_self=3, cpu_total=7)
        operator = FakeEvent("aten::add", parent=leaf, cpu_self=2, cpu_total=2, shapes=[[2, 2]])
        self._events = [root, layer, leaf, operator]
        FakeProfiler.last = self

    def __enter__(self):
        if FakeProfiler.fail_enter is not None:
            raise FakeProfiler.fail_enter
        FakeProfiler.active = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        FakeProfiler.active = False

    def events(self):
        return self._events

    def export_chrome_trace(self, path):
        if FakeProfiler.export_error is not None:
            raise FakeProfiler.export_error
        events = FakeProfiler.trace_events or [{"name": "fixture", "cat": "cpu_op", "dur": 1}]
        Path(path).write_text(json.dumps({"traceEvents": events}))


FakeProfiler.last = None


def _device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _matched_models():
    device = _device()
    uncached, cached, _ = build_matched_models(MODEL_CONFIG, device=device)
    uncached.to(device=device, dtype=torch.float64)
    cached.to(device=device, dtype=torch.float64)
    return uncached, cached, device


def _run(model, ids, *, arm, level="ops", budget=3):
    backend = RecordingBackend()
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(regions, "_record_function", backend)
        output = regions.run_instrumented_generation(
            model, ids.clone(), arm=arm, max_new_tokens=budget, level=level,
        )
    assert backend.stack == []
    return output, backend.events


def _names(events):
    return [name for name, _ in events]


def _under(events, name, *parents):
    return any(event_name == name and all(parent in ancestors for parent in parents)
               for event_name, ancestors in events)


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _tree_equal(first, second)
    else:
        assert left == right


def _method_layout(model):
    return [
        (module, name, name in module.__dict__, module.__dict__.get(name))
        for module in model.modules()
        for name in ("forward", "prefill", "decode")
    ]


def test_core_has_no_profiling_dependency_and_helper_is_experiment_owned():
    root = Path(__file__).parents[2]
    assert not (root / "llm_foundations/profiling_regions.py").exists()
    core_files = [path for path in (root / "llm_foundations").rglob("*.py")
                  if "experiments" not in path.parts]
    for path in core_files:
        source = path.read_text(encoding="utf-8")
        assert "profiling_regions" not in source
        assert "record_function" not in source
    for path in (Path(__file__).parent / "profiling_regions.py",
                 Path(__file__).parent / "kv_cache_profiling.py",
                 Path(__file__).parent / "kv_cache_profiling_report.py"):
        assert path.is_file()


def test_context_scopes_are_default_off_nested_and_exception_safe(monkeypatch):
    backend = RecordingBackend()
    monkeypatch.setattr(regions, "_record_function", backend)
    with regions.region("outside"):
        pass
    with regions.annotation_scope("off"):
        with regions.region("disabled"):
            pass
    assert backend.events == []

    with regions.annotation_scope("blocks"):
        with regions.region("outer"):
            with regions.region("head/0", level="ops"):
                pass
            with regions.annotation_scope("ops"):
                with regions.region("nested"):
                    pass
            with regions.annotation_scope("off"):
                with regions.region("masked"):
                    pass
            with regions.region("restored"):
                pass
    assert _names(backend.events) == ["outer", "nested", "restored"]

    with pytest.raises(LookupError, match="scope-failure"):
        with regions.annotation_scope("ops"):
            with regions.region("failure"):
                raise LookupError("scope-failure")
    assert regions.current_annotation_level() == "off"
    with regions.region("after-error"):
        pass
    assert "after-error" not in _names(backend.events)

    with regions.annotation_scope("blocks"):
        copied = contextvars.copy_context()

    def record_copied_context():
        with regions.region("copied-context"):
            pass

    copied.run(record_copied_context)
    assert regions.current_annotation_level() == "off"
    assert "copied-context" in _names(backend.events)
    with regions.region("caller-stays-off"):
        pass
    assert "caller-stays-off" not in _names(backend.events)


def test_instance_wrappers_restore_exact_attribute_layout_and_hooks_on_failure(monkeypatch):
    uncached, cached, _ = _matched_models()
    original_forward = uncached.forward
    original_prefill = cached.prefill
    hook_handle = uncached.register_forward_hook(lambda module, inputs, output: None)
    original_hooks = dict(uncached._forward_hooks)
    saved_forward = lambda token_ids: original_forward(token_ids)
    uncached.forward = saved_forward
    try:
        with pytest.raises(RuntimeError, match="instrumentation-probe"):
            with regions.instrument_model(uncached, arm="uncached", level="ops"):
                assert uncached.__dict__["forward"] is not saved_forward
                raise RuntimeError("instrumentation-probe")
        assert uncached.__dict__["forward"] is saved_forward
        assert uncached._forward_hooks == original_hooks
    finally:
        del uncached.forward
    assert "forward" not in uncached.__dict__
    assert uncached.forward.__func__ is original_forward.__func__
    hook_handle.remove()

    with regions.instrument_model(uncached, arm="uncached", level="blocks"):
        with pytest.raises(RuntimeError, match="nested instrumentation"):
            with regions.instrument_model(uncached, arm="uncached", level="ops"):
                pass
        with regions.instrument_model(cached, arm="cached", level="blocks"):
            assert "prefill" in cached.__dict__
    assert "forward" not in uncached.__dict__
    assert "prefill" not in cached.__dict__

    with regions.instrument_model(cached, arm="cached", level="blocks"):
        assert "prefill" in cached.__dict__
        assert "decode" in cached.__dict__
    assert "prefill" not in cached.__dict__
    assert "decode" not in cached.__dict__
    assert cached.prefill.__func__ is original_prefill.__func__


def test_real_tiny_models_preserve_math_state_and_emit_truthful_scopes(monkeypatch):
    uncached, cached, device = _matched_models()
    ids = torch.tensor(PROMPT, dtype=torch.long, device=device)
    uncached.train()
    cached.train()
    uncached.blocks[0].ffn.eval()
    cached.blocks[0].ffn.eval()
    modes_before = [[(module, module.training) for module in model.modules()]
                    for model in (uncached, cached)]
    states_before = [{key: value.detach().clone() for key, value in model.state_dict().items()}
                     for model in (uncached, cached)]
    grad_before = torch.is_grad_enabled()

    baseline_u = profiling._call_arm(uncached, ids.clone(), 3, False)
    baseline_c = profiling._call_arm(cached, ids.clone(), 3, True)
    observed_u, events_u = _run(uncached, ids, arm="uncached")
    observed_c, events_c = _run(cached, ids, arm="cached")

    assert torch.equal(baseline_u, observed_u)
    assert torch.equal(baseline_c, observed_c)
    assert torch.equal(observed_u, observed_c)
    with torch.no_grad():
        baseline_logits_u = uncached(ids)
        with regions.annotation_scope("ops"), regions.instrument_model(
            uncached, arm="uncached", level="ops",
        ):
            instrumented_logits_u = uncached(ids)
        baseline_logits_c, baseline_cache = cached.prefill(ids)
        with regions.annotation_scope("ops"), regions.instrument_model(
            cached, arm="cached", level="ops",
        ):
            instrumented_logits_c, instrumented_cache = cached.prefill(ids)
        decode_id = torch.tensor([4], dtype=torch.long, device=device)
        baseline_decode, baseline_cache = cached.decode(decode_id, baseline_cache)
        with regions.annotation_scope("ops"), regions.instrument_model(
            cached, arm="cached", level="ops",
        ):
            instrumented_decode, instrumented_cache = cached.decode(decode_id, instrumented_cache)
    assert torch.equal(baseline_logits_u, instrumented_logits_u)
    _tree_equal((baseline_logits_c, baseline_cache), (instrumented_logits_c, instrumented_cache))
    _tree_equal((baseline_decode, baseline_cache), (instrumented_decode, instrumented_cache))
    assert torch.equal(ids, torch.tensor(PROMPT, dtype=torch.long, device=device))
    assert torch.is_grad_enabled() is grad_before
    assert [[(module, module.training) for module in model.modules()]
            for model in (uncached, cached)] == modes_before
    for before, model in zip(states_before, (uncached, cached)):
        after = model.state_dict()
        assert before.keys() == after.keys()
        for key, value in before.items():
            assert torch.equal(value, after[key])
    for model in (uncached, cached):
        parameter = next(model.parameters())
        assert parameter.device == device
        assert parameter.dtype == torch.float64

    names_u = _names(events_u)
    names_c = _names(events_c)
    assert names_u.count("generation") == names_c.count("generation") == 1
    assert names_u.count("prefill") == names_c.count("prefill") == 1
    assert names_u.count("prefix_forward/1") == names_u.count("prefix_forward/2") == 1
    assert names_c.count("decode/1") == names_c.count("decode/2") == 1
    assert not any(name.startswith("decode/") for name in names_u)
    assert not any(name.startswith("prefix_forward/") for name in names_c)
    assert sum(name.startswith("selector/") for name in names_u) == 3
    assert sum(name.startswith("selector/") for name in names_c) == 3
    assert names_u.count("layer/0") == 3 and names_u.count("layer/1") == 3
    assert names_c.count("layer/0") == 3 and names_c.count("layer/1") == 3
    assert names_u.count("head/0") == names_c.count("head/0") == 6
    assert names_u.count("attention") == names_c.count("attention") == 6
    assert _under(events_u, "embedding", "prefill")
    assert _under(events_c, "embedding", "prefill")
    assert not _under(events_c, "embedding", "decode/1")
    assert _under(events_u, "ln1", "layer/0")
    assert _under(events_c, "ffn", "layer/1")
    for events in (events_u, events_c):
        assert not any(name in {"residual1", "residual2", "append_ids", "cache_k_cat",
                                "cache_v_cat", "qkv_projection", "qk_scores", "softmax"}
                       for name, _ in events)

    blocks_output, blocks_events = _run(uncached, ids, arm="uncached", level="blocks")
    assert torch.equal(blocks_output, observed_u)
    assert not any(name.startswith("head/") for name, _ in blocks_events)
    cached_blocks_output, cached_blocks_events = _run(cached, ids, arm="cached", level="blocks")
    assert torch.equal(cached_blocks_output, observed_c)
    assert not any(name.startswith("head/") for name, _ in cached_blocks_events)


def test_zero_and_one_token_budgets_and_selector_failure_cleanup(monkeypatch):
    uncached, cached, device = _matched_models()
    ids = torch.tensor(PROMPT, dtype=torch.long, device=device)
    for model, arm in ((uncached, "uncached"), (cached, "cached")):
        before_attrs = dict(model.__dict__)
        output, zero_events = _run(model, ids, arm=arm, budget=0)
        assert torch.equal(output, ids)
        assert not any(name in {"prefill", "decode/1", "prefix_forward/1", "selector/0"}
                       for name, _ in zero_events)
        assert not any(name in model.__dict__ and model.__dict__[name] != before_attrs.get(name)
                       for name in ("forward", "prefill", "decode"))
        one_output, one_events = _run(model, ids, arm=arm, budget=1)
        assert one_output.numel() == ids.numel() + 1
        one_names = _names(one_events)
        assert one_names.count("prefill") == one_names.count("selector/0") == 1
        assert not any(name.startswith(("decode/", "prefix_forward/")) for name in one_names)

    boom = ValueError("selector-failure")
    original = regions.greedy_token

    def fail_selector(logits):
        raise boom

    monkeypatch.setattr(regions, "greedy_token", fail_selector)
    with regions.annotation_scope("ops"):
        with pytest.raises(ValueError) as caught:
            regions.run_instrumented_generation(
                cached, ids.clone(), arm="cached", max_new_tokens=2, level="ops",
            )
        assert caught.value is boom
        assert regions.current_annotation_level() == "ops"
    assert regions.current_annotation_level() == "off"
    assert "prefill" not in cached.__dict__ and "decode" not in cached.__dict__
    monkeypatch.setattr(regions, "greedy_token", original)


def test_cached_method_exception_restores_all_instance_state_and_training_modes():
    _, cached, device = _matched_models()
    ids = torch.tensor(PROMPT, dtype=torch.long, device=device)
    cached.train()
    cached.blocks[0].ffn.eval()
    failure = LookupError("decode-method-failed")

    def failing_decode(token_ids, caches):
        raise failure

    cached.decode = failing_decode
    method_layout = _method_layout(cached)
    modes_before = [(module, module.training) for module in cached.modules()]
    grad_before = torch.is_grad_enabled()
    with regions.annotation_scope("ops"):
        with pytest.raises(LookupError) as caught:
            regions.run_instrumented_generation(
                cached, ids.clone(), arm="cached", max_new_tokens=2, level="ops",
            )
        assert caught.value is failure
        assert regions.current_annotation_level() == "ops"
    assert regions.current_annotation_level() == "off"
    assert _method_layout(cached) == method_layout
    assert [(module, module.training) for module in cached.modules()] == modes_before
    assert torch.is_grad_enabled() is grad_before
    assert cached.__dict__["decode"] is failing_decode


def test_metric_export_maps_fields_and_keeps_nulls_and_ancestry():
    root = FakeEvent("generation", cpu_self=10, cpu_total=50)
    layer = FakeEvent("layer/0", parent=root, cpu_self=4, cpu_total=30)
    attention = FakeEvent("attention", parent=layer, cpu_self=3, cpu_total=14,
                          device_self=0, device_total=2, device_type="CUDA")
    ffn = FakeEvent("ffn", parent=layer, cpu_self=5, cpu_total=8)
    op = FakeEvent("aten::matmul", parent=attention, cpu_self=1, cpu_total=1,
                   device_self=0, device_total=0, shapes=[[2, 3], [3, 4]],
                   memory_self=0, memory_total=0, device_type="CUDA")
    rows, operators, mapping = profiling._aggregate_events(
        [root, layer, attention, ffn, op], record_shapes=True,
        profile_memory=True, device_type="cuda",
    )
    attention_row = next(row for row in rows if row["name"] == "attention")
    assert json.loads(attention_row["ancestry"]) == ["generation", "layer/0"]
    assert attention_row["cpu_self_us"] == 3
    assert attention_row["cpu_total_us"] == 14
    assert next(row for row in rows if row["name"] == "ffn")["device_self_us"] == "null"
    operator = next(row for row in operators if row["name"] == "aten::matmul")
    assert operator["device_self_us"] == 0
    assert operator["memory_self_bytes"] == 0
    assert operator["input_shapes"] == "[[2, 3], [3, 4]]"
    assert mapping["device_self"]["source"] == "self_device_time_total"
    assert mapping["cpu_self"]["unit"] == "us"
    assert mapping["memory_self"]["unit"] == "bytes"

    _, legacy_operators, legacy_mapping = profiling._aggregate_events(
        [LegacyEvent()], record_shapes=False, profile_memory=False, device_type="cuda",
    )
    assert legacy_operators[0]["device_self_us"] == 3
    assert legacy_operators[0]["device_total_us"] == 7
    assert legacy_mapping["device_self"]["source"] == "self_cuda_time_total"

    _, cpu_operators, _ = profiling._aggregate_events(
        [FakeEvent("aten::add", cpu_self=0, cpu_total=0)], record_shapes=False,
        profile_memory=False, device_type="cpu",
    )
    cpu_row = cpu_operators[0]
    assert cpu_row["cpu_self_us"] == 0
    assert cpu_row["device_self_us"] == "null"
    assert "CPU" in cpu_row["device_self_unavailable_reason"]


def _write_rows(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _assert_notebook_cell_ids(notebook):
    if notebook.get("nbformat_minor", 0) < 5:
        return
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(isinstance(cell_id, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id)
               for cell_id in cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def _make_report_run(root: Path, name: str, *, cuda=False):
    run = root / name
    run.mkdir()
    region_rows = [
        {"name": "generation", "ancestry": "", "count": 1, "cpu_self_us": 4,
         "cpu_total_us": 30, "device_self_us": "null",
         "device_self_unavailable_reason": "no CUDA activity"},
        {"name": "ffn", "ancestry": '["generation","layer/0"]', "count": 2,
         "cpu_self_us": 7, "cpu_total_us": 12,
         "device_self_us": 2 if cuda else "null",
         "device_total_us": 4 if cuda else "null",
         "memory_self_bytes": 12, "memory_total_bytes": 16},
        {"name": "attention", "ancestry": '["generation","prefill","layer/0"]',
         "count": 1, "cpu_self_us": "null",
         "cpu_self_unavailable_reason": "fixture missing field", "cpu_total_us": 25,
         "device_self_us": "null"},
        {"name": "attention", "ancestry": '["generation","prefill","layer/1"]',
         "count": 1, "cpu_self_us": 0, "cpu_total_us": 30,
         "device_self_us": "null"},
    ]
    operator_rows = [
        {"name": "aten::add", "count": 4, "cpu_self_us": 8,
         "cpu_total_us": 8, "device_self_us": "null", "memory_self_bytes": 0},
        {"name": "aten::matmul", "count": 2, "cpu_self_us": 5,
         "cpu_total_us": 5, "device_self_us": 3 if cuda else "null"},
    ]
    _write_rows(run / "regions.csv", region_rows)
    _write_rows(run / "operators.csv", operator_rows)
    trace_events = ([{"name": "sgemm_fixture", "cat": "kernel", "ph": "X", "ts": 1, "dur": 6}]
                    if cuda else [{"name": "cpu_fixture", "cat": "cpu_op", "ph": "X", "ts": 1, "dur": 6}])
    (run / "trace.json").write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")
    manifest = {
        "selected_arm": "cached", "device": "cuda:0" if cuda else "cpu",
        "dtype": "torch.float64", "effective_tokens": 2,
        "case": {"prompt_length": 3, "requested_tokens": 2, "capacity": 8},
        "capture": {"complete": True, "status": "complete",
                     "cuda_kernel_event_count": 1 if cuda else None},
        "event_fields": {"cpu_self": {"unit": "us"}, "device_self": {"unit": "us"}},
        "measurements": {"latency": "No normal latency is derived"},
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run


def test_offline_report_embeds_tables_graphs_units_and_missing_data(tmp_path):
    cpu_run = _make_report_run(tmp_path, "cpu-fixture")
    sibling = _make_report_run(tmp_path, "cuda-fixture", cuda=True)
    sentinel = sibling / "sentinel"
    sentinel.write_text("untouched")
    first_report = report.render_profile_report(cpu_run)
    first_bytes = first_report.read_bytes()
    graph_root = cpu_run / "graphs"
    assert (graph_root / "region_cpu_self_total.svg").is_file()
    assert (graph_root / "operators_cpu_self.svg").is_file()
    assert not (graph_root / "device_kernel_times.svg").exists()

    notebook = json.loads(first_report.read_text())
    _assert_notebook_cell_ids(notebook)
    markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "7 us" in markdown and "12 us" in markdown
    assert "ancestry" in markdown.lower()
    assert "CPU-only" in markdown
    assert "No compute kernels" in markdown
    assert "inclusive totals are not additive" in markdown
    assert "No normal latency" in markdown
    assert "device_self_unavailable_reason" in markdown or "no CUDA activity" in markdown
    assert "CPU memory self (bytes)" in markdown
    assert "12 bytes" in markdown and "0 bytes" in markdown
    assert "null - fixture missing field" in markdown
    assert "25 us" in markdown and "0 us" in markdown
    svg_path = graph_root / "region_cpu_self_total.svg"
    svg_root = ElementTree.fromstring(svg_path.read_text())
    svg_text = [element.text or "" for element in svg_root.iter()
                if element.tag.endswith("text")]
    svg_titles = [element.text or "" for element in svg_root.iter()
                  if element.tag.endswith("title")]
    assert "N/A" in svg_text
    assert "0" in svg_text
    assert any("layer/0" in title for title in svg_titles)
    assert any("layer/1" in title for title in svg_titles)
    attachments = [attachment for cell in notebook["cells"]
                   for attachment in cell.get("attachments", {}).values()]
    assert attachments
    embedded = attachments[0]["image/svg+xml"]
    assert base64.b64decode(embedded).startswith(b"<svg")
    assert report.render_profile_report(cpu_run).read_bytes() == first_bytes

    cuda_report = report.render_profile_report(sibling)
    cuda_notebook = json.loads(cuda_report.read_text())
    cuda_markdown = "\n".join("".join(cell["source"]) for cell in cuda_notebook["cells"])
    assert (sibling / "graphs/device_kernel_times.svg").is_file()
    assert "sgemm_fixture" in cuda_markdown
    assert "6 us" in cuda_markdown
    assert sentinel.read_text() == "untouched"


def _patch_fake_harness(monkeypatch, *, active_behavior=None):
    uncached = FakeHarnessModel("uncached")
    cached = FakeHarnessModel("cached")
    monkeypatch.setattr(
        profiling, "build_matched_models",
        lambda config, *, device: (uncached, cached, {"synthetic_mapping": True}),
    )
    equivalence_activity = []

    def fake_equivalence(*args, **kwargs):
        equivalence_activity.append(FakeProfiler.active)
        return {"allclose": True, "checks": [], "atol": kwargs["atol"], "rtol": kwargs["rtol"]}

    monkeypatch.setattr(profiling, "check_equivalence", fake_equivalence)
    arm_calls = []

    def fake_call_arm(model, ids, budget, cached_arm):
        arm_calls.append((FakeProfiler.active, cached_arm, ids))
        count = min(budget, MODEL_CONFIG.max_seq_len - ids.numel())
        suffix = torch.ones((count,), dtype=ids.dtype, device=ids.device)
        return torch.cat((ids, suffix), dim=0) if count else ids

    monkeypatch.setattr(profiling, "_call_arm", fake_call_arm)
    real_runner = regions.run_instrumented_generation
    runner_calls = []
    active_failure = LookupError("active-capture")

    def tracked_runner(model, ids, **kwargs):
        runner_calls.append((FakeProfiler.active, ids, kwargs["arm"]))
        if FakeProfiler.active and active_behavior == "raise":
            raise active_failure
        if FakeProfiler.active and active_behavior == "cuda_error":
            raise RuntimeError("cuda-runtime")
        if FakeProfiler.active and active_behavior == "parity":
            suffix = torch.full((2,), 2, dtype=ids.dtype, device=ids.device)
            return torch.cat((ids, suffix), dim=0)
        return real_runner(model, ids, **kwargs)

    monkeypatch.setattr(profiling, "run_instrumented_generation", tracked_runner)
    monkeypatch.setattr(profiling, "_environment", lambda device, dtype: {"device": str(device), "dtype": str(dtype)})
    monkeypatch.setattr(profiling.torch.profiler, "profile", FakeProfiler)
    FakeProfiler.active = False
    FakeProfiler.fail_enter = None
    FakeProfiler.export_error = None
    FakeProfiler.trace_events = []
    FakeProfiler.last = None
    return {
        "models": (uncached, cached),
        "arm_calls": arm_calls,
        "runner_calls": runner_calls,
        "equivalence_activity": equivalence_activity,
        "active_failure": active_failure,
    }


def _valid_capture_config(device="cpu"):
    return BenchmarkConfig(
        model=MODEL_CONFIG, seed=20261023, dtype="float64", device=device,
        warmup=0, repeats=99, prompt_lengths=(2,), generated_tokens=6,
        generation_lengths=(0,), generation_prompt_length=3,
    )


def test_capture_validation_precedes_run_creation(monkeypatch, tmp_path):
    _patch_fake_harness(monkeypatch)
    invalid = (
        profiling.ProfileConfig(arm="other"),
        profiling.ProfileConfig(level="off"),
        profiling.ProfileConfig(prompt_length=9),
        profiling.ProfileConfig(max_new_tokens=-1),
        profiling.ProfileConfig(warmup=-1),
    )
    for profile in invalid:
        with pytest.raises((ValueError, TypeError)):
            profiling.capture_profile(_valid_capture_config(), profile, output_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_capture_uses_prepared_input_once_and_preserves_existing_run(monkeypatch, tmp_path):
    fake = _patch_fake_harness(monkeypatch)
    old_run = tmp_path / "old-run"
    old_run.mkdir()
    sentinel = old_run / "sentinel.txt"
    sentinel.write_text("historical user data")
    prepare_activity = []
    prepared_inputs = []
    boundary_order = []
    real_prepare = profiling._prepare_capture_input

    def prepare(prompt_ids):
        prepare_activity.append(FakeProfiler.active)
        boundary_order.append(("prepare", FakeProfiler.active))
        result = real_prepare(prompt_ids)
        prepared_inputs.append(result)
        return result

    def synchronize(device):
        boundary_order.append(("sync", FakeProfiler.active))

    monkeypatch.setattr(profiling, "_prepare_capture_input", prepare)
    monkeypatch.setattr(profiling, "_sync", synchronize)
    profile = profiling.ProfileConfig(
        arm="cached", prompt_length=3, max_new_tokens=2, level="ops", warmup=1,
    )
    run = profiling.capture_profile(_valid_capture_config(), profile, output_root=tmp_path)
    run2 = profiling.capture_profile(_valid_capture_config(), profile, output_root=tmp_path)
    assert run != old_run and run2 != run
    assert prepare_activity == [False, False]
    assert len(prepared_inputs) == 2
    assert len(fake["runner_calls"]) == 2
    assert all(active for active, _, _ in fake["runner_calls"])
    assert all(call[1] is prepared for call, prepared in zip(fake["runner_calls"], prepared_inputs))
    assert len({id(ids) for _, _, ids in fake["arm_calls"]}) == len(fake["arm_calls"])
    assert fake["equivalence_activity"] == [False, False]
    assert boundary_order == [
        ("prepare", False), ("sync", False), ("sync", True),
        ("prepare", False), ("sync", False), ("sync", True),
    ]
    assert {path.name for path in run.iterdir()} >= {
        "trace.json", "regions.csv", "operators.csv", "manifest.json", "report.ipynb", "graphs",
    }
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["capture"]["complete"] is True
    assert manifest["checks"]["profiled_ids_match_unprofiled"] is True
    assert manifest["artifact_availability"]["report"]["available"] is True
    assert manifest["profiler"]["effective"]["activities"] == ["CPU"]
    repository_root = Path(__file__).parents[2]
    assert "llm_foundations/experiments/profiling_regions.py" in manifest["source_sha256"]
    assert "llm_foundations/profiling_regions.py" not in manifest["source_sha256"]
    for relative_path, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((repository_root / relative_path).read_bytes()).hexdigest() == digest
    assert sentinel.read_text() == "historical user data"
    assert not FakeProfiler.active


def test_zero_budget_capture_manifest_has_no_model_or_selector_work(monkeypatch, tmp_path):
    fake = _patch_fake_harness(monkeypatch)
    profile = profiling.ProfileConfig(
        arm="uncached", prompt_length=3, max_new_tokens=0, level="blocks", warmup=0,
    )
    run = profiling.capture_profile(_valid_capture_config(), profile, output_root=tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["effective_tokens"] == 0
    assert manifest["expected_work_counts"]["prefill"] == 0
    assert manifest["expected_work_counts"]["selector"] == 0
    assert manifest["checks"]["selected_wrapper_identity"] is True
    assert manifest["checks"]["profiled_ids_match_unprofiled"] is True
    assert fake["models"][0].calls == []
    assert fake["runner_calls"][0][0] is True


def test_capture_and_trace_export_errors_preserve_primary_state(monkeypatch, tmp_path):
    fake = _patch_fake_harness(monkeypatch, active_behavior="raise")
    trace_error = OSError("trace-export")
    FakeProfiler.export_error = trace_error
    rng_before = torch.get_rng_state().clone()
    grad_before = torch.is_grad_enabled()
    modes_before = [model.training for model in fake["models"]]
    with regions.annotation_scope("ops"):
        with pytest.raises(RuntimeError) as caught:
            profiling.capture_profile(
                _valid_capture_config(),
                profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=2, warmup=0),
                output_root=tmp_path,
            )
        assert regions.current_annotation_level() == "ops"
    assert caught.value.__cause__ is fake["active_failure"]
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.is_grad_enabled() is grad_before
    assert [model.training for model in fake["models"]] == modes_before
    assert not FakeProfiler.active
    run = next(tmp_path.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["capture"]["complete"] is False
    assert "active-capture" in manifest["capture"]["reason"]
    assert "trace-export" in manifest["artifact_availability"]["trace"]["reason"]
    assert manifest["artifact_availability"]["report"]["available"] is True
    assert not (run / "trace.json").exists()


def test_parity_export_only_and_report_failures_are_marked_incomplete(monkeypatch, tmp_path):
    _patch_fake_harness(monkeypatch, active_behavior="parity")
    with pytest.raises(RuntimeError) as mismatch:
        profiling.capture_profile(
            _valid_capture_config(),
            profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=2, warmup=0),
            output_root=tmp_path,
        )
    assert isinstance(mismatch.value.__cause__, AssertionError)
    mismatch_run = next(tmp_path.iterdir())
    mismatch_manifest = json.loads((mismatch_run / "manifest.json").read_text())
    assert mismatch_manifest["capture"]["complete"] is False
    assert mismatch_manifest["artifact_availability"]["trace"]["available"] is True

    runs_before = set(tmp_path.iterdir())
    FakeProfiler.export_error = OSError("export-only")
    with pytest.raises(RuntimeError) as export_failure:
        profiling.capture_profile(
            _valid_capture_config(),
            profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=0, warmup=0),
            output_root=tmp_path,
        )
    assert isinstance(export_failure.value.__cause__, OSError)
    export_run = (set(tmp_path.iterdir()) - runs_before).pop()
    export_manifest = json.loads((export_run / "manifest.json").read_text())
    assert export_manifest["capture"]["complete"] is False
    assert export_manifest["artifact_availability"]["trace"]["available"] is False
    assert export_manifest["artifact_availability"]["report"]["available"] is True
    assert not (export_run / "trace.json").exists()

    runs_before = set(tmp_path.iterdir())
    _patch_fake_harness(monkeypatch)
    report_error = OSError("report-render-only")
    monkeypatch.setattr(profiling, "render_profile_report", lambda run: (_ for _ in ()).throw(report_error))
    with pytest.raises(RuntimeError) as report_failure:
        profiling.capture_profile(
            _valid_capture_config(),
            profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=0, warmup=0),
            output_root=tmp_path,
        )
    assert report_failure.value.__cause__ is report_error
    report_run = (set(tmp_path.iterdir()) - runs_before).pop()
    report_manifest = json.loads((report_run / "manifest.json").read_text())
    assert report_manifest["artifact_availability"]["report"]["available"] is False
    assert "report-render-only" in report_manifest["capture"]["reason"]


def test_primary_capture_error_survives_report_render_failure(monkeypatch, tmp_path):
    fake = _patch_fake_harness(monkeypatch, active_behavior="raise")
    report_error = OSError("report-render-after-capture")
    monkeypatch.setattr(profiling, "render_profile_report", lambda run: (_ for _ in ()).throw(report_error))
    with pytest.raises(RuntimeError) as caught:
        profiling.capture_profile(
            _valid_capture_config(),
            profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=2, warmup=0),
            output_root=tmp_path,
        )
    assert caught.value.__cause__ is fake["active_failure"]
    run = next(tmp_path.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["artifact_availability"]["report"]["available"] is False
    assert "active-capture" in manifest["capture"]["reason"]
    assert "report-render-after-capture" in manifest["capture"]["reason"]


def test_device_policy_and_kernel_evidence_never_claim_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert profiling._resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        profiling._resolve_device("cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert profiling._resolve_device("auto") == torch.device("cuda:0")
    unavailable = profiling._capture_status("cuda", [], expected_cuda_work=True)
    assert unavailable["complete"] is False
    assert unavailable["status"] == "unsupported_or_incomplete"
    assert unavailable["cuda_kernel_event_count"] == 0


def test_cuda_capture_runtime_failure_never_retries_cpu(monkeypatch, tmp_path):
    _patch_fake_harness(monkeypatch, active_behavior="cuda_error")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    resolved = []
    monkeypatch.setattr(
        profiling, "_resolve_device",
        lambda requested: (resolved.append(requested), torch.device("cuda:0"))[1],
    )
    monkeypatch.setattr(profiling, "_move_to_device", lambda tensor, device: tensor)
    sync_calls = []
    monkeypatch.setattr(profiling, "_sync", lambda device: sync_calls.append(str(device)))
    with pytest.raises(RuntimeError) as caught:
        profiling.capture_profile(
            _valid_capture_config(device="cuda"),
            profiling.ProfileConfig(arm="cached", prompt_length=3, max_new_tokens=2, warmup=0),
            output_root=tmp_path,
        )
    assert "cuda-runtime" in str(caught.value.__cause__)
    assert resolved == ["cuda"]
    assert sync_calls == ["cuda:0"]
    assert FakeProfiler.active is False
    run = next(tmp_path.iterdir())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["capture"]["complete"] is False
    assert manifest["capture"]["status"] == "unsupported_or_incomplete"
    assert manifest["artifact_availability"]["report"]["available"] is True


def test_cuda_kernel_evidence_filters_transfers_and_trace_categories_are_authoritative():
    transfer = FakeEvent("Memcpy DtoD", device_type="CUDA")
    memset = FakeEvent("Memset", device_type="CUDA")
    annotation = FakeEvent("generation", device_type="CUDA", category="user_annotation")
    clone = FakeEvent("aten::clone", device_type="CUDA")
    compute = FakeEvent("gemm", device_type="CUDA", category="kernel")
    for event in (transfer, memset, annotation, clone):
        assert not profiling._is_cuda_kernel(event)
    assert profiling._is_cuda_kernel(compute)
    status = profiling._capture_status(
        "cuda", [compute], expected_cuda_work=True,
        trace_events=[{"name": "Memcpy DtoD", "cat": "gpu_memcpy"}],
    )
    assert status["complete"] is False
    assert status["cuda_kernel_event_count"] == 0
    assert status["evidence_source"] == "chrome_trace.cat"


def test_real_cpu_profiler_region_and_native_operator_smoke():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as active:
        with regions.annotation_scope("blocks"):
            with regions.region("tiny_cpu_smoke"):
                value = torch.ones((2, 2)) @ torch.ones((2, 2))
    names = {event.name for event in active.events()}
    assert "tiny_cpu_smoke" in names
    assert any("matmul" in name or name == "aten::mm" for name in names)
    assert value.shape == (2, 2)


def test_teaching_notebook_is_unexecuted_default_off_and_teaches_report():
    path = Path(__file__).with_name("kv_cache_profiling.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    _assert_notebook_cell_ids(notebook)
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        compile("".join(cell["source"]), f"{path}:cell", "exec")
    full_text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_PROFILE = False" in full_text
    assert "report.ipynb" in full_text
    assert "display(" in full_text or "FileLink" in full_text
    assert "residual1" not in full_text
    assert "cache_k_cat" not in full_text
    assert "prefill" in full_text and "decode/1" in full_text
    assert re.search(r"^\s*attention\n\s+head/0 \.\.\. head/H-1", full_text, re.MULTILINE)
    assert not re.search(r"^\s*ffn\n\s+head/", full_text, re.MULTILINE)
    assert "P=8" in full_text and "P=96" in full_text and "B=4" in full_text
