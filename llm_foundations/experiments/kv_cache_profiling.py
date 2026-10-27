"""Capture một lần generation có chú thích và xuất metric profiler có provenance."""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import platform
import re
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from llm_foundations.experiments.kv_cache_benchmark import (
    BenchmarkConfig,
    _call_arm,
    _rand_ids,
    build_matched_models,
    check_equivalence,
)
from llm_foundations.experiments.profiling_regions import (
    annotation_scope,
    run_instrumented_generation,
)
from llm_foundations.experiments.kv_cache_profiling_report import render_profile_report


@dataclass(frozen=True)
class ProfileConfig:
    """Cấu hình cho một lần capture duy nhất."""

    arm: str = "cached"
    prompt_length: int = 8
    max_new_tokens: int = 4
    level: str = "blocks"
    warmup: int = 2
    record_shapes: bool = False
    profile_memory: bool = False
    with_stack: bool = False


_REGION_LEAVES = frozenset({
    "generation", "prefill", "final_norm", "lm_head", "ln1", "attention",
    "ln2", "ffn", "embedding",
})
_REGION_PATTERNS = (
    re.compile(r"^(?:layer|head|selector|decode|prefix_forward)/\d+$"),
)
_SOURCE_FILES = (
    "llm_foundations/experiments/profiling_regions.py",
    "llm_foundations/experiments/kv_cache_profiling_report.py",
    "llm_foundations/experiments/kv_cache_profiling.py",
    "llm_foundations/experiments/test_kv_cache_profiling.py",
    "llm_foundations/experiments/kv_cache_profiling.ipynb",
    "llm_foundations/experiments/kv_cache_profiling_implementation_report.md",
    "llm_foundations/experiments/kv_cache_benchmark.py",
    "llm_foundations/experiments/test_kv_cache_benchmark.py",
    "llm_foundations/experiments/kv_cache_benchmark.ipynb",
    "llm_foundations/generation.py",
    "llm_foundations/kv_cache/generation.py",
    "llm_foundations/decoder_only_lm.py",
    "llm_foundations/kv_cache/decoder_only_lm.py",
    "llm_foundations/decoder_block.py",
    "llm_foundations/kv_cache/decoder_block.py",
    "llm_foundations/pre_norm_attention_residual.py",
    "llm_foundations/multi_head_attention.py",
    "llm_foundations/kv_cache/multi_head_attention.py",
    "llm_foundations/self_attention.py",
    "llm_foundations/cached_self_attention.py",
    "llm_foundations/scaled_attention.py",
    "llm_foundations/ffn.py",
    "llm_foundations/decoding.py",
    "llm_foundations/embeddings/input_embedding.py",
    "llm_foundations/layer_norm.py",
    "llm_foundations/lm_head.py",
    "llm_foundations/causal_mask.py",
    "llm_foundations/embedding.py",
    "llm_foundations/softmax.py",
)
_FIELD_OPTIONS = {
    "cpu_self": ("self_cpu_time_total", "self_cpu_time"),
    "cpu_total": ("cpu_time_total", "cpu_time"),
    "device_self": ("self_device_time_total", "self_cuda_time_total", "self_device_time", "self_cuda_time"),
    "device_total": ("device_time_total", "cuda_time_total", "device_time", "cuda_time"),
    "memory_self": ("self_cpu_memory_usage", "self_memory_usage"),
    "memory_total": ("cpu_memory_usage", "memory_usage"),
    "device_memory_self": ("self_device_memory_usage", "self_cuda_memory_usage"),
    "device_memory_total": ("device_memory_usage", "cuda_memory_usage"),
}


def _validate_config(config: BenchmarkConfig, profile: ProfileConfig) -> None:
    config.validate()
    if profile.arm not in {"cached", "uncached"}:
        raise ValueError("arm phải là 'cached' hoặc 'uncached'")
    if profile.level not in {"blocks", "ops"}:
        raise ValueError("level phải là 'blocks' hoặc 'ops'")
    if isinstance(profile.prompt_length, bool) or not isinstance(profile.prompt_length, int):
        raise TypeError("prompt_length phải là int")
    if not 1 <= profile.prompt_length <= config.model.max_seq_len:
        raise ValueError("prompt_length phải nằm trong sức chứa model")
    if isinstance(profile.max_new_tokens, bool) or not isinstance(profile.max_new_tokens, int):
        raise TypeError("max_new_tokens phải là int")
    if profile.max_new_tokens < 0:
        raise ValueError("max_new_tokens phải không âm")
    if isinstance(profile.warmup, bool) or not isinstance(profile.warmup, int):
        raise TypeError("warmup phải là int")
    if profile.warmup < 0:
        raise ValueError("warmup phải không âm")
    for name in ("record_shapes", "profile_memory", "with_stack"):
        if not isinstance(getattr(profile, name), bool):
            raise TypeError(f"{name} phải là bool")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không khả dụng; không chuyển sang CPU")
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device)


def _prepare_capture_input(prompt_ids: torch.Tensor) -> torch.Tensor:
    return prompt_ids.clone()


def _region_name(name: str) -> bool:
    return name in _REGION_LEAVES or any(pattern.fullmatch(name) for pattern in _REGION_PATTERNS)


def _safe_field(event: Any, candidates: tuple[str, ...]) -> tuple[Any, str | None]:
    for field in candidates:
        try:
            value = getattr(event, field)
        except Exception:
            continue
        if value is not None:
            return value, field
    return None, None


def _event_name(event: Any) -> str:
    value = getattr(event, "name", None)
    if value is None:
        value = getattr(event, "key", "<unknown>")
    return str(value)


def _parent_event(event: Any) -> Any | None:
    for field in ("cpu_parent", "parent"):
        try:
            parent = getattr(event, field)
        except Exception:
            continue
        if parent is not None:
            return parent
    return None


def _ancestry(event: Any) -> str:
    names: list[str] = []
    parent = _parent_event(event)
    seen: set[int] = set()
    while parent is not None and id(parent) not in seen:
        seen.add(id(parent))
        name = _event_name(parent)
        if _region_name(name):
            names.append(name)
        parent = _parent_event(parent)
    return json.dumps(list(reversed(names)), ensure_ascii=False, separators=(",", ":"))


def _field_mapping(events: list[Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for metric, candidates in _FIELD_OPTIONS.items():
        source = None
        for event in events:
            _, source = _safe_field(event, candidates)
            if source is not None:
                break
        unit = "bytes" if "memory" in metric else "us"
        mapping[metric] = {"source": source, "unit": unit, "available": source is not None}
    return mapping


def _metric_value(
    events: list[Any], metric: str, candidates: tuple[str, ...], *, device_type: str,
    reason: str | None = None,
) -> tuple[Any, str | None]:
    if metric.startswith("device_") and device_type != "cuda":
        return None, reason or "CUDA activity was not requested for CPU capture"
    values: list[float] = []
    for event in events:
        value, _ = _safe_field(event, candidates)
        if value is None:
            return None, reason or f"installed profiler exposes no {metric} value for every event"
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return None, reason or f"installed profiler returned a non-numeric {metric} value"
    if not values:
        return None, reason or "no matching profiler events"
    return sum(values), None


def _row_metrics(
    events: list[Any], *, record_shapes: bool, profile_memory: bool, device_type: str,
) -> dict[str, Any]:
    counts = []
    for event in events:
        count = getattr(event, "count", 1)
        try:
            counts.append(int(count))
        except (TypeError, ValueError):
            counts.append(1)
    result: dict[str, Any] = {"count": sum(counts)}
    metric_specs = (
        ("cpu_self_us", "cpu_self", _FIELD_OPTIONS["cpu_self"]),
        ("cpu_total_us", "cpu_total", _FIELD_OPTIONS["cpu_total"]),
        ("device_self_us", "device_self", _FIELD_OPTIONS["device_self"]),
        ("device_total_us", "device_total", _FIELD_OPTIONS["device_total"]),
    )
    for column, metric, candidates in metric_specs:
        value, reason = _metric_value(events, metric, candidates, device_type=device_type)
        result[column] = value if value is not None else "null"
        if reason:
            result[column.replace("_us", "_unavailable_reason")] = reason
    if record_shapes:
        shapes = []
        for event in events:
            value = getattr(event, "input_shapes", None)
            if value is not None:
                normalized = json.dumps(value, sort_keys=True, default=str)
                if normalized not in shapes:
                    shapes.append(normalized)
        result["input_shapes"] = " | ".join(shapes) if shapes else "null"
        if not shapes:
            result["input_shapes_unavailable_reason"] = "profiler supplied no input shapes"
    if profile_memory:
        for column, metric, candidates in (
            ("memory_self_bytes", "memory_self", _FIELD_OPTIONS["memory_self"]),
            ("memory_total_bytes", "memory_total", _FIELD_OPTIONS["memory_total"]),
            ("device_memory_self_bytes", "device_memory_self", _FIELD_OPTIONS["device_memory_self"]),
            ("device_memory_total_bytes", "device_memory_total", _FIELD_OPTIONS["device_memory_total"]),
        ):
            value, reason = _metric_value(events, metric, candidates, device_type=device_type)
            result[column] = value if value is not None else "null"
            if reason:
                result[column.replace("_bytes", "_unavailable_reason")] = reason
    return result


def _aggregate_events(
    events: list[Any], *, record_shapes: bool, profile_memory: bool, device_type: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    mapping = _field_mapping(events)
    regions: dict[tuple[str, str], list[Any]] = {}
    operators: dict[tuple[str, str], list[Any]] = {}
    for event in events:
        name = _event_name(event)
        if _region_name(name):
            regions.setdefault((name, _ancestry(event)), []).append(event)
        elif name not in {"ProfilerStep#0", "ProfilerStep"}:
            shapes = getattr(event, "input_shapes", None)
            shape_key = json.dumps(shapes, sort_keys=True, default=str) if record_shapes and shapes else ""
            operators.setdefault((name, shape_key), []).append(event)

    region_rows = []
    for (name, ancestry), grouped in sorted(regions.items()):
        region_rows.append({"name": name, "ancestry": ancestry, **_row_metrics(
            grouped, record_shapes=record_shapes, profile_memory=profile_memory, device_type=device_type,
        )})
    operator_rows = []
    for (name, shape_key), grouped in sorted(operators.items()):
        row = {"name": name, **_row_metrics(
            grouped, record_shapes=record_shapes, profile_memory=profile_memory, device_type=device_type,
        )}
        if not record_shapes:
            row.pop("input_shapes", None)
        elif shape_key:
            row["input_shapes"] = shape_key
        operator_rows.append(row)
    return region_rows, operator_rows, mapping


def _is_cuda_device(event: Any) -> bool:
    device_type = getattr(event, "device_type", None)
    device_label = str(getattr(device_type, "name", device_type)).upper()
    return device_label.endswith("CUDA") or device_label.endswith("GPU")


def _activity_metadata(event: Any) -> tuple[bool, Any]:
    for field in ("category", "cat", "activity_type", "activity_kind", "kind"):
        value = event.get(field) if isinstance(event, dict) else getattr(event, field, None)
        if value is not None:
            return True, value
    return False, None


def _category_is_kernel(category: Any) -> bool:
    label = str(getattr(category, "name", category)).strip().lower().replace("-", "_")
    if any(marker in label for marker in ("memcpy", "memset", "transfer", "user_annotation", "annotation")):
        return False
    return "kernel" in label or label in {"concurrent_kernel", "cuda_kernel"}


def _is_cuda_kernel(event: Any) -> bool:
    if not _is_cuda_device(event):
        return False
    has_metadata, metadata = _activity_metadata(event)
    if has_metadata:
        return _category_is_kernel(metadata)
    name = _event_name(event).strip().lower()
    excluded = ("memcpy", "memset", "copy", "dtoh", "htod", "dtod", "clone", "aten::", "user_annotation", "record_function")
    return not any(marker in name for marker in excluded) and not _region_name(name)


def _has_cuda_kernels(events: list[Any]) -> bool:
    return any(_is_cuda_kernel(event) for event in events)


def _trace_kernel_evidence(trace_events: list[Any]) -> tuple[int, bool]:
    categories = [event.get("cat") for event in trace_events if isinstance(event, dict) and event.get("cat") is not None]
    if not categories:
        return 0, False
    return sum(1 for category in categories if _category_is_kernel(category)), True


def _capture_status(
    device_type: str,
    events: list[Any],
    *,
    expected_cuda_work: bool,
    trace_events: list[Any] | None = None,
) -> dict[str, Any]:
    if device_type != "cuda":
        return {
            "complete": True,
            "status": "complete",
            "cuda_kernel_event_count": None,
            "evidence_source": "not_applicable_cpu_capture",
            "reason": None,
        }
    trace_count, trace_categories_present = _trace_kernel_evidence(trace_events or [])
    if trace_categories_present:
        kernel_count = trace_count
        evidence_source = "chrome_trace.cat"
    else:
        kernel_events = [event for event in events if _is_cuda_kernel(event)]
        kernel_count = len(kernel_events)
        has_activity_metadata = any(_activity_metadata(event)[0] for event in kernel_events)
        if has_activity_metadata:
            evidence_source = "torch_profiler.events_activity_metadata"
        elif kernel_count:
            evidence_source = "torch_profiler.events_device_type_and_filtered_name"
        else:
            evidence_source = "no_kernel_event_evidence"
    if not expected_cuda_work:
        return {
            "complete": True,
            "status": "complete",
            "cuda_kernel_event_count": kernel_count,
            "evidence_source": evidence_source,
            "reason": "zero effective token budget; no CUDA kernel was expected",
        }
    complete = kernel_count > 0
    return {
        "complete": complete,
        "status": "complete" if complete else "unsupported_or_incomplete",
        "cuda_kernel_event_count": kernel_count,
        "evidence_source": evidence_source,
        "reason": None if complete else "CUDA activity was requested but no compute-kernel evidence was exposed",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        fields = ["name", "count"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in _SOURCE_FILES
        if (root / relative).is_file()
    }


def _new_run_dir(output_root: str | Path) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    while True:
        run = root / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        try:
            run.mkdir()
            return run
        except FileExistsError:
            continue


def _export_trace(profiler: Any | None, path: Path) -> tuple[list[Any] | None, Exception | None]:
    if profiler is None:
        return None, None
    try:
        profiler.export_chrome_trace(str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        trace_events = payload.get("traceEvents") if isinstance(payload, dict) else payload
        if not isinstance(trace_events, list):
            raise ValueError("Chrome trace has no event list")
        return trace_events, None
    except Exception as exc:
        path.unlink(missing_ok=True)
        return None, exc


def _environment(device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "thread_count": torch.get_num_threads(),
        "dtype": str(dtype),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _write_artifacts(
    run: Path, *, events: list[Any], trace_available: bool, trace_reason: str | None,
    manifest: dict[str, Any], profile: ProfileConfig, device_type: str,
) -> None:
    regions, operators, mapping = _aggregate_events(
        events,
        record_shapes=profile.record_shapes,
        profile_memory=profile.profile_memory,
        device_type=device_type,
    )
    _write_csv(run / "regions.csv", regions)
    _write_csv(run / "operators.csv", operators)
    manifest["event_fields"] = mapping
    manifest["artifacts"] = {
        "trace": "trace.json",
        "regions": "regions.csv",
        "operators": "operators.csv",
        "manifest": "manifest.json",
        "report": "report.ipynb",
        "graphs": "graphs/",
    }
    manifest["artifact_availability"] = {
        "trace": {"available": trace_available, "reason": trace_reason},
        "regions": {"available": True, "reason": None},
        "operators": {"available": True, "reason": None},
        "manifest": {"available": True, "reason": None},
        "report": {"available": False, "reason": "report rendering pending"},
        "graphs": {"available": False, "reason": "report rendering pending"},
    }
    if not trace_available:
        manifest["capture"]["complete"] = False
        manifest["capture"]["status"] = "unsupported_or_incomplete" if device_type == "cuda" else "failed"
        prior_reason = manifest["capture"].get("reason")
        manifest["capture"]["reason"] = "; ".join(value for value in (prior_reason, trace_reason) if value)
    manifest["source_sha256"] = _source_hashes()
    manifest_path = run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    try:
        render_profile_report(run)
    except Exception as exc:
        manifest["artifact_availability"]["report"] = {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
        manifest["artifact_availability"]["graphs"] = {"available": False, "reason": "report rendering failed"}
        manifest["capture"]["complete"] = False
        manifest["capture"]["status"] = "unsupported_or_incomplete" if device_type == "cuda" else "failed"
        prior_reason = manifest["capture"].get("reason")
        report_reason = f"report rendering failed: {type(exc).__name__}: {exc}"
        manifest["capture"]["reason"] = "; ".join(value for value in (prior_reason, report_reason) if value)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        raise
    graph_files = sorted(path.relative_to(run).as_posix() for path in (run / "graphs").glob("*.svg"))
    manifest["artifacts"]["graphs"] = graph_files
    manifest["artifact_availability"]["report"] = {"available": True, "reason": None}
    manifest["artifact_availability"]["graphs"] = {
        "available": bool(graph_files),
        "reason": None if graph_files else "no graph had available metric data",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def _base_manifest(
    config: BenchmarkConfig, profile: ProfileConfig, *, device: torch.device, dtype: torch.dtype,
    prompt_ids: torch.Tensor, prompt_seed: int, weight_mapping: dict[str, Any], effective_tokens: int,
) -> dict[str, Any]:
    ids = [int(value) for value in prompt_ids.detach().cpu().tolist()]
    return {
        "schema_version": 1,
        "benchmark_config": asdict(config),
        "profile_config": asdict(profile),
        "ignored_benchmark_fields": ["warmup", "repeats", "prompt_lengths", "generated_tokens", "generation_lengths", "generation_prompt_length"],
        "case": {
            "arm": profile.arm,
            "prompt_length": profile.prompt_length,
            "requested_tokens": profile.max_new_tokens,
            "capacity": config.model.max_seq_len,
            "effective_tokens": effective_tokens,
            "decode_steps": max(effective_tokens - 1, 0) if profile.arm == "cached" else None,
            "prefix_forward_calls": max(effective_tokens - 1, 0) if profile.arm == "uncached" else None,
            "final_cache_length": profile.prompt_length + max(effective_tokens - 1, 0) if profile.arm == "cached" and effective_tokens else None,
        },
        "effective_tokens": effective_tokens,
        "expected_work_counts": {
            "prefill": int(effective_tokens > 0),
            "decode": max(effective_tokens - 1, 0) if profile.arm == "cached" else 0,
            "prefix_forward": max(effective_tokens - 1, 0) if profile.arm == "uncached" else 0,
            "selector": effective_tokens,
            "append_ids": effective_tokens,
            "cached_k_cat": max(effective_tokens - 1, 0) * config.model.num_layers * config.model.num_heads if profile.arm == "cached" else 0,
            "cached_v_cat": max(effective_tokens - 1, 0) * config.model.num_layers * config.model.num_heads if profile.arm == "cached" else 0,
        },
        "selected_arm": profile.arm,
        "device": str(device),
        "dtype": str(dtype),
        "prompt": {
            "base_seed": config.seed,
            "seed": prompt_seed,
            "seed_derivation": "config.seed + 3101",
            "ids": ids,
            "sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        },
        "weights": {"seed": 20261023, "mapping": weight_mapping},
        "checks": {
            "equivalence_continuation": {
                "seed": config.seed + 10291,
                "ids": [int(value) for value in _rand_ids(
                    config.seed + 10291, min(2, config.model.max_seq_len - profile.prompt_length),
                    config.model, torch.device("cpu"),
                ).tolist()],
            },
        },
        "capture": {},
        "environment": _environment(device, dtype),
        "profiler": {
            "requested": {
                "activities": ["CPU", "CUDA"] if device.type == "cuda" else ["CPU"],
                "record_shapes": profile.record_shapes,
                "profile_memory": profile.profile_memory,
                "with_stack": profile.with_stack,
                "schedule": None,
            },
            "effective": {},
        },
        "measurements": {
            "cpu_time": "Profiler event self and inclusive total durations are microseconds; parent totals already include nested work",
            "device_time": "CUDA/device event durations are microseconds and may overlap CPU execution; do not add them to CPU time as wall time",
            "cuda_kernel_evidence": "Chrome trace category 'kernel' is authoritative when trace categories are present; otherwise use explicit event activity/category metadata or CUDA-device FunctionEvent names after excluding memcpy, memset, aten operations and user annotations",
            "operators": "Operator rows group event name and requested input shapes; kernel count is not a bottleneck measurement",
            "memory": "Profiler memory values describe allocation/free accounting, not HBM traffic or bandwidth",
            "latency": "No wall-time latency is derived from this capture; use the separate repeated benchmark for ordinary timing",
            "limitations": "One trace is descriptive and cannot establish a bottleneck or causal explanation",
        },
    }


def capture_profile(
    config: BenchmarkConfig,
    profile: ProfileConfig,
    *,
    output_root: str | Path,
) -> Path:
    """Capture một public greedy wrapper call và ghi trace, bảng metric cùng manifest.

    Args:
        config: Cấu hình model, seed, dtype, thiết bị và tolerance hiện có.
        profile: Cấu hình arm và workload cho lần capture đơn.
        output_root: Thư mục cha tường minh cho một artifact run mới.
    """
    _validate_config(config, profile)
    if not output_root:
        raise ValueError("output_root phải được chỉ định")
    device = _resolve_device(config.device)
    dtype = getattr(torch, config.dtype)
    effective_tokens = min(profile.max_new_tokens, config.model.max_seq_len - profile.prompt_length)
    prompt_seed = config.seed + 3101
    prompt_ids = _move_to_device(
        _rand_ids(prompt_seed, profile.prompt_length, config.model, torch.device("cpu")), device,
    )
    continuation_length = min(2, config.model.max_seq_len - profile.prompt_length)
    continuation_ids = _move_to_device(
        _rand_ids(config.seed + 10291, continuation_length, config.model, torch.device("cpu")), device,
    )

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    annotation_context = annotation_scope("off")
    annotation_context.__enter__()
    selected_model = None
    unprofiled_ids = None
    run = None
    profiler_context = None
    events: list[Any] = []
    trace_events: list[Any] | None = None
    trace_available = False
    trace_error: Exception | None = None
    capture_error: Exception | None = None
    manifest: dict[str, Any] = {}
    try:
        uncached_model, cached_model, weight_mapping = build_matched_models(config.model, device=device)
        uncached_model.to(device=device, dtype=dtype)
        cached_model.to(device=device, dtype=dtype)
        selected_model = cached_model if profile.arm == "cached" else uncached_model
        manifest = _base_manifest(
            config, profile, device=device, dtype=dtype, prompt_ids=prompt_ids,
            prompt_seed=prompt_seed, weight_mapping=weight_mapping, effective_tokens=effective_tokens,
        )
        if effective_tokens:
            equivalence = check_equivalence(
                uncached_model, cached_model, prompt_ids, continuation_ids,
                atol=config.atol, rtol=config.rtol,
            )
            if not equivalence["allclose"]:
                raise AssertionError("cached/uncached logit equivalence failed before profiling")
            uncached_ids = _call_arm(uncached_model, prompt_ids.clone(), profile.max_new_tokens, False)
            cached_ids = _call_arm(cached_model, prompt_ids.clone(), profile.max_new_tokens, True)
            greedy_equal = bool(torch.equal(uncached_ids, cached_ids))
            if not greedy_equal:
                raise AssertionError("cached/uncached greedy output parity failed before profiling")
            unprofiled_ids = cached_ids if profile.arm == "cached" else uncached_ids
            manifest["checks"].update({
                "logit_equivalence": equivalence,
                "greedy_output_parity": greedy_equal,
                "unprofiled_selected_ids": [int(value) for value in unprofiled_ids.detach().cpu().tolist()],
            })
        else:
            unprofiled_ids = _call_arm(selected_model, prompt_ids.clone(), profile.max_new_tokens, profile.arm == "cached")
            wrapper_identity = bool(torch.equal(unprofiled_ids, prompt_ids))
            if not wrapper_identity:
                raise AssertionError("zero-budget wrapper must return the prompt unchanged")
            manifest["checks"].update({
                "logit_equivalence": {"skipped": "zero effective budget performs no model work"},
                "greedy_output_parity": {"skipped": "zero effective budget has no selector or new token"},
                "selected_wrapper_identity": wrapper_identity,
                "unprofiled_selected_ids": [int(value) for value in unprofiled_ids.detach().cpu().tolist()],
            })
        for _ in range(profile.warmup):
            _call_arm(selected_model, prompt_ids.clone(), profile.max_new_tokens, profile.arm == "cached")
        capture_input = _prepare_capture_input(prompt_ids)
        run = _new_run_dir(output_root)
        if device.type == "cuda" and not hasattr(torch.profiler.ProfilerActivity, "CUDA"):
            capture_error = RuntimeError("CUDA profiler activity is unsupported by this PyTorch build")
        else:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            manifest["profiler"]["effective"] = {
                "activities": [activity.name for activity in activities],
                "record_shapes": profile.record_shapes,
                "profile_memory": profile.profile_memory,
                "with_stack": profile.with_stack,
                "schedule": None,
            }
            try:
                _sync(device)
                with torch.profiler.profile(
                    activities=activities,
                    record_shapes=profile.record_shapes,
                    profile_memory=profile.profile_memory,
                    with_stack=profile.with_stack,
                ) as profiler_context:
                    with annotation_scope(profile.level):
                        captured_ids = run_instrumented_generation(
                            selected_model,
                            capture_input,
                            arm=profile.arm,
                            max_new_tokens=profile.max_new_tokens,
                            level=profile.level,
                        )
                    _sync(device)
            except Exception as exc:
                capture_error = exc

        if profiler_context is not None:
            try:
                events = list(profiler_context.events())
            except Exception as exc:
                if capture_error is None:
                    capture_error = exc
            trace_events, trace_error = _export_trace(profiler_context, run / "trace.json")
            trace_available = trace_error is None
            if trace_error is not None and capture_error is None:
                capture_error = trace_error
        else:
            trace_error = capture_error or RuntimeError("profiler context was not created")
        trace_reason = str(trace_error) if trace_error is not None else None

        if capture_error is None:
            if not torch.equal(captured_ids, unprofiled_ids):
                capture_error = AssertionError("profiled greedy IDs differ from unprofiled IDs")
            else:
                manifest["checks"]["profiled_ids_match_unprofiled"] = True
        if capture_error is None:
            manifest["capture"] = _capture_status(
                device.type, events, expected_cuda_work=effective_tokens > 0,
                trace_events=trace_events if trace_available else None,
            )
        else:
            partial_status = _capture_status(
                device.type, events, expected_cuda_work=effective_tokens > 0,
                trace_events=trace_events if trace_available else None,
            )
            manifest["capture"] = {
                **partial_status,
                "complete": False,
                "status": "unsupported_or_incomplete" if device.type == "cuda" else "failed",
                "reason": f"{type(capture_error).__name__}: {capture_error}",
            }
            manifest["checks"]["profiled_ids_match_unprofiled"] = {"not_completed": str(capture_error)}
        if trace_error is not None:
            manifest["capture"]["trace_export_error"] = f"{type(trace_error).__name__}: {trace_error}"
        if not manifest.get("profiler", {}).get("effective"):
            manifest["profiler"]["effective"] = {"activities": [], "reason": str(capture_error)}
        try:
            _write_artifacts(
                run,
                events=events,
                trace_available=trace_available,
                trace_reason=trace_reason,
                manifest=manifest,
                profile=profile,
                device_type=device.type,
            )
        except Exception as exc:
            if capture_error is None:
                capture_error = exc
        if capture_error is not None:
            raise RuntimeError(f"Profiler capture failed without device fallback; incomplete artifacts: {run}") from capture_error
        if device.type == "cuda" and not manifest["capture"]["complete"]:
            raise RuntimeError(f"CUDA capture is incomplete; inspect manifest: {run}")
        return run
    finally:
        try:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        finally:
            annotation_context.__exit__(*sys.exc_info())
