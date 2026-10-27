"""Kết xuất notebook và SVG xác định từ artifact profiler của một run."""
from __future__ import annotations

import base64
import csv
import html
import json
import math
from pathlib import Path
from typing import Any


_NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_run_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path.parent != root.resolve() and root.resolve() not in path.parents:
        raise ValueError("artifact path escaped the selected run directory")
    return path


def _number(value: Any) -> float | None:
    if value is None or value == "" or str(value).lower() == "null":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format(value: Any, *, unit: str = "") -> str:
    number = _number(value)
    if number is None:
        return "null"
    shown = str(int(number)) if number.is_integer() else f"{number:.3g}"
    return f"{shown} {unit}".rstrip()


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True).replace("|", "\\|")


def _metric_display(row: dict[str, str], column: str, unit: str = "us") -> str:
    value = row.get(column)
    if _number(value) is None:
        metric = column.removesuffix("_us").removesuffix("_bytes")
        reason_key = metric + "_unavailable_reason"
        reason = row.get(reason_key, "")
        return f"null - {reason}" if reason else "null"
    return _format(value, unit=unit)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["No exported rows"] + [""] * (len(headers) - 1)]
    header = "| " + " | ".join(_escape(value) for value in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_escape(value) for value in row) + " |" for row in rows]
    return "\n".join((header, divider, *body))


def _svg_bars(
    title: str,
    axis: str,
    labels: list[str],
    series: list[tuple[str, list[float | None]]],
) -> str:
    width = 1000
    left = 300
    right = 90
    top = 72
    row_height = 30
    bottom = 72
    height = top + row_height * max(len(labels), 1) + bottom
    plot_width = width - left - right
    max_value = max(
        (value for _, values in series for value in values if value is not None),
        default=0.0,
    )
    max_value = max(max_value, 1.0)
    colors = ("#2b6cb0", "#dd6b20", "#38a169")
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="32" font-size="20" font-family="sans-serif">{_escape(title)}</text>',
        f'<text x="{left}" y="{height - 18}" font-size="13" font-family="sans-serif">{_escape(axis)}</text>',
    ]
    bar_height = max(5, min(10, (row_height - 5) // max(len(series), 1)))
    for row_index, label in enumerate(labels):
        y_center = top + row_index * row_height + row_height / 2
        output.append(f'<text x="{left - 10}" y="{y_center + 4:.1f}" text-anchor="end" font-size="12" font-family="sans-serif">{_escape(label[:42])}<title>{_escape(label)}</title></text>')
        for series_index, (series_name, values) in enumerate(series):
            if row_index >= len(values):
                continue
            bar_y = y_center - len(series) * bar_height / 2 + series_index * bar_height
            value = values[row_index]
            if value is None:
                output.append(f'<text x="{left + 5}" y="{bar_y + bar_height - 2:.1f}" font-size="10" font-family="sans-serif">N/A</text>')
                continue
            value = max(value, 0.0)
            bar_width = plot_width * value / max_value
            color = colors[series_index % len(colors)]
            if value == 0:
                output.append(f'<circle cx="{left}" cy="{bar_y + (bar_height - 1) / 2:.1f}" r="2" fill="{color}"/>')
            else:
                output.append(f'<rect x="{left}" y="{bar_y:.1f}" width="{bar_width:.2f}" height="{bar_height - 1}" fill="{color}"/>')
            output.append(f'<text x="{left + bar_width + 5:.2f}" y="{bar_y + bar_height - 2:.1f}" font-size="10" font-family="sans-serif">{value:.3g}</text>')
    legend_x = left
    for index, (name, _) in enumerate(series):
        color = colors[index % len(colors)]
        output.append(f'<rect x="{legend_x}" y="{height - 47}" width="12" height="12" fill="{color}"/>')
        output.append(f'<text x="{legend_x + 17}" y="{height - 37}" font-size="12" font-family="sans-serif">{_escape(name)}</text>')
        legend_x += 140
    output.append("</svg>")
    return "\n".join(output) + "\n"


def _region_leaf_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    paths = []
    for row in rows:
        raw_ancestry = row.get("ancestry", "")
        try:
            ancestry = json.loads(raw_ancestry) if raw_ancestry else []
        except json.JSONDecodeError:
            ancestry = [part for part in raw_ancestry.split("/") if part]
        if not isinstance(ancestry, list):
            ancestry = [str(ancestry)]
        paths.append((row, (*ancestry, row.get("name", ""))))
    leaves = []
    for row, path in paths:
        has_semantic_child = any(
            len(other_path) > len(path) and other_path[:len(path)] == path
            for _, other_path in paths
        )
        if not has_semantic_child:
            leaves.append(row)
    return leaves


def _trace_kernels(trace_path: Path) -> list[tuple[str, float]]:
    if not trace_path.is_file():
        return []
    payload = _read_json(trace_path)
    events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        return []
    kernels: list[tuple[str, float]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", "")).lower().replace("-", "_")
        name = str(event.get("name", ""))
        excluded = ("memcpy", "memset", "transfer", "annotation")
        if "kernel" not in category or any(marker in category or marker in name.lower() for marker in excluded):
            continue
        duration = _number(event.get("dur"))
        if duration is not None and duration >= 0:
            kernels.append((name, duration))
    return kernels


def _cell(source: str, attachments: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    cell = {
        "cell_type": "markdown",
        "id": "kv-profile-report-main",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if attachments:
        cell["attachments"] = attachments
    return cell


def render_profile_report(run_dir: str | Path) -> Path:
    """Kết xuất báo cáo notebook/graph từ đúng artifact đã xuất của run.

    Args:
        run_dir: Thư mục chứa manifest, CSV và trace tùy chọn của một run.
    """
    root = Path(run_dir).resolve()
    manifest_path = _safe_run_file(root, "manifest.json")
    regions_path = _safe_run_file(root, "regions.csv")
    operators_path = _safe_run_file(root, "operators.csv")
    trace_path = _safe_run_file(root, "trace.json")
    manifest = _read_json(manifest_path)
    region_rows = _read_csv(regions_path)
    operator_rows = _read_csv(operators_path)
    graph_root = root / "graphs"
    graph_root.mkdir(exist_ok=True)

    cpu_region_rows = [row for row in _region_leaf_rows(region_rows)
                       if _number(row.get("cpu_self_us")) is not None
                       or _number(row.get("cpu_total_us")) is not None]
    cpu_region_rows.sort(key=lambda row: (
        _number(row.get("cpu_self_us")) is None,
        -(_number(row.get("cpu_self_us")) or 0.0),
    ))
    cpu_region_rows = cpu_region_rows[:16]
    if cpu_region_rows:
        labels = [f'{row.get("name", "")} [{row.get("ancestry", "")}]' for row in cpu_region_rows]
        series = [
            ("CPU self", [_number(row.get("cpu_self_us")) for row in cpu_region_rows]),
            ("CPU inclusive total", [_number(row.get("cpu_total_us")) for row in cpu_region_rows]),
        ]
        region_graph = graph_root / "region_cpu_self_total.svg"
        region_graph.write_text(_svg_bars("Semantic leaf regions: CPU self vs inclusive total", "Profiler time (us)", labels, series), encoding="utf-8")
    else:
        region_graph = None

    top_operators = [row for row in operator_rows if _number(row.get("cpu_self_us")) is not None]
    top_operators.sort(key=lambda row: _number(row.get("cpu_self_us")) or 0.0, reverse=True)
    top_operators = top_operators[:16]
    if top_operators:
        op_labels = [row.get("name", "") for row in top_operators]
        op_values = [_number(row.get("cpu_self_us")) or 0.0 for row in top_operators]
        operator_graph = graph_root / "operators_cpu_self.svg"
        operator_graph.write_text(_svg_bars("Top operators by CPU self time", "Profiler time (us)", op_labels, [("CPU self", op_values)]), encoding="utf-8")
    else:
        operator_graph = None

    device = str(manifest.get("device", "unknown"))
    capture = manifest.get("capture", {})
    cpu_only = not device.lower().startswith("cuda")
    device_operator_rows = [] if cpu_only else [
        row for row in operator_rows if _number(row.get("device_self_us")) is not None
    ]
    if device_operator_rows:
        device_operator_rows.sort(key=lambda row: _number(row.get("device_self_us")) or 0.0, reverse=True)
        device_operator_rows = device_operator_rows[:16]
        device_graph = graph_root / "device_operator_self.svg"
        device_graph.write_text(_svg_bars(
            "Device operator self time (not wall latency)", "Device profiler time (us)",
            [row.get("name", "") for row in device_operator_rows],
            [("Device self", [_number(row.get("device_self_us")) or 0.0 for row in device_operator_rows])],
        ), encoding="utf-8")
    else:
        device_graph = None

    kernels = [] if cpu_only else _trace_kernels(trace_path)
    if kernels:
        kernel_graph = graph_root / "device_kernel_times.svg"
        kernel_graph.write_text(_svg_bars(
            "Observed CUDA kernel event durations", "Trace event duration (us)",
            [f"{index}: {name}" for index, (name, _) in enumerate(kernels[:16])],
            [("Kernel event", [duration for _, duration in kernels[:16]])],
        ), encoding="utf-8")
    else:
        kernel_graph = None

    graph_paths = [path for path in (region_graph, operator_graph, device_graph, kernel_graph) if path is not None]
    report_graphs = []
    attachments: dict[str, dict[str, str]] = {}
    for path in graph_paths:
        relative = path.relative_to(root).as_posix()
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        attachment_name = path.name
        attachments[attachment_name] = {"image/svg+xml": encoded}
        report_graphs.append((relative, attachment_name, path.stem.replace("_", " ")))

    region_headers = ["Region", "Ancestry", "Count", "CPU self", "CPU inclusive total", "Device self"]
    region_memory_columns = [
        ("memory_self_bytes", "CPU memory self (bytes)"),
        ("memory_total_bytes", "CPU memory total (bytes)"),
        ("device_memory_self_bytes", "Device memory self (bytes)"),
        ("device_memory_total_bytes", "Device memory total (bytes)"),
    ]
    region_memory_columns = [(column, label) for column, label in region_memory_columns
                             if any(column in row for row in region_rows)]
    region_headers.extend(label for _, label in region_memory_columns)
    region_table_rows = []
    for row in region_rows:
        values = [
            row.get("name", ""), row.get("ancestry", ""), row.get("count", ""),
            _metric_display(row, "cpu_self_us"),
            _metric_display(row, "cpu_total_us"),
            _metric_display(row, "device_self_us"),
        ]
        values.extend(_metric_display(row, column, unit="bytes") for column, _ in region_memory_columns)
        region_table_rows.append(values)
    operator_headers = ["Operator", "Count", "CPU self", "CPU total", "Input shapes"]
    operator_memory_columns = [
        ("memory_self_bytes", "CPU memory self (bytes)"),
        ("memory_total_bytes", "CPU memory total (bytes)"),
        ("device_memory_self_bytes", "Device memory self (bytes)"),
        ("device_memory_total_bytes", "Device memory total (bytes)"),
    ]
    operator_memory_columns = [(column, label) for column, label in operator_memory_columns
                               if any(column in row for row in top_operators)]
    operator_headers.extend(label for _, label in operator_memory_columns)
    operator_table_rows = []
    for row in top_operators:
        shapes = row.get("input_shapes", "not requested")
        if str(shapes).lower() == "null" and row.get("input_shapes_unavailable_reason"):
            shapes = f"null - {row['input_shapes_unavailable_reason']}"
        values = [
            row.get("name", ""), row.get("count", ""),
            _metric_display(row, "cpu_self_us"),
            _metric_display(row, "cpu_total_us"),
            shapes,
        ]
        values.extend(_metric_display(row, column, unit="bytes") for column, _ in operator_memory_columns)
        operator_table_rows.append(values)

    if cpu_only:
        device_status = "CPU-only capture: CUDA/device metrics are null where unavailable. No compute kernels are present because no CUDA kernel capture was requested."
    elif kernels:
        device_status = f"{len(kernels)} CUDA kernel events were present in this run's trace; durations are individual profiler events, not end-to-end latency."
    elif _number(capture.get("cuda_kernel_event_count")):
        count = int(capture["cuda_kernel_event_count"])
        device_status = f"{count} kernel events were observed in the manifest, but per-kernel durations are absent from the exported trace; no kernel-duration graph is shown."
    else:
        device_status = "No compute kernels were available in the exported trace; no kernel graph is shown."
    region_table = _markdown_table(region_headers, region_table_rows)
    operator_table = _markdown_table(operator_headers, operator_table_rows)
    markdown = [
        f"# KV-cache profiler report - `{root.name}`\n\n",
        "This report was rendered offline from this run's exported manifest, CSV files, and trace. It does not rerun the model or start a notebook kernel.\n\n",
        f"- Arm: `{manifest.get('selected_arm', 'unknown')}`\n",
        f"- Device: `{device}`; dtype: `{manifest.get('dtype', 'unknown')}`\n",
        f"- Prompt / requested budget / capacity: `{manifest.get('case', {}).get('prompt_length', 'unknown')}` / `{manifest.get('case', {}).get('requested_tokens', 'unknown')}` / `{manifest.get('case', {}).get('capacity', 'unknown')}`\n",
        f"- Effective tokens: `{manifest.get('effective_tokens', 'unknown')}`\n",
        f"- Capture status: `{capture.get('status', 'unknown')}`; complete: `{capture.get('complete', 'unknown')}`\n\n",
        "## Semantic region table\n\n",
        "CPU and device durations are profiler microseconds (`us`), not ordinary wrapper latency. CPU self excludes child events; CPU inclusive total includes nested work. Ancestry distinguishes repeated scopes. Rows are not additive: inclusive totals are not additive across nested parent/child rows, and CPU plus device time is never a latency sum. The region graph uses semantic leaves to avoid double-counting nested inclusive totals.\n\n",
        region_table + "\n\n",
        "## Operator table\n\n",
        "Top exported operators are ordered by CPU self time. Native operator names/shapes are evidence from the profiler; counts or shapes alone do not establish a bottleneck.\n\n",
        operator_table + "\n\n",
        f"## Device and kernel availability\n\n{device_status}\n\n",
        _markdown_table(["Kernel event", "Trace duration (us)"],
                        [[name, _format(duration, unit="us")] for name, duration in kernels[:16]]) + "\n\n" if kernels else "",
        "Kernel event durations may overlap and are not summed as generation latency. Allocation metrics describe allocator accounting, not HBM traffic or bandwidth. No normal latency is derived; use the separate repeated wall-time benchmark.\n\n",
    ]
    for relative, attachment_name, caption in report_graphs:
        markdown.append(f"### {caption}\n\n![{caption}](attachment:{attachment_name})\n\nStandalone graph: `{relative}`\n\n")
    markdown.extend([
        "## Missing data\n\n",
        "A `null` cell means the profiler did not expose that metric or the activity was unavailable; inspect the CSV's `*_unavailable_reason` field. A numeric zero is preserved as a measured zero. Device/kernel graphs are emitted only when corresponding exported device/kernel values exist.\n",
    ])
    cell = _cell("".join(markdown), attachments or None)
    notebook = {
        "cells": [cell],
        "metadata": _NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    report_path = _safe_run_file(root, "report.ipynb")
    report_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path
