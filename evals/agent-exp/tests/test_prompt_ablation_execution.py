"""Kiểm tra lịch C, durability vòng đời, integrity và replay offline."""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from diagnostic_subexp.doctoral_defense_prompt_ablation import cli
from diagnostic_subexp.doctoral_defense_prompt_ablation.artifacts import (
    derive_result_score,
    execute_prompt_ablation,
    load_prompt_ablation,
    prepare_prompt_ablation,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import (
    PHASES,
    VARIANTS,
    PromptAblationResult,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.metrics import summarize_prompt_ablation
from diagnostic_subexp.doctoral_defense_prompt_ablation.schedule import build_prompt_definition
from contracts import DecisionOutcome
from src.rag.agent.schema import Decision



class Client:
    """Ghi lại request bytes và trả một decision tất định."""

    def __init__(self, *, stop: bool = True) -> None:
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}, "custom": 7}
        self.stop = stop
        self.calls: list[tuple[list, dict]] = []

    def invoke(self, prompt, **kwargs):
        self.calls.append((prompt.to_messages(), kwargs))
        decision = '{"stop": true, "dieu": null}' if self.stop else '{"stop": false, "dieu": 41}'
        return SimpleNamespace(
            content=decision,
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def prepare(tmp_path: Path, policies: list[str] | None = None):
    """Prepare bằng chứng C nhúng mà không cần đường dẫn dataset sinh ra."""
    return prepare_prompt_ablation(tmp_path / "run", policies or ["first", "llm"])


def test_schedule_pairs_rounds_exclusions_and_injected_execution(tmp_path: Path):
    """C giữ đủ hai phase, 66 treatment pair và cô lập theo từng request."""
    path, manifest, _ = prepare(tmp_path)
    manifest, cases, inputs, prompts, _ = load_prompt_ablation(path)
    assert manifest.artifact_schema_version == 3
    assert len(manifest.trials) == 132
    assert len(inputs) == 10 and len(prompts) == 20
    assert manifest.expected_policy_result_count == 264
    assert Counter(t.phase_id for t in manifest.trials) == {PHASES[0]: 60, PHASES[1]: 72}
    assert len(manifest.exclusions) == 2
    pairs = defaultdict(list)
    for trial in manifest.trials:
        pairs[trial.prompt_treatment_pair_id].append(trial)
    assert len(pairs) == 66
    for members in pairs.values():
        assert {trial.prompt_variant_id for trial in members} == set(VARIANTS)
        expected = VARIANTS if members[0].batch_id == "batch-0" else VARIANTS[::-1]
        assert [trial.prompt_variant_id for trial in members] == expected
        assert members[0].presented_candidates == members[1].presented_candidates
        assert members[0].observation_hash == members[1].observation_hash
    client = Client()
    execute_prompt_ablation(path, manifest, cases, inputs, prompts, {"llm": client})
    assert len(client.calls) == 132
    prompt_by_id = {trace.prompt_trace_id: trace for trace in prompts}
    for trial, (messages, kwargs) in zip(manifest.trials, client.calls):
        prompt = prompt_by_id[trial.prompt_trace_id]
        assert [(message.type, message.content) for message in messages] == [
            (item.role, item.content) for item in prompt.effective_messages
        ]
        assert kwargs["extra_body"]["grammar"] == prompt.grammar_text
    assert "grammar" not in client.extra_body
    summary = summarize_prompt_ablation(*load_prompt_ablation(path))
    assert summary["status"] == "complete"
    assert len(summary["c2_macro_accuracy"]) == 4


def test_offline_replay_after_renderer_drift(tmp_path: Path, monkeypatch):
    """Reload dùng message đã nhúng và không import prompt production."""
    path, manifest, _ = prepare(tmp_path, ["first"])
    import builtins

    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("src.rag.agent.prompt") or name == "src.rag.agent.gate":
            raise AssertionError("offline replay imported production code")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    loaded = load_prompt_ablation(path)
    assert summarize_prompt_ablation(*loaded)["status"] == "incomplete"


def test_reload_rejects_tampered_evidence(tmp_path: Path):
    """Sửa registry, prompt trace hoặc trial đã nhúng đều bị từ chối."""
    path, _, _ = prepare(tmp_path, ["first"])
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["registry_snapshot"]["variants"][0]["included_group_ids"] = []
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_prompt_ablation(path)

    path, _, _ = prepare(tmp_path / "trace", ["first"])
    rows = [json.loads(line) for line in (path / "prompt_traces.jsonl").read_text().splitlines()]
    rows[0]["effective_messages"][0]["content"] += "drift"
    (path / "prompt_traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        load_prompt_ablation(path)

    path, _, _ = prepare(tmp_path / "trial", ["first"])
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["trials"][0]["case_role"] = "general-follow-guard"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_prompt_ablation(path)


def test_saved_phase_selection_only(tmp_path: Path):
    """Một phase C được chọn giữ đúng số trial và trace của nó."""
    base = build_prompt_definition()
    for phase, expected_trials, expected_inputs in ((PHASES[0], 60, 5), (PHASES[1], 72, 6)):
        payload = dict(base)
        payload["phases"] = [phase]
        path, manifest, _ = prepare_prompt_ablation(
            tmp_path / phase, ["first"], definition=payload
        )
        assert len(manifest.trials) == expected_trials
        assert len(load_prompt_ablation(path)[2]) == expected_inputs


def test_error_missing_and_switch_denominators(tmp_path: Path):
    """Lỗi tổng hợp được tách khỏi slot missing và valid."""
    path, manifest, _ = prepare(tmp_path, ["llm"])
    manifest, cases, inputs, prompts, _ = load_prompt_ablation(path)
    by_case = {case.case_id: case for case in cases}
    results = []
    for index, trial in enumerate(manifest.trials[:3]):
        outcome = DecisionOutcome(
            status="ok",
            decision=Decision(stop=True, dieu=None),
            error=None,
            latency_ms=0.0,
            usage=None,
        )
        if index == 1:
            outcome = outcome.model_copy(update={"decision": Decision(stop=False, dieu=3)})
        if index == 2:
            outcome = DecisionOutcome(
                status="error",
                decision=None,
                error={"category": "timeout", "message": "timeout"},
                latency_ms=0.0,
                usage=None,
            )
        results.append(
            PromptAblationResult(
                run_id=manifest.run_id,
                trial_id=trial.trial_id,
                policy="llm",
                input_trace_id=trial.input_trace_id,
                prompt_trace_id=trial.prompt_trace_id,
                request_fingerprint=trial.request_fingerprint,
                outcome=outcome,
                **derive_result_score(by_case[trial.case_id], trial.presented_candidates, outcome),
            )
        )
    summary = summarize_prompt_ablation(manifest, cases, inputs, prompts, results)
    coverage = summary["coverage_by_policy"]["llm"]
    assert coverage["scheduled"] == 132
    assert coverage["valid"] == 2
    assert coverage["errors"] == 1


def test_prepare_rejects_execution_config_drift(tmp_path: Path):
    """Guard identity C/A từ chối execution config bị đổi."""
    base = build_prompt_definition()
    payload = dict(base)
    payload["execution_config"] = {**base["execution_config"], "temperature": 0.7}
    with pytest.raises(ValueError):
        prepare_prompt_ablation(tmp_path / "drift", ["first"], definition=payload)
    assert not (tmp_path / "drift").exists()


def test_prepare_writes_no_generated_intermediate_or_result_local_source(tmp_path: Path):
    """Prepare chỉ tạo trace và manifest result-native."""
    path, _, _ = prepare(tmp_path, ["first"])
    assert sorted(item.name for item in path.iterdir()) == [
        "input_traces.jsonl",
        "manifest.json",
        "prompt_traces.jsonl",
        "report.md",
        "summary.json",
    ]
    assert not (path / "sources").exists()


def test_distinct_trace_layers_and_complete_pair_roster(tmp_path: Path):
    """Input trace không chứa treatment identity, prompt trace thì có."""
    path, manifest, _ = prepare(tmp_path, ["first"])
    input_rows = [json.loads(line) for line in (path / "input_traces.jsonl").read_text().splitlines()]
    prompt_rows = [json.loads(line) for line in (path / "prompt_traces.jsonl").read_text().splitlines()]
    assert len(input_rows) == 10
    assert len(prompt_rows) == 20
    assert all("prompt_variant_id" not in row and "effective_messages" not in row for row in input_rows)
    assert manifest.input_trace_ids == [row["input_trace_id"] for row in input_rows]
    assert manifest.prompt_trace_ids == [row["prompt_trace_id"] for row in prompt_rows]


def test_rejects_cross_object_retarget_and_trace_inventory_drift(tmp_path: Path):
    """ID chéo object và trace row thiếu không được gán lại âm thầm."""
    path, _, _ = prepare(tmp_path, ["first"])
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["trials"][0]["prompt_trace_id"] = manifest["trials"][1]["prompt_trace_id"]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_prompt_ablation(path)

    path, _, _ = prepare(tmp_path / "missing", ["first"])
    lines = (path / "prompt_traces.jsonl").read_text().splitlines()
    (path / "prompt_traces.jsonl").write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        load_prompt_ablation(path)


def test_result_retarget_and_stale_scores_are_rejected(tmp_path: Path):
    """Result ID và điểm suy ra vẫn gắn với trial đã lưu."""
    path, manifest, _ = prepare(tmp_path, ["first"])
    manifest, cases, inputs, prompts, _ = load_prompt_ablation(path)
    execute_prompt_ablation(path, manifest, cases, inputs, prompts, {})
    rows = [json.loads(line) for line in (path / "results_first.jsonl").read_text().splitlines()]
    rows[0]["decision_correct"] = not rows[0]["decision_correct"]
    (path / "results_first.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        load_prompt_ablation(path)


def test_every_result_is_flushed_before_next_trial(tmp_path: Path, monkeypatch):
    """Mỗi result C được ghi bền vững trước invocation kế tiếp."""
    path, manifest, _ = prepare(tmp_path, ["first"])
    manifest, cases, inputs, prompts, _ = load_prompt_ablation(path)
    synced: list[str] = []
    real_fsync = os.fsync

    def fsync(fd):
        rows = (path / "results_first.jsonl").read_text().splitlines()
        assert len(rows) == len(synced) + 1
        synced.append(json.loads(rows[-1])["trial_id"])
        real_fsync(fd)

    monkeypatch.setattr("diagnostic_subexp.doctoral_defense_prompt_ablation.artifacts.os.fsync", fsync)
    execute_prompt_ablation(path, manifest, cases, inputs, prompts, {})
    assert synced == [trial.trial_id for trial in manifest.trials]


def test_cli_first_lifecycle_never_constructs_client(tmp_path: Path, monkeypatch):
    """Policy first chạy offline mà không dựng LLM client."""
    path, _, _ = prepare(tmp_path, ["first"])
    monkeypatch.setattr(cli, "_build_clients", lambda _manifest: {})
    assert cli.main(["run", "--run-dir", str(path)]) == 0
    assert cli.main(["summarize", "--run-dir", str(path)]) == 0
    manifest, _, _, _, results = load_prompt_ablation(path)
    assert manifest.status == "complete"
    assert len(results) == 132
