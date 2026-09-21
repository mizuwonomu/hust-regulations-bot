"""Kiểm tra gửi request thật, durability và reload offline của diagnostic A/B."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(AGENT_EXP_ROOT.parents[1]))

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.article_42_observation_block_order.artifacts import (
    load_observation_run,
)
from diagnostic_subexp.article_42_observation_block_order.metrics import (
    summarize_observation_run,
)
from diagnostic_subexp.article_42_observation_block_order.report import render_observation_report
from diagnostic_subexp.article_42_observation_block_order.schedule import read_observation_spec
from diagnostic_subexp.article_42_observation_block_order.transforms import (
    reconstruct_observation_request,
)
from diagnostic_subexp.candidate_pair_41_3_position import cli as candidate_cli
from diagnostic_subexp.candidate_pair_41_3_position.artifacts import (
    DiagnosticArtifactError,
    load_candidate_run,
)
from diagnostic_subexp.candidate_pair_41_3_position.metrics import summarize_candidate_run
from diagnostic_subexp.candidate_pair_41_3_position.report import render_candidate_report
from diagnostic_subexp.candidate_pair_41_3_position.schedule import read_candidate_spec
from diagnostic_subexp.shared import execution as diagnostic_policy
from diagnostic_subexp.shared.execution import execute_diagnostic_request, request_from_trace
from artifacts import read_snapshot
from seed_cases import policy_input, read_cases
from src.rag.agent.gate import decide_next


class _FakeClient:
    """Client chỉ ghi lại request, không chạm mạng hay store."""

    def __init__(self, responder):
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.responder = responder
        self.calls = 0
        self.invocations: list[tuple[list, dict]] = []

    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        self.invocations.append((prompt.to_messages(), dict(extra_body or {})))
        result = self.responder(self.calls)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(content=result)


def _messages(records) -> list[tuple[str, str]]:
    return [(record.role, record.content) for record in records]


def _base_case(spec, cases):
    return next(case for case in cases if case.case_id == spec.source_case_identity.case_id)


def _prepare(tmp_path: Path, kind: str, policies=("first", "llm")):
    """Prepare A hoặc B và trả về CLI, loader, metric, report và run parts."""

    def parts(outcome):
        return outcome.run_dir, outcome.manifest, outcome.plan

    if kind == "a":
        outcome = candidate_cli.prepare_run(None, tmp_path / "run", list(policies))
        return (
            candidate_cli,
            load_candidate_run,
            summarize_candidate_run,
            render_candidate_report,
            read_candidate_spec,
            parts(outcome),
        )
    outcome = observation_cli.prepare_run(None, tmp_path / "run", list(policies))
    return (
        observation_cli,
        load_observation_run,
        summarize_observation_run,
        render_observation_report,
        read_observation_spec,
        parts(outcome),
    )


def _follow_all(_call: int) -> str:
    return '{"stop": false, "dieu": 41}'


def test_saved_messages_and_grammar_are_delivered_exactly(tmp_path: Path):
    _, loader, _, _, read_spec, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    spec = read_spec(None)
    case = _base_case(spec, cases)
    trace = next(trace for trace in traces if trace.arm_id == "a1-pair-23-3-first")
    request = request_from_trace(trace, case)

    client = _FakeClient(lambda _call: '{"stop": false, "dieu": 41}')
    outcome = execute_diagnostic_request(request, policy="llm", client=client)
    assert outcome.status == "ok"
    assert outcome.decision is not None and outcome.decision.dieu == 41
    messages, extra_body = client.invocations[0]
    assert [(message.type, message.content) for message in messages] == _messages(
        trace.effective_messages
    )
    assert extra_body["grammar"] == trace.grammar_text
    assert extra_body["chat_template_kwargs"] == {"enable_thinking": False}


def test_original_control_parity_with_decide_next(tmp_path: Path):
    _, loader, _, _, read_spec, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    spec = read_spec(None)
    case = _base_case(spec, cases)
    trace = next(trace for trace in traces if trace.arm_id == "a0-original-control")

    direct = _FakeClient(lambda _call: '{"stop": false, "dieu": 41}')
    decide_next(case.question, case.observation, list(case.candidates), client=direct)
    stored = _FakeClient(lambda _call: '{"stop": false, "dieu": 41}')
    execute_diagnostic_request(request_from_trace(trace, case), policy="llm", client=stored)

    direct_messages, direct_extra = direct.invocations[0]
    stored_messages, stored_extra = stored.invocations[0]
    assert [(message.type, message.content) for message in direct_messages] == _messages(
        trace.effective_messages
    )
    assert direct_messages == stored_messages
    assert direct_extra["grammar"] == stored_extra["grammar"] == trace.grammar_text


def test_a_changes_the_final_candidate_text_only(tmp_path: Path):
    _, loader, _, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    control = next(trace for trace in traces if trace.arm_id == "a0-original-control")
    treatment = next(trace for trace in traces if trace.arm_id == "a1-pair-23-3-first")
    assert [(m.role, m.content) for m in control.effective_messages[:-1]] == [
        (m.role, m.content) for m in treatment.effective_messages[:-1]
    ]
    left = control.effective_messages[-1].content.splitlines()
    right = treatment.effective_messages[-1].content.splitlines()
    assert len(left) == len(right)
    differences = [(a, b) for a, b in zip(left, right) if a != b]
    assert len(differences) == 1
    assert "Điều" in differences[0][0] and "Điều" in differences[0][1]


def test_b_changes_the_observation_segment_only(tmp_path: Path):
    _, loader, _, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "b")
    _, cases, traces, _ = loader(run_dir)
    control = next(trace for trace in traces if trace.arm_id == "b0-original-control")
    treatment = next(trace for trace in traces if trace.arm_id == "b1-reversed-blocks")
    assert [(m.role, m.content) for m in control.effective_messages[:-1]] == [
        (m.role, m.content) for m in treatment.effective_messages[:-1]
    ]
    control_final = control.effective_messages[-1].content
    treatment_final = treatment.effective_messages[-1].content
    assert control_final != treatment_final
    assert control_final.split("Unvisited candidates:")[1] == treatment_final.split(
        "Unvisited candidates:"
    )[1]
    university = "NCS được bảo vệ luận án cấp Đại học"
    foundation = "Điều 41 Quy chế này"
    assert treatment_final.index(university) < treatment_final.index(foundation)


def test_first_policy_makes_zero_model_calls(tmp_path: Path):
    _, loader, _, _, read_spec, (run_dir, manifest, plan) = _prepare(
        tmp_path, "a", policies=("first",)
    )
    _, cases, traces, _ = loader(run_dir)
    spec = read_spec(None)
    case = _base_case(spec, cases)
    trace = traces[0]
    client = _FakeClient(
        lambda _call: (_ for _ in ()).throw(AssertionError("no model call expected"))
    )
    outcome = execute_diagnostic_request(request_from_trace(trace, case), policy="first", client=client)
    assert client.calls == 0
    assert outcome.status == "ok"
    assert outcome.decision is not None
    assert outcome.decision.dieu == trace.presented_candidates[0]


def test_failure_durability_keeps_distinct_outcomes(tmp_path: Path):
    cli, loader, summarize, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, results = loader(run_dir)
    assert results == []

    def responder(call: int):
        if call == 1:
            return '{"stop": false, "dieu": 41}'
        if call == 2:
            return '{"stop": true, "dieu": null}'
        if call == 3:
            return '{"stop": false, "dieu": 99}'
        if call == 4:
            return httpx.ConnectError("offline")
        return '{"stop": false, "dieu": 41}'

    client = _FakeClient(responder)
    cli.execute_schedule(run_dir, manifest, cases, traces, {"llm": client})
    status = cli.finalize_run(run_dir)
    assert status == "completed_with_errors"

    manifest2, cases2, traces2, results2 = loader(run_dir)
    summary = summarize(manifest2, cases2, traces2, results2)
    assert summary.scheduled == 60
    assert summary.valid == 58
    assert summary.errors == 2
    assert summary.missing == 0
    llm_results = [result for result in results2 if result.policy == "llm"]
    categories = {
        result.outcome.error.category
        for result in llm_results
        if result.outcome.status == "error"
    }
    assert categories == {"invalid_candidate", "transport"}
    for result in llm_results:
        if result.outcome.status == "error":
            assert result.outcome.decision is None
            assert result.selected_article is None
            assert result.selected_position is None
    stop_rows = [
        row for row in summary.action_counts if row.policy == "llm" and row.batch_id is None
    ]
    assert sum(row.action_counts.get("STOP", 0) for row in stop_rows) == 1


def test_second_execution_is_refused(tmp_path: Path):
    cli, loader, _, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    cli.execute_schedule(run_dir, manifest, cases, traces, {"llm": _FakeClient(_follow_all)})
    with pytest.raises(DiagnosticArtifactError, match="already contains result records"):
        cli.execute_run(run_dir)


def test_offline_replay_survives_prompt_drift(tmp_path: Path, monkeypatch):
    cli, loader, summarize, render, _, (run_dir, manifest, plan) = _prepare(tmp_path, "b")
    _, cases, traces, _ = loader(run_dir)
    cli.execute_schedule(run_dir, manifest, cases, traces, {"llm": _FakeClient(_follow_all)})
    assert cli.finalize_run(run_dir) == "complete"
    before = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    def _drift(*args, **kwargs):
        raise AssertionError("current prompt rendering must not be required for replay")

    monkeypatch.setattr(diagnostic_policy, "build_effective_messages", _drift)
    monkeypatch.setattr(diagnostic_policy, "render_effective_prompt", _drift)

    manifest2, cases2, traces2, results2 = loader(run_dir)
    summary2 = summarize(manifest2, cases2, traces2, results2)
    assert summary2.model_dump(mode="json") == before
    report = render(manifest2, traces2, results2, summary2)
    for trace in traces2:
        assert trace.original_observation in report
        assert trace.transformed_observation in report
    assert not (run_dir / "debug").exists()


def test_interrupted_run_keeps_missing_slots_distinct_from_errors(tmp_path: Path):
    cli, loader, summarize, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    client = _FakeClient(_follow_all)
    cli.execute_schedule(
        run_dir,
        manifest.model_copy(update={"policies": ["llm"]}),
        cases,
        traces,
        {"llm": client},
    )
    assert client.calls == len(manifest.trials)
    assert cli.finalize_run(run_dir) == "incomplete"
    manifest2, cases2, traces2, results2 = loader(run_dir)
    summary = summarize(manifest2, cases2, traces2, results2)
    assert summary.scheduled == 60
    assert summary.valid == 30
    assert summary.errors == 0
    assert summary.missing == 30
    assert all(result.policy == "llm" for result in results2)


def test_original_control_fingerprint_is_shared_across_interventions(tmp_path: Path):
    _, _, _, _, _, (a_run, _, _) = _prepare(tmp_path / "a", "a")
    _, _, _, _, _, (b_run, _, _) = _prepare(tmp_path / "b", "b")
    _, _, a_traces, _ = load_candidate_run(a_run)
    _, _, b_traces, _ = load_observation_run(b_run)
    a_control = next(trace for trace in a_traces if trace.arm_id == "a0-original-control")
    b_control = next(trace for trace in b_traces if trace.arm_id == "b0-original-control")
    assert a_control.request_fingerprint == b_control.request_fingerprint
    assert a_control.input_trace_id != b_control.input_trace_id


def test_reconstruct_diagnostic_input_matches_the_stored_trace():
    from diagnostic_subexp.article_42_observation_block_order.schedule import (
        plan_observation_run,
        resolve_observation_sources,
    )

    spec = read_observation_spec(None)
    sources = resolve_observation_sources(spec)
    cases = read_cases(sources["case"])
    snapshot = read_snapshot(sources["snapshot"])
    plan = plan_observation_run(spec, cases, snapshot, ["first", "llm"])
    case = _base_case(spec, cases)
    for arm in spec.arms:
        trace = next(item for item in plan.traces if item.arm_id == arm.arm_id)
        request = reconstruct_observation_request(policy_input(case), arm, spec)
        assert request.effective_messages == trace.effective_messages
        assert request.grammar_text == trace.grammar_text
        assert request.observation == trace.transformed_observation
        assert request.presented_candidates == trace.presented_candidates


def test_reload_rejects_a_tampered_result(tmp_path: Path):
    cli, loader, _, _, _, (run_dir, manifest, plan) = _prepare(tmp_path, "a")
    _, cases, traces, _ = loader(run_dir)
    cli.execute_schedule(run_dir, manifest, cases, traces, {"llm": _FakeClient(_follow_all)})
    path = run_dir / "results_llm.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    lines[0]["decision_correct"] = not lines[0]["decision_correct"]
    path.write_text(
        "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(DiagnosticArtifactError, match="stale decision_correct"):
        load_candidate_run(run_dir)
