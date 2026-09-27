"""Kiểm tra source run hop-0, successor export và xác thực source artifact"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from stop_policy_synthetic import reviewed_corpus

from stop_policy_eval import cli as stop_cli
from stop_policy_eval.artifacts import execute_run, finalize_run, load_run, prepare_run
from stop_policy_eval.cases import read_stop_cases, write_new_stop_cases
from stop_policy_eval.contracts import CollectedArticleReference, StopPolicyCase
from stop_policy_eval.later_hop import (
    export_later_hop,
    verify_later_hop_source_provenance,
    verify_export_bundle,
)
from stop_policy_eval.schedule import build_hop0_capture_plan
from stop_policy_eval.artifacts import StopArtifactError


class _FollowClient:
    """Client giả tạo một FOLLOW hop-0 theo request đã đóng băng"""

    model_name = "fake-citation-agent"
    openai_api_base = "http://fake.invalid/v1"
    temperature = 0.0
    max_tokens = 4096
    request_timeout = 60.0
    max_retries = 2
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def __init__(self, followed_dieu: int):
        self.followed_dieu = followed_dieu
        self.calls = 0

    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        return SimpleNamespace(
            content=json.dumps({"stop": False, "dieu": self.followed_dieu}),
            usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
        )


def _source_run(tmp_path: Path):
    reviewed = reviewed_corpus(tmp_path / "fixture", name="hop0-source")
    source_cases = []
    for case in reviewed.cases:
        if case.source_hop != 0:
            continue
        payload = case.model_dump(mode="json")
        payload.update(
            {
                "label_status": "draft",
                "label_reason": None,
                "label_observation_hash": None,
                "fewshot_overlap": None,
                "split": "dev" if case.question_id == 1 else "heldout",
            }
        )
        source_cases.append(StopPolicyCase.model_validate(payload))

    cases_path = tmp_path / "hop0-source-cases.jsonl"
    write_new_stop_cases(cases_path, source_cases)
    source_cases = read_stop_cases(cases_path)
    plan = build_hop0_capture_plan(
        source_cases,
        reviewed.snapshot,
        reviewed.corpus,
        cases_path=cases_path,
        snapshot_path=reviewed.seeds_path,
        split="dev",
        policy="llm",
    )
    run_dir = tmp_path / "hop0-source-run"
    manifest = prepare_run(plan, run_dir)
    client = _FollowClient(plan.traces[0].candidates[0])
    execute_run(run_dir, client_factory=lambda: client)
    assert finalize_run(run_dir) == "complete"
    return reviewed, cases_path, run_dir, client


def test_hop0_source_run_and_export_bind_actual_follow_to_successor_state(tmp_path):
    reviewed, cases_path, run_dir, client = _source_run(tmp_path)
    manifest, cases, traces, results = load_run(run_dir)
    assert manifest.run_purpose == "later_hop_capture"
    assert manifest.policies == ["llm"]
    assert manifest.repeats == 1
    assert client.calls == 1
    assert len(results) == 1
    result = results[0]
    assert result.outcome.status == "ok"
    assert result.outcome.decision.stop is False

    followed_dieu = result.outcome.decision.dieu
    output_dir = tmp_path / "hop1-export"
    export = export_later_hop(
        run_dir,
        output_dir,
        article_fetcher=lambda dieu: (
            f"Điều {dieu}. Điều được follow\nNội dung dẫn tiếp Điều 32."
        ),
    )

    assert export.source_run_id == manifest.run_id
    assert len(export.states) == 1
    assert export.terminal_outcomes == []
    state = export.states[0]
    assert state.source_run_id == manifest.run_id
    assert state.source_policy == "llm"
    assert state.source_trial_id == result.trial_id
    assert state.parent_case_id == result.trial_id.rsplit(":r", 1)[0]
    assert state.followed_dieu == followed_dieu
    assert state.candidates == [32]
    assert len(state.collected_article_refs) == 1
    assert state.collected_article_refs[0].dieu == followed_dieu
    assert len(state.followed_article_sha256) == 64
    assert not (run_dir / "summary.json").exists()
    assert "Source-only run" in (run_dir / "report.md").read_text(encoding="utf-8")
    exported_state = json.loads(
        (output_dir / "later_hop_states.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert exported_state["state_hash"] == state.state_hash

    verify_later_hop_source_provenance(
        export.states,
        manifest,
        cases,
        results,
    )
    output_path = tmp_path / "hop1-reviewed.jsonl"
    assert stop_cli.main(
        [
            "import-later-hop",
            "--seeds",
            str(reviewed.seeds_path),
            "--cases",
            str(cases_path),
            "--states",
            str(output_dir / "later_hop_states.jsonl"),
            "--source-run-dir",
            str(run_dir),
            "--output",
            str(output_path),
        ]
    ) == 0
    imported = read_stop_cases(output_path)
    hop1 = next(case for case in imported if case.source_hop == 1)
    assert hop1.label_status == "draft"
    assert hop1.followed_article_sha256 == state.followed_article_sha256
    assert hop1.source_state_hash == state.state_hash


def test_export_records_empty_successor_frontier_as_terminal_not_stop_case(tmp_path):
    _, _, run_dir, _ = _source_run(tmp_path)
    output_dir = tmp_path / "terminal-export"
    export = export_later_hop(
        run_dir,
        output_dir,
        article_fetcher=lambda dieu: f"Điều {dieu}. Điều không dẫn tiếp",
    )

    assert export.states == []
    assert len(export.terminal_outcomes) == 1
    terminal = export.terminal_outcomes[0]
    assert terminal.terminal_reason == "empty_frontier"
    assert terminal.candidates == []
    assert terminal.source_hop == 1
    assert (output_dir / "later_hop_states.jsonl").read_bytes() == b""
    assert (output_dir / "terminal_outcomes.jsonl").read_text(encoding="utf-8").count(
        "empty_frontier"
    ) == 1
    verify_export_bundle(output_dir / "later_hop_states.jsonl", run_dir)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_run_id", "run-other", "source_run_id mismatch"),
        ("source_policy", "first", "source_policy is not scheduled"),
        ("source_trial_id", "trial-other", "source trial/parent mismatch"),
        ("parent_case_id", "case-other", "source trial/parent mismatch"),
        ("followed_dieu", 31, "no matching valid FOLLOW result"),
    ],
)
def test_later_hop_import_rejects_provenance_not_in_source_artifacts(
    tmp_path, field, value, message
):
    reviewed, _, run_dir, _ = _source_run(tmp_path)
    manifest, cases, _, results = load_run(run_dir)
    output = export_later_hop(
        run_dir,
        tmp_path / f"tamper-source-{field}",
        article_fetcher=lambda dieu: f"Điều {dieu}. Dẫn tiếp Điều 32.",
    )
    state = output.states[0]
    tampered = state.model_copy(update={field: value})

    with pytest.raises(StopArtifactError, match=message):
        verify_later_hop_source_provenance([tampered], manifest, cases, results)


def test_later_hop_source_provenance_rejects_non_successor_hop(tmp_path):
    _, _, run_dir, _ = _source_run(tmp_path)
    manifest, cases, _, results = load_run(run_dir)
    output = export_later_hop(
        run_dir,
        tmp_path / "tampered-hop-export",
        article_fetcher=lambda dieu: f"Điều {dieu}. Dẫn tiếp Điều 32.",
    )
    tampered = output.states[0].model_copy(update={"source_hop": 2})

    with pytest.raises(StopArtifactError, match="one-edge states"):
        verify_later_hop_source_provenance([tampered], manifest, cases, results)


def test_later_hop_source_provenance_rejects_incomplete_collected_article_refs(tmp_path):
    _, _, run_dir, _ = _source_run(tmp_path)
    manifest, cases, _, results = load_run(run_dir)
    output = export_later_hop(
        run_dir,
        tmp_path / "tampered-lineage-export",
        article_fetcher=lambda dieu: f"Điều {dieu}. Dẫn tiếp Điều 32.",
    )
    state = output.states[0]
    unrelated = CollectedArticleReference(dieu=999, sha256="a" * 64)
    tampered = state.model_copy(
        update={"collected_article_refs": [unrelated, *state.collected_article_refs]}
    )

    with pytest.raises(StopArtifactError, match="do not cover each hop"):
        verify_later_hop_source_provenance([tampered], manifest, cases, results)


@pytest.mark.parametrize("source_kind", ["cases", "seeds"])
def test_later_hop_import_requires_exact_source_run_inputs(
    tmp_path, capsys, source_kind
):
    reviewed, cases_path, run_dir, _ = _source_run(tmp_path)
    export = export_later_hop(
        run_dir,
        tmp_path / "source-input-hash-export",
        article_fetcher=lambda dieu: f"Điều {dieu}. Dẫn tiếp Điều 32.",
    )
    cases_input = cases_path
    seeds_input = reviewed.seeds_path
    if source_kind == "cases":
        cases_input = tmp_path / "different-cases.jsonl"
        cases_input.write_bytes(cases_path.read_bytes() + b"\n")
    else:
        seeds_input = tmp_path / "different-seeds.json"
        seeds_input.write_bytes(reviewed.seeds_path.read_bytes() + b"\n")
    output = tmp_path / "must-not-import.jsonl"

    exit_code = stop_cli.main(
        [
            "import-later-hop",
            "--seeds",
            str(seeds_input),
            "--cases",
            str(cases_input),
            "--states",
            str(export.output_dir / "later_hop_states.jsonl"),
            "--source-run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert "does not match the source run" in capsys.readouterr().err
    assert not output.exists()
