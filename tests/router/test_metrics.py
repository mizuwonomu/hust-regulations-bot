"""Test chấm offline: accuracy, recall, confusion, consistency, latency, usage, cost và report.

Fixture được viết tay với số liệu biết trước nên oracle không phụ thuộc code production
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from evals.router.adapters import TokenUsage, RouteTrial
from evals.router.corpus import CORPUS_SIZE
from evals.router.metrics import (
    SUMMARY_NAME,
    ScoringError,
    main,
    read_trials,
    render_report,
    score_run,
    write_summary,
)
from evals.router.run import TRIALS_NAME, run_live

PRICING = {
    "groq": {
        "model_id": "groq-model",
        "usd_per_million_input": 1.0,
        "usd_per_million_output": 2.0,
        "source_url": "https://example.invalid/groq",
        "observed_at": "2026-09-25",
    },
    "jev": {
        "model_id": "jev-model",
        "usd_per_million_input": 1.0,
        "usd_per_million_output": 2.0,
        "source_url": "https://example.invalid/jev",
        "observed_at": "2026-09-25",
    },
}


def _row(
    case_index: int,
    arm: str,
    repeat: int,
    gold: str,
    executed: str | None,
    *,
    raw: str | None = None,
    format_valid: bool | None = None,
    usage: tuple[int | None, int | None] | None = (10, 2),
    latency_ms: float = 100.0,
    model: str | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    case_id: int | None = None,
    group: str = "primary",
) -> dict:
    if format_valid is None:
        format_valid = True if executed in {"RAG", "chat"} and error_type is None else None
    return {
        "case_index": case_index,
        "id": case_index + 1 if case_id is None else case_id,
        "group": group,
        "repeat": repeat,
        "gold": gold,
        "captured_at": "2026-09-25T00:00:00+00:00",
        "arm": arm,
        "raw_answer": raw if raw is not None else executed,
        "executed_branch": executed,
        "format_valid": format_valid,
        "usage": None
        if usage is None
        else {"input_tokens": usage[0], "output_tokens": usage[1]},
        "latency_ms": latency_ms,
        "request_id": f"{arm}-req",
        "model": model or f"{arm}-model",
        "attempts": 1,
        "error_type": error_type,
        "error_message": error_message,
        "detail": None,
    }


def _manifest(
    *,
    rows: int = 4,
    repeats: int = 3,
    arms: tuple[str, ...] = ("groq", "jev"),
    gold_counts: dict[str, int] | None = None,
    corpus_path: str | None = None,
    corpus_sha256: str = "0" * 64,
) -> dict:
    return {
        "created_at": "2026-09-25T00:00:00+00:00",
        "corpus": {
            "path": corpus_path,
            "sha256": corpus_sha256,
            "rows": rows,
            "gold_counts": gold_counts or {"RAG": rows // 2, "chat": rows - rows // 2},
        },
        "schedule": {
            "repeats": repeats,
            "arms": list(arms),
            "calls": rows * repeats * len(arms),
            "sha256": "schedule-sha",
            "rule": "arm order alternates by (case_index + repeat_index) % 2",
        },
        "request_fingerprint": "fingerprint",
        "timeout_seconds": 30.0,
        "pricing": PRICING,
    }


def _setup(tmp_path: Path, trials: list[dict], **manifest_kwargs) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    trials_path = run_dir / "trials.jsonl"
    manifest_path.write_text(
        json.dumps(_manifest(**manifest_kwargs), ensure_ascii=False), encoding="utf-8"
    )
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trials),
        encoding="utf-8",
    )
    return manifest_path, trials_path


def _unanimous(arms: tuple[str, ...] = ("groq", "jev")) -> list[dict]:
    """4 case (2 RAG, 2 chat) x 3 repeat x 2 arm, mọi nhánh đều đúng"""
    trials = []
    for arm in arms:
        for case_index in range(4):
            gold = "RAG" if case_index < 2 else "chat"
            for repeat in range(3):
                trials.append(_row(case_index, arm, repeat, gold, gold))
    return trials


class _CorpusAdapter:
    """Adapter giả trả đúng gold của corpus test 30 dòng, dùng cho test bind gold"""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def route(self, query: str) -> RouteTrial:
        index = int(query.rsplit(" ", 1)[-1])
        branch = "RAG" if index % 2 == 0 else "chat"
        return RouteTrial(
            arm=self.arm,
            raw_answer=branch,
            executed_branch=branch,
            format_valid=True,
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            latency_ms=1.0,
            request_id=f"{self.arm}-{index}",
            model="real-model",
            attempts=1,
        )


def _corpus_adapters() -> dict[str, _CorpusAdapter]:
    return {"groq": _CorpusAdapter("groq"), "jev": _CorpusAdapter("jev")}


def _write_30_case_corpus(tmp_path: Path) -> Path:
    rows = [
        {
            "id": index + 1,
            "group": "primary",
            "query": f"câu hỏi số {index}",
            "type": "RAG" if index % 2 == 0 else "chat",
        }
        for index in range(CORPUS_SIZE)
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


# Fixture chuẩn: mọi thứ đúng


def test_unanimous_run_scores_perfectly(tmp_path) -> None:
    manifest_path, trials_path = _setup(tmp_path, _unanimous())

    summary = score_run(manifest_path, trials_path)

    assert summary.run["recorded_trials"] == 24
    assert summary.run["scheduled_calls"] == 24
    for arm in ("groq", "jev"):
        stats = summary.arms[arm]
        assert stats["scheduled"] == 12
        assert stats["recorded"] == 12
        assert stats["missing"] == 0
        assert stats["errors"] == 0
        assert stats["unexpected"] == 0
        assert stats["decisions"] == 12
        assert stats["correct"] == 12
        assert stats["scheduled_accuracy"] == 1.0
        assert stats["decision_accuracy"] == 1.0
        assert stats["recall_scheduled"] == {"RAG": 1.0, "chat": 1.0}
        assert stats["recall_decided"] == {"RAG": 1.0, "chat": 1.0}
        assert stats["recall_scheduled_by_gold"] == {"RAG": 6, "chat": 6}
        assert stats["recall_decided_by_gold"] == {"RAG": 6, "chat": 6}
        assert stats["confusion"] == {
            "RAG": {"RAG": 6, "chat": 0},
            "chat": {"RAG": 0, "chat": 6},
        }
        assert stats["rag_to_chat"] == 0
        assert stats["format_valid"] == {"valid": 12, "invalid": 0, "unknown": 0}
        assert stats["consistency_mean"] == 1.0
        assert stats["consistency_by_gold"] == {"RAG": 1.0, "chat": 1.0}
        assert stats["consistency_eligible"] == 4
        assert stats["consistency_total"] == 4
        assert stats["latency_ms"] == {"p50": 100.0, "p95": 100.0, "max": 100.0}
        assert stats["usage"] == {
            "input_tokens": 120,
            "output_tokens": 24,
            "missing_usage": 0,
        }
        # 120 token vào x 1.0 + 24 token ra x 2.0 trên mỗi triệu token
        assert stats["cost_usd"] == pytest.approx((120 * 1.0 + 24 * 2.0) / 1_000_000)
        assert stats["cost_complete"] is True
        assert stats["cost_blockers"] == []
        assert stats["model_ids"] == [f"{arm}-model"]
    assert summary.disagreements == []
    assert summary.format_invalid == []
    assert summary.errors == []
    assert summary.missing == []


def test_two_of_three_split_scores_one_third(tmp_path) -> None:
    trials = [
        _row(0, "groq", 0, "RAG", "RAG"),
        _row(0, "groq", 1, "RAG", "RAG"),
        _row(0, "groq", 2, "RAG", "chat"),
        _row(1, "groq", 0, "chat", "chat"),
        _row(1, "groq", 1, "chat", "chat"),
        _row(1, "groq", 2, "chat", "chat"),
    ]
    manifest_path, trials_path = _setup(
        tmp_path,
        trials,
        rows=2,
        gold_counts={"RAG": 1, "chat": 1},
        arms=("groq",),
    )

    stats = score_run(manifest_path, trials_path).arms["groq"]

    assert stats["consistency_mean"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert stats["consistency_eligible"] == 2
    assert stats["correct"] == 5
    assert stats["scheduled_accuracy"] == pytest.approx(5 / 6)
    assert stats["recall_scheduled"] == {"RAG": pytest.approx(2 / 3), "chat": 1.0}
    assert stats["recall_decided"] == {"RAG": pytest.approx(2 / 3), "chat": 1.0}
    assert stats["confusion"]["RAG"] == {"RAG": 2, "chat": 1}
    assert stats["rag_to_chat"] == 1


def test_consistently_wrong_answers_are_consistent_but_incorrect(tmp_path) -> None:
    trials = [
        _row(0, "groq", repeat, "RAG", "chat") for repeat in range(3)
    ] + [_row(1, "groq", repeat, "chat", "chat") for repeat in range(3)]
    manifest_path, trials_path = _setup(
        tmp_path,
        trials,
        rows=2,
        gold_counts={"RAG": 1, "chat": 1},
        arms=("groq",),
    )

    summary = score_run(manifest_path, trials_path)
    stats = summary.arms["groq"]

    assert stats["consistency_mean"] == 1.0
    assert stats["scheduled_accuracy"] == pytest.approx(0.5)
    assert stats["recall_scheduled"] == {"RAG": 0.0, "chat": 1.0}
    assert stats["recall_decided"] == {"RAG": 0.0, "chat": 1.0}
    assert stats["rag_to_chat"] == 3
    assert stats["rag_to_chat_rate"] == 1.0
    assert [row["case_index"] for row in summary.disagreements] == [0, 0, 0]
    assert {row["arm"] for row in summary.disagreements} == {"groq"}
    assert summary.disagreements[0]["id"] == 1
    assert summary.disagreements[0]["group"] == "primary"


def test_debug_fields_change_annotations_but_not_scoring(tmp_path) -> None:
    trials = [_row(0, "groq", repeat, "RAG", "chat") for repeat in range(3)]
    manifest_path, trials_path = _setup(
        tmp_path,
        trials,
        rows=1,
        gold_counts={"RAG": 1, "chat": 0},
        arms=("groq",),
    )
    renamed = [
        {**row, "id": 999, "group": "nhóm-khác"} for row in trials
    ]
    other_dir = tmp_path / "run2"
    other_dir.mkdir()
    other_manifest = other_dir / "manifest.json"
    other_trials = other_dir / "trials.jsonl"
    other_manifest.write_text(
        json.dumps(_manifest(rows=1, gold_counts={"RAG": 1, "chat": 0}, arms=("groq",))),
        encoding="utf-8",
    )
    other_trials.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in renamed),
        encoding="utf-8",
    )

    original = score_run(manifest_path, trials_path)
    changed = score_run(other_manifest, other_trials)

    assert original.arms == changed.arms
    assert original.run["recorded_trials"] == changed.run["recorded_trials"]
    assert [row["id"] for row in original.disagreements] == [1, 1, 1]
    assert [row["id"] for row in changed.disagreements] == [999, 999, 999]
    assert {row["group"] for row in changed.disagreements} == {"nhóm-khác"}


# Format malformed, error, missing usage và missing trial


def test_malformed_groq_output_is_reported_separately_from_branch(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    trials[0] = _row(
        0,
        "groq",
        0,
        "RAG",
        "chat",
        raw="Chat về quy chế",
        format_valid=False,
    )
    manifest_path, trials_path = _setup(tmp_path, trials, arms=("groq",))

    summary = score_run(manifest_path, trials_path)
    stats = summary.arms["groq"]

    assert stats["format_valid"] == {"valid": 11, "invalid": 1, "unknown": 0}
    assert stats["decisions"] == 12
    assert len(summary.format_invalid) == 1
    invalid = summary.format_invalid[0]
    assert invalid["case_index"] == 0
    assert invalid["raw_answer"] == "Chat về quy chế"
    assert invalid["executed_branch"] == "chat"
    assert len(summary.disagreements) == 1


def test_api_error_stays_in_scheduled_denominator(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    trials[0] = _row(
        0,
        "groq",
        0,
        "RAG",
        None,
        usage=None,
        error_type="RateLimitError",
        error_message="quá nhiều request",
    )
    manifest_path, trials_path = _setup(tmp_path, trials, arms=("groq",))

    summary = score_run(manifest_path, trials_path)
    stats = summary.arms["groq"]

    assert stats["errors"] == 1
    assert stats["decisions"] == 11
    assert stats["correct"] == 11
    assert stats["scheduled_accuracy"] == pytest.approx(11 / 12)
    assert stats["decision_accuracy"] == 1.0
    assert stats["recall_scheduled"]["RAG"] == pytest.approx(5 / 6)
    # Trial lỗi nằm trong denominator scheduled nhưng không nằm trong denominator decided
    assert stats["recall_decided"]["RAG"] == 1.0
    assert stats["recall_decided_by_gold"] == {"RAG": 5, "chat": 6}
    assert stats["consistency_eligible"] == 3
    assert stats["consistency_mean"] == 1.0
    assert [row["error_type"] for row in summary.errors] == ["RateLimitError"]
    assert summary.errors[0]["case_index"] == 0


def test_missing_usage_makes_cost_unknown_but_latency_still_counted(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    trials[0] = _row(0, "groq", 0, "RAG", "RAG", usage=None, latency_ms=300.0)
    trials[1] = _row(0, "groq", 1, "RAG", "RAG", usage=(None, 5))
    manifest_path, trials_path = _setup(tmp_path, trials, arms=("groq",))

    stats = score_run(manifest_path, trials_path).arms["groq"]

    # Cả usage null và usage chỉ có một nửa đều tính là thiếu, nên cost là không đầy đủ
    assert stats["usage"]["missing_usage"] == 2
    assert stats["usage"]["input_tokens"] == 100
    assert stats["usage"]["output_tokens"] == 25
    assert stats["cost_usd"] is None
    assert stats["cost_complete"] is False
    assert stats["latency_ms"] == {"p50": 100.0, "p95": pytest.approx(190.0), "max": 300.0}


def test_missing_trials_are_listed_with_corpus_annotations(tmp_path) -> None:
    corpus = _write_30_case_corpus(tmp_path)
    trials = [
        _row(index, "groq", repeat, "RAG" if index % 2 == 0 else "chat", "RAG" if index % 2 == 0 else "chat")
        for index in range(CORPUS_SIZE)
        for repeat in range(1)
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    trials_path = run_dir / "trials.jsonl"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                rows=CORPUS_SIZE,
                repeats=1,
                arms=("groq",),
                gold_counts={"RAG": 15, "chat": 15},
                corpus_path=str(corpus),
                corpus_sha256=_sha256(corpus),
            )
        ),
        encoding="utf-8",
    )
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in trials[:-3]),
        encoding="utf-8",
    )

    summary = score_run(manifest_path, trials_path)
    stats = summary.arms["groq"]

    assert stats["scheduled"] == 30
    assert stats["recorded"] == 27
    assert stats["missing"] == 3
    assert [row["case_index"] for row in summary.missing] == [27, 28, 29]
    assert [row["id"] for row in summary.missing] == [28, 29, 30]
    assert {row["group"] for row in summary.missing} == {"primary"}
    assert stats["scheduled_accuracy"] == pytest.approx(27 / 30)


def test_missing_annotations_fall_back_to_null_when_corpus_changed(tmp_path) -> None:
    corpus = _write_30_case_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / "manifest.json"
    trials_path = run_dir / "trials.jsonl"
    manifest_path.write_text(
        json.dumps(
            _manifest(
                rows=CORPUS_SIZE,
                repeats=1,
                arms=("groq",),
                gold_counts={"RAG": 15, "chat": 15},
                corpus_path=str(corpus),
                corpus_sha256="f" * 64,
            )
        ),
        encoding="utf-8",
    )
    trials_path.write_text("", encoding="utf-8")

    summary = score_run(manifest_path, trials_path)

    assert len(summary.missing) == 30
    assert all(row["id"] is None and row["group"] is None for row in summary.missing)


# Guard artifact hỏng


def test_rejects_broken_artifacts(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    manifest_path, trials_path = _setup(tmp_path, trials, arms=("groq",))

    with pytest.raises(ScoringError, match="trùng key"):
        score_run(manifest_path, _write_lines(tmp_path / "dup.jsonl", [trials[0], trials[0]]))

    with pytest.raises(ScoringError, match="không thuộc manifest"):
        score_run(manifest_path, _write_lines(tmp_path / "arm.jsonl", [{**trials[0], "arm": "khác"}]))

    with pytest.raises(ScoringError, match="repeat ngoài phạm vi"):
        score_run(manifest_path, _write_lines(tmp_path / "repeat.jsonl", [{**trials[0], "repeat": 9}]))

    with pytest.raises(ScoringError, match="case index ngoài phạm vi"):
        score_run(manifest_path, _write_lines(tmp_path / "index.jsonl", [{**trials[0], "case_index": 99}]))

    with pytest.raises(ScoringError, match="gold lạ"):
        score_run(manifest_path, _write_lines(tmp_path / "gold.jsonl", [{**trials[0], "gold": "FAQ"}]))

    broken = tmp_path / "broken.jsonl"
    broken.write_text("{oops}\n", encoding="utf-8")
    with pytest.raises(ScoringError, match="không parse được"):
        read_trials(broken)
    with pytest.raises(ScoringError, match="thiếu file trials"):
        read_trials(tmp_path / "không-có.jsonl")

    foreign = tmp_path / "foreign.json"
    foreign.write_text('{"khác": 1}', encoding="utf-8")
    with pytest.raises(ScoringError, match="thiếu field corpus"):
        score_run(foreign, trials_path)


def _write_lines(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


# Ghi summary và CLI


def test_summary_is_written_exclusively(tmp_path, capsys) -> None:
    manifest_path, trials_path = _setup(tmp_path, _unanimous())

    assert main(["--manifest", str(manifest_path), "--trials", str(trials_path)]) == 0
    output = capsys.readouterr().out
    assert "Đã ghi" in output
    assert "scheduled accuracy 1.0000" in output
    summary_path = manifest_path.parent / SUMMARY_NAME
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "run",
        "cases",
        "arms",
        "disagreements",
        "format_invalid",
        "errors",
        "missing",
    }
    assert payload["run"]["recorded_trials"] == 24
    assert set(payload["arms"]) == {"groq", "jev"}

    assert main(["--manifest", str(manifest_path), "--trials", str(trials_path)]) == 1
    assert "không ghi đè" in capsys.readouterr().err

    with pytest.raises(ScoringError, match="không ghi đè"):
        write_summary(score_run(manifest_path, trials_path), manifest_path)


def test_cost_is_unknown_when_scheduled_trials_are_absent(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    manifest_path, trials_path = _setup(tmp_path, trials[:-1], arms=("groq",))

    stats = score_run(manifest_path, trials_path).arms["groq"]

    assert stats["missing"] == 1
    assert stats["usage"]["missing_usage"] == 0
    # Trial thiếu nghĩa là usage chưa biết, nên cost của arm là không đầy đủ
    assert stats["cost_usd"] is None
    assert stats["cost_complete"] is False


def test_empty_artifact_reports_unknown_cost(tmp_path) -> None:
    manifest_path, trials_path = _setup(
        tmp_path,
        [],
        rows=2,
        gold_counts={"RAG": 1, "chat": 1},
        arms=("groq",),
    )

    stats = score_run(manifest_path, trials_path).arms["groq"]

    assert stats["missing"] == 6
    assert stats["cost_usd"] is None
    assert stats["cost_complete"] is False


def test_missing_field_fails_as_scoring_error(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    manifest_path, _ = _setup(tmp_path, trials, arms=("groq",))

    for field in ("format_valid", "error_type", "latency_ms", "usage"):
        row = {key: value for key, value in trials[0].items() if key != field}
        with pytest.raises(ScoringError, match=f"thiếu field {field}"):
            score_run(manifest_path, _write_lines(tmp_path / f"no-{field}.jsonl", [row]))


def test_case_outcomes_cover_the_whole_corpus(tmp_path) -> None:
    trials = [
        _row(0, "groq", 0, "RAG", "RAG"),
        _row(0, "groq", 1, "RAG", "RAG"),
        _row(0, "groq", 2, "RAG", "chat"),
        _row(0, "jev", 0, "RAG", "RAG"),
        _row(0, "jev", 1, "RAG", None, usage=None, error_type="APITimeoutError"),
        # Case 1 không có trial nào
    ]
    manifest_path, trials_path = _setup(
        tmp_path,
        trials,
        rows=2,
        gold_counts={"RAG": 1, "chat": 1},
    )

    summary = score_run(manifest_path, trials_path)

    assert [case["case_index"] for case in summary.cases] == [0, 1]
    first = summary.cases[0]
    assert first["id"] == 1
    assert first["group"] == "primary"
    assert first["gold"] == "RAG"
    assert first["arms"]["groq"] == {
        "branches": ["RAG", "RAG", "chat"],
        "decisions": 3,
        "correct": 2,
        "errors": 0,
        "missing": 0,
        "consistency": pytest.approx(1 / 3),
    }
    assert first["arms"]["jev"] == {
        "branches": ["RAG", None],
        "decisions": 1,
        "correct": 1,
        "errors": 1,
        "missing": 1,
        "consistency": None,
    }
    second = summary.cases[1]
    assert second["arms"]["groq"]["branches"] == []
    assert second["arms"]["groq"]["missing"] == 3
    assert second["arms"]["groq"]["consistency"] is None
    assert summary.to_dict()["cases"] == summary.cases


def test_cost_is_unknown_when_served_model_differs_or_is_unknown(tmp_path) -> None:
    trials = _unanimous(arms=("groq",))
    other_model = [{**row, "model": "model-khác"} for row in trials]
    manifest_path, _ = _setup(tmp_path, trials, arms=("groq",))

    stats = score_run(
        manifest_path, _write_lines(tmp_path / "other-model.jsonl", other_model)
    ).arms["groq"]
    # Giá đã niêm yết không áp dụng cho model phục vụ khác
    assert stats["cost_usd"] is None
    assert stats["cost_blockers"] == ["served_model_not_priced"]

    unknown_model = [{**row, "model": None} for row in trials]
    stats = score_run(
        manifest_path, _write_lines(tmp_path / "no-model.jsonl", unknown_model)
    ).arms["groq"]
    assert stats["cost_usd"] is None
    assert stats["cost_blockers"] == ["served_model_unknown"]


def test_rejects_gold_mismatch_against_hashed_corpus(tmp_path) -> None:
    corpus = _write_30_case_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(
        corpus_path=corpus,
        output_dir=run_dir,
        repeats=1,
        adapters=_corpus_adapters(),
    )
    trials_path = run_dir / TRIALS_NAME
    rows = read_trials(trials_path)
    # Giữ gold hợp lệ nhưng gán sai case
    rows[0]["gold"] = "chat" if rows[0]["gold"] == "RAG" else "RAG"
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ScoringError, match="khác corpus đã hash"):
        score_run(run_dir / "manifest.json", trials_path)


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [
        (lambda row: {**row, "case_index": True}, "case_index không phải số nguyên"),
        (lambda row: {**row, "repeat": "0"}, "repeat không phải số nguyên"),
        (lambda row: {**row, "arm": 1}, "arm không phải string"),
        (lambda row: {**row, "executed_branch": "maybe"}, "executed_branch lạ"),
        (lambda row: {**row, "format_valid": "yes"}, "format_valid không phải bool"),
        (lambda row: {**row, "error_type": 5}, "error_type không phải string"),
        (
            lambda row: {**row, "error_type": "APITimeoutError", "executed_branch": "RAG"},
            "vừa có error_type",
        ),
        (
            lambda row: {**row, "error_type": "APITimeoutError", "executed_branch": None, "usage": []},
            "usage không phải object",
        ),
        (
            lambda row: {**row, "usage": {"input_tokens": 1}},
            "usage thiếu field output_tokens",
        ),
        (
            lambda row: {**row, "usage": {"input_tokens": True, "output_tokens": 2}},
            "input_tokens không phải số nguyên",
        ),
        (lambda row: {**row, "latency_ms": "12"}, "latency_ms không phải số"),
    ],
)
def test_rejects_invalid_trial_types(tmp_path, mutate, pattern: str) -> None:
    trials = _unanimous(arms=("groq",))
    manifest_path, _ = _setup(tmp_path, trials, arms=("groq",))

    with pytest.raises(ScoringError, match=pattern):
        score_run(manifest_path, _write_lines(tmp_path / "bad.jsonl", [mutate(trials[0])]))


def test_rejects_json_array_line(tmp_path) -> None:
    manifest_path, trials_path = _setup(tmp_path, _unanimous(arms=("groq",)), arms=("groq",))
    trials_path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(ScoringError, match="không phải JSON object"):
        score_run(manifest_path, trials_path)


def test_render_report_marks_unknown_values(tmp_path) -> None:
    manifest_path, trials_path = _setup(
        tmp_path,
        [],
        rows=2,
        gold_counts={"RAG": 1, "chat": 1},
        arms=("groq",),
    )

    report = render_report(score_run(manifest_path, trials_path))

    assert "n/a" in report
    assert "missing 6" in report


# Integration với artifact thật của runner


def test_run_artifacts_with_failures_can_be_scored(tmp_path) -> None:
    corpus = _write_30_case_corpus(tmp_path)
    run_dir = tmp_path / "run"

    class _Adapter:
        def __init__(self, arm: str) -> None:
            self.arm = arm

        def route(self, query: str) -> RouteTrial:
            index = int(query.rsplit(" ", 1)[-1])
            if index % 3 == 0:
                return RouteTrial(
                    arm=self.arm,
                    raw_answer=None,
                    executed_branch=None,
                    format_valid=None,
                    usage=None,
                    latency_ms=5.0,
                    request_id=None,
                    model=None,
                    attempts=1,
                    error_type="APITimeoutError",
                    error_message="hết giờ",
                )
            # Cố tình trả nhãn ngược gold để accuracy có giá trị biết trước
            branch = "chat" if index % 2 == 0 else "RAG"
            return RouteTrial(
                arm=self.arm,
                raw_answer=branch,
                executed_branch=branch,
                format_valid=True,
                usage=TokenUsage(input_tokens=100, output_tokens=3),
                latency_ms=12.0,
                request_id=f"{self.arm}-{index}",
                model="real-model",
                attempts=1,
            )

    run_live(
        corpus_path=corpus,
        output_dir=run_dir,
        repeats=1,
        adapters={"groq": _Adapter("groq"), "jev": _Adapter("jev")},
    )
    trials_path = run_dir / "trials.jsonl"
    lines = trials_path.read_text(encoding="utf-8").splitlines()
    trials_path.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")

    summary = score_run(run_dir / "manifest.json", trials_path)
    stats = summary.arms["groq"]

    assert summary.run["recorded_trials"] == 58
    assert summary.run["scheduled_calls"] == 60
    assert stats["scheduled"] == 30
    assert stats["errors"] == 10
    assert stats["decisions"] == 19
    assert stats["correct"] == 0
    assert stats["scheduled_accuracy"] == 0.0
    assert stats["decision_accuracy"] == 0.0
    assert len(summary.disagreements) == 38
    # Thiếu trial thì lấy id/group từ corpus vì hash còn khớp
    assert [row["case_index"] for row in summary.missing] == [29, 29]
    assert {row["id"] for row in summary.missing} == {30}
    # Lỗi không có usage nên cost của cả hai arm là không đầy đủ
    assert stats["cost_usd"] is None
    assert stats["cost_complete"] is False
    assert stats["model_ids"] == ["real-model"]
    # Chỉ 1 repeat nên consistency không tính được
    assert stats["consistency_mean"] is None
    assert stats["consistency_eligible"] == 0
    assert summary.format_invalid == []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
