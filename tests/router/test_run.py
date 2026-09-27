"""Test runner: budget call của dry run, schedule tất định, không ghi đè run, resume theo fingerprint.

Adapter trong test đều là fake - không có network call nào
"""

from __future__ import annotations

import fcntl
import json
from pathlib import Path

import pytest

from evals.router.adapters import TokenUsage, RouteTrial, groq_prompt_sha256, jev_criteria_sha256
from evals.router.corpus import CORPUS_SIZE, DEFAULT_CORPUS_PATH, corpus_sha256
from evals.router.run import (
    MANIFEST_NAME,
    TRIALS_NAME,
    RunError,
    build_adapters,
    build_manifest,
    build_schedule,
    check_api_keys,
    main,
    pricing_snapshot,
    request_fingerprint,
    run_dry,
    run_live,
    schedule_sha256,
)

ROUTER_MODEL = "qwen/qwen3.6-27b"


class FakeAdapter:
    """Adapter giả: ghi lại query, có thể trả branch cố định hoặc lỗi theo query"""

    def __init__(self, arm: str, *, branch="RAG", branch_for=None, fail_for=None) -> None:
        self.arm = arm
        self.queries: list[str] = []
        self._branch = branch
        self._branch_for = branch_for
        self._fail_for = fail_for

    def route(self, query: str) -> RouteTrial:
        self.queries.append(query)
        failure = self._fail_for(query) if self._fail_for is not None else None
        if failure is not None:
            return RouteTrial(
                arm=self.arm,
                raw_answer=None,
                executed_branch=None,
                format_valid=None,
                usage=None,
                latency_ms=0.4,
                request_id=None,
                model=None,
                attempts=1,
                error_type=type(failure).__name__,
                error_message=str(failure),
            )
        branch = self._branch_for(query) if self._branch_for is not None else self._branch
        return RouteTrial(
            arm=self.arm,
            raw_answer=branch,
            executed_branch=branch,
            format_valid=True,
            usage=TokenUsage(input_tokens=10, output_tokens=2),
            latency_ms=0.5,
            request_id=f"{self.arm}-req",
            model=f"{self.arm}-model",
            attempts=1,
        )


def _fake_adapters(**kwargs) -> dict[str, FakeAdapter]:
    return {
        "groq": FakeAdapter("groq", **kwargs),
        "jev": FakeAdapter("jev", **kwargs),
    }


def _write_corpus(
    tmp_path: Path,
    *,
    name: str = "corpus.json",
    ids: list[int] | None = None,
    groups: list[str] | None = None,
    labels: list[str] | None = None,
) -> Path:
    """Ghi corpus hợp lệ 30 dòng, query giữ nguyên giữa các biến thể debug"""
    labels = labels or ["chat" if index % 2 else "RAG" for index in range(CORPUS_SIZE)]
    ids = ids or list(range(1, CORPUS_SIZE + 1))
    groups = groups or ["primary"] * CORPUS_SIZE
    rows = [
        {
            "id": ids[index],
            "group": groups[index],
            "query": f"câu hỏi số {index}",
            "type": labels[index],
        }
        for index in range(CORPUS_SIZE)
    ]
    path = tmp_path / name
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _read_rows(trials_path: Path) -> list[dict]:
    return [json.loads(line) for line in trials_path.read_text(encoding="utf-8").splitlines() if line]


# Dry run và schedule


def test_dry_run_reports_180_calls_without_touching_adapters(tmp_path, capsys, monkeypatch) -> None:
    def explode():
        raise AssertionError("dry run không được dựng adapter thật")

    monkeypatch.setattr("evals.router.run.build_adapters", explode)
    corpus = _write_corpus(tmp_path)

    result = run_dry(corpus, repeats=3)

    output = capsys.readouterr().out
    assert len(result["slots"]) == CORPUS_SIZE * 3 * 2 == 180
    assert f"= 180 call" in output
    assert "case 00 (id 1, primary, gold RAG): r0 groq -> jev" in output
    assert "r1 jev -> groq" in output


def test_schedule_alternates_arm_order_deterministically() -> None:
    slots = build_schedule(2, 3)

    assert [(slot.arm) for slot in slots[:6]] == ["groq", "jev", "jev", "groq", "groq", "jev"]
    assert [(slot.arm) for slot in slots[6:]] == ["jev", "groq", "groq", "jev", "jev", "groq"]
    assert slots == build_schedule(2, 3)
    assert len({slot.key for slot in slots}) == len(slots)


def test_cli_dry_run_exit_code_and_live_requires_output_dir(capsys) -> None:
    assert main(["--dry-run", "--repeats", "1"]) == 0
    assert "= 60 call" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--live"])
    with pytest.raises(SystemExit):
        main(["--resume", "--dry-run"])


# Live run, manifest và refusal to overwrite


def test_live_run_writes_manifest_and_trials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-value")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret-value")
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    adapters = _fake_adapters(branch="RAG")

    result = run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=adapters)

    rows = _read_rows(run_dir / TRIALS_NAME)
    assert len(rows) == CORPUS_SIZE * 2 == 60
    assert {(row["case_index"], row["arm"], row["repeat"]) for row in rows} == {
        slot.key for slot in build_schedule(CORPUS_SIZE, 1)
    }
    assert len(result["rows"]) == 60
    first = rows[0]
    assert first["arm"] == "groq"
    assert first["case_index"] == 0
    assert first["id"] == 1
    assert first["group"] == "primary"
    assert first["repeat"] == 0
    assert first["gold"] == "RAG"
    assert first["executed_branch"] == "RAG"
    assert first["format_valid"] is True
    assert first["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert first["model"] == "groq-model"
    assert first["attempts"] == 1
    assert result["models"] == {"groq": ["groq-model"], "jev": ["jev-model"]}

    raw_manifest = (run_dir / MANIFEST_NAME).read_text(encoding="utf-8")
    assert "groq-secret-value" not in raw_manifest
    assert "typesafe-secret-value" not in raw_manifest


def test_manifest_records_bindings(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)

    manifest = build_manifest(
        corpus_path=corpus,
        cases=[],
        repeats=3,
        slots=build_schedule(CORPUS_SIZE, 3),
        created_at="2026-09-25T00:00:00+00:00",
    )

    assert manifest["created_at"] == "2026-09-25T00:00:00+00:00"
    assert manifest["corpus"]["sha256"] == corpus_sha256(corpus)
    assert manifest["schedule"] == {
        "repeats": 3,
        "arms": ["groq", "jev"],
        "calls": 180,
        "sha256": schedule_sha256(build_schedule(CORPUS_SIZE, 3)),
        "rule": "arm order alternates by (case_index + repeat_index) % 2",
    }
    assert manifest["request_fingerprint"] == request_fingerprint(
        corpus_path=corpus,
        slots=build_schedule(CORPUS_SIZE, 3),
        repeats=3,
    )
    assert manifest["groq"]["prompt_sha256"] == groq_prompt_sha256()
    assert manifest["jev"]["criteria_sha256"] == jev_criteria_sha256()
    assert manifest["groq"] == {
        "model": ROUTER_MODEL,
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_retries": 0,
        "prompt_sha256": manifest["groq"]["prompt_sha256"],
    }
    assert manifest["jev"]["model"] == "jev-1.13.0"
    assert manifest["jev"]["question_id"] == "route"
    assert set(manifest["jev"]["criteria"]) == {"RAG", "chat"}
    assert manifest["timeout_seconds"] == 30.0
    assert manifest["git"]["revision"] is None or len(manifest["git"]["revision"]) == 40
    assert manifest["packages"]["langchain-typesafe"] == "0.0.1a3"
    for arm in ("groq", "jev"):
        pricing = manifest["pricing"][arm]
        assert pricing["observed_at"] == "2026-09-25"
        assert pricing["source_url"].startswith("https://")
        assert isinstance(pricing["usd_per_million_input"], float)
        assert isinstance(pricing["usd_per_million_output"], float)


def test_live_refuses_to_overwrite_existing_run(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(RunError, match="không ghi đè"):
        run_live(
            corpus_path=corpus,
            output_dir=run_dir,
            repeats=1,
            adapters=_fake_adapters(),
        )


def test_live_trial_order_follows_schedule(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"

    result = run_live(
        corpus_path=corpus,
        output_dir=run_dir,
        repeats=3,
        adapters=_fake_adapters(),
    )

    expected = [
        (slot.case_index, slot.arm, slot.repeat_index) for slot in build_schedule(CORPUS_SIZE, 3)
    ]
    rows = _read_rows(run_dir / TRIALS_NAME)
    assert [(row["case_index"], row["arm"], row["repeat"]) for row in rows] == expected
    assert [(row["case_index"], row["arm"], row["repeat"]) for row in result["rows"]] == expected


def test_publish_leaves_no_staging_dir_and_creates_both_artifacts(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"

    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())

    assert sorted(entry.name for entry in run_dir.iterdir()) == [MANIFEST_NAME, TRIALS_NAME]
    assert not (tmp_path / f".{run_dir.name}.staging").exists()
    assert (tmp_path / f".{run_dir.name}.lock").exists()


def test_resume_tolerates_torn_last_line_but_not_missing_file(tmp_path, capsys) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    trials_path = run_dir / TRIALS_NAME
    rows = _read_rows(trials_path)
    # Dòng cuối bị đứt (không có newline) như khi tiến trình chết giữa lúc ghi
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[:-1]) + '{"case_index": 29,',
        encoding="utf-8",
    )
    resume_adapters = _fake_adapters()

    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=resume_adapters)

    assert "bỏ dòng cuối bị đứt" in capsys.readouterr().out
    assert len(_read_rows(trials_path)) == 60
    # Dòng bị cắt chính là call cuối của case 29, chỉ nó bị gọi lại
    assert resume_adapters["groq"].queries == ["câu hỏi số 29"]
    assert resume_adapters["jev"].queries == []

    # File rỗng là hợp lệ: chưa ghi được trial nào thì gọi lại toàn bộ là đúng
    trials_path.write_text("", encoding="utf-8")
    refill_adapters = _fake_adapters()
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=refill_adapters)
    assert len(refill_adapters["groq"].queries) + len(refill_adapters["jev"].queries) == 60


def test_resume_refuses_when_trials_file_is_missing(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    (run_dir / TRIALS_NAME).unlink()
    resume_adapters = _fake_adapters()

    # Mất file artifact phải fail, nếu không resume sẽ gọi lại toàn bộ schedule đã trả tiền
    with pytest.raises(RunError, match="artifact hỏng"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=resume_adapters)
    assert resume_adapters["groq"].queries == []
    assert resume_adapters["jev"].queries == []


def test_resume_refuses_on_complete_but_corrupt_last_line(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    trials_path = run_dir / TRIALS_NAME
    rows = _read_rows(trials_path)

    # Dòng hoàn chỉnh (có newline) nhưng JSON hỏng không phải torn write, phải fail
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[:-1]) + "{oops}\n",
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="không parse được"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())


def test_run_lock_blocks_second_writer(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    lock_path = tmp_path / f".{run_dir.name}.lock"

    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RunError, match="đang được process khác ghi"):
            run_live(
                corpus_path=corpus,
                output_dir=run_dir,
                repeats=1,
                resume=True,
                adapters=_fake_adapters(),
            )


def test_resume_rejects_rows_with_unhashable_key(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    trials_path = run_dir / TRIALS_NAME
    rows = _read_rows(trials_path)
    rows[0]["case_index"] = []
    trials_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="case_index không phải số nguyên"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())


def test_build_adapters_wires_requested_models_without_network(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-value")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret-value")

    adapters = build_adapters()

    assert set(adapters) == {"groq", "jev"}
    assert adapters["groq"].arm == "groq"
    assert adapters["groq"].model == ROUTER_MODEL
    assert adapters["groq"].timeout == 30.0
    assert adapters["jev"].arm == "jev"
    assert adapters["jev"].model == "jev-1.13.0"


def test_pricing_snapshot_fails_fast_for_unknown_model() -> None:
    with pytest.raises(RunError, match="chưa có giá cho model"):
        pricing_snapshot(groq_model="model-chưa-có-giá")


def test_cli_live_passes_resume_flag_through(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "groq-secret-value")
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret-value")
    calls: list[dict] = []

    def fake_run_live(**kwargs) -> dict:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("evals.router.run.run_live", fake_run_live)
    run_dir = tmp_path / "run"

    exit_code = main(
        [
            "--live",
            "--resume",
            "--output-dir",
            str(run_dir),
            "--repeats",
            "2",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "corpus_path": str(DEFAULT_CORPUS_PATH),
            "output_dir": str(run_dir),
            "repeats": 2,
            "resume": True,
        }
    ]
    assert capsys.readouterr().err == ""


def test_cli_live_reports_missing_keys_without_traceback(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"

    exit_code = main(
        ["--live", "--corpus", str(corpus), "--output-dir", str(run_dir), "--repeats", "1"]
    )

    assert exit_code == 1
    assert "thiếu API key" in capsys.readouterr().err
    assert not run_dir.exists()


def test_live_preflight_fails_before_creating_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-secret-value")
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"

    with pytest.raises(RunError, match="thiếu API key.*Groq"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=None)
    assert not run_dir.exists()


def test_check_api_keys_ignores_blank_values() -> None:
    with pytest.raises(RunError, match="TYPESAFE_API_KEY"):
        check_api_keys({"GROQ_API_KEY": "a", "TYPESAFE_API_KEY": "   "})
    check_api_keys({"GROQ_API_KEY": "a", "TYPESAFE_API_KEY": "b"})


# Resume


def test_resume_finishes_only_missing_calls(tmp_path, capsys) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    rows = _read_rows(run_dir / TRIALS_NAME)
    kept = rows[:-2]
    (run_dir / TRIALS_NAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    resume_adapters = _fake_adapters()

    run_live(
        corpus_path=corpus,
        output_dir=run_dir,
        repeats=1,
        resume=True,
        adapters=resume_adapters,
    )

    resumed = _read_rows(run_dir / TRIALS_NAME)
    assert len(resumed) == 60
    assert len({(row["case_index"], row["arm"], row["repeat"]) for row in resumed}) == 60
    assert len(resume_adapters["groq"].queries) + len(resume_adapters["jev"].queries) == 2
    assert "còn 2 call" in capsys.readouterr().out


def test_resume_rejects_duplicate_trial_keys(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    rows = _read_rows(run_dir / TRIALS_NAME)
    (run_dir / TRIALS_NAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*rows, rows[0]]),
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="trùng key"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())


def test_resume_rejects_trial_gold_mismatch(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())
    rows = _read_rows(run_dir / TRIALS_NAME)
    rows[0]["gold"] = "chat"
    (run_dir / TRIALS_NAME).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(RunError, match="khác corpus"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())


def test_resume_rejects_changed_repeats(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters=_fake_adapters())

    with pytest.raises(RunError, match="resume không khớp schedule.repeats"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=2, resume=True, adapters=_fake_adapters())


def test_resume_rejects_missing_or_foreign_manifest(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RunError, match="thiếu"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())
    (run_dir / MANIFEST_NAME).write_text('{"khác": 1}', encoding="utf-8")
    with pytest.raises(RunError, match="không phải manifest"):
        run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=_fake_adapters())


# Debug id/group không được ảnh hưởng gì ngoài hash file nguồn


def test_debug_fields_do_not_change_requests_or_resume_identity(tmp_path) -> None:
    corpus_a = _write_corpus(tmp_path, name="a.json")
    corpus_b = _write_corpus(
        tmp_path,
        name="b.json",
        ids=[500 + index for index in range(CORPUS_SIZE)],
        groups=["nhóm-khác"] * CORPUS_SIZE,
    )
    adapters_a = _fake_adapters()
    adapters_b = _fake_adapters()
    run_dir_a = tmp_path / "run_a"
    run_dir_b = tmp_path / "run_b"

    run_live(corpus_path=corpus_a, output_dir=run_dir_a, repeats=1, adapters=adapters_a)
    run_live(corpus_path=corpus_b, output_dir=run_dir_b, repeats=1, adapters=adapters_b)

    assert adapters_a["groq"].queries == adapters_b["groq"].queries
    assert adapters_a["jev"].queries == adapters_b["jev"].queries
    rows_a = _read_rows(run_dir_a / TRIALS_NAME)
    rows_b = _read_rows(run_dir_b / TRIALS_NAME)
    assert len(rows_a) == len(rows_b)
    assert [(row["case_index"], row["arm"], row["repeat"]) for row in rows_a] == [
        (row["case_index"], row["arm"], row["repeat"]) for row in rows_b
    ]
    assert [row["gold"] for row in rows_a] == [row["gold"] for row in rows_b]
    assert [row["id"] for row in rows_b][:4] == [500, 500, 501, 501]
    assert [row["group"] for row in rows_b][:4] == ["nhóm-khác"] * 4
    manifest_a = json.loads((run_dir_a / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest_b = json.loads((run_dir_b / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest_a["schedule"]["sha256"] == manifest_b["schedule"]["sha256"]
    assert manifest_a["corpus"]["sha256"] != manifest_b["corpus"]["sha256"]

    with pytest.raises(RunError, match="resume không khớp corpus.sha256"):
        run_live(corpus_path=corpus_b, output_dir=run_dir_a, repeats=1, resume=True, adapters=_fake_adapters())


# Lỗi provider vẫn ghi được artifact và resume không gọi lại


def test_failures_are_persisted_and_resume_does_not_repeat_them(tmp_path) -> None:
    corpus = _write_corpus(tmp_path)
    run_dir = tmp_path / "run"

    def fail_for(query: str) -> Exception | None:
        return TimeoutError("hết giờ") if query.endswith(("0", "2", "4")) else None

    groq = FakeAdapter("groq", branch_for=lambda query: "chat" if query.endswith("1") else "RAG", fail_for=fail_for)
    jev = FakeAdapter("jev", branch_for=lambda query: "RAG", fail_for=fail_for)
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, adapters={"groq": groq, "jev": jev})

    rows = _read_rows(run_dir / TRIALS_NAME)
    errors = [row for row in rows if row["error_type"]]
    assert errors
    assert all(row["executed_branch"] is None for row in errors)
    assert all(row["error_type"] == "TimeoutError" for row in errors)
    assert all(row["usage"] is None for row in errors)
    assert all(row["error_message"] == "hết giờ" for row in errors)

    resume_adapters = _fake_adapters()
    run_live(corpus_path=corpus, output_dir=run_dir, repeats=1, resume=True, adapters=resume_adapters)

    assert resume_adapters["groq"].queries == []
    assert resume_adapters["jev"].queries == []
    assert len(_read_rows(run_dir / TRIALS_NAME)) == 60
