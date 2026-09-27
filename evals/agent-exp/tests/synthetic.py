"""Provide deterministic synthetic records for offline agent experiment tests."""

from __future__ import annotations

import json
from pathlib import Path

from artifacts import sha256_text
from contracts import DecisionOutcome, GateCase, Trial, TrialError
from src.rag.agent.schema import Decision


QUESTION_1 = "Quy trình đăng ký học tập của hệ kỹ sư thế nào?"
QUESTION_2 = "Điều kiện bảo vệ luận án tiến sĩ?"
CONTEXT_21 = "Điều 21. Đăng ký học tập chương trình kỹ sư\nQuy trình đăng ký học tập thực hiện theo Điều 10 của Quy chế này."
CONTEXT_10 = "Điều 10. Đăng ký học tập chương trình đại học\nĐăng ký học tập là quyền và nghĩa vụ của sinh viên."
CONTEXT_42 = "Điều 42. Đánh giá luận án tiến sĩ\nĐiều kiện bảo vệ cấp cơ sở quy định tại Điều 41 và cấp Đại học tại Điều 40 của Quy chế này."


DEFAULT_ROWS = [
    {"id": 1, "question": QUESTION_1, "contexts": [CONTEXT_21, CONTEXT_10], "seed_dieu": [10, 21]},
    {"id": 2, "question": QUESTION_2, "contexts": [CONTEXT_42], "seed_dieu": [42]},
]


def make_baseline(rows: list[dict], *, agent: bool = False) -> dict:
    """Build a synthetic single-pass baseline payload."""
    return {
        "config": {"rerank_ratio": 0.45, "agent": agent},
        "agent_trajectory": None,
        "results": [
            {
                "id": row["id"],
                "query": row["question"],
                "retrieved_contexts": row["contexts"],
                "hop_scores": {"retrieved_dieu": row["seed_dieu"]},
                "agent_trace": row.get("agent_trace"),
            }
            for row in rows
        ],
    }


def make_dataset(rows: list[dict]) -> list[dict]:
    """Build a dataset roster matching synthetic baseline rows."""
    return [{"id": row["id"], "user_input": row["question"]} for row in rows]


def write_inputs(
    tmp_path: Path,
    *,
    rows: list[dict] | None = None,
    whitelist: list[int] | None = None,
    agent: bool = False,
) -> tuple[Path, Path, Path]:
    """Write synthetic baseline, dataset, and whitelist inputs."""
    rows = rows if rows is not None else DEFAULT_ROWS
    whitelist = whitelist if whitelist is not None else [3, 10, 19, 20, 21, 40, 41, 42, 45]
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline_path = tmp_path / "baseline.json"
    dataset_path = tmp_path / "dataset.json"
    whitelist_path = tmp_path / "internal_dieu.json"
    baseline_path.write_text(json.dumps(make_baseline(rows, agent=agent), ensure_ascii=False), encoding="utf-8")
    dataset_path.write_text(json.dumps(make_dataset(rows), ensure_ascii=False), encoding="utf-8")
    whitelist_path.write_text(json.dumps(whitelist), encoding="utf-8")
    return baseline_path, dataset_path, whitelist_path


def make_case(
    case_id: str,
    *,
    expected_action: str,
    candidates: list[int],
    acceptable_dieu: set[int] | None = None,
    label_status: str = "approved",
    label_reason: str | None = "approved label",
) -> GateCase:
    """Build one synthetic approved or draft gate case."""
    observation = f"observation {case_id}"
    return GateCase(
        case_id=case_id,
        dataset_id="synthetic",
        question_id=case_id,
        snapshot_id="snapshot",
        snapshot_hash="snapshot-hash",
        question=f"question {case_id}",
        observation=observation,
        observation_hash=sha256_text(observation),
        candidates=candidates,
        source_hop=0,
        source_run_id=None,
        source_policy=None,
        expected_action=expected_action,
        acceptable_dieu=acceptable_dieu or set(),
        label_reason=label_reason,
        label_status=label_status,
        split="dev",
        fewshot_overlap=None,
    )


def make_trial(trial_id: str, case: GateCase, *, repeat_id: int = 0) -> Trial:
    """Build one trial matching a synthetic case."""
    return Trial(
        trial_id=trial_id,
        case_id=case.case_id,
        repeat_id=repeat_id,
        condition="original",
        permutation_id="original",
        seed_order=[],
        candidate_order=list(case.candidates),
        observation_hash=case.observation_hash,
    )


def ok_outcome(dieu: int | None, *, latency_ms: float = 1.0, usage=None) -> DecisionOutcome:
    """Build a valid follow or STOP outcome."""
    decision = Decision(stop=True, dieu=None) if dieu is None else Decision(stop=False, dieu=dieu)
    return DecisionOutcome(status="ok", decision=decision, error=None, latency_ms=latency_ms, usage=usage)


def error_outcome(category: str, message: str = "failure", *, latency_ms: float = 2.0) -> DecisionOutcome:
    """Build a classified policy error outcome."""
    return DecisionOutcome(
        status="error",
        decision=None,
        error=TrialError(category=category, message=message),
        latency_ms=latency_ms,
        usage=None,
    )
