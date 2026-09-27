"""Chọn split đã duyệt và dựng trace cùng lịch replay tất định cho STOP policy

Module chỉ nhận case đã qua validate review: case draft, overlap chưa quyết,
split chưa gán và lớp yêu cầu còn thiếu đều bị từ chối hoặc loại tường minh
"""

from __future__ import annotations

from pathlib import Path

from contracts import SeedSnapshot, SourceFile

from stop_policy_eval.cases import StopCorpus, source_file, validate_reviewed_cases
from stop_policy_eval.contracts import (
    BASELINE_VARIANT,
    REQUIREMENT_CLASSES,
    SPLIT_ORDER,
    VARIANT_RECIPES,
    StopExclusion,
    StopPlan,
    StopPolicyCase,
    StopReviewSelection,
    StopTrial,
    resolve_variant,
)
from stop_policy_eval.execution import VALID_POLICIES, render_baseline_trace

UNASSIGNED_SPLIT_ERROR = "split must be dev or heldout"


def _validate_policies(policies: list[str]) -> list[str]:
    policy_list = list(policies)
    if not policy_list:
        raise ValueError("at least one policy is required")
    if len(set(policy_list)) != len(policy_list):
        raise ValueError("policies must not contain duplicates")
    for policy in policy_list:
        if policy not in VALID_POLICIES:
            raise ValueError(f"unsupported policy: {policy!r}")
    return policy_list


def _validate_repeats(repeats: int) -> int:
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")
    return repeats


def _validate_split(split: str) -> str:
    if split not in SPLIT_ORDER:
        raise ValueError(f"{UNASSIGNED_SPLIT_ERROR}: {split!r}")
    return split


def _select_split(
    selection: StopReviewSelection,
    split: str,
) -> list[StopPolicyCase]:
    """Lấy case đủ điều kiện của một split, giữ nguyên thứ tự case file"""
    return [case for case in selection.eligible_cases if case.split == split]


def select_stop_cases(
    cases: list[StopPolicyCase],
    snapshot: SeedSnapshot,
    corpus: StopCorpus | None,
    *,
    split: str,
) -> tuple[StopReviewSelection, list[StopPolicyCase]]:
    """Validate review rồi tách case chạy được của split được yêu cầu"""
    canonical_split = _validate_split(split)
    selection = validate_reviewed_cases(snapshot, corpus, cases)
    return selection, _select_split(selection, canonical_split)


def _trials_for(cases: list[StopPolicyCase], *, repeats: int, variant: str) -> list[StopTrial]:
    trials: list[StopTrial] = []
    for case in cases:
        trace = render_baseline_trace(case, variant)
        for repeat_id in range(repeats):
            trials.append(
                StopTrial(
                    trial_id=f"{case.case_id}:r{repeat_id}",
                    input_trace_id=trace.input_trace_id,
                    case_id=case.case_id,
                    variant=variant,
                    repeat_id=repeat_id,
                    input_fingerprint=trace.input_fingerprint,
                )
            )
    return trials


def plan_selected_trials(
    cases: list[StopPolicyCase],
    snapshot: SeedSnapshot,
    *,
    split: str,
    repeats: int,
    variant: str,
    corpus: StopCorpus | None = None,
    require_complete: bool = False,
) -> list[StopTrial]:
    """Dựng lại lịch replay từ case đã duyệt, không đọc result hay output model"""
    canonical_variant = resolve_variant(variant)
    repeat_count = _validate_repeats(repeats)
    selection, selected = select_stop_cases(cases, snapshot, corpus, split=split)
    if require_complete:
        readiness = _readiness_for_split(selection, split)
        if not readiness.complete:
            raise ValueError(
                f"split {split} cannot be prepared; missing requirement classes: "
                f"{readiness.missing()}"
            )
    return _trials_for(selected, repeats=repeat_count, variant=canonical_variant)


def _readiness_for_split(selection: StopReviewSelection, split: str):
    for readiness in selection.readiness:
        if readiness.split == split:
            return readiness
    raise ValueError(f"{split}: no readiness matrix was reported for this split")


def _source_pair(cases_path: Path, snapshot_path: Path) -> tuple[SourceFile, SourceFile]:
    return source_file(cases_path), source_file(snapshot_path)


def build_stop_plan(
    cases: list[StopPolicyCase],
    snapshot: SeedSnapshot,
    corpus: StopCorpus | None,
    *,
    cases_path: Path,
    snapshot_path: Path,
    split: str,
    policies: list[str],
    repeats: int,
    variant: str,
) -> StopPlan:
    """Validate variant, policy, split rồi dựng lịch và trace replay tất định

    Params:
    - cases: case đã review, giữ nguyên thứ tự file
    - snapshot: snapshot capture canonical mà case tham chiếu
    - corpus: corpus đã băm để kiểm source binding; None khi dựng lại lịch offline
    - cases_path, snapshot_path: nguồn canonical sẽ được manifest tham chiếu
    - split: dev hoặc heldout, phải đủ bốn lớp yêu cầu
    - policies: first và/hoặc llm
    - repeats: số lần lặp cho mỗi case
    - variant: prompt variant đã đăng ký
    """
    canonical_variant = resolve_variant(variant)
    policy_list = _validate_policies(policies)
    repeat_count = _validate_repeats(repeats)
    selection, selected = select_stop_cases(cases, snapshot, corpus, split=split)
    readiness = _readiness_for_split(selection, split)
    if not readiness.complete:
        raise ValueError(
            f"split {split} cannot be prepared; missing requirement classes: {readiness.missing()}"
        )
    if not selected:
        raise ValueError(f"split {split} has no runnable approved case")

    exclusions = list(selection.exclusions)
    exclusions.extend(
        StopExclusion(case_id=case.case_id, reason="split_not_selected")
        for case in selection.eligible_cases
        if case.split != split
    )
    trials = _trials_for(selected, repeats=repeat_count, variant=canonical_variant)
    traces = [render_baseline_trace(case, canonical_variant) for case in selected]
    cases_source, snapshot_source = _source_pair(Path(cases_path), Path(snapshot_path))

    return StopPlan(
        split=split,
        variant=canonical_variant,
        policies=policy_list,
        repeats=repeat_count,
        cases_source=cases_source,
        snapshot_source=snapshot_source,
        corpus_sha256=(
            corpus.sha256 if corpus is not None else snapshot.dataset_source.sha256
        ),
        selected_case_ids=[case.case_id for case in selected],
        traces=traces,
        trials=trials,
        exclusions=exclusions,
        readiness=readiness,
        run_purpose="evaluation",
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials) * len(policy_list),
    )


def build_hop0_capture_plan(
    cases: list[StopPolicyCase],
    snapshot: SeedSnapshot,
    corpus: StopCorpus | None,
    *,
    cases_path: Path,
    snapshot_path: Path,
    split: str,
    policy: str = "llm",
    variant: str = BASELINE_VARIANT,
) -> StopPlan:
    """Lên đúng một run nguồn hop-0, không cần action labels đã duyệt"""
    canonical_split = _validate_split(split)
    policy_list = _validate_policies([policy])
    canonical_variant = resolve_variant(variant)
    selection = validate_reviewed_cases(snapshot, corpus, cases)
    readiness = _readiness_for_split(selection, canonical_split)
    selected = [
        case
        for case in cases
        if case.source_hop == 0 and case.split == canonical_split and case.candidates
    ]
    if not selected:
        raise ValueError(
            f"split {canonical_split} has no hop-0 source cases with a nonempty frontier"
        )
    selected_ids = {case.case_id for case in selected}
    exclusions: list[StopExclusion] = []
    for case in cases:
        if case.case_id in selected_ids:
            continue
        if not case.candidates:
            reason = "no_candidates"
        elif case.source_hop > 0:
            reason = "hop_not_selected"
        else:
            reason = "split_not_selected"
        exclusions.append(StopExclusion(case_id=case.case_id, reason=reason))

    traces = [render_baseline_trace(case, canonical_variant) for case in selected]
    trials = _trials_for(selected, repeats=1, variant=canonical_variant)
    cases_source, snapshot_source = _source_pair(Path(cases_path), Path(snapshot_path))
    return StopPlan(
        split=canonical_split,
        run_purpose="later_hop_capture",
        variant=canonical_variant,
        policies=policy_list,
        repeats=1,
        cases_source=cases_source,
        snapshot_source=snapshot_source,
        corpus_sha256=(
            corpus.sha256 if corpus is not None else snapshot.dataset_source.sha256
        ),
        selected_case_ids=[case.case_id for case in selected],
        traces=traces,
        trials=trials,
        exclusions=exclusions,
        readiness=readiness,
        expected_trial_count=len(trials),
        expected_policy_result_count=len(trials),
    )


__all__ = [
    "REQUIREMENT_CLASSES",
    "UNASSIGNED_SPLIT_ERROR",
    "VALID_POLICIES",
    "VARIANT_RECIPES",
    "build_hop0_capture_plan",
    "build_stop_plan",
    "plan_selected_trials",
    "resolve_variant",
    "select_stop_cases",
]
