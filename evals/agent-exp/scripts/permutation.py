"""Lập lịch order thuần và dựng lại input deterministic cho permutation replay."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

try:
    from contracts import (
        Exclusion,
        GateCase,
        GateInput,
        PermutationConfig,
        PermutationPlan,
        SeedSnapshot,
        Trial,
    )
    from seed_cases import rebuild_case_state, validate_permutation_cases
except ModuleNotFoundError:
    from .contracts import (
        Exclusion,
        GateCase,
        GateInput,
        PermutationConfig,
        PermutationPlan,
        SeedSnapshot,
        Trial,
    )
    from .seed_cases import rebuild_case_state, validate_permutation_cases


SEED_ORDER_CONDITIONS = frozenset({"seed-order", "combined"})
ROTATION_CONDITIONS = frozenset({"candidate-order", "combined"})
# Lịch chỉ thay đổi order đã lưu, không đọc nhãn hoặc gọi model


def question_identity(question_id: Any) -> tuple[type[Any], Any]:
    if isinstance(question_id, bool) or not isinstance(question_id, (int, str)):
        raise ValueError(f"question id must be an integer or string: {question_id!r}")
    return type(question_id), question_id


_question_identity = question_identity


def _observation_hash(observation: str) -> str:
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def cyclic_rotations(items: Sequence[Any]) -> list[list[Any]]:
    """Trả về mọi cyclic rotation duy nhất, giữ order gốc ở đầu."""
    values = list(items)
    rotations: list[list[Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for offset in range(len(values) or 1):
        rotation = values[offset:] + values[:offset]
        token = tuple(rotation)
        if token not in seen:
            seen.add(token)
            rotations.append(rotation)
    return rotations


def is_cyclic_rotation(order: Sequence[Any], original: Sequence[Any]) -> bool:
    """Kiểm tra một order có là cyclic rotation của order gốc hay không."""
    expected = list(original)
    actual = list(order)
    if len(actual) != len(expected):
        return False
    if not expected:
        return not actual
    return any(
        actual == expected[offset:] + expected[:offset]
        for offset in range(len(expected))
    )


def _original_seed_order(case: GateCase, row: Any) -> list[int]:
    if case.source_hop != 0 or row is None:
        return []
    return list(range(len(row.contexts)))


def _permutation_id(
    condition: str,
    seed_order: list[int],
    candidate_order: list[int],
) -> str:
    if condition == "repeat":
        return "original"
    seed_token = ",".join(str(index) for index in seed_order)
    candidate_token = ",".join(str(article_id) for article_id in candidate_order)
    if condition == "candidate-order":
        return f"c[{candidate_token}]"
    if condition == "seed-order":
        return f"s[{seed_token}]"
    return f"s[{seed_token}]-c[{candidate_token}]"


def _seed_orders(condition: str, seed_length: int) -> list[list[int]]:
    if condition in SEED_ORDER_CONDITIONS:
        return cyclic_rotations(range(seed_length))
    return [list(range(seed_length))]


def _candidate_orders(condition: str, candidates: list[int]) -> list[list[int]]:
    if condition in ROTATION_CONDITIONS:
        return cyclic_rotations(candidates)
    return [list(candidates)]


def _case_exclusion(case: GateCase) -> Exclusion | None:
    if not case.candidates:
        return Exclusion(case_id=case.case_id, reason="no_candidates")
    if case.label_status != "approved":
        return Exclusion(case_id=case.case_id, reason="draft")
    if case.expected_action == "unresolved":
        return Exclusion(case_id=case.case_id, reason="unresolved")
    if case.expected_action not in {"follow", "stop"}:
        raise ValueError(f"{case.case_id}: unsupported semantic label {case.expected_action!r}")
    return None


def build_permutation_plan(
    cases: list[GateCase],
    snapshot: SeedSnapshot,
    config: PermutationConfig,
    *,
    policy_count: int = 1,
) -> PermutationPlan:
    """Lập mọi order được phép mà không đọc nhãn hoặc output của model."""
    if not isinstance(config, PermutationConfig):
        raise TypeError("config must be a PermutationConfig")
    if isinstance(policy_count, bool) or not isinstance(policy_count, int) or policy_count <= 0:
        raise ValueError("policy_count must be a positive integer")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")

    row_by_question = {_question_identity(row.id): row for row in snapshot.rows}
    seen_case_ids: set[str] = set()
    trials: list[Trial] = []
    exclusions: list[Exclusion] = []

    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)

        exclusion = _case_exclusion(case)
        if exclusion is not None:
            exclusions.append(exclusion)
            continue

        row = row_by_question.get(_question_identity(case.question_id))
        if config.condition in SEED_ORDER_CONDITIONS:
            if case.source_hop != 0 or row is None:
                exclusions.append(Exclusion(case_id=case.case_id, reason="later_hop_seed_order"))
                continue
            if not row.contexts:
                exclusions.append(Exclusion(case_id=case.case_id, reason="empty_seed_contexts"))
                continue

        seed_length = len(row.contexts) if case.source_hop == 0 and row is not None else 0
        order_configs: list[tuple[list[int], list[int]]] = []
        seen_orders: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        for seed_order in _seed_orders(config.condition, seed_length):
            for candidate_order in _candidate_orders(config.condition, list(case.candidates)):
                token = (tuple(seed_order), tuple(candidate_order))
                if token in seen_orders:
                    continue
                seen_orders.add(token)
                order_configs.append((seed_order, candidate_order))

        prepared: list[tuple[list[int], list[int], str]] = []
        for seed_order, candidate_order in order_configs:
            if config.condition in SEED_ORDER_CONDITIONS:
                if row is None:
                    raise ValueError(f"{case.case_id}: seed-order case has no snapshot row")
                observation = _rebuild_seed_observation(case, snapshot, row, seed_order)
                observation_hash = _observation_hash(observation)
            else:
                observation_hash = case.observation_hash
            prepared.append((seed_order, candidate_order, observation_hash))

        for repeat_id in range(config.repeats):
            for seed_order, candidate_order, observation_hash in prepared:
                permutation_id = _permutation_id(
                    config.condition,
                    seed_order,
                    candidate_order,
                )
                trials.append(
                    Trial(
                        trial_id=f"{case.case_id}:{config.condition}:{permutation_id}:r{repeat_id}",
                        case_id=case.case_id,
                        repeat_id=repeat_id,
                        condition=config.condition,
                        permutation_id=permutation_id,
                        seed_order=list(seed_order),
                        candidate_order=list(candidate_order),
                        observation_hash=observation_hash,
                    )
                )

    expected = len(trials)
    return PermutationPlan(
        config=config,
        trials=trials,
        exclusions=exclusions,
        expected_trial_count=expected,
        expected_policy_result_count=expected * policy_count,
    )


def plan_permutation_run(
    snapshot: SeedSnapshot,
    cases: list[GateCase],
    *,
    snapshot_hash: str,
    config: PermutationConfig,
    policy_count: int = 1,
) -> PermutationPlan:
    """Kiểm tra case permutation rồi dựng và ghép lịch order."""
    selection = validate_permutation_cases(snapshot, cases, snapshot_hash=snapshot_hash)
    plan = build_permutation_plan(
        selection.eligible_cases,
        snapshot,
        config,
        policy_count=policy_count,
    )
    if not selection.exclusions:
        return plan
    return PermutationPlan(
        config=plan.config,
        trials=plan.trials,
        exclusions=[*selection.exclusions, *plan.exclusions],
        expected_trial_count=plan.expected_trial_count,
        expected_policy_result_count=plan.expected_policy_result_count,
    )


def _rebuild_seed_observation(
    case: GateCase,
    snapshot: SeedSnapshot,
    row: Any,
    seed_order: list[int],
) -> str:
    if sorted(seed_order) != list(range(len(row.contexts))):
        raise ValueError(f"{case.case_id}: seed_order must permute the original context indices")
    reordered = row.model_copy(
        update={"contexts": [row.contexts[index] for index in seed_order]}
    )
    observation, candidates = rebuild_case_state(snapshot, reordered)
    if set(candidates) != set(case.candidates):
        raise ValueError(f"{case.case_id}: seed rotation changed derived candidate membership")
    return observation


def reconstruct_trial_input(
    case: GateCase,
    snapshot: SeedSnapshot,
    trial: Trial,
) -> GateInput:
    """Dựng một trial input và kiểm tra invariant cùng hash của condition."""
    if trial.case_id != case.case_id:
        raise ValueError(f"{trial.trial_id}: trial and case identifiers do not match")

    row_by_question = {_question_identity(row.id): row for row in snapshot.rows}
    row = row_by_question.get(_question_identity(case.question_id))
    expected_seed_order = _original_seed_order(case, row)

    if trial.condition == "original":
        if trial.seed_order not in ([], expected_seed_order):
            raise ValueError(f"{trial.trial_id}: original trials must keep the original seed order")
        if trial.candidate_order != case.candidates:
            raise ValueError(f"{trial.trial_id}: original trials must keep the original candidate order")
        if trial.observation_hash != case.observation_hash:
            raise ValueError(f"{trial.trial_id}: original trials must keep the original observation hash")
        return GateInput(
            question=case.question,
            observation=case.observation,
            candidates=list(case.candidates),
        )

    if trial.condition == "repeat":
        if trial.seed_order != expected_seed_order:
            raise ValueError(f"{trial.trial_id}: repeat trials must keep the original seed order")
        if trial.candidate_order != case.candidates:
            raise ValueError(f"{trial.trial_id}: repeat trials must keep the original candidate order")
        if trial.observation_hash != case.observation_hash:
            raise ValueError(f"{trial.trial_id}: repeat trials must keep the original observation hash")
        return GateInput(
            question=case.question,
            observation=case.observation,
            candidates=list(case.candidates),
        )

    if trial.condition == "candidate-order":
        if trial.seed_order != expected_seed_order:
            raise ValueError(
                f"{trial.trial_id}: candidate-order trials must keep the original seed order"
            )
        if not is_cyclic_rotation(trial.candidate_order, case.candidates):
            raise ValueError(f"{trial.trial_id}: candidate order is not a rotation of the case")
        if trial.observation_hash != case.observation_hash:
            raise ValueError(
                f"{trial.trial_id}: candidate-order trials must keep the original observation hash"
            )
        return GateInput(
            question=case.question,
            observation=case.observation,
            candidates=list(trial.candidate_order),
        )

    if trial.condition in SEED_ORDER_CONDITIONS:
        if case.source_hop != 0 or row is None or not row.contexts:
            raise ValueError(
                f"{trial.trial_id}: seed-order conditions require a hop-0 full-context case"
            )
        observation = _rebuild_seed_observation(case, snapshot, row, trial.seed_order)
        if trial.condition == "seed-order":
            if trial.candidate_order != case.candidates:
                raise ValueError(
                    f"{trial.trial_id}: seed-order trials must keep the original candidate list"
                )
            candidates = list(case.candidates)
        else:
            if not is_cyclic_rotation(trial.candidate_order, case.candidates):
                raise ValueError(f"{trial.trial_id}: candidate order is not a rotation of the case")
            candidates = list(trial.candidate_order)
        if _observation_hash(observation) != trial.observation_hash:
            raise ValueError(
                f"{trial.trial_id}: trial observation hash does not match the reconstruction"
            )
        return GateInput(
            question=case.question,
            observation=observation,
            candidates=candidates,
        )

    raise ValueError(f"{trial.trial_id}: unsupported trial condition {trial.condition!r}")


__all__ = [
    "build_permutation_plan",
    "cyclic_rotations",
    "is_cyclic_rotation",
    "plan_permutation_run",
    "question_identity",
    "reconstruct_trial_input",
]
