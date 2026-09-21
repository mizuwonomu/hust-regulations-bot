"""Áp transform thứ tự candidate cho intervention A.

Module chỉ đổi presented order của arm A và giữ nguyên membership, question,
observation và grammar bytes.
"""

from __future__ import annotations

from contracts import GateInput

from diagnostic_subexp.candidate_pair_41_3_position.contracts import (
    CandidateArmSpec,
    CandidateTransformTrace,
    derive_pair_metadata,
)
from diagnostic_subexp.shared.contracts import sha256_text


def transform_candidate_order(
    source_input: GateInput,
    arm: CandidateArmSpec,
) -> CandidateTransformTrace:
    """Đổi presented order nhưng giữ nguyên membership và observation."""
    original = list(source_input.candidates)
    presented = list(arm.presented_candidates)
    if sorted(original) != sorted(presented):
        raise ValueError(f"arm {arm.arm_id} changes candidate membership")
    if len(set(presented)) != len(presented):
        raise ValueError(f"arm {arm.arm_id} repeats a presented candidate")
    if len(presented) != len(original):
        raise ValueError(f"arm {arm.arm_id} changes the presented candidate count")
    positions, orientation = derive_pair_metadata(presented)
    if arm.pair_positions and arm.pair_positions != positions:
        raise ValueError(f"arm {arm.arm_id} declared pair_positions do not match its order")
    if arm.relative_pair_order is not None and arm.relative_pair_order != orientation:
        raise ValueError(f"arm {arm.arm_id} declared relative_pair_order does not match its order")
    observation_hash = sha256_text(source_input.observation)
    changed = [] if presented == original else ["presented_candidates"]
    return CandidateTransformTrace(
        original_candidates=original,
        presented_candidates=presented,
        pair_positions=positions,
        relative_pair_order=orientation,
        original_observation_hash=observation_hash,
        transformed_observation_hash=observation_hash,
        declared_changed_fields=changed,
        verified_unchanged_fields=["question", "observation", "grammar_candidates"],
    )


__all__ = ["transform_candidate_order"]
