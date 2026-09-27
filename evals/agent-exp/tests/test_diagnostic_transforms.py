"""Kiểm tra offline transform candidate và observation block của diagnostic A/B."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(AGENT_EXP_ROOT.parents[1]))

from artifacts import read_snapshot
from diagnostic_subexp.article_42_observation_block_order.contracts import ObservationSuiteSpec
from diagnostic_subexp.article_42_observation_block_order.schedule import (
    build_observation_definition,
    plan_observation_run,
    read_observation_spec,
    resolve_observation_sources,
)
from diagnostic_subexp.article_42_observation_block_order.transforms import (
    split_line_spans,
    transform_observation_blocks,
)
from diagnostic_subexp.candidate_pair_41_3_position.schedule import (
    plan_candidate_run,
    read_candidate_spec,
    resolve_candidate_sources,
)
from diagnostic_subexp.candidate_pair_41_3_position.transforms import transform_candidate_order
from diagnostic_subexp.shared.contracts import sha256_text
from seed_cases import policy_input, read_cases

A_SPEC_PATH = AGENT_EXP_ROOT / "scripts" / "diagnostic_subexp" / "candidate_pair_41_3_position"
B_SPEC_PATH = AGENT_EXP_ROOT / "scripts" / "diagnostic_subexp" / "article_42_observation_block_order"


def _frozen_observation_hash() -> str:
    """Băm observation đóng băng từ case canonical thay vì hardcode."""
    spec = read_observation_spec(None)
    sources = resolve_observation_sources(spec)
    cases = read_cases(sources["case"])
    case = next(
        item for item in cases if item.case_id == spec.source_case_identity.case_id
    )
    return sha256_text(case.observation)


def _load(path: Path):
    if path == A_SPEC_PATH:
        spec = read_candidate_spec(None)
        sources = resolve_candidate_sources(spec)
        cases = read_cases(sources["case"])
        snapshot = read_snapshot(sources["snapshot"])
        plan = plan_candidate_run(spec, cases, snapshot, ["first", "llm"])
        return spec, cases, plan
    spec = read_observation_spec(None)
    sources = resolve_observation_sources(spec)
    cases = read_cases(sources["case"])
    snapshot = read_snapshot(sources["snapshot"])
    plan = plan_observation_run(spec, cases, snapshot, ["first", "llm"])
    return spec, cases, plan


@pytest.fixture(scope="module")
def b_fixture():
    spec, cases, plan = _load(B_SPEC_PATH)
    trace_by_arm = {trace.arm_id: trace for trace in plan.traces}
    return spec, cases, trace_by_arm


@pytest.fixture(scope="module")
def a_fixture():
    spec, cases, plan = _load(A_SPEC_PATH)
    return spec, cases, plan


def test_b0_identity_transform_matches_the_frozen_observation(b_fixture):
    _, _, traces = b_fixture
    trace = traces["b0-original-control"]
    assert trace.transformed_observation == trace.original_observation
    assert trace.transformed_observation_hash == _frozen_observation_hash()
    assert [item.original_line_number for item in trace.line_origin_map] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [item.output_line_number for item in trace.line_origin_map] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert trace.moved_blocks == []
    assert trace.before_block_order == trace.after_block_order


def test_b1_follows_the_exact_line_origin_mapping(b_fixture):
    _, _, traces = b_fixture
    trace = traces["b1-reversed-blocks"]
    assert [item.original_line_number for item in trace.line_origin_map] == [1, 2, 5, 3, 4, 6, 7, 8]
    blocks = {block.block_id: block for block in trace.blocks}
    assert blocks["b42-foundation-timing"].before_position == 1
    assert blocks["b42-foundation-timing"].after_position == 2
    assert blocks["b42-university-defense"].before_position == 2
    assert blocks["b42-university-defense"].after_position == 1
    assert blocks["b42-foundation-timing"].after_byte_span == [401, 946]
    assert blocks["b42-university-defense"].after_byte_span == [114, 401]
    assert trace.moved_blocks[0].block_id == "b42-foundation-timing"
    assert trace.declared_changed_fields == ["observation_block_order"]


def test_b1_preserves_total_bytes_and_every_source_byte_once(b_fixture):
    _, _, traces = b_fixture
    trace = traces["b1-reversed-blocks"]
    original = trace.original_observation.encode("utf-8")
    transformed = trace.transformed_observation.encode("utf-8")
    assert len(transformed) == len(original)
    assert sorted(transformed) == sorted(original)
    for block in trace.blocks:
        assert transformed.decode("utf-8").count(block.text) == 1


def test_b1_keeps_the_page_split_sentence_atomic(b_fixture):
    _, _, traces = b_fixture
    trace = traces["b1-reversed-blocks"]
    assert trace.page_split_group is not None
    group = trace.page_split_group
    assert group.before_line_numbers == [3, 4]
    assert group.after_line_numbers == [4, 5]
    assert group.after_line_numbers[1] - group.after_line_numbers[0] == 1
    lines = split_line_spans(trace.transformed_observation)
    assert sha256_text(lines[3].text) == group.line_sha256[0]
    assert sha256_text(lines[4].text) == group.line_sha256[1]


def test_b1_keeps_unchanged_sections_byte_identical(b_fixture):
    _, _, traces = b_fixture
    trace = traces["b1-reversed-blocks"]
    original = trace.original_observation.encode("utf-8")
    transformed = trace.transformed_observation.encode("utf-8")
    sections = {section.section_id: section for section in trace.unchanged_context_sections}
    assert set(sections) == {"obs-preamble", "ctx45-section"}
    for section in sections.values():
        start, end = section.source_byte_span
        out_start, out_end = section.output_byte_span
        assert original[start:end] == transformed[out_start:out_end]
        assert sha256_text(original[start:end].decode("utf-8")) == section.text_sha256
    assert sections["ctx45-section"].referenced_candidates == [8, 43]


def test_spec_rejects_a_split_page_split_block(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    lines = split_line_spans(case.observation)
    payload = json.loads(json.dumps(build_observation_definition()))
    payload["observation_blocks"][0]["source_line_count"] = 1
    payload["observation_blocks"][0]["source_line_numbers"] = [3]
    payload["observation_blocks"][0]["source_byte_span"] = [lines[2].start, lines[2].end]
    payload["observation_blocks"][0]["source_text_sha256"] = sha256_text(lines[2].text)
    with pytest.raises(ValueError, match="page split group must match its block source lines"):
        ObservationSuiteSpec.model_validate(payload)


def test_transform_rejects_reversing_the_page_split_lines(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    lines = split_line_spans(case.observation)
    foundation = spec.observation_blocks[0]
    second_line = foundation.model_copy(
        update={
            "block_id": "b42-foundation-second-line",
            "source_line_count": 1,
            "source_line_numbers": [4],
            "source_byte_span": [lines[3].start, lines[3].end],
            "source_text_sha256": sha256_text(lines[3].text),
        }
    )
    narrowed = [
        foundation.model_copy(
            update={
                "source_line_count": 1,
                "source_line_numbers": [3],
                "source_byte_span": [lines[2].start, lines[2].end],
                "source_text_sha256": sha256_text(lines[2].text),
            }
        ),
        second_line,
        spec.observation_blocks[1],
    ]
    reversed_transform = spec.observation_transforms[1].model_copy(
        update={
            "ordered_block_ids": [
                second_line.block_id,
                foundation.block_id,
                spec.observation_blocks[1].block_id,
            ]
        }
    )
    identity_transform = spec.observation_transforms[0].model_copy(
        update={"ordered_block_ids": [block.block_id for block in narrowed]}
    )
    broken = spec.model_copy(
        update={
            "observation_blocks": narrowed,
            "observation_transforms": [reversed_transform, identity_transform],
            "arms": [
                arm.model_copy(update={"observation_transform_id": "b-reversed-blocks"})
                for arm in spec.arms
            ],
        }
    )
    with pytest.raises(ValueError, match="page split group changed its line count"):
        transform_observation_blocks(policy_input(case), broken.arms[0], broken)


def test_transform_rejects_a_moved_header(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    lines = split_line_spans(case.observation)
    moved_prefix = spec.observation_transforms[0].model_copy(
        update={
            "prefix_byte_span": [lines[1].start, lines[1].end],
            "prefix_sha256": sha256_text(lines[1].text),
        }
    )
    broken = spec.model_copy(update={"observation_transforms": [moved_prefix, spec.observation_transforms[1]]})
    with pytest.raises(ValueError, match="prefix span does not match"):
        transform_observation_blocks(policy_input(case), spec.arms[0], broken)


def test_transform_rejects_an_edited_context_45(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    edited = spec.observation_transforms[0].model_copy(update={"suffix_sha256": "0" * 64})
    broken = spec.model_copy(update={"observation_transforms": [edited, spec.observation_transforms[1]]})
    with pytest.raises(ValueError, match="suffix hash does not match"):
        transform_observation_blocks(policy_input(case), spec.arms[0], broken)


def test_transform_rejects_a_block_hash_mismatch(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    edited = list(spec.observation_blocks)
    edited[0] = edited[0].model_copy(update={"source_text_sha256": "0" * 64})
    broken = spec.model_copy(update={"observation_blocks": edited})
    with pytest.raises(ValueError, match="does not match its declared hash"):
        transform_observation_blocks(policy_input(case), spec.arms[0], broken)


def test_candidate_transform_preserves_membership_and_observation(a_fixture):
    spec, cases, plan = a_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    source_input = policy_input(case)
    for arm in spec.arms:
        trace = transform_candidate_order(source_input, arm)
        assert trace.presented_candidates == arm.presented_candidates
        assert sorted(trace.presented_candidates) == sorted(source_input.candidates)
        assert trace.original_observation_hash == case.observation_hash
        assert trace.transformed_observation_hash == trace.original_observation_hash
        expected = [] if arm.presented_candidates == case.candidates else ["presented_candidates"]
        assert trace.declared_changed_fields == expected
    assert {trace.arm_id for trace in plan.traces} == {arm.arm_id for arm in spec.arms}


def test_candidate_transform_rejects_a_membership_change(a_fixture):
    spec, cases, _ = a_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    arm = spec.arms[1].model_copy(update={"presented_candidates": [8, 3, 41, 40, 44]})
    with pytest.raises(ValueError, match="changes candidate membership"):
        transform_candidate_order(policy_input(case), arm)


def test_observation_originals_are_never_modified(b_fixture):
    spec, cases, _ = b_fixture
    case = next(case for case in cases if case.case_id == spec.source_case_identity.case_id)
    for arm in spec.arms:
        trace = transform_observation_blocks(policy_input(case), arm, spec)
        assert trace.original_observation == case.observation
        assert sha256_text(trace.original_observation) == _frozen_observation_hash()
        assert sorted(trace.transformed_observation.encode("utf-8")) == sorted(
            case.observation.encode("utf-8")
        )
