"""Áp transform thứ tự observation block cho intervention B.

Module dựng bằng chứng trước/sau đầy đủ cho Article 42: block atomic, line
origin, page-split proof, section không đổi và byte multiset bất biến.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contracts import GateInput

from diagnostic_subexp.article_42_observation_block_order.contracts import (
    ObservationArmSpec,
    ObservationBlockTrace,
    ObservationLineOrigin,
    ObservationMovedBlock,
    ObservationPageSplitProof,
    ObservationSectionTrace,
    ObservationSuiteSpec,
    ObservationTransformTrace,
)
from diagnostic_subexp.shared.contracts import DiagnosticGateRequest, sha256_text
from diagnostic_subexp.shared.execution import build_effective_messages, build_grammar_text


def _merge_fields(left: list[str], right: list[str]) -> list[str]:
    """Hợp nhất hai danh sách field theo thứ tự xuất hiện."""
    merged = list(left)
    for item in right:
        if item not in merged:
            merged.append(item)
    return merged


# Field đã kiểm tra không đổi khi B giữ nguyên thứ tự candidate
CANDIDATE_UNCHANGED_FIELDS = ["question", "observation", "grammar_candidates"]


@dataclass(frozen=True, slots=True)
class LineSpan:
    """Một dòng UTF-8 với số dòng 1-based và span byte end-exclusive."""

    line_number: int
    start: int
    end: int
    text: str


def split_line_spans(text: str) -> list[LineSpan]:
    """Tách dòng theo byte, giữ nguyên line terminator của từng dòng."""
    data = text.encode("utf-8")
    spans: list[LineSpan] = []
    start = 0
    number = 0
    for index, byte in enumerate(data):
        if byte == 0x0A:
            number += 1
            spans.append(
                LineSpan(number, start, index + 1, data[start : index + 1].decode("utf-8"))
            )
            start = index + 1
    if start < len(data):
        number += 1
        spans.append(LineSpan(number, start, len(data), data[start:].decode("utf-8")))
    return spans


def _span_bytes(data: bytes, span: list[int], *, field_name: str) -> bytes:
    start, end = span
    if end > len(data):
        raise ValueError(f"{field_name} exceeds the observation length: {span}")
    return data[start:end]


def _span_text(text: str, span: list[int], *, field_name: str) -> str:
    return _span_bytes(text.encode("utf-8"), span, field_name=field_name).decode("utf-8")


def _transform_by_id(spec: ObservationSuiteSpec, transform_id: str):
    for transform in spec.observation_transforms:
        if transform.transform_id == transform_id:
            return transform
    raise ValueError(f"unknown observation transform: {transform_id!r}")


def _unchanged_regions(total: int, block_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(block_spans):
        if start < cursor:
            raise ValueError("atomic block byte spans must not overlap")
        if start > cursor:
            regions.append((cursor, start))
        cursor = end
    if cursor < total:
        regions.append((cursor, total))
    return regions


def _line_numbers_for_span(lines: list[LineSpan], span: list[int], *, field_name: str) -> list[int]:
    start, end = span
    numbers = [line.line_number for line in lines if line.start >= start and line.end <= end]
    if not numbers:
        raise ValueError(f"{field_name} does not cover a complete line")
    first, last = min(numbers), max(numbers)
    if numbers != list(range(first, last + 1)):
        raise ValueError(f"{field_name} does not cover contiguous lines")
    if lines[first - 1].start != start or lines[last - 1].end != end:
        raise ValueError(f"{field_name} must align with line boundaries")
    return numbers


def _first_line_number(lines: list[LineSpan], span: tuple[int, int]) -> int:
    start, end = span
    for line in lines:
        if line.start >= start and line.end <= end:
            return line.line_number
    raise ValueError(f"span {list(span)} does not start at a line boundary")


def _build_line_origin_map(
    output_lines: list[LineSpan],
    original_by_number: dict[int, LineSpan],
    segments: list[tuple[str, int, int, int]],
) -> list[ObservationLineOrigin]:
    mapping: list[ObservationLineOrigin] = []
    segment_index = 0
    offset = 0
    for line in output_lines:
        while segment_index < len(segments) and line.start >= segments[segment_index][2]:
            segment_index += 1
            offset = 0
        if segment_index >= len(segments):
            raise ValueError(f"output line {line.line_number} falls outside every segment")
        segment_id, out_start, out_end, original_first_line = segments[segment_index]
        if not (out_start <= line.start and line.end <= out_end):
            raise ValueError(f"output line {line.line_number} splits a diagnostic segment")
        original_number = original_first_line + offset
        original = original_by_number.get(original_number)
        if original is None:
            raise ValueError(f"output line {line.line_number} maps outside the original observation")
        if original.text != line.text:
            raise ValueError(
                f"output line {line.line_number} text differs from original line {original_number}"
            )
        mapping.append(
            ObservationLineOrigin(
                output_line_number=line.line_number,
                original_line_number=original_number,
                segment_id=segment_id,
                line_sha256=sha256_text(line.text),
            )
        )
        offset += 1
    for segment_id, out_start, out_end, _ in segments:
        segment_lines = [line for line in output_lines if out_start <= line.start and line.end <= out_end]
        if not segment_lines:
            raise ValueError(f"segment {segment_id} covers no complete output line")
    return mapping


def _build_page_split_proof(
    spec: ObservationSuiteSpec,
    after_order: list[str],
    output_lines: list[LineSpan],
    output_span: dict[str, list[int]],
    original_by_number: dict[int, LineSpan],
) -> ObservationPageSplitProof | None:
    group = spec.page_split_group
    if group is None:
        return None
    if group.block_id not in after_order:
        raise ValueError("page split group block is missing from the transform")
    span = output_span[group.block_id]
    after_numbers = [
        line.line_number for line in output_lines if span[0] <= line.start and line.end <= span[1]
    ]
    if len(after_numbers) != len(group.source_line_numbers):
        raise ValueError("page split group changed its line count after the transform")
    if after_numbers != list(range(after_numbers[0], after_numbers[0] + len(after_numbers))):
        raise ValueError("page split group lines are no longer contiguous")
    for original_number, output_number, expected_hash in zip(
        group.source_line_numbers,
        after_numbers,
        group.line_sha256,
    ):
        original = original_by_number[original_number]
        output = output_lines[output_number - 1]
        if sha256_text(original.text) != expected_hash or output.text != original.text:
            raise ValueError("page split group lines changed during the transform")
    return ObservationPageSplitProof(
        group_id=group.group_id,
        block_id=group.block_id,
        line_sha256=list(group.line_sha256),
        before_line_numbers=list(group.source_line_numbers),
        after_line_numbers=after_numbers,
    )


def transform_observation_blocks(
    source_input: GateInput,
    arm: ObservationArmSpec,
    spec: ObservationSuiteSpec,
) -> ObservationTransformTrace:
    """Đổi thứ tự các atomic block và trả bằng chứng trước/sau đầy đủ."""
    if list(arm.presented_candidates) != list(source_input.candidates):
        raise ValueError(f"observation arm {arm.arm_id} must keep the original candidate order")

    transform = _transform_by_id(spec, arm.observation_transform_id)
    original = source_input.observation
    data = original.encode("utf-8")
    original_lines = split_line_spans(original)
    lines_by_number = {line.line_number: line for line in original_lines}

    block_by_id = {block.block_id: block for block in spec.observation_blocks}
    before_order = [block.block_id for block in spec.observation_blocks]
    if set(transform.ordered_block_ids) != set(before_order):
        raise ValueError(
            f"transform {transform.transform_id} must order exactly the declared atomic blocks"
        )

    block_text: dict[str, str] = {}
    block_span: dict[str, list[int]] = {}
    for block in spec.observation_blocks:
        text = _span_text(original, block.source_byte_span, field_name=block.block_id)
        if sha256_text(text) != block.source_text_sha256:
            raise ValueError(f"atomic block {block.block_id} does not match its declared hash")
        if original.count(text) != 1:
            raise ValueError(f"atomic block {block.block_id} is missing or ambiguous")
        declared_lines = _line_numbers_for_span(
            original_lines, block.source_byte_span, field_name=block.block_id
        )
        if declared_lines != block.source_line_numbers:
            raise ValueError(f"atomic block {block.block_id} does not match its declared lines")
        block_text[block.block_id] = text
        block_span[block.block_id] = list(block.source_byte_span)

    regions = _unchanged_regions(len(data), list(block_span.values()))
    if len(regions) != 2:
        raise ValueError("the declared atomic blocks must leave exactly one prefix and one suffix")
    prefix_region, suffix_region = regions
    if list(prefix_region) != transform.prefix_byte_span:
        raise ValueError("the declared prefix span does not match the unchanged prefix region")
    if list(suffix_region) != transform.suffix_byte_span:
        raise ValueError("the declared suffix span does not match the unchanged suffix region")

    prefix_bytes = _span_bytes(data, transform.prefix_byte_span, field_name="prefix")
    suffix_bytes = _span_bytes(data, transform.suffix_byte_span, field_name="suffix")
    if sha256_text(prefix_bytes.decode("utf-8")) != transform.prefix_sha256:
        raise ValueError("prefix hash does not match the observation")
    if sha256_text(suffix_bytes.decode("utf-8")) != transform.suffix_sha256:
        raise ValueError("suffix hash does not match the observation")

    original_layout = (
        prefix_bytes
        + b"".join(_span_bytes(data, block_span[bid], field_name=bid) for bid in before_order)
        + suffix_bytes
    )
    if original_layout != data:
        raise ValueError("prefix, atomic blocks and suffix must tile the original observation exactly")

    output_bytes = (
        prefix_bytes
        + b"".join(
            _span_bytes(data, block_span[bid], field_name=bid) for bid in transform.ordered_block_ids
        )
        + suffix_bytes
    )
    if len(output_bytes) != len(data):
        raise ValueError("the transformed observation changed the total byte length")
    if sorted(output_bytes) != sorted(data):
        raise ValueError("the transformed observation changed the source byte multiset")
    transformed = output_bytes.decode("utf-8")

    section_by_span = {}
    for section in spec.observation_sections:
        key = tuple(section.source_byte_span)
        if key in section_by_span:
            raise ValueError(f"duplicate unchanged section span: {list(key)}")
        section_by_span[key] = section
    if set(section_by_span) != {tuple(prefix_region), tuple(suffix_region)}:
        raise ValueError("unchanged sections must describe exactly the prefix and suffix regions")
    prefix_section = section_by_span[tuple(prefix_region)]
    suffix_section = section_by_span[tuple(suffix_region)]
    for section in (prefix_section, suffix_section):
        text = _span_text(original, section.source_byte_span, field_name=section.section_id)
        if sha256_text(text) != section.source_text_sha256:
            raise ValueError(f"unchanged section {section.section_id} does not match its hash")
        if original.count(text) != 1:
            raise ValueError(f"unchanged section {section.section_id} is missing or ambiguous")

    output_span: dict[str, list[int]] = {}
    cursor = len(prefix_bytes)
    for block_id in transform.ordered_block_ids:
        length = len(_span_bytes(data, block_span[block_id], field_name=block_id))
        output_span[block_id] = [cursor, cursor + length]
        cursor += length
    if cursor != len(output_bytes) - len(suffix_bytes):
        raise ValueError("ordered atomic blocks do not tile the transformed interior")

    transformed_lines = split_line_spans(transformed)
    blocks: list[ObservationBlockTrace] = []
    moved: list[ObservationMovedBlock] = []
    for block in spec.observation_blocks:
        before_position = before_order.index(block.block_id) + 1
        after_position = transform.ordered_block_ids.index(block.block_id) + 1
        blocks.append(
            ObservationBlockTrace(
                block_id=block.block_id,
                text=block_text[block.block_id],
                text_sha256=block.source_text_sha256,
                referenced_candidates=list(block.referenced_candidates),
                before_position=before_position,
                after_position=after_position,
                before_byte_span=list(block.source_byte_span),
                after_byte_span=output_span[block.block_id],
            )
        )
        if before_position != after_position:
            moved.append(
                ObservationMovedBlock(
                    block_id=block.block_id,
                    before_position=before_position,
                    after_position=after_position,
                )
            )

    segments: list[tuple[str, int, int, int]] = [
        (
            prefix_section.section_id,
            0,
            len(prefix_bytes),
            _first_line_number(original_lines, prefix_region),
        )
    ]
    for block_id in transform.ordered_block_ids:
        block = block_by_id[block_id]
        segments.append(
            (
                block_id,
                output_span[block_id][0],
                output_span[block_id][1],
                block.source_line_numbers[0],
            )
        )
    segments.append(
        (
            suffix_section.section_id,
            len(output_bytes) - len(suffix_bytes),
            len(output_bytes),
            _first_line_number(original_lines, suffix_region),
        )
    )

    line_origin_map = _build_line_origin_map(transformed_lines, lines_by_number, segments)

    sections = [
        ObservationSectionTrace(
            section_id=prefix_section.section_id,
            source_byte_span=list(prefix_section.source_byte_span),
            output_byte_span=[0, len(prefix_bytes)],
            text_sha256=prefix_section.source_text_sha256,
            referenced_candidates=list(prefix_section.referenced_candidates),
        ),
        ObservationSectionTrace(
            section_id=suffix_section.section_id,
            source_byte_span=list(suffix_section.source_byte_span),
            output_byte_span=[len(output_bytes) - len(suffix_bytes), len(output_bytes)],
            text_sha256=suffix_section.source_text_sha256,
            referenced_candidates=list(suffix_section.referenced_candidates),
        ),
    ]

    page_split = _build_page_split_proof(
        spec,
        transform.ordered_block_ids,
        transformed_lines,
        output_span,
        lines_by_number,
    )

    changed = [] if transform.ordered_block_ids == before_order else ["observation_block_order"]
    return ObservationTransformTrace(
        transform_id=transform.transform_id,
        original_observation=original,
        transformed_observation=transformed,
        before_block_order=before_order,
        after_block_order=list(transform.ordered_block_ids),
        blocks=blocks,
        moved_blocks=moved,
        line_origin_map=line_origin_map,
        sections=sections,
        page_split_group=page_split,
        declared_changed_fields=list(changed),
        verified_unchanged_fields=_merge_fields(
            CANDIDATE_UNCHANGED_FIELDS,
            [
                "question",
                "presented_candidates",
                "grammar_candidates",
                "unchanged_sections",
            ],
        ),
    )


def reconstruct_observation_request(
    source_input: GateInput,
    arm: ObservationArmSpec,
    spec: ObservationSuiteSpec,
) -> DiagnosticGateRequest:
    """Dựng request hiệu lực label-free cho một arm B."""
    observation_trace = transform_observation_blocks(source_input, arm, spec)
    observation = observation_trace.transformed_observation
    return DiagnosticGateRequest(
        question=source_input.question,
        observation=observation,
        presented_candidates=list(arm.presented_candidates),
        grammar_candidates=list(spec.grammar_candidates),
        effective_messages=build_effective_messages(
            source_input.question,
            observation,
            list(arm.presented_candidates),
        ),
        grammar_text=build_grammar_text(list(spec.grammar_candidates)),
    )


def verify_reconstructed_observation(trace: Any, observation_trace: Any) -> None:
    """Đối chiếu toàn bộ bằng chứng observation đã lưu với bản dựng lại."""
    if trace.original_observation != observation_trace.original_observation:
        raise ValueError("trace original observation does not match the bundle")
    if trace.transformed_observation != observation_trace.transformed_observation:
        raise ValueError("trace transformed observation does not match the bundle")
    if trace.before_block_order != list(observation_trace.before_block_order):
        raise ValueError("trace before_block_order does not match the bundle")
    if trace.after_block_order != list(observation_trace.after_block_order):
        raise ValueError("trace after_block_order does not match the bundle")
    if [block.model_dump(mode="json") for block in trace.blocks] != [
        block.model_dump(mode="json") for block in observation_trace.blocks
    ]:
        raise ValueError("trace block records do not match the bundle")
    if [block.model_dump(mode="json") for block in trace.moved_blocks] != [
        block.model_dump(mode="json") for block in observation_trace.moved_blocks
    ]:
        raise ValueError("trace moved block records do not match the bundle")
    if [item.model_dump(mode="json") for item in trace.line_origin_map] != [
        item.model_dump(mode="json") for item in observation_trace.line_origin_map
    ]:
        raise ValueError("trace line-origin map does not match the bundle")
    if [item.model_dump(mode="json") for item in trace.unchanged_context_sections] != [
        item.model_dump(mode="json") for item in observation_trace.sections
    ]:
        raise ValueError("trace unchanged sections do not match the bundle")
    if trace.page_split_group != observation_trace.page_split_group:
        raise ValueError("trace page-split proof does not match the bundle")
    if trace.transformed_observation_hash != sha256_text(observation_trace.transformed_observation):
        raise ValueError("trace transformed observation hash does not match the bundle")


def verify_observation_trace(
    trace: Any,
    source_input: GateInput,
    arm: ObservationArmSpec,
    spec: ObservationSuiteSpec,
) -> None:
    """Dựng lại transform observation và từ chối mọi khác biệt bằng chứng."""
    observation_trace = transform_observation_blocks(source_input, arm, spec)
    verify_reconstructed_observation(trace, observation_trace)
    if trace.declared_changed_fields != list(observation_trace.declared_changed_fields):
        raise ValueError("trace declared changed fields do not match the bundle")
    if trace.verified_unchanged_fields != list(observation_trace.verified_unchanged_fields):
        raise ValueError("trace declared unchanged fields do not match the bundle")
__all__ = [
    "LineSpan",
    "reconstruct_observation_request",
    "split_line_spans",
    "transform_observation_blocks",
    "verify_observation_trace",
    "verify_reconstructed_observation",
]
