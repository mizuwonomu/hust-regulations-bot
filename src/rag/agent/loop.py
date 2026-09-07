"""Vòng lặp retrieval có gate cho citation agent.

Loop sở hữu một mapping thu thập (key -> nguyên văn, key là membership) và
frontier; extractor chỉ parse mention từ nguyên văn.
Trả về full parent texts cho synthesizer/eval, gate chỉ thấy compact observation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langsmith import traceable

from src.rag.agent.schema import Decision
from src.rag.agent.tools import (
    CitationMention,
    article_title,
    dieu_from_title,
    source_dieu,
)

logger = logging.getLogger(__name__)

# (contexts, dieu set) - seam hai giá trị giữ nguyên từ eval baseline
# Trả về kết quả retrieval ban đầu, contract nhận question dạng str, trả về tuple nội dung các điều cha,
# và set số Điều tương ứng
SeedFn = Callable[[str], tuple[list[str], set[int]]] 

# Contract nhận câu hỏi, observation, số Điều đã ghé -> trả về Decision hop hay stop
DecideFn = Callable[[str, str, list[int]], Decision]

# (source_dieu, article_text) -> mentions theo thứ tự xuất hiện
ExtractFn = Callable[[int, str], list[CitationMention]]


def _trace_inputs(inputs: dict) -> dict:
    return {
        "question": inputs.get("question"),
        "max_follow_articles": inputs.get("max_follow_articles"),
        "internal_dieu_count": len(inputs.get("internal_dieu", ())),
    }


def _trace_outputs(output: tuple[list[str], set[int]]) -> dict:
    contexts, collected_dieu = output
    return {
        "context_count": len(contexts),
        "collected_dieu": sorted(collected_dieu),
    }


def _build_gate_observation(
    collected: dict[int, str | None],
    mentions_by_source: dict[int, list[CitationMention]],
) -> str:
    """Dựng compact observation chỉ từ title và các dòng chứa dẫn chiếu.

    Params:
    - collected: Mapping (key nội bộ -> nguyên văn) đã thu thập, theo thứ tự;
      entry membership-only (value None) không được render.
    - mentions_by_source: Mention đã parse theo key nội bộ.

    Header chỉ mang định danh vô hại: số Điều, hoặc ? cho seed không parse được;
    title chỉ render khi là heading Điều nhận diện được, thiếu title thì bỏ hẳn.
    """
    sections = ["Các Điều đã thu thập:"]
    for key, article in collected.items():
        if article is None:
            continue
        # Label theo nguồn thật: seed trùng ID dưới key âm vẫn hiện đúng số Điều
        source = source_dieu(key, article)
        title = article_title(article)
        # Loại self-reference: excerpt trùng target của chính article này là noise
        excerpts = list(
            dict.fromkeys(
                m.excerpt
                for m in mentions_by_source.get(key, [])
                if m.target_dieu != source
            )
        )
        label = str(source) if source > 0 else "?"
        header = f"[Context {label}] {title}" if title is not None else f"[Context {label}]"
        lines = [header]
        lines.extend(f"- {excerpt}" for excerpt in excerpts)
        sections.append("\n".join(lines))
    return "\n".join(sections)


def _build_frontier(
    collected: dict[int, str | None],
    extract_citations_fn: ExtractFn,
    internal_dieu: set[int],
) -> tuple[list[int], dict[int, list[CitationMention]]]:
    """Parse mention trên toàn bộ context đã thu thập và lọc frontier.

    Params:
    - collected: Mapping (key nội bộ -> nguyên văn) đã thu thập; membership-only
      entry (value None) không có nguyên văn để parse.
    - extract_citations_fn: Parser canonical dùng chung.
    - internal_dieu: Whitelist các Điều nội bộ.

    Trả về candidates theo thứ tự xuất hiện đầu tiên cùng mentions theo key.
    Key nội bộ là số Điều thật hoặc số âm duy nhất, nên mention của các seed
    không parse được không bao giờ đè nhau; membership tức key đã có trong mapping.
    Self-reference trên header không thành candidate; traversal forward-only:
    chỉ quét article đã thu thập.
    """
    mentions_by_source: dict[int, list[CitationMention]] = {}
    candidates: list[int] = []
    seen: set[int] = set()

    for key, article in collected.items():
        if article is None:
            continue
        # Source_dieu trung thực cho mention: key dương là nguồn chính thức,
        # Seed trùng ID dưới key âm vẫn giữ số Điều thật, không parse được là 0
        source = source_dieu(key, article)
        mentions = extract_citations_fn(source, article)
        mentions_by_source[key] = mentions
        for mention in mentions:
            target = mention.target_dieu
            if (target == source) or (target in collected) or (target in seen) or (target not in internal_dieu):
                continue
            seen.add(target)
            candidates.append(target)

    return candidates, mentions_by_source


@traceable(
    name="citation_agent_retrieve",
    run_type="chain",
    process_inputs=_trace_inputs,
    process_outputs=_trace_outputs,
)
def agent_retrieve(
    question: str,
    *,
    retrieve_seed_fn: SeedFn,
    get_article_fn: Callable[[int], str | None],
    extract_citations_fn: ExtractFn,
    decide_fn: DecideFn,
    internal_dieu: set[int],
    max_follow_articles: int,
    telemetry: dict[str, Any] | None = None,
) -> tuple[list[str], set[int]]:
    """Retrieve seed rồi follow citation theo quyết định của gate.

    Params:
    - question: Câu hỏi gốc của người dùng
    - retrieve_seed_fn: Retrieval single-pass cho hop 0
    - get_article_fn: Hàm lấy nguyên văn Điều theo số
    - extract_citations_fn: Parser canonical (source_dieu, text) -> mentions
    - decide_fn: Gate đã bind client
    - internal_dieu: Whitelist các Điều nội bộ
    - max_follow_articles: Số Điều follow thành công tối đa, không tính seed
    - telemetry: Dict tùy chọn được loop điền trajectory (seed, gate, follow)
      để caller đo lường mà không phải parse log
    """
    if (
        isinstance(max_follow_articles, bool)
        or not isinstance(max_follow_articles, int)
        or max_follow_articles < 0
    ):
        raise ValueError("max_follow_articles phải là số nguyên không âm")

    seed_contexts, seed_dieu = retrieve_seed_fn(question)
    # Một mapping duy nhất (key -> nguyên văn) giữ state loop: key là membership
    # (visited), value None là entry membership-only cho seed_dieu không có
    # context riêng - không render, không thay bằng excerpt
    # Key nội bộ duy nhất cho từng seed: số Điều thật, hoặc số âm theo vị trí khi
    # không parse được hoặc trùng ID đã có, để mọi chuỗi seed đều được giữ nguyên
    collected: dict[int, str | None] = {}
    for index, article in enumerate(seed_contexts):
        dieu = dieu_from_title(article)
        if dieu > 0 and dieu not in collected:
            collected[dieu] = article
        else:
            collected[-(index + 1)] = article
    for dieu in seed_dieu:
        if dieu > 0 and dieu not in collected:
            collected[dieu] = None
    gate_calls = 0
    follow_count = 0
    termination_reason = "follow_limit_reached"
    # Seed thật được chốt ngay sau hop 0, follow không được tính vào seed
    seed_dieu = sorted(key for key in collected if key > 0)
    gate_decisions: list[dict[str, Any]] = []

    logger.info(
        "agent_start seed_dieu=%s max_follow_articles=%d",
        seed_dieu,
        max_follow_articles,
    )

    while follow_count < max_follow_articles:
        candidates, mentions_by_source = _build_frontier(
            collected, extract_citations_fn, internal_dieu
        )

        if not candidates:
            termination_reason = "no_candidates"
            break

        observation = _build_gate_observation(collected, mentions_by_source)
        gate_calls += 1
        logger.info(
            "gate_call=%d candidates=%s observation_chars=%d",
            gate_calls,
            candidates,
            len(observation),
        )

        decision = decide_fn(question, observation, candidates)
        gate_decisions.append(
            {
                "call": gate_calls,
                "candidates": list(candidates),
                "stop": decision.stop,
                "follow_dieu": None if decision.stop else decision.dieu,
            }
        )
        logger.info(
            "gate_call=%d follow_request=%s",
            gate_calls,
            None if decision.stop else decision.dieu,
        )

        if decision.stop:
            termination_reason = "gate_stop"
            break

        dieu = decision.dieu
        if dieu not in candidates:
            termination_reason = "invalid_decision"
            logger.warning("invalid_decision gate_call=%d dieu=%s", gate_calls, dieu)
            break

        article = get_article_fn(dieu)
        if article is None:
            termination_reason = "article_not_found"
            logger.warning("article_not_found gate_call=%d dieu=%s", gate_calls, dieu)
            break

        follow_count += 1
        collected[dieu] = article
        logger.info(
            "article_fetched dieu=%d follow_count=%d collected_dieu=%s",
            dieu,
            follow_count,
            sorted(key for key in collected if key > 0),
        )

    logger.info(
        "agent_finished reason=%s gate_calls=%d follow_count=%d collected_dieu=%s",
        termination_reason,
        gate_calls,
        follow_count,
        sorted(key for key in collected if key > 0),
    )

    # Tóm tắt riêng cho nhánh budget cạn: build lại frontier để biết candidate nào
    # bị bỏ dở; event tên riêng, tránh tiền tố gate_call= gây hiểu nhầm vừa gọi LLM
    # Không kèm observation_chars vì không có observation mới nào được gửi gate
    if termination_reason == "follow_limit_reached":
        remaining_candidates, _ = _build_frontier(
            collected, extract_citations_fn, internal_dieu
        )
        logger.info(
            "follow_limit_summary reason=%s gate_calls=%d follow_count=%d "
            "remaining_candidates=%s remaining_candidate_count=%d",
            termination_reason,
            gate_calls,
            follow_count,
            remaining_candidates,
            len(remaining_candidates),
        )

    if telemetry is not None:
        telemetry.update(
            {
                "seed_dieu": seed_dieu,
                "collected_dieu": sorted(key for key in collected if key > 0),
                "gate_calls": gate_calls,
                "follow_count": follow_count,
                "termination_reason": termination_reason,
                "gate_decisions": gate_decisions,
            }
        )

    collected_contexts = [article for article in collected.values() if article is not None]
    return collected_contexts, {key for key in collected if key > 0}
