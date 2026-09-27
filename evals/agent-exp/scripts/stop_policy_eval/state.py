"""Dựng frontier và observation STOP từ snapshot mà không khởi tạo retrieval store"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.ingestion.reference_parser import _REFERENCE

_ARTICLE_HEADING = re.compile(
    r"^(?:#{1,6}\s+)?(?:[A-ZÀ-Ỹ0-9][A-ZÀ-Ỹ0-9\s]*-\s+)?Điều\s+(\d+)\s*\."
)


@dataclass(frozen=True, slots=True)
class CitationMention:
    """Lưu số Điều nguồn, Điều đích và dòng chứa dẫn chiếu"""

    source_dieu: int
    target_dieu: int
    excerpt: str


def _dieu_from_title(article: str) -> int:
    """Đọc số Điều từ heading đầu tiên của article"""
    first_line = article.split("\n", 1)[0].strip()
    match = _ARTICLE_HEADING.match(first_line)
    return int(match.group(1)) if match is not None else 0


def _article_title(article: str) -> str | None:
    """Trả heading đầu tiên nếu article có format Điều canonical"""
    first_line = article.split("\n", 1)[0].strip()
    match = _ARTICLE_HEADING.match(first_line)
    if match is None:
        return None
    return re.sub(r"^#{1,6}\s+", "", first_line).strip()


def _source_dieu(key: int, article: str) -> int:
    """Lấy số Điều thật kể cả khi seed bị giữ dưới key âm"""
    return key if key > 0 else max(0, _dieu_from_title(article))


def _normalize_seed(row: Any) -> dict[int, str | None]:
    """Tạo mapping ordered từ context và Điều membership của snapshot row"""
    collected: dict[int, str | None] = {}
    for index, article in enumerate(row.contexts):
        dieu = _dieu_from_title(article)
        if dieu > 0 and dieu not in collected:
            collected[dieu] = article
        else:
            collected[-(index + 1)] = article
    for dieu in row.seed_dieu:
        if dieu > 0 and dieu not in collected:
            collected[dieu] = None
    return collected


def _extract_mentions(source_dieu: int, article: str) -> list[CitationMention]:
    """Trích dòng dẫn chiếu theo regex canonical và giữ thứ tự xuất hiện"""
    mentions: list[CitationMention] = []
    seen: set[tuple[int, str]] = set()
    for line in article.replace("\r\n", "\n").split("\n"):
        for match in _REFERENCE.finditer(line):
            key = (int(match.group("dieu")), line.strip())
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                CitationMention(
                    source_dieu=source_dieu,
                    target_dieu=key[0],
                    excerpt=key[1],
                )
            )
    return mentions


def _build_frontier(
    collected: dict[int, str | None],
    internal_dieu: set[int],
) -> tuple[list[int], dict[int, list[CitationMention]]]:
    """Dựng candidates theo thứ tự dẫn chiếu từ context đã thu thập"""
    mentions_by_source: dict[int, list[CitationMention]] = {}
    candidates: list[int] = []
    seen: set[int] = set()
    for key, article in collected.items():
        if article is None:
            continue
        source = _source_dieu(key, article)
        mentions = _extract_mentions(source, article)
        mentions_by_source[key] = mentions
        for mention in mentions:
            target = mention.target_dieu
            if (
                target == source
                or target in collected
                or target in seen
                or target not in internal_dieu
            ):
                continue
            seen.add(target)
            candidates.append(target)
    return candidates, mentions_by_source


def _build_observation(
    collected: dict[int, str | None],
    mentions_by_source: dict[int, list[CitationMention]],
) -> str:
    """Render observation gọn từ heading và các dòng dẫn chiếu"""
    sections = ["Các Điều đã thu thập:"]
    for key, article in collected.items():
        if article is None:
            continue
        source = _source_dieu(key, article)
        title = _article_title(article)
        excerpts = list(
            dict.fromkeys(
                mention.excerpt
                for mention in mentions_by_source.get(key, [])
                if mention.target_dieu != source
            )
        )
        label = str(source) if source > 0 else "?"
        header = f"[Context {label}] {title}" if title is not None else f"[Context {label}]"
        lines = [header]
        lines.extend(f"- {excerpt}" for excerpt in excerpts)
        sections.append("\n".join(lines))
    return "\n".join(sections)


def rebuild_collected_state(
    collected: dict[int, str | None],
    internal_dieu: set[int],
) -> tuple[str, list[int]]:
    """Dựng lại gate state từ mapping collected đã advance đến đúng một hop"""
    candidates, mentions_by_source = _build_frontier(collected, internal_dieu)
    observation = _build_observation(collected, mentions_by_source)
    return observation, candidates


def rebuild_case_state(snapshot: Any, row: Any) -> tuple[str, list[int]]:
    """Tính lại observation và candidates chỉ từ snapshot đã đóng băng"""
    return rebuild_collected_state(_normalize_seed(row), snapshot.internal_dieu)
