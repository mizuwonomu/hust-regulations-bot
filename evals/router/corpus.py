"""Loader corpus định tuyến: schema nghiêm ngặt, giữ id/group chỉ như metadata debug.

Corpus là một JSON array đúng 30 object, mỗi object có đúng bốn key: id, query, type, group
- id: số nguyên, là marker debug/thứ tự do người viết corpus cấp; không bắt buộc liên tục
- group: string tự do, nhãn debug; không có vocabulary cố định
- query: string khác rỗng sau khi trim; đây là thứ duy nhất được gửi cho model
- type: đúng 'RAG' hoặc 'chat', dùng cho scoring

Không sort, group, dedupe hay join theo id/group: thứ tự dòng trong file là case index ổn định
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn

RouteLabel = Literal["RAG", "chat"]

CORPUS_SIZE = 30
ROUTE_LABELS: tuple[RouteLabel, ...] = ("RAG", "chat")
CASE_FIELDS = frozenset({"id", "query", "type", "group"})
DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parent / "datasets" / "corpus_classification.json"
)


class CorpusError(ValueError):
    """Corpus vi phạm data contract"""


@dataclass(frozen=True)
class ClassificationCase:
    """Một case đã validate: id/group là debug, type là gold label đã freeze"""

    id: int
    query: str
    type: RouteLabel
    group: str


def load_corpus(path: Path | str = DEFAULT_CORPUS_PATH) -> list[ClassificationCase]:
    """Đọc và validate corpus, giữ nguyên thứ tự dòng của file

    path: file JSON cần đọc; mặc định là corpus đã freeze của experiment
    """
    raw = _read_bytes(path)
    if not raw.strip():
        _reject(path, "file rỗng")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _reject(path, f"JSON không hợp lệ ({exc})")
    if not isinstance(payload, list):
        _reject(path, "root phải là JSON array")
    if len(payload) != CORPUS_SIZE:
        _reject(path, f"cần đúng {CORPUS_SIZE} dòng, thấy {len(payload)}")
    return _parse_rows(path, payload)


def corpus_sha256(path: Path | str = DEFAULT_CORPUS_PATH) -> str:
    """SHA256 của nội dung file thô - đổi bất kỳ byte nào cũng đổi hash

    path: file corpus cần hash
    """
    return hashlib.sha256(_read_bytes(path)).hexdigest()


def gold_counts(cases: list[ClassificationCase]) -> dict[str, int]:
    """Đếm số case theo gold label, luôn trả đủ hai nhãn kể cả khi bằng 0

    cases: danh sách case đã load
    """
    counts = dict.fromkeys(ROUTE_LABELS, 0)
    for case in cases:
        counts[case.type] += 1
    return counts


def _parse_rows(path: Path | str, payload: list[object]) -> list[ClassificationCase]:
    cases: list[ClassificationCase] = []
    # Query trùng bị chặn để một dòng không thể vô tình tính hai lần vào metric
    seen_queries: dict[str, int] = {}
    for row_index, row in enumerate(payload):
        cases.append(_parse_case(path, row_index, row, seen_queries))
    return cases


def _parse_case(
    path: Path | str,
    row_index: int,
    row: object,
    seen_queries: dict[str, int],
) -> ClassificationCase:
    if not isinstance(row, dict):
        _reject(path, row_index, "mỗi dòng phải là object")
    keys = set(row)
    if missing := CASE_FIELDS - keys:
        _reject(path, row_index, f"thiếu key {sorted(missing)}")
    if extra := keys - CASE_FIELDS:
        _reject(path, row_index, f"key lạ {sorted(extra)}")

    case_id = row["id"]
    # bool là subclass của int nên phải loại riêng
    if isinstance(case_id, bool) or not isinstance(case_id, int):
        _reject(path, row_index, f"id phải là số nguyên, thấy {case_id!r}")

    raw_query = row["query"]
    if not isinstance(raw_query, str):
        _reject(path, row_index, f"query phải là string, thấy {raw_query!r}")
    query = raw_query.strip()
    if not query:
        _reject(path, row_index, "query rỗng sau khi trim")

    label = row["type"]
    if not isinstance(label, str):
        _reject(path, row_index, f"type phải là string, thấy {label!r}")
    if label not in ROUTE_LABELS:
        _reject(path, row_index, f"type phải là 'RAG' hoặc 'chat', thấy {label!r}")

    group = row["group"]
    if not isinstance(group, str):
        _reject(path, row_index, f"group phải là string, thấy {group!r}")

    if (first_row := seen_queries.get(query)) is not None:
        _reject(path, row_index, f"query trùng với dòng {first_row}")
    seen_queries[query] = row_index

    return ClassificationCase(id=case_id, query=query, type=label, group=group)


def _read_bytes(path: Path | str) -> bytes:
    try:
        return Path(path).read_bytes()
    except OSError as exc:
        _reject(path, f"không đọc được file ({exc})")


def _reject(path: Path | str, *parts: object) -> NoReturn:
    raise CorpusError(f"{path}: " + ": ".join(str(part) for part in parts))
