"""Test schema nghiêm ngặt của corpus loader và smoke trên corpus thật đã freeze."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from evals.router.corpus import (
    CORPUS_SIZE,
    DEFAULT_CORPUS_PATH,
    ROUTE_LABELS,
    CorpusError,
    corpus_sha256,
    gold_counts,
    load_corpus,
)


def _case(index: int, *, label: str = "RAG", group: str = "primary", query: str | None = None) -> dict:
    """Tạo một dòng corpus hợp lệ, query mặc định là duy nhất theo index"""
    return {
        "id": index,
        "query": query if query is not None else f"câu hỏi số {index}",
        "type": label,
        "group": group,
    }


def _valid_rows() -> list[dict]:
    """30 dòng hợp lệ, 15 RAG / 15 chat"""
    labels = ["RAG" if index % 2 == 0 else "chat" for index in range(CORPUS_SIZE)]
    return [_case(index + 1, label=labels[index]) for index in range(CORPUS_SIZE)]


def _dump(tmp_path: Path, rows: object, name: str = "corpus.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def _patch(rows: list[dict], patch: dict) -> list[dict]:
    rows[0] = {**rows[0], **patch}
    return rows


def _drop_key(rows: list[dict], key: str) -> list[dict]:
    rows[0].pop(key)
    return rows


def _replace_row(rows: list[dict], value: object) -> list[dict]:
    rows[0] = value
    return rows


INVALID_ROWS = [
    ("missing-key", lambda rows: _drop_key(rows, "group"), "thiếu key"),
    ("extra-key", lambda rows: _patch(rows, {"notes": "debug"}), "key lạ"),
    ("bool-id", lambda rows: _patch(rows, {"id": True}), "id phải là số nguyên"),
    ("string-id", lambda rows: _patch(rows, {"id": "1"}), "id phải là số nguyên"),
    ("float-id", lambda rows: _patch(rows, {"id": 1.5}), "id phải là số nguyên"),
    ("null-id", lambda rows: _patch(rows, {"id": None}), "id phải là số nguyên"),
    ("int-group", lambda rows: _patch(rows, {"group": 3}), "group phải là string"),
    ("null-group", lambda rows: _patch(rows, {"group": None}), "group phải là string"),
    ("int-query", lambda rows: _patch(rows, {"query": 42}), "query phải là string"),
    ("blank-query", lambda rows: _patch(rows, {"query": "   "}), "query rỗng"),
    ("lowercase-label", lambda rows: _patch(rows, {"type": "rag"}), "type phải là"),
    ("mixed-case-label", lambda rows: _patch(rows, {"type": "Rag"}), "type phải là"),
    ("unknown-label", lambda rows: _patch(rows, {"type": "FAQ"}), "type phải là"),
    ("non-string-label", lambda rows: _patch(rows, {"type": ["RAG"]}), "type phải là string"),
    ("duplicate-query", lambda rows: _patch(rows, {"query": rows[1]["query"]}), "query trùng"),
    ("duplicate-after-trim", lambda rows: _patch(rows, {"query": f"  {rows[1]['query']}  "}), "query trùng"),
    ("row-not-object", lambda rows: _replace_row(rows, "hello"), "mỗi dòng phải là object"),
    ("too-few-rows", lambda rows: rows[:-1], f"cần đúng {CORPUS_SIZE} dòng"),
    ("too-many-rows", lambda rows: [*rows, _case(99, query="câu hỏi thêm")], f"cần đúng {CORPUS_SIZE} dòng"),
]


@pytest.mark.parametrize(
    ("mutate", "pattern"),
    [pytest.param(case[1], case[2], id=case[0]) for case in INVALID_ROWS],
)
def test_rejects_invalid_rows(tmp_path: Path, mutate, pattern: str) -> None:
    """Mỗi fixture âm phải fail đúng lý do của nó"""
    path = _dump(tmp_path, mutate(_valid_rows()))
    with pytest.raises(CorpusError) as excinfo:
        load_corpus(path)
    assert pattern in str(excinfo.value)


def test_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    with pytest.raises(CorpusError, match="file rỗng"):
        load_corpus(path)


def test_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("[{'id': 1},", encoding="utf-8")
    with pytest.raises(CorpusError, match="JSON không hợp lệ"):
        load_corpus(path)


def test_rejects_non_array_root(tmp_path: Path) -> None:
    path = _dump(tmp_path, {"cases": _valid_rows()})
    with pytest.raises(CorpusError, match="root phải là JSON array"):
        load_corpus(path)


def test_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="không đọc được file"):
        load_corpus(tmp_path / "khong-ton-tai.json")


def test_loads_path_passed_as_string(tmp_path: Path) -> None:
    path = _dump(tmp_path, _valid_rows())
    assert len(load_corpus(str(path))) == CORPUS_SIZE


def test_keeps_row_order_and_debug_fields(tmp_path: Path) -> None:
    """id không liên tục và group tự do vẫn được nhận, giữ nguyên, không sort lại"""
    rows = _valid_rows()
    for offset, row in enumerate(rows):
        row["id"] = 1000 - offset * 3
        row["group"] = f"nhóm-{offset}"
    rows[5]["query"] = "  đăng ký học phần  "

    cases = load_corpus(_dump(tmp_path, rows))

    assert [case.id for case in cases] == [1000 - offset * 3 for offset in range(CORPUS_SIZE)]
    assert [case.group for case in cases] == [f"nhóm-{offset}" for offset in range(CORPUS_SIZE)]
    assert [case.query for case in cases] == [
        row["query"].strip() for row in rows
    ]
    assert cases[5].query == "đăng ký học phần"


def test_real_corpus_smoke() -> None:
    cases = load_corpus(DEFAULT_CORPUS_PATH)
    # Đếm nhãn trực tiếp từ file nguồn để oracle không phụ thuộc code production
    raw_rows = json.loads(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8"))
    raw_counts = Counter(row["type"] for row in raw_rows)

    assert raw_counts == {"RAG": 15, "chat": 15}
    assert gold_counts(cases) == dict(raw_counts)
    assert len(cases) == CORPUS_SIZE
    assert {case.type for case in cases} <= set(ROUTE_LABELS)
    assert len({case.query for case in cases}) == CORPUS_SIZE
    assert all(case.query and case.query == case.query.strip() for case in cases)


def test_real_corpus_preserves_source_debug_fields() -> None:
    raw = json.loads(DEFAULT_CORPUS_PATH.read_text(encoding="utf-8"))
    cases = load_corpus(DEFAULT_CORPUS_PATH)

    assert [(case.id, case.group, case.query, case.type) for case in cases] == [
        (row["id"], row["group"], row["query"].strip(), row["type"]) for row in raw
    ]


def test_corpus_hash_tracks_debug_only_edits(tmp_path: Path) -> None:
    """Hash bám byte thô: đổi chỉ id/group vẫn đổi hash, cùng byte thì bằng nhau"""
    rows = _valid_rows()
    original = tmp_path / "original.json"
    original.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    same = tmp_path / "same.json"
    same.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    edited_rows = [dict(row) for row in rows]
    for offset, row in enumerate(edited_rows):
        row["id"] = 500 + offset
        row["group"] = "khác"
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(edited_rows, ensure_ascii=False), encoding="utf-8")

    assert corpus_sha256(original) == corpus_sha256(same)
    assert corpus_sha256(original) != corpus_sha256(edited)
