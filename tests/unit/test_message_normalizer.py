"""Unit tests cho normalize_message - module gộp DUY NHẤT cho message normalization
"""

from __future__ import annotations

import json

import pytest

from src.database.message_normalizer import normalize_message


def _envelope(message_type: str, content) -> dict:
    """Envelope chuẩn mà LangChain (PostgresChatMessageHistory) lưu vào jsonb:
    {"type": ..., "data": {"content": ...}} — production-shaped row."""
    return {"type": message_type, "data": {"content": content}}


class TestRoleMapping:
    def test_human_maps_to_user(self):
        assert normalize_message(_envelope("human", "xin chào")) == {
            "role": "user",
            "content": "xin chào",
        }

    def test_ai_maps_to_ai(self):
        assert normalize_message(_envelope("ai", "chào bạn")) == {
            "role": "ai",
            "content": "chào bạn",
        }

    @pytest.mark.parametrize(
        ("message_type", "expected_role"),
        [("user", "user"), ("assistant", "ai")],
    )
    def test_alias_types_map_too(self, message_type: str, expected_role: str):
        """user/assistant là alias trong role_map của code cũ — giữ hành vi."""
        assert normalize_message(_envelope(message_type, "x"))["role"] == expected_role


class TestJsonStringRow:
    """BUG GUARD: row có thể là JSON string (jsonb cast ::text, driver khác).

    Bản sync cũ parse thành công rồi vẫn return None vì check nhầm raw_message
    (vẫn là str) thay vì parsed_message — đây là lý do task này tồn tại.
    """

    @pytest.mark.parametrize(
        ("message_type", "expected_role"),
        [("human", "user"), ("ai", "ai")],
    )
    def test_json_string_is_parsed_not_dropped(
        self, message_type: str, expected_role: str
    ):
        raw = json.dumps(_envelope(message_type, "nội dung từ json string"))
        assert normalize_message(raw) == {
            "role": expected_role,
            "content": "nội dung từ json string",
        }


class TestContentShapes:
    def test_top_level_content_accepted_without_data_key(self):
        assert normalize_message({"type": "human", "content": "top level"}) == {
            "role": "user",
            "content": "top level",
        }

    def test_list_content_joined_with_newlines(self):
        content = ["dòng 1", {"text": "dòng 2"}, {"không text": "bỏ qua"}, "dòng 3"]
        assert normalize_message(_envelope("ai", content)) == {
            "role": "ai",
            "content": "dòng 1\ndòng 2\ndòng 3",
        }

    def test_list_with_no_text_entries_joins_to_empty_string(self):
        content = [{"không text": 1}, 42, None]
        assert normalize_message(_envelope("human", content)) == {
            "role": "user",
            "content": "",
        }

    def test_empty_list_content_joins_to_empty_string(self):
        assert normalize_message(_envelope("human", [])) == {
            "role": "user",
            "content": "",
        }


class TestUnusableRows:
    """Mọi row không dùng được phải trả None, không raise."""

    @pytest.mark.parametrize(
        ("raw", "case_id"),
        [
            (None, "none-input"),
            ("không phải json", "non-json-string"),
            (42, "non-dict-scalar"),
            (_envelope("system", "nội dung"), "unmapped-role-system"),
            ({"type": "human", "data": {}}, "no-content"),
            ({"type": "human", "data": {"content": None}}, "null-content"),
            ({"data": {"content": "x"}}, "no-type"),
        ],
        ids=[
            "none-input",
            "non-json-string",
            "non-dict-scalar",
            "unmapped-role-system",
            "no-content",
            "null-content",
            "no-type",
        ],
    )
    def test_unusable_rows_return_none(self, raw, case_id: str):
        assert normalize_message(raw) is None
