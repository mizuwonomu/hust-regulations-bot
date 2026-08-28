"""Unit tests cho 4 event schema của SSE stream.

Mỗi model là phần `data:` của một frame; tên event trên wire = tên frame
(sources / token / done / error). Test semantic: parse lại bằng json.loads
chứ không so byte - byte framing (event:/data:) là việc của route, không
phải của schema. SourcesEvent phải tái dùng đúng SourceItem + serialize_sources
của JSON endpoint để shape nguồn không drift khỏi source_panel.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from src.api.schemas.chat import SourceItem
from src.api.schemas.chat_stream import (
    DoneEvent,
    ErrorEvent,
    SourcesEvent,
    TokenEvent,
)


def test_sources_event_reuses_json_endpoint_source_shape():
    """SourcesEvent bọc list[SourceItem] - item trên wire đúng 3 khóa
    {title, content, doc_id} mà source_panel.py đọc, không thừa không thiếu"""
    event = SourcesEvent(
        sources=[SourceItem(title="Điều 10", content="nội dung", doc_id="abc")]
    )
    data = json.loads(event.to_json())

    assert data == {
        "sources": [{"title": "Điều 10", "content": "nội dung", "doc_id": "abc"}]
    }


def test_sources_event_from_docs_maps_through_serialize_sources():
    """from_docs đi qua serialize_sources của JSON endpoint - một hàm, hai consumer"""
    doc = SimpleNamespace(
        page_content="nội dung cha",
        metadata={"title": "Điều 12", "doc_id": "doc-42"},
    )
    event = SourcesEvent.from_docs([doc])

    assert event.sources == [
        SourceItem(title="Điều 12", content="nội dung cha", doc_id="doc-42")
    ]
    data = json.loads(event.to_json())
    assert list(data["sources"][0].keys()) == ["title", "content", "doc_id"]


def test_sources_event_chitchat_empty_context_is_empty_list():
    """Chitchat trả context rỗng - sources phải là list rỗng, không null"""
    event = SourcesEvent.from_docs([])
    assert json.loads(event.to_json()) == {"sources": []}


def test_token_event_carries_single_delta():
    """TokenEvent là một delta text duy nhất - một khóa, đúng tên event token"""
    event = TokenEvent(text="Xin chào")
    data = json.loads(event.to_json())

    assert data == {"text": "Xin chào"}


def test_done_event_reports_memory_persisted_false():
    """DoneEvent(memory_persisted=False) serialize ra false - client đọc khóa này
    để quyết định soft warning"""
    data = json.loads(DoneEvent(memory_persisted=False).to_json())

    assert data == {"memory_persisted": False}


def test_error_event_carries_message():
    """ErrorEvent mang message cho mid-stream error sau khi 200 đã commit"""
    data = json.loads(ErrorEvent(message="chain gãy giữa stream").to_json())

    assert data == {"message": "chain gãy giữa stream"}


def test_event_names_match_wire_contract():
    """Tên event trên wire khớp 4 frame type - route đọc ClassVar này để
    đóng khung `event: <name>`, sai tên ở đây là vỡ contract với client"""
    assert SourcesEvent.event == "sources"
    assert TokenEvent.event == "token"
    assert DoneEvent.event == "done"
    assert ErrorEvent.event == "error"
