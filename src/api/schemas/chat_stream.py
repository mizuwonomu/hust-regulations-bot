"""
    Event schema cho SSE token stream

    Bốn frame type, tên event trên wire trùng tên class bỏ hậu tố Event:
    sources / token / done / error. Mỗi model serialize thành chuỗi JSON -
    chính là phần `data:` của frame SSE.

    SourcesEvent tái dùng SourceItem + serialize_sources của JSON endpoint 
    để shape nguồn trên wire khớp từng bit với ChatResponse 
    - source_panel.py đọc {title, content, doc_id} chung cho cả
    hai đường.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel

from src.api.schemas.chat import SourceItem, serialize_sources


class StreamEvent(BaseModel):
    """Nền chung của 4 frame: biết tên event của mình và serialize ra JSON"""

    event: ClassVar[str]

    def to_json(self) -> str:
        # Model_dump_json là nguồn duy nhất của phần data: - không so byte ở client
        return self.model_dump_json()


class SourcesEvent(StreamEvent):
    """Frame nguồn: list parent document dùng làm context, emit MỘT lần duy nhất"""

    event: ClassVar[str] = "sources"

    sources: list[SourceItem]

    # Đầu vào route là Langchain Document, nên khi dựng SourcesEvent 
    # phải map Document -> SourceItem qua hàm serialize
    @classmethod
    def from_docs(cls, docs) -> SourcesEvent:
        #qua serialize_sources của JSON endpoint - một hàm, hai consumer, không drift
        # data của một frame SSE bắt buộc là chuỗi (byte trên dây)
        # List source sẽ được serialize thành chuỗi JSON 
        return cls(sources=serialize_sources(docs))


class TokenEvent(StreamEvent):
    """Frame delta: một mảnh text của answer, emit liên tục trong lúc gen"""

    event: ClassVar[str] = "token"

    text: str


class DoneEvent(StreamEvent):
    """Frame kết thúc: báo cờ persist memory để client quyết định soft warning"""

    event: ClassVar[str] = "done"

    memory_persisted: bool


class ErrorEvent(StreamEvent):
    """Frame lỗi: mid-stream error sau khi 200 đã commit - lỗi đi qua wire
    dưới dạng event chứ không thể raise nữa"""

    event: ClassVar[str] = "error"

    message: str
