"""
    API data contracts cho chat endpoint: request body và shape của answer + sources
"""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Body của một lượt chat. conversation_id nằm ở path, KHÔNG nhận từ body"""

    question: str


class SourceItem(BaseModel):
    """Một parent document đã được dùng làm context cho câu trả lời"""

    title: str
    content: str
    doc_id: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


def serialize_sources(docs) -> list[SourceItem]:
    """Map parent Document (result["context"]) (context đã parse từ full result Doc sau invoke) 
    sang shape mà UI đọc được

    Khớp với frontend/components/source_panel.py: title lấy từ metadata,
    content là page_content. Chitchat trả context rỗng -> sources rỗng
    """
    if not docs:
        return []

    items: list[SourceItem] = []
    for index, doc in enumerate(docs):
        metadata = getattr(doc, "metadata", {}) or {}
        items.append(
            SourceItem(
                title=str(metadata.get("title") or f"Nguồn tài liệu #{index + 1}"),
                content=doc.page_content,
                doc_id=str(metadata.get("doc_id") or ""),
            )
        )

    return items
