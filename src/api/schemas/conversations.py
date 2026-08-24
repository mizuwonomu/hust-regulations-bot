"""
    API data contracts cho 2 read endpoints:
    GET /conversations và GET /conversations/{id}/messages.
"""

from __future__ import annotations

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    """Một nút trong sidebar: id + title"""

    conversation_id: str
    title: str | None = None


class ConversationListResponse(BaseModel):
    """Danh sách sidebar, payload của `GET /conversations` """
    conversations: list[ConversationSummary]


class ConversationMessage(BaseModel):
    """Một tin nhắn đã chuẩn hoá: role user/ai + content"""

    role: str
    content: str


class ConversationMessagesResponse(BaseModel):
    """Toàn bộ message của một `conversation_id` """
    conversation_id: str
    messages: list[ConversationMessage]
