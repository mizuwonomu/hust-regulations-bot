"""
    Định nghĩa API data contracts: Fields, types, status, etc. (input/output shapes)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

TitleGenerationStatus = Literal["scheduled", "already_ready", "not_ready"] 
"""Kết quả status của việc POST scheduling request

   - scheduled: request đã accept, background job được thêm vào
   - already_ready: title đã tồn tại không cần LLM sinh thêm
   - not_ready: chưa thể schedule do first exchange chưa tồn tại (tức chưa có messages) hoặc conversation chưa tồn tại
"""

TitleStatus = Literal["pending", "ready", "missing", "failed"]
#POST /title-generation -> Xác nhận liệu backend đã chấp nhận/schedule việc sinh title
#GET /title -> Xác nhận title đã available hay chưa

class TitleGenerationResponse(BaseModel):
    """Validate conv_id và status scheduling sinh title"""

    conversation_id: str
    status: TitleGenerationStatus

class ConversationTitleResponse(BaseModel):
    """Validate status của title đã avaliable hay chưa,"""

    conversation_id: str
    status: TitleStatus
    title: str | None = None
