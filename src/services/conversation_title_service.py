"""
    Orchestrator async tạo title cho backend FastAPI.

    Module kết nối database state với LLM title generator: validates user hiện tại, ownership cho conversation, 
    loads first exchange, upsert title, và lưu vào database
"""
from __future__ import annotations

import logging

from psycopg_pool import AsyncConnectionPool

from src.database.conversation_queries_async import (
    fetch_conversation_owner_and_title,
    fetch_first_exchange,
    upsert_conversation_title,
)
from src.services.title_generator import generate_title_async

logger = logging.getLogger(__name__)


async def generate_conversation_title(
    pool: AsyncConnectionPool,
    conversation_id: str,
    user_id: str,
) -> None:
    try:
        async with pool.connection() as conn:
            conversation = await fetch_conversation_owner_and_title(conn, conversation_id)
            if conversation is not None and conversation[0] != user_id: #nếu conversation tồn tại nhưng thuộc về user khác -> không sinh title
                return

            if conversation is not None and conversation[1]: #nếu conversation tồn tại và đã có title -> không sinh title
                return

            #backend mới là sources of truth
            #giả sử frontend gọi vô số lần post /title-generation thì backend vẫn check các điều kiện -> idempotent hơn
            first_exchange = await fetch_first_exchange(conn, conversation_id) # chỉ khi conversation title chưa tồn tại -> lấy các messages đã được chuẩn hoá
            if first_exchange is None:
                return

            first_question, first_answer = first_exchange
            title = await generate_title_async(first_question, first_answer) #chỉ khi có 2 normalized messages trong chat_history -> mới invoke title chain
            if not title:
                return

            await upsert_conversation_title(conn, conversation_id, user_id, title) #rồi mới upsert và tạo title
            await conn.commit()
    except Exception:
        logger.exception("Failed to generate conversation title")
