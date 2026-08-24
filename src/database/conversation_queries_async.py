"""
    Các SQL queries được thực hiện async, các tác vụ khác nhau được thực hiện độc lập.
    Giả sử như user A đang ở bước upsert title, thì user B ở bước fetch title 
    không cần phải đợi event user A hoàn thành.
"""

from __future__ import annotations
from typing import Literal, overload

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from src.database.message_normalizer import normalize_message

TitleState = Literal["pending", "ready", "missing", "failed"]
"""Trả về title state để polling

    - pending: đã gửi request để sinh title
    - ready: title đã sẵn sàng
    - missing: không tìm được hội thoại tương ứng với user id
    - failed: sinh title thất bại
"""

@overload
async def get_conversation_title_state(
    pool: AsyncConnectionPool,
    conversation_id: str,
    user_id: str,
    include_title: Literal[False] = False,
) -> TitleState: 
    """
        Trả về 1 trong 4 status title khi biến include là False.
        Route POST title-generation chỉ cần state nên gọi không cần include title
    """
    ...


@overload
async def get_conversation_title_state(
    pool: AsyncConnectionPool,
    conversation_id: str,
    user_id: str,
    include_title: Literal[True],
) -> tuple[TitleState, str | None]:
    """
        Trả về state lẫn title khi biến include là True.
        Route GET title-generation cần cả state lẫn title => unpack state, title 
    """
    ...


async def get_conversation_title_state(
    pool: AsyncConnectionPool,
    conversation_id: str,
    user_id: str,
    include_title: bool = False,
):
    """Implementation để trả về state status cùng với các type checker"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT user_id, title
                FROM conversations
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            )
            row = await cur.fetchone()

    if row is None: #Nếu title row chưa có trong conversations
        has_history = await has_chat_history(pool, conversation_id)
        state: TitleState = "pending" if has_history else "missing" #Có message -> có hội thoại, chưa có title row. Chưa hề có message -> conversation chưa hề tồn tại
        title: str | None = None
    
    elif row[0] != user_id: #Nếu có row, nhưng user_id hiện tại khác -> vẫn set thành missing tránh leak title của user khác 
        state = "missing"
        title = None

    else: #Nếu có row title -> gửi state sẵn sàng 
        title = row[1]
        state = "ready" if title else "pending"

    if include_title:
        return state, title
    return state


async def has_chat_history(
    pool: AsyncConnectionPool,
    conversation_id: str,
) -> bool:
    """
        Verifier để kiểm tra với conversation ID hiện tại đã có messages chưa.
    """ 
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT 1
                FROM chat_history
                WHERE session_id = %s
                LIMIT 1
                """,
                (conversation_id,),
            )
            row = await cur.fetchone()

    return row is not None


async def fetch_conversation_owner_and_title(
    conn: AsyncConnection,
    conversation_id: str
) -> tuple[str, str | None] | None:
    """Truy cập vào bảng conversations để lấy title, được tạo ra với mỗi hội thoại, tức tên cuộc hội thoại"""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT user_id, title
            FROM conversations
            WHERE conversation_id = %s
            """,
            (conversation_id,),
        )
        row = await cur.fetchone()

    if row is None:
        return None

    return row[0], row[1]

async def fetch_first_exchange(
    conn: AsyncConnection,
    conversation_id: str,
) -> tuple[str, str] | None:
    """Fetch các messages và chuẩn hoá messages' roles"""

    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT message
            FROM chat_history
            WHERE session_id = %s
            ORDER BY id ASC
            LIMIT 8
            """,
            (conversation_id,),
        )
        rows = await cur.fetchall()

    user_message = None
    ai_message = None

    for row in rows:
        normalized = normalize_message(row[0])
        if not normalized:
            continue

        #lặp qua các rows, nếu đã tìm được 2 messages của user với ai đã được chuẩn hóa
        # -> sẽ break vòng lặp và trả về tuple 2 strings làm input cho chain title
        if normalized["role"] == "user" and user_message is None: 
            user_message = normalized["content"]
        elif normalized["role"] == "ai" and user_message is not None:
            ai_message = normalized["content"]
            break

    if not user_message or not ai_message:
        return None

    return user_message, ai_message


async def upsert_conversation_title(
    conn: AsyncConnection,
    conversation_id: str,
    user_id: str,
    title: str,
) -> None:
    """Upsert title của hội thoại và nhét user_id, conv_id"""
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE
                SET title = EXCLUDED.title,
                    updated_at = NOW()
            """,
            (conversation_id, user_id, title),
        )

async def fetch_conversation_owner_and_title_pool(
    pool: AsyncConnectionPool,
    conversation_id: str,
) -> tuple[str, str | None] | None:
    """Wrapper mỏng lấy owner + title theo pool, thay vì connection"""
    async with pool.connection() as conn:
        return await fetch_conversation_owner_and_title(conn, conversation_id)


async def list_user_conversations(
    pool: AsyncConnectionPool,
    user_id: str,
) -> list[tuple[str, str | None]]:
    """List (conversation_id, title) của một user, mới nhất trước.

    Giữ nguyên ngữ nghĩa của bản sync `get_user_conversations`
    (conversation_queries.py) để UI không đổi hành vi: order by created_at
    DESC, title có thể None (placeholder là quyết định hiển thị của sidebar,
    không phải của tầng dữ liệu). Filter user_id chính là gate"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT conversation_id, title
                FROM conversations
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,),
            )
            rows = await cur.fetchall()

    return [(row[0], row[1]) for row in rows]


async def fetch_conversation_messages(
    pool: AsyncConnectionPool,
    conversation_id: str,
    user_id: str,
) -> list[dict]:
    """Load messages của một conversation, đã chuẩn hoá {role, content}"""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT ch.message
                FROM chat_history ch
                JOIN conversations c
                  ON ch.session_id = c.conversation_id
                WHERE c.conversation_id = %s AND c.user_id = %s
                ORDER BY ch.id ASC
                """,
                (conversation_id, user_id),
            )
            rows = await cur.fetchall()

    messages: list[dict] = []
    for row in rows:
        normalized = normalize_message(row[0])
        if normalized:
            messages.append(normalized)
    return messages