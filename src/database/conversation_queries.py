"""Các SQL queries thực hiện async"""

from src.database.message_normalizer import normalize_message

def insert_title_conversations(conn, conv_id: str, user_id: str, title: str):
    """Upsert title của hội thoại và nhét user_id, conv_id"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, title)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO UPDATE
                SET title = EXCLUDED.title,
                updated_at = NOW()
            """,
            (conv_id, user_id, title),
        )
    conn.commit()

def fetch_conversation_owner(conn, conversation_id: str) -> str | None:
    """Trả user_id sở hữu conversation, None nếu conversation chưa tồn tại"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()

    return row[0] if row else None


def claim_conversation(conn, conversation_id: str, user_id: str) -> None:
    """Eager-create row conversation với title NULL (không bao giờ là placeholder)

    Idempotent: nếu row đã có thì không đụng gì, kể cả title đã sinh xong
    Ownership phải được check TRƯỚC khi gọi hàm này
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, title)
            VALUES (%s, %s, NULL)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            (conversation_id, user_id),
        )


def get_user_conversations(conn, user_id: str):
    with conn.cursor() as cur:
        #truy cập vào bảng conversations để lấy title 
        #title được tạo ra với mỗi hội thoại, tức tên cuộc hội thoại 
        cur.execute(
            "SELECT conversation_id, title FROM conversations WHERE user_id = %s ORDER BY created_at DESC",
            (user_id,)
        ) #lấy conversation gần nhất được tạo

        return cur.fetchall()

def get_conversation_messages(conn, conversation_id: str) -> list[dict]:
    """Load lịch sử tin nhắn và chuẩn hoá messages để render UI"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT message FROM chat_history WHERE session_id = %s ORDER BY id ASC",
            (conversation_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return []
    
    messages: list[dict] = []
    for row in rows:
        normalized = normalize_message(row[0])
        if normalized:
            messages.append(normalized)

    return messages