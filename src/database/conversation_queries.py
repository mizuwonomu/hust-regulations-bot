import json

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
    
#vì khi lấy object messages của langchain cho session state của streamlit
#metadata của dict không phải dạng thông thường là {"role": ..., "content": ...} cho streamlit
#->phải chuẩn hoá về đúng dạng role và content từ property message của object langchain
def _normalize_message(raw_message) -> dict | None:
    if raw_message is None:
        return None
    
    parsed_message = raw_message
    if isinstance(raw_message, str):
        try:
            parsed_message = json.loads(raw_message)
        except json.JSONDecodeError:
            return None
        
    if not isinstance(raw_message, dict):
        return None
    
    message_type = parsed_message.get("type")
    message_data = parsed_message.get("data", {})
    content = message_data.get("content", parsed_message.get("content"))

    if isinstance(content, list):
        text_chunks = []
        for chunk in content:
            if isinstance(chunk, str):
                text_chunks.append(chunk)
            elif isinstance(chunk, dict):
                maybe_text = chunk.get("text")
                if maybe_text:
                    text_chunks.append(str(maybe_text))

        content = "\n".join(text_chunks)

    role_map = {
        "human": "user",
        "user": "user",
        "ai": "ai",
        "assistant": "ai"
    }

    role = role_map.get(message_type)

    if role is None or content is None:
        return None
    
    return {"role": role, "content": str(content)}


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
        normalized = _normalize_message(row[0])
        if normalized:
            messages.append(normalized)

    return messages