"""
    Module message normalization để render ra Streamlit
"""

from __future__ import annotations

import json

_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "ai": "ai",
    "assistant": "ai",
}


def normalize_message(raw_message) -> dict | None:
    """Chuẩn hoá một row message từ chat_history về {"role", "content"}.

    - Nhận dict (psycopg trả jsonb dạng dict) hoặc str (JSON) — string sẽ
      được parse; parse fail thì row không dùng được.
    - `content` đọc từ `data.content` (envelope LangChain), fallback content
      top-level. List content được flatten: string giữ nguyên, dict lấy
      `text`, entry không có gì bị bỏ qua, nối nhau bằng newline.
    - Role map: human/user -> "user", ai/assistant -> "ai".

    Trả None khi row không dùng được: input None, JSON không parse được,
    non-dict scalar, role không map được (vd "system"), hoặc thiếu content.
    """
    if raw_message is None:
        return None

    parsed_message = raw_message
    if isinstance(raw_message, str):
        try:
            parsed_message = json.loads(raw_message)
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed_message, dict):
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

    role = _ROLE_MAP.get(message_type)

    if role is None or content is None:
        return None

    return {"role": role, "content": str(content)}