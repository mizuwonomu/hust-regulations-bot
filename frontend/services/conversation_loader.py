"""
    Load một conversation cũ vào Streamlit session state, lấy message qua HTTP
"""

from __future__ import annotations

import streamlit as st

from frontend.services.conversation_client import fetch_messages


def load_conversation_into_state(conversation_id: str, user_id: str) -> bool:
    """Đổ message của conversation vào session state. Trả True nếu thành công.

    Trả bool chứ không None: sidebar chỉ được `st.rerun()` khi load thành công.
    Rerun sau khi fail sẽ quay lại đúng nhánh fail đó -> vòng lặp rerun.

    `fetch_messages` trả None cho CẢ hai trường hợp không đọc được: backend
    chết (transport) và 404 (không tồn tại / không phải của user này). Với UI
    thì cả hai đều là "không mở được cuộc trò chuyện", nên gộp một thông báo.
    """
    messages = fetch_messages(conversation_id, user_id)

    if messages is None:
        st.error("Không tải được cuộc trò chuyện này.")
        return False

    ui_messages = []
    for msg in messages:
        role = msg["role"]
        entry = {"role": role, "content": msg["content"]}
        if role == "ai":
            entry["sources"] = []
        ui_messages.append(entry)

    st.session_state.messages = ui_messages
    st.session_state.conv_id = conversation_id
    return True