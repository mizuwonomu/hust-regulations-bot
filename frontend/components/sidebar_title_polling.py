"""
    Streamlit polling helper cho title đã được tạo ở backend
    
    Chỉ chạy khi title sync state là pending, check FastAPI title endpoint từ client
    và trigger rerun mỗi khi sidebar nên reload title đã lưu cho cuộc hội thoại.
"""
from __future__ import annotations

import streamlit as st

from frontend.services.title_generation_client import get_title_status


@st.fragment(run_every="3s")
def poll_sidebar_title() -> None:
    if not st.session_state.get("pending_sidebar_title_sync"):
        return

    conversation_id = st.session_state.get("conv_id")
    user_id = st.session_state.get("user_id")
    if not conversation_id or not user_id:
        st.session_state.pending_sidebar_title_sync = False
        return

    payload = get_title_status(conversation_id, user_id)
    if payload is None:
        return

    status = payload.get("status")
    if status == "ready":
        st.session_state.pending_sidebar_title_sync = False
        st.rerun()
    elif status in {"missing", "failed"}:
        st.session_state.pending_sidebar_title_sync = False
