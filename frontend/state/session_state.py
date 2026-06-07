"""
Streamlit init state
"""

import uuid
import streamlit as st

def bootstrap_session_state() -> None:
    """Khởi tạo các session state ban đầu"""
    defaults = {
        "user_id": "user_vjp_pro_1", #đầu tiên sẽ cố định user_id để test
        "conv_id": str(uuid.uuid4()), #khởi tạo conversation_id nếu chưa có hay lần đầu vào web
        "messages": [], #không có messages, khởi tạo hero section
        "title_generation_started": set(), #trigger bắt đầu khởi tạo title
        "selected_conversation_id": None, #id của conversation được chọn từ sidebar
        "conversation_selectbox_id": None, #conv id của riêng selectbox -> ngăn chặn overlap với conv_id mới khi tạo new chat 
        "load_selected_conversation": False, #flag để check nếu có yêu cầu load conversation hiện tại được chọn
        "last_context": [],
        "pending_sidebar_title_sync": False, #flag check nếu message state hiện tại > 2 -> rerun để render sidebar title
    }
    for key,value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation_state() -> None:
    """Reset state khi reload, hoặc trigger new chat"""
    st.session_state.messages = []
    st.session_state.conv_id = str(uuid.uuid4())
    st.session_state.selected_conversation_id = None
    st.session_state.conversation_selectbox_id = None
    st.session_state.load_selected_conversation = False
    st.session_state.pending_sidebar_title_sync = False