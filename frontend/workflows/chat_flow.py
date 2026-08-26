"""
    Module orchestrate frontend Streamlit.

    Khi đủ lượng messages trong state và là first exchange -> trigger sinh title
    và poll sync sidebar để rerun, trả về title hiển thị ra UI
"""
import streamlit as st

from frontend.components.source_panel import render_sources
from frontend.services.chat_client import send_message

def handle_query(question: str, deps):
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    session_id = st.session_state.conv_id
    user_id = st.session_state.user_id

    result = send_message(session_id, user_id, question)

    if result is None:
        # Hard fail (404 / transport): KHÔNG append AI message và KHÔNG trigger
        # title -> để user hỏi lại. User message đã append thì giữ nguyên.
        with st.chat_message("ai"):
            st.error("Không nhận được câu trả lời từ máy chủ. Vui lòng thử lại.")
        return

    with st.chat_message("ai"):
        st.markdown(result.answer)
        render_sources(result.sources)
        if not result.memory_persisted:
            # Soft degrade: answer đã sinh thành công server-side, chỉ phần ghi
            # history thất bại -> cảnh báo, không vứt bỏ lượt chat này.
            st.warning(
                "Câu trả lời đã hiển thị nhưng không được lưu vào lịch sử hội thoại."
            )

    # Lưu câu trả lời của AI vào state để hiển thị
    st.session_state.messages.append({
        "role": "ai",
        "content": result.answer,
        "sources": result.sources,  # avoid losing sources when reload
    })

    message_count = len(st.session_state.messages)  # 2 = first exchange (user + AI)
    is_first_exchange = message_count == 2
    already_scheduled = session_id in st.session_state.title_generation_started

    if is_first_exchange and not already_scheduled:
        status = deps.title_generation_scheduler(session_id, user_id)
        if status in {"scheduled", "already_ready"}:
            st.session_state.title_generation_started.add(session_id)
            st.session_state.pending_sidebar_title_sync = status == "scheduled"
