import streamlit as st
from frontend.services.feedback_repo import save_feedback

def render_feedback():
    if not (st.session_state.messages and st.session_state.messages[-1]["role"] == "ai"):
        return

    msg_len = len(st.session_state.messages)

    @st.dialog("👉 Tại sao bạn không thích câu này?")
    def feedback_dialog():
        reasons = st.multiselect(
            "Chọn vấn đề bạn gặp phải:",
            ["Thông tin không chính xác", "Thiếu thông tin", "Thông tin thừa thãi", "Văn phong không phù hợp"],
            key=f"reasons_{msg_len}",
        )
        other_comment = st.text_area("Ghi rõ hơn (nếu có):", key=f"comment_{msg_len}")
        if st.button("Gửi đánh giá chi tiết", key=f"btn_gb_{msg_len}"):
            last_msg = st.session_state.messages[-1]
            last_user_msg = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {"content": "Unknown"}
            save_feedback(last_user_msg["content"], last_msg["content"], "Dislike", reason=", ".join(reasons), comment=other_comment)
            st.success("Đã ghi nhận, cảm ơn ý kiến của bạn🙏! Phản hồi này sẽ được dùng để cải thiện chất lượng bot!")

    st.write("---")
    st.caption("Bạn thấy câu trả lời thế nào? (Hãy phản hồi!)")
    col_fb, _ = st.columns([1, 4])
    with col_fb:
        feedback = st.feedback("thumbs", key=f"fb_{msg_len}")

    if feedback == 0:
        feedback_dialog()
    elif feedback == 1:
        last_msg = st.session_state.messages[-1]
        last_user_msg = st.session_state.messages[-2] if len(st.session_state.messages) > 1 else {"content": "Unknown"}
        save_feedback(last_user_msg["content"], last_msg["content"], "Like")
        st.toast("Cảm ơn bạn đã đóng góp ý kiến🙏!", icon="💾")