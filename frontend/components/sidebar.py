import streamlit as st
from src.database.conversation_queries import get_user_conversations
from frontend.services.conversation_loader import load_conversation_into_state

#Sentinel đại diện cho "cuộc trò chuyện mới chưa lưu" (đang ở Hero / vừa New Chat)
NEW_CHAT_SENTINEL = "__new_chat__"


def render_sidebar(db_connection_factory):
    with st.sidebar:
        st.header("💬 Cuộc trò chuyện")
        st.divider()

        conn = db_connection_factory()
        rows = get_user_conversations(conn, st.session_state.user_id)

        options = [(conv_id, title or "Cuộc trò chuyện chưa có tiêu đề") for conv_id, title in rows]
        current_conv_id = st.session_state.conv_id

        option_ids = [conv_id for conv_id, _ in options]
        option_title_map = {conv_id: title for conv_id, title in options}

        # Mấu chốt kiến trúc: selectbox phải LUÔN có 1 option đại diện cho conv đang mở.
        # Khi conv đang mở chưa lưu vào DB (Hero / vừa New Chat), nó không nằm trong list,
        # nên ta chèn 1 sentinel ở đầu. Nhờ vậy "giá trị nên hiển thị" (display_id) LUÔN tồn
        # tại trong options -> selected_id == display_id khi user CHƯA click, và chỉ khác khi
        # user CHỦ ĐỘNG chọn conv khác. Không cần on_change, không cần theo dõi prev-state.
        is_new_chat = current_conv_id not in option_ids
        if is_new_chat:
            option_ids = [NEW_CHAT_SENTINEL] + option_ids
            option_title_map[NEW_CHAT_SENTINEL] = "✨ Cuộc trò chuyện mới"
            display_id = NEW_CHAT_SENTINEL
        else:
            display_id = current_conv_id

        if not option_ids:
            st.caption("Chưa có cuộc trò chuyện nào để tải lại.")
            return

        # Dùng index (KHÔNG dùng key persistent): mỗi rerun selectbox được đặt mặc định về
        # conv đang mở qua index. Vì display_id luôn ∈ option_ids nên index luôn hợp lệ.
        # Không set st.session_state[key] thủ công (sẽ nuốt cú click của user trong cùng run).
        default_index = option_ids.index(display_id)

        selected_id = st.selectbox(
            "Chọn cuộc trò chuyện",
            options=option_ids,
            index=default_index,
            format_func=lambda cid: option_title_map[cid],
        )

        # User chủ động chọn conv khác khi selected_id khác conv đang mở VÀ không phải sentinel.
        if selected_id != display_id and selected_id != NEW_CHAT_SENTINEL:
            load_conversation_into_state(selected_id, db_connection_factory)
            st.rerun()