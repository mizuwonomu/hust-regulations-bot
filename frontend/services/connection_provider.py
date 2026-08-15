"""
    Streamlit-side connection provider (giai đoạn chuyển tiếp).

    src/database/connection.py giờ là hàm thuần (không streamlit). Streamlit
    vẫn cần singleton theo process + liveness check nên bọc cache_resource ở
    đây — y như model_loader.py. Khi frontend chuyển hẳn sang HTTP thì xóa
"""

from __future__ import annotations

import streamlit as st

from src.database.connection import get_db_connection, is_connection_alive


@st.cache_resource(validate=is_connection_alive)
def get_cached_db_connection():
    """Trả về connection sync được cache theo process, tự kiểm tra còn sống"""
    return get_db_connection()
