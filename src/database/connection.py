"""
    Low-level sync connection opener — KHÔNG phụ thuộc Streamlit

    - Frontend: bọc cache_resource ở frontend/services/connection_provider.py
    - Memory path (get_session_history): hiện tạm gọi get_db_connection()
      trực tiếp; sẽ chuyển sang sync pool (src/database/sync_connection.py)
"""

import os
import psycopg
from dotenv import load_dotenv
from psycopg.errors import OperationalError

load_dotenv()
#check if connection still working - if not, connect cái mới vứt đi cache connect cũ
def is_connection_alive(conn) -> bool:
    try:
        if conn.closed:
            return False
        #nếu connection có vẻ mở nhưng db server đã sập
        conn.execute("SELECT 1")
        return True

    except (OperationalError, psycopg.Error):
        return False


def get_db_connection():
    """Mở một connection sync MỚI mỗi lần gọi (không cache, không pool)"""
    DATABASE_URL = os.environ.get("DATABASE_URL")

    conn = psycopg.connect(DATABASE_URL)
    return conn