"""Cung cấp thời gian UTC và chuẩn hóa lỗi cho STOP evaluation"""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> str:
    """Trả timestamp UTC theo định dạng ISO"""
    return datetime.now(UTC).isoformat()


def sanitize_error(error: Exception) -> str:
    """Chuẩn hóa thông báo lỗi và giới hạn độ dài đầu ra"""
    message = " ".join(str(error).split())
    if not message:
        message = type(error).__name__
    return message[:500]
