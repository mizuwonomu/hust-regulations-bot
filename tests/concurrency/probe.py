"""Sampler nhỏ đo pool sync trong lúc test — vẽ timeline đo được.

Lấy mẫu get_stats() của ConnectionPool theo chu kỳ ngắn để dựng timeline
requests_waiting / pool_available:

- test (b) dùng để xác nhận đã thật sự gây starvation (request xếp hàng
  chứ không ngẫu nhiên chạy tuần tự) — requests_waiting phải có lúc > 0.
- test (c) dùng để xác nhận không leak connection — pool_available phải trở
  về đủ max_size sau khi một request bị hủy giữa chừng.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class PoolProbe:
    """Bám theo một ConnectionPool sync bằng thread riêng, ghi timeline."""

    def __init__(self, pool, interval: float = 0.005):
        self._pool = pool
        self._interval = interval
        self._stop = threading.Event()
        self._timeline: list[dict] = []
        self._thread = threading.Thread(
            target=self._run, name="pool-probe", daemon=True
        )

    def start(self) -> PoolProbe:
        """Bắt đầu lấy mẫu (thread daemon, không chặn test)."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Dừng lấy mẫu và chờ thread kết thúc."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                stats = self._pool.get_stats()
                self._timeline.append(
                    {
                        "t": time.monotonic(),
                        "waiting": stats.get("requests_waiting", 0),
                        "available": stats.get("pool_available", 0),
                    }
                )
            except Exception:
                logger.debug("pool-probe: get_stats() lỗi, bỏ mẫu", exc_info=True)
            self._stop.wait(self._interval)

    def get_stats(self) -> dict:
        """Tổng hợp timeline thành các chỉ số kiểm tra được."""
        waiting = [m["waiting"] for m in self._timeline]
        available = [m["available"] for m in self._timeline]
        return {
            "n_samples": len(self._timeline),
            "t": [m["t"] for m in self._timeline],
            "requests_waiting": {
                "max": max(waiting) if waiting else 0,
                "timeline": waiting,
            },
            "pool_available": {
                "min": min(available) if available else None,
                "max": max(available) if available else None,
                "timeline": available,
            },
            "saw_contention": bool(waiting) and max(waiting) > 0,
        }
