"""Sampler đo latency của async endpoint NGAY TRÊN event loop — khác PoolProbe.

PoolProbe (probe.py) đo pool sync bằng thread riêng, vì get_stats() là sync
và không bao giờ chạm event loop — thread là đúng chỗ cho nó. LoopProbe đo
CHÍNH event loop: nó phải await một HTTP request trên CÙNG loop mà app chạy.
Sample từ một thread khác sẽ đo một loop KHÁC và vẫn pass kể cả khi loop của
app bị block hoàn toàn.
"""

from __future__ import annotations

import asyncio
import time


class LoopProbe:
    """Đo latency của một GET async endpoint trong lúc caller gây load.

    start() tạo sampling task và nhường control một lần để sample đầu tiên
    rơi xuống TRƯỚC khi caller bắt đầu gây load — nếu không, baseline đo
    thiếu phần đầu. stop() set event và await task (timeout-based wait thay
    vì bare sleep để stop phản hồi tức thì).
    """

    def __init__(
        self,
        client,
        path: str,
        headers: dict | None = None,
        interval: float = 0.01,
    ):
        self._client = client
        self._path = path
        self._headers = headers
        self._interval = interval
        self._latencies: list[float] = []
        self._errors = 0
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())
        # nhường control 1 lần: sample đầu tiên chạy trước khi caller gây load
        await asyncio.sleep(0)

    async def stop(self) -> None:
        if self._task is not None:
            self._stop.set()
            await self._task
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                response = await self._client.get(self._path, headers=self._headers)
                if response.status_code == 200:
                    self._latencies.append(time.monotonic() - t0)
                else:
                    self._errors += 1
            except Exception:
                # exception = lỗi transport/loop — vẫn đếm là lỗi, không crash probe
                self._errors += 1

            # chờ stop event hoặc interval — timeout-based, không bare sleep
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass

    def get_stats(self) -> dict:
        """Percentile summary; mọi latency đo bằng GIÂY. Không raise khi 0 sample."""
        if not self._latencies:
            return {
                "n_samples": 0,
                "p50": None,
                "p95": None,
                "max": None,
                "errors": self._errors,
                "timeline": [],
            }

        sorted_lats = sorted(self._latencies)
        n = len(sorted_lats)

        def percentile(p: float) -> float:
            return sorted_lats[min(n - 1, int(n * p))]

        return {
            "n_samples": n,
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": sorted_lats[-1],
            "errors": self._errors,
            "timeline": self._latencies,
        }
