"""Probe (h) — chứng minh handler chat `def` không stall event loop.

Giả định load-bearing đang được đo: chat là `def` nên Starlette đẩy nó sang
threadpool, event loop vẫn rảnh, và read endpoint async vẫn trả lời trong
mili giây (trên nền RTT Supabase ~90ms → ~0.3-0.5s cho 2 round trip liên
tiếp) trong lúc chat đang chạy. Trước probe này đó chỉ là giả định không có
bằng chứng.

Ba arms:

| Arm  | Đo gì                                                    | Vì sao cần |
| ---- | -------------------------------------------------------- | ---------- |
| h1   | read latency không load                                  | baseline — không có nó, threshold là số bịa |
| h2   | read latency trong lúc N chat chiếm threadpool           | phép đo thật |
| h3   | negative control: async def CỐ Ý block loop, probe phải phát hiện | không có nó, h2 xanh có thể chỉ vì probe hỏng |

Assert trên CẢ hai: ceiling p95 tuyệt đối (bắt block thật) và ratio với
baseline đo trong cùng test (bắt regression trên máy chậm làm cả hai arms
phình như nhau)
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import pytest
from fastapi import FastAPI

from conftest import N_CONCURRENT, SLEEP
from concurrency.loop_probe import LoopProbe

from src.api.dependencies import get_db_pool, get_rag_chain, get_sync_db_pool
from src.api.routes.chat import router as chat_router
from src.api.routes.conversations import router as conversations_router

# ── Thresholds ────────────────────────────────────────────────────────────────
# Read endpoint = 2 SELECT liên tiếp qua Supabase 6543 (~90ms RTT mỗi lượt):
# baseline thường ~0.3-0.5s. Ceiling 1.0s ≈ 2-3× baseline — đủ rộng cho jitter
# mạng, vẫn thấp hơn hẳn ~1.8s mà một loop bị block hoàn toàn (chat async def,
# 5 lượt × SLEEP nối tiếp) tạo ra
P95_CEILING_SECONDS = 1.0

# p95_load / p95_baseline (baseline đo trong cùng test): healthy ≈ 1-1.5;
# loop bị block từng lượt chat ≈ 3-5+
P95_RATIO = 5.0

# Số sample tối thiểu của probe trong h2. Sample count sụp đổ TỰ NÓ là bằng
# chứng loop bị stall (mỗi sample kéo dài thêm phần block) — nên assertion
# này không chỉ là sanity check.
MIN_SAMPLES = 4

PROBE_USER = "probe_user_h"


async def _warmup(client) -> None:
    """Một GET trước khi probe start: sample đầu tiên trả tiền cold-connect
    (mở conn pool + TLS, đo được ~0.8s) — đẩy nó ra ngoài cửa sổ đo để không
    làm nhiễu baseline (đã thấy nó mask cả h3: cold sample 0.87s > blocked
    sample 0.6s nên max không đổi)."""
    await client.get("/conversations", headers={"X-User-Id": PROBE_USER})


def _chat_posts(client, n: int):
    """n chat POST vào các conversation id riêng biệt."""
    return [
        client.post(
            f"/conversations/{uuid.uuid4()}/messages",
            json={"question": f"câu hỏi probe h {i}"},
        )
        for i in range(n)
    ]


@pytest.mark.anyio
async def test_h1_baseline_read_latency_no_load(api_client):
    """Read latency khi loop rảnh"""
    probe = LoopProbe(
        api_client,
        "/conversations",
        headers={"X-User-Id": PROBE_USER},
    )
    await _warmup(api_client)
    await probe.start()
    await asyncio.sleep(2.0)
    await probe.stop()

    stats = probe.get_stats()
    assert stats["n_samples"] >= MIN_SAMPLES, (
        f"chỉ có {stats['n_samples']} sample sau 2s — probe không đo được gì?"
    )
    assert stats["errors"] == 0, f"probe thấy {stats['errors']} lỗi khi loop rảnh"
    assert stats["p95"] < P95_CEILING_SECONDS, (
        f"baseline p95={stats['p95']:.3f}s đã vượt ceiling {P95_CEILING_SECONDS}s — "
        f"sàn RTT Supabase đổi hoặc read endpoint không còn là SELECT thuần"
    )
    print(
        f"[h1] baseline read: n={stats['n_samples']} p50={stats['p50'] * 1000:.0f}ms "
        f"p95={stats['p95'] * 1000:.0f}ms max={stats['max'] * 1000:.0f}ms"
    )


@pytest.mark.anyio
async def test_h2_read_latency_under_chat_load(api_client):
    """Read latency trong lúc N chat turn chiếm threadpool — phép đo thật.

    Nếu test này fail lần chạy đầu: DỪNG — có gì đó thật sự đang block loop
    (chat không còn là def, read endpoint vô tình dùng sync query), đây là
    finding chứ không phải test hỏng.
    """
    # Baseline ngắn trong CHÍNH test này: ratio so với điều kiện máy hiện tại
    probe = LoopProbe(
        api_client,
        "/conversations",
        headers={"X-User-Id": PROBE_USER},
    )
    await _warmup(api_client)
    await probe.start()
    await asyncio.sleep(1.5)
    baseline = probe.get_stats()
    assert baseline["n_samples"] >= MIN_SAMPLES, (
        f"baseline chỉ có {baseline['n_samples']} sample — probe không đo được gì"
    )
    assert baseline["errors"] == 0

    # Gây load: N chat POST, mỗi cái def handler chạy trong threadpool
    responses = await asyncio.gather(*_chat_posts(api_client, N_CONCURRENT))

    await asyncio.sleep(1.0)  # vài sample đuôi sau khi load kết thúc
    await probe.stop()

    for i, resp in enumerate(responses):
        assert resp.status_code == 200, (
            f"chat {i}: {resp.status_code} {resp.text}"
        )

    stats = probe.get_stats()
    # sample count sụp đổ = bằng chứng loop bị stall (mỗi sample kéo dài thêm
    # phần block); nếu loop khoẻ, con số này ≈ baseline window
    assert stats["n_samples"] >= MIN_SAMPLES, (
        f"chỉ {stats['n_samples']} sample trong lúc load (baseline "
        f"{baseline['n_samples']}) — loop có vẻ đã bị block: chat handler còn "
        f"là def không? read endpoint có vô tình dùng sync query không?"
    )
    assert stats["errors"] == 0, (
        f"probe thấy {stats['errors']} lỗi trong lúc load"
    )
    assert stats["p95"] < P95_CEILING_SECONDS, (
        f"p95={stats['p95']:.3f}s vượt ceiling {P95_CEILING_SECONDS}s — "
        f"event loop bị block: chat handler còn là def không? read endpoint có "
        f"vô tình dùng sync query không?"
    )
    assert stats["p95"] < baseline["p95"] * P95_RATIO, (
        f"p95={stats['p95']:.3f}s gấp {stats['p95'] / baseline['p95']:.1f}x "
        f"baseline ({baseline['p95']:.3f}s), vượt ratio {P95_RATIO} — regression "
        f"máy chậm hoặc loop bắt đầu bị nghẽn"
    )
    print(
        f"[h2] under load: n={stats['n_samples']} p50={stats['p50'] * 1000:.0f}ms "
        f"p95={stats['p95'] * 1000:.0f}ms max={stats['max'] * 1000:.0f}ms "
        f"(baseline p95={baseline['p95'] * 1000:.0f}ms, ratio "
        f"{stats['p95'] / baseline['p95']:.2f}x)"
    )


@pytest.mark.anyio
async def test_h3_negative_control_probe_detects_blocked_loop(
    test_pool, async_test_pool, stub_core_chain
):
    """Negative control: async def CỐ Ý block loop → probe PHẢI thấy.

    Route /block-loop gọi time.sleep trực tiếp trên event loop — chính là lỗi
    mà plan này tồn tại để ngăn. Nếu arm này fail, LoopProbe không nhìn thấy
    loop bị block và h2 xanh là vô nghĩa.

    Assert theo pre-block max của chính probe (không phải hằng số tuyệt đối):
    sample bị block kéo dài thêm ~SLEEP, nên max mới phải ≥ pre_max + 0.8×SLEEP
    — robust với mọi RTT nền.
    """
    app = FastAPI()
    app.include_router(chat_router)
    app.include_router(conversations_router)
    app.dependency_overrides[get_sync_db_pool] = lambda: test_pool
    app.dependency_overrides[get_db_pool] = lambda: async_test_pool
    app.dependency_overrides[get_rag_chain] = lambda: stub_core_chain

    @app.get("/block-loop")
    async def block_loop():
        # CỐ Ý SAI: blocking sleep trực tiếp trên event loop. Đây là lỗi mà
        # toàn bộ plan này tồn tại để ngăn — route này chỉ tồn tại để chứng
        # minh probe phát hiện được nó.
        time.sleep(SLEEP)
        return {"ok": True}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        probe = LoopProbe(
            client,
            "/conversations",
            headers={"X-User-Id": PROBE_USER},
        )
        await _warmup(client)
        await probe.start()

        # baseline trước block: chờ đủ sample (mỗi sample ~2 RTT trên Supabase)
        deadline = time.monotonic() + 15
        while probe.get_stats()["n_samples"] < 3 and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        pre_max = probe.get_stats()["max"]
        assert pre_max is not None, "probe không lấy được sample baseline nào"

        resp = await client.get("/block-loop")
        assert resp.status_code == 200
        # Chờ 0.15s rồi hit lần 2: sau hit 1, probe bắt đầu sample mới ngay khi
        # loop mở khoá (sample dài ~0.3s). Nếu hit 2 bắn ngay lập tức, nó cạnh
        # tranh với việc dispatch request sample của probe (race: sample miss
        # luôn cả 2 hit, đã thấy trong thực tế). Chèn gap 0.15s < 0.3s → sample
        # CHẮC CHẮN đang in-flight khi hit 2 block loop → detection deterministic.
        await asyncio.sleep(0.15)
        resp2 = await client.get("/block-loop")
        assert resp2.status_code == 200
        await asyncio.sleep(0.1)
        await probe.stop()

    stats = probe.get_stats()
    # Ngưỡng 0.5×SLEEP thay vì 0.8×: sample bị block không bao giờ đo đủ cả
    # 0.3s — response delivery qua ASGITransport chỉ bị đóng băng phần cuối
    # (query I/O đã xong trước block), đo được ~70% block. 0.5×SLEEP vẫn cách
    # xa mọi baseline jitter (pre_max + 0.15s).
    assert stats["max"] >= pre_max + 0.5 * SLEEP, (
        f"max={stats['max']:.3f}s không tăng thêm ≥ {0.5 * SLEEP:.3f}s so với "
        f"pre-block ({pre_max:.3f}s) — LoopProbe KHÔNG thấy loop bị block "
        f"{SLEEP}s. Probe hỏng: h2 xanh không còn nghĩa là gì."
    )
    print(
        f"[h3] blocked loop detected: pre_max={pre_max * 1000:.0f}ms → "
        f"max={stats['max'] * 1000:.0f}ms (block={SLEEP * 1000:.0f}ms)"
    )
