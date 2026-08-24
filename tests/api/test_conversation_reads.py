"""Endpoint tests cho 2 read endpoints (step 1 của cutover).

Ownership: đọc conversation của user khác → 404 chứ không 403 (không lộ sự
tồn tại — giống chat.py:43-45). Gate tầng SQL (fetch_conversation_messages
JOIN conversations + filter user_id) là load-bearing: test cuối gọi thẳng
hàm query, bỏ qua handler, chứng minh handler tương lai quên gate vẫn không
thể leak messages của user khác.

Hai X-User-Id riêng biệt (OWNER / INTRUDER) xuyên suốt. conversation_id dùng
uuid4 vì schema test dùng cột session_id UUID.
"""

from __future__ import annotations

import uuid

import pytest

from src.database.conversation_queries_async import fetch_conversation_messages

OWNER = "test_owner"
INTRUDER = "test_intrud"


def _cid() -> str:
    return str(uuid.uuid4())


@pytest.mark.anyio
async def test_list_empty_for_user_without_conversations(api_client):
    resp = await api_client.get("/conversations", headers={"X-User-Id": OWNER})
    assert resp.status_code == 200
    assert resp.json() == {"conversations": []}


@pytest.mark.anyio
async def test_list_returns_only_callers_conversations(api_client, seed_conversation):
    mine = _cid()
    theirs = _cid()
    seed_conversation(mine, OWNER, title="của tôi")
    seed_conversation(theirs, INTRUDER, title="của họ")

    resp = await api_client.get("/conversations", headers={"X-User-Id": OWNER})
    assert resp.status_code == 200
    body = resp.json()

    ids = [c["conversation_id"] for c in body["conversations"]]
    assert mine in ids
    assert theirs not in ids
    # chỉ có đúng conversation của caller
    assert len(ids) == 1


@pytest.mark.anyio
async def test_list_preserves_null_title_on_the_wire(api_client, seed_conversation):
    """title null phải nguyên vẹn trên wire — placeholder hiển thị thuộc
    sidebar, không phải backend (giống routes/titles.py trả title: null)."""
    cid = _cid()
    seed_conversation(cid, OWNER, title=None)

    resp = await api_client.get("/conversations", headers={"X-User-Id": OWNER})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversations"] == [{"conversation_id": cid, "title": None}]


@pytest.mark.anyio
async def test_messages_normalized_roles_in_order_and_echo(api_client, seed_conversation):
    cid = _cid()
    seed_conversation(cid, OWNER, n_pairs=3)

    resp = await api_client.get(
        f"/conversations/{cid}/messages", headers={"X-User-Id": OWNER}
    )
    assert resp.status_code == 200
    body = resp.json()

    # echo conversation_id + role chuẩn hoá user/ai theo đúng thứ tự insert
    assert body["conversation_id"] == cid
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "ai", "user", "ai", "user", "ai"]
    assert body["messages"][0]["content"] == "câu hỏi 0"
    assert body["messages"][1]["content"] == "trả lời 0"
    assert body["messages"][5]["content"] == "trả lời 2"


@pytest.mark.anyio
async def test_owned_conversation_without_messages_is_200_empty(
    api_client, seed_conversation
):
    """Tồn tại + của mình + chưa có message → 200 kèm list rỗng, KHÔNG 404.

    Đây chính là lý do handler vẫn phải lookup owner trước: riêng SQL join
    không phân biệt được case này với 404 — hai case này chỉ tách được nhờ
    lớp handler.
    """
    cid = _cid()
    seed_conversation(cid, OWNER, n_pairs=0)

    resp = await api_client.get(
        f"/conversations/{cid}/messages", headers={"X-User-Id": OWNER}
    )
    assert resp.status_code == 200
    assert resp.json() == {"conversation_id": cid, "messages": []}


@pytest.mark.anyio
async def test_reading_others_conversation_is_404_and_leaks_nothing(
    api_client, seed_conversation
):
    """Conversation của user khác → 404, và body không lộ conversation id."""
    cid = _cid()
    seed_conversation(cid, INTRUDER, n_pairs=1)

    resp = await api_client.get(
        f"/conversations/{cid}/messages", headers={"X-User-Id": OWNER}
    )
    assert resp.status_code == 404
    assert cid not in resp.text


@pytest.mark.anyio
async def test_unknown_conversation_id_is_404(api_client):
    resp = await api_client.get(
        f"/conversations/{_cid()}/messages", headers={"X-User-Id": OWNER}
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_fetch_messages_direct_call_enforces_ownership_in_sql(
    api_client, async_test_pool, seed_conversation
):
    """LOAD-BEARING: gọi thẳng fetch_conversation_messages, bỏ qua handler.

    Non-owner nhận list rỗng, owner nhận đủ — gate nằm trong SQL (JOIN +
    filter c.user_id), không uỷ thác cho caller. Nếu test này fail, một
    handler tương lai quên gate sẽ mở lại IDOR.
    """
    cid = _cid()
    seed_conversation(cid, OWNER, n_pairs=2)

    intruder_msgs = await fetch_conversation_messages(async_test_pool, cid, INTRUDER)
    assert intruder_msgs == []

    owner_msgs = await fetch_conversation_messages(async_test_pool, cid, OWNER)
    assert len(owner_msgs) == 4
    assert [m["role"] for m in owner_msgs] == ["user", "ai", "user", "ai"]
