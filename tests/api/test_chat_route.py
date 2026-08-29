"""Route tests cho POST /conversations/{id}/messages (JSON path, đã de-wrap).

Route tự đọc history (read_history ở borrow-1) rồi đưa vào input của invoke,
memory ghi bằng persist_turn tường minh SAU khi invoke xong. Các biên:

- C1: history được đọc + turn mới được persist - 200, memory_persisted=true,
  2 row cuối trong DB là cặp question/answer của lượt này.
- C2: persist_turn nổ -> vẫn 200 kèm answer, memory_persisted=false (lỗi
  memory không vứt answer đã tốn tiền LLM).
- C3: owner mismatch -> 404, không lộ sự tồn tại của conversation.
"""

from __future__ import annotations

import uuid

import pytest

import src.api.routes.chat as chat_module

DEFAULT_USER = "user_vjp_pro_1"  # default identity của get_current_user_id
OTHER = "user_other"  # <= 15 ký tự: cột conversations.user_id là varchar(15)


def _cid() -> str:
    return str(uuid.uuid4())


@pytest.mark.anyio
async def test_reads_history_and_persists_turn(client, seed_conversation, fetch_messages):
    """C1: 200 + memory_persisted=true + 2 row cuối là turn mới"""
    cid = _cid()
    seed_conversation(cid, DEFAULT_USER, n_pairs=1)

    response = await client.post(
        f"/conversations/{cid}/messages",
        json={"question": "hỏi tiếp"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Stub trả lời: hỏi tiếp"
    assert body["memory_persisted"] is True

    messages = await fetch_messages(cid)
    assert messages[-2:] == [
        {"role": "user", "content": "hỏi tiếp"},
        {"role": "ai", "content": "Stub trả lời: hỏi tiếp"},
    ]


@pytest.mark.anyio
async def test_persist_failure_returns_200_with_false_flag(client, monkeypatch):
    """C2: persist_turn nổ -> vẫn 200, answer vẫn trong body, cờ false"""

    def _boom(*args, **kwargs):
        raise RuntimeError("DB sập lúc ghi")

    monkeypatch.setattr(chat_module, "persist_turn", _boom)
    cid = _cid()

    response = await client.post(
        f"/conversations/{cid}/messages",
        json={"question": "hỏi tiếp"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Stub trả lời: hỏi tiếp"
    assert body["memory_persisted"] is False


@pytest.mark.anyio
async def test_owner_mismatch_is_404(client, seed_conversation):
    """C3: caller mặc định (user_vjp_pro_1) khác owner -> 404, cid không lộ"""
    cid = _cid()
    seed_conversation(cid, OTHER)

    response = await client.post(
        f"/conversations/{cid}/messages",
        json={"question": "xin chào"},
    )

    assert response.status_code == 404
    assert cid not in response.text
