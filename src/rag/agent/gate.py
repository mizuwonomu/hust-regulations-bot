"""Orchestrate một quyết định dừng hoặc follow của citation gate.

Đọc câu hỏi và observation rồi trả về Decision nghiêm ngặt.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.rag.agent.prompt import build_decision_grammar, render_gate_prompt
from src.rag.agent.schema import Decision


def decide_next(
    question: str,
    observation: str,
    candidates: list[int],
    *,
    client: ChatOpenAI,
) -> Decision:
    """Hỏi gate một lần và validate quyết định trả về.

    Params:
    - question: Câu hỏi gốc của người dùng.
    - observation: Nội dung các Điều đã thu thập.
    - candidates: Các Điều hợp lệ chưa được thăm.
    - client: Client đã bind tới llama-server.
    """
    if not candidates:
        raise ValueError("candidates rỗng")

    prompt = render_gate_prompt(question, observation, candidates)
    grammar = build_decision_grammar(candidates)

    request_extra_body = dict(client.extra_body or {})
    request_extra_body["grammar"] = grammar
    response = client.invoke(prompt, extra_body=request_extra_body)

    decision = Decision.model_validate_json(response.content)
    if not decision.stop and decision.dieu not in candidates:
        raise ValueError(f"gate chọn Điều {decision.dieu} ngoài candidates {candidates}")

    return decision
