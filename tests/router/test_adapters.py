"""Test hai adapter định tuyến: parity prompt production, rule fallback, và type guard của TypeSafe.

Toàn bộ test dùng fake client/transport - không có network call nào
"""

from __future__ import annotations

import json

import httpx
import httpx2
import pytest
from langchain_core.messages import AIMessage
from langchain_typesafe import (
    Choice,
    ChoiceAnswer,
    ClassifierResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    TypeSafeClassifier,
    Usage,
)

from evals.router.adapters import (
    GROQ_ROUTER_PROMPT,
    JEV_MODEL,
    JEV_ROUTE_INSTRUCTION,
    PRODUCTION_QA_CHAIN_PATH,
    ROUTE_CRITERIA,
    GroqRouterAdapter,
    JevRouterAdapter,
    PromptParityError,
    RouteContractError,
    TokenUsage,
    build_route_questions,
    check_production_prompt_parity,
    executed_branch_from_text,
    extract_route_choice,
    format_is_valid,
    production_route_decision_source,
    production_router_template,
    validate_route_questions,
)

QUERY = "Sinh viên bị cảnh báo học tập mức 2 được đăng ký tối đa bao nhiêu tín chỉ?"


class FakeChat:
    """Fake chat model: trả một AIMessage cố định hoặc raise, và ghi lại input"""

    def __init__(self, message: AIMessage | None = None, error: Exception | None = None) -> None:
        self.message = message
        self.error = error
        self.calls: list[object] = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return self.message


class FakeClassifier:
    """Fake TypeSafe classifier: trả response cố định hoặc raise, và ghi lại request"""

    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[object] = []

    def invoke(self, input, config=None, **kwargs):
        self.calls.append(input)
        if self.error is not None:
            raise self.error
        return self.response


def _choice_answer(choice: str = "RAG", **overrides) -> ChoiceAnswer:
    payload = {
        "type": "choice",
        "choice": choice,
        "probabilities": {"RAG": 0.8, "chat": 0.2},
        "confidence": 0.72,
    }
    return ChoiceAnswer(**{**payload, **overrides})


def _response(**overrides) -> ClassifierResponse:
    payload = {
        "model": JEV_MODEL,
        "answers": {"route": _choice_answer()},
        "usage": Usage(input_tokens=120, output_tokens=4),
        "request_id": "req-abc",
    }
    return ClassifierResponse(**{**payload, **overrides})


def _validation_error(questions: object) -> str:
    with pytest.raises(RouteContractError) as excinfo:
        validate_route_questions(questions)  # type: ignore[arg-type]
    return str(excinfo.value)


# Prompt parity với production


def test_snapshot_matches_production_prompt() -> None:
    assert production_router_template() == GROQ_ROUTER_PROMPT


def test_parity_is_not_fooled_by_tokens_in_a_comment() -> None:
    source = PRODUCTION_QA_CHAIN_PATH.read_text(encoding="utf-8")
    # Rule bị đảo, nhưng một comment vẫn chứa đúng các token cũ
    broken = source.replace(
        "return chat_chain\n        else:\n            return rag_chain",
        "# if \"chat\" in decision: return chat_chain\n            return rag_chain\n        else:\n            return rag_chain",
        1,
    )
    assert broken != source

    with pytest.raises(PromptParityError, match="rule route_decision"):
        check_production_prompt_parity(broken)


def test_parity_ignores_comment_only_changes() -> None:
    source = PRODUCTION_QA_CHAIN_PATH.read_text(encoding="utf-8")
    commented = source.replace(
        "#clean string (xóa khoảng trắng thừa nếu có)",
        "#clean string (xóa khoảng trắng thừa nếu có) - ghi chú thêm",
        1,
    )
    assert commented != source

    # Comment không đổi hành vi nên parity vẫn phải pass
    check_production_prompt_parity(commented)


def test_parity_accepts_unmodified_production_source() -> None:
    check_production_prompt_parity()


def test_parity_failure_reports_missing_template() -> None:
    with pytest.raises(PromptParityError, match="không tìm thấy router_template"):
        production_router_template("router_template_other = 'x'")


def test_parity_failure_reports_missing_rule() -> None:
    with pytest.raises(PromptParityError, match="không tìm thấy route_decision"):
        production_route_decision_source("def other():\n    return 1\n")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "M là một chuyên gia phân loại câu hỏi",
            "M là chuyên gia phân loại câu hỏi",
        ),
        ('decision.strip().lower()', "decision.lower()"),
        ('if "chat" in decision:', 'if "chat" not in decision:'),
        ('if "chat" in decision:', 'if "chat" in decision or True:'),
        (
            "return chat_chain\n        else:\n            return rag_chain",
            "return rag_chain\n        else:\n            return chat_chain",
        ),
    ],
)
def test_parity_detects_drift_in_production_source(old: str, new: str) -> None:
    source = PRODUCTION_QA_CHAIN_PATH.read_text(encoding="utf-8")
    assert old in source
    with pytest.raises(PromptParityError):
        check_production_prompt_parity(source.replace(old, new, 1))


# Rule chọn nhánh và format validity của arm Groq


@pytest.mark.parametrize(
    ("raw_text", "branch", "valid"),
    [
        ("chat", "chat", True),
        ("RAG", "RAG", True),
        (" chat \n", "chat", True),
        ("CHAT", "chat", False),
        ("Chat", "chat", False),
        ("chatbot", "chat", False),
        ("Đây là câu hỏi chat xã giao", "chat", False),
        ("RAG vì đây là câu hỏi quy chế", "RAG", False),
        ("", "RAG", False),
    ],
)
def test_groq_branch_and_format_rules(raw_text: str, branch: str, valid: bool) -> None:
    assert executed_branch_from_text(raw_text) == branch
    assert format_is_valid(raw_text) is valid


# Arm Groq: chỉ query được gửi đi, usage lấy trước khi đổi text


def test_groq_sends_only_query_at_the_placeholder() -> None:
    fake = FakeChat(AIMessage(content="RAG"))
    adapter = GroqRouterAdapter(llm=fake, model="test-model")

    adapter.route(QUERY)

    (messages,) = fake.calls
    assert len(messages) == 1
    assert messages[0].content == GROQ_ROUTER_PROMPT.replace("{question}", QUERY)


def test_groq_captures_usage_and_latency() -> None:
    message = AIMessage(
        content="RAG",
        id="lc_run--abc-0",
        usage_metadata={"input_tokens": 123, "output_tokens": 2, "total_tokens": 125},
        response_metadata={"model_name": "qwen/qwen3.6-27b"},
    )
    adapter = GroqRouterAdapter(llm=FakeChat(message), model="requested-model")

    trial = adapter.route(QUERY)

    assert trial.usage == TokenUsage(input_tokens=123, output_tokens=2)
    # LLM tiêm vào không đi qua HTTP nên không có id/model của provider
    assert trial.request_id is None
    assert trial.model is None
    assert trial.executed_branch == "RAG"
    assert trial.format_valid is True
    assert trial.error_type is None
    assert trial.latency_ms >= 0


def test_groq_keeps_usage_when_content_is_block_list() -> None:
    message = AIMessage(
        content=[{"type": "reasoning", "reasoning": "nghĩ"}, {"type": "text", "text": "RAG"}],
        usage_metadata={"input_tokens": 7, "output_tokens": 1, "total_tokens": 8},
    )
    adapter = GroqRouterAdapter(llm=FakeChat(message), model="m")

    trial = adapter.route(QUERY)

    assert trial.raw_answer == "RAG"
    assert trial.usage == TokenUsage(input_tokens=7, output_tokens=1)


def test_groq_missing_usage_stays_null() -> None:
    adapter = GroqRouterAdapter(llm=FakeChat(AIMessage(content="chat")), model="m")

    trial = adapter.route(QUERY)

    assert trial.usage is None
    assert trial.request_id is None
    assert trial.model is None


def test_groq_error_becomes_error_trial_not_route() -> None:
    adapter = GroqRouterAdapter(llm=FakeChat(error=TimeoutError("hết giờ")), model="m")

    trial = adapter.route(QUERY)

    assert trial.executed_branch is None
    assert trial.format_valid is None
    assert trial.raw_answer is None
    assert trial.usage is None
    assert trial.error_type == "TimeoutError"
    assert trial.error_message == "hết giờ"
    assert trial.attempts == 1


# Arm Jev: guard request trước network call


@pytest.mark.parametrize(
    ("questions", "pattern"),
    [
        ({"route": Noul(instructions="đây là RAG?")}, "phải đúng type Choice"),
        ({"route": Score(instructions="mức độ", criteria=["a", "b"])}, "phải đúng type Choice"),
        ({"route": "RAG"}, "phải đúng type Choice"),
        (
            {"route": Choice(instructions="x", criteria={"RAG": "r", "khác": "k"})},
            "criteria phải đúng",
        ),
        (
            {"route": Choice(instructions="x", criteria={"RAG": "r"})},
            "criteria phải đúng",
        ),
        (
            {"route": build_route_questions()["route"], "extra": build_route_questions()["route"]},
            "chỉ được có đúng một câu hỏi",
        ),
        ({"khác": build_route_questions()["route"]}, "chỉ được có đúng một câu hỏi"),
    ],
)
def test_rejects_bad_route_questions(questions: dict, pattern: str) -> None:
    assert pattern in _validation_error(questions)


def test_rejects_choice_subclass() -> None:
    class SneakyChoice(Choice):
        pass

    questions = {"route": SneakyChoice(instructions="x", criteria=dict(ROUTE_CRITERIA))}

    assert "phải đúng type Choice" in _validation_error(questions)


def test_rejects_mutated_criteria_content_and_instruction() -> None:
    mutated_criteria = dict(ROUTE_CRITERIA)
    mutated_criteria["RAG"] = "câu nào cũng là RAG"
    mutated_instructions = {"route": Choice(instructions="đoán xem", criteria=dict(ROUTE_CRITERIA))}

    with pytest.raises(RouteContractError, match="đúng nội dung đã freeze"):
        validate_route_questions(
            {"route": Choice(instructions=JEV_ROUTE_INSTRUCTION, criteria=mutated_criteria)}
        )
    with pytest.raises(RouteContractError, match="instructions phải đúng"):
        validate_route_questions(mutated_instructions)


def test_valid_route_questions_pass() -> None:
    validate_route_questions(build_route_questions())


def test_route_time_guard_fires_before_network_call() -> None:
    fake = FakeClassifier(_response())
    adapter = JevRouterAdapter(classifier=fake)
    # Phá cấu trúc câu hỏi để chứng minh guard chạy trước khi gọi classifier
    adapter._questions = {"route": Noul(instructions="x")}

    with pytest.raises(RouteContractError, match="phải đúng type Choice"):
        adapter.route(QUERY)
    assert fake.calls == []


def test_jev_request_carries_only_query_and_one_choice() -> None:
    fake = FakeClassifier(_response())
    adapter = JevRouterAdapter(classifier=fake)

    adapter.route(QUERY)
    adapter.route("câu khác")

    first, second = fake.calls
    assert set(first) == {"state", "questions"}
    assert first["state"] == QUERY
    assert second["state"] == "câu khác"
    assert first["questions"] == second["questions"]
    question = first["questions"]["route"]
    assert type(question) is Choice
    assert set(question.criteria) == {"RAG", "chat"}
    assert question.criteria == ROUTE_CRITERIA
    assert question.instructions == JEV_ROUTE_INSTRUCTION
    payload = {name: q.model_dump() for name, q in first["questions"].items()}
    assert json.loads(json.dumps(payload))["route"]["criteria"] == ROUTE_CRITERIA
    assert "group" not in json.dumps(payload, ensure_ascii=False)
    assert "id" not in json.loads(json.dumps(payload))["route"]


# Arm Jev: guard response sau network call


@pytest.mark.parametrize(
    ("response", "pattern"),
    [
        (
            _response(answers={"route": NoulAnswer(type="noul", noul=0.9)}),
            "nouls/scores",
        ),
        (
            _response(
                answers={
                    "route": ScoreAnswer(
                        type="score",
                        score=1.0,
                        legend={0: "a", 1: "b"},
                        probabilities={0: 0.5, 1: 0.5},
                        confidence=0.5,
                    )
                }
            ),
            "nouls/scores",
        ),
        (
            _response(answers={"route": _choice_answer(), "extra": _choice_answer("chat")}),
            "chỉ được có answer",
        ),
        (_response(answers={"intent": _choice_answer()}), "chỉ được có answer"),
        (_response(answers={"route": _choice_answer("FAQ")}), "label phải thuộc"),
        ({"choices": {"route": {"choice": "RAG"}}}, "phải là ClassifierResponse"),
    ],
)
def test_rejects_bad_responses(response: object, pattern: str) -> None:
    with pytest.raises(RouteContractError) as excinfo:
        extract_route_choice(response)
    assert pattern in str(excinfo.value)


def test_rejects_choice_answer_subclass() -> None:
    class SneakyAnswer(ChoiceAnswer):
        pass

    answer = SneakyAnswer(
        type="choice",
        choice="RAG",
        probabilities={"RAG": 1.0, "chat": 0.0},
        confidence=1.0,
    )

    with pytest.raises(RouteContractError, match="phải đúng type ChoiceAnswer"):
        extract_route_choice(_response(answers={"route": answer}))


def test_jev_trial_preserves_probabilities_and_usage() -> None:
    fake = FakeClassifier(_response())
    adapter = JevRouterAdapter(classifier=fake)

    trial = adapter.route(QUERY)

    assert trial.executed_branch == "RAG"
    assert trial.raw_answer == "RAG"
    assert trial.format_valid is True
    assert trial.usage == TokenUsage(input_tokens=120, output_tokens=4)
    assert trial.request_id == "req-abc"
    assert trial.model == JEV_MODEL
    assert trial.detail == {"probabilities": {"RAG": 0.8, "chat": 0.2}, "confidence": 0.72}


def test_jev_missing_usage_stays_null() -> None:
    adapter = JevRouterAdapter(classifier=FakeClassifier(_response(usage=Usage())))

    trial = adapter.route(QUERY)

    assert trial.usage is None


def test_jev_error_becomes_error_trial_not_route() -> None:
    adapter = JevRouterAdapter(classifier=FakeClassifier(error=TimeoutError("hết giờ")))

    trial = adapter.route(QUERY)

    assert trial.executed_branch is None
    assert trial.error_type == "TimeoutError"
    assert trial.usage is None
    assert trial.detail is None


# End-to-end arm Groq qua ChatGroq thật + MockTransport: id và model phải lấy từ wire


def test_groq_wire_path_records_provider_id_and_served_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-groq-{len(seen)}",
                "model": f"served/substitute-{len(seen)}",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "RAG"},
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
            headers={"x-request-id": "header-id"},
        )

    adapter = GroqRouterAdapter(
        model="qwen/qwen3.6-27b",
        transport=httpx.MockTransport(handler),
    )

    first = adapter.route(QUERY)
    second = adapter.route("câu khác")

    assert first.request_id == "chatcmpl-groq-1"
    assert first.model == "served/substitute-1"
    assert second.request_id == "chatcmpl-groq-2"
    assert second.model == "served/substitute-2"
    assert first.usage == TokenUsage(input_tokens=11, output_tokens=2)
    assert first.executed_branch == "RAG"
    assert seen[0]["model"] == "qwen/qwen3.6-27b"
    # langchain_groq đổi temperature 0 thành 1e-08 khi gửi, giống hệt production
    assert seen[0]["temperature"] == 1e-08
    assert seen[0]["reasoning_effort"] == "none"
    assert QUERY in seen[0]["messages"][0]["content"]


def test_groq_wire_path_error_response_becomes_error_trial(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    adapter = GroqRouterAdapter(model="m", transport=httpx.MockTransport(handler))

    trial = adapter.route(QUERY)

    assert trial.executed_branch is None
    assert trial.usage is None
    assert trial.request_id is None
    assert trial.model is None
    assert trial.error_type is not None


# End-to-end qua client thật của TypeSafe với MockTransport


def _fake_typesafe_client(handler) -> object:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _typesafe_classifier(handler, *, model: str, api_key: str) -> TypeSafeClassifier:
    """Classifier thật của langchain_typesafe nhưng đi qua MockTransport, không network"""
    return TypeSafeClassifier(
        model=model,
        api_key=api_key,
        client=_fake_typesafe_client(handler),
    )


def _json_response(payload: dict, status: int = 200, request_id: str = "req-wire") -> httpx2.Response:
    return httpx2.Response(
        status,
        json=payload,
        headers={"x-typesafe-request-id": request_id},
    )


def test_jev_wire_response_is_recorded() -> None:
    seen: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return _json_response(
            {
                "model": JEV_MODEL,
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "chat",
                        "probabilities": {"RAG": 0.1, "chat": 0.9},
                        "confidence": 0.88,
                    }
                },
                "usage": {"input_tokens": 55, "output_tokens": 2},
            }
        )

    adapter = JevRouterAdapter(
        classifier=_typesafe_classifier(handler, model=JEV_MODEL, api_key="test-key")
    )

    trial = adapter.route(QUERY)

    assert seen[0]["state"] == QUERY
    assert set(seen[0]["questions"]) == {"route"}
    assert seen[0]["questions"]["route"]["type"] == "choice"
    assert seen[0]["questions"]["route"]["criteria"] == ROUTE_CRITERIA
    assert trial.executed_branch == "chat"
    assert trial.raw_answer == "chat"
    assert trial.usage == TokenUsage(input_tokens=55, output_tokens=2)
    assert trial.request_id == "req-wire"
    assert trial.model == JEV_MODEL


def test_jev_wire_rate_limit_becomes_error_trial() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _json_response({"error": "chậm lại"}, status=429)

    adapter = JevRouterAdapter(
        classifier=_typesafe_classifier(handler, model=JEV_MODEL, api_key="test-key")
    )

    trial = adapter.route(QUERY)

    assert trial.executed_branch is None
    assert trial.error_type == "TypeSafeRateLimitError"


def test_jev_wire_timeout_becomes_error_trial() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("quá hạn")

    adapter = JevRouterAdapter(
        classifier=_typesafe_classifier(handler, model=JEV_MODEL, api_key="test-key")
    )

    trial = adapter.route(QUERY)

    assert trial.executed_branch is None
    assert trial.error_type == "TypeSafeAPITimeoutError"
    assert trial.usage is None
