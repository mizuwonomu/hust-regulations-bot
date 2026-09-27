"""Offline policy isolation checks for first and LLM adapters."""

from __future__ import annotations

import httpx
import openai
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from policies import first, llm
import run_experiments

from src.rag.agent.schema import Decision


class _GateClient:
    """Provide the invoke and extra-body surface used by the real gate."""

    def __init__(self, content=None, error=None):
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.content = content
        self.error = error
        self.calls = 0

    def invoke(self, prompt, extra_body=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


def test_first_selects_first_without_client_factory(monkeypatch):
    monkeypatch.setattr(llm, "create_llm_client", lambda: (_ for _ in ()).throw(AssertionError()))
    decision = first.decide("q", "observation", [30, 20])
    assert decision == Decision(stop=False, dieu=30)


def test_first_rejects_empty_candidates():
    with pytest.raises(ValueError, match="non-empty"):
        first.decide("q", "observation", [])


def test_llm_adapter_uses_existing_gate_boundary(monkeypatch):
    calls = []

    def fake_decide_next(question, observation, candidates, *, client):
        calls.append((question, observation, list(candidates), client))
        return Decision(stop=False, dieu=20)

    monkeypatch.setattr(llm, "decide_next", fake_decide_next)
    client = object()
    decision = llm.decide("q", "observation", [20, 30], client=client)
    assert decision == Decision(stop=False, dieu=20)
    assert calls == [("q", "observation", [20, 30], client)]


def test_llm_stop_is_preserved(monkeypatch):
    monkeypatch.setattr(
        llm,
        "decide_next",
        lambda *args, **kwargs: Decision(stop=True, dieu=None),
    )
    assert llm.decide("q", "observation", [20], client=object()).stop is True


def test_llm_factory_is_explicit_and_lazy(monkeypatch):
    created = object()
    calls = []
    monkeypatch.setattr(llm, "create_llm_client", lambda: calls.append(True) or created)
    assert calls == []
    assert llm.create_client() is created
    assert calls == [True]


def test_llm_adapter_propagates_gate_failure(monkeypatch):
    monkeypatch.setattr(
        llm,
        "decide_next",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("transport down")),
    )
    with pytest.raises(RuntimeError, match="transport down"):
        llm.decide("q", "observation", [20], client=SimpleNamespace())


def test_connect_error_is_classified_as_transport():
    # Phân loại theo kiểu lỗi transport, không phụ thuộc câu chữ của thông báo
    for error in (
        httpx.ConnectError("no route"),
    ):
        assert run_experiments._classify_exception(error) == "transport"


def test_timeout_precedes_transport_classification():
    for error in (
        openai.APITimeoutError(
            request=httpx.Request("POST", "http://127.0.0.1:8080/v1")
        ),
    ):
        assert run_experiments._classify_exception(error) == "timeout"


def test_llm_real_gate_accepts_follow_and_stop_with_fake_client():
    # Chỉ giả client gọi ra ngoài, giữ parser và kiểm tra candidate của gate thật
    follow_client = _GateClient('{"stop": false, "dieu": 20}')
    stop_client = _GateClient('{"stop": true, "dieu": null}')
    assert llm.decide("q", "observation", [20, 30], client=follow_client) == Decision(stop=False, dieu=20)
    assert llm.decide("q", "observation", [20, 30], client=stop_client) == Decision(stop=True, dieu=None)
    assert follow_client.calls == 1
    assert stop_client.calls == 1


def test_llm_real_gate_rejects_malformed_json_and_invalid_candidate():
    with pytest.raises(ValidationError):
        llm.decide("q", "observation", [20], client=_GateClient("not json"))
    with pytest.raises(ValueError, match="ngoài candidates"):
        llm.decide("q", "observation", [20], client=_GateClient('{"stop": false, "dieu": 99}'))
    with pytest.raises(ValueError, match="non-empty list"):
        llm.decide("q", "observation", [], client=_GateClient('{"stop": true, "dieu": null}'))


def test_llm_real_gate_propagates_transport_and_timeout_errors():
    transport = httpx.ConnectError("offline")
    timeout = openai.APITimeoutError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1"))
    with pytest.raises(httpx.ConnectError):
        llm.decide("q", "observation", [20], client=_GateClient(error=transport))
    with pytest.raises(openai.APITimeoutError):
        llm.decide("q", "observation", [20], client=_GateClient(error=timeout))


def test_runner_error_outcome_is_compact_and_not_stop():
    # Lỗi gọi model tạo error với decision null, không được chuyển thành STOP
    outcome = run_experiments._outcome_from_exception(
        RuntimeError("line one\n" + "x" * 900),
        latency_ms=4.0,
    )
    assert outcome.status == "error"
    assert outcome.decision is None
    assert outcome.error is not None
    assert outcome.error.category == "unexpected"
    assert "\n" not in outcome.error.message
    assert len(outcome.error.message) <= 500


def test_runner_classifies_validation_error_as_schema():
    with pytest.raises(ValidationError):
        Decision.model_validate_json("not json")
    try:
        Decision.model_validate_json("not json")
    except ValidationError as error:
        assert run_experiments._classify_exception(error) == "schema"
