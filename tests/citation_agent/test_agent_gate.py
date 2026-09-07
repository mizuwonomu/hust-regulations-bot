"""Unit test decide_next (gate) với fake client: không gọi llama-server thật.

Fake client surface như ChatOpenAI production: extra_body seed sẵn
chat_template_kwargs, invoke ghi lại call và trả canned response.
"""

from types import SimpleNamespace

import pytest

from src.rag.agent.gate import decide_next
from src.rag.agent.prompt import build_decision_grammar
from src.rag.agent.schema import Decision


class FakeClient:
    """Client giả: extra_body seed như production, invoke ghi call rồi trả response"""

    def __init__(self, responses):
        self.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.calls = []
        self._responses = list(responses)

    def invoke(self, prompt, extra_body=None, **kwargs):
        self.calls.append({"prompt": prompt, "extra_body": dict(extra_body or {})})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def payload(json_text):
    return SimpleNamespace(content=json_text)


class TestValidPayloads:
    def test_valid_stop_payload(self):
        client = FakeClient([payload('{"stop": true, "dieu": null}')])
        decision = decide_next("q", "obs", [20], client=client)
        assert decision == Decision(stop=True, dieu=None)
        assert len(client.calls) == 1

    def test_valid_follow_target_in_candidates(self):
        client = FakeClient([payload('{"stop": false, "dieu": 20}')])
        decision = decide_next("q", "obs", [20, 26], client=client)
        assert decision == Decision(stop=False, dieu=20)


class TestInvalidPayloads:
    def test_follow_outside_candidates_rejected(self):
        client = FakeClient([payload('{"stop": false, "dieu": 99}')])
        with pytest.raises(ValueError):
            decide_next("q", "obs", [20], client=client)

    def test_malformed_json_rejected(self):
        client = FakeClient([payload("không phải json")])
        with pytest.raises(ValueError):
            decide_next("q", "obs", [20], client=client)

    def test_invalid_decision_state_rejected(self):
        # stop=True kèm dieu khác None là state không hợp lệ theo schema
        client = FakeClient([payload('{"stop": true, "dieu": 3}')])
        with pytest.raises(ValueError):
            decide_next("q", "obs", [20], client=client)


class TestPerCallBindings:
    def test_each_call_receives_its_own_grammar(self):
        client = FakeClient(
            [
                payload('{"stop": true, "dieu": null}'),
                payload('{"stop": false, "dieu": 30}'),
            ]
        )
        decide_next("q", "obs 1", [20], client=client)
        decide_next("q", "obs 2", [30, 40], client=client)

        assert client.calls[0]["extra_body"]["grammar"] == build_decision_grammar([20])
        assert client.calls[1]["extra_body"]["grammar"] == build_decision_grammar([30, 40])

    def test_rendered_prompt_reaches_client(self):
        client = FakeClient([payload('{"stop": true, "dieu": null}')])
        decide_next("câu hỏi riêng?", "obs riêng", [20], client=client)

        prompt = client.calls[0]["prompt"]
        final_message = prompt.to_messages()[-1]
        assert "câu hỏi riêng?" in final_message.content
        assert "obs riêng" in final_message.content

    def test_preexisting_extra_body_keys_preserved(self):
        client = FakeClient([payload('{"stop": true, "dieu": null}')])
        decide_next("q", "obs", [20], client=client)

        request_extra_body = client.calls[0]["extra_body"]
        assert request_extra_body["chat_template_kwargs"] == {"enable_thinking": False}
        assert "grammar" in request_extra_body

    def test_client_extra_body_not_mutated(self):
        client = FakeClient(
            [
                payload('{"stop": true, "dieu": null}'),
                payload('{"stop": false, "dieu": 20}'),
            ]
        )
        decide_next("q", "obs 1", [20], client=client)
        decide_next("q", "obs 2", [20], client=client)

        # Gate phải truyền bản copy per-request, không ghi grammar vào client.extra_body
        assert client.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


class TestFailureContract:
    def test_transport_error_propagates(self):
        client = FakeClient([RuntimeError("connection refused")])
        with pytest.raises(RuntimeError, match="connection refused"):
            decide_next("q", "obs", [20], client=client)
        assert len(client.calls) == 1

    def test_empty_candidates_rejected_before_invoke(self):
        client = FakeClient([payload('{"stop": true, "dieu": null}')])
        with pytest.raises(ValueError):
            decide_next("q", "obs", [], client=client)
        assert client.calls == [] # không invoke client
