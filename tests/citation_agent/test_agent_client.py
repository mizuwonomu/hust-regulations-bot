"""Unit test client factory create_llm_client: chỉ mock constructor, không network.

Không health check, không inference: mock nhận plain kwargs là bằng chứng duy nhất.
"""

from types import SimpleNamespace

import pytest

from src.rag.agent import llm_client


@pytest.fixture
def fake_chat_openai(monkeypatch):
    """Thay ChatOpenAI bằng factory ghi lại kwargs, trả về object trơ"""
    created = []

    def factory(**kwargs):
        client = SimpleNamespace(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(llm_client, "ChatOpenAI", factory)
    return created


class TestBaseUrlValidation:
    @pytest.mark.parametrize(
        "url",
        ["http://127.0.0.1:8080/v1", "https://localhost:1234/v1", "  http://127.0.0.1:8080/v1  "],
    )
    def test_valid_prefix_accepted(self, url, fake_chat_openai):
        llm_client.create_llm_client(base_url=url, temperature=0.0, max_completion_tokens=10)
        assert len(fake_chat_openai) == 1

    @pytest.mark.parametrize(
        "bad_url",
        ["", "   ", None, 123, "ftp://127.0.0.1:8080/v1", "127.0.0.1:8080/v1", "localhost/v1"],
    )
    def test_invalid_values_rejected(self, bad_url, fake_chat_openai):
        with pytest.raises(ValueError):
            llm_client.create_llm_client(
                base_url=bad_url, temperature=0.0, max_completion_tokens=10
            )
        assert fake_chat_openai == []


class TestConstruction:
    def test_constructor_receives_contract_kwargs(self, fake_chat_openai):
        llm_client.create_llm_client(
            base_url="http://127.0.0.1:8080/v1",
            temperature=0.0,
            max_completion_tokens=4096,
        )
        assert len(fake_chat_openai) == 1
        kwargs = vars(fake_chat_openai[0])
        # Alias nội bộ + placeholder key local (không phải secret từ .env)
        assert kwargs["model"] == "citation-agent"
        assert kwargs["api_key"] == "dummy"
        # Transport settings: temperature và cap token completion
        assert kwargs["temperature"] == 0.0
        assert kwargs["max_completion_tokens"] == 4096
        # extra_body phải chứa chat_template_kwargs
        assert "chat_template_kwargs" in kwargs["extra_body"]

    def test_construction_does_no_inference_or_health_check(self, fake_chat_openai):
        # Mock chỉ nhận plain kwargs; mọi hành vi network đều không thể xảy ra
        client = llm_client.create_llm_client(
            base_url="http://127.0.0.1:8080/v1",
            temperature=0.0,
            max_completion_tokens=10,
        )
        assert client is fake_chat_openai[0]
        assert all(
            isinstance(value, (str, int, float, dict)) for value in vars(client).values()
        )
