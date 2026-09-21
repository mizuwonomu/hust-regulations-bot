"""Kiểm tra prompt production tĩnh và treatment C do eval sở hữu."""

from __future__ import annotations

from types import SimpleNamespace

from diagnostic_subexp.doctoral_defense_prompt_ablation.contracts import PromptBaselineReference
from diagnostic_subexp.doctoral_defense_prompt_ablation.schedule import (
    build_prompt_definition,
    build_prompt_input_traces,
    build_prompt_traces,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.variants import (
    baseline_references,
    prompt_registry,
)
from diagnostic_subexp.shared.contracts import DiagnosticExecutionConfig, MessageRecord
from src.rag.agent.gate import decide_next
from src.rag.agent.prompt import build_decision_grammar, render_gate_prompt


REFERENCES = [
    PromptBaselineReference.model_validate(row) for row in baseline_references()
]


def test_definition_is_built_in_memory_without_historical_hashes():
    """Definition C không chứa path/hash của dataset hay prompt trung gian."""
    definition = build_prompt_definition()
    assert definition["registry_snapshot"] == prompt_registry()
    assert definition["baseline_references"] == baseline_references()
    serialized = str(definition)
    for forbidden in (
        "fixed_diagnostic_c_v1.json",
        "baseline_prompt_reference.jsonl",
        "prompt_pre_refactor.py",
        "pre_refactor_source_hash",
        "baseline_reference_hash",
        "a_suite_hash",
        "source_revision",
    ):
        assert forbidden not in serialized
    assert all("prompt_source_hashes" not in row for row in baseline_references())


def test_production_baseline_is_byte_equivalent_to_all_ten_references():
    """Renderer production duy nhất vẫn khớp composition baseline đóng băng."""
    for reference in REFERENCES:
        rendered = [
            MessageRecord(role=message.type, content=message.content)
            for message in render_gate_prompt(
                reference.question, reference.observation, reference.presented_candidates
            ).to_messages()
        ]
        assert rendered == reference.effective_messages
        assert build_decision_grammar(reference.grammar_candidates) == reference.grammar_text
        assert len(rendered) == 14


def test_evaluation_registry_builds_exact_baseline_and_ablation_messages():
    """C sở hữu sáu cặp ví dụ và chỉ loại bỏ group doctoral-defense."""
    config = DiagnosticExecutionConfig(
        model_alias="citation-agent",
        base_url="http://127.0.0.1:8080/v1",
        temperature=0.0,
        max_completion_tokens=4096,
        enable_thinking=False,
        timeout_seconds=None,
        max_retries=None,
        extra_request_options={"chat_template_kwargs": {"enable_thinking": False}},
    )
    inputs = build_prompt_input_traces(REFERENCES, config)
    traces = build_prompt_traces(inputs, REFERENCES, prompt_registry(), config, render=False)
    for reference in REFERENCES:
        rows = [
            trace
            for trace in traces
            if trace.input_trace_id
            == next(
                item for item in inputs if item.reference_id == reference.reference_id
            ).input_trace_id
        ]
        assert {len(trace.effective_messages) for trace in rows} == {8, 14}


def test_gate_exposes_only_request_scoped_baseline_prompt():
    """Gate call không có tham số variant và không sửa client settings."""
    calls = []
    options = {"chat_template_kwargs": {"enable_thinking": False}, "custom": 7}

    class Client:
        extra_body = options

        def invoke(self, prompt, **kwargs):
            calls.append((prompt.to_messages(), kwargs))
            return SimpleNamespace(content='{"stop": false, "dieu": 3}')

    client = Client()
    decide_next("literal {question}", "literal {observation}", [3, 41], client=client)
    assert len(calls[0][0]) == 14
    assert calls[0][1]["extra_body"]["grammar"] == build_decision_grammar([3, 41])
    assert client.extra_body == options
    assert "prompt_variant_id" not in decide_next.__annotations__


def test_production_source_contains_no_experiment_contract():
    """Prompt production không chứa registry evaluation hay variant identity."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[3] / "src/rag/agent/prompt.py").read_text()
    for forbidden in (
        "prompt_variant_id",
        "variant_id",
        "baseline-v1",
        "PromptExamplePair",
        "GatePromptVariant",
        "prompt_registry",
    ):
        assert forbidden not in source
