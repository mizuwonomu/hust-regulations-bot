"""Kiểm tra số học summary C tất định trên trace result-native."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from diagnostic_subexp.doctoral_defense_prompt_ablation.artifacts import (
    execute_prompt_ablation,
    load_prompt_ablation,
    prepare_prompt_ablation,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.metrics import summarize_prompt_ablation


class FakeClient:
    """Trả một decision stop tất định cho mọi request C."""

    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    def invoke(self, prompt, **kwargs):
        return SimpleNamespace(
            content='{"stop": true, "dieu": null}',
            usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        )


def test_c_summary_has_separate_phase_and_variant_rows(tmp_path: Path):
    """Summary giữ đủ 132 slot và cả hai treatment variant."""
    path, manifest, _ = prepare_prompt_ablation(tmp_path / "run", ["first", "llm"])
    manifest, cases, inputs, prompts, _ = load_prompt_ablation(path)
    execute_prompt_ablation(path, manifest, cases, inputs, prompts, {"llm": FakeClient()})
    loaded = load_prompt_ablation(path)
    summary = summarize_prompt_ablation(*loaded)
    assert summary["artifact_schema_version"] == 3
    assert summary["coverage_by_policy"]["first"]["scheduled"] == 132
    assert summary["coverage_by_policy"]["llm"]["valid"] == 132
    assert summary["coverage_by_policy"]["llm"]["errors"] == 0
    variants = {
        variant for row in summary["per_case_arm_effects"] for variant in row["variants"]
    }
    assert variants == {"baseline-v1", "without-doctoral-defense-v1"}
