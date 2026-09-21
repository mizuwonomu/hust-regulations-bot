"""Kiểm tra boundary chỉ-đọc cho các result root lịch sử và fresh recreation.

Historical result root chỉ được kiểm tra bằng `Path.exists()`. Test dùng thư
mục tạm, không bao giờ mở manifest hay summary của bundle lịch sử thật.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
SCRIPTS = AGENT_EXP_ROOT / "scripts"
A_CLI = SCRIPTS / "diagnostic_subexp" / "candidate_pair_41_3_position" / "cli.py"
B_CLI = SCRIPTS / "diagnostic_subexp" / "article_42_observation_block_order" / "cli.py"
C_CLI = SCRIPTS / "diagnostic_subexp" / "doctoral_defense_prompt_ablation" / "cli.py"
A_ENTRY = SCRIPTS / "run_fixed_diagnostics_ab.py"
C_ENTRY = SCRIPTS / "run_prompt_ablation.py"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.candidate_pair_41_3_position import cli as candidate_cli
from diagnostic_subexp.shared.contracts import (
    ARTIFACT_ORIGIN,
    HISTORICAL_RESULT_ROOTS,
    PrepareOutcome,
)


def _seed_existing_result(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "manifest.json").write_text("not json", encoding="utf-8")
    sentinel = path / "sentinel.jsonl"
    sentinel.write_text("not json", encoding="utf-8")
    sentinel.chmod(0o000)


def _restore(path: Path) -> None:
    sentinel = path / "sentinel.jsonl"
    if sentinel.exists():
        sentinel.chmod(0o644)


def _guard_children(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = path.resolve()
    real_open = Path.open
    real_read_text = Path.read_text

    def inside(candidate: Path) -> bool:
        resolved = candidate.resolve()
        return resolved == root or root in resolved.parents

    def guarded_open(self: Path, *args, **kwargs):
        if inside(self):
            raise AssertionError(f"existing result child was read: {self}")
        return real_open(self, *args, **kwargs)

    def guarded_read_text(self: Path, *args, **kwargs):
        if inside(self):
            raise AssertionError(f"existing result child was read: {self}")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)


def test_existing_output_path_is_reported_without_child_reads(tmp_path: Path, monkeypatch):
    """Path đã tồn tại trả existing-result thành công và không đọc file con."""
    existing = tmp_path / "existing"
    _seed_existing_result(existing)
    try:
        _guard_children(existing, monkeypatch)
        outcome = candidate_cli.prepare_run(None, existing, ["first"])
        assert outcome == PrepareOutcome(status="existing-result", run_dir=existing)
        monkeypatch.undo()
        assert (existing / "manifest.json").read_text(encoding="utf-8") == "not json"
    finally:
        monkeypatch.undo()
        _restore(existing)


def test_absent_path_creates_a_fresh_schema_v3_run(tmp_path: Path):
    """Path trống được tạo mới với artifact_origin và hash của chính run."""
    output_dir = tmp_path / "fresh"
    outcome = candidate_cli.prepare_run(None, output_dir, ["first"])
    assert outcome.status == "created"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_origin"] == ARTIFACT_ORIGIN
    assert set(manifest["sources"]) == {"case", "snapshot", "inventory"}
    assert "migration" not in manifest
    assert "source_revision" not in manifest
    assert not (output_dir / "sources").exists()
    assert not (output_dir / "results_first.jsonl").exists()
    for reference in manifest["sources"].values():
        assert reference["required_for_replay"] is True
        assert len(reference["sha256"]) == 64


def test_absent_path_without_git_metadata_still_creates_fresh_run(tmp_path: Path):
    """Filesystem absence một mình đủ để prepare, không cần Git phân biệt."""
    output_dir = tmp_path / "absent-looks-like-deletion"
    outcome = observation_cli.prepare_run(None, output_dir, ["first"])
    assert outcome.status == "created"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_origin"] == ARTIFACT_ORIGIN
    assert manifest["suite_id"] == "fixed-diagnostic-b-v1"


def test_second_prepare_reports_existing_result_and_does_not_overwrite(tmp_path: Path):
    """Prepare lần hai giữ nguyên artifact của run đã tạo."""
    output_dir = tmp_path / "fresh"
    candidate_cli.prepare_run(None, output_dir, ["first"])
    before = {
        path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()
    }
    outcome = candidate_cli.prepare_run(None, output_dir, ["first"])
    assert outcome.status == "existing-result"
    after = {path.name: path.read_bytes() for path in output_dir.iterdir() if path.is_file()}
    assert after == before


def test_fresh_run_embeds_definition_hashes_from_its_own_prepare(tmp_path: Path):
    """Definition nhúng không chứa hash nguồn lịch sử; manifest tự ghi hash."""
    output_dir = tmp_path / "fresh"
    candidate_cli.prepare_run(None, output_dir, ["first"])
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    definition = json.dumps(manifest["experiment_definition"], ensure_ascii=False)
    for forbidden in ("case_hash", "snapshot_hash", "inventory_hash", "prompt_source_hash"):
        assert forbidden not in definition
    source_path = REPO_ROOT / manifest["sources"]["case"]["path"]
    assert manifest["sources"]["case"]["sha256"] == hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()


def test_existing_malformed_result_survives_prepare_via_subprocess(tmp_path: Path):
    """CLI subprocess báo existing-result với bundle con hỏng và unreadable."""
    for cli_path in (A_CLI, B_CLI, C_CLI):
        existing = tmp_path / cli_path.parent.name
        _seed_existing_result(existing)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(cli_path),
                    "prepare",
                    "--output-dir",
                    str(existing),
                    "--policies",
                    "first",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            assert completed.returncode == 0, (cli_path, completed.stderr)
            assert "Existing result" in completed.stdout
            assert (existing / "manifest.json").read_text(encoding="utf-8") == "not json"
        finally:
            _restore(existing)


def test_dual_suite_prepare_reports_existing_root_without_reading_children(tmp_path: Path):
    """Output root đã tồn tại thì dual-suite prepare không mở child nào."""
    output_root = tmp_path / "ab"
    _seed_existing_result(output_root)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(A_ENTRY),
                "prepare",
                "--output-root",
                str(output_root),
                "--policies",
                "first",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.count("existing-result") == 2
        assert (output_root / "manifest.json").read_text(encoding="utf-8") == "not json"
    finally:
        _restore(output_root)


def test_prompt_entrypoint_reports_existing_result(tmp_path: Path):
    """Entrypoint C cũng chỉ kiểm tra Path.exists trên output path."""
    existing = tmp_path / "c-run"
    _seed_existing_result(existing)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(C_ENTRY),
                "prepare",
                "--output-dir",
                str(existing),
                "--policies",
                "first",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert completed.returncode == 0, completed.stderr
        assert "Existing result" in completed.stdout
        assert (existing / "manifest.json").read_text(encoding="utf-8") == "not json"
    finally:
        _restore(existing)


def test_historical_roots_are_declared_as_path_only_boundaries():
    """Danh sách historical root là constant repo-relative, không có hash gate."""
    assert HISTORICAL_RESULT_ROOTS == (
        "evals/agent-exp/results/fixed-diagnostic-ab-live-v1/a-candidate-position",
        "evals/agent-exp/results/fixed-diagnostic-ab-live-v1/b-observation-order",
        "evals/agent-exp/results/prompt-ablation/c-llm-v1",
    )
    for relative in HISTORICAL_RESULT_ROOTS:
        assert not Path(relative).is_absolute()
        assert ".." not in Path(relative).parts


def test_implementation_never_reads_historical_children():
    """Code implementation không nhắc tới đường dẫn con của historical bundle."""
    scan_roots = [
        SCRIPTS / "run_fixed_diagnostics_ab.py",
        SCRIPTS / "run_prompt_ablation.py",
    ]
    scan_roots.extend(
        path
        for path in (SCRIPTS / "diagnostic_subexp").rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "contracts.py"
    )
    for path in scan_roots:
        text = path.read_text(encoding="utf-8")
        assert "fixed-diagnostic-ab-live-v1" not in text, path
        assert "prompt-ablation/c-llm-v1" not in text, path
