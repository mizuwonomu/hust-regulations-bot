"""Kiểm tra ranh giới contract giữa các package diagnostic.

Test này đóng vai trò hợp đồng tĩnh: module nào được sở hữu module nào, import
nào bị cấm, và fresh artifact schema v3 từ chối payload thiếu, sai nguồn hoặc
không phải fresh run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
SCRIPTS = AGENT_EXP_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO_ROOT))

from diagnostic_subexp.article_42_observation_block_order import cli as observation_cli
from diagnostic_subexp.article_42_observation_block_order.artifacts import (
    ObservationArtifactError,
    load_observation_run,
)
from diagnostic_subexp.candidate_pair_41_3_position import cli as candidate_cli
from diagnostic_subexp.candidate_pair_41_3_position.artifacts import (
    DiagnosticArtifactError,
    load_candidate_run,
)
from diagnostic_subexp.doctoral_defense_prompt_ablation.artifacts import (
    PromptAblationArtifactError,
    load_prompt_ablation,
    prepare_prompt_ablation,
)
from diagnostic_subexp.shared.contracts import ARTIFACT_ORIGIN, structured_hash

SUBPEXP_ROOT = SCRIPTS / "diagnostic_subexp"
A_ROOT = SUBPEXP_ROOT / "candidate_pair_41_3_position"
B_ROOT = SUBPEXP_ROOT / "article_42_observation_block_order"
C_ROOT = SUBPEXP_ROOT / "doctoral_defense_prompt_ablation"
SHARED_ROOT = SUBPEXP_ROOT / "shared"

EXPECTED_A_MODULES = {
    "__init__.py",
    "artifacts.py",
    "cli.py",
    "contracts.py",
    "metrics.py",
    "report.py",
    "schedule.py",
    "transforms.py",
}
EXPECTED_B_MODULES = EXPECTED_A_MODULES
EXPECTED_C_MODULES = {
    "__init__.py",
    "artifacts.py",
    "cli.py",
    "contracts.py",
    "metrics.py",
    "report.py",
    "schedule.py",
    "variants.py",
}
EXPECTED_SHARED_MODULES = {
    "__init__.py",
    "contracts.py",
    "execution.py",
    "provenance.py",
    "source_refs.py",
}


def _python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def _module_names(root: Path) -> set[str]:
    return {path.name for path in root.glob("*.py")}


def test_package_module_ownership_matches_the_plan():
    """Mỗi package chỉ chứa đúng module thuộc sở hữu của nó."""
    assert _module_names(A_ROOT) == EXPECTED_A_MODULES
    assert _module_names(B_ROOT) == EXPECTED_B_MODULES
    assert _module_names(C_ROOT) == EXPECTED_C_MODULES
    assert _module_names(SHARED_ROOT) == EXPECTED_SHARED_MODULES


def test_cross_package_imports_are_rejected_by_construction():
    """A/B/C chỉ được nói chuyện qua shared, không import chéo."""
    forbidden = {
        A_ROOT: ("article_42_observation_block_order", "doctoral_defense_prompt_ablation"),
        B_ROOT: ("candidate_pair_41_3_position", "doctoral_defense_prompt_ablation"),
        C_ROOT: ("candidate_pair_41_3_position", "article_42_observation_block_order"),
        SHARED_ROOT: (
            "candidate_pair_41_3_position",
            "article_42_observation_block_order",
            "doctoral_defense_prompt_ablation",
        ),
    }
    for root, names in forbidden.items():
        for path in _python_files(root):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for name in names:
                    assert name not in stripped, f"{path} imports {name}: {stripped}"


def test_diagnostic_code_has_no_flat_import_fallback_or_star_import():
    """Không còn import phẳng, import sao, fallback ModuleNotFoundError."""
    for path in _python_files(SUBPEXP_ROOT):
        text = path.read_text(encoding="utf-8")
        assert "import *" not in text, f"{path} uses a star import"
        assert "except ModuleNotFoundError" not in text, f"{path} has an import fallback"
        assert "diagnostic_contracts" not in text, f"{path} references a flat legacy module"
        assert "prompt_ablation_contracts" not in text, f"{path} references a flat legacy module"


def test_no_historical_migration_module_or_metadata_remains():
    """Không còn migration module, import cũ hay field migration trong code."""
    legacy = (
        "migrate_results_v3",
        "pre_migration_manifest_sha256",
        "deleted_result_local_source_paths",
        "deleted_generated_input_hashes",
        "previous_artifact_schema_version",
    )
    for path in _python_files(SUBPEXP_ROOT) + sorted(SCRIPTS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        for token in legacy:
            assert token not in text, f"{path} still references {token}"
    assert not (SCRIPTS / "migrate_results_v3.py").exists()
    assert not (AGENT_EXP_ROOT / "tests" / "test_migration_v3.py").exists()


def test_repository_root_depth_is_centralized():
    """Chỉ shared.source_refs được tính độ sâu repository root."""
    for path in _python_files(SUBPEXP_ROOT):
        if path == SHARED_ROOT / "source_refs.py":
            continue
        assert "parents[" not in path.read_text(encoding="utf-8"), f"{path} computes repo depth"
    for entrypoint in (
        SCRIPTS / "run_fixed_diagnostics_ab.py",
        SCRIPTS / "run_prompt_ablation.py",
    ):
        assert "parents[" not in entrypoint.read_text(encoding="utf-8")


def _run_a(tmp_path: Path) -> Path:
    return candidate_cli.prepare_run(None, tmp_path / "a", ["first"]).run_dir


def _run_b(tmp_path: Path) -> Path:
    return observation_cli.prepare_run(None, tmp_path / "b", ["first"]).run_dir


def _rewrite_manifest(run_dir: Path, payload: dict) -> None:
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize("kind", ["a", "b"])
def test_fresh_manifest_requires_artifact_origin(tmp_path: Path, kind: str):
    """Fresh loader từ chối manifest thiếu hoặc sai artifact_origin."""
    run_dir = _run_a(tmp_path) if kind == "a" else _run_b(tmp_path)
    loader = load_candidate_run if kind == "a" else load_observation_run
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["artifact_origin"] == ARTIFACT_ORIGIN

    for value in (None, "historical-bundle"):
        broken = json.loads(json.dumps(payload))
        if value is None:
            broken.pop("artifact_origin")
        else:
            broken["artifact_origin"] = value
        _rewrite_manifest(run_dir, broken)
        with pytest.raises((DiagnosticArtifactError, ObservationArtifactError)):
            loader(run_dir)


@pytest.mark.parametrize("kind", ["a", "b"])
def test_fresh_schema_rejects_missing_definition_or_sources(tmp_path: Path, kind: str):
    """Manifest thiếu definition hoặc base source không được reload."""
    run_dir = _run_a(tmp_path) if kind == "a" else _run_b(tmp_path)
    loader = load_candidate_run if kind == "a" else load_observation_run
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    broken.pop("experiment_definition")
    _rewrite_manifest(run_dir, broken)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError)):
        loader(run_dir)

    broken = json.loads(json.dumps(payload))
    broken.pop("sources")
    _rewrite_manifest(run_dir, broken)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError)):
        loader(run_dir)

    broken = json.loads(json.dumps(payload))
    broken["sources"].pop("inventory")
    _rewrite_manifest(run_dir, broken)
    with pytest.raises(
        (DiagnosticArtifactError, ObservationArtifactError), match="exactly case, snapshot and inventory"
    ):
        loader(run_dir)


@pytest.mark.parametrize("kind", ["a", "b"])
def test_fresh_schema_rejects_hash_or_path_tampering(tmp_path: Path, kind: str):
    """Hash sai hoặc path tuyệt đối phải chặn reload."""
    run_dir = _run_a(tmp_path) if kind == "a" else _run_b(tmp_path)
    loader = load_candidate_run if kind == "a" else load_observation_run
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    broken = json.loads(json.dumps(payload))
    broken["sources"]["case"]["sha256"] = "0" * 64
    _rewrite_manifest(run_dir, broken)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError), match="hash mismatch"):
        loader(run_dir)

    broken = json.loads(json.dumps(payload))
    broken["sources"]["case"]["path"] = "/tmp/gate_cases_fixed_v1.jsonl"
    _rewrite_manifest(run_dir, broken)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError)):
        loader(run_dir)


@pytest.mark.parametrize("kind", ["a", "b"])
def test_fresh_schema_rejects_source_definition_disagreement(tmp_path: Path, kind: str):
    """Source reference trỏ file canonical khác path trong definition phải bị chặn."""
    run_dir = _run_a(tmp_path) if kind == "a" else _run_b(tmp_path)
    loader = load_candidate_run if kind == "a" else load_observation_run
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot = payload["sources"]["snapshot"]
    payload["sources"]["case"] = {
        "role": "case",
        "path": snapshot["path"],
        "sha256": snapshot["sha256"],
        "required_for_replay": True,
    }
    _rewrite_manifest(run_dir, payload)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError), match="disagrees"):
        loader(run_dir)


def test_fresh_manifest_hashes_its_own_definition_and_registry(tmp_path: Path):
    """Manifest ghi structured hash của definition và registry của chính run."""
    from diagnostic_subexp.shared.provenance import sha256_file
    from diagnostic_subexp.shared.source_refs import scripts_root

    for kind in ("a", "b"):
        run_dir = _run_a(tmp_path / kind) if kind == "a" else _run_b(tmp_path / kind)
        payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        assert payload["definition_sha256"] == structured_hash(payload["experiment_definition"])
        assert payload["executed_module_hashes"]
        for relative, digest in payload["executed_module_hashes"].items():
            assert digest == sha256_file(scripts_root() / relative)

    run_dir, _, _ = prepare_prompt_ablation(tmp_path / "c", ["first"])
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["definition_sha256"] == structured_hash(payload["experiment_definition"])
    assert payload["registry_sha256"] == structured_hash(payload["registry_snapshot"])
    assert payload["executed_module_hashes"]
    for relative, digest in payload["executed_module_hashes"].items():
        assert digest == sha256_file(scripts_root() / relative)


@pytest.mark.parametrize("kind", ["a", "b"])
def test_fresh_manifest_rejects_a_tampered_definition_hash(tmp_path: Path, kind: str):
    """Sửa definition mà không cập nhật definition_sha256 phải bị từ chối."""
    run_dir = _run_a(tmp_path) if kind == "a" else _run_b(tmp_path)
    loader = load_candidate_run if kind == "a" else load_observation_run
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["experiment_definition"]["suite_id"] = "tampered-suite"
    _rewrite_manifest(run_dir, payload)
    with pytest.raises((DiagnosticArtifactError, ObservationArtifactError), match="definition hash"):
        loader(run_dir)


def test_c_manifest_rejects_a_tampered_registry_hash(tmp_path: Path):
    """Sửa registry snapshot mà không cập nhật registry_sha256 phải bị từ chối."""
    run_dir, _, _ = prepare_prompt_ablation(tmp_path / "c", ["first"])
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["registry_snapshot"]["groups"][0]["group_id"] = "tampered-group"
    (run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PromptAblationArtifactError, match="registry hash"):
        load_prompt_ablation(run_dir)


def test_c_manifest_requires_fresh_contract_fields(tmp_path: Path):
    """C fresh manifest phải có artifact_origin, sources và registry."""
    run_dir, _, _ = prepare_prompt_ablation(tmp_path / "c", ["first"])
    payload = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert payload["artifact_origin"] == ARTIFACT_ORIGIN
    assert set(payload["sources"]) == {"case", "snapshot", "inventory"}

    broken = json.loads(json.dumps(payload))
    broken["registry_snapshot"] = None
    (run_dir / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(PromptAblationArtifactError):
        load_prompt_ablation(run_dir)

    broken = json.loads(json.dumps(payload))
    broken["sources"]["snapshot"]["sha256"] = "0" * 64
    (run_dir / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(PromptAblationArtifactError, match="hash mismatch"):
        load_prompt_ablation(run_dir)


def test_no_result_local_sources_lookup_exists():
    """Loader không được chứa nhánh tìm nguồn trong `sources/` cục bộ."""
    for path in _python_files(SUBPEXP_ROOT):
        text = path.read_text(encoding="utf-8")
        assert 'run_dir / "sources"' not in text, f"{path} searches a result-local sources dir"
        assert '"/sources"' not in text, f"{path} searches a result-local sources dir"
