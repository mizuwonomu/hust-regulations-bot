"""Kiểm tra quy tắc resolve source canonical của diagnostic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AGENT_EXP_ROOT.parents[1]
sys.path.insert(0, str(AGENT_EXP_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from diagnostic_subexp.shared.contracts import SourceReference
from diagnostic_subexp.shared.source_refs import (
    canonical_repository_path,
    resolve_repository_source,
    scripts_root,
    sha256_file,
    source_reference_payload,
    validate_source_reference,
)
from diagnostic_subexp.shared.provenance import (
    executed_module_hashes,
    execution_provenance,
)

CASE_PATH = "evals/agent-exp/datasets/gate_cases_fixed_v1.jsonl"
# Hash được tính tại chỗ, không hardcode hash nguồn lịch sử
CASE_HASH = sha256_file(REPO_ROOT / CASE_PATH)


def test_canonical_path_rejects_absolute_paths():
    """Manifest không được chứa đường dẫn tuyệt đối."""
    with pytest.raises(ValueError, match="repository-relative"):
        canonical_repository_path("/etc/passwd")
    with pytest.raises(ValueError, match="repository-relative"):
        SourceReference(role="case", path="/etc/passwd", sha256=CASE_HASH)


def test_canonical_path_rejects_escaping_paths():
    """Manifest không được chứa path thoát khỏi repository."""
    with pytest.raises(ValueError, match="escapes repository"):
        canonical_repository_path("../outside.json")
    with pytest.raises(ValueError):
        SourceReference(role="case", path="evals/../../outside.json", sha256=CASE_HASH)


def test_resolve_requires_the_exact_repository_relative_file():
    """Nguồn replay thiếu tại path canonical phải là lỗi, không fallback."""
    with pytest.raises(FileNotFoundError, match="required replay source is missing"):
        resolve_repository_source("evals/agent-exp/datasets/does-not-exist.jsonl")


def test_validate_rejects_a_hash_mismatch():
    """Hash sai phải chặn replay thay vì đọc file cùng tên nơi khác."""
    reference = SourceReference(role="case", path=CASE_PATH, sha256="0" * 64)
    with pytest.raises(ValueError, match="source hash mismatch"):
        validate_source_reference(reference)


def test_executed_module_hashes_resolve_package_relative_paths():
    """Module hash resolve từ scripts root, module thiếu là lỗi không bỏ qua."""
    relative = "diagnostic_subexp/shared/contracts.py"
    hashes = executed_module_hashes([relative])
    assert hashes == {relative: sha256_file(scripts_root() / relative)}
    with pytest.raises(FileNotFoundError, match="executed module is missing"):
        executed_module_hashes(["diagnostic_subexp/shared/does_not_exist.py"])
    with pytest.raises(ValueError, match="at least one module path"):
        executed_module_hashes([])


def test_execution_provenance_requires_revision_dirty_state_and_hashes():
    """Execution provenance luôn có revision, dirty state và hash module thật."""
    provenance = execution_provenance(["diagnostic_subexp/shared/contracts.py"])
    assert provenance["execution_revision"]
    assert isinstance(provenance["execution_dirty"], bool)
    assert provenance["executed_module_hashes"]
    for digest in provenance["executed_module_hashes"].values():
        assert len(digest) == 64


def test_source_reference_payload_only_accepts_shared_roles():
    """Payload tham chiếu nguồn nền chỉ nhận ba role dùng chung."""
    payload = source_reference_payload("case", CASE_PATH, CASE_HASH)
    assert payload == {
        "role": "case",
        "path": CASE_PATH,
        "sha256": CASE_HASH,
        "required_for_replay": True,
    }
    with pytest.raises(ValueError, match="unsupported shared source role"):
        source_reference_payload("spec", CASE_PATH, CASE_HASH)


def test_resolver_ignores_same_named_files_outside_repository(tmp_path: Path, monkeypatch):
    """Resolver chỉ đọc từ repository root, không tìm theo basename hay cwd."""
    decoy = tmp_path / "evals" / "agent-exp" / "datasets"
    decoy.mkdir(parents=True)
    (decoy / "gate_cases_fixed_v1.jsonl").write_text("decoy", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    resolved = resolve_repository_source(CASE_PATH)
    assert resolved == REPO_ROOT / CASE_PATH
    assert resolved.read_text(encoding="utf-8") != "decoy"


def test_source_reference_requires_replay_for_shared_roles():
    """Ba nguồn nền dùng chung không được phép tắt kiểm tra replay."""
    with pytest.raises(ValueError, match="required for replay"):
        SourceReference(
            role="case",
            path=CASE_PATH,
            sha256=CASE_HASH,
            required_for_replay=False,
        )
