"""Helper hash và provenance thực thi dùng chung cho các harness diagnostic.

Module path trong `executed_module_hashes` là package-relative so với
`evals/agent-exp/scripts`; mọi module thiếu đều phải làm prepare thất bại thay
vì bị bỏ qua âm thầm.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

from diagnostic_subexp.shared.source_refs import repository_root, scripts_root


def sha256_file(path: Path) -> str:
    """Trả digest SHA-256 của bytes trong một file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(arguments: list[str]) -> str | None:
    """Đọc một giá trị git mà không đổi trạng thái thực thi đã ghi."""
    try:
        completed = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            cwd=repository_root(),
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_dirty() -> bool | None:
    """Trả việc repository có thay đổi chưa commit hay không."""
    value = git_value(["status", "--porcelain"])
    return None if value is None else bool(value)


def resolve_module_path(module_path: str | Path) -> Path:
    """Resolve một module package-relative so với scripts root, thiếu là lỗi."""
    candidate = Path(module_path)
    resolved = candidate if candidate.is_absolute() else scripts_root() / candidate
    if not resolved.is_file():
        raise FileNotFoundError(f"executed module is missing: {module_path}")
    return resolved


def executed_module_hashes(module_paths: list[str | Path]) -> dict[str, str]:
    """Băm đúng các module package-relative đã thực thi, không bỏ qua module thiếu."""
    if not module_paths:
        raise ValueError("executed module provenance requires at least one module path")
    return {
        str(Path(module_path).as_posix()): sha256_file(resolve_module_path(module_path))
        for module_path in module_paths
    }


def execution_provenance(module_paths: list[str | Path]) -> dict[str, Any]:
    """Ghi revision, dirty state và hash của đúng các module đã thực thi."""
    revision = git_value(["rev-parse", "HEAD"])
    if revision is None:
        raise RuntimeError("fresh-run provenance requires a git revision")
    dirty = git_dirty()
    if dirty is None:
        raise RuntimeError("fresh-run provenance requires a git dirty state")
    return {
        "execution_revision": revision,
        "execution_dirty": dirty,
        "executed_module_hashes": executed_module_hashes(module_paths),
    }
