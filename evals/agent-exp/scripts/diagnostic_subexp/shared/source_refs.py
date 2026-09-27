"""Tham chiếu nguồn canonical tương đối repository cho replay kết quả."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

def repository_root() -> Path:
    """Resolve repository root cho việc phân giải nguồn canonical."""
    return Path(__file__).resolve().parents[5]


def scripts_root() -> Path:
    """Resolve thư mục `evals/agent-exp/scripts` chứa các module diagnostic."""
    return repository_root() / "evals/agent-exp/scripts"


_SHARED_ROLE_BY_LABEL = {
    "case": "case",
    "snapshot": "snapshot",
    "inventory": "inventory",
}


def canonical_repository_path(path: str | Path) -> str:
    """Kiểm tra và trả một đường dẫn nguồn tương đối repository."""
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError(f"source path must be repository-relative: {path}")
    root = repository_root().resolve()
    resolved = (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes repository: {path}") from exc
    return relative.as_posix()


def resolve_repository_source(path: str | Path, *, required_for_replay: bool = True) -> Path:
    """Resolve đúng một nguồn canonical từ repository root."""
    relative = canonical_repository_path(path)
    resolved = repository_root() / relative
    if required_for_replay and not resolved.is_file():
        raise FileNotFoundError(f"required replay source is missing: {relative}")
    return resolved


def sha256_file(path: Path) -> str:
    """Băm một file nguồn mà không tìm ở vị trí khác."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_reference_payload(label: str, path: str | Path, expected_hash: str) -> dict[str, Any]:
    """Dựng payload tham chiếu nguồn nền canonical nghiêm ngặt."""
    role = _SHARED_ROLE_BY_LABEL.get(label, label)
    if role not in {"case", "snapshot", "inventory"}:
        raise ValueError(f"unsupported shared source role: {role}")
    return {
        "role": role,
        "path": canonical_repository_path(path),
        "sha256": expected_hash,
        "required_for_replay": True,
    }


def validate_source_reference(reference: Any) -> Path:
    """Resolve và kiểm tra hash một model hoặc mapping source reference."""
    path = canonical_repository_path(reference.path if hasattr(reference, "path") else reference["path"])
    expected = reference.sha256 if hasattr(reference, "sha256") else reference["sha256"]
    resolved = resolve_repository_source(path)
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"source hash mismatch for {path}: expected {expected}, got {actual}")
    return resolved
