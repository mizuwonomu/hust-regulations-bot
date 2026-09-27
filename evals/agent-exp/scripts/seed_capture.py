"""Capture seed hop-0 trực tiếp từ single-pass retrieval thành snapshot đóng băng.

Chỉ `id` và `user_input` được đọc như semantic input. `link`, `group`, `type`,
`response` và `retrieved_contexts` của corpus không tham gia retrieval, nhãn,
định danh hay membership. Hashing bytes của corpus gốc chỉ dùng cho provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

try:
    from artifacts import sha256_bytes, sha256_file
    from contracts import (
        SEED_SCHEMA_VERSION_V2,
        SeedRow,
        SeedSnapshot,
        SourceFile,
    )
    from seed_cases import (
        canonical_json,
        validate_dataset,
        validate_whitelist,
    )
except ModuleNotFoundError:
    from .artifacts import sha256_bytes, sha256_file
    from .contracts import (
        SEED_SCHEMA_VERSION_V2,
        SeedRow,
        SeedSnapshot,
        SourceFile,
    )
    from .seed_cases import (
        canonical_json,
        validate_dataset,
        validate_whitelist,
    )

from evals.common.single_pass_retrieval import SinglePassSettings


REPO_ROOT = Path(__file__).resolve().parents[3]
#File nguồn quyết định hành vi retrieval, dùng cho provenance và content identity
CAPTURE_SOURCE_PATHS = (
    "evals/common/single_pass_retrieval.py",
    "evals/agent-exp/scripts/seed_capture.py",
    "evals/agent-exp/scripts/contracts.py",
    "evals/agent-exp/scripts/seed_cases.py",
    "src/rag/config.py",
    "src/rag/embedding_utils.py",
    "src/rag/reranker_utils.py",
)


class CaptureRuntime(Protocol):
    """Seam runtime tối thiểu mà capture cần: metadata và một lượt retrieve."""

    metadata: dict[str, Any]

    def retrieve(self, question: str) -> tuple[list[str], set[int]]:
        ...


def _parse_json(path: Path, raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: file is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}") from exc


def _git_value(arguments: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())


def capture_source_hashes() -> dict[str, str]:
    """Băm source retrieval và config/model-loader đang chạy capture."""
    hashes: dict[str, str] = {}
    for relative in CAPTURE_SOURCE_PATHS:
        path = REPO_ROOT / relative
        if path.exists():
            hashes[relative] = sha256_file(path)
    return hashes


def _unavailable_metadata(revision: str | None, dirty: bool | None) -> dict[str, str]:
    unavailable = {
        "store_content_fingerprint": "vector and parent store contents are not fingerprinted",
        "embedding_model_weights": "configured identifier recorded, served weights not verified",
        "reranker_model_weights": "configured identifier recorded, served weights not verified",
        "rewrite_model_weights": "configured model id recorded, served weights not verified",
    }
    if revision is None:
        unavailable["source_revision"] = "git revision unavailable"
    if dirty is None:
        unavailable["working_tree_modified"] = "git status unavailable"
    return unavailable


def _snapshot_id(
    *,
    dataset_id: str,
    dataset_hash: str,
    whitelist_hash: str,
    retrieval_config: dict[str, Any],
    source_hashes: dict[str, str],
    rows: list[SeedRow],
    internal_dieu: set[int],
) -> str:
    """Băm payload semantic capture, bỏ timestamp, output path và run identity."""
    payload = {
        "schema_version": SEED_SCHEMA_VERSION_V2,
        "source_kind": "direct_retrieval",
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_hash,
        "whitelist_sha256": whitelist_hash,
        "retrieval_config": retrieval_config,
        "captured_source_hashes": source_hashes,
        "rows": [row.model_dump(mode="json") for row in rows],
        "internal_dieu": sorted(internal_dieu),
    }
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"seeds-{digest[:16]}"


def capture_seeds(
    dataset_path: Path,
    whitelist_path: Path,
    *,
    settings: SinglePassSettings,
    runtime_factory: Any,
    before_runtime: Any = None,
) -> SeedSnapshot:
    """Validate input, chạy retrieval một lần cho mỗi câu và dựng snapshot v2.

    Hash của corpus/whitelist được lấy từ chính bytes đã parse và provenance code
    được cố định trước khi runtime chạy, nên file đổi giữa capture không thể tạo
    snapshot mang câu hỏi cũ nhưng hash file mới.

    Params:
    - before_runtime: hook tùy chọn chạy sau khi input/provenance hợp lệ và trước
      khi dựng dependency live, ví dụ để nạp env file
    """
    dataset_path = Path(dataset_path)
    whitelist_path = Path(whitelist_path)

    #Đọc và băm bytes gốc trước, rồi parse đúng bytes đó để hash khớp nội dung
    dataset_bytes = dataset_path.read_bytes()
    whitelist_bytes = whitelist_path.read_bytes()
    dataset_data = _parse_json(dataset_path, dataset_bytes)
    whitelist_data = _parse_json(whitelist_path, whitelist_bytes)
    dataset_hash = sha256_bytes(dataset_bytes)
    whitelist_hash = sha256_bytes(whitelist_bytes)

    questions = validate_dataset(dataset_path, dataset_data)
    if not questions:
        raise ValueError(f"{dataset_path}: dataset must not be empty")
    internal_dieu = validate_whitelist(whitelist_path, whitelist_data)

    #Provenance phải là ảnh chụp trước khi runtime chạy, không phải sau
    source_hashes = capture_source_hashes()
    revision = _git_value(["rev-parse", "HEAD"])
    dirty = _git_dirty()
    capture_metadata: dict[str, Any] = {
        "capture_id": uuid.uuid4().hex,
        "captured_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "working_tree_modified": dirty,
        "python_version": sys.version.split()[0],
        "source_hashes": source_hashes,
        "runtime_metadata": {},
    }

    #Runtime chỉ được dựng sau khi input, provenance và hook env đã xong
    if before_runtime is not None:
        before_runtime()
    runtime = runtime_factory(settings)

    rows: list[SeedRow] = []
    for identity, question in questions.items():
        contexts, seed_dieu = runtime.retrieve(question)
        rows.append(
            SeedRow(
                id=identity[1],
                question=question,
                contexts=list(contexts),
                seed_dieu=set(seed_dieu),
            )
        )

    dataset_id = dataset_path.stem
    retrieval_config = dict(settings.model_dump(mode="json"))
    capture_metadata["runtime_metadata"] = dict(getattr(runtime, "metadata", {}) or {})

    return SeedSnapshot(
        schema_version=SEED_SCHEMA_VERSION_V2,
        snapshot_id=_snapshot_id(
            dataset_id=dataset_id,
            dataset_hash=dataset_hash,
            whitelist_hash=whitelist_hash,
            retrieval_config=retrieval_config,
            source_hashes=source_hashes,
            rows=rows,
            internal_dieu=internal_dieu,
        ),
        dataset_id=dataset_id,
        source_kind="direct_retrieval",
        baseline_source=None,
        capture_metadata=capture_metadata,
        dataset_source=SourceFile(path=str(dataset_path), sha256=dataset_hash),
        whitelist_source=SourceFile(path=str(whitelist_path), sha256=whitelist_hash),
        retrieval_config=retrieval_config,
        unavailable_metadata=_unavailable_metadata(revision, dirty),
        internal_dieu=internal_dieu,
        rows=rows,
    )


def publish_captured_snapshot(snapshot: SeedSnapshot, output_path: Path) -> None:
    """Ghi snapshot capture đã validate bằng cơ chế không ghi đè."""
    if snapshot.schema_version != SEED_SCHEMA_VERSION_V2:
        raise ValueError("only direct retrieval snapshots can be published by capture")
    if snapshot.source_kind != "direct_retrieval" or snapshot.baseline_source is not None:
        raise ValueError("only direct retrieval snapshots can be published by capture")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = snapshot.model_dump(mode="json")
    temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as target:
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        #os.link tạo đích theo kiểu atomic và thất bại nếu đích đã tồn tại
        try:
            os.link(temporary, output_path)
        except FileExistsError:
            raise ValueError(f"{output_path}: destination already exists") from None
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "CAPTURE_SOURCE_PATHS",
    "CaptureRuntime",
    "capture_seeds",
    "capture_source_hashes",
    "publish_captured_snapshot",
]
