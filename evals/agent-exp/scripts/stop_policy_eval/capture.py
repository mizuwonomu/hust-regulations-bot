"""Capture hop-0 STOP state từ live retrieval và publish bundle label review một lần

Chỉ `id` và `user_input` là retrieval input. Draft case được dựng lại từ chính
snapshot đã capture nên observation và candidate order khớp boundary của loop;
bundle chỉ được publish sau khi mọi row và mọi draft validate xong
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from contracts import SeedSnapshot
from seed_capture import capture_seeds

from stop_policy_eval.cases import (
    cases_bytes,
    hop0_case_id,
    parse_stop_corpus,
    read_stop_corpus,
    snapshot_content_bytes,
    snapshot_content_hash,
)
from diagnostic_subexp.shared.contracts import (
    canonical_json,
    sha256_bytes,
    sha256_text,
)

from stop_policy_eval.contracts import (
    STOP_POLICY_SCHEMA_VERSION,
    StopCaptureManifest,
    StopPolicyCase,
)
from stop_policy_eval.state import rebuild_case_state

BUNDLE_FILES: tuple[str, ...] = (
    "seeds.json",
    "cases_draft.jsonl",
    "cases_review.jsonl",
    "label_review.md",
)
MANIFEST_FILE = "capture_manifest.json"
REPO_ROOT = Path(__file__).resolve().parents[4]
STOP_CAPTURE_PROVENANCE_PATHS: tuple[str, ...] = (
    "evals/agent-exp/scripts/stop_policy_eval/capture.py",
    "evals/agent-exp/scripts/stop_policy_eval/state.py",
    "evals/agent-exp/scripts/stop_policy_eval/cases.py",
    "evals/agent-exp/scripts/stop_policy_eval/contracts.py",
    "evals/agent-exp/scripts/stop_policy_eval/cli.py",
    "evals/agent-exp/scripts/seed_cases.py",
    "src/ingestion/reference_parser.py",
    "src/rag/agent/loop.py",
    "src/rag/agent/tools.py",
    "src/rag/agent/schema.py",
)


@dataclass(frozen=True, slots=True)
class StopCaptureOutcome:
    """Bundle capture đã publish cùng manifest, snapshot và draft case của nó"""

    capture_dir: Path
    manifest: StopCaptureManifest
    snapshot: SeedSnapshot
    draft_cases: list[StopPolicyCase]


def _required_text(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"capture metadata is missing {key}")
    return value


def _capture_source_hashes(source_hashes: dict[str, Any]) -> dict[str, str]:
    """Bổ sung hash các module STOP và source tạo state gate canonical"""
    captured = {str(name): str(digest) for name, digest in source_hashes.items()}
    for relative in STOP_CAPTURE_PROVENANCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"required capture source is missing: {relative}")
        content = path.read_bytes()
        if not content:
            raise ValueError(f"required capture source is empty: {relative}")
        captured[relative] = sha256_bytes(content)
    return {name: captured[name] for name in sorted(captured)}


def prepare_draft_cases(
    snapshot: SeedSnapshot,
    corpus_bytes: bytes,
    *,
    source_label: str = "<corpus bytes>",
) -> list[StopPolicyCase]:
    """Dựng draft hop-0 không nhãn từ snapshot và bytes corpus đã băm

    Params:
    - snapshot: snapshot v2 vừa capture từ direct retrieval
    - corpus_bytes: bytes gốc của corpus, dùng để bind row hash và dataset hash
    - source_label: nhãn đường dẫn cho thông báo lỗi
    """
    corpus = parse_stop_corpus(corpus_bytes, path_label=source_label)
    if snapshot.dataset_source.sha256 != corpus.sha256:
        raise ValueError(
            f"{source_label}: corpus bytes do not match the captured snapshot dataset hash"
        )
    corpus_rows = corpus.by_identity()
    snapshot_rows = {_identity(row.id): row for row in snapshot.rows}
    if set(corpus_rows) != set(snapshot_rows):
        missing = sorted(repr(key[1]) for key in set(snapshot_rows) - set(corpus_rows))
        extra = sorted(repr(key[1]) for key in set(corpus_rows) - set(snapshot_rows))
        raise ValueError(
            f"{source_label}: corpus rows do not match the captured snapshot, "
            f"missing={missing}, extra={extra}"
        )

    snapshot_hash = snapshot_content_hash(snapshot)
    drafts: list[StopPolicyCase] = []
    for row in snapshot.rows:
        corpus_row = corpus_rows[_identity(row.id)]
        if corpus_row.question != row.question:
            raise ValueError(
                f"{source_label}: corpus question for id {row.id!r} does not match the snapshot row"
            )
        observation, candidates = rebuild_case_state(snapshot, row)
        drafts.append(
            StopPolicyCase(
                case_id=hop0_case_id(snapshot.dataset_id, row.id),
                dataset_id=snapshot.dataset_id,
                question_id=row.id,
                snapshot_id=snapshot.snapshot_id,
                snapshot_hash=snapshot_hash,
                question=row.question,
                observation=observation,
                observation_hash=sha256_text(observation),
                candidates=candidates,
                source_hop=0,
                expected_action="unresolved" if candidates else "no_candidates",
                acceptable_dieu=[],
                label_reason=None,
                label_status="draft",
                label_observation_hash=None,
                semantic_group_id=None,
                split="unassigned",
                fewshot_overlap=None,
                overlap_notes=None,
                source_row_hash=corpus_row.row_hash,
                source_dataset_hash=corpus.sha256,
            )
        )
    return drafts


def _identity(question_id: Any) -> tuple[type[Any], Any]:
    return type(question_id), question_id


def _review_state_note(case: StopPolicyCase) -> str:
    if not case.candidates:
        return "no_candidates - empty frontier, mechanically excluded; never label this STOP"
    return "candidates remain - label follow, stop or unresolved against the observation below"


def _review_section(case: StopPolicyCase) -> list[str]:
    candidates = ", ".join(f"Điều {dieu}" for dieu in case.candidates) or "(none)"
    return [
        f"## {case.case_id}",
        "",
        f"- Dataset: {case.dataset_id}",
        f"- Question id: {case.question_id!r}",
        f"- Source hop: {case.source_hop}",
        f"- Snapshot: {case.snapshot_id}",
        f"- Source row hash: {case.source_row_hash}",
        f"- Observation hash: {case.observation_hash}",
        f"- State: {_review_state_note(case)}",
        f"- Candidates in the order the gate will see them: {candidates}",
        "",
        "Question (exact):",
        "",
        "```text",
        case.question,
        "```",
        "",
        "Observation (exact gate input):",
        "",
        "```text",
        case.observation,
        "```",
        "",
        "Review fields to fill in `cases_review.jsonl`:",
        "",
        "- `expected_action`: `follow` when a remaining citation edge is still needed, `stop` when",
        "  none is needed, `unresolved` when the observation is not clear enough to decide",
        "- `acceptable_dieu`: empty for `stop` and `unresolved`; a non-empty subset of the listed",
        "  candidates for `follow`",
        f"- `label_observation_hash`: copy `{case.observation_hash}` exactly",
        "- `label_reason`: non-blank rationale comparing the question with the excerpts above",
        "- `semantic_group_id`: one group per original question, paraphrase or hop chain",
        "- `split`: `dev` or `heldout`, identical for every case in the group",
        "- `fewshot_overlap`: `false` only when the case shares no semantic content with the",
        "  current few-shot demonstrations",
        "- `overlap_notes`: what overlaps, when `fewshot_overlap` is `true`",
        "",
    ]


def build_review_markdown(cases: list[StopPolicyCase]) -> str:
    """Render hướng dẫn review chỉ từ state đã capture, không có nhãn hay output model"""
    lines = [
        "# STOP policy label review",
        "",
        "Edit `cases_review.jsonl` in place; this file is read-only guidance",
        "",
        f"- Draft cases: {len(cases)}",
        f"- Mechanical exclusions (empty frontier): {sum(1 for case in cases if not case.candidates)}",
        "- Approve a label only against the exact question, observation and ordered candidates below",
        "",
    ]
    for case in cases:
        lines.extend(_review_section(case))
    return "\n".join(lines) + "\n"


def _capture_manifest(
    snapshot: SeedSnapshot,
    *,
    generated_files: dict[str, str],
) -> StopCaptureManifest:
    metadata = dict(snapshot.capture_metadata or {})
    source_hashes = metadata.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("capture metadata is missing the captured source hashes")
    unavailable = dict(snapshot.unavailable_metadata)
    if snapshot.capture_metadata is None:
        raise ValueError("capture manifests require capture metadata from the direct retrieval run")
    return StopCaptureManifest(
        schema_version=STOP_POLICY_SCHEMA_VERSION,
        capture_status="captured",
        capture_id=_required_text(metadata, "capture_id"),
        captured_at=_required_text(metadata, "captured_at"),
        dataset_source=snapshot.dataset_source,
        inventory_source=snapshot.whitelist_source,
        snapshot_id=snapshot.snapshot_id,
        snapshot_hash=snapshot_content_hash(snapshot),
        row_count=len(snapshot.rows),
        generated_files=generated_files,
        retrieval_config=snapshot.retrieval_config,
        captured_source_hashes=_capture_source_hashes(source_hashes),
        runtime_metadata=dict(metadata.get("runtime_metadata") or {}),
        source_revision=metadata.get("source_revision"),
        working_tree_modified=metadata.get("working_tree_modified"),
        python_version=_required_text(metadata, "python_version"),
        unavailable_metadata=unavailable,
    )


def _fsync_directory(path: Path) -> None:
    handle = os.open(path, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _rename_directory_no_replace(source: Path, target: Path) -> None:
    """Đổi tên thư mục chỉ khi đích chưa tồn tại, nếu thiếu primitive thì dừng"""
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, "destination already exists", str(target))
        raise OSError(error_number, os.strerror(error_number), str(target))


def publish_bundle(output_dir: Path, files: dict[str, bytes]) -> Path:
    """Publish bundle bằng một atomic no-replace directory rename

    Destination phải chưa tồn tại; rename no-replace từ chối cả thư mục rỗng cạnh tranh
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir}: destination already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, payload in files.items():
            if os.sep in name or name in {"", ".", ".."}:
                raise ValueError(f"bundle entry must be a plain file name: {name!r}")
            with (temporary / name).open("wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
        _fsync_directory(temporary)
        try:
            _rename_directory_no_replace(temporary, output_dir)
        except FileExistsError as exc:
            raise ValueError(f"{output_dir}: destination already exists") from exc
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
    return output_dir


def capture_corpus(
    dataset_path: Path,
    inventory_path: Path,
    *,
    output_dir: Path,
    settings: Any,
    runtime_factory: Any,
    before_runtime: Callable[[], None] | None = None,
) -> StopCaptureOutcome:
    """Capture corpus thành một bundle đóng băng rồi publish nó một lần

    Params:
    - dataset_path: corpus gốc; chỉ id và user_input đi vào retrieval
    - inventory_path: whitelist internal Điều của corpus
    - output_dir: đích mới, tuyệt đối không ghi đè
    - settings: `SinglePassSettings` hiệu lực của retrieval
    - runtime_factory: factory runtime retrieval, chỉ được gọi sau preflight
    - before_runtime: hook được gọi sau khi input/provenance hợp lệ và trước runtime
    """
    dataset_path = Path(dataset_path)
    inventory_path = Path(inventory_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir}: destination already exists")

    # Preflight input và đích xong mới dựng provenance và dependency live
    corpus = read_stop_corpus(dataset_path)
    corpus_bytes = corpus.raw
    inventory_bytes = inventory_path.read_bytes()

    snapshot = capture_seeds(
        dataset_path,
        inventory_path,
        settings=settings,
        runtime_factory=runtime_factory,
        before_runtime=before_runtime,
    )
    if snapshot.dataset_source.sha256 != corpus.sha256:
        raise ValueError(f"{dataset_path}: corpus bytes changed during capture")
    if snapshot.whitelist_source.sha256 != sha256_bytes(inventory_bytes):
        raise ValueError(f"{inventory_path}: inventory bytes changed during capture")

    drafts = prepare_draft_cases(snapshot, corpus_bytes, source_label=str(dataset_path))
    seeds_payload = snapshot_content_bytes(snapshot)
    cases_payload = cases_bytes(drafts)
    review_payload = build_review_markdown(drafts).encode("utf-8")
    generated = {
        "seeds.json": seeds_payload,
        "cases_draft.jsonl": cases_payload,
        "cases_review.jsonl": cases_payload,
        "label_review.md": review_payload,
    }
    manifest = _capture_manifest(
        snapshot,
        generated_files={name: sha256_bytes(payload) for name, payload in generated.items()},
    )
    files = dict(generated)
    files[MANIFEST_FILE] = canonical_json(manifest.model_dump(mode="json"))
    publish_bundle(output_dir, files)
    return StopCaptureOutcome(
        capture_dir=output_dir,
        manifest=manifest,
        snapshot=snapshot,
        draft_cases=drafts,
    )


def read_capture_manifest(capture_dir: Path) -> StopCaptureManifest:
    """Đọc capture manifest và kiểm lại hash của mọi file đã sinh"""
    capture_dir = Path(capture_dir)
    path = capture_dir / MANIFEST_FILE
    try:
        payload = read_json_file(path)
    except ValueError as exc:
        raise ValueError(f"{path}: cannot read the capture manifest: {exc}") from exc
    try:
        manifest = StopCaptureManifest.model_validate(payload)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid capture manifest: {exc}") from exc
    for name, expected in manifest.generated_files.items():
        target = capture_dir / name
        if not target.is_file():
            raise ValueError(f"{target}: a generated capture file is missing")
        actual = sha256_bytes(target.read_bytes())
        if actual != expected:
            raise ValueError(f"{target}: generated file hash mismatch, expected {expected}")
    return manifest


def read_json_file(path: Path) -> Any:
    """Đọc JSON từ một file với thông báo lỗi có số dòng"""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}") from exc


__all__ = [
    "BUNDLE_FILES",
    "MANIFEST_FILE",
    "StopCaptureOutcome",
    "build_review_markdown",
    "capture_corpus",
    "prepare_draft_cases",
    "publish_bundle",
    "read_capture_manifest",
    "read_json_file",
    "snapshot_content_bytes",
    "snapshot_content_hash",
]
