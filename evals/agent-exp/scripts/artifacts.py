"""Persist experiment inputs, manifests, results, debug records, and reloads."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from contracts import (
        GateCase,
        ResultRecord,
        RunManifest,
        SeedSnapshot,
        Summary,
    )
    from seed_cases import read_cases, validate_replay
except ModuleNotFoundError:
    from .contracts import GateCase, ResultRecord, RunManifest, SeedSnapshot, Summary
    from .seed_cases import read_cases, validate_replay


class ArtifactCorruptionError(ValueError):
    """Signal a malformed persisted JSON or JSONL artifact."""


def sha256_bytes(value: bytes) -> str:
    """Hash bytes with SHA-256."""
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash UTF-8 text with SHA-256."""
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    """Hash a file's bytes with SHA-256."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(path)


def write_snapshot(snapshot: SeedSnapshot, output_path: Path) -> None:
    """Write one validated seed snapshot as JSON."""
    _atomic_write_json(output_path, snapshot.model_dump(mode="json"))


def read_snapshot(path: Path) -> SeedSnapshot:
    """Read and validate one seed snapshot JSON file."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as exc:
        raise ArtifactCorruptionError(f"{path}: invalid JSON at line {exc.lineno}") from exc
    try:
        return SeedSnapshot.model_validate(payload)
    except ValueError as exc:
        raise ArtifactCorruptionError(f"{path}: invalid seed snapshot: {exc}") from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_source_path(path: str, *, run_dir: Path | None = None) -> Path:
    """Resolve a committed relative input path without silently replacing it."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    candidates = [Path.cwd() / candidate, _repo_root() / candidate]
    if run_dir is not None:
        run_dir = Path(run_dir)
        candidates.extend(
            [
                run_dir / candidate,
                run_dir.parent / candidate,
                run_dir.parent.parent / candidate,
            ]
        )
    for resolved in candidates:
        if resolved.exists():
            return resolved
    return candidates[0]


def _manifest_payload(manifest: RunManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")


def write_manifest(run_dir: Path, manifest: RunManifest) -> None:
    """Atomically update a run manifest."""
    _atomic_write_json(Path(run_dir) / "manifest.json", _manifest_payload(manifest))


def _run_directory_name(manifest: RunManifest) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_run_id = manifest.run_id.replace("/", "_").replace("\\", "_")
    return f"{timestamp}_{manifest.experiment}_{safe_run_id}"


def create_run(output_root: Path, manifest: RunManifest) -> Path:
    """Create a non-overwriting run directory and save its running manifest."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / _run_directory_name(manifest)
    run_dir.mkdir()
    write_manifest(run_dir, manifest)
    return run_dir


def _result_path(run_dir: Path, policy: str) -> Path:
    return Path(run_dir) / f"results_{policy}.jsonl"


def append_result(run_dir: Path, result: ResultRecord) -> None:
    """Append one compact policy result and flush it to disk."""
    path = _result_path(run_dir, result.policy)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
    with path.open("a", encoding="utf-8") as target:
        target.write(line)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def append_debug_record(run_dir: Path, policy: str, payload: dict[str, Any]) -> None:
    """Append a verbose local decision record for one policy."""
    debug_dir = Path(run_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"decisions_{policy}.jsonl"
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(payload, ensure_ascii=False, default=str))
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def append_policy_log(run_dir: Path, policy: str, message: str) -> None:
    """Append a local human-readable policy log line."""
    debug_dir = Path(run_dir) / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"run_{policy}.log"
    with path.open("a", encoding="utf-8") as target:
        target.write(message.rstrip("\n"))
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())


def _read_manifest(run_dir: Path) -> RunManifest:
    path = Path(run_dir) / "manifest.json"
    try:
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError as exc:
        raise ArtifactCorruptionError(f"{path}: invalid JSON at line {exc.lineno}") from exc
    try:
        return RunManifest.model_validate(payload)
    except ValueError as exc:
        raise ArtifactCorruptionError(f"{path}: invalid manifest: {exc}") from exc


def _read_result_file(
    path: Path,
    *,
    policy: str,
    manifest: RunManifest,
) -> list[ResultRecord]:
    if not path.exists():
        return []

    planned = {trial.trial_id: trial for trial in manifest.trials}
    seen: set[tuple[str, str]] = set()
    records: list[ResultRecord] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ArtifactCorruptionError(f"{path}: blank line at {line_number}")
            try:
                payload = json.loads(line)
                record = ResultRecord.model_validate(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ArtifactCorruptionError(
                    f"{path}: invalid result at line {line_number}"
                ) from exc
            if record.policy != policy:
                raise ArtifactCorruptionError(
                    f"{path}: line {line_number} has policy {record.policy!r}"
                )
            if record.run_id != manifest.run_id:
                raise ArtifactCorruptionError(
                    f"{path}: line {line_number} has a different run_id"
                )
            planned_trial = planned.get(record.trial.trial_id)
            if planned_trial is None:
                raise ArtifactCorruptionError(
                    f"{path}: line {line_number} references an unknown trial"
                )
            if record.trial.model_dump(mode="json") != planned_trial.model_dump(mode="json"):
                raise ArtifactCorruptionError(
                    f"{path}: line {line_number} changes the saved trial input"
                )
            key = (record.policy, record.trial.trial_id)
            if key in seen:
                raise ArtifactCorruptionError(
                    f"{path}: duplicate result for trial {record.trial.trial_id}"
                )
            seen.add(key)
            records.append(record)
    return records


def _verify_source(source_path: str, expected_hash: str, *, run_dir: Path) -> Path:
    path = resolve_source_path(source_path, run_dir=run_dir)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ArtifactCorruptionError(
            f"{path}: source hash mismatch, expected {expected_hash}, got {actual_hash}"
        )
    return path


def load_run(run_dir: Path) -> tuple[RunManifest, list[GateCase], list[ResultRecord]]:
    """Reload a run and verify its committed inputs and result schedule."""
    run_dir = Path(run_dir)
    manifest = _read_manifest(run_dir)
    snapshot_path = _verify_source(
        manifest.snapshot_source.path,
        manifest.snapshot_source.sha256,
        run_dir=run_dir,
    )
    cases_path = _verify_source(
        manifest.cases_source.path,
        manifest.cases_source.sha256,
        run_dir=run_dir,
    )
    snapshot = read_snapshot(snapshot_path)
    cases = read_cases(cases_path)
    snapshot_hash = sha256_file(snapshot_path)
    replay = validate_replay(snapshot, cases, snapshot_hash=snapshot_hash)
    if [item.model_dump(mode="json") for item in replay.exclusions] != [
        item.model_dump(mode="json") for item in manifest.exclusions
    ]:
        raise ArtifactCorruptionError("manifest exclusions differ from recomputed case exclusions")

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ArtifactCorruptionError("cases source contains duplicate case_id values")
    for trial in manifest.trials:
        case = case_by_id.get(trial.case_id)
        if case is None:
            raise ArtifactCorruptionError(
                f"manifest trial {trial.trial_id} references an unknown case"
            )
        if trial.observation_hash != case.observation_hash:
            raise ArtifactCorruptionError(
                f"manifest trial {trial.trial_id} changes the observation hash"
            )
        if trial.candidate_order != case.candidates:
            raise ArtifactCorruptionError(
                f"manifest trial {trial.trial_id} changes candidate order"
            )

    results: list[ResultRecord] = []
    for policy in manifest.policies:
        results.extend(
            _read_result_file(
                _result_path(run_dir, policy),
                policy=policy,
                manifest=manifest,
            )
        )

    eligible_case_ids = {case.case_id for case in replay.eligible_cases}
    if any(trial.case_id not in eligible_case_ids for trial in manifest.trials):
        raise ArtifactCorruptionError("manifest contains an ineligible trial")
    return manifest, cases, results


def finalize_run(run_dir: Path, manifest: RunManifest, summary: Summary) -> None:
    """Atomically save a summary and the final manifest state."""
    _atomic_write_json(Path(run_dir) / "summary.json", summary.model_dump(mode="json"))
    write_manifest(run_dir, manifest)


__all__ = [
    "ArtifactCorruptionError",
    "append_debug_record",
    "append_policy_log",
    "append_result",
    "create_run",
    "finalize_run",
    "load_run",
    "read_snapshot",
    "resolve_source_path",
    "sha256_bytes",
    "sha256_file",
    "sha256_text",
    "write_manifest",
    "write_snapshot",
]
