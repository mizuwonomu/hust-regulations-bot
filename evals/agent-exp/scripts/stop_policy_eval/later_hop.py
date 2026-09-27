"""Advance một source FOLLOW thành một frozen later-hop state hoặc terminal outcome"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from artifacts import read_snapshot
from contracts import SeedSnapshot
from diagnostic_subexp.shared.contracts import canonical_json, sha256_text, structured_hash

from stop_policy_eval.artifacts import StopArtifactError, load_run
from stop_policy_eval.capture import publish_bundle
from stop_policy_eval.cases import read_later_hop_states, resolve_source_file
from stop_policy_eval.contracts import (
    CollectedArticleReference,
    LaterHopState,
    LaterHopTerminalOutcome,
    StopPolicyCase,
    StopResult,
    StopRunManifest,
    frozen_state_payload,
)
from stop_policy_eval.state import _dieu_from_title, _normalize_seed, rebuild_collected_state

STATE_FILE = "later_hop_states.jsonl"
TERMINAL_FILE = "terminal_outcomes.jsonl"
EXPORT_MANIFEST_FILE = "later_hop_export_manifest.json"


@dataclass(frozen=True, slots=True)
class LaterHopExportOutcome:
    """Kết quả export successor states và frontier terminal từ một source run"""

    output_dir: Path
    source_run_id: str
    states: list[LaterHopState]
    terminal_outcomes: list[LaterHopTerminalOutcome]
    skipped_results: int


def build_local_article_fetcher() -> Callable[[int], str | None]:
    """Dựng article lookup một lần từ Chroma map và parent doc store"""
    from langchain_chroma import Chroma
    from langchain_classic.storage import EncoderBackedStore, LocalFileStore

    from src.rag.agent.tools import build_article_map, get_article
    from src.rag.config import CHROMA_COLLECTION, CHROMA_PATH, DOC_STORE_PATH

    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION,
        persist_directory=CHROMA_PATH,
    )
    article_map = build_article_map(vector_store)
    file_store = LocalFileStore(DOC_STORE_PATH)
    doc_store = EncoderBackedStore(
        store=file_store,
        key_encoder=lambda value: value,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )
    return lambda dieu: get_article(
        dieu,
        article_map=article_map,
        doc_store=doc_store,
    )


def _jsonl_bytes(records: list[LaterHopState] | list[LaterHopTerminalOutcome]) -> bytes:
    rows = []
    for record in records:
        payload = record.model_dump(mode="json")
        if isinstance(record, LaterHopState):
            payload["state_hash"] = record.state_hash
        rows.append(canonical_json(payload) + b"\n")
    return b"".join(rows)


def _snapshot_for_source_run(manifest: StopRunManifest) -> SeedSnapshot:
    snapshot_path = resolve_source_file(manifest.snapshot_source)
    try:
        return read_snapshot(snapshot_path)
    except (OSError, ValueError) as exc:
        raise StopArtifactError(f"cannot read source snapshot: {exc}") from exc


def _verify_source_run(
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    results: list[StopResult],
) -> None:
    if manifest.run_purpose != "later_hop_capture":
        raise StopArtifactError("source run purpose must be later_hop_capture")
    if len(manifest.policies) != 1 or manifest.repeats != 1:
        raise StopArtifactError("later-hop source runs require one policy and one repeat")
    if manifest.status not in {"complete", "completed_with_errors"}:
        raise StopArtifactError(f"source run is not complete: {manifest.status}")

    case_by_id = {case.case_id: case for case in cases}
    result_by_trial = {
        result.trial_id: result for result in results if result.policy == manifest.policies[0]
    }
    for trial in manifest.trials:
        case = case_by_id.get(trial.case_id)
        if case is None or case.source_hop != 0:
            raise StopArtifactError(
                f"source trial {trial.trial_id} must reference a hop-0 parent case"
            )
        result = result_by_trial.get(trial.trial_id)
        if result is None or result.run_id != manifest.run_id:
            raise StopArtifactError(f"source trial {trial.trial_id} has no matching result")
        if result.input_trace_id != trial.input_trace_id:
            raise StopArtifactError(f"source trial {trial.trial_id} result changes its input trace")


def export_later_hop(
    source_run_dir: Path,
    output_dir: Path,
    *,
    article_fetcher: Callable[[int], str | None] | None = None,
) -> LaterHopExportOutcome:
    """Fetch exactly one followed article per valid source FOLLOW, then freeze its next state"""
    source_run_dir = Path(source_run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"{output_dir}: destination already exists")
    manifest, cases, traces, results = load_run(source_run_dir)
    _verify_source_run(manifest, cases, results)
    snapshot = _snapshot_for_source_run(manifest)
    fetch_article = article_fetcher or build_local_article_fetcher()

    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    case_by_id = {case.case_id: case for case in cases}
    trace_by_id = {trace.input_trace_id: trace for trace in traces}
    result_by_trial = {
        result.trial_id: result for result in results if result.policy == manifest.policies[0]
    }
    snapshot_rows = {
        (type(row.id), row.id): row for row in snapshot.rows
    }
    states: list[LaterHopState] = []
    terminals: list[LaterHopTerminalOutcome] = []
    skipped = 0

    for trial in manifest.trials:
        result = result_by_trial[trial.trial_id]
        outcome = result.outcome
        decision = outcome.decision
        if (
            outcome.status != "ok"
            or decision is None
            or decision.stop
            or decision.dieu is None
        ):
            skipped += 1
            continue
        trace = trace_by_id[trial.input_trace_id]
        if decision.dieu not in trace.candidates:
            raise StopArtifactError(
                f"source trial {trial.trial_id} followed an article outside its frozen candidates"
            )
        parent = case_by_id[trial.case_id]
        row = snapshot_rows.get((type(parent.question_id), parent.question_id))
        if row is None:
            raise StopArtifactError(f"source parent {parent.case_id} is absent from the snapshot")
        collected = _normalize_seed(row)
        if decision.dieu in collected:
            raise StopArtifactError(
                f"source trial {trial.trial_id} followed an article already in collected state"
            )
        article_text = fetch_article(decision.dieu)
        if not isinstance(article_text, str) or not article_text:
            raise StopArtifactError(
                f"source trial {trial.trial_id}: followed article {decision.dieu} was not found"
            )
        if _dieu_from_title(article_text) != decision.dieu:
            raise StopArtifactError(
                f"source trial {trial.trial_id}: fetched article title does not match "
                f"followed Điều {decision.dieu}"
            )

        article_hash = hashlib.sha256(article_text.encode("utf-8")).hexdigest()
        collected[decision.dieu] = article_text
        observation, candidates = rebuild_collected_state(collected, snapshot.internal_dieu)
        reference = CollectedArticleReference(dieu=decision.dieu, sha256=article_hash)
        refs = [*parent.collected_article_refs, reference]
        state_payload = {
            "dataset_id": snapshot.dataset_id,
            "question_id": parent.question_id,
            "source_hop": 1,
            "question": parent.question,
            "observation": observation,
            "candidates": candidates,
            "source_run_id": manifest.run_id,
            "source_policy": result.policy,
            "source_trial_id": trial.trial_id,
            "parent_case_id": parent.case_id,
            "followed_dieu": decision.dieu,
            "followed_article_sha256": article_hash,
            "collected_article_refs": [item.model_dump(mode="json") for item in refs],
        }
        if candidates:
            states.append(LaterHopState.model_validate(state_payload))
        else:
            terminal_hash = structured_hash(
                frozen_state_payload(
                    **state_payload
                )
            )
            terminals.append(
                LaterHopTerminalOutcome(
                    **state_payload,
                    observation_hash=sha256_text(observation),
                    terminal_reason="empty_frontier",
                    terminal_state_hash=terminal_hash,
                )
            )

    states_bytes = _jsonl_bytes(states)
    terminals_bytes = _jsonl_bytes(terminals)
    source_manifest_bytes = (source_run_dir / "manifest.json").read_bytes()
    result_hashes = {}
    for policy in manifest.policies:
        path = source_run_dir / f"results_{policy}.jsonl"
        if path.is_file():
            result_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    export_manifest = {
        "schema_version": 1,
        "source_run_id": manifest.run_id,
        "source_run_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        "source_result_hashes": result_hashes,
        "states_sha256": hashlib.sha256(states_bytes).hexdigest(),
        "terminal_outcomes_sha256": hashlib.sha256(terminals_bytes).hexdigest(),
        "state_count": len(states),
        "terminal_count": len(terminals),
        "skipped_result_count": skipped,
    }
    publish_bundle(
        output_dir,
        {
            STATE_FILE: states_bytes,
            TERMINAL_FILE: terminals_bytes,
            EXPORT_MANIFEST_FILE: canonical_json(export_manifest),
        },
    )
    return LaterHopExportOutcome(
        output_dir=output_dir,
        source_run_id=manifest.run_id,
        states=states,
        terminal_outcomes=terminals,
        skipped_results=skipped,
    )


def verify_export_bundle(states_path: Path, source_run_dir: Path) -> None:
    """Kiểm byte hash của export bundle đã sinh trước khi import"""
    states_path = Path(states_path)
    bundle_dir = states_path.parent
    manifest_path = bundle_dir / EXPORT_MANIFEST_FILE
    if states_path.name != STATE_FILE or not manifest_path.is_file():
        raise StopArtifactError(f"{states_path}: expected a later-hop export bundle")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StopArtifactError(f"{manifest_path}: invalid export manifest") from exc
    state_bytes = states_path.read_bytes()
    terminal_path = bundle_dir / TERMINAL_FILE
    if not terminal_path.is_file():
        raise StopArtifactError(f"{terminal_path}: terminal outcome file is missing")
    if hashlib.sha256(state_bytes).hexdigest() != payload.get("states_sha256"):
        raise StopArtifactError(f"{states_path}: exported state file hash mismatch")
    if hashlib.sha256(terminal_path.read_bytes()).hexdigest() != payload.get(
        "terminal_outcomes_sha256"
    ):
        raise StopArtifactError(f"{terminal_path}: terminal outcome file hash mismatch")
    states = read_later_hop_states(states_path, allow_empty=True)
    if len(states) != payload.get("state_count"):
        raise StopArtifactError(f"{states_path}: state count differs from export manifest")
    terminal_rows = []
    for line_number, line in enumerate(
        terminal_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise StopArtifactError(f"{terminal_path}: blank line at {line_number}")
        try:
            terminal_rows.append(LaterHopTerminalOutcome.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise StopArtifactError(
                f"{terminal_path}: invalid terminal outcome at line {line_number}"
            ) from exc
    if len(terminal_rows) != payload.get("terminal_count"):
        raise StopArtifactError(f"{terminal_path}: terminal count differs from export manifest")
    source_run_dir = Path(source_run_dir)
    source_manifest = source_run_dir / "manifest.json"
    if not source_manifest.is_file():
        raise StopArtifactError(f"{source_manifest}: source run manifest is missing")
    if payload.get("source_run_id") != json.loads(
        source_manifest.read_text(encoding="utf-8")
    ).get("run_id"):
        raise StopArtifactError("later-hop export refers to a different source run")
    expected_manifest_hash = payload.get("source_run_manifest_sha256")
    if hashlib.sha256(source_manifest.read_bytes()).hexdigest() != expected_manifest_hash:
        raise StopArtifactError("source run manifest changed after later-hop export")
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected_result_names = {
        f"results_{policy}.jsonl" for policy in source_payload.get("policies", [])
    }
    source_result_hashes = payload.get("source_result_hashes", {})
    if set(source_result_hashes) != expected_result_names:
        raise StopArtifactError("export manifest source result roster differs from the run manifest")
    for name, digest in source_result_hashes.items():
        if Path(name).name != name:
            raise StopArtifactError(f"export manifest contains an unsafe result path: {name}")
        result_path = source_run_dir / name
        if not result_path.is_file() or hashlib.sha256(result_path.read_bytes()).hexdigest() != digest:
            raise StopArtifactError(f"source result artifact changed after export: {name}")


def verify_later_hop_source_provenance(
    states: list[LaterHopState],
    manifest: StopRunManifest,
    cases: list[StopPolicyCase],
    results: list[StopResult],
) -> None:
    """Xác thực mọi state import bằng policy, trial và FOLLOW result trong source run"""
    _verify_source_run(manifest, cases, results)
    trial_by_id = {trial.trial_id: trial for trial in manifest.trials}
    case_by_id = {case.case_id: case for case in cases}
    result_by_slot = {(result.policy, result.trial_id): result for result in results}
    for state in states:
        if state.source_run_id != manifest.run_id:
            raise StopArtifactError(f"state {state.state_hash}: source_run_id mismatch")
        if state.source_policy not in manifest.policies:
            raise StopArtifactError(f"state {state.state_hash}: source_policy is not scheduled")
        trial = trial_by_id.get(state.source_trial_id)
        if trial is None or trial.case_id != state.parent_case_id:
            raise StopArtifactError(f"state {state.state_hash}: source trial/parent mismatch")
        parent = case_by_id.get(trial.case_id)
        if parent is None or parent.source_hop != 0:
            raise StopArtifactError(f"state {state.state_hash}: source parent is not hop-0")
        if state.source_hop != 1:
            raise StopArtifactError(
                f"state {state.state_hash}: this exporter accepts one-edge states from hop-0 source runs"
            )
        if len(state.collected_article_refs) != state.source_hop:
            raise StopArtifactError(
                f"state {state.state_hash}: collected article references do not cover each hop"
            )
        if (type(parent.question_id), parent.question_id) != (
            type(state.question_id),
            state.question_id,
        ):
            raise StopArtifactError(f"state {state.state_hash}: source question id mismatch")
        result = result_by_slot.get((state.source_policy, state.source_trial_id))
        decision = None if result is None else result.outcome.decision
        if (
            result is None
            or result.run_id != manifest.run_id
            or result.outcome.status != "ok"
            or decision is None
            or decision.stop
            or decision.dieu != state.followed_dieu
        ):
            raise StopArtifactError(
                f"state {state.state_hash}: no matching valid FOLLOW result in source artifacts"
            )
        if decision.dieu not in parent.candidates:
            raise StopArtifactError(
                f"state {state.state_hash}: followed article is outside the parent candidates"
            )
        if state.collected_article_refs[:-1] != parent.collected_article_refs:
            raise StopArtifactError(
                f"state {state.state_hash}: collected article lineage differs from its parent"
            )


__all__ = [
    "EXPORT_MANIFEST_FILE",
    "STATE_FILE",
    "TERMINAL_FILE",
    "LaterHopExportOutcome",
    "build_local_article_fetcher",
    "export_later_hop",
    "verify_export_bundle",
    "verify_later_hop_source_provenance",
]
