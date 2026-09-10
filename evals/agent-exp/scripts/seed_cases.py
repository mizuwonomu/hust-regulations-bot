"""Import frozen seeds and prepare loop-compatible initial-gate cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from contracts import (
        Exclusion,
        GateCase,
        GateInput,
        ReplaySelection,
        SeedRow,
        SeedSnapshot,
    )
except ModuleNotFoundError:
    from .contracts import (
        Exclusion,
        GateCase,
        GateInput,
        ReplaySelection,
        SeedRow,
        SeedSnapshot,
    )


SCHEMA_VERSION = 1


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _question_identity(value: Any) -> tuple[type[Any], Any]:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"question id must be an integer or string: {value!r}")
    return type(value), value


def _positive_article_list(value: Any, *, path: Path, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field_name} must be a JSON array")

    parsed: list[int] = []
    seen: set[int] = set()
    for index, article_id in enumerate(value):
        if isinstance(article_id, bool) or not isinstance(article_id, int) or article_id <= 0:
            raise ValueError(
                f"{path}: {field_name}[{index}] must be a positive integer"
            )
        if article_id in seen:
            raise ValueError(
                f"{path}: {field_name}[{index}] duplicates article id {article_id}"
            )
        seen.add(article_id)
        parsed.append(article_id)
    return parsed


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON at line {exc.lineno}") from exc


def _validate_dataset(path: Path, data: Any) -> dict[tuple[type[Any], Any], str]:
    if not isinstance(data, list):
        raise ValueError(f"{path}: dataset must be a JSON array")

    questions: dict[tuple[type[Any], Any], str] = {}
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: dataset row {index} must be an object")
        if "id" not in row:
            raise ValueError(f"{path}: dataset row {index} is missing id")
        identity = _question_identity(row["id"])
        if identity in questions:
            raise ValueError(f"{path}: duplicate dataset id at row {index}: {row['id']!r}")
        question = row.get("user_input")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{path}: dataset row {index} field user_input must be non-empty text"
            )
        questions[identity] = question
    return questions


def _validate_baseline(path: Path, data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict):
        raise ValueError(f"{path}: baseline must be a JSON object")

    config = data.get("config")
    if not isinstance(config, dict) or config.get("agent") is not False:
        raise ValueError(
            f"{path}: source is not proven to be a single-pass baseline"
        )
    if data.get("agent_trajectory") is not None:
        raise ValueError(f"{path}: agent trajectory metadata is not allowed for seeds")

    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{path}: baseline results must be a JSON array")

    identities: set[tuple[type[Any], Any]] = set()
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(results):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: baseline row {index} must be an object")
        if "id" not in row:
            raise ValueError(f"{path}: baseline row {index} is missing id")
        identity = _question_identity(row["id"])
        if identity in identities:
            raise ValueError(f"{path}: duplicate baseline id at row {index}: {row['id']!r}")
        identities.add(identity)

        question = row.get("query")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"{path}: baseline row {index} field query must be non-empty text"
            )
        if "retrieved_contexts" not in row:
            raise ValueError(f"{path}: baseline row {index} is missing retrieved_contexts")
        contexts = row["retrieved_contexts"]
        if not isinstance(contexts, list):
            raise ValueError(
                f"{path}: baseline row {index} retrieved_contexts must be a list"
            )
        for context_index, context in enumerate(contexts):
            if not isinstance(context, str) or not context:
                raise ValueError(
                    f"{path}: baseline row {index} retrieved_contexts[{context_index}] "
                    "must be non-empty text"
                )

        hop_scores = row.get("hop_scores")
        if not isinstance(hop_scores, dict):
            raise ValueError(f"{path}: baseline row {index} is missing hop_scores")
        if "retrieved_dieu" not in hop_scores:
            raise ValueError(
                f"{path}: baseline row {index} is missing hop_scores.retrieved_dieu"
            )
        seed_dieu = _positive_article_list(
            hop_scores["retrieved_dieu"],
            path=path,
            field_name=f"results[{index}].hop_scores.retrieved_dieu",
        )
        if row.get("agent_trace") is not None:
            raise ValueError(
                f"{path}: baseline row {index} contains agent_trace and is not a seed"
            )

        validated.append(
            {
                "id": row["id"],
                "question": question,
                "contexts": list(contexts),
                "seed_dieu": seed_dieu,
            }
        )

    baseline_identities = {
        _question_identity(row["id"])
        for row in validated
    }
    if not baseline_identities:
        raise ValueError(f"{path}: baseline results must not be empty")
    return validated


def _load_whitelist(path: Path) -> set[int]:
    data = _load_json(path)
    values = _positive_article_list(data, path=path, field_name="internal_dieu")
    if not values:
        raise ValueError(f"{path}: internal_dieu whitelist must not be empty")
    return set(values)


def _unavailable_metadata(data: dict[str, Any], config: dict[str, Any]) -> dict[str, str]:
    unavailable: dict[str, str] = {}
    for name in ("source_revision", "retrieval_model", "embedding_model", "reranker_model"):
        if name not in data and name not in config:
            unavailable[name] = "not recorded in the supplied baseline metadata"
    return unavailable


def _snapshot_id(
    *,
    baseline_hash: str,
    dataset_hash: str,
    whitelist_hash: str,
    rows: list[SeedRow],
    internal_dieu: set[int],
) -> str:
    payload = {
        "baseline_sha256": baseline_hash,
        "dataset_sha256": dataset_hash,
        "whitelist_sha256": whitelist_hash,
        "rows": [row.model_dump(mode="json") for row in rows],
        "internal_dieu": sorted(internal_dieu),
    }
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"seeds-{digest[:16]}"


def import_seeds(
    baseline_path: Path,
    dataset_path: Path,
    whitelist_path: Path,
) -> SeedSnapshot:
    """Import and validate a reproduced single-pass seed baseline."""
    baseline_path = Path(baseline_path)
    dataset_path = Path(dataset_path)
    whitelist_path = Path(whitelist_path)

    baseline_data = _load_json(baseline_path)
    dataset_data = _load_json(dataset_path)
    dataset_questions = _validate_dataset(dataset_path, dataset_data)
    baseline_rows = _validate_baseline(baseline_path, baseline_data)
    internal_dieu = _load_whitelist(whitelist_path)

    baseline_ids = {_question_identity(row["id"]) for row in baseline_rows}
    dataset_ids = set(dataset_questions)
    if baseline_ids != dataset_ids:
        missing = sorted(
            [repr(identity[1]) for identity in dataset_ids - baseline_ids]
        )
        extra = sorted(
            [repr(identity[1]) for identity in baseline_ids - dataset_ids]
        )
        raise ValueError(
            f"{baseline_path}: baseline and dataset IDs differ, missing={missing}, extra={extra}"
        )

    rows: list[SeedRow] = []
    for index, row in enumerate(baseline_rows):
        identity = _question_identity(row["id"])
        dataset_question = dataset_questions[identity]
        if row["question"] != dataset_question:
            raise ValueError(
                f"{baseline_path}: row {index} question does not match dataset id {row['id']!r}"
            )
        rows.append(
            SeedRow(
                id=row["id"],
                question=dataset_question,
                contexts=row["contexts"],
                seed_dieu=set(row["seed_dieu"]),
            )
        )

    config = baseline_data["config"]
    baseline_hash = _file_hash(baseline_path)
    dataset_hash = _file_hash(dataset_path)
    whitelist_hash = _file_hash(whitelist_path)
    snapshot = SeedSnapshot(
        schema_version=SCHEMA_VERSION,
        snapshot_id=_snapshot_id(
            baseline_hash=baseline_hash,
            dataset_hash=dataset_hash,
            whitelist_hash=whitelist_hash,
            rows=rows,
            internal_dieu=internal_dieu,
        ),
        dataset_id=dataset_path.stem,
        baseline_source={"path": str(baseline_path), "sha256": baseline_hash},
        dataset_source={"path": str(dataset_path), "sha256": dataset_hash},
        whitelist_source={"path": str(whitelist_path), "sha256": whitelist_hash},
        retrieval_config=dict(config),
        unavailable_metadata=_unavailable_metadata(baseline_data, config),
        internal_dieu=internal_dieu,
        rows=rows,
    )
    return snapshot


def normalize_seed(row: SeedRow) -> dict[int, str | None]:
    """Normalize seed contexts with the same membership semantics as the loop."""
    from src.rag.agent.tools import dieu_from_title

    collected: dict[int, str | None] = {}
    for index, article in enumerate(row.contexts):
        dieu = dieu_from_title(article)
        if dieu > 0 and dieu not in collected:
            collected[dieu] = article
        else:
            collected[-(index + 1)] = article
    for dieu in row.seed_dieu:
        if dieu > 0 and dieu not in collected:
            collected[dieu] = None
    return collected


def _observation_hash(observation: str) -> str:
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def _case_id(dataset_id: str, question_id: Any) -> str:
    token = hashlib.sha256(_canonical_json(question_id)).hexdigest()[:12]
    return f"{dataset_id}:hop0:q-{token}"


def prepare_cases(snapshot: SeedSnapshot, *, snapshot_hash: str) -> list[GateCase]:
    """Derive hop-0 gate observations and draft labels from a seed snapshot."""
    # Nạp helper khi dựng case để import seed không kéo theo module truy cập store
    from src.rag.agent.loop import _build_frontier, _build_gate_observation
    from src.rag.agent.tools import extract_citation_mentions

    if not isinstance(snapshot_hash, str) or not snapshot_hash.strip():
        raise ValueError("snapshot_hash must be a non-empty string")

    cases: list[GateCase] = []
    for row in snapshot.rows:
        collected = normalize_seed(row)
        candidates, mentions_by_source = _build_frontier(
            collected,
            extract_citations_fn=extract_citation_mentions,
            internal_dieu=snapshot.internal_dieu,
        )
        observation = _build_gate_observation(collected, mentions_by_source)
        cases.append(
            GateCase(
                case_id=_case_id(snapshot.dataset_id, row.id),
                dataset_id=snapshot.dataset_id,
                question_id=row.id,
                snapshot_id=snapshot.snapshot_id,
                snapshot_hash=snapshot_hash,
                question=row.question,
                observation=observation,
                observation_hash=_observation_hash(observation),
                candidates=candidates,
                source_hop=0,
                source_run_id=None,
                source_policy=None,
                expected_action="unresolved" if candidates else "no_candidates",
                acceptable_dieu=set(),
                label_reason=None,
                label_status="draft",
                split="dev",
                fewshot_overlap=None,
            )
        )
    return cases


def _rebuild_case_state(
    snapshot: SeedSnapshot,
    row: SeedRow,
) -> tuple[str, list[int]]:
    from src.rag.agent.loop import _build_frontier, _build_gate_observation
    from src.rag.agent.tools import extract_citation_mentions

    collected = normalize_seed(row)
    candidates, mentions_by_source = _build_frontier(
        collected,
        extract_citations_fn=extract_citation_mentions,
        internal_dieu=snapshot.internal_dieu,
    )
    observation = _build_gate_observation(collected, mentions_by_source)
    return observation, candidates


def validate_replay(
    snapshot: SeedSnapshot,
    cases: list[GateCase],
    *,
    snapshot_hash: str,
) -> ReplaySelection:
    """Recompute case inputs and separate executable labels from exclusions."""
    if not isinstance(snapshot_hash, str) or not snapshot_hash.strip():
        raise ValueError("snapshot_hash must be a non-empty string")

    rows = {_question_identity(row.id): row for row in snapshot.rows}
    expected_case_ids = {_case_id(snapshot.dataset_id, row.id) for row in snapshot.rows}
    seen_case_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)
    if seen_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - seen_case_ids)
        extra = sorted(seen_case_ids - expected_case_ids)
        raise ValueError(
            f"case roster does not match snapshot rows, missing={missing}, extra={extra}"
        )

    eligible: list[GateCase] = []
    exclusions: list[Exclusion] = []

    for case in cases:
        if case.dataset_id != snapshot.dataset_id:
            raise ValueError(f"{case.case_id}: dataset_id does not match snapshot")
        if case.snapshot_id != snapshot.snapshot_id:
            raise ValueError(f"{case.case_id}: snapshot_id does not match snapshot")
        if case.snapshot_hash != snapshot_hash:
            raise ValueError(f"{case.case_id}: snapshot hash is stale")

        identity = _question_identity(case.question_id)
        row = rows.get(identity)
        if row is None:
            raise ValueError(f"{case.case_id}: question_id is absent from snapshot")
        expected_case_id = _case_id(snapshot.dataset_id, row.id)
        if case.case_id != expected_case_id:
            raise ValueError(
                f"{case.case_id}: case_id does not match the canonical hop-0 identity {expected_case_id}"
            )
        if case.source_hop != 0 or case.source_run_id is not None or case.source_policy is not None:
            raise ValueError(f"{case.case_id}: only hop-0 cases are supported")
        if case.question != row.question:
            raise ValueError(f"{case.case_id}: question does not match snapshot")

        observation, candidates = _rebuild_case_state(snapshot, row)
        if case.observation != observation:
            raise ValueError(f"{case.case_id}: observation is stale or mismatched")
        if case.observation_hash != _observation_hash(observation):
            raise ValueError(f"{case.case_id}: observation_hash is invalid")
        if case.candidates != candidates:
            raise ValueError(f"{case.case_id}: candidate order is stale or mismatched")

        if case.expected_action == "no_candidates":
            exclusions.append(Exclusion(case_id=case.case_id, reason="no_candidates"))
        elif case.label_status != "approved":
            exclusions.append(Exclusion(case_id=case.case_id, reason="draft"))
        elif case.expected_action == "unresolved":
            exclusions.append(Exclusion(case_id=case.case_id, reason="unresolved"))
        else:
            eligible.append(case)

    return ReplaySelection(eligible_cases=eligible, exclusions=exclusions)


def policy_input(case: GateCase) -> GateInput:
    """Project a case to the identical label-free input for both policies."""
    if not case.candidates:
        raise ValueError(f"{case.case_id}: policy input requires non-empty candidates")
    return GateInput(
        question=case.question,
        observation=case.observation,
        candidates=list(case.candidates),
    )


def write_cases(cases: list[GateCase], output_path: Path) -> None:
    """Write ordered gate cases as one validated JSON object per line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for case in cases:
            target.write(json.dumps(case.model_dump(mode="json"), ensure_ascii=False))
            target.write("\n")
        target.flush()
    temporary.replace(output_path)


def read_cases(path: Path) -> list[GateCase]:
    """Read and validate an ordered JSONL case dataset."""
    path = Path(path)
    cases: list[GateCase] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{path}: blank line at {line_number}")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}: invalid JSON at line {line_number}") from exc
                try:
                    cases.append(GateCase.model_validate(payload))
                except ValueError as exc:
                    raise ValueError(f"{path}: invalid case at line {line_number}: {exc}") from exc
    except OSError:
        raise
    return cases


__all__ = [
    "import_seeds",
    "normalize_seed",
    "policy_input",
    "prepare_cases",
    "read_cases",
    "validate_replay",
    "write_cases",
]
