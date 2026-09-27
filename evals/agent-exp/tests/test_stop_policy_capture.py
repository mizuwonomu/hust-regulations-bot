"""Kiểm tra capture bundle STOP-policy bằng runtime giả, không gọi retrieval thật"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.common.single_pass_retrieval import SinglePassSettings
from seed_capture import CAPTURE_SOURCE_PATHS

from artifacts import read_snapshot
from stop_policy_synthetic import (
    CORPUS_ANSWER_MARKER,
    CORPUS_GOLD_MARKER,
    CORPUS_LINK_MARKER,
    QUESTION_ONE,
    QUESTION_SIX,
    capture_bundle,
    default_corpus_rows,
    write_corpus,
)

from stop_policy_eval.capture import (
    capture_corpus,
    prepare_draft_cases,
    publish_bundle,
    read_capture_manifest,
    snapshot_content_hash,
)
from stop_policy_eval.cases import hop0_case_id, read_stop_corpus
from stop_policy_eval.contracts import STOP_POLICY_SCHEMA_VERSION
from diagnostic_subexp.shared.contracts import sha256_bytes, sha256_text

EXPECTED_QUESTIONS = [row["user_input"] for row in default_corpus_rows()]
REPO_ROOT = Path(__file__).resolve().parents[3]

STOP_CAPTURE_PROVENANCE_PATHS = {
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
}


class _RaisingRuntime:
    """Runtime giả luôn ném lỗi đã chỉ định khi được gọi retrieval"""

    metadata = {"retrieval": "fake"}

    def __init__(self, error: Exception):
        self.error = error

    def retrieve(self, question: str):
        raise self.error


def _tmp_siblings(path: Path) -> list[str]:
    return sorted(item.name for item in path.parent.iterdir() if ".tmp" in item.name)


def test_only_id_and_user_input_reach_the_runtime(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    assert fixture.runtime.questions == EXPECTED_QUESTIONS
    corpus = read_stop_corpus(fixture.corpus_path)
    assert [row.question for row in corpus.rows] == fixture.runtime.questions
    assert CORPUS_ANSWER_MARKER not in "".join(fixture.runtime.questions)


def test_corpus_annotations_change_provenance_not_retrieval_or_labels(tmp_path: Path):
    fixture = capture_bundle(tmp_path / "base")
    rows = json.loads(fixture.corpus_path.read_text(encoding="utf-8"))
    for row in rows:
        row["response"] = f"{CORPUS_ANSWER_MARKER} đã đổi"
        row["retrieved_contexts"] = f"{CORPUS_GOLD_MARKER} đã đổi"
        row["link"] = f"{CORPUS_LINK_MARKER} đã đổi"
        row["group"] = "Other"
        row["type"] = "other"
    annotated = capture_bundle(tmp_path / "annotated", rows=rows)

    assert annotated.runtime.questions == fixture.runtime.questions
    label_fields = (
        "case_id",
        "question_id",
        "question",
        "observation",
        "observation_hash",
        "candidates",
        "expected_action",
        "acceptable_dieu",
        "label_status",
        "label_observation_hash",
        "semantic_group_id",
        "split",
        "fewshot_overlap",
    )
    before = [case.model_dump(mode="json") for case in fixture.drafts]
    after = [case.model_dump(mode="json") for case in annotated.drafts]
    for left, right in zip(before, after):
        assert {name: left[name] for name in label_fields} == {
            name: right[name] for name in label_fields
        }
        assert left["source_row_hash"] != right["source_row_hash"]
        assert left["source_dataset_hash"] != right["source_dataset_hash"]
    assert (
        fixture.outcome.manifest.dataset_source.sha256
        != annotated.outcome.manifest.dataset_source.sha256
    )


def test_an_invalid_corpus_is_rejected_before_runtime_construction(tmp_path: Path):
    rows = json.loads(write_corpus(tmp_path)[0].read_text(encoding="utf-8"))
    del rows[0]["user_input"]
    with pytest.raises(ValueError, match="user_input"):
        capture_bundle(tmp_path / "broken-input", rows=rows)


def test_invalid_inventory_is_rejected_before_runtime_construction(tmp_path: Path):
    corpus_path, inventory_path = write_corpus(tmp_path / "empty", whitelist=[])
    calls: list[bool] = []

    def factory(settings):
        calls.append(True)
        raise AssertionError("the runtime must not be constructed for an invalid inventory")

    with pytest.raises(ValueError, match="must not be empty"):
        capture_corpus(
            corpus_path,
            inventory_path,
            output_dir=tmp_path / "empty-capture",
            settings=SinglePassSettings(),
            runtime_factory=factory,
        )
    assert calls == []
    assert not (tmp_path / "empty-capture").exists()


def test_one_retrieval_call_per_row_in_input_order(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    assert fixture.factory.calls == 1
    assert len(fixture.runtime.questions) == 6
    assert len(set(fixture.runtime.questions)) == 6
    assert [case.question for case in fixture.drafts] == fixture.runtime.questions


def test_hop0_state_preserves_canonical_order_and_observation_bytes(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    snapshot = fixture.outcome.snapshot
    snapshot_hash = snapshot_content_hash(snapshot)
    expected_candidates = {1: [30], 2: [31], 3: [33], 4: [35], 5: [], 6: [34]}
    for draft in fixture.drafts:
        assert draft.case_id == hop0_case_id(snapshot.dataset_id, draft.question_id)
        assert draft.candidates == expected_candidates[draft.question_id]
        assert draft.question == snapshot.rows[draft.question_id - 1].question
        assert draft.observation_hash == sha256_text(draft.observation)
        assert draft.snapshot_hash == snapshot_hash

    assert fixture.case_for(1).observation == (
        "Các Điều đã thu thập:\n"
        "[Context 21] Điều 21. Điều kiện tốt nghiệp kỹ sư\n"
        "- Quy trình xét tốt nghiệp theo Điều 30 Quy chế này."
    )


def test_hop0_state_matches_canonical_rebuilder_in_isolated_process(tmp_path: Path):
    fixture = capture_bundle(tmp_path / "fixture")
    snapshot = fixture.outcome.snapshot
    row = snapshot.rows[0].model_copy(
        update={
            "contexts": [
                "Điều 10. Nguồn thứ nhất\r\n"
                "Dẫn Điều 20 và Điều 20; tự dẫn Điều 10.\r\n"
                "Dẫn Điều 21, Điều 30 và Điều 99.",
                "Tiêu đề không hợp lệ\r\nDẫn Điều 22 và Điều 21.",
                "Điều 10. Nguồn trùng\r\nDẫn Điều 21 và Điều 23.",
                "Cũng không hợp lệ\r\nDẫn Điều 24.",
            ],
            "seed_dieu": {30, 31},
        }
    )
    parity_snapshot = snapshot.model_copy(
        update={
            "internal_dieu": {10, 20, 21, 22, 23, 24, 25, 30, 31},
            "rows": [row],
        }
    )
    input_path = tmp_path / "parity-input.json"
    input_path.write_text(
        json.dumps(parity_snapshot.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )
    script = f"""
import builtins
import json
import socket
import sys

network_calls = []
dotenv_calls = []
def block_network(*args, **kwargs):
    network_calls.append(True)
    raise AssertionError("network access is forbidden")
def block_dotenv(*args, **kwargs):
    dotenv_calls.append(True)
    raise AssertionError("dotenv loading is forbidden")
socket.socket.connect = block_network
socket.create_connection = block_network
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    module = original_import(name, globals, locals, fromlist, level)
    if name == "dotenv" or name.startswith("dotenv."):
        for module_name in ("dotenv", "dotenv.main"):
            dotenv_module = sys.modules.get(module_name)
            if dotenv_module is not None:
                dotenv_module.load_dotenv = block_dotenv
    return module
builtins.__import__ = guarded_import
sys.path.insert(0, {str(REPO_ROOT / "evals/agent-exp/scripts")!r})
from contracts import SeedSnapshot
from seed_cases import normalize_seed, rebuild_case_state
from stop_policy_eval import state
from src.rag.agent.loop import _build_frontier as production_frontier
from src.rag.agent.loop import _build_gate_observation
from src.rag.agent.tools import extract_citation_mentions

snapshot = SeedSnapshot.model_validate(json.load(open({str(input_path)!r}, encoding="utf-8")))
row = snapshot.rows[0]
local_observation, local_candidates = state.rebuild_case_state(snapshot, row)
canonical_observation, canonical_candidates = rebuild_case_state(snapshot, row)
local_keys = state._normalize_seed(row)
canonical_keys = normalize_seed(row)
followed_article = "Điều 20. Điều mới sau follow\\r\\nCăn cứ Điều 25."
local_collected = state._normalize_seed(row)
canonical_collected = normalize_seed(row)
local_collected[20] = followed_article
canonical_collected[20] = followed_article
local_next_observation, local_next_candidates = state.rebuild_collected_state(
    local_collected, snapshot.internal_dieu
)
canonical_next_candidates, mentions = production_frontier(
    canonical_collected, extract_citations_fn=extract_citation_mentions,
    internal_dieu=snapshot.internal_dieu,
)
canonical_next_observation = _build_gate_observation(canonical_collected, mentions)
print(json.dumps({{
    "local": {{
        "observation_hex": local_observation.encode("utf-8").hex(),
        "candidates": local_candidates,
        "negative_keys": sorted(key for key in local_keys if key < 0),
    }},
    "canonical": {{
        "observation_hex": canonical_observation.encode("utf-8").hex(),
        "candidates": canonical_candidates,
        "negative_keys": sorted(key for key in canonical_keys if key < 0),
    }},
    "local_next": {{
        "observation_hex": local_next_observation.encode("utf-8").hex(),
        "candidates": local_next_candidates,
    }},
    "canonical_next": {{
        "observation_hex": canonical_next_observation.encode("utf-8").hex(),
        "candidates": canonical_next_candidates,
    }},
    "network_calls": len(network_calls),
    "dotenv_calls": len(dotenv_calls),
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    local = payload["local"]
    canonical = payload["canonical"]

    assert local["observation_hex"] == canonical["observation_hex"]
    assert bytes.fromhex(local["observation_hex"]).find(b"\r") == -1
    assert local["candidates"] == canonical["candidates"] == [20, 21, 22, 23, 24]
    assert local["candidates"].count(20) == 1
    assert local["negative_keys"] == canonical["negative_keys"] == [-4, -3, -2]
    assert b"[Context ?]" in bytes.fromhex(local["observation_hex"])
    assert b"[Context 30]" not in bytes.fromhex(local["observation_hex"])
    assert payload["local_next"] == payload["canonical_next"]
    assert payload["local_next"]["candidates"] == [21, 22, 23, 24, 25]
    assert payload["network_calls"] == 0
    assert payload["dotenv_calls"] == 0


def test_empty_frontiers_are_mechanical_exclusions_not_stop(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    empty = fixture.case_for(5)
    assert empty.candidates == []
    assert empty.expected_action == "no_candidates"
    assert empty.expected_action != "stop"
    assert empty.label_status == "draft"
    assert empty.acceptable_dieu == []


def test_nonempty_frontiers_stay_unreviewed_drafts(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    case = fixture.case_for(1)
    assert case.candidates == [30]
    assert case.expected_action == "unresolved"
    assert case.label_status == "draft"
    assert case.split == "unassigned"
    assert case.semantic_group_id is None
    assert case.fewshot_overlap is None
    assert case.label_observation_hash is None
    assert case.label_reason is None
    for field in (
        "source_run_id",
        "source_policy",
        "source_trial_id",
        "parent_case_id",
        "followed_dieu",
        "source_state_hash",
    ):
        assert getattr(case, field) is None


def test_draft_and_review_case_files_are_byte_identical(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    capture_dir = fixture.capture_dir
    assert (capture_dir / "cases_draft.jsonl").read_bytes() == (
        capture_dir / "cases_review.jsonl"
    ).read_bytes()
    lines = (capture_dir / "cases_draft.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    assert [json.loads(line)["question_id"] for line in lines] == [1, 2, 3, 4, 5, 6]


def test_review_markdown_exposes_exact_gate_inputs_only(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    text = (fixture.capture_dir / "label_review.md").read_text(encoding="utf-8")
    case = fixture.case_for(1)
    assert case.observation in text
    assert case.observation_hash in text
    assert "Điều 30" in text
    assert case.case_id in text
    assert case.source_row_hash in text
    for forbidden in (CORPUS_ANSWER_MARKER, CORPUS_GOLD_MARKER, CORPUS_LINK_MARKER):
        assert forbidden not in text
    assert "expected_action" in text and "semantic_group_id" in text and "fewshot_overlap" in text
    assert "no_candidates" in text


def test_existing_destination_fails_before_the_runtime_and_keeps_bytes(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_bytes(b"keep me")
    before = {item.name: item.read_bytes() for item in existing.iterdir()}
    calls: list[bool] = []

    def factory(settings):
        calls.append(True)
        return fixture.runtime

    with pytest.raises(ValueError, match="destination already exists"):
        capture_corpus(
            fixture.corpus_path,
            fixture.inventory_path,
            output_dir=existing,
            settings=fixture.settings,
            runtime_factory=factory,
        )
    assert calls == []
    assert {item.name: item.read_bytes() for item in existing.iterdir()} == before


def test_competing_empty_destination_survives_atomic_publish_race(tmp_path: Path, monkeypatch):
    import stop_policy_eval.capture as capture_module

    destination = tmp_path / "raced-capture"
    original_fsync = capture_module._fsync_directory

    def create_competitor_before_publish(path: Path) -> None:
        destination.mkdir()
        original_fsync(path)

    monkeypatch.setattr(capture_module, "_fsync_directory", create_competitor_before_publish)

    with pytest.raises(ValueError, match="destination already exists"):
        publish_bundle(destination, {"capture.json": b"complete"})

    assert list(destination.iterdir()) == []
    assert _tmp_siblings(destination) == []


def test_publish_fails_closed_when_atomic_no_replace_is_unavailable(
    tmp_path: Path, monkeypatch
):
    import stop_policy_eval.capture as capture_module

    destination = tmp_path / "unsupported-publish"

    def unavailable(source: Path, target: Path) -> None:
        raise OSError("atomic no-replace directory publication is unavailable")

    monkeypatch.setattr(capture_module, "_rename_directory_no_replace", unavailable)

    with pytest.raises(OSError, match="atomic no-replace"):
        publish_bundle(destination, {"capture.json": b"complete"})

    assert not destination.exists()
    assert _tmp_siblings(destination) == []


def test_runtime_error_and_interrupt_publish_no_destination(tmp_path: Path):
    for error in (RuntimeError("retrieval down"), KeyboardInterrupt()):
        name = f"failing-{type(error).__name__}"
        corpus_path, inventory_path = write_corpus(tmp_path / f"{name}-input", name="corpus")
        destination = tmp_path / name
        with pytest.raises(type(error)):
            capture_corpus(
                corpus_path,
                inventory_path,
                output_dir=destination,
                settings=SinglePassSettings(),
                runtime_factory=lambda settings, error=error: _RaisingRuntime(error),
            )
        assert not destination.exists()
        assert _tmp_siblings(destination) == []


def test_identical_retrieval_yields_same_snapshot_identity_but_new_capture_identity(tmp_path: Path):
    first = capture_bundle(tmp_path / "first")
    second = capture_bundle(tmp_path / "second")
    assert first.outcome.snapshot.snapshot_id == second.outcome.snapshot.snapshot_id
    assert first.outcome.manifest.capture_id != second.outcome.manifest.capture_id
    assert first.drafts[0].source_dataset_hash == second.drafts[0].source_dataset_hash
    assert first.drafts[0].source_row_hash == second.drafts[0].source_row_hash


def test_capture_manifest_binds_sources_and_generated_files(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    manifest = fixture.outcome.manifest
    assert manifest.schema_version == STOP_POLICY_SCHEMA_VERSION
    assert manifest.capture_status == "captured"
    assert manifest.row_count == 6
    assert manifest.snapshot_id == fixture.outcome.snapshot.snapshot_id
    assert manifest.snapshot_hash == snapshot_content_hash(fixture.outcome.snapshot)
    assert manifest.dataset_source.sha256 == read_stop_corpus(fixture.corpus_path).sha256
    assert manifest.inventory_source.sha256 == sha256_bytes(fixture.inventory_path.read_bytes())
    assert manifest.captured_source_hashes
    adapter = Path(__file__).resolve().parents[3] / "evals/common/single_pass_retrieval.py"
    assert manifest.captured_source_hashes["evals/common/single_pass_retrieval.py"] == (
        sha256_bytes(adapter.read_bytes())
    )
    expected_sources = set(CAPTURE_SOURCE_PATHS) | STOP_CAPTURE_PROVENANCE_PATHS
    assert set(manifest.captured_source_hashes) == expected_sources
    for relative, digest in manifest.captured_source_hashes.items():
        source = REPO_ROOT / relative
        assert source.is_file(), relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, relative
    assert manifest.runtime_metadata == {"retrieval": "fake"}
    assert manifest.retrieval_config.get("rerank_ratio") is not None
    for name, digest in manifest.generated_files.items():
        assert digest == sha256_bytes((fixture.capture_dir / name).read_bytes()), name
    assert manifest.captured_at and manifest.python_version
    if manifest.source_revision is None:
        assert "source_revision" in manifest.unavailable_metadata


def test_inspect_reads_the_bundle_without_the_original_inputs(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    fixture.corpus_path.unlink()
    fixture.inventory_path.unlink()
    manifest = read_capture_manifest(fixture.capture_dir)
    assert manifest == fixture.outcome.manifest
    snapshot = read_snapshot(fixture.capture_dir / "seeds.json")
    assert snapshot.snapshot_id == fixture.outcome.snapshot.snapshot_id
    assert snapshot_content_hash(snapshot) == manifest.snapshot_hash


def test_prepare_draft_cases_rejects_a_corpus_that_changed(tmp_path: Path):
    fixture = capture_bundle(tmp_path)
    rows = json.loads(fixture.corpus_path.read_text(encoding="utf-8"))
    rows[0]["user_input"] = f"{QUESTION_ONE} đã đổi"
    with pytest.raises(ValueError):
        prepare_draft_cases(fixture.outcome.snapshot, json.dumps(rows).encode("utf-8"))
    assert QUESTION_SIX in (fixture.capture_dir / "label_review.md").read_text(encoding="utf-8")
