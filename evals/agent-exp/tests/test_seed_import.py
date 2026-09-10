"""Tests import seed: round-trip, từ chối dữ liệu sai, và isolation khỏi dotenv/model."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from artifacts import read_snapshot, sha256_file, write_snapshot
from seed_cases import import_seeds


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"id": 1, "user_input": "Câu hỏi một"},
                {"id": "q2", "user_input": "Câu hỏi hai"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "config": {"agent": False, "rerank_ratio": 0.45},
                "agent_trajectory": None,
                "results": [
                    {
                        "id": 1,
                        "query": "Câu hỏi một",
                        "retrieved_contexts": [
                            "Điều 10. A\nTheo Điều 20",
                            "Điều 11. B",
                        ],
                        "hop_scores": {"retrieved_dieu": [10, 11]},
                        "agent_trace": None,
                    },
                    {
                        "id": "q2",
                        "query": "Câu hỏi hai",
                        "retrieved_contexts": [],
                        "hop_scores": {"retrieved_dieu": []},
                        "agent_trace": None,
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    whitelist_path = tmp_path / "internal.json"
    whitelist_path.write_text("[10, 11, 20]", encoding="utf-8")
    return baseline_path, dataset_path, whitelist_path


def test_import_and_round_trip_preserve_order_and_membership(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    snapshot = import_seeds(baseline, dataset, whitelist)

    assert [row.id for row in snapshot.rows] == [1, "q2"]
    assert snapshot.rows[0].contexts == ["Điều 10. A\nTheo Điều 20", "Điều 11. B"]
    assert snapshot.rows[0].seed_dieu == {10, 11}
    assert snapshot.rows[1].contexts == []
    assert snapshot.rows[1].seed_dieu == set()
    assert snapshot.internal_dieu == {10, 11, 20}

    output = tmp_path / "seeds.json"
    write_snapshot(snapshot, output)
    loaded = read_snapshot(output)
    assert loaded.model_dump(mode="json") == snapshot.model_dump(mode="json")
    assert sha256_file(output) == sha256_file(output)


def test_agent_final_source_is_rejected(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["config"]["agent"] = True
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="single-pass"):
        import_seeds(baseline, dataset, whitelist)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda payload: payload["results"].append(payload["results"][0]), "duplicate baseline id"),
        (lambda payload: payload["results"][0].update(retrieved_contexts=[None]), "retrieved_contexts"),
        (lambda payload: payload["results"][0]["hop_scores"].update(retrieved_dieu=[10, 10]), "duplicates article id"),
    ],
)
def test_malformed_baseline_rows_are_rejected(tmp_path, change, message):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    change(payload)
    baseline.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        import_seeds(baseline, dataset, whitelist)


def test_duplicate_whitelist_is_rejected(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    whitelist.write_text("[10, 10]", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates article id"):
        import_seeds(baseline, dataset, whitelist)


def test_dataset_alignment_is_required(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload[0]["user_input"] = "Câu hỏi khác"
    dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match dataset"):
        import_seeds(baseline, dataset, whitelist)


def test_snapshot_id_is_deterministic_for_unchanged_inputs(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    first = import_seeds(baseline, dataset, whitelist)
    second = import_seeds(baseline, dataset, whitelist)
    assert first.snapshot_id == second.snapshot_id


def test_context_order_and_membership_ids_remain_independent(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["results"][0]["retrieved_contexts"] = [
        "Điều 10. First",
        "Điều 11. Second",
        "Điều 12. Third",
    ]
    payload["results"][0]["hop_scores"]["retrieved_dieu"] = [10]
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Cố tình để thứ tự và số lượng contexts khác tập IDs để phát hiện việc ghép bằng zip
    snapshot = import_seeds(baseline, dataset, whitelist)
    row = snapshot.rows[0]
    assert row.contexts == ["Điều 10. First", "Điều 11. Second", "Điều 12. Third"]
    assert row.seed_dieu == {10}


@pytest.mark.parametrize("bad_id", [0, -1, True, 2.5, "3"])
def test_seed_article_ids_require_positive_strict_integers(tmp_path, bad_id):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["results"][0]["hop_scores"]["retrieved_dieu"] = [bad_id]
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # Bool là lớp con của int trong Python nhưng không phải article ID hợp lệ
    with pytest.raises(ValueError, match="positive integer"):
        import_seeds(baseline, dataset, whitelist)


@pytest.mark.parametrize("whitelist_payload", ["{\"10\": true}", "[10, 10]", "[10, 0]"])
def test_whitelist_must_be_list_unique_and_positive(tmp_path, whitelist_payload):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    whitelist.write_text(whitelist_payload, encoding="utf-8")
    with pytest.raises(ValueError):
        import_seeds(baseline, dataset, whitelist)


def test_dataset_roster_mismatch_is_rejected(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(dataset.read_text(encoding="utf-8"))
    payload.pop()
    dataset.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="IDs differ"):
        import_seeds(baseline, dataset, whitelist)


def test_agent_trace_and_agent_config_are_not_seed_provenance(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["results"][0]["agent_trace"] = {"hop": 0}
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    # Context cuối của agent không phải seed single-pass dù vẫn có đủ text và IDs
    with pytest.raises(ValueError, match="agent_trace"):
        import_seeds(baseline, dataset, whitelist)

    payload["results"][0]["agent_trace"] = None
    payload["config"]["agent"] = True
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="single-pass"):
        import_seeds(baseline, dataset, whitelist)

    payload["config"] = {"agent": False}
    payload.pop("config")
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="single-pass"):
        import_seeds(baseline, dataset, whitelist)


def test_empty_retrieval_is_valid_snapshot(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    payload = json.loads(baseline.read_text(encoding="utf-8"))
    payload["results"][0]["retrieved_contexts"] = []
    payload["results"][0]["hop_scores"]["retrieved_dieu"] = []
    baseline.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    snapshot = import_seeds(baseline, dataset, whitelist)
    assert snapshot.rows[0].contexts == []
    assert snapshot.rows[0].seed_dieu == set()


def test_unavailable_source_metadata_records_reasons(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    snapshot = import_seeds(baseline, dataset, whitelist)
    assert set(snapshot.unavailable_metadata) == {
        "source_revision",
        "retrieval_model",
        "embedding_model",
        "reranker_model",
    }
    assert all(snapshot.unavailable_metadata.values())


def test_source_hashes_match_source_bytes(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    snapshot = import_seeds(baseline, dataset, whitelist)
    # Đối chiếu hash với bytes của nguồn để phát hiện provenance bị ghi sai
    assert snapshot.baseline_source.sha256 == hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert snapshot.dataset_source.sha256 == hashlib.sha256(dataset.read_bytes()).hexdigest()
    assert snapshot.whitelist_source.sha256 == hashlib.sha256(whitelist.read_bytes()).hexdigest()


def test_import_modules_stay_isolated_in_fresh_process(tmp_path):
    baseline, dataset, whitelist = _write_inputs(tmp_path)
    output = tmp_path / "snapshot.json"
    code = (
        "import json, sys;"
        "sys.path.insert(0, 'evals/agent-exp/scripts'); sys.path.insert(0, '.');"
        "import run_experiments as cli;"
        "rc=cli.main(['import-seeds','--baseline',sys.argv[1],'--dataset',sys.argv[2],"
        "'--internal-dieu',sys.argv[3],'--output',sys.argv[4]]);"
        "print(json.dumps({'rc':rc,'llm':'policies.llm' in sys.modules,'dotenv':'dotenv' in sys.modules," 
        "'model':'sentence_transformers' in sys.modules,'db':'chromadb' in sys.modules}))"
    )
    # Dùng process mới để module đã nạp từ pytest không che lỗi isolation
    completed = subprocess.run(
        [sys.executable, "-c", code, str(baseline), str(dataset), str(whitelist), str(output)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ".", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {
        "rc": 0,
        "llm": False,
        "dotenv": False,
        "model": False,
        "db": False,
    }
