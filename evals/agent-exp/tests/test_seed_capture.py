"""Offline contract tests cho direct capture seed và command capture-seeds.

Chỉ dùng fake runtime: không network, không store, không model, không sleep.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import evals.common.single_pass_retrieval as spr
import run_experiments as cli
import seed_capture
from artifacts import read_snapshot
from contracts import GateCase
from evals.common.single_pass_retrieval import QueryExpansion, SinglePassSettings
from run_experiments import main
from seed_capture import capture_seeds, publish_captured_snapshot
from seed_cases import read_cases, write_cases

BREADTH_QUESTION = "Quy trình đăng ký học tập của hệ kỹ sư thế nào?"
SEQUENTIAL_QUESTION = "Điều kiện bảo vệ luận án tiến sĩ?"
EMPTY_QUESTION = "Câu hỏi không có kết quả?"

CONTEXT_21 = (
    "Điều 21. Đăng ký học tập chương trình kỹ sư\n"
    "Quy trình đăng ký học tập thực hiện theo Điều 10 của Quy chế này."
)
CONTEXT_42 = (
    "Điều 42. Đánh giá luận án tiến sĩ\n"
    "Điều kiện bảo vệ cấp cơ sở quy định tại Điều 41 và cấp Đại học tại Điều 40 của Quy chế này."
)
WHITELIST = [10, 21, 40, 41, 42]

DEFAULT_ANSWERS = {
    BREADTH_QUESTION: ([CONTEXT_21], {21}),
    SEQUENTIAL_QUESTION: ([CONTEXT_42], {42}),
    EMPTY_QUESTION: ([], set()),
}

FRESH_PROCESS_CODE = r'''
import importlib.abc, json, sys

FORBIDDEN = {
    "ragas", "dotenv", "chromadb", "sentence_transformers",
    "langchain_groq", "torch",
}


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ImportError("forbidden: " + fullname)
        return None


sys.meta_path.insert(0, Guard())
sys.path.insert(0, "evals/agent-exp/scripts")
sys.path.insert(0, ".")

import run_experiments as cli
from evals.common.single_pass_retrieval import SinglePassSettings
from seed_capture import capture_seeds, publish_captured_snapshot

help_rc = None
try:
    cli.main(["capture-seeds", "--help"])
except SystemExit as exc:
    help_rc = exc.code


class Runtime:
    metadata = {"runtime": "fake"}

    def retrieve(self, question):
        return (["\u0110i\u1ec1u 21. A\nTheo \u0110i\u1ec1u 10"], {21})


snapshot = capture_seeds(
    sys.argv[1],
    sys.argv[2],
    settings=SinglePassSettings(),
    runtime_factory=lambda settings: Runtime(),
)
publish_captured_snapshot(snapshot, sys.argv[3])
print(json.dumps({
    "help_rc": help_rc,
    "forbidden": sorted(m for m in FORBIDDEN if m in sys.modules),
}))
'''


class FakeRuntime:
    """Runtime giả ghi lại câu hỏi nhận được và trả kết quả đã chỉ định."""

    def __init__(
        self,
        answers: dict[str, tuple[list[str], set[int]]],
        *,
        metadata: dict | None = None,
        fail_on: str | None = None,
        interrupt_on: str | None = None,
    ):
        self.answers = answers
        self.metadata = metadata if metadata is not None else {"runtime": "fake"}
        self.fail_on = fail_on
        self.interrupt_on = interrupt_on
        self.questions: list[str] = []

    def retrieve(self, question: str) -> tuple[list[str], set[int]]:
        self.questions.append(question)
        if self.interrupt_on is not None and question == self.interrupt_on:
            raise KeyboardInterrupt
        if self.fail_on is not None and question == self.fail_on:
            raise RuntimeError("retrieval exploded")
        return self.answers[question]


class FakeFactory:
    """Factory giả đếm số lần runtime được dựng."""

    def __init__(self, runtime: FakeRuntime):
        self.runtime = runtime
        self.calls = 0
        self.settings: SinglePassSettings | None = None

    def __call__(self, settings: SinglePassSettings) -> FakeRuntime:
        self.calls += 1
        self.settings = settings
        return self.runtime


def _write_json(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _corpus_row(row_id, question: str) -> dict:
    """Dựng row corpus có annotation nhiễu mà capture không được đọc."""
    return {
        "id": row_id,
        "user_input": question,
        "response": "câu trả lời tham chiếu không được dùng",
        "retrieved_contexts": ["ngữ cảnh gold giả không được dùng"],
        "link": "999 -> 111",
        "group": "nhóm-không-tồn-tại",
        "type": "multi_hop",
    }


def _write_inputs(
    tmp_path: Path,
    *,
    rows: list[dict],
    whitelist: list | None = None,
    dataset_name: str = "corpus.json",
    whitelist_name: str = "internal.json",
) -> tuple[Path, Path]:
    dataset = _write_json(tmp_path / dataset_name, rows)
    whitelist_path = _write_json(
        tmp_path / whitelist_name, WHITELIST if whitelist is None else whitelist
    )
    return dataset, whitelist_path


def _capture(tmp_path: Path, *, rows=None, answers=None, **runtime_kwargs):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=rows if rows is not None else [_corpus_row(1, BREADTH_QUESTION)]
    )
    runtime = FakeRuntime(
        DEFAULT_ANSWERS if answers is None else answers, **runtime_kwargs
    )
    factory = FakeFactory(runtime)
    settings = SinglePassSettings(rerank_ratio=0.5)
    snapshot = capture_seeds(
        dataset, whitelist, settings=settings, runtime_factory=factory
    )
    return snapshot, dataset, whitelist, runtime, factory, settings


def test_capture_uses_only_id_and_question_not_corpus_annotations(tmp_path):
    rows = [
        _corpus_row(1, BREADTH_QUESTION),
        {**_corpus_row("q2", EMPTY_QUESTION), "link": "không phải arrow", "group": None},
    ]
    dataset, whitelist = _write_inputs(tmp_path, rows=rows)
    runtime = FakeRuntime(DEFAULT_ANSWERS)
    factory = FakeFactory(runtime)
    snapshot = capture_seeds(
        dataset,
        whitelist,
        settings=SinglePassSettings(rerank_ratio=0.5),
        runtime_factory=factory,
    )

    # Arrow, group, type, response và retrieved_contexts không vào runtime
    assert runtime.questions == [BREADTH_QUESTION, EMPTY_QUESTION]
    assert snapshot.source_kind == "direct_retrieval"
    assert snapshot.baseline_source is None
    assert snapshot.dataset_id == "corpus"
    assert [row.id for row in snapshot.rows] == [1, "q2"]
    assert snapshot.rows[0].contexts == [CONTEXT_21]
    assert snapshot.rows[0].seed_dieu == {21}
    # Kết quả rỗng là một row hợp lệ, không phải lỗi
    assert snapshot.rows[1].contexts == []
    assert snapshot.rows[1].seed_dieu == set()


def test_membership_only_ids_survive_without_zipping_to_sorted_ids(tmp_path):
    answers = {BREADTH_QUESTION: ([CONTEXT_21], {21, 99})}
    snapshot, *_ = _capture(tmp_path, answers=answers)
    row = snapshot.rows[0]
    assert row.contexts == [CONTEXT_21]
    # Membership độc lập: 99 không suy ra được từ context nào
    assert row.seed_dieu == {21, 99}


def test_overlapping_ids_between_two_corpora_stay_separate(tmp_path):
    breadth_dir = tmp_path / "breadth"
    sequential_dir = tmp_path / "sequential"
    breadth_dir.mkdir()
    sequential_dir.mkdir()
    breadth, whitelist_a = _write_inputs(
        breadth_dir, rows=[_corpus_row(1, BREADTH_QUESTION)], dataset_name="breadth.json"
    )
    sequential, whitelist_b = _write_inputs(
        sequential_dir,
        rows=[_corpus_row(1, SEQUENTIAL_QUESTION)],
        dataset_name="sequential.json",
    )

    runtime = FakeRuntime(DEFAULT_ANSWERS)
    factory = FakeFactory(runtime)
    settings = SinglePassSettings(rerank_ratio=0.5)
    first = capture_seeds(breadth, whitelist_a, settings=settings, runtime_factory=factory)
    second = capture_seeds(
        sequential, whitelist_b, settings=settings, runtime_factory=factory
    )

    assert first.dataset_id == "breadth"
    assert second.dataset_id == "sequential"
    assert first.snapshot_id != second.snapshot_id
    assert first.rows[0].id == second.rows[0].id == 1
    assert first.rows[0].question != second.rows[0].question


@pytest.mark.parametrize(
    ("payload", "whitelist"),
    [
        ({"not": "an array"}, WHITELIST),
        ([], WHITELIST),
        ([{"id": 1, "user_input": "a"}, {"id": 1, "user_input": "b"}], WHITELIST),
        ([{"user_input": "thiếu id"}], WHITELIST),
        ([{"id": True, "user_input": "bool id"}], WHITELIST),
        ([{"id": 1, "user_input": "   "}], WHITELIST),
        ([{"id": 1}], WHITELIST),
        ([{"id": 1, "user_input": "ok"}], [10, 10]),
        ([{"id": 1, "user_input": "ok"}], []),
    ],
)
def test_invalid_inputs_fail_before_runtime_construction(tmp_path, payload, whitelist):
    dataset, whitelist_path = _write_inputs(tmp_path, rows=[], whitelist=whitelist)
    _write_json(dataset, payload)
    runtime = FakeRuntime(DEFAULT_ANSWERS)
    factory = FakeFactory(runtime)

    with pytest.raises(ValueError):
        capture_seeds(
            dataset,
            whitelist_path,
            settings=SinglePassSettings(),
            runtime_factory=factory,
        )
    assert factory.calls == 0
    assert runtime.questions == []


def test_runtime_is_built_once_and_called_once_per_row_in_order(tmp_path):
    rows = [
        _corpus_row(2, SEQUENTIAL_QUESTION),
        _corpus_row(1, BREADTH_QUESTION),
        _corpus_row(3, EMPTY_QUESTION),
    ]
    snapshot, _, _, runtime, factory, _ = _capture(tmp_path, rows=rows)

    assert factory.calls == 1
    assert runtime.questions == [SEQUENTIAL_QUESTION, BREADTH_QUESTION, EMPTY_QUESTION]
    assert [row.id for row in snapshot.rows] == [2, 1, 3]


def test_runtime_failure_aborts_without_publishing_partial_snapshot(tmp_path):
    rows = [_corpus_row(1, BREADTH_QUESTION), _corpus_row(2, EMPTY_QUESTION)]
    dataset, whitelist = _write_inputs(tmp_path, rows=rows)
    runtime = FakeRuntime(DEFAULT_ANSWERS, fail_on=EMPTY_QUESTION)
    factory = FakeFactory(runtime)
    output = tmp_path / "snapshot.json"

    with pytest.raises(RuntimeError):
        capture_seeds(
            dataset,
            whitelist,
            settings=SinglePassSettings(),
            runtime_factory=factory,
        )
    assert not output.exists()
    assert runtime.questions == [BREADTH_QUESTION, EMPTY_QUESTION]


def test_provenance_is_truthful_and_content_identity_is_stable(tmp_path):
    snapshot, dataset, whitelist, _, _, settings = _capture(tmp_path)

    assert snapshot.schema_version == 2
    assert snapshot.source_kind == "direct_retrieval"
    assert snapshot.baseline_source is None
    assert snapshot.retrieval_config["rerank_ratio"] == 0.5
    assert snapshot.retrieval_config["retriever_k"] == settings.retriever_k
    assert snapshot.capture_metadata is not None
    assert snapshot.capture_metadata["capture_id"]
    assert snapshot.capture_metadata["captured_at"]
    assert snapshot.capture_metadata["source_hashes"]
    assert "store_content_fingerprint" in snapshot.unavailable_metadata
    assert snapshot.dataset_source.sha256 == cli.sha256_file(dataset)
    assert snapshot.whitelist_source.sha256 == cli.sha256_file(whitelist)

    # Cùng input và cùng config phải cho cùng snapshot_id dù run id/timestamp khác
    _, _, _, _, _, _ = _capture(tmp_path)
    second = capture_seeds(
        dataset,
        whitelist,
        settings=SinglePassSettings(rerank_ratio=0.5),
        runtime_factory=FakeFactory(FakeRuntime(DEFAULT_ANSWERS)),
    )
    assert second.snapshot_id == snapshot.snapshot_id
    assert second.capture_metadata["capture_id"] != snapshot.capture_metadata["capture_id"]

    # Ratio khác là một phần của content identity
    third = capture_seeds(
        dataset,
        whitelist,
        settings=SinglePassSettings(rerank_ratio=0.9),
        runtime_factory=FakeFactory(FakeRuntime(DEFAULT_ANSWERS)),
    )
    assert third.snapshot_id != snapshot.snapshot_id


def test_publish_refuses_overwrite_and_leaves_no_temporary_file(tmp_path):
    snapshot, *_ = _capture(tmp_path)
    output = tmp_path / "out" / "seeds.json"
    publish_captured_snapshot(snapshot, output)
    original = output.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        publish_captured_snapshot(snapshot, output)
    assert output.read_text(encoding="utf-8") == original
    assert [path.name for path in output.parent.iterdir()] == [output.name]


def test_published_snapshot_reloads_as_v2(tmp_path):
    snapshot, _, _, _, _, _ = _capture(tmp_path)
    output = tmp_path / "seeds.json"
    publish_captured_snapshot(snapshot, output)

    loaded = read_snapshot(output)
    assert loaded.schema_version == 2
    assert loaded.source_kind == "direct_retrieval"
    assert loaded.model_dump(mode="json") == snapshot.model_dump(mode="json")


def test_cli_rejects_existing_destination_before_runtime(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    output.write_text("đã tồn tại", encoding="utf-8")
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert factory.calls == 0
    assert output.read_text(encoding="utf-8") == "đã tồn tại"


def test_cli_capture_runtime_failure_is_not_a_success(monkeypatch, tmp_path):
    rows = [_corpus_row(1, BREADTH_QUESTION), _corpus_row(2, EMPTY_QUESTION)]
    dataset, whitelist = _write_inputs(tmp_path, rows=rows)
    output = tmp_path / "seeds.json"
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS, fail_on=EMPTY_QUESTION))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
        ]
    )
    assert code == 1
    assert not output.exists()
    assert [path.name for path in tmp_path.iterdir() if path.name.endswith(".tmp")] == []


def test_cli_capture_interrupt_returns_130_without_destination(monkeypatch, tmp_path):
    rows = [_corpus_row(1, BREADTH_QUESTION), _corpus_row(2, EMPTY_QUESTION)]
    dataset, whitelist = _write_inputs(tmp_path, rows=rows)
    output = tmp_path / "seeds.json"
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS, interrupt_on=EMPTY_QUESTION))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
        ]
    )
    assert code == 130
    assert not output.exists()


def test_cli_capture_records_ratio_in_retrieval_config(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
            "--ratio",
            "0.6",
        ]
    )
    assert code == 0
    snapshot = read_snapshot(output)
    assert snapshot.retrieval_config["rerank_ratio"] == 0.6
    assert factory.settings is not None
    assert factory.settings.rerank_ratio == 0.6


ENV_NAME = "HUST_CAPTURE_ENV_TEST"


def _write_env_file(tmp_path: Path, value: str = "loaded-from-file") -> Path:
    path = tmp_path / "runtime.env"
    path.write_text(f"{ENV_NAME}={value}\n", encoding="utf-8")
    return path


def test_cli_env_file_loads_after_validation_before_runtime(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    env_file = _write_env_file(tmp_path)
    monkeypatch.delenv(ENV_NAME, raising=False)
    observed: dict = {}

    class EnvAwareFactory(FakeFactory):
        def __call__(self, settings):
            observed["env_at_runtime"] = os.environ.get(ENV_NAME)
            return super().__call__(settings)

    factory = EnvAwareFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ]
    )
    assert code == 0
    # Env phải đã được nạp trước khi dependency live được dựng
    assert observed["env_at_runtime"] == "loaded-from-file"
    assert read_snapshot(output).source_kind == "direct_retrieval"


def test_cli_env_file_is_not_loaded_when_input_is_invalid(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(tmp_path, rows=[])
    output = tmp_path / "seeds.json"
    env_file = _write_env_file(tmp_path)
    monkeypatch.delenv(ENV_NAME, raising=False)
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ]
    )
    assert code == 2
    assert ENV_NAME not in os.environ
    assert factory.calls == 0
    assert not output.exists()


def test_cli_env_file_is_not_loaded_when_destination_exists(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    output.write_text("đã tồn tại", encoding="utf-8")
    env_file = _write_env_file(tmp_path)
    monkeypatch.delenv(ENV_NAME, raising=False)
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
            "--env-file",
            str(env_file),
        ]
    )
    assert code == 2
    assert ENV_NAME not in os.environ
    assert factory.calls == 0


def test_cli_missing_env_file_returns_data_error(monkeypatch, tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)

    code = main(
        [
            "capture-seeds",
            "--dataset",
            str(dataset),
            "--internal-dieu",
            str(whitelist),
            "--output",
            str(output),
            "--env-file",
            str(tmp_path / "missing.env"),
        ]
    )
    assert code == 2
    assert factory.calls == 0
    assert not output.exists()


def test_capture_pins_source_bytes_and_provenance_before_runtime(tmp_path, monkeypatch):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    original_dataset = dataset.read_bytes()
    original_whitelist = whitelist.read_bytes()
    order: dict[str, bool] = {}

    def fake_source_hashes():
        order["source_hashed"] = True
        return {"evals/common/single_pass_retrieval.py": "deadbeef"}

    monkeypatch.setattr(seed_capture, "capture_source_hashes", fake_source_hashes)

    class MutatingRuntime(FakeRuntime):
        def retrieve(self, question):
            # Đổi file ngay giữa capture để bắt lỗi hash lấy sau khi chạy xong
            dataset.write_text(
                json.dumps([_corpus_row(1, "Câu hỏi khác")], ensure_ascii=False),
                encoding="utf-8",
            )
            whitelist.write_text("[99]", encoding="utf-8")
            return super().retrieve(question)

    class OrderFactory(FakeFactory):
        def __call__(self, settings):
            order["source_hashed_before_runtime"] = order.get("source_hashed", False)
            return super().__call__(settings)

    runtime = MutatingRuntime(DEFAULT_ANSWERS)
    factory = OrderFactory(runtime)
    snapshot = capture_seeds(
        dataset, whitelist, settings=SinglePassSettings(), runtime_factory=factory
    )

    assert order["source_hashed_before_runtime"] is True
    # Hash phải khớp đúng bytes đã parse, không phải bytes mới trên đĩa
    assert snapshot.dataset_source.sha256 == hashlib.sha256(original_dataset).hexdigest()
    assert snapshot.whitelist_source.sha256 == hashlib.sha256(original_whitelist).hexdigest()
    assert snapshot.dataset_source.sha256 != cli.sha256_file(dataset)
    assert snapshot.whitelist_source.sha256 != cli.sha256_file(whitelist)
    assert snapshot.rows[0].question == BREADTH_QUESTION
    assert snapshot.internal_dieu == set(WHITELIST)
    assert snapshot.capture_metadata["source_hashes"] == {
        "evals/common/single_pass_retrieval.py": "deadbeef"
    }


def test_capture_runtime_metadata_comes_from_builder_config(tmp_path):
    from src.rag.config import EMBEDDING_MODEL, RERANKER_MODEL

    class _Chain:
        def invoke(self, payload):
            return QueryExpansion(reasoning="fake", queries=["q"])

    class _Ensemble:
        def map(self):
            return self

        def invoke(self, sub_queries):
            return [[] for _ in sub_queries]

    class _Store:
        def mget(self, ids):
            return []

    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )

    def factory(settings):
        # Dùng builder thật, chỉ thay dependency nặng bằng stub
        return spr.build_single_pass_runtime(
            settings,
            embedding_loader=lambda: "embeddings",
            reranker_loader=lambda: object(),
            retriever_builder=lambda **kwargs: (_Ensemble(), _Store(), "vector"),
            rewrite_chain_builder=lambda settings: _Chain(),
        )

    snapshot = capture_seeds(
        dataset,
        whitelist,
        settings=SinglePassSettings(rerank_ratio=0.5),
        runtime_factory=factory,
    )

    metadata = snapshot.capture_metadata["runtime_metadata"]
    assert metadata["embedding_model"] == EMBEDDING_MODEL
    assert metadata["reranker_model"] == RERANKER_MODEL
    assert metadata["rerank_ratio"] == 0.5
    assert snapshot.retrieval_config["rerank_ratio"] == 0.5
    # Rỗng vẫn là row hợp lệ
    assert snapshot.rows[0].contexts == []
    assert snapshot.rows[0].seed_dieu == set()


def test_live_capture_factory_calls_the_real_builder(monkeypatch):
    seen: dict = {}

    def fake_builder(settings):
        seen["settings"] = settings
        return "runtime"

    monkeypatch.setattr(spr, "build_single_pass_runtime", fake_builder)
    settings = SinglePassSettings(rerank_ratio=0.6)
    assert cli._live_capture_runtime_factory(settings) == "runtime"
    assert seen["settings"] is settings


def test_capture_and_prepare_cases_keep_corpora_separate(tmp_path, monkeypatch):
    breadth_dir = tmp_path / "breadth"
    sequential_dir = tmp_path / "sequential"
    breadth_dir.mkdir()
    sequential_dir.mkdir()
    breadth, whitelist_a = _write_inputs(
        breadth_dir, rows=[_corpus_row(1, BREADTH_QUESTION)], dataset_name="breadth.json"
    )
    sequential, whitelist_b = _write_inputs(
        sequential_dir,
        rows=[_corpus_row(1, SEQUENTIAL_QUESTION)],
        dataset_name="sequential.json",
    )
    runtime = FakeRuntime(DEFAULT_ANSWERS)
    factory = FakeFactory(runtime)
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)
    settings = SinglePassSettings(rerank_ratio=0.5)

    breadth_snapshot = capture_seeds(
        breadth, whitelist_a, settings=settings, runtime_factory=factory
    )
    sequential_snapshot = capture_seeds(
        sequential, whitelist_b, settings=settings, runtime_factory=factory
    )
    breadth_output = tmp_path / "seeds_breadth.json"
    sequential_output = tmp_path / "seeds_sequential.json"
    publish_captured_snapshot(breadth_snapshot, breadth_output)
    publish_captured_snapshot(sequential_snapshot, sequential_output)

    # Reload qua đúng artifact path rồi dựng case bằng frontier helper thật
    breadth_cases_path = tmp_path / "cases_breadth.jsonl"
    sequential_cases_path = tmp_path / "cases_sequential.jsonl"
    assert main(["prepare-cases", "--seeds", str(breadth_output), "--output", str(breadth_cases_path)]) == 0
    assert main(["prepare-cases", "--seeds", str(sequential_output), "--output", str(sequential_cases_path)]) == 0

    breadth_cases = read_cases(breadth_cases_path)
    sequential_cases = read_cases(sequential_cases_path)
    assert breadth_cases[0].dataset_id == "breadth"
    assert sequential_cases[0].dataset_id == "sequential"
    assert breadth_cases[0].case_id != sequential_cases[0].case_id
    assert breadth_cases[0].candidates == [10]
    assert sequential_cases[0].candidates == [41, 40]
    # Chưa review thì nhãn phải là draft/unresolved, không tự approve
    assert breadth_cases[0].label_status == "draft"
    assert breadth_cases[0].expected_action == "unresolved"
    assert sequential_cases[0].label_status == "draft"
    assert sequential_cases[0].expected_action == "unresolved"


def test_direct_snapshot_runs_initial_selection_and_permutation(tmp_path, monkeypatch):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, SEQUENTIAL_QUESTION)]
    )
    factory = FakeFactory(FakeRuntime(DEFAULT_ANSWERS))
    monkeypatch.setattr(cli, "_live_capture_runtime_factory", factory)
    snapshot = capture_seeds(
        dataset,
        whitelist,
        settings=SinglePassSettings(rerank_ratio=0.5),
        runtime_factory=factory,
    )
    snapshot_path = tmp_path / "seeds.json"
    publish_captured_snapshot(snapshot, snapshot_path)
    cases_path = tmp_path / "cases.jsonl"
    assert main(["prepare-cases", "--seeds", str(snapshot_path), "--output", str(cases_path)]) == 0

    draft = read_cases(cases_path)[0]
    approved = GateCase.model_validate(
        {
            **draft.model_dump(mode="python"),
            "expected_action": "follow",
            "acceptable_dieu": [41],
            "label_reason": "Điều 41 cần cho câu hỏi",
            "label_status": "approved",
        }
    )
    write_cases([approved], cases_path)

    initial_code = main(
        [
            "run",
            "--experiment",
            "initial-selection",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--repeats",
            "1",
            "--output-root",
            str(tmp_path / "results-initial"),
        ]
    )
    assert initial_code == 0

    permutation_code = main(
        [
            "run",
            "--experiment",
            "permutation",
            "--condition",
            "candidate-order",
            "--schedule",
            "rotate",
            "--seeds",
            str(snapshot_path),
            "--cases",
            str(cases_path),
            "--policies",
            "first",
            "--repeats",
            "1",
            "--output-root",
            str(tmp_path / "results-permutation"),
        ]
    )
    assert permutation_code == 0

    assert main(["summarize", "--run-dir", str(next((tmp_path / "results-initial").iterdir()))]) == 0
    assert main(["summarize", "--run-dir", str(next((tmp_path / "results-permutation").iterdir()))]) == 0


def test_capture_cli_help_lists_command_without_live_imports():
    with pytest.raises(SystemExit) as help_exit:
        main(["capture-seeds", "--help"])
    assert help_exit.value.code == 0


def test_capture_fake_runtime_runs_isolated_in_fresh_process(tmp_path):
    dataset, whitelist = _write_inputs(
        tmp_path, rows=[_corpus_row(1, BREADTH_QUESTION)]
    )
    output = tmp_path / "seeds.json"
    repo_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS_CODE, str(dataset), str(whitelist), str(output)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(repo_root),
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report == {"help_rc": 0, "forbidden": []}
    assert read_snapshot(output).source_kind == "direct_retrieval"
