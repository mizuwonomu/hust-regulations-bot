"""Characterization tests cho single-pass retrieval dùng chung.

Fixture dùng fake thuần: không network, không model, không store, không sleep.
Kỳ vọng được viết tay từ hành vi retrieval trước khi tách shared module.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tenacity import wait_none

import evals.common.single_pass_retrieval as spr
from evals.common.single_pass_retrieval import (
    QueryExpansion,
    RateLimitError,
    SinglePassSettings,
    extract_dieu_number,
    retrieve_parent_contexts,
    rewrite_into_subqueries,
)

V2_EVALUATOR_PATH = (
    Path(__file__).resolve().parents[3] / "evals/v2/scripts/run_evals_retrieval.py"
)


@dataclass(frozen=True)
class FakeDoc:
    """Bề mặt tối thiểu mà retrieval đọc từ một child hoặc parent doc."""

    page_content: str
    metadata: dict = field(default_factory=dict)


class FakeRewriteChain:
    def __init__(self, queries: list[str]):
        self.queries = queries
        self.calls: list[dict] = []

    def invoke(self, payload: dict) -> QueryExpansion:
        self.calls.append(payload)
        return QueryExpansion(reasoning="fake", queries=list(self.queries))


class FakeEnsembleRetriever:
    def __init__(self, by_subquery: dict[str, list[FakeDoc]]):
        self.by_subquery = by_subquery
        self.seen_subqueries: list[str] = []

    def map(self) -> FakeEnsembleRetriever:
        return self

    def invoke(self, sub_queries: list[str]) -> list[list[FakeDoc]]:
        self.seen_subqueries = list(sub_queries)
        return [self.by_subquery.get(query, []) for query in sub_queries]


class FakeReranker:
    def __init__(self, scores_by_content: dict[str, float]):
        self.scores_by_content = scores_by_content
        self.predict_calls = 0

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.predict_calls += 1
        return [self.scores_by_content[content] for _, content in pairs]


class FakeDocStore:
    def __init__(self, parents: dict[str, FakeDoc]):
        self.parents = parents
        self.requested_ids: list[str] = []

    def mget(self, ids: list[str]) -> list[FakeDoc | None]:
        self.requested_ids = list(ids)
        return [self.parents.get(parent_id) for parent_id in ids]


def _child(content: str, parent_id: str) -> FakeDoc:
    return FakeDoc(page_content=content, metadata={"doc_id": parent_id})


def _parent(parent_id: str, dieu: int) -> FakeDoc:
    return FakeDoc(
        page_content=f"Body {dieu}",
        metadata={"doc_id": parent_id, "title": f"Điều {dieu}. Title {dieu}"},
    )


def _settings(**overrides) -> SinglePassSettings:
    base = {
        "rerank_ratio": 0.5,
        "rerank_max_children": 6,
        "max_parents": 4,
    }
    base.update(overrides)
    return SinglePassSettings(**base)


def _describe() -> tuple[FakeRewriteChain, FakeEnsembleRetriever, FakeReranker, FakeDocStore]:
    """Dựng fixture có dedup, score xáo trộn và nhiều parent hơn cap."""
    children = {
        "child one": _child("child one", "p1"),
        "child two": _child("child two", "p2"),
        "child three": _child("child three", "p3"),
        "child four": _child("child four", "p4"),
        "child five": _child("child five", "p5"),
        "child six": _child("child six", "p6"),
    }
    # child three lặp lại ở sub-query hai: dedup phải giữ lần xuất hiện đầu
    duplicate = _child("child three", "p3")
    ensemble = FakeEnsembleRetriever(
        {
            "q1": [children["child one"], children["child three"], children["child two"]],
            "q2": [children["child six"], duplicate, children["child four"], children["child five"]],
        }
    )
    # Score cố tình không theo thứ tự ensemble: c6 cao thứ hai, c5 dưới cutoff
    reranker = FakeReranker(
        {
            "child one": 1.0,
            "child two": 0.8,
            "child three": 0.6,
            "child four": 0.5,
            "child five": 0.4,
            "child six": 0.9,
        }
    )
    store = FakeDocStore(
        {f"p{index}": _parent(f"p{index}", index) for index in range(1, 7)}
    )
    return FakeRewriteChain(["q1", "q2"]), ensemble, reranker, store


def test_contexts_order_membership_and_parent_cap_are_preserved():
    rewrite_chain, ensemble, reranker, store = _describe()
    contexts, dieu = retrieve_parent_contexts(
        "câu hỏi gốc",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(),
        sleeper=lambda label: None,
    )

    assert rewrite_chain.calls == [{"question": "câu hỏi gốc"}]
    assert ensemble.seen_subqueries == ["q1", "q2"]
    # Cap parent=4 áp sau fetch nên p4 đã lấy nhưng bị bỏ
    assert store.requested_ids == ["p1", "p6", "p2", "p3", "p4"]
    assert contexts == [
        "Điều 1. Title 1\nBody 1",
        "Điều 6. Title 6\nBody 6",
        "Điều 2. Title 2\nBody 2",
        "Điều 3. Title 3\nBody 3",
    ]
    assert dieu == {1, 2, 3, 6}


def test_ratio_boundary_is_inclusive_and_settings_are_not_shared():
    rewrite_chain, ensemble, reranker, store = _describe()
    _, inclusive = retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(rerank_ratio=0.5),
        sleeper=lambda label: None,
    )
    # c4 có score đúng bằng top1 * ratio nên được fetch; cap parent mới bỏ nó sau đó
    assert store.requested_ids == ["p1", "p6", "p2", "p3", "p4"]
    assert inclusive == {1, 2, 3, 6}

    rewrite_chain, ensemble, reranker, store = _describe()
    _, strict = retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(rerank_ratio=0.95),
        sleeper=lambda label: None,
    )
    # Ratio khác chỉ ảnh hưởng runtime của chính nó, không qua global dùng chung
    assert strict == {1}
    assert store.requested_ids == ["p1"]


def test_missing_parent_is_dropped_without_renumbering_siblings():
    rewrite_chain, ensemble, reranker, store = _describe()
    del store.parents["p2"]
    contexts, dieu = retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(),
        sleeper=lambda label: None,
    )
    assert store.requested_ids == ["p1", "p6", "p2", "p3", "p4"]
    assert contexts == [
        "Điều 1. Title 1\nBody 1",
        "Điều 6. Title 6\nBody 6",
        "Điều 3. Title 3\nBody 3",
        "Điều 4. Title 4\nBody 4",
    ]
    assert dieu == {1, 3, 4, 6}


def test_child_cap_is_applied_before_parent_fetch():
    rewrite_chain, ensemble, reranker, store = _describe()
    retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(rerank_max_children=2),
        sleeper=lambda label: None,
    )
    assert store.requested_ids == ["p1", "p6"]


def test_empty_rewrite_falls_back_to_original_question():
    rewrite_chain = FakeRewriteChain([])
    ensemble = FakeEnsembleRetriever({"câu hỏi gốc": [_child("only", "p1")]})
    reranker = FakeReranker({"only": 0.9})
    store = FakeDocStore({"p1": _parent("p1", 1)})

    contexts, dieu = retrieve_parent_contexts(
        "câu hỏi gốc",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(),
        sleeper=lambda label: None,
    )
    assert rewrite_chain.calls == [{"question": "câu hỏi gốc"}]
    assert ensemble.seen_subqueries == ["câu hỏi gốc"]
    assert contexts == ["Điều 1. Title 1\nBody 1"]
    assert dieu == {1}


def test_empty_child_retrieval_is_a_valid_empty_result():
    rewrite_chain, ensemble, reranker, store = _describe()
    ensemble.by_subquery = {"q1": [], "q2": []}
    contexts, dieu = retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(),
        sleeper=lambda label: None,
    )
    # Rỗng thì không predict và không fetch parent
    assert reranker.predict_calls == 0
    assert store.requested_ids == []
    assert contexts == []
    assert dieu == set()


def test_top1_is_retained_even_when_all_scores_are_low():
    rewrite_chain = FakeRewriteChain(["q1"])
    ensemble = FakeEnsembleRetriever(
        {"q1": [_child("high", "p1"), _child("low", "p2"), _child("lower", "p3")]}
    )
    reranker = FakeReranker({"high": -0.5, "low": -0.6, "lower": -0.7})
    store = FakeDocStore({f"p{index}": _parent(f"p{index}", index) for index in (1, 2, 3)})

    _, dieu = retrieve_parent_contexts(
        "q",
        rewrite_chain,
        ensemble,
        reranker,
        store,
        settings=_settings(),
        sleeper=lambda label: None,
    )
    assert store.requested_ids == ["p1"]
    assert dieu == {1}


def test_article_id_extraction_prefers_metadata_over_title():
    with_metadata = FakeDoc(
        page_content="x",
        metadata={"title": "Điều 7. Sai", "Điều": "Điều 42. Đúng"},
    )
    title_only = FakeDoc(page_content="x", metadata={"title": "Điều 7. Fallback"})
    no_number = FakeDoc(page_content="x", metadata={"title": "Mở đầu"})

    assert extract_dieu_number(with_metadata) == 42
    assert extract_dieu_number(title_only) == 7
    assert extract_dieu_number(no_number) is None


def test_rate_limit_error_is_classified_and_rewrapped(monkeypatch):
    class RateLimitedChain:
        def invoke(self, payload):
            raise RuntimeError("429 Too Many Requests")

    class BrokenChain:
        def invoke(self, payload):
            raise ValueError("prompt sai")

    monkeypatch.setattr(rewrite_into_subqueries.retry, "wait", wait_none())
    with pytest.raises(RateLimitError):
        rewrite_into_subqueries("q", RateLimitedChain())
    with pytest.raises(ValueError):
        rewrite_into_subqueries("q", BrokenChain())


def _load_v2_with_stubs(monkeypatch):
    """Import v2 thật với stub cho dotenv/RAGAS/provider, không chạy live."""

    def _module(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        return module

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        _module(
            "dotenv",
            load_dotenv=lambda *a, **k: None,
            dotenv_values=lambda *a, **k: {},
        ),
    )
    monkeypatch.setitem(
        sys.modules, "langchain_groq", _module("langchain_groq", ChatGroq=_Dummy)
    )
    monkeypatch.setitem(
        sys.modules, "ragas", _module("ragas", experiment=lambda *a, **k: (lambda fn: fn))
    )
    monkeypatch.setitem(
        sys.modules, "ragas.cache", _module("ragas.cache", DiskCacheBackend=_Dummy)
    )
    monkeypatch.setitem(
        sys.modules,
        "ragas.dataset_schema",
        _module("ragas.dataset_schema", SingleTurnSample=_Dummy),
    )
    monkeypatch.setitem(
        sys.modules, "ragas.llms", _module("ragas.llms", LangchainLLMWrapper=_Dummy)
    )
    monkeypatch.setitem(
        sys.modules,
        "ragas.metrics",
        _module("ragas.metrics", ContextPrecision=_Dummy, ContextRecall=_Dummy),
    )
    #Loader reranker thật kéo torch nên thay bằng stub cho test này
    monkeypatch.setitem(
        sys.modules,
        "src.rag.reranker_utils",
        _module("src.rag.reranker_utils", load_reranker=lambda: "reranker-loader"),
    )

    spec = importlib.util.spec_from_file_location(
        "_v2_retrieval_under_test", V2_EVALUATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _retrieve_call_sites(source: str) -> list[ast.Call]:
    """Tìm mọi call site dùng retrieve_parent_contexts, kể cả qua partial."""
    sites: list[ast.Call] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id == "retrieve_parent_contexts":
            sites.append(node)
        elif node.func.id == "partial" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Name) and first.id == "retrieve_parent_contexts":
                sites.append(node)
    return sites


def test_v2_evaluator_binds_shared_retrieval_and_forwards_settings(monkeypatch):
    from src.rag.config import QUERY_REWRITE_MODEL, RERANK_MAX_CHILDREN, RERANK_RATIO

    module = _load_v2_with_stubs(monkeypatch)
    # Binding thật, không phải trùng tên: cùng một function object với shared module
    assert module.retrieve_parent_contexts is spr.retrieve_parent_contexts
    assert module.SinglePassSettings is spr.SinglePassSettings
    assert module.build_single_pass_runtime is spr.build_single_pass_runtime
    assert not hasattr(module, "_rerank_ratio_filter")
    # Surface cũ mà run_ratio_sweep.py đang dùng vẫn còn
    assert module.RERANK_MAX_CHILDREN == RERANK_MAX_CHILDREN
    assert module.RERANK_RATIO == RERANK_RATIO
    assert module.QUERY_REWRITE_MODEL == QUERY_REWRITE_MODEL
    assert module.load_reranker() == "reranker-loader"

    # Mọi call site phải forward settings hiệu lực, không dựa vào global
    source = V2_EVALUATOR_PATH.read_text(encoding="utf-8")
    sites = _retrieve_call_sites(source)
    assert len(sites) == 2
    for site in sites:
        keywords = {keyword.arg for keyword in site.keywords}
        assert "settings" in keywords


def test_runtime_builder_records_config_models_and_honors_injected_deps():
    from src.rag.config import EMBEDDING_MODEL, RERANKER_MODEL

    calls: dict = {}

    def embedding_loader():
        calls["embedding"] = calls.get("embedding", 0) + 1
        return "embeddings"

    def reranker_loader():
        calls["reranker"] = calls.get("reranker", 0) + 1
        return "reranker"

    def retriever_builder(**kwargs):
        calls["retriever"] = kwargs
        return "ensemble", "store", "vector"

    def rewrite_chain_builder(settings):
        calls["rewrite"] = settings
        return "chain"

    settings = SinglePassSettings(rerank_ratio=0.5)
    runtime = spr.build_single_pass_runtime(
        settings,
        embedding_loader=embedding_loader,
        reranker_loader=reranker_loader,
        retriever_builder=retriever_builder,
        rewrite_chain_builder=rewrite_chain_builder,
    )

    assert calls["embedding"] == 1
    assert calls["reranker"] == 1
    assert calls["retriever"]["k"] == settings.retriever_k
    assert calls["retriever"]["weights"] == settings.hybrid_weights
    assert calls["retriever"]["embedding_model"] == "embeddings"
    assert calls["rewrite"] is settings
    assert runtime.ensemble_retriever == "ensemble"
    assert runtime.vector_store == "vector"
    assert runtime.rewrite_chain == "chain"
    # Metadata ghi model từ config, không phải lời khai settings
    assert runtime.metadata["embedding_model"] == EMBEDDING_MODEL
    assert runtime.metadata["reranker_model"] == RERANKER_MODEL
    assert runtime.metadata["rerank_ratio"] == 0.5


def test_unsupported_model_override_is_rejected_before_loading():
    with pytest.raises(ValueError, match="embedding_model override"):
        SinglePassSettings(embedding_model="someone-else/model")
    with pytest.raises(ValueError, match="reranker_model override"):
        SinglePassSettings(reranker_model="someone-else/model")

    # model_construct bỏ qua validator nên builder vẫn phải là chốt chặn cuối
    bypassed = SinglePassSettings.model_construct(embedding_model="someone-else/model")
    with pytest.raises(ValueError, match="embedding_model override"):
        spr.build_single_pass_runtime(
            bypassed,
            embedding_loader=lambda: "embeddings",
            reranker_loader=lambda: "reranker",
            retriever_builder=lambda **kwargs: ("e", "s", "v"),
            rewrite_chain_builder=lambda settings: "chain",
        )


def test_shared_module_import_stays_free_of_providers_and_model_loaders():
    # Kiểm tra trong process mới để module đã nạp từ pytest không che lỗi
    code = (
        "import json, sys;"
        "sys.path.insert(0, '.');"
        "import evals.common.single_pass_retrieval as spr;"
        "print(json.dumps({m: m in sys.modules for m in "
        "['langchain_groq','langchain_chroma','sentence_transformers','torch','chromadb']}))"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[3]),
        env={"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert set(report.values()) == {False}
