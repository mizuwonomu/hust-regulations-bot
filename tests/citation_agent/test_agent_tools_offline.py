"""Unit test offline cho helpers của tools.py: build_article_map, get_article với fake,
và nhận diện heading Điều theo contract.

Không mở Chroma hay doc store thật: fake store chỉ cung cấp đúng surface mà
helper tiêu thụ (get() -> {"metadatas": [...]}, mget(keys)).
"""

from types import SimpleNamespace

import pytest

from src.rag.agent.tools import (
    article_title,
    build_article_map,
    dieu_from_title,
    get_article,
)


class FakeVectorStore:
    """Chroma giả: chỉ cần surface get() trả {"metadatas": [...]}"""

    def __init__(self, metadatas):
        self._metadatas = metadatas

    def get(self):
        return {"metadatas": self._metadatas}


class FakeDocStore:
    """Parent store giả: ghi lại mọi lệnh mget để assert về truy vấn"""

    def __init__(self, docs=None):
        self.docs = dict(docs or {})
        self.mget_calls = []

    def mget(self, keys):
        self.mget_calls.append(list(keys))
        return [self.docs.get(key) for key in keys]


def parent_doc(title, content):
    return SimpleNamespace(metadata={"title": title}, page_content=content)


class TestBuildArticleMap:
    # Map dieu -> parent doc_id được quyết định toàn bộ bởi metadata của fake
    def test_maps_dieu_to_parent_doc_id(self):
        store = FakeVectorStore(
            [
                {"dieu": 3, "doc_id": "parent-3", "title": "x"},
                {"dieu": 10, "doc_id": "parent-10", "title": "y"},
            ]
        )
        assert build_article_map(store) == {3: "parent-3", 10: "parent-10"}

    def test_metadata_without_dieu_key_skipped(self):
        store = FakeVectorStore(
            [
                {"doc_id": "parent-orphan"},
                {"dieu": 5, "doc_id": "parent-5"},
            ]
        )
        article_map = build_article_map(store)
        assert article_map == {5: "parent-5"}
        assert None not in article_map

    def test_multiple_children_same_dieu_single_entry(self):
        store = FakeVectorStore(
            [
                {"dieu": 10, "doc_id": "parent-10"},
                {"dieu": 10, "doc_id": "parent-10"},
            ]
        )
        article_map = build_article_map(store)
        assert len(article_map) == 1
        assert article_map[10] == "parent-10"


class TestGetArticleInjected:
    # Map + store được inject: không rebuild, không đụng store khi map miss
    def test_hit_returns_title_newline_content_exactly(self):
        doc_store = FakeDocStore({"parent-10": parent_doc("Điều 10. Tiêu đề", "Nội dung Điều 10")})
        result = get_article(10, article_map={10: "parent-10"}, doc_store=doc_store)
        assert result == "Điều 10. Tiêu đề\nNội dung Điều 10"

    def test_map_miss_returns_none_without_store_query(self):
        doc_store = FakeDocStore()
        result = get_article(10, article_map={20: "parent-20"}, doc_store=doc_store)
        assert result is None
        assert doc_store.mget_calls == []

    def test_store_miss_returns_none(self):
        doc_store = FakeDocStore({}) # mget trả [None]
        result = get_article(10, article_map={10: "parent-10"}, doc_store=doc_store)
        assert result is None

    def test_injected_deps_reused_not_rebuilt(self, monkeypatch):
        def forbidden_build(vector_store):
            raise AssertionError("build_article_map không được chạy lại khi map đã được inject")

        monkeypatch.setattr("src.rag.agent.tools.build_article_map", forbidden_build)
        doc_store = FakeDocStore({"parent-10": parent_doc("Điều 10. Tiêu đề", "Nội dung")})
        result = get_article(10, article_map={10: "parent-10"}, doc_store=doc_store)
        assert result is not None
        assert doc_store.mget_calls == [["parent-10"]]

    def test_empty_map_is_valid_injected_map(self, monkeypatch):
        # {} là map hợp lệ: lookup miss trả None, tuyệt đối không mở Chroma
        def forbidden_chroma(*args, **kwargs):
            raise AssertionError("map rỗng inject không được trigger lazy build")

        monkeypatch.setattr("src.rag.agent.tools.Chroma", forbidden_chroma)
        doc_store = FakeDocStore()
        result = get_article(10, article_map={}, doc_store=doc_store)
        assert result is None
        assert doc_store.mget_calls == []


class TestGetArticleLazyPath:
    # Lazy path chỉ được test qua mock: không bao giờ mở chroma_db/ hay doc_store_pdr/
    def test_lazy_map_build_goes_through_vector_store(self, monkeypatch):
        fake_store = FakeVectorStore([{"dieu": 10, "doc_id": "parent-10"}])
        chroma_calls = []

        def fake_chroma(**kwargs):
            chroma_calls.append(kwargs)
            return fake_store

        monkeypatch.setattr("src.rag.agent.tools.Chroma", fake_chroma)
        doc_store = FakeDocStore({"parent-10": parent_doc("Điều 10. Tiêu đề", "Nội dung")})
        result = get_article(10, doc_store=doc_store)
        assert len(chroma_calls) == 1
        assert result == "Điều 10. Tiêu đề\nNội dung"

    def test_lazy_doc_store_build_is_mocked_not_real(self, monkeypatch):
        opened_paths = []

        def fake_local_file_store(path):
            opened_paths.append(path)
            return object()

        def fake_encoder_backed_store(**kwargs):
            return SimpleNamespace(mget=lambda keys: [parent_doc("Điều 20. Tiêu đề", "Nội dung")])

        monkeypatch.setattr("src.rag.agent.tools.LocalFileStore", fake_local_file_store)
        monkeypatch.setattr("src.rag.agent.tools.EncoderBackedStore", fake_encoder_backed_store)
        result = get_article(20, article_map={20: "parent-20"})
        assert result == "Điều 20. Tiêu đề\nNội dung"
        assert opened_paths == ["doc_store_pdr"]


class TestArticleHeadingRecognition:
    # Heading hợp lệ: dòng đầu, markdown prefix tuỳ chọn, chapter prefix "<title> - "
    @pytest.mark.parametrize(
        "heading",
        [
            "Điều 42. Đánh giá luận án tiến sĩ",
            "### Điều 42. Đánh giá luận án tiến sĩ",
            "ĐÀO TẠO TIẾN SĨ - Điều 42. Đánh giá luận án tiến sĩ",
        ],
    )
    def test_valid_headings_recognized(self, heading):
        assert dieu_from_title(heading + "\nbody") == 42
        assert article_title(heading + "\nbody") is not None

    def test_empty_first_line_never_promotes_body_line(self):
        assert dieu_from_title("\nĐiều 20. Giả heading") == 0
        assert article_title("\nĐiều 20. Giả heading") is None

    # Contract: câu body chỉ nhắc Điều KHÔNG được nhận là heading - regex yêu cầu
    # dấu chấm sau số Điều và prefix chương IN HOA nên cả hai fixture âm bị từ chối
    @pytest.mark.parametrize(
        "line",
        [
            "Điều 20 quy định cách xử lý vi phạm",
            "Theo quy định - Điều 20 được áp dụng",
        ],
    )
    def test_body_sentences_must_not_be_headings(self, line):
        text = line + "\nbody"
        assert dieu_from_title(text) == 0
        assert article_title(text) is None
