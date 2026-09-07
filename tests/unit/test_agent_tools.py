"""Unit test cho tool get_article trong src/rag/agent/tools.py.

Chạy trên chroma_db + doc_store_pdr thật để guard
đúng contract cửa A (map dieu -> doc_id) và cửa B (doc store) mà agent
loop sẽ dựa vào: hit trả str đầy đủ, miss trả None thay vì raise.
"""

from src.rag.agent.tools import get_article


class TestHappyPath:
    # Cửa A + B thông: tra số Điều có thật phải trả về nguyên văn
    def test_hit_returns_string_with_marker(self):
        result = get_article(10)

        assert result is not None
        assert isinstance(result, str)
        assert "Điều 10" in result


class TestMiss:
    # Cửa A trượt: số Điều không tồn tại, agent loop dựa vào None để biết "hợp hụt"
    def test_missing_dieu_returns_none_without_raise(self):
        result = get_article(999)

        assert result is None


class TestContentShape:
    # Kết quả phải là title + "\n" + page_content, không phải title nối cứng vào content
    def test_result_starts_with_title_then_newline(self):
        result = get_article(10)
        assert result is not None

        title_line, sep, body = result.partition("\n")

        assert sep == "\n"
        assert "Điều 10." in title_line
        assert body.strip() != ""
