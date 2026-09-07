"""Unit test extract_citation_mentions: pure, store-free, không DB.

Assert đúng contract: trả về list CitationMention theo thứ tự xuất hiện,
mỗi record mang source_dieu + excerpt là dòng gốc sau strip(), matching per line.
"""

from src.rag.agent.schema import CitationMention
from src.rag.agent.tools import extract_citation_mentions


def targets(mentions):
    return [m.target_dieu for m in mentions]


class TestExtractionTable:
    # Bảng case số 1-7 trong tests.md section A
    def test_khoan_prefix_swallowed(self):
        mentions = extract_citation_mentions(5, "...thực hiện theo khoản 1 Điều 10")
        assert targets(mentions) == [10]

    def test_multiple_khoan_prefix_swallowed(self):
        mentions = extract_citation_mentions(5, "theo khoản 2, khoản 3 và khoản 4 Điều 19")
        assert targets(mentions) == [19]

    def test_two_targets_first_appearance_order(self):
        mentions = extract_citation_mentions(5, "Điều 12 quy định A, đối chiếu Điều 15 thì B")
        assert targets(mentions) == [12, 15]

    def test_bare_citation(self):
        mentions = extract_citation_mentions(5, "...thực hiện theo Điều 20 Quy chế này")
        assert targets(mentions) == [20]

    def test_same_target_same_line_dedup(self):
        mentions = extract_citation_mentions(5, "theo Điều 10 quy định A, nhắc lại Điều 10 ở đây")
        assert len(mentions) == 1
        assert mentions[0].target_dieu == 10

    def test_no_citation_returns_empty(self):
        assert extract_citation_mentions(5, "nội dung không dẫn chiếu điều nào") == []

    def test_regulation_name_without_dieu_number(self):
        assert extract_citation_mentions(5, "theo Quy chế công tác sinh viên đại học") == []


class TestMentionRecord:
    # Mỗi mention là CitationMention, mang đủ source_dieu và excerpt của dòng gốc
    def test_record_carries_source_and_excerpt(self):
        mentions = extract_citation_mentions(5, "  ...thực hiện theo khoản 1 Điều 10  ")
        assert len(mentions) == 1
        mention = mentions[0]
        assert isinstance(mention, CitationMention)
        assert mention.source_dieu == 5
        assert mention.target_dieu == 10
        assert mention.excerpt == "...thực hiện theo khoản 1 Điều 10"

    def test_multiple_targets_one_line_one_record_per_target(self):
        mentions = extract_citation_mentions(5, "dẫn Điều 12 và Điều 15 trong cùng dòng")
        assert targets(mentions) == [12, 15]
        assert mentions[0].excerpt == mentions[1].excerpt == "dẫn Điều 12 và Điều 15 trong cùng dòng"


class TestLineSemantics:
    # Matching per line: dòng khác nhau là mention khác nhau dù cùng target
    def test_same_target_two_lines_both_retained(self):
        text = "dòng đầu dẫn Điều 10 về điểm a\ndòng sau cũng dẫn Điều 10 về điểm b"
        mentions = extract_citation_mentions(5, text)
        assert targets(mentions) == [10, 10]
        assert mentions[0].excerpt == "dòng đầu dẫn Điều 10 về điểm a"
        assert mentions[1].excerpt == "dòng sau cũng dẫn Điều 10 về điểm b"

    def test_repeated_identical_pair_dedup(self):
        text = "dẫn chiếu Điều 10 ở đây\ndẫn chiếu Điều 10 ở đây"
        mentions = extract_citation_mentions(5, text)
        assert len(mentions) == 1
        assert mentions[0].excerpt == "dẫn chiếu Điều 10 ở đây"

    def test_crlf_normalized_in_excerpt(self):
        text = "cảnh báo theo Điều 10\r\náp dụng Điều 12 khi vi phạm"
        mentions = extract_citation_mentions(5, text)
        assert targets(mentions) == [10, 12]
        assert all("\r" not in m.excerpt for m in mentions)


class TestSourceAndSelf:
    # Unknown source dùng 0; self reference KHÔNG bị lọc ở tầng extractor
    def test_unknown_source_zero(self):
        mentions = extract_citation_mentions(0, "theo Điều 10 Quy chế này")
        assert len(mentions) == 1
        assert mentions[0].source_dieu == 0
        assert mentions[0].target_dieu == 10

    def test_self_reference_returned(self):
        text = "Điều 10. Tiêu đề\nthực hiện theo Điều 10 tại điểm b"
        mentions = extract_citation_mentions(10, text)
        assert targets(mentions) == [10, 10]
        assert mentions[0].excerpt == "Điều 10. Tiêu đề"
        assert mentions[1].excerpt == "thực hiện theo Điều 10 tại điểm b"
