"""Unit test cho 2 hàm thuần của hop-recall metric (parse_link, compute_hop_recall).

Chạy offline, không cần API call - metric phải deterministic để so sánh được
giữa các run bất kể judge model đổi thế nào
"""

import pytest

from evals.v2.scripts.run_evals_retrieval import compute_hop_recall, parse_link


class TestParseLink:
    def test_parse_standard_link(self):
        gold_dieu, second_hop = parse_link("21 -> 10")
        assert gold_dieu == {21, 10}
        assert second_hop == 10

    def test_parse_link_without_spaces(self):
        gold_dieu, second_hop = parse_link("33->3")
        assert gold_dieu == {33, 3}
        assert second_hop == 3

    def test_parse_malformed_link_raises(self):
        with pytest.raises(ValueError, match="định dạng"):
            parse_link("21 10")

    def test_parse_non_numeric_link_raises(self):
        with pytest.raises(ValueError, match="số không hợp lệ"):
            parse_link("21 -> abc")


class TestComputeHopRecall:
    def test_both_gold_retrieved(self):
        full_recall, hit = compute_hop_recall({21, 10}, 10, {21, 10})
        assert full_recall == 1.0
        assert hit is True

    def test_second_hop_missed(self):
        full_recall, hit = compute_hop_recall({21, 10}, 10, {21})
        assert full_recall == 0.5
        assert hit is False

    def test_second_hop_hit_source_missed(self):
        full_recall, hit = compute_hop_recall({21, 10}, 10, {10, 5})
        assert full_recall == 0.5
        assert hit is True

    def test_nothing_retrieved(self):
        full_recall, hit = compute_hop_recall({21, 10}, 10, set())
        assert full_recall == 0.0
        assert hit is False

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError, match="rỗng"):
            compute_hop_recall(set(), 10, {10})
