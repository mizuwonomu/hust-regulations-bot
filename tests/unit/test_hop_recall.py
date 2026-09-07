"""Unit test cho 3 hàm thuần của hop-recall metric (parse_link, compute_hop_recall, compute_missing_gold_recovery).

Chạy offline, không cần API call - metric phải deterministic để so sánh được
giữa các run bất kể judge model đổi thế nào
"""

import pytest

from evals.v2.scripts.run_evals_retrieval import (
    compute_hop_recall,
    compute_missing_gold_recovery,
    parse_link,
)


class TestParseLink:
    def test_parse_standard_link(self):
        source, target = parse_link("21 -> 10")
        assert source == 21
        assert target == 10

    def test_parse_link_without_spaces(self):
        source, target = parse_link("33->3")
        assert source == 33
        assert target == 3

    def test_parse_malformed_link_raises(self):
        with pytest.raises(ValueError, match="định dạng"):
            parse_link("21 10")

    def test_parse_non_numeric_link_raises(self):
        with pytest.raises(ValueError, match="số không hợp lệ"):
            parse_link("21 -> abc")


class TestComputeHopRecall:
    def test_both_gold_retrieved(self):
        (
            recall,
            precision,
            f1,
            all_hit,
            source_hit,
            target_hit,
        ) = compute_hop_recall({21, 10}, 21, 10, {21, 10})
        assert recall == 1.0
        assert precision == 1.0
        assert f1 == 1.0
        assert all_hit is True
        assert source_hit is True
        assert target_hit is True

    def test_full_gold_with_extra_articles(self):
        # Ví dụ chuẩn: gold 2 Điều, lấy đúng cả 2 nhưng thêm 3 Điều ngoài gold
        (
            recall,
            precision,
            f1,
            all_hit,
            source_hit,
            target_hit,
        ) = compute_hop_recall({21, 10}, 21, 10, {21, 10, 5, 7, 9})
        assert recall == 1.0
        assert precision == 0.4
        assert f1 == pytest.approx(2 * 0.4 * 1.0 / (0.4 + 1.0))
        assert all_hit is True
        assert source_hit is True
        assert target_hit is True

    def test_target_missed(self):
        (
            recall,
            precision,
            f1,
            all_hit,
            source_hit,
            target_hit,
        ) = compute_hop_recall({21, 10}, 21, 10, {21})
        assert recall == 0.5
        assert precision == 1.0
        assert f1 == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))
        assert all_hit is False
        assert source_hit is True
        assert target_hit is False

    def test_target_hit_source_missed(self):
        (
            recall,
            precision,
            f1,
            all_hit,
            source_hit,
            target_hit,
        ) = compute_hop_recall({21, 10}, 21, 10, {10, 5})
        assert recall == 0.5
        assert precision == 0.5
        assert f1 == pytest.approx(0.5)
        assert all_hit is False
        assert source_hit is False
        assert target_hit is True

    def test_nothing_retrieved(self):
        (
            recall,
            precision,
            f1,
            all_hit,
            source_hit,
            target_hit,
        ) = compute_hop_recall({21, 10}, 21, 10, set())
        assert recall == 0.0
        # R rỗng thì precision/f1 quy ước bằng 0 thay vì chia 0
        assert precision == 0.0
        assert f1 == 0.0
        assert all_hit is False
        assert source_hit is False
        assert target_hit is False

    def test_empty_gold_raises(self):
        with pytest.raises(ValueError, match="rỗng"):
            compute_hop_recall(set(), 21, 10, {10})


class TestComputeMissingGoldRecovery:
    def test_recovers_missing_gold(self):
        missing, recovered = compute_missing_gold_recovery(
            {13, 5}, {22, 13}, {13, 5, 22}
        )
        assert missing == [22]
        assert recovered == [22]

    def test_no_recovery_when_agent_misses(self):
        missing, recovered = compute_missing_gold_recovery(
            {13, 5}, {22, 13}, {13, 5, 3}
        )
        assert missing == [22]
        assert recovered == []

    def test_not_applicable_when_baseline_complete(self):
        missing, recovered = compute_missing_gold_recovery(
            {22, 13}, {22, 13}, {22, 13}
        )
        assert missing == []
        assert recovered == []

    def test_baseline_beyond_gold_ignored(self):
        # Baseline retrieves chỉ gold bị bỏ sót mới tính, Điều ngoài gold không liên quan
        missing, recovered = compute_missing_gold_recovery(
            {13, 5}, {22, 13}, {13, 22}
        )
        assert missing == [22]
        assert recovered == [22]
