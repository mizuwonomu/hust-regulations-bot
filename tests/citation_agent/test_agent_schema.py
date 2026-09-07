"""Unit test Decision schema và CitationMention: contract dữ liệu công khai.

Test hành vi public (state hợp lệ, extra/strict, frozen) chứ không test nội bộ Pydantic.
"""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from src.rag.agent.schema import CitationMention, Decision


class TestDecisionValidStates:
    # Chỉ hai state hợp lệ: (stop=True, dieu=None) và (stop=False, dieu dương)
    def test_stop_decision_valid(self):
        decision = Decision(stop=True, dieu=None)
        assert decision.stop is True
        assert decision.dieu is None

    def test_follow_decision_valid(self):
        decision = Decision(stop=False, dieu=20)
        assert decision.stop is False
        assert decision.dieu == 20


class TestDecisionStrict:
    # Thiếu trường, trường thừa, ép kiểu ngầm đều bị từ chối
    def test_missing_stop_rejected(self):
        with pytest.raises(ValidationError):
            Decision(dieu=None)

    def test_missing_dieu_rejected(self):
        # Nullable không có nghĩa là optional
        with pytest.raises(ValidationError):
            Decision(stop=True)

    def test_unknown_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            Decision(stop=True, dieu=None, note="thừa")

    def test_string_dieu_not_coerced(self):
        with pytest.raises(ValidationError):
            Decision(stop=False, dieu="20")

    def test_string_stop_not_coerced(self):
        with pytest.raises(ValidationError):
            Decision(stop="true", dieu=None)

    def test_float_dieu_not_coerced(self):
        with pytest.raises(ValidationError):
            Decision(stop=False, dieu=20.0)

    def test_bool_dieu_not_coerced(self):
        # Bool là subclass của int nhưng strict mode phải từ chối thay vì ép thành 1
        with pytest.raises(ValidationError):
            Decision(stop=False, dieu=True)


class TestDecisionStateInvariant:
    def test_stop_with_non_null_dieu_rejected(self):
        with pytest.raises(ValidationError):
            Decision(stop=True, dieu=20)

    def test_follow_with_null_dieu_rejected(self):
        with pytest.raises(ValidationError):
            Decision(stop=False, dieu=None)

    @pytest.mark.parametrize("bad_dieu", [0, -1, -100])
    def test_follow_with_non_positive_dieu_rejected(self, bad_dieu):
        with pytest.raises(ValidationError):
            Decision(stop=False, dieu=bad_dieu)


class TestCitationMention:
    # Frozen-ness là yêu cầu hành vi duy nhất: gán field phải raise
    def test_construction_works(self):
        mention = CitationMention(source_dieu=10, target_dieu=20, excerpt="theo Điều 20")
        assert mention.source_dieu == 10
        assert mention.target_dieu == 20
        assert mention.excerpt == "theo Điều 20"

    def test_mutation_raises(self):
        mention = CitationMention(source_dieu=10, target_dieu=20, excerpt="theo Điều 20")
        with pytest.raises(FrozenInstanceError):
            mention.target_dieu = 30
