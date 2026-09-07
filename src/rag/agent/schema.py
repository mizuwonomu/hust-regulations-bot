"""Contract cho quyết định dừng hoặc follow của citation agent."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Decision(BaseModel):
    """
    Params:
    - stop: Quyết định dừng việc tìm tiếp các Điều hay không.
    - dieu: Số Điều cần retrieve tại hop hiện tại.
    """

    model_config = ConfigDict(extra="forbid", strict=True) # Bỏ trường thừa, không ép kiểu ngầm

    stop: bool
    dieu: int | None

    @field_validator("dieu")
    @classmethod
    def _dieu_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("dieu phải là số nguyên dương")
        return value

    @model_validator(mode="after")
    def _check_state_invariant(self) -> Decision:
        if self.stop and self.dieu is not None:
            raise ValueError("stop=True thì dieu phải là None")
        if not self.stop and self.dieu is None:
            raise ValueError("stop=False thì phải kèm dieu để follow")
        return self


@dataclass(frozen=True, slots=True)
class CitationMention:
    """
    Trajectory cho extract citations.

    Params:
    - source_dieu: Số Điều chứa lần dẫn chiếu
    - target_dieu: Số Điều được dẫn chiếu tới
    - excerpt: Dòng nguyên văn chứa lần dẫn chiếu đó
    """

    source_dieu: int
    target_dieu: int
    excerpt: str