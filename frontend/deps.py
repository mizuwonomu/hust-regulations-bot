from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True) #freeze các định dạng để không được thay đổi
class AppDeps:
    """Gom các dependencies của frontend"""
    title_generation_scheduler: Callable[[str, str], str | None]
