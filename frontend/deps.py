from dataclasses import dataclass
from typing import Callable, Any


@dataclass(frozen=True) #freeze các định dạng để không được thay đổi
class AppDeps:
    """Gom các dependencies như chain, các services"""
    rag_chain: Any
    db_connection_factory: Callable[[], Any]
    title_generation_scheduler: Callable[[str, str], str | None]