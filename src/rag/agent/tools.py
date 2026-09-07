"""Tools và helpers cho citation agent: trích dẫn chiếu, tra nguyên văn Điều."""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath('.'))

import pickle
import re

from langchain_chroma import Chroma
from langchain_classic.storage import EncoderBackedStore, LocalFileStore

from src.ingestion.reference_parser import _REFERENCE
from src.rag.agent.schema import CitationMention
from src.rag.config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    DOC_STORE_PATH,
)

# Nhận diện heading Điều hẹp: chỉ dòng đầu heading, format theo corpus "Điều N. Tên điều"
_ARTICLE_HEADING = re.compile(
    r"^(?:#{1,6}\s+)?(?:[A-ZÀ-Ỹ0-9][A-ZÀ-Ỹ0-9\s]*-\s+)?Điều\s+(\d+)\s*\."
)


def _article_heading_line(article_text: str) -> re.Match | None:
    """Match heading Điều trên đúng dòng đầu tiên của article, anchor đầu dòng.

    Params:
    - article_text: Nguyên văn một Điều (title + content)
    """
    first_line = article_text.split("\n", 1)[0]
    if not first_line.strip():
        return None
    return _ARTICLE_HEADING.match(first_line.strip())


def article_title(article_text: str) -> str | None:
    """Trả về dòng heading Điều đã nhận diện, hoặc None nếu không phải heading.

    Params:
    - article_text: Nguyên văn một Điều (title + content)
    """
    match = _article_heading_line(article_text)
    if match is None:
        return None

    # Bỏ markdown heading prefix để render header gọn
    return re.sub(r"^#{1,6}\s+", "", match.string).strip()


def dieu_from_title(article_text: str) -> int:
    """Suy ra số Điều của một article từ dòng title đầu tiên.
    Tức xác định điều nguồn hiện tại của cả một article text.
    Chỉ nhận dòng heading Điều thật, body text chứa "Điều" không được tính là title.

    Params:
    - article_text: Nguyên văn một Điều (title + content)
    """
    match = _article_heading_line(article_text)
    return int(match.group(1)) if match else 0


def source_dieu(key: int, article_text: str) -> int:
    """Số Điều nguồn trung thực của một entry trong mapping thu thập.

    Params:
    - key: Key nội bộ, dương là nguồn chính thức, âm là seed trùng/không parse được.
    - article_text: Nguyên văn của entry, dùng để suy nguồn thật cho key âm.
    """
    return key if key > 0 else max(0, dieu_from_title(article_text))



def extract_citation_mentions(source_dieu: int, text: str) -> list[CitationMention]:
    """Trích các lần dẫn chiếu trong một Điều bằng regex canonical duy nhất.

    Params:
    - source_dieu: Số Điều của text, gắn vào từng mention.
    - text: Nguyên văn một Điều (title + content).

    Trả về mention theo thứ tự xuất hiện, khử trùng lặp (target, excerpt).
    """
    mentions: list[CitationMention] = []
    seen: set[tuple[int, str]] = set()

    lines = text.replace("\r\n", "\n").split("\n")
    for line in lines:
        for match in _REFERENCE.finditer(line):
            key = (int(match.group("dieu")), line.strip())
            if key in seen:
                continue
            seen.add(key)
            mentions.append(
                CitationMention(
                    source_dieu=source_dieu,
                    target_dieu=key[0],
                    excerpt=key[1],
                )
            )

    return mentions



def build_article_map(vector_store) -> dict[int, str]:
    """Helper prebuild map child metadata với child dieu -> doc_id của parent Điều đó.
    Sẽ được build 1 lần tại trước vòng lặp caller/wiring."""
    doc_data = vector_store.get()

    article_map = {}
    doc_metadatas = doc_data["metadatas"]
    for md in doc_metadatas:
        dieu_num = md.get("dieu")

        if dieu_num is None:
            continue

        article_map[dieu_num] = md["doc_id"]

    return article_map


def get_article(dieu: int, *, article_map=None, doc_store=None) -> str | None:
    """Tool tra số Điều và trả về nguyên văn nội dung Điều đó."""

    # Lazy build nếu chưa tồn tại article_map
    if article_map is None:
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=CHROMA_PATH,
        )
        article_map = build_article_map(vector_store)

    target_doc_id = article_map.get(dieu)

    if target_doc_id is None:
        return None

    if doc_store is None:
        fs = LocalFileStore(DOC_STORE_PATH)
        doc_store = EncoderBackedStore(
            store=fs,
            key_encoder=lambda x: x,
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads
        )

    parent_doc = doc_store.mget([target_doc_id])[0] # Map về full nội dung Điều
    if parent_doc is None:
        return None

    full_doc_content = parent_doc.metadata["title"] + "\n" + parent_doc.page_content
    return full_doc_content
