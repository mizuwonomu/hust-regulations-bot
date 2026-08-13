import json
import uuid
import re
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.ingestion.degree_map import BAC_FIELDS

_REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = _REPO_ROOT / "data_quyche" / "QCDT_2025_DHBK.md"
DEGREE_SCOPE_PATH = _REPO_ROOT / "data_quyche" / "degree_scope.json"

#Số Điều nằm trong header dạng "Điều 24. Điểm trung bình toàn khóa..."
_DIEU_NUMBER = re.compile(r"Điều\s+(\d+)")

markdown_documents: str = MARKDOWN_PATH.read_text(encoding="utf-8")

headers_to_split_on = [
    ("#", "Chương"),
    ("###", "Điều"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

markdown_chunk_documents = markdown_splitter.split_text(markdown_documents)

def load_degree_scope() -> dict[int, dict]:
    """Bảng bậc đào tạo theo số điều"""
    if not DEGREE_SCOPE_PATH.exists():
        raise FileNotFoundError(
            f"Thiếu {DEGREE_SCOPE_PATH}. Chạy: python -m src.ingestion.degree_map"
        )

    records = json.loads(DEGREE_SCOPE_PATH.read_text(encoding="utf-8"))
    return {r["dieu"]: r for r in records}


def build_child_metadata(parent_id: str, title: str, dieu_header: str, degree_scope: dict) -> dict:
    """Build child metadata bao gồm:
     - doc_id: id của parent ứng với child
     - title: 'Nội dung chương X' - 'Điều Y' 
     - Các flag boolean như áp dụng đại học (ap_dung_dh), ap_dung_ks, ap_dung_ths, ap_dung_ts 
    """
    metadata = {"doc_id": parent_id, "title": title}

    match = _DIEU_NUMBER.search(dieu_header or "")
    if not match:
        #Phần nội dung nằm trước header Điều đầu tiên -> không gắn cờ bậc
        return metadata

    dieu = int(match.group(1))
    metadata["dieu"] = dieu

    scope = degree_scope.get(dieu)
    if scope is None:
        return metadata

    metadata["chuong"] = scope["chuong"]
    for field in BAC_FIELDS:
        metadata[field] = scope[field]

    return metadata


_GFM_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")

def _is_gfm_table_separator_line(line: str) -> bool:
    """GFM table delimiter row: cells are only --- or :---: style (spaces allowed around |)."""
    s = line.strip()
    if "|" not in s:
        return False

    cells = [c.strip() for c in s.split("|") if c.strip()]

    if not cells:
        return False

    return all(_GFM_SEPARATOR_CELL.match(c) for c in cells)

def _chunk_contains_gfm_separator(chunk: str) -> bool:
    return any(_is_gfm_table_separator_line(ln) for ln in chunk.split("\n"))

def extract_block_header(block_text: str) -> str:
    """First markdown table header row + GFM separator row, or empty if not a table block."""
    lines = [l.strip() for l in block_text.strip().split("\n")]

    for i in range(len(lines) - 1):
        if "|" in lines[i] and _is_gfm_table_separator_line(lines[i + 1]):
            return f"{lines[i]}\n{lines[i + 1]}\n"

    return ""


def split_parent_into_blocks(text):
    text = text.replace('\r\n', '\n')
    """Tách parent thành list các khối: text và table riêng biệt"""

    table_pattern = r'((?:^[ \t]*\|.*(?:\n|$))+)'
    blocks = re.split(table_pattern, text, flags=re.MULTILINE)

    return [b for b in blocks if b.strip()] #loại bỏ các block rỗng


def get_pdr_data():
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    all_child_docs = []
    all_parent_pairs = []
    degree_scope = load_degree_scope()

    for doc in markdown_chunk_documents:
        chuong = doc.metadata.get('Chương', '')
        dieu = doc.metadata.get('Điều', '')

        #Tạo metadata 'title' to sync with qa_chain
        doc.metadata['title'] = f"{chuong} - {dieu}"
        context_prefix = f"Ngữ cảnh: {doc.metadata['title']}\n"
        parent_id = str(uuid.uuid4())

        #chia parent thành các khối
        blocks = split_parent_into_blocks(doc.page_content)

        for i, block in enumerate(blocks):
        #kiểm tra nếu block này là table
            header = extract_block_header(block)

            #thêm lead-in: vài dòng text ở trước bảng để chú thích
            lead_in = ""

            #Nếu là bảng, lấy 2 dòng cuối của khối text trước nó (không lấy từ block bảng liền kề)
            if header and i > 0:
                prev_block = blocks[i - 1].strip()
                if prev_block and not extract_block_header(prev_block):
                    prev_lines = [l for l in prev_block.split("\n") if l.strip()]
                    lead_in = " ".join(prev_lines[-2:])
                    lead_in = f"Câu dẫn: {lead_in}\n"

            #chia nhỏ block thành các child chunks
            block_children = child_splitter.split_text(block)

            for chunk_content in block_children:
            #nếu block là bảng, tiêm header của chính nó vào đầu chunk

                if header:
                    if not _chunk_contains_gfm_separator(chunk_content):
                        final_content = f"{context_prefix}{lead_in}Dữ liệu bảng:\n{header}{chunk_content}"
                    else:
                        final_content = f"{context_prefix}{lead_in}{chunk_content}"

                else:
                    final_content = f"{context_prefix}{chunk_content}"

                #tạo document object cho child
                #tránh việc convert sẵn từ split_documents, rồi lặp qua list đó, trích xuất page_content
                #rồi mới cộng thêm header, dán ngược lại vào object
                c_doc = Document(
                    page_content=final_content,
                    metadata=build_child_metadata(
                        parent_id, doc.metadata['title'], dieu, degree_scope
                    )
                )

                all_child_docs.append(c_doc)

        all_parent_pairs.append((parent_id, doc))

    return all_child_docs, all_parent_pairs
    