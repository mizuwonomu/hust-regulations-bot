"""
Trích các dẫn chiếu chéo (cross-reference) giữa các Điều trong quy chế

Đọc `data_quyche/QCDT_2025_DHBK.md` và xuất JSON: mỗi record là MỘT lần
Điều nguồn dẫn chiếu tới một (Điều, khoản) đích

Cách nhận diện: bám cụm "…khoản N Điều M" / "…điểm a khoản N Điều M" /
"Điều M" trần..
"""

import argparse
import json
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = _REPO_ROOT / "data_quyche" / "QCDT_2025_DHBK.md"
DEFAULT_OUTPUT = _REPO_ROOT / "data_quyche" / "cross_references.json"

#Header: splitter.py tách theo "# Chương" và "### Điều" -> bám đúng quy ước đó
_CHUONG_HEADER = re.compile(r"^#\s+CHƯƠNG\s+([IVXLC]+)\s*$", re.IGNORECASE)
_DIEU_HEADER = re.compile(r"^###\s+Điều\s+(\d+)\s*\.", re.IGNORECASE)

#Một lần dẫn chiếu. Phần "điểm a," / "khoản 1, khoản 2 và khoản 3" đứng trước
#"Điều M" được gom vào group(1) để bóc ra từng khoản.
#Cho phép lặp vì thực tế có "điểm b, điểm c khoản 2 Điều 16"
_REFERENCE = re.compile(
    r"(?P<prefix>(?:(?:điểm)\s+[a-zăâđêôơư]\s*,?\s*|(?:khoản)\s+\d+\s*(?:,\s*|\s+và\s+)?)*)"
    r"Điều\s+(?P<dieu>\d+)",
    re.IGNORECASE,
)
_KHOAN_IN_PREFIX = re.compile(r"khoản\s+(\d+)", re.IGNORECASE)

#Điều 47 và Điều 48 (Hiệu lực thi hành) là 2 Điều meta:
#chúng không quy định nội dung mà nói về hiệu lực áp dụng của các điều khác
_APPLICABILITY_SOURCES = {47, 48}

_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100} 

#parse phần mã la mã của chương sang int
def _roman_to_int(s: str) -> int:
    s = s.upper()
    total = 0
    for i, ch in enumerate(s):
        value = _ROMAN[ch]
        if i + 1 < len(s) and value < _ROMAN[s[i + 1]]:
            total -= value
        else:
            total += value
    return total


def classify_reference(source_dieu: int) -> str:
    """
    Phân biệt hiệu lực thi hành và phụ thuộc ngữ nghĩa

    - "dependency": Điều nguồn CẦN Điều đích mới đủ nghĩa
      (VD: Điều 24 dẫn khoản 6 Điều 12 để lấy thang xếp loại).
      Traverse XUÔI: khớp nguồn -> kéo đích về làm context

    - "applicability": Điều đích tự đủ nghĩa, Điều nguồn chỉ giới hạn
      PHẠM VI ÁP DỤNG của nó theo khóa tuyển sinh / học kỳ
      (VD: Điều 48 nói khoản 2 Điều 18 chỉ áp dụng từ khóa 2022).
      Traverse NGƯỢC: khớp đích -> kéo nguồn về để biết điều kiện thời gian
    """
    return "applicability" if source_dieu in _APPLICABILITY_SOURCES else "dependency"


def classify_scope(source_chuong: int | None, target_chuong: int | None) -> str:
    """Dẫn chiếu nằm trong cùng một Chương hay bắc qua Chương khác
    """
    if source_chuong is None or target_chuong is None:
        return "unknown"
    return "same_chapter" if source_chuong == target_chuong else "cross_chapter"


def build_dieu_to_chuong(lines: list[str]) -> dict[int, int]:
    """Điều nào thuộc Chương nào — cần cho cả source lẫn target"""
    mapping: dict[int, int] = {}
    current_chuong: int | None = None

    for line in lines:
        m_chuong = _CHUONG_HEADER.match(line.strip())
        if m_chuong:
            current_chuong = _roman_to_int(m_chuong.group(1))
            continue

        m_dieu = _DIEU_HEADER.match(line.strip())
        if m_dieu and current_chuong is not None:
            mapping[int(m_dieu.group(1))] = current_chuong

    return mapping


def extract_references(markdown_text: str) -> list[dict]:
    lines = markdown_text.replace("\r\n", "\n").split("\n")
    dieu_to_chuong = build_dieu_to_chuong(lines)

    references: list[dict] = []
    seen: set[tuple] = set()
    current_dieu: int | None = None

    for line in lines:
        stripped = line.strip()

        if _CHUONG_HEADER.match(stripped):
            continue

        m_dieu = _DIEU_HEADER.match(stripped)
        if m_dieu:
            #Dòng header là ĐỊNH NGHĨA của Điều, không phải dẫn chiếu -> chỉ đổi ngữ cảnh
            current_dieu = int(m_dieu.group(1))
            continue

        if current_dieu is None:
            continue

        for match in _REFERENCE.finditer(line):
            target_dieu = int(match.group("dieu"))

            #Tự dẫn chiếu chính mình không tạo cạnh mới trong đồ thị, bởi ta đã lấy nguyên 1 điều hoàn chỉnh
            if target_dieu == current_dieu:
                continue

            #Điều đích không tồn tại trong văn bản -> nhiều khả năng là số hiệu
            #của văn bản luật khác được viện dẫn, không phải nội bộ quy chế
            if target_dieu not in dieu_to_chuong:
                continue

            khoan_list = [int(k) for k in _KHOAN_IN_PREFIX.findall(match.group("prefix"))]
            #Không có khoản (VD "thực hiện theo điều 20") -> dẫn chiếu cả điều
            targets = khoan_list or [None]

            for khoan in targets:
                key = (current_dieu, target_dieu, khoan)
                if key in seen:
                    continue
                seen.add(key)

                references.append({
                    "source_dieu": current_dieu,
                    "source_chuong": dieu_to_chuong.get(current_dieu),
                    "target_dieu": target_dieu,
                    "target_khoan": khoan,
                    "target_chuong": dieu_to_chuong.get(target_dieu),
                    "ref_type": classify_reference(current_dieu),
                    "scope": classify_scope(
                        dieu_to_chuong.get(current_dieu),
                        dieu_to_chuong.get(target_dieu),
                    ),
                    "topic": "",
                })

    return references


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract cross-references between Điều")
    parser.add_argument("--markdown", type=str, default=str(MARKDOWN_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    text = Path(args.markdown).read_text(encoding="utf-8")
    references = extract_references(text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(references, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    cross_chapter = [r for r in references if r["scope"] == "cross_chapter"]
    applicability = [r for r in references if r["ref_type"] == "applicability"]
    print(f"Saved {len(references)} references to: {out_path}")
    print(f"  cross-chapter: {len(cross_chapter)}  same-chapter: {len(references) - len(cross_chapter)}")
    print(f"  dependency: {len(references) - len(applicability)}  applicability: {len(applicability)}")


if __name__ == "__main__":
    main()