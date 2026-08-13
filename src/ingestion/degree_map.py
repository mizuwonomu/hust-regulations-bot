"""
Suy ra mỗi Điều áp dụng cho những bậc đào tạo nào

Suy theo hai nguồn bổ sung cho nhau:
  1. Chương — phần nền. Chương II=ĐH, III=KS, IV=ThS, V=TS; Chương I
     (quy định chung) và Chương VI (tổ chức thực hiện) áp dụng cho mọi bậc.
  2. Các khoản trong Điều có tham chiếu liên bậc, gom theo điều

"""

import os
import sys
sys.path.append(os.path.abspath('.'))

import argparse
import json
from pathlib import Path

from src.ingestion.reference_parser import (
    MARKDOWN_PATH,
    build_dieu_to_chuong,
    extract_references,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "data_quyche" / "degree_scope.json"

#Tên trường đi thẳng vào child metadata -> giữ nguyên khi dùng ở where clause
BAC_FIELDS = ("ap_dung_dh", "ap_dung_ks", "ap_dung_ths", "ap_dung_ts")
_ALL_BAC = set(BAC_FIELDS)

#Chương I: quy định chung, Chương VI: tổ chức thực hiện -> áp dụng mọi bậc
_CHUONG_TO_BAC: dict[int, set[str]] = {
    1: _ALL_BAC,
    2: {"ap_dung_dh"},
    3: {"ap_dung_ks"},
    4: {"ap_dung_ths"},
    5: {"ap_dung_ts"},
    6: _ALL_BAC,
}


def base_flags(chuong: int | None) -> set[str]:
    """Bậc suy từ Chương — phần nền, chưa tính phần được Điều khác mượn"""
    if chuong is None:
        return set()
    return set(_CHUONG_TO_BAC.get(chuong, set()))


def build_degree_map(markdown_text: str) -> list[dict]:
    dieu_to_chuong = build_dieu_to_chuong(markdown_text.replace("\r\n", "\n").split("\n"))
    references = extract_references(markdown_text)

    flags = {dieu: base_flags(chuong) for dieu, chuong in dieu_to_chuong.items()}

    #Điều nào bật thêm cờ nhờ được điều nào mượn, tức được tham chiếu tới 
    extended_by: dict[int, set[int]] = {dieu: set() for dieu in dieu_to_chuong}

    for ref in references:
        if ref["ref_type"] != "dependency":
            continue

        target = ref["target_dieu"]
        borrower_flags = base_flags(ref["source_chuong"])
        added = borrower_flags - flags.get(target, set())

        if added:
            flags[target] |= added
            extended_by[target].add(ref["source_dieu"])

    result = []
    for dieu in sorted(dieu_to_chuong):
        record = {"dieu": dieu, "chuong": dieu_to_chuong[dieu]}
        record.update({field: field in flags[dieu] for field in BAC_FIELDS})
        record["extended_by"] = sorted(extended_by[dieu])
        result.append(record)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive per-Điều degree-level scope")
    parser.add_argument("--markdown", type=str, default=str(MARKDOWN_PATH))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    text = Path(args.markdown).read_text(encoding="utf-8")
    degree_map = build_degree_map(text)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(degree_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    extended = [r for r in degree_map if r["extended_by"]]
    print(f"Saved degree scope for {len(degree_map)} Điều to: {out_path}")
    for field in BAC_FIELDS:
        print(f"  {field}: {sum(1 for r in degree_map if r[field])}")
    print(f"  mở rộng nhờ đồ thị dẫn chiếu: {len(extended)} Điều")
    for r in extended:
        on = [f for f in BAC_FIELDS if r[f]]
        print(f"    Điều {r['dieu']} (Chương {r['chuong']}) <- mượn bởi {r['extended_by']} -> {on}")


if __name__ == "__main__":
    main()