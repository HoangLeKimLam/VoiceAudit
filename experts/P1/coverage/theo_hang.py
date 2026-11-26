"""Tóm bảng T1.x theo HÀNG Constraint (không theo nhóm ca): mỗi hàng bao nhiêu ca, bao nhiêu ca đúng hết ở cả vi và en,
và lỗi thuộc loại nào (sai quyết định · thiếu / thừa cảnh báo của chính hàng · cảnh báo lạ từ hàng khác). Chỉ đọc.

  .venv/Scripts/python.exe experts/P1/coverage/theo_hang.py [tag]
"""
from __future__ import annotations

import collections
import json
import os
import sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".tmp_v6", "results")
HANG = (("dungluong.", "Dung lượng ≤ 250 MB"), ("dinhdang.", "Định dạng / mã hoá"), ("bitrate.", "Tốc độ bit nén"), ("tanso.", "Tần số lấy mẫu ≥ 16 kHz"),
        ("dosau.", "Độ sâu bit ≥ 16"), ("kenh.", "Số kênh / hai kênh"), ("dodai.", "Độ dài 3 s – 10 phút"),
        ("tinhieu.bangtan", "Dải tần thực ≥ 6 kHz"), ("tinhieu.g711", "Dải tần thực ≥ 6 kHz"), ("tinhieu.amrnb", "Dải tần thực ≥ 6 kHz"),
        ("tinhieu.rong", "Dải tần thực ≥ 6 kHz"), ("tinhieu.quanho", "Âm lượng ≥ −40 dBFS"), ("tinhieu.amluong", "Âm lượng ≥ −40 dBFS"),
        ("tinhieu.muc_noi", "Âm lượng ≥ −40 dBFS"), ("tinhieu.catdinh", "Cắt đỉnh"), ("tinhieu.matTH", "Mất tín hiệu"),
        ("tinhieu.tiengnoi", "Lượng tiếng nói ≥ 2 s và ≥ 30 %"), ("nguoi.", "Số người nói ≤ 4"), ("loai.", "Loại âm thanh"))
DAU_CUOI = "0,3 s đầu và cuối không được soi"


def hang_cua(ca: str) -> str:
    for p, h in HANG:
        if ca.startswith(p):
            return h
    raise KeyError(ca)


def loai_loi(s: str) -> str:
    if s.startswith("quyet dinh"):
        return "sai quyết định"
    if s.startswith("canh bao moi khong co o tep goc"):
        return "cảnh báo lạ (hàng khác)"
    if s.startswith("thieu canh bao"):
        return "thiếu cảnh báo"
    if s.startswith("canh bao thua"):
        return "cảnh báo thừa"
    return "lệch mốc / khác"


def main(tag: str = "p1_v1_c4") -> None:
    rows = json.load(open(os.path.join(RES, f"k1_axes_{tag}.json"), encoding="utf-8"))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    g = collections.defaultdict(lambda: {"n": 0, "ok": 0, "ca": collections.defaultdict(lambda: {"vi": [0, 0], "en": [0, 0]}), "loi": collections.Counter()})
    for r in rows:
        h = g[hang_cua(r["ca"])]
        h["n"] += 1
        h["ok"] += bool(r["ok"])
        c = h["ca"][r["ca"]][r["lang"]]
        c[0] += bool(r["ok"])
        c[1] += 1
        for s in r["loi"]:
            for part in str(s).split("; "):
                h["loi"][loai_loi(part)] += 1
    print(f"| hàng | ca | ca đúng hết (vi và en) | tệp đúng | sai quyết định | lỗi khác |\n|---|---|---|---|---|---|")
    for _, ten in dict.fromkeys((h, h) for _, h in HANG):
        h = g[ten]
        du = sum(all(k == n for k, n in c.values() if n) for c in h["ca"].values())
        khac = ", ".join(f"{k} {v}" for k, v in h["loi"].most_common() if k != "sai quyết định")
        print(f"| {ten} | {len(h['ca'])} | {du}/{len(h['ca'])} | {h['ok']}/{h['n']} | {h['loi']['sai quyết định']} | {khac or '–'} |")
    nhan = [r for r in rows if r["decision"] in ("nhan", "canh_bao")]
    print(f"\nĐầu và cuối 0,3 s: {sum(DAU_CUOI in r['warnings'] for r in nhan)}/{len(nhan)} tệp được nhận có cảnh báo bắt buộc")
    print("thiếu một ngôn ngữ ở ca nào:", [ca for h in g.values() for ca, c in h["ca"].items() if not (c["vi"][1] and c["en"][1])] or "không")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:2])
