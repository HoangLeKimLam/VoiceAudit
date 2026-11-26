"""Hiệu chỉnh tai người nghe (Huấn chốt 25/11, NHAT_KY 2.21): nghe mù 20 tệp v4 dev biết SNR thật → nhãn on_nen ứng với bao nhiêu dB.
Chọn (seed 2514): mỗi mức 6 / 8 / 10 / 12 / 15 / 20 dB lấy 3 tệp snr (|SNR_A thật − mức| ≤ 0,5 dB, 3 loại nền khác nhau, 3 gốc khác nhau)
+ 2 tệp goc (vi + en). Xáo thứ tự. Trang chỉ hiện số thứ tự (không tên tệp / thư mục). Không chép dữ liệu.

  .venv/Scripts/python.exe experts/P1/build_cond/nghe_hieu_chinh.py run | cham
"""
from __future__ import annotations

import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import huan_luyen as H  # noqa: E402
from cham_dieu_kien import V6, jl  # noqa: E402
from nghe_tay import page  # noqa: E402

SEED, MUC = 2514, (6, 8, 10, 12, 15, 20)
K = os.path.join(V6, H.V4, "_kiem")
OUT, MAP, KQ = (os.path.join(K, n) for n in ("NGHE_HIEU_CHINH_2511.html", "NGHE_HIEU_CHINH_2511_map.json", "nghe_hieu_chinh_2511.json"))


def chon(da: dict, ca: dict, rng: random.Random) -> list[dict]:
    rows = []
    for t in MUC:
        pool = [f for f in sorted(da) if f.startswith("snr/") and abs(da[f]["snr"] - t) <= 0.5]
        rng.shuffle(pool)
        loai, base, got = set(), set(), []
        for f in pool:
            lo = ca[f].split(".")[0]
            if lo not in loai and da[f]["base"] not in base:
                got.append({"ten": f, "muc": t, "snr": round(da[f]["snr"], 2), "loai": lo})
                loai.add(lo)
                base.add(da[f]["base"])
            if len(got) == 3:
                break
        assert len(got) == 3, (t, len(got))
        rows += got
    for lg in ("vi", "en"):
        f = rng.choice([f for f in sorted(da) if f.startswith("goc/") and f"_{lg}_" in f])
        rows.append({"ten": f, "muc": "goc", "snr": round(da[f]["snr"], 2), "loai": "goc"})
    rng.shuffle(rows)
    return rows


def run() -> None:
    for p in (OUT, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    da = H.dap_an(H.V4)
    ca = {c["file"]: c["ca"] for c in json.load(open(os.path.join(V6, H.V4, "cases.json"), encoding="utf-8"))}
    rows = chon(da, ca, random.Random(SEED))
    assert len(rows) == 20 and len({r["ten"] for r in rows}) == 20 and all(os.path.exists(os.path.join(V6, H.V4, r["ten"])) for r in rows)
    html = page([{"ten": r["ten"]} for r in rows], "..", "nghe_hieu_chinh_2511", "(không có)")
    for i, r in enumerate(rows, 1):                     # mù: ô hiển thị chỉ còn số thứ tự (data-f / src vẫn là đường dẫn)
        html = html.replace(f"<td>{i}</td><td>{r['ten']}<br>", f"<td>{i}</td><td>tệp {i}<br>", 1)
    assert not re.search(r"<td>(snr|goc)/", html)
    html = html.replace("vào <code>p1_tunhien_v1/_kiem/</code>", "vào <code>p1_dieukien_v4/_kiem/</code>")
    open(OUT, "w", encoding="utf-8").write(html)
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/nghe_hieu_chinh.py", "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"xong: {OUT} · {MAP}")
    print("thành phần:", collections.Counter((r["muc"], r["loai"]) for r in rows))


def cham() -> None:
    kq = json.load(open(KQ, encoding="utf-8"))["ket_qua"]
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    by = collections.defaultdict(list)
    for r in rows:
        by[kq.get(r["ten"], {}).get("on_nen")].append(r["snr"])
    for lab in ("khong", "nhe", "ro", "che_loi", None):
        if by[lab]:
            v = sorted(by[lab])
            print(f"{lab}: n={len(v)} · SNR thật {v}")
    print("\ntheo mức:", {t: [kq.get(r["ten"], {}).get("on_nen") for r in rows if r["muc"] == t] for t in (*MUC, "goc")})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "cham": cham}[sys.argv[1]]()
