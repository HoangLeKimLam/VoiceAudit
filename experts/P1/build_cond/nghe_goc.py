"""Kiểm nhãn bản gốc (Huấn duyệt 25/11, NHAT_KY 2.22): nghe mù 12 tệp goc v4 dev.
6 base có snr_bh_db của goc thấp nhất + 6 base đối chứng ngẫu nhiên (goc ≥ 35 dB), mỗi base 1 tệp goc. Seed 2515, xáo thứ tự,
trang chỉ hiện số thứ tự. Base nghe `ro` / `che_loi` → bản gốc ồn → ghi `_kiem/base_goc_on.json` (bỏ khỏi chấm, báo cả hai thước).

  .venv/Scripts/python.exe experts/P1/build_cond/nghe_goc.py run | cham
"""
from __future__ import annotations

import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import huan_luyen as H  # noqa: E402
from cham_dieu_kien import V6  # noqa: E402
from nghe_tay import page  # noqa: E402

SEED, N_THAP, N_DC, DC_MIN = 2515, 6, 6, 35.0
K = os.path.join(V6, H.V4, "_kiem")
OUT, MAP, KQ, BO_ON = (os.path.join(K, n) for n in ("NGHE_GOC_2511.html", "NGHE_GOC_2511_map.json", "nghe_goc_2511.json", "base_goc_on.json"))


def chon(da: dict, M: dict, rng: random.Random) -> list[dict]:
    goc = {}
    for f in sorted(da):
        if f.startswith("goc/") and M.get(f, {}).get("snr_bh_db") is not None:
            b = da[f]["base"]
            if b not in goc or M[f]["snr_bh_db"] < goc[b][1]:
                goc[b] = (f, M[f]["snr_bh_db"])
    thap = sorted(goc.items(), key=lambda t: t[1][1])[:N_THAP]
    sach = sorted(b for b, (_, s) in goc.items() if s >= DC_MIN)
    dc = rng.sample(sach, N_DC)
    rows = [{"ten": goc[b][0], "base": b, "snr_bh": goc[b][1], "nhom": "thap"} for b, _ in thap] + \
           [{"ten": goc[b][0], "base": b, "snr_bh": goc[b][1], "nhom": "doi_chung"} for b in dc]
    rng.shuffle(rows)
    return rows


def run() -> None:
    for p in (OUT, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rows = chon(H.dap_an(H.V4), H.so_do(H.V4), random.Random(SEED))
    assert len(rows) == 12 and len({r["base"] for r in rows}) == 12                     # kiểm TRƯỚC khi ghi
    assert all(os.path.exists(os.path.join(V6, H.V4, r["ten"])) for r in rows)
    html = page([{"ten": r["ten"]} for r in rows], "..", "nghe_goc_2511", "(không có)")
    for i, r in enumerate(rows, 1):
        html = html.replace(f"<td>{i}</td><td>{r['ten']}<br>", f"<td>{i}</td><td>tệp {i}<br>", 1)
    assert not re.search(r"<td>goc/", html)
    html = html.replace("vào <code>p1_tunhien_v1/_kiem/</code>", "vào <code>p1_dieukien_v4/_kiem/</code>")
    open(OUT, "w", encoding="utf-8").write(html)
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/nghe_goc.py", "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"xong: {OUT}")
    print("nhóm:", {n: sum(r["nhom"] == n for r in rows) for n in ("thap", "doi_chung")})


def cham() -> None:
    if os.path.exists(BO_ON):
        raise SystemExit(f"Đã có {BO_ON}; không ghi đè")
    kq = json.load(open(KQ, encoding="utf-8"))["ket_qua"]
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    for r in sorted(rows, key=lambda r: r["snr_bh"]):
        print(f"{r['nhom']:9s} snr_bh {r['snr_bh']:6.2f} · {kq.get(r['ten'], {}).get('on_nen')} · {r['base']}")
    on = sorted(r["base"] for r in rows if kq.get(r["ten"], {}).get("on_nen") in ("ro", "che_loi"))
    json.dump({"luat": "NHAT_KY 2.22: goc nghe ro / che_loi → bản gốc ồn", "base": on}, open(BO_ON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("bản gốc ồn:", on)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "cham": cham}[sys.argv[1]]()
