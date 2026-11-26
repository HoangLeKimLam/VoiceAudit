"""Huấn nghe tay tệp thật dev mà mô hình SNR mlp_F1 chặn (luật chốt trước: NHAT_KY 2.21, seed 2513):
  16 tệp mlp_F1 chặn, WADA ≥ 10 dB (không xác nhận) · 4 tệp chặn, WADA < 10 (xác nhận) · 4 tệp không chặn. Xáo thứ tự, trang mù.
Trang đặt ở p1_tunhien_v1/_kiem/, audio trỏ về tệp gốc (../vi/…, ../en/…): không chép dữ liệu.

  .venv/Scripts/python.exe experts/P1/build_cond/nghe_tay_snr.py run | cham
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cham_v3 as C  # noqa: E402
import huan_luyen as H  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, pct  # noqa: E402
from cham_tram import cases_of  # noqa: E402
from nghe_tay import page  # noqa: E402

SEED = 2513
K = os.path.join(V6, NAT, "_kiem")
OUT, MAP, KQ = (os.path.join(K, n) for n in ("NGHE_TAY_SNR_2511.html", "NGHE_TAY_SNR_2511_map.json", "nghe_tay_snr_2511.json"))


def du_doan() -> dict:
    """mlp_F1 khớp trên cả v4 dev (như huan_luyen.bao_kem) → {tệp thật dev accept: dB dự đoán}."""
    da4, M4, A4 = H.dap_an(H.V4), H.so_do(H.V4), H.dac_trung(H.V4)
    files = sorted(f for f in da4 if f in M4 and f in A4)
    import numpy as np
    y = np.array([np.clip(da4[f]["snr"], -10, 40) for f in files])
    X, cols = H.ma_tran(files, M4, A4, "F1")
    model = H.mo_hinh("mlp", cols).fit(X, y)
    cn, MN, MNr, AN = cases_of(NAT), C.gop(NAT), H.so_do(NAT), H.dac_trung(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2 and f in AN and f in MNr]
    p = model.predict(H.ma_tran(acc, MNr, AN, "F1", cols)[0])
    return {f: (float(v), AN[f]["wada_db"]) for f, v in zip(acc, p)}


def run() -> None:
    for p in (OUT, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    d = du_doan()
    rng = random.Random(SEED)
    chan_kxn = sorted(f for f, (p, w) in d.items() if p < 10 and w >= 10)
    chan_xn = sorted(f for f, (p, w) in d.items() if p < 10 and w < 10)
    qua = sorted(f for f, (p, w) in d.items() if p >= 10)
    assert len(chan_kxn) == 16, len(chan_kxn)                     # khớp số đã báo ở 2.20 (75 chặn, 59 xác nhận)
    rows = [{"ten": f, "nhom": "chan_khong_xac_nhan"} for f in chan_kxn] + \
           [{"ten": f, "nhom": "chan_xac_nhan"} for f in rng.sample(chan_xn, 4)] + [{"ten": f, "nhom": "khong_chan"} for f in rng.sample(qua, 4)]
    rng.shuffle(rows)
    rows = [{**r, "lang": r["ten"][:2], "du_doan_db": round(d[r["ten"]][0], 2), "wada_db": d[r["ten"]][1]} for r in rows]
    assert len({r["ten"] for r in rows}) == 24 and all(os.path.exists(os.path.join(V6, NAT, r["ten"])) for r in rows)   # kiểm trước khi ghi
    html = page([{"ten": r["ten"]} for r in rows], "..", "nghe_tay_snr_2511", "(không có)")
    open(OUT, "w", encoding="utf-8").write(html)
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/nghe_tay_snr.py", "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"xong: {OUT} (24 tệp) · {MAP}")
    print("thành phần:", collections.Counter((r["nhom"], r["lang"]) for r in rows))


def cham() -> None:
    """Sau khi Huấn lưu nghe_tay_snr_2511.json: chặn oan = tệp bị chặn mà người nghe ghi on_nen khong / nhe."""
    kq = json.load(open(KQ, encoding="utf-8"))["ket_qua"]
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    for nhom in ("chan_khong_xac_nhan", "chan_xac_nhan", "khong_chan"):
        r = [x for x in rows if x["nhom"] == nhom and x["ten"] in kq]
        print(nhom, collections.Counter(kq[x["ten"]].get("on_nen") for x in r),
              "· on_nen khong/nhe:", pct(sum(kq[x["ten"]].get("on_nen") in ("khong", "nhe") for x in r), len(r)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "cham": cham}[sys.argv[1]]()
