"""Chẩn đoán hàng «Dải tần thực ≥ 6 kHz» (chỉ đọc). Với mỗi tệp T1.x hàng dải tần (p1_v1_c4): in các dải 250 Hz mà bandwidth_hz của K1
coi là «còn năng lượng», mức LTAS so với mức 1–3 kHz, và dải lớn nhất của chuỗi liền từ 0 Hz (ứng viên đo «liền mạch»).
Thêm 24 tệp gốc làm đối chứng bằng rộng.

  .venv/Scripts/python.exe experts/P1/coverage/chan_doan_bangtan.py [ca_prefix ...]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from vaudit import k1  # noqa: E402

DIR = os.path.join(ROOT, "data_labels", "v6", "k1axes", "p1_v1")
RES = os.path.join(ROOT, ".tmp_v6", "results", "k1_axes_p1_v1_c4.json")


def bang(y) -> tuple:
    """Như k1.bandwidth_hz nhưng trả cả danh sách dải đạt và mức tương đối."""
    n, SR = 1024, k1.SR
    k = len(y) // n
    S = np.abs(np.fft.rfft(y[:k * n].reshape(k, n) * np.hanning(n), axis=1)) ** 2
    ltas = 10 * np.log10(S.mean(0) + 1e-20)
    f = np.fft.rfftfreq(n, 1 / SR)
    ref = np.median(ltas[(f >= 1000) & (f <= 3000)])
    dat, muc = [], []
    for a, b in zip(np.arange(0, SR / 2, 250), np.arange(250, SR / 2 + 1, 250)):
        sel = (f >= a) & (f < b)
        lv = float(np.median(ltas[sel]) - ref)
        e = 10 * np.log10(S[:, sel].mean(1) + 1e-20)
        dyn = float(np.percentile(e, k1.BW_DYN_Q[0]) - np.percentile(e, k1.BW_DYN_Q[1]))
        muc.append(round(lv))
        if lv > -55 and dyn >= k1.BW_DYN_DB:
            dat.append(int(b))
    lien = 0
    for b in range(250, SR // 2 + 1, 250):
        if b in dat or (b + 250 in dat and b - 250 <= lien):     # cho phép hở đúng một dải
            lien = b if b in dat else lien
        else:
            break
    return (max(dat) if dat else 0), lien, dat, muc


def main(pre: tuple) -> None:
    rows = json.load(open(RES, encoding="utf-8"))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    sel = [r for r in rows if r["ca"].startswith(pre)]
    sel += [{"file": f"_goc__{b}.wav", "ca": "_goc", "base": b, "ok": True} for b in sorted({r["base"] for r in rows})]
    for r in sel:
        y = k1.decode(os.path.join(DIR, r["file"]), 1)[:, 0]
        top, lien, dat, muc = bang(np.asarray(y, float))
        tren4 = [b for b in dat if b > 4000]
        print(f"{r['ca'][8:28]:20s} {r['base']} ok={int(r['ok'])} K1={top} lien={lien} dải>4k đạt={tren4[:3]}..{tren4[-2:]} "
              f"mức 3,5–8k={muc[14:32]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(tuple(sys.argv[1:]) or ("tinhieu.g711", "tinhieu.amrnb", "tinhieu.bangtan", "tinhieu.rong"))
