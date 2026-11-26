"""NHAT_KY 2.22 (khám phá, CHỈ v4 dev): sai lớn có dồn vào bản gốc vốn đã ồn không (nhãn SNR chỉ tính nền thêm vào)?
Mỗi base: lệch TB của mlp_F1 (OOF) trên tệp 8–16 dB, và SNR Brouhaha / WADA của chính tệp goc cùng base (nếu có).

  .venv/Scripts/python.exe experts/P1/build_cond/phan_tich_goc.py
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import huan_luyen as H  # noqa: E402


def main() -> None:
    da, M, A = H.dap_an(H.V4), H.so_do(H.V4), H.dac_trung(H.V4)
    files = sorted(f for f in da if f in M)
    y = np.array([da[f]["snr"] for f in files])
    X, cols = H.ma_tran(files, M, {}, "F1")
    p = H.oof(files, y, X, cols, "mlp", {f: da[f]["base"] for f in files})
    lech = collections.defaultdict(list)
    for f, pp, yy in zip(files, p, y):
        if 8 <= yy <= 16:
            lech[da[f]["base"]].append(pp - yy)
    goc = collections.defaultdict(list)
    for f in files:
        if f.startswith("goc/"):
            goc[da[f]["base"]].append((M[f].get("snr_bh_db"), A.get(f, {}).get("wada_db")))
    rows = sorted(((np.mean(v), len(v), b) for b, v in lech.items() if len(v) >= 5))
    print("base có lệch TB âm nhất (đoán thấp hơn nhãn) · goc cùng base (snr_bh, wada):")
    for m, n, b in rows[:12]:
        print(f"  {m:+6.2f} dB n={n:3d} {b[:45]:45s} {goc.get(b, '—')}")
    print("base lệch dương nhất:")
    for m, n, b in rows[-5:]:
        print(f"  {m:+6.2f} dB n={n:3d} {b[:45]:45s} {goc.get(b, '—')}")
    g = [(np.mean(lech[b]), np.mean([s for s, _ in goc[b] if s is not None])) for b in goc if len(lech.get(b, [])) >= 5]
    if len(g) > 5:
        a, s = np.array(g).T
        print(f"\nSpearman(lệch TB base, snr_bh goc) = {np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(s)))[0, 1]:+.3f} · n={len(g)}")
    print(f"tổng base {len(lech)} · base có goc trong dev {sum(1 for b in lech if b in goc)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
