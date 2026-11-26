"""NHAT_KY 2.22 (khám phá, CHỈ v4 dev fold 0–2): sai số SNR dồn vào đâu trước khi chọn bộ ước mạnh hơn.
- MAE sau hiệu chỉnh tuyến tính (OOF) của từng số đo đơn lẻ trong F1, theo loại nền.
- MAE của mlp_F1 (OOF, train v4) theo loại nền / độ dài / ngôn ngữ, chỉ tệp SNR thật 8–16 dB (vùng quyết định).

  .venv/Scripts/python.exe experts/P1/build_cond/phan_tich_sai.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import huan_luyen as H  # noqa: E402
from cham_dieu_kien import V6  # noqa: E402


def main() -> None:
    da, M = H.dap_an(H.V4), H.so_do(H.V4)
    ca = {c["file"]: c["ca"] for c in json.load(open(os.path.join(V6, H.V4, "cases.json"), encoding="utf-8"))}
    files = sorted(f for f in da if f in M)
    y = np.array([da[f]["snr"] for f in files])
    base = {f: da[f]["base"] for f in files}
    loai = np.array([ca[f].split(".")[0] for f in files])
    dur = np.array([M[f].get("nghi_s", 0) + 0 for f in files])
    X, cols = H.ma_tran(files, M, {}, "F1")
    vung = (y >= 8) & (y <= 16)
    print(f"v4 dev {len(files)} tệp · vùng 8–16 dB {vung.sum()}")

    print("\n## Số đo đơn lẻ (ridge 1 cột, OOF) · MAE dB trên vùng 8–16")
    for j, c in enumerate(cols):
        ok = ~np.isnan(X[:, j])
        if ok.sum() < 0.9 * len(files):
            continue
        p = H.oof([f for f, o in zip(files, ok) if o], y[ok], X[ok][:, [j]], [c], "ridge", base)
        e = np.abs(p - y[ok])
        print(f"{c:28s} MAE toàn bộ {e.mean():5.2f} · vùng {e[vung[ok]].mean():5.2f}")

    p = H.oof(files, y, X, cols, "mlp", base)
    e = np.abs(p - y)
    print(f"\n## mlp_F1 OOF · MAE toàn {e.mean():.2f} · vùng {e[vung].mean():.2f} · lệch TB vùng {(p - y)[vung].mean():+.2f}")
    for k in sorted(set(loai)):
        m = vung & (loai == k)
        if m.sum():
            print(f"  {k:18s} n={m.sum():4d} MAE {e[m].mean():5.2f} lệch {(p - y)[m].mean():+5.2f}")
    lang = np.array(["vi" if "_vi_" in f else "en" for f in files])
    for k in ("vi", "en"):
        m = vung & (lang == k)
        print(f"  {k:18s} n={m.sum():4d} MAE {e[m].mean():5.2f} lệch {(p - y)[m].mean():+5.2f}")
    d = np.array([M[f].get("loi_noi_bh_s") or 0 for f in files])
    for a, b in ((0, 5), (5, 10), (10, 20), (20, 31)):
        m = vung & (d >= a) & (d < b)
        if m.sum():
            print(f"  lời {a:2d}–{b:2d} s        n={m.sum():4d} MAE {e[m].mean():5.2f}")
    top = collections.Counter(base[f] for f, ee, v in zip(files, e, vung) if v and ee > 4)
    print("\nbase sai > 4 dB nhiều nhất:", top.most_common(5))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
