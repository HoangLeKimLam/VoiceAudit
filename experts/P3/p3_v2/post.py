# -*- coding: utf-8 -*-
"""Hau xu ly diem khung tho cua P3 v2 -> diem nop (H3), ham thuan + CLI do nhanh tren OOF.

Giai ma he thong (v6.boundary.fusion.decode_regions): dong = doan lien tuc >= nguong dai 0,4-8 s; nguong = phan vi diem MAX cua tep that.
=> bien diem khung s thanh s' = mo (opening) cua trung binh truot: max(s') = muc cao nhat ma tep co mot doan >= 0,4 s vuot qua, nen
nguong dat tren tep that tinh dung "co doan dai >= 0,4 s" (gai ngan cua tep that khong day nguong len) va tep sua duoc bat khi co doan.
  smooth(s, k_mean) -> close(k_close) (lap khe ngan) -> open(K_OPEN = 21 khung > 0,4 s)
CLI: python experts/P3/p3_v2/post.py --raw .tmp_v6/cache/p3v2_raw_v1.pkl [--grid]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
K_OPEN = 21


def close1d(x: np.ndarray, k: int) -> np.ndarray:
    return x if k <= 1 else minimum_filter1d(maximum_filter1d(x, k, mode="nearest"), k, mode="nearest")


def open1d(x: np.ndarray, k: int) -> np.ndarray:
    return x if k <= 1 else maximum_filter1d(minimum_filter1d(x, k, mode="nearest"), k, mode="nearest")


def post(s: np.ndarray, k_mean: int = 5, k_close: int = 5, k_open: int = K_OPEN) -> np.ndarray:
    """(T,) diem tho -> (T,) diem nop (ham thuan)."""
    s = np.asarray(s, np.float32)
    if k_mean > 1:
        s = uniform_filter1d(s, k_mean, mode="nearest")
    return open1d(close1d(s, k_close), k_open)


def plateau(o: np.ndarray, sm: np.ndarray, alpha: float = 0.5, low: float = 0.05, max_len: int = 400, reach: int = 50) -> np.ndarray:
    """Nang bac co tre (ham thuan): moi vung lien tuc o >= low co dinh p; vung mo rong = khung lien thong voi dinh ma diem tho lam muot sm >= alpha * p
    (khong qua `reach` khung ngoai vung o >= alpha * p, tong <= max_len khung) duoc gan gia tri p. max(ket qua) = max(o) => nguong tep that khong doi;
    chi bien doan (dinh vi, Req 10 / 11) duoc lay theo noi diem tho tut, thay vi noi diem da mo cat nguong."""
    out = o.astype(np.float32).copy()
    T = len(o)
    edges = np.diff(np.r_[0, (o >= low).astype(int), 0])
    for s0, e0 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        p = float(o[s0:e0].max())
        core = np.where(o[s0:e0] >= alpha * p)[0]
        a, b = s0 + int(core[0]), s0 + int(core[-1]) + 1
        lo_lim, hi_lim = max(0, a - reach), min(T, b + reach)
        while a > lo_lim and sm[a - 1] >= alpha * p:
            a -= 1
        while b < hi_lim and sm[b] >= alpha * p:
            b += 1
        if b - a <= max_len:
            out[a:b] = np.maximum(out[a:b], p)
    return out


def post2(s: np.ndarray, k_mean: int = 9, k_close: int = 9, k_open: int = K_OPEN, alpha: float = 0.5, k_sm: int = 3) -> np.ndarray:
    """post() roi nang bac co tre theo diem tho lam muot k_sm khung (ham thuan)."""
    s = np.asarray(s, np.float32)
    return plateau(post(s, k_mean, k_close, k_open), uniform_filter1d(s, k_sm, mode="nearest") if k_sm > 1 else s, alpha)


def post_raw_mx(s: np.ndarray, k_mean: int = 9, k_close: int = 9, alpha: float = 0.5) -> np.ndarray:
    """Nhanh KHONG mo (nguong tren diem tho): max(diem tho, nang bac cua ban da mo) (ham thuan). max(ket qua) = max(diem tho) va
    vung nang bac chi quanh doan >= 21 khung da co -> khong sinh dong moi tu gai ngan, chi noi bien doan that."""
    s = np.asarray(s, np.float32)
    r = post(s, k_mean, k_close, 1)
    return np.maximum(r, plateau(post(s, k_mean, k_close, K_OPEN), uniform_filter1d(s, 3, mode="nearest"), alpha))


def build(raw: dict, ids: list, col: int, **kw) -> dict:
    """{id: diem nop} cho cot col; tep khong co diem tho -> 0 (khong ra dong)."""
    out = {}
    for i in ids:
        r = raw.get(i)
        out[i] = post(r[:, col].astype(np.float32), **kw) if r is not None else np.zeros(1, np.float32)
    return out


def selftest() -> None:
    s = np.zeros(100, np.float32)
    s[10:15] = 1.0                     # gai 5 khung: bi mo xoa
    s[40:70] = 0.8                     # doan 30 khung: giu
    p = post(s, k_mean=1, k_close=1)
    assert p[12] == 0 and abs(p[55] - 0.8) < 1e-6 and p.max() == np.float32(0.8)
    s2 = s.copy()
    s2[50:52] = 0.0                    # khe 2 khung trong doan: dong lai
    assert post(s2, k_mean=1, k_close=5)[55] > 0.79
    o = post(s, 1, 1)
    sm = s.copy()
    sm[36:40] = 0.6                    # diem tho con cao ngay truoc doan da mo
    pl = plateau(o, sm, alpha=0.5)
    assert pl.max() == o.max() and pl[37] > 0.79 and pl[30] == 0
    print("post selftest dat")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    from evalp3 import P3Eval
    raw = pickle.load(open(a.raw, "rb"))
    E = P3Eval()
    grid = [(1, 1), (5, 5), (9, 9), (5, 15), (11, 11)] if a.grid else [(5, 5)]
    for km, kc in grid:
        den = build(raw, E.ids, 0, k_mean=km, k_close=kc)
        voi = build(raw, E.ids, 1, k_mean=km, k_close=kc)
        res = E.run(den=den, voi=voi)
        print(f"--- k_mean {km} k_close {kc}")
        print(E.summary(res))
        print("   typed:", " ".join(f"{s[-2:]}:{k}={a_ / max(n, 1):.2f}" for (s, k), (a_, n) in sorted(res["typed"].items()) if ":" not in k),
              "| FA:", " ".join(f"{s[-2:]}:{k[3:]}={v:.4f}" for s in sorted(res["fa"]) for k, v in sorted(res["fa"][s].items()) if "cuc_bo" in k))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
