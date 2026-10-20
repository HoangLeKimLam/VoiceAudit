# -*- coding: utf-8 -*-
"""HOP NHAT DIEM KHUNG cua nhieu nguon OOF (FT tinh chinh + TCN dong bang ...) thanh MOT OOF moi, cung dinh dang (T, n_cot) float16 [ghep, xoa, khu on, doi giong, doan ghep, giong may]
-> dung tiep fuse9 / cmp_oof khong can sua. Moi cot lay cac nguon CO cot do (tcn: 3 cot, tcn_reg: 4 cot, ft / tcn_r2: 6 cot); do dai khung lay min.
Che do: prob (trung binh xac suat) | logit (trung binh logit - hop voi diem bao hoa gan 1) | max.
  python -m v6.boundary.ens6 --out fuse1 --src ft3:0.5 tcn_r2:0.5 [--mode logit] [--cols 6]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

from v6.boundary import detect as D  # noqa: E402

EPS = 1e-4


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def combine(arrs: list, weights: list, mode: str) -> np.ndarray:
    """arrs: cac mang (T,) cung do dai; weights cung so luong."""
    w = np.asarray(weights, np.float32) / float(sum(weights))
    if mode == "max":
        return np.max(np.stack(arrs), 0)
    if mode == "logit":
        z = sum(wi * logit(a) for wi, a in zip(w, arrs))
        return 1.0 / (1.0 + np.exp(-z))
    return sum(wi * a for wi, a in zip(w, arrs))


def fuse(srcs: list, mode: str, n_cols: int) -> dict:
    """srcs: [(oof dict, trong so)]; tra {id: (T, c) float16}, c = so cot toi da co o it nhat mot nguon (<= n_cols)."""
    ids = set.intersection(*[set(o) for o, _ in srcs])
    out = {}
    for i in ids:
        T = min(len(o[i]) for o, _ in srcs)
        cmax = min(n_cols, max(o[i].shape[1] for o, _ in srcs))
        cols = []
        for c in range(cmax):
            have = [(o[i][:T, c].astype(np.float32), w) for o, w in srcs if o[i].shape[1] > c]
            cols.append(combine([a for a, _ in have], [w for _, w in have], mode))
        out[i] = np.stack(cols, 1).astype(np.float16)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs="+", required=True, help="tag:trong_so")
    ap.add_argument("--mode", choices=["prob", "logit", "max"], default="logit")
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()
    srcs = [(pickle.load(open(D.tagged(D.OOFP, s.split(":")[0]), "rb")), float(s.split(":")[1])) for s in a.src]
    out = fuse(srcs, a.mode, a.cols)
    p = D.tagged(D.OOFP, a.out)
    pickle.dump(out, open(p + ".tmp", "wb"))
    os.replace(p + ".tmp", p)
    print(f"{len(out)} tep ({a.mode}) -> {p}", flush=True)


if __name__ == "__main__":
    main()
