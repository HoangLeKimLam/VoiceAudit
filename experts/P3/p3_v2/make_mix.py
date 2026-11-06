# -*- coding: utf-8 -*-
"""Tao cac nguon gop cua ban nop p3_v2 (OOF):
  .tmp_v6/cache/p3v2_raw_ens55.pkl / ens73.pkl = sigmoid(w logit(LightGBM st2) + (1 - w) logit(TCN s1)), w = 0,5 / 0,7 (moi khung)
  experts/P3/p3_v2/heads_vi_mix.pkl            = [B nen, sigmoid(0,7 logit B giong + 0,3 logit A2 giong)] cho tep goc co .wav
CLI: python experts/P3/p3_v2/make_mix.py [--lgb st2] [--seq s1]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")


def lg(p):
    p = np.clip(np.asarray(p, np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mix(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    """Gop logit hai mang xac suat (cat ve do dai ngan hon) (ham thuan)."""
    n = min(len(a), len(b))
    return (1 / (1 + np.exp(-(w * lg(a[:n]) + (1 - w) * lg(b[:n]))))).astype(np.float16)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lgb", default="st2")
    ap.add_argument("--seq", default="s1")
    a = ap.parse_args()
    A = pickle.load(open(os.path.join(CACHE, f"p3v2_raw_{a.lgb}.pkl"), "rb"))
    B = pickle.load(open(os.path.join(CACHE, f"p3v2_seq_{a.seq}.pkl"), "rb"))["frames"]
    for w, tag in ((0.5, "ens55"), (0.7, "ens73")):
        out = {i: mix(A[i].astype(np.float32), B[i].astype(np.float32), w) for i in A if i in B}
        pickle.dump(out, open(os.path.join(CACHE, f"p3v2_raw_{tag}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("->", tag, len(out))
    r = pickle.load(open(os.path.join(CACHE, "p3v2_heads_raw.pkl"), "rb"))
    sg = lambda x: 1 / (1 + np.exp(-x))                                     # noqa: E731
    hv = {i: np.array([r["B"][n, 0], sg(0.7 * lg(r["B"][n, 1]) + 0.3 * lg(r["A2"][n, 1]))], np.float32) for n, i in enumerate(r["ids"]) if r["goc"][n]}
    pickle.dump(hv, open(os.path.join(HERE, "heads_vi_mix.pkl"), "wb"))
    print("-> heads_vi_mix", len(hv))


if __name__ == "__main__":
    main()
