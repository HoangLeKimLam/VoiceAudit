# -*- coding: utf-8 -*-
"""P3 v2 — dau CA TEP nen (02_nen_toan_cuc) / giong (02_giong_toan_cuc, gom chuyen giong AI) hoc 5 fold THEO NGUOI NOI.

Hai nhanh, roi ghep bang hoi quy logistic tren diem OOF (cung 5 fold; mo hinh ghep fold f chi hoc tren OOF cua tep fold != f):
  A. LightGBM tren thong ke VAT LY ca tep tu dac trung khung (feats.py): phan vi nang luong, san nhieu 16 dai, khung lang (do phang,
     lo pho, nang luong tuyet doi), bang thong (RO95 / RO99, H4 / H6 / H7 tren khung noi), F0 / do bat tuan hoan, MFCC, nhay bac (gating).
  B. Hoi quy logistic tren vector gop XLS-R (6144) (giong p3_v1).
Hoc tren: tep natvi / naten (goc + bien the app_*) + benign co .wav; tep chi sua CUC BO (doan) khong dung cho dau ca tep (khong duong, khong am).
Am cung: WebRTC NS <= 2 (nhan 01) va benign — trong so cao.
Ra: experts/P3/p3_v2/heads_p3v2[_tag].pkl {id: (2,) float32} cho tep natvi / naten goc (+ benign) co .wav;
    tep khong co .wav giu diem h2 nen (bao so tep).
CLI: python experts/P3/p3_v2/train_heads.py [--tag ""] [--jobs 20]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
FF = os.path.join(CACHE, "p3v2_ff")
POOL = os.path.join(BASE, "data_labels", "realistic_v4", "xlsr_pooled")
SETS = ("v6natvi", "v6naten", "v6benign")


def file_stats(ff: np.ndarray) -> np.ndarray:
    """(T, 40) -> vector thong ke ca tep (ham thuan). Ten: stat_names()."""
    from feats import COL
    from scipy.ndimage import minimum_filter1d
    ff = ff.astype(np.float64)
    B, E = ff[:, :16], ff[:, COL["E"]]
    q = lambda a, p: np.quantile(a, p, axis=0)                             # noqa: E731
    eq = q(E, [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98])
    low = E <= np.quantile(E, 0.2)
    sp = E >= np.quantile(E, 0.6)
    V = ff[:, COL["V"]] > 0.5
    fl = q(B, 0.1)
    bl = B[low].mean(0) if low.sum() > 2 else fl
    bs = B[sp].mean(0) if sp.sum() > 2 else q(B, 0.9)
    bstd = B[low].std(0) if low.sum() > 2 else np.zeros(16)
    m25 = minimum_filter1d(E, 25, mode="nearest")
    dE = np.abs(np.diff(E))
    sel = lambda c, m: ff[m, COL[c]] if m.sum() > 2 else ff[:, COL[c]]    # noqa: E731
    st = ff[V, COL["F0"]] if V.sum() > 5 else np.zeros(1)
    dst = np.abs(np.diff(ff[:, COL["F0"]]))[(V[1:] & V[:-1])] if (V[1:] & V[:-1]).sum() > 5 else np.zeros(1)
    M = ff[:, COL["M1"]:COL["M12"] + 1]
    out = np.r_[
        eq, eq[6] - eq[2], [np.mean(E < t) for t in (-90, -80, -70, -60)],
        fl, bl - bs, bs - bs.mean(), bstd, [bstd.mean()],
        [m25.std(), np.quantile(m25, 0.9) - np.quantile(m25, 0.1), np.mean(dE > 15), np.mean(dE > 25), np.quantile(dE, 0.99)],
        [sel("FLAT", low).mean(), sel("FLAT", low).std(), sel("FLAT", sp).mean(), sel("HOLE", low).mean(), sel("HOLE", sp).mean(),
         np.quantile(sel("HOLE", low), 0.9)],
        [np.quantile(sel(c, sp), p) for c in ("RO95", "RO99", "H4", "H6", "H7", "CEN") for p in (0.1, 0.5, 0.9)],
        [np.quantile(sel(c, low), 0.5) for c in ("H4", "H6", "H7", "RO99")],
        [np.median(st), np.quantile(st, 0.9) - np.quantile(st, 0.1), V.mean(), np.median(sel("AP", V)), np.quantile(sel("AP", V), 0.9),
         np.median(dst), np.quantile(dst, 0.9)],
        M.mean(0), M.std(0), M[sp].mean(0) if sp.sum() > 2 else M.mean(0), M[low].mean(0) if low.sum() > 2 else M.mean(0),
        [np.log(len(ff) * 0.02)],
    ]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=-100.0).astype(np.float32)


def stat_job(i: str):
    return i, file_stats(np.load(os.path.join(FF, i + ".npy")))


def head_y(meta: dict) -> tuple:
    """(y_nen, y_giong, co_sua_cuc_bo_02) (ham thuan)."""
    l2 = set(meta.get("level2", []))
    loc = any(e.get("level2", "").startswith("02") and e.get("kind") == "segment" for e in meta.get("edits", []))
    return int("02_nen_toan_cuc" in l2), int("02_giong_toan_cuc" in l2), loc


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--jobs", type=int, default=20)
    a = ap.parse_args()
    import lightgbm as lgb
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from v6.boundary import detect as D
    t0 = time.time()
    items = D.load_items()
    have = {x[:-4] for x in os.listdir(FF)}
    ids = sorted(i for i, it in items.items() if it["set"] in SETS and i in have and os.path.exists(os.path.join(POOL, i + ".npy")))
    with Pool(a.jobs) as pool:
        S = dict(pool.imap_unordered(stat_job, ids, chunksize=32))
    XS = np.stack([S[i] for i in ids])
    XP = np.stack([np.load(os.path.join(POOL, i + ".npy")).astype(np.float32) for i in ids])
    XP = np.nan_to_num(XP)
    print(f"{len(ids)} tep · thong ke {XS.shape[1]} · gop {XP.shape[1]} · {time.time() - t0:.0f}s", flush=True)
    fo = np.array([items[i]["fold"] for i in ids])
    goc = np.array([items[i]["cfg"] == "goc" for i in ids])
    real = np.array([items[i]["real"] for i in ids])
    Y = np.array([head_y(items[i]["meta"]) for i in ids])
    hardneg = real & np.array([items[i]["kind"].startswith("nen_") or items[i]["set"] == "v6benign" for i in ids])
    oofA = np.zeros((len(ids), 2), np.float32)
    oofB = np.zeros((len(ids), 2), np.float32)
    params = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.6, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=5.0, num_threads=a.jobs, verbose=-1, seed=3)
    for k in range(2):
        use = Y[:, 2] == 0                                                  # bo tep chi sua cuc bo 02
        y = Y[:, k]
        w = np.where(real, 3.0, 1.0) * np.where(hardneg, 3.0, 1.0) * np.where(goc, 1.0, 0.5)
        for f in range(5):
            tr = use & (fo != f)
            te = fo == f
            m = lgb.train(params, lgb.Dataset(XS[tr], y[tr], weight=w[tr]), num_boost_round=600)
            oofA[te, k] = m.predict(XS[te])
            sc = StandardScaler().fit(XP[tr])
            lr = LogisticRegression(C=0.005, max_iter=600)
            lr.fit(sc.transform(XP[tr]), y[tr], sample_weight=w[tr])
            oofB[te, k] = lr.predict_proba(sc.transform(XP[te]))[:, 1]
        print(f"dau {k} xong · {time.time() - t0:.0f}s", flush=True)
    out = {"ids": ids, "A": oofA, "B": oofB, "fold": fo, "goc": goc, "real": real, "Y": Y}
    p = os.path.join(CACHE, f"p3v2_heads_raw{a.tag}.pkl")
    pickle.dump(out, open(p, "wb"))
    print("->", p, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
