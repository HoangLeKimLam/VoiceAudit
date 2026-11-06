# -*- coding: utf-8 -*-
"""P3 v2 — nhanh dau ca tep B2: tai tao TAI DUOC phuong phap dau cua p3_v1 (train_heads.py cu), thay dac trung .wav mat ma (G / Svoi cua
p3_wav_feats.pkl) bang dac trung co ma (feats_pause.py), 5 fold THEO NGUOI NOI tren tep cham natvi / naten goc (Ctx9.ids).

  X = XLS-R gop (6144) + thong ke khung XLS-R lang / noi (z_q mean, std, z_a - z_q, ty le std) + diem dau h2 + thong ke diem khung fuse1
      (+ pho khung lang / noi 640 cho giong)
  nen  : LR(C=0,005) nhan 02_nen_toan_cuc, trong so tep that x8 (WebRTC NS <= 2 ca tep x25)
  giong: hai LR rieng (doi giong DSP; chuyen giong AI), tron theo ngon ngu nhu p3_v1 (en: 0,15 h2 + 0,85 DSP; vi: hieu chinh 0,4 h2 + 0,4 DSP + 0,2 AI)
Ra: experts/P3/p3_v2/heads_b2.pkl {id: (2,)} cho Ctx9.ids. CLI: python experts/P3/p3_v2/train_heads_b2.py"""
from __future__ import annotations

import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")


def zf_stats(z: np.ndarray, pr: np.ndarray, hp: np.ndarray) -> np.ndarray:
    """Nhu p3_v1.extract_zf_stats (ham thuan): khung lang / noi theo chuan buoc nhay XLS-R + thong ke diem fuse1 + h2."""
    z = z.astype(np.float32)
    act = np.linalg.norm(np.abs(np.diff(z, axis=0, prepend=z[:1])), axis=1)
    q = act <= np.quantile(act, 0.25)
    a = act >= np.quantile(act, 0.60)
    q = q if q.sum() >= 2 else np.ones(len(z), bool)
    a = a if a.sum() >= 2 else np.ones(len(z), bool)
    pr = pr.astype(np.float32)
    st = np.r_[hp, pr.mean(0), pr.std(0), pr.max(0), np.quantile(pr, 0.5, axis=0), np.quantile(pr, 0.9, axis=0), np.quantile(pr, 0.98, axis=0)]
    return np.r_[z[q].mean(0), z[q].std(0), z[a].mean(0) - z[q].mean(0), z[q].std(0) / (z[a].std(0) + 1e-4), st]


def calibrate_oof(p: np.ndarray, real: np.ndarray, langs: np.ndarray, folds: np.ndarray, q: float = 0.0025) -> np.ndarray:
    """Hieu chinh theo (ngon ngu, fold) chi dung tep THAT cua fold KHAC (nhu p3_v1) (ham thuan)."""
    out = np.zeros_like(p, dtype=np.float32)
    for lg in ("vi", "en"):
        for f in range(5):
            r = real & (langs == lg) & (folds != f)
            te = (langs == lg) & (folds == f)
            if r.sum() < 5 or te.sum() == 0:
                continue
            th, med = float(np.quantile(p[r], 1 - q)), float(np.quantile(p[r], 0.5))
            out[te] = 1 / (1 + np.exp(-2.0 * np.clip((p[te] - th) / max(th - med, 1e-5), -15, 15)))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from v6.boundary import fuse9 as F9
    from v6.boundary import fusion as F
    ctx = F9.Ctx9("fuse1", "fuse1", "h2", F.NAT_SETS)
    ids = ctx.ids
    N = len(ids)
    folds = np.array([ctx.fold[i] for i in ids])
    langs = np.array([ctx.lang[i] for i in ids])
    real = np.array([ctx.items[i]["real"] for i in ids])
    kinds = np.array([ctx.items[i]["kind"] for i in ids])
    y_nen = np.array([int(any(e.get("level2") == "02_nen_toan_cuc" for e in ctx.items[i]["meta"].get("edits", []))) for i in ids])
    y_dsp, y_ai = (~real) & (kinds == "giong_toan_cuc"), (~real) & (kinds == "doi_giong_ai")
    XZ = np.nan_to_num(np.stack([zf_stats(np.load(os.path.join(V4, "xlsr_frames", i + ".npy")), ctx.PB[i], ctx.HP[i]) for i in ids]))
    XP = np.nan_to_num(np.stack([np.load(os.path.join(V4, "xlsr_pooled", i + ".npy")).astype(np.float32) for i in ids]))
    pz = pickle.load(open(os.path.join(CACHE, "p3v2_pause.pkl"), "rb"))
    pm = {i: n for n, i in enumerate(pz["ids"])}
    XW = np.stack([pz["X"][pm[i]][:640] if i in pm else np.zeros(640, np.float32) for i in ids])
    H = np.stack([ctx.HP[i][:2] for i in ids]).astype(np.float32)
    print(f"{N} tep cham · XZ {XZ.shape[1]} · XP {XP.shape[1]} · XW {XW.shape[1]}", flush=True)
    o_nen, o_dsp, o_ai = (np.zeros(N, np.float32) for _ in range(3))
    for f in range(5):
        tr, te = folds != f, folds == f
        Xn = np.hstack([XP, XZ])
        sc = StandardScaler().fit(Xn[tr])
        w = np.where(real[tr], 8.0, 1.0)
        w[real[tr] & (kinds[tr] == "nen_toan_cuc")] = 25.0
        o_nen[te] = LogisticRegression(C=0.005, max_iter=500).fit(sc.transform(Xn[tr]), y_nen[tr], sample_weight=w).predict_proba(sc.transform(Xn[te]))[:, 1]
        sp = StandardScaler().fit(XP[tr])
        rg = RobustScaler().fit(np.hstack([XZ[tr], XW[tr]]))
        g = lambda m: np.hstack([sp.transform(XP[m]), np.clip(rg.transform(np.hstack([XZ[m], XW[m]])), -8, 8) * 2.5])     # noqa: E731
        Xtr, Xte = g(tr), g(te)
        w = np.where(real[tr], 10.0, 1.0)
        w[real[tr] & (H[tr, 1] > 0.3)] = 40.0
        m1 = ~y_ai[tr]
        o_dsp[te] = LogisticRegression(C=0.005, max_iter=500).fit(Xtr[m1], y_dsp[tr][m1].astype(int), sample_weight=w[m1]).predict_proba(Xte)[:, 1]
        m2 = ~y_dsp[tr]
        o_ai[te] = LogisticRegression(C=0.01, max_iter=500).fit(Xtr[m2], y_ai[tr][m2].astype(int), sample_weight=w[m2]).predict_proba(Xte)[:, 1]
        print(f"fold {f} xong", flush=True)
    c_h2, c_dsp, c_ai = (calibrate_oof(p, real, langs, folds) for p in (H[:, 1], o_dsp, o_ai))
    giong = np.where(langs == "en", 0.15 * H[:, 1] + 0.85 * o_dsp, 0.40 * c_h2 + 0.40 * c_dsp + 0.20 * c_ai).astype(np.float32)
    out = {i: np.array([o_nen[n], giong[n]], np.float32) for n, i in enumerate(ids)}
    pickle.dump(out, open(os.path.join(HERE, "heads_b2.pkl"), "wb"))
    print("->", os.path.join(HERE, "heads_b2.pkl"), len(out))


if __name__ == "__main__":
    main()
