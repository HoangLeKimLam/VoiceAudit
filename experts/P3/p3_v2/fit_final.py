# -*- coding: utf-8 -*-
"""Hoc mo hinh CUOI cua P3 v2 tren TOAN BO du lieu (khong giu fold — chi de san pham chay score.py; moi so do la OOF) -> weights/.
TCN chuoi hoc rieng: python experts/P3/p3_v2/train_seq.py --final --epochs 10   (-> weights/seq.pt, seq_norm.npz)

Cau hinh = ban OOF da chon (card.json, compose.py):
  khung : LightGBM (ctxfeat bo F_* + bo 40 tuyet doi, xep chong fuse1) — den.txt / voi.txt ; gop logit voi TCN theo FRAME_MIX
  dau   : A2 = LightGBM [pho khung lang/noi + thong ke tep] (nenA.txt, giongA.txt) · B = LR tren XLS-R gop (headsB.npz)
          B2 = LR [XLS-R gop + thong ke khung XLS-R lang/noi + h2 + thong ke fuse1] (+ pho 640 cho giong DSP) (headsB2.npz)
          vi: nen = B ; giong = sigmoid(0,7 logit B + 0,3 logit A2) · en: nen = B2 ; giong = 0,15 h2 + 0,85 B2-DSP
Ra: weights/den.txt voi.txt nenA.txt giongA.txt headsB.npz headsB2.npz final.json.  CLI: python experts/P3/p3_v2/fit_final.py [--jobs 8]"""
from __future__ import annotations

import argparse
import json
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
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
W = os.path.join(HERE, "weights")
FINAL = {
    "frame_mix": {"den": {"vi": 0.5, "en": 0.5}, "voi": {"vi": 0.5, "en": 0.7}},          # trong so logit LightGBM (con lai: TCN)
    "post": {"den": {"vi": ["post2", 9, 9, 21], "en": ["post2", 9, 9, 21]}, "voi": {"vi": ["raw_mx", 9, 9], "en": ["post2", 9, 9, 21]}},
    "heads": {"vi": {"nen": "B", "giong": "0.7B+0.3A2"}, "en": {"nen": "B2", "giong": "0.15h2+0.85B2dsp"}},
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import warnings
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=300)
    a = ap.parse_args()
    import lightgbm as lgb
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from train_frames import feat_names, file_job, pool_ids
    from train_heads import stat_job
    from train_heads_b2 import zf_stats
    from v6.boundary import detect as D
    os.makedirs(W, exist_ok=True)
    t0 = time.time()
    items = D.load_items()
    F1 = pickle.load(open(os.path.join(CACHE, "boundary_oof_fuse1.pkl"), "rb"))
    # ---- khung
    ids = [i for i in pool_ids(items) if items[i]["cfg"] == "goc" and i in F1]
    tasks = [(i, items[i]["meta"], items[i]["cfg"], items[i]["real"], "train", F1[i], True, True) for i in ids]
    Xs, YD, YV, Wt = [], [], [], []
    with Pool(a.jobs) as pool:
        for i, X, yd, yv, w in pool.imap_unordered(file_job, tasks, chunksize=16):
            Xs.append(X.astype(np.float32)); YD.append(yd); YV.append(yv); Wt.append(w)
    X, yd, yv, w = np.vstack(Xs), np.concatenate(YD), np.concatenate(YV), np.concatenate(Wt)
    del Xs
    prm = dict(objective="binary", learning_rate=0.06, num_leaves=63, min_data_in_leaf=300, feature_fraction=0.5, bagging_fraction=0.7,
               bagging_freq=1, lambda_l2=2.0, num_threads=a.jobs, verbose=-1, seed=7)
    for nm, y in (("den", yd), ("voi", yv)):
        lgb.train(prm, lgb.Dataset(X, y.astype(np.float32), weight=w, feature_name=feat_names(True, True, True)), num_boost_round=a.rounds).save_model(
            os.path.join(W, f"{nm}.txt"))
    del X
    print(f"khung xong {time.time() - t0:.0f}s", flush=True)
    # ---- dau A2, B (cung tap hoc voi train_heads / train_heads2)
    raw = pickle.load(open(os.path.join(CACHE, "p3v2_heads_raw.pkl"), "rb"))
    hid = raw["ids"]
    pz = pickle.load(open(os.path.join(CACHE, "p3v2_pause.pkl"), "rb"))
    pm = {i: n for n, i in enumerate(pz["ids"])}
    with Pool(a.jobs) as pool:
        S = dict(pool.imap_unordered(stat_job, hid, chunksize=32))
    PZ = np.stack([pz["X"][pm[i]] for i in hid])
    XS = np.hstack([PZ, np.stack([S[i] for i in hid])])
    XP = np.nan_to_num(np.stack([np.load(os.path.join(V4, "xlsr_pooled", i + ".npy")).astype(np.float32) for i in hid]))
    real, goc, Y = raw["real"], raw["goc"], raw["Y"]
    hard = real & np.array([items[i]["kind"].startswith("nen_") or items[i]["set"] == "v6benign" for i in hid])
    wh = np.where(real, 3.0, 1.0) * np.where(hard, 3.0, 1.0) * np.where(goc, 1.0, 0.5)
    use = Y[:, 2] == 0
    prmA = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=5.0, num_threads=a.jobs, verbose=-1, seed=5)
    sc = StandardScaler().fit(XP[use])
    Bz = {"mu": sc.mean_, "sd": sc.scale_}
    wB = np.where(real, 3.0, 1.0) * np.where(goc, 1.0, 0.5)
    for k, nm in ((0, "nenA"), (1, "giongA")):
        lgb.train(prmA, lgb.Dataset(XS[use], Y[use, k], weight=wh[use]), num_boost_round=800).save_model(os.path.join(W, f"{nm}.txt"))
        lr = LogisticRegression(C=0.005, max_iter=600).fit(sc.transform(XP[use]), Y[use, k], sample_weight=wB[use] * np.where(hard[use], 3.0, 1.0))
        Bz[f"w{k}"], Bz[f"b{k}"] = lr.coef_[0], lr.intercept_[0]
    np.savez(os.path.join(W, "headsB.npz"), **Bz)
    print(f"dau A2 / B xong {time.time() - t0:.0f}s", flush=True)
    # ---- dau B2 (tren tep cham natvi / naten goc nhu train_heads_b2)
    h2 = pickle.load(open(os.path.join(CACHE, "heads2_h2.pkl"), "rb"))
    h2m = {i: n for n, i in enumerate(h2["ids"])}
    ev = [i for i in hid if items[i]["cfg"] == "goc" and items[i]["set"] in ("v6natvi", "v6naten") and i in h2m]
    ev_all = [i for i, it in items.items() if it["set"] in ("v6natvi", "v6naten") and it["cfg"] == "goc" and i in F1 and i in h2m]
    XZ = np.nan_to_num(np.stack([zf_stats(np.load(os.path.join(V4, "xlsr_frames", i + ".npy")), F1[i], h2["P"][h2m[i]]) for i in ev_all]))
    XPe = np.nan_to_num(np.stack([np.load(os.path.join(V4, "xlsr_pooled", i + ".npy")).astype(np.float32) for i in ev_all]))
    XW = np.stack([pz["X"][pm[i]][:640] if i in pm else np.zeros(640, np.float32) for i in ev_all])
    re_ = np.array([items[i]["real"] for i in ev_all])
    kd = np.array([items[i]["kind"] for i in ev_all])
    yn = np.array([int(any(e.get("level2") == "02_nen_toan_cuc" for e in items[i]["meta"].get("edits", []))) for i in ev_all])
    ydsp, yai = (~re_) & (kd == "giong_toan_cuc"), (~re_) & (kd == "doi_giong_ai")
    Xn = np.hstack([XPe, XZ])
    scn = StandardScaler().fit(Xn)
    wn = np.where(re_, 8.0, 1.0)
    wn[re_ & (kd == "nen_toan_cuc")] = 25.0
    lrn = LogisticRegression(C=0.005, max_iter=500).fit(scn.transform(Xn), yn, sample_weight=wn)
    sp = StandardScaler().fit(XPe)
    rg = RobustScaler().fit(np.hstack([XZ, XW]))
    Xg = np.hstack([sp.transform(XPe), np.clip(rg.transform(np.hstack([XZ, XW])), -8, 8) * 2.5])
    wg = np.where(re_, 10.0, 1.0)
    H1 = np.array([h2["P"][h2m[i]][1] for i in ev_all])
    wg[re_ & (H1 > 0.3)] = 40.0
    m1 = ~yai
    lrd = LogisticRegression(C=0.005, max_iter=500).fit(Xg[m1], ydsp[m1].astype(int), sample_weight=wg[m1])
    np.savez(os.path.join(W, "headsB2.npz"), n_mu=scn.mean_, n_sd=scn.scale_, n_w=lrn.coef_[0], n_b=lrn.intercept_[0], p_mu=sp.mean_, p_sd=sp.scale_,
             r_c=rg.center_, r_s=rg.scale_, d_w=lrd.coef_[0], d_b=lrd.intercept_[0])
    cfg = dict(FINAL, frames={"drop_file_ctx": True, "rel_only": True, "stack_fuse1": True, "rounds": a.rounds},
               train_files={"frames": len(ids), "heads_A2_B": int(use.sum()), "heads_B2": len(ev_all)}, time_s=round(time.time() - t0))
    json.dump(cfg, open(os.path.join(W, "final.json"), "w", encoding="utf-8"), indent=1)
    print("->", W, cfg["train_files"], f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
