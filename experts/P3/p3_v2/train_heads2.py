# -*- coding: utf-8 -*-
"""P3 v2 — dau ca tep nhanh A2 (LightGBM) va C (LR) tren pho khung lang / noi chi tiet (feats_pause.py) + thong ke ca tep (train_heads.file_stats),
5 fold THEO NGUOI NOI, cung quy uoc nhan / trong so voi train_heads.py (bo tep chi sua cuc bo). Gop vao tep tho cua train_heads (them khoa A2, C).
CLI: python experts/P3/p3_v2/train_heads2.py [--jobs 8]"""
import argparse, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE); sys.path.insert(0, HERE)
import numpy as np
CACHE = os.path.join(BASE, ".tmp_v6", "cache")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    import lightgbm as lgb
    from multiprocessing import Pool
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from train_heads import stat_job
    from v6.boundary import detect as D
    t0 = time.time()
    raw = pickle.load(open(os.path.join(CACHE, "p3v2_heads_raw.pkl"), "rb"))
    ids = raw["ids"]
    pz = pickle.load(open(os.path.join(CACHE, "p3v2_pause.pkl"), "rb"))
    pm = {i: n for n, i in enumerate(pz["ids"])}
    XPa = np.stack([pz["X"][pm[i]] for i in ids])
    with Pool(a.jobs) as pool:
        S = dict(pool.imap_unordered(stat_job, ids, chunksize=32))
    XS = np.hstack([XPa, np.stack([S[i] for i in ids])])
    items = D.load_items(keep=("level2",))
    fo, goc, real, Y = raw["fold"], raw["goc"], raw["real"], raw["Y"]
    hardneg = real & np.array([items[i]["kind"].startswith("nen_") or items[i]["set"] == "v6benign" for i in ids])
    w = np.where(real, 3.0, 1.0) * np.where(hardneg, 3.0, 1.0) * np.where(goc, 1.0, 0.5)
    use = Y[:, 2] == 0
    A2, C = np.zeros((len(ids), 2)), np.zeros((len(ids), 2))
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
               bagging_freq=1, lambda_l2=5.0, num_threads=a.jobs, verbose=-1, seed=5)
    for k in range(2):
        for f in range(5):
            tr, te = use & (fo != f), fo == f
            m = lgb.train(prm, lgb.Dataset(XS[tr], Y[tr, k], weight=w[tr]), num_boost_round=800)
            A2[te, k] = m.predict(XS[te])
            sc = StandardScaler().fit(XPa[tr])
            lr = LogisticRegression(C=0.01, max_iter=800).fit(np.clip(sc.transform(XPa[tr]), -6, 6), Y[tr, k], sample_weight=w[tr])
            C[te, k] = lr.predict_proba(np.clip(sc.transform(XPa[te]), -6, 6))[:, 1]
        print(f"dau {k} xong {time.time() - t0:.0f}s", flush=True)
    raw["A2"], raw["C"] = A2.astype(np.float32), C.astype(np.float32)
    pickle.dump(raw, open(os.path.join(CACHE, "p3v2_heads_raw.pkl"), "wb"))
    print("-> them A2, C")


if __name__ == "__main__":
    main()
