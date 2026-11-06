# -*- coding: utf-8 -*-
"""P3 v2 — hoc kenh khung den (khu on chuyen nghiep) / voi (doi giong) bang LightGBM tren dac trung ngu canh (ctxfeat.py),
5 fold THEO NGUOI NOI (item["fold"] = v6.boundary.detect.person_fold): mo hinh fold f chi hoc tren tep fold != f.

Nghia cua diem khung: den(t) = P(khung t da qua khu on chuyen nghiep / manh), voi(t) = P(khung t da bi doi giong) — tep sua ca tep
la duong o MOI khung (giai ma he thong chi ra dong doan 0,4-8 s; tep dai ca tep do dau nen / giong lo).
Am: tep that (ke ca WebRTC NS <= 2, nhan 01: trong so cao), khung ngoai doan sua, tep cua thao tac kia.
Hoc them: bien the ung dung (app_*) cua cung tep (cung nguoi -> cung fold, trong so 0,5) va bo benign (tep that qua xu ly vo hai).

Ra: .tmp_v6/cache/p3v2_raw_<tag>.pkl  {id: (T, 2) float16} diem tho (chua hau xu ly) cho tep natvi / naten goc + benign co .wav.
CLI: python experts/P3/p3_v2/train_frames.py [--tag v1] [--rounds 300] [--jobs 20]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
FF = os.path.join(CACHE, "p3v2_ff")
FR = os.path.join(BASE, "data_labels", "realistic_v4", "xlsr_frames")
SETS = ("v6natvi", "v6naten", "v6benign")
DT = 0.02
IGN = 0.04


def frame_labels(meta: dict, T: int) -> tuple:
    """(y_den, y_voi, bo_qua, trong_so_ky) theo khung tu meta.edits (ham thuan). Doan: khung co tam trong [start, end];
    bo qua khung cach bien <= 40 ms. Tep sua ca tep: duong moi khung. WebRTC NS <= 2 (edits_khong_tinh): khung trong doan -> trong so cao."""
    t = (np.arange(T) + 0.5) * DT
    yd, yv, ign, ky = (np.zeros(T, bool) for _ in range(4))
    for e in meta.get("edits", []):
        l2 = e.get("level2", "")
        if not l2.startswith("02"):
            continue
        tgt = yd if "_nen_" in l2 else yv
        if e.get("kind") == "global" or "toan_cuc" in l2:
            tgt[:] = True
        else:
            a, b = float(e["start"]), float(e["end"])
            tgt |= (t >= a) & (t <= b)
            ign |= (np.abs(t - a) <= IGN) | (np.abs(t - b) <= IGN)
    for e in meta.get("edits_khong_tinh", []):
        if e.get("kind") == "segment":
            ky |= (t >= float(e["start"])) & (t <= float(e["end"]))
        else:
            ky[:] = True
    return yd, yv, ign, ky


def stack_feats(P: np.ndarray, T: int) -> np.ndarray:
    """Diem fuse1 (T, 6) -> (T, 12): ghep / xoa / den / voi tho, trung binh 11 khung, mo 21 khung (den, voi), max 51 khung (den, voi) (ham thuan)."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
    P = np.asarray(P, np.float32)[:T, :4]
    if len(P) < T:
        P = np.vstack([P, np.repeat(P[-1:], T - len(P), 0)])
    op = maximum_filter1d(minimum_filter1d(P[:, 2:4], 21, axis=0, mode="nearest"), 21, axis=0, mode="nearest")
    return np.hstack([P, uniform_filter1d(P[:, 2:4], 11, axis=0, mode="nearest"), op, maximum_filter1d(P[:, 2:4], 51, axis=0, mode="nearest"),
                      uniform_filter1d(P[:, :2], 11, axis=0, mode="nearest")])


def keep_cols(drop: bool, rel: bool) -> list:
    """Chi so cot ctxfeat giu lai: drop = bo boi canh ca tep F_*; rel = bo 40 dac trung vat ly TUYET DOI (chi giu tuong doi cuc bo - tep)."""
    import ctxfeat
    from feats import FCOLS
    return [k for k, n in enumerate(ctxfeat.names()) if not (drop and n.startswith("F_")) and not (rel and k < len(FCOLS))]


def feat_names(stack: bool, drop, rel: bool = False) -> list:
    import ctxfeat
    nm = ctxfeat.names()
    n = [nm[k] for k in keep_cols(bool(drop), rel)]
    return n + (["s_ghep", "s_xoa", "s_den", "s_voi", "s_den11", "s_voi11", "s_denO", "s_voiO", "s_denM", "s_voiM", "s_ghep11", "s_xoa11"] if stack else [])


def file_job(args: tuple):
    i, meta, cfg, real, mode = args[:5]
    P, drop = (args[5], args[6]) if len(args) > 5 else (None, False)
    rel = bool(args[7]) if len(args) > 7 else False
    import ctxfeat
    ff = np.load(os.path.join(FF, i + ".npy"))
    z = np.load(os.path.join(FR, i + ".npy"))
    X = ctxfeat.ctx_feats(ff, z)
    T = len(X)
    if drop or rel:
        X = X[:, keep_cols(drop, rel)]
    if P is not None:
        X = np.hstack([X, stack_feats(P, T)])
    if mode == "full":
        return i, X
    yd, yv, ign, ky = frame_labels(meta, T)
    whole = yd.all() or yv.all()
    step = 2 if real and cfg == "goc" else 4 if whole else 3
    sel = np.zeros(T, bool)
    sel[np.random.default_rng(zlib.crc32(i.encode())).integers(0, step)::step] = True
    sel |= (yd | yv) & ~whole                    # moi khung trong doan sua cuc bo
    sel |= ky                                    # moi khung trong doan WebRTC NS <= 2
    sel &= ~ign
    w = np.ones(T, np.float32)
    if real:
        w[:] = 6.0 if ky.any() else 3.0
        w[ky] = 10.0
    elif (yd | yv).any() and not whole:
        w[yd | yv] = 2.0
    if cfg != "goc":
        w *= 0.5
    return i, X[sel], yd[sel], yv[sel], w[sel]


def turn_plan(items: dict, ids: list, n_per_lang: int, seed: int = 0) -> list:
    """Ke hoach am tong hop "doi luot nguoi noi" A-B-A: A, B la hai tep THAT goc khac nguoi, CUNG fold, cung ngon ngu (khong ro ri fold).
    Tra [(sid, fold, [(id, khung_dau, khung_cuoi), ...])] (ham thuan tren items / do dai)."""
    rng = np.random.default_rng(seed)
    by = {}
    for i in ids:
        it = items[i]
        if it["real"] and it["cfg"] == "goc" and it["set"] in ("v6natvi", "v6naten"):
            by.setdefault((it["set"], it["fold"]), []).append(i)
    tot = {}
    for (st, f), v in by.items():
        tot[st] = tot.get(st, 0) + len(v)
    plan = []
    for (st, f), v in sorted(by.items()):
        n = int(round(n_per_lang * len(v) / tot[st]))
        for k in range(n):
            a, b = rng.choice(len(v), 2, replace=False)
            A, B = v[a], v[b]
            if items[A]["meta"].get("person") == items[B]["meta"].get("person"):
                continue
            la, lb = items[A]["_T"], items[B]["_T"]
            if la < 150 or lb < 40:
                continue
            seg = int(rng.integers(20, min(400, lb) + 1))          # doan nguoi B 0,4-8 s
            b0 = int(rng.integers(0, lb - seg + 1))
            cut = int(rng.integers(50, la - 50 + 1))
            if rng.random() < 0.5:                                   # A - B - A (chen luot)
                parts = [(A, 0, cut), (B, b0, b0 + seg), (A, cut, la)]
            else:                                                    # A - B (doi luot cuoi tep)
                parts = [(A, 0, cut), (B, b0, min(lb, b0 + seg + 200))]
            plan.append((f"turn_{st[-2:]}_{f}_{k}", f, parts))
    return plan


def synth_job(args: tuple):
    """Dac trung cua mot tep tong hop (noi khung ff / z / fuse1 cua cac phan); nhan 0 moi khung, trong so nhu tep that."""
    sid, f, parts, Ps, drop, rel, mode = args
    import ctxfeat
    ff = np.vstack([np.load(os.path.join(FF, i + ".npy"))[a:b] for i, a, b in parts])
    z = np.vstack([np.load(os.path.join(FR, i + ".npy"))[a:b] for i, a, b in parts])
    X = ctxfeat.ctx_feats(ff, z)
    T = len(X)
    if drop or rel:
        X = X[:, keep_cols(drop, rel)]
    if Ps is not None:
        P = np.vstack([np.asarray(P_, np.float32)[a:b] for P_, (i, a, b) in zip(Ps, parts)])
        X = np.hstack([X, stack_feats(P, T)])
    if mode == "full":
        return sid, X
    sel = np.zeros(T, bool)
    sel[::2] = True
    return sid, X[sel], np.zeros(sel.sum(), bool), np.zeros(sel.sum(), bool), np.full(sel.sum(), 3.0, np.float32)


def _job(t: tuple):
    return synth_job(t) if isinstance(t[2], list) else file_job(t)


def pool_ids(items: dict) -> list:
    have = {x[:-4] for x in os.listdir(FF)}
    return sorted(i for i, it in items.items() if it["set"] in SETS and i in have and os.path.exists(os.path.join(FR, i + ".npy")))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=20)
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--stack", action="store_true", help="them diem fuse1 lam dac trung (chi tep co OOF fuse1: goc + benign)")
    ap.add_argument("--drop-file-ctx", action="store_true", help="bo dac trung boi canh ca tep F_* (buoc dinh vi)")
    ap.add_argument("--rel-only", action="store_true", help="bo 40 dac trung vat ly tuyet doi (chong lech mien)")
    ap.add_argument("--aug-turns", type=int, default=0, help="so tep tong hop doi luot nguoi noi moi ngon ngu (am cung)")
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--min-leaf", type=int, default=300)
    a = ap.parse_args()
    import lightgbm as lgb
    from multiprocessing import Pool
    from v6.boundary import detect as D
    t0 = time.time()
    items = D.load_items()
    ids = pool_ids(items)
    F1 = pickle.load(open(os.path.join(CACHE, "boundary_oof_fuse1.pkl"), "rb")) if a.stack else {}
    if a.no_app or a.stack:
        ids = [i for i in ids if items[i]["cfg"] == "goc" and (not a.stack or i in F1)]
    print(f"{len(ids)} tep hoc (co .wav + khung)", flush=True)
    ext = lambda i: (F1.get(i) if a.stack else None, a.drop_file_ctx, a.rel_only)          # noqa: E731
    tasks = [(i, items[i]["meta"], items[i]["cfg"], items[i]["real"], "train", *ext(i)) for i in ids]
    Xs, YD, YV, Wt, FO, LG = [], [], [], [], [], []
    sfold = {}
    if a.aug_turns:
        for i in ids:
            items[i]["_T"] = np.load(os.path.join(FR, i + ".npy"), mmap_mode="r").shape[0]
        plan = turn_plan(items, ids, a.aug_turns)
        sfold = {sid: f for sid, f, _ in plan}
        tasks += [(sid, f, parts, [F1[i] for i, _, _ in parts] if a.stack else None, a.drop_file_ctx, a.rel_only, "train") for sid, f, parts in plan]
        print(f"+ {len(plan)} tep tong hop doi luot", flush=True)
    with Pool(a.jobs) as pool:
        for i, X, yd, yv, w in pool.imap_unordered(_job, tasks, chunksize=16):
            Xs.append(X.astype(np.float32))
            YD.append(yd)
            YV.append(yv)
            Wt.append(w)
            FO.append(np.full(len(X), sfold[i] if i in sfold else items[i]["fold"], np.int8))
    X = np.vstack(Xs)
    del Xs
    yd, yv, w, fo = np.concatenate(YD), np.concatenate(YV), np.concatenate(Wt), np.concatenate(FO)
    print(f"ma tran hoc {X.shape} · den+ {yd.mean():.3f} · voi+ {yv.mean():.3f} · {time.time() - t0:.0f}s", flush=True)
    import ctxfeat
    params = dict(objective="binary", learning_rate=0.06, num_leaves=a.leaves, min_data_in_leaf=a.min_leaf, feature_fraction=0.5, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=2.0, num_threads=a.jobs, verbose=-1, seed=7)
    ev = [i for i in ids if items[i]["cfg"] == "goc"]
    out = {}
    imp = np.zeros((2, X.shape[1]))
    for f in range(5):
        tr = fo != f
        models = []
        for k, y in enumerate((yd, yv)):
            ds = lgb.Dataset(X[tr], y[tr].astype(np.float32), weight=w[tr], feature_name=feat_names(a.stack, a.drop_file_ctx, a.rel_only), free_raw_data=True)
            m = lgb.train(params, ds, num_boost_round=a.rounds)
            imp[k] += m.feature_importance("gain")
            models.append(m)
        te = [(i, None, None, None, "full", *ext(i)) for i in ev if items[i]["fold"] == f]
        with Pool(a.jobs) as pool:
            for i, Xf in pool.imap_unordered(file_job, te, chunksize=16):
                out[i] = np.stack([m.predict(Xf, num_threads=4) for m in models], 1).astype(np.float16)
        print(f"fold {f} xong · {len(te)} tep cham · {time.time() - t0:.0f}s", flush=True)
    p = os.path.join(CACHE, f"p3v2_raw_{a.tag}.pkl")
    pickle.dump(out, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    nm = feat_names(a.stack, a.drop_file_ctx, a.rel_only)
    for k, lab in enumerate(("den", "voi")):
        top = np.argsort(-imp[k])[:25]
        print(lab, "dac trung manh:", ", ".join(f"{nm[j]}:{imp[k, j] / imp[k].sum():.3f}" for j in top))
    print("->", p, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
