# -*- coding: utf-8 -*-
"""Dau CA FILE v2 tren VECTOR GOP THO XLS-R (xlsr_pooled.py: tang 8 / 12 / 16, trung binh + do lech chuan, khong tru trung vi): 3 dau
  nen_toan   02 khu on ca file          giong_toan  02 doi giong ca file          tts_file  03 giong may ca file
Moi dau: chuan hoa -> PCA (fit trong fold hoc) -> (hoi quy logistic + HGB), diem = trung binh hai xac suat da xep hang. Fold theo NGUOI NOI.
Ra: .tmp_v6/cache/heads2_<tag>.pkl = {ids, P (n, 3)}; in AUC tung dau cho vi / en (tep goc) va so voi dau v1 (vector gop PCA 64 da tru trung vi).
  python -m v6.boundary.heads2 run [--tag h2] [--layers 12] [--pca 96]
  python -m v6.boundary.heads2 final [--tag h2]                          -> heads_final_<tag>.joblib (san pham, toan bo tap hoc)
  python -m v6.boundary.heads2 final --per-fold [--folds 0 1 2 3 4]      -> heads_final_<tag>_f{k}.joblib: hoc KHONG co fold nguoi noi k (cung tap hoc voi fold k cua run)
                                                                          + so diem voi OOF heads2_<tag>.pkl tren tep fold k (trung neu tap hoc khong doi) -> pipeline fold=k
  python -m v6.boundary.heads2 check [--tag h2] [--n-check 300]           -> .tmp_v6/results/heads2_<tag>_perfold.md (lech mo hinh fold so voi OOF)
  python -m v6.boundary.heads2 selftest                                   (du lieu gia)
LUU Y 11/10: heads_final_h2_f{k} (hoc 10/10) KHONG trung OOF heads2_h2.pkl (03/10): tap hoc / nhan da doi (nghi do manifest cover sua 07/10); xem heads2_h2_perfold.md."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from v6.boundary import detect as D  # noqa: E402
from v6.boundary import fusion as F  # noqa: E402
from v6.natural import req_check as RC  # noqa: E402

POOL = os.path.join(D.V4, "xlsr_pooled")
CACHE = os.path.join(BASE, ".tmp_v6", "cache")
LAYER_COLS = {8: slice(0, 2048), 12: slice(2048, 4096), 16: slice(4096, 6144)}
TARGETS = ("nen_toan", "giong_toan", "tts_file")


def labels(meta: dict) -> list:
    l2 = set(meta.get("level2", []))
    tts = any(e.get("op") == "may_ca_file" for e in meta.get("edits", []))
    return [int("02_nen_toan_cuc" in l2), int("02_giong_toan_cuc" in l2), int(tts)]


def load_pooled(ids: list, layers: tuple) -> np.ndarray:
    cols = [LAYER_COLS[k] for k in layers]
    X = np.zeros((len(ids), sum(c.stop - c.start for c in cols)), np.float32)
    for n, i in enumerate(ids):
        v = np.load(os.path.join(POOL, i + ".npy")).astype(np.float32)
        X[n] = np.concatenate([v[c] for c in cols])
    return X


def load_pooled_cached(ids: list, layers: tuple) -> np.ndarray:
    """Nhu load_pooled nhung luu ma tran vao .tmp_v6/cache/heads2_pool_L<tang>_<sha1 danh sach id>.npy (doc ~30k tep nho mat ~7 phut; lan sau vai giay)."""
    import hashlib
    key = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:12]
    p = os.path.join(CACHE, f"heads2_pool_L{'-'.join(map(str, layers))}_{key}.npy")
    if os.path.exists(p):
        X = np.load(p)
        if X.shape[0] == len(ids):
            return X
    X = load_pooled(ids, layers)
    np.save(p + ".tmp.npy", X)
    os.replace(p + ".tmp.npy", p)
    return X


def rank01(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Xep hang v theo phan bo cua ref (da sap xep) -> [0, 1]."""
    return np.searchsorted(np.sort(ref), v) / max(1, len(ref))


def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, n_pca: int) -> np.ndarray:
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=min(n_pca, Xtr.shape[0] - 1), random_state=0).fit(sc.transform(Xtr))
    A, B = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lr = LogisticRegression(C=0.05, max_iter=300, class_weight="balanced").fit(A, ytr)
    gb = F.fit_hgb(A, ytr, 150, 0.07)
    p1, p2 = lr.predict_proba(B)[:, 1], gb.predict_proba(B)[:, 1]
    r1, r2 = rank01(p1, lr.predict_proba(A)[:, 1]), rank01(p2, gb.predict_proba(A)[:, 1])
    return (r1 + r2) / 2


def run(tag: str, layers: tuple, n_pca: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    ids = sorted(i for i in items if os.path.exists(os.path.join(POOL, i + ".npy")))
    print(f"{len(ids)} / {len(items)} tep co vector gop tho", flush=True)
    X = load_pooled(ids, layers)
    Y = np.array([labels(items[i]["meta"]) for i in ids])
    fold = np.array([items[i]["fold"] for i in ids])
    in_train = np.array([items[i]["set"] in F.TRAIN_SETS for i in ids])
    P = np.zeros((len(ids), len(TARGETS)), np.float32)
    for f in range(5):
        tr, te = in_train & (fold != f), fold == f
        for h in range(len(TARGETS)):
            if Y[tr, h].sum() < 5:
                continue
            P[te, h] = fit_predict(X[tr], Y[tr, h], X[te], n_pca)
        print(f"  fold {f + 1}/5", flush=True)
    pickle.dump({"ids": ids, "P": P, "layers": layers}, open(os.path.join(CACHE, f"heads2_{tag}.pkl"), "wb"))
    old = pickle.load(open(os.path.join(CACHE, "fusion_v1_all.pkl"), "rb"))
    old_id = {i: n for n, i in enumerate(old["ids"])}
    L = [f"# Dau ca file v2 (vector gop tho, tang {layers}, PCA {n_pca}) - tep goc tu nhien\n", "| Dau | Tap | n duong | n that | AUC v2 | AUC v1 (PCA 64 tru trung vi) |", "|---|---|---|---|---|---|"]
    for s, name in (("v6natvi", "vi"), ("v6naten", "en")):
        sel = [k for k, i in enumerate(ids) if items[i]["set"] == s and items[i]["cfg"] == "goc"]
        real = [k for k in sel if items[ids[k]]["real"]]
        for h, t in enumerate(TARGETS):
            pos = [k for k in sel if Y[k, h]]
            if len(pos) < 5:
                continue
            a2 = RC.auc(P[pos, h], P[real, h])
            a1 = RC.auc([old["P"][old_id[ids[k]]][h] for k in pos], [old["P"][old_id[ids[k]]][h] for k in real]) if h < 2 else float("nan")
            L.append(f"| {t} | {name} | {len(pos)} | {len(real)} | **{a2:.3f}** | {a1:.3f} |")
    open(os.path.join(D.RES, f"heads2_{tag}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ mo hinh CUOI (hoc tren toan bo du lieu hoc) cho san pham v6/product
def fit_final(X: np.ndarray, y: np.ndarray, n_pca: int) -> dict:
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(n_pca, X.shape[0] - 1), random_state=0).fit(sc.transform(X))
    A = pca.transform(sc.transform(X))
    lr = LogisticRegression(C=0.05, max_iter=300, class_weight="balanced").fit(A, y)
    gb = F.fit_hgb(A, y, 150, 0.07)
    return {"sc": sc, "pca": pca, "lr": lr, "gb": gb, "ref_lr": np.sort(lr.predict_proba(A)[:, 1]), "ref_gb": np.sort(gb.predict_proba(A)[:, 1])}


def fit_final_shared(X: np.ndarray, Y: np.ndarray, n_pca: int) -> dict:
    """Nhu fit_final cho tung dau nhung chuan hoa + PCA hoc MOT lan (cung X, cung hat giong -> giong het ban hoc rieng tung dau), nhanh gap ~3 lan.
    Tra {ten dau: mo hinh} cho cac dau co >= 5 mau duong."""
    sc = StandardScaler().fit(X)
    pca = PCA(n_components=min(n_pca, X.shape[0] - 1), random_state=0).fit(sc.transform(X))
    A = pca.transform(sc.transform(X))
    out = {}
    for h, t in enumerate(TARGETS):
        y = Y[:, h]
        if y.sum() < 5:
            continue
        lr = LogisticRegression(C=0.05, max_iter=300, class_weight="balanced").fit(A, y)
        gb = F.fit_hgb(A, y, 150, 0.07)
        out[t] = {"sc": sc, "pca": pca, "lr": lr, "gb": gb, "ref_lr": np.sort(lr.predict_proba(A)[:, 1]), "ref_gb": np.sort(gb.predict_proba(A)[:, 1])}
    return out


def predict_final(m: dict, x: np.ndarray) -> np.ndarray:
    """Diem 0..1 (trung binh hai thu hang theo tap hoc) - CUNG thang voi diem OOF cua run() de dung chung nguong theo vu oan."""
    B = m["pca"].transform(m["sc"].transform(np.atleast_2d(x)))
    return (np.searchsorted(m["ref_lr"], m["lr"].predict_proba(B)[:, 1]) / len(m["ref_lr"]) + np.searchsorted(m["ref_gb"], m["gb"].predict_proba(B)[:, 1]) / len(m["ref_gb"])) / 2


def final(tag: str, layers: tuple, n_pca: int, per_fold: bool = False, folds: tuple = (0, 1, 2, 3, 4), n_check: int = 300) -> None:
    """Mo hinh cuoi (per_fold False) hoac mo hinh tung fold k (hoc tren tep TRAIN_SETS co fold nguoi noi != k = dung tap hoc cua fold k trong run())."""
    import joblib
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    items = D.load_items(keep=("edits", "level2"))
    ids = sorted(i for i in items if os.path.exists(os.path.join(POOL, i + ".npy")) and items[i]["set"] in F.TRAIN_SETS)
    if not per_fold:
        X = load_pooled(ids, layers)
        Y = np.array([labels(items[i]["meta"]) for i in ids])
        models = {t: fit_final(X, Y[:, h], n_pca) for h, t in enumerate(TARGETS) if Y[:, h].sum() >= 5}
        joblib.dump({"models": models, "layers": layers, "targets": TARGETS}, os.path.join(CACHE, f"heads_final_{tag}.joblib"), compress=3)
        print(f"da luu heads_final_{tag}.joblib: {sorted(models)} tren {len(ids)} tep", flush=True)
        return
    oof_p = os.path.join(CACHE, f"heads2_{tag}.pkl")
    oof = pickle.load(open(oof_p, "rb")) if os.path.exists(oof_p) else None
    if oof is not None:
        if tuple(oof.get("layers", layers)) != tuple(layers):
            sys.exit(f"heads2_{tag}.pkl hoc tren tang {oof.get('layers')} khac --layers {layers}")
        have = set(oof["ids"])                                      # tap hoc cua OOF = tep CO vector gop LUC chay run() (h2: truoc khi co bo aug) -> giu dung tap do
        n0, ids = len(ids), [i for i in ids if i in have]
        print(f"chi giu {len(ids)}/{n0} tep co trong heads2_{tag}.pkl (tap hoc cua OOF)", flush=True)
    X = load_pooled_cached(ids, layers)
    Y = np.array([labels(items[i]["meta"]) for i in ids])
    fold = np.array([items[i]["fold"] for i in ids])
    print(f"nap {len(ids)} vector gop xong ({time.time() - t0:.0f} s)", flush=True)
    for k in folds:
        tr = fold != k
        models = fit_final_shared(X[tr], Y[tr], n_pca)
        dst = os.path.join(CACHE, f"heads_final_{tag}_f{k}.joblib")
        joblib.dump({"models": models, "layers": layers, "targets": TARGETS, "fold": k, "fold_def": "detect.person_fold", "n_train": int(tr.sum())}, dst + ".tmp", compress=3)
        os.replace(dst + ".tmp", dst)
        msg = ""
        if oof is not None:                                         # truy vet: diem cua mo hinh fold k tren tep fold k phai = diem OOF da luu (cung tap hoc, cung hat giong)
            par = oof_parity(models, oof, items, k, layers, n_check)
            msg = f"; so voi OOF tren tep fold {k}: " + " / ".join(f"{t} trung vi {v['med']:.1e} max {v['max']:.1e}" for t, v in par.items())
        print(f"da luu {os.path.basename(dst)}: {sorted(models)} tren {int(tr.sum())} tep{msg} ({time.time() - t0:.0f} s)", flush=True)


def oof_parity(models: dict, oof: dict, items: dict, k: int, layers: tuple, n_check: int = 300) -> dict:
    """{dau: {n, med, p95, max, rho}}: lech |diem mo hinh fold k - diem OOF| tren <= n_check tep fold k (rho = tuong quan hang Spearman). Trung khi tap hoc / nhan
    luc chay run() giong luc hoc mo hinh fold; lech > 0 = tap hoc / nhan da doi (h2: nghi do manifest cover sua 07/10, sau heads2_h2.pkl ngay 03/10) -> nguong thr_fold dat tren OOF chi gan dung."""
    from scipy.stats import spearmanr
    oid = {i: n for n, i in enumerate(oof["ids"])}
    pool_k = [i for i in sorted(items) if items[i]["fold"] == k and i in oid]
    te = pool_k[::max(1, len(pool_k) // n_check)][:n_check]
    Xt = load_pooled(te, layers)
    out = {}
    for h, t in enumerate(TARGETS):
        if t not in models:
            continue
        p, o = predict_final(models[t], Xt), oof["P"][[oid[i] for i in te], h].astype(np.float64)
        d = np.abs(p - o)
        out[t] = {"n": len(te), "med": float(np.median(d)), "p95": float(np.quantile(d, 0.95)), "max": float(d.max()),
                  "rho": float(spearmanr(p, o)[0]) if np.std(p) > 0 and np.std(o) > 0 else float("nan")}
    return out


def perfold_check(tag: str, layers: tuple, n_check: int = 300) -> list:
    """Do lech heads_final_<tag>_f{k}.joblib so voi OOF heads2_<tag>.pkl -> .tmp_v6/results/heads2_<tag>_perfold.md (bang)."""
    import joblib
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    oof = pickle.load(open(os.path.join(CACHE, f"heads2_{tag}.pkl"), "rb"))
    L = [f"# Dau ca tep theo fold {tag} so voi OOF heads2_{tag}.pkl (tep fold k, <= {n_check} tep / fold)\n",
         "Lech = |diem mo hinh fold k - diem OOF| (thang 0..1 cua predict_final). Trung (lech ~ 0) chi khi tap hoc va nhan khong doi tu luc chay run().\n",
         "| fold | n hoc | dau | n | lech trung vi | p95 | toi da | Spearman |", "|---|---|---|---|---|---|---|---|"]
    for k in range(5):
        p = os.path.join(CACHE, f"heads_final_{tag}_f{k}.joblib")
        if not os.path.exists(p):
            L.append(f"| {k} | thieu {os.path.basename(p)} | | | | | | |")
            continue
        m = joblib.load(p)
        for t, v in oof_parity(m["models"], oof, items, k, layers, n_check).items():
            L.append(f"| {k} | {m.get('n_train', '?')} | {t} | {v['n']} | {v['med']:.4f} | {v['p95']:.4f} | {v['max']:.4f} | {v['rho']:.4f} |")
    open(os.path.join(D.RES, f"heads2_{tag}_perfold.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return L


# ------------------------------------------------------------------ kiem thu (du lieu gia, khong doc cache)
def _toy(n: int = 400, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 20)).astype(np.float32)
    Y = np.stack([(X[:, 0] + rng.normal(0, 0.5, n) > 1).astype(int), (X[:, 1] > 1.2).astype(int), np.zeros(n, int)], 1)
    Y[:3, 2] = 1                                                   # < 5 duong -> khong hoc dau tts_file
    return X, Y


def test_shared_fit_equals_separate_fit() -> None:
    X, Y = _toy()
    sh = fit_final_shared(X, Y, 8)
    assert sorted(sh) == ["giong_toan", "nen_toan"]
    for h, t in enumerate(TARGETS[:2]):
        sep = fit_final(X, Y[:, h], 8)
        assert np.allclose(predict_final(sh[t], X[:50]), predict_final(sep, X[:50]))


def test_final_scores_match_cv_scores_on_same_split() -> None:
    """Mo hinh 'fold k' (fit_final_shared tren fold != k) cho diem TRUNG fit_predict cua run() tren fold k -> co so cho so sanh oof_parity."""
    X, Y = _toy(500, 1)
    fold = np.arange(len(X)) % 5
    tr, te = fold != 2, fold == 2
    m = fit_final_shared(X[tr], Y[tr], 8)
    assert np.allclose(predict_final(m["nen_toan"], X[te]), fit_predict(X[tr], Y[tr, 0], X[te], 8))


def selftest() -> None:
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name, flush=True)
    print(f"{n} test dat")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "final", "check", "selftest"])
    ap.add_argument("--tag", default="h2")
    ap.add_argument("--layers", type=int, nargs="+", default=[12])
    ap.add_argument("--pca", type=int, default=96)
    ap.add_argument("--per-fold", action="store_true", help="final: mot mo hinh moi fold nguoi noi (khong thay fold do) cho danh gia ngoai mau")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-check", type=int, default=300, help="check: so tep moi fold de so voi OOF")
    a = ap.parse_args()
    if a.step == "run":
        run(a.tag, tuple(a.layers), a.pca)
    elif a.step == "final":
        final(a.tag, tuple(a.layers), a.pca, a.per_fold, tuple(a.folds))
    elif a.step == "check":
        perfold_check(a.tag, tuple(a.layers), a.n_check)
    else:
        selftest()
