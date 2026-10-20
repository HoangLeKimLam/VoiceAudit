# -*- coding: utf-8 -*-
"""v6c / B2 - BO DO RANH GIOI THEO KHUNG 20 ms tren XLS-R L12 (da nen PCA 64) + GIAI MA THANH DONG THAO TAC + GHEP VOI v6b.

Lop khung: 0 khong · 1 ranh gioi ghep/chen (moc dau + cuoi doan chen; giong may v4) · 2 moi noi xoa loi · 3 ranh gioi sua 02 cuc bo
(khu on / doi giong mot doan: lop 'nhieu' de bo phan loai khong nham). Nhan khung = 1 neu cach moc that <= 60 ms.
Dac trung khung: z (64) + log1p(buoc nhay chuan hoa theo trung vi tep) (64) + tuong phan cuc bo +-5 khung (64) = 192.
Hoc HistGradientBoosting (khung duong + 4 % khung am), 5 fold theo nguoi noi (crc32 % 5, v4: speaker_fold) -> diem khung OOF.
Giai ma: lam tron 3 khung, dinh (>= nguong, cach nhau >= 0,3 s); nguong = phan vi 1 - FPR cua diem max tren tep THAT cua cac fold
KHAC fold dang xet. Hai dinh ghep cach <= 5 s gop thanh MOT dong (start-end); xoa loi: dong +-0,25 s quanh moi noi; toi da 2 dong
moi 14 s tep. Ghep voi v6b: nhan = 04 neu K1 04, nguoi lai 03 neu co dong 03 nao (v6b hoac bo do moi), nguoi lai nhan v6b.
  python -m v6.boundary.detect train            hoc + diem OOF   -> .tmp_v6/cache/boundary_oof.pkl
  python -m v6.boundary.detect decode --fpr 0.03 0.05 0.10 [--base v6b]   -> engine_v6c_f<..>.json (report_natural / req_check doc duoc)"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import zlib

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

V4 = os.path.join(BASE, "data_labels", "realistic_v4")
FR = os.path.join(V4, "xlsr_frames")
RES = os.path.join(BASE, ".tmp_v6", "results")
OOFP = os.path.join(BASE, ".tmp_v6", "cache", "boundary_oof.pkl")
DT, POS_R, NEG_KEEP = 0.02, 0.06, 0.04
CFGS = {"goc", "app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi"}
MAXD = 0.3 / DT
PAIR_S, ROW_HALF = 5.0, 0.25


def feats(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.float32)
    d = np.abs(np.diff(z, axis=0, prepend=z[:1]))
    j = np.log1p(d / (np.median(d, axis=0, keepdims=True) + 1e-6))
    k = 5
    c = np.cumsum(np.pad(z, ((k + 1, k), (0, 0)), mode="edge"), axis=0)
    ma = (c[2 * k + 1:] - c[:-2 * k - 1]) / (2 * k + 1)
    return np.c_[z, j, z - ma[:len(z)]]


def joints(meta: dict) -> list[tuple[float, int]]:
    out = []
    for x in meta.get("edits", []):
        if x.get("kind") != "segment":
            continue
        l2 = x["level2"]
        if l2.startswith("03"):
            if x.get("sub") == "cat_bo":
                out.append((x["joint"], 2))
            else:
                out += [(x["start"], 1), (x["end"], 1)]
        elif x.get("start") is not None:
            out += [(x["start"], 3), (x["end"], 3)]
    return out


def labels(meta: dict, T: int) -> np.ndarray:
    y = np.zeros(T, np.int64)
    t = (np.arange(T) + 0.5) * DT
    for jt, c in joints(meta):
        y[np.abs(t - jt) <= POS_R] = c
    return y


def person_fold(meta: dict) -> int:
    from v6 import common as C
    return C.assign_fold([str(meta.get("person") or meta.get("speaker"))])


def load_items(keep: tuple | None = None) -> dict:
    """id -> dict(set, kind, cfg, meta, fold, real) cho moi tep co khung XLS-R. keep: chi giu cac khoa meta nay (vd ("edits", "level2") cho huan luyen: nhe hon ~8 lan)."""
    items = {}
    for s, tag in (("natvi", "v6natvi"), ("naten", "v6naten"), ("amilong", "v6amilong"), ("cover", "v6cover"), ("cover2", "v6cover2"), ("benign", "v6benign"), ("chain", "v6chain"),
                   ("ttsvi", "v6ttsvi"), ("ttsen", "v6ttsen"), ("aug", "v6aug")):
        mp = os.path.join(BASE, "data_labels", "v6", s, "manifest.json")
        if not os.path.exists(mp):
            continue
        for e in json.load(open(mp, encoding="utf-8")):
            m = e["meta"]
            # quyet dinh 18 (20/10): bo ban ghi tam loai (N7 khong ro muc) va ban ghi trai chinh sach nhai giong — ca hoc lan cham
            if e["cfg"] in CFGS and not m.get("ngoai_pham_vi") and not m.get("tam_loai") and not m.get("chinh_sach") and \
                    (not m.get("lockbox") or tag == "v6amilong") and \
                    os.path.exists(os.path.join(FR, e["id"] + ".npy")):
                items[e["id"]] = {"set": tag, "kind": e["kind"], "cfg": e["cfg"], "meta": {k: m[k] for k in keep if k in m} if keep else m, "fold": person_fold(m),
                                  "real": not m.get("level2")}
    for r in pickle.load(open(os.path.join(V4, "features_v4.pkl"), "rb")).values():
        if os.path.exists(os.path.join(FR, r["id"] + ".npy")):
            items[r["id"]] = {"set": "v4", "kind": r["kind"], "cfg": "goc", "meta": {k: r["meta"][k] for k in keep if k in r["meta"]} if keep else r["meta"], "fold": person_fold(r["meta"]),
                              "real": not r["meta"].get("level2")}
    return items


def smooth(p: np.ndarray) -> np.ndarray:
    return np.convolve(p, np.ones(3) / 3, mode="same")


def tagged(path: str, tag: str) -> str:
    return path if not tag else path.replace(".pkl", f"_{tag}.pkl").replace(".joblib", f"_{tag}.joblib")


def train(with_ami: bool = False, tag: str = "", with_chain: bool = False, with_benign: bool = False) -> None:
    """with_ami: cho bo do HOC them tep dai AMI (khong lockbox TS3003; trong so 5), chia fold theo cuoc hop -> kiem tra thich nghi mien.
    with_chain: hoc them bo chain (nhieu dieu kien chong nhau) · with_benign: hoc them bo benign (tep THAT qua xu ly vo hai = mau am ro rang)."""
    import joblib
    sys.stdout.reconfigure(encoding="utf-8")
    items = load_items()
    ids = sorted(i for i in items if not (items[i]["set"] == "v6amilong" and items[i]["meta"].get("lockbox") and with_ami))
    print(f"{len(ids)} tep co khung XLS-R: "
          + str({s: sum(items[i]['set'] == s for i in ids) for s in ('v4', 'v6natvi', 'v6naten', 'v6amilong', 'v6cover')}), flush=True)
    held = {s for s, w in (("v6benign", with_benign), ("v6chain", with_chain)) if not w}
    train_ids = [i for i in ids if items[i]["set"] not in held and (with_ami or items[i]["set"] != "v6amilong")]      # mac dinh: AMI dai, benign, chain chi de CHAM
    rng = np.random.default_rng(0)
    cache = os.path.join(os.path.dirname(OOFP), f"boundary_trainset{'_' + tag if tag else ''}.npz")
    if os.path.exists(cache) and not with_ami:
        z = np.load(cache)
        X, Y, F, W = z["X"], z["Y"], z["F"], z["W"]
    else:
        Xs, Ys, Fs, Ws = [], [], [], []
        for n, i in enumerate(train_ids):
            zf = np.load(os.path.join(FR, i + ".npy"))
            y = labels(items[i]["meta"], len(zf))
            keep = (y > 0) | (rng.random(len(y)) < NEG_KEEP)
            Xs.append(feats(zf)[keep]); Ys.append(y[keep]); Fs.append(np.full(int(keep.sum()), items[i]["fold"]))
            Ws.append(np.full(int(keep.sum()), 5.0 if items[i]["set"] == "v6amilong" else 1.0, np.float32))
            if (n + 1) % 4000 == 0:
                print(f"  nap {n + 1}/{len(train_ids)}", flush=True)
        X, Y, F, W = np.vstack(Xs), np.concatenate(Ys), np.concatenate(Fs), np.concatenate(Ws)
        del Xs
        if not with_ami:
            np.savez(cache, X=X, Y=Y, F=F, W=W)
    print(f"mau hoc {len(Y)} khung: " + str(np.bincount(Y).tolist()), flush=True)
    oof, op = {}, tagged(OOFP, tag)
    for f in range(5):
        tr = F != f
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        m.fit(X[tr], Y[tr], sample_weight=W[tr])
        joblib.dump(m, tagged(os.path.join(os.path.dirname(OOFP), f"boundary_m{f}.joblib"), tag), compress=3)
        te = [i for i in ids if items[i]["fold"] == f]
        for i in te:
            p = m.predict_proba(feats(np.load(os.path.join(FR, i + ".npy"))))
            oof[i] = np.stack([smooth(p[:, c]) for c in (1, 2, 3)], 1).astype(np.float16)
        print(f"  fold {f + 1}/5 xong ({len(te)} tep cham)", flush=True)
        os.makedirs(os.path.dirname(op), exist_ok=True)
        pickle.dump(oof, open(op + ".tmp", "wb"))
        os.replace(op + ".tmp", op)
    print("train xong", flush=True)


def decode_file(p: np.ndarray, th: tuple, dur: float, with_warn: bool = False):
    """p: (T, 3) diem lam tron (ghep, xoa, 02) -> cac dong 03; TOI DA 2 DONG MOI KHUC 16 S (buoc 14 s, nhu Constraint).
    with_warn=True: tra them co 'co the con cho sua khac' (con ung vien vuot nguong bi bo vi qua 2 dong / khuc)."""
    cand = []
    for c, name in ((0, "ghep"), (1, "xoa")):
        pk, pr = find_peaks(p[:, c].astype(float), height=th[c], distance=MAXD)
        cand += [(float(pr["peak_heights"][k]) - th[c], name, (pk[k] + 0.5) * DT) for k in range(len(pk))]
    cand.sort(reverse=True)
    rows, used = [], set()
    g = sorted([c for c in cand if c[1] == "ghep"], key=lambda x: x[2])
    for k, (m1, _, t1) in enumerate(g):
        if k in used:
            continue
        nxt = next((q for q in range(k + 1, len(g)) if q not in used and g[q][2] - t1 <= PAIR_S), None)
        if nxt is not None:
            used |= {k, nxt}
            rows.append({"level2": "03_ghep_giong_that", "kind": "segment", "start": t1, "end": g[nxt][2],
                         "p": 1.0, "margin": max(m1, g[nxt][0]), "via": "ranh_gioi_ghep"})
        else:
            used.add(k)
            rows.append({"level2": "03_ghep_giong_that", "kind": "segment", "start": max(0.0, t1 - ROW_HALF),
                         "end": min(dur, t1 + ROW_HALF), "p": 1.0, "margin": m1, "via": "ranh_gioi_ghep"})
    for m1, name, t in cand:
        if name == "xoa":
            rows.append({"level2": "03_ghep_giong_that", "kind": "segment", "start": max(0.0, t - ROW_HALF),
                         "end": min(dur, t + ROW_HALF), "p": 1.0, "margin": m1, "via": "ranh_gioi_xoa"})
    by_chunk: dict[int, list] = {}
    for r in rows:
        by_chunk.setdefault(int(((r["start"] + r["end"]) / 2) // 14.0), []).append(r)
    kept = [r for v in by_chunk.values() for r in sorted(v, key=lambda x: -x["margin"])[:2]]
    kept.sort(key=lambda r: r["start"])
    return (kept, len(kept) < len(rows)) if with_warn else kept


def _ov(a, b) -> float:
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = min(a[1] - a[0], b[1] - b[0])
    return i / u if u > 0 else 0.0


def merge_rows(base_rows: list, new_rows: list) -> list:
    out = list(base_rows)
    for r in new_rows:
        if not any(x["level2"] == r["level2"] and x.get("start") is not None and _ov((x["start"], x["end"]), (r["start"], r["end"])) > 0.3
                   for x in out):
            out.append(r)
    return out


def decode(fprs: list, base_tag: str, gate_fprs: list | None = None) -> None:
    """gate_fprs: giu dong cua cong v6b chi khi (diem cong - nguong) >= phan vi 1 - g cua tep THAT (fold khac); g = 0,09 ~ giu nguyen."""
    sys.stdout.reconfigure(encoding="utf-8")
    oof = pickle.load(open(OOFP, "rb"))
    items = load_items()
    base = json.load(open(os.path.join(RES, f"engine_{base_tag}.json"), encoding="utf-8"))
    score = {i: oof[i].astype(float).max(0) for i in oof}
    for fpr, gf in [(f, g) for f in fprs for g in (gate_fprs or [None])]:
        thr = {}
        for f in range(5):
            real = [score[i] for i in oof if items[i]["real"] and items[i]["fold"] != f and items[i]["set"] != "v4"]
            thr[f] = tuple(float(np.quantile([s[c] for s in real], 1 - fpr)) for c in (0, 1))
        res = json.loads(json.dumps(base))
        res["tag"] = f"v6c_f{int(round(fpr * 100)):02d}" + ("" if gf is None else f"_g{int(round(gf * 100)):02d}")
        n_new = n_flip = 0
        gthr = {}
        if gf is not None:
            for f in range(5):
                mr = [r["gate"] - r["gate_thr"] for sn, rs_ in base["records"].items() if sn != "v4" for r in rs_
                      if r["expected"] == "01" and r.get("fold") != f and "gate" in r]
                gthr[f] = float(np.quantile(mr, 1 - gf))
        for sname, recs in res["records"].items():
            if sname == "v4":
                continue
            for r in recs:
                it = items.get(r["id"])
                if gf is not None and "gate" in r and r["gate"] - r["gate_thr"] < gthr.get(r.get("fold"), -1e9):
                    k1 = r["id"] in items and items[r["id"]]["meta"].get("k1", {}).get("decision") in ("04", "tu_choi")
                    r["rows"] = []
                    r["final"] = "04" if k1 else "01"
                    r["correct"] = r["final"] == r["expected"]
                if it is None or r["id"] not in oof:
                    continue
                dur = len(oof[r["id"]]) * DT
                rows = decode_file(oof[r["id"]].astype(float), thr[it["fold"]], dur)
                if not rows:
                    continue
                n_new += 1
                old = r["final"]
                r["rows"] = merge_rows(r["rows"], rows)
                if old != "04" or not any(x.get("kind") == "gate" for x in r["rows"]):
                    pass
                k1_04 = it["meta"].get("k1", {}).get("decision") in ("04", "tu_choi")
                r["final"] = "04" if k1_04 else "03"
                r["correct"] = r["final"] == r["expected"]
                n_flip += r["final"] != old
        res["created"] = "v6c = v6b + bo do ranh gioi XLS-R"
        p = os.path.join(RES, f"engine_{res['tag']}.json")
        json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"FPR {fpr:.0%}: {n_new} tep co dong ranh gioi, {n_flip} tep doi nhan -> {p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "decode"])
    ap.add_argument("--fpr", type=float, nargs="+", default=[0.03, 0.05, 0.10])
    ap.add_argument("--base", default="v6b")
    ap.add_argument("--gate-fpr", type=float, nargs="+", default=None)
    ap.add_argument("--with-ami", action="store_true", help="train: hoc them tep dai AMI (khong lockbox), de kiem tra thich nghi mien")
    ap.add_argument("--tag", default="", help="train: hau to ten tep ket qua (de khong de len ban chinh)")
    ap.add_argument("--with-chain", action="store_true", help="train: hoc them bo chain (nhieu dieu kien chong nhau)")
    ap.add_argument("--with-benign", action="store_true", help="train: hoc them bo benign (tep that qua xu ly vo hai = mau am)")
    a = ap.parse_args()
    train(a.with_ami, a.tag, a.with_chain, a.with_benign) if a.step == "train" else decode(a.fpr, a.base, a.gate_fpr)
