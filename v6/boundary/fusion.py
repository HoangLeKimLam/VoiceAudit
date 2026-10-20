# -*- coding: utf-8 -*-
"""v7 - HOP NHAT cac chuyen gia tren XLS-R L12 (da nen PCA 64, khung 20 ms) thanh nhan tep + dong thao tac, DAY DU 4 loai nhan:
  03 moi noi     bo do ranh gioi (detect.py): dong ghep / chen / xoa loi
  02 ca file     2 dau HGB tren vector gop (trung binh + do lech chuan 64 chieu PCA + log do dai): khu on ca file · doi giong ca file
  02 mot doan    lop 'ranh gioi 02' cua bo do (cap dinh) + BO KIEM CHUNG kieu doan (khu on / doi giong / khong phai 02)
  nhan tep       04 neu K1 04 -> 03 neu co dong 03 -> 02 neu co dong 02 -> 01   (dung thu tu Scope)
  rang buoc      <= 2 dong DOAN moi khuc 14 s (gop 03 + 02 cuc bo), co 'warn_more' neu con ung vien bi bo
Moi chuyen gia co NGUONG RIENG dat o ti le vu oan tren tep THAT tu nhien (goc) cua cac fold KHAC (khong nhin fold dang cham).
4 diem van hanh co dinh (khong chon theo ket qua): A (1 ; 0,5 ; 0,5 ; 0,5 %) ~ tong 2,5 % cho Req 4 · B (2 ; 1 ; 1 ; 1) · C (3 ; 2 ; 2 ; 2) · D (5 ; 3 ; 3 ; 3),
thu tu: (moi noi 03, khu on ca file, doi giong ca file, 02 mot doan).
Mo hinh hoc theo fold NGUOI NOI (crc32 % 5); tep ngoai tap hoc (benign, chain, amilong, v4) cham bang mo hinh fold cua nguoi do.
  python -m v6.boundary.fusion run --tag all [--smoke]   -> .tmp_v6/results/fusion_<tag>.md, engine_v7<A-D>.json (+ _extra.json)
  python -m v6.natural.req_check v7A v7B v7C v7D          -> bang Requirement chuan cho natvi / naten"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

from v6.boundary import detect as D  # noqa: E402
from v6.natural import req_check as RC  # noqa: E402
from v6.natural.report_natural import edit_hit, iou  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
TRAIN_SETS = ("v6natvi", "v6naten", "v6cover", "v6cover2", "v6ttsvi", "v6ttsen", "v6aug")
NAT_SETS = ("v6natvi", "v6naten")
OPS = {"A": (0.010, 0.005, 0.005, 0.005), "B": (0.02, 0.01, 0.01, 0.01), "C": (0.03, 0.02, 0.02, 0.02), "D": (0.05, 0.03, 0.03, 0.03)}
SEG_MAX_S, SEG_MIN_S, PAIR_MAX_S = 8.0, 0.4, 8.0
CHUNK_S, MAX_ROWS = 14.0, 2
LOCAL02 = ("02_nen_cuc_bo", "02_giong_cuc_bo")
BENIGN_OK02 = ("ns2", "ns3")                      # WebRTC NS: khu on vo hai, ra 01 hoac 02 deu khong bi ket toi
VERSION = "v1"


def expected_of(meta: dict) -> str:
    l2 = meta.get("level2", [])
    return "03" if any(x.startswith("03") for x in l2) else "02" if any(x.startswith("02") for x in l2) else "01"


def head_labels(meta: dict) -> list:
    l2 = set(meta.get("level2", []))
    return [int("02_nen_toan_cuc" in l2), int("02_giong_toan_cuc" in l2)]


def load_z(i: str) -> np.ndarray:
    return np.load(os.path.join(D.FR, i + ".npy"))


def pooled_all(ids: list, suffix: str) -> np.ndarray:
    path = os.path.join(CACHE, f"pooled_xlsr{suffix}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        if list(z["ids"]) == ids:
            return z["X"]
    X = np.zeros((len(ids), 129), np.float32)
    for n, i in enumerate(ids):
        f = load_z(i).astype(np.float32)
        X[n] = np.r_[f.mean(0), f.std(0), np.log(max(len(f) * D.DT, 0.05))]
        if (n + 1) % 5000 == 0:
            print(f"  gop {n + 1}/{len(ids)}", flush=True)
    np.savez(path, ids=np.array(ids, dtype=object), X=X)
    return X


def seg_feat(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Dac trung doan [a, b]: (trung binh trong - trung binh ngoai) + do lech chuan trong + log do dai."""
    i0 = max(0, int(a / D.DT))
    i1 = min(len(z), max(int(b / D.DT), i0 + 2))
    inn = z[i0:i1].astype(np.float32)
    out = np.concatenate([z[:i0], z[i1:]]).astype(np.float32)
    if len(out) == 0:
        out = inn
    return np.r_[inn.mean(0) - out.mean(0), inn.std(0), np.log(max(b - a, 0.05))]


def seg_windows(e: dict) -> list:
    """Cua so (a, b, lop) lay tu mot thao tac that: 02 cuc bo -> lop 0 / 1; 03 -> lop 2 (khong phai 02)."""
    if e["level2"] in LOCAL02:
        if SEG_MIN_S < e["end"] - e["start"] <= SEG_MAX_S:
            return [(e["start"], e["end"], 0 if e["level2"] == "02_nen_cuc_bo" else 1)]
        return []
    if not e["level2"].startswith("03"):
        return []
    if e.get("sub") == "cat_bo":
        j = e.get("joint", (e["start"] + e["end"]) / 2)
        return [(max(0.0, j - 0.6), j + 0.6, 2)]
    return [(e["start"], e["end"] if e["end"] - e["start"] <= SEG_MAX_S else e["start"] + 2.0, 2)]


def seg_training(items: dict, rng: np.random.Generator):
    """Mau cho bo kiem chung kieu doan: duong (02 nen / doi giong mot doan), am (doan 03 + cua so ngau nhien tren tep that)."""
    X, y, fold, pos_len = [], [], [], []
    for it in items.values():
        pos_len += [e["end"] - e["start"] for e in it["meta"].get("edits", []) if e.get("level2") in LOCAL02 and e.get("start") is not None]
    pos_len = [x for x in pos_len if SEG_MIN_S < x <= SEG_MAX_S] or [1.5]
    for i, it in items.items():
        if it["set"] not in TRAIN_SETS:
            continue
        wins = [w for e in it["meta"].get("edits", []) if e.get("kind") == "segment" for w in seg_windows(e)]
        take_real = it["real"] and rng.random() < 0.15
        if not wins and not take_real:
            continue
        z = load_z(i)
        dur = len(z) * D.DT
        for a, b, c in wins:
            X.append(seg_feat(z, a, b)); y.append(c); fold.append(it["fold"])
        if take_real and dur > 4:
            L = float(rng.choice(pos_len))
            a = float(rng.uniform(0.3, max(0.4, dur - L - 0.3)))
            X.append(seg_feat(z, a, a + L)); y.append(2); fold.append(it["fold"])
    return np.array(X, np.float32), np.array(y), np.array(fold)


def fit_hgb(X: np.ndarray, y: np.ndarray, iters: int, lr: float) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=iters, learning_rate=lr, class_weight="balanced", random_state=0).fit(X, y)


def train_models(items: dict, ids: list, X: np.ndarray, tag: str, suffix: str) -> dict:
    """2 dau ca file (OOF theo fold) + bo kiem chung doan (mot mo hinh moi fold)."""
    path = os.path.join(CACHE, f"fusion_{VERSION}_{tag}{suffix}.pkl")
    if os.path.exists(path):
        cached = pickle.load(open(path, "rb"))
        if cached["ids"] == ids:                     # tap tep doi (them bo moi) -> khong dung lai diem cu, hoc lai
            return cached
        print("danh sach tep da doi: hoc lai dau ca file", flush=True)
    fold = np.array([items[i]["fold"] for i in ids])
    in_train = np.array([items[i]["set"] in TRAIN_SETS for i in ids])
    Y = np.array([head_labels(items[i]["meta"]) for i in ids])
    P = np.zeros((len(ids), 2), np.float32)
    for f in range(5):
        tr, te = in_train & (fold != f), fold == f
        for h in range(2):
            P[te, h] = fit_hgb(X[tr], Y[tr, h], 200, 0.07).predict_proba(X[te])[:, 1]
        print(f"  dau ca file: fold {f + 1}/5", flush=True)
    SX, Sy, Sf = seg_training(items, np.random.default_rng(0))
    print(f"bo kiem chung doan: {len(Sy)} mau, lop (nen, giong, khong 02) = {np.bincount(Sy).tolist()}", flush=True)
    seg = {f: fit_hgb(SX[Sf != f], Sy[Sf != f], 150, 0.08) for f in range(5)}
    out = {"ids": ids, "P": P, "seg": seg}
    pickle.dump(out, open(path, "wb"))
    return out


def decode02_local(p: np.ndarray, th: float, dur: float) -> list:
    """Cap dinh cua lop 'ranh gioi 02' -> doan (a, b, bien) dai 0,4-8 s; moi dinh dung toi da mot lan."""
    from scipy.signal import find_peaks
    pk, pr = find_peaks(p, height=th, distance=D.MAXD)
    t = [((k + 0.5) * D.DT, float(h) - th) for k, h in zip(pk, pr["peak_heights"])]
    segs, used = [], set()
    for k in range(len(t)):
        if k in used:
            continue
        q = next((q for q in range(k + 1, len(t)) if q not in used and SEG_MIN_S <= t[q][0] - t[k][0] <= min(PAIR_MAX_S, 0.5 * dur)), None)
        if q is not None:
            used |= {k, q}
            segs.append((t[k][0], t[q][0], min(t[k][1], t[q][1])))
    return segs


REGION_LEVEL2 = ("02_nen_cuc_bo", "02_giong_cuc_bo", "03_ghep_giong_that", "03_chen_giong_may")


def decode_regions(probs: list, ths: list, levels: tuple = REGION_LEVEL2) -> list:
    """Dong MOT DOAN tu cac kenh VUNG cua TCN (tcn.py --regions / --regions2): moi kenh la mot mang diem theo khung; doan lien tuc vuot nguong,
    dai 0,4-8 s. probs / ths / levels cung thu tu (khu on, doi giong, doan ghep, giong may)."""
    rows = []
    for p, th, lvl in zip(probs, ths, levels):
        edges = np.diff(np.r_[0, (p >= th).astype(int), 0])
        for s0, e0 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            a, b = s0 * D.DT, e0 * D.DT
            if SEG_MIN_S <= b - a <= SEG_MAX_S:
                rows.append({"level2": lvl, "kind": "segment", "start": a, "end": b, "p": float(p[s0:e0].mean()), "nm": float(p[s0:e0].max()), "via": "tcn_vung"})
    return rows


def cap_rows(rows: list) -> tuple:
    """Chong lan > 30 % (cung loai) va <= MAX_ROWS dong doan moi khuc CHUNK_S, xep theo bien vuot nguong. Tra (dong giu, co_bo)."""
    seg = sorted([r for r in rows if r["kind"] == "segment"], key=lambda r: -r["nm"])
    keep, dropped = [], False
    for r in seg:
        mid = int(((r["start"] + r["end"]) / 2) // CHUNK_S)
        dup = any(x["level2"] == r["level2"] and D._ov((r["start"], r["end"]), (x["start"], x["end"])) > 0.3 for x in keep)
        if dup:
            continue
        if sum(int(((x["start"] + x["end"]) / 2) // CHUNK_S) == mid for x in keep) >= MAX_ROWS:
            dropped = True
            continue
        keep.append(r)
    return [r for r in rows if r["kind"] != "segment"] + sorted(keep, key=lambda r: r["start"]), dropped


def decide(P: np.ndarray, get_z, hp: np.ndarray, th: dict, seg_model, k1_04: bool) -> tuple:
    """P: (T, 3) diem khung (ghep, xoa, 02 cuc bo); hp: (nen_toan, giong_toan). Tra (rows, final, warn_more)."""
    dur = len(P) * D.DT
    r3, warn = D.decode_file(P, (th["ghep"], th["xoa"]), dur, with_warn=True)
    rows = [{**r, "kind": "segment", "nm": r["margin"]} for r in r3]
    for key, name, thk in ((0, "02_nen_toan_cuc", "nen"), (1, "02_giong_toan_cuc", "giong")):
        if hp[key] >= th[thk]:
            rows.append({"level2": name, "kind": "global", "p": float(hp[key]), "nm": float(hp[key] - th[thk])})
    cands = decode02_local(P[:, 2], th["loc"], dur)
    if cands:
        z = get_z()
        for a, b, m in cands:
            pr = seg_model.predict_proba(seg_feat(z, a, b)[None])[0]
            c = int(np.argmax(pr))
            if c < 2:
                rows.append({"level2": LOCAL02[c], "kind": "segment", "start": a, "end": b, "p": float(pr[c]), "nm": float(m), "via": "ranh_gioi_02"})
    rows, dropped = cap_rows(rows)
    final = "04" if k1_04 else "03" if any(r["level2"].startswith("03") for r in rows) else "02" if rows else "01"
    return rows, final, bool(warn or dropped)


def thresholds(items: dict, nat: list, bs: dict, HP: dict, fprs: tuple) -> dict:
    """Nguong moi fold: phan vi 1 - FPR tren tep THAT tu nhien cua cac fold KHAC."""
    out = {}
    q = lambda vals, fpr: float(np.quantile(vals, 1 - fpr))                                  # noqa: E731
    for f in range(5):
        use = [i for i in nat if items[i]["fold"] != f]
        out[f] = {"ghep": q([bs[i][0] for i in use], fprs[0]), "xoa": q([bs[i][1] for i in use], fprs[0]),
                  "nen": q([HP[i][0] for i in use], fprs[1]), "giong": q([HP[i][1] for i in use], fprs[2]),
                  "loc": q([bs[i][2] for i in use], fprs[3])}
    return out


def group_name(it: dict) -> str:
    m = it["meta"]
    if it["set"] == "v6cover":
        return f"cover:{m['axis']['truc']}={m['axis']['gia_tri']}"
    if it["set"] == "v6cover2":
        return f"cover2:{m['cond']}"
    if it["set"] == "v6benign":
        return f"benign:{m['cond']}"
    if it["set"] == "v6chain":
        return "chain"
    return it["set"] + (" goc" if it["cfg"] == "goc" else " 4 kenh")


def groups_of(it: dict) -> list:
    """Cac nhom bao cao ma tep thuoc ve: nhom chinh + (bo chain) tung gia tri cua tung truc."""
    g = [group_name(it)]
    if it["set"] == "v6chain":
        g += [f"chain:{k}={v}" for k, v in it["meta"].get("axis", {}).items()]
    return g


def make_record(i: str, it: dict, rows: list, final: str, warn: bool) -> dict:
    exp = expected_of(it["meta"])
    return {"id": i, "set": it["set"], "cfg": it["cfg"], "kind": it["kind"], "final": final, "expected": exp, "correct": final == exp,
            "rows": rows, "_m": it["meta"], "_g": it["meta"].get("person") or it["meta"].get("speaker"), "lockbox": False, "warn_more": warn,
            "groups": groups_of(it)}


def edit_key(e: dict) -> str:
    return e["level2"] + (":" + e["sub"] if e.get("sub") in ("thay_that", "chen_that", "cat_bo") else "")


def pct(v: float) -> str:
    return f"{100 * v:.0f} %"


def per_op_table(rs: list) -> list:
    """Theo tung loai thao tac that: so thao tac, tep bi bao co sua, khoanh dung cho (bat ky dong chong), dung loai, moc <= 500 ms, IoU."""
    st = defaultdict(lambda: {"n": 0, "flag": 0, "loc": 0, "typed": 0, "m500": [], "iou": []})
    for r in rs:
        rows = r["rows"] if r["final"] != "04" else []
        for e, h in zip(r["_m"].get("edits", []), edit_hit(r, r["_m"])):
            s = st[edit_key(e)]
            s["n"] += 1
            s["flag"] += r["final"] in ("02", "03")
            s["typed"] += bool(h[0])
            if e.get("kind") == "segment":
                s["loc"] += any(x.get("start") is not None and iou((x["start"], x["end"]), (e["start"], e["end"])) > 0 for x in rows)
                if h[0]:
                    s["m500"].append(bool(h[1])); s["iou"].append(h[2])
    L = ["| Loai thao tac that | n | tep bi bao co sua | khoanh dung cho | dung loai (+cho) | moc <= 500 ms | IoU |", "|---|---|---|---|---|---|---|"]
    for k in sorted(st):
        s = st[k]
        local = "toan_cuc" not in k and "giong_may" not in k
        L.append(f"| {k} | {s['n']} | {pct(s['flag'] / s['n'])} | {pct(s['loc'] / s['n']) if local else '-'} | {pct(s['typed'] / s['n'])} | "
                 f"{pct(np.mean(s['m500'])) if s['m500'] else '-'} | {np.mean(s['iou']):.2f} |" if s["iou"] else
                 f"| {k} | {s['n']} | {pct(s['flag'] / s['n'])} | {pct(s['loc'] / s['n']) if local else '-'} | {pct(s['typed'] / s['n'])} | - | - |")
    real = [r for r in rs if r["expected"] == "01"]
    if real:
        c = Counter(l2 for r in real for l2 in {x["level2"] for x in r["rows"]})
        L.append(f"\nTep that ({len(real)}): ra 01 {np.mean([r['final'] == '01' for r in real]):.1%} · co dong: "
                 + " · ".join(f"{k} {v / len(real):.1%}" for k, v in sorted(c.items())))
    return L


def benign_table(groups: dict) -> list:
    L = ["| Nguon vo hai | n | ra 01 | ra 02 | ra 03 | ra 04 | bi ket toi sai |", "|---|---|---|---|---|---|---|"]
    for g in sorted(k for k in groups if k.startswith("benign:")):
        rs, c = groups[g], Counter(r["final"] for r in groups[g])
        ok02 = g.split(":")[1] in BENIGN_OK02
        wrong = sum(1 for r in rs if r["final"] == "03" or (not ok02 and r["final"] == "02") or any(x["level2"].startswith("02_giong") for x in r["rows"]))
        L.append(f"| {g[7:]} | {len(rs)} | " + " | ".join(pct(c[k] / len(rs)) for k in ("01", "02", "03", "04")) + f" | {pct(wrong / len(rs))} |")
    return L


def head_auc(items: dict, ids: list, HP: dict, PR: dict) -> list:
    nat = [i for i in ids if items[i]["set"] in NAT_SETS and items[i]["cfg"] == "goc"]
    real = [i for i in nat if items[i]["real"]]
    specs = (("khu on ca file", lambda i: HP[i][0], {"02_nen_toan_cuc"}), ("doi giong ca file", lambda i: HP[i][1], {"02_giong_toan_cuc"}),
             ("02 mot doan (dinh khung)", lambda i: float(PR[i][:, 2].max()), set(LOCAL02)))
    L = []
    for name, sel, keys in specs:
        pos = [i for i in nat if keys & set(items[i]["meta"].get("level2", []))]
        L.append(f"AUC {name}: **{RC.auc([sel(i) for i in pos], [sel(i) for i in real]):.3f}** (n duong {len(pos)}, n that {len(real)})")
    return L


def run(tag: str, smoke: bool) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items()
    ids = sorted(items)
    if smoke:
        ids = ids[::40]
        items = {i: items[i] for i in ids}
    suffix = "_smoke" if smoke else ""
    print(f"{len(ids)} tep", flush=True)
    X = pooled_all(ids, suffix)
    M = train_models(items, ids, X, tag, suffix)
    HP = {i: M["P"][n] for n, i in enumerate(ids)}
    oof = pickle.load(open(D.tagged(D.OOFP, tag), "rb"))
    bm = {f: joblib.load(D.tagged(os.path.join(os.path.dirname(D.OOFP), f"boundary_m{f}.joblib"), tag)) for f in range(5)}

    def frame_probs(i: str) -> np.ndarray:
        if i in oof:
            return oof[i]
        pr = bm[items[i]["fold"]].predict_proba(D.feats(load_z(i)))
        return np.stack([D.smooth(pr[:, c]) for c in (1, 2, 3)], 1).astype(np.float16)

    PR = {i: frame_probs(i) for i in ids}
    nat = [i for i in ids if items[i]["set"] in NAT_SETS and items[i]["real"] and items[i]["cfg"] == "goc"]
    bs = {i: PR[i].astype(float).max(0) for i in nat}
    md = ["# Hop nhat v7 (4 chuyen gia tren XLS-R L12)\n", "**Chan doan chuyen gia (tep goc tu nhien, vi + en)**\n", "\n\n".join(head_auc(items, ids, HP, PR))]
    for name, fprs in OPS.items():
        th = thresholds(items, nat, bs, HP, fprs)
        groups, recs = defaultdict(list), defaultdict(list)
        for i in ids:
            it = items[i]
            k1_04 = it["meta"].get("k1", {}).get("decision") in ("04", "tu_choi")
            rows, final, warn = decide(PR[i].astype(float), lambda i=i: load_z(i), HP[i], th[it["fold"]], M["seg"][it["fold"]], k1_04)
            r = make_record(i, it, rows, final, warn)
            groups[group_name(it)].append(r)
            recs[it["set"]].append({k: v for k, v in r.items() if k != "_m"})
        md.append(f"\n## Diem van hanh {name}: vu oan (moi noi 03, khu on ca file, doi giong ca file, 02 mot doan) = {tuple(100 * x for x in fprs)} %\n")
        md += ["| Nhom | n | 1 F1 | 2 khong bo sot | 3 dang tin | 4 vu oan | 5 khu on vo hai | 7a | 7b | 9 (04) | 10 moc | 11 IoU |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for g in sorted(groups, key=lambda x: (not x.startswith("v6"), x)):
            if g.startswith("benign:"):
                continue
            m = RC.metrics(groups[g])
            md.append(f"| {g} | {len(groups[g])} | " + " | ".join("-" if not np.isfinite(m.get(k, np.nan)) else f"{m[k]:.2f}"
                                                                   for k in ("1", "2", "3", "4", "5", "7a", "7b", "9", "10", "11")) + " |")
        for g in ("v6natvi goc", "v6naten goc"):
            if g in groups:
                md += [f"\n**Theo loai thao tac - {g}**\n"] + per_op_table(groups[g])
        md += ["\n**Nguon vo hai (benign)**\n"] + benign_table(groups)
        main = {s: recs[s] for s in NAT_SETS if s in recs}
        json.dump({"tag": f"v7{name}", "records": main}, open(os.path.join(D.RES, f"engine_v7{name}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
        extra = {s: v for s, v in recs.items() if s not in NAT_SETS}
        json.dump({"tag": f"v7{name}", "records": extra}, open(os.path.join(D.RES, f"engine_v7{name}_extra.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
        print("\n".join(md[-60:]), flush=True)
    open(os.path.join(D.RES, f"fusion_{tag}{suffix}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run"])
    ap.add_argument("--tag", default="all")
    ap.add_argument("--smoke", action="store_true", help="chi 1/40 so tep, de thu duong chay")
    a = ap.parse_args()
    run(a.tag, a.smoke)
