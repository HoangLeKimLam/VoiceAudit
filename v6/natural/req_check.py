# -*- coding: utf-8 -*-
"""M17 - DOI CHIEU CHI TIEU DA DANG KY (Requirement 1-11) theo DUNG NGUYEN VAN ban nop (v3, 10/10) + CHAN DOAN CONG 01.
  python -m v6.natural.req_check <tag_engine> [<tag> ...] [--ci [N]] [--strata] [--old <md cu>] [--sets v6natvi v6naten]
                                 [--cluster spec|i2] [--lockbox engine|common]
  (moi tag: .tmp_v6/results/engine_<tag>.json; --ci = khoang tin cay 95 % bootstrap theo CUM, mac dinh 2000 lan;
   --strata = bang phan tang + bang o du lieu COND-10; --old = bang cu de dat canh so moi, mac dinh req_check_<tag>_old_1010.md)
Pham vi (req_report): bang chinh THEO BAN NOP = tep goc + cau hinh kenh >= 24 kbps (kbps tu v6/phone/channels.py), kenh
< 24 kbps bao rieng; bang 'tep goc' de so voi ban cu; trong pham vi, khong LOCKBOX; phan tang tren tep goc (tru 'kenh').
Luat cham (6Muc, muc 3 Requirement + Dieu kien cham):
  - CHI loai khoi cham tep bi Constraint chuyen 04 / tu choi (K1 decision 04 | tu_choi va final 04); bao so tep bi loai theo ly do.
    Tep ra 04 o buoc phan tich (luat 5) VAN CHAM: Req 1 tinh la sai nhan, Req 2 tinh la khong gan 02/03, Req 9 la tu so.
  - Loai thao tac cua DONG lay tu level2 + via: '03_chen_giong_may' = giong may (doan / ca file theo kind),
    '03_ghep_*' via 'ranh_gioi_xoa' = xoa loi, con lai '03_ghep_*' = ghep loi that, '02_*' = level2.
    Loai cua THAO TAC THAT: '03_chen_giong_may' = giong may, '03_ghep_*' sub 'cat_bo' = xoa loi, con lai = ghep.
  - 7a / 7b / 8a: F1 theo tung loai (precision tren dong cung loai, recall tren thao tac that cung loai).
    Trung: ghep / giong may = dong cung loai giao vung (IoU > 0); xoa loi = moi noi cach vung khoanh <= 500 ms.
  - Req 10: thao tac that co MOT dong CUNG LOAI hai mep lech <= 500 ms (xoa loi: moi noi cach vung khoanh <= 500 ms).
    Req 11 (cach doc DA CHON = lac quan nhat, ghi ro trong bao cao): trung binh, tren thao tac that (khong xoa loi) CO dong
    cung loai giao vung, cua IoU tot nhat; thao tac bo sot va dong sai khong vao mau so (da phat o 7a / 8a / 10).
    Bao them hai cach doc khac: '11r' theo DONG (moi dong doan cung loai voi mot thao tac that cuc bo trong tep: IoU tot
    nhat voi thao tac cung loai, 0 neu khong giao) va '11m' theo THAO TAC (moi thao tac that khong xoa, 0 neu bo sot).
  - Chi tieu KHONG DO DUOC (in 'khong do duoc (n=..)', khong bao gio DAT): mau so < 10, khong co mau duong, kenh lien quan tat
    (Req 6 khi kenh giong may tat; Req 9 khi khong co tep 04 phan tich nao - SUY TU DU LIEU, khong doc tu cau hinh).
  - Macro-F1 (Req 1) khi bootstrap: tap nhan co dinh theo tap duoc cham (nhan co ho tro / co du doan); F1 khong xac dinh
    trong mot lan boc lai tinh = 0 (khong bo nhan).
  - Khoang tin cay: bootstrap theo cum (cum = meta.cluster / hosts.cluster cho nguon vss (video), con lai meta.person;
    'voxp:None' -> moi host_file mot cum), boc lai cum co hoan lai, moi chi tieu la ti so cua tong -> tinh lai tu tong cum.
Ra: .tmp_v6/results/req_check_<tag>.md (+ .json). Phan trinh bay: v6.natural.req_report. Kiem thu: python -m v6.natural.test_req_check"""
from __future__ import annotations

import math
import os
import sys
import warnings
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.natural.report_natural import MAIN, Report, edit_hit, iou, macro_f1  # noqa: E402,F401  (xuat lai cho ma cu)

RES = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQ = [("1", "Macro-F1 (01/02/03)", 0.86, ">="), ("2", "Khong bo sot tep sua", 0.90, ">="), ("3", "Bao 'co sua' dang tin", 0.97, ">="),
       ("4", "Khong vu oan tep that", 0.03, "<="), ("5", "Khong ket toi khu on vo hai", 0.92, ">="), ("6", "Khong nham loi that thanh giong may", 0.05, "<="),
       ("7a", "Bat ghep loi that (F1)", 0.80, ">="), ("7b", "Bat xoa loi (F1)", 0.65, ">="), ("8a", "Bat giong may mot doan (F1)", 0.86, ">="),
       ("8b", "Bat giong may ca file", 0.90, ">="),
       ("9", "Ti le 04", 0.08, "<="), ("10", "Moc <= 500 ms", 0.78, ">="), ("11", "IoU trung binh", 0.80, ">=")]
EXTRA = [("01", "Tep that ra 01 (bao them)"), ("33", "Tep co thao tac 03 ra dung 03 (bao them)"),
         ("11r", "Req 11 doc theo dong (dong cung loai khong giao = 0)"),
         ("11m", "Req 11 doc theo thao tac (thao tac bo sot = 0)")]
MIN_N = 10                      # mau so < 10 -> khong do duoc
TOL = 0.5                       # 500 ms
BOOT_N = 2000
CONSTRAINT_DEC = ("04", "tu_choi")
LABELS3 = ("01", "02", "03")
TYPED = (("7a", "ghep"), ("7b", "xoa"), ("8a", "may"))
FEATS = tuple([f"cm_{e}_{f}" for e in LABELS3 for f in ("01", "02", "03", "04")] +
              ["n2", "k2", "n3", "k3", "n4", "k4", "n5", "k5", "a5", "n6", "k6", "n8b", "k8b", "n9", "k9",
               "n10", "k10", "s11", "n11", "n01", "k01", "n33", "k33", "s11r", "n11r", "n11m"] +
              [f"{p}_{t}" for _, t in TYPED for p in ("tp", "fp", "nr", "hr")])


# ---------------- loai thao tac / dong ----------------
def edit_type(e: dict) -> str:
    """Loai cua thao tac THAT: ghep | xoa | may | may_file | level2 (02_*)."""
    l2 = str(e.get("level2", ""))
    if l2 == "03_chen_giong_may":
        return "may_file" if e.get("kind") == "global" else "may"
    if l2.startswith("03"):
        return "xoa" if e.get("sub") == "cat_bo" or "xoa" in l2 else "ghep"
    return l2


def row_type(x: dict) -> str:
    """Loai cua DONG he thong bao: ghep | xoa | may | may_file | level2 (02_*)."""
    l2 = str(x.get("level2", ""))
    if l2 == "03_chen_giong_may":
        return "may_file" if x.get("kind") == "global" else "may"
    if l2.startswith("03"):
        return "xoa" if x.get("via") == "ranh_gioi_xoa" or "xoa" in l2 else "ghep"
    return l2


def is_segment(x: dict) -> bool:
    return x.get("kind") != "global" and x.get("start") is not None and x.get("end") is not None


def joint_of(e: dict) -> float:
    return float(e["joint"]) if e.get("joint") is not None else (float(e["start"]) + float(e["end"])) / 2


def joint_dist(j: float, x: dict) -> float:
    """Khoang cach tu moi noi toi vung khoanh (0 neu nam trong vung)."""
    return max(0.0, float(x["start"]) - j, j - float(x["end"]))


def seg_match(e: dict, x: dict, typ: str) -> bool:
    """Dong x (cung loai) trung thao tac e: xoa loi = moi noi cach vung <= 500 ms, con lai = giao vung."""
    if typ == "xoa":
        return joint_dist(joint_of(e), x) <= TOL + 1e-9
    return iou((x["start"], x["end"]), (e["start"], e["end"])) > 0


def edge_ok(e: dict, x: dict, typ: str) -> bool:
    """Req 10: hai mep lech <= 500 ms (xoa loi: moi noi cach vung khoanh <= 500 ms)."""
    if typ == "xoa":
        return joint_dist(joint_of(e), x) <= TOL + 1e-9
    return max(abs(float(x["start"]) - float(e["start"])), abs(float(x["end"]) - float(e["end"]))) <= TOL + 1e-9


def live_rows(r: dict) -> list:
    """Tep ra 04 khong co phan Thao tac (chi doan kha nghi) -> khong dong nao."""
    return list(r.get("rows") or []) if r.get("final") != "04" else []


def edit_detected(e: dict, rows: list) -> bool:
    """Thao tac that duoc bat dung loai: ca file = co dong ca file cung loai; mot doan = dong cung loai trung (seg_match)."""
    typ = edit_type(e)
    if not is_segment(e):
        return any(row_type(x) == typ and x.get("kind") == "global" for x in rows)
    return any(row_type(x) == typ and is_segment(x) and seg_match(e, x, typ) for x in rows)


# ---------------- Constraint ----------------
def k1_of(r: dict) -> dict:
    return r.get("k1") or (r.get("_m") or {}).get("k1") or {}


def constraint_reasons(r: dict) -> list | None:
    """Ly do Constraint neu tep bi Constraint chuyen 04 / tu choi (khong cham), None neu tep duoc cham."""
    if r.get("final") != "04":
        return None
    if r.get("constraint_reasons"):
        return list(r["constraint_reasons"])
    k1 = k1_of(r)
    if k1.get("decision") in CONSTRAINT_DEC:
        return list(k1.get("reasons") or []) or [f"K1 {k1.get('decision')} (khong ghi ly do)"]
    return None


def split_scored(rs: list) -> tuple[list, list]:
    """(tep duoc cham, tep bi Constraint loai)."""
    sc, ex = [], []
    for r in rs:
        (ex if constraint_reasons(r) is not None else sc).append(r)
    return sc, ex


# ---------------- dong gop tung tep ----------------
def contrib(r: dict) -> dict:
    """Dong gop cua MOT tep duoc cham vao cac tu so / mau so (moi chi tieu la ti so cua tong -> bootstrap theo cum duoc)."""
    c = dict.fromkeys(FEATS, 0.0)
    exp, fin = r.get("expected"), r.get("final")
    edits = list((r.get("_m") or {}).get("edits") or [])
    rows = live_rows(r)
    etypes = [edit_type(e) for e in edits]
    rtypes = [row_type(x) for x in rows]
    flagged = fin in ("02", "03")
    if exp in LABELS3 and fin in ("01", "02", "03", "04"):
        c[f"cm_{exp}_{fin}"] = 1.0
    if exp in ("02", "03"):
        c["n2"], c["k2"] = 1.0, float(flagged)
    if flagged:
        c["n3"], c["k3"] = 1.0, float(exp in ("02", "03"))
    if exp == "01":
        c["n4"], c["k4"] = 1.0, float(flagged)
        c["n01"], c["k01"] = 1.0, float(fin == "01")
    only_nr = all(t == "02_nen_toan_cuc" for t in etypes) if edits else r.get("kind") == "nen_toan_cuc"
    if only_nr:                                                # Req 5: chi khu on ca file
        c["n5"] = 1.0
        c["k5"] = float(not (fin == "03" or any(t.startswith("02_giong") for t in rtypes)))
        c["a5"] = float(fin == "04")
    if "ghep" in etypes and not ({"may", "may_file"} & set(etypes)):   # Req 6: ghep loi that (khong kem giong may)
        c["n6"], c["k6"] = 1.0, float(bool({"may", "may_file"} & set(rtypes)))
    if "may_file" in etypes:                                   # Req 8b
        c["n8b"], c["k8b"] = 1.0, float("may_file" in rtypes)
    if any(t in ("ghep", "xoa", "may", "may_file") for t in etypes):
        c["n33"], c["k33"] = 1.0, float(fin == "03")
    c["n9"], c["k9"] = 1.0, float(fin == "04")
    for _, typ in TYPED:                                       # 7a / 7b / 8a
        if typ == "may" and "may_file" in etypes:
            continue                                           # tep giong may CA FILE: dong doan khong tinh vao 8a
        true = [e for e, t in zip(edits, etypes) if t == typ and is_segment(e)]
        rws = [x for x, t in zip(rows, rtypes) if t == typ and is_segment(x)]
        used = set()
        for e in true:
            hit = {k for k, x in enumerate(rws) if seg_match(e, x, typ)}
            used |= hit
            c[f"hr_{typ}"] += float(bool(hit))
        c[f"nr_{typ}"] += len(true)
        c[f"tp_{typ}"] += len(used)
        c[f"fp_{typ}"] += len(rws) - len(used)
    for e, typ in zip(edits, etypes):                          # Req 10 / 11
        if not is_segment(e):
            continue
        same = [x for x, t in zip(rows, rtypes) if t == typ and is_segment(x)]
        c["n10"] += 1
        c["k10"] += float(any(edge_ok(e, x, typ) for x in same))
        if typ != "xoa":
            ov = [iou((x["start"], x["end"]), (e["start"], e["end"])) for x in same]
            ov = [v for v in ov if v > 0]
            c["n11m"] += 1                                     # 11m: cung tu so s11, thao tac bo sot = 0
            if ov:
                c["s11"] += max(ov)
                c["n11"] += 1
    loc = {t: [e for e, te in zip(edits, etypes) if te == t and is_segment(e)] for t in set(etypes) if t != "xoa"}
    for x, t in zip(rows, rtypes):                             # 11r: dong doan cung loai voi thao tac that cuc bo
        if is_segment(x) and loc.get(t):
            c["n11r"] += 1
            c["s11r"] += max(iou((x["start"], x["end"]), (e["start"], e["end"])) for e in loc[t])
    return c


def contrib_matrix(rs: list) -> np.ndarray:
    return np.array([[c[k] for k in FEATS] for c in map(contrib, rs)], float).reshape(len(rs), len(FEATS))


# ---------------- tu tong -> chi tieu (chay duoc tren vo huong hoac mang bootstrap) ----------------
def _div(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    np.divide(a, b, out=out, where=b > 0)
    return out


def _f1_terms(c: dict, lab: str):
    tp = np.asarray(c[f"cm_{lab}_{lab}"], float)
    fp = sum(np.asarray(c[f"cm_{e}_{lab}"], float) for e in LABELS3 if e != lab)
    fn = sum(np.asarray(c[f"cm_{lab}_{f}"], float) for f in ("01", "02", "03", "04") if f != lab)
    return 2 * tp, 2 * tp + fp + fn


def active_labels(c: dict) -> tuple:
    """Nhan co ho tro hoac co du doan trong tap (F1 xac dinh) - co dinh cho moi lan boc lai."""
    return tuple(lab for lab in LABELS3 if float(np.sum(_f1_terms(c, lab)[1])) > 0)


def _macro_f1(c: dict, labels: tuple | None = None):
    """labels=None: trung binh tren nhan co F1 xac dinh; labels cho truoc: F1 khong xac dinh -> 0 (khong bo nhan)."""
    f1s = []
    for lab in (LABELS3 if labels is None else labels):
        f = _div(*_f1_terms(c, lab))
        f1s.append(f if labels is None else np.nan_to_num(f))
    if not f1s:
        return np.full(np.shape(c["n9"]), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(f1s), axis=0)


def finish(c: dict, labels: tuple | None = None) -> tuple[dict, dict]:
    """Tong dong gop -> (gia tri chi tieu, mau so n). Gia tri NaN khi mau so = 0. labels: tap nhan co dinh cho Req 1."""
    v, n = {}, {}
    v["1"], n["1"] = _macro_f1(c, labels), c["n9"]
    for key, num, den in (("2", "k2", "n2"), ("3", "k3", "n3"), ("4", "k4", "n4"), ("5", "k5", "n5"), ("6", "k6", "n6"),
                          ("8b", "k8b", "n8b"), ("9", "k9", "n9"), ("10", "k10", "n10"), ("11", "s11", "n11"),
                          ("01", "k01", "n01"), ("33", "k33", "n33"), ("11r", "s11r", "n11r"), ("11m", "s11", "n11m")):
        v[key], n[key] = _div(c[num], c[den]), c[den]
    for key, typ in TYPED:
        tp, fp, nr, hr = (np.asarray(c[f"{p}_{typ}"], float) for p in ("tp", "fp", "nr", "hr"))
        P, R = _div(tp, tp + fp), _div(hr, nr)
        Pz, Rz = np.nan_to_num(P), np.nan_to_num(R)
        F = np.where(Pz + Rz > 0, 2 * Pz * Rz / np.where(Pz + Rz > 0, Pz + Rz, 1.0), 0.0)   # khong co dong nao -> F1 = 0
        v[key], v[key + "_P"], v[key + "_R"] = np.where(nr > 0, F, np.nan), P, R
        n[key], n[key + "_P"], n[key + "_R"] = nr, tp + fp, nr
    return v, n


def sums(rs: list) -> dict:
    X = contrib_matrix(rs)
    return dict(zip(FEATS, X.sum(0) if len(X) else np.zeros(len(FEATS))))


def _scal(x) -> float:
    return float(np.asarray(x).reshape(-1)[0]) if np.size(x) else float("nan")


# ---------------- ngu canh (kenh tat) va danh gia DAT ----------------
def infer_ctx(rs: list, params: dict | None = None) -> dict:
    """tts_off: kenh giong may tat (params ftts = 0 hoac khong co dong giong may nao); a04_off: khong co tep 04 phan tich
    nao (SUY TU DU LIEU, khong doc tu cau hinh he thong)."""
    params = params or {}
    any_tts = any(row_type(x) in ("may", "may_file") for r in rs for x in (r.get("rows") or []))
    tts_off = (params.get("ftts") is not None and float(params.get("ftts") or 0) == 0.0) or not any_tts
    a04 = sum(1 for r in rs if r.get("final") == "04" and constraint_reasons(r) is None)
    return {"tts_off": bool(tts_off), "a04_off": a04 == 0,
            "tts_why": "params ftts=0" if params.get("ftts") is not None and float(params.get("ftts") or 0) == 0.0
            else ("khong co dong giong may nao" if not any_tts else "")}


def vacancy(ctx: dict | None) -> dict:
    """key -> (kieu, ly do): 'kdd' = khong do duoc; 'fail' = gia tri that nhung khong the DAT (kenh tat)."""
    ctx = ctx or {}
    out = {}
    if ctx.get("tts_off"):
        out["6"] = ("kdd", "kenh giong may tat")
        out["8a"] = ("fail", "kenh giong may tat")
        out["8b"] = ("fail", "kenh giong may tat")
    if ctx.get("a04_off"):
        out["9"] = ("kdd", "khong co tep 04 phan tich nao, suy tu du lieu")
    return out


def judge(v: float, n: float, tgt: float, op: str, vac: tuple | None = None, lo: float = math.nan,
          hi: float = math.nan, min_n: int = MIN_N) -> tuple[str, str]:
    """(trang thai theo uoc luong, trang thai theo khoang tin cay). Khong bao gio DAT khi khong do duoc."""
    n = 0 if n is None or not np.isfinite(n) else int(round(n))
    if vac and vac[0] == "kdd":
        s = f"khong do duoc ({vac[1]}, n={n})"
        return s, s
    if n < min_n or not np.isfinite(v):
        s = f"khong do duoc (n={n})"
        return s, s
    ok = v >= tgt if op == ">=" else v <= tgt
    s = "DAT" if ok and not vac else ("chua" + (f" ({vac[1]})" if vac else ""))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return s, "-"
    ok_ci = (lo >= tgt if op == ">=" else hi <= tgt) and not vac
    return s, "DAT" if ok_ci else "chua"


# ---------------- cum + bootstrap ----------------
def person_of(r: dict) -> str:
    """Nguoi noi: meta.person; 'xxx:None' / thieu -> moi host_file mot nguoi (khong gop thanh mot nguoi gia)."""
    m = r.get("_m") or {}
    p = m.get("person") or m.get("speaker") or r.get("_g")
    if p is None or str(p) in ("", "None") or str(p).endswith(":None"):
        return f"{m.get('source')}:host:{m.get('host_file') or r.get('id')}"
    return str(p)


def cluster_of(r: dict, hosts_cluster: dict | None = None) -> str:
    """Cum bootstrap: vss -> video (meta.cluster / hosts.cluster), con lai person_of (voxp:None -> moi host_file mot cum)."""
    m = r.get("_m") or {}
    if m.get("source") == "vss":
        c = m.get("cluster") or (hosts_cluster or {}).get(m.get("host_file"))
        if c:
            return str(c)
    return person_of(r)


def boot_arrays(X: np.ndarray, clusters: list, n_boot: int = BOOT_N, seed: int = 0, chunk: int = 250) -> dict | None:
    """Bootstrap theo cum: boc lai cum co hoan lai, cong dong gop -> mang (n_boot,) cho tung chi tieu. < 2 cum -> None."""
    if len(X) == 0:
        return None
    uniq, inv = np.unique(np.asarray(clusters, dtype=object).astype(str), return_inverse=True)
    G = len(uniq)
    if G < 2:
        return None
    C = np.zeros((G, X.shape[1]))
    np.add.at(C, inv, X)
    labels = active_labels(dict(zip(FEATS, C.sum(0))))
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, G, size=(b, G))
        W = np.bincount((idx + (np.arange(b) * G)[:, None]).ravel(), minlength=b * G).reshape(b, G).astype(float)
        parts.append(W @ C)
    S = np.vstack(parts)
    v, _ = finish({k: S[:, j] for j, k in enumerate(FEATS)}, labels)
    return v


def ci_from(arr) -> tuple[float, float]:
    if arr is None:
        return math.nan, math.nan
    a = np.asarray(arr, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return math.nan, math.nan
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(rs: list, clusters: list | None = None, n_boot: int = 0, ctx: dict | None = None, seed: int = 0,
             keep_boot: bool = False) -> dict:
    """Cham mot tap tep (DA bo LOCKBOX / ngoai pham vi). Tu loai tep bi Constraint. Tra dict:
    v (gia tri), n (mau so), ci (lo, hi), st (trang thai), excl (Counter ly do), n_scored, n_excl, persons, boot (neu keep_boot)."""
    keep = [constraint_reasons(r) is None for r in rs]
    sc = [r for r, k in zip(rs, keep) if k]
    excl = Counter()
    for r, k in zip(rs, keep):
        if not k:
            for why in constraint_reasons(r):
                excl[why] += 1
    X = contrib_matrix(sc)
    v, n = finish(dict(zip(FEATS, X.sum(0) if len(X) else np.zeros(len(FEATS)))))
    v, n = {k: _scal(x) for k, x in v.items()}, {k: _scal(x) for k, x in n.items()}
    cl = [c for c, k in zip(clusters, keep) if k] if clusters is not None else None
    B = boot_arrays(X, cl, n_boot, seed) if (n_boot and cl is not None) else None
    ci = {k: ci_from(B[k]) if B is not None else (math.nan, math.nan) for k in v}
    vac = vacancy(ctx)
    st = {}
    for k, _, tgt, op in REQ:
        st[k] = judge(v[k], n[k], tgt, op, vac.get(k), *ci[k])
    for k, _ in EXTRA:
        st[k] = ("khong do duoc (n=%d)" % int(n[k]), "-") if n[k] < MIN_N else ("", "")
    persons = len({person_of(r) for r in sc})
    out = {"v": v, "n": n, "ci": ci, "st": st, "excl": excl, "n_scored": len(sc), "n_excl": len(rs) - len(sc),
           "n_cum": len(set(cl)) if cl else 0, "persons": persons, "vac": vac}
    if keep_boot:
        out["boot"] = B
    return out


# ---------------- API cu (giu tuong thich cho v6.boundary.*) ----------------
def metrics(rs, ctx: dict | None = None) -> dict:
    """Gia tri Requirement (float, NaN neu mau so 0) theo luat ban nop; tu loai tep bi Constraint chuyen 04.
    Them '_n' (mau so), '_vac' (chi tieu khong do duoc / kenh tat), '_excl' (so tep bi Constraint loai) - dang dict, khong vo huong."""
    rs = list(rs)
    ctx = infer_ctx(rs) if ctx is None else ctx
    e = evaluate(rs, ctx=ctx)
    m = dict(e["v"])
    m["_n"], m["_vac"], m["_excl"] = e["n"], e["vac"], {"n": e["n_excl"], "ly_do": dict(e["excl"])}
    return m


def typed_metrics(rs) -> dict:
    """6 = tep ghep loi that bi gan giong may / tep ghep loi that · 8a = F1 giong may MOT DOAN · 8b = tep giong may CA FILE
    duoc gan giong may ca file / tep giong may ca file (dong ca file trong tep giong may ca file khong tinh vao 8a)."""
    m = metrics(rs, ctx={})
    return {k: m[k] for k in ("6", "8a", "8a_R", "8a_P", "8b")}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    r = np.argsort(np.argsort(np.r_[pos, neg])) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def gate_diag(rs) -> list:
    L = []
    real = [r["gate"] - r["gate_thr"] for r in rs if r["expected"] == "01" and "gate" in r]
    ed = [r for r in rs if r["expected"] in ("02", "03") and "gate" in r]
    ms = [r["gate"] - r["gate_thr"] for r in ed]
    if not real or not ms:
        return ["(khong co diem cong trong ban ghi)"]
    L.append(f"AUC cong (tep sua vs tep that): **{auc(ms, real):.3f}** (n that {len(real)}, n sua {len(ms)})")
    q = [(f, float(np.quantile(real, 1 - f))) for f in (0.03, 0.05, 0.10, 0.20, 0.30)]
    L.append("TPR cua cong tai vu oan: " + " · ".join(f"{int(f * 100)} %: **{np.mean(np.array(ms) > t):.0%}**" for f, t in q))
    c = Counter(r["final"] for r in ed)
    gate_block = sum(1 for r, mg in zip(ed, ms) if mg < 0)
    L.append(f"Tep sua ({len(ed)}): bi CONG CHAN (-> 01) {gate_block} · ra 04 {c['04']} · ra 02 {c['02']} · ra 03 {c['03']}")
    by = {}
    for r, mg in zip(ed, ms):
        by.setdefault(r["kind"].split("_ky")[0].split("_vung")[0], []).append(mg)
    L.append("Ti le qua cong theo loai: " + " · ".join(f"{k} {np.mean(np.array(v) >= 0):.0%}" for k, v in sorted(by.items())))
    return L


def main(argv: list | None = None) -> None:
    from v6.natural import req_report
    req_report.main(sys.argv[1:] if argv is None else list(argv))


if __name__ == "__main__":
    main()
