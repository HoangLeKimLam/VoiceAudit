# -*- coding: utf-8 -*-
"""HIEU CHINH dau ca tep (giao dien I1): diem OOF cua heads2 (.tmp_v6/cache/heads2_<heads>.pkl, fold theo NGUOI NOI = detect.person_fold) -> XAC SUAT
"tep co thao tac ca file loai nay", RIENG tung ngon ngu (vi / en) va tung dau (nen = 02 khu on ca file, giong = 02 doi giong ca file, ttsf = 03 giong may ca file).
Cach lam: voi moi (ngon ngu, dau) hoc 2 bo hieu chinh - hoi quy dang dieu (isotonic) va Platt (logistic tren logit diem) - CHEO THEO FOLD NGUOI NOI (hoc tren 4 fold, du doan
fold con lai) -> chon cach co Brier cheo thap hon; bo hieu chinh luu la ban hoc tren TOAN BO tep. Do tin cay: bang do tin cay (10 khoang, ti le duong + khoang Wilson 95 %),
ECE (+ khoang tin cay bootstrap theo NGUOI NOI), Brier, vu oan tren tep THAT o p >= 0,6 (luat ca tep cua hop dong), ti le roi vao vung xam 0,4-0,6.
LUU Y: xac suat phu thuoc ti le tep co thao tac trong tap hieu chinh (tep tong hop natvi / naten); ghi 'prevalence' trong tep de nguoi doc biet dieu kien.
  python -m v6.product.calib fit [--heads h2] [--cfg all|goc]     -> .tmp_v6/results/calib_<heads>.json + calib_<heads>.md
  python -m v6.product.calib show --heads h2 --key nen --lang vi --score 0.93
  python -m v6.product.calib report --heads h2                   (viet lai calib_<heads>.md tu json, khong hoc lai)
  python -m v6.product.calib selftest                            (kiem thu tren du lieu gia, khong doc cache)
Giao dien:  cal = load("h2")  ->  apply(cal, "nen", 0.93, lang="vi") -> float trong [0, 1]  (lang None -> bo hieu chinh gop 'all'); KeyError neu khong co bo nao.
            issue_of(cal, "ttsf", "vi") -> mo ta neu bo hieu chinh KHONG TIN CAY (bang do tin cay cheo lech > 0,2 lam doi vung quyet dinh) hoac None;
            pipeline.Product khong dung xac suat cua bo khong tin cay (head_prob[khoa] = None -> ticket ghi 'chua hieu chinh', giu nguong diem)."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pickle
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

RES = os.path.join(BASE, ".tmp_v6", "results")
CACHE = os.path.join(BASE, ".tmp_v6", "cache")
KEYS = (("nen", 0), ("giong", 1), ("ttsf", 2))          # khoa cua Product.analyse_channel det["head"] -> cot cua heads2 P / heads2.labels
NAT_SETS = ("v6natvi", "v6naten")
MIN_POS, MIN_NEG = 10, 30                                # it hon -> khong hoc rieng ngon ngu do (dung bo 'all')
RULE_P, GREY = 0.6, (0.4, 0.6)                           # luat hop dong: bao thao tac ca tep khi p >= 0,6; vung xam 0,4-0,6
N_BINS, N_BOOT = 10, 200
REL_GAP, REL_MIN_N = 0.2, 30                             # bo hieu chinh khong tin cay: khoang >= 30 tep lech > 0,2 lam doi vung quyet dinh (reliability_issue)
EPS = 1e-4


def path_of(tag: str) -> str:
    return os.path.join(RES, f"calib_{tag}.json")


def load(tag: str) -> dict:
    """Doc .tmp_v6/results/calib_<tag>.json (tag = ten dau ca tep, vd h2). FileNotFoundError neu chua hieu chinh."""
    with open(path_of(tag), encoding="utf-8") as fh:
        cal = json.load(fh)
    if cal.get("heads") != tag:
        raise ValueError(f"calib_{tag}.json ghi heads={cal.get('heads')!r}, khong khop {tag!r}")
    return cal


def _pick(cal: dict, key: str, lang: str | None) -> dict:
    if ":" in key and lang is None:                      # cho phep khoa dang "vi:nen"
        lang, key = key.split(":", 1)
    tab = cal["cal"]
    for lg in ([lang] if lang else []) + ["all"]:
        c = tab.get(lg, {}).get(key)
        if c and c.get("method") in ("isotonic", "platt"):
            return c
    raise KeyError(f"khong co bo hieu chinh cho dau {key!r} (ngon ngu {lang!r} va 'all')")


def _apply_one(c: dict, s: np.ndarray) -> np.ndarray:
    if c["method"] == "isotonic":
        return np.interp(s, np.asarray(c["x"], float), np.asarray(c["y"], float))
    z = np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS)))
    return 1.0 / (1.0 + np.exp(-(c["a"] * z + c["b"])))


def apply(cal: dict, key: str, score: float, lang: str | None = None) -> float:
    """Diem tho cua dau (cung thang heads2.predict_final) -> xac suat trong [0, 1]."""
    p = float(_apply_one(_pick(cal, key, lang), np.asarray([float(score)]))[0])
    return min(1.0, max(0.0, p))


def method_of(cal: dict, key: str, lang: str | None = None) -> str:
    """'isotonic' / 'platt' (+ ' (gop all)' neu ngon ngu nay khong du du lieu)."""
    c = _pick(cal, key, lang)
    own = cal["cal"].get(lang or "all", {}).get(key) is c
    return c["method"] + ("" if own else " (gop all)")


def zone(p: float) -> int:
    """Vung quyet dinh cua luat hop dong: 2 = bao (p >= 0,6), 1 = vung xam (0,4-0,6), 0 = khong bao."""
    return 2 if p >= RULE_P else 1 if p >= GREY[0] else 0


def reliability_issue(c: dict) -> str | None:
    """Bo hieu chinh KHONG TIN CAY neu bang do tin cay CHEO co khoang n >= REL_MIN_N ma |p trung binh - ti le quan sat| > REL_GAP, p nam ngoai khoang Wilson
    cua ti le quan sat VA lech do doi vung quyet dinh (bao / vung xam / khong bao) - vd vi / ttsf [0,6;0,7): p 0,67 nhung quan sat 0,19 -> bao sai giong may ca file.
    Lech khong doi vung (vd vi / nen [0,7;0,8) du tin hon p) khong tinh. Tra mo ta khoang lech dau tien hoac None."""
    for lo, hi, n, pm, obs, wl, wh in c.get("rel") or []:
        if n >= REL_MIN_N and abs(pm - obs) > REL_GAP and not (wl <= pm <= wh) and zone(pm) != zone(obs):
            return f"khoang [{lo:.1f};{hi:.1f}) n={n}: p {pm:.2f}, quan sat {obs:.2f} ({wl:.2f}-{wh:.2f})"
    return None


def issue_of(cal: dict, key: str, lang: str | None = None) -> str | None:
    """reliability_issue cua bo hieu chinh thuc dung cho (khoa, ngon ngu) (ke ca khi roi ve 'all'); KeyError neu khong co bo nao."""
    return reliability_issue(_pick(cal, key, lang))


# ------------------------------------------------------------------ hoc bo hieu chinh
def fit_isotonic(s: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True).fit(s, y)
    return {"method": "isotonic", "x": [round(float(v), 6) for v in iso.X_thresholds_], "y": [round(float(v), 6) for v in iso.y_thresholds_]}


def fit_platt(s: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression
    z = np.log(np.clip(s, EPS, 1 - EPS) / (1 - np.clip(s, EPS, 1 - EPS)))[:, None]
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(z, y)
    return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    tot = 0.0
    for k in range(n_bins):
        m = b == k
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / max(1, len(p)))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list:
    """[[lo, hi, n, p trung binh, ti le duong, Wilson thap, Wilson cao]] cho moi khoang co tep."""
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for k in range(n_bins):
        m = b == k
        if m.any():
            lo, hi = wilson(int(y[m].sum()), int(m.sum()))
            out.append([k / n_bins, (k + 1) / n_bins, int(m.sum()), round(float(p[m].mean()), 4), round(float(y[m].mean()), 4), round(lo, 4), round(hi, 4)])
    return out


def boot_ece(p: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Khoang 95 % cua ECE: bootstrap lay mau lai theo NGUOI NOI (cum), khong theo tep."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx = [np.flatnonzero(inv == g) for g in range(len(ug))]
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.integers(0, len(ug), len(ug))])
        vals.append(ece(p[pick], y[pick]))
    return (round(float(np.quantile(vals, 0.025)), 4), round(float(np.quantile(vals, 0.975)), 4))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def fit_one(s: np.ndarray, y: np.ndarray, fold: np.ndarray, groups: np.ndarray, real: np.ndarray) -> dict:
    """Mot (ngon ngu, dau): cheo theo fold -> chon cach (Brier) -> hoc lai tren toan bo; tra dict luu duoc vao JSON."""
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    base = {"n": int(len(y)), "n_pos": n_pos, "n_real": int(real.sum()), "prevalence": round(n_pos / max(1, len(y)), 4)}
    if n_pos < MIN_POS or n_neg < MIN_NEG:
        return {**base, "method": "none", "reason": f"qua it du lieu (duong {n_pos}, am {n_neg})"}
    cv = {}
    for name, fit in (("isotonic", fit_isotonic), ("platt", fit_platt)):
        p = np.full(len(y), np.nan)
        for f in np.unique(fold):
            tr, te = fold != f, fold == f
            if y[tr].sum() == 0 or y[tr].sum() == tr.sum():
                p[te] = y[tr].mean() if tr.any() else 0.0
                continue
            p[te] = _apply_one(fit(s[tr], y[tr]), s[te])
        cv[name] = p
    best = min(cv, key=lambda k: brier(cv[k], y))
    p = np.clip(cv[best], 0.0, 1.0)
    fin = (fit_isotonic if best == "isotonic" else fit_platt)(s, y)
    pos = y == 1
    stats = {"brier": round(brier(p, y), 5), "brier_other": round(brier(np.clip(cv[[k for k in cv if k != best][0]], 0, 1), y), 5),
             "ece": round(ece(p, y), 4), "ece_ci": boot_ece(p, y, groups),
             "fa_real_p06": round(float(np.mean(p[real] >= RULE_P)), 4) if real.any() else None,
             "fa_real_p06_ci": [round(v, 4) for v in wilson(int((p[real] >= RULE_P).sum()), int(real.sum()))] if real.any() else None,
             "recall_p06": round(float(np.mean(p[pos] >= RULE_P)), 4),
             "grey_all": round(float(np.mean((p >= GREY[0]) & (p < GREY[1]))), 4),
             "grey_real": round(float(np.mean((p[real] >= GREY[0]) & (p[real] < GREY[1]))), 4) if real.any() else None}
    rel = reliability(p, y)
    return {**base, **fin, "cv": stats, "rel": rel, "issue": reliability_issue({"rel": rel})}


def calibrate(S: np.ndarray, Y: np.ndarray, lang: np.ndarray, fold: np.ndarray, groups: np.ndarray, real: np.ndarray) -> dict:
    """S, Y: (n, 3) diem / nhan theo KEYS; tra {lang | 'all': {key: bo hieu chinh}}."""
    out = {}
    for lg in sorted(set(lang.tolist())) + ["all"]:
        m = np.ones(len(lang), bool) if lg == "all" else lang == lg
        out[lg] = {k: fit_one(S[m, c].astype(float), Y[m, c].astype(int), fold[m], groups[m], real[m]) for k, c in KEYS}
    return out


def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fit(heads: str, cfg_mode: str) -> dict:
    from v6.boundary import detect as D
    from v6.boundary import heads2 as H
    sys.stdout.reconfigure(encoding="utf-8")
    src = os.path.join(CACHE, f"heads2_{heads}.pkl")
    h = pickle.load(open(src, "rb"))
    items = D.load_items(keep=("edits", "level2", "lang", "person", "speaker", "source"))
    hid = {i: n for n, i in enumerate(h["ids"])}
    ids = [i for i in h["ids"] if i in items and items[i]["set"] in NAT_SETS and (cfg_mode == "all" or items[i]["cfg"] == "goc")]
    lang = np.array([str(items[i]["meta"].get("lang") or ("en" if items[i]["set"] == "v6naten" else "vi"))[:2] for i in ids])
    S = np.stack([h["P"][hid[i]] for i in ids]).astype(float)
    Y = np.array([H.labels(items[i]["meta"]) for i in ids])
    fold = np.array([items[i]["fold"] for i in ids])
    groups = np.array([str(items[i]["meta"].get("person") or items[i]["meta"].get("speaker")) for i in ids])
    real = np.array([items[i]["real"] for i in ids])
    cal = {"heads": heads, "made": datetime.datetime.now().isoformat(timespec="seconds"),
           "source": {"oof": os.path.basename(src), "sha1": sha1_file(src), "sets": list(NAT_SETS), "cfg": cfg_mode, "folds": "detect.person_fold (crc32(person) % 5)"},
           "rule": {"report_p": RULE_P, "grey": list(GREY)}, "keys": {k: c for k, c in KEYS},
           "note": "xac suat co dieu kien theo ti le thao tac cua tap hieu chinh (natvi / naten tong hop); app NS mac dinh = 01 (khong phai duong cua 'nen')",
           "cal": calibrate(S, Y, lang, fold, groups, real)}
    with open(path_of(heads) + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(cal, fh, indent=1, ensure_ascii=False)
    os.replace(path_of(heads) + ".tmp", path_of(heads))
    md = report(cal)
    open(os.path.join(RES, f"calib_{heads}.md"), "w", encoding="utf-8").write(md)
    print(md)
    return cal


def report(cal: dict) -> str:
    L = [f"# Hieu chinh dau ca tep {cal['heads']} (OOF fold nguoi noi, {cal['source']['cfg']}; {cal['made']})\n",
         f"Nguon: {cal['source']['oof']} sha1 {cal['source']['sha1'][:12]}. Luat hop dong: bao thao tac ca tep khi p >= {RULE_P}; vung xam {GREY[0]}-{GREY[1]}.\n",
         "| ngon ngu | dau | cach | n / duong / that | ti le duong | ECE cheo [95 %] | Brier (cach kia) | vu oan that p>=0,6 [Wilson] | bat p>=0,6 | vung xam (that) | tin cay |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lg, d in cal["cal"].items():
        for k, c in d.items():
            if c["method"] == "none":
                L.append(f"| {lg} | {k} | khong ({c['reason']}) | {c['n']} / {c['n_pos']} / {c['n_real']} | {c['prevalence']:.3f} | | | | | | |")
                continue
            v = c["cv"]
            fa = f"{v['fa_real_p06']:.3f} [{v['fa_real_p06_ci'][0]:.3f}-{v['fa_real_p06_ci'][1]:.3f}]" if v["fa_real_p06"] is not None else "-"
            L.append(f"| {lg} | {k} | {c['method']} | {c['n']} / {c['n_pos']} / {c['n_real']} | {c['prevalence']:.3f} | {v['ece']:.3f} [{v['ece_ci'][0]:.3f}-{v['ece_ci'][1]:.3f}] | "
                     f"{v['brier']:.4f} ({v['brier_other']:.4f}) | {fa} | {v['recall_p06']:.3f} | {v['grey_all']:.3f} ({v['grey_real'] if v['grey_real'] is not None else '-'}) | "
                     f"{'KHONG: ' + iss if (iss := reliability_issue(c)) else 'co'} |")
    L += ["", "Ghi chu: khoang ECE la bootstrap phan vi theo NGUOI NOI; ECE lay mau lai bi thien lech LEN (luon >= 0, tang do nhieu) nen khoang co the nam tren diem uoc luong "
          "(vd all / giong). Xac suat la co dieu kien theo ti le thao tac cua tap tong hop; ttsf tieng Anh khong co tep duong -> dung bo 'all' (pipeline ghi head_prob_method).",
          f"Cot 'tin cay' (tinh lai tu bang do tin cay luc doc, khong hoc lai): KHONG = co khoang >= {REL_MIN_N} tep lech > {REL_GAP} lam doi vung quyet dinh "
          "(bao >= 0,6 / vung xam / khong bao); pipeline KHONG dung xac suat cua bo do (head_prob = None, ticket ghi 'chua hieu chinh', giu nguong diem)."]
    L += ["", "## Bang do tin cay (p cheo theo fold)", ""]
    for lg, d in cal["cal"].items():
        for k, c in d.items():
            if c.get("rel"):
                L.append(f"- **{lg} / {k}**: " + " · ".join(f"[{r[0]:.1f},{r[1]:.1f}) n={r[2]} p={r[3]:.2f} quan sat {r[4]:.2f} ({r[5]:.2f}-{r[6]:.2f})" for r in c["rel"]))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ kiem thu (du lieu gia)
def _fake(n: int = 1500, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    y = (rng.random((n, 3)) < np.array([0.2, 0.15, 0.04])).astype(int)
    S = np.clip(0.35 + 0.4 * y + rng.normal(0, 0.15, (n, 3)), 0, 1)
    lang = np.where(rng.random(n) < 0.6, "vi", "en")
    y[lang == "en", 2] = 0                                          # khong co giong may tieng Anh -> phai roi ve bo 'all'
    groups = np.array([f"p{k}" for k in rng.integers(0, 120, n)])
    fold = np.array([int(g[1:]) % 5 for g in groups])
    real = y.sum(1) == 0
    return S, y, lang, fold, groups, real


def test_calibrated_monotone_and_bounded() -> None:
    S, Y, lang, fold, groups, real = _fake()
    cal = {"heads": "fake", "cal": calibrate(S, Y, lang, fold, groups, real)}
    for lg in ("vi", "en", None):
        ps = [apply(cal, "nen", s, lang=lg) for s in np.linspace(0, 1, 21)]
        assert all(0.0 <= p <= 1.0 for p in ps) and all(b >= a - 1e-9 for a, b in zip(ps, ps[1:])), ps
    assert apply(cal, "nen", 0.95, "vi") > apply(cal, "nen", 0.2, "vi")


def test_fallback_to_all_when_language_lacks_positives() -> None:
    S, Y, lang, fold, groups, real = _fake()
    cal = {"heads": "fake", "cal": calibrate(S, Y, lang, fold, groups, real)}
    assert cal["cal"]["en"]["ttsf"]["method"] == "none"
    assert method_of(cal, "ttsf", "en").endswith("(gop all)")
    assert abs(apply(cal, "ttsf", 0.8, "en") - apply(cal, "ttsf", 0.8, None)) < 1e-12
    assert abs(apply(cal, "en:nen", 0.7) - apply(cal, "nen", 0.7, "en")) < 1e-12


def test_missing_key_raises() -> None:
    cal = {"heads": "x", "cal": {"all": {"nen": {"method": "none"}}}}
    try:
        apply(cal, "nen", 0.5)
    except KeyError:
        return
    raise AssertionError("phai KeyError khi khong co bo hieu chinh")


def test_cross_fit_stats_sane() -> None:
    S, Y, lang, fold, groups, real = _fake(seed=3)
    c = fit_one(S[:, 0], Y[:, 0], fold, groups, real)
    assert c["method"] in ("isotonic", "platt") and 0 <= c["cv"]["ece"] < 0.1, c["cv"]
    assert c["cv"]["ece_ci"][0] <= c["cv"]["ece"] + 1e-9 and sum(r[2] for r in c["rel"]) == len(S)
    assert 0 <= c["cv"]["fa_real_p06"] <= 0.2 and c["cv"]["recall_p06"] >= 0.3
    lo, hi = wilson(5, 100)
    assert lo < 0.05 < hi


def test_reliability_flag_only_when_decision_zone_changes() -> None:
    ttsf = {"method": "isotonic", "x": [0, 1], "y": [0, 1], "rel": [[0.0, 0.1, 11425, 0.01, 0.0, 0.0, 0.01], [0.6, 0.7, 32, 0.67, 0.19, 0.09, 0.35]]}     # calib_h2 vi / ttsf
    nen = {"method": "platt", "a": 1.0, "b": 0.0, "rel": [[0.5, 0.6, 86, 0.55, 0.71, 0.61, 0.79], [0.7, 0.8, 37, 0.75, 0.97, 0.86, 1.0]]}           # calib_h2 vi / nen: lech nhung cung vung
    small = {"method": "platt", "a": 1.0, "b": 0.0, "rel": [[0.6, 0.7, 12, 0.65, 0.1, 0.02, 0.4]]}                                                     # n < 30: khong du de ket luan
    assert "0.67" in reliability_issue(ttsf) and reliability_issue(nen) is None and reliability_issue(small) is None
    cal = {"heads": "x", "cal": {"vi": {"ttsf": ttsf, "nen": nen}, "all": {"ttsf": ttsf}}}
    assert issue_of(cal, "ttsf", "vi") and issue_of(cal, "ttsf", "en") and issue_of(cal, "nen", "vi") is None      # en roi ve 'all' -> cung khong tin cay
    assert zone(0.6) == 2 and zone(0.4) == 1 and zone(0.39) == 0
    S, Y, lang, fold, groups, real = _fake(seed=2)
    c = fit_one(S[:, 0], Y[:, 0], fold, groups, real)
    assert "issue" in c and c["issue"] == reliability_issue(c)


def test_json_roundtrip_and_load_checks_tag() -> None:
    import tempfile
    S, Y, lang, fold, groups, real = _fake(n=600, seed=5)
    cal = {"heads": "zz", "cal": calibrate(S, Y, lang, fold, groups, real)}
    global RES
    old = RES
    with tempfile.TemporaryDirectory() as d:
        RES = d
        try:
            json.dump(cal, open(path_of("zz"), "w", encoding="utf-8"))
            back = load("zz")
            assert abs(apply(back, "giong", 0.66, "vi") - apply(cal, "giong", 0.66, "vi")) < 1e-9
            json.dump({**cal, "heads": "other"}, open(path_of("yy"), "w", encoding="utf-8"))
            try:
                load("yy")
                raise AssertionError("phai tu choi tep ghi heads khac")
            except ValueError:
                pass
        finally:
            RES = old
    assert report({**cal, "made": "-", "source": {"oof": "-", "sha1": "0" * 40, "cfg": "all"}}).startswith("# Hieu chinh")


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
    ap.add_argument("step", choices=["fit", "show", "report", "selftest"], nargs="?", default="selftest")
    ap.add_argument("--heads", default="h2")
    ap.add_argument("--cfg", choices=["all", "goc"], default="all", help="all = goc + 4 kenh ung dung (app NS mac dinh = 01); goc = chi tep goc")
    ap.add_argument("--key", default="nen")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--score", type=float, default=0.5)
    a = ap.parse_args()
    if a.step == "fit":
        fit(a.heads, a.cfg)
    elif a.step == "report":                                     # viet lai calib_<heads>.md tu tep json (khong hoc lai)
        sys.stdout.reconfigure(encoding="utf-8")
        md = report(load(a.heads))
        open(os.path.join(RES, f"calib_{a.heads}.md"), "w", encoding="utf-8").write(md)
        print(md)
    elif a.step == "show":
        c = load(a.heads)
        print(f"{a.key} {a.lang} {a.score} -> p = {apply(c, a.key, a.score, a.lang):.4f} ({method_of(c, a.key, a.lang)})")
    else:
        selftest()
