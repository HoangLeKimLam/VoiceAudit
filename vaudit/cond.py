# -*- coding: utf-8 -*-
"""Uoc luong DIEU KIEN GHI AM bang mo hinh hoc tren vector gop tho XLS-R (tang 8 / 12 / 16; v6/k1x/cond_learn.py) - 3 dong Constraint:
  snr    tieng noi so voi on < 10 dB  -> 04 “ồn quá”;  10-15 dB -> canh bao “tiếng ồn cao”
  nen    nhac / TV / loa > 10 % nang luong -> 04 “nền che tiếng nói”
  vang   RT60 > 1,0 s -> canh bao “phòng vang”
Do chinh xac (CV theo nguoi noi, 243 tep co so, dieu kien mo phong): SNR AUC 0,84 · nen AUC 0,91 · vang AUC 0,997.
Nguong (10/10, WP2): CALIB ben duoi = max(nguong OOF tren mau mo phong, phan vi tren tep goc tu nhien HOP LE khong dung de hoc) de
  04 chan <= 2 % tep tu nhien hop le va canh bao on / vang o muc hop ly - so lieu: .tmp_v6/results/cond_calib.md (python -m vaudit.cond calib).
GIOI HAN (ghi vao tuyen bo gioi han / ticket): do NHAY dong SNR thap - mo phong OOF: 5 dB -> 04 'ồn quá' 26 %, 8 dB -> 14 %,
  12 dB -> canh bao 'tiếng ồn cao' 21 % => luat hop dong '< 10 dB -> 04' PHAN LON KHONG duoc thuc thi; nguong uu tien khong chan nham tep tot.
Thu thay the 12/10: SNR mu SQUIM (torchaudio SQUIM_OBJECTIVE, da huan luyen san DNS 2020, uoc luong SI-SDR) tren cung cach tron (40 tep co so,
  189 mau + 50 tep tu nhien; .tmp_v6/results/wp2_diag/squim_exp.py): AUC 0,82 (khong hon 0,84); tep goc vi 'sach' uoc luong trung vi 8 dB (57 % < 10 dB),
  tep tu nhien vi 52 % < 10 dB (en 20 %) -> nguong hop dong '< 10 dB' se chan nua tep vi hop le => KHONG dung; giu bo hoc XLS-R.
Chu ra la CUM TU HOP DONG nguyen van (vaudit.k1.WARN / reason); so do chi tiet -> measures.
Vector gop 6 144 chieu = [tang 8: mean, std][tang 12: ...][tang 16: ...] (xlsr_pooled.py / pipeline.encode).
  python -m vaudit.cond calib [--limit N]     hieu chuan nguong (doc .tmp_v6/cache/cond_learn.npz + xlsr_pooled cua natvi / naten) -> md + json
  python -m vaudit.cond test                  test (khong can du lieu lon)"""
from __future__ import annotations

import os
import sys

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cond_models.joblib")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_M: dict = {}
# Nguong da hieu chuan (None -> dung nguong trong joblib). Ghi tay tu ket qua 'calib' (xem docstring).
# 10/10 (cond_calib.md): nguong joblib da dat muc tieu tren tep goc tu nhien hop le khong dung de hoc (936 tep: vi 500 cung nguoi khac tep,
# en 436; 224 en khac nguoi): 04 'on qua' 0,4 % vi / 0 % en, 04 'nen che' 0,2 % / 0 %, canh bao 'tieng on cao' 2,8 % / 0 %, 'phong vang' 0 %
# -> giu nguyen (max(OOF, tu nhien) = nguong joblib). 135/150 canh bao on cu den tu SNR mu cua K1 (da bo), khong phai tu mo hinh nay.
# Gioi han: SNR 5 dB mo phong chi bi chan 26 % (AUC 0,84) - dong 'on qua' bat it.
CALIB: dict = {"snr": {"thr_04": 0.985, "thr_hi": 0.384}, "nen": {"thr_04": 1.431}, "vang": {"thr_warn": -0.542}}


def _load() -> dict:
    if "m" not in _M:
        import joblib
        _M["m"] = joblib.load(PATH)["models"] if os.path.exists(PATH) else {}
    return _M["m"]


def thr(task: str, key: str) -> float | None:
    """Nguong dang dung: CALIB neu da hieu chuan, khong thi nguong goc trong joblib."""
    v = CALIB.get(task, {}).get(key)
    if v is not None:
        return float(v)
    m = _load().get(task)
    return None if m is None or key not in m else float(m[key])


def score_many(task: str, X: np.ndarray) -> np.ndarray | None:
    """Diem dieu kien cho NHIEU vector gop (n, 6144); None neu chua co mo hinh."""
    m = _load().get(task)
    if m is None:
        return None
    B = m["pca"].transform(m["sc"].transform(np.atleast_2d(X.astype(np.float32))))
    p = m["gb"].predict_proba(B)
    return (m["lr"].decision_function(B) / 3 + np.log(np.clip(p[:, 1], 1e-4, 1 - 1e-4) / np.clip(p[:, 0], 1e-4, 1)) / 2) / 2


def score(task: str, vec: np.ndarray) -> float | None:
    s = score_many(task, vec)
    return None if s is None else float(s[0])


def decide(s: dict) -> tuple:
    """Luat tu diem {snr, nen, vang} -> (warnings, reasons) theo cum tu hop dong. Tach rieng de test khong can mo hinh."""
    from .k1 import WARN, reason
    W, R = [], []
    if s.get("snr") is not None and thr("snr", "thr_04") is not None:
        if s["snr"] > thr("snr", "thr_04"):
            R.append(reason("on_qua", "tiếng nói so với tiếng ồn dưới 10 dB (ước lượng bằng mô hình XLS-R)"))
        elif s["snr"] > thr("snr", "thr_hi"):
            W.append(WARN["on_cao"])
    if s.get("nen") is not None and thr("nen", "thr_04") is not None and s["nen"] > thr("nen", "thr_04"):
        R.append(reason("nen_che", "nhạc, TV, loa hoặc tiếng động vượt 10 % năng lượng (ước lượng bằng mô hình XLS-R)"))
    if s.get("vang") is not None and thr("vang", "thr_warn") is not None and s["vang"] > thr("vang", "thr_warn"):
        W.append(WARN["phong_vang"])
    return W, R


def assess(vec: np.ndarray) -> tuple:
    """(measures, warnings, reasons) cho mot kenh; reasons khong rong -> nhan 04 (cung dang k1.analyse_channel)."""
    m = _load()
    s = {t: score(t, vec) for t in ("snr", "nen", "vang") if t in m}
    M = {"diem_dieu_kien": {t: None if v is None else round(v, 3) for t, v in s.items()},
         "nguong_dieu_kien": {"snr_04": thr("snr", "thr_04"), "snr_canh_bao": thr("snr", "thr_hi"), "nen_04": thr("nen", "thr_04"),
                              "vang_canh_bao": thr("vang", "thr_warn")}}
    W, R = decide(s)
    return M, W, R


# ------------------------------------------------------------------ hieu chuan (nhe: chi doc vector da cache, khong nap XLS-R)
def _oof(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Diem OOF 5 fold theo nguoi noi, cung cong thuc mo hinh cuoi (chuan hoa -> PCA 64 -> LR + HGB)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        s = StandardScaler().fit(X[tr])
        pca = PCA(n_components=min(64, int(tr.sum()) - 1), random_state=0).fit(s.transform(X[tr]))
        A, B = pca.transform(s.transform(X[tr])), pca.transform(s.transform(X[te]))
        lr = LogisticRegression(C=0.05, max_iter=300, class_weight="balanced").fit(A, y[tr])
        gb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, class_weight="balanced", random_state=0).fit(A, y[tr])
        p = gb.predict_proba(B)
        sc[te] = (lr.decision_function(B) / 3 + np.log(np.clip(p[:, 1], 1e-4, 1 - 1e-4) / np.clip(p[:, 0], 1e-4, 1)) / 2) / 2
    return sc


TASKS = {"snr": (("snr",), lambda v: v < 10, lambda v: v >= 12), "nen": (("nhac", "tv"), lambda v: v >= 15, lambda v: v <= 10),
         "vang": (("vang",), lambda v: v >= 1.5, lambda v: v <= 1.0)}


def natural_scores(limit: int | None, exclude_base: set, exclude_person: set) -> dict:
    """Diem 3 dieu kien tren tep goc tu nhien HOP LE (natvi + naten, K1 da luu nhan / canh_bao, khong lockbox) KHONG dung de hoc cond."""
    import json
    import zlib
    pool = os.path.join(BASE, "data_labels", "realistic_v4", "xlsr_pooled")
    ids, persons, langs = [], [], []
    for st in ("natvi", "naten"):
        man = json.load(open(os.path.join(BASE, "data_labels", "v6", st, "manifest.json"), encoding="utf-8"))
        for e in man:
            p = str(e["meta"].get("person") or e["meta"].get("speaker"))
            if (e["kind"] == "nguyen_ban" and e["cfg"] == "goc" and not e["meta"].get("lockbox") and not e["meta"].get("ngoai_pham_vi")
                    and e["meta"].get("k1", {}).get("decision") in ("nhan", "canh_bao") and e["id"] not in exclude_base and p not in exclude_person
                    and os.path.exists(os.path.join(pool, e["id"] + ".npy"))):
                ids.append(e["id"]); persons.append(p); langs.append(st)
    if limit:
        ids, persons, langs = ids[:limit], persons[:limit], langs[:limit]
    X = np.stack([np.load(os.path.join(pool, i + ".npy")).astype(np.float32).ravel() for i in ids]) if ids else np.zeros((0, 6144), np.float32)
    out = {"ids": ids, "lang": np.array(langs), "half": np.array([zlib.crc32(p.encode()) % 2 for p in persons])}
    for t in ("snr", "nen", "vang"):
        out[t] = score_many(t, X) if len(X) else np.zeros(0)
    return out


def calib(limit: int | None = None) -> dict:
    """Chon nguong: 04 = max(OOF mo phong q98 am, tu nhien q98); canh bao on = max(OOF q90 am, tu nhien q90); vang = max(OOF q95, tu nhien q95).
    Tu nhien = TAT CA tep goc hop le khong dung de hoc (khong chia doi; sua docstring 11/10 - ban cu ghi 'chia doi theo nguoi' nhung code
    khong lam). Ti le bao cao tren chinh tap chon nguong -> KHONG phai ngoai mau; chi 'gan ngoai mau' vi nguong cuoi = nguong joblib (hoc tren
    mau mo phong). Ti le vi LAC QUAN: nguoi vi da co tep khac trong mau hoc."""
    import json
    sys.stdout.reconfigure(encoding="utf-8")
    z = np.load(os.path.join(BASE, ".tmp_v6", "cache", "cond_learn.npz"), allow_pickle=True)
    X, name, val, fold = z["X"].astype(np.float32), z["name"], z["val"], z["fold"]
    strict = natural_scores(limit, set(map(str, z["base"])), set(map(str, z["person"])))    # khac nguoi voi mau hoc (chi en: moi nguoi vi da dung)
    loose = natural_scores(limit, set(map(str, z["base"])), set())                        # khac TEP, co the cung nguoi (vi + en)
    L = ["# Hieu chuan nguong vaudit.cond (WP2, 10/10)\n", f"Mau mo phong: {len(X)} (OOF 5 fold theo nguoi). Tep goc tu nhien hop le khong dung de hoc: "
         f"khac nguoi {len(strict['ids'])} (en); khac tep {len(loose['ids'])} (vi {int((loose['lang'] == 'natvi').sum())}, en {int((loose['lang'] == 'naten').sum())}"
         " - vi deu la nguoi da co tep khac trong mau hoc -> uoc luong lac quan).\n"]
    res = {}
    for task, (axes, pos_f, neg_f) in TASKS.items():
        pos = np.array([n in axes and pos_f(v) for n, v in zip(name, val)])
        neg = np.array([(n in axes and neg_f(v)) or n == "clean" for n, v in zip(name, val)])
        sel = pos | neg
        sc = _oof(X[sel], pos[sel].astype(int), fold[sel])
        ns, vs, yneg = name[sel], val[sel], ~pos[sel]
        q = {"thr_04": 0.98, "thr_hi": 0.90} if task == "snr" else {"thr_04": 0.98} if task == "nen" else {"thr_warn": 0.95}
        old = {k: float(_load()[task][k]) for k in q}                     # nguong goc trong joblib
        nat_all = np.r_[strict[task], loose[task]]
        final = {k: float(max(np.quantile(sc[yneg], qq), np.quantile(nat_all, qq) if len(nat_all) else -np.inf)) for k, qq in q.items()}
        sets = {"khac nguoi (en)": strict[task], "khac tep vi": loose[task][loose["lang"] == "natvi"], "khac tep en": loose[task][loose["lang"] == "naten"]}
        res[task] = {"cu": old, "cuoi": final, "phan_vi_tu_nhien": {nm: [float(np.quantile(v, x)) for x in (0.5, 0.9, 0.98, 1.0)] for nm, v in sets.items() if len(v)}}
        L += [f"\n## {task}\n", "| nguong | joblib cu | cuoi = max(OOF, tu nhien) | " + " | ".join(f"{nm}: % > cu / cuoi" for nm in sets) + " |",
              "|---|---|---|" + "---|" * len(sets)]
        for k in q:
            L.append(f"| {k} | {old[k]:.3f} | {final[k]:.3f} | " + " | ".join(
                f"{100 * np.mean(v > old[k]):.1f} / {100 * np.mean(v > final[k]):.1f} %" if len(v) else "-" for v in sets.values()) + " |")
        L.append("\nPhan vi diem tu nhien (50 / 90 / 98 / max): " + "; ".join(f"{nm} " + " / ".join(f"{x:.2f}" for x in v) for nm, v in res[task]["phan_vi_tu_nhien"].items()))
        L += ["", "| dieu kien mo phong (OOF) | n | " + " | ".join(f"% > {k} cu | % > {k} cuoi" for k in q) + " |", "|---|---|" + "---|---|" * len(q)]
        for nm, v in sorted({(a, b) for a, b in zip(ns, vs)}):
            k_ = (ns == nm) & (vs == v)
            L.append(f"| {nm} {v:g} | {int(k_.sum())} | " + " | ".join(f"{100 * np.mean(sc[k_] > old[k]):.0f} % | {100 * np.mean(sc[k_] > final[k]):.0f} %" for k in q) + " |")
    L.append("\nGhi chu: vector XLS-R cua mau mo phong trich tren GPU fp32 (cond_learn.extract); san pham co the chay int8 tren CPU -> diem co the lech nhe.")
    out = os.path.join(BASE, ".tmp_v6", "results", "cond_calib")
    open(out + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(res, open(out + ".json", "w", encoding="utf-8"), indent=1)
    print("\n".join(L))
    return res


# ------------------------------------------------------------------ test
def test_decide_vocabulary() -> None:
    from .k1 import CONTRACT_WARNINGS
    saved = {k: dict(v) for k, v in CALIB.items()}
    try:
        CALIB.update(snr={"thr_04": 1.0, "thr_hi": 0.0}, nen={"thr_04": 1.0}, vang={"thr_warn": 0.0})
        assert decide({"snr": -1.0, "nen": -1.0, "vang": -1.0}) == ([], [])
        W, R = decide({"snr": 0.5, "nen": -1.0, "vang": 0.5})
        assert W == ["tiếng ồn cao", "phòng vang"] and not R and all(w in CONTRACT_WARNINGS for w in W)
        W, R = decide({"snr": 2.0, "nen": 2.0, "vang": None})
        assert not W and R[0].startswith("04 · “ồn quá”") and R[1].startswith("04 · “nền che tiếng nói”")
    finally:
        CALIB.clear(); CALIB.update(saved)


def test_thresholds_ordered() -> None:
    if not _load():
        print("    (bo qua: khong co cond_models.joblib)")
        return
    assert thr("snr", "thr_hi") < thr("snr", "thr_04"), (thr("snr", "thr_hi"), thr("snr", "thr_04"))
    M, W, R = assess(np.zeros(6144, np.float32))
    assert set(M["diem_dieu_kien"]) == {"snr", "nen", "vang"} and all(isinstance(w, str) for w in W + R)


def run_tests() -> None:
    import time
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  ok   {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:      # noqa: BLE001
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(tests) - fails}/{len(tests)} test dat")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    a = sys.argv[1:]
    if a and a[0] == "calib":
        calib(int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    else:
        run_tests()
