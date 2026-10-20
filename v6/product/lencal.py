# -*- coding: utf-8 -*-
"""NGUONG THEO DO DAI TEP (K7, 18/10): sua vu oan tang theo do dai.

  Van de: nguong moi kenh doan (ghep / xoa = dinh ranh gioi; den / voi = vung khu on / doi giong) = phan vi 1 - q cua diem LON NHAT
  moi tep THAT tu nhien (fuse9.Ctx9.th) - tep hieu chinh dai trung vi 8,8 s (toi da 22 s). Tep dai d co ~d / 9 lan nhieu khung hon -> xac
  suat co it nhat mot lan vuot nguong ngau nhien tang theo d (do tren tep that dai 1-10 phut: vu oan 32 % -> 75 %, e2e_v9F_long).
  Cach sua (thong ke quet, duoi mu): cat tep that thanh khuc BLOCK_S giay, lay diem lon nhat moi khuc, tren thang logit duoi tren
  ~ mu voi thang sigma (uoc luong hop ly cuc dai tren phan vuot nguong TAIL_Q) -> ti le lan vuot nguong / giay lambda(t) ~ exp(-t / sigma)
  -> giu so lan vu oan ky vong / tep KHONG DOI theo do dai:
      logit T(d) = logit T + sigma * ln(d / d_ref)      (duoi mu; Pareto tong quat xi > 0: shift_logit)
  d_ref = do dai TRUNG VI tep hieu chinh (8,8 s) va HAI CHIEU (tep ngan hon cung ha nguong): chon tren OOF natvi / naten (lencal regress
  18/10: hai chieu 8,8 s tot hon mot chieu 10 s / 20 s ca hai ngon ngu - Req 2 en giu 0,741, vi 0,550; Req 4 en 6,7 %, vi 7,9 % so voi 6,8 / 9,4 %).
  Duoi: Pareto tong quat tu phan vi 0,8 (lencal check 18/10: phang theo do dai hon duoi mu; vi den / voi co xi ~ 0,15).
  sigma uoc luong theo (ngon ngu, kenh, fold): fold k = chi tep that cua cac fold KHAC (nhu thr_fold[k]); 'all' = moi fold (mo hinh cuoi).
  Dau ca tep (nen / giong / ttsf) KHONG dieu chinh: diem tren vector gop ca tep, khong phai thong ke quet.
  Kiem dinh (dev, KHONG dung tap kiem thu): chuoi gia dai d ghep tu diem OOF cua tep that fold k (ghep trong khong gian DIEM, khong tao moi
  noi am thanh) giai ma bang dung bo giai ma san pham (detect.decode_file + fusion.decode_regions) voi thr_fold[k] cu / da dieu chinh;
  them tep that AMI dai (amilong, KHONG lockbox) neu co trong OOF.
CLI: python -m v6.product.lencal fit   [--bound fuse1 --thr thresholds_v9F.json]    -> .tmp_v6/results/lencal_v9F.json
     python -m v6.product.lencal regress                                             -> lencal_v9F_regress.md (gia tren tep NGAN, OOF)
     python -m v6.product.lencal check [--shuffles 6]                                -> .tmp_v6/results/lencal_v9F_check.md
     python -m v6.product.lencal selftest"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

RES = os.path.join(BASE, ".tmp_v6", "results")
D_REF, BLOCK_S, TAIL_Q = 10.0, 4.0, 0.80
DT = 0.02
EPS = 1e-6
CH = {"ghep": 0, "xoa": 1, "den": 2, "voi": 3, "spl": 4, "ttsr": 5}      # cot cua mang diem fuse1 (T, 6)
SEG_CH = ("ghep", "xoa", "den", "voi", "spl", "ttsr")                     # chi kenh DOAN (dau ca tep khong dieu chinh)
DURS = (10, 30, 60, 120, 240, 400, 560)


def logit(p: float) -> float:
    p = min(max(float(p), EPS), 1 - EPS)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def block_max(p: np.ndarray, block_s: float = BLOCK_S) -> np.ndarray:
    """Diem lon nhat moi khuc block_s giay khong chong; khuc cuoi giu neu >= nua khuc (ham thuan)."""
    n = int(round(block_s / DT))
    out = [p[a:a + n].max() for a in range(0, len(p), n) if len(p[a:a + n]) >= n // 2]
    return np.asarray(out, float)


def fit_sigma(vals: np.ndarray, tail_q: float = TAIL_Q, tail: str = "exp") -> dict:
    """Duoi tren thang logit cua diem lon nhat moi khuc: u = phan vi tail_q, phan vuot ~ mu (tail='exp', xi = 0) hoac Pareto tong quat
    (tail='gpd', xi in [0; 0,5]). Kem chan doan: ti le vuot o vai muc (thuc te vs mo hinh) de biet duoi khop khong."""
    x = np.array([logit(v) for v in vals])
    u = float(np.quantile(x, tail_q))
    ex = x[x > u] - u
    xi, sig = 0.0, float(ex.mean()) if len(ex) else float("nan")
    if tail == "gpd" and len(ex) >= 50:
        from scipy.stats import genpareto
        c, _, scale = genpareto.fit(ex, floc=0)
        xi, sig = float(min(max(c, 0.0), 0.5)), float(scale)
        if c < 0:                                   # duoi nhe hon mu -> dung mu (bao thu: dieu chinh manh hon)
            sig = float(ex.mean())
    diag = {}
    for q in (0.95, 0.99, 0.995):
        t = float(np.quantile(x, q))
        diag[str(q)] = {"thuc_te": round(1 - q, 4), "mo_hinh": round((1 - tail_q) * _sf(t - u, sig, xi), 4)}
    return {"sigma": round(sig, 4), "xi": round(xi, 4), "u": round(u, 4), "n": int(len(x)), "n_tail": int(len(ex)), "tail": tail, "diag": diag}


def _sf(z: float, sig: float, xi: float) -> float:
    """P(phan vuot > z) cua duoi mu (xi = 0) / Pareto tong quat."""
    if not sig > 0:
        return float("nan")
    if xi < 1e-9:
        return math.exp(-z / sig)
    return max(1 + xi * z / sig, 1e-12) ** (-1 / xi)


def shift_logit(t0: float, r: float, prm) -> float:
    """Nguong logit moi khi tep dai gap r lan d_ref: giu ti le vuot / tep (ham thuan). prm = sigma (so) hoac {sigma, xi, u}."""
    if not isinstance(prm, dict):
        prm = {"sigma": float(prm), "xi": 0.0, "u": t0}
    sig, xi, u = prm["sigma"], prm.get("xi", 0.0), prm.get("u", t0)
    if xi < 1e-9:
        return t0 + sig * math.log(r)
    inner = max(1 + xi * (t0 - u) / sig, 1e-6)
    return u + (sig / xi) * (r ** xi * inner - 1)


def adjust(thr: dict, dur: float, prm: dict, d_ref: float = D_REF, two_sided: bool = False) -> dict:
    """Nguong da dieu chinh theo do dai (ham thuan). prm = {kenh: sigma | {sigma, xi, u}}; kenh thieu / dau ca tep giu nguyen.
    two_sided: tep ngan hon d_ref cung HA nguong (vu oan / tep deu theo do dai; d_ref = do dai trung vi tep hieu chinh)."""
    if dur is None or dur <= 0 or (dur <= d_ref and not two_sided) or dur == d_ref:
        return dict(thr)
    out = dict(thr)
    for k in SEG_CH:
        p = prm.get(k)
        sig = p.get("sigma") if isinstance(p, dict) else p
        if k in thr and sig and sig > 0:
            out[k] = sigmoid(shift_logit(logit(thr[k]), dur / d_ref, p))
    return out


def sigma_for(cal: dict, lang: str, fold) -> dict:
    """{kenh: sigma} cua bo hieu chinh cho (ngon ngu, fold); fold None / khong co -> 'all'."""
    by = cal["sigma"].get(lang) or next(iter(cal["sigma"].values()))
    f = str(fold) if fold is not None and str(fold) in by else "all"
    return {k: {"sigma": v["sigma"], "xi": v.get("xi", 0.0), "u": v.get("u")} for k, v in by[f].items()}


# ------------------------------------------------------------------ du lieu OOF
def load_ctx(bound: str = "fuse1") -> dict:
    from v6.boundary import detect as D
    from v6.boundary import fusion as F
    items = D.load_items(keep=("level2", "lang", "source", "lockbox", "k1"))
    ob = pickle.load(open(D.tagged(D.OOFP, bound), "rb"))
    lang = lambda i: str(items[i]["meta"].get("lang") or ("en" if items[i]["set"] == "v6naten" else "vi"))[:2]   # noqa: E731  (nhu Ctx9)
    real = [i for i, it in items.items() if it["real"] and it["set"] in F.NAT_SETS and it["cfg"] == "goc" and i in ob]
    ami = [i for i, it in items.items() if it["set"] == "v6amilong" and it["real"] and not it["meta"].get("lockbox") and i in ob]
    return {"items": items, "P": {i: ob[i].astype(np.float32) for i in real + ami}, "real": real, "ami": ami,
            "lang": {i: lang(i) for i in real + ami}, "fold": {i: items[i]["fold"] for i in real + ami}}


def fit(ctx: dict, thr_file: str, tail_q: float = TAIL_Q, tail: str = "exp") -> dict:
    thr = json.load(open(os.path.join(RES, thr_file), encoding="utf-8"))
    chans = [k for k in SEG_CH if any(k in v for v in thr["thr"].values())]
    d_med = round(float(np.median([len(ctx["P"][i]) * DT for i in ctx["real"]])), 1)
    out = {"made_by": "v6.product.lencal fit", "thr_file": thr_file, "d_ref": d_med, "two_sided": True, "block_s": BLOCK_S, "tail_q": tail_q,
           "tail": tail,
           "channels": chans, "n_real": len(ctx["real"]),
           "dur_real_s": {"median": round(float(np.median([len(ctx['P'][i]) * DT for i in ctx['real']])), 2),
                          "max": round(float(max(len(ctx['P'][i]) * DT for i in ctx['real'])), 2)},
           "sigma": {}}
    for lg in sorted({ctx["lang"][i] for i in ctx["real"]}):
        out["sigma"][lg] = {}
        for f in ("all", "0", "1", "2", "3", "4"):
            ids = [i for i in ctx["real"] if ctx["lang"][i] == lg and (f == "all" or str(ctx["fold"][i]) != f)]
            out["sigma"][lg][f] = {k: fit_sigma(np.concatenate([block_max(ctx["P"][i][:, CH[k]]) for i in ids]), tail_q, tail) for k in chans}
    return out


# ------------------------------------------------------------------ kiem dinh dev
def decode_any(P: np.ndarray, T: dict) -> dict:
    """{kenh: co dong} khi giai ma chuoi diem P bang nguong T - dung bo giai ma san pham (khong dau ca tep)."""
    from v6.boundary import detect as D
    from v6.boundary import fusion as F
    dur = len(P) * DT
    rows = D.decode_file(P[:, :3], (T["ghep"], T["xoa"]), dur)
    hit = {"ghep": any(r["via"] == "ranh_gioi_ghep" for r in rows), "xoa": any(r["via"] != "ranh_gioi_ghep" for r in rows)}
    for k, c in (("den", 2), ("voi", 3), ("spl", 4), ("ttsr", 5)):
        if k in T and c < P.shape[1]:
            hit[k] = bool(F.decode_regions([P[:, c]], [T[k]], (F.REGION_LEVEL2[c - 2],)))
    return hit


def pseudo_files(ids: list, P: dict, dur: float, rng: np.random.Generator) -> list:
    """Ghep (trong khong gian DIEM) cac tep that da xao tron thanh chuoi dai >= dur giay; moi tep dung toi da mot lan / luot."""
    order = list(rng.permutation(ids))
    out, cur, n = [], [], 0
    need = int(dur / DT)
    for i in order:
        cur.append(P[i])
        n += len(P[i])
        if n >= need:
            out.append(np.concatenate(cur)[:need])
            cur, n = [], 0
    return out


def check(ctx: dict, cal: dict, thr: dict, shuffles: int) -> list:
    rng = np.random.default_rng(0)
    L = [f"# Kiem dinh nguong theo do dai (dev: chuoi gia ghep diem OOF tep that fold k, nguong thr_fold[k]) - {thr['name']}",
         "", f"duoi: {cal.get('tail', 'exp')} tu phan vi {cal['tail_q']}; {cal['thr_file']}, d_ref {cal['d_ref']} s, khuc {cal['block_s']} s",
         "", "| ngon ngu | do dai | n chuoi | vu oan CU | vu oan MOI | MOI theo kenh |", "|---|---|---|---|---|---|"]
    for lg in sorted({ctx["lang"][i] for i in ctx["real"]}):
        for d in DURS:
            n, old, new, per = 0, 0, 0, defaultdict(int)
            for f in range(5):
                ids = [i for i in ctx["real"] if ctx["lang"][i] == lg and ctx["fold"][i] == f]
                T = thr["thr_fold"][str(f)][lg]
                Ta = adjust(T, d, sigma_for(cal, lg, f), cal["d_ref"], cal.get("two_sided", False))
                for _ in range(shuffles):
                    for P in pseudo_files(ids, ctx["P"], d, rng):
                        ho, hn = decode_any(P, T), decode_any(P, Ta)
                        n += 1
                        old += any(ho.values())
                        new += any(hn.values())
                        for k, v in hn.items():
                            per[k] += v
            L.append(f"| {lg} | {d} s | {n} | {old / max(n, 1):.3f} | {new / max(n, 1):.3f} | "
                     + " ".join(f"{k} {v / max(n, 1):.3f}" for k, v in sorted(per.items())) + " |")
    if ctx["ami"]:
        L += ["", "## Tep that AMI dai (amilong, khong lockbox, OOF) - nguong 'all' (khong co fold rieng cho AMI)", "",
              "| tep | do dai | vu oan CU | vu oan MOI |", "|---|---|---|---|"]
        o = nw = 0
        for i in ctx["ami"]:
            P, lg = ctx["P"][i], ctx["lang"][i]
            T = thr["thr"].get(lg) or thr["thr"]["en"]
            d = len(P) * DT
            Ta = adjust(T, d, sigma_for(cal, lg, None), cal["d_ref"], cal.get("two_sided", False))
            ho, hn = any(decode_any(P, T).values()), any(decode_any(P, Ta).values())
            o += ho
            nw += hn
            L.append(f"| {i} | {d:.0f} s | {int(ho)} | {int(hn)} |")
        L.append(f"| **tong** | {len(ctx['ami'])} tep | {o}/{len(ctx['ami'])} | {nw}/{len(ctx['ami'])} |")
    return L


# ------------------------------------------------------------------ hoi quy tren tep NGAN (OOF natvi / naten goc, khong XLS-R)
def regress(thr: dict, cal: dict, bound: str = "fuse1", variants: tuple = ((None, False),)) -> list:
    """Chi so Req tren tep goc tu nhien (OOF, nguong thr_fold[k] nhu san pham dinh tuyen fold) KHONG / CO nguong theo do dai:
    do cai gia cua lencal tren tep ngan (tep 10-22 s cung bi nang nhe)."""
    from v6.boundary import detect as D
    from v6.boundary import fuse9 as F9
    from v6.boundary import fusion as F
    from v6.boundary.rowmerge import merge
    from v6.natural import req_check as RC
    ctx = F9.Ctx9(bound, bound, thr["params"]["heads"], F.NAT_SETS)

    def recs(adj, keep=lambda d: True) -> dict:
        by = defaultdict(list)
        for i in ctx.ids:
            lg, f = ctx.lang[i], ctx.fold[i]
            dur = len(ctx.PB[i]) * DT
            if not keep(dur):
                continue
            T = thr["thr_fold"][str(f)].get(lg) or next(iter(thr["thr_fold"][str(f)].values()))
            if adj:
                T = adjust(T, dur, sigma_for(cal, lg, f), adj[0] or cal["d_ref"], adj[1])
            rows = D.decode_file(ctx.PB[i], (T["ghep"], T["xoa"]), dur)
            for r in rows:
                r.update(kind="segment", nm=r["margin"] + (T["ghep"] if r["via"] == "ranh_gioi_ghep" else T["xoa"]))
            use = [(k, c) for k, c in (("den", 0), ("voi", 1), ("spl", 2), ("ttsr", 3)) if k in T and 2 + c < ctx.PR[i].shape[1]]
            rows = merge(rows, F.decode_regions([ctx.PR[i][:, 2 + c] for _, c in use], [T[k] for k, _ in use],
                                                tuple(F.REGION_LEVEL2[c] for _, c in use)) if use else [])
            for h, (k, name) in enumerate((("nen", "02_nen_toan_cuc"), ("giong", "02_giong_toan_cuc"))):
                if k in T and ctx.HP[i][h] >= T[k]:
                    rows = merge(rows, [{"level2": name, "kind": "global", "p": float(ctx.HP[i][h]), "nm": float(ctx.HP[i][h] - T[k])}])
            rows, dropped = F.cap_rows(rows)
            final = "04" if ctx.k1_04[i] else "03" if any(x["level2"].startswith("03") for x in rows) else "02" if rows else "01"
            by[ctx.items[i]["set"]].append(F.make_record(i, ctx.items[i], rows, final, dropped))
        return by

    L = ["# Hoi quy tren tep NGAN: OOF natvi / naten goc, nguong thr_fold[k]; cac bien the (d_ref, hai chieu)", "",
         "| bien the | bo | nhom do dai | n | Req 1 | Req 2 | Req 3 | Req 4 |", "|---|---|---|---|---|---|---|---|"]
    f = lambda m, k: f"{m.get(k, float('nan')):.3f}"                    # noqa: E731
    groups = (("tat ca", lambda d: True), ("<= 6 s", lambda d: d <= 6), ("6-10 s", lambda d: 6 < d <= 10), ("10-15 s", lambda d: 10 < d <= 15),
              ("> 15 s", lambda d: d > 15))
    for v in (False,) + tuple(variants):
        name_v = "CU" if v is False else f"d_ref {v[0] or cal['d_ref']} s{' hai chieu' if v[1] else ''}"
        for gname, keep in groups:
            by = recs(v, keep)
            for s_ in sorted(by):
                m = RC.metrics(by[s_])
                L.append(f"| {name_v} | {s_} | {gname} | {len(by[s_])} | {f(m, '1')} | {f(m, '2')} | {f(m, '3')} | {f(m, '4')} |")
    return L


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "check", "regress", "selftest"])
    ap.add_argument("--bound", default="fuse1")
    ap.add_argument("--thr", default="thresholds_v9F.json")
    ap.add_argument("--shuffles", type=int, default=6)
    ap.add_argument("--tail", choices=["exp", "gpd"], default="gpd")
    ap.add_argument("--tail-q", type=float, default=TAIL_Q)
    ap.add_argument("--tag", default="", help="hau to tep hieu chinh (so sanh bien the tren dev)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
        return
    out_cal = os.path.join(RES, a.thr.replace("thresholds_", "lencal_").replace(".json", a.tag + ".json"))
    ctx = load_ctx(a.bound) if a.cmd in ("fit", "check") else None
    if a.cmd == "fit":
        cal = fit(ctx, a.thr, a.tail_q, a.tail)
        json.dump(cal, open(out_cal, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        for lg, by in cal["sigma"].items():
            print(lg, {k: (v["sigma"], v["xi"], v["diag"]) for k, v in by["all"].items()})
        print("->", out_cal)
    else:
        cal = json.load(open(out_cal, encoding="utf-8"))
        thr = json.load(open(os.path.join(RES, a.thr), encoding="utf-8"))
        L = check(ctx, cal, thr, a.shuffles) if a.cmd == "check" else regress(thr, cal, a.bound, ((None, cal.get("two_sided", False)),))
        p = out_cal.replace(".json", f"_{a.cmd}.md")
        open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
        print("\n".join(L))


# ------------------------------------------------------------------ kiem thu (khong doc OOF)
def selftest() -> None:
    p = np.zeros(1000)
    p[150], p[650] = 0.9, 0.5
    bm = block_max(p, 4.0)                           # 200 khung / khuc -> 5 khuc
    assert list(bm) == [0.9, 0.0, 0.0, 0.5, 0.0], bm
    rng = np.random.default_rng(1)
    s = fit_sigma(1 / (1 + np.exp(-rng.exponential(0.7, 20000))))   # logit ~ mu(0,7) -> sigma ~ 0,7
    assert abs(s["sigma"] - 0.7) < 0.05, s
    T = {"ghep": 0.9, "den": 0.5, "nen": 0.9}
    assert adjust(T, 8.0, {"ghep": 0.7, "den": 0.7}) == T                         # tep ngan: giu nguyen
    Ta = adjust(T, 100.0, {"ghep": 0.7, "den": 0.7})
    assert abs(logit(Ta["ghep"]) - (logit(0.9) + 0.7 * math.log(10))) < 1e-9 and Ta["nen"] == 0.9    # dau ca tep khong doi
    assert Ta["den"] > 0.5 and adjust(T, 100.0, {})["den"] == 0.5
    assert adjust(T, 5.0, {"den": 0.7}, 10.0, two_sided=True)["den"] < 0.5 and adjust(T, 5.0, {"den": 0.7}, 10.0)["den"] == 0.5
    cal = {"sigma": {"vi": {"all": {"ghep": {"sigma": 0.5}}, "2": {"ghep": {"sigma": 0.6}}}}}
    sg = lambda lg, f: {k: v["sigma"] for k, v in sigma_for(cal, lg, f).items()}     # noqa: E731
    assert sg("vi", 2) == {"ghep": 0.6} and sg("vi", None) == {"ghep": 0.5} and sg("en", 7) == {"ghep": 0.5}
    assert abs(shift_logit(2.0, 10.0, {"sigma": 0.7, "xi": 0.0, "u": 1.0}) - (2.0 + 0.7 * math.log(10))) < 1e-9
    g = shift_logit(2.0, 10.0, {"sigma": 0.7, "xi": 0.3, "u": 1.0})
    assert g > 2.0 + 0.7 * math.log(10)                                        # duoi nang hon -> dich manh hon
    assert abs(shift_logit(2.0, 1.0, {"sigma": 0.7, "xi": 0.3, "u": 1.0}) - 2.0) < 1e-9     # r = 1: giu nguyen
    sg = fit_sigma(1 / (1 + np.exp(-rng.exponential(0.7, 20000))), 0.8, "gpd")
    assert sg["xi"] < 0.1 and abs(sg["sigma"] - 0.7) < 0.08, sg                 # du lieu mu -> gpd ~ mu
    print("lencal selftest dat")


if __name__ == "__main__":
    main()
