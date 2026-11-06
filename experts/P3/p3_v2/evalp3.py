# -*- coding: utf-8 -*-
"""Bo cham nhanh cua P3 (cung thuoc voi v6.product.ghep_thu, khong ghi pkl tam): nap Ctx9 nen (fuse1 + h2) MOT lan, thay cot den / voi va
dau nen / giong trong bo nho, roi tinh lai nguong + giai ma dung Ctx9.records (phan vi tep THAT cung ngon ngu, fold KHAC).

Ra theo bo (v6natvi / v6naten): Req 1-5, 10, 11 (req_check.metrics), % dung theo kieu (nhu ghep_thu), TPR DUNG LOAI theo thao tac 02
(tep co dong dung level2 va khong bi 03), vu oan tep that theo kenh, AUC dau ca tep tren tep that vs tep sua cung loai.

CLI:  python experts/P3/p3_v2/evalp3.py [--oof experts/P3/x/oof.pkl] [--heads experts/P3/x/heads.pkl] [--name ten]
      python experts/P3/p3_v2/evalp3.py selftest
Dung trong Python:  E = P3Eval(); res = E.run(den=..., voi=..., nen=..., giong=...); print(E.table(res))
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

FREG, FH, F03 = 0.005, 0.0025, 0.005
P3_KINDS = ("nen_toan_cuc", "nen_cuc_bo", "nen_cuc_bo_ky", "giong_toan_cuc", "doi_giong_ai", "giong_cuc_bo", "giong_cuc_bo_ky",
            "da_ghep_nen", "da_ghep_giong", "nguyen_ban", "nguyen_ban_vsasv")


def auc(pos, neg) -> float:
    """AUC Mann-Whitney (ham thuan)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.r_[pos, neg])
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kind_label(it: dict) -> str:
    """Kieu bao cao: tep that mang ten kieu nen_* (WebRTC NS <= 2, nay 01) tach thanh '<kieu>@01' (ham thuan)."""
    k = it["kind"]
    return f"{k}@01" if it["real"] and k.startswith("nen_") else k


class P3Eval:
    def __init__(self, sets: tuple | None = None):
        from v6.boundary import fuse9 as F9
        from v6.boundary import fusion as F
        self.F9, self.F = F9, F
        self.ctx = F9.Ctx9("fuse1", "fuse1", "h2", sets or F.NAT_SETS)
        c = self.ctx
        self.PR0 = c.PR                                    # cung doi tuong voi PB (fuse1)
        self.HP0 = dict(c.HP)
        self.ids = c.ids
        self.kind = {i: kind_label(c.items[i]) for i in self.ids}

    # ---------- thay diem ----------
    def _install(self, den=None, voi=None, nen=None, giong=None, mode: str = "replace") -> None:
        c = self.ctx
        if den is None and voi is None:
            c.PR = self.PR0
        else:
            PR = {}
            for i in self.ids:
                Q = self.PR0[i].copy()
                for col, src in ((2, den), (3, voi)):
                    if src is None or i not in src:
                        continue
                    v = np.asarray(src[i], np.float32).ravel()
                    n = min(len(v), len(Q))
                    Q[:n, col] = v[:n] if mode == "replace" else 1 / (1 + np.exp(-(_lg(Q[:n, col]) + _lg(v[:n])) / 2))
                PR[i] = Q
            c.PR = PR
        HP = {}
        for i in self.ids:
            h = np.array(self.HP0[i], np.float32).copy()
            if nen is not None and i in nen:
                h[0] = float(np.ravel(nen[i])[0])
            if giong is not None and i in giong:
                h[1] = float(np.ravel(giong[i])[0])
            HP[i] = h
        c.HP = HP
        c.mr = {i: c.PR[i][:, 2:].max(0) for i in c.real}
        c._th, c._rr = {k: v for k, v in c._th.items() if k[0][0] == "b"}, {}

    def run(self, den=None, voi=None, nen=None, giong=None, mode: str = "replace", freg: float = FREG, fh: float = FH, f03: float = F03) -> dict:
        from v6.natural import req_check as RC
        self._install(den, voi, nen, giong, mode)
        recs = self.ctx.records(f03, freg, fh, 0.0)
        out = {"sets": {}, "kind": {}, "typed": {}, "fa": {}, "auc": {}, "recs": recs}
        by = defaultdict(list)
        for r in recs:
            by[r["set"]].append(r)
        for s, rs in by.items():
            m = RC.metrics(rs)
            out["sets"][s] = {k: float(m[k]) for k in ("1", "2", "3", "4", "5", "10", "11") if k in m}
            kk, ty, fa, nr = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter(), 0
            for r in rs:
                if r["final"] == "04":
                    continue
                k = self.kind[r["id"]]
                kk[k][0] += r["correct"]
                kk[k][1] += 1
                lv = {x["level2"] + ("/toan" if x.get("kind") == "global" else "") for x in r["rows"]}
                if r["expected"] == "01":
                    nr += 1
                    fa.update(lv)
                for e in r["_m"].get("edits", []):
                    if e["level2"].startswith("02"):
                        key = e["level2"] + (":" + k if k in ("doi_giong_ai", "giong_cuc_bo_ky", "da_ghep_nen", "da_ghep_giong") else "")
                        want = e["level2"] + ("/toan" if e.get("kind") == "global" else "")
                        ty[key][0] += int(want in lv and (r["final"] != "03" or r["expected"] == "03"))
                        ty[key][1] += 1
            out["kind"].update({(s, k): tuple(v) for k, v in kk.items()})
            out["typed"].update({(s, k): tuple(v) for k, v in ty.items()})
            out["fa"][s] = {k: v / max(nr, 1) for k, v in fa.items()}
            lg = "vi" if s == "v6natvi" else "en"
            real = [i for i in self.ctx.real if self.ctx.lang[i] == lg]
            for h, nm, kinds in ((0, "nen", ("nen_toan_cuc",)), (1, "giong", ("giong_toan_cuc", "doi_giong_ai"))):
                pos = [self.ctx.HP[r["id"]][h] for r in rs if r["kind"] in kinds and not self.ctx.items[r["id"]]["real"]]
                out["auc"][(s, nm)] = auc(pos, [self.ctx.HP[i][h] for i in real])
        return out

    # ---------- bao cao ----------
    @staticmethod
    def table(res: dict, base: dict | None = None, kinds: tuple = P3_KINDS) -> str:
        L = ["| bo | Req1 | Req2 | Req3 | Req4 | Req5 | Req10 | Req11 |", "|---|---|---|---|---|---|---|---|"]
        for s in sorted(res["sets"]):
            for lab, rr in (("nen", base), ("MOI", res)) if base else (("", res),):
                m = rr["sets"][s]
                L.append(f"| {s} {lab} | " + " | ".join(f"{m.get(k, float('nan')):.3f}" for k in ("1", "2", "3", "4", "5", "10", "11")) + " |")
        L += ["", "| bo | kieu | n | dung" + (" nen" if base else "") + " |", "|---|---|---|---|"]
        for (s, k) in sorted(res["kind"]):
            if not any(k.startswith(x) for x in kinds):
                continue
            a, n = res["kind"][(s, k)]
            b = f" {base['kind'][(s, k)][0] / max(base['kind'][(s, k)][1], 1):.2f} ->" if base and (s, k) in base["kind"] else ""
            L.append(f"| {s} | {k} | {n} |{b} {a / max(n, 1):.2f} |")
        L += ["", "| bo | thao tac (dung loai) | n | TPR |", "|---|---|---|---|"]
        for (s, k) in sorted(res["typed"]):
            a, n = res["typed"][(s, k)]
            b = f" {base['typed'][(s, k)][0] / max(base['typed'][(s, k)][1], 1):.2f} ->" if base and (s, k) in base["typed"] else ""
            L.append(f"| {s} | {k} | {n} |{b} {a / max(n, 1):.2f} |")
        L += ["", "| bo | kenh | vu oan tep that |", "|---|---|---|"]
        for s in sorted(res["fa"]):
            for k, v in sorted(res["fa"][s].items()):
                if k.startswith("02"):
                    L.append(f"| {s} | {k} | {v:.4f} |")
        L += ["", "AUC dau: " + " · ".join(f"{s}:{h} {v:.3f}" for (s, h), v in sorted(res["auc"].items()))]
        return "\n".join(L)

    @staticmethod
    def summary(res: dict) -> str:
        """Mot dong: dung theo kieu P3 + Req 4 / 5 hai bo."""
        g = lambda s, k: res["kind"].get((s, k), (0, 1))                      # noqa: E731
        parts = []
        for s, t in (("v6natvi", "vi"), ("v6naten", "en")):
            m = res["sets"][s]
            ks = " ".join(f"{k[:9]}={g(s, k)[0] / max(g(s, k)[1], 1):.2f}" for k in ("nen_toan_cuc", "nen_cuc_bo", "nen_cuc_bo_ky@01", "giong_toan_cuc",
                                                                                 "doi_giong_ai", "giong_cuc_bo", "giong_cuc_bo_ky") if (s, k) in res["kind"])
            parts.append(f"{t}: R1 {m['1']:.3f} R4 {m['4']:.3f} R5 {m.get('5', float('nan')):.3f} | {ks}")
        return "\n".join(parts)


def ci_typed(res: dict, n_boot: int = 400, seed: int = 0) -> dict:
    """KTC 95 % (bootstrap theo cum nguoi noi) cho TPR dung loai cua tung thao tac 02 va vu oan tep that theo kenh.
    Tra {(bo, khoa): (gia tri, thap, cao)} (ham thuan tren res["recs"])."""
    rng = np.random.default_rng(seed)
    rows = defaultdict(list)                                  # (bo, khoa) -> [(cum, trung, n)]
    for r in res["recs"]:
        if r["final"] == "04":
            continue
        lv = {x["level2"] + ("/toan" if x.get("kind") == "global" else "") for x in r["rows"]}
        g = r["_g"] or r["id"]
        if r["expected"] == "01":
            for k in ("02_nen_cuc_bo", "02_giong_cuc_bo", "02_nen_toan_cuc/toan", "02_giong_toan_cuc/toan"):
                rows[(r["set"], "FA:" + k)].append((g, float(k in lv), 1.0))
        for e in r["_m"].get("edits", []):
            if e["level2"].startswith("02"):
                want = e["level2"] + ("/toan" if e.get("kind") == "global" else "")
                hit = want in lv and (r["final"] != "03" or r["expected"] == "03")
                rows[(r["set"], e["level2"])].append((g, float(hit), 1.0))
    out = {}
    for key, v in rows.items():
        gs = sorted({x[0] for x in v})
        gi = {g: n for n, g in enumerate(gs)}
        num, den = np.zeros(len(gs)), np.zeros(len(gs))
        for g, h, n in v:
            num[gi[g]] += h
            den[gi[g]] += n
        b = rng.integers(0, len(gs), (n_boot, len(gs)))
        bs = num[b].sum(1) / np.maximum(den[b].sum(1), 1)
        out[key] = (float(num.sum() / max(den.sum(), 1)), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975)))
    return out


def _lg(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def load_frames(path: str) -> tuple:
    d = pickle.load(open(path, "rb"))
    den = {i: np.asarray(v)[:, 0] for i, v in d.items()}
    voi = {i: np.asarray(v)[:, 1] for i, v in d.items() if np.asarray(v).shape[1] > 1}
    return den, voi


def load_heads(path: str) -> tuple:
    d = pickle.load(open(path, "rb"))
    if "ids" in d:
        d = {i: np.asarray(d["P"])[n] for n, i in enumerate(d["ids"])}
    return {i: np.ravel(v)[0] for i, v in d.items()}, {i: np.ravel(v)[1] for i, v in d.items() if np.size(v) > 1}


def selftest() -> None:
    assert abs(auc([1, 2, 3], [0, 0.5]) - 1.0) < 1e-9 and abs(auc([0], [1]) - 0.0) < 1e-9 and abs(auc([1, 0], [1, 0]) - 0.5) < 1e-9
    assert kind_label({"kind": "nen_cuc_bo_ky", "real": True}) == "nen_cuc_bo_ky@01"
    assert kind_label({"kind": "nen_toan_cuc", "real": False}) == "nen_toan_cuc"
    assert np.allclose(_lg(np.array([0.5])), 0.0)
    print("evalp3 selftest dat")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof")
    ap.add_argument("--heads")
    ap.add_argument("--name", default="thu")
    a = ap.parse_args()
    E = P3Eval()
    base = E.run()
    den = voi = nen = giong = None
    if a.oof:
        den, voi = load_frames(a.oof)
    if a.heads:
        nen, giong = load_heads(a.heads)
    res = E.run(den, voi, nen, giong)
    txt = f"# evalp3 `{a.name}`\n\n" + E.table(res, base)
    p = os.path.join(BASE, ".tmp_v6", "results", f"evalp3_{a.name}.md")
    open(p, "w", encoding="utf-8").write(txt + "\n")
    print(E.summary(base))
    print(E.summary(res))
    print("->", p)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
