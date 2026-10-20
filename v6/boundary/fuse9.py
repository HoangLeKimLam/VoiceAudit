# -*- coding: utf-8 -*-
"""v9 - SAN PHAM NHANH: hop nhat cac chuyen gia chi dung XLS-R dong bang (khung L12 + vector gop tho), KHONG dung dac trung v5 / v6b (FileFeat RTF 5,5 > Req 15):
  03 moi noi (ghep / chen / xoa)   TCN nhan ranh gioi            (boundary_oof_<bound>.pkl, cot ghep / xoa)
  02 mot doan (khu on / doi giong) TCN nhan VUNG                 (boundary_oof_<reg>.pkl,   cot vung khu on / vung doi giong)
  02 ca file (khu on / doi giong), 03 giong may ca file          dau HGB + LR tren vector gop tho XLS-R (heads2_<heads>.pkl)
  nhan tep = 04 (K1) > 03 > 02 > 01 ; <= 2 dong doan moi khuc 14 s (cap_rows)
Moi chuyen gia co NGUONG RIENG tai ti le vu oan tren tep THAT tu nhien (goc) cua cac fold KHAC.
  python -m v6.boundary.fuse9 sweep [--bound tcn --reg tcn_reg --heads h3]      -> .tmp_v6/results/fuse9_<heads>[_<bound>[_<reg>]].md / .json (hau to khi khac tcn / tcn_reg)
  python -m v6.boundary.fuse9 selftest                                          (Ctx9 gia: thr_fold = nguong nghien cuu, params truy vet)
  python -m v6.boundary.fuse9 emit --name A --f03 0.02 --freg 0.01 --fh 0.01 --ftts 0.01  -> engine_v9A.json (+ _extra.json) cho moi tep co khung
  python -m v6.boundary.fuse9 perfold --names F FM                             -> oof_perfold_v9F.md (chi so OOF theo fold nguoi noi cua tep nguong da co)
  python -m v6.boundary.fuse9 thr --bound fuse1 --reg fuse1 --heads h2 --name F --f03 0.01 --freg 0.03 --fh 0.005 --ftts 0   -> thresholds_v9F.json
      (params day du + sha1 OOF / checkpoint + thr_fold[k] cho Product fold=k + chi so OOF tai diem van hanh; --ftts 0.01 --typed: bat kenh giong may)"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

from v6.boundary import detect as D  # noqa: E402
from v6.boundary import fusion as F  # noqa: E402
from v6.boundary.rowmerge import merge, snap_merge, typed_merge  # noqa: E402
from v6.natural import req_check as RC  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
F03 = (0.005, 0.01, 0.02, 0.03, 0.05)
FREG = (0.01, 0.03, 0.06)
FH = (0.005, 0.01, 0.02)
FTTS = (0.0, 0.01)
FSPL = (0.0, 0.01, 0.03)       # vung doan ghep loi that (chi khi TCN --regions2)
SNAP = ((False, False), (True, False), (True, True))   # (snap, typed): snap = dong ranh gioi chong len vung cung loai lay ranh gioi cua VUNG (dinh vi chinh xac hon);
                                                       # typed = them: dong ghep chong len vung GIONG MAY / doan ghep that lay LOAI cua vung (Req 6, 8a)


class Ctx9:
    def __init__(self, bound: str, reg: str, heads: str, sets: tuple, include_all_cfg: bool = False, per_lang: bool = True, drop_real_src: tuple = ()):
        self.items = D.load_items()
        self.tags = (bound, reg, heads)
        ob = pickle.load(open(D.tagged(D.OOFP, bound), "rb"))
        orr = ob if reg == bound else pickle.load(open(D.tagged(D.OOFP, reg), "rb"))      # fuse1: bien = vung = cung tep OOF -> nap mot lan
        h = pickle.load(open(os.path.join(CACHE, f"heads2_{heads}.pkl"), "rb"))
        hid = {i: n for n, i in enumerate(h["ids"])}
        self.ids = [i for i, it in self.items.items() if it["set"] in sets and (include_all_cfg or it["cfg"] == "goc") and i in ob and i in orr and i in hid]
        self.fold = {i: self.items[i]["fold"] for i in self.ids}
        # ngon ngu cua tep (K1x nhan dang vi / en ~ 100 % / 83 %): nguong dat RIENG tung ngon ngu vi phan bo diem tep that khac nhau (kenh, ban ghi)
        self.lang = {i: (str(self.items[i]["meta"].get("lang") or ("en" if self.items[i]["set"] == "v6naten" else "vi"))[:2] if per_lang else "all") for i in self.ids}
        self.PB = {i: ob[i].astype(np.float32) for i in self.ids}
        self.PR = self.PB if orr is ob else {i: orr[i].astype(np.float32) for i in self.ids}
        self.HP = {i: h["P"][hid[i]] for i in self.ids}
        self.drop = {i for i in self.ids if self.items[i]["real"] and str(self.items[i]["meta"].get("source")) in drop_real_src}     # tep that cua nguon nghi da duoc dung (video YouTube)
        self.real = [i for i in self.ids if self.items[i]["real"] and self.items[i]["set"] in F.NAT_SETS and self.items[i]["cfg"] == "goc" and i not in self.drop]
        self.k1_04 = {i: self.items[i]["meta"].get("k1", {}).get("decision") in ("04", "tu_choi") for i in self.ids}
        self._th, self._r3, self._rr = {}, {}, {}
        self.mb = {i: self.PB[i][:, :2].max(0) for i in self.real}
        self.nreg = next(iter(self.PR.values())).shape[1] - 2
        self.mr = {i: self.PR[i][:, 2:].max(0) for i in self.real}

    def th(self, key: str, vals: dict, q: float) -> dict:
        """{(ngon ngu, fold): nguong} = phan vi 1 - q cua diem tep THAT cung ngon ngu o cac fold KHAC."""
        k = (key, q)
        if k not in self._th:
            out = {}
            for lg in set(self.lang[i] for i in self.real):
                for f in range(5):
                    v = [vals[i] for i in self.real if self.fold[i] != f and self.lang[i] == lg]
                    out[(lg, f)] = float(np.quantile(v, 1 - q))
            self._th[k] = out
        return self._th[k]

    def t(self, th: dict, i: str) -> float:
        return th.get((self.lang[i], self.fold[i]), th[(next(iter({k[0] for k in th})), self.fold[i])])

    def rows03(self, q: float) -> dict:
        if q not in self._r3:
            t = [self.th(f"b{c}", {i: self.mb[i][c] for i in self.real}, q) for c in (0, 1)]
            out = {}
            for i in self.ids:
                rows = D.decode_file(self.PB[i], (self.t(t[0], i), self.t(t[1], i)), len(self.PB[i]) * D.DT)
                for r in rows:
                    c = 0 if r["via"] == "ranh_gioi_ghep" else 1
                    r.update(kind="segment", nm=r["margin"] + self.t(t[c], i))
                out[i] = rows
            self._r3[q] = out
        return self._r3[q]

    def rowsreg(self, qs: tuple) -> dict:
        """Dong tu cac kenh VUNG cua TCN; qs = (vu oan khu on / doi giong, vu oan vung doan ghep, vu oan vung giong may); 0 = tat."""
        if qs not in self._rr:
            ths = []
            for c in range(self.nreg):
                q = qs[0] if c < 2 else qs[1] if c == 2 else qs[2]
                ths.append(self.th(f"r{c}", {i: self.mr[i][c] for i in self.real}, q) if q > 0 else None)
            out = {}
            for i in self.ids:
                use = [c for c in range(self.nreg) if ths[c] is not None]
                out[i] = F.decode_regions([self.PR[i][:, 2 + c] for c in use], [self.t(ths[c], i) for c in use], tuple(F.REGION_LEVEL2[c] for c in use))
            self._rr[qs] = out
        return self._rr[qs]

    def export_thresholds(self, f03: float, freg: float, fh: float, ftts: float, fspl: float = 0.0, exclude_fold: int | None = None) -> dict:
        """Nguong cho san pham (v6/product): phan vi 1 - q cua diem tep THAT tu nhien cung ngon ngu tren TOAN BO fold (mo hinh cuoi hoc tren toan bo du lieu).
        exclude_fold = k: chi tep that cua cac fold KHAC k (= nguong nghien cuu Ctx9.th cho fold k) -> san pham che do dinh tuyen fold (danh gia ngoai mau)."""
        out = {}
        real = [i for i in self.real if exclude_fold is None or self.fold[i] != exclude_fold]
        for lg in sorted({self.lang[i] for i in real}):
            rl = [i for i in real if self.lang[i] == lg]
            q = lambda vals, qq: float(np.quantile(vals, 1 - qq))          # noqa: E731
            d = {"ghep": q([self.mb[i][0] for i in rl], f03), "xoa": q([self.mb[i][1] for i in rl], f03)}
            if freg:
                d["den"], d["voi"] = q([self.mr[i][0] for i in rl], freg), q([self.mr[i][1] for i in rl], freg)
            if fspl and self.nreg > 2:
                d["spl"] = q([self.mr[i][2] for i in rl], fspl)
            if ftts and self.nreg > 3:
                d["ttsr"] = q([self.mr[i][3] for i in rl], ftts)
            if fh:
                d["nen"], d["giong"] = q([self.HP[i][0] for i in rl], fh), q([self.HP[i][1] for i in rl], fh)
            if ftts:
                d["ttsf"] = q([self.HP[i][2] for i in rl], ftts)
            out[lg] = d
        return out

    def export_refs(self) -> dict:
        """Bang phan vi (0..100) cua diem tung chuyen gia tren tep THAT tu nhien cung ngon ngu: dung de ghi 'phan vi so voi tep that' trong ho so."""
        qs = np.arange(0, 101)
        out = {}
        for lg in sorted({self.lang[i] for i in self.real}):
            rl = [i for i in self.real if self.lang[i] == lg]
            d = {"ghep": [self.mb[i][0] for i in rl], "xoa": [self.mb[i][1] for i in rl]}
            for k, c in (("den", 0), ("voi", 1), ("spl", 2), ("ttsr", 3)):
                if c < self.nreg:
                    d[k] = [self.mr[i][c] for i in rl]
            d["nen"], d["giong"], d["ttsf"] = ([self.HP[i][h] for i in rl] for h in (0, 1, 2))
            out[lg] = {k: [round(float(x), 5) for x in np.percentile(v, qs)] for k, v in d.items()}
        return out

    def records(self, f03: float, freg: float, fh: float, ftts: float, ids: list | None = None, fspl: float = 0.0, snap: bool = False, typed: bool = False) -> list:
        r3, rr = self.rows03(f03), self.rowsreg((freg, fspl, ftts))
        heads = [(h, name, self.th(f"h{h}", {j: self.HP[j][h] for j in self.real}, q))
                 for h, (name, q) in enumerate((("02_nen_toan_cuc", fh), ("02_giong_toan_cuc", fh), ("03_chen_giong_may", ftts))) if q > 0]
        out = []
        for i in ids or self.ids:
            rows = typed_merge(list(r3[i]), rr[i]) if typed else snap_merge(list(r3[i]), rr[i]) if snap else merge(list(r3[i]), rr[i])
            for h, name, t in heads:
                tv = self.t(t, i)
                if self.HP[i][h] >= tv:
                    rows = merge(rows, [{"level2": name, "kind": "global", "p": float(self.HP[i][h]), "nm": float(self.HP[i][h] - tv)}])
            rows, dropped = F.cap_rows(rows)
            final = "04" if self.k1_04[i] else "03" if any(x["level2"].startswith("03") for x in rows) else "02" if rows else "01"
            out.append(F.make_record(i, self.items[i], rows, final, dropped))
        return out


def sweep_suffix(bound: str, reg: str) -> str:
    """Hau to ten tep sweep: tcn / tcn_reg (mac dinh cu) -> '' (fuse9_<heads>.md nhu truoc); khac -> _<bound>[_<reg>] de KHONG ghi de ket qua cua mo hinh khac
    (fuse9_h2.md ngay 04/10 la cua fuse1 / fuse1, ghi de ban tcn - ban tcn chep tay thanh fuse9_h2_tcn.md)."""
    if (bound, reg) == ("tcn", "tcn_reg"):
        return ""
    return f"_{bound}" + ("" if reg == bound else f"_{reg}")


def sweep(bound: str, reg: str, heads: str, drop_src: tuple = ()) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = Ctx9(bound, reg, heads, F.NAT_SETS, drop_real_src=drop_src)
    sfx = sweep_suffix(bound, reg) + ("_novss" if drop_src else "")
    results = []
    for f03, freg, fh, ftts, fspl, (snap, typed) in itertools.product(F03, FREG, FH, FTTS if ctx.nreg > 3 else (0.0,), FSPL if ctx.nreg > 2 else (0.0,),
                                                                       SNAP if ctx.nreg > 2 else ((False, False),)):
        if typed and not ftts:
            continue                                       # phan loai giong may can kenh giong may
        recs = [r for r in ctx.records(f03, freg, fh, ftts, fspl=fspl, snap=snap, typed=typed) if r["id"] not in ctx.drop]
        by = {s: [r for r in recs if r["set"] == s] for s in F.NAT_SETS}
        results.append({"f03": f03, "freg": freg, "fh": fh, "ftts": ftts, "fspl": fspl, "snap": snap, "typed": typed, **{s: {k: float(v) for k, v in RC.metrics(by[s]).items() if np.isscalar(v)} for s in by}})
    json.dump(results, open(os.path.join(D.RES, f"fuse9_{heads}{sfx}.json"), "w"), indent=1)
    L = [f"# San pham nhanh v9 (bien {bound}, vung {reg}, dau ca file {heads}) - duong bien tren tep goc tu nhien\n"]
    for s, name in (("v6natvi", "tieng Viet"), ("v6naten", "tieng Anh")):
        L += [f"\n## {name}\n", "| vu oan <= | f03 | freg | fh | ftts | fspl / snap / typed | macro-F1 | khong bo sot | dang tin | vu oan | 6 | 7a | 7b | 8a | 8b | 04 | moc 500 ms | IoU |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for cap in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
            ok = [r for r in results if r[s]["4"] <= cap and r[s]["9"] <= 0.08]
            if ok:
                b = max(ok, key=lambda r: r[s]["1"])
                m = b[s]
                nan = float('nan')
                L.append(f"| {cap:.0%} | {b['f03']} | {b['freg']} | {b['fh']} | {b['ftts']} | {b['fspl']}/{int(b['snap'])}/{int(b['typed'])} | {m['1']:.2f} | {m['2']:.2f} | {m['3']:.2f} | {m['4']:.3f} | "
                         f"{m.get('6', nan):.2f} | {m['7a']:.2f} | {m['7b']:.2f} | {m.get('8a', nan):.2f} | {m.get('8b', nan):.2f} | {m['9']:.2f} | {m['10']:.2f} | {m['11']:.2f} |")
    if ctx.nreg > 3:
        nan = float("nan")
        for s, name in (("v6natvi", "tieng Viet"), ("v6naten", "tieng Anh")):
            L += ["", f"### Giong may - {name} (chon theo 8a, voi Req 6 <= 5 %)", "",
                  "| vu oan <= | ftts | fspl / snap / typed | 6 | 8a (R / P) | 8b | macro-F1 | khong bo sot |", "|---|---|---|---|---|---|---|---|"]
            for cap in (0.05, 0.08, 0.10, 0.15):
                ok = [r for r in results if r[s]["4"] <= cap and r[s]["9"] <= 0.08 and r[s].get("6", 0.0) <= 0.05 and np.isfinite(r[s].get("8a", nan))]
                if ok:
                    r = max(ok, key=lambda r: (r[s]["8a"], r[s]["1"]))
                    m = r[s]
                    L.append(f"| {cap:.0%} | {r['ftts']} | {r['fspl']}/{int(r['snap'])}/{int(r['typed'])} | {m.get('6', nan):.3f} | {m['8a']:.2f} ({m.get('8a_R', nan):.2f} / {m.get('8a_P', nan):.2f}) | "
                             f"{m.get('8b', nan):.2f} | {m['1']:.2f} | {m['2']:.2f} |")
    open(os.path.join(D.RES, f"fuse9_{heads}{sfx}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def sha1_of(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    from v6.product.pipeline import sha1_file
    return sha1_file(path)


def trace_params(bound: str, reg: str, heads: str) -> dict:
    """Truy vet: sha1 cac tep OOF da dung de dat nguong + sha1 MOI checkpoint san pham se nap (mo hinh cuoi va 5 fold) -> Product tu choi neu khac."""
    from v6.product import pipeline as PL
    oof = {"bound": D.tagged(D.OOFP, bound), "reg": D.tagged(D.OOFP, reg), "heads": os.path.join(CACHE, f"heads2_{heads}.pkl")}
    ck, missing = {}, []
    for fold in (None, 0, 1, 2, 3, 4):
        for _, p in PL.expected_ckpts(bound, reg, heads, fold):
            if os.path.exists(p):
                ck[os.path.basename(p)] = sha1_of(p)
            elif os.path.basename(p) not in missing:
                missing.append(os.path.basename(p))
    return {"oof": {k: os.path.basename(p) for k, p in oof.items()}, "oof_sha1": {k: sha1_of(p) for k, p in oof.items()}, "ckpt_sha1": ck, "ckpt_missing": missing,
            "folds": "detect.person_fold (crc32(person) % 5)"}


def _num(v) -> float | None:
    try:
        return round(float(v), 4) if np.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None


PF_KEYS = ("1", "2", "3", "4", "6", "8a", "8b", "9")


def metrics_by_fold(ctx: Ctx9, recs: list) -> dict:
    """{bo: {fold: {n, n_real, Req ...}}} - chi so OOF theo FOLD NGUOI NOI (review 11/10: fold 0 chua ~56 % natvi goc -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)."""
    out = {}
    for s in F.NAT_SETS:
        out[s] = {}
        for k in range(5):
            rr = [r for r in recs if r["set"] == s and ctx.fold[r["id"]] == k]
            try:
                m = RC.metrics(rr) if rr else {}
            except (ValueError, ZeroDivisionError, KeyError):                # fold thieu kieu tep nao do
                m = {}
            out[s][str(k)] = {"n": len(rr), "n_real": sum(r["expected"] == "01" for r in rr), **{kk: _num(m.get(kk)) for kk in PF_KEYS if kk in m}}
    return out


def perfold_md(name: str, mf: dict) -> list:
    L = [f"\n## {name}\n", "| bo | fold | n | tep that | 1 macro-F1 | 2 khong bo sot | 3 dang tin | 4 vu oan | 6 | 8a | 8b | 9 (04) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s, d in mf.items():
        for k, m in d.items():
            L.append(f"| {s} | {k} | {m['n']} | {m['n_real']} | " + " | ".join("-" if m.get(kk) is None else f"{m[kk]:.3f}" for kk in PF_KEYS) + " |")
    return L


def perfold(names: list) -> None:
    """Chi so OOF theo fold nguoi noi cho cac tep nguong DA CO (khong viet lai tep nguong): python -m v6.boundary.fuse9 perfold --names F FM
    -> .tmp_v6/results/oof_perfold_v9<ten dau>.md (cac tep cung bound / reg / heads dung chung mot Ctx9)."""
    sys.stdout.reconfigure(encoding="utf-8")
    L, ctxs = ["# Chi so OOF nghien cuu theo fold nguoi noi (detect.person_fold; mo hinh fold + nguong tep that cac fold khac; tep goc natvi / naten)"], {}
    for name in names:
        j = json.load(open(os.path.join(D.RES, f"thresholds_v9{name}.json"), encoding="utf-8"))
        p = j["params"]
        tags = (p.get("bound", "tcn"), p.get("reg", "tcn_reg"), p.get("heads", "h2"))
        if tags not in ctxs:
            ctxs[tags] = Ctx9(*tags, F.NAT_SETS)
        ctx = ctxs[tags]
        recs = ctx.records(p["f03"], p["freg"], p["fh"], p["ftts"], fspl=p.get("fspl", 0.0), snap=p.get("snap", False), typed=p.get("typed", False))
        L += perfold_md(f"{name} ({'/'.join(tags)}; f03 {p['f03']} freg {p['freg']} fh {p['fh']} ftts {p['ftts']}{' typed' if p.get('typed') else ''})", metrics_by_fold(ctx, recs))
    dst = os.path.join(D.RES, f"oof_perfold_v9{names[0]}.md")
    open(dst, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def thr(bound: str, reg: str, heads: str, name: str, f03: float, freg: float, fh: float, ftts: float, fspl: float, snap: bool = False, typed: bool = False,
        with_metrics: bool = True, ctx: Ctx9 | None = None) -> dict:
    """Tep nguong san pham thresholds_v9<name>.json: thr (tep that moi fold, cho mo hinh cuoi) + thr_fold[k] (tep that cac fold khac k, cho Product fold=k) + params day du
    (bound, reg, heads, diem van hanh, snap / typed) + truy vet sha1 (OOF, checkpoint) + chi so OOF nghien cuu tai CUNG diem van hanh (natvi / naten goc).
    ctx: Ctx9 da nap (cung bound / reg / heads, bo F.NAT_SETS) de dat nhieu diem van hanh ma chi doc OOF mot lan."""
    import datetime
    sys.stdout.reconfigure(encoding="utf-8")
    if typed and not ftts:
        raise ValueError("typed can kenh giong may (ftts > 0)")
    if ctx is None:
        ctx = Ctx9(bound, reg, heads, F.NAT_SETS)
    elif getattr(ctx, "tags", (bound, reg, heads)) != (bound, reg, heads):
        raise ValueError(f"ctx nap cho {ctx.tags}, khong phai {(bound, reg, heads)}")
    t = ctx.export_thresholds(f03, freg, fh, ftts, fspl)
    tf = {str(k): ctx.export_thresholds(f03, freg, fh, ftts, fspl, exclude_fold=k) for k in range(5)}
    params = {"bound": bound, "reg": reg, "heads": heads, "f03": f03, "freg": freg, "fh": fh, "ftts": ftts, "fspl": fspl, "snap": snap, "typed": typed,
              "nreg": ctx.nreg, **trace_params(bound, reg, heads)}
    out = {"name": name, "made": datetime.datetime.now().isoformat(timespec="seconds"), "params": params, "thr": t, "thr_fold": tf, "refs": ctx.export_refs()}
    if with_metrics:
        recs = ctx.records(f03, freg, fh, ftts, fspl=fspl, snap=snap, typed=typed)
        out["oof_metrics"] = {s: {k: _num(v) for k, v in RC.metrics([r for r in recs if r["set"] == s]).items() if np.isscalar(v)} for s in F.NAT_SETS}
        out["oof_n"] = {s: sum(r["set"] == s for r in recs) for s in F.NAT_SETS}
        out["oof_metrics_fold"] = metrics_by_fold(ctx, recs)
        out["oof_metrics_note"] = "nghien cuu: mo hinh fold + nguong cac fold khac, tep goc tu nhien (ke ca nguon vss / edacc), khong khoang tin cay"
    dst = os.path.join(D.RES, f"thresholds_v9{name}.json")
    with open(dst + ".tmp", "w", encoding="utf-8") as fh_:
        json.dump(out, fh_, indent=1, ensure_ascii=False)
    os.replace(dst + ".tmp", dst)
    print(json.dumps({"thr": t, "params": {k: v for k, v in params.items() if k != "ckpt_sha1"}, "oof_metrics": out.get("oof_metrics")}, indent=1, ensure_ascii=False), flush=True)
    return out


def emit(bound: str, reg: str, heads: str, name: str, f03: float, freg: float, fh: float, ftts: float, fspl: float = 0.0, snap: bool = False, typed: bool = False) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ctx = Ctx9(bound, reg, heads, ("v6natvi", "v6naten", "v6cover", "v6cover2", "v6benign", "v6chain", "v6ttsvi", "v6ttsen", "v6amilong", "v4"), True)
    recs = ctx.records(f03, freg, fh, ftts, fspl=fspl, snap=snap, typed=typed)
    by = {}
    for r in recs:
        by.setdefault(r["set"], []).append({k: v for k, v in r.items() if k != "_m"})
    for part, keep in (("", F.NAT_SETS), ("_extra", None)):
        d = {s: v for s, v in by.items() if (s in keep if keep else s not in F.NAT_SETS)}
        json.dump({"tag": f"v9{name}", "params": {"bound": bound, "reg": reg, "heads": heads, "f03": f03, "freg": freg, "fh": fh, "ftts": ftts, "fspl": fspl, "snap": snap, "typed": typed}, "records": d},
                  open(os.path.join(D.RES, f"engine_v9{name}{part}.json"), "w", encoding="utf-8"), ensure_ascii=False, default=float)
    print(f"v9{name}: {len(recs)} tep", {s: len(v) for s, v in by.items()}, flush=True)


# ------------------------------------------------------------------ kiem thu (Ctx9 gia, khong doc OOF / manifest)
def _fake_ctx(seed: int = 0, n: int = 400, nreg: int = 4) -> Ctx9:
    rng = np.random.default_rng(seed)
    ctx = object.__new__(Ctx9)
    ctx.tags = ("fuse1", "fuse1", "h2")
    ctx.ids = [f"f{k}" for k in range(n)]
    ctx.items = {i: {"set": "v6natvi" if k % 2 else "v6naten", "cfg": "goc", "real": k % 3 != 0, "kind": "x", "meta": {}} for k, i in enumerate(ctx.ids)}
    ctx.fold = {i: k % 5 for k, i in enumerate(ctx.ids)}
    ctx.lang = {i: "vi" if ctx.items[i]["set"] == "v6natvi" else "en" for i in ctx.ids}
    ctx.real = [i for i in ctx.ids if ctx.items[i]["real"]]
    ctx.nreg = nreg
    ctx.mb = {i: rng.uniform(0, 1, 2) for i in ctx.real}
    ctx.mr = {i: rng.uniform(0, 1, nreg) for i in ctx.real}
    ctx.HP = {i: rng.uniform(0, 1, 3) for i in ctx.ids}
    ctx.drop, ctx._th, ctx._r3, ctx._rr = set(), {}, {}, {}
    return ctx


def test_sweep_suffix_keeps_legacy_names() -> None:
    assert sweep_suffix("tcn", "tcn_reg") == "" and sweep_suffix("fuse1", "fuse1") == "_fuse1" and sweep_suffix("tcn", "tcn_r2") == "_tcn_tcn_r2"


def test_thr_fold_equals_research_thresholds() -> None:
    """thr_fold[k] (Product fold=k) phai TRUNG nguong nghien cuu Ctx9.th cho fold k (tep that cung ngon ngu cac fold khac)."""
    ctx = _fake_ctx()
    for k in range(5):
        tf = ctx.export_thresholds(0.02, 0.03, 0.01, 0.01, exclude_fold=k)
        for lg in ("vi", "en"):
            assert abs(tf[lg]["ghep"] - ctx.th("b0", {i: ctx.mb[i][0] for i in ctx.real}, 0.02)[(lg, k)]) < 1e-12
            assert abs(tf[lg]["voi"] - ctx.th("r1", {i: ctx.mr[i][1] for i in ctx.real}, 0.03)[(lg, k)]) < 1e-12
            assert abs(tf[lg]["ttsf"] - ctx.th("h2", {i: ctx.HP[i][2] for i in ctx.real}, 0.01)[(lg, k)]) < 1e-12
    full = ctx.export_thresholds(0.02, 0.03, 0.01, 0.0)
    assert "ttsr" not in full["vi"] and "ttsf" not in full["vi"] and "den" in full["en"]


def test_thr_writes_traceable_params() -> None:
    import tempfile
    global trace_params
    old_res, old_trace = D.RES, trace_params
    with tempfile.TemporaryDirectory() as d:
        D.RES = d
        trace_params = lambda b, r, h: {"oof": {"bound": b}, "oof_sha1": {"bound": "0" * 40}, "ckpt_sha1": {"x.pt": "1" * 40}, "ckpt_missing": [], "folds": "detect.person_fold"}  # noqa: E731
        try:
            ctx = _fake_ctx(1)
            out = thr("fuse1", "fuse1", "h2", "T", 0.01, 0.03, 0.005, 0.01, 0.0, True, True, with_metrics=False, ctx=ctx)
            back = json.load(open(os.path.join(d, "thresholds_v9T.json"), encoding="utf-8"))
            p = back["params"]
            assert (p["bound"], p["reg"], p["heads"], p["typed"], p["snap"], p["ftts"]) == ("fuse1", "fuse1", "h2", True, True, 0.01) and p["ckpt_sha1"] == {"x.pt": "1" * 40}
            assert sorted(back["thr_fold"]) == ["0", "1", "2", "3", "4"] and back["thr"] == out["thr"] and "ttsr" in back["thr"]["vi"]
            for bad in ((("tcn", "tcn_reg", "h2"), {}), (("fuse1", "fuse1", "h2"), {"typed": True, "ftts": 0.0})):
                try:
                    kw = {"f03": 0.01, "freg": 0.03, "fh": 0.005, "ftts": 0.01, "fspl": 0.0, "snap": True, "typed": False, **bad[1]}
                    thr(*bad[0], "U", with_metrics=False, ctx=ctx, **kw)
                except ValueError:
                    continue
                raise AssertionError(f"phai tu choi {bad}")
        finally:
            D.RES, trace_params = old_res, old_trace


def test_metrics_by_fold_groups_by_person_fold() -> None:
    ctx = _fake_ctx(2, n=50)
    recs = [{"id": i, "set": ctx.items[i]["set"], "expected": "01" if ctx.items[i]["real"] else "03"} for i in ctx.ids]
    old = RC.metrics
    RC.metrics = lambda rr: {"1": len(rr) / 100, "4": sum(r["expected"] == "01" for r in rr) / max(1, len(rr)), "x": "khong phai so"}
    try:
        mf = metrics_by_fold(ctx, recs)
    finally:
        RC.metrics = old
    for s in F.NAT_SETS:
        for k in range(5):
            want = [r for r in recs if r["set"] == s and ctx.fold[r["id"]] == k]
            m = mf[s][str(k)]
            assert m["n"] == len(want) and m["n_real"] == sum(r["expected"] == "01" for r in want) and abs(m["1"] - len(want) / 100) < 1e-9 and "x" not in m, m
    assert any("| v6natvi | 0 |" in x for x in perfold_md("T", mf))


def selftest() -> None:
    n = 0
    for name_, fn in list(globals().items()):
        if name_.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name_, flush=True)
    print(f"{n} test dat")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sweep", "emit", "thr", "perfold", "selftest"])
    ap.add_argument("--names", nargs="+", default=["F"], help="perfold: ten tep nguong thresholds_v9<ten>.json da co")
    ap.add_argument("--bound", default="tcn")
    ap.add_argument("--reg", default="tcn_reg")
    ap.add_argument("--heads", default="h2")
    ap.add_argument("--name", default="A")
    ap.add_argument("--f03", type=float, default=0.02)
    ap.add_argument("--freg", type=float, default=0.01)
    ap.add_argument("--fh", type=float, default=0.01)
    ap.add_argument("--ftts", type=float, default=0.01)
    ap.add_argument("--fspl", type=float, default=0.0)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--typed", action="store_true", help="phan loai dong ghep theo vung giong may / doan ghep that (can --regions2 va ftts > 0); ngam hieu --snap")
    ap.add_argument("--no-vss", action="store_true", help="sweep: bo tep THAT nguon vss (video YouTube, nghi da bi dung) khoi nguong va vu oan")
    ap.add_argument("--no-metrics", action="store_true", help="thr: khong tinh chi so OOF tai diem van hanh (nhanh hon)")
    a = ap.parse_args()
    snap = a.snap or a.typed
    if a.step == "selftest":
        selftest()
    elif a.step == "thr":
        thr(a.bound, a.reg, a.heads, a.name, a.f03, a.freg, a.fh, a.ftts, a.fspl, snap, a.typed, not a.no_metrics)
    elif a.step == "sweep":
        sweep(a.bound, a.reg, a.heads, ("vss",) if a.no_vss else ())
    elif a.step == "perfold":
        perfold(a.names)
    else:
        emit(a.bound, a.reg, a.heads, a.name, a.f03, a.freg, a.fh, a.ftts, a.fspl, snap, a.typed)
