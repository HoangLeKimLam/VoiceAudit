# -*- coding: utf-8 -*-
"""GHEP THU (lane tu kiem truoc khi nop, 19/10): cam cot diem / dau ca tep cua MOT lane vao he thong tren OOF va do chi so HE THONG truoc / sau.

  Nen = fuse1 (diem khung (T, 6): ghep, xoa, den, voi, spl, ttsr) + h2 (dau ca tep (N, 3): nen, giong, ttsf) - dung bo dang chay san pham.
  Lane nop OOF dung hop dong H3 ({id: (T, k) float16}, 5 fold theo nguoi noi) -> thay (replace) hoac trung binh logit (mean) vao cot chi dinh;
  dau ca tep: {"ids": [...], "P": (N, k)} hoac {id: p}. Tep thieu trong OOF lane -> giu diem nen (bao so tep thieu).
  Danh gia nhu fuse9.Ctx9.records: nguong = phan vi tep THAT tu nhien cung ngon ngu o cac fold KHAC, tai DUNG diem van hanh cua tep nguong
  (mac dinh F: f03 / freg / fh / ftts / fspl / snap / typed lay tu thresholds_v9F.json params) hoac --budget (ngan sach vu oan tung kenh).
  Ra: Req 1-4, 9 (+ 6, 7a, 7b, 8a, 8b khi do duoc) theo bo, % dung theo kieu, vu oan tep that THEO KENH (de kiem ngan sach), so sanh voi nen.
CLI: python -m v6.product.ghep_thu --name p3_thu1 --col den=experts/P3/x/oof_x.pkl:0 --col voi=experts/P3/x/oof_x.pkl:1 [--mode replace|mean]
     python -m v6.product.ghep_thu --name p5_thu --head ttsf=experts/P5/y/heads_y.pkl:0 [--f03 0.005 --freg 0.005 --fh 0.0025 --ftts 0.0025]
     python -m v6.product.ghep_thu --name nen                                  (khong cam gi: tai tao nen - kiem cong cu)
  -> .tmp_v6/results/ghep_thu_<name>.md ; OOF ghep tam: .tmp_v6/cache/boundary_oof_gt_<name>.pkl, heads2_gt_<name>.pkl (xoa duoc)
NANG VUA: doc 2 OOF (~1-2 GB RAM). Chay qua run_serial.sh."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
RES = os.path.join(BASE, ".tmp_v6", "results")
COLS = {"ghep": 0, "xoa": 1, "den": 2, "voi": 3, "spl": 4, "ttsr": 5}
HEADS = {"nen": 0, "giong": 1, "ttsf": 2}
EPS = 1e-4


def parse_spec(s: str, table: dict) -> tuple:
    """'den=duong/dan.pkl:0' -> ('den', cot dich, duong dan, cot nguon) (ham thuan)."""
    k, rest = s.split("=", 1)
    path, _, c = rest.rpartition(":")
    if k not in table:
        raise ValueError(f"kenh '{k}' khong co trong {sorted(table)}")
    if not path:
        path, c = rest, "0"
    return k, table[k], path, int(c)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def combine(base: np.ndarray, new: np.ndarray, mode: str) -> np.ndarray:
    """Cot moi = new (replace) hoac sigmoid(trung binh logit) (mean); do dai lech -> cat ve ngan hon roi dem phan con lai tu nen (ham thuan)."""
    n = min(len(base), len(new))
    out = base.astype(np.float32).copy()
    if mode == "replace":
        out[:n] = new[:n]
    else:
        out[:n] = 1 / (1 + np.exp(-(_logit(base[:n]) + _logit(new[:n])) / 2))
    return out


def merge_frames(base: dict, specs: list, mode: str) -> tuple:
    """{id: (T, 6)} da cam cot lane; tra (oof moi, {kenh: so tep thieu})."""
    lane = {}
    for k, c, path, sc in specs:
        lane.setdefault(path, pickle.load(open(path, "rb")))
    out, miss = {}, Counter()
    for i, P in base.items():
        Q = P.astype(np.float32).copy()
        for k, c, path, sc in specs:
            L = lane[path].get(i)
            if L is None:
                miss[k] += 1
                continue
            L = np.asarray(L, np.float32)
            Q[:, c] = combine(Q[:, c], L[:, sc] if L.ndim == 2 else L, mode)
        out[i] = Q.astype(np.float16)
    return out, dict(miss)


def merge_heads(base: dict, specs: list, mode: str) -> tuple:
    hid = {i: n for n, i in enumerate(base["ids"])}
    P = np.asarray(base["P"], np.float32).copy()
    miss = Counter()
    for k, c, path, sc in specs:
        h = pickle.load(open(path, "rb"))
        get = (lambda i, h=h: h.get(i)) if "ids" not in h else (lambda i, h=h, m={x: n for n, x in enumerate(h["ids"])}: (
            np.asarray(h["P"])[m[i]] if i in m else None))
        for i, n in hid.items():
            v = get(i)
            if v is None:
                miss[k] += 1
                continue
            v = float(np.ravel(v)[sc]) if np.ndim(v) else float(v)
            P[n, c] = v if mode == "replace" else float(1 / (1 + np.exp(-(_logit(np.array([P[n, c]]))[0] + _logit(np.array([v]))[0]) / 2)))
    return {**base, "P": P}, dict(miss)


def evaluate(tag: str, heads: str, op: dict) -> dict:
    """Chi so he thong OOF tai diem van hanh op (fuse9.Ctx9). Tra {bo: metrics, 'kieu': {(bo, kieu): (dung, n)}, 'fa_kenh': {bo: {kenh: ti le}}}."""
    from v6.boundary import fuse9 as F9
    from v6.boundary import fusion as F
    from v6.natural import req_check as RC
    ctx = F9.Ctx9(tag, tag, heads, F.NAT_SETS)
    recs = ctx.records(op["f03"], op["freg"], op["fh"], op["ftts"], fspl=op.get("fspl", 0.0), snap=op.get("snap", False), typed=op.get("typed", False))
    out = {"kieu": {}, "fa_kenh": {}}
    by = defaultdict(list)
    for r in recs:
        by[r["set"]].append(r)
    for s, rs in by.items():
        out[s] = {k: float(v) for k, v in RC.metrics(rs).items() if np.isscalar(v)}
        kk = defaultdict(lambda: [0, 0])
        fa = Counter()
        n_real = 0
        for r in rs:
            if r["final"] == "04":
                continue
            kk[r["kind"]][0] += r["correct"]
            kk[r["kind"]][1] += 1
            if r["expected"] == "01":
                n_real += 1
                for lv in {x["level2"] + ("/toan" if x.get("kind") == "global" else "") for x in r["rows"]}:
                    fa[lv] += 1
        out["kieu"].update({(s, k): tuple(v) for k, v in kk.items()})
        out["fa_kenh"][s] = {k: v / max(n_real, 1) for k, v in fa.items()}
    return out


def report(name: str, base: dict, new: dict, miss: dict, op: dict) -> list:
    f = lambda m, k: f"{m.get(k, float('nan')):.3f}"                    # noqa: E731
    L = [f"# Ghep thu `{name}` - OOF natvi / naten goc, diem van hanh {op}", "", f"Tep thieu trong OOF lane (giu diem nen): {miss or 'khong'}", "",
         "| bo | | Req 1 | Req 2 | Req 3 | Req 4 | Req 9 | 7a | 7b | 6 | 8a |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(k for k in base if k.startswith("v6")):
        for lab, m in (("nen", base[s]), ("MOI", new[s])):
            L.append(f"| {s} | {lab} | {f(m, '1')} | {f(m, '2')} | {f(m, '3')} | {f(m, '4')} | {f(m, '9')} | {f(m, '7a')} | {f(m, '7b')} | "
                     f"{f(m, '6')} | {f(m, '8a')} |")
    L += ["", "## Vu oan tep that theo kenh (ty le tep that co dong cua kenh) - kiem ngan sach", "", "| bo | kenh | nen | MOI |", "|---|---|---|---|"]
    for s in sorted(new["fa_kenh"]):
        for k in sorted(set(base["fa_kenh"].get(s, {})) | set(new["fa_kenh"][s])):
            L.append(f"| {s} | {k} | {base['fa_kenh'].get(s, {}).get(k, 0):.3f} | {new['fa_kenh'][s].get(k, 0):.3f} |")
    L += ["", "## % dung theo kieu (o tut > 5 diem danh dau !)", "", "| bo | kieu | n | nen | MOI | |", "|---|---|---|---|---|---|"]
    for (s, k) in sorted(new["kieu"]):
        b, n_ = base["kieu"].get((s, k), (0, 1)), new["kieu"][(s, k)]
        pb, pn = b[0] / max(b[1], 1), n_[0] / max(n_[1], 1)
        L.append(f"| {s} | {k} | {n_[1]} | {pb:.2f} | {pn:.2f} | {'!' if pn < pb - 0.05 else ''} |")
    return L


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--col", action="append", default=[], help="kenh=duong_dan_oof.pkl:cot_nguon (kenh: " + " ".join(COLS) + ")")
    ap.add_argument("--head", action="append", default=[], help="kenh=duong_dan.pkl:cot_nguon (kenh: " + " ".join(HEADS) + ")")
    ap.add_argument("--mode", choices=["replace", "mean"], default="replace")
    ap.add_argument("--thr", default="thresholds_v9F.json", help="lay diem van hanh (params) tu tep nguong nay")
    for k in ("f03", "freg", "fh", "ftts", "fspl"):
        ap.add_argument(f"--{k}", type=float, default=None, help="ghi de ngan sach vu oan cua kenh (vd 0.005)")
    a = ap.parse_args()
    from v6.boundary import detect as D
    params = json.load(open(os.path.join(RES, a.thr), encoding="utf-8"))["params"]
    op = {k: params.get(k, 0.0) for k in ("f03", "freg", "fh", "ftts", "fspl", "snap", "typed")}
    op.update({k: getattr(a, k) for k in ("f03", "freg", "fh", "ftts", "fspl") if getattr(a, k) is not None})
    bound, heads = params.get("bound", "fuse1"), params.get("heads", "h2")
    base_eval = evaluate(bound, heads, op)
    tag, htag, miss = bound, heads, {}
    if a.col:
        merged, miss = merge_frames(pickle.load(open(D.tagged(D.OOFP, bound), "rb")), [parse_spec(s, COLS) for s in a.col], a.mode)
        tag = f"gt_{a.name}"
        pickle.dump(merged, open(D.tagged(D.OOFP, tag), "wb"))
    if a.head:
        hm, mh = merge_heads(pickle.load(open(os.path.join(CACHE, f"heads2_{heads}.pkl"), "rb")), [parse_spec(s, HEADS) for s in a.head], a.mode)
        htag = f"gt_{a.name}"
        pickle.dump(hm, open(os.path.join(CACHE, f"heads2_{htag}.pkl"), "wb"))
        miss.update({f"head:{k}": v for k, v in mh.items()})
    new_eval = base_eval if (tag, htag) == (bound, heads) else evaluate(tag, htag, op)
    L = report(a.name, base_eval, new_eval, miss, op)
    p = os.path.join(RES, f"ghep_thu_{a.name}.md")
    open(p, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print("->", p)


# ---------------- kiem thu (khong doc OOF that) ----------------
def selftest() -> None:
    assert parse_spec("den=a/b.pkl:1", COLS) == ("den", 2, "a/b.pkl", 1)
    assert parse_spec("ttsf=x.pkl", HEADS) == ("ttsf", 2, "x.pkl", 0)
    try:
        parse_spec("abc=x.pkl:0", COLS)
        raise AssertionError
    except ValueError:
        pass
    b = np.array([0.2, 0.8, 0.5], np.float32)
    assert np.allclose(combine(b, np.array([0.9, 0.1]), "replace"), [0.9, 0.1, 0.5])
    m = combine(b, np.array([0.8, 0.2, 0.5]), "mean")
    assert np.allclose(m, [0.5, 0.5, 0.5], atol=1e-4)                      # logit doi xung -> 0,5
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lane.pkl")
        pickle.dump({"a": np.full((4, 2), 0.9, np.float16)}, open(p, "wb"))
        base = {"a": np.zeros((5, 6), np.float16), "b": np.zeros((3, 6), np.float16)}
        out, miss = merge_frames(base, [("den", 2, p, 1)], "replace")
        assert float(out["a"][0, 2]) > 0.89 and float(out["a"][4, 2]) == 0.0 and miss == {"den": 1} and float(out["b"].max()) == 0.0
        hp = os.path.join(d, "h.pkl")
        pickle.dump({"ids": ["a"], "P": np.array([[0.7]])}, open(hp, "wb"))
        hb, mh = merge_heads({"ids": ["a", "b"], "P": np.zeros((2, 3))}, [("ttsf", 2, hp, 0)], "replace")
        assert abs(hb["P"][0, 2] - 0.7) < 1e-6 and hb["P"][1, 2] == 0 and mh == {"ttsf": 1}
    print("ghep_thu selftest dat")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
