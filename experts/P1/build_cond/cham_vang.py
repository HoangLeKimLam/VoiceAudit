"""Chấm kiểm H-vang + H-lặng theo luật chốt trước (NHAT_KY 2.16). Đọc dieu_kien_vang.jsonl (do_vang.py), dieu_kien_tram.jsonl (nen_a),
dieu_kien.jsonl cũ (C50, SNR Brouhaha, tỉ lệ lời Brouhaha). Đáp án tổng hợp trọng số A từ cham_tram.dap_an_syn. Không đọc fold test.
Ra: p1_tunhien_v1/_kiem/CHAM_VANG_2411.md

  .venv/Scripts/python.exe experts/P1/build_cond/cham_vang.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, SYN, V6, jl, late, pct  # noqa: E402
from cham_tram import BIEN, NS04, cases_of, dap_an_syn  # noqa: E402

CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_VANG_2411.md")
C50_NHOM = ((-1e9, 10), (10, 20), (20, 35), (35, 1e9))
LOI_NHOM = ((0, 0.5), (0.5, 0.85), (0.85, 1.01))


def gop(bo: str) -> dict:
    """file → mọi số đo cần (chỉ tệp có đủ 3 nguồn và có C50)."""
    k = lambda n: {r["file"]: r for r in jl(os.path.join(V6, bo, "_kiem", n))}  # noqa: E731
    cu, tr, va = k("dieu_kien.jsonl"), k("dieu_kien_tram.jsonl"), k("dieu_kien_vang.jsonl")
    return {f: {"c50": cu[f]["c50_bh_db"], "bh": cu[f]["snr_bh_db"], "loi_bh": cu[f]["loi_noi_bh_s"] / cu[f]["dur_s"],
                "nen_a": tr[f]["nen_a"], **va[f]} for f in va if f in cu and f in tr and cu[f]["c50_bh_db"] is not None}


def ns_snr(snr: float) -> float:
    return late(snr)                                   # % không phải giọng khi SNR = snr


def dac_trung(m: dict) -> dict:
    """Biến đầu vào của từng ứng viên (hàng của ma trận khớp, chưa có hệ số)."""
    c1 = [m["nen_a_loi"], late(m["c50"]), 1.0]
    return {"KA": [m["nen_a"], late(m["c50"]), 1.0], "C1": c1,
            "C2": [ns_snr(m["snr_nghi"]), 1.0] if m["snr_nghi"] is not None else None,
            "C3": [late(m["bh"]), 1.0] if m["bh"] is not None else None}


def khop(X: list, y: list) -> np.ndarray:
    return np.linalg.lstsq(np.array(X, float), np.array(y, float), rcond=None)[0]


def lam(dev: list, M: dict, da: dict) -> dict:
    """Khớp hệ số mỗi ứng viên trên dev → hàm chan(m) → bool."""
    coef = {}
    for u in ("KA", "C1", "C2", "C3"):
        rows = [(dac_trung(M[f])[u], da[f]["a"]) for f in dev if dac_trung(M[f])[u] is not None]
        coef[u] = khop([r[0] for r in rows], [r[1] for r in rows])
    ns = lambda u, m: float(np.dot(coef[u], dac_trung(m)[u]))  # noqa: E731
    f = {u: (lambda m, u=u: ns(u, m) > NS04) for u in ("KA", "C1", "C3")}
    f["C2"] = lambda m: ns("C2", m) > NS04 if m["snr_nghi"] is not None else f["C1"](m)
    f["C4"] = lambda m: f["C1"](m) and (f["C3"](m) if m["bh"] is not None else True)
    return {"chan": f, "coef": coef}


def run() -> None:
    cs, M = cases_of(SYN), gop(SYN)
    cn, MN = cases_of(NAT), gop(NAT)
    da, _ = dap_an_syn(cs)
    dev = [f for f in da if f in M]
    L_ = lam(dev, M, da)
    chan = L_["chan"]
    nghe = json.load(open(os.path.join(V6, NAT, "_kiem", "nghe_tay_2411.json"), encoding="utf-8"))["ket_qua"]
    hmap = [r for r in json.load(open(os.path.join(V6, NAT, "_kiem", "NGHE_TAY_2411_map.json"), encoding="utf-8"))["tep"] if r["loai"] == "that"]
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN]
    vang = [f for f in dev if cs[f]["ca"].startswith("vang.")]
    L = ["# Kiểm H-vang + H-lặng (luật chốt trước: NHAT_KY 2.16)", "",
         f"- Dev tổng hợp {len(dev)} (chỉ vang {len(vang)}) · tệp thật Huấn nghe {len(hmap)} · accept p1_tunhien_v1 {len(acc)}",
         f"- C2 thiếu khung nghỉ (dùng C1): dev {sum(M[f]['snr_nghi'] is None for f in dev)} · thật {sum(MN[f]['snr_nghi'] is None for f in acc)}",
         "- Hệ số: " + " · ".join(f"{u} {np.round(c, 3).tolist()}" for u, c in L_["coef"].items()), "",
         "| ứng viên | (a) đúng, đáp án A | (a) đáp án toàn dải | (a′) chặn tệp chỉ vang | (b) chặn oan chặt | (b) chặn oan rộng | (b) thả sai chặt | (b) thả sai rộng | (c) bị chặn | (c) vi | (c) en |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for u, fn in chan.items():
        a = {}
        for dk in ("a", "full"):
            g = [f for f in dev if not BIEN[0] <= da[f][dk] <= BIEN[1]]
            a[dk] = (sum(fn(M[f]) == (da[f][dk] > NS04) for f in g), len(g))
        av = (sum(fn(M[f]) for f in vang), len(vang))
        b = {}
        for cach, ON in (("chat", ("che_loi",)), ("rong", ("ro", "che_loi"))):
            hm = [(fn(MN[r["nguon"].split("/", 1)[1]]), nghe[r["ten"]]["on_nen"] in ON) for r in hmap]
            b[cach] = ((sum(c and not o for c, o in hm), sum(not o for _, o in hm)), (sum(o and not c for c, o in hm), sum(o for _, o in hm)))
        cc = {lg: (sum(fn(MN[f]) for f in acc if lg in ("", f[:2])), sum(lg in ("", f[:2]) for f in acc)) for lg in ("", "vi", "en")}
        d50 = [(sum(fn(MN[f]) for f in acc if lo <= MN[f]["c50"] < hi), sum(lo <= MN[f]["c50"] < hi for f in acc)) for lo, hi in C50_NHOM]
        dloi = [(sum(fn(MN[f]) for f in acc if lo <= MN[f]["loi_bh"] < hi), sum(lo <= MN[f]["loi_bh"] < hi for f in acc)) for lo, hi in LOI_NHOM]
        res[u] = {"a": a["a"], "av": av, "b": b, "c": cc[""], "d50": d50, "dloi": dloi}
        L.append(f"| {u} | {pct(*a['a'])} | {pct(*a['full'])} | {pct(*av)} | {pct(*b['chat'][0])} | {pct(*b['rong'][0])} | {pct(*b['chat'][1])} | "
                 f"{pct(*b['rong'][1])} | {pct(*cc[''])} | {pct(*cc['vi'])} | {pct(*cc['en'])} |")
    L += ["", "## (d) tỉ lệ bị chặn theo nhóm (tệp thật accept)", "",
          "| ứng viên | C50 < 10 | 10–20 | 20–35 | ≥ 35 | lời < 0,5 | 0,5–0,85 | ≥ 0,85 |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {u} | " + " | ".join(pct(*x) for x in r["d50"] + r["dloi"]) + " |" for u, r in res.items()]
    r0, ti = res["KA"], lambda x: x[0] / max(x[1], 1)  # noqa: E731
    ok = [u for u, r in res.items() if ti(r["a"]) >= ti(r0["a"]) - 0.05 and ti(r["av"]) <= ti(r0["av"])]
    chon = min(ok, key=lambda u: (ti(res[u]["b"]["rong"][0]), ti(res[u]["c"]))) if ok else None
    hv = "C2" in ok and ti(res["C2"]["d50"][0]) <= ti(r0["d50"][0]) / 2
    hl = ti(res["C1"]["dloi"][0]) <= ti(r0["dloi"][0]) / 2
    L += ["", f"- Đạt điều kiện chọn ((a) ≥ KA − 5 điểm và (a′) ≤ KA): {', '.join(ok) or 'không có'} → **chọn {chon}**",
          f"- H-vang: C2 {'đạt' if 'C2' in ok else 'KHÔNG đạt'} điều kiện; nhóm C50 < 10: KA {100 * ti(r0['d50'][0]):.0f} % → C2 "
          f"{100 * ti(res['C2']['d50'][0]):.0f} % → **{'ĐÚNG' if hv else 'CHƯA ĐỦ'}**",
          f"- H-lặng: nhóm lời < 0,5: KA {100 * ti(r0['dloi'][0]):.0f} % → C1 {100 * ti(res['C1']['dloi'][0]):.0f} % → **{'ĐÚNG' if hl else 'CHƯA ĐỦ'}**",
          "- (c) là cận trên của chặn oan (tệp thật có thể ồn thật). Chưa mở test. Cài K1 vẫn cần chặn oan ≤ 2 %."]
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    m = {"c50": 50.0, "bh": 20.0, "nen_a": 5.0, "nen_a_loi": 4.0, "snr_nghi": None, "loi_bh": 0.9}
    d = dac_trung(m)
    assert d["C2"] is None and len(d["KA"]) == 3 and abs(d["C3"][0] - late(20.0)) < 1e-12
    M = {f"x{i}": {**m, "nen_a": v, "nen_a_loi": v, "snr_nghi": s, "bh": s} for i, (v, s) in enumerate([(1, 30), (5, 15), (20, 6), (40, 2), (60, 0)])}
    da = {f: {"a": M[f]["nen_a"], "full": M[f]["nen_a"]} for f in M}
    L_ = lam(list(M), M, da)
    assert [L_["chan"]["KA"](M[f]) for f in M] == [False, False, True, True, True]
    assert L_["chan"]["C2"]({**m, "nen_a_loi": 50.0}) is True       # thiếu khung nghỉ → dùng C1
    print("cham_vang selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
