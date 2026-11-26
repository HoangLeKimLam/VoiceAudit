"""Chấm kiểm H-trầm theo luật chốt trước (NHAT_KY 2.16 «Kiểm H-trầm»). Đọc dieu_kien_tram.jsonl (do_tram.py) + C50 cũ.
  (a) dev tổng hợp p1_dieukien_v2 fold 0–2: đúng 04/accept theo đáp án của chính ứng viên và theo đáp án toàn dải (bỏ SNR đáp án 9–11 dB)
  (b) 16 tệp thật Huấn nghe tay (dev): chặn oan / thả sai, 2 cách đếm
  (c) tỉ lệ bị chặn trên tệp accept p1_tunhien_v1
Không đọc fold test. Ra: p1_tunhien_v1/_kiem/CHAM_TRAM_2411.md

  .venv/Scripts/python.exe experts/P1/build_cond/cham_tram.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, SYN, V6, fold, jl, late, pct, wilson  # noqa: E402
from do_tram import LOC  # noqa: E402

UV = {"K0": "full", "K100": "hp100", "K300": "hp300", "KA": "a"}
NS04 = 100 / 11                                   # ns > 9,09 % ⇔ SNR < 10 dB (nền > 10 % cũng nằm trong)
BIEN = (late(11), late(9))                        # đáp án SNR 9–11 dB → bỏ khỏi (a)
CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_TRAM_2411.md")


def tach(y: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """tệp trộn y = k·gốc + nền → (k·gốc, nền), k bình phương tối thiểu."""
    n = min(len(y), len(g))
    k = float(np.dot(y[:n], g[:n]) / np.dot(g[:n], g[:n]))
    return k * g[:n], y[:n] - k * g[:n]


def ns_loc(s: np.ndarray, n: np.ndarray, f, san_db: float | None = None, y: np.ndarray | None = None) -> float:
    """% không phải giọng sau lọc f. s = k·gốc (gồm cả sàn nền của gốc, SNR sàn = san_db).
    y cho trước → E(f(n)) / E(f(y)) (định nghĩa «nền»); không thì (E(f(n)) + sàn) / (E(f(s)) + E(f(n)))."""
    es, en = float(np.sum(f(s) ** 2)), float(np.sum(f(n) ** 2))
    if y is not None:
        return 100 * en / max(float(np.sum(f(y) ** 2)), 1e-20)
    floor = es * 10 ** (-san_db / 10) / (1 + 10 ** (-san_db / 10)) if san_db is not None else 0.0
    return 100 * (en + floor) / max(es + en, 1e-20)


def dap_an_syn(cases: dict) -> tuple[dict, list]:
    """file → {cách: ns đáp án %} cho fold dev; kèm danh sách lệch kiểm độc lập (toàn dải vs provenance)."""
    prov = {p["file"].split(f"{SYN}/", 1)[-1]: p for p in jl(os.path.join(V6, SYN, "provenance.jsonl"))}
    goc = {c["base"]: f for f, c in cases.items() if c["ca"] == "goc"}
    out, lech = {}, []
    for f, c in cases.items():
        if fold(c["base"]) >= 3:
            continue
        ca, p = c["ca"], prov[f]
        if ca == "goc":
            v = late(p["tham_so"]["snr_san_goc_db"])
            out[f] = {k: v for k in LOC}
        elif ca.startswith("vang."):
            out[f] = {k: 0.0 for k in LOC}
        elif ca.startswith(("snr.", "nen.")):
            y, _ = sf.read(os.path.join(V6, SYN, f), dtype="float64")
            g, _ = sf.read(os.path.join(V6, SYN, goc[c["base"]]), dtype="float64")
            s, n = tach(y, g)
            # Khớp định nghĩa provenance (sửa 24/11 trước khi chấm): snr tính cả nền sẵn của gốc (sàn, coi như không đổi qua lọc,
            # giống ca `goc`); nen = E(nền) / E(tệp).
            san = p["tham_so"].get("snr_san_goc_db") if ca.startswith("snr.") else None
            out[f] = {k: ns_loc(s, n, fn, san, y[:len(s)] if san is None else None) for k, fn in LOC.items()}
            if ca.startswith("snr."):
                ok = abs(10 * np.log10((100 - out[f]["full"]) / out[f]["full"]) - p["do_lai"]["snr_that_uoc_db"]) <= 0.3
                ref = p["do_lai"]["snr_that_uoc_db"]
            else:
                ok, ref = abs(out[f]["full"] - p["do_lai"]["ti_le_nen_pct"]) <= 0.3, p["do_lai"]["ti_le_nen_pct"]
            if not ok:
                lech.append((f, round(out[f]["full"], 2), ref))
    return out, lech


def khop(rows: list) -> np.ndarray:
    """rows: (nen_X, c50, ns_đáp_án) → hệ số (a, b, c) của ns = a·nen + b·late(C50) + c."""
    X = np.array([[r[0], late(r[1]), 1.0] for r in rows])
    return np.linalg.lstsq(X, np.array([r[2] for r in rows]), rcond=None)[0]


def ns_uoc(coef, nen: float, c50: float) -> float:
    return float(np.clip(coef[0] * nen + coef[1] * late(c50) + coef[2], 0.01, 99.99))


def load_meas(bo: str) -> dict:
    old = {r["file"]: r for r in jl(os.path.join(V6, bo, "_kiem", "dieu_kien.jsonl"))}
    return {r["file"]: {**r, "c50": old[r["file"]]["c50_bh_db"], "nen_cu": old[r["file"]]["nen_dfn_pct"]}
            for r in jl(os.path.join(V6, bo, "_kiem", "dieu_kien_tram.jsonl")) if r["file"] in old}


def cases_of(bo: str) -> dict:
    c = json.load(open(os.path.join(V6, bo, "cases.json"), encoding="utf-8"))
    return {x["file"]: x for x in (c if isinstance(c, list) else c["cases"])}


def run() -> None:
    cs, ms = cases_of(SYN), load_meas(SYN)
    cn, mn = cases_of(NAT), load_meas(NAT)
    allm = list(ms.values()) + list(mn.values())
    tai_lap = sum(abs(m["nen_full"] - m["nen_cu"]) <= 0.5 for m in allm)
    da, lech = dap_an_syn(cs)
    dev = [f for f in da if f in ms and ms[f]["c50"] is not None]
    nghe = json.load(open(os.path.join(V6, NAT, "_kiem", "nghe_tay_2411.json"), encoding="utf-8"))["ket_qua"]
    hmap = [r for r in json.load(open(os.path.join(V6, NAT, "_kiem", "NGHE_TAY_2411_map.json"), encoding="utf-8"))["tep"] if r["loai"] == "that"]
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in mn and mn[f]["c50"] is not None]
    L = ["# Kiểm H-trầm (luật chốt trước: NHAT_KY 2.16)", "",
         f"- Tái lập K0: |nen_full − nen_dfn_pct cũ| ≤ 0,5 điểm ở {pct(tai_lap, len(allm))} (cần ≥ 95 %)",
         f"- Kiểm đáp án độc lập (toàn dải vs provenance): lệch {len(lech)}/{sum(cs[f]['ca'].startswith(('snr.', 'nen.')) for f in da)} tệp snr/nen dev"
         + (f"; ví dụ {lech[:3]}" if lech else ""),
         f"- Dev tổng hợp: {len(dev)} tệp · tệp thật Huấn nghe: {len(hmap)} · tệp accept p1_tunhien_v1 có số đo: {len(acc)}", "",
         "| ứng viên | hệ số (a, b, c) | (a) đúng, đáp án riêng | (a) đúng, đáp án toàn dải | (b) chặn oan chặt | (b) chặn oan rộng | (b) thả sai chặt | (b) thả sai rộng | (c) bị chặn |",
         "|---|---|---|---|---|---|---|---|---|"]
    res = {}
    for ten, k in UV.items():
        coef = khop([(ms[f][f"nen_{k}"], ms[f]["c50"], da[f][k]) for f in dev])
        chan = lambda m: ns_uoc(coef, m[f"nen_{k}"], m["c50"]) > NS04  # noqa: E731
        a = {}
        for dk in (k, "full"):
            g = [f for f in dev if not BIEN[0] <= da[f][dk] <= BIEN[1]]
            a[dk] = (sum(chan(ms[f]) == (da[f][dk] > NS04) for f in g), len(g))
        b = {}
        for cach, ON in (("chat", ("che_loi",)), ("rong", ("ro", "che_loi"))):
            hm = [(chan(mn[r["nguon"].split("/", 1)[1]]), nghe[r["ten"]]["on_nen"] in ON) for r in hmap]
            b[cach] = ((sum(c and not o for c, o in hm), sum(not o for _, o in hm)), (sum(o and not c for c, o in hm), sum(o for _, o in hm)))
        cc = (sum(chan(mn[f]) for f in acc), len(acc))
        res[ten] = {"a": a[k], "a_full": a["full"], "b": b, "c": cc}
        L.append(f"| {ten} | {', '.join(f'{x:.3f}' for x in coef)} | {pct(*a[k])} | {pct(*a['full'])} | {pct(*b['chat'][0])} | {pct(*b['rong'][0])} | "
                 f"{pct(*b['chat'][1])} | {pct(*b['rong'][1])} | {pct(*cc)} |")
    r0 = res["K0"]
    ok = [t for t in UV if res[t]["a"][0] / res[t]["a"][1] >= r0["a"][0] / r0["a"][1] - 0.05]
    chon = min(ok, key=lambda t: (res[t]["b"]["rong"][0][0] / max(res[t]["b"]["rong"][0][1], 1), res[t]["c"][0])) if ok else None
    c0 = r0["c"][0] / r0["c"][1]
    giam = {t: c0 - res[t]["c"][0] / res[t]["c"][1] for t in ("K100", "K300", "KA")}
    ket = ("ĐÚNG" if any(res[t]["c"][0] / res[t]["c"][1] <= c0 / 2 for t in giam)
           else "SAI" if all(v < 0.05 for v in giam.values()) else "CHƯA KẾT LUẬN")
    L += ["", f"- Ứng viên đạt điều kiện (a) (≥ K0 − 5 điểm): {', '.join(ok) or 'không có'} → **chọn {chon}**",
          f"- (c) giảm so với K0: " + " · ".join(f"{t} {100 * v:.1f} điểm" for t, v in giam.items()) + f" → **H-trầm {ket}**",
          "- Chưa mở test. Chặn oan để cài K1 vẫn phải ≤ 2 %."]
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    rng = np.random.default_rng(1)
    g, n = rng.standard_normal(8000), rng.standard_normal(8000)
    s2, n2 = tach(0.7 * g + 0.2 * n, g)
    assert abs(float(np.dot(s2, g) / np.dot(g, g)) - 0.7) < 0.02 and abs(ns_loc(s2, n2, LOC["full"]) - 100 * 0.04 / 0.53) < 1
    coef = khop([(x, 50.0, 2 * x + 1) for x in (1.0, 5.0, 9.0, 20.0)] + [(3.0, 0.0, 2 * 3 + 1 + 0)])
    assert abs(ns_uoc(coef, 4.0, 50.0) - 9.0) < 0.5, coef
    lo, hi = wilson(0, 10)
    assert lo == 0 and hi > 0.2
    print("cham_tram selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
