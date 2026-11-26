# -*- coding: utf-8 -*-
"""Chấm bộ đo điều kiện (do_dieu_kien.py) theo quy trình chốt 23/11 TRƯỚC khi xem số:
  Bước 1 (bộ tổng hợp p1_dieukien_v2, đáp án biết trước): fold crc32(base) % 5; dev 0–2 khớp đường quy đổi số đo → đơn vị thật
      snr: thật ≈ a·đo + b (Brouhaha và DeepFilterNet; chọn bộ có MAE dev nhỏ hơn) · nen: thật ≈ a·đo + b · vang: log RT60 ≈ a·C50 + b
      rồi áp NGƯỠNG HỢP ĐỒNG (SNR < 10 dB → 04 · nền > 10 % → 04 · RT60 > 1,0 s → cảnh báo), chấm trên test (bỏ ca biên).
  Bước 2 (ghi âm thật p1_tunhien_v1, đáp án accept): áp nguyên đường quy đổi + ngưỡng → tỉ lệ chặn oan / cảnh báo, Wilson 95 %.
      Không chỉnh gì theo bước 2.
  python experts/P1/build_cond/cham_dieu_kien.py run | selftest   → data_labels/v6/p1_dieukien_v2/_kiem/CHAM_DIEU_KIEN.md"""
from __future__ import annotations

import json
import os
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chay_k1 import wilson  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V6 = os.path.join(ROOT, "data_labels", "v6")
SYN, NAT = "p1_dieukien_v2", "p1_tunhien_v1"
NGUONG = {"snr": 10.0, "nen": 10.0, "vang": 1.0}   # hợp đồng, KHÔNG chỉnh


def fold(base: str) -> int:
    return zlib.crc32(base.encode()) % 5


def fit(x, y) -> tuple[float, float]:
    a, b = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return float(a), float(b)


def jl(path: str) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def load(bo: str) -> tuple[dict, dict]:
    d = os.path.join(V6, bo)
    cases = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    cases = {c["file"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}
    return cases, {r["file"]: r for r in jl(os.path.join(d, "_kiem", "dieu_kien.jsonl"))}


def truth_syn() -> dict:
    """file → giá trị thật của trục mình (snr dB thật ước · nền % · RT60 s) từ provenance."""
    out = {}
    for p in jl(os.path.join(V6, SYN, "provenance.jsonl")):
        f, dl, ts = p["file"].split(f"{SYN}/", 1)[-1], p["do_lai"], p["tham_so"]
        if p["ca"].startswith("snr."):
            out[f] = ("snr", dl["snr_that_uoc_db"])
        elif p["ca"].startswith("nen."):
            out[f] = ("nen", dl["ti_le_nen_pct"])
        elif p["ca"].startswith("vang."):
            out[f] = ("vang", dl["rt60_rir_schroeder_s"])
        elif p["ca"] == "goc":
            out[f] = ("goc", ts.get("snr_san_goc_db"))
    return out


# Bản 2 (đổi sau khi xem bước 1 bản 1, CHỈ dựa trên dev; NHAT_KY 2.15): DeepFilterNet tính cả vang là «không phải lời nói»
# (RT60 0,5 s → 15 % nền) còn Brouhaha SNR bền với vang nhưng nén thang (8–12 dB đều ra ~15 dB). Nên:
#   ns% (năng lượng không phải giọng) = a·nen_dfn + b·late(C50) + c   (late = % năng lượng vang muộn suy từ C50, để trừ phần vang DFN bắt nhầm)
#   SNR = 10·log10((100 − ns) / ns)  (cùng một đại lượng: nền > 10 % ⇔ SNR < 9,54 dB) · RT60 = exp(a·C50 + b) khớp trên C50 < 15 dB
C50_INFO_DB = 15.0


def late(c50: float) -> float:
    return 100.0 / (1.0 + 10 ** (c50 / 10))


def ns_truth(ax: str, v: float | None) -> float:
    """% năng lượng không phải giọng THẬT của tệp tổng hợp theo trục của nó."""
    if ax in ("snr", "goc"):
        return 0.0 if v is None else late(v)            # late(x) = 100 / (1 + 10^(x/10)) cũng là % ồn khi SNR = x
    return float(v) if ax == "nen" else 0.0              # vang: không thêm ồn / nhạc


def calibrate(cases: dict, meas: dict, tru: dict) -> dict:
    dev = [f for f in tru if f in meas and fold(cases[f]["base"]) < 3 and meas[f]["c50_bh_db"] is not None]
    X = np.array([[meas[f]["nen_dfn_pct"], late(meas[f]["c50_bh_db"]), 1.0] for f in dev])
    y = np.array([ns_truth(*tru[f]) for f in dev])
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    xy = [(meas[f]["c50_bh_db"], tru[f][1]) for f in dev if tru[f][0] == "vang" and meas[f]["c50_bh_db"] < C50_INFO_DB]
    return {"ns": [float(c) for c in coef], "c50_bh_db": fit([x for x, _ in xy], [np.log(v) for _, v in xy]),
            "n_dev": len(dev), "n_dev_vang": len(xy)}


def estimate(m: dict, cal: dict) -> dict:
    if m.get("c50_bh_db") is None:
        return {"snr": None, "nen": None, "vang": None}
    a, b, c = cal["ns"]
    ns = float(np.clip(a * m["nen_dfn_pct"] + b * late(m["c50_bh_db"]) + c, 0.01, 99.99))
    rt = float(np.exp(cal["c50_bh_db"][0] * m["c50_bh_db"] + cal["c50_bh_db"][1]))
    return {"snr": 10 * np.log10((100 - ns) / ns), "nen": ns, "vang": rt}


def flags(e: dict) -> dict:
    return {"snr": e["snr"] is not None and e["snr"] < NGUONG["snr"], "nen": e["nen"] is not None and e["nen"] > NGUONG["nen"],
            "vang": e["vang"] is not None and e["vang"] > NGUONG["vang"]}


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / max(n, 1):.1f} % ({100 * lo:.1f}–{100 * hi:.1f})"


def step1(cases: dict, meas: dict, tru: dict, cal: dict) -> list:
    a, b, c = cal["ns"]
    L = ["## Bước 1: bộ tổng hợp `p1_dieukien_v2` (đường quy đổi khớp trên dev, chấm trên test)", "",
         f"- Không phải giọng (%) = {a:.3f}·nen_dfn + {b:.3f}·late(C50) + {c:.2f} (khớp trên {cal['n_dev']} tệp dev mọi trục); SNR suy từ đó",
         f"- RT60 = exp({cal['c50_bh_db'][0]:.4f}·C50 + {cal['c50_bh_db'][1]:.3f}) (khớp trên {cal['n_dev_vang']} tệp vang dev có C50 < {C50_INFO_DB:g} dB)", "",
         "| trục | tập | n | MAE | quyết định đúng theo ngưỡng hợp đồng (bỏ ca biên) |", "|---|---|---|---|---|"]
    for ax, unit in (("snr", "dB"), ("nen", "điểm %"), ("vang", "s")):
        for name, sel in (("dev", lambda f: fold(cases[f]["base"]) < 3), ("test", lambda f: fold(cases[f]["base"]) >= 3)):
            fs = [f for f, (a, _) in tru.items() if a == ax and f in meas and sel(f)]
            est = {f: estimate(meas[f], cal)[ax] for f in fs}
            fs = [f for f in fs if est[f] is not None]
            err = [abs(est[f] - tru[f][1]) for f in fs]
            ok = [f for f in fs if not cases[f]["ca_bien"]]
            right = sum(flags(estimate(meas[f], cal))[ax] == (cases[f]["mong_doi"] != "accept") for f in ok)
            L.append(f"| {ax} | {name} | {len(fs)} | {np.mean(err):.2f} {unit} | {pct(right, len(ok))} |")
    L += ["", "Báo động nhầm trên tệp **gốc sạch** (SNR sẵn có ≥ 25 dB, không thêm gì) và chéo trục (tệp chỉ biến đổi trục khác):", "",
          "| tệp | n | snr < 10 dB | nền > 10 % | RT60 > 1,0 s |", "|---|---|---|---|---|"]
    for grp in ("goc", "snr", "nen", "vang"):
        fs = [f for f, (a, _) in tru.items() if a == grp and f in meas]
        fl = [flags(estimate(meas[f], cal)) for f in fs]
        L.append(f"| {grp} | {len(fs)} | " + " | ".join(pct(sum(x[k] for x in fl), len(fl)) for k in ("snr", "nen", "vang")) + " |")
    return L


def step2(cal: dict) -> list:
    cases, meas = load(NAT)
    k1 = {r["file"]: r for r in jl(os.path.join(V6, NAT, "_kiem", "k1_ket_qua.jsonl"))}
    fs = [f for f, c in cases.items() if c["mong_doi"] == "accept" and f in meas]
    L = ["", "## Bước 2: ghi âm thật `p1_tunhien_v1` (đáp án accept), áp nguyên quy đổi + ngưỡng hợp đồng", "",
         f"Đã đo {len(fs)} / {sum(c['mong_doi'] == 'accept' for c in cases.values())} tệp accept. «04 mới» = ồn quá hoặc nền che; "
         "«K1 đang nhận» = tệp K1 hiện không chặn (cột cuối: số tệp bị chặn THÊM).", "",
         "| tập | lang | n | ồn quá (SNR < 10) | nền che (> 10 %) | 04 mới (ồn hoặc nền) | cảnh báo vang | 04 mới trong tệp K1 đang nhận |",
         "|---|---|---|---|---|---|---|---|"]
    for name, sel in (("dev", lambda c: fold(c["base"]) < 3), ("test", lambda c: fold(c["base"]) >= 3), ("tất cả", lambda c: True)):
        for lang in ("en", "vi", "chung"):
            g = [f for f in fs if sel(cases[f]) and (lang == "chung" or f.startswith(lang + "/"))]
            fl = {f: flags(estimate(meas[f], cal)) for f in g}
            new = [f for f in g if fl[f]["snr"] or fl[f]["nen"]]
            acc = [f for f in g if k1.get(f, {}).get("decision") != "04"]
            L.append(f"| {name} | {lang} | {len(g)} | {pct(sum(fl[f]['snr'] for f in g), len(g))} | {pct(sum(fl[f]['nen'] for f in g), len(g))} | "
                     f"{pct(len(new), len(g))} | {pct(sum(fl[f]['vang'] for f in g), len(g))} | {pct(len([f for f in new if f in acc]), len(acc))} |")
    est = [estimate(meas[f], cal) for f in fs]
    L += ["", "Phân bố ước lượng trên tệp thật (trung vị · phân vị 10 · phân vị 90):"]
    for ax in ("snr", "nen", "vang"):
        v = np.array([e[ax] for e in est if e[ax] is not None])
        L.append(f"- {ax}: {np.median(v):.2f} · {np.percentile(v, 10):.2f} · {np.percentile(v, 90):.2f}")
    return L


def run() -> str:
    cases, meas = load(SYN)
    tru = truth_syn()
    cal = calibrate(cases, meas, tru)
    L = ["# Chấm bộ đo điều kiện (Brouhaha + DeepFilterNet3), quy trình chốt 23/11 trước khi xem số", "",
         "Ngưỡng hợp đồng giữ nguyên. Wilson 95 % trong ngoặc. Fold = crc32(base) % 5: dev 0–2, test 3–4.", ""]
    L += step1(cases, meas, tru, cal)
    L += step2(cal) if os.path.exists(os.path.join(V6, NAT, "_kiem", "dieu_kien.jsonl")) else ["", "(Bước 2: chưa có số đo tệp thật)"]
    out = os.path.join(V6, SYN, "_kiem", "CHAM_DIEU_KIEN.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({k: v for k, v in cal.items()}, open(os.path.join(V6, SYN, "_kiem", "hieu_chinh.json"), "w", encoding="utf-8"), indent=1)
    return "\n".join(L)


def selftest() -> None:
    assert fit([0, 1, 2], [1, 3, 5]) == (2.0, 1.0) or np.allclose(fit([0, 1, 2], [1, 3, 5]), (2, 1))
    cal = {"ns": [1.0, 0.0, 0.0], "c50_bh_db": (0.0, 0.0)}
    e = estimate({"nen_dfn_pct": 10.0, "c50_bh_db": 5.0}, cal)
    assert abs(e["snr"] - 9.542) < 1e-3 and flags(e) == {"snr": True, "nen": False, "vang": False}, e   # nền 10 % = SNR 9,54 dB; RT60 e^0 = 1,0
    assert flags(estimate({"nen_dfn_pct": 10.1, "c50_bh_db": 5.0}, cal))["nen"] and estimate({"c50_bh_db": None}, cal)["snr"] is None
    assert abs(late(0.0) - 50) < 1e-9 and abs(ns_truth("snr", 10.0) - 100 / 11) < 1e-9 and ns_truth("vang", 1.2) == 0.0
    assert fold("x") == zlib.crc32(b"x") % 5
    print("cham_dieu_kien selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(run()) if sys.argv[1:2] == ["run"] else selftest()
