"""Tệp xác nhận hàng cứng cho p1_tunhien_v1 (luật chốt trước: NHAT_KY 2.16 «Sửa dữ liệu bước 1» ý c; Huấn chốt cách đếm 24/11).
Một dòng / tệp accept: K1 04 vì hàng cứng nào + phép đo độc lập + xác nhận; ồn: snr_nghi (do_vang.py) và on_duoi10.
Không sửa cases.json. Ra: p1_tunhien_v1/_kiem/xac_nhan_hang_cung.jsonl

  .venv/Scripts/python.exe experts/P1/build_cond/xac_nhan.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6, jl  # noqa: E402
from cham_tram import cases_of  # noqa: E402

OUT = os.path.join(V6, NAT, "_kiem", "xac_nhan_hang_cung.jsonl")
BANG_HEP_DB, CAT_DINH_PCT, SNR_LOI = -40.0, 5.0, -60.0


def bang_hep_db(y: np.ndarray, sr: int) -> float:
    """10·log10(E[6–7,8 kHz] / E[0,3–3,4 kHz]) trên phổ FFT cả tệp."""
    p = np.abs(np.fft.rfft(y)) ** 2
    f = np.fft.rfftfreq(len(y), 1 / sr)
    hi, lo = p[(f >= 6000) & (f <= 7800)].sum(), p[(f >= 300) & (f <= 3400)].sum()
    return float(10 * np.log10(max(hi, 1e-30) / max(lo, 1e-30)))


def cat_dinh_pct(y: np.ndarray) -> float:
    pk = float(np.max(np.abs(y)))
    return 100 * float(np.mean(np.abs(y) >= 0.99 * pk)) if pk > 0 else 0.0


def nen_khoi(tram: dict | None, music) -> dict:
    """Hàng «âm phi tiếng nói ≤ 10 % năng lượng» (Huấn 24/11: làm đúng Constraint): phần dư DeepFilterNet / năng lượng tệp."""
    a, full = (tram or {}).get("nen_a"), (tram or {}).get("nen_full")
    return {"nen_a": a, "nen_full": full, "nen_tren10_a": (a > 10) if a is not None else None,
            "nen_tren10_full": (full > 10) if full is not None else None, "nhac_nen_audioset": music,
            "cach": "do_tram (DeepFilterNet, phần dư / tệp); nhac_nen_audioset chỉ tham khảo",
            "chua_kiem_dinh": True}  # 24/11: phần dư thô gồm cả vang + lỗi tách (tệp Huấn nghe «khong» có nen_a trung vị ~9,5 %) → chưa dùng để xác nhận tới khi kiểm trên p1_dieukien_v3


def mot(f: str, reasons: list, snr_nghi) -> dict:
    out = {"file": f, "k1_04": [r for r in reasons], "xac_nhan": {}}
    y = None
    for r in reasons:
        if "quá nhỏ" in r:
            out["xac_nhan"]["qua_nho"] = {"cach": "công thức K1 (Huấn chốt 24/11)", "dung": True}
        elif "băng hẹp" in r or "cắt đỉnh nặng" in r:
            if y is None:
                y, sr = sf.read(os.path.join(V6, NAT, f), dtype="float64")
            if "băng hẹp" in r:
                v = bang_hep_db(y, sr)
                out["xac_nhan"]["bang_hep"] = {"do": round(v, 1), "nguong": f"≤ {BANG_HEP_DB} dB", "dung": v <= BANG_HEP_DB}
            else:
                v = cat_dinh_pct(y)
                out["xac_nhan"]["cat_dinh"] = {"do": round(v, 2), "nguong": f"> {CAT_DINH_PCT} %", "dung": v > CAT_DINH_PCT}
    ok = snr_nghi is not None and snr_nghi > SNR_LOI
    out["on"] = {"snr_nghi": snr_nghi, "on_duoi10": (snr_nghi < 10) if ok else None, "cach": "do_vang.snr_nghi (DeepFilterNet, khung nghỉ)"}
    return out


def run() -> None:
    cases = cases_of(NAT)
    acc = {f for f, c in cases.items() if c["mong_doi"] == "accept"}
    k1 = {r["file"]: r for r in jl(os.path.join(V6, NAT, "_kiem", "k1_ket_qua.jsonl")) if r["file"] in acc}
    vang = {r["file"]: r for r in jl(os.path.join(V6, NAT, "_kiem", "dieu_kien_vang.jsonl"))}
    tram = {r["file"]: r for r in jl(os.path.join(V6, NAT, "_kiem", "dieu_kien_tram.jsonl"))}
    rows = []
    for f in sorted(acc):
        reasons = k1[f]["reasons"] if f in k1 and k1[f]["decision"] == "04" else []
        ev = (k1[f]["measures"]["kenh"][0].get("su_kien_am") or {}) if f in k1 else {}
        rows.append({**mot(f, reasons, vang.get(f, {}).get("snr_nghi")), "nen": nen_khoi(tram.get(f), ev.get("music_mean"))})
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"{OUT}: {len(rows)} tệp accept (thiếu K1: {sum(f not in k1 for f in acc)})")
    for h in ("qua_nho", "bang_hep", "cat_dinh"):
        x = [r["xac_nhan"][h]["dung"] for r in rows if h in r["xac_nhan"]]
        print(f"  {h}: K1 04 {len(x)} · đo độc lập xác nhận {sum(x)}")
    o = [r["on"]["on_duoi10"] for r in rows]
    print(f"  ồn (snr_nghi < 10 dB): {sum(v is True for v in o)} · ≥ 10: {sum(v is False for v in o)} · không đo được: {sum(v is None for v in o)}")
    for k in ("nen_tren10_a", "nen_tren10_full"):
        v = [r["nen"][k] for r in rows]
        print(f"  nền {k}: True {sum(x is True for x in v)} · False {sum(x is False for x in v)} · không đo {sum(x is None for x in v)}")
    print(f"  K1 04 lý do khác (không phải hàng cứng đo được): {sum(1 for r in rows if r['k1_04'] and not r['xac_nhan'])}")


def selftest() -> None:
    sr = 16000
    t = np.arange(sr) / sr
    rng = np.random.default_rng(0)
    rong = rng.standard_normal(sr)
    hep = np.sin(2 * np.pi * 1000 * t)
    assert bang_hep_db(rong, sr) > -5 and bang_hep_db(hep, sr) < -60
    c = np.clip(np.sin(2 * np.pi * 200 * t) * 2, -1, 1)
    assert cat_dinh_pct(c) > 30 and cat_dinh_pct(rong * 0.1) < 5   # sin thuần có ~9 % mẫu ≥ 0,99 đỉnh nên không dùng làm ca «không cắt»
    r = mot("x.wav", [], -173.2)
    assert r["on"]["on_duoi10"] is None and mot("x.wav", [], 8.0)["on"]["on_duoi10"] is True
    assert mot("x.wav", ["“quá nhỏ”"], 20.0)["xac_nhan"]["qua_nho"]["dung"] is True
    n = nen_khoi({"nen_a": 12.0, "nen_full": 8.0}, 0.7)
    assert n["nen_tren10_a"] is True and n["nen_tren10_full"] is False and nen_khoi(None, None)["nen_tren10_a"] is None
    print("xac_nhan selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
