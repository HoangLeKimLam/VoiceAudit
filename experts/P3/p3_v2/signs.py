# -*- coding: utf-8 -*-
"""SIGNS (P3 v2) — dau hieu DINH LUONG DO THAT tu .wav cho ho so K9 (P1): moi dong 02 >= 2 dau hieu co so + phan vi so voi kho tep THAT.

signs(wav, row, ctx=None) -> [{"name", "val", "unit", "giai_thich", "phan_vi_that"}]
  row: {level2, kind (global / segment), start, end, ...}; ctx tuy chon: {"ff": dac trung khung feats.frame_feats, "lang": "vi"/"en"}.
Dau hieu (tinh tren dac trung khung feats.py, khung 20 ms):
  khu on mot doan : buoc san nhieu trong - ngoai doan (dB, trung vi 16 dai, phan vi 10 theo thoi gian; ngoai = 2 s hai ben) · buoc san dai cao >= 4,5 kHz
  doi giong mot doan: lech cao do trong - ngoai doan (nua cung, trung vi khung huu thanh) · khoang cach MFCC trong - ngoai (don vi do lech chuan ngoai)
  khu on ca tep   : do sau khoang lang (dB: trung vi khung noi - phan vi 5) · do phang pho khoang lang (log, cang am cang "sach")
  doi giong ca tep : bang thong khung noi (kHz, RO99 trung vi) · do bat tuan hoan YIN khung huu thanh (trung vi)
phan_vi_that = phan vi (0-100) cua gia tri theo HUONG nghi van trong phan bo cung dau hieu tren tep that cung ngon ngu
  (kho signs_ref.json; dau hieu doan do tren doan ngau nhien 1,5 s cua tep that). Thieu kho -> None.
CLI: python experts/P3/p3_v2/signs.py build-ref     (-> experts/P3/p3_v2/signs_ref.json, tu .tmp_v6/cache/p3v2_ff)
     python experts/P3/p3_v2/signs.py selftest"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

from feats import COL, frame_feats, n_frames, read_wav  # noqa: E402

REF = os.path.join(HERE, "signs_ref.json")
DT = 0.02
# ten dau hieu -> (don vi, huong nghi van: +1 gia tri cang lon cang nghi, -1 cang nho cang nghi, 0 tri tuyet doi)
SIGNS = {
    "buoc_san_nhieu": ("dB", -1), "buoc_san_dai_cao": ("dB", -1), "lech_cao_do": ("nua cung", 0), "khoang_cach_mfcc": ("do lech chuan", 1),
    "do_sau_khoang_lang": ("dB", 1), "do_phang_khoang_lang": ("log", -1), "bang_thong_noi": ("kHz", -1), "bat_tuan_hoan": ("", 1),
}


def _inout(T: int, a: float, b: float, ctx_s: float = 2.0, gap: float = 0.1) -> tuple:
    t = (np.arange(T) + 0.5) * DT
    ins = (t >= a) & (t <= b)
    out = ((t >= a - ctx_s) & (t < a - gap)) | ((t > b + gap) & (t <= b + ctx_s))
    if out.sum() < 10:
        out = ~ins & ((t < a - gap) | (t > b + gap))
    return ins, out


def seg_values(ff: np.ndarray, a: float, b: float) -> dict:
    """Dau hieu doan [a, b] (giay) tren dac trung khung (ham thuan)."""
    ff = ff.astype(np.float64)
    ins, out = _inout(len(ff), a, b)
    if ins.sum() < 3 or out.sum() < 3:
        return {}
    B = ff[:, :16]
    fi, fo = np.quantile(B[ins], 0.1, axis=0), np.quantile(B[out], 0.1, axis=0)
    V = ff[:, COL["V"]] > 0.5
    st = ff[:, COL["F0"]]
    vi_, vo = ins & V, out & V
    M = ff[:, COL["M1"]:COL["M12"] + 1]
    return {"buoc_san_nhieu": float(np.median(fi - fo)), "buoc_san_dai_cao": float(np.median((fi - fo)[12:])),
            "lech_cao_do": float(np.median(st[vi_]) - np.median(st[vo])) if vi_.sum() >= 3 and vo.sum() >= 3 else 0.0,
            "khoang_cach_mfcc": float(np.linalg.norm(M[ins].mean(0) - M[out].mean(0)) / (M[out].std(0).mean() + 1e-6))}


def file_values(ff: np.ndarray) -> dict:
    """Dau hieu ca tep (ham thuan)."""
    ff = ff.astype(np.float64)
    E = ff[:, COL["E"]]
    lo, sp = E <= np.quantile(E, 0.25), E >= np.quantile(E, 0.6)
    V = ff[:, COL["V"]] > 0.5
    return {"do_sau_khoang_lang": float(np.median(E[sp]) - np.quantile(E, 0.05)), "do_phang_khoang_lang": float(ff[lo, COL["FLAT"]].mean()),
            "bang_thong_noi": float(np.median(ff[sp, COL["RO99"]])), "bat_tuan_hoan": float(np.median(ff[V, COL["AP"]])) if V.sum() > 3 else 0.5}


def percentile(val: float, ref: list, direction: int) -> float | None:
    """Phan vi theo huong nghi van (ham thuan): +1 -> % tep that co gia tri <= val; -1 -> % >= val; 0 -> theo |val|."""
    if not ref:
        return None
    r = np.asarray(ref, float)
    if direction == 0:
        r, val = np.abs(r), abs(val)
    p = np.mean(r <= val) if direction >= 0 else np.mean(r >= val)
    return round(100.0 * float(p), 1)


GIAI = {
    "buoc_san_nhieu": "Sàn nhiễu trong đoạn so với 2 s hai bên (âm = trong đoạn sạch hơn bất thường)",
    "buoc_san_dai_cao": "Như trên, riêng dải ≥ 4,5 kHz",
    "lech_cao_do": "Cao độ trung vị trong đoạn lệch so với hai bên (≥ 2 nửa cung là ngưỡng đổi giọng)",
    "khoang_cach_mfcc": "Âm sắc (MFCC) trong đoạn khác hai bên, tính theo độ lệch chuẩn của hai bên",
    "do_sau_khoang_lang": "Chênh mức khung nói với khoảng lặng sâu nhất (khử ồn mạnh làm khoảng lặng gần như câm)",
    "do_phang_khoang_lang": "Độ phẳng phổ khoảng lặng (nhiễu nền tự nhiên phẳng hơn; âm nhiều = nền bị gọt)",
    "bang_thong_noi": "Tần số chứa 99 % năng lượng khung nói (hạ giọng bằng lấy mẫu lại làm hụt dải cao)",
    "bat_tuan_hoan": "Độ bất tuần hoàn YIN của khung hữu thanh (bộ biến giọng làm hài âm kém sạch)",
}


def _ref() -> dict:
    try:
        return json.load(open(REF, encoding="utf-8"))
    except Exception:                                                    # noqa: BLE001
        return {}


def signs(wav: str, row: dict, ctx: dict | None = None) -> list:
    ctx = ctx or {}
    ff = ctx.get("ff")
    if ff is None:
        x = read_wav(wav)
        ff = frame_feats(x, n_frames(len(x)))
    lang = (ctx.get("lang") or "vi")[:2]
    l2, seg = row.get("level2", ""), row.get("kind") == "segment"
    if seg:
        v = seg_values(ff, float(row["start"]), float(row["end"]))
        keys = ("buoc_san_nhieu", "buoc_san_dai_cao") if "_nen_" in l2 else ("lech_cao_do", "khoang_cach_mfcc")
    else:
        v = file_values(ff)
        keys = ("do_sau_khoang_lang", "do_phang_khoang_lang") if "_nen_" in l2 else ("bang_thong_noi", "bat_tuan_hoan")
    ref = _ref().get(lang, {})
    out = []
    for k in keys:
        if k not in v:
            continue
        unit, d = SIGNS[k]
        out.append({"name": k, "val": round(v[k], 3), "unit": unit, "giai_thich": GIAI[k], "phan_vi_that": percentile(v[k], ref.get(k, []), d)})
    return out


def build_ref() -> None:
    """Kho phan vi: dau hieu tren tep THAT natvi / naten goc (doan ngau nhien 1,5 s cho dau hieu doan)."""
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    sys.path.insert(0, BASE)
    from v6.boundary import detect as D
    items = D.load_items(keep=("lang",))
    FF = os.path.join(BASE, ".tmp_v6", "cache", "p3v2_ff")
    rng = np.random.default_rng(0)
    ref = {"vi": {k: [] for k in SIGNS}, "en": {k: [] for k in SIGNS}}
    for i, it in items.items():
        if not it["real"] or it["cfg"] != "goc" or it["set"] not in ("v6natvi", "v6naten") or not os.path.exists(os.path.join(FF, i + ".npy")):
            continue
        lg = "en" if it["set"] == "v6naten" else "vi"
        ff = np.load(os.path.join(FF, i + ".npy"))
        for k, v in file_values(ff).items():
            ref[lg][k].append(round(v, 3))
        dur = len(ff) * DT
        if dur > 3:
            a = float(rng.uniform(0.5, dur - 2.0))
            for k, v in seg_values(ff, a, a + 1.5).items():
                ref[lg][k].append(round(v, 3))
    json.dump(ref, open(REF, "w", encoding="utf-8"))
    print("->", REF, {lg: len(ref[lg]["do_sau_khoang_lang"]) for lg in ref})


def selftest() -> None:
    rng = np.random.default_rng(1)
    sr = 16000
    t = np.arange(sr * 6) / sr
    x = 0.2 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > 0) + rng.normal(0, 0.01, len(t))
    a, b = int(2.0 * sr), int(4.0 * sr)
    x[a:b] = 0.2 * np.sin(2 * np.pi * 150 * t[a:b]) * (np.sin(2 * np.pi * 2 * t[a:b]) > 0) + rng.normal(0, 0.0005, b - a)  # doan "khu on"
    ff = frame_feats(x, n_frames(len(x)))
    v = seg_values(ff, 2.0, 4.0)
    assert v["buoc_san_nhieu"] < -15, v
    s = signs("", {"level2": "02_nen_cuc_bo", "kind": "segment", "start": 2.0, "end": 4.0}, {"ff": ff})
    assert len(s) >= 2 and all(isinstance(d["val"], float) for d in s)
    assert percentile(-20, [-1, -2, 0, 1], -1) == 100.0 and percentile(3, [-1, -2, 0, 1], -1) == 0.0 and percentile(5, [1, 2, 3, 4], 1) == 100.0
    print("signs selftest dat", s[0]["val"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "build-ref":
        build_ref()
    else:
        selftest()
