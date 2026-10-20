# -*- coding: utf-8 -*-
"""N2 / K1x - DO CAC UOC LUONG TIN HIEU cua cac dong Constraint con thieu, tren tep tu nhien that + dieu kien ap CO KIEM SOAT:
  snr    tieng dong that SNR 5 / 8 / 10 / 12 / 15 / 25 dB       -> WADA, uoc luong theo ASR cua k1.py
  vang   RT60 0,3 / 0,6 / 1,0 / 1,5 / 2,0 s                      -> RT60 mu, anh huong len SNR
  nhac   nhac nen tong hop 5 / 10 / 15 / 20 / 30 / 50 % nang luong  (Constraint <= 10 %)  -> AST
  tv     giong nguoi khac 8 / 15 / 30 % nang luong               -> AST
  chong  hai nguoi noi chong 0 / 5 / 10 / 20 / 40 % thoi gian   -> pyannote segmentation 3.0
  ngonngu tep goc vi / en                                         -> whisper-small
Moi tep co so duoc ap dieu kien TRONG BO NHO (khong ghi wav), do cac dac trung, luu .tmp_v6/results/k1x_signal.json.
  python -m v6.k1x.eval_signal [--per-lang 20] [--seed 5] [--axes snr vang nhac tv chong ngonngu]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.environ.setdefault("NAT_LANG", "cover")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from vaudit import k1, k1x  # noqa: E402
from v6.coverage import build_axes as BA  # noqa: E402

SR = 16000
RES = os.path.join(BASE, ".tmp_v6", "results")
OUT = os.path.join(RES, "k1x_signal.json")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
CONDS = {"snr": [("snr", v) for v in (5, 8, 10, 12, 15, 25)], "vang": [("vang", v) for v in (0.3, 0.6, 1.0, 1.5, 2.0)],
         "nhac": [("nhac", v) for v in (5, 10, 15, 20, 30, 50)], "tv": [("tv", v) for v in (8, 15, 30)]}
CHONG = (0, 5, 10, 20, 40)


def snr_asr(y: np.ndarray) -> float | None:
    """Uoc luong SNR theo ASR: giong het k1.analyse_channel (vung ASR nghe ra chu vs vung khong co)."""
    db = k1.frame_db(y)
    sp_e, _ = k1.speech_mask(db)
    sp_a = k1.asr_speech_mask(y)
    k = min(len(sp_e), len(sp_a))
    sp = np.zeros(len(db), bool)
    sp[:k] = sp_e[:k] & sp_a[:k]
    nz = np.zeros(len(db), bool)
    nz[:k] = ~sp_a[:k]
    pw = 10 ** (db / 10)
    if not sp.any():
        return 0.0
    if nz.sum() * 0.03 < 1.0:
        return None
    pn = pw[nz].mean()
    return float(10 * np.log10(max(pw[sp].mean() - pn, 1e-12) / (pn + 1e-12)))


def bases(per_lang: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for lang in ("natvi", "naten"):
        man = [e for e in json.load(open(os.path.join(BASE, "data_labels", "v6", lang, "manifest.json"), encoding="utf-8"))
               if e["cfg"] == "goc" and e["kind"] == "nguyen_ban" and not e["meta"].get("ngoai_pham_vi") and not e["meta"].get("lockbox")
               and e["meta"].get("k1", {}).get("decision") not in ("04", "tu_choi", "loi_k1") and 5 <= e["meta"].get("k1", {}).get("measures", {}).get("do_dai_s", 8) <= 20]
        out += rng.sample(man, min(per_lang, len(man)))
    return out


def load(e: dict) -> np.ndarray:
    return sf.read(os.path.join(V4, e["wav"]), dtype="float64")[0]


def common(y: np.ndarray, with_asr: bool) -> dict:
    y32 = y.astype(np.float32)
    r = {"wada": k1x.wada_snr(y32), **k1x.rt60_blind(y32), **k1x.overlap_share(y32), **k1x.audio_events(y32)}
    if with_asr:
        r["snr_asr"] = snr_asr(y32)
    return r


def overlap_mix(a: np.ndarray, b: np.ndarray, share: float) -> tuple:
    """Dat b de ti le (thoi gian chong) / (thoi gian hop) = share; tra (am thanh, ti le that)."""
    def trim(y, n):
        m = BA.speech_frames(y)
        i = np.where(m)[0]
        s, e = (i[0] * 160, i[-1] * 160 + 320) if len(i) else (0, len(y))
        return y[s:min(e, s + n * SR)]
    a, b = trim(a, 6), trim(b, 6)
    b = b * BA.speech_rms(a) / (BA.speech_rms(b) + 1e-9)
    L = int(share / (1 + share) * (len(a) + len(b))) if share > 0 else 0
    L = min(L, len(a) - 1, len(b) - 1)
    off = len(a) - L if share > 0 else len(a) + int(0.3 * SR)
    y = np.zeros(max(len(a), off + len(b)))
    y[:len(a)] += a
    y[off:off + len(b)] += b
    return y, L / len(y)


def run(axes: list, per_lang: int, seed: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    base = bases(per_lang, seed)
    pool = [load(e) for e in base[:30]]
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else []
    done = {(r["axis"], r["v"], r["base"]) for r in out}
    t0 = time.time()
    print(f"{len(base)} tep co so; truc {axes}", flush=True)
    for bi, e in enumerate(base):
        y0 = load(e)
        todo = [("clean", 0)] + [c for a in axes if a in CONDS for c in CONDS[a]]
        for name, v in todo:
            if (name, v, e["id"]) in done:
                continue
            y = y0 if name == "clean" else BA.apply(name, v, y0.copy(), [], pool, np.random.default_rng(abs(hash((e["id"], name, v))) % (2 ** 32)))[0]
            rec = {"axis": name, "v": v, "base": e["id"], "lang": e["meta"].get("lang"), "dur": len(y) / SR, **common(y, with_asr=True)}
            if name == "clean" and "ngonngu" in axes:
                rec["lang_probs"] = k1x.language_probs(y.astype(np.float32))
            out.append(rec)
        if "chong" in axes:
            b2 = base[(bi + 7) % len(base)]
            for s in CHONG:
                if ("chong", s, e["id"]) in done:
                    continue
                y, true = overlap_mix(y0, load(b2), s / 100)
                out.append({"axis": "chong", "v": s, "base": e["id"], "true_share": true, **k1x.overlap_share(y.astype(np.float32))})
        json.dump(out, open(OUT + ".tmp", "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(OUT + ".tmp", OUT)
        print(f"  {bi + 1}/{len(base)} tep co so, {len(out)} ket qua, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--axes", nargs="+", default=["snr", "vang", "nhac", "tv", "chong", "ngonngu"])
    a = ap.parse_args()
    run(a.axes, a.per_lang, a.seed)
