# -*- coding: utf-8 -*-
"""KHO THAM CHIEU CHO HO SO v9: phan vi (0..100) cua (a) diem cac chuyen gia tren tep THAT tu nhien (tu thresholds_v9<op>.json, muc 'refs') va
(b) cac so do tin hieu (signals.py) tren tep THAT tu nhien (tinh o day: moi tep goc that, vi + en, 3 doan ngau nhien dai 0,8-4 s).
Moi 'dau hieu' trong ho so = (gia tri, trung vi that, phan vi) co ghi NGUON kho. Chi dung tep that cua cac bo tu nhien, goc, trong pham vi.
  python -m v6.product.refbank build [--n 400]  -> .tmp_v6/results/refbank_signals.json"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from v6.product import signals as S  # noqa: E402

RES = os.path.join(BASE, ".tmp_v6", "results")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
PATH = os.path.join(RES, "refbank_signals.json")


def build(n: int, seed: int = 4) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(seed)
    pool = {"vi": [], "en": []}
    for lang, short in (("vi", "natvi"), ("en", "naten")):
        man = [e for e in json.load(open(os.path.join(BASE, "data_labels", "v6", short, "manifest.json"), encoding="utf-8"))
               if e["cfg"] == "goc" and e["kind"] == "nguyen_ban" and not e["meta"].get("ngoai_pham_vi") and not e["meta"].get("lockbox")
               and e["meta"].get("k1", {}).get("decision") in ("nhan", "canh_bao")]
        pool[lang] = rng.sample(man, min(n // 2, len(man)))
    acc = {lg: {"nf_file": [], "tilt_file": [], "noise_step": [], "inout_nf": [], "inout_tilt": []} for lg in pool}
    for lg, es in pool.items():
        for k, e in enumerate(es):
            y, _ = sf.read(os.path.join(V4, e["wav"]), dtype="float32")
            dur = len(y) / S.SR
            acc[lg]["nf_file"].append(S.nf_db(y))
            acc[lg]["tilt_file"].append(S.tilt_db(y))
            for _ in range(3):
                L = rng.uniform(0.8, 4.0)
                if dur < L + 4.0:
                    continue
                a = rng.uniform(1.5, dur - L - 1.5)
                for key, v in (("noise_step", S.noise_step(y, a, a + L)), ("inout_nf", S.inout_nf(y, a, a + L)), ("inout_tilt", S.inout_tilt(y, a, a + L))):
                    if np.isfinite(v):
                        acc[lg][key].append(v)
            if (k + 1) % 50 == 0:
                print(f"  {lg} {k + 1}/{len(es)}", flush=True)
    out = {lg: {k: [round(float(x), 4) for x in np.percentile(v, np.arange(0, 101))] for k, v in d.items() if len(v) > 20} for lg, d in acc.items()}
    json.dump({"nguon": f"{n} tep that tu nhien (vi + en), 3 doan ngau nhien moi tep", "tables": out}, open(PATH, "w"), indent=1)
    print({lg: {k: round(d[50], 3) for k, d in t.items()} for lg, t in out.items()})


def rank(table: list, v: float) -> dict:
    """phan vi cua v trong bang 101 diem phan vi: duoi = ti le tep that <= v, tren = ti le tep that >= v."""
    t = np.asarray(table, float)
    return {"trung_vi": float(t[50]), "duoi": float(np.searchsorted(t, v, "right") / len(t)), "tren": float(1 - np.searchsorted(t, v, "left") / len(t))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build"])
    ap.add_argument("--n", type=int, default=400)
    build(ap.parse_args().n)
