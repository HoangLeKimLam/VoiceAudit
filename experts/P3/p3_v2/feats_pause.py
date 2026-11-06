# -*- coding: utf-8 -*-
"""Dac trung CA TEP chi tiet tan so cho dau nen / giong (P3 v2): pho cong suat log 128 dai (62,5 Hz moi dai, 0-8 kHz) theo nhom khung
  lang (25 % nang luong thap nhat) / noi (30 % cao nhat):
  mean_lang (128) · std theo thoi gian khung lang (128: "musical noise") · mean_noi - mean_lang (128: SNR tung dai)
  · |delta theo thoi gian| trung binh khung lang (128) · phan vi 10 theo thoi gian moi dai tru phan vi 50 (128)
  + XLS-R PCA-64 tren khung lang / noi: mean_lang, std_lang, mean_noi - mean_lang, std_lang / std_noi (256).
Ra: .tmp_v6/cache/p3v2_pause.pkl {"ids": [...], "X": (N, 896) float32}.
CLI: python experts/P3/p3_v2/feats_pause.py [--jobs 8]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
FR = os.path.join(V4, "xlsr_frames")


def pause_feats(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Tin hieu float 16 kHz + khung XLS-R (T, 64) -> vector 896 (ham thuan)."""
    from feats import NFFT, frames_of
    T = len(z)
    fr = frames_of(x.astype(np.float64), T, NFFT) * np.hanning(NFFT)[None, :]
    P = np.abs(np.fft.rfft(fr, NFFT)) ** 2 + 1e-10
    L = 10 * np.log10(P[:, 1:257].reshape(T, 128, 2).mean(2))            # (T, 128)
    E = 10 * np.log10(P[:, 3:].sum(1))
    lo = E <= np.quantile(E, 0.25)
    hi = E >= np.quantile(E, 0.7)
    if lo.sum() < 3:
        lo = np.ones(T, bool)
    if hi.sum() < 3:
        hi = np.ones(T, bool)
    ml, mh = L[lo].mean(0), L[hi].mean(0)
    dl = np.abs(np.diff(L, axis=0))[lo[1:]].mean(0) if lo[1:].sum() > 1 else np.zeros(128)
    q10 = np.quantile(L, 0.1, axis=0) - np.quantile(L, 0.5, axis=0)
    z = z.astype(np.float64)
    zl, zh = z[lo[:len(z)]], z[hi[:len(z)]]
    out = np.r_[ml, L[lo].std(0), mh - ml, dl, q10, zl.mean(0), zl.std(0), zh.mean(0) - zl.mean(0), zl.std(0) / (zh.std(0) + 1e-3)]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=-100.0).astype(np.float32)


def job(args):
    i, wav = args
    from feats import read_wav
    try:
        return i, pause_feats(read_wav(wav), np.load(os.path.join(FR, i + ".npy")))
    except Exception as e:                                              # noqa: BLE001
        return i, None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    from multiprocessing import Pool
    from feats import todo
    from v6.boundary import detect as D
    items = D.load_items(keep=("level2",))
    tasks = [(i, p) for i, p in todo(["natvi", "naten", "benign"], "all") if i in items and os.path.exists(os.path.join(FR, i + ".npy"))]
    print(len(tasks), "tep", flush=True)
    res = {}
    with Pool(a.jobs) as pool:
        for i, v in pool.imap_unordered(job, tasks, chunksize=16):
            if v is not None:
                res[i] = v
    ids = sorted(res)
    pickle.dump({"ids": ids, "X": np.stack([res[i] for i in ids])}, open(os.path.join(CACHE, "p3v2_pause.pkl"), "wb"))
    print("->", len(ids))


def selftest() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, 16000 * 3)
    x[16000:32000] += 0.3 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    v = pause_feats(x, rng.normal(0, 1, (149, 64)))
    assert v.shape == (896,) and np.isfinite(v).all()
    assert v[256 + 3] > 10                                               # dai 220 Hz: noi >> lang
    print("feats_pause selftest dat")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
