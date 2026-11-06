# -*- coding: utf-8 -*-
"""Dac trung VAT LY theo khung 20 ms tu .wav (16 kHz) cho P3 v2 — khop dung luoi khung XLS-R (khung t ~ mau [320 t, 320 t + 400)).

Moi khung (cot, xem FCOLS):
  B0..B15   nang luong 16 dai mel 0-8 kHz (dB, cua so Hann 512 mau tam tai 320 t + 200)
  E         nang luong toan dai (dB) · FLAT do phang pho (log) · CEN tam pho (kHz) · RO95 / RO99 tan so cat 95 / 99 % nang luong (kHz)
  H4 H6 H7  ti le nang luong tren 4 / 6 / 7 kHz so voi 0,3-4 kHz (dB)
  F0        cao do YIN (nua cung so voi 100 Hz; 0 khi khong huu thanh) · AP do bat tuan hoan YIN (0..1) · V co huu thanh
  M1..M12   MFCC 1-12 (bo M0)
  HOLE      ti le o pho (0,3-8 kHz) thap hon trung vi dai cua khung do 30 dB (lo pho do khu on cat sau)
Ra: .tmp_v6/cache/p3v2_ff/<id>.npy (T, 40) float16 ; T = so khung XLS-R cua tep (neu co) hoac (N - 400) // 320 + 1.
CLI: python experts/P3/p3_v2/feats.py run [--sets natvi naten benign] [--cfg all|goc] [--jobs 20]
     python experts/P3/p3_v2/feats.py selftest
Nhe: chi numpy / scipy; ~30-60 ms moi tep 10 s."""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

V4 = os.path.join(BASE, "data_labels", "realistic_v4")
FR = os.path.join(V4, "xlsr_frames")
OUT = os.path.join(BASE, ".tmp_v6", "cache", "p3v2_ff")
SR, HOP, NFFT = 16000, 320, 512
FCOLS = [f"B{k}" for k in range(16)] + ["E", "FLAT", "CEN", "RO95", "RO99", "H4", "H6", "H7", "F0", "AP", "V"] + [f"M{k}" for k in range(1, 13)] + ["HOLE"]
COL = {c: n for n, c in enumerate(FCOLS)}
EPS = 1e-10


def n_frames(n_samples: int) -> int:
    return max(1, (n_samples - 400) // HOP + 1)


def mel_fb(n_mels: int = 16, nfft: int = NFFT, sr: int = SR, fmin: float = 0.0, fmax: float = 8000.0) -> np.ndarray:
    """Bo loc tam giac mel (n_mels, nfft // 2 + 1) (ham thuan)."""
    m = lambda f: 2595 * np.log10(1 + f / 700)                         # noqa: E731
    im = lambda x: 700 * (10 ** (x / 2595) - 1)                       # noqa: E731
    pts = im(np.linspace(m(fmin), m(fmax), n_mels + 2))
    fr = np.linspace(0, sr / 2, nfft // 2 + 1)
    fb = np.zeros((n_mels, len(fr)))
    for k in range(n_mels):
        lo, c, hi = pts[k], pts[k + 1], pts[k + 2]
        fb[k] = np.clip(np.minimum((fr - lo) / (c - lo + EPS), (hi - fr) / (hi - c + EPS)), 0, None)
    return fb


FB16 = mel_fb(16)
FB40 = mel_fb(40)
DCT = np.cos(np.pi / 40 * (np.arange(40)[None, :] + 0.5) * np.arange(13)[:, None])


def frames_of(x: np.ndarray, T: int, L: int) -> np.ndarray:
    """Cat T khung dai L tam tai 320 t + 200 (dem 0 hai dau) (ham thuan)."""
    pad = L
    xp = np.pad(x, (pad, pad + L))
    starts = HOP * np.arange(T) + 200 - L // 2 + pad
    idx = starts[:, None] + np.arange(L)[None, :]
    return xp[idx]


def yin(x: np.ndarray, T: int, tmin: int = 32, tmax: int = 320, W: int = 512, th: float = 0.15) -> tuple:
    """YIN vector hoa: (f0 Hz (0 neu khong huu thanh), do bat tuan hoan d'(tau*)) (ham thuan)."""
    L = W + tmax + 2
    fr = frames_of(x, T, L).astype(np.float64)
    fr -= fr.mean(1, keepdims=True)
    nf = 1 << int(np.ceil(np.log2(L + W)))
    A = np.fft.rfft(fr, nf)
    Bw = np.fft.rfft(fr[:, :W], nf)
    r = np.fft.irfft(A * np.conj(Bw), nf)[:, :tmax + 1]            # r(tau) = sum_j x_j x_{j+tau}, j < W
    cs = np.concatenate([np.zeros((T, 1)), np.cumsum(fr ** 2, 1)], 1)
    e0 = cs[:, W][:, None]
    taus = np.arange(tmax + 1)
    et = cs[:, taus + W] - cs[:, taus]
    d = np.maximum(e0 + et - 2 * r, 0)
    d[:, 0] = 0
    cm = np.cumsum(d[:, 1:], 1) / np.arange(1, tmax + 1)
    dn = np.ones_like(d)
    dn[:, 1:] = d[:, 1:] / (cm + EPS)
    sub = dn[:, tmin:tmax]
    below = sub < th
    first = np.where(below.any(1), below.argmax(1), sub.argmin(1))
    # di toi cuc tieu dia phuong sau diem dau tien duoi nguong
    k = first.copy()
    for _ in range(40):
        nxt = np.minimum(k + 1, sub.shape[1] - 1)
        mv = sub[np.arange(T), nxt] < sub[np.arange(T), k]
        if not mv.any():
            break
        k = np.where(mv, nxt, k)
    tau = k + tmin
    a = dn[np.arange(T), np.clip(tau - 1, 1, tmax)]
    b = dn[np.arange(T), tau]
    c = dn[np.arange(T), np.clip(tau + 1, 1, tmax)]
    den = a - 2 * b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-9, 0.5 * (a - c) / den, 0.0)
    tf = tau + np.clip(off, -1, 1)
    ap = np.clip(b, 0, 1)
    f0 = np.where(ap < 0.3, SR / tf, 0.0)
    return f0, ap


def frame_feats(x: np.ndarray, T: int) -> np.ndarray:
    """(T, 40) dac trung khung tu tin hieu float [-1, 1] (ham thuan)."""
    x = x.astype(np.float64)
    fr = frames_of(x, T, NFFT) * np.hanning(NFFT)[None, :]
    P = np.abs(np.fft.rfft(fr, NFFT)) ** 2 + EPS                       # (T, 257)
    f = np.linspace(0, SR / 2, NFFT // 2 + 1)
    B = 10 * np.log10(P @ FB16.T + EPS)
    tot = P[:, 3:].sum(1)
    E = 10 * np.log10(tot + EPS)
    lp = np.log(P[:, 3:])
    flat = lp.mean(1) - np.log(P[:, 3:].mean(1))
    cen = (P[:, 3:] * f[3:]).sum(1) / tot / 1000
    cum = np.cumsum(P[:, 3:], 1) / tot[:, None]
    ro95 = f[3:][np.minimum((cum < 0.95).sum(1), len(f) - 4)] / 1000
    ro99 = f[3:][np.minimum((cum < 0.99).sum(1), len(f) - 4)] / 1000
    lo = P[:, (f >= 300) & (f < 4000)].sum(1) + EPS
    h = [10 * np.log10(P[:, f >= fc].sum(1) / lo + EPS) for fc in (4000, 6000, 7000)]
    f0, ap = yin(x, T)
    st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1) / 100.0), 0.0)
    v = (f0 > 0).astype(np.float64)
    mel = np.log(P @ FB40.T + EPS)
    mf = mel @ DCT.T
    band = P[:, (f >= 300)]
    med = np.median(band, 1, keepdims=True)
    hole = (band < med * 1e-3).mean(1)
    out = np.column_stack([B, E, flat, cen, ro95, ro99, *h, st, ap, v, mf[:, 1:13], hole])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=-100.0).astype(np.float16)


def read_wav(p: str) -> np.ndarray:
    from scipy.io import wavfile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, x = wavfile.read(p)
    if x.ndim > 1:
        x = x.mean(1)
    if x.dtype.kind == "i":
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x.astype(np.float32)


def job(args: tuple) -> str:
    i, wav = args
    o = os.path.join(OUT, i + ".npy")
    if os.path.exists(o):
        return "co"
    try:
        x = read_wav(wav)
        fz = os.path.join(FR, i + ".npy")
        T = np.load(fz, mmap_mode="r").shape[0] if os.path.exists(fz) else n_frames(len(x))
        np.save(o, frame_feats(x, T))
        return "ok"
    except Exception as e:                                              # noqa: BLE001
        return f"loi {i}: {e}"


def todo(sets: list, cfg: str) -> list:
    out = []
    for s in sets:
        for e in json.load(open(os.path.join(BASE, "data_labels", "v6", s, "manifest.json"), encoding="utf-8")):
            if cfg == "goc" and e["cfg"] != "goc":
                continue
            p = os.path.normpath(os.path.join(V4, e["wav"]))
            if os.path.exists(p):
                out.append((e["id"], p))
    return out


def run(sets: list, cfg: str, jobs: int) -> None:
    from collections import Counter
    from multiprocessing import Pool
    os.makedirs(OUT, exist_ok=True)
    tasks = todo(sets, cfg)
    print(f"{len(tasks)} tep", flush=True)
    c = Counter()
    with Pool(jobs) as pool:
        for n, r in enumerate(pool.imap_unordered(job, tasks, chunksize=8)):
            c[r if not r.startswith("loi") else "loi"] += 1
            if r.startswith("loi") and c["loi"] <= 5:
                print(r, flush=True)
            if (n + 1) % 2000 == 0:
                print(f"  {n + 1}/{len(tasks)} {dict(c)}", flush=True)
    print(dict(c))


def selftest() -> None:
    t = np.arange(SR * 2) / SR
    x = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.3 * np.sin(2 * np.pi * 400 * t)
    T = n_frames(len(x))
    F = frame_feats(x, T).astype(np.float32)
    assert F.shape == (T, len(FCOLS)) == (99, 40)
    f0 = 100 * 2 ** (F[10:-10, COL["F0"]] / 12)
    assert np.all(np.abs(f0 - 200) < 4), f0[:5]
    assert np.all(F[10:-10, COL["V"]] == 1)
    rng = np.random.default_rng(0)
    nz = rng.normal(0, 0.05, SR)
    G = frame_feats(nz, n_frames(len(nz))).astype(np.float32)
    assert G[5:-5, COL["FLAT"]].mean() > F[10:-10, COL["FLAT"]].mean()          # nhieu trang phang hon hai hoa
    assert G[5:-5, COL["V"]].mean() < 0.2
    assert abs(mel_fb(16).sum(0)[50] - 1) < 0.2
    print("feats selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("cmd", choices=["run"])
        ap.add_argument("--sets", nargs="+", default=["natvi", "naten", "benign"])
        ap.add_argument("--cfg", default="all", choices=["all", "goc"])
        ap.add_argument("--jobs", type=int, default=20)
        a = ap.parse_args()
        run(a.sets, a.cfg, a.jobs)
