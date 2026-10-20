# -*- coding: utf-8 -*-
"""So do tin hieu RE (numpy) dung lam 'dau hieu' cho ho so: san nhieu, do nghieng pho, bac san nhieu o mep, chenh trong / ngoai doan.
Moi so do co BANG THAM CHIEU tu tep THAT tu nhien (refbank.py) de ghi phan vi + trung vi. Am thanh 16 kHz mono float."""
from __future__ import annotations

import numpy as np

SR = 16000
WIN = 0.02


def frame_db(y: np.ndarray, win: float = WIN) -> np.ndarray:
    n = max(1, int(win * SR))
    k = len(y) // n
    if k == 0:
        return np.array([-120.0])
    f = y[:k * n].reshape(k, n).astype(np.float64)
    return 10 * np.log10((f ** 2).mean(1) + 1e-12)


def nf_db(y: np.ndarray) -> float:
    """San nhieu: muc (dB) cua 15 % khung yen nhat."""
    db = frame_db(y)
    return float(np.mean(np.sort(db)[:max(1, int(0.15 * len(db)))]))


def tilt_db(y: np.ndarray) -> float:
    """Do nghieng pho (dB): nang luong 4-8 kHz tru nang luong 0-2 kHz tren cac khung to nhat (50 %), pho trung binh."""
    n = 512
    k = len(y) // n
    if k < 2:
        return float("nan")
    fr = y[:k * n].reshape(k, n) * np.hanning(n)
    S = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    loud = np.argsort(S.sum(1))[k // 2:]
    f = np.fft.rfftfreq(n, 1 / SR)
    P = S[loud].mean(0) + 1e-12
    return float(10 * np.log10(P[(f >= 4000) & (f < 8000)].mean()) - 10 * np.log10(P[f < 2000].mean()))


def seg(y: np.ndarray, a: float, b: float) -> np.ndarray:
    return y[max(0, int(a * SR)):max(0, int(b * SR))]


def noise_step(y: np.ndarray, a: float, b: float, ctx: float = 1.0) -> float:
    """|bac san nhieu| (dB) giua ngay truoc mep a va ngay sau mep b (moi ben ctx giay)."""
    left, right = seg(y, a - ctx, a - 0.05), seg(y, b + 0.05, b + ctx)
    if len(left) < SR // 5 or len(right) < SR // 5:
        return float("nan")
    return abs(nf_db(right) - nf_db(left))


def inout_nf(y: np.ndarray, a: float, b: float, ctx: float = 2.0) -> float:
    """San nhieu TRONG doan tru san nhieu NGOAI doan (dB; hai ben moi ben ctx giay)."""
    inn = seg(y, a, b)
    out = np.concatenate([seg(y, a - ctx, a), seg(y, b, b + ctx)])
    if len(inn) < SR // 5 or len(out) < SR // 2:
        return float("nan")
    return nf_db(inn) - nf_db(out)


def inout_tilt(y: np.ndarray, a: float, b: float, ctx: float = 2.0) -> float:
    """|Do nghieng pho TRONG doan - NGOAI doan| (dB)."""
    inn = seg(y, a, b)
    out = np.concatenate([seg(y, a - ctx, a), seg(y, b, b + ctx)])
    if len(inn) < SR // 2 or len(out) < SR:
        return float("nan")
    return abs(tilt_db(inn) - tilt_db(out))
