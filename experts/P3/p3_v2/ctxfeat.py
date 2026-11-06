# -*- coding: utf-8 -*-
"""Dac trung NGU CANH theo khung cho P3 v2 (ham thuan, khong doc dia ngoai load_file): tu dac trung vat ly khung (feats.py, (T, 40)) va khung
XLS-R PCA-64 (T, 64) -> (T, D) gom:
  - tho (40) · san nhieu cuc bo (min 25 / 51 khung) tru san ca tep (phan vi 10) theo 16 dai · trung binh dai cuc bo tru trung binh tep
  - nang luong: min / max / trung binh cuc bo tru phan vi tep, dai dong cuc bo · F0 cuc bo (khung huu thanh) tru trung vi tep, ti le huu thanh
  - MFCC / HOLE / FLAT / RO99 / H6 / H7 cuc bo tru trung binh tep · buoc nhay phai-trai (15 khung) cua san nhieu, F0, MFCC, XLS-R
  - XLS-R: trung binh cuc bo 25 khung tru trung binh tep (64) · boi canh ca tep (san 16 dai, dai dong, F0 trung vi, HOLE / FLAT tep)
Ten cot: names()."""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feats import COL, FCOLS  # noqa: E402

W = 25          # 0,5 s
W2 = 51         # 1 s
J = 15          # nua cua so buoc nhay


def _lr_jump(x: np.ndarray, k: int = J) -> np.ndarray:
    """|trung binh k khung ben phai - k khung ben trai| tai moi khung; x (T,) hoac (T, d) -> (T,) (chuan L2 khi nhieu chieu) (ham thuan)."""
    x2 = x if x.ndim == 2 else x[:, None]
    T = len(x2)
    c = np.vstack([np.zeros((1, x2.shape[1])), np.cumsum(x2, 0)])
    t = np.arange(T)
    lo, hi = np.clip(t - k, 0, T), np.clip(t + k, 0, T)
    left = (c[t] - c[lo]) / np.maximum(t - lo, 1)[:, None]
    right = (c[hi] - c[t]) / np.maximum(hi - t, 1)[:, None]
    d = right - left
    return np.sqrt((d ** 2).sum(1)) if x.ndim == 2 else np.abs(d[:, 0])


def _masked_mean(x: np.ndarray, m: np.ndarray, w: int) -> np.ndarray:
    num = uniform_filter1d(x * m, w, axis=0, mode="nearest")
    den = uniform_filter1d(m.astype(np.float64), w, axis=0, mode="nearest")
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan)


def names() -> list:
    n = list(FCOLS)
    n += [f"fl25_{k}" for k in range(16)] + [f"fl51_{k}" for k in range(16)] + [f"mB25_{k}" for k in range(16)]
    n += ["Emin25", "Emax25", "Emean25", "Edyn25", "Emin51", "f0loc", "vfrac", "aploc"]
    n += [f"mM25_{k}" for k in range(12)] + ["hole25", "flat25", "ro99_25", "h6_25", "h7_25", "flq25"]
    n += ["j_floor", "j_f0", "j_mfcc", "j_z", "j_E", "j_h7"]
    n += [f"zl{k}" for k in range(64)]
    n += [f"F_floor{k}" for k in range(16)] + ["F_dyn", "F_f0", "F_hole", "F_flat", "F_E90", "F_vfrac", "F_dur"]
    return n


def ctx_feats(ff: np.ndarray, z: np.ndarray) -> np.ndarray:
    """(T, 40) + (T, 64) -> (T, len(names())) float32 (ham thuan)."""
    ff = ff.astype(np.float64)
    z = z.astype(np.float64)
    T = min(len(ff), len(z))
    ff, z = ff[:T], z[:T]
    B = ff[:, :16]
    E = ff[:, COL["E"]]
    st, V = ff[:, COL["F0"]], ff[:, COL["V"]] > 0.5
    floorF = np.quantile(B, 0.1, axis=0)
    e10, e90 = np.quantile(E, 0.1), np.quantile(E, 0.9)
    fl25 = minimum_filter1d(B, W, axis=0, mode="nearest") - floorF
    fl51 = minimum_filter1d(B, W2, axis=0, mode="nearest") - floorF
    mB = uniform_filter1d(B, W, axis=0, mode="nearest") - B.mean(0)
    emin = minimum_filter1d(E, W, mode="nearest")
    emax = maximum_filter1d(E, W, mode="nearest")
    f0med = np.median(st[V]) if V.sum() > 5 else 0.0
    f0loc = np.nan_to_num(_masked_mean(st, V, W) - f0med, nan=0.0)
    ap = ff[:, COL["AP"]]
    aploc = np.nan_to_num(_masked_mean(ap, V, W) - (ap[V].mean() if V.sum() > 5 else 0.5), nan=0.0)
    M = ff[:, COL["M1"]:COL["M12"] + 1]
    mM = uniform_filter1d(M, W, axis=0, mode="nearest") - M.mean(0)
    loc = lambda c: uniform_filter1d(ff[:, COL[c]], W, mode="nearest") - ff[:, COL[c]].mean()          # noqa: E731
    low = (E <= np.quantile(E, 0.3)).astype(np.float64)
    flq = np.nan_to_num(_masked_mean(ff[:, COL["FLAT"]], low, W2) - ff[low > 0, COL["FLAT"]].mean(), nan=0.0)
    zl = uniform_filter1d(z, W, axis=0, mode="nearest") - z.mean(0)
    jf = _lr_jump(minimum_filter1d(B, 9, axis=0, mode="nearest").mean(1))
    jf0 = _lr_jump(np.where(V, st, f0med) - f0med)
    out = [ff, fl25, fl51, mB,
           np.column_stack([emin - e10, emax - e90, uniform_filter1d(E, W, mode="nearest") - E.mean(), emax - emin,
                            minimum_filter1d(E, W2, mode="nearest") - e10, f0loc, uniform_filter1d(V.astype(float), W, mode="nearest"), aploc]),
           mM,
           np.column_stack([loc("HOLE"), loc("FLAT"), loc("RO99"), loc("H6"), loc("H7"), flq]),
           np.column_stack([jf, jf0, _lr_jump(M), _lr_jump(z), _lr_jump(E), _lr_jump(ff[:, COL["H7"]])]),
           zl]
    fctx = np.r_[floorF, e90 - e10, f0med, ff[:, COL["HOLE"]].mean(), ff[:, COL["FLAT"]].mean(), e90, V.mean(), np.log(T * 0.02)]
    out.append(np.repeat(fctx[None, :], T, 0))
    X = np.hstack(out).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=-100.0)


def selftest() -> None:
    rng = np.random.default_rng(0)
    ff = rng.normal(0, 1, (300, 40)).astype(np.float32)
    ff[:, COL["V"]] = 1
    z = rng.normal(0, 1, (300, 64)).astype(np.float32)
    X = ctx_feats(ff, z)
    assert X.shape == (300, len(names())), (X.shape, len(names()))
    assert np.isfinite(X).all()
    x = np.r_[np.zeros(50), np.ones(50)]
    j = _lr_jump(x, 10)
    assert j.argmax() == 50 and abs(j[50] - 1) < 1e-9 and j[10] == 0
    print("ctxfeat selftest dat", X.shape)


if __name__ == "__main__":
    selftest()
