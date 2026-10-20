# -*- coding: utf-8 -*-
"""Anh bang chung (K9): moi thao tac mot PNG — dang song + dai pho log-mel cua CHINH tep, to dung doan,
truc thoi gian mm:ss,cc, tieu de loai · pham vi · moc · muc do; tep dai (> 20 s) va dong cuc bo: them dai tong quan
ca tep + phong to [a − 3 s, b + 3 s]. Anh tong quan ca tep: dang song + pho + xac suat khung K4 (4 lop cuc bo)
+ moi thao tac. Chi ve lai so da co trong ho so (khong tinh quyet dinh moi)."""
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from v6.io.common import CLS, EDGE_S, LONG_S, SEV_COLOR, mmss

SR = 16000
PAD = 3.0
MAX_COLS = 2400
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11})
_TIME_FMT = FuncFormatter(lambda x, _: mmss(x))
LANE_NAMES = ["khử ồn đoạn", "đổi giọng đoạn", "ghép lời thật", "giọng máy đoạn"]      # ten loai theo Scope (khong noi kieu con 'chen')


def _envelope(y: np.ndarray, lo: float, hi: float, bins: int = 2000) -> tuple:
    seg = y[int(lo * SR):int(hi * SR)]
    step = max(1, len(seg) // bins)
    n = max(1, len(seg) // step)
    s = seg[: n * step].reshape(n, step) if len(seg) >= step else seg.reshape(1, -1)
    return lo + (np.arange(len(s)) * step) / SR, s.min(1), s.max(1)


def _logmel(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import librosa
    seg = y[int(lo * SR):int(hi * SR)].astype(np.float32)
    if len(seg) < 512:
        seg = np.pad(seg, (0, 512 - len(seg)))
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=seg, sr=SR, n_fft=512, hop_length=160, n_mels=64),
                            ref=np.max)
    if S.shape[1] > MAX_COLS:                       # gop cot cho tep dai (anh van dung ti le thoi gian)
        k = int(np.ceil(S.shape[1] / MAX_COLS))
        m = S.shape[1] // k
        S = S[:, : m * k].reshape(S.shape[0], m, k).mean(2)
    return S


def _spec_axis(ax, S: np.ndarray, lo: float, hi: float) -> None:
    import librosa
    ax.imshow(S, origin="lower", aspect="auto", extent=[lo, hi, 0, S.shape[0]], cmap="magma", vmin=-80, vmax=0,
              interpolation="nearest")
    mel_hz = librosa.mel_frequencies(n_mels=S.shape[0] + 2, fmax=SR / 2)[1:-1]
    ticks = [int(np.argmin(np.abs(mel_hz - f))) for f in (500, 2000, 4000, 7900)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0,5", "2", "4", "8 kHz"])
    ax.set_ylabel("phổ mel")


def _unscanned(ax, dur: float, lo: float, hi: float) -> None:
    for u, v in ((0.0, EDGE_S), (dur - EDGE_S, dur)):
        if v > lo and u < hi:
            ax.axvspan(max(u, lo), min(v, hi), facecolor="none", edgecolor="#9a9a9a", hatch="///", lw=0)


def _time_axis(ax, lo: float, hi: float, label: bool) -> None:
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(_TIME_FMT)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    if label:
        ax.set_xlabel("thời gian (mm:ss,cc tính từ đầu tệp)")
    else:
        ax.tick_params(labelbottom=False)


def render_op(y: np.ndarray, dur: float, op: dict, P: np.ndarray | None, grid_t: np.ndarray | None,
              path: str, kenh_label: str = "") -> dict:
    """Mot anh cho mot thao tac. Tra thong tin ve anh (vung to, khung nhin, kich thuoc, thoi gian ve)."""
    t0 = time.time()
    a, b = float(op["bat_dau_s"]), float(op["ket_thuc_s"])
    glob_op = op["pham_vi"] == "cả file"
    zoom = dur > LONG_S and not glob_op
    lo, hi = (max(0.0, a - PAD), min(dur, b + PAD)) if zoom else (0.0, dur)
    col = SEV_COLOR[op["muc_do"]]
    ratios = ([0.35] if zoom else []) + [1.0, 1.6, 0.55]
    fig, ax = plt.subplots(len(ratios), 1, figsize=(10, 1.15 * sum(ratios) + 1.0),
                           gridspec_kw={"height_ratios": ratios})
    k0 = 0
    if zoom:
        tt, mn, mx = _envelope(y, 0.0, dur, 1500)
        ax[0].fill_between(tt, 0, np.maximum(np.abs(mn), np.abs(mx)), color="#8a8a8a", lw=0)
        ax[0].axvspan(lo, hi, color="#1f5fbf", alpha=0.16, lw=0)
        ax[0].axvspan(a, b, color=col, alpha=0.9, lw=0)
        ax[0].set_yticks([])
        ax[0].set_ylabel("cả tệp", fontsize=8)
        _time_axis(ax[0], 0.0, dur, False)
        ax[0].tick_params(labelbottom=True, labelsize=7)
        k0 = 1
    w, s_, p_ = ax[k0], ax[k0 + 1], ax[k0 + 2]
    tt, mn, mx = _envelope(y, lo, hi)
    w.fill_between(tt, mn, mx, color="#3a3a3a", lw=0)
    w.set_ylabel("biên độ")
    _spec_axis(s_, _logmel(y, lo, hi), lo, hi)
    for x in (w, s_, p_):
        x.axvspan(max(a, lo), min(b, hi), color=col, alpha=0.10 if glob_op else 0.24, lw=0)
        _unscanned(x, dur, lo, hi)
    if not glob_op:
        for x in (w, s_):
            x.axvline(a, color=col, lw=1.5)
            x.axvline(b, color=col, lw=1.5)
    if not glob_op and P is not None and grid_t is not None and len(grid_t):
        j = CLS.index(op["nhan_cap2"])
        sel = (grid_t >= lo) & (grid_t <= hi)
        p_.plot(grid_t[sel], P[: len(grid_t)][sel, j], color=col, lw=1.5)
        p_.axhline(0.3, color="#9a9a9a", lw=0.8, ls="--")
        p_.set_ylabel("xác suất\nkhung K4", fontsize=8)
    else:
        p_.text(0.5, 0.5, "thao tác cả file — tô toàn bộ bản ghi", ha="center", va="center",
                transform=p_.transAxes, fontsize=9, color="#555")
        p_.set_yticks([])
    p_.set_ylim(0, 1)
    for x in (w, s_):
        _time_axis(x, lo, hi, False)
    _time_axis(p_, lo, hi, True)
    ax[0].set_title(f"{op['id']}{kenh_label} · {op['loai']} · {op['pham_vi']} · {op['bat_dau']} – {op['ket_thuc']} · "
                    f"{op['muc_do']}", loc="left", color=col)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=100)
    size = [int(v * 100) for v in fig.get_size_inches()]
    plt.close(fig)
    return {"tep": os.path.basename(path), "vung_to_s": [a, b], "khung_nhin_s": [lo, hi], "phong_to": zoom,
            "px": size, "giay": round(time.time() - t0, 2)}


def render_overview(y: np.ndarray, dur: float, ops: list, P: np.ndarray | None, grid_t: np.ndarray | None,
                    path: str, title: str, lanes: list | None = None) -> dict:
    """Anh tong quan ca tep: dang song · pho · xac suat khung cac lop cuc bo (chu ky bang chung) · moi thao tac.
    lanes = ten lan K4 (mac dinh LANE_NAMES); ho so v9 ghi ' · không soi' cho lop cua nhanh tat (cot diem de trong)."""
    t0 = time.time()
    has_p = P is not None and grid_t is not None and len(grid_t) > 1
    ratios = [1.0, 1.4] + ([0.9] if has_p else []) + [0.35]
    fig, ax = plt.subplots(len(ratios), 1, figsize=(10, 1.15 * sum(ratios) + 1.0), gridspec_kw={"height_ratios": ratios})
    tt, mn, mx = _envelope(y, 0.0, dur)
    ax[0].fill_between(tt, mn, mx, color="#3a3a3a", lw=0)
    ax[0].set_ylabel("biên độ")
    _spec_axis(ax[1], _logmel(y, 0.0, dur), 0.0, dur)
    k = 2
    if has_p:
        n = min(len(grid_t), len(P))
        half = (grid_t[1] - grid_t[0]) / 2
        ax[2].imshow(P[:n, 1:].T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1, interpolation="nearest",
                     extent=[grid_t[0] - half, grid_t[n - 1] + half, -0.5, P.shape[1] - 1.5])
        ax[2].set_yticks(range(P.shape[1] - 1))
        ax[2].set_yticklabels(list(lanes or LANE_NAMES)[:P.shape[1] - 1], fontsize=7)
        ax[2].set_ylabel("K4", fontsize=8)
        k = 3
    lane = ax[k]
    lane.set_yticks([])
    lane.set_ylabel("thao tác", fontsize=8)
    n_glob = sum(1 for o in ops if o["pham_vi"] == "cả file")
    gi = 0
    for op in ops:
        col = SEV_COLOR[op["muc_do"]]
        a, b = float(op["bat_dau_s"]), float(op["ket_thuc_s"])
        glob_op = op["pham_vi"] == "cả file"
        for x in ax[:k]:
            x.axvspan(a, b, color=col, alpha=0.08 if glob_op else 0.22, lw=0)
        if glob_op:                                   # nhieu thao tac ca file: xep chong len nhau theo tung soc ngang, khong de len nhau
            h = 0.4 / max(1, n_glob)
            y0 = 0.55 + gi * h
            gi += 1
            ymin, ymax, yt = y0, y0 + h - 0.02, y0 + h / 2
        else:
            ymin, ymax, yt = 0.05, 0.5, 0.27
        lane.axvspan(a, b, ymin=ymin, ymax=ymax, color=col, alpha=0.85, lw=0)
        lane.text((a + b) / 2, yt, op["id"], ha="center", va="center", fontsize=7, color="white", transform=lane.get_xaxis_transform())
    for x in ax:
        _unscanned(x, dur, 0.0, dur)
        _time_axis(x, 0.0, dur, x is lane)
    ax[0].set_title(title, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return {"tep": os.path.basename(path), "giay": round(time.time() - t0, 2)}
