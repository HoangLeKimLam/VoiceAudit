# -*- coding: utf-8 -*-
"""DAC TRUNG KHUNG 100 ms + BO DO VET CAT cho bo loc nhe (v6.scrape.filter) va kiem dinh (v6.scrape.validate_cuts).

Khong torch. Doc am thanh theo khoi (FrameAccumulator.feed), giu MANG KHUNG nho (vai MB cho tep 6 h):
  rms_db · flat (do phang pho 100-7800 Hz) · band (ty le 300-3400 Hz) · roll (tan so cuon 99 %) · peaks (3 dinh 60-4000 Hz)
  · edge (bang thong: tan so cao nhat > dinh - 50 dB) · spec16 (16 dai log nang luong, hinh dang nen / doi nguoi noi) · dmax (buoc mau lon nhat / khung) · vad (webrtcvad muc 2)
  + vet lang so (|x| < -80 dBFS >= 40 ms) dem xuyen khoi.
Sau finish(): speech_mask · noise_floor · music_mask · detect_cuts (a lang so trong loi noi, b buoc nen on, c nhay am luong
loi noi khong doi nguoi, d click o cuc tieu nang luong) · bw_switches (nhay bang thong >= 2 kHz keo dai > 30 s).
Nguong lay tu filter_spec C1-C5 (audit1_scrape); CHUA hieu chinh tren du lieu that -> xem validate_cuts.
FineEnvelope / phases_from_env8 / save_env8: duong bao 100 ms o 8 pha chinh xac (buoc 12,5 ms) cho van tay qc.dedup.

CLI: khong co (thu vien); kiem thu: python -m v6.scrape.test_scrape"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16000
HOP = 1600                 # 100 ms
NFFT = 2048
FR_S = HOP / SR
DS_THR = 1e-4              # -80 dBFS
DS_MIN = int(0.04 * SR)    # 40 ms
_F = np.fft.rfftfreq(NFFT, 1 / SR)
_BAND = (_F >= 100) & (_F <= 7800)
_VOICE = (_F >= 300) & (_F <= 3400)
_ALL = (_F >= 60) & (_F <= 8000)
_PEAK = np.where((_F >= 60) & (_F <= 4000))[0]
_EDGES16 = np.unique(np.round(np.geomspace(4, NFFT // 2, 17)).astype(int))
_WIN = np.hanning(HOP).astype(np.float32)

# nguong (filter_spec)
VAD_MARGIN_DB = 6.0        # co webrtcvad: van doi nang luong > nen + 6 dB
NRG_MARGIN_DB = 9.0        # khong co webrtcvad: nang luong > nen + 9 dB, phang < 0,4, ty le 300-3400 Hz > 0,4
FLAT_MUSIC = 0.05
# pilot 10/10 (42 tep that): nguong filter_spec ban dau (6 dB / 0,10 / 8 dB / x6) ban 30-900 vet / gio tren ban ghi
# hop / lop hoc -> nang len va doi hai phia ON DINH (IQR <= STAB_DB); van CHUA hieu chinh bang nghe (xem validate_cuts)
NF_STEP_DB = 10.0
NF_SPEC_DIST = 0.15
LOUD_STEP_DB = 10.0
STAB_DB = 6.0
GATED_DS_PER_H = 60.0      # > 60 doan lang so / gio = chuoi co cong nhieu (Zoom / OBS / Opus DTX): vet 'a' vo nghia
EDGE_REL_DB = 50.0         # bang thong khung = tan so cao nhat con > dinh pho - 50 dB (lam tron 9 bin)
SPK_SAME_DIST = 0.08
CLICK_RATIO = 10.0
CLICK_ABS = 0.05
BW_STEP_HZ = 2000.0


def _vad():
    try:
        import webrtcvad
        return webrtcvad.Vad(2)
    except ImportError:   # pragma: no cover - may khong co webrtcvad -> luat nang luong
        return None


@dataclass
class FrameAccumulator:
    """Nhan khoi float32 16 kHz mono theo thu tu; tinh dac trung tung khung 100 ms."""
    use_vad: bool = True
    keep_windows: bool = False          # giu 1 cua so 3 s nhieu loi noi nhat moi 60 s (dem nguoi noi)
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _cols: dict = field(default_factory=lambda: {k: [] for k in ("rms_db", "flat", "band", "roll", "peaks", "spec16",
                                                                 "dmax", "vad", "clip", "edge")})
    _n_samples: int = 0
    _ds_run: int = 0
    _ds_events: list = field(default_factory=list)
    windows: list = field(default_factory=list)
    _vad_obj: object = None
    _win_buf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _win_t0: float = 0.0

    def __post_init__(self) -> None:
        self._vad_obj = _vad() if self.use_vad else None

    # ---- vet lang so dem xuyen khoi ----
    def _digital_silence(self, x: np.ndarray, t0: int) -> None:
        q = np.abs(x) < DS_THR
        dq = np.diff(np.concatenate([[0], q.astype(np.int8), [0]]))
        starts, ends = np.where(dq == 1)[0], np.where(dq == -1)[0]
        carry, new_carry = self._ds_run, 0
        if carry and not (starts.size and starts[0] == 0):      # doan lang mang sang da ket thuc o bien khoi
            if carry >= DS_MIN:
                self._ds_events.append(((t0 - carry) / SR, carry / SR))
        for s, e in zip(starts, ends):
            run, a = e - s, t0 + s
            if s == 0 and carry:
                run, a = run + carry, t0 - carry
            if e == len(x):                                      # con keo dai sang khoi sau
                new_carry = run
                continue
            if run >= DS_MIN:
                self._ds_events.append((a / SR, run / SR))
        self._ds_run = new_carry

    def feed(self, x: np.ndarray) -> None:
        x = np.asarray(x, np.float32).ravel()
        self._digital_silence(x, self._n_samples)
        self._n_samples += len(x)
        buf = np.concatenate([self._buf, x])
        k = len(buf) // HOP
        if k:
            self._frames(buf[:k * HOP].reshape(k, HOP))
        self._buf = buf[k * HOP:]
        if self.keep_windows:
            self._window_reservoir(x)

    def _frames(self, fr: np.ndarray) -> None:
        c = self._cols
        rms = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1))
        c["rms_db"].append((20 * np.log10(rms + 1e-10)).astype(np.float32))
        c["clip"].append((np.abs(fr) >= 0.999).sum(axis=1).astype(np.int32))
        c["dmax"].append(np.abs(np.diff(fr, axis=1)).max(axis=1).astype(np.float32))
        P = np.abs(np.fft.rfft(fr * _WIN, NFFT, axis=1)) ** 2 + 1e-12
        pb = P[:, _BAND]
        c["flat"].append((np.exp(np.mean(np.log(pb), axis=1)) / np.mean(pb, axis=1)).astype(np.float32))
        c["band"].append((P[:, _VOICE].sum(1) / P[:, _ALL].sum(1)).astype(np.float32))
        cs = np.cumsum(P[:, _ALL], axis=1)
        idx = np.argmax(cs >= 0.99 * cs[:, -1:], axis=1)
        c["roll"].append(_F[_ALL][idx].astype(np.float32))
        pdb = 10 * np.log10(P[:, _ALL])
        cs9 = np.cumsum(np.pad(pdb, ((0, 0), (4, 4)), mode="edge"), axis=1)
        sm = (cs9[:, 8:] - np.pad(cs9, ((0, 0), (1, 0)))[:, :-9]) / 9
        above = sm > sm.max(axis=1, keepdims=True) - EDGE_REL_DB
        last = above.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1)
        c["edge"].append(_F[_ALL][last].astype(np.float32))
        pk = _PEAK[np.argsort(P[:, _PEAK], axis=1)[:, -3:]]
        c["peaks"].append(np.sort(pk, axis=1).astype(np.int16))
        s16 = np.stack([P[:, a:b].sum(1) for a, b in zip(_EDGES16[:-1], _EDGES16[1:])], axis=1)
        c["spec16"].append(np.log10(s16).astype(np.float32))
        if self._vad_obj is not None:
            pcm = (np.clip(fr, -1, 1) * 32767).astype(np.int16)
            v = np.zeros(len(fr), np.int8)
            for i, f in enumerate(pcm):
                hits = sum(self._vad_obj.is_speech(f[j:j + 480].tobytes(), SR) for j in (0, 480, 960))
                v[i] = 1 if hits >= 2 else 0
            c["vad"].append(v)

    def _window_reservoir(self, x: np.ndarray) -> None:
        """Moi 60 s giu cua so 3 s co nang luong trung vi cao nhat (uu tien loi noi) cho dem nguoi noi."""
        self._win_buf = np.concatenate([self._win_buf, x])
        while len(self._win_buf) >= 60 * SR:
            blk, self._win_buf = self._win_buf[:60 * SR], self._win_buf[60 * SR:]
            w = 3 * SR
            best, bi = -1e9, 0
            for i in range(0, len(blk) - w + 1, SR):
                e = np.median(np.abs(blk[i:i + w:160]))
                if e > best:
                    best, bi = e, i
            if best > 10 ** (-45 / 20):
                self.windows.append((self._win_t0 + bi / SR, blk[bi:bi + w].copy()))
            self._win_t0 += 60.0

    def finish(self) -> dict[str, np.ndarray]:
        if self._ds_run >= DS_MIN:
            self._ds_events.append(((self._n_samples - self._ds_run) / SR, self._ds_run / SR))
            self._ds_run = 0
        out = {}
        for k, v in self._cols.items():
            if not v:
                out[k] = np.zeros((0, 3) if k == "peaks" else (0, len(_EDGES16) - 1) if k == "spec16" else 0)
                continue
            out[k] = np.concatenate(v)
        out["ds_events"] = np.array(self._ds_events, np.float64).reshape(-1, 2)
        out["dur_s"] = np.array(self._n_samples / SR)
        return out


# ---------------- duong bao min (8 pha) cho van tay trung lap (v6.scrape.qc) ----------------
SUB = 200                  # mau / khoi con (12,5 ms)
SUB_PER_FRAME = HOP // SUB  # 8 khoi con = 1 khung 100 ms


class FineEnvelope:
    """rms_db cua so 100 ms, BUOC 12,5 ms (= 8 pha CHINH XAC cua duong bao 100 ms) tu cac khoi lien tiep.
    env8[k + 8 m] = 10 log10(trung binh x^2 tren mau [200 k + 1 600 m, +1 600)) -> pha k = env8[k::8]."""

    def __init__(self) -> None:
        self._rest = np.zeros(0, np.float32)
        self._pw: list[np.ndarray] = []

    def feed(self, x: np.ndarray) -> None:
        buf = np.concatenate([self._rest, np.asarray(x, np.float32).ravel()])
        k = len(buf) // SUB
        if k:
            self._pw.append((buf[:k * SUB].reshape(k, SUB).astype(np.float64) ** 2).mean(axis=1))
        self._rest = buf[k * SUB:]

    def finish(self) -> np.ndarray:
        pw = np.concatenate(self._pw) if self._pw else np.zeros(0)
        if len(pw) < SUB_PER_FRAME:
            return np.zeros(0, np.float32)
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(pw, SUB_PER_FRAME).mean(axis=1)      # tong chinh xac (khong cumsum: lang so = 0 dung)
        return (10 * np.log10(win + 1e-20)).astype(np.float32)


def phases_from_env8(env8: np.ndarray, n_phase: int = SUB_PER_FRAME) -> list[np.ndarray]:
    """n_phase pha deu (n_phase <= 8, chia het 8 tot nhat) tu env8; pha 0 = duong bao 100 ms thuong."""
    n_phase = max(1, min(int(n_phase), SUB_PER_FRAME))
    e = np.asarray(env8, np.float32)
    return [e[int(round(k * SUB_PER_FRAME / n_phase))::SUB_PER_FRAME] for k in range(n_phase)]


def save_env8(path: str, env8: np.ndarray) -> None:
    """Ghi nguyen tu (float16, ~160 B / giay am thanh)."""
    import os
    tmp = path[:-4] + ".tmp.npy" if path.endswith(".npy") else path + ".tmp.npy"
    np.save(tmp, np.asarray(env8, np.float16))
    os.replace(tmp, path)


# ---------------- quyet dinh tren mang khung ----------------
def _rolling_pct(x: np.ndarray, block: int = 300, q: float = 10) -> np.ndarray:
    """Phan vi q theo khoi block khung, noi suy ve tung khung (nen on bien doi cham)."""
    n = len(x)
    if n == 0:
        return x
    cent, vals = [], []
    for i in range(0, n, block):
        seg = x[i:i + block]
        cent.append(i + len(seg) / 2)
        vals.append(np.percentile(seg, q))
    return np.interp(np.arange(n), cent, vals).astype(np.float32)


def _median3(m: np.ndarray) -> np.ndarray:
    if len(m) < 3:
        return m
    p = np.pad(m.astype(np.int8), 1, mode="edge")
    return ((p[:-2] + p[1:-1] + p[2:]) >= 2)


def speech_mask(F: dict) -> tuple[np.ndarray, np.ndarray]:
    nf = _rolling_pct(F["rms_db"])
    if len(F.get("vad", [])) == len(F["rms_db"]) and len(F["rms_db"]):
        sp = (F["vad"] > 0) & (F["rms_db"] > nf + VAD_MARGIN_DB)
    else:
        sp = (F["rms_db"] > nf + NRG_MARGIN_DB) & (F["flat"] < 0.4) & (F["band"] > 0.4)
    return _median3(sp), nf


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Cac doan True lien tuc [a, b) theo chi so khung."""
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def music_mask(F: dict, speech: np.ndarray) -> np.ndarray:
    """Khung nhac / am sac: dinh pho on dinh (+-1 bin) >= 0,5 s; bo dinh xuat hien > 30 % tep (u dien luoi, quat)."""
    pk = F["peaks"].astype(np.int32)
    n = len(pk)
    if n < 5:
        return np.zeros(n, bool)
    vals, cnt = np.unique(pk.ravel(), return_counts=True)
    persistent = set(vals[cnt > 0.3 * n].tolist()) | set((vals[cnt > 0.3 * n] + 1).tolist()) | \
        set((vals[cnt > 0.3 * n] - 1).tolist())
    keep = ~np.isin(pk, list(persistent)) if persistent else np.ones_like(pk, bool)
    match = np.zeros(n, bool)
    for i in range(1, n):   # >= 2/3 dinh khop khung truoc (+-1 bin), chi xet dinh khong thuong truc
        a, b = pk[i], pk[i - 1]
        ok = (np.abs(a[:, None] - b[None, :]) <= 1).any(axis=1) & keep[i]
        match[i] = ok.sum() >= 2
    stable5 = np.convolve(match.astype(int), np.ones(5, int), mode="same") >= 5
    stable10 = np.convolve(match.astype(int), np.ones(10, int), mode="same") >= 10
    return ((~speech) & (F["flat"] < FLAT_MUSIC) & stable5) | stable10


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _dedupe(ev: list[dict], gap_s: float = 2.0) -> list[dict]:
    ev = sorted(ev, key=lambda e: e["t"])
    out: list[dict] = []
    for e in ev:
        if out and e["t"] - out[-1]["t"] < gap_s:
            continue
        out.append(e)
    return out


def _step_events(idx: np.ndarray, val: np.ndarray, spec: np.ndarray, k: int, thr: float, max_span: float,
                 want_same: bool | None, typ: str) -> list[dict]:
    """Buoc nhay tren day khung con (chi so idx): trung vi k khung truoc/sau lech >= thr dB."""
    ev = []
    n = len(idx)
    if n < 2 * k:
        return ev
    score = np.zeros(n)
    for j in range(k, n - k + 1):
        if (idx[j - 1] - idx[j - k]) * FR_S > max_span or (idx[j + k - 1] - idx[j]) * FR_S > max_span:
            continue
        score[j] = abs(np.median(val[j - k:j]) - np.median(val[j:j + k]))
    for j in range(k, n - k + 1):
        if score[j] < thr or score[j] < score[max(k, j - k):j + k].max():
            continue
        # buoc phai DOT NGOT: 3 khung sat truoc/sau da lech >= 60 % (khong phai troi cham)
        pre, post = val[j - k:j], val[j:j + k]
        if np.subtract(*np.percentile(pre, [75, 25])) > STAB_DB or np.subtract(*np.percentile(post, [75, 25])) > STAB_DB:
            continue                                      # mot phia khong on dinh -> troi / doi nguoi, khong phai buoc
        near = abs(np.median(val[max(0, j - 3):j]) - np.median(val[j:j + 3]))
        if near < 0.6 * score[j]:
            continue
        d = _cos_dist(spec[j - k:j].mean(0), spec[j:j + k].mean(0))
        if want_same is True and d >= SPK_SAME_DIST:      # nhay am luong NHUNG doi pho -> doi nguoi noi, khong tinh
            continue
        if want_same is False and d < NF_SPEC_DIST:       # buoc nen on phai kem doi hinh dang pho nen
            continue
        t = 0.5 * (idx[j - 1] + idx[j] + 1) * FR_S
        ev.append({"t": round(float(t), 2), "type": typ, "db": round(float(score[j]), 1), "spec_dist": round(d, 3),
                   "lo": round(float((idx[j - 1] + 1) * FR_S), 2), "hi": round(float(idx[j] * FR_S), 2)})   # buoc nam trong [lo, hi]
    return ev


def gated_floor(F: dict) -> tuple[bool, float]:
    """Chuoi ghi co cong nhieu: so doan lang so >= 40 ms / gio (ca trong khoang lang) > GATED_DS_PER_H."""
    hours = max(len(F["rms_db"]) * FR_S / 3600, 1e-9)
    rate = len(F["ds_events"]) / hours
    return rate > GATED_DS_PER_H, round(rate, 1)


def detect_cuts(F: dict, speech: np.ndarray, edge_s: float = 1.0, gated: bool | None = None) -> list[dict]:
    """Vet cat a-d (filter_spec C5). Tra danh sach {t, type, ...} da gop trong 2 s.
    gated: trang thai 'chuoi co cong' cua TEP GOC (validate_cuts truyen tu doan chua sua - vet cay day dac khong duoc
    tu kich hoat luat cong); None = tu do tren F."""
    n = len(F["rms_db"])
    dur = n * FR_S
    ev: list[dict] = []
    if gated is None:
        gated, _ = gated_floor(F)
    for t, d in ([] if gated else F["ds_events"]):
        i0, i1 = int(t / FR_S), int((t + d) / FR_S)
        before = speech[max(0, i0 - 10):i0].any()
        after = speech[i1 + 1:i1 + 11].any()
        if edge_s <= t <= dur - edge_s and before and after:
            ev.append({"t": round(float(t + d / 2), 2), "type": "a_digital_silence", "ms": round(1000 * float(d))})
    ns = np.where(~speech)[0]
    ev += _step_events(ns, F["rms_db"][ns], F["spec16"][ns], 10, NF_STEP_DB, 15.0, False, "b_noise_floor_step")
    sp = np.where(speech)[0]
    ev += _step_events(sp, F["rms_db"][sp], F["spec16"][sp], 30, LOUD_STEP_DB, 6.0, True, "c_loudness_jump")
    if n > 10:
        lin = 10 ** (F["rms_db"] / 20)
        loc = np.sqrt(np.convolve(lin ** 2, np.ones(11) / 11, mode="same"))
        med = np.convolve(F["rms_db"], np.ones(11) / 11, mode="same")
        cl = (F["dmax"] > CLICK_RATIO * loc) & (F["dmax"] > CLICK_ABS) & (F["rms_db"] <= med)
        for i in np.where(cl)[0]:
            t = i * FR_S
            if edge_s <= t <= dur - edge_s:
                ev.append({"t": round(float(t + FR_S / 2), 2), "type": "d_click", "ratio": round(float(F["dmax"][i] / loc[i]), 1)})
    return _dedupe([e for e in ev if edge_s <= e["t"] <= dur - edge_s])


def bw_switches(F: dict, speech: np.ndarray, win: int = 300, stride: int = 50) -> list[dict]:
    """Doi chat luong livestream: trung vi roll-off 30 s truoc/sau lech >= 2 kHz."""
    sp = np.where(speech)[0]
    out = []
    if len(sp) < 2 * win:
        return out
    r = F.get("edge", F["roll"])[sp]
    best = None
    for j in range(win, len(sp) - win, stride):
        d = float(np.median(r[j:j + win]) - np.median(r[j - win:j]))
        if abs(d) >= BW_STEP_HZ:
            if best is None or abs(d) > abs(best[1]):
                best = (j, d)
        elif best is not None:
            out.append({"t": round(float(sp[best[0]] * FR_S), 1), "type": "bw_switch", "hz": round(best[1])})
            best = None
    if best is not None:
        out.append({"t": round(float(sp[best[0]] * FR_S), 1), "type": "bw_switch", "hz": round(best[1])})
    return out


def analyse_array(y: np.ndarray, use_vad: bool = True, block_s: float = 60.0) -> tuple[dict, np.ndarray, np.ndarray]:
    """Tien ich: chay FrameAccumulator tren mang trong RAM (kiem thu / validate_cuts). Tra (F, speech, noise_floor)."""
    acc = FrameAccumulator(use_vad=use_vad)
    b = int(block_s * SR)
    for i in range(0, len(y), b):
        acc.feed(y[i:i + b])
    F = acc.finish()
    sp, nf = speech_mask(F)
    return F, sp, nf
