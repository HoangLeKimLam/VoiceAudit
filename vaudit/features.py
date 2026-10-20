# -*- coding: utf-8 -*-
"""Ngan hang dac trung v4: PHO + ANH PHO + 3 model PRETRAINED (dong bang).

Tinh 1 lan / file (FileFeat), roi cat theo cua so:
  - boundary(t):  so sanh cua so TRAI [t-w,t] vs PHAI [t,t+w]  -> bat MOI NOI
  - inside(a,b):  doan [a,b] SO VOI phan con lai cua file       -> bat DOAN BI SUA (loai gi)
  - global():     ca file                                        -> thao tac TOAN CUC
"""
import os

import librosa
import numpy as np
import scipy.fft
import torch
from skimage.feature import graycomatrix, graycoprops
from skimage.metrics import structural_similarity

from . import SR

HOP = 160           # 10 ms / khung
NFFT = 512
_models = {}


# ---------------- pretrained (nap 1 lan) ----------------
def _spk():
    if "spk" not in _models:
        from resemblyzer import VoiceEncoder
        _models["spk"] = VoiceEncoder("cpu")
    return _models["spk"]


def _spoof():
    if "spoof" not in _models:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        # Da so sanh tren file THAT co phong+nhieu: MelodyMachine bao nham ~50% cua so,
        # Gustking (XLS-R da ngon ngu, cua so 2 s) chi ~7% -> chon Gustking.
        name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        fe = AutoFeatureExtractor.from_pretrained(name)
        m = AutoModelForAudioClassification.from_pretrained(name).eval()
        fi = [i for i, l in m.config.id2label.items() if "fake" in l.lower()][0]
        _models["spoof"] = (fe, m, fi)
    return _models["spoof"]


LONG_S, CHUNK_S, OVL_S = 20.0, 16.0, 2.0   # tep > 20 s: chay mo hinh theo khuc 16 s chong 2 s (tep v4 <= 17 s khong doi)


def chunk_frames(y, fn):
    """Chay fn(doan song) -> (so khung x D) theo khuc 16 s chong 2 s, ghep lai theo thoi gian tuyet doi.
    Moi khung lay tu khuc ma no nam gan tam nhat. Tra ve (ma tran khung, buoc khung giay)."""
    n, L, step = len(y), int(CHUNK_S * SR), int((CHUNK_S - OVL_S) * SR)
    starts = list(range(0, max(1, n - int(OVL_S * SR)), step))
    parts = []
    for s in starts:
        h = fn(y[s:s + L])
        dt = min(L, n - s) / SR / h.shape[0]
        parts.append((s / SR, dt, h))
    dt = parts[0][1]
    T = int(round(n / SR / dt))
    out = np.zeros((T, parts[0][2].shape[1]), np.float32)
    best = np.full(T, np.inf)
    for s0, d, h in parts:
        tc = s0 + (np.arange(h.shape[0]) + 0.5) * d
        idx = np.clip((tc / dt).astype(int), 0, T - 1)
        dist = np.abs(tc - (s0 + h.shape[0] * d / 2))
        take = dist < best[idx]
        out[idx[take]], best[idx[take]] = h[take], dist[take]
    return out, dt


def _w2v_hidden(y):
    from .asr import _load
    _, model = _load()

    def run(seg):
        x = torch.from_numpy(((seg - seg.mean()) / (seg.std() + 1e-7)).astype(np.float32))[None]
        with torch.no_grad():
            return model.wav2vec2(x).last_hidden_state[0].numpy()
    if len(y) / SR > LONG_S:
        return chunk_frames(y, run)
    hs = run(y)
    return hs, len(y) / SR / hs.shape[0]


# ---------------- tien ich pho ----------------
def _linear_fbank(n_filt=40, n_fft=NFFT, sr=SR):
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = np.linspace(0, sr / 2, n_filt + 2)
    fb = np.zeros((n_filt, len(freqs)))
    for i in range(n_filt):
        l, c, r = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((freqs - l) / (c - l + 1e-9), (r - freqs) / (r - c + 1e-9)), 0, None)
    return fb


_LFB = _linear_fbank()


def _cpp(power_frames):
    """Cepstral Peak Prominence moi khung (do ro cau truc hai am, 70-400 Hz)."""
    logp = np.log(power_frames + 1e-10)
    cep = np.fft.irfft(logp, axis=0)
    q = np.arange(cep.shape[0]) / SR
    band = (q >= 1 / 400) & (q <= 1 / 70)
    seg = cep[band]
    qq = q[band]
    peak = seg.max(axis=0)
    A = np.vstack([qq, np.ones_like(qq)]).T
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    line_at_peak = coef[0] * qq[seg.argmax(axis=0)] + coef[1]
    return peak - line_at_peak


class FileFeat:
    """Tinh san cac chuoi theo khung 10 ms + embedding pretrained cho 1 file."""

    def __init__(self, y: np.ndarray, spoof: bool = True):
        self.y = np.asarray(y, float)
        self.dur = len(self.y) / SR
        S = librosa.stft(self.y, n_fft=NFFT, hop_length=HOP)
        self.mag, self.phase = np.abs(S), np.angle(S)
        P = self.mag ** 2
        self.logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=SR, n_mels=64))
        lf = np.log(_LFB @ P + 1e-10)
        self.lfcc = scipy.fft.dct(lf, axis=0, norm="ortho")[:20]
        self.mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=SR, n_mels=40)), n_mfcc=20)
        self.dmfcc = librosa.feature.delta(self.mfcc)
        self.rms_db = 10 * np.log10(P.mean(axis=0) + 1e-12)
        freqs = librosa.fft_frequencies(sr=SR, n_fft=NFFT)
        self.hf_ratio = 10 * np.log10(P[freqs >= 4000].sum(0) / (P.sum(0) + 1e-12) + 1e-12)
        self.centroid = librosa.feature.spectral_centroid(S=self.mag, sr=SR)[0]
        self.flatness = librosa.feature.spectral_flatness(S=self.mag)[0]
        self.rolloff = librosa.feature.spectral_rolloff(S=self.mag, sr=SR)[0]
        self.contrast = librosa.feature.spectral_contrast(S=self.mag, sr=SR).mean(axis=0)
        self.cpp = _cpp(P)
        # flux phuc (bat dut gay PHA tai moi noi, khong chi bien do)
        Sc = self.mag * np.exp(1j * self.phase)
        pred = np.abs(Sc[:, 1:-1]) * np.exp(1j * (2 * self.phase[:, 1:-1] - self.phase[:, :-2]))
        self.cflux = np.concatenate([[0], np.abs(Sc[:, 2:] - pred).sum(0), [0]])
        self.cflux /= (np.median(self.cflux) + 1e-9)
        f0 = librosa.yin(self.y, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=HOP)
        n = self.mag.shape[1]
        f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
        voiced = self.rms_db > np.percentile(self.rms_db, 40)
        self.f0 = np.where(voiced, f0, np.nan)
        self.n = n
        # pretrained 1: speaker-embedding theo luoi (resemblyzer, gop lo)
        self._spk_grid()
        # pretrained 2: wav2vec2-vi (chinh la mo hinh ASR, lay tang an cuoi)
        self.w2v, self.w2v_step = _w2v_hidden(self.y)
        # pretrained 3: diem 'giong gia' theo cua so 1 s
        self.spoof_t, self.spoof_s = self._spoof_grid() if spoof else (np.array([0.0]), np.array([0.0]))

    # --- chuyen giay <-> khung ---
    def fr(self, t):
        return int(np.clip(round(t * SR / HOP), 0, self.n))

    def _spk_grid(self, win=0.8, hop=0.1):
        from resemblyzer import audio as rz
        enc = _spk()
        mel = rz.wav_to_mel_spectrogram(self.y.astype(np.float32))
        w, h = int(win * 100), int(hop * 100)
        starts = list(range(0, max(1, len(mel) - w), h)) or [0]
        batch = np.stack([np.pad(mel[s:s + w], ((0, max(0, w - len(mel[s:s + w]))), (0, 0))) for s in starts])
        with torch.no_grad():   # chia lo 512 cua so de tep dai khong tran bo nho (ket qua y het)
            emb = np.concatenate([enc(torch.from_numpy(batch[i:i + 512]).float()).numpy()
                                  for i in range(0, len(batch), 512)])
        self.spk_t = np.array([(s + w / 2) / 100 for s in starts])
        self.spk_e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def _spoof_grid(self, win=2.0, hop=0.5):
        fe, m, fi = _spoof()
        W, H = int(win * SR), int(hop * SR)
        starts = list(range(0, max(1, len(self.y) - W + 1), H)) or [0]
        segs = [self.y[s:s + W].astype(np.float32) for s in starts]
        if len(segs) <= 40:     # tep v4 (<= 17 s): giu nguyen mot lo nhu cu
            inp = fe(segs, sampling_rate=SR, return_tensors="pt", padding=True)
            with torch.no_grad():
                p = torch.softmax(m(**inp).logits, -1)[:, fi].numpy()
        else:                   # tep dai: chia lo 16 cua so, moi cua so dai bang nhau nen khong doi ket qua
            ps = []
            for i in range(0, len(segs), 16):
                inp = fe(segs[i:i + 16], sampling_rate=SR, return_tensors="pt", padding=True)
                with torch.no_grad():
                    ps.append(torch.softmax(m(**inp).logits, -1)[:, fi].numpy())
            p = np.concatenate(ps)
        return np.array([(s + W / 2) / SR for s in starts]), p

    # --- trich cua so ---
    def _span(self, arr, a, b):
        return arr[..., self.fr(a):max(self.fr(a) + 1, self.fr(b))]

    def _rest(self, arr, a, b):
        return np.concatenate([arr[..., :self.fr(a)], arr[..., self.fr(b):]], axis=-1)

    def _spk_mean(self, a, b, inside=True):
        sel = (self.spk_t >= a) & (self.spk_t <= b)
        if not inside:
            sel = ~sel & ((self.spk_t < a - 0.4) | (self.spk_t > b + 0.4))
        if sel.sum() == 0:
            idx = np.argmin(np.abs(self.spk_t - (a + b) / 2))
            sel = np.zeros_like(self.spk_t, bool); sel[idx] = True
        v = self.spk_e[sel].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def _w2v_mean(self, a, b, inside=True):
        i0, i1 = int(a / self.w2v_step), max(int(a / self.w2v_step) + 1, int(b / self.w2v_step))
        if inside:
            v = self.w2v[i0:i1].mean(0)
        else:
            rest = np.concatenate([self.w2v[:i0], self.w2v[i1:]])
            v = rest.mean(0) if len(rest) else self.w2v.mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    def _spoof_at(self, a, b):
        sel = (self.spoof_t >= a - 0.75) & (self.spoof_t <= b + 0.75)  # cua so 2 s co tam nam gan doan
        return float(self.spoof_s[sel].max()) if sel.any() else float(self.spoof_s[np.argmin(np.abs(self.spoof_t - (a + b) / 2))])


# ---------------- ANH PHO ----------------
def _glcm(img):
    q = np.clip((img - img.min()) / (np.ptp(img) + 1e-9) * 15, 0, 15).astype(np.uint8)
    g = graycomatrix(q, [1], [0, np.pi / 2], levels=16, symmetric=True, normed=True)
    return {k: float(graycoprops(g, k).mean()) for k in ("contrast", "homogeneity", "energy", "correlation")}


def _mod_hi_ratio(img):
    """Ti le nang luong dieu bien THOI GIAN cao (>8 Hz) trong anh pho -- giong may thuong 'min' hon."""
    if img.shape[1] < 8:
        return np.nan
    F = np.abs(np.fft.rfft(img - img.mean(axis=1, keepdims=True), axis=1)) ** 2
    mod_hz = np.fft.rfftfreq(img.shape[1], d=HOP / SR)
    return float(F[:, mod_hz > 8].sum() / (F.sum() + 1e-12))


def _nanmed(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


# ---------------- 3 loai trich dac trung ----------------
def boundary(ff: FileFeat, t: float, w: float = 0.25) -> dict:
    L = lambda arr: ff._span(arr, t - w, t)  # noqa: E731
    R = lambda arr: ff._span(arr, t, t + w)  # noqa: E731
    out = {}
    for name, arr in (("mel", ff.logmel), ("lfcc", ff.lfcc), ("mfcc", ff.mfcc)):
        out[f"B_{name}_dist"] = float(np.linalg.norm(L(arr).mean(-1) - R(arr).mean(-1)))
    out["B_rms_step"] = abs(_nanmed(L(ff.rms_db)) - _nanmed(R(ff.rms_db)))
    wl, wr = ff._span(ff.rms_db, t - 0.6, t), ff._span(ff.rms_db, t, t + 0.6)
    out["B_noisefloor_step"] = abs(np.percentile(wl, 10) - np.percentile(wr, 10)) if wl.size and wr.size else np.nan
    out["B_hf_step"] = abs(_nanmed(L(ff.hf_ratio)) - _nanmed(R(ff.hf_ratio)))
    out["B_centroid_step"] = abs(_nanmed(L(ff.centroid)) - _nanmed(R(ff.centroid))) / 1000
    out["B_cflux_peak"] = float(ff._span(ff.cflux, t - 0.03, t + 0.03).max()) if ff._span(ff.cflux, t - 0.03, t + 0.03).size else np.nan
    fl, frr = _nanmed(L(ff.f0)), _nanmed(R(ff.f0))
    out["B_f0_jump_st"] = abs(12 * np.log2(fl / frr)) if np.isfinite(fl) and np.isfinite(frr) and fl > 0 and frr > 0 else np.nan
    imgL, imgR = L(ff.logmel), R(ff.logmel)
    k = min(imgL.shape[1], imgR.shape[1])
    out["IMG_ssim_LR"] = float(structural_similarity(imgL[:, -k:], imgR[:, :k],
                                                     data_range=float(np.ptp(ff.logmel)) + 1e-6, win_size=3)) if k >= 3 else np.nan
    out["P_spk_dist_LR"] = float(1 - ff._spk_mean(t - 0.9, t - 0.1) @ ff._spk_mean(t + 0.1, t + 0.9))
    out["P_w2v_dist_LR"] = float(1 - ff._w2v_mean(t - w, t) @ ff._w2v_mean(t, t + w))
    return out


def inside(ff: FileFeat, a: float, b: float) -> dict:
    S = lambda arr: ff._span(arr, a, b)  # noqa: E731
    Rr = lambda arr: ff._rest(arr, a, b)  # noqa: E731
    out = {}
    for name, arr in (("centroid", ff.centroid / 1000), ("flatness", ff.flatness), ("rolloff", ff.rolloff / 1000),
                      ("contrast", ff.contrast), ("cpp", ff.cpp), ("hf", ff.hf_ratio)):
        out[f"I_{name}_d"] = _nanmed(S(arr)) - _nanmed(Rr(arr))
    out["I_noisefloor_d"] = (float(np.percentile(S(ff.rms_db), 10) - np.percentile(Rr(ff.rms_db), 10))
                             if Rr(ff.rms_db).size and S(ff.rms_db).size else np.nan)   # doan rong (sat cuoi tep) -> NaN
    out["I_lfcc_dist"] = float(np.linalg.norm(S(ff.lfcc).mean(-1) - Rr(ff.lfcc).mean(-1)))
    out["I_mfcc_dist"] = float(np.linalg.norm(S(ff.mfcc).mean(-1) - Rr(ff.mfcc).mean(-1)))
    out["I_dmfcc_var_ratio"] = float(np.log((S(ff.dmfcc).var() + 1e-9) / (Rr(ff.dmfcc).var() + 1e-9)))
    fs, frr = S(ff.f0), Rr(ff.f0)
    ms, mr = _nanmed(fs), _nanmed(frr)
    out["I_f0_shift_st"] = 12 * np.log2(ms / mr) if np.isfinite(ms) and np.isfinite(mr) and ms > 0 and mr > 0 else np.nan
    out["I_f0_shift_abs"] = abs(out["I_f0_shift_st"]) if np.isfinite(out["I_f0_shift_st"]) else np.nan
    # anh pho
    imgS, imgR = S(ff.logmel), Rr(ff.logmel)
    gS, gR = (_glcm(imgS), _glcm(imgR)) if imgS.shape[1] >= 4 and imgR.shape[1] >= 4 else ({}, {})
    for k in ("contrast", "homogeneity", "energy", "correlation"):
        out[f"IMG_glcm_{k}_d"] = gS.get(k, np.nan) - gR.get(k, np.nan)
    out["IMG_modhi_d"] = _mod_hi_ratio(imgS) - _mod_hi_ratio(imgR) if imgR.shape[1] >= 8 else np.nan
    # pretrained
    out["P_spk_dist_in"] = float(1 - ff._spk_mean(a, b) @ ff._spk_mean(a, b, inside=False))
    out["P_w2v_dist_in"] = float(1 - ff._w2v_mean(a, b) @ ff._w2v_mean(a, b, inside=False))
    out["P_spoof_in"] = ff._spoof_at(a, b)
    out["P_spoof_d"] = out["P_spoof_in"] - float(np.median(ff.spoof_s))
    return out


def global_feats(ff: FileFeat) -> dict:
    out = {}
    for name, arr in (("centroid", ff.centroid / 1000), ("flatness", ff.flatness), ("rolloff", ff.rolloff / 1000),
                      ("contrast", ff.contrast), ("cpp", ff.cpp), ("hf", ff.hf_ratio), ("cflux", ff.cflux)):
        if name != "cflux":
            # cflux da chuan hoa theo CHINH trung vi cua no -> trung vi luon ~1; gia tri 'tach lop'
            # truoc day (AUC 0.995) chi la epsilon 1e-9 ro do lon flux goc -> KHONG phai dac trung.
            out[f"G_{name}_med"] = _nanmed(arr)
        out[f"G_{name}_iqr"] = float(np.subtract(*np.nanpercentile(arr, [75, 25])))
    out["G_noisefloor_db"] = float(np.percentile(ff.rms_db, 10))
    out["G_dynamic_db"] = float(np.percentile(ff.rms_db, 90) - np.percentile(ff.rms_db, 10))
    quiet = ff.rms_db <= np.percentile(ff.rms_db, 20)
    out["G_quiet_flat"] = _nanmed(ff.flatness[quiet])
    out["G_quiet_specvar"] = float(np.var(ff.logmel[:, quiet], axis=1).mean()) if quiet.sum() > 2 else np.nan
    out["G_f0_med"] = _nanmed(ff.f0)
    f0v = ff.f0[np.isfinite(ff.f0)]
    out["G_f0_cv"] = float(np.std(f0v) / (np.mean(f0v) + 1e-9)) if len(f0v) > 5 else np.nan
    out["G_dmfcc_var"] = float(ff.dmfcc.var())
    g = _glcm(ff.logmel)
    for k, v in g.items():
        out[f"IMG_glcm_{k}"] = v
    out["IMG_modhi"] = _mod_hi_ratio(ff.logmel)
    out["P_spoof_med"] = float(np.median(ff.spoof_s))
    out["P_spoof_max"] = float(np.max(ff.spoof_s))
    for i, v in enumerate(ff.lfcc.mean(-1)[:12]):
        out[f"G_lfcc{i}"] = float(v)
    return out
