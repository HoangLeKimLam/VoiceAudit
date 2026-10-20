# -*- coding: utf-8 -*-
"""Bo chuyen Resemble-Enhance 0.0.1 (ResembleAI/resemble-enhance, MIT) — kind denoise_ai, vai tro HELD-OUT
(khu on AI ho khac DeepFilterNet: UNet denoiser + bo lam dep latent CFM, 44,1 kHz).

Env rieng D:/va_gen/envs/resenh (cai --no-deps: METADATA ghim deepspeed 0.12.4 + torch 2.1.1; deepspeed CHI dung khi
hoc -> gia lap module truoc khi nap). Trong so: D:/va_gen/models/resemble_enhance/enhancer_stage2 (tai bang curl
trong setup_envs.sh, KHONG git clone vao site-packages nhu download() goc).
Viec: {job_id, src_wav, seed[, n_candidates, params]}. Dau ra tra ve O SR CUA NGUON, dung do dai nguon (khop thoi
gian voi host); params ghi model_sr=44100. Ung vien k (luoi MODES): enhance lambd 0.9 / chi denoise / enhance
lambd 0.5 / enhance nfe 64 tau 0.7. params tuy chon: mode (enhance|denoise), nfe, lambd, tau, solver.
CFM ngau nhien -> seed (run_engine dat torch.manual_seed) quyet dinh dau ra.
Che do enhance tai tao pha (xcorr voi nguon ~0,3) va lech vai ms: adapter bu lag (best_lag, |lag| <= 50 ms) va ghi
params.waveform_aligned=False -> WP7b chi dung enhance cho 02 CA TEP (hoac noi doan co crossfade); 'denoise' giu pha.
Thu nhanh: python -m v6.tts_multi.engines.smoke run --engine resenh
"""
from __future__ import annotations

import dataclasses
import os
import pathlib
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf

from v6.tts_multi.engines.base import MODELS, BaseEngine, ascii_wav, pick, resample

RUN_DIR = MODELS + "/resemble_enhance/enhancer_stage2"
CKPT_REL = "ds/G/default/mp_rank_00_model_states.pt"
MODES = [
    {"mode": "enhance", "nfe": 32, "lambd": 0.9, "tau": 0.5, "solver": "midpoint"},
    {"mode": "denoise"},
    {"mode": "enhance", "nfe": 32, "lambd": 0.5, "tau": 0.5, "solver": "midpoint"},
    {"mode": "enhance", "nfe": 64, "lambd": 0.9, "tau": 0.7, "solver": "midpoint"},
]


def _stub_deepspeed() -> None:
    """Module deepspeed gia (chi ten) de nap duoc resemble_enhance.*.train / utils khi suy luan."""
    if "deepspeed" in sys.modules:
        return
    ds = types.ModuleType("deepspeed")
    acc = types.ModuleType("deepspeed.accelerator")
    rt = types.ModuleType("deepspeed.runtime")
    eng = types.ModuleType("deepspeed.runtime.engine")
    utl = types.ModuleType("deepspeed.runtime.utils")

    class DeepSpeedConfig:  # noqa: D401 - chi de nap module
        def __init__(self, *a, **kw):
            raise RuntimeError("deepspeed gia: chi suy luan, khong hoc")

    class DeepSpeedEngine:
        pass

    def _no(*a, **kw):
        raise RuntimeError("deepspeed gia: chi suy luan, khong hoc")

    ds.DeepSpeedConfig, ds.init_distributed = DeepSpeedConfig, _no
    acc.get_accelerator = _no
    eng.DeepSpeedEngine = DeepSpeedEngine
    utl.clip_grad_norm_ = _no
    ds.accelerator, ds.runtime, rt.engine, rt.utils = acc, rt, eng, utl
    sys.modules.update({"deepspeed": ds, "deepspeed.accelerator": acc, "deepspeed.runtime": rt,
                        "deepspeed.runtime.engine": eng, "deepspeed.runtime.utils": utl})


MAX_LAG_SEC = 0.05          # do lech thoi gian toi da tim (enhance lech ~6,6 ms o smoke 11/10)
LAG_WIN_SEC = 10.0


def best_lag(ref: np.ndarray, out: np.ndarray, sr: int, max_lag_sec: float = MAX_LAG_SEC) -> tuple[int, float]:
    """Lag L (mau) lam cuc dai tuong quan chuan hoa sum ref[n] * out[n + L], |L| <= max_lag; tra (L, xcorr).
    Dung FFT tren toi da LAG_WIN_SEC dau."""
    n = min(len(ref), len(out), int(sr * LAG_WIN_SEC))
    if n < 16:
        return 0, 0.0
    a = ref[:n] - ref[:n].mean()
    b = out[:n] - out[:n].mean()
    m = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, m)) * np.fft.rfft(b, m), m)     # cc[L] = sum a[n] b[n+L]
    ml = min(int(sr * max_lag_sec), n - 1)
    lags = np.concatenate([np.arange(0, ml + 1), np.arange(-ml, 0)])
    vals = np.concatenate([cc[: ml + 1], cc[m - ml:]])
    i = int(np.argmax(vals))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) + 1e-12
    return int(lags[i]), float(vals[i] / den)


def shift(out: np.ndarray, lag: int) -> np.ndarray:
    """out_moi[n] = out[n + lag] (dem 0 o bien), giu do dai."""
    if lag == 0:
        return out
    z = np.zeros_like(out)
    if lag > 0:
        z[: len(out) - lag] = out[lag:]
    else:
        z[-lag:] = out[: len(out) + lag]
    return z


def _load_state(torch, path: str, device: str) -> dict:
    """state_dict 'module' cua checkpoint deepspeed; nap thang len device. Thu weights_only truoc; checkpoint chinh
    thuc (nguon tin cay, MIT) co doi tuong khac tensor thi moi noi long."""
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        obj = torch.load(path, map_location=device, weights_only=False)
    return obj["module"]


class Engine(BaseEngine):
    kind = "denoise_ai"

    def load(self) -> None:
        _stub_deepspeed()
        import torch
        from resemble_enhance.enhancer.hparams import HParams
        from resemble_enhance.enhancer.enhancer import Enhancer
        from resemble_enhance.inference import inference
        run_dir = Path(self.opts.get("run_dir", RUN_DIR))
        ckpt = run_dir / CKPT_REL
        if not ckpt.exists():
            raise FileNotFoundError(f"thieu {ckpt} (bash v6/tts_multi/setup_envs.sh resenh)")
        posix = pathlib.PosixPath                  # hparams.yaml luu fg_dir/bg_dir/rir_dir (duong du lieu hoc) la PosixPath
        pathlib.PosixPath = pathlib.WindowsPath if os.name == "nt" else posix
        try:
            hp = HParams.load(run_dir)
        finally:
            pathlib.PosixPath = posix
        if getattr(hp, "denoiser_run_dir", None) is not None:     # duong luc hoc; trong so denoiser nam san trong G
            hp = dataclasses.replace(hp, denoiser_run_dir=None)
        self._torch, self._inference = torch, inference
        self.model = Enhancer(hp)
        self.model.load_state_dict(_load_state(torch, str(ckpt), self.device))
        self.model = self.model.eval().to(self.device)
        self.msr = int(hp.wav_rate)

    def convert(self, job: dict, k: int, seed: int):
        p = {**pick(MODES, k), **job.get("params", {})}
        y, sr = sf.read(ascii_wav(job["src_wav"]), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        t = self._torch.from_numpy(np.ascontiguousarray(y))
        with self._torch.inference_mode():
            if p["mode"] == "denoise":
                out, osr = self._inference(model=self.model.denoiser, dwav=t, sr=sr, device=self.device)
            elif p["mode"] == "enhance":
                self.model.configurate_(nfe=int(p["nfe"]), solver=str(p["solver"]), lambd=float(p["lambd"]),
                                        tau=float(p["tau"]))
                out, osr = self._inference(model=self.model, dwav=t, sr=sr, device=self.device)
            else:
                raise ValueError(f"mode khong hop le: {p['mode']}")
        z = resample(out.detach().float().cpu().numpy().astype(np.float32), int(osr), sr)
        z = np.pad(z, (0, max(0, len(y) - len(z))))[: len(y)]      # dung do dai nguon
        lag, xc = best_lag(y, z, sr)                                 # enhance: bu lech thoi gian (~ms) so voi nguon
        z = shift(z, lag)
        keys = ("mode", "nfe", "lambd", "tau", "solver")
        params = {**{key: p[key] for key in keys if key in p}, "model": "ResembleAI/resemble-enhance enhancer_stage2",
                  "model_sr": int(osr), "lag_comp_samples": lag, "xcorr_src": round(xc, 3),
                  # enhance tai tao pha + dai cao: khop thoi gian sau bu lag nhung KHONG khop mau -> chi dung ca tep
                  # (02 whole) hoac noi doan co crossfade; denoise giu pha (xcorr ~0,99)
                  "waveform_aligned": p["mode"] == "denoise"}
        return z, sr, params

