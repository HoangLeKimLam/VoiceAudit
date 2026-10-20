# -*- coding: utf-8 -*-
"""Bo chuyen DeepFilterNet3 (deepfilternet 0.5.6, MIT/Apache-2.0) — kind denoise_ai cho thao tac 'khu on'.

Viec: {job_id, src_wav, seed[, n_candidates, params]}. Mo hinh chay o 48 kHz; dau ra tra ve O SR CUA NGUON
(giu khop thoi gian voi tep host; ghi model_sr=48000 trong params). Ung vien k doi atten_lim_db
(None = khu het, 18 / 12 dB = khu nhe hon) -> nhieu muc khu on khac nhau.
Ban 0.5.6 nap 'torchaudio.backend.common.AudioMetaData' (bi bo o torchaudio moi) -> chen module gia truoc khi nap df.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import soundfile as sf

from v6.tts_multi.engines.base import MODELS, BaseEngine, ascii_wav, pick, resample

ATTEN = [None, 18.0, 12.0, 24.0]


def _shim_torchaudio() -> None:
    try:
        import torchaudio.backend.common  # noqa: F401
        return
    except Exception:  # noqa: BLE001 - torchaudio moi khong con module nay
        pass
    import torchaudio
    backend = types.ModuleType("torchaudio.backend")
    common = types.ModuleType("torchaudio.backend.common")

    class AudioMetaData:  # chi can ten de nap duoc df.io (ta khong dung df.io.load_audio)
        def __init__(self, sample_rate=0, num_frames=0, num_channels=0, bits_per_sample=0, encoding=""):
            self.sample_rate, self.num_frames, self.num_channels = sample_rate, num_frames, num_channels
            self.bits_per_sample, self.encoding = bits_per_sample, encoding

    common.AudioMetaData = AudioMetaData
    backend.common = common
    sys.modules["torchaudio.backend"] = backend
    sys.modules["torchaudio.backend.common"] = common
    torchaudio.backend = backend


class Engine(BaseEngine):
    kind = "denoise_ai"

    def load(self) -> None:
        _shim_torchaudio()
        import torch
        import importlib
        dfe = importlib.import_module("df.enhance")      # "import df.enhance as" tra ve HAM enhance (df/__init__ che)
        dfu = importlib.import_module("df.utils")
        self._torch = torch
        self._enhance = dfe.enhance
        cache = MODELS + "/deepfilternet"            # mac dinh appdirs -> C:; dua ve D:
        dfu.get_cache_dir = dfe.get_cache_dir = lambda: cache
        self.model, self.df_state, _ = dfe.init_df(log_level="ERROR", log_file=None)
        self.model = self.model.to(self.device).eval() if self.device == "cuda" else self.model
        self.msr = int(self.df_state.sr())

    def convert(self, job: dict, k: int, seed: int):
        p = job.get("params", {})
        atten = p.get("atten_lim_db", pick(ATTEN, k))
        y, sr = sf.read(ascii_wav(job["src_wav"]), dtype="float32")
        x = resample(y, sr, self.msr)
        t = self._torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        with self._torch.no_grad():
            out = self._enhance(self.model, self.df_state, t, atten_lim_db=atten)
        z = resample(out.squeeze(0).cpu().numpy().astype(np.float32), self.msr, sr)[: len(y)]
        params = {"atten_lim_db": atten, "model": "DeepFilterNet3", "model_sr": self.msr}
        return z, sr, params
