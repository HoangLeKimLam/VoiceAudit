# -*- coding: utf-8 -*-
"""Bo chuyen StyleTTS2 LibriTTS (goi pip styletts2 0.1.6; ma MIT; trong so yl4579/StyleTTS2-LibriTTS) — HELD-OUT en.

Env rieng D:/va_gen/envs/styletts2 (am vi bang gruut, khong can espeak). Trong so tai qua cached_path ve
CACHED_PATH_CACHE_ROOT (D:/hf-cache/cached_path); nltk punkt/punkt_tab ve NLTK_DATA (D:/hf-cache/nltk_data).
Dieu khoan mo hinh: phai cong bo la giong tong hop / chi dung giong duoc phep KHI CONG BO — o day chi noi bo,
nhai nguoi LibriSpeech (LibriTTS lay tu LibriSpeech). Ung vien k doi alpha / beta / diffusion_steps / embedding_scale.
"""
from __future__ import annotations

import os

import numpy as np

from v6.tts_multi.engines.base import BaseEngine, as_list, concat_refs, pick

ALPHA = [0.3, 0.2, 0.4, 0.3]
BETA = [0.7, 0.6, 0.8, 0.7]
STEPS = [5, 10, 5, 8]
EMB = [1.0, 1.0, 1.5, 1.2]


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        nd = os.environ.get("NLTK_DATA", "D:/hf-cache/nltk_data")
        os.makedirs(nd, exist_ok=True)
        os.makedirs(os.environ.get("CACHED_PATH_CACHE_ROOT", "D:/hf-cache/cached_path"), exist_ok=True)
        import nltk
        for pkg in ("punkt", "punkt_tab"):
            nltk.download(pkg, download_dir=nd, quiet=True)
        import torch
        from styletts2 import tts
        # Checkpoint chinh thuc (yl4579, HF/GitHub) luu bang pickle cu co 'getattr' -> torch >= 2.6 chan khi
        # weights_only=True. Chi noi long trong luc nap mo hinh nay roi tra lai (nguon tin cay, ghi trong notes).
        orig = torch.load
        torch.load = lambda *a, **kw: orig(*a, **{**kw, "weights_only": kw.get("weights_only", False)})
        try:
            self.tts = tts.StyleTTS2()
        finally:
            torch.load = orig
        self.device = self.tts.device
        self.sr = 24000
        self._style = {}

    def synth(self, job: dict, k: int, seed: int):
        refs = as_list(job.get("ref_wav"))
        if not refs:
            raise ValueError("StyleTTS2 can ref_wav (nhai giong)")
        ref = concat_refs(refs, sr=24000)
        if ref not in self._style:
            self._style = {ref: self.tts.compute_style(ref)}
        p = job.get("params", {})
        alpha = float(p.get("alpha", pick(ALPHA, k)))
        beta = float(p.get("beta", pick(BETA, k)))
        steps = int(p.get("diffusion_steps", pick(STEPS, k)))
        emb = float(p.get("embedding_scale", pick(EMB, k)))
        y = self.tts.inference(job["text"], ref_s=self._style[ref], output_sample_rate=self.sr, alpha=alpha, beta=beta,
                               diffusion_steps=steps, embedding_scale=emb)
        params = {"alpha": alpha, "beta": beta, "diffusion_steps": steps, "embedding_scale": emb, "cloned": True,
                  "model": "yl4579/StyleTTS2-LibriTTS epochs_2nd_00020"}
        return np.asarray(y, np.float32), self.sr, params
