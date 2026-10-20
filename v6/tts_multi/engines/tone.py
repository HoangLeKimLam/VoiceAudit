# -*- coding: utf-8 -*-
"""May GIA (khong mo hinh) de kiem vong chay run_engine: sinh am 'giong noi' tong hop tu f0 + nhieu,
do dai ty le voi so tu. Chi dung trong test_engines.py (khong co trong engines.json).
opts.fail_on: job_id gay loi co y (kiem ghi .err.json); opts.oom_device: gia lap het bo nho khi nap tren may do."""
from __future__ import annotations

import numpy as np

from v6.tts_multi.engines.base import BaseEngine


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        if self.opts.get("oom_device") == self.device:      # gia lap het bo nho GPU khi nap (kiem fallback CPU)
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB (gia)")
        self.sr = 16000
        self.loaded = True

    def _make(self, n_sec: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        t = np.arange(int(n_sec * self.sr)) / self.sr
        f0 = 120 + 20 * np.sin(2 * np.pi * 3 * t)
        y = 0.3 * np.sin(2 * np.pi * np.cumsum(f0) / self.sr) + 0.01 * rng.standard_normal(len(t))
        return y.astype(np.float32)

    def synth(self, job: dict, k: int, seed: int):
        if job["job_id"] == self.opts.get("fail_on"):
            raise RuntimeError("loi co y")
        n = max(1, len(job["text"].split()))
        return self._make(0.3 * n, seed), self.sr, {"cloned": False, "n_words": n}

    def convert(self, job: dict, k: int, seed: int):
        return self._make(1.0, seed), self.sr, {}
