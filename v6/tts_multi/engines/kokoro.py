# -*- coding: utf-8 -*-
"""Bo chuyen Kokoro-82M (hexgrad/Kokoro-82M, Apache-2.0, en, 24 kHz) — GIONG CO SAN, khong nhai giong.

Env rieng D:/va_gen/envs/kokoro (kokoro 0.9.4 + misaki[en]; espeak-ng qua espeakng-loader).
Giong co san la cua mo hinh (chinh sach dong y muc 5: duoc dung). Muon thanh giong nguoi trong kho -> noi tiep VC
(openvoice_v2 / seedvc). Ung vien k chon giong co san khac nhau (VOICES_A / VOICES_B) + toc do.
params tuy chon: voice, speed, accent ('a' My, 'b' Anh).
"""
from __future__ import annotations

import numpy as np

from v6.tts_multi.engines.base import BaseEngine, pick

VOICES_A = ["af_heart", "am_michael", "af_bella", "am_fenrir", "af_nicole", "am_puck", "af_sarah", "am_echo"]
VOICES_B = ["bf_emma", "bm_george", "bf_isabella", "bm_fable"]
SPEED = [1.0, 0.95, 1.05, 1.0]


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        from kokoro import KPipeline
        self._KP = KPipeline
        self.pipes = {}
        self.sr = 24000

    def _pipe(self, accent: str):
        if accent not in self.pipes:
            self.pipes[accent] = self._KP(lang_code=accent, repo_id="hexgrad/Kokoro-82M", device=self.device)
        return self.pipes[accent]

    def synth(self, job: dict, k: int, seed: int):
        p = job.get("params", {})
        voice = p.get("voice")
        accent = p.get("accent") or ("b" if voice and voice[:1] == "b" else "a")
        if not voice:
            pool = VOICES_A if accent == "a" else VOICES_B
            voice = pool[(seed + k) % len(pool)]
        speed = float(p.get("speed", pick(SPEED, k)))
        parts = [np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, np.float32)
                 for _, _, a in self._pipe(accent)(job["text"], voice=voice, speed=speed) if a is not None]
        if not parts:
            raise RuntimeError("Kokoro khong sinh am thanh")
        params = {"voice": voice, "accent": accent, "speed": speed, "cloned": False}
        return np.concatenate(parts), self.sr, params
