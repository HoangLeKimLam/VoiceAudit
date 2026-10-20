# -*- coding: utf-8 -*-
"""Bo chuyen MMS-TTS (facebook/mms-tts-vie, VITS, CC-BY-NC-4.0) qua transformers trong .venv chinh.

KHONG nhai giong (mot giong co dinh) -> bac 'giong may thap' ; muon giong nguoi trong kho thi noi tiep may VC
(openvoice_v2 / seedvc / freevc24) voi target_ref_wav la ban ghi KHAC cua cung nguoi.
Ung vien k doi noise_scale / noise_scale_duration / speaking_rate (VITS khong tat dinh -> seed quan trong).
"""
from __future__ import annotations

import numpy as np

from v6.tts_multi.engines.base import BaseEngine, pick

NOISE = [0.667, 0.5, 0.8, 0.6]
NOISE_W = [0.8, 0.6, 1.0, 0.7]
RATE = [1.0, 0.95, 1.05, 1.0]


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel
        mid = self.opts.get("model_id", "facebook/mms-tts-vie")
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = VitsModel.from_pretrained(mid).to(self.device).eval()
        self.sr = int(self.model.config.sampling_rate)
        self._torch = torch

    def synth(self, job: dict, k: int, seed: int):
        p = job.get("params", {})
        m = self.model
        m.noise_scale = float(p.get("noise_scale", pick(NOISE, k)))
        m.noise_scale_duration = float(p.get("noise_scale_duration", pick(NOISE_W, k)))
        m.speaking_rate = float(p.get("speaking_rate", pick(RATE, k)))
        ids = self.tok(job["text"].lower(), return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            y = m(**ids).waveform[0].float().cpu().numpy()
        params = {"noise_scale": m.noise_scale, "noise_scale_duration": m.noise_scale_duration,
                  "speaking_rate": m.speaking_rate, "cloned": False, "voice": "mms-vie-default"}
        return np.asarray(y, np.float32), self.sr, params
