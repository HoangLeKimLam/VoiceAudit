# -*- coding: utf-8 -*-
"""Bo chuyen VieNeu-TTS v3 Turbo (pnnbao-ump/VieNeu-TTS-v3-Turbo, Apache-2.0, vi+en, 48 kHz) — vai tro HELD-OUT.

Env rieng D:/va_gen/envs/vieneu (vieneu 3.8.3, KHONG cai extra [watermark]; goi infer(apply_watermark=False)
-> ghi watermark=false). GPU: backend pytorch; CPU: ONNX. Nhai giong tu ref_audio (thu vien tu khu on + cat <= 8 s).
Ung vien k doi temperature / top_p / top_k.
"""
from __future__ import annotations

import numpy as np

from v6.tts_multi.engines.base import BaseEngine, as_list, concat_refs, pick

TEMP = [0.8, 0.7, 0.9, 0.75]
TOP_P = [0.95, 0.9, 0.95, 0.92]
TOP_K = [25, 20, 40, 25]


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        from vieneu import Vieneu
        backend = "pytorch" if self.device == "cuda" else "onnx"
        self.tts = Vieneu(mode="v3turbo", backbone_repo=self.opts.get("model_id", "pnnbao-ump/VieNeu-TTS-v3-Turbo"),
                          device=self.device, backend=backend, max_streams=1, max_batch_size=4)
        self.sr = int(getattr(self.tts, "sample_rate", 48000))
        self.backend = backend

    def synth(self, job: dict, k: int, seed: int):
        refs = as_list(job.get("ref_wav"))
        if not refs:
            raise ValueError("VieNeu can ref_wav (nhai giong)")
        ref = concat_refs(refs, sr=24000)
        p = job.get("params", {})
        temp = float(p.get("temperature", pick(TEMP, k)))
        top_p = float(p.get("top_p", pick(TOP_P, k)))
        top_k = int(p.get("top_k", pick(TOP_K, k)))
        denoise = bool(p.get("denoise_ref", True))
        y = self.tts.infer(job["text"], ref_audio=ref, denoise=denoise, temperature=temp, top_p=top_p, top_k=top_k,
                           apply_watermark=False)
        params = {"temperature": temp, "top_p": top_p, "top_k": top_k, "denoise_ref": denoise, "watermark": False,
                  "backend": self.backend, "cloned": True, "model": self.opts.get("model_id", "")}
        return np.asarray(y, np.float32), self.sr, params
