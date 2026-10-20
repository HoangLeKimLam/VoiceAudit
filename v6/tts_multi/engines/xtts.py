# -*- coding: utf-8 -*-
"""Bo chuyen XTTS (coqui-tts 0.27.5, .venv chinh): viXTTS (vi, models/viXTTS) va XTTS-v2 (en, coqui/XTTS-v2).

Nap TRUC TIEP Xtts tu thu muc checkpoint (nhu vaudit/tts.py, nhung KHONG dung trang thai chung cua no) va mo
cac nut chinh: temperature, top_p, speed, danh sach tep tham chieu (nhieu tep cung nguoi -> latents on dinh hon).
Ung vien k lay tham so tu luoi (TEMP / TOP_P / SPEED) -> cac ung vien khac nhau ve ngu dieu.
viXTTS: the card bao kem voi cau < 10 tu -> WP7b nen tong hop cau mang (carrier) >= 10 tu roi cat.
Thu nhanh: python -m v6.tts_multi.engines.smoke run --engine vixtts
"""
from __future__ import annotations

import os
import re

import numpy as np
import soundfile as sf

from v6.tts_multi.engines.base import BaseEngine, as_list, ascii_wav, free_memory, pick

TEMP = [0.65, 0.75, 0.55, 0.85]
TOP_P = [0.85, 0.80, 0.90, 0.85]
SPEED = [1.0, 0.95, 1.05, 1.0]
LATIN = {"ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "nl", "pl", "pt", "ru", "tr", "zh", "ko", "ja"}


def _patch_coqui(vi_limit: int = 224) -> None:
    """Doc am thanh bang soundfile (tranh torchcodec/ffmpeg), them 'vi' vao bo tach tu XTTS."""
    import librosa
    import torch
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer, multilingual_cleaners
    from TTS.tts.models import xtts as xtts_module

    def _load_audio(audiopath, sampling_rate):
        y, sr = sf.read(audiopath, always_2d=True)
        y = y.mean(axis=1)
        if sr != sampling_rate:
            y = librosa.resample(y.astype(float), orig_sr=sr, target_sr=sampling_rate)
        return torch.from_numpy(np.clip(y, -1, 1).astype(np.float32)).unsqueeze(0)

    def _preprocess(self, txt, lang):
        if lang == "vi":
            return re.sub(r"\s+", " ", txt).strip()
        if lang in LATIN:
            return multilingual_cleaners(txt, lang)
        raise NotImplementedError(f"Language '{lang}' is not supported.")

    xtts_module.load_audio = _load_audio
    if not getattr(VoiceBpeTokenizer, "_va_patched", False):
        VoiceBpeTokenizer.preprocess_text = _preprocess
        VoiceBpeTokenizer.char_limits = {**VoiceBpeTokenizer(None).char_limits, "vi": vi_limit}
        VoiceBpeTokenizer._va_patched = True


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        import torch
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        _patch_coqui()
        mdir = self.opts["model_dir"]
        cfg = XttsConfig()
        cfg.load_json(os.path.join(mdir, "config.json"))
        import TTS.tts.models.xtts as xmod
        orig_load = xmod.load_fsspec
        if self.device == "cuda":
            # Khoi tao + nap checkpoint THANG len GPU (torch.load doc tung tensor roi chuyen ngay): tranh giu ~1,9 GB
            # ban CPU -> truoc day tien trinh chet 0xC0000005 trong torch.load khi commit RAM may can.
            with torch.device("cuda"):
                self.model = Xtts.init_from_config(cfg)
            xmod.load_fsspec = lambda path, map_location=None, **kw: orig_load(path, map_location="cuda", **kw)
        else:
            self.model = Xtts.init_from_config(cfg)
        try:
            self.model.load_checkpoint(cfg, checkpoint_dir=mdir, eval=True, use_deepspeed=False)
        finally:
            xmod.load_fsspec = orig_load
        if self.device == "cuda":
            self.model.cuda()
        free_memory()
        self.out_sr = int(getattr(cfg.audio, "output_sample_rate", 24000))
        self._lat_cache = {}
        self._torch = torch

    def _latents(self, refs: list, cond_len: int):
        key = (tuple(refs), cond_len)
        if key not in self._lat_cache:
            if len(self._lat_cache) > 32:
                self._lat_cache.clear()
            self._lat_cache[key] = self.model.get_conditioning_latents(audio_path=refs, gpt_cond_len=cond_len,
                                                                       max_ref_length=30)
        return self._lat_cache[key]

    def synth(self, job: dict, k: int, seed: int):
        refs = [ascii_wav(p) for p in as_list(job.get("ref_wav"))]
        if not refs:
            raise ValueError("XTTS can ref_wav (nhai giong)")
        p = job.get("params", {})
        cond_len = int(p.get("gpt_cond_len", self.opts.get("gpt_cond_len", 12)))
        temp = float(p.get("temperature", pick(TEMP, k)))
        top_p = float(p.get("top_p", pick(TOP_P, k)))
        speed = float(p.get("speed", pick(SPEED, k)))
        gpt_lat, spk = self._latents(refs, cond_len)
        with self._torch.inference_mode():
            out = self.model.inference(job["text"], job["lang"], gpt_lat, spk, temperature=temp, top_p=top_p,
                                       speed=speed, repetition_penalty=float(p.get("repetition_penalty", 10.0)),
                                       enable_text_splitting=bool(p.get("split", False)))
        params = {"temperature": temp, "top_p": top_p, "speed": speed, "gpt_cond_len": cond_len,
                  "n_refs": len(refs), "cloned": True, "model": self.opts.get("model_id", "")}
        return np.asarray(out["wav"], dtype=np.float32), self.out_sr, params
