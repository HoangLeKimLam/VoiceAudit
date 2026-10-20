# -*- coding: utf-8 -*-
"""Bo chuyen ZipVoice (env rieng D:/va_gen/envs/zipvoice, ma nguon D:/va_gen/src/ViZipvoice @7c09abc):
  vizipvoice       : contextboxai/ViZipvoice (vi, Apache-2.0, tokenizer ky tu, chuan hoa soe-vinorm)
  zipvoice_distill : k2-fsa/ZipVoice zipvoice_distill (en, EmiliaTokenizer qua piper_phonemize; trong so hoc tu
                     Emilia -> giay phep trong so khong ro, coi la phi thuong mai)
Vocoder Vocos 24 kHz tu D:/va_gen/models/vocos-mel-24khz. Ung vien k doi num_step / guidance_scale / speed / t_shift.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

from v6.tts_multi.engines.base import VA_GEN, BaseEngine, as_list, concat_refs, pick

SRC = VA_GEN + "/src/ViZipvoice"
GRID = {
    "vizipvoice": {"num_step": [16, 16, 12, 20], "guidance_scale": [1.0, 1.2, 0.9, 1.1]},
    "zipvoice_distill": {"num_step": [8, 8, 6, 10], "guidance_scale": [3.0, 3.5, 2.5, 3.0]},
}
SPEED = [1.0, 0.95, 1.05, 1.0]
T_SHIFT = [0.5, 0.6, 0.45, 0.5]


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        import torch
        self._torch = torch
        self.variant = self.opts["variant"]
        mdir = self.opts["model_dir"]
        voc = self.opts.get("vocoder_path")
        if self.variant == "vizipvoice":
            from zipvoice.vizipvoice import ViZipVoiceTTS
            self.tts = ViZipVoiceTTS(model_dir=mdir, checkpoint_name=self.opts["checkpoint_name"], vocoder_path=voc,
                                     device=self.device, use_fp16=self.device == "cuda", num_threads=2)
            self.sr = self.tts.sampling_rate
            return
        from zipvoice.bin.infer_zipvoice import get_vocoder
        from zipvoice.models.zipvoice_distill import ZipVoiceDistill
        from zipvoice.tokenizer.tokenizer import EmiliaTokenizer
        from zipvoice.utils.checkpoint import load_checkpoint
        from zipvoice.utils.feature import VocosFbank
        self.tokenizer = EmiliaTokenizer(token_file=os.path.join(mdir, "tokens.txt"))
        with open(os.path.join(mdir, "model.json"), encoding="utf-8") as fh:
            mcfg = json.load(fh)
        self.model = ZipVoiceDistill(**mcfg["model"], vocab_size=self.tokenizer.vocab_size, pad_id=self.tokenizer.pad_id)
        load_checkpoint(filename=os.path.join(mdir, "model.pt"), model=self.model, strict=True)
        self.dev = torch.device(self.device)
        self.model.to(self.dev).eval()
        self.vocoder = get_vocoder(voc).to(self.dev).eval()
        self.fbank = VocosFbank()
        self.sr = int(mcfg["feature"]["sampling_rate"])

    def synth(self, job: dict, k: int, seed: int):
        refs = as_list(job.get("ref_wav"))
        if not refs or not job.get("ref_text"):
            raise ValueError("ZipVoice can ref_wav + ref_text")
        ref = concat_refs(refs, sr=24000)
        p = job.get("params", {})
        g = GRID[self.variant]
        num_step = int(p.get("num_step", pick(g["num_step"], k)))
        gs = float(p.get("guidance_scale", pick(g["guidance_scale"], k)))
        speed = float(p.get("speed", pick(SPEED, k)))
        t_shift = float(p.get("t_shift", pick(T_SHIFT, k)))
        tmpd = tempfile.mkdtemp(prefix="zv_", dir=VA_GEN + "/tmp")
        out = os.path.join(tmpd, "o.wav")
        try:
            if self.variant == "vizipvoice":
                self.tts.synthesize(prompt_wav=ref, prompt_text=job["ref_text"], text=job["text"], output_path=out,
                                    num_step=num_step, guidance_scale=gs, speed=speed, t_shift=t_shift, seed=seed)
            else:
                from lhotse.utils import fix_random_seed
                from zipvoice.bin.infer_zipvoice import generate_sentence
                fix_random_seed(seed)
                with self._torch.inference_mode():
                    generate_sentence(save_path=out, prompt_text=job["ref_text"], prompt_wav=ref, text=job["text"],
                                      model=self.model, vocoder=self.vocoder, tokenizer=self.tokenizer,
                                      feature_extractor=self.fbank, device=self.dev, num_step=num_step,
                                      guidance_scale=gs, speed=speed, t_shift=t_shift, sampling_rate=self.sr)
            y, sr = sf.read(out, dtype="float32")
        finally:
            for f in os.listdir(tmpd):
                try:
                    os.remove(os.path.join(tmpd, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmpd)
            except OSError:
                pass
        params = {"num_step": num_step, "guidance_scale": gs, "speed": speed, "t_shift": t_shift, "cloned": True,
                  "model": self.opts.get("model_id", ""), "n_refs": len(refs)}
        return np.asarray(y, np.float32), int(sr), params
