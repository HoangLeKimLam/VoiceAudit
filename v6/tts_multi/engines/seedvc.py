# -*- coding: utf-8 -*-
"""Bo chuyen Seed-VC (Plachtaa/seed-vc @51383ef, GPL-3.0, repo DA LUU TRU) — doi giong khong can hoc them.

Env rieng D:/va_gen/envs/seedvc (torch 2.8.0+cu128 thay cho 2.4.0 ghim trong requirements). Mo hinh offline
DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned (22,05 kHz) + whisper-small + CAMPPlus + BigVGAN;
KHONG dung che do --f0-condition (mo hinh hat 44,1 kHz khac). Nap 'inference.py' cua repo lam module
(cwd = thu muc repo vi no dung duong tuong doi ./checkpoints), roi lap lai vong sinh cua main() cho tung viec.
Ung vien k doi diffusion_steps / inference_cfg_rate.

GIOI HAN DO DAI (sua 11/10): DiT nhan MOT cua so 30 s = 2580 khung mel (22050//256*30) cho CA tham chieu (prompt)
VA nguon. Ban truoc bao loi khi nguon + tham chieu > 30 s (vd tham chieu 10 s -> nguon toi da ~20 s). Nay:
  1) tham chieu duoc cat con min(MAX_REF_SEC, 30 - nguon - le) nhung khong ngan hon MIN_REF_SEC -> nguon toi
     ~24,8 s di qua MOT lan (khong co moi noi);
  2) nguon dai hon: chia khung theo cua so con lai, chong OVERLAP_FRAMES khung, noi bang crossfade cos^2
     (dung cach cua main() goc); whisper > 30 s cung chia doan chong 5 s nhu goc. Khong con gioi han do dai nguon.
params ghi ref_sec_used, n_chunks (>1 = co moi noi crossfade 16 khung ~0,19 s; WP7b nen biet khi cat ghep).
Kiem so hoc (khong can mo hinh): python -m v6.tts_multi.test_engines (test_seedvc_window).
"""
from __future__ import annotations

import importlib
import math
import os
import sys
import types

import numpy as np

from v6.tts_multi.engines.base import VA_GEN, BaseEngine, as_list, ascii_wav, concat_refs, pick

SRC = VA_GEN + "/src/seed-vc"
STEPS = [30, 25, 40, 30]
CFG_RATE = [0.7, 0.6, 0.8, 0.7]
SR, HOP = 22050, 256
CTX_SEC = 30
MAX_REF_SEC = 25           # nhu goc
MIN_REF_SEC = 5.0          # tham chieu ngan hon lam giam do giong; nguon dai hon (30 - 5 - le) thi chia khung
MARGIN_FRAMES = 8          # du phong lech so khung mel (padding cua to_mel)
OVERLAP_FRAMES = 16        # nhu goc
WHISPER_SEC, WHISPER_OVL = 30, 5
VOC_PIECE, VOC_CTX = 430, 32   # giai ma BigVGAN theo mieng ~5 s + 32 khung (~0,37 s) ngu canh moi ben


def max_ctx_frames(sr: int = SR, hop: int = HOP) -> int:
    return sr // hop * CTX_SEC


def fit_ref_seconds(src_sec: float, ref_sec: float, sr: int = SR, hop: int = HOP) -> float:
    """Do dai tham chieu dung that: <= MAX_REF_SEC, <= ref_sec co san, va du ngan de nguon lot vua cua so neu
    duoc (nhung >= MIN_REF_SEC). Tra so giay."""
    fps = sr / hop
    room = (max_ctx_frames(sr, hop) - MARGIN_FRAMES - math.ceil(src_sec * fps) - 1) / fps
    return float(max(0.0, min(ref_sec, MAX_REF_SEC, max(MIN_REF_SEC, room))))


def chunk_spans(n_cond: int, window: int, overlap: int = OVERLAP_FRAMES) -> list[tuple[int, int]]:
    """Cac doan [s, e) khung nguon, giong vong while cua main() goc (buoc = do dai doan - overlap)."""
    if window <= overlap:
        raise ValueError(f"cua so nguon {window} khung <= overlap {overlap}: tham chieu qua dai")
    spans, s = [], 0
    while s < n_cond:
        e = min(s + window, n_cond)
        spans.append((s, e))
        if e >= n_cond:
            break
        s = e - overlap
    return spans


def crossfade(prev_tail: np.ndarray, chunk: np.ndarray, overlap: int) -> np.ndarray:
    """Nhu crossfade() goc: chunk[:overlap] tron voi duoi doan truoc (cos^2). Tra ban moi (khong sua chunk)."""
    out = np.array(chunk, dtype=np.float32, copy=True)
    fo = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fi = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    n = min(overlap, len(out))
    out[:n] = out[:n] * fi[:n] + (prev_tail[-overlap:] * fo)[:n]
    return out


def stitch(waves: list, overlap_wave: int) -> np.ndarray:
    """Noi song cac doan: doan k > 0 bat dau bang overlap_wave mau trung voi duoi doan truoc (tron crossfade)."""
    if len(waves) == 1:
        return np.asarray(waves[0], np.float32)
    parts, prev = [], None
    for i, w in enumerate(waves):
        w = np.asarray(w, np.float32)
        last = i == len(waves) - 1
        body = w if last else w[:-overlap_wave]
        parts.append(body if prev is None else crossfade(prev, body, overlap_wave))
        if not last:
            prev = w[-overlap_wave:]
    return np.concatenate(parts)


def vocoder_spans(n: int, piece: int = 0, ctx: int = 0) -> list[tuple[int, int, int, int]]:
    """Chia mel n khung thanh cac mieng <= piece khung (+ ctx khung ngu canh moi ben, cat bo sau khi giai ma).
    Tra (bat dau_co_ngu_canh, ket thuc_co_ngu_canh, so khung cat trai, so khung cat phai). BigVGAN la mang chap ->
    ngu canh ~0,4 s du de noi lien; mot lan giai ma 24 s mel ton > 3 GB VRAM tam (OOM 11/10)."""
    piece = piece or VOC_PIECE
    ctx = ctx or VOC_CTX
    if n <= piece:
        return [(0, n, 0, 0)]
    out = []
    for s in range(0, n, piece):
        e = min(s + piece, n)
        cs, ce = max(0, s - ctx), min(n, e + ctx)
        out.append((cs, ce, s - cs, ce - e))
    return out


def whisper_spans(n16: int, sec: int = WHISPER_SEC, ovl: int = WHISPER_OVL) -> list[tuple[int, int, int]]:
    """Doan mau 16 kHz cho whisper (giong goc): (bat dau, ket thuc, so khung 50 Hz bo o dau do chong)."""
    if n16 <= 16000 * sec:
        return [(0, n16, 0)]
    out, t = [(0, 16000 * sec, 0)], 16000 * sec
    while t < n16:
        s = t - 16000 * ovl
        e = min(t + 16000 * (sec - ovl), n16)
        out.append((s, e, 50 * ovl))
        t = e
    return out


class Engine(BaseEngine):
    kind = "vc"

    def load(self) -> None:
        os.chdir(SRC)
        if SRC not in sys.path:
            sys.path.insert(0, SRC)
        self.inf = importlib.import_module("inference")
        import torch
        self._torch = torch
        args = types.SimpleNamespace(fp16=self.device == "cuda", f0_condition=False, checkpoint=None, config=None)
        self.inf.device = torch.device(self.device)
        (self.model, self.semantic_fn, _f0, self.vocoder_fn, self.campplus, self.to_mel,
         mel_args) = self.inf.load_models(args)
        self.sr = int(mel_args["sampling_rate"])
        self.hop = HOP

    def _semantic(self, w16):
        parts = [self.semantic_fn(w16[:, s:e])[:, drop:] for s, e, drop in whisper_spans(w16.size(-1))]
        return parts[0] if len(parts) == 1 else self._torch.cat(parts, dim=1)

    def _vocode(self, mel) -> np.ndarray:
        """BigVGAN theo mieng (vocoder_spans); mel [1, 80, T] -> song float32 dai T*hop."""
        hop, parts = self.hop, []
        for cs, ce, tl, tr in vocoder_spans(mel.size(2)):
            w = self.vocoder_fn(mel[:, :, cs:ce]).squeeze().float().cpu().numpy()
            parts.append(w[tl * hop: len(w) - tr * hop])
        return np.concatenate(parts).astype(np.float32)

    def convert(self, job: dict, k: int, seed: int):
        import librosa
        torch = self._torch
        ta = __import__("torchaudio")
        p = job.get("params", {})
        steps = int(p.get("diffusion_steps", pick(STEPS, k)))
        cfg_rate = float(p.get("inference_cfg_rate", pick(CFG_RATE, k)))
        length_adjust = float(p.get("length_adjust", 1.0))
        dev = torch.device(self.device)
        sr = self.sr
        src = librosa.load(ascii_wav(job["src_wav"]), sr=sr)[0]
        ref = librosa.load(concat_refs(as_list(job["target_ref_wav"]), sr=16000), sr=sr)[0]
        ref_sec = fit_ref_seconds(len(src) * length_adjust / sr, len(ref) / sr, sr, self.hop)
        ref = ref[: int(sr * ref_sec)]
        src_t = torch.tensor(src).unsqueeze(0).float().to(dev)
        ref_t = torch.tensor(ref).unsqueeze(0).float().to(dev)
        with torch.no_grad():
            s_alt = self._semantic(ta.functional.resample(src_t, sr, 16000))
            ref16 = ta.functional.resample(ref_t, sr, 16000)
            s_ori = self.semantic_fn(ref16)
            mel = self.to_mel(src_t)
            mel2 = self.to_mel(ref_t)
            tl = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(dev)
            tl2 = torch.LongTensor([mel2.size(2)]).to(dev)
            feat2 = ta.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0, sample_frequency=16000)
            style2 = self.campplus((feat2 - feat2.mean(dim=0, keepdim=True)).unsqueeze(0))
            cond = self.model.length_regulator(s_alt, ylens=tl, n_quantizers=3, f0=None)[0]
            prompt = self.model.length_regulator(s_ori, ylens=tl2, n_quantizers=3, f0=None)[0]
            spans = chunk_spans(cond.size(1), max_ctx_frames(sr, self.hop) - mel2.size(2))
            waves = []
            for s, e in spans:
                cat = torch.cat([prompt, cond[:, s:e]], dim=1)
                with torch.autocast(device_type=dev.type,
                                    dtype=torch.float16 if self.device == "cuda" else torch.float32):
                    vc = self.model.cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(dev), mel2, style2, None,
                                                  steps, inference_cfg_rate=cfg_rate)
                    vc = vc[:, :, mel2.size(-1):]
                waves.append(self._vocode(vc.float()))
                del vc, cat
            y = stitch(waves, OVERLAP_FRAMES * self.hop)
        params = {"diffusion_steps": steps, "inference_cfg_rate": cfg_rate, "length_adjust": length_adjust,
                  "model": "Plachta/Seed-VC DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned",
                  "n_target_refs": len(as_list(job["target_ref_wav"])), "ref_sec_used": round(len(ref) / sr, 2),
                  "n_chunks": len(spans)}
        return np.asarray(y, np.float32), sr, params
