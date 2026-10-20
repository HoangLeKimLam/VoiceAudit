# -*- coding: utf-8 -*-
"""Bo chuyen doi giong (VC) cua coqui-tts trong .venv chinh: OpenVoice v2 (tone colour converter), FreeVC24, kNN-VC.

Viec VC: {job_id, src_wav, target_ref_wav, seed}. target_ref_wav = ban ghi KHAC cua nguoi dich (kho cho phep);
kNN-VC can nhieu phut giong dich (danh sach tep) — it hon ~1 phut thi chat luong giam ro.
Ung vien k: kNN-VC doi topk (4/8/2) (tat dinh voi cung tham chieu); OpenVoice doi tau (nhiet do lay mau
enc_q: 0,3 mac dinh / 0,5 / 0,15 / 0,7) + seed; FreeVC lay mau z trong enc_p theo seed (chi khac nho) -> khi
n_candidates > so tep tham chieu, params.diversity = 'seed_only' de WP7b biet cac ung vien gan trung nhau.
Nhieu tep trong target_ref_wav: OpenVoice/FreeVC dung lan luot tung tep cho k > 0.
Ghi chu ho: kNN-VC (held-out) va FreeVC24 (train) CUNG bo ma noi dung WavLM-Large (bo giai ma khac: kNN + HiFi-GAN
prematched vs VITS) -> held-out VC chi "khac ho" o phan giai ma; ghi trong notes engines.json.
kNN-VC + tham chieu DAI (12/10): coqui chay WavLM tren CA tep mot lan, attention O(T^2) -> tep ~300 s (~15000 khung)
can ~14 GB VRAM moi lop (WP7b cho knnvc tep toi 300 s; tep ~80 s co le van vua 6 GB, tep > ~120-150 s thi khong).
Nay tap so khop lam theo knn_matching_set:
  1. VAD (torchaudio, trigger 7 = mac dinh get_matching_set cua coqui) MOT lan tren CA tep dich, tren CPU, dau roi lat
     nguoc cat cuoi — dung cach coqui lam tren ca tep. Luot dau khong thay tieng -> giu nguyen tep ('none'); luot cuoi
     khong thay -> chi cat dau ('front'). (Truoc day coqui gap tep VAD rong -> 'cannot reshape tensor of 0 elements'.)
  2. Cat song da cat lang thanh mieng <= KNN_PIECE_SEC (split_pieces).
  3. get_features(mieng, vad_trigger_level=0) tung mieng (KHONG VAD lai tung mieng: ban giua 12/10 VAD tung mieng 15 s
     -> mieng im lang ra rong -> RuntimeError; do tren jobs knnvc cua WP7b: 20/240 viec (ban jobs truoc 18:00) va
     11/165 viec, 7 tep dich AMI (ban jobs 18:00); tep khong hong cung mat khung o bien mieng: VIVOS 129,5 s 5928 vs
     6442 khung, do boi reviewer),
     bo mieng < KNN_MIN_SAMPLES, ghep -> CPU. Bo nho dem tap so khop theo tep dich (KNN_CACHE muc).
Phan con lai giu dung coqui: get_features(nguon, khong VAD) -> match(topk) -> vocoder HiFi-GAN cua Synthesizer.
Khac coqui con lai: bien moi mieng 15 s WavLM mat ngu canh (<= 1 khung / mieng; 127 tep dich WP7b: 665 597 khung vs
666 409 khi coqui chay ca tep, -0,12 %; do 12/10 voi get_features + VAD that, WavLM gia).
CODE_REVS: doi hanh vi adapter -> tang so; smoke ghi code_rev, run_engine tu choi chay khi smoke cu hon ma (xem do).
"""
from __future__ import annotations

import importlib
import os

import numpy as np
import soundfile as sf

from v6.tts_multi.engines.base import BaseEngine, as_list, ascii_wav, pick, to_mono_f32

TOPK = [4, 8, 2]
TAU = [0.3, 0.5, 0.15, 0.7]     # OpenVoice v2: 0,3 = mac dinh cua coqui/OpenVoice
KNN_PIECE_SEC = 15.0            # mieng tham chieu cho WavLM (750 khung: attention ~36 MB / lop)
KNN_MIN_PIECE_SEC = 1.0         # mieng cuoi ngan hon -> gop vao mieng truoc
KNN_CACHE = 2                   # so tap so khop giu lai (300 s ~ 61 MB moi tap, tren CPU)
KNN_VAD_TRIGGER = 7.0           # = vad_trigger_level mac dinh cua KNNVC.get_matching_set (coqui)
KNN_MIN_SAMPLES = 400           # WavLM can >= 400 mau (25 ms) moi ra 1 khung; ngan hon -> bo / khong cat lang
SR_KNN = 16000
# Phien ban hanh vi theo ho (family); khong co -> "1". Tang khi doi cach sinh -> smoke cu bi coi la cu (run_engine).
CODE_REVS = {"knnvc": "2"}      # 2 = VAD ca tep + WavLM theo mieng khong VAD (12/10); 1 = coqui nguyen ban (11/10)


def vad_trim(x, sr: int = SR_KNN, trigger: float = KNN_VAD_TRIGGER, vad=None):
    """Cat lang dau + cuoi CA tep nhu KNNVC.get_features cua coqui (Vad, lat nguoc, Vad, lat lai), tren CPU.
    x: tensor (1, T). Luot dau con < KNN_MIN_SAMPLES mau -> tra nguyen x ('none'); luot cuoi con qua it -> chi cat
    dau ('front'); du -> 'both'. Tra (song, cach_cat). vad: thay bo VAD (test)."""
    import torch
    if vad is None:
        import torchaudio
        vad = torchaudio.transforms.Vad(sample_rate=sr, trigger_level=trigger)
    x = x.detach().cpu()
    front = vad(x)
    if front.shape[-1] < KNN_MIN_SAMPLES:
        return x, "none"
    back = vad(torch.flip(front, (-1,)))
    if back.shape[-1] < KNN_MIN_SAMPLES:
        return front, "front"
    return torch.flip(back, (-1,)), "both"


def knn_matching_set(model, waves: list, sr: int = SR_KNN, vad=None) -> tuple:
    """Tap so khop kNN-VC (N, dim, CPU) tu cac song dich 16 kHz (mang 1-D): VAD ca tep -> mieng <= KNN_PIECE_SEC ->
    model.get_features(mieng, vad_trigger_level=0). Goi trong torch.inference_mode(). Tra (tap, so_mieng, [cach_cat])."""
    import torch
    feats, cuts = [], []
    for y in waves:
        x, how = vad_trim(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))[None], sr, vad=vad)
        cuts.append(how)
        for p in split_pieces(x[0].numpy(), sr):
            if len(p) >= KNN_MIN_SAMPLES:
                feats.append(model.get_features(torch.from_numpy(np.ascontiguousarray(p))[None], vad_trigger_level=0))
    if not feats:
        raise ValueError(f"kNN-VC: tep dich khong co doan nao >= {KNN_MIN_SAMPLES} mau cho WavLM")
    return torch.concat(feats, dim=0).cpu(), len(feats), cuts


def split_pieces(y: np.ndarray, sr: int, piece_sec: float = KNN_PIECE_SEC,
                 min_sec: float = KNN_MIN_PIECE_SEC) -> list[np.ndarray]:
    """Cat song thanh cac mieng <= piece_sec (mieng cuoi < min_sec gop vao mieng truoc). Khong mat mau nao."""
    n = int(piece_sec * sr)
    if len(y) <= n:
        return [y]
    pieces = [y[i:i + n] for i in range(0, len(y), n)]
    if len(pieces[-1]) < int(min_sec * sr):
        pieces = pieces[:-2] + [np.concatenate(pieces[-2:])]
    return pieces


def _patch_wavlm(device: str) -> None:
    """WavLM-Large (1,26 GB) cua coqui: tao mo hinh + nap checkpoint THANG len `device` (tung tensor), khong giu ban
    CPU ~2,5 GB (khoi tao + checkpoint). Truoc day kNN-VC chet 0xC0000005 / 'CUDA out of memory' trong torch.load
    khi commit RAM may con ~2 GB (cung kieu loi XTTS da gap)."""
    if device != "cuda":
        return
    import torch
    import TTS.vc.layers.freevc.wavlm as W
    from TTS.vc.layers.freevc.wavlm.wavlm import WavLM, WavLMConfig

    def _get_wavlm(_device="cpu"):
        path = os.path.join(W.get_user_data_dir("tts"), "wavlm", "WavLM-Large.pt")
        if not os.path.exists(path):
            return orig(device)                      # chua tai -> de coqui tai ve (duong goc)
        ckpt = torch.load(path, map_location="cuda", weights_only=W.is_pytorch_at_least_2_4())
        with torch.device("cuda"):
            m = WavLM(WavLMConfig(ckpt["cfg"]))
        m.load_state_dict(ckpt["model"])
        del ckpt
        return m.eval()

    orig = W.get_wavlm
    W.get_wavlm = _get_wavlm
    for mod in ("TTS.vc.models.knnvc", "TTS.vc.models.freevc"):
        try:
            setattr(importlib.import_module(mod), "get_wavlm", _get_wavlm)
        except (ImportError, AttributeError):
            pass


class Engine(BaseEngine):
    kind = "vc"

    @classmethod
    def code_rev(cls, cfg: dict) -> str:
        fam = (cfg.get("opts") or {}).get("family") or cfg.get("family", "")
        return CODE_REVS.get(fam, "1")

    def load(self) -> None:
        from TTS.api import TTS
        import TTS.model as tmodel
        if self.opts.get("family") in ("knnvc", "freevc"):
            _patch_wavlm(self.device)
        orig = tmodel.load_fsspec
        if self.device == "cuda":     # checkpoint chinh (FreeVC ...) nap thang len GPU, khong giu ban CPU (commit RAM)
            tmodel.load_fsspec = lambda path, map_location=None, **kw: orig(path, map_location="cuda", **kw)
        try:
            self.api = TTS(self.opts["coqui_id"], progress_bar=False).to(self.device)
        finally:
            tmodel.load_fsspec = orig
        self.sr = int(self.api.voice_converter.output_sample_rate)
        self.family = self.opts.get("family", "")
        self._ms_cache = {}

    def _knn_matching_set(self, targets: list):
        """Tap so khop kNN-VC (N, 1024, CPU) tu cac tep dich 16 kHz (knn_matching_set: VAD ca tep, WavLM theo mieng).
        Goi trong torch.inference_mode(). Tra (tap, so_mieng, [cach_cat VAD moi tep], lay_tu_bo_nho_dem)."""
        cache = getattr(self, "_ms_cache", None)
        if cache is None:
            cache = self._ms_cache = {}
        key = tuple(targets)                          # ascii_wav: ten tep da gom mtime + kich thuoc
        if key in cache:
            ms, n, cuts = cache[key]
            return ms, n, cuts, True
        waves = [sf.read(t, dtype="float32")[0] for t in targets]
        ms, n, cuts = knn_matching_set(self.api.voice_converter.vc_model, waves, SR_KNN)
        while len(cache) >= KNN_CACHE:
            cache.pop(next(iter(cache)))
        cache[key] = (ms, n, cuts)
        return ms, n, cuts, False

    def _knn_convert(self, src: str, targets: list, topk: int):
        """Nhu Synthesizer.voice_conversion cua coqui cho kNN-VC, nhung tap so khop lam theo mieng + co bo nho dem."""
        import torch
        syn = self.api.voice_converter
        model = syn.vc_model
        with torch.inference_mode():
            ms, n_pieces, cuts, hit = self._knn_matching_set(targets)
            q = model.get_features(torch.from_numpy(sf.read(src, dtype="float32")[0])[None])   # nguon: khong VAD (= coqui)
            out = model.match(q, ms, topk=topk)
            if syn.vocoder_model is not None:
                out = syn.vocoder_model.inference(out)
        return out.squeeze(), {"n_target_pieces": n_pieces, "matching_frames": int(ms.shape[0]),
                               "matching_set_cached": hit, "target_vad": cuts}

    def convert(self, job: dict, k: int, seed: int):
        p = job.get("params", {})
        targets = [ascii_wav(t, sr=16000) for t in as_list(job["target_ref_wav"])]
        if not targets:
            raise ValueError("VC can target_ref_wav")
        src = ascii_wav(job["src_wav"], sr=16000)
        params = {"family": self.family}
        if self.family == "knnvc":
            # kNN-VC doc tep bang torchaudio.load -> torchcodec hong voi lop torch_cu; dua tensor 16 kHz thay duong dan.
            # Het cac tep dich lam tap so khop: VAD ca tep roi cat mieng <= KNN_PIECE_SEC (tep 300 s khong het VRAM).
            topk = int(p.get("topk", pick(TOPK, k)))
            params["topk"] = topk
            if k >= len(TOPK) and "topk" not in p:
                params["diversity"] = f"duplicate_of_k{k % len(TOPK)}"      # kNN-VC tat dinh: lap lai luoi topk
            y, info = self._knn_convert(src, targets, topk)
            params.update(info, n_target_refs=len(targets), src_sr_in=16000)
            return to_mono_f32(y), self.sr, params                          # tensor tren GPU -> numpy
        use = targets if len(targets) == 1 or k == 0 else [targets[k % len(targets)]]   # OpenVoice / FreeVC
        grid = len(TAU) if self.family == "openvoice" else 1                 # so ung vien khac nhau ro khong ke seed
        if self.family == "openvoice":
            params["tau"] = float(p.get("tau", pick(TAU, k)))
            self.api.voice_converter.vc_model.tau = params["tau"]
        if k >= max(grid, len(targets)):
            params["diversity"] = "seed_only"
        y = self.api.voice_conversion(source_wav=src, target_wav=use)
        params.update({"n_target_refs": len(use), "src_sr_in": 16000})
        return to_mono_f32(y), self.sr, params
