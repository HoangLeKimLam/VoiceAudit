# -*- coding: utf-8 -*-
"""Bo chuyen F5-TTS 1.1.22 (env rieng D:/va_gen/envs/f5): f5_vi = zalopay/vietnamese-tts (F5TTS_Base, CC-BY-4.0),
f5_en = SWivid F5TTS_v1_Base (CC-BY-NC-4.0, du lieu Emilia).

Ung vien k doi nfe_step / cfg_strength / speed / sway (luoi o duoi); seed = job.seed + k.
Tham chieu: F5 tu cat ref <= ~12 s; ref_text BAT BUOC (khong de F5 goi Whisper tu nhan dang).
Ban zalopay goc 5,4 GB (kem trang thai toi uu) -> rut EMA ra safetensors MOT LAN (doc tung storage, it RAM):
  bash .tmp_v6/wf/run_serial.sh D:/va_gen/envs/f5/Scripts/python.exe -m v6.tts_multi.engines.f5 --convert \
      D:/va_gen/models/f5_vi_zalopay/model_1290000.pt D:/va_gen/models/f5_vi_zalopay/model_1290000_ema.safetensors
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from v6.tts_multi.engines.base import BaseEngine, as_list, concat_refs, pick

NFE = [32, 32, 24, 40]
CFG = [2.0, 2.5, 1.7, 2.2]
SPEED = [1.0, 0.95, 1.05, 1.0]
SWAY = [-1.0, -1.0, -0.8, -1.0]
MAX_REF_SEC = 11.5


class Engine(BaseEngine):
    kind = "tts"

    def load(self) -> None:
        from f5_tts.api import F5TTS
        o = self.opts
        ckpt = o["ckpt_file"]
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"thieu {ckpt} (xem --convert trong docstring)")
        self.tts = F5TTS(model=o["model"], ckpt_file=ckpt, vocab_file=o.get("vocab_file", ""),
                         vocoder_local_path=o.get("vocoder_local_path"), device=self.device)
        self.sr = int(self.tts.target_sample_rate)

    def synth(self, job: dict, k: int, seed: int):
        refs = as_list(job.get("ref_wav"))
        if not refs or not job.get("ref_text"):
            raise ValueError("F5 can ref_wav + ref_text")
        ref = concat_refs(refs, sr=24000)
        import soundfile as sf
        ref_sec = sf.info(ref).duration
        if ref_sec > MAX_REF_SEC:      # F5 tu cat ref > 12 s -> ref_text khong con khop am thanh
            raise ValueError(f"ref {ref_sec:.1f} s > {MAX_REF_SEC} s: chon doan tham chieu ngan hon (ref_text phai khop)")
        p = job.get("params", {})
        nfe = int(p.get("nfe_step", pick(NFE, k)))
        cfg = float(p.get("cfg_strength", pick(CFG, k)))
        speed = float(p.get("speed", pick(SPEED, k)))
        sway = float(p.get("sway_sampling_coef", pick(SWAY, k)))
        quiet = lambda *a, **kw: None  # noqa: E731 - tat log cua F5
        wav, sr, _ = self.tts.infer(ref_file=ref, ref_text=job["ref_text"], gen_text=job["text"], show_info=quiet,
                                    progress=None, nfe_step=nfe, cfg_strength=cfg, speed=speed,
                                    sway_sampling_coef=sway, remove_silence=False, seed=seed)
        params = {"nfe_step": nfe, "cfg_strength": cfg, "speed": speed, "sway_sampling_coef": sway,
                  "cloned": True, "model": self.opts.get("model_id", ""), "n_refs": len(refs)}
        return np.asarray(wav, np.float32), int(sr), params


def _lazy_pickle(zf, prefix: str):
    """Mo data.pkl cua checkpoint zip ma KHONG doc storage: tensor -> ('T', (key, dtype), offset, size, stride)."""
    import collections
    import pickle

    import torch

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
                return lambda storage, offset, size, stride, *a, **kw: ("T", storage, offset, tuple(size), tuple(stride))
            if mod == "torch" and name.endswith("Storage"):
                return torch.serialization.StorageType(name)
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod.startswith(("builtins", "collections", "torch", "numpy", "_codecs")):
                return super().find_class(mod, name)
            raise pickle.UnpicklingError(f"khong cho phep lop {mod}.{name}")

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            return (str(key), stype.dtype)

    with zf.open(prefix + "data.pkl") as fh:
        return _U(fh).load()


def convert_ckpt(src: str, dst: str) -> dict:
    """Rut ema_model_state_dict tu .pt 5,4 GB -> safetensors, doc TUNG storage (khong nap ca tep, khong mmap
    — mmap copy-on-write tren Windows bi tinh vao commit memory)."""
    import zipfile

    import torch
    from safetensors.torch import save_file
    zf = zipfile.ZipFile(src)
    prefix = next(n for n in zf.namelist() if n.endswith("data.pkl"))[: -len("data.pkl")]
    obj = _lazy_pickle(zf, prefix)
    lazy = obj.get("ema_model_state_dict", obj)
    sd, cache = {}, {}
    for name, v in lazy.items():
        if not (isinstance(v, tuple) and v and v[0] == "T"):
            continue
        _, (key, dtype), off, size, stride = v
        if key not in cache:
            cache = {key: torch.frombuffer(bytearray(zf.read(f"{prefix}data/{key}")), dtype=dtype)}
        sd[name] = cache[key].as_strided(size, stride, off).clone()
    cache.clear()
    tmp = dst + ".tmp"
    save_file(sd, tmp)
    os.replace(tmp, dst)
    return {"n_tensors": len(sd), "bytes": os.path.getsize(dst), "step": obj.get("step")}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="F5: tien ich checkpoint")
    ap.add_argument("--convert", nargs=2, metavar=("SRC_PT", "DST_SAFETENSORS"), required=True)
    a = ap.parse_args(argv)
    print(convert_ckpt(*a.convert))
    return 0


if __name__ == "__main__":
    sys.exit(main())
