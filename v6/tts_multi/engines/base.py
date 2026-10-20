# -*- coding: utf-8 -*-
"""Lop goc + tien ich chung cho cac bo chuyen (adapter) may sinh giong. Chi numpy + soundfile o muc module
(torch nap muon) -> nap duoc tu moi env rieng.

Moi adapter: class Engine(BaseEngine) voi
    load()                          nap mo hinh (mot lan / tien trinh)
    synth(job, k, seed) -> (wav float32 mono, sr, params)       cho kind tts
    convert(job, k, seed) -> (...)                                cho kind vc / denoise_ai
    close()                         giai phong (del mo hinh + gc + cuda.empty_cache)
"""
from __future__ import annotations

import gc
import hashlib
import os
import random
import sys

import numpy as np
import soundfile as sf

VA_GEN = "D:/va_gen"
MODELS = VA_GEN + "/models"
REF_CACHE = VA_GEN + "/tmp/refs"           # ban sao ASCII cua tep tham chieu (thu vien C hay hong voi duong Unicode)


def torch_mod():
    """torch neu da nap duoc (khong ep nap)."""
    return sys.modules.get("torch")


def pick_device(want: str = "auto") -> str:
    if want in ("cpu", "cuda"):
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    t = torch_mod()
    if t is not None:
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)


def free_memory() -> None:
    gc.collect()
    t = torch_mod()
    if t is not None and t.cuda.is_available():
        t.cuda.empty_cache()


def peak_vram_mb() -> float | None:
    t = torch_mod()
    if t is None or not t.cuda.is_available():
        return None
    return round(t.cuda.max_memory_allocated() / 2 ** 20, 1)


def _proc_commit(field: str) -> float | None:
    """Truong commit (PagefileUsage / PeakPagefileUsage, MB) cua tien trinh hien tai (Windows; noi khac None)."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    try:
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return round(getattr(c, field) / 2 ** 20, 1)
    except (OSError, AttributeError, ctypes.ArgumentError):
        return None                                   # chi so phu, khong bao gio lam hong lan chay


def peak_commit_mb() -> float | None:
    """Dinh commit (PeakPagefileUsage) cua tien trinh — chi so quan trong tren may it commit RAM (Windows)."""
    return _proc_commit("PeakPagefileUsage")


def commit_mb() -> float | None:
    """Commit hien tai (PagefileUsage) cua tien trinh, MB."""
    return _proc_commit("PagefileUsage")


def free_commit_mb() -> float | None:
    """Commit con lai cua CA MAY khi KHONG noi pagefile (GlobalMemoryStatusEx.ullAvailPageFile = gioi han commit
    hien tai - tong commit), MB. Pagefile o day do Windows tu quan ly: vuot muc nay -> pagefile no them tren C:
    (11/10 da day C: ve 0 byte). Noi khac / loi -> None."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class MSE(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
            (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                              "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]
    try:
        m = MSE()
        m.dwLength = ctypes.sizeof(MSE)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MSE)]
        k32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        if not k32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return round(m.ullAvailPageFile / 2 ** 20, 1)
    except (OSError, AttributeError, ctypes.ArgumentError):
        return None


def is_ascii(path: str) -> bool:
    try:
        path.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ascii_wav(path: str, sr: int | None = None, max_sec: float | None = None) -> str:
    """Ban wav PCM16 mono tren duong ASCII (D:/va_gen/tmp/refs) cho `path` (wav/flac/...).
    Doi sr neu `sr` duoc cho, cat toi da max_sec. Bo nho dem theo (duong, mtime, sr, max_sec)."""
    path = os.path.abspath(path)
    st = os.stat(path)
    key = hashlib.md5(f"{path}|{st.st_mtime_ns}|{st.st_size}|{sr}|{max_sec}".encode("utf-8")).hexdigest()[:16]
    out = os.path.join(REF_CACHE, f"{key}.wav").replace("\\", "/")
    if os.path.exists(out):
        return out
    y, s = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr and s != sr:
        y = resample(y, s, sr)
        s = sr
    if max_sec:
        y = y[: int(max_sec * s)]
    os.makedirs(REF_CACHE, exist_ok=True)
    tmp = out + ".tmp.wav"
    sf.write(tmp, np.clip(y, -1, 1), s, subtype="PCM_16")
    os.replace(tmp, out)
    return out


def concat_refs(paths: list, sr: int = 16000, gap: float = 0.15) -> str:
    """Noi nhieu tep tham chieu cung nguoi thanh mot tep ASCII (cho may chi nhan mot tep)."""
    if len(paths) == 1:
        return ascii_wav(paths[0], sr=sr)
    key = hashlib.md5("|".join(os.path.abspath(p) for p in paths).encode("utf-8")).hexdigest()[:16]
    out = os.path.join(REF_CACHE, f"cat_{key}_{sr}.wav").replace("\\", "/")
    if os.path.exists(out):
        return out
    parts = []
    for p in paths:
        y, _ = sf.read(ascii_wav(p, sr=sr), dtype="float32")
        parts += [y, np.zeros(int(gap * sr), np.float32)]
    os.makedirs(REF_CACHE, exist_ok=True)
    sf.write(out, np.concatenate(parts[:-1]), sr, subtype="PCM_16")
    return out


def resample(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y
    try:
        import soxr
        return soxr.resample(y, sr_in, sr_out).astype(np.float32)
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out)
    except ImportError:
        from scipy.signal import resample_poly
        g = np.gcd(sr_in, sr_out)
        return resample_poly(y, sr_out // g, sr_in // g).astype(np.float32)


def to_mono_f32(wav) -> np.ndarray:
    """torch.Tensor / list / ndarray -> float32 1-D."""
    if hasattr(wav, "detach"):
        wav = wav.detach().float().cpu().numpy()
    y = np.squeeze(np.asarray(wav, dtype=np.float32))
    if y.ndim > 1:                                   # nhieu kenh: trung binh theo truc ngan (truc kenh)
        y = y.mean(axis=int(np.argmin(y.shape)))
    return np.atleast_1d(y)


def as_list(x) -> list:
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def pick(grid: list, k: int):
    """Phan tu thu k (vong) cua luoi tham so -> ung vien khac nhau co tham so khac nhau."""
    return grid[k % len(grid)]


class BaseEngine:
    kind = "tts"
    CODE_REV = "1"          # phien ban hanh vi adapter; doi cach sinh -> tang (smoke cu bi run_engine tu choi)

    @classmethod
    def code_rev(cls, cfg: dict) -> str:
        """Phien ban hanh vi cua adapter cho may `cfg` (adapter nhieu ho co the tra theo ho). Smoke ghi lai so nay."""
        return str(cls.CODE_REV)

    def __init__(self, name: str, cfg: dict, device: str = "auto"):
        self.name = name
        self.cfg = cfg
        self.device = pick_device(device)
        self.opts = dict(cfg.get("opts", {}))

    def load(self) -> None:  # pragma: no cover - adapter viet de
        raise NotImplementedError

    def synth(self, job: dict, k: int, seed: int):  # pragma: no cover
        raise NotImplementedError(f"{self.name} khong phai may TTS")

    def convert(self, job: dict, k: int, seed: int):  # pragma: no cover
        raise NotImplementedError(f"{self.name} khong phai may VC / khu on")

    def run(self, job: dict, k: int, seed: int):
        return self.synth(job, k, seed) if self.kind == "tts" else self.convert(job, k, seed)

    def close(self) -> None:
        for a in [a for a in vars(self) if a not in ("name", "cfg", "device", "opts")]:
            setattr(self, a, None)
        free_memory()
