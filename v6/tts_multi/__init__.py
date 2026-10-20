# -*- coding: utf-8 -*-
"""Nhieu may sinh giong (TTS / VC / khu on AI) cho du lieu VoiceAudit — moi may chay trong python RIENG cua no.

Giao dien I3 (WP7a viet, WP7b goi):
    <env_python> -m v6.tts_multi.run_engine --engine <ten> --jobs <jobs.jsonl> --out <thu_muc>
  TTS : moi dong jobs {job_id, text, lang, ref_wav, ref_text, seed, n_candidates}
  VC  : moi dong jobs {job_id, src_wav, target_ref_wav, seed[, n_candidates]}
  denoise_ai: {job_id, src_wav, seed[, n_candidates]}
  ra  : <out>/<job_id>_c<k>.wav (mono, sr goc cua may) + <out>/<job_id>_c<k>.json {job_id, k, engine, sr, seconds, params}
  ma thoat: 0 xong; 1 loi nap khong phai bo nho (thieu tep / module, hoac 'smoke cu': adapter doi sau smoke -> chay
  lai smoke truoc); 2 co viec loi (<job_id>_c<k>.err.json); 3 khong phuc hoi: CUDA / het bo nho, hoac THIEU COMMIT RAM
  truoc khi nap (summary.fatal 'thieu commit RAM ...', khong sinh gi) -> giai phong bo nho roi chay lai (resume).
  Muc commit RAM ranh can de may chiu nap: commit_required_mb(ten) (dung cho est_mem_gb cua hang doi).
Danh muc: v6/tts_multi/engines.json {ten: {kind, langs, env_python, license, role, notes, ...}}.

Module nay CHI dung thu vien chuan (nap duoc tu moi env rieng). Dung:
    from v6.tts_multi import registry, command, commit_required_mb
    argv, env = command("f5_vi", "jobs.jsonl", "out_dir")   # roi: bash run_serial.sh <argv...> voi env
    gb = commit_required_mb("vixtts") / 1024                  # commit ranh toi thieu (GB) de run_engine chiu nap
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).replace("\\", "/")
PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(PKG_DIR, "engines.json")
MAIN_PY = ROOT + "/.venv/Scripts/python.exe"
TORCH_CU = ROOT + "/.tmp_v6/torch_cu"            # lop torch CUDA cho .venv chinh (dat TRUOC trong PYTHONPATH)
VA_GEN = "D:/va_gen"

# Bien moi truong chung: moi bo nho dem tren D: (C: chi con ~7 GB)
BASE_ENV = {
    "HF_HOME": "D:/hf-cache", "TORCH_HOME": "D:/hf-cache/torch", "TTS_HOME": "D:/hf-cache/tts",
    "XDG_CACHE_HOME": "D:/hf-cache/xdg", "TMP": "D:/tmp_va", "TEMP": "D:/tmp_va", "PYTHONIOENCODING": "utf-8",
    "COQUI_TOS_AGREED": "1", "PYTHONUTF8": "1", "CACHED_PATH_CACHE_ROOT": "D:/hf-cache/cached_path",
    "NLTK_DATA": "D:/hf-cache/nltk_data", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
}


COMMIT_MARGIN_FRAC = 0.10      # du phong commit so voi dinh smoke (viec that dai hon cau smoke)
COMMIT_MARGIN_MB = 256.0


def commit_peak_mb(cfg: dict) -> float | None:
    """Commit (MB) mot tien trinh may can, theo engines.json: min_commit_mb (dat tay) hoac smoke.peak_commit_mb (do
    that). Chua co so do -> None."""
    v = cfg.get("min_commit_mb")
    if v is None:
        v = (cfg.get("smoke") or {}).get("peak_commit_mb")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def commit_extra_mb(need: float, cur: float = 0.0) -> float:
    """Commit ranh can THEM khi tien trinh dang dung `cur` MB: (need - cur) * 1,1 + 256 MB (cong thuc run_engine)."""
    return round(max(0.0, need - (cur or 0.0)) * (1 + COMMIT_MARGIN_FRAC) + COMMIT_MARGIN_MB, 1)


def commit_required_mb(name: str, reg: dict | None = None) -> float | None:
    """Commit ranh (MB) toi thieu de run_engine chiu nap may `name` (chan tren: coi commit dang dung = 0; thuc te
    tien trinh da dung ~100-300 MB nen can it hon chut). Chua co so do -> None (run_engine khong chan)."""
    reg = registry() if reg is None else reg
    need = commit_peak_mb(reg[name])
    return None if need is None else commit_extra_mb(need, 0.0)


PATH_KEYS = ("env_python", "model_dir", "ckpt_file", "vocab_file", "config_file", "log")


def _rooted(cfg):
    """Duong TUONG DOI trong danh muc (vd '.venv/Scripts/python.exe', 'models/viXTTS') -> theo goc kho nay; duong tuyet doi
    (moi truong rieng cua may, vd D:/va_gen) giu nguyen (ham thuan)."""
    if isinstance(cfg, dict):
        return {k: (ROOT + "/" + v if k in PATH_KEYS and isinstance(v, str) and v and not os.path.isabs(v) else _rooted(v))
                for k, v in cfg.items()}
    return cfg


def registry(path: str = REGISTRY_PATH) -> dict:
    """Doc danh muc may (bo cac khoa bat dau bang '_' la ghi chu)."""
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    return {k: _rooted(v) for k, v in reg.items() if not k.startswith("_")}


def engine_env(cfg: dict, base: dict | None = None) -> dict:
    """Bien moi truong de chay mot may: ROOT trong PYTHONPATH (de -m v6.tts_multi.run_engine), lop torch_cu neu can."""
    env = dict(os.environ if base is None else base)
    env.update(BASE_ENV)
    parts = [TORCH_CU] if cfg.get("gpu_overlay") else []
    parts.append(ROOT)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    env.update(cfg.get("env", {}))
    return env


def command(name: str, jobs: str, out: str, extra: list | None = None, reg: dict | None = None) -> tuple[list, dict]:
    """(argv, env) de chay may `name` tren jobs -> out. Nho goi qua run_serial.sh (khoa may)."""
    reg = registry() if reg is None else reg
    if name not in reg:
        raise KeyError(f"khong co may '{name}' trong engines.json")
    cfg = reg[name]
    argv = [cfg["env_python"], "-m", "v6.tts_multi.run_engine", "--engine", name, "--jobs", jobs, "--out", out]
    return argv + list(extra or []), engine_env(cfg)
