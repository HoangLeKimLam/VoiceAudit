# -*- coding: utf-8 -*-
"""Chay MOT may sinh giong tren mot danh sach viec (giao dien I3), trong python cua chinh may do.

Dung (LUON qua khoa may):
  bash .tmp_v6/wf/run_serial.sh <env_python> -m v6.tts_multi.run_engine --engine <ten> --jobs <jobs.jsonl> --out <dir>
      [--device auto|cuda|cpu] [--limit N] [--summary <tep.json>] [--fail-fast] [--allow-cpu-fallback]
      [--commit-wait-min M] [--allow-pagefile-growth] [--allow-stale-smoke]
  (PYTHONPATH phai co thu muc goc cua kho; .venv chinh + GPU: dat .tmp_v6/torch_cu truoc — xem v6.tts_multi.command)

jobs.jsonl, moi dong:
  tts        {job_id, text, lang, ref_wav, ref_text, seed, n_candidates[, params]}   (ref_wav: tep hoac danh sach)
  vc         {job_id, src_wav, target_ref_wav, seed[, n_candidates, params]}
  denoise_ai {job_id, src_wav, seed[, n_candidates, params]}
Ra: <out>/<job_id>_c<k>.wav (mono float, sr goc) + <out>/<job_id>_c<k>.json {job_id, k, engine, sr, seconds, params}.
  .json ghi SAU .wav (thay the nguyen tu) -> co .json = xong; chay lai bo qua viec da co (resume).
  Loi mot viec -> <out>/<job_id>_c<k>.err.json, chay tiep (tru --fail-fast). Seed ung vien k = seed + k.
Duong dan tuong doi trong jobs (ref_wav, src_wav, target_ref_wav) tinh theo cwd luc chay lenh.
Mot tien trinh = mot may = mot mo hinh tren GPU; cuoi cung giai phong (close + gc + empty_cache).
Loi khi NAP may: ghi summary {load_error, fatal (neu bo nho/CUDA)} roi thoat (3 = fatal, 1 = loi khac).
--allow-cpu-fallback (+ --device auto, mac dinh TAT): nap GPU het bo nho -> nap lai CPU, ghi 'fallback' trong summary.
  Tat mac dinh: o may nay 'CUDA out of memory' luc nap la do het commit RAM; CPU chi lam te hon + giu khoa lau.
KIEM COMMIT TRUOC KHI NAP (mac dinh BAT): can = smoke.peak_commit_mb (hoac min_commit_mb) trong engines.json - commit
  dang dung + 10 % + 256 MB; commit con lai cua may (khong noi pagefile) < can -> KHONG nap, ghi summary fatal
  'thieu commit RAM' (ma 3). Ly do: pagefile do Windows tu quan ly, vuot commit la no pagefile tren C: (11/10 da day C:).
  --commit-wait-min M: cho toi da M phut (hoi 15 s / lan) cho du commit. --allow-pagefile-growth: bo kiem nay.
  Uoc luong ngoai (hang doi): v6.tts_multi.commit_required_mb(ten) = muc can khi commit dang dung = 0 (chan tren).
KIEM SMOKE CON HOP LE (mac dinh BAT): adapter co phien ban hanh vi Engine.code_rev(cfg) (base.CODE_REV / CODE_REVS);
  smoke trong engines.json ghi code_rev luc do (thieu = "1"). Khac nhau = adapter da doi sau smoke -> KHONG nap, ghi
  summary load_error 'smoke cu' + stale_smoke (ma 1): chay lai smoke truoc (smoke tu them --allow-stale-smoke).
  May chua co smoke: khong chan.
--summary: ghi {engine, n_new, n_skip, n_err, load_sec, gen_sec, audio_sec, peak_vram_mb, peak_commit_mb, device, outputs,
  commit, code_rev, [fatal, load_error, fallback, stale_smoke]}.
Ma thoat: 0 xong; 1 loi nap khong phai bo nho (thieu tep / module, smoke cu); 2 co viec loi; 3 loi khong phuc hoi
  (CUDA hong / het bo nho luc nap hoac chay, THIEU COMMIT RAM truoc khi nap: summary.fatal bat dau 'thieu commit RAM')
  -> giai phong bo nho roi chay lai lenh de tiep tuc (resume).
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback

import numpy as np
import soundfile as sf

from v6.tts_multi import COMMIT_MARGIN_FRAC, COMMIT_MARGIN_MB, commit_extra_mb, commit_peak_mb  # noqa: F401

PKG = "v6.tts_multi"
SILENT_DB = -50.0          # RMS duoi muc nay coi la cam
MIN_SEC = 0.3              # ngan hon coi la hong

REQUIRED = {
    "tts": ("job_id", "text", "lang"),            # ref_wav: may nhai giong tu kiem; may giong co san bo qua
    "vc": ("job_id", "src_wav", "target_ref_wav"),
    "denoise_ai": ("job_id", "src_wav"),
}
PATH_FIELDS = ("ref_wav", "src_wav", "target_ref_wav")   # duong tuong doi -> tuyet doi theo cwd LUC DOC jobs
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED")
COMMIT_POLL_SEC = 15.0         # COMMIT_MARGIN_FRAC / _MB: o v6.tts_multi (dung chung voi commit_required_mb)


def _abs_paths(job: dict) -> dict:
    """Doi cac truong duong dan tuong doi sang tuyet doi (mot so may doi cwd khi nap, vd Seed-VC)."""
    fix = {}
    for f in PATH_FIELDS:
        v = job.get(f)
        if isinstance(v, str) and v:
            fix[f] = os.path.abspath(v)
        elif isinstance(v, (list, tuple)):
            fix[f] = [os.path.abspath(x) for x in v]
    return {**job, **fix}


def load_registry() -> dict:
    from v6.tts_multi import registry
    return registry()


def read_jobs(path: str, kind: str) -> list[dict]:
    """Doc + kiem jobs.jsonl (bo dong trong). Loi ro rang neu thieu truong / trung job_id."""
    jobs, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)
            miss = [f for f in REQUIRED[kind] if f not in job]
            if miss:
                raise ValueError(f"{path}:{i} thieu truong {miss} (may kind={kind})")
            jid = str(job["job_id"])
            if jid in seen:
                raise ValueError(f"{path}:{i} trung job_id {jid}")
            if any(c in jid for c in '\\/:*?"<>|'):
                raise ValueError(f"{path}:{i} job_id co ky tu khong hop le: {jid}")
            seen.add(jid)
            job.setdefault("seed", 0)
            job.setdefault("n_candidates", 1)
            jobs.append(_abs_paths(job))
    return jobs


def out_paths(out_dir: str, job_id: str, k: int) -> tuple[str, str]:
    stem = os.path.join(out_dir, f"{job_id}_c{k}")
    return stem + ".wav", stem + ".json"


def pending(job: dict, out_dir: str) -> list[int]:
    """Cac k chua xong (thieu .json hoac thieu .wav)."""
    todo = []
    for k in range(int(job["n_candidates"])):
        w, j = out_paths(out_dir, job["job_id"], k)
        if not (os.path.exists(w) and os.path.exists(j)):
            todo.append(k)
    return todo


def sanity(y: np.ndarray, sr: int) -> dict:
    """Do nhanh: do dai, RMS dBFS, dinh, co NaN, co cam."""
    finite = bool(np.all(np.isfinite(y)))
    yy = np.nan_to_num(y)
    rms = float(np.sqrt(np.mean(yy ** 2))) if len(yy) else 0.0
    rms_db = 20 * np.log10(rms + 1e-12)
    sec = len(yy) / float(sr)
    return {"seconds": round(sec, 3), "rms_db": round(rms_db, 2), "peak": round(float(np.max(np.abs(yy))) if len(yy) else 0.0, 4),
            "finite": finite, "sane": bool(finite and sec >= MIN_SEC and rms_db > SILENT_DB)}


def write_atomic_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def write_output(out_dir: str, engine: str, job: dict, k: int, y: np.ndarray, sr: int, params: dict) -> dict:
    """Ghi .wav (float, nguyen tu) roi .json. Tra ban ghi json."""
    from v6.tts_multi.engines.base import to_mono_f32
    y = to_mono_f32(y)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak > 1.0:                                   # chong vo tieng khi ghi; ghi lai he so
        y = y / peak * 0.99
        params = {**params, "renorm_peak": round(peak, 4)}
    wav_p, js_p = out_paths(out_dir, job["job_id"], k)
    tmp = wav_p + ".tmp.wav"
    sf.write(tmp, y, int(sr), subtype="FLOAT")
    os.replace(tmp, wav_p)
    chk = sanity(y, sr)
    rec = {"job_id": job["job_id"], "k": k, "engine": engine, "sr": int(sr), "seconds": chk["seconds"],
           "params": {**params, "sanity": chk}}
    write_atomic_json(js_p, rec)
    return rec


FATAL_MARKERS = ("CUDA error", "CUBLAS_STATUS", "cuDNN error", "paging file is too small", "MemoryError")


def is_fatal(exc: BaseException) -> bool:
    """Loi lam hong ca tien trinh (ngu canh CUDA, het bo nho) -> khong chay tiep viec sau."""
    msg = f"{type(exc).__name__}: {exc}"
    return isinstance(exc, MemoryError) or any(m in msg for m in FATAL_MARKERS)


def make_engine(name: str, cfg: dict, device: str):
    mod = importlib.import_module(f"{PKG}.engines.{cfg['module']}")
    return mod.Engine(name, cfg, device)


def is_oom(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(m in msg for m in OOM_MARKERS)


def load_engine(name: str, cfg: dict, device: str, log=print, allow_cpu_fallback: bool = False):
    """Tao + nap may. Mac dinh: loi nap nem ra (run() bien loi bo nho thanh 'fatal', ma thoat 3).
    CHI khi allow_cpu_fallback (--allow-cpu-fallback) va --device auto: nap GPU het bo nho -> giai phong, nap lai
    tren CPU. Mac dinh TAT vi tren may nay 'CUDA out of memory' khi nap thuong do het commit RAM (VRAM con trong):
    nap lai tren CPU con ton commit hon va giu khoa may lau qua 10 phut. Tra (engine, ghi_chu_fallback | None)."""
    from v6.tts_multi.engines.base import free_memory
    eng = make_engine(name, cfg, device)
    try:
        eng.load()
        return eng, None
    except Exception as exc:  # noqa: BLE001 - chi bat het bo nho GPU khi duoc phep fallback, con lai nem lai
        if not (allow_cpu_fallback and device == "auto" and eng.device == "cuda" and is_oom(exc)):
            eng.close()
            raise
        note = f"cuda->cpu: {type(exc).__name__}: {str(exc)[:200]}"
        log(f"[{name}] nap GPU het bo nho -> thu lai tren CPU ({note[:120]})")
        eng.close()
        del eng
        free_memory()
    eng = make_engine(name, cfg, "cpu")
    eng.load()
    return eng, note


commit_need_mb = commit_peak_mb      # ten cu: commit (MB) mot tien trinh may can theo engines.json (None = chua do)


def smoke_rev_check(cfg: dict) -> dict:
    """Smoke trong engines.json con ung voi ma adapter hien tai? {ok, code_rev, smoke_rev}. code_rev = Engine.code_rev
    (nap module adapter: chi numpy/soundfile o muc module). Smoke khong ghi code_rev = "1"; chua co smoke -> ok.
    Khong nap duoc module -> ok (loi that se hien luc nap may)."""
    try:
        cur = str(importlib.import_module(f"{PKG}.engines.{cfg['module']}").Engine.code_rev(cfg))
    except Exception as exc:  # noqa: BLE001 - de load_engine bao loi nap day du
        return {"ok": True, "code_rev": None, "smoke_rev": None, "why": f"{type(exc).__name__}: {exc}"[:200]}
    sm = cfg.get("smoke")
    if not isinstance(sm, dict) or not sm:
        return {"ok": True, "code_rev": cur, "smoke_rev": None}
    got = str(sm.get("code_rev") or "1")
    return {"ok": got == cur, "code_rev": cur, "smoke_rev": got}


def commit_check(cfg: dict, wait_min: float = 0.0, log=print, free_fn=None, cur_fn=None, sleep=time.sleep,
                 clock=time.time) -> dict:
    """Du commit RAM de nap may khong? can_them = (can - commit dang dung cua tien trinh nay) * 1,1 + 256 MB, so voi
    commit con lai cua may (khong noi pagefile). Cho toi da wait_min phut. Tra {ok, need_mb, free_mb, waited_sec[, why]}.
    free_fn / cur_fn / sleep / clock thay duoc trong test. Khong do duoc (khong phai Windows / chua co so) -> ok."""
    from v6.tts_multi.engines.base import commit_mb, free_commit_mb
    free_fn = free_commit_mb if free_fn is None else free_fn
    cur_fn = commit_mb if cur_fn is None else cur_fn
    need = commit_need_mb(cfg)
    if need is None:
        return {"ok": True, "need_mb": None, "free_mb": free_fn(), "waited_sec": 0.0, "why": "chua co so do commit"}
    t0 = clock()
    while True:
        free, cur = free_fn(), cur_fn() or 0.0
        if free is None:
            return {"ok": True, "need_mb": None, "free_mb": None, "waited_sec": 0.0, "why": "khong do duoc commit"}
        extra = commit_extra_mb(need, cur)
        waited = round(clock() - t0, 1)
        if free >= extra:
            return {"ok": True, "need_mb": extra, "free_mb": free, "waited_sec": waited}
        if waited >= wait_min * 60:
            return {"ok": False, "need_mb": extra, "free_mb": free, "waited_sec": waited}
        log(f"[commit] can ~{extra:.0f} MB, may con {free:.0f} MB -> cho {COMMIT_POLL_SEC:.0f} s")
        sleep(COMMIT_POLL_SEC)


def run(name: str, jobs_path: str, out_dir: str, device: str = "auto", limit: int | None = None,
        fail_fast: bool = False, reg: dict | None = None, log=print, allow_cpu_fallback: bool = False,
        check_commit: bool = True, commit_wait_min: float = 0.0, commit_fns: dict | None = None,
        check_smoke_rev: bool = True) -> dict:
    reg = load_registry() if reg is None else reg
    if name not in reg:
        raise KeyError(f"khong co may '{name}' trong engines.json")
    cfg = reg[name]
    kind = cfg["kind"]
    jobs = read_jobs(jobs_path, kind)
    os.makedirs(out_dir, exist_ok=True)
    todo = [(j, k) for j in jobs for k in pending(j, out_dir)]
    n_all = sum(int(j["n_candidates"]) for j in jobs)
    n_skip = n_all - len(todo)
    if limit is not None:
        todo = todo[:limit]
    rev = smoke_rev_check(cfg)
    summ = {"engine": name, "kind": kind, "jobs": len(jobs), "candidates": n_all, "n_skip": n_skip, "n_new": 0, "n_err": 0,
            "load_sec": 0.0, "gen_sec": 0.0, "audio_sec": 0.0, "peak_vram_mb": None, "device": None, "outputs": [], "errors": [],
            "code_rev": rev["code_rev"]}
    if not todo:
        log(f"[{name}] khong con viec ({summ['n_skip']} da co)")
        return summ
    if check_smoke_rev and not rev["ok"]:            # adapter doi sau smoke: ket qua smoke (ok, VRAM, commit) khong con dung
        msg = (f"smoke cu: adapter da doi (code_rev {rev['code_rev']}, smoke do voi {rev['smoke_rev']}); chay lai smoke "
               "(v6.tts_multi.engines.smoke run --engine ...) truoc, hoac --allow-stale-smoke")
        summ.update(load_error=msg, stale_smoke=rev)
        summ["errors"].append(msg)
        log(f"[{name}] {msg}")
        return summ
    from v6.tts_multi.engines.base import free_memory, peak_commit_mb, peak_vram_mb, set_seed
    if check_commit:                                # truoc khi nap: du commit RAM khong (tranh no pagefile tren C:)
        chk = commit_check(cfg, commit_wait_min, log, **(commit_fns or {}))
        summ["commit"] = chk
        if not chk["ok"]:
            msg = (f"thieu commit RAM: can ~{chk['need_mb']:.0f} MB, may con {chk['free_mb']:.0f} MB (khong nap de "
                   "khong no pagefile tren C:); dong bot ung dung roi chay lai, hoac --allow-pagefile-growth")
            summ.update(fatal=msg, load_error=msg)
            summ["errors"].append(msg)
            log(f"[{name}] {msg}")
            return summ
    t0 = time.time()
    try:
        eng, fallback = load_engine(name, cfg, device, log, allow_cpu_fallback)
    except Exception as exc:  # noqa: BLE001 - loi nap: ghi summary (fatal neu bo nho/CUDA) thay vi chet khong dau vet
        msg = f"{type(exc).__name__}: {exc}"
        summ.update(load_error=msg[:1000], load_trace=traceback.format_exc()[-3000:],
                    load_sec=round(time.time() - t0, 2), peak_vram_mb=peak_vram_mb(), peak_commit_mb=peak_commit_mb())
        if is_fatal(exc) or is_oom(exc):
            summ["fatal"] = msg[:300]
        summ["errors"].append(msg[:1000])
        log(f"[{name}] LOI NAP: {msg[:300]}" + (" (khong phuc hoi; chay lai sau khi giai phong bo nho)"
                                                 if summ.get("fatal") else ""))
        if fail_fast:
            raise
        free_memory()
        return summ
    if fallback:
        summ["fallback"] = fallback
    try:
        summ["device"] = eng.device
        summ["load_sec"] = round(time.time() - t0, 2)
        log(f"[{name}] nap xong {summ['load_sec']} s tren {eng.device}; {len(todo)} ung vien can sinh")
        for i, (job, k) in enumerate(todo, 1):
            seed = int(job.get("seed", 0)) + k
            t1 = time.time()
            try:
                set_seed(seed)
                y, sr, params = eng.run(job, k, seed)
                dt = time.time() - t1
                rec = write_output(out_dir, name, job, k, y, sr, {"seed": seed, "gen_sec": round(dt, 3), **params})
                summ["n_new"] += 1
                summ["gen_sec"] += dt
                summ["audio_sec"] += rec["seconds"]
                summ["outputs"].append({"job_id": job["job_id"], "k": k, "seconds": rec["seconds"],
                                        "sane": rec["params"]["sanity"]["sane"], "gen_sec": round(dt, 3)})
                log(f"[{name}] {i}/{len(todo)} {job['job_id']}_c{k}: {rec['seconds']:.2f} s am thanh, {dt:.1f} s sinh, "
                    f"sane={rec['params']['sanity']['sane']}")
            except Exception as exc:  # noqa: BLE001 - ghi loi tung viec, chay tiep
                summ["n_err"] += 1
                err = {"job_id": job["job_id"], "k": k, "engine": name, "error": f"{type(exc).__name__}: {exc}",
                       "trace": traceback.format_exc()[-3000:]}
                summ["errors"].append(err["error"])
                write_atomic_json(out_paths(out_dir, job["job_id"], k)[1].replace(".json", ".err.json"), err)
                log(f"[{name}] LOI {job['job_id']}_c{k}: {err['error']}")
                if fail_fast:
                    raise
                if is_fatal(exc):            # ngu canh CUDA hong / het commit RAM: dung, chay lai se tiep tuc (resume)
                    summ["fatal"] = err["error"][:300]
                    log(f"[{name}] DUNG: loi khong phuc hoi trong tien trinh nay; chay lai lenh de tiep tuc")
                    break
                free_memory()
    finally:
        summ["peak_vram_mb"] = peak_vram_mb()
        summ["peak_commit_mb"] = peak_commit_mb()
        eng.close()
        del eng
        free_memory()
    summ["gen_sec"] = round(summ["gen_sec"], 2)
    summ["audio_sec"] = round(summ["audio_sec"], 2)
    return summ


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--allow-cpu-fallback", action="store_true",
                    help="(--device auto) nap GPU het bo nho -> nap lai CPU; mac dinh tat (ma thoat 3)")
    ap.add_argument("--commit-wait-min", type=float, default=0.0,
                    help="cho toi da M phut cho du commit RAM truoc khi nap (mac dinh 0 = thoat 3 ngay)")
    ap.add_argument("--allow-pagefile-growth", action="store_true",
                    help="bo kiem commit truoc khi nap (chap nhan pagefile no them tren C:)")
    ap.add_argument("--allow-stale-smoke", action="store_true",
                    help="van chay khi smoke trong engines.json cu hon ma adapter (smoke tu dung co nay)")
    a = ap.parse_args(argv)
    import faulthandler
    faulthandler.enable()                          # loi C (0xC0000005) van in ngan xep python vao log
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except (AttributeError, ValueError):
        pass
    from v6.tts_multi import BASE_ENV
    for k, v in BASE_ENV.items():                  # bo nho dem tren D: ke ca khi goi tay
        os.environ.setdefault(k, v)
    os.makedirs(os.environ["TMP"], exist_ok=True)
    t0 = time.time()
    summ = run(a.engine, os.path.abspath(a.jobs), os.path.abspath(a.out), a.device, a.limit, a.fail_fast,
               allow_cpu_fallback=a.allow_cpu_fallback, check_commit=not a.allow_pagefile_growth,
               commit_wait_min=a.commit_wait_min, check_smoke_rev=not a.allow_stale_smoke)
    summ["wall_sec"] = round(time.time() - t0, 2)
    if a.summary:
        write_atomic_json(a.summary, summ)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("outputs", "load_trace")}, ensure_ascii=False))
    return exit_code(summ)


def exit_code(summ: dict) -> int:
    """0 xong; 1 loi nap khong phai bo nho (thieu tep / module, smoke cu); 2 co viec loi; 3 fatal (bo nho / CUDA,
    thieu commit RAM truoc khi nap)."""
    if summ.get("fatal"):
        return 3
    if summ.get("load_error"):
        return 1
    return 0 if summ["n_err"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
