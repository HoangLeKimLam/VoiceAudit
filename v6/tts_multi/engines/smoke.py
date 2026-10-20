# -*- coding: utf-8 -*-
"""Thu nhanh (smoke) tung may: 2 cau vi (nhai VIVOS) hoac 2 cau en (nhai LibriSpeech), 2 viec VC, 1 viec khu on.
Do: giay sinh, giay am thanh, RTF, VRAM dinh (torch.cuda.max_memory_allocated), am thanh co hop ly khong
(do dai, khong cam, toc do tu; VC: resemblyzer sim(ra, dich) > sim(ra, nguon) + 0,05 = doi giong that;
khu on: LSD toi ban sach giam). Ghi ket qua vao truong 'smoke' cua v6/tts_multi/engines.json.

Dung (qua khoa may; tien trinh nay nhe, tien trinh con nap mo hinh):
  bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.tts_multi.engines.smoke prepare
  bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.tts_multi.engines.smoke run --engine f5_vi [--device cpu]
  ... smoke asr [--engine ...]        # CER bang ASR CPU (doc dung chu?) cho dau ra TTS da ok
  ... smoke show                      # bang tom tat tu engines.json
Tham chieu chi tu kho cho phep (quyet dinh 2): VIVOS test VIVOSDEV05 (vi), LibriSpeech test-clean 1089 (en);
nguon VC: VIVOSDEV18 (nu, khac gioi dich) / LibriSpeech 1188 (nguoi KHAC); nguon khu on: VIVOSDEV01 + nhieu trang.
Tep tham chieu va cau sinh ra la cac ban ghi khac nhau.
kNN-VC (jobs_vc_knn.jsonl) them: tep dich dai 129,5 s (het VIVOSDEV05) va, neu WP7b da tao, tep dich AMI 224 s
long_amimee071_300 (data_labels/v6/vcmulti/_work/refs; AMI = kho cho phep) — tep co doan im lang dai ma ban 12/10
(VAD tung mieng) da hong; nguon en LibriSpeech 1188.
Kiem commit RAM truoc (nhu run_engine): khong du -> BO QUA may do, GIU nguyen ket qua smoke cu trong engines.json
(khong ghi ok=False chi vi may dang thieu bo nho); --allow-pagefile-growth de bo kiem. Dau ra cu duoc chuyen vao
out/<may>/_prev truoc khi chay va TRA LAI neu tien trinh con tu choi nap (commit tut giua hai lan kiem); chay that
xong (ok hay hong) thi xoa _prev. Smoke ghi code_rev (phien ban adapter) -> run_engine biet smoke con hop le khong;
tien trinh con cua smoke chay voi --allow-stale-smoke.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import soundfile as sf

from v6.tts_multi import REGISTRY_PATH, ROOT, engine_env, registry
from v6.tts_multi.engines.base import as_list

SMOKE = "D:/va_gen/smoke"
LOGS = "D:/va_gen/logs"
PREV = "_prev"                    # thu muc con giu dau ra smoke cu trong luc chay lai
VIVOS = ROOT + "/data/vivos_labeled"
# Tep dich dai cua WP7b (AMI, nguoi MEE071, 224 s, nhieu doan im lang): ca kho cho phep; khong co -> bo viec nay
KNN_AMI_LONG = ROOT + "/data_labels/v6/vcmulti/_work/refs/long_amimee071_300.wav"
LIBRI = ROOT + "/data/librispeech/LibriSpeech/test-clean"
TIMEOUT_SEC = 9 * 60
SENT = {
    "vi": ["Hôm nay trời mưa nhẹ nên chúng tôi quyết định ở nhà đọc sách và nấu một bữa cơm thật ngon.",
           "Cuộc họp sẽ bắt đầu lúc chín giờ sáng thứ hai, mọi người nhớ mang theo báo cáo của tuần trước nhé."],
    "en": ["The meeting will start at nine in the morning, so please bring the report from last week.",
           "We walked along the river for an hour and talked about the new project until it got dark."],
}
SEC_PER_WORD = (0.12, 1.2)        # toc do hop ly (giay / tu) cho cau TTS
# Nguon VC vi: nu, f0 ~241 Hz, resemblyzer 0,585 voi dich VIVOSDEV05 (nam, ~122 Hz) — do 11/10 tren 4 cau dau.
# Truoc day dung VIVOSDEV01 (nam, sim 0,83 voi dich) -> qua giong dich, khong kiem duoc VC co doi giong that.
VI_VC_SRC = "VIVOSDEV18"
VC_SIM_MARGIN = 0.05              # VC dat: sim(ra, dich) - sim(ra, nguon) >= bien nay (resemblyzer)


def _vivos_ref(spk: str, min_sec: float = 6.0, max_sec: float = 10.0) -> tuple[list, str]:
    rows = [json.loads(x) for x in open(f"{VIVOS}/metadata_test.jsonl", encoding="utf-8") if spk in x]
    rows = sorted([r for r in rows if r["speaker_id"] == spk], key=lambda r: r["file"])
    files, texts, tot = [], [], 0.0
    for r in rows:
        if tot + r["duration"] > max_sec:
            continue
        files.append(f"{VIVOS}/{r['file']}")
        texts.append(r["sentence"])
        tot += r["duration"] + 0.15
        if tot >= min_sec:
            break
    return files, " ".join(texts)


def _libri_ref(spk_dir: str, min_sec: float = 6.0, max_sec: float = 10.5) -> tuple[list, str]:
    trans = {}
    for t in glob.glob(f"{LIBRI}/{spk_dir}/*/*.trans.txt"):
        for line in open(t, encoding="utf-8"):
            uid, txt = line.strip().split(" ", 1)
            trans[uid] = txt.lower()
    for uid in sorted(trans):
        f = glob.glob(f"{LIBRI}/{spk_dir}/*/{uid}.flac")[0]
        d = sf.info(f).duration
        if min_sec <= d <= max_sec:
            return [f], trans[uid]
    raise RuntimeError(f"khong tim duoc ref {min_sec}-{max_sec} s cho {spk_dir}")


def _write_jsonl(path: str, rows: list) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare() -> dict:
    """Tao tep tham chieu + jobs smoke trong D:/va_gen/smoke."""
    os.makedirs(f"{SMOKE}/refs", exist_ok=True)
    from v6.tts_multi.engines.base import concat_refs
    vi_files, vi_text = _vivos_ref("VIVOSDEV05")
    en_files, en_text = _libri_ref("1089")
    vi_ref = concat_refs(vi_files, sr=16000)
    out = {}
    for lang, ref, rtxt in (("vi", vi_ref, vi_text), ("en", en_files[0], en_text)):
        rows = [{"job_id": f"smoke_{lang}_{i}", "text": s, "lang": lang, "ref_wav": ref, "ref_text": rtxt,
                 "seed": 1234, "n_candidates": 1} for i, s in enumerate(SENT[lang])]
        _write_jsonl(f"{SMOKE}/jobs_tts_{lang}.jsonl", rows)
        out[lang] = {"ref": ref, "ref_text": rtxt, "ref_files": vi_files if lang == "vi" else en_files}
    vi_src = concat_refs(_vivos_ref(VI_VC_SRC, 3.0, 6.0)[0], sr=16000)
    en_src = _libri_ref("1188", 3.0, 8.0)[0][0]
    vi_long = concat_refs(_vivos_ref(VI_VC_SRC, 24.0, 26.5)[0], sr=16000)   # nguon dai: kiem cua so 30 s Seed-VC
    vc = [{"job_id": "smoke_vc_vi", "src_wav": vi_src, "target_ref_wav": vi_ref, "seed": 7},
          {"job_id": "smoke_vc_en", "src_wav": en_src, "target_ref_wav": en_files[0], "seed": 7},
          {"job_id": "smoke_vc_vi_long", "src_wav": vi_long, "target_ref_wav": vi_ref, "seed": 7}]
    _write_jsonl(f"{SMOKE}/jobs_vc.jsonl", vc)
    # kNN-VC: WP7b dua MOT tep dich dai toi 300 s -> them viec tham chieu dai ~130 s (het VIVOSDEV05) cho rieng knnvc
    long_ref = concat_refs(_vivos_ref("VIVOSDEV05", 300.0, 300.0)[0], sr=16000)
    knn = vc + [{"job_id": "smoke_vc_vi_longref", "src_wav": vi_src, "target_ref_wav": long_ref, "seed": 7}]
    if os.path.exists(KNN_AMI_LONG):              # tep AMI dai co doan im lang (ban VAD tung mieng da hong o day)
        knn.append({"job_id": "smoke_vc_en_amilong", "src_wav": en_src, "target_ref_wav": KNN_AMI_LONG, "seed": 7})
    _write_jsonl(f"{SMOKE}/jobs_vc_knn.jsonl", knn)
    out["vc_knn"] = [j["job_id"] for j in knn]
    dn_src = concat_refs(_vivos_ref("VIVOSDEV01", 3.0, 6.0)[0], sr=16000)
    y, sr = sf.read(dn_src, dtype="float32")             # nguon + nhieu trang ~10 dB SNR cho may khu on
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(y)).astype(np.float32)
    noise *= np.sqrt(np.mean(y ** 2) / (np.mean(noise ** 2) * 10 ** (10 / 10)))
    noisy = f"{SMOKE}/refs/noisy_vivosdev01.wav"
    sf.write(noisy, np.clip(y + noise, -1, 1), sr, subtype="PCM_16")
    _write_jsonl(f"{SMOKE}/jobs_dn.jsonl", [{"job_id": "smoke_dn_vi", "src_wav": noisy, "clean_wav": dn_src, "seed": 3,
                                             "n_candidates": 4}])      # 4 = moi che do cua resenh
    out["vc"], out["dn"] = vc, noisy
    return out


def jobs_for(cfg: dict) -> str:
    if cfg["kind"] == "vc":
        return f"{SMOKE}/jobs_vc_knn.jsonl" if cfg.get("family") == "knnvc" else f"{SMOKE}/jobs_vc.jsonl"
    if cfg["kind"] == "denoise_ai":
        return f"{SMOKE}/jobs_dn.jsonl"
    return f"{SMOKE}/jobs_tts_{'vi' if 'vi' in cfg['langs'] else 'en'}.jsonl"


_ENC = {}


def speaker_embed(path: str) -> np.ndarray:
    """Vector nguoi noi resemblyzer (CPU, .venv chinh), chuan hoa L2."""
    from resemblyzer import VoiceEncoder, preprocess_wav
    from v6.tts_multi.engines.base import resample
    if "enc" not in _ENC:
        _ENC["enc"] = VoiceEncoder("cpu", verbose=False)
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    e = _ENC["enc"].embed_utterance(preprocess_wav(resample(y.mean(axis=1), sr, 16000), 16000))
    return e / (np.linalg.norm(e) + 1e-9)


def _logspec(y: np.ndarray, n_fft: int = 512, hop: int = 256) -> np.ndarray:
    n = 1 + max(0, len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    fr = np.pad(y, (0, max(0, n_fft - len(y))))[idx] * np.hanning(n_fft)
    return np.log10(np.abs(np.fft.rfft(fr, axis=1)) ** 2 + 1e-8)


def lsd(ref_path: str, test, sr_test: int | None = None) -> float:
    """Log-spectral distance (dB-ish, khong phu thuoc pha) giua tep sach va tin hieu thu (duong dan hoac mang)."""
    from v6.tts_multi.engines.base import resample
    r, sr = sf.read(ref_path, dtype="float32", always_2d=True)
    r = r.mean(axis=1)
    if isinstance(test, str):
        t, sr_test = sf.read(test, dtype="float32", always_2d=True)
        t = t.mean(axis=1)
    else:
        t = np.asarray(test, np.float32)
    t = resample(t, int(sr_test), sr)
    n = min(len(r), len(t))
    a, b = _logspec(r[:n]), _logspec(t[:n])
    return float(np.mean(np.sqrt(np.mean((10 * (a - b)) ** 2, axis=1))))


def _vc_check(job: dict, wav: str, embed) -> dict:
    """Doi giong that? sim(ra, dich) phai hon sim(ra, nguon) it nhat VC_SIM_MARGIN."""
    tgt = as_list(job["target_ref_wav"])
    e_out, e_src = embed(wav), embed(job["src_wav"])
    e_tgt = np.mean([embed(t) for t in tgt], axis=0)
    e_tgt = e_tgt / (np.linalg.norm(e_tgt) + 1e-9)
    st, ss = float(np.dot(e_out, e_tgt)), float(np.dot(e_out, e_src))
    return {"sim_out_tgt": round(st, 3), "sim_out_src": round(ss, 3), "sim_src_tgt": round(float(np.dot(e_src, e_tgt)), 3),
            "conv_ok": st - ss >= VC_SIM_MARGIN}


def _dn_check(job: dict, wav: str) -> dict:
    """Khu on co lam tin hieu gan ban sach hon ban on khong (LSD, khong phu thuoc pha)."""
    li, lo = lsd(job["clean_wav"], job["src_wav"]), lsd(job["clean_wav"], wav)
    return {"lsd_in": round(li, 2), "lsd_out": round(lo, 2), "dn_ok": lo < li}


def judge(cfg: dict, jobs_path: str, out_dir: str, summ: dict, embed=None) -> dict:
    """Danh gia dau ra: sane (do dai + khong cam); TTS: toc do tu; VC: do dai + doi giong that (resemblyzer);
    denoise_ai: do dai + LSD toi ban sach giam (neu job co clean_wav). embed: ham vector nguoi noi (test thay duoc)."""
    embed = speaker_embed if embed is None else embed
    jobs = {j["job_id"]: j for j in map(json.loads, open(jobs_path, encoding="utf-8"))}
    checks = []
    for o in summ.get("outputs", []):
        stem = os.path.join(out_dir, f"{o['job_id']}_c{o['k']}")
        rec = json.load(open(stem + ".json", encoding="utf-8"))
        job = jobs[o["job_id"]]
        san = rec["params"]["sanity"]
        c = {"job": f"{o['job_id']}_c{o['k']}", "sec": rec["seconds"], "rms_db": san["rms_db"], "sane": san["sane"]}
        if cfg["kind"] == "tts":
            spw = rec["seconds"] / max(1, len(job["text"].split()))
            c["sec_per_word"] = round(spw, 3)
            c["rate_ok"] = SEC_PER_WORD[0] <= spw <= SEC_PER_WORD[1]
        else:
            c["src_sec"] = round(sf.info(job["src_wav"]).duration, 3)
            c["len_ratio"] = round(rec["seconds"] / max(1e-3, c["src_sec"]), 3)
            c["rate_ok"] = 0.8 <= c["len_ratio"] <= 1.25
            if cfg["kind"] == "vc" and c["sane"]:
                c.update(_vc_check(job, stem + ".wav", embed))
            elif cfg["kind"] == "denoise_ai" and c["sane"] and job.get("clean_wav"):
                c.update(_dn_check(job, stem + ".wav"))
        checks.append(c)
    ok = (bool(checks) and summ.get("n_err", 1) == 0
          and all(c["sane"] and c["rate_ok"] and c.get("conv_ok", True) and c.get("dn_ok", True) for c in checks))
    res = {"ok": ok, "checks": checks}
    by_lang = {}                                      # VC theo ngon ngu (vd doi giong tot en nhung yeu vi)
    for c in checks:
        if "conv_ok" in c:
            lang = c["job"].split("_")[2]
            by_lang[lang] = by_lang.get(lang, True) and c["conv_ok"]
    if by_lang:
        res["conv_ok_by_lang"] = by_lang
    return res


def update_registry(name: str, smoke: dict, path: str = REGISTRY_PATH) -> None:
    with open(path, encoding="utf-8") as fh:
        reg = json.load(fh)
    reg[name]["smoke"] = smoke
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def _files(d: str) -> list:
    return [f for f in glob.glob(f"{d}/*") if os.path.isfile(f)]


def stash_outputs(out_dir: str) -> str:
    """Chuyen dau ra smoke cu (tep) vao out_dir/_prev (xoa _prev cu truoc). Tra duong _prev."""
    prev = os.path.join(out_dir, PREV)
    shutil.rmtree(prev, ignore_errors=True)
    os.makedirs(prev, exist_ok=True)
    for f in _files(out_dir):
        os.replace(f, os.path.join(prev, os.path.basename(f)))
    return prev


def restore_outputs(out_dir: str) -> int:
    """Bo tep moi trong out_dir, tra dau ra cu tu _prev ve. Tra so tep tra ve (khong co _prev -> 0)."""
    prev = os.path.join(out_dir, PREV)
    if not os.path.isdir(prev):
        return 0
    for f in _files(out_dir):
        os.remove(f)
    old = _files(prev)
    for f in old:
        os.replace(f, os.path.join(out_dir, os.path.basename(f)))
    shutil.rmtree(prev, ignore_errors=True)
    return len(old)


def run_smoke(name: str, device: str = "auto", fresh: bool = True, allow_growth: bool = False,
              commit_fns: dict | None = None, _run=None) -> dict:
    """Chay smoke mot may (tien trinh con) roi ghi ket qua vao engines.json. Thieu commit RAM (kiem nhu run_engine, o
    day hoac trong tien trinh con) -> tra {skipped, commit}, giu dau ra cu (tra tu _prev) va KHONG ghi registry.
    commit_fns: thay ham do commit; _run: thay subprocess.run (test)."""
    from v6.tts_multi.run_engine import commit_check
    run_fn = subprocess.run if _run is None else _run
    cfg = registry()[name]
    if not allow_growth:
        chk = commit_check(cfg, 0.0, log=lambda *_: None, **(commit_fns or {}))
        if not chk["ok"]:
            return {"skipped": f"thieu commit RAM: can ~{chk['need_mb']:.0f} MB, may con {chk['free_mb']:.0f} MB "
                               "(giu ket qua smoke cu)", "commit": chk}
    jobs = jobs_for(cfg)
    if not os.path.exists(jobs):
        prepare()
    out_dir = f"{SMOKE}/out/{name}"
    os.makedirs(out_dir, exist_ok=True)
    if fresh:                                              # do lai tu dau (thoi gian + VRAM that); cu -> _prev
        stash_outputs(out_dir)
    summ_p = f"{out_dir}/summary.json"
    if os.path.exists(summ_p):                             # --keep: summary cu khong duoc lan voi lan chay nay
        os.remove(summ_p)
    argv = [cfg["env_python"], "-m", "v6.tts_multi.run_engine", "--engine", name, "--jobs", jobs, "--out", out_dir,
            "--summary", summ_p, "--device", device, "--allow-stale-smoke"] + (
        ["--allow-pagefile-growth"] if allow_growth else [])
    os.makedirs(LOGS, exist_ok=True)
    log_p = f"{LOGS}/smoke_{name}.log"
    res = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "device_req": device}
    try:
        with open(log_p, "w", encoding="utf-8") as lf:
            rc = run_fn(argv, env=engine_env(cfg), cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                        timeout=TIMEOUT_SEC).returncode
        res["rc"] = rc
    except subprocess.TimeoutExpired:
        res.update(rc=None, ok=False, error=f"qua {TIMEOUT_SEC} s")
    except FileNotFoundError as exc:
        res.update(rc=None, ok=False, error=f"khong co python env: {exc}")
    summ = json.load(open(summ_p, encoding="utf-8")) if os.path.exists(summ_p) else None
    if summ is not None and (summ.get("commit") or {}).get("ok") is False:   # con tu choi nap (thieu commit): giu cu
        n_back = restore_outputs(out_dir)
        return {"skipped": summ.get("fatal"), "commit": summ["commit"], "log": log_p, "restored_files": n_back}
    shutil.rmtree(os.path.join(out_dir, PREV), ignore_errors=True)        # chay that (ok hay hong): bo dau ra cu
    if summ is not None:
        res.update({k: summ.get(k) for k in ("device", "load_sec", "gen_sec", "audio_sec", "peak_vram_mb", "peak_commit_mb", "fatal", "load_error",
                                             "n_new", "n_err", "commit", "code_rev")})
        res["rtf"] = round(summ["gen_sec"] / summ["audio_sec"], 3) if summ.get("audio_sec") else None
        res.update(judge(cfg, jobs, out_dir, summ))
        if summ.get("errors"):
            res["error"] = summ["errors"][0][:400]
    elif "error" not in res:
        tail = open(log_p, encoding="utf-8", errors="replace").read()[-1500:]
        res.update(ok=False, error="khong co summary; cuoi log: " + tail)
    res["log"] = log_p
    update_registry(name, res)
    return res


def _norm(text: str, lang: str) -> str:
    import re
    t = text.lower() if lang == "vi" else text.upper()
    t = re.sub(r"[^\w' ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def asr_check(names: list | None = None) -> dict:
    """CER cua dau ra TTS smoke bang ASR CPU co san (vi: wav2vec2-base-vietnamese-250h; en: wav2vec2-base-960h).
    Ghi smoke.cer (trung binh) + smoke.cer_items vao engines.json. Chi la kiem 'doc dung chu', khong do do tu nhien."""
    import vaudit.asr as asr
    from v6.tts_multi.engines.base import resample
    reg = registry()
    out = {}
    for lang, model in (("vi", "nguyenvulebinh/wav2vec2-base-vietnamese-250h"), ("en", "facebook/wav2vec2-base-960h")):
        todo = [n for n, c in reg.items() if c["kind"] == "tts" and (c.get("smoke") or {}).get("ok")
                and jobs_for(c).endswith(f"_{lang}.jsonl") and (names is None or n in names)]
        if not todo:
            continue
        asr.ASR_NAME = model
        asr._state.clear()
        jobs = {j["job_id"]: j for j in map(json.loads, open(f"{SMOKE}/jobs_tts_{lang}.jsonl", encoding="utf-8"))}
        for n in todo:
            items = []
            for jid, j in jobs.items():
                w = f"{SMOKE}/out/{n}/{jid}_c0.wav"
                if not os.path.exists(w):
                    continue
                y, sr = sf.read(w, dtype="float32")
                hyp, _ = asr.transcribe_words(resample(y, sr, 16000), 16000)
                c = float(asr.cer(_norm(j["text"], lang), _norm(hyp, lang)))
                items.append({"job": jid, "cer": round(c, 3), "hyp": hyp[:200]})
            if items:
                s = dict(reg[n]["smoke"])
                s["cer"] = round(float(np.mean([i["cer"] for i in items])), 3)
                s["cer_items"] = items
                s["cer_asr"] = model
                update_registry(n, s)
                out[n] = s["cer"]
    return out


def show() -> None:
    from v6.tts_multi.run_engine import smoke_rev_check
    for name, cfg in registry().items():
        s = cfg.get("smoke") or {}
        rev = smoke_rev_check(cfg)
        stale = "" if rev["ok"] else f" CU(smoke rev {rev['smoke_rev']} != ma {rev['code_rev']})"
        print(f"{name:18s} {cfg['kind']:10s} {','.join(cfg['langs']):6s} {cfg['role']:8s} ok={s.get('ok')!s:5s} "
              f"dev={s.get('device')} vram={s.get('peak_vram_mb')} rtf={s.get('rtf')}{stale} {(s.get('error') or '')[:80]}")


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="smoke test may sinh giong")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    r = sub.add_parser("run")
    r.add_argument("--engine", required=True, nargs="+")
    r.add_argument("--device", default="auto")
    r.add_argument("--keep", action="store_true", help="khong xoa dau ra cu (resume)")
    r.add_argument("--allow-pagefile-growth", action="store_true", help="bo kiem commit RAM (pagefile co the no tren C:)")
    sub.add_parser("show")
    q = sub.add_parser("asr")
    q.add_argument("--engine", nargs="*", default=None)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False, indent=1))
    elif a.cmd == "run":
        for e in a.engine:
            res = run_smoke(e, a.device, fresh=not a.keep, allow_growth=a.allow_pagefile_growth)
            print(json.dumps({e: res}, ensure_ascii=False, indent=1))
    elif a.cmd == "asr":
        print(json.dumps(asr_check(a.engine), ensure_ascii=False))
    else:
        show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
