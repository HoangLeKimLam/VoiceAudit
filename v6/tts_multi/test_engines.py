# -*- coding: utf-8 -*-
"""Kiem v6.tts_multi (khong nap mo hinh that): danh muc engines.json, doc jobs, vong chay run_engine voi may gia
'tone' (ghi dung ten tep I3, resume, ghi loi tung viec), tien ich am thanh (ban ASCII, mono), lenh + bien moi truong.

Chay: bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.tts_multi.test_engines
"""
from __future__ import annotations

import json
import os
import py_compile
import shutil
import sys
import tempfile

import numpy as np
import soundfile as sf

from v6.tts_multi import PKG_DIR, ROOT, TORCH_CU, command, registry
from v6.tts_multi import run_engine as RE
from v6.tts_multi.engines import base as B

KINDS = {"tts", "vc", "denoise_ai"}
ROLES = {"train", "heldout"}
ALLOWED_LANGS = {"vi", "en"}
FORBIDDEN = ("hynt", "ViVoice-1000h", "EraX", "viterbox", "VoxCPM")
TONE = {"tone": {"kind": "tts", "langs": ["vi"], "env_python": sys.executable, "license": "-", "role": "train",
                 "notes": "", "module": "tone", "opts": {"fail_on": "bad"}}}


def _tmp() -> str:
    os.makedirs("D:/tmp_va", exist_ok=True)
    return tempfile.mkdtemp(prefix="tts_multi_test_", dir="D:/tmp_va")


def _jobs(path: str, rows: list) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def test_registry_schema():
    reg = registry()
    assert len(reg) >= 10, len(reg)
    for name, c in reg.items():
        for f in ("kind", "langs", "env_python", "license", "role", "notes", "module", "family"):
            assert f in c, (name, f)
        assert c["kind"] in KINDS, name
        assert c["role"] in ROLES, name
        assert set(c["langs"]) <= ALLOWED_LANGS and c["langs"], name
        assert os.path.exists(os.path.join(PKG_DIR, "engines", c["module"] + ".py")), name
        blob = json.dumps(c, ensure_ascii=False)
        assert not any(b.lower() in blob.lower() for b in FORBIDDEN), (name, "mo hinh bi cam")
        assert c["env_python"].replace("\\", "/").startswith(("D:/va_gen/envs/", ROOT + "/.venv/")), name
    assert {"vi", "en"} <= {lang for c in reg.values() if c["kind"] == "tts" for lang in c["langs"]}
    assert any(c["role"] == "heldout" and c["kind"] == "tts" for c in reg.values())
    assert any(c["role"] == "heldout" and c["kind"] == "vc" for c in reg.values())
    # held-out khong chung ho voi may train cung loai
    for kind in ("tts", "vc"):
        tr = {c["family"] for c in reg.values() if c["kind"] == kind and c["role"] == "train"}
        ho = {c["family"] for c in reg.values() if c["kind"] == kind and c["role"] == "heldout"}
        assert not (tr & ho), (kind, tr & ho)


def test_all_modules_compile():
    for f in os.listdir(os.path.join(PKG_DIR, "engines")):
        if f.endswith(".py"):
            py_compile.compile(os.path.join(PKG_DIR, "engines", f), doraise=True)
    py_compile.compile(os.path.join(PKG_DIR, "run_engine.py"), doraise=True)


def test_read_jobs_validation():
    d = _tmp()
    try:
        p = _jobs(f"{d}/ok.jsonl", [{"job_id": "a", "text": "xin chao", "lang": "vi"}, {}][:1])
        js = RE.read_jobs(p, "tts")
        assert js[0]["seed"] == 0 and js[0]["n_candidates"] == 1
        for bad, kind in (([{"job_id": "a", "text": "x"}], "tts"),                       # thieu lang
                          ([{"job_id": "a", "text": "x", "lang": "vi"}] * 2, "tts"),     # trung id
                          ([{"job_id": "a/b", "text": "x", "lang": "vi"}], "tts"),       # ky tu xau
                          ([{"job_id": "a", "src_wav": "x.wav"}], "vc")):               # thieu target_ref_wav
            try:
                RE.read_jobs(_jobs(f"{d}/bad.jsonl", bad), kind)
            except ValueError:
                continue
            raise AssertionError(f"phai bao loi: {bad}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_run_loop_outputs_resume_and_errors():
    d = _tmp()
    try:
        jobs = _jobs(f"{d}/j.jsonl", [
            {"job_id": "u1", "text": "mot hai ba bon", "lang": "vi", "ref_wav": "x", "ref_text": "", "seed": 5,
             "n_candidates": 2},
            {"job_id": "bad", "text": "loi", "lang": "vi", "seed": 1, "n_candidates": 1}])
        out = f"{d}/out"
        s = RE.run("tone", jobs, out, device="cpu", reg=TONE, log=lambda *_: None)
        assert s["n_new"] == 2 and s["n_err"] == 1 and s["n_skip"] == 0, s
        for k in (0, 1):
            w, j = RE.out_paths(out, "u1", k)
            rec = json.load(open(j, encoding="utf-8"))
            assert set(rec) >= {"job_id", "k", "engine", "sr", "seconds", "params"}, rec
            assert rec["job_id"] == "u1" and rec["k"] == k and rec["engine"] == "tone" and rec["sr"] == 16000
            assert rec["params"]["seed"] == 5 + k
            y, sr = sf.read(w)
            assert sr == 16000 and y.ndim == 1 and abs(len(y) / sr - rec["seconds"]) < 1e-2
            assert rec["params"]["sanity"]["sane"]
        assert os.path.exists(f"{out}/bad_c0.err.json")
        a = sf.read(RE.out_paths(out, "u1", 0)[0])[0]
        b = sf.read(RE.out_paths(out, "u1", 1)[0])[0]
        assert not np.allclose(a, b), "ung vien k khac seed phai khac nhau"
        s2 = RE.run("tone", jobs, out, device="cpu", reg=TONE, log=lambda *_: None)
        assert s2["n_new"] == 0 and s2["n_skip"] == 2 and s2["n_err"] == 1, s2    # chi viec loi chay lai
        os.remove(RE.out_paths(out, "u1", 1)[1])                                    # mat .json -> lam lai
        s3 = RE.run("tone", jobs, out, device="cpu", reg=TONE, limit=1, log=lambda *_: None)
        assert s3["n_new"] == 1 and s3["outputs"][0]["k"] == 1, s3
        assert not [f for f in os.listdir(out) if ".tmp" in f], "con tep tam"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_relative_paths_become_absolute():
    d = _tmp()
    old = os.getcwd()
    try:
        os.chdir(d)
        p = _jobs(f"{d}/rel.jsonl", [{"job_id": "v", "src_wav": "a/src.wav", "target_ref_wav": ["r1.wav", "r2.wav"]}])
        job = RE.read_jobs(p, "vc")[0]
        assert job["src_wav"] == os.path.join(d, "a", "src.wav"), job
        assert job["target_ref_wav"] == [os.path.join(d, "r1.wav"), os.path.join(d, "r2.wav")], job
        assert os.path.isabs(RE.read_jobs(p, "vc")[0]["src_wav"])
    finally:
        os.chdir(old)
        shutil.rmtree(d, ignore_errors=True)


def test_load_falls_back_to_cpu_on_cuda_oom():
    d = _tmp()
    orig = B.pick_device
    try:
        B.pick_device = lambda want="auto": "cuda" if want == "auto" else want      # gia lap may co GPU
        reg = {"tone": {**TONE["tone"], "opts": {"oom_device": "cuda"}}}
        jobs = _jobs(f"{d}/j.jsonl", [{"job_id": "a", "text": "mot hai", "lang": "vi", "seed": 1}])
        s = RE.run("tone", jobs, f"{d}/out", device="auto", reg=reg, log=lambda *_: None, allow_cpu_fallback=True)
        assert s["device"] == "cpu" and s["n_new"] == 1 and "cuda->cpu" in s.get("fallback", ""), s
        for dev in ("auto", "cuda"):                         # mac dinh (khong opt-in) / ep cuda -> KHONG chuyen CPU
            s = RE.run("tone", jobs, f"{d}/out_{dev}", device=dev, reg=reg, log=lambda *_: None)
            assert s["n_new"] == 0 and "fallback" not in s and "out of memory" in s.get("fatal", ""), s
            assert RE.exit_code(s) == 3 and s["load_error"], s
        assert RE.is_oom(RuntimeError("CUDA out of memory. Tried")) and not RE.is_oom(ValueError("x"))
    finally:
        B.pick_device = orig
        shutil.rmtree(d, ignore_errors=True)


def test_load_error_writes_summary_and_exit_code():
    d = _tmp()
    try:
        jobs = _jobs(f"{d}/j.jsonl", [{"job_id": "a", "text": "mot hai", "lang": "vi", "seed": 1}])
        reg = {"tone": {**TONE["tone"], "opts": {"oom_device": "cpu"}}}
        summ_p = f"{d}/s.json"                                                   # qua main(): summary + ma thoat 3
        orig = RE.load_registry
        RE.load_registry = lambda: reg
        try:
            rc = RE.main(["--engine", "tone", "--jobs", jobs, "--out", f"{d}/o", "--device", "cpu", "--summary", summ_p])
        finally:
            RE.load_registry = orig
        s = json.load(open(summ_p, encoding="utf-8"))
        assert rc == 3 and s["fatal"] and s["n_new"] == 0, (rc, s)
        assert RE.exit_code({"n_err": 0, "load_error": "ModuleNotFoundError: x"}) == 1
        assert RE.exit_code({"n_err": 2}) == 2 and RE.exit_code({"n_err": 0}) == 0
        assert RE.is_fatal(OSError("The paging file is too small for this operation"))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_seedvc_window():
    from v6.tts_multi.engines import seedvc as S
    fps = S.SR / S.HOP
    ctx = S.max_ctx_frames()
    assert ctx == 2580
    for src_sec, ref_sec in ((25, 10), (20, 10), (5, 25), (3, 6), (28, 10), (24, 30), (60, 10), (180, 8)):
        r = S.fit_ref_seconds(src_sec, ref_sec)
        assert r >= min(ref_sec, S.MIN_REF_SEC) - 1e-6, (src_sec, ref_sec, r)
        assert r <= min(ref_sec, S.MAX_REF_SEC) + 1e-9
        mel2 = int(np.ceil(r * fps)) + 1                                         # mel cua tham chieu (lam tron len)
        n_cond = int(np.ceil(src_sec * fps)) + 1
        spans = S.chunk_spans(n_cond, ctx - mel2)                                # khong duoc nem loi
        assert spans[0][0] == 0 and spans[-1][1] == n_cond
        assert all(e - s <= ctx - mel2 for s, e in spans)
        assert all(spans[i + 1][0] == spans[i][1] - S.OVERLAP_FRAMES for i in range(len(spans) - 1))
        if src_sec + min(ref_sec, S.MIN_REF_SEC) <= 29.5:                        # vua cua so -> mot lan, khong moi noi
            assert len(spans) == 1, (src_sec, ref_sec, r, spans)
    assert S.fit_ref_seconds(20, 10) > 9.5                                       # nguon 20 s giu gan du 10 s tham chieu
    assert len(S.chunk_spans(100, 50)) == 3                                      # 0-50, 34-84, 68-100
    try:
        S.chunk_spans(100, S.OVERLAP_FRAMES)
    except ValueError:
        pass
    else:
        raise AssertionError("cua so <= overlap phai bao loi")
    # noi song: do dai = tong - overlap*(so doan - 1); vung khong chong giu nguyen
    ov = 64
    w = [np.ones(300, np.float32), 2 * np.ones(300, np.float32), 3 * np.ones(200, np.float32)]
    y = S.stitch(w, ov)
    assert len(y) == 800 - 2 * ov, len(y)
    assert np.allclose(y[:300 - ov], 1) and np.allclose(y[300:300 - ov + 300 - ov], 2) and np.allclose(y[-100:], 3)
    assert np.all(np.diff(y[300 - ov:300]) >= -1e-6)                            # crossfade 1 -> 2 don dieu
    assert np.array_equal(S.stitch([w[0]], ov), w[0])
    ws = S.whisper_spans(16000 * 70)                                             # 0-30, 25-55, 50-70 (bo 5 s chong)
    assert ws == [(0, 480000, 0), (400000, 880000, 250), (800000, 1120000, 250)], ws
    assert S.whisper_spans(16000 * 20) == [(0, 320000, 0)]
    vs = S.vocoder_spans(1000, 430, 32)                                          # mieng giai ma BigVGAN
    assert vs == [(0, 462, 0, 32), (398, 892, 32, 32), (828, 1000, 32, 0)], vs
    assert sum((ce - cs) - tl - tr for cs, ce, tl, tr in vs) == 1000
    assert S.vocoder_spans(300) == [(0, 300, 0, 0)]


def test_resenh_lag_compensation():
    from v6.tts_multi.engines import resenh as R
    rng = np.random.default_rng(0)
    sr = 16000
    x = rng.standard_normal(sr * 2).astype(np.float32)
    for lag in (0, 106, -73):
        out = R.shift(x, -lag)                                                   # out[n] = x[n - lag]
        L, xc = R.best_lag(x, out, sr)
        assert L == lag and xc > 0.9, (lag, L, xc)
        al = R.shift(out, L)
        n = sr - 200
        assert np.allclose(al[200:n], x[200:n]), lag
    assert len(R.shift(x, 50)) == len(x)
    assert R.best_lag(x[:5], x[:5], sr) == (0, 0.0)


def test_resenh_deepspeed_stub_and_modes():
    from v6.tts_multi.engines import resenh as R
    names = ("deepspeed", "deepspeed.accelerator", "deepspeed.runtime", "deepspeed.runtime.engine",
             "deepspeed.runtime.utils")
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    try:
        R._stub_deepspeed()
        from deepspeed import DeepSpeedConfig                                    # noqa: F401 - nap duoc ten
        from deepspeed.runtime.engine import DeepSpeedEngine                     # noqa: F401
        try:
            DeepSpeedConfig({})
        except RuntimeError:
            pass
        else:
            raise AssertionError("deepspeed gia phai chan viec hoc")
        assert {m["mode"] for m in R.MODES} == {"enhance", "denoise"} and len(R.MODES) >= 2
        assert all(0 <= m.get("lambd", 0) <= 1 and 0 < m.get("nfe", 1) <= 128 for m in R.MODES)
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)


def test_sanity_flags():
    sr = 16000
    assert not RE.sanity(np.zeros(sr, np.float32), sr)["sane"]
    assert not RE.sanity(0.1 * np.ones(100, np.float32), sr)["sane"]                   # qua ngan
    assert not RE.sanity(np.array([np.nan] * sr, np.float32), sr)["sane"]
    assert RE.sanity(0.1 * np.sin(np.arange(sr) / 5).astype(np.float32), sr)["sane"]


def test_write_output_renormalises_clipping():
    d = _tmp()
    try:
        rec = RE.write_output(d, "x", {"job_id": "j"}, 0, 2.0 * np.ones(16000, np.float32), 16000, {})
        y, _ = sf.read(RE.out_paths(d, "j", 0)[0])
        assert np.max(np.abs(y)) <= 1.0 and rec["params"]["renorm_peak"] == 2.0
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ascii_wav_and_mono():
    d = _tmp()
    try:
        uni = os.path.join(d, "t\u01b0_th\u1eed")
        os.makedirs(uni)
        src = os.path.join(uni, "a.flac")
        st = np.stack([np.sin(np.arange(8000) / 3), np.zeros(8000)], 1).astype(np.float32) * 0.5
        sf.write(src, st, 8000)
        out = B.ascii_wav(src, sr=16000)
        assert B.is_ascii(out) and out.endswith(".wav")
        y, sr = sf.read(out)
        assert sr == 16000 and y.ndim == 1 and abs(len(y) - 16000) <= 2
        assert B.ascii_wav(src, sr=16000) == out                                       # bo nho dem
        cat = B.concat_refs([src, src], sr=16000, gap=0.1)
        assert abs(sf.info(cat).duration - 2.1) < 0.01
        assert B.to_mono_f32(np.ones((2, 50))).shape == (50,)
        assert B.to_mono_f32(np.ones((50, 2))).shape == (50,)
        assert B.to_mono_f32(np.ones((1, 50))).shape == (50,)
        assert B.pick([1, 2, 3], 4) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_command_env():
    argv, env = command("vixtts", "j.jsonl", "o")
    assert argv[1:4] == ["-m", "v6.tts_multi.run_engine", "--engine"] and argv[-2:] == ["--out", "o"]
    pp = env["PYTHONPATH"].split(os.pathsep)
    assert pp[0] == TORCH_CU and pp[-1] == ROOT
    argv, env = command("f5_vi", "j.jsonl", "o")
    assert env["PYTHONPATH"] == ROOT and argv[0].startswith("D:/va_gen/envs/f5")
    assert env["HF_HOME"].startswith("D:") and env["TMP"].startswith("D:")
    try:
        command("khong_co", "j", "o")
    except KeyError:
        return
    raise AssertionError("phai KeyError")


def test_smoke_judge():
    from v6.tts_multi.engines import smoke as S
    d = _tmp()
    try:
        jobs = _jobs(f"{d}/j.jsonl", [{"job_id": "s", "text": "mot hai ba bon nam sau", "lang": "vi"}])
        RE.write_output(d, "x", {"job_id": "s"}, 0, 0.1 * np.sin(np.arange(32000) / 4).astype(np.float32), 16000, {})
        summ = {"n_err": 0, "outputs": [{"job_id": "s", "k": 0}]}
        r = S.judge({"kind": "tts"}, jobs, d, summ)
        assert r["ok"] and r["checks"][0]["sec_per_word"] == round(2.0 / 6, 3), r
        RE.write_output(d, "x", {"job_id": "s"}, 0, np.zeros(32000, np.float32), 16000, {})
        assert not S.judge({"kind": "tts"}, jobs, d, summ)["ok"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_smoke_judge_vc_similarity_and_denoise():
    from v6.tts_multi.engines import smoke as S
    d = _tmp()
    try:
        sr = 16000
        t = np.arange(sr * 2) / sr
        src = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        tgt = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(f"{d}/src.wav", src, sr)
        sf.write(f"{d}/tgt.wav", tgt, sr)
        vecs = {"src": np.array([1.0, 0.0]), "tgt": np.array([0.0, 1.0])}

        def fake_embed(path):                       # tep ra mang "giong" theo f0 chiem uu the
            y, r = sf.read(path, dtype="float32")
            sp = np.abs(np.fft.rfft(y))
            f = np.argmax(sp) * r / len(y)
            v = vecs["tgt"] if f > 225 else vecs["src"]
            return v / np.linalg.norm(v)

        jobs = _jobs(f"{d}/vc.jsonl", [{"job_id": "smoke_vc_vi", "src_wav": f"{d}/src.wav",
                                        "target_ref_wav": f"{d}/tgt.wav"}])
        summ = {"n_err": 0, "outputs": [{"job_id": "smoke_vc_vi", "k": 0}]}
        RE.write_output(d, "vc", {"job_id": "smoke_vc_vi"}, 0, tgt, sr, {})        # doi giong that
        r = S.judge({"kind": "vc"}, jobs, d, summ, embed=fake_embed)
        assert r["ok"] and r["checks"][0]["conv_ok"] and r["conv_ok_by_lang"] == {"vi": True}, r
        RE.write_output(d, "vc", {"job_id": "smoke_vc_vi"}, 0, src, sr, {})        # tra lai nguon -> phai TRUOT
        r = S.judge({"kind": "vc"}, jobs, d, summ, embed=fake_embed)
        assert not r["ok"] and not r["checks"][0]["conv_ok"] and r["conv_ok_by_lang"] == {"vi": False}, r
        # khu on: LSD toi ban sach phai giam
        rng = np.random.default_rng(1)
        noisy = src + 0.1 * rng.standard_normal(len(src)).astype(np.float32)
        sf.write(f"{d}/noisy.wav", noisy, sr)
        jobs = _jobs(f"{d}/dn.jsonl", [{"job_id": "smoke_dn_vi", "src_wav": f"{d}/noisy.wav",
                                        "clean_wav": f"{d}/src.wav"}])
        summ = {"n_err": 0, "outputs": [{"job_id": "smoke_dn_vi", "k": 0}]}
        RE.write_output(d, "dn", {"job_id": "smoke_dn_vi"}, 0, src + 0.005 * rng.standard_normal(len(src)), sr, {})
        r = S.judge({"kind": "denoise_ai"}, jobs, d, summ)
        assert r["ok"] and r["checks"][0]["lsd_out"] < r["checks"][0]["lsd_in"], r
        RE.write_output(d, "dn", {"job_id": "smoke_dn_vi"}, 0, noisy + 0.2 * rng.standard_normal(len(src)), sr, {})
        assert not S.judge({"kind": "denoise_ai"}, jobs, d, summ)["ok"]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_commit_helpers_real():
    free, cur, peak = B.free_commit_mb(), B.commit_mb(), B.peak_commit_mb()
    if os.name != "nt":
        assert free is None and cur is None
        return
    assert free is not None and free > 0, free
    assert cur is not None and peak is not None and 0 < cur <= peak + 1e-6, (cur, peak)


def test_commit_check_logic():
    cfg = {"smoke": {"peak_commit_mb": 3000.0}}
    need = round((3000 - 100) * (1 + RE.COMMIT_MARGIN_FRAC) + RE.COMMIT_MARGIN_MB, 1)
    r = RE.commit_check(cfg, 0, log=lambda *_: None, free_fn=lambda: 9000.0, cur_fn=lambda: 100.0)
    assert r["ok"] and r["need_mb"] == need and r["free_mb"] == 9000.0, r
    r = RE.commit_check(cfg, 0, log=lambda *_: None, free_fn=lambda: need - 1, cur_fn=lambda: 100.0)
    assert not r["ok"] and r["waited_sec"] == 0, r
    assert RE.commit_check({**cfg, "min_commit_mb": 500}, 0, log=lambda *_: None, free_fn=lambda: 900.0,
                           cur_fn=lambda: 0.0)["ok"]                                   # min_commit_mb uu tien smoke
    assert RE.commit_check({}, 0, log=lambda *_: None, free_fn=lambda: 1.0)["ok"]      # chua co so do -> khong chan
    assert RE.commit_check(cfg, 0, log=lambda *_: None, free_fn=lambda: None)["ok"]    # khong do duoc -> khong chan
    t = [0.0]                                                                           # cho: commit tang dan
    frees = iter([1000.0, 2000.0, 5000.0])
    r = RE.commit_check(cfg, 1, log=lambda *_: None, free_fn=lambda: next(frees), cur_fn=lambda: 100.0,
                        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert r["ok"] and r["waited_sec"] == 2 * RE.COMMIT_POLL_SEC, r
    t[0] = 0.0                                                                          # het han cho -> tu choi
    r = RE.commit_check(cfg, 0.5, log=lambda *_: None, free_fn=lambda: 10.0, cur_fn=lambda: 0.0,
                        sleep=lambda s: t.__setitem__(0, t[0] + s), clock=lambda: t[0])
    assert not r["ok"] and r["waited_sec"] >= 30, r


def test_run_refuses_load_when_commit_short():
    d = _tmp()
    try:
        reg = {"tone": {**TONE["tone"], "smoke": {"peak_commit_mb": 5000.0}}}
        jobs = _jobs(f"{d}/j.jsonl", [{"job_id": "a", "text": "mot hai", "lang": "vi", "seed": 1}])
        short = {"free_fn": lambda: 1000.0, "cur_fn": lambda: 50.0}
        s = RE.run("tone", jobs, f"{d}/o", device="cpu", reg=reg, log=lambda *_: None, commit_fns=short)
        assert s["n_new"] == 0 and "thieu commit RAM" in s["fatal"] and not s["commit"]["ok"], s
        assert RE.exit_code(s) == 3 and not [f for f in os.listdir(f"{d}/o") if f.endswith(".wav")]
        s = RE.run("tone", jobs, f"{d}/o", device="cpu", reg=reg, log=lambda *_: None, commit_fns=short,
                   check_commit=False)                                                  # --allow-pagefile-growth
        assert s["n_new"] == 1 and "commit" not in s and RE.exit_code(s) == 0, s
        ok = {"free_fn": lambda: 99999.0, "cur_fn": lambda: 50.0}
        s = RE.run("tone", jobs, f"{d}/o2", device="cpu", reg=reg, log=lambda *_: None, commit_fns=ok)
        assert s["n_new"] == 1 and s["commit"]["ok"], s
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_smoke_keeps_registry_when_commit_short():
    from v6.tts_multi.engines import smoke as S
    reg = {"big": {"kind": "tts", "langs": ["vi"], "env_python": sys.executable, "module": "tone",
                   "smoke": {"ok": True, "peak_commit_mb": 6500.0}}}
    orig_reg, orig_upd = S.registry, S.update_registry
    calls = []
    try:
        S.registry = lambda: reg
        S.update_registry = lambda *a, **kw: calls.append(a)
        r = S.run_smoke("big", commit_fns={"free_fn": lambda: 4000.0, "cur_fn": lambda: 100.0})
        assert "thieu commit RAM" in r["skipped"] and not r["commit"]["ok"], r
        assert not calls, "khong duoc ghi de smoke cu khi chi thieu bo nho"
    finally:
        S.registry, S.update_registry = orig_reg, orig_upd


def test_smoke_restores_outputs_when_child_refuses():
    """Tien trinh con tu choi nap (commit tut giua hai lan kiem) -> dau ra cu tra lai, registry giu nguyen;
    chay that -> _prev bi xoa, registry ghi ket qua moi (kem code_rev); tien trinh con luon co --allow-stale-smoke."""
    from v6.tts_multi.engines import smoke as S
    d = _tmp()
    reg = {"big": {"kind": "tts", "langs": ["vi"], "env_python": sys.executable, "module": "tone",
                   "smoke": {"ok": True, "peak_commit_mb": 10.0}}}
    saved = (S.registry, S.update_registry, S.SMOKE, S.LOGS)
    calls, argvs = [], []

    def child(summ: dict, new_files: tuple = ()):
        def fake_run(argv, **kw):
            argvs.append(argv)
            out = argv[argv.index("--out") + 1]
            for f in new_files:
                open(os.path.join(out, f), "w").close()
            with open(argv[argv.index("--summary") + 1], "w", encoding="utf-8") as fh:
                json.dump(summ, fh)
            return type("R", (), {"returncode": 3 if summ.get("fatal") else 0})()
        return fake_run

    try:
        S.registry, S.update_registry, S.SMOKE, S.LOGS = (lambda: reg), (lambda *a, **kw: calls.append(a)), d, d
        _jobs(f"{d}/jobs_tts_vi.jsonl", [{"job_id": "s", "text": "mot hai ba", "lang": "vi"}])
        out = f"{d}/out/big"
        os.makedirs(out)
        for f in ("smoke_vi_0_c0.wav", "smoke_vi_0_c0.json"):
            with open(f"{out}/{f}", "w") as fh:
                fh.write("cu")
        roomy = {"free_fn": lambda: 1e6, "cur_fn": lambda: 0.0}
        refuse = {"commit": {"ok": False, "need_mb": 9e3, "free_mb": 1e3}, "fatal": "thieu commit RAM: ...",
                  "n_new": 0, "n_err": 0, "outputs": []}
        r = S.run_smoke("big", commit_fns=roomy, _run=child(refuse, ("loi_c0.err.json",)))
        assert r.get("skipped") and r["restored_files"] == 2 and not calls, r
        assert sorted(os.listdir(out)) == ["smoke_vi_0_c0.json", "smoke_vi_0_c0.wav"], os.listdir(out)
        assert open(f"{out}/smoke_vi_0_c0.wav").read() == "cu"
        assert "--allow-stale-smoke" in argvs[-1]
        done = {"commit": {"ok": True}, "n_new": 0, "n_err": 0, "outputs": [], "code_rev": "1", "gen_sec": 0.0,
                "audio_sec": 0.0}
        r = S.run_smoke("big", commit_fns=roomy, _run=child(done))
        assert not os.path.exists(f"{out}/{S.PREV}") and len(calls) == 1 and calls[0][1]["code_rev"] == "1", calls
        assert not r["ok"] and "smoke_vi_0_c0.wav" not in os.listdir(out), r          # khong co dau ra -> hong
    finally:
        S.registry, S.update_registry, S.SMOKE, S.LOGS = saved
        shutil.rmtree(d, ignore_errors=True)


def test_run_refuses_stale_smoke():
    """Adapter doi sau smoke (code_rev khac) -> run_engine khong nap (ma 1, stale_smoke); co --allow-stale-smoke ->
    chay; smoke khong ghi code_rev = "1"; chua co smoke -> khong chan."""
    d = _tmp()
    try:
        jobs = _jobs(f"{d}/j.jsonl", [{"job_id": "a", "text": "mot hai", "lang": "vi", "seed": 1}])
        stale = {"tone": {**TONE["tone"], "smoke": {"ok": True, "code_rev": "0"}}}
        s = RE.run("tone", jobs, f"{d}/o", device="cpu", reg=stale, log=lambda *_: None, check_commit=False)
        assert s["n_new"] == 0 and "smoke cu" in s["load_error"] and RE.exit_code(s) == 1, s
        assert s["stale_smoke"] == {"ok": False, "code_rev": "1", "smoke_rev": "0"} and s["code_rev"] == "1", s
        assert not [f for f in os.listdir(f"{d}/o") if f.endswith((".wav", ".err.json"))]
        s = RE.run("tone", jobs, f"{d}/o", device="cpu", reg=stale, log=lambda *_: None, check_commit=False,
                   check_smoke_rev=False)
        assert s["n_new"] == 1 and RE.exit_code(s) == 0, s
        for sm in ({"ok": True}, {"ok": True, "code_rev": "1"}, None):
            cfg = {**TONE["tone"], **({"smoke": sm} if sm is not None else {})}
            assert RE.smoke_rev_check(cfg)["ok"], sm
        knn = {"module": "coqui_vc", "family": "knnvc", "opts": {"family": "knnvc"}}
        assert not RE.smoke_rev_check({**knn, "smoke": {"ok": True}})["ok"]           # smoke 11/10 truoc ban VAD moi
        assert RE.smoke_rev_check({**knn, "smoke": {"ok": True, "code_rev": "2"}})["ok"]
        r = RE.smoke_rev_check({"module": "khong_co_module", "smoke": {"ok": True}})
        assert r["ok"] and r["code_rev"] is None and "why" in r, r                        # loi that de luc nap bao
        summ_p = f"{d}/s.json"                                                             # qua main(): ma thoat 1
        orig = RE.load_registry
        RE.load_registry = lambda: stale
        try:
            rc = RE.main(["--engine", "tone", "--jobs", jobs, "--out", f"{d}/o2", "--device", "cpu", "--summary", summ_p,
                          "--allow-pagefile-growth"])
            rc2 = RE.main(["--engine", "tone", "--jobs", jobs, "--out", f"{d}/o3", "--device", "cpu",
                           "--allow-pagefile-growth", "--allow-stale-smoke"])
        finally:
            RE.load_registry = orig
        assert rc == 1 and json.load(open(summ_p, encoding="utf-8"))["stale_smoke"]["smoke_rev"] == "0", rc
        assert rc2 == 0, rc2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_commit_required_matches_runner():
    from v6.tts_multi import commit_extra_mb, commit_required_mb
    reg = {"x": {"smoke": {"peak_commit_mb": 3000.0}}, "y": {}, "z": {"min_commit_mb": 800, "smoke": {"peak_commit_mb": 1}}}
    assert commit_required_mb("x", reg) == round(3000 * 1.1 + 256, 1) == commit_extra_mb(3000.0, 0.0)
    assert commit_required_mb("y", reg) is None and commit_required_mb("z", reg) == round(800 * 1.1 + 256, 1)
    r = RE.commit_check(reg["x"], 0, log=lambda *_: None, free_fn=lambda: 9000.0, cur_fn=lambda: 0.0)
    assert r["need_mb"] == commit_required_mb("x", reg), r                     # cung cong thuc voi run_engine
    for name, c in registry().items():                                          # danh muc that: so hop ly
        v = commit_required_mb(name)
        assert v is None or 1000 < v < 12000, (name, v)


def test_knnvc_long_target_pieces_and_cache():
    import torch
    from v6.tts_multi.engines import coqui_vc as C
    sr = 16000
    y = np.arange(int(40.5 * sr), dtype=np.float32)
    ps = C.split_pieces(y, sr)
    assert [len(p) for p in ps] == [15 * sr, 15 * sr, int(10.5 * sr)] and np.array_equal(np.concatenate(ps), y)
    assert [len(p) for p in C.split_pieces(np.zeros(int(30.4 * sr), np.float32), sr)] == [15 * sr, int(15.4 * sr)]
    assert len(C.split_pieces(np.zeros(5 * sr, np.float32), sr)) == 1
    d = _tmp()
    try:
        t = np.arange(sr * 40) / sr
        sf.write(f"{d}/tgt.wav", (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
        for i in (2, 3):
            sf.write(f"{d}/tgt{i}.wav", (0.3 * np.sin(2 * np.pi * 90 * i * t[:sr * 2])).astype(np.float32), sr)
        sf.write(f"{d}/src.wav", (0.3 * np.sin(2 * np.pi * 120 * t[:sr * 2])).astype(np.float32), sr)
        calls = {"feat": [], "vad": 0}

        class FakeKnn:                                  # thay vc_model coqui: ghi lai cach goi
            def get_matching_set(self, wavs):
                raise AssertionError("khong duoc goi get_matching_set (VAD tung mieng)")

            def get_features(self, x, vad_trigger_level=0):
                assert torch.is_inference_mode_enabled(), "WavLM phai chay trong inference_mode"
                assert x.dim() == 2 and vad_trigger_level == 0, (x.shape, vad_trigger_level)
                calls["feat"].append(x.shape[-1])
                return torch.zeros(x.shape[-1] // 320, 4)

            def match(self, q, ms, topk=None):
                calls["topk"] = topk
                return q[None] + 1

        class FakeVoc:
            def inference(self, f):
                return f.reshape(1, 1, -1)

        class FakeSyn:
            vc_model, vocoder_model, output_sample_rate = FakeKnn(), FakeVoc(), sr

        class FakeApi:
            voice_converter = FakeSyn()

        def fake_vad(x, sr=16000, trigger=7.0, vad=None):                         # VAD ca tep: dem so lan, khong cat
            calls["vad"] += 1
            return x, "both"

        eng = C.Engine("knnvc", {"opts": {"family": "knnvc", "coqui_id": "x"}}, "cpu")
        eng.api, eng.sr, eng.family = FakeApi(), sr, "knnvc"
        job = {"job_id": "a", "src_wav": f"{d}/src.wav", "target_ref_wav": f"{d}/tgt.wav"}
        orig_vad = C.vad_trim
        C.vad_trim = fake_vad
        try:
            y0, sr0, p0 = eng.convert(job, 0, 1)
            assert calls["feat"] == [15 * sr, 15 * sr, 10 * sr, 2 * sr], calls    # 40 s -> 3 mieng + nguon 2 s
            assert calls["vad"] == 1, calls                                       # VAD MOT lan / tep dich
            assert p0["n_target_pieces"] == 3 and not p0["matching_set_cached"] and p0["topk"] == calls["topk"] == 4
            assert p0["target_vad"] == ["both"], p0
            assert y0.ndim == 1 and len(y0) > 0 and sr0 == sr and p0["matching_frames"] == 40 * sr // 320
            _, _, p1 = eng.convert(job, 1, 2)                                      # cung tep dich -> bo nho dem
            assert len(calls["feat"]) == 5 and calls["vad"] == 1 and p1["matching_set_cached"] and p1["topk"] == 8
            assert p1["target_vad"] == ["both"], p1
            for i in (2, 3):
                eng.convert({**job, "target_ref_wav": [f"{d}/tgt{i}.wav"]}, 0, 1)
            assert len(eng._ms_cache) <= C.KNN_CACHE and calls["vad"] == 3
        finally:
            C.vad_trim = orig_vad
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert C.Engine.code_rev({"opts": {"family": "knnvc"}}) == C.CODE_REVS["knnvc"] != "1"
    assert C.Engine.code_rev({"opts": {"family": "openvoice"}}) == "1" == B.BaseEngine.code_rev({})


def _speechy(sec: float, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Tin hieu giong tieng noi: am tiet 0,15-0,35 s (chuoi hai, f0 100-220 Hz, muc khac nhau) xen khoang lang
    0,05-0,25 s. VAD torchaudio (trigger 7) cat no nhu tieng VIVOS that (do 12/10); chuoi hai deu dan thi KHONG
    (bo uoc luong on bam theo -> chi doan sau lang dai moi kich)."""
    r = np.random.default_rng(seed)
    out, t = [], 0.0
    while t < sec:
        d, g, f0 = r.uniform(0.15, 0.35), r.uniform(0.05, 0.25), r.uniform(100, 220)
        tt = np.arange(int(d * sr)) / sr
        ph = 2 * np.pi * np.cumsum(f0 * (1 + 0.1 * tt / d)) / sr
        y = sum(np.sin(h * ph) / h for h in range(1, 30))
        out += [r.uniform(0.05, 0.2) * y * np.sin(np.pi * tt / d) ** 0.5 / np.max(np.abs(y)), np.zeros(int(g * sr))]
        t += d + g
    y = np.concatenate(out)[: int(sec * sr)]
    return (y + 1e-4 * r.standard_normal(len(y))).astype(np.float32)


def test_knnvc_matching_set_real_coqui_vad():
    """KNNVC.get_features THAT cua coqui + VAD torchaudio THAT + WavLM gia (CPU): tep dich co doan gan im 15 s va duoi
    nho (kieu AMI) khong duoc lam hong tap so khop; ban cu (VAD tung mieng) hong dung o tep nay."""
    import torch
    try:
        import TTS.vc.models.knnvc as K
        from TTS.vc.configs.knnvc_config import KNNVCConfig
    except ImportError as exc:
        print(f"  (bo qua: khong co coqui/torchaudio: {exc})")
        return
    from v6.tts_multi.engines import coqui_vc as C

    class StubWavLM(torch.nn.Module):                   # thay WavLM-Large 1,26 GB: so khung theo conv 400 / 320
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def extract_features(self, wav, output_layer=None, ret_layer_results=False):
            n = wav.shape[-1]
            if n < C.KNN_MIN_SAMPLES:
                raise RuntimeError(f"WavLM gia: {n} mau < 400")
            return (torch.zeros(1, (n - 400) // 320 + 1, 8),)

    orig = K.get_wavlm
    K.get_wavlm = lambda *a, **kw: StubWavLM()
    try:
        model = K.KNNVC(KNNVCConfig())
    finally:
        K.get_wavlm = orig
    sr = 16000
    rng = np.random.default_rng(1)

    def quiet(sec: float, db: float) -> np.ndarray:
        return (10 ** (db / 20) * rng.standard_normal(int(sec * sr))).astype(np.float32)

    frames = lambda n: (n - 400) // 320 + 1                                        # noqa: E731
    # lang dau 1 s (VAD uoc luong on o 0,35 s dau), mieng 15-30 s gan im -> ban VAD tung mieng hong, duoi nho 6 s.
    y = np.concatenate([quiet(1, -80), _speechy(14), quiet(15, -80), _speechy(10, seed=2), quiet(6, -70)])
    with torch.inference_mode():
        x, how = C.vad_trim(torch.from_numpy(y)[None])
        assert how == "both" and 30 * sr < x.shape[-1] < len(y), (how, x.shape)     # cat duoi nho, giu doan giua
        ms, n, cuts = C.knn_matching_set(model, [y])
        assert ms.shape[0] > 0 and ms.device.type == "cpu" and cuts == ["both"] and n == len(C.split_pieces(
            x[0].numpy(), sr)), (ms.shape, n, cuts)
        whole = model.get_matching_set([torch.from_numpy(y)[None]])               # coqui ca tep (khong vua VRAM that)
        assert abs(ms.shape[0] - whole.shape[0]) <= n, (ms.shape, whole.shape)      # chi lech <= 1 khung / mieng
        pieces = [torch.from_numpy(np.ascontiguousarray(p))[None] for p in C.split_pieces(y, sr)]
        try:
            model.get_matching_set(pieces)                                         # ban 12/10: VAD tung mieng
        except RuntimeError:
            pass
        else:
            raise AssertionError("tep thu phai lam hong ban VAD tung mieng (neu khong, test khong con y nghia)")
        sil = quiet(20, -80)                                                       # ca tep im: VAD rong -> giu nguyen
        ms, n, cuts = C.knn_matching_set(model, [sil])
        assert cuts == ["none"] and ms.shape[0] == sum(frames(len(p)) for p in C.split_pieces(sil, sr)), (cuts, ms.shape)
        try:
            model.get_matching_set([torch.from_numpy(sil)[None]])
        except RuntimeError:
            pass
        else:
            raise AssertionError("coqui ca tep im phai hong (ly do co nhanh 'none')")
        lead = np.concatenate([quiet(3, -80), _speechy(5, seed=3)])                # lang dau 3 s phai bi cat
        ms, n, cuts = C.knn_matching_set(model, [lead, y])
        assert cuts == ["both", "both"] and ms.shape[0] < frames(len(lead)) + whole.shape[0], (cuts, ms.shape)
        try:
            C.knn_matching_set(model, [np.zeros(100, np.float32)])
        except ValueError:
            pass
        else:
            raise AssertionError("tep qua ngan phai bao loi ro")


def _git_bash() -> str | None:
    """bash cua Git for Windows (KHONG lay bash.exe cua WSL trong System32)."""
    for p in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
        if os.path.exists(p):
            return p
    p = shutil.which("bash")
    return p if p and "system32" not in p.lower() else None


def test_setup_envs_clone_selftest():
    import subprocess
    bash = _git_bash()
    if bash is None or shutil.which("git") is None:
        print("  (bo qua: khong co Git bash / git)")
        return
    r = subprocess.run([bash, os.path.join(PKG_DIR, "setup_envs.sh"), "selftest"], capture_output=True, text=True,
                       timeout=120, encoding="utf-8", errors="replace")
    assert r.returncode == 0 and "selftest OK" in r.stdout and r.stdout.count("PASS") == 4, (r.returncode, r.stdout,
                                                                                             r.stderr[-500:])


def _main() -> int:
    fails = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)):
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{'OK' if not fails else 'LOI'}: {fails} test hong")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(_main())
