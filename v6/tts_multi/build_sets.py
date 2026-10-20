# -*- coding: utf-8 -*-
"""WP7b / B5 - DIEU PHOI cac bo ttsmulti_vi · ttsmulti_en · vcmulti: ke hoach -> sinh (I3) -> cham -> ghep -> kenh app
-> kiem id -> K1 -> dac trung XLS-R (hang doi) + bo nghe ABX.

  plan       v6.tts_multi.plan / vc_sets (nhe)
  gen        chay MOT may sinh qua giao dien I3 trong tien trinh rieng (env_python cua engines.json):
             <env_python> -m <module|v6.tts_multi.run_engine> --engine <ten> --jobs <jobs.jsonl> --out <cand/ten>
             chi gui job con thieu ung vien; kiem dau ra (<job>_c<k>.wav + .json {job_id, k, engine, sr, seconds, params}).
             NANG (nap mo hinh TTS/VC): moi may mot viec trong hang doi, qua run_serial.sh; KHONG long run_serial ben trong.
  qa         qa.run_tts_qa / vc_sets.run_vc_qa (NANG: ASR + resemblyzer)
  splice     splice.run_splice / vc_sets.run_vc_splice (hang doi dung --keep-failed: muc truot cong VAN vao manifest,
             meta.qa_pass=False + edits[0].qa.fail; loc sau theo co do — khong bo ngam sau nhieu gio GPU)
  channels   bien the kenh app (v6.phone.channels.MAIN_PHONE) cho tap con moi loai: id '<id>__<cfg>'
             (khu on mac dinh cua app = dieu kien kenh, nhan KHONG doi — DECISIONS.md muc 4)
  ids        v6.common.check_new_ids (tien to rieng v6ttsmvi_ / v6ttsmen_ / v6vcm_)
  k1         vaudit.k1.check tung tep -> meta.k1 (giong build_natural k1; tiep tuc duoc)
  feat-frames / feat-pooled   XLS-R L12 khung PCA64 / gop 6144 (v6.boundary.xlsr_frames / xlsr_pooled, GPU) cho RIENG
             cac bo nay (thay danh sach tep; dung PCA _pca64.npz da co, khong fit lai)
  abx        bo nghe mu that-vs-may: <root>/listen_kit/tts/<set>/ {audio/, phieu.csv, dap_an.csv, HUONG_DAN.txt};
             abx-score cham phieu da dien (ty le doan dung, ty le 'that' theo may / muc / thao tac)
  status     dem muc / job / ung vien / qua cong / manifest
  queue      IN RA + ghi <work>/queue.json (khong chay) danh sach viec nang {name, cmd, device, est_mem_gb, est_minutes,
             depends_on, produces} theo cac jobs/*.jsonl da lap (+ --hybrid-pairs mms_vie+seedvc:40); uoc luong lay tu
             smoke cua engines.json; moi may: gen -> qacal (qa --limit 30: mau CHIA DEU thao tac x muc, KIEM TAY
             summary.json) -> qa (xet cong lai ca muc qacal theo qa.GATES hien tai); gen tu choi may co smoke ok=False
Ma thoat: plan loi (o rong / da co dau ra cu) -> 2; gen: ma cua tien trinh may (1 nap, 2 job hong, 3 OOM) hoac 4 khi
  con thieu ung vien / metadata I3 sai -> hang doi dung lai, chay lai chi gui job con thieu.
CLI: python -m v6.tts_multi.build_sets <buoc> --set ttsmulti_vi|ttsmulti_en|vcmulti [--root R] [--registry F] ...
     vd: ... gen --set ttsmulti_vi --engine vixtts · ... gen --set ttsmulti_vi --engine seedvc --hybrid vixtts+seedvc"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6 import common as C  # noqa: E402
from v6.tts_multi import plan as P  # noqa: E402
from v6.tts_multi import qa as Q  # noqa: E402
from v6.tts_multi import splice as SP  # noqa: E402
from v6.tts_multi import vc_sets as V  # noqa: E402

SR = 16000
RUN_MODULE = "v6.tts_multi.run_engine"
I3_META_KEYS = ("job_id", "k", "engine", "sr", "seconds", "params")
LISTEN = os.path.join(BASE, "listen_kit", "tts")
QA_CAL_N = 30                       # so muc cham thu (buoc qacal_*) sau lan sinh that dau tien cua moi may
QA_SEC_TTS, QA_SEC_VC = 1.5, 4.0    # giay CPU / ung vien (uoc luong): TTS = ASR + giong + DNSMOS + F0; VC = ASR 2 lan
                                    # (doan doi giong + doan nguon) + giong nguoi dich / nguoi nguon + DNSMOS


# ---------------- sinh (I3) ----------------
def _read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def check_outputs(out_dir: str, jobs: list[dict]) -> dict:
    """Moi job co du n_candidates (VC: >= 1) tep <job>_c<k>.wav + .json dung truong I3? Tra {ok, missing, bad_meta}."""
    missing, bad = [], []
    for j in jobs:
        n = int(j.get("n_candidates", 1))
        for k in range(n):
            w = os.path.join(out_dir, f"{j['job_id']}_c{k}.wav")
            if not os.path.exists(w):
                missing.append(f"{j['job_id']}_c{k}")
                continue
            mj = os.path.splitext(w)[0] + ".json"
            try:
                with open(mj, encoding="utf-8") as fh:
                    m = json.load(fh)
                if any(key not in m for key in I3_META_KEYS) or m["job_id"] != j["job_id"]:
                    bad.append(os.path.basename(mj))
            except (OSError, ValueError):
                bad.append(os.path.basename(mj))
    return {"ok": not missing and not bad, "missing": missing, "bad_meta": bad}


def _engine_env(ent: dict) -> dict:
    """Moi truong cho tien trinh may: v6.tts_multi.engine_env (WP7a: bo nho dem D:, torch_cu neu gpu_overlay,
    BASE trong PYTHONPATH); chua co -> BASE + PYTHONPATH cu."""
    try:
        from v6.tts_multi import engine_env
        return engine_env(ent)
    except ImportError:
        env = dict(os.environ)
        env["PYTHONPATH"] = BASE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env


def run_engine(engine: str, jobs_path: str, out_dir: str, registry: str | None = None, log_path: str | None = None,
               timeout: float | None = None, ignore_smoke: bool = False) -> dict:
    """Chay mot may qua giao dien I3 (tien trinh rieng, cwd = BASE, PYTHONPATH co BASE). Chi job con thieu.
    May co smoke.ok=False trong engines.json (WP7a) -> tu choi (chay lai smoke tren GPU ranh truoc) tru ignore_smoke."""
    reg, src = P.load_registry(registry)
    if engine not in reg:
        raise KeyError(f"may {engine!r} khong co trong {src}")
    ent = reg[engine]
    sm = ent.get("smoke")
    if isinstance(sm, dict) and sm.get("ok") is False and not ignore_smoke:
        raise RuntimeError(f"{engine}: smoke ok=False ({sm.get('date')}, {str(sm.get('fatal') or sm.get('load_error'))[:80]})"
                           " -> chay lai smoke (WP7a) hoac --ignore-smoke")
    py = ent.get("env_python") or sys.executable
    if not os.path.exists(py):
        raise FileNotFoundError(f"{engine}: env_python {py} chua cai (v6/tts_multi/setup_envs.sh)")
    jobs = _read_jsonl(jobs_path)
    todo = [j for j in jobs if not check_outputs(out_dir, [j])["ok"]]
    res = {"engine": engine, "jobs": len(jobs), "todo": len(todo), "registry": src}
    if not todo:
        chk = check_outputs(out_dir, jobs)
        return {**res, "returncode": 0, "ok": chk["ok"], "missing": len(chk["missing"]), "bad_meta": chk["bad_meta"][:10]}
    os.makedirs(out_dir, exist_ok=True)
    todo_path = os.path.join(out_dir, f"_todo_{P.slug(engine)}.jsonl")
    _write_jsonl(todo_path, todo)
    env = _engine_env(ent)
    cmd = [py, "-m", ent.get("runner_module", RUN_MODULE), "--engine", engine, "--jobs", todo_path, "--out", out_dir]
    log = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
    try:
        r = subprocess.run(cmd, cwd=BASE, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    finally:
        if log_path:
            log.close()
    chk = check_outputs(out_dir, jobs)
    return {**res, "returncode": r.returncode, "ok": chk["ok"], "missing": len(chk["missing"]),
            "bad_meta": chk["bad_meta"][:10]}


def gen(set_name: str, engine: str, root: str | None = None, registry: str | None = None,
        hybrid: str | None = None, timeout: float | None = None, ignore_smoke: bool = False) -> dict:
    """gen bo tts/vc: jobs/<engine>.jsonl -> cand/<engine>/; hybrid '<tts>+<vc>': jobs/hybrid_<..>.jsonl -> cand/<..>/
    (may chay = engine = may VC)."""
    D = P.set_dirs(set_name, root)
    key = hybrid or engine
    jp = os.path.join(D["jobs"], f"hybrid_{hybrid}.jsonl" if hybrid else f"{engine}.jsonl")
    os.makedirs(D["logs"], exist_ok=True)
    return run_engine(engine, jp, os.path.join(D["cand"], key), registry,
                      os.path.join(D["logs"], f"gen_{P.slug(key)}.log"), timeout, ignore_smoke)


# ---------------- kenh app ----------------
def add_channels(set_name: str, root: str | None = None, per_kind: int = 10, cfgs: list[str] | None = None,
                 seed: int = 2026) -> dict:
    """Tap con moi (loai, cau hinh kenh chinh) qua v6.phone.channels.apply -> dong '<id>__<cfg>' (nhan giu nguyen)."""
    from v6.phone import channels as PC
    D = P.set_dirs(set_name, root)
    man = SP.load_manifest(D["manifest"])
    have = {m["id"] for m in man}
    cfgs = cfgs or list(PC.MAIN_PHONE)
    by_kind = defaultdict(list)
    for m in man:
        if m["cfg"] == "goc":
            by_kind[m["kind"]].append(m)
    new = 0
    for kind, rows in sorted(by_kind.items()):
        for ci, cfg in enumerate(cfgs):
            pick = random.Random(f"{seed}|{kind}|{cfg}").sample(rows, min(per_kind, len(rows)))
            for j, e in enumerate(pick):
                vid = f"{e['id']}__{cfg}"
                if vid in have:
                    continue
                y, _ = sf.read(C.abs_wav(e["wav"]), dtype="float32")
                z, cm = PC.apply(y, cfg, seed + ci * 7919 + j)
                path = os.path.join(os.path.dirname(C.abs_wav(e["wav"])), vid + ".wav")
                sf.write(path, z, SR, subtype="PCM_16")
                man.append({"id": vid, "kind": e["kind"], "cfg": cfg, "src": e["id"], "wav": C.rel_v4(path),
                            "meta": {**e["meta"], "kenh": cfg, "kenh_meta": cm}})
                have.add(vid)
                new += 1
    SP.save_manifest(D["manifest"], man)
    return {"set": set_name, "new_variants": new, "n_manifest": len(man)}


# ---------------- id / K1 / dac trung ----------------
def check_ids(set_name: str, root: str | None = None) -> dict:
    D = P.set_dirs(set_name, root)
    ids = [m["id"] for m in SP.load_manifest(D["manifest"])]
    C.check_new_ids(ids)
    return {"set": set_name, "ok": len(ids), "prefix": C.id_prefix(ids[0]) if ids else None}


def run_k1(set_name: str, root: str | None = None, limit: int | None = None) -> dict:
    """vaudit.k1.check -> meta.k1 (dang build_natural.step_k1), luu moi 50 tep, tiep tuc duoc."""
    from vaudit import k1
    D = P.set_dirs(set_name, root)
    man = SP.load_manifest(D["manifest"])
    n = 0
    for e in man:
        if "k1" in e["meta"]:
            continue
        if limit is not None and n >= limit:
            break
        r = k1.check(C.abs_wav(e["wav"]))
        e["meta"]["k1"] = {"decision": r["decision"], "reasons": r["reasons"],
                           "warnings": [w for w in r["warnings"] if not w.startswith("0,3")],
                           "measures": {k: v for k, v in r["measures"].items() if k != "kenh"},
                           "kenh": r["measures"].get("kenh")}
        n += 1
        if n % 50 == 0:
            SP.save_manifest(D["manifest"], man)
    SP.save_manifest(D["manifest"], man)
    return {"set": set_name, "k1_new": n, "k1_total": sum("k1" in e["meta"] for e in man), "n": len(man)}


def xlsr_list(sets: list[str], root: str | None = None, cfgs: set | None = None) -> list[tuple[str, str]]:
    """[(id, wav tuong doi V4)] cua cac bo nay voi cfg trong tap kenh cua xlsr_frames (goc + 4 kenh chinh)."""
    keep = cfgs or {"goc", "app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi"}
    out = []
    for s in sets:
        for e in SP.load_manifest(P.set_dirs(s, root)["manifest"]):
            if e["cfg"] in keep:
                out.append((e["id"], e["wav"]))
    return out


def run_xlsr(which: str, sets: list[str], root: str | None = None, limit: int = 0) -> None:
    """GPU: goi main() cua xlsr_frames / xlsr_pooled voi danh sach tep cua RIENG cac bo nay (PCA da co)."""
    files = xlsr_list(sets, root)
    if which == "frames":
        from v6.boundary import xlsr_frames as X
        if not os.path.exists(os.path.join(X.OUT, "_pca64.npz")):
            raise FileNotFoundError("chua co _pca64.npz: khong fit PCA moi tren bo giong may (lech phan bo)")
    else:
        from v6.boundary import xlsr_pooled as X
    X.file_list = lambda: list(files)
    sys.argv = [X.__name__] + (["--limit", str(limit)] if limit else [])
    X.main()


# ---------------- bo nghe ABX ----------------
ABX_RMS_DB = -26.0


def abx_match_real(real: np.ndarray, tricks: list) -> tuple[np.ndarray, str | None]:
    """Tep gia bac ky co the da ma hoa lai CA tep ('tay_<codec>_<kbps>' trong thu_thuat) -> ma hoa ben THAT cung codec,
    de nguoi nghe khong phan biet nho dau vet codec (khong lien quan giong may). Tra (real, the codec | None)."""
    for t in tricks:
        if not str(t).startswith("tay_"):
            continue
        for name, kbps, sr in SP.REENCODE:
            if t == f"tay_{name}_{kbps}":
                from v6.phone.channels import codec
                return codec(np.asarray(real, np.float32), name, kbps, sr).astype(np.float32), t
    return np.asarray(real, np.float32), None


def _rms_norm(y: np.ndarray, db: float = ABX_RMS_DB) -> np.ndarray:
    """Dua hai ben cap ABX ve cung muc RMS (bo goi y do to); chan dinh 0,99."""
    y = np.asarray(y, np.float32)
    r = float(np.sqrt(np.mean(np.square(y)))) if len(y) else 0.0
    if r <= 1e-8:
        return y
    z = y * (10 ** (db / 20) / r)
    pk = float(np.max(np.abs(z)))
    return z / pk * 0.99 if pk > 0.99 else z


def abx_kit(set_name: str, root: str | None = None, n_per_engine: int = 30, out_root: str | None = None,
            seed: int = 2026) -> dict:
    """Cap that (ban goc chua sua cua CHINH host) - gia (tep da ghep), thu tu A/B ngau nhien; phieu.csv cho nguoi nghe,
    dap_an.csv tach rieng cho nguoi to chuc. Chi nghe tai may (chinh sach DECISIONS / consent: khong sao chep, khong gui)."""
    D = P.set_dirs(set_name, root)
    items = {f"{it['prefix']}_{it['item_id']}": it for it in P.read_plan(D["plan"])}
    man = [m for m in SP.load_manifest(D["manifest"]) if m["cfg"] == "goc" and m["id"] in items]
    rng = random.Random(seed)
    by_eng = defaultdict(list)
    for m in man:
        by_eng[m["meta"].get("engine", "?")].append(m)
    out = os.path.join(out_root or LISTEN, set_name)
    aud = os.path.join(out, "audio")
    os.makedirs(aud, exist_ok=True)
    form, key = [], []
    for eng, rows in sorted(by_eng.items()):
        for m in rng.sample(rows, min(n_per_engine, len(rows))):
            pid = f"p{len(form) + 1:04d}"
            real, codec_tag = abx_match_real(Q.load16(items[m["id"]]["host"]["wav"]), m["meta"].get("thu_thuat") or [])
            fake, _ = sf.read(C.abs_wav(m["wav"]), dtype="float32")
            real, fake = _rms_norm(real), _rms_norm(fake)
            fake_is = rng.choice("AB")
            pair = {"A": fake if fake_is == "A" else real, "B": real if fake_is == "A" else fake}
            for side, y in pair.items():
                sf.write(os.path.join(aud, f"{pid}_{side}.wav"), y, SR, subtype="PCM_16")
            form.append({"pair_id": pid, "file_A": f"audio/{pid}_A.wav", "file_B": f"audio/{pid}_B.wav",
                         "tep_co_giong_may_A_B": "", "tu_tin_1_5": "", "tu_nhien_A_1_5": "", "tu_nhien_B_1_5": "",
                         "ghi_chu": ""})
            e = m["meta"]["edits"][0]
            key.append({"pair_id": pid, "may_la": fake_is, "id": m["id"], "engine": eng, "role": m["meta"].get("engine_role"),
                        "tier": m["meta"].get("muc_che_giau"), "op": e.get("op"), "level2": e.get("level2"),
                        "ref_stratum": m["meta"].get("ref_stratum"), "start": e.get("start"), "end": e.get("end"),
                        "ma_hoa_ca_hai_ben": codec_tag})
    for name, rows in (("phieu.csv", form), ("dap_an.csv", key)):
        with open(os.path.join(out, name), "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["pair_id"])
            w.writeheader()
            w.writerows(rows)
    with open(os.path.join(out, "HUONG_DAN.txt"), "w", encoding="utf-8") as fh:
        fh.write("Bo nghe mu that / giong may (WP7b). Moi cap A/B: mot tep that (ban goc) va mot tep co giong may hoac doi giong.\n"
                 "Nghe bang tai nghe, theo thu tu bat ky; dien cot tep_co_giong_may_A_B (A hoac B), tu_tin_1_5 (1 = doan bua,"
                 " 5 = chac chan), tu_nhien_A_1_5 / tu_nhien_B_1_5 (5 = rat tu nhien). KHONG mo dap_an.csv.\n"
                 "Chi nghe tren may nay: khong sao chep, khong tai len, khong gui am thanh (chinh sach nhai giong / du lieu).\n"
                 "Cham: python -m v6.tts_multi.build_sets abx-score --set <bo> --form <phieu_da_dien.csv>\n")
    return {"set": set_name, "pairs": len(form), "out": out, "engines": dict(Counter(k["engine"] for k in key))}


def abx_score(form_csv: str, key_csv: str) -> dict:
    """Ty le doan dung tep may va ty le 'bi tuong la that' theo may / muc / thao tac (phieu da dien)."""
    with open(key_csv, encoding="utf-8-sig") as fh:
        key = {r["pair_id"]: r for r in csv.DictReader(fh)}
    acc: dict[str, list] = defaultdict(list)
    nat: dict[str, list] = defaultdict(list)
    with open(form_csv, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            ans = (r.get("tep_co_giong_may_A_B") or "").strip().upper()
            k = key.get(r["pair_id"])
            if not k or ans not in ("A", "B"):
                continue
            ok = ans == k["may_la"]
            for g in ("all", f"engine:{k['engine']}", f"tier:{k['tier']}", f"op:{k['op']}"):
                acc[g].append(ok)
                v = r.get(f"tu_nhien_{k['may_la']}_1_5", "").strip()
                if v.isdigit():
                    nat[g].append(int(v))
    return {g: {"n": len(v), "doan_dung": round(float(np.mean(v)), 3), "tuong_la_that": round(1 - float(np.mean(v)), 3),
                "tu_nhien_tep_may": round(float(np.mean(nat[g])), 2) if nat[g] else None,
                "wilson95": [round(x, 3) for x in C.wilson(int(sum(v)), len(v))]} for g, v in sorted(acc.items())}


# ---------------- hang doi viec nang ----------------
RUN_SERIAL = BASE.replace("\\", "/") + "/.tmp_v6/wf/run_serial.sh"
MAIN_PY = BASE.replace("\\", "/") + "/.venv/Scripts/python.exe"


def _est_audio_s(job: dict) -> float:
    """Uoc luong giay am thanh mot ung vien: TTS ~0,3 s / tu (vi) | 0,35 s / tu (en); VC = do dai nguon."""
    if "text" in job:
        return max(1.0, len(str(job["text"]).split()) * (0.3 if job.get("lang") == "vi" else 0.35))
    try:
        return float(sf.info(job["src_wav"]).duration)
    except (OSError, RuntimeError, KeyError):
        return 10.0


def queue_cmds(set_name: str, root: str | None = None, registry: str | None = None,
               hybrids: list[tuple[str, str, int]] | None = None) -> list[dict]:
    """Danh sach viec NANG (khong chay) cho mot bo da lap ke hoach: gen moi may (rieng tung may) -> qa moi may
    [-> hybrid (tts, vc, n): ke hoach -> gen VC -> qa] -> splice -> channels -> ids -> k1 -> feat-frames / feat-pooled
    (GPU) + abx. Uoc luong tu smoke cua engines.json (rtf, load_sec, peak_commit_mb); thieu -> than trong (rtf 1, 6 GB)."""
    D = P.set_dirs(set_name, root)
    reg, _ = P.load_registry(registry)
    opt = (f' --root "{root}"' if root else "") + (f' --registry "{registry}"' if registry else "")
    base = f'bash "{RUN_SERIAL}" "{MAIN_PY}" -m v6.tts_multi.build_sets'
    out, qa_names = [], []
    for tts, vc, n in hybrids or []:               # tang TTS -> VC (03): job VC chi co sau khi qa may tts xong
        key = f"{tts}+{vc}"
        sm = reg.get(vc, {}).get("smoke") or {}
        hp, gn, qn = (f"hybridplan_{set_name}_{P.slug(key)}", f"gen_{set_name}_{P.slug(key)}",
                      f"qa_{set_name}_{P.slug(key)}")
        out += [{"name": hp, "cmd": f"{base} hybrid --set {set_name} --engine {tts} --vc {vc} --n {n}{opt}",
                 "device": "cpu", "est_mem_gb": 0.5, "est_minutes": 1.0, "depends_on": [f"qa_{set_name}_{P.slug(tts)}"],
                 "produces": f"{D['jobs']}/hybrid_{key}.jsonl + muc '<item>_h{P.slug(vc)}' trong plan.jsonl"},
                {"name": gn, "cmd": f"{base} gen --set {set_name} --engine {vc} --hybrid {key}{opt}", "device": "gpu",
                 "est_mem_gb": round(float(sm.get("peak_commit_mb") or 6000) / 1024 + 0.5, 1),
                 "est_minutes": round((float(sm.get("load_sec") or 120) + n * 1.5 * 8 * float(sm.get("rtf") or 1)) / 60,
                                      1), "depends_on": [hp], "produces": f"{D['cand']}/{key}/"},
                {"name": qn, "cmd": f"{base} qa --set {set_name} --engine {key}{opt}", "device": "cpu",
                 "est_mem_gb": 3.0, "est_minutes": round(max(2.0, n * 1.5 * 0.9 / 60), 1), "depends_on": [gn],
                 "produces": f"{D['qa']}/selection.json (muc lai)"}]
        qa_names.append(qn)
    for jp in sorted(glob.glob(os.path.join(D["jobs"], "*.jsonl"))):
        key = os.path.splitext(os.path.basename(jp))[0]
        hybrid = key[len("hybrid_"):] if key.startswith("hybrid_") else None
        if hybrid and hybrid in {f"{t}+{v}" for t, v, _ in hybrids or []}:
            continue                               # da liet ke o tren (ke hoach lai -> gen -> qa)
        eng = hybrid.split("+")[-1] if hybrid else key
        ent = reg.get(eng, {})
        jobs = _read_jsonl(jp)
        n_c = sum(int(j.get("n_candidates", 1)) for j in jobs)
        audio = sum(_est_audio_s(j) * int(j.get("n_candidates", 1)) for j in jobs)
        sm = ent.get("smoke") or {}
        rtf = float(sm.get("rtf") or 1.0)
        knn = ent.get("family") == "knnvc" or float(ent.get("ref_seconds") or 0) > 60   # mau giong ~300 s moi job:
        mins = (float(sm.get("load_sec") or 120) + audio * rtf * 1.3 + (len(jobs) * 5.0 if knn else 0.0)) / 60
        name = f"gen_{set_name}_{P.slug(key)}"
        cmd = f"{base} gen --set {set_name} --engine {eng}" + (f" --hybrid {hybrid}" if hybrid else "") + opt
        dev = "gpu" if (ent.get("gpu_overlay") or sm.get("device") == "cuda") else "cpu"
        out.append({"name": name, "cmd": cmd, "device": dev,
                    "est_mem_gb": round(float(sm.get("peak_commit_mb") or 6000) / 1024 + 0.5, 1),
                    "est_minutes": round(mins, 1), "depends_on": [f"qa_{set_name}_{P.slug(hybrid.split('+')[0])}"]
                    if hybrid else [], "produces": f"{D['cand']}/{key}/<job>_c<k>.wav (+.json); {n_c} ung vien"})
        qn = f"qa_{set_name}_{P.slug(key)}"
        cal = f"qacal_{set_name}_{P.slug(key)}"
        qa_names.append(qn)
        q_eng = f" --engine {hybrid or eng}"
        qa_s = QA_SEC_VC if (ent.get("kind") == "vc" and not hybrid) else QA_SEC_TTS
        per_item = 1.3 * n_c / max(1, len(jobs))           # ~ung vien moi muc (may_ca_file nhieu phan)
        out.append({"name": cal, "cmd": f"{base} qa --set {set_name}{q_eng} --limit {QA_CAL_N}{opt}", "device": "cpu",
                    "est_mem_gb": 3.0, "est_minutes": round(max(2.0, QA_CAL_N * per_item * qa_s / 60 + 1.5), 1),
                    "depends_on": [name],
                    "produces": f"{D['qa']}/summary.json (mau {QA_CAL_N} muc chia deu thao tac x muc): KIEM TAY ty le "
                                f"qua + fail_gate + spk_z/cer p10-50-90 truoc khi chay {qn} (cong chua tung chay tren dau "
                                "ra that; sua qa.GATES neu lech — buoc qa sau xet cong lai ca cac muc nay)"})
        out.append({"name": qn, "cmd": f"{base} qa --set {set_name}{q_eng}{opt}", "device": "cpu", "est_mem_gb": 3.0,
                    "est_minutes": round(max(2.0, n_c * qa_s / 60 + 1.5), 1), "depends_on": [cal],
                    "produces": f"{D['qa']}/selection.json, {D['qa']}/<engine>.jsonl (moi diem moi ung vien)"})
    n_items = len(P.read_plan(D["plan"])) if os.path.exists(D["plan"]) else 0
    tail = [("splice", "splice --keep-failed", "cpu", 2.0, max(3.0, n_items * 1.5 / 60), qa_names,
             f"{D['set']}/<kind>/<id>.wav + manifest.json"),
            ("channels", "channels --per-kind 10", "cpu", 1.5, 15.0, ["splice"], "bien the kenh app '<id>__<cfg>'"),
            ("ids", "ids", "cpu", 1.0, 2.0, ["channels"], "kiem tien to id (v6.common.check_new_ids)"),
            ("k1", "k1", "cpu", 2.0, max(5.0, n_items * 1.3 * 0.8 / 60), ["ids"], "meta.k1 trong manifest.json"),
            ("feat_frames", "feat-frames", "gpu", 5.0, max(5.0, n_items * 1.3 * 0.25 / 60), ["k1"],
             "data_labels/realistic_v4/xlsr_frames/<id>.npy (PCA64 co san)"),
            ("feat_pooled", "feat-pooled", "gpu", 5.0, max(5.0, n_items * 1.3 * 0.25 / 60), ["feat_frames"],
             "data_labels/realistic_v4/xlsr_pooled/<id>.npy"),
            ("abx", "abx --n 30", "cpu", 1.0, 3.0, ["splice"], f"listen_kit/tts/{set_name}/")]
    for nm, step, dev, mem, mins, deps, prod in tail:
        out.append({"name": f"{nm}_{set_name}", "cmd": f"{base} {step} --set {set_name}{opt}", "device": dev,
                    "est_mem_gb": mem, "est_minutes": round(mins, 1),
                    "depends_on": [d if d.startswith(("qa_", "gen_")) else f"{d}_{set_name}" for d in deps],
                    "produces": prod})
    return out


# ---------------- trang thai ----------------
def status(set_name: str, root: str | None = None) -> dict:
    D = P.set_dirs(set_name, root)
    items = P.read_plan(D["plan"]) if os.path.exists(D["plan"]) else []
    selp = os.path.join(D["qa"], "selection.json")
    sel = json.load(open(selp, encoding="utf-8")) if os.path.exists(selp) else {}
    man = SP.load_manifest(D["manifest"])
    per = defaultdict(lambda: {"items": 0, "jobs": 0, "cand": 0, "qa": 0, "pass": 0})
    for it in items:
        p = per[it["engine"]]
        p["items"] += 1
        for part in it["parts"]:
            for jid in part["jobs"]:
                p["jobs"] += 1
                p["cand"] += len(Q.candidates(D["cand"], it["engine"], jid))
        s = sel.get(it["item_id"])
        p["qa"] += int(bool(s))
        p["pass"] += int(bool(s and s["pass"]))
    return {"set": set_name, "engines": dict(per), "manifest": len(man),
            "manifest_cfg": dict(Counter(m["cfg"] for m in man)), "k1": sum("k1" in m["meta"] for m in man)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Dieu phoi bo giong may nhieu may (WP7b)")
    ap.add_argument("step", choices=["plan", "gen", "qa", "splice", "hybrid", "channels", "ids", "k1", "feat-frames",
                                     "feat-pooled", "abx", "abx-score", "status", "queue"])
    ap.add_argument("--set", required=True, choices=sorted(P.SETS))
    ap.add_argument("--root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--engine", default=None)
    ap.add_argument("--hybrid", default=None, help="gen: khoa '<tts>+<vc>' (engine = may VC)")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--n-cand", type=int, default=3)
    ap.add_argument("--refs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--reencode-p", type=float, default=0.5)
    ap.add_argument("--per-kind", type=int, default=10)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--vc", default="seedvc")
    ap.add_argument("--form", default=None)
    ap.add_argument("--timeout", type=float, default=None)
    ap.add_argument("--ignore-smoke", action="store_true", help="gen: van chay may co smoke ok=False")
    ap.add_argument("--hybrid-pairs", default="", help="queue: '<tts>+<vc>:<n>,...' (vd mms_vie+seedvc:40)")
    ap.add_argument("--no-write", action="store_true", help="queue: chi in, khong ghi <work>/queue.json")
    P.add_plan_args(ap)
    a = ap.parse_args()
    s, vc = a.set, a.set == "vcmulti"
    code = 0
    if a.step == "plan":
        try:
            out = V.make_vc_plan(a.per_cell, root=a.root, registry=a.registry, **P.plan_kwargs(a)) if vc else \
                P.make_plan(s, a.per_cell, a.n_cand, a.refs, root=a.root, registry=a.registry, **P.plan_kwargs(a))
        except P.PlanError as ex:
            out, code = {"loi": str(ex), "summary": ex.summary}, 2
    elif a.step == "gen":
        out = gen(s, a.engine, a.root, a.registry, a.hybrid, a.timeout, a.ignore_smoke)
        code = gen_exit_code(out)
    elif a.step == "qa":
        out = V.run_vc_qa(a.root, engine=a.engine, limit=a.limit) if vc else Q.run_tts_qa(s, a.root, a.engine, a.limit)
    elif a.step == "splice":
        out = V.run_vc_splice(a.root, a.keep_failed, a.reencode_p) if vc else \
            SP.run_splice(s, a.root, a.keep_failed, a.reencode_p, a.limit)
    elif a.step == "hybrid":
        out = V.plan_hybrid(s, a.vc, a.n, a.root, a.registry, keep_failed=a.keep_failed,
                            tts_engines=a.engine.split(",") if a.engine else None)
    elif a.step == "channels":
        out = add_channels(s, a.root, a.per_kind)
    elif a.step == "ids":
        out = check_ids(s, a.root)
    elif a.step == "k1":
        out = run_k1(s, a.root, a.limit)
    elif a.step in ("feat-frames", "feat-pooled"):
        run_xlsr(a.step.split("-")[1], [s], a.root, a.limit or 0)
        out = {"done": a.step}
    elif a.step == "abx":
        out = abx_kit(s, a.root, a.n)
    elif a.step == "abx-score":
        out = abx_score(a.form, os.path.join(LISTEN, s, "dap_an.csv"))
    elif a.step == "queue":
        pairs = []
        for tok in filter(None, a.hybrid_pairs.split(",")):
            k, _, n = tok.partition(":")
            t, _, v = k.partition("+")
            pairs.append((t, v, int(n or 40)))
        out = queue_cmds(s, a.root, a.registry, pairs)
        if not a.no_write:
            Q._atomic_json(os.path.join(P.set_dirs(s, a.root)["work"], "queue.json"), out)
    else:
        out = status(s, a.root)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    if code:
        sys.exit(code)


def gen_exit_code(res: dict) -> int:
    """Ma thoat cua buoc gen cho hang doi: ma cua tien trinh may (run_engine: 1 loi nap, 2 job hong, 3 OOM / CUDA) neu
    khac 0; tien trinh 0 nhung con thieu ung vien / metadata sai -> 4; du -> 0 (chay lai chi gui job con thieu)."""
    rc = int(res.get("returncode") or 0)
    if rc:
        return rc
    return 4 if (res.get("ok") is False or int(res.get("missing") or 0) > 0 or res.get("bad_meta")) else 0


if __name__ == "__main__":
    main()
