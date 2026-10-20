# -*- coding: utf-8 -*-
"""Kiem thu WP7b (plan / qa / splice / vc_sets / build_sets) + MAY GIA theo giao dien I3 (am don + on, khong mo hinh).

  may gia     python -m v6.tts_multi.test_pipeline fake-engine --engine <ten> --jobs <jobs.jsonl> --out <dir>
              TTS: moi tu = mot cum am hai (so am tiet x 0,11 s), cach 0,12 s, 22,05 kHz, cao do doi theo seed + k
              VC : doc src_wav, doi am sac (dieu bien + loc), giu do dai, ghi 24 kHz. Ghi <job>_c<k>.wav + .json (I3).
  unit        du lieu tong hop trong thu muc tam (khong doc du lieu that, khong nap mo hinh): cau chu moi, CER, quota /
              fold / mau giong / cam nguon, chon ung vien, noi equal-power, cat doan dich, chuoi day du tren may gia
  e2e         chuoi DAY DU tren host THAT (natural_v6) voi may gia qua TIEN TRINH RIENG (I3): plan -> gen -> qa (resemblyzer
              + DNSMOS + pyworld that, ASR 'oracle' vi am don khong doc duoc) -> splice (3 muc, tai ma hoa ffmpeg) -> kenh app
              -> ABX -> VC (02) -> TTS->VC (03); + thu ASR that (vi + en) tren 1 host moi ngon ngu
CLI: python -m v6.tts_multi.test_pipeline [unit|e2e|all]      (mac dinh all; chay qua .tmp_v6/wf/run_serial.sh)
     WP7B_IDS=1 -> them kiem v6.common.check_new_ids tren id that cua bo e2e"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import zlib

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6 import common as C  # noqa: E402
from v6.tts_multi import build_sets as B  # noqa: E402
from v6.tts_multi import plan as P  # noqa: E402
from v6.tts_multi import qa as Q  # noqa: E402
from v6.tts_multi import splice as SP  # noqa: E402
from v6.tts_multi import vc_sets as V  # noqa: E402

SR = 16000
TEST_ROOT = os.path.join(BASE, ".tmp_v6", "tts_multi_test")
I3_JOB_TTS = {"job_id", "text", "lang", "ref_wav", "ref_text", "seed", "n_candidates"}
I3_JOB_VC = {"job_id", "src_wav", "target_ref_wav", "seed"}


# ---------------- may gia (I3) ----------------
def _tone_words(words: list[str], seed: int, sr: int, lang: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = [np.zeros(int(0.2 * sr))]
    for w in words:
        n = max(1, P.n_syll([w], lang))
        d = 0.11 * n + 0.04
        t = np.arange(int(d * sr)) / sr
        f0 = 110 + (zlib.crc32(w.encode("utf-8")) % 80) + 3 * (seed % 7)
        x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 5)) * np.hanning(len(t)) * 0.25
        out += [x, np.zeros(int(0.12 * sr))]
    out.append(np.zeros(int(0.1 * sr)))
    y = np.concatenate(out)
    return (y + rng.normal(0, 3e-4, len(y))).astype(np.float32)


def fake_engine(engine: str, jobs_path: str, out_dir: str) -> int:
    """May gia dung giao dien I3 (TTS: truong text; VC: truong src_wav)."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    drop = os.environ.get("WP7B_FAKE_DROP") == "1"            # kiem thu ma thoat: bo ung vien cuoi moi job
    for j in B._read_jsonl(jobs_path):
        nc = int(j.get("n_candidates", 1))
        for k in range(nc):
            if drop and nc > 1 and k == nc - 1:
                continue
            if "text" in j:
                sr = 22050
                y = _tone_words(P.norm_text(j["text"], j.get("lang", "vi")).split(), int(j["seed"]) + k, sr,
                                j.get("lang", "vi"))
                params = {"fake": True, "seed": int(j["seed"]) + k}
            else:
                sr = 24000
                x = Q.load16(j["src_wav"])
                from scipy.signal import butter, resample_poly, sosfilt
                x = resample_poly(x, 3, 2)
                t = np.arange(len(x)) / sr
                y = (sosfilt(butter(2, [300, 3400], "bandpass", fs=sr, output="sos"), x) * (1 + 0.2 * np.sin(2 * np.pi * 6 * t)))
                y = (y * 0.9 / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)
                params = {"fake": True, "seed": int(j["seed"]) + k, "vc": True}
            base = os.path.join(out_dir, f"{j['job_id']}_c{k}")
            sf.write(base + ".wav", y, sr)
            with open(base + ".json", "w", encoding="utf-8") as fh:
                json.dump({"job_id": j["job_id"], "k": k, "engine": engine, "sr": sr, "seconds": round(len(y) / sr, 3),
                           "params": params}, fh)
            n += 1
    return n


def fake_registry(path: str) -> str:
    ent = {"env_python": sys.executable, "runner_module": "v6.tts_multi.test_pipeline", "license": "test", "notes": "may gia"}
    reg = {"fake_tts": {**ent, "kind": "tts", "langs": ["vi", "en"], "role": "train"},
           "fake_ho": {**ent, "kind": "tts", "langs": ["vi", "en"], "role": "heldout"},
           "fake_vc": {**ent, "kind": "vc", "langs": ["vi", "en"], "role": "train"}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(reg, fh)
    return path


# ---------------- du lieu tong hop ----------------
def synth_hosts(d: str, lang: str = "en", n_person: int = 10, n_file: int = 3) -> list[dict]:
    """Host tong hop: 14 'tu' am hai tren nen on -45 dB, 16 kHz; AMI gia (cum = cuoc hop: 2 tep M1, 1 tep M2)."""
    vocab = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()
    os.makedirs(d, exist_ok=True)
    rng = random.Random(7)
    out = []
    for p in range(n_person):
        for f in range(n_file):
            words, t, ys = [], 0.4, [np.zeros(int(0.4 * SR))]
            for i in range(14):
                w = vocab[(p + f + i) % len(vocab)]
                y = _tone_words([w], p * 10 + f, SR, lang)[int(0.2 * SR):-int(0.22 * SR)]
                words.append([w, round(t, 3), round(t + len(y) / SR, 3)])
                gap = rng.choice([0.08, 0.1, 0.3])
                ys += [y, np.zeros(int(gap * SR))]
                t += len(y) / SR + gap
            ys.append(np.zeros(int(0.5 * SR)))
            y = np.concatenate(ys)
            y = (y + np.random.default_rng(p * 31 + f).normal(0, 0.005, len(y))).astype(np.float32)
            name = f"ami_syn_{p:02d}_{f}"
            path = os.path.join(d, name + ".wav")
            sf.write(path, y, SR)
            out.append({"file": name, "source": "ami", "cluster": f"ami:SY{p:02d}{'a' if f < 2 else 'b'}",
                        "person": f"ami:P{p:02d}", "dur": round(len(y) / SR, 3), "text": " ".join(w[0] for w in words),
                        "words": words, "wav": path, "lang": lang, "extra": False, "ref": None, "info": {}})
    return out


def _fake_scorers() -> Q.Scorers:
    def emb(y):
        f = np.abs(np.fft.rfft(np.asarray(y[:SR * 2], float), 512))
        return f[:64] / (np.linalg.norm(f[:64]) + 1e-9)
    return Q.Scorers(asr=Q.oracle_asr, embed=emb, dnsmos=lambda y: (3.4, 4.0, 3.1),
                     f0=lambda y: np.full(max(1, len(y) // 160), 120.0), asr_mode="oracle")


# ---------------- kiem tra dong manifest ----------------
def validate_entry(m: dict, items: dict) -> None:
    w = C.abs_wav(m["wav"])
    info = sf.info(w)
    assert info.samplerate == SR and info.channels == 1, (m["id"], info.samplerate)
    meta = m["meta"]
    assert C.id_prefix(m["id"]) in ("v6ttsmvi", "v6ttsmen", "v6vcm"), m["id"]
    assert meta["source"] in P.HOST_SOURCES["vi"] + P.HOST_SOURCES["en"] + ("vivos", "librispeech"), meta["source"]
    assert isinstance(meta["fold"], int) and 0 <= meta["person_fold"] < 5
    assert meta["person_fold"] == C.assign_fold([meta["person"]])
    plain = {k: v for k, v in meta.items() if k not in ("lockbox", "lockbox_reason")}
    want = C.lockbox_reason({"id": m["id"], "meta": plain}, "")
    assert bool(meta.get("lockbox")) == bool(want) and meta.get("lockbox_reason") == want, (m["id"], want)
    e = meta["edits"][0]
    assert e["level2"] in meta["level2"] and meta["level1"], meta
    assert 0 <= e["start"] < e["end"] <= info.duration + 0.06, (m["id"], e["start"], e["end"], info.duration)
    if e["level2"] == "03_chen_giong_may":
        assert e["op"] in ("may_thay", "may_chen", "may_ca_file") and e["tier"] in P.TIERS
        assert e["kind"] == ("global" if e["op"] == "may_ca_file" else "segment")
        assert "qa" in e and e["ref_stratum"] in P.STRATA
        assert meta["ref_cung_phien"] == (e["ref_stratum"] in P.SAME_STRATA)
        if e["kind"] == "segment":
            assert e["end"] - e["start"] <= P.MAX_SEG_S + 0.2, (m["id"], e["end"] - e["start"])
        it = items.get(m["id"])
        if it:
            assert all(r["src_file"] != it["host"]["file"] for r in it["refs"]), "mau giong lay tu chinh tep host"
    else:
        assert e["level2"] in ("02_giong_toan_cuc", "02_giong_cuc_bo") and e["op"] == "doi_giong_ai"
        assert e["target_person"] != meta["person"] and meta["speakers"] == [e["target_person"]], meta.get("speakers")


# ---------------- unit ----------------
def test_norm_and_syll() -> None:
    assert P.norm_text("Hà Nội, <unk>đã 12!", "vi") == "hà nội đã"
    assert P.norm_text("It's OK-ish [noise]", "en") == "it's ok ish"
    assert P.n_syll(["mười lăm", "người"], "vi") == 3 and P.n_syll(["hello", "a"], "en") == 3


def test_new_phrase() -> None:
    rng = random.Random(1)
    for lang in ("vi", "en"):
        for n in (1, 2, 3, 6, 10):
            for _ in range(20):
                w = P.new_phrase(lang, n, ["hai"] if n == 3 else [], rng)
                assert w and abs(len(w) - n) <= max(2, n // 2), (lang, n, w)
        assert P.new_phrase(lang, 1, ["không"], rng) != ["không"]


def test_cer() -> None:
    assert Q.cer("abc", "abc") == 0.0 and abs(Q.cer("abcd", "abed") - 0.25) < 1e-9
    assert Q.cer("hà nội", "ha noi") > 0                         # dau thanh duoc tinh
    assert Q.cer("", "") == 0.0 and Q.cer("", "x") == 1.0


def test_plan_quota_refs_folds() -> None:
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"))
        reg = fake_registry(os.path.join(d, "reg.json"))
        s = P.make_plan("ttsmulti_en", per_cell=1, n_cand=2, root=d, registry=reg, hosts=hosts, extra=[], pairs={})
        assert s["n_items"] == 18 and s["max_per_person"] <= s["person_cap"], s
        assert set(s["ref_stratum"]) <= set(P.STRATA) and s["roles"]["heldout"] == 9, s
        D = P.set_dirs("ttsmulti_en", d)
        items = P.read_plan(D["plan"])
        for it in items:
            assert all(r["src_file"] != it["host"]["file"] and os.path.exists(r["wav"]) for r in it["refs"])
            ok = P.SAME_STRATA if it["want_same_session"] else P.MAIN_STRATA
            assert it["ref_stratum"] in ok, (it["want_same_session"], it["ref_stratum"])
            assert all(r["ltas_dist_db"] is not None for r in it["refs"]), it["refs"]
            assert all(2.7 <= r["dur"] <= 10.3 for r in it["refs"]), it["refs"]
            assert it["person_fold"] == C.assign_fold([it["host"]["person"]])
            if it["op"] != "may_ca_file":
                e = it["edit"]
                assert e["est_s"] <= P.MAX_SEG_S and len(e["carrier"].split()) >= P.CARRIER_MIN - 1, e
                tgt = e["carrier"].split()[e["carrier_lo"]:e["carrier_hi"] + 1]
                assert " ".join(tgt) == e["inserted"], (tgt, e["inserted"])
        for eng in ("fake_tts", "fake_ho"):
            for j in B._read_jsonl(os.path.join(D["jobs"], f"{eng}.jsonl")):
                assert set(j) == I3_JOB_TTS, j
        bad = [dict(hosts[0], source="vss", person="vss:x")]
        try:
            P.make_plan("ttsmulti_en", 1, root=d, registry=reg, hosts=bad + hosts, extra=[], pairs={})
            raise AssertionError("nguon vss phai bi tu choi")
        except ValueError as ex:
            assert not isinstance(ex, P.PlanError), ex
        try:                                             # bo het nguon -> moi o rong -> PlanError, khong ghi gi
            P.make_plan("ttsmulti_en", 1, root=d, registry=reg, hosts=hosts, extra=[], pairs={}, exclude_sources=("ami",))
            raise AssertionError("o rong phai bao PlanError")
        except P.PlanError as ex:
            assert ex.summary["n_items"] == 0 and len(ex.summary["empty_cells"]) == 18, ex.summary
        assert len(P.read_plan(D["plan"])) == 18, "ke hoach cu phai giu nguyen khi PlanError"
        s2 = P.make_plan("ttsmulti_en", 1, root=d, registry=reg, hosts=hosts, extra=[], pairs={},
                         exclude_sources=("ami",), allow_empty_cells=True)
        assert s2["n_items"] == 0 and s2["exclude_sources"] == ["ami"] and s2["archived_to"], s2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_ref_strata() -> None:
    """Tang mau giong: vox/celeb/voxp uoc luong theo hang parquet; EdAcc / AMI theo cum; VIVOS mot phien."""
    def h(file, src, cl, p):
        return {"file": file, "source": src, "cluster": cl, "person": p}
    a = h("vox_test-00000-of-00038_00010", "vox", "vox:id1", "vox:id1")
    assert P.row_index(a["file"]) == (0, 10) and P.row_index("voxp_test-00003_00012") == (3, 12)
    assert P.ref_stratum(a, h("vox_test-00000-of-00038_00013", "vox", "vox:id1", "vox:id1")) == "khac_tep_ke"
    assert P.ref_stratum(a, h("vox_test-00000-of-00038_00030", "vox", "vox:id1", "vox:id1")) == "khac_tep_ke"
    assert P.ref_stratum(a, h("vox_test-00000-of-00038_00060", "vox", "vox:id1", "vox:id1")) == "khac_tep_xa"
    # khac manh, khong co hang toan cuc -> KHONG biet -> coi la ke (bao thu); co 'grow' -> theo khoang cach that
    assert P.ref_stratum(a, h("vox_test-00001-of-00038_00011", "vox", "vox:id1", "vox:id1")) == "khac_tep_ke"
    b, c = dict(a, grow=700), h("vox_test-00001-of-00038_00002", "vox", "vox:id1", "vox:id1")
    assert P.ref_stratum(b, dict(c, grow=705)) == "khac_tep_ke" and P.ref_stratum(b, dict(c, grow=760)) == "khac_tep_xa"
    # voxp: ma bai phat bieu (audio_id) quyet dinh, bat ke hang parquet
    v1 = dict(h("voxp_test-00000_00001", "voxp", "voxp:7", "voxp:7"), session="voxp:20170703-0900-PLENARY-26-en_x")
    v2 = dict(h("voxp_test-00000_00900", "voxp", "voxp:7", "voxp:7"), session="voxp:20170703-0900-PLENARY-26-en_x")
    v3 = dict(h("voxp_test-00000_00002", "voxp", "voxp:7", "voxp:7"), session="voxp:20180101-0900-PLENARY-3-en_y")
    assert P.ref_stratum(v1, v2) == "cung_phien" and P.ref_stratum(v1, v3) == "khac_phien"
    m = P.parquet_meta("voxp")                         # parquet VoxPopuli that (chi metadata + audio_id)
    if m is not None:
        hv = P.annotate_rows({"file": "voxp_test-00000_00000", "source": "voxp"})
        assert hv["grow"] == 0 and hv["session"].startswith("voxp:20170703-0900-PLENARY-26-en_"), hv
    e1, e2 = h("edacc_test-00000_00002", "edacc", "edacc:C08", "edacc:C08-A"), h("edacc_test-00000_00090", "edacc",
                                                                                 "edacc:C08", "edacc:C08-A")
    assert P.ref_stratum(e1, e2) == "cung_phien"
    assert P.ref_stratum(h("ami_x_1", "ami", "ami:ES2004a", "ami:M1"), h("ami_x_2", "ami", "ami:ES2004c", "ami:M1")) \
        == "khac_phien"
    assert P.ref_stratum(h("V01_1", "vivos", "vivos:V01", "vivos:V01"), h("V01_2", "vivos", "vivos:V01", "vivos:V01")) \
        == "cung_phien"
    y = np.random.default_rng(0).normal(0, 0.1, SR)
    assert P.ltas_dist_db(y, y) == 0.0 and P.ltas_dist_db(y, SP.lowpass(y, 2000)) > 3.0
    assert P.ltas_dist_db(y[:100], y) is None


def test_main_stratum_excludes_same_session() -> None:
    """Nguoi chi co mot phien (kieu EdAcc) KHONG duoc vao tang chinh, chi vao tang so sanh; tang so sanh <= P_SAME."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"), n_person=10, n_file=3)
        single = {x["person"] for x in hosts if int(x["person"][-2:]) < 5}
        for x in hosts:                                  # 5 nguoi mot cum (= mot cuoc hoi thoai)
            if x["person"] in single:
                x["cluster"] = "ami:ONE" + x["person"][-2:]
        reg = fake_registry(os.path.join(d, "reg.json"))
        s = P.make_plan("ttsmulti_en", per_cell=10, n_cand=1, root=d, registry=reg, hosts=hosts, extra=[], pairs={},
                        engines=["fake_tts"], seed=11, person_cap_frac=1.0, ops=("may_thay",), tiers=("vung",))
        items = P.read_plan(P.set_dirs("ttsmulti_en", d)["plan"])
        same = [it for it in items if it["want_same_session"]]
        assert same and all(it["ref_stratum"] in P.SAME_STRATA for it in same), [it["ref_stratum"] for it in same]
        assert all(it["want_same_session"] for it in items if it["host"]["person"] in single), "mot phien -> so sanh"
        assert all(it["ref_stratum"] in P.MAIN_STRATA for it in items if not it["want_same_session"])
        assert s["comparison_share"] <= P.P_SAME + 1e-9 and s["ref_main_share"] >= 1 - P.P_SAME - 1e-9, s
        per = {}
        for it in items:
            per.setdefault((it["tier"], it["op"]), []).append(it["want_same_session"])
        assert all(sum(v) <= math.floor(P.P_SAME * len(v) + 1e-9) for v in per.values()), per
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_round_robin_scarce_supply() -> None:
    """Nhieu may + nguon KHAN HIEM (40 host < 108 muc muc tieu): moi o may x muc x thao tac deu co muc, chenh lech
    <= 1, tang chinh >= 0,7, nhieu fold; nguon < so o -> PlanError va khong ghi plan (12/10: may dau khong lay het)."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"), n_person=10, n_file=4)
        regp = fake_registry(os.path.join(d, "reg.json"))
        reg = json.load(open(regp, encoding="utf-8"))
        reg["fake_x"] = dict(reg["fake_tts"])
        json.dump(reg, open(regp, "w", encoding="utf-8"))
        engs = ["fake_ho", "fake_tts", "fake_x"]
        s = P.make_plan("ttsmulti_en", per_cell=4, n_cand=1, root=d, registry=regp, hosts=hosts, extra=[], pairs={},
                        engines=engs, person_cap_frac=1.0)
        cells = [f"{e}|{t}|{o}" for e in engs for t in P.TIERS for o in P.OPS]
        assert s["n_items"] == 40 and s["underfilled"] and not s["empty_cells"], s
        assert all(s["cells"].get(c, 0) >= 1 for c in cells) and s["cell_max"] - s["cell_min"] <= 1, s["cells"]
        assert s["ref_main_share"] >= 0.7 and len(s["folds"]) >= 3, s
        assert len({it["host"]["file"] for it in P.read_plan(P.set_dirs("ttsmulti_en", d)["plan"])}) == 40
        d2 = os.path.join(d, "r2")
        try:
            P.make_plan("ttsmulti_en", per_cell=4, n_cand=1, root=d2, registry=regp, hosts=hosts[:9], extra=[],
                        pairs={}, engines=engs, person_cap_frac=1.0)
            raise AssertionError("9 host < 27 o phai bao PlanError")
        except P.PlanError as ex:
            assert ex.summary["empty_cells"] and ex.summary["supply"]["main_persons"] >= 1, ex.summary
        assert not os.path.exists(P.set_dirs("ttsmulti_en", d2)["plan"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_archive_and_force() -> None:
    """Lap lai ke hoach: ban cu (plan, jobs) doi sang _work/_old; da co cand/ -> tu choi tru force (doi ca cand)."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"))
        reg = fake_registry(os.path.join(d, "reg.json"))
        kw = dict(per_cell=1, n_cand=1, root=d, registry=reg, hosts=hosts, extra=[], pairs={}, engines=["fake_tts"])
        assert P.make_plan("ttsmulti_en", **kw)["archived_to"] is None
        D = P.set_dirs("ttsmulti_en", d)
        B._write_jsonl(os.path.join(D["jobs"], "hybrid_fake_tts+fake_vc.jsonl"), [{"job_id": "old"}])
        s = P.make_plan("ttsmulti_en", **kw)
        old = s["archived_to"]
        assert old and os.path.exists(os.path.join(old, "_work", "plan.jsonl"))
        assert os.path.exists(os.path.join(old, "_work", "jobs", "hybrid_fake_tts+fake_vc.jsonl"))
        assert sorted(os.listdir(D["jobs"])) == ["fake_tts.jsonl"], os.listdir(D["jobs"])
        os.makedirs(os.path.join(D["cand"], "fake_tts"))
        open(os.path.join(D["cand"], "fake_tts", "x_c0.wav"), "w").close()
        try:
            P.make_plan("ttsmulti_en", **kw)
            raise AssertionError("da co cand/ phai tu choi")
        except P.PlanError as ex:
            assert "force" in str(ex)
        s = P.make_plan("ttsmulti_en", force=True, **kw)
        assert os.path.exists(os.path.join(s["archived_to"], "_work", "cand", "fake_tts", "x_c0.wav"))
        assert not os.path.exists(D["cand"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_speaker_fallback_and_long_ref() -> None:
    """QA: nguoi chi con mau giong -> so voi mau giong (spk_basis='mau_giong'); mau giong dai VC ghi du danh sach tep."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"), n_person=1, n_file=2)
        refs = Q.PersonRefs({hosts[0]["person"]: hosts}, _fake_scorers(), "en")
        e = _fake_scorers().embed(Q.load16(hosts[1]["wav"]))
        both = {hosts[0]["file"], hosts[1]["file"]}
        assert refs.speaker(hosts[0]["person"], both, e)["spk_cos"] is None
        r = refs.speaker(hosts[0]["person"], both, e, {hosts[0]["file"]})
        assert r["spk_basis"] == "mau_giong" and r["n_genuine"] == 1 and r["spk_cos"] > 0.99, r
        assert refs.speaker(hosts[0]["person"], {hosts[0]["file"]}, e)["spk_basis"] == "ban_ghi_khac"
        for _ in range(2):                               # lan 2: tep noi da co -> van ghi du danh sach tep
            t = V.target_ref(hosts[0]["person"], hosts, 300.0, os.path.join(d, "refs"), random.Random(0))
            assert set(t["src_file"].split(",")) == both and t["text_src"] == "none", t
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_resolve_excludes() -> None:
    assert P.DEFAULT_EXCLUDE == ("celeb",) and P.resolve_excludes(None) == ("celeb",)
    assert P.resolve_excludes(None, "celeb") == () and P.resolve_excludes("ami,celeb", "celeb") == ("ami",)
    assert P.resolve_excludes("") == ()


def _qa_row(**kw) -> dict:
    r = {"n_words": 5, "cer": 0.0, "target_ok": True, "spk_z": -5.0, "spk_cos": 0.3, "spk_basis": "ban_ghi_khac",
         "dnsmos_sig": 3.5, "clip_frac": 0.0, "max_sil": 0.2, "pros_zmax": 1.0, "dur_ratio": 1.0,
         "dnsmos_ovr": 3.0, "host_ovr": 3.0, "pros_l2": 1.0}
    r.update(kw)
    return r


def test_gates_fixed_voice_and_hybrid_base() -> None:
    """May giong co san (clone=False): khong co cong giong nguoi; tang lai chon goc khong can g2_spk (khong --keep-failed)."""
    r = _qa_row()
    g1, g0 = Q.gates(r, "vi", True), Q.gates(r, "vi", False)
    assert g1["g2_spk"] is False and not all(g1.values())
    assert "g2_spk" not in g0 and all(g0.values()), g0
    assert Q.gates(_qa_row(spk_basis="mau_giong", spk_z=None, spk_cos=0.75), "vi")["g2_spk"] is False
    assert Q.gates(_qa_row(spk_basis="mau_giong", spk_z=None, spk_cos=0.85), "vi")["g2_spk"] is True
    rows = [dict(_qa_row(), wav="a", spk_cos=0.9), dict(_qa_row(), wav="b", spk_cos=0.1)]
    c0, c1 = Q.composite(rows, clone=False), Q.composite(rows, clone=True)
    assert c0[0] == c0[1] and c1[0] < c1[1], (c0, c1)
    item = {"pass": False, "parts": {"0": {"gates": {"g1_cer": True, "g2_spk": False}}}}
    assert not V.base_ok(item) and V.base_ok(item, needs_vc=True)
    assert not V.base_ok({"pass": False, "parts": {"0": {"gates": {"g1_cer": False, "g2_spk": True}}}}, True)
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:                                                 # chuoi STRICT that: may co dinh -> qa strict -> plan_hybrid
        hosts = synth_hosts(os.path.join(d, "h"))
        regp = fake_registry(os.path.join(d, "reg.json"))
        reg = json.load(open(regp, encoding="utf-8"))
        reg["fake_fix"] = dict(reg["fake_tts"], needs_vc=True)
        json.dump(reg, open(regp, "w", encoding="utf-8"))
        P.make_plan("ttsmulti_en", per_cell=3, n_cand=2, root=d, registry=regp, hosts=hosts, extra=[], pairs={},
                    engines=["fake_fix"])
        D = P.set_dirs("ttsmulti_en", d)
        items = P.read_plan(D["plan"])
        assert items and all(it["clone"] is False for it in items)
        fake_engine("fake_fix", os.path.join(D["jobs"], "fake_fix.jsonl"), os.path.join(D["cand"], "fake_fix"))
        by = {}
        for h in hosts:
            by.setdefault(h["person"], []).append(h)
        Q.run_tts_qa("ttsmulti_en", d, scorers=_fake_scorers(), gate_mode="strict", by_person=by)
        sel = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
        parts = [p for v in sel.values() for p in v["parts"].values() if p]
        assert parts and all("g2_spk" not in p["gates"] for p in parts), parts[0]["gates"]
        n_ok = sum(V.base_ok(v, True) for v in sel.values())
        assert n_ok >= 1, [p["fail"] for p in parts]
        h = V.plan_hybrid("ttsmulti_en", "fake_vc", n=2, root=d, registry=regp)       # strict: khong keep_failed
        assert h["new_items"] == min(2, n_ok), (h, n_ok)
        hy = [it for it in P.read_plan(D["plan"]) if it.get("gen_chain")]
        assert all(it["clone"] is True and it["host_reuse_of"] == it["hybrid_of"] for it in hy), hy[:1]
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qa_summary_groups() -> None:
    p_ok = {"spk_basis": "ban_ghi_khac", "fail": [], "spk_z": 0.1, "cer": 0.0}
    p_mg = {"spk_basis": "mau_giong", "fail": ["g2_spk"], "spk_z": None, "cer": 0.1}
    sel = {"a": {"engine": "e", "pass": True, "parts": {"0": p_ok}},
           "b": {"engine": "e", "pass": False, "parts": {"0": p_mg}},
           "c": {"engine": "e", "pass": False, "parts": {"0": None}}}
    s = Q.qa_summary(sel)["engines"]["e"]
    assert s["items"] == 3 and s["mau_giong"] == 1 and s["no_cand"] == 1 and s["fail_gate"] == {"g2_spk": 1}, s
    assert s["pass_rate_ban_ghi_khac"] == 0.5 and s["spk_z_p10_50_90"] == [0.1, 0.1, 0.1], s


def test_abx_helpers() -> None:
    y = (np.random.default_rng(1).normal(0, 0.05, SR)).astype(np.float32)
    a, b = B._rms_norm(y), B._rms_norm(y * 4)
    assert abs(np.sqrt(np.mean(a ** 2)) - np.sqrt(np.mean(b ** 2))) < 1e-4
    same, tag = B.abx_match_real(y, ["tron_20ms"])
    assert tag is None and np.array_equal(same, y)
    z, tag = B.abx_match_real(y, ["khop_ltas", "tay_mp3_96"])
    assert tag == "tay_mp3_96" and abs(len(z) - len(y)) < SR // 10, (tag, len(z))


def test_smoke_guard() -> None:
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        regp = fake_registry(os.path.join(d, "reg.json"))
        reg = json.load(open(regp, encoding="utf-8"))
        reg["fake_tts"]["smoke"] = {"ok": False, "date": "x", "fatal": "CUDA OOM"}
        json.dump(reg, open(regp, "w", encoding="utf-8"))
        jp = os.path.join(d, "j.jsonl")
        B._write_jsonl(jp, [{"job_id": "j0", "text": "a b", "lang": "en", "ref_wav": "", "ref_text": "", "seed": 1,
                             "n_candidates": 1}])
        try:
            B.run_engine("fake_tts", jp, os.path.join(d, "out"), regp)
            raise AssertionError("may smoke ok=False phai bi tu choi")
        except RuntimeError as ex:
            assert "smoke" in str(ex)
        assert "fake_tts" not in P.engines_for(reg, "tts", "en")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_gen_exit_code() -> None:
    """gen khong bao 'xong' khi may hong / thieu ung vien: ma thoat != 0 (CLI) de hang doi dung lai."""
    assert B.gen_exit_code({"returncode": 3, "ok": False, "missing": 5}) == 3
    assert B.gen_exit_code({"returncode": 0, "ok": False, "missing": 2}) == 4
    assert B.gen_exit_code({"returncode": 0, "ok": True, "missing": 0, "bad_meta": []}) == 0
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"))
        reg = fake_registry(os.path.join(d, "reg.json"))
        P.make_plan("ttsmulti_en", per_cell=1, n_cand=2, root=d, registry=reg, hosts=hosts, extra=[], pairs={},
                    engines=["fake_tts"])
        import subprocess
        env = dict(os.environ, WP7B_FAKE_DROP="1", PYTHONIOENCODING="utf-8")
        cmd = [sys.executable, "-m", "v6.tts_multi.build_sets", "gen", "--set", "ttsmulti_en", "--engine", "fake_tts",
               "--root", d, "--registry", reg]
        r = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 4 and '"missing"' in r.stdout, (r.returncode, r.stdout[-400:], r.stderr[-400:])
        env.pop("WP7B_FAKE_DROP")
        r = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, (r.returncode, r.stdout[-400:], r.stderr[-400:])          # chay lai: bu job thieu
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_vc_targets_partition() -> None:
    """Nguoi dich VC: cung phan LOCKBOX va cung person_fold voi host khi co (toi uu theo target_rank); nguoi dich vao
    meta.speakers -> lockbox cua dong manifest = host hoac nguoi dich."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"), n_person=16, n_file=2)
        reg = fake_registry(os.path.join(d, "reg.json"))
        s = V.make_vc_plan(per_cell=2, langs=("en",), root=d, registry=reg, engines=["fake_vc"],
                           hosts_by_lang={"en": hosts}, extra_by_lang={"en": []}, person_cap_frac=1.0)
        assert s["n_items"] == 12 and not s["empty_cells"], s
        pool = {}
        for h in hosts:
            pool.setdefault(h["person"], []).append(h)
        items = P.read_plan(P.set_dirs("vcmulti", d)["plan"])
        for it in items:
            h, tp = it["host"], it["target"]["person"]
            best = min(V.target_rank(h, p, pool, {}, "en")[:2] for p in pool if p != h["person"])
            got = V.target_rank(h, tp, pool, {}, "en")[:2]
            assert got == best, (it["item_id"], got, best)
            assert it["target"]["same_lockbox"] == (got[0] == 0) and it["target"]["same_fold"] == (got[1] == 0)
        assert s["same_lockbox"] == len(items), s                     # 16 nguoi tong hop: luon co nguoi cung phan
        it = items[0]
        sel = {"pass": True, "parts": {"0": {"job_id": "j", "k": 0}}}
        y = Q.load16(it["host"]["wav"])
        e = V.vc_entry(it, y, {"level2": "02_giong_toan_cuc", "op": "doi_giong_ai", "kind": "global", "start": 0.0,
                               "end": len(y) / SR}, [], sel, os.path.join(d, "out"))
        m = e["meta"]
        assert m["speakers"] == [it["target"]["person"]] and m["target_person_fold"] == it["target"]["person_fold"]
        assert bool(m.get("lockbox")) == bool(C.lockbox_reason({"id": e["id"], "meta": dict(m, lockbox=None)}))
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_peak_outside_edit_identical() -> None:
    """Doan doi giong / doan chen qua to: CHI doan bi ha muc / nen; ngoai vung sua trung tung mau voi host (12/10)."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        rng = np.random.default_rng(3)
        y = (np.sin(2 * np.pi * 150 * np.arange(4 * SR) / SR) * 0.9 + rng.normal(0, 0.01, 4 * SR)).astype(np.float32)
        hp, zp = os.path.join(d, "host.wav"), os.path.join(d, "z.wav")
        sf.write(hp, y, SR, subtype="PCM_16")
        z = rng.normal(0, 0.02, int(1.5 * SR))
        z[SR // 2:SR // 2 + 300] = 0.5                     # muc tieng noi thap + mot xung -> khop muc (~x7) vuot 1
        sf.write(zp, z.astype(np.float32), SR, subtype="FLOAT")
        it = {"host": {"wav": hp, "file": "host", "cluster": "ami:SY99a", "person": "ami:P99", "source": "ami",
                       "text": "x"}, "op": "vc_cuc_bo", "tier": "trung_binh", "seg": {"a": 1.5, "b": 2.5, "words": "x"},
              "seg_pad": 0.25, "engine": "fake_vc", "role": "train", "seed": 1, "prefix": "v6vcm", "item_id": "vbtb_t_00000",
              "lang": "en", "fold": 0, "person_fold": 0,
              "target": {"person": "ami:P98", "source": "ami", "ref": {"ref_id": "r"}, "same_gender": False}}
        sel = {"pass": True, "parts": {"0": {"wav": zp, "job_id": "j", "k": 0}}}
        out, edit, tricks = V.build_vc(it, sel, random.Random(0), 0.0)
        assert edit.get("ha_dinh_doan_db", 0) < 0, edit                 # doan bi ha muc, khong phai ca tep
        e = V.vc_entry(it, out, edit, tricks, sel, os.path.join(d, "out"))
        w, _ = sf.read(C.abs_wav(e["wav"]), dtype="int16")
        h0, _ = sf.read(hp, dtype="int16")
        f = edit["blend_ms"] / 1000
        a, b = int((edit["start"] - f) * SR) - 2, int((edit["end"] + f) * SR) + 2
        assert np.array_equal(w[:a], h0[:a]) and np.array_equal(w[b:], h0[b:]), "ngoai doan doi giong phai giu nguyen"
        assert "chuan_hoa_ca_tep_db" not in e["meta"]["edits"][0]
        loud = y.copy()
        loud[SR:SR + 800] = 1.4
        g, info = SP.finalize_peak(loud, (SR - 10, SR + 810), False)
        assert np.array_equal(g[:SR - 10], loud[:SR - 10]) and np.array_equal(g[SR + 810:], loud[SR + 810:])
        assert float(np.max(np.abs(g))) <= SP.FILE_PEAK and info["nen_dinh_trong_doan"] == 800, info
        g2, info2 = SP.finalize_peak(loud, None, True)
        assert info2["chuan_hoa_ca_tep_db"] < 0 and float(np.max(np.abs(g2))) <= SP.FILE_PEAK + 1e-6
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_qa_stratified_and_regate() -> None:
    """qacal --limit lay mau chia deu thao tac x muc; lan qa sau XET CONG LAI tu diem luu (doi GATES co hieu luc),
    nhat ky qa/<engine>.jsonl khong trung dong sau nhieu lan cham."""
    its = [{"op": o, "tier": t, "i": i} for o in P.OPS for t in P.TIERS for i in range(5)]
    pick = Q.stratified(its, 9, lambda x: (x["op"], x["tier"]))
    assert len({(x["op"], x["tier"]) for x in pick}) == 9 and len(Q.stratified(its, 100, lambda x: x["op"])) == 45
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    old = dict(Q.GATES)
    try:
        hosts = synth_hosts(os.path.join(d, "h"))
        reg = fake_registry(os.path.join(d, "reg.json"))
        P.make_plan("ttsmulti_en", per_cell=1, n_cand=2, root=d, registry=reg, hosts=hosts, extra=[], pairs={},
                    engines=["fake_tts"])
        D = P.set_dirs("ttsmulti_en", d)
        fake_engine("fake_tts", os.path.join(D["jobs"], "fake_tts.jsonl"), os.path.join(D["cand"], "fake_tts"))
        by = {}
        for h in hosts:
            by.setdefault(h["person"], []).append(h)
        s1 = Q.run_tts_qa("ttsmulti_en", d, limit=3, scorers=_fake_scorers(), gate_mode="strict", by_person=by)
        sel = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
        assert s1["n_items"] == 3 and len({k.split("_")[0] for k in sel}) == 3, sorted(sel)
        Q.run_tts_qa("ttsmulti_en", d, scorers=_fake_scorers(), gate_mode="strict", by_person=by)
        n_rows = sum(1 for _ in open(os.path.join(D["qa"], "fake_tts.jsonl"), encoding="utf-8"))
        Q.GATES["cer_max"] = {"vi": -1.0, "en": -1.0}         # cong moi: khong ai qua
        Q.GATES["cer_max_long"] = {"vi": -1.0, "en": -1.0}
        s3 = Q.run_tts_qa("ttsmulti_en", d, scorers=_fake_scorers(), gate_mode="strict", by_person=by)
        sel = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
        assert all(not v["pass"] for v in sel.values()) and s3["regated"] >= 1, s3
        assert all("g1_cer" in (p["fail"] or []) for v in sel.values() for p in v["parts"].values()), sel
        n2 = sum(1 for _ in open(os.path.join(D["qa"], "fake_tts.jsonl"), encoding="utf-8"))
        assert n_rows == n2 == 9 * 2, (n_rows, n2)                 # 9 muc x 2 ung vien, khong trung
    finally:
        Q.GATES.clear()
        Q.GATES.update(old)
        shutil.rmtree(d, ignore_errors=True)


def test_phrase_plausible_and_pause_insert() -> None:
    """Loi moi: ngay <= 31, gio sang 5-11 (khong 'ngay mot tram'); may_chen uu tien khoang lang."""
    rng = random.Random(4)
    bad_vi = re.compile(r"ngày (một trăm|hai trăm|năm trăm|một nghìn|bốn mươi lăm)|khoảng (một trăm|hai trăm|năm trăm"
                        r"|một nghìn) giờ")
    for _ in range(400):
        w = " ".join(P.new_phrase("vi", rng.choice([3, 5, 6, 9]), ["hai"], rng))
        assert not bad_vi.search(w), w
        e = " ".join(P.new_phrase("en", rng.choice([5, 6, 9]), ["two"], rng))
        assert "around a hundred" not in e and "around a thousand" not in e, e
    ws, t = [], 0.5
    for i in range(20):
        ws.append([f"w{i}", round(t, 3), round(t + 0.25, 3)])
        t += 0.25 + (0.4 if i == 9 else 0.05)                  # mot khoang lang duy nhat sau tu 9
    h = {"words": ws, "dur": t + 0.6}
    hits = [P.pick_edit(h, "may_chen", "en", {"syll_dur": 0.2}, random.Random(k)) for k in range(40)]
    hits = [x for x in hits if x]
    at = [x for x in hits if x["at_pause"] and x["k"] == 10]
    assert hits and len(at) >= 0.9 * len(hits), [(x["k"], x["at_pause"]) for x in hits]


def test_composite_select() -> None:
    def row(c, spk, ok):
        return {"wav": f"w{c}", "cer": c, "spk_cos": spk, "dnsmos_ovr": 3.0, "host_ovr": 3.0, "pros_l2": 1.0,
                "dur_ratio": 1.0, "gates": {"g": ok}}
    rows = [row(0.3, 0.9, False), row(0.05, 0.8, True), row(0.02, 0.7, True)]
    for r, c in zip(rows, Q.composite(rows)):
        r["composite"] = c
    best = Q.select(rows)
    assert best["pass"] and best["wav"] in ("w0.05", "w0.02") and best["n_pass"] == 2
    none = [dict(r, gates={"g": False}) for r in rows]
    b2 = Q.select(none)
    assert not b2["pass"] and b2["fail"] == ["g"]
    assert Q.select(none, "record")["gate_mode"] == "record"


def test_eq_join_and_cut() -> None:
    a, b = np.ones(1600), -np.ones(1600)
    out, j = SP.eq_join([a, b], 10.0)
    assert len(out) == 3200 - 160 and abs(j[0] - (1600 - 80)) <= 1
    y = _tone_words(["one", "two", "three", "four"], 3, SR, "en")
    _, words = Q.oracle_asr(y, "en", "one two three four")
    seg, info = SP.cut_target(y, words, "one two three four", 1, 2, "en")
    assert info["cut_how"] == "asr_khop" and 0.2 < len(seg) / SR < 1.2, (len(seg) / SR, info)
    bed, src = SP.room_tone(y, words, SR, random.Random(0))
    assert len(bed) == SR and np.mean(np.abs(bed) > 0) > 0.95, src


def test_chain_synthetic() -> None:
    """Chuoi plan -> may gia (trong tien trinh) -> qa (bo cham gia) -> splice tren du lieu tong hop."""
    d = tempfile.mkdtemp(prefix="wp7b_", dir=os.path.join(BASE, ".tmp_v6"))
    try:
        hosts = synth_hosts(os.path.join(d, "h"))
        reg = fake_registry(os.path.join(d, "reg.json"))
        P.make_plan("ttsmulti_en", per_cell=1, n_cand=2, root=d, registry=reg, hosts=hosts, extra=[], pairs={},
                    engines=["fake_tts"])
        D = P.set_dirs("ttsmulti_en", d)
        fake_engine("fake_tts", os.path.join(D["jobs"], "fake_tts.jsonl"), os.path.join(D["cand"], "fake_tts"))
        by = {}
        for h in hosts:
            by.setdefault(h["person"], []).append(h)
        s = Q.run_tts_qa("ttsmulti_en", d, scorers=_fake_scorers(), gate_mode="record", by_person=by)
        assert s["n_items"] == 9, s
        r = SP.run_splice("ttsmulti_en", d, keep_failed=True, reencode_p=0.0)
        assert r["new"] == 9, r
        items = {f"{it['prefix']}_{it['item_id']}": it for it in P.read_plan(D["plan"])}
        man = SP.load_manifest(D["manifest"])
        for m in man:
            validate_entry(m, items)
        assert {m["meta"]["muc_che_giau"] for m in man} == set(P.TIERS)
        assert SP.run_splice("ttsmulti_en", d, keep_failed=True)["skipped"]["da_co"] == 9     # tiep tuc duoc
        qj = B.queue_cmds("ttsmulti_en", d, reg, [("fake_tts", "fake_vc", 3)])
        names = {j["name"]: j for j in qj}
        need = {"gen_ttsmulti_en_faketts", "qa_ttsmulti_en_faketts", "hybridplan_ttsmulti_en_faketts+fakevc",
                "gen_ttsmulti_en_faketts+fakevc", "qa_ttsmulti_en_faketts+fakevc", "splice_ttsmulti_en", "k1_ttsmulti_en",
                "feat_frames_ttsmulti_en", "abx_ttsmulti_en"}
        assert need <= set(names), sorted(names)
        assert names["hybridplan_ttsmulti_en_faketts+fakevc"]["depends_on"] == ["qa_ttsmulti_en_faketts"]
        # diem kiem tay sau lan sinh that dau tien: gen -> qacal (--limit) -> qa; splice giu muc truot cong (co qa_pass)
        assert names["qacal_ttsmulti_en_faketts"]["depends_on"] == ["gen_ttsmulti_en_faketts"]
        assert f"--limit {B.QA_CAL_N}" in names["qacal_ttsmulti_en_faketts"]["cmd"]
        assert names["qa_ttsmulti_en_faketts"]["depends_on"] == ["qacal_ttsmulti_en_faketts"]
        assert "--keep-failed" in names["splice_ttsmulti_en"]["cmd"]
        assert all("qa_pass" in m["meta"] for m in man)
        assert set(names["splice_ttsmulti_en"]["depends_on"]) == {"qa_ttsmulti_en_faketts",
                                                                  "qa_ttsmulti_en_faketts+fakevc"}
        assert all(j["cmd"].startswith("bash ") and "run_serial.sh" in j["cmd"] for j in qj)
        assert all(set(j) == {"name", "cmd", "device", "est_mem_gb", "est_minutes", "depends_on", "produces"} for j in qj)
        deps = {d for j in qj for d in j["depends_on"]}
        assert deps <= set(names), deps - set(names)
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------- e2e tren host that ----------------
def _small_real_hosts(lang: str, n_person: int = 5) -> list[dict]:
    """Host that nho: bo nguon mac dinh bi loai (celeb); moi nguoi 5 tep TRAI DEU theo hang parquet (de co ca mau giong
    tang chinh 'khac_tep_xa' lan tang so sanh)."""
    hs = [h for h in P.load_hosts(lang) if h["source"] not in P.DEFAULT_EXCLUDE]
    by: dict[str, list[dict]] = {}
    for h in hs:
        by.setdefault(h["person"], []).append(h)
    good = [p for p, v in sorted(by.items()) if len(v) >= 5 and sum(1 for h in v if 6 <= h["dur"] <= 18) >= 2
            and max(h.get("grow", 0) for h in v) - min(h.get("grow", 0) for h in v) >= 4 * P.FAR_GAP]
    rng = random.Random(5)
    pick = rng.sample(good, min(n_person, len(good)))
    out = []
    for p in pick:
        v = sorted(by[p], key=lambda h: h.get("grow", 0))
        out += [v[round(i * (len(v) - 1) / 4)] for i in range(5)]
    return out


def test_e2e_fake_engine() -> None:
    root = os.path.join(TEST_ROOT, time.strftime("%m%d_%H%M%S"))
    os.makedirs(root, exist_ok=True)
    reg = fake_registry(os.path.join(root, "engines_fake.json"))
    hosts = _small_real_hosts("vi")
    by = {}
    for h in hosts:
        by.setdefault(h["person"], []).append(h)
    s = P.make_plan("ttsmulti_vi", per_cell=1, n_cand=2, root=root, registry=reg, hosts=hosts, extra=[], pairs={},
                    engines=["fake_tts"])
    print("  plan", s["n_items"], "muc,", s["n_jobs"], "job", s["cells"], flush=True)
    assert s["n_items"] >= 6 and s["n_jobs"] >= 6, s
    g = B.gen("ttsmulti_vi", "fake_tts", root, reg)
    print("  gen", g, flush=True)
    assert g["returncode"] == 0 and g["ok"], g
    S = Q.Scorers(asr=Q.oracle_asr, asr_mode="oracle")            # resemblyzer + DNSMOS + pyworld THAT
    q = Q.run_tts_qa("ttsmulti_vi", root, scorers=S, gate_mode="record", by_person=by)
    print("  qa", q, flush=True)
    r = SP.run_splice("ttsmulti_vi", root, keep_failed=True, reencode_p=1.0)
    print("  splice", r, flush=True)
    assert r["new"] == s["n_items"], r
    D = P.set_dirs("ttsmulti_vi", root)
    items = {f"{it['prefix']}_{it['item_id']}": it for it in P.read_plan(D["plan"])}
    man = SP.load_manifest(D["manifest"])
    for m in man:
        validate_entry(m, items)
    sel = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
    one = next(iter(sel.values()))["parts"]["0"]
    assert one["spk_cos"] is not None and one["dnsmos_ovr"] > 0 and one["prosody"]["f0_med_st"] is not None, one
    c = B.add_channels("ttsmulti_vi", root, per_kind=1, cfgs=["app_whatsapp_telegram"])
    print("  channels", c, flush=True)
    assert c["new_variants"] >= 3
    k = B.abx_kit("ttsmulti_vi", root, n_per_engine=4, out_root=os.path.join(root, "listen_kit", "tts"))
    print("  abx", k, flush=True)
    assert k["pairs"] == 4 and os.path.exists(os.path.join(k["out"], "phieu.csv"))
    _e2e_vc(root, reg, by, hosts)
    _e2e_hybrid(root, reg, by)
    st = B.status("ttsmulti_vi", root)
    print("  status", json.dumps(st, ensure_ascii=False), flush=True)
    if os.environ.get("WP7B_IDS") == "1":
        print("  ids", B.check_ids("ttsmulti_vi", root), B.check_ids("vcmulti", root), flush=True)
    print(f"  e2e root: {root}", flush=True)


def _e2e_vc(root: str, reg: str, by: dict, hosts: list[dict]) -> None:
    s = V.make_vc_plan(per_cell=1, langs=("vi",), root=root, registry=reg, engines=["fake_vc"],
                       hosts_by_lang={"vi": hosts}, extra_by_lang={"vi": []})
    print("  vc plan", s, flush=True)
    D = P.set_dirs("vcmulti", root)
    for j in B._read_jsonl(os.path.join(D["jobs"], "fake_vc.jsonl")):
        assert set(j) == I3_JOB_VC, j
    g = B.gen("vcmulti", "fake_vc", root, reg)
    assert g["returncode"] == 0 and g["ok"], g
    S = Q.Scorers(asr=Q.oracle_asr, asr_mode="oracle")
    V.run_vc_qa(root, S, gate_mode="record", by_person=by, src_text_oracle=True)
    r = V.run_vc_splice(root, keep_failed=True, reencode_p=0.0)
    print("  vc splice", r, flush=True)
    assert r["new"] == s["n_items"] == len(V.VC_OPS) * len(V.VC_TIERS), r
    pool = {}
    for h in hosts:
        pool.setdefault(h["person"], []).append(h)
    for it in P.read_plan(D["plan"]):                  # nguoi dich toi uu theo target_rank (lockbox, fold, kho X-10)
        best = min(V.target_rank(it["host"], p, pool, {}, "vi")[:3] for p in pool if p != it["host"]["person"])
        assert V.target_rank(it["host"], it["target"]["person"], pool, {}, "vi")[:3] == best, (it["item_id"], best)
    for m in SP.load_manifest(D["manifest"]):
        validate_entry(m, {})
        if m["meta"]["edits"][0]["kind"] == "segment":
            e = m["meta"]["edits"][0]
            y, _ = sf.read(C.abs_wav(m["wav"]), dtype="float32")
            h = Q.load16(next(h for h in hosts if h["file"] == m["meta"]["host_file"])["wav"])
            f = float(e.get("blend_ms") or 0) / 1000
            a, b = int((e["start"] - f) * SR) - 2, int((e["end"] + f) * SR) + 2
            assert len(y) == len(h) and np.array_equal(y[:a], h[:a]) and np.array_equal(y[b:], h[b:]), \
                (m["id"], float(np.max(np.abs(y[:a] - h[:a]))), "ngoai doan doi giong phai giu nguyen")


def _e2e_hybrid(root: str, reg: str, by: dict) -> None:
    h = V.plan_hybrid("ttsmulti_vi", "fake_vc", n=1, root=root, registry=reg, keep_failed=True)
    print("  hybrid plan", h, flush=True)
    assert h["new_items"] == 1
    D = P.set_dirs("ttsmulti_vi", root)
    key = h["jobs_files"][0][len("hybrid_"):-len(".jsonl")]
    g = B.gen("ttsmulti_vi", "fake_vc", root, reg, hybrid=key)
    assert g["returncode"] == 0 and g["ok"], g
    Q.run_tts_qa("ttsmulti_vi", root, scorers=Q.Scorers(asr=Q.oracle_asr, asr_mode="oracle"), gate_mode="record",
                 by_person=by)
    r = SP.run_splice("ttsmulti_vi", root, keep_failed=True, reencode_p=0.0)
    assert r["new"] == 1, r
    m = [x for x in SP.load_manifest(D["manifest"]) if "+" in x["meta"]["engine"]]
    assert m and m[0]["meta"]["edits"][0]["gen_chain"] == ["fake_tts", "fake_vc"], m[:1]
    assert m[0]["meta"]["level2"] == ["03_chen_giong_may"]


def test_real_asr_smoke() -> None:
    """ASR that (vaudit.asr) qua bo chuyen cua qa: 1 host vi + 1 host en, CER so voi loi hosts.jsonl (cung ASR) ~ 0."""
    S = Q.Scorers()
    for lang in ("vi", "en"):
        h = next(x for x in P.load_hosts(lang) if 4 <= x["dur"] <= 8)
        text, words = S.asr(Q.load16(h["wav"]), lang)
        c = Q.cer(P.norm_text(h["text"], lang), P.norm_text(text, lang))
        print(f"  asr {lang}: {h['file']} cer={c:.3f} n_words={len(words)}", flush=True)
        assert words and c < 0.2, (lang, c, text[:80])


UNIT = [test_norm_and_syll, test_new_phrase, test_cer, test_ref_strata, test_plan_quota_refs_folds,
        test_main_stratum_excludes_same_session, test_round_robin_scarce_supply, test_archive_and_force,
        test_speaker_fallback_and_long_ref, test_composite_select, test_eq_join_and_cut, test_chain_synthetic,
        test_resolve_excludes, test_gates_fixed_voice_and_hybrid_base, test_qa_summary_groups, test_abx_helpers,
        test_smoke_guard, test_gen_exit_code, test_vc_targets_partition, test_peak_outside_edit_identical,
        test_qa_stratified_and_regate, test_phrase_plausible_and_pause_insert]
E2E = [test_e2e_fake_engine, test_real_asr_smoke]


def run(tests: list) -> int:
    bad = 0
    for t in tests:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as ex:                                # noqa: BLE001 — bao cao moi bai kiem
            import traceback
            bad += 1
            print(f"FAIL {t.__name__}: {type(ex).__name__} {ex}", flush=True)
            traceback.print_exc()
    print(f"{len(tests) - bad}/{len(tests)} dat", flush=True)
    return bad


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--engine" in sys.argv:                                  # goi theo I3 (build_sets.run_engine) hoac 'fake-engine'
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd", nargs="?")
        ap.add_argument("--engine", required=True)
        ap.add_argument("--jobs", required=True)
        ap.add_argument("--out", required=True)
        a = ap.parse_args()
        print(f"may gia {a.engine}: {fake_engine(a.engine, a.jobs, a.out)} tep")
        return
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = UNIT if mode == "unit" else E2E if mode == "e2e" else UNIT + E2E
    sys.exit(1 if run(tests) else 0)


if __name__ == "__main__":
    main()
