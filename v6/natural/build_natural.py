# -*- coding: utf-8 -*-
"""M8 - BO TEST LON TREN LOI NOI TU NHIEN TIENG VIET + KENH DIEN THOAI (cham he thong hien tai v5 qua v6.engine).

Nguon ban goc (khong qua phong ao: ban ghi that): VietSuperSpeech 4 kenh sach (hoi thoai, vlog, podcast) · VoxVietnam test
(noi tu nhien, 1 406 nguoi) · Vietnam-Celeb test (phong van, on that). Moc tu = ASR wav2vec2 tieng Viet (vaudit.asr).
Cum thong ke (bootstrap): VietSuperSpeech = video (ban ghi), hai bo con lai = nguoi noi.

Thao tac (nhan dung dinh dang v4 de v5 cham duoc):
  nguyen_ban                                     -> 01
  nen_toan_cuc / nen_cuc_bo   ho khu on N1..N5 (v6.tools.backends) + N7 WebRTC NS (thu vien that)      -> 02
  giong_toan_cuc / giong_cuc_bo  ho doi giong V1..V6                                                    -> 02
  ghep_thay   thay 1-12 tu bang doan GIONG THAT cua cung nguoi (cung ban ghi 60 % / ban ghi khac 40 %),
              noi dung khac loi goc (dung Scope moi)                                                    -> 03
  ghep_chen   chen 1-8 tu that cua cung nguoi vao khoang giua hai tu                                    -> 03
  xoa_loi     cat 1-12 tu roi noi; nhan = vung 0,5 s quanh moi noi (Output: vung <= 1 s chua moi noi)  -> 03
  Moi noi: crossfade 0 / 5 / 20 ms. Doan sua <= 5 s, <= 50 % tep, cach hai dau tep >= 0,5 s.
Kenh (v6.phone.channels): moi tep goc giu cfg 'goc'; tap con phan tang theo loai dua qua 4 kenh dien thoai chinh
  (Zalo tin nhan GIA DINH · WhatsApp/Telegram · goi app ghi lai · VoLTE ghi cuoc goi) va tap nho qua du 27 cau hinh.

Buoc (moi buoc mot tien trinh, ghi tiep duoc):
  hosts     cat + ASR -> data/natural_v6/hosts/<nguon>/*.wav + hosts.jsonl
  make      sinh thao tac + kenh -> data_labels/v6/natvi/{<loai>/*.wav, manifest.json} (id 'v6nat_')
  k1        vaudit.k1.check tung tep -> meta.k1
  feat      v6.common.extract_features -> realistic_v4/features_natvi.pkl
Sau do:  python -m v6.engine --ext v6natvi --tag natvi_v5   va   python -m v6.natural.report_natural natvi_v5

TIENG ANH (NAT_LANG=en, bo v6naten, id 'v6naten_'): EdAcc (hoi thoai cuoc goi video, ~40 giong Anh) · AMI ihm (hop that)
· VoxPopuli en (phat bieu). Moc tu = wav2vec2-base-960h, chi giu ban goc ASR khop transcript goc (CER <= 0,35).
Khong co phan VSASV. Cac buoc nhu tren:  NAT_LANG=en python -m v6.natural.build_natural hosts|make|k1|feat|merge"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import random
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402  (setup_env truoc khi nap torch/transformers)
from vaudit import SR  # noqa: E402
from vaudit.edits import gap_after, gap_before, join  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAT = os.path.join(BASE, "data", "natural_v6")
HOSTS = os.path.join(NAT, "hosts")
OUTD = os.path.join(BASE, "data_labels", "v6", "natvi")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
NAME = "v6natvi"
ID = "v6nat_"
LANG = os.environ.get("NAT_LANG", "vi")   # 'en': bo tieng Anh tu nhien v6naten (EdAcc · AMI · VoxPopuli), cung thao tac
if LANG == "en":
    HOSTS = os.path.join(NAT, "hosts_en")
    OUTD = os.path.join(BASE, "data_labels", "v6", "naten")
    NAME, ID = "v6naten", "v6naten_"
elif LANG in ("ttsvi", "ttsen"):          # bo GIONG MAY tu nhien (viXTTS nhai giong chinh nguoi noi host), tap rieng -> khong lam doi khoa cache v6b
    HOSTS = os.path.join(NAT, "hosts_en") if LANG == "ttsen" else HOSTS
    OUTD = os.path.join(BASE, "data_labels", "v6", LANG)
    NAME, ID = "v6" + LANG, "v6" + LANG + "_"
elif LANG == "amilong":                   # tep dai nhieu cho sua tu cuoc hop AMI lien mach (v6.boundary.build_long)
    OUTD = os.path.join(BASE, "data_labels", "v6", "amilong")
    NAME, ID = "v6amilong", "v6amilong_"
elif LANG == "chain":                     # bo SINH THEO CAU HINH (nhieu dieu kien chong nhau, ap sau khi sua)
    OUTD = os.path.join(BASE, "data_labels", "v6", "chain")
    NAME, ID = "v6chain", "v6chain_"
elif LANG == "benign":                    # bo NGUON BAO NHAM LANH TINH (ung dung goi / dien thoai tren tep that)
    OUTD = os.path.join(BASE, "data_labels", "v6", "benign")
    NAME, ID = "v6benign", "v6benign_"
elif LANG == "cover2":                    # bo PHU bien cua chinh cho sua (mep / 1 am tiet / gan nua tep / hai cho sat nhau)
    OUTD = os.path.join(BASE, "data_labels", "v6", "cover2")
    NAME, ID = "v6cover2", "v6cover2_"
elif LANG == "cover":                     # bo PHU: dieu kien bien cua Constraint ap len tep goc TU NHIEN (v6.coverage.build_axes)
    OUTD = os.path.join(BASE, "data_labels", "v6", "cover")
    NAME, ID = "v6cover", "v6cover_"
EN_ASR = "facebook/wav2vec2-base-960h"    # chi de lay moc tu / kiem TTS; K1 + dac trung van dung ASR tieng Viet nhu v5
EN_MAX_CER = 0.35                         # ASR lech transcript goc qua muc nay -> moc tu khong dang tin, bo ban goc
EN_PER_PERSON = 40                        # tran ban goc moi nguoi noi (EdAcc / AMI xep theo cuoc hoi thoai)
EDGE_S, MAX_EDIT_S, MIN_WORDS = 0.5, 5.0, 8

PLAN = {"nguyen_ban": 600, "nen_toan_cuc": 400, "nen_cuc_bo": 400, "giong_toan_cuc": 400, "giong_cuc_bo": 400,
        "ghep_thay": 400, "ghep_chen": 300, "xoa_loi": 400}
SCALE = float(os.environ.get("NAT_SCALE", "1"))   # chay thu nho: NAT_SCALE=0.02
PHONE_PER_KIND = 100          # moi (loai, kenh dien thoai chinh); nguyen_ban x1,5
ALL_CFG_PER_KIND = 5          # moi (loai, cau hinh) trong du 27 cau hinh
if LANG in ("ttsvi", "ttsen"):
    PHONE_PER_KIND, ALL_CFG_PER_KIND = 30, 0     # tiet kiem GPU: moi (loai, kenh chinh) 30 tep, khong lam du 27 cau hinh
DEN_FAMS = ["N1", "N2", "N3", "N4", "N5", "N7"]
VOC_FAMS = ["V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass
class NHost:
    file: str
    speaker: str      # cum thong ke
    person: str       # nguoi noi (lay donor cung nguoi)
    source: str
    text: str
    syll: list
    path: str         # am thanh doc tu dia khi can (2 700 ban goc nap san ton ~3,5 GB RAM)
    words: list
    info: dict = None  # tieng Anh: giong / L1 / gioi tinh cua nguoi noi (neu bo du lieu co)

    @property
    def dry(self) -> np.ndarray:
        return sf.read(self.path, dtype="float32")[0].astype(float)


# ------------------------------------------------------------------ hosts
def _iter_vss():
    seen = set()
    for root in (os.path.join(NAT, "vietsuperspeech", "audio"), os.path.join(BASE, "data", "real_pilot", "vietsuperspeech", "audio")):
        for f in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
            b = os.path.basename(f)
            ch = os.path.relpath(f, root).split(os.sep)[0].split("_part")[0].replace("asr_dataset_", "").replace("asr_segments_", "")
            if "nguoivietdailynews" in ch or b in seen:
                continue
            seen.add(b)
            video = b.rsplit("_seg", 1)[0]
            y, sr = sf.read(f, dtype="float32", always_2d=True)
            yield f"vss_{b[:-4]}", "vss", f"vss:{ch}:{video}"[:120], f"vss:{ch}", y.mean(1), sr


def _iter_parquet(sub, src, spk_col):
    """Doc tung lo 16 dong (tep parquet 300-400 MB, doc ca tep mot lan tran RAM)."""
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(NAT, sub, "data", "test*.parquet"))):
        pf = pq.ParquetFile(p)
        ac = next(c for c in pf.schema_arrow.names if "audio" in c)
        i = -1
        for batch in pf.iter_batches(batch_size=16, columns=[ac, spk_col]):
          for row in batch.to_pylist():
            i += 1
            v = row[ac]
            if isinstance(v, dict) and v.get("bytes") is not None:
                y, sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32", always_2d=True)
                y = y.mean(1)
            elif isinstance(v, dict) and v.get("array") is not None:
                y, sr = np.asarray(v["array"], np.float32), int(v.get("sampling_rate") or 16000)
            else:
                continue
            spk = str(row[spk_col])
            yield f"{src}_{os.path.basename(p)[:-8]}_{i:05d}", src, f"{src}:{spk}", f"{src}:{spk}", y, sr


def _iter_parquet_en(sub, src, cols, ids, info=()):
    """Parquet tieng Anh -> (fid, nguon, cum, nguoi, y, sr, transcript goc, {cot info}). ids(row) -> (cum, nguoi noi)."""
    import pyarrow.parquet as pq

    def rows(p):
        i = -1
        # pre_buffer=False + doc dem 8 MB: mac dinh pyarrow nap truoc ca cot audio cua tep (~400 MB/tep x 10 tep xen ke = 5 GB)
        pf = pq.ParquetFile(p, pre_buffer=False, buffer_size=1 << 23)
        for batch in pf.iter_batches(batch_size=8, columns=["audio"] + cols + list(info)):
            for row in batch.to_pylist():
                i += 1
                yield p, i, row
    # doc XEN KE cac tep parquet (moi tep chua vai cuoc hoi thoai / hop) -> ban goc rai tren nhieu cum ngay tu dau
    its = [rows(p) for p in sorted(glob.glob(os.path.join(NAT, sub, "**", "*test*.parquet"), recursive=True))]
    while its:
        for it in list(its):
            nxt = next(it, None)
            if nxt is None:
                its.remove(it)
                continue
            p, i, row = nxt
            v = row["audio"]
            if not isinstance(v, dict) or v.get("bytes") is None:
                continue
            y, sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32", always_2d=True)
            cluster, person = ids(row)
            yield (f"{src}_{os.path.basename(p)[:-8].split('-of-')[0]}_{i:05d}", src, f"{src}:{cluster}", f"{src}:{person}",
                   y.mean(1), sr, str(row[cols[0]] or ""),
                   {k: row[k] for k in info if row.get(k) not in (None, "None", "")})


def _host_iters():
    if LANG == "vi":
        return {"vss": _iter_vss(), "vox": _iter_parquet("voxvietnam", "vox", "speaker"),
                "celeb": _iter_parquet("vietnam_celeb", "celeb", "label")}
    # EdAcc: cum = cuoc hoi thoai (2 nguoi chung mot cuoc goi), nguoi = speaker · AMI: cum = cuoc hop, nguoi = speaker_id
    # (nguoi AMI du 4 buoi a-d -> co doan cho 'ban ghi khac') · VoxPopuli: cum = nguoi = speaker_id
    return {"edacc": _iter_parquet_en("en_edacc", "edacc", ["text", "speaker"],
                                      lambda r: (str(r["speaker"]).rsplit("-", 1)[0], r["speaker"]), ("accent", "l1", "gender")),
            "ami": _iter_parquet_en("en_ami", "ami", ["text", "meeting_id", "speaker_id"],
                                    lambda r: (r["meeting_id"], r["speaker_id"])),
            "voxp": _iter_parquet_en("en_voxpopuli", "voxp", ["normalized_text", "speaker_id"],
                                     lambda r: (r["speaker_id"], r["speaker_id"]), ("accent", "gender"))}


def _use_lang_asr() -> None:
    """Tieng Anh: vaudit.asr dung wav2vec2-base-960h (moc tu, kiem TTS). Chi goi trong hosts / make, khong goi o k1 / feat."""
    if LANG in ("en", "ttsen"):
        import vaudit.asr as asr
        asr.ASR_NAME = EN_ASR
        asr._state.clear()
        os.environ["TTS_LANG"] = "en"


def _norm_en(t: str) -> str:
    """Chu thuong, chi giu a-z va dau nhay (so sanh voi dau ra ASR 960h)."""
    import re
    return " ".join(re.sub(r"[^a-z' ]+", " ", re.sub(r"<[^>]*>|\[[^\]]*\]", " ", t.lower())).split())


def step_hosts(limit_per_src: int) -> None:
    import librosa
    from vaudit.asr import cer, transcribe_words
    _use_lang_asr()
    os.makedirs(HOSTS, exist_ok=True)
    idx_path = os.path.join(HOSTS, "hosts.jsonl")
    old = [json.loads(l) for l in open(idx_path, encoding="utf-8")] if os.path.exists(idx_path) else []
    done = {r["file"] for r in old}
    per_person = defaultdict(int)             # tieng Anh: <= EN_PER_PERSON ban goc / nguoi -> nhieu nguoi noi hon
    for r in old:
        per_person[r["person"]] += 1
    gens = _host_iters()
    n_bad_asr = defaultdict(int)
    with open(idx_path, "a", encoding="utf-8") as out:
        for src, gen in gens.items():
            kept = sum(1 for f in done if f.startswith(src + "_"))
            for fid, source, cluster, person, y, sr, *extra in gen:
                ref, info = (extra[0], extra[1]) if extra else (None, {})
                if kept >= limit_per_src:
                    break
                if fid in done or (LANG == "en" and per_person[person] >= EN_PER_PERSON):
                    continue
                if sr != SR:
                    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
                dur = len(y) / SR
                if not 4.0 <= dur <= 20.0:
                    continue
                text, words = transcribe_words(y)
                if LANG == "en":
                    words = [type(w)(w.text.lower(), w.start, w.end) for w in words]
                    text = " ".join(w.text for w in words)
                    if not ref or not _norm_en(ref) or cer(_norm_en(ref), text) > EN_MAX_CER:
                        n_bad_asr[src] += 1
                        continue
                if len(words) < MIN_WORDS:
                    continue
                d = os.path.join(HOSTS, source); os.makedirs(d, exist_ok=True)
                sf.write(os.path.join(d, fid + ".wav"), y, SR)
                out.write(json.dumps({"file": fid, "source": source, "cluster": cluster, "person": person, "dur": round(dur, 3),
                                      "text": text, "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in words],
                                      **({"ref": ref, "info": info} if ref else {})}, ensure_ascii=False) + "\n")
                out.flush()
                kept += 1
                per_person[person] += 1
                if kept % 200 == 0:
                    print(f"  hosts {src}: {kept} (bo vi ASR lech transcript: {n_bad_asr[src]})", flush=True)
            gen.close()                        # giai phong bo doc parquet cua nguon nay truoc khi sang nguon sau
            print(f"hosts {src}: {kept} (bo vi ASR lech transcript: {n_bad_asr[src]})", flush=True)


def load_hosts() -> list[NHost]:
    from vaudit.asr import Word
    hs = []
    for l in open(os.path.join(HOSTS, "hosts.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        words = [Word(t, a, b) for t, a, b in r["words"]]
        hs.append(NHost(r["file"], r["cluster"], r["person"], r["source"], r["text"], [w.text for w in words],
                        os.path.join(HOSTS, r["source"], r["file"] + ".wav"), words, r.get("info") or {}))
    return hs


# ------------------------------------------------------------------ thao tac
def _span_len(rng: random.Random) -> int:
    return rng.choices([1, 2, 3, rng.randint(4, 6), rng.randint(7, 12)], weights=[30, 20, 15, 20, 15])[0]


def _pick_span(h: NHost, n: int, rng: random.Random):
    """Cum tu [k..m] dai n, nam trong vung soi, <= MAX_EDIT_S va <= 50 % tep."""
    dur = len(h.dry) / SR
    for _ in range(30):
        n2 = max(1, min(n, len(h.words) - 2))
        k = rng.randint(1, len(h.words) - 1 - n2)
        m = k + n2 - 1
        a, b = gap_before(h.words, k), gap_after(h.words, m, dur)
        if a >= EDGE_S and b <= dur - EDGE_S and b - a <= min(MAX_EDIT_S, 0.5 * dur):
            return k, m, a, b
        n = max(1, n - 1)
    return None


def _rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def _donor_seg(h: NHost, by_person, by_cluster, n: int, rng: random.Random):
    same = [d for d in by_cluster[h.speaker] if d.file != h.file and d.person == h.person]   # cum hoi thoai/hop co 2+ nguoi
    other = [d for d in by_person[h.person] if d.speaker != h.speaker]
    pool, where = (same, "cung_ban_ghi") if same and (rng.random() < 0.6 or not other) else (other, "ban_ghi_khac")
    if not pool:
        return None
    d = rng.choice(pool)
    sp = _pick_span(d, n, rng)
    if sp is None:
        return None
    k, m, a, b = sp
    return d, d.dry[int(a * SR):int(b * SR)], " ".join(d.syll[k:m + 1]), where


def _meta(h: NHost, edits: list, expected: str) -> dict:
    l2 = sorted({e["level2"] for e in edits})
    l1 = sorted({{"02": "02_ai_hau_ky", "03": "03_ghep_chen"}[x[:2]] for x in l2}) or ["01_nguyen_ban"]
    return {"host_file": h.file, "speaker": h.speaker, "person": h.person, "source": h.source, "orig_text": h.text,
            "expected_text": expected, "level1": l1, "level2": l2, "edits": edits, "lang": LANG,
            **({"host_info": h.info} if h.info else {})}


def _process(op: str, y: np.ndarray, fam: str, rng: np.random.Generator):
    """Khu on / doi giong bang backends co QC; N7 = WebRTC NS muc 2-3 (thu vien that)."""
    if fam == "N7":
        from v6.phone.channels import webrtc
        lvl = int(rng.integers(2, 4))
        return webrtc(np.asarray(y, np.float32), lvl, False).astype(float), {"tool": "WebRTC NS", "ns_level": lvl, "qc_ok": True,
                                                                           "family": "N7"}
    from v6.tools import backends as B
    steps = int(rng.choice(B.STEPS)) if op == "giong" else None
    for _ in range(3):
        try:
            z, p = B.apply(op, np.asarray(y, float), SR, fam, rng, steps=steps)
        except B.BackendError:
            continue
        if p.get("qc_ok", True) and p.get("strength_ok", True):
            return z, {**p, "family": fam}
    return None, None


def make_one(kind: str, h: NHost, by_person, by_cluster, rng: random.Random, nrng: np.random.Generator):
    y = h.dry.astype(float)
    dur = len(y) / SR
    fade = rng.choice([0.0, 5.0, 20.0])
    if kind == "nguyen_ban":
        return y, _meta(h, [], h.text)
    if kind in ("nen_toan_cuc", "giong_toan_cuc"):
        op = kind.split("_")[0]
        fam = rng.choice(DEN_FAMS if op == "nen" else VOC_FAMS)
        z, p = _process(op, y, fam, nrng)
        if z is None:
            return None
        return z, _meta(h, [dict(level2=f"02_{op}_toan_cuc", kind="global",
                                 **{k: v for k, v in p.items() if isinstance(v, (int, float, str, bool))})], h.text)
    if kind in ("nen_cuc_bo", "giong_cuc_bo"):
        op = kind.split("_")[0]
        sp = _pick_span(h, rng.randint(2, 10), rng)
        if sp is None:
            return None
        k, m, a, b = sp
        fam = rng.choice(DEN_FAMS if op == "nen" else VOC_FAMS)
        seg = y[int(a * SR):int(b * SR)]
        z, p = _process(op, seg, fam, nrng)
        if z is None:
            return None
        from vaudit.edits import blend_process
        out = blend_process(y, a, b, np.asarray(z, float)[:len(seg)], SR)
        return out, _meta(h, [dict(level2=f"02_{op}_cuc_bo", kind="segment", start=a, end=b, family=fam,
                                   words=" ".join(h.syll[k:m + 1]))], h.text)
    if kind == "xoa_loi":
        sp = _pick_span(h, _span_len(rng), rng)
        if sp is None:
            return None
        k, m, a, b = sp
        out, joints = join([y[:int(a * SR)], y[int(b * SR):]], SR, fade)
        t = joints[0] / SR
        return out, _meta(h, [dict(level2="03_ghep_giong_that", sub="cat_bo", op="xoa_loi", kind="segment",
                                   start=max(0.0, t - 0.25), end=min(len(out) / SR, t + 0.25), joint=t,
                                   deleted=" ".join(h.syll[k:m + 1]), n_words=m - k + 1, fade_ms=fade)],
                          " ".join(h.syll[:k] + h.syll[m + 1:]))
    if kind in ("ghep_thay", "ghep_chen"):
        n = _span_len(rng) if kind == "ghep_thay" else rng.choice([1, 1, 2, 3, rng.randint(4, 8)])
        dn = _donor_seg(h, by_person, by_cluster, n, rng)
        if dn is None:
            return None
        donor, seg, dwords, where = dn
        if kind == "ghep_thay":
            sp = _pick_span(h, _span_len(rng), rng)
            if sp is None:
                return None
            k, m, a, b = sp
            ref = y[int(a * SR):int(b * SR)]
            removed, expected_mid = " ".join(h.syll[k:m + 1]), dwords
            pre, post = h.syll[:k], h.syll[m + 1:]
        else:
            j = rng.randint(2, len(h.words) - 2)
            a = b = gap_before(h.words, j)
            if not EDGE_S <= a <= dur - EDGE_S:
                return None
            ref = y[int(h.words[j - 1].start * SR):int(h.words[j].end * SR)]
            removed, expected_mid = "", dwords
            pre, post = h.syll[:j], h.syll[j:]
        seg = seg * (_rms(ref) / (_rms(seg) + 1e-9))
        out, joints = join([y[:int(a * SR)], seg, y[int(b * SR):]], SR, fade)
        s, e = joints[0] / SR, joints[1] / SR
        if e - s > MAX_EDIT_S or (e - s) > 0.5 * len(out) / SR:
            return None
        sub = "thay_that" if kind == "ghep_thay" else "chen_that"
        return out, _meta(h, [dict(level2="03_ghep_giong_that", sub=sub, op=kind, kind="segment", start=s, end=e,
                                   removed=removed, inserted=dwords, donor_file=donor.file, donor_where=where,
                                   n_words=len(dwords.split()), fade_ms=fade)],
                          " ".join(pre + expected_mid.split() + post))
    raise ValueError(kind)


TTS_PLAN = {"may_thay": 250, "may_chen": 150, "may_ca_file": 100}
VSASV_PLAN = {"doi_giong_ai": 300, "nguyen_ban_vsasv": 200, "ngoai_pham_vi_replay": 100, "ngoai_pham_vi_doi_khang": 100}
TMP_TTS = os.path.join(BASE, ".tmp_v6", "tts_nat")


def _tts_phrase(text: str, h: NHost, seed: int):
    """viXTTS nhai giong ban goc doc `text`; kiem lai bang ASR (CER <= 0,35: loi noi tu nhien, tu ngan)."""
    from vaudit.asr import cer, transcribe_words
    from vaudit.tts import clone_speak
    ref = h.dry[: int(min(len(h.dry) / SR, 10.0) * SR)]
    best = None
    for s in range(2):
        y = clone_speak(text, np.asarray(ref, float), TMP_TTS, seed=seed + s)
        t, _ = transcribe_words(y)
        c = cer(text, t)
        if best is None or c < best[1]:
            best = (y, c)
        if c <= 0.2:
            break
    return best if best and best[1] <= 0.35 else None


def _splice(y, a, b, seg, ref, fade):
    seg = seg * (_rms(ref) / (_rms(seg) + 1e-9))
    out, joints = join([y[:int(a * SR)], seg, y[int(b * SR):]], SR, fade)
    return out, joints[0] / SR, joints[1] / SR


def make_tts(kind: str, h: NHost, by_person, by_cluster, rng: random.Random):
    y = h.dry.astype(float)
    dur = len(y) / SR
    fade = rng.choice([0.0, 5.0, 20.0])
    if kind == "may_ca_file":
        if len(h.text) > 200:
            return None
        r = _tts_phrase(h.text, h, rng.randint(0, 999))
        if r is None:
            return None
        z, c = r
        z = z * (_rms(y) / (_rms(z) + 1e-9))
        if not 3.0 <= len(z) / SR <= 20.0:
            return None
        return z, _meta(h, [dict(level2="03_chen_giong_may", op="may_ca_file", kind="global", tts="viXTTS",
                                 tts_cer=round(c, 3))], h.text)
    if kind == "may_thay":
        sp = _pick_span(h, rng.choice([1, 2, 3, rng.randint(4, 8)]), rng)
        if sp is None:
            return None
        k, m, a, b = sp
        same = rng.random() < 0.5
        text = " ".join(h.syll[k:m + 1])
        if not same:
            dn = _donor_seg(h, by_person, by_cluster, m - k + 1, rng)
            if dn is not None:
                text = dn[2]
            else:
                same = True
        r = _tts_phrase(text, h, rng.randint(0, 999))
        if r is None:
            return None
        out, s_, e_ = _splice(y, a, b, r[0], y[int(a * SR):int(b * SR)], fade)
        if e_ - s_ > MAX_EDIT_S:
            return None
        return out, _meta(h, [dict(level2="03_chen_giong_may", sub="thay_the", op="may_thay", kind="segment", start=s_,
                                   end=e_, removed=" ".join(h.syll[k:m + 1]), inserted=text, giu_noi_dung=same,
                                   tts="viXTTS", tts_cer=round(r[1], 3), n_words=len(text.split()), fade_ms=fade)],
                          " ".join(h.syll[:k] + text.split() + h.syll[m + 1:]))
    if kind == "may_chen":
        dn = _donor_seg(h, by_person, by_cluster, rng.choice([1, 2, 3, rng.randint(4, 6)]), rng)
        if dn is None:
            return None
        text = dn[2]
        j = rng.randint(2, len(h.words) - 2)
        t = gap_before(h.words, j)
        if not EDGE_S <= t <= dur - EDGE_S:
            return None
        r = _tts_phrase(text, h, rng.randint(0, 999))
        if r is None:
            return None
        ref = y[int(h.words[j - 1].start * SR):int(h.words[j].end * SR)]
        out, s_, e_ = _splice(y, t, t, r[0], ref, fade)
        if e_ - s_ > MAX_EDIT_S:
            return None
        return out, _meta(h, [dict(level2="03_chen_giong_may", sub="chen_them", op="may_chen", kind="segment", start=s_,
                                   end=e_, inserted=text, tts="viXTTS", tts_cer=round(r[1], 3),
                                   n_words=len(text.split()), fade_ms=fade)],
                          " ".join(h.syll[:j] + text.split() + h.syll[j:]))
    raise ValueError(kind)


def _load_man():
    mp = os.path.join(OUTD, "manifest.json")
    return json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else []


def _save_man(man):
    mp = os.path.join(OUTD, "manifest.json")
    import time
    with open(mp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=0)
    for attempt in range(20):                  # Windows: trinh diet virus / chi muc giu tep vua ghi -> thu lai
        try:
            os.replace(mp + ".tmp", mp)
            return
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)
    raise PermissionError(f"khong ghi duoc {mp} sau 20 lan thu")


def _gen_kind(kind, n, hosts, rng, man, maker):
    """Sinh toi da n tep 'goc' cho mot loai; chay lai thi tiep tuc tu so da co (luu manifest moi 50 tep)."""
    k = sum(1 for e in man if e["kind"] == kind and e["cfg"] == "goc")
    used = {e["meta"]["host_file"] for e in man if e["kind"] == kind}
    per_cluster = defaultdict(int)
    for e in man:
        if e["kind"] == kind and e["cfg"] == "goc":
            per_cluster[e["meta"]["speaker"]] += 1
    pool = [h for h in hosts if h.file not in used]
    rng.shuffle(pool)
    for h in pool:
        if k >= n:
            break
        if per_cluster[h.speaker] >= 6:            # tran moi cum -> nhieu cum doc lap
            continue
        try:
            r = maker(kind, h)
        except Exception as ex:                    # mot ban goc hong khong lam dung ca buoc
            print(f"    bo {h.file}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
            continue
        if r is None:
            continue
        wav, meta = r
        vid = f"{ID}{kind}_{k:04d}"
        man.append(_write(vid, kind, "goc", np.asarray(wav, np.float32), {**meta, "kenh": "goc"}, [h.speaker], vid))
        per_cluster[h.speaker] += 1
        k += 1
        if k % 50 == 0:
            _save_man(man)
    _save_man(man)
    print(f"  {kind:<24} {k}/{n}", flush=True)


def _add_channels(man, kinds, seed):
    from v6.phone import channels as PC
    have = {e["id"] for e in man}
    extra = [c for c in PC.CONFIGS if c not in PC.MAIN_PHONE and c != "goc"]
    for kind in kinds:
        items = [e for e in man if e["kind"] == kind and e["cfg"] == "goc"]
        if not items:
            continue
        n_phone = max(1, int(PHONE_PER_KIND * SCALE * (1.5 if kind.startswith("nguyen_ban") else 1)))
        jobs = [(cfg, e, seed + ci * 7919 + j) for ci, cfg in enumerate(PC.MAIN_PHONE)
                for j, e in enumerate(random.Random(seed + ci).sample(items, min(n_phone, len(items))))]
        jobs += [(cfg, e, seed + 100003 + ci * 131 + j) for ci, cfg in enumerate(extra)
                 for j, e in enumerate(random.Random(seed + 500 + ci).sample(items, min(ALL_CFG_PER_KIND, len(items))))]
        for cfg, e, sd in jobs:
            if f"{e['id']}__{cfg}" not in have:
                man.append(_variant(e, cfg, PC, sd))
        _save_man(man)
        print(f"  kenh {kind}: {len(jobs)} bien the", flush=True)


def _vsasv_rows():
    import pyarrow.parquet as pq
    for p in sorted(glob.glob(os.path.join(NAT, "vsasv", "data", "*.parquet"))):
        for b in pq.ParquetFile(p).iter_batches(batch_size=32, columns=["audio", "label", "utt_type", "file"]):
            for r in b.to_pylist():
                a = r["audio"]
                if a.get("array") is None:
                    continue
                y = np.asarray(a["array"], np.float32)
                sr = int(a.get("sampling_rate") or SR)
                if len(y) / sr >= 3.5:
                    yield r["utt_type"], str(r["label"]), str(r["file"]), y, sr


def add_vsasv(man):
    """VSASV: doi giong bang AI ca tep (02 doi giong ca file) · ban that nguoi noi khac · ngoai pham vi (phat lai, doi khang)."""
    import librosa
    want = {"voice_conversion": "doi_giong_ai", "bonafide": "nguyen_ban_vsasv", "replay": "ngoai_pham_vi_replay",
            "adversarial_attack": "ngoai_pham_vi_doi_khang"}
    got = defaultdict(int)
    for e in man:
        if e["kind"] in VSASV_PLAN and e["cfg"] == "goc":
            got[e["kind"]] += 1
    per_spk = defaultdict(int)
    for ut, spk, f, y, sr in _vsasv_rows():
        kind = want.get(ut)
        if kind is None or got[kind] >= max(1, int(VSASV_PLAN[kind] * SCALE)) or per_spk[(kind, spk)] >= 40:
            continue
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        cl = f"vsasv:{spk}"
        edits = [dict(level2="02_giong_toan_cuc", kind="global", op="doi_giong_ai", family="VC (VSASV)")] \
            if kind == "doi_giong_ai" else []
        meta = {"host_file": f"vsasv_{f}", "speaker": cl, "person": cl, "source": "vsasv", "orig_text": "",
                "expected_text": "", "level1": ["02_ai_hau_ky"] if edits else ["01_nguyen_ban"],
                "level2": [x["level2"] for x in edits], "edits": edits, "lang": "vi", "vsasv_type": ut,
                "ngoai_pham_vi": kind.startswith("ngoai_pham_vi")}
        vid = f"{ID}{kind}_{got[kind]:04d}"
        man.append(_write(vid, kind, "goc", y, {**meta, "kenh": "goc"}, [cl], vid))
        got[kind] += 1
        per_spk[(kind, spk)] += 1
    _save_man(man)
    print("  vsasv:", dict(got), flush=True)


NARROW_N = 150   # tieng Anh: Earnings22 = cuoc goi dien thoai BANG HEP that (8 kHz) -> ngoai Constraint, ky vong 04


def add_narrowband(man):
    """Earnings22: doan 4-20 s, <= 8 doan / cuoc goi -> 'ngoai_pham_vi_bang_hep' (ban ghi that, khong sua)."""
    import librosa
    import pyarrow.parquet as pq
    got = sum(1 for e in man if e["kind"] == "ngoai_pham_vi_bang_hep")
    per = defaultdict(int)
    for e in man:
        if e["kind"] == "ngoai_pham_vi_bang_hep":
            per[e["meta"]["speaker"]] += 1
    have = {e["meta"]["host_file"] for e in man if e["kind"] == "ngoai_pham_vi_bang_hep"}
    n = max(1, int(NARROW_N * SCALE))
    for p in sorted(glob.glob(os.path.join(NAT, "en_earnings22", "**", "*.parquet"), recursive=True)):
        for b in pq.ParquetFile(p).iter_batches(batch_size=16, columns=["audio", "file_id", "segment_id", "transcription"]):
            for r in b.to_pylist():
                hf = f"earn22_{r['file_id']}_{r['segment_id']}"
                cl = f"earn22:{r['file_id']}"
                if got >= n or hf in have or per[cl] >= 8:
                    continue
                y, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32", always_2d=True)
                y = librosa.resample(y.mean(1), orig_sr=sr, target_sr=SR) if sr != SR else y.mean(1)
                if not 4.0 <= len(y) / SR <= 20.0:
                    continue
                meta = {"host_file": hf, "speaker": cl, "person": cl, "source": "earnings22", "orig_text": r["transcription"],
                        "expected_text": r["transcription"], "level1": ["01_nguyen_ban"], "level2": [], "edits": [],
                        "lang": LANG, "ngoai_pham_vi": True, "ly_do_ngoai_pham_vi": "bang hep 8 kHz (cuoc goi dien thoai that)"}
                vid = f"{ID}ngoai_pham_vi_bang_hep_{got:04d}"
                man.append(_write(vid, "ngoai_pham_vi_bang_hep", "goc", y, {**meta, "kenh": "goc"}, [cl], vid))
                got += 1
                per[cl] += 1
    _save_man(man)
    print(f"  bang hep (Earnings22): {got}", flush=True)


def step_make(seed: int, part: str) -> None:
    _use_lang_asr()
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    hosts = load_hosts()
    by_person, by_cluster = defaultdict(list), defaultdict(list)
    for h in hosts:
        by_person[h.person].append(h)
        by_cluster[h.speaker].append(h)
    print(f"{len(hosts)} ban goc · {len(by_cluster)} cum · {len(by_person)} nguoi/kenh", flush=True)
    os.makedirs(OUTD, exist_ok=True)
    man = _load_man()
    if part in ("edits", "all"):
        for kind, n in PLAN.items():
            _gen_kind(kind, max(1, int(round(n * SCALE))), hosts, rng, man,
                      lambda k, h: make_one(k, h, by_person, by_cluster, rng, nrng))
        if LANG == "vi":                           # VSASV chi co tieng Viet
            add_vsasv(man)
        else:
            add_narrowband(man)
        _add_channels(man, list(PLAN) + (["doi_giong_ai", "nguyen_ban_vsasv"] if LANG == "vi" else []), seed)
    if part in ("tts", "all"):
        for kind, n in TTS_PLAN.items():
            _gen_kind(kind, max(1, int(round(n * SCALE))), hosts, rng, man,
                      lambda k, h: make_tts(k, h, by_person, by_cluster, rng))
        _add_channels(man, list(TTS_PLAN), seed + 1)
    print(f"-> {len(man)} tep", flush=True)


def _write(vid, kind, cfg, wav, meta, spks, src):
    d = os.path.join(OUTD, kind); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, vid + ".wav")
    peak = float(np.abs(wav).max())
    if peak > 0.999:
        wav = wav / peak * 0.999
    sf.write(p, wav, SR)
    meta = {**meta, "fold": C.assign_fold(spks)}
    return {"id": vid, "kind": kind, "cfg": cfg, "src": src, "wav": os.path.relpath(p, V4).replace("\\", "/"), "meta": meta}


def _variant(e, cfg, PC, seed):
    y, _ = sf.read(os.path.join(V4, e["wav"]), dtype="float32")
    z, cm = PC.apply(y, cfg, seed)
    return _write(f"{e['id']}__{cfg}", e["kind"], cfg, z, {**e["meta"], "kenh": cfg, "kenh_meta": cm},
                  [e["meta"]["speaker"]], e["id"])


def step_k1() -> None:
    from vaudit import k1
    mp = os.path.join(OUTD, "manifest.json")
    man = json.load(open(mp, encoding="utf-8"))
    # Tien trinh chet (segfault do het bo nho / thu vien C) ngay tren mot tep -> dem so lan; chi bo qua tep sau 3 lan
    # (thuong la het RAM chu khong phai loi cua tep, nen khong danh dau loi ngay).
    cur, cnt_p = os.path.join(OUTD, "_k1_dang_xu_ly.txt"), os.path.join(OUTD, "_k1_so_lan_sap.json")
    cnt = json.load(open(cnt_p, encoding="utf-8")) if os.path.exists(cnt_p) else {}
    if os.path.exists(cur):
        crashed = open(cur, encoding="utf-8").read().strip()
        cnt[crashed] = cnt.get(crashed, 0) + 1
        json.dump(cnt, open(cnt_p, "w", encoding="utf-8"))
    for i, e in enumerate(man):
        if e["meta"].get("k1", {}).get("decision") == "loi_k1" and cnt.get(e["id"], 0) < 3:
            del e["meta"]["k1"]                        # danh dau loi o lan truoc -> thu lai
        if "k1" in e["meta"]:
            continue
        if cnt.get(e["id"], 0) >= 3:
            e["meta"]["k1"] = {"decision": "loi_k1", "reasons": ["K1 lam sap tien trinh 3 lan tren tep nay"], "warnings": [],
                               "measures": {}, "kenh": None}
            print(f"  bo qua tep lam sap K1 3 lan: {e['id']}", flush=True)
            continue
        with open(cur, "w", encoding="utf-8") as fh:
            fh.write(e["id"])
        r = k1.check(os.path.join(V4, e["wav"]))
        e["meta"]["k1"] = {"decision": r["decision"], "reasons": r["reasons"],
                           "warnings": [w for w in r["warnings"] if not w.startswith("0,3")],
                           "measures": {k: v for k, v in r["measures"].items() if k != "kenh"}, "kenh": r["measures"].get("kenh")}
        if i % 200 == 0:
            _save_man(man)
            print(f"  k1 {i}/{len(man)}", flush=True)
    _save_man(man)
    if os.path.exists(cur):
        os.remove(cur)
    print("k1 xong", flush=True)


def step_feat(threads: int, shard: tuple = (0, 1)) -> None:
    """Trich theo thu tu uu tien (~8 s/tep tren CPU): tep goc -> 50 tep dau moi (loai, kenh dien thoai chinh) -> phan con lai
    cua 4 kenh chinh -> 22 kenh phu. features() luu moi 20 tep nen co the cham phan da xong trong khi van trich tiep."""
    from v6.phone.channels import MAIN_PHONE
    man = json.load(open(os.path.join(OUTD, "manifest.json"), encoding="utf-8"))
    rank = defaultdict(int)

    def prio(e):
        if e["cfg"] == "goc":
            return 0
        if e["cfg"] in MAIN_PHONE:
            rank[(e["kind"], e["cfg"])] += 1
            return 1 if rank[(e["kind"], e["cfg"])] <= 50 else 2
        return 3
    order = sorted(man, key=prio)
    k, n = shard
    if n == 1:
        C.extract_features(order, C.feat_path_for(NAME), threads=threads)
        return
    # Chia xen ke theo vi tri trong danh sach uu tien -> moi phan van lam tep goc truoc. Phan 0 qua extract_features
    # (khoa heavy_job + sidecar) vao pkl chinh; phan khac goi thang features() vao pkl rieng, 'merge' gop lai sau.
    mine = [e for i, e in enumerate(order) if i % n == k]
    if k == 0:
        C.extract_features(mine, C.feat_path_for(NAME), threads=threads)
    else:
        import torch
        torch.set_num_threads(threads)
        from run_codec_variants import features
        main_done = set(pickle.load(open(C.feat_path_for(NAME), "rb"))) if os.path.exists(C.feat_path_for(NAME)) else set()
        features([e for e in mine if e["id"] not in main_done], C.feat_path_for(NAME)[:-4] + f"_s{k}.pkl")


def step_merge() -> None:
    """Gop pkl cac phan phu (_s* CPU, _g* GPU) vao pkl chinh, roi ghi lai sidecar cho CA manifest bang C._finish
    (dong bo nhan, kiem cache, sha256 pkl; khong nap mo hinh). Chay lai bao nhieu lan cung duoc, ke ca khi GPU dang trich:
    phan phu giu nguyen, lan gop sau lay them phan moi."""
    main = C.feat_path_for(NAME)
    cache = pickle.load(open(main, "rb")) if os.path.exists(main) else {}
    for p in sorted(glob.glob(main[:-4] + "_s*.pkl") + glob.glob(main[:-4] + "_g*.pkl")):
        part = pickle.load(open(p, "rb"))
        new = {k: v for k, v in part.items() if k not in cache}
        cache.update(new)
        print(f"  gop {os.path.basename(p)}: {len(part)} ({len(new)} moi)", flush=True)
    with open(main + ".tmp", "wb") as fh:
        pickle.dump(cache, fh)
    os.replace(main + ".tmp", main)
    del cache
    man = json.load(open(os.path.join(OUTD, "manifest.json"), encoding="utf-8"))
    entries = [C._normalize_entry(m) for m in man]
    items = [{"id": e["id"], "wav": e["wav"], "sha256": C.sha256_file(C.abs_wav(e["wav"])), "fold": int(e["meta"]["fold"])}
             for e in entries]
    s = C._finish(main, entries, items, {"feat_path": main, "n": len(entries)})
    print(f"pkl chinh: {s['n'] - s['missing']}/{s['n']} tep · thieu cache {s['missing_cache']} · "
          f"dong bo nhan {s['meta_synced']} · {s['status']}", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["hosts", "make", "k1", "feat", "merge"])
    ap.add_argument("--shard", default="0/1", help="feat: phan k/n (chay song song nhieu tien trinh)")
    ap.add_argument("--part", choices=["edits", "tts", "all"], default="edits",
                    help="make: thao tac thuong / giong may / ca hai")
    ap.add_argument("--limit", type=int, default=2500, help="so ban goc toi da moi nguon (hosts)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"hosts": lambda: step_hosts(a.limit), "make": lambda: step_make(a.seed, a.part), "k1": step_k1,
     "feat": lambda: step_feat(a.threads, tuple(int(x) for x in a.shard.split("/"))), "merge": step_merge}[a.step]()


if __name__ == "__main__":
    main()
