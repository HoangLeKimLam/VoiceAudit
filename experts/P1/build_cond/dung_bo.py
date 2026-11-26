# -*- coding: utf-8 -*-
"""Dựng bộ tệp kiểm điều kiện K1 `p1_dieukien_v1`: SNR · nhạc nền · vang · loại âm thanh (các hàng Constraint `k1_axes` chưa phủ).

Mỗi tệp = lời nói sạch (FLEURS vi/en, TẠM) + một biến đổi có tham số biết trước → đáp án (mong_doi, ly_do) suy ra từ luật Constraint
(`VoiceAudit.html`, mục Constraint). Không gán nhãn tay. Mỗi tệp ghi một dòng `provenance.jsonl` (nguồn + sha256 + tham số + số đo lại).

Đầu ra (data_labels/v6/p1_dieukien_v1/):
  manifest.json      H1 (CHUNG §4): {id, kind, cfg, src, wav, meta{lang, source, person, speaker, level1, level2[], edits[], k1_ca}}
  cases.json         quy ước k1axes: {file, ca, mong_doi accept|canh_bao|04, ly_do, ca_bien, base}
  provenance.jsonl   nguồn gốc từng tệp
  KIEM_NGHE.md       2 tệp ngẫu nhiên / ô để người kiểm bằng tai
  <kind>/<id>.wav    16 kHz, mono, PCM 16-bit

HẠN CHẾ: FLEURS không có mã người nói → person = từng tệp (fold theo người chưa đúng) → bộ này CHỈ để thử script / thử K1, không nộp.

CLI:  python experts/P1/build_cond/dung_bo.py build [--n 12] [--seed 0] [--bo p1_dieukien_v2]
      python experts/P1/build_cond/dung_bo.py selftest            (hàm thuần, không cần dữ liệu)"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from vaudit import k1  # noqa: E402  (frame_db, speech_mask: cùng VAD năng lượng với K1)
from vaudit.events import license_ok  # noqa: E402  (lọc CC BY / BY-SA / CC0 / PD, bỏ NC / ND)

SR = 16000
RAW = os.path.join(ROOT, "data", "p1_raw")
BO = "p1_dieukien_v1"                   # đổi bằng --bo (build() đặt lại BO / OUT / RIR_DIR)
OUT = os.path.join(ROOT, "data_labels", "v6", BO)
RIR_DIR = os.path.join(RAW, "rir_pra")  # RIR tự sinh (pyroomacoustics), ghi lại để provenance có sha256; bộ khác v1 → rir_pra/<bo>/
# Nhạc thánh ca trong MUSAN (soát LICENSE + ANNOTATIONS 23/11): Stabat Mater (Tudor Consort) + album gospel «Adoro Te» (Angelo Mele).
# Đều có lời hát nên no_vocals đã loại; giữ danh sách để loại TƯỜNG MINH (Constraint: không nội dung tôn giáo).
THANH_CA = {f"music-fma-wa-{i:04d}" for i in range(83, 93)} | {f"music-jamendo-{i:04d}" for i in range(18, 24)}
LEVEL_DBFS = -23.0                     # mức RMS chuẩn hoá: xa ngưỡng «quá nhỏ» −40 dBFS, không cắt đỉnh
MIN_BASE_S = 5.0
MIN_BASE_SNR_DB = 25.0                 # Huấn chốt 08/11: gốc FLEURS phải có SNR sẵn có ≥ 25 dB (+ bù ồn, xem snr_add_for)
XF = int(0.01 * SR)                    # crossfade 10 ms ở chỗ nối vòng nguồn ồn / nhạc (fit_len) → không «tách»
LOAI_MAX_ZERO_S = 0.5                  # đoạn loai không được chứa im lặng số 0 dài hơn (bài nhạc hết giữa đoạn)

# 23/11 (bộ v2): nới dải để hiệu chỉnh bộ đo (Brouhaha / DeepFilterNet) quanh ngưỡng hợp đồng; v1 dựng với (5..20) · (5..30) · (0,5 1,0 1,5)
SNR_DB = (0, 5, 8, 10, 12, 15, 20)     # Constraint: < 10 → 04 «ồn quá» · 10–15 → cảnh báo «tiếng ồn cao»
NEN_PCT = (2, 5, 10, 15, 20, 30, 50)   # > 10 % năng lượng → 04 «nền che tiếng nói»
RT60_S = (0.3, 0.5, 0.8, 1.0, 1.2, 1.5)  # > 1,0 s → cảnh báo «phòng vang»
RT60_TOL = 0.1
T30_T20_TOL = 0.2                      # |T30 − T20| ≤ 20 % T20: EDC thẳng → RT60 đo tin được (ISO 3382); RIRS_NOISES cũ lệch 2–10×
# DEMAND: bỏ môi trường có tiếng người nói rõ (căng-tin, họp, nhà hàng, quảng trường) để tệp chỉ vi phạm trục SNR,
# không vô tình thêm «nhiều người nói» / «nói chồng».
DEMAND_BO = ("PCAFETER", "OMEETING", "PRESTO", "SPSQUARE")


# ----------------------------------------------------------------------------- hàm thuần (có selftest)
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.square(x, dtype=np.float64)))


def loopable(x: np.ndarray, w: int = XF) -> np.ndarray:
    """Bản mới dài len(x) − w, w mẫu đầu = trộn dần đuôi → đầu: lặp lại liền mạch (cuối bản = x[−w−1], đầu bản kế = x[−w])."""
    if len(x) <= 2 * w:
        return x
    r = np.linspace(0.0, 1.0, w)
    return np.r_[x[:w] * r + x[-w:] * (1 - r), x[w:-w]]


def fit_len(x: np.ndarray, n: int, offset: int) -> np.ndarray:
    """Lặp / cắt x thành đúng n mẫu, bắt đầu tại offset (để mỗi tệp lấy một đoạn khác của nguồn ồn / nhạc); chỗ nối có crossfade."""
    x = loopable(x)
    reps = int(np.ceil((offset % len(x) + n) / len(x))) + 1
    return np.tile(x, reps)[offset % len(x): offset % len(x) + n]


def native_snr(y: np.ndarray) -> float | None:
    """SNR sẵn có (công thức như k1.py dòng snr_db, mặt nạ năng lượng thay ASR): (công suất khung tiếng nói − nền) / nền.
    Thô: tính trên khung tiếng nói, còn SNR đích tính trên cả tệp → dùng để lọc gốc và bù ồn, không thay bộ đo SNR của K1."""
    db = k1.frame_db(y)
    sp, _ = k1.speech_mask(db)
    ns = ~sp & (db > k1.ZERO_DB)
    if not sp.any() or ns.sum() < 10:
        return None
    pw = 10 ** (db / 10)
    pn = pw[ns].mean()
    return float(10 * np.log10(max(pw[sp].mean() - pn, 1e-12) / pn))


def combine_snr(a_db: float, b_db: float) -> float:
    """Hai nguồn ồn độc lập cộng công suất: SNR tổng = −10·log10(10^(−a/10) + 10^(−b/10))."""
    return float(-10 * np.log10(10 ** (-a_db / 10) + 10 ** (-b_db / 10)))


def snr_add_for(target_db: float, base_db: float) -> float:
    """SNR của ồn THÊM vào để SNR thật (gộp ồn sẵn có của gốc) = target_db: nghịch đảo combine_snr. Cần base_db > target_db."""
    if base_db <= target_db:
        raise ValueError(f"gốc ({base_db:.1f} dB) không sạch hơn đích ({target_db} dB)")
    return float(-10 * np.log10(10 ** (-target_db / 10) - 10 ** (-base_db / 10)))


def longest_zero_run(y: np.ndarray, thr: float = 1e-4) -> float:
    z = np.r_[0, (np.abs(y) < thr).astype(np.int8), 0]
    d = np.diff(z)
    runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    return float(runs.max() / SR) if len(runs) else 0.0


def gain_for_snr(s: np.ndarray, n: np.ndarray, snr_db: float) -> float:
    """g sao cho 10·log10(E_s / E_{g·n}) = snr_db (năng lượng trên cả tệp)."""
    return float(np.sqrt(energy(s) / (energy(n) * 10 ** (snr_db / 10))))


def gain_for_ratio(s: np.ndarray, m: np.ndarray, r: float) -> float:
    """g sao cho E_{g·m} / (E_s + E_{g·m}) = r  ⇔  g = √(r/(1−r) · E_s/E_m)."""
    return float(np.sqrt(r / (1 - r) * energy(s) / energy(m)))


def rt60_schroeder(h: np.ndarray, sr: int = SR, lo_db: float = -25.0) -> float | None:
    """RT60 từ đáp ứng xung bằng tích phân ngược Schroeder; hồi quy tuyến tính EDC trong [−5, lo_db] dB, ngoại suy ra 60 dB.
    lo_db = −25 → T20 (dùng gán nhãn) · −35 → T30 (kiểm EDC có thẳng không)."""
    e = np.cumsum(np.square(h[::-1], dtype=np.float64))[::-1]
    edc = 10 * np.log10(e / e[0] + 1e-20)
    idx = np.where((edc <= -5) & (edc >= lo_db))[0]
    if len(idx) < 10:
        return None
    slope = np.polyfit(idx / sr, edc[idx], 1)[0]          # dB / s (âm)
    return float(-60 / slope) if slope < 0 else None


def reverb(s: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Chập lời với RIR, căn đường truyền thẳng (đỉnh RIR) về 0 để không trễ; giữ nguyên độ dài."""
    k = int(np.argmax(np.abs(h)))
    return fftconvolve(s, h[k:])[: len(s)]


def normalize(x: np.ndarray, dbfs: float = LEVEL_DBFS) -> tuple[np.ndarray, float]:
    """Đưa RMS về dbfs; nếu đỉnh > 0,99 thì hạ tiếp cho khỏi cắt đỉnh. Trả (tín hiệu, mức RMS thực tế dBFS)."""
    y = x * (10 ** (dbfs / 20) / rms(x))
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = y * (0.99 / pk)
    return y, 20 * np.log10(rms(y))


def expected(truc: str, v: float) -> tuple[str, str | None, bool]:
    """(mong_doi, ly_do, ca_bien) theo luật Constraint. ca_bien = đúng ngưỡng (hợp đồng không nói rõ phía nào)."""
    if truc == "snr":
        if v < 10:
            return "04", "ồn quá", False
        if v <= 15:
            return "canh_bao", "tiếng ồn cao", v in (10, 15)
        return "accept", None, False
    if truc == "nen":
        return ("04", "nền che tiếng nói", False) if v > 10 else ("accept", None, v == 10)
    if truc == "vang":                 # v = RT60 đo trên RIR; ca_bien khi trong sai số đo ±RT60_TOL quanh 1,0 s
        return ("canh_bao", "phòng vang", abs(v - 1.0) <= RT60_TOL) if v > 1.0 else ("accept", None, abs(v - 1.0) <= RT60_TOL)
    if truc == "loai":
        return "04", "không phải ghi âm lời nói", False
    if truc == "goc":
        return "accept", None, False
    raise ValueError(truc)


# ----------------------------------------------------------------------------- đọc nguồn
_SHA: dict = {}


def sha256(path: str) -> str:
    if path not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def load(path: str) -> np.ndarray:
    """Đọc wav/flac → mono float64 16 kHz (kênh đầu nếu nhiều kênh: RIR / DEMAND đa kênh)."""
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x[:, 0]
    return resample_poly(x, SR, sr) if sr != SR else x


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace("\\", "/")


def fleurs_bases(lang: str, n: int, rng: random.Random) -> list[dict]:
    """n câu FLEURS (dev + test) dài ≥ MIN_BASE_S và SNR sẵn có ≥ MIN_BASE_SNR_DB, mỗi câu một sentence id khác nhau, cân bằng giới.
    SNR sẵn có chỉ tính khi xét tới (lười) → không phải đo cả 1 000+ tệp."""
    d = os.path.join(RAW, "fleurs", lang)
    rows = []
    for split in ("dev", "test"):
        with open(os.path.join(d, f"{split}.tsv"), encoding="utf-8") as fh:
            for r in (ln.rstrip("\n").split("\t") for ln in fh):   # không dùng csv: câu có dấu " làm csv gộp cột
                wav = os.path.join(d, split, r[1])
                if int(r[5]) / SR >= MIN_BASE_S and os.path.exists(wav):
                    rows.append({"sid": r[0], "file": r[1], "gender": r[6], "wav": wav, "split": split})
    rng.shuffle(rows)
    snr0: dict = {}

    def clean(r: dict) -> bool:
        if r["wav"] not in snr0:
            snr0[r["wav"]] = native_snr(load(r["wav"]))
        return snr0[r["wav"]] is not None and snr0[r["wav"]] >= MIN_BASE_SNR_DB

    picked, seen = [], set()
    for g in ("MALE", "FEMALE") * n:
        r = next((r for r in rows if r["gender"] == g and r["sid"] not in seen and clean(r)), None)
        if r and len(picked) < n:
            picked.append({**r, "snr_san_db": round(snr0[r["wav"]], 2)})
            seen.add(r["sid"])
    print(f"{lang}: đo SNR sẵn có {len(snr0)} tệp, {sum(v is not None and v >= MIN_BASE_SNR_DB for v in snr0.values())} đạt ≥ "
          f"{MIN_BASE_SNR_DB:g} dB → chọn {len(picked)}", flush=True)
    return picked


MUSAN_ID = re.compile(r"\b(?:music|noise|speech)-[a-z-]+-\d{4}\b")
LIC_WORD = re.compile(r"(public domain|cc0|cc by|attribution|creative commons)", re.I)


def license_text_ok(t: str | None) -> bool:
    """Giấy phép viết tự do (MUSAN): cho CC BY / BY-SA / CC0 / Public Domain; bỏ NC / ND (DECISIONS 1, như vaudit.events.license_ok)."""
    if not t:
        return False
    s = t.lower()
    toks = set(re.split(r"[\s\-/()\[\],.:]+", s))
    if toks & {"nc", "nd"} or re.search(r"non-?commercial|no-?deriv|nonderiv", s):
        return False
    return bool(LIC_WORD.search(s))


def parse_musan_license(text: str) -> tuple[dict, str | None]:
    """({mã tệp: dòng giấy phép}, giấy phép chung). Khối ngăn bằng dòng '====='; giấy phép của khối = dòng đầu khớp LIC_WORD
    hoặc dạng 'CC BY…'; khối không có mã tệp = giấy phép chung (đầu tệp rfm, cả tệp librivox) áp cho mã không có giấy phép riêng."""
    per, common = {}, None
    for block in re.split(r"\n=+\s*\n?", text):
        lic = next((ln.strip() for ln in block.splitlines() if LIC_WORD.search(ln) or ln.strip().upper().startswith("CC ")), None)
        ids = MUSAN_ID.findall(block)
        if not ids:
            common = common or lic
        for i in ids:
            if lic:
                per[i] = lic
    return per, common


def musan_meta(sub: str) -> dict:
    """{mã tệp: {license, vocals}} cho musan/<sub>/*: giấy phép từ LICENSE, có lời hát (Y/N) từ ANNOTATIONS (cột 3)."""
    out = {}
    for root, _, files in os.walk(os.path.join(RAW, "musan", "musan", sub)):
        per, common = ({}, None)
        if "LICENSE" in files:
            per, common = parse_musan_license(open(os.path.join(root, "LICENSE"), encoding="utf-8", errors="replace").read())
        voc = {}
        if "ANNOTATIONS" in files:
            for ln in open(os.path.join(root, "ANNOTATIONS"), encoding="utf-8", errors="replace"):
                p = ln.split()
                if len(p) >= 3:
                    voc[p[0]] = p[2]
        for f in files:
            if f.endswith(".wav"):
                k = f[:-4]
                out[os.path.join(root, f)] = {"license": per.get(k, common), "vocals": voc.get(k)}
    return out


def musan_pool(sub: str, min_s: float, no_vocals: bool = False) -> list[dict]:
    """Tệp MUSAN dùng được: giấy phép cho phép, đủ dài; no_vocals → chỉ nhạc không lời (tránh thêm giọng hát / người nói thứ hai)."""
    pool = []
    for path, m in sorted(musan_meta(sub).items()):
        if not license_text_ok(m["license"]) or (no_vocals and m["vocals"] != "N") or os.path.basename(path)[:-4] in THANH_CA:
            continue
        if sf.info(path).duration >= min_s:
            pool.append({"path": path, "license": m["license"], "src": f"MUSAN/{sub}", "ann": m["vocals"]})
    return pool


def demand_pool() -> list[dict]:
    d = os.path.join(RAW, "demand")
    return [{"path": os.path.join(d, e, "ch01.wav"), "license": "CC BY 4.0", "src": f"DEMAND/{e}"}
            for e in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, e)) and e not in DEMAND_BO
            and os.path.exists(os.path.join(d, e, "ch01.wav"))]


def sim_rir(nrng: np.random.Generator, design_s: float) -> dict | None:
    """Một RIR phòng hộp (pyroomacoustics, phương pháp ảnh): kích thước ngẫu nhiên, hấp thụ đều theo inverse_sabine(design_s),
    nguồn cao 1,6 m, micro 1,2 m cách nguồn 1–3 m. None nếu phòng không đạt được RT60 thiết kế."""
    import pyroomacoustics as pra
    dim = [nrng.uniform(5, 12), nrng.uniform(4, 10), nrng.uniform(2.6, 4.0)]
    try:
        ab, order = pra.inverse_sabine(design_s, dim)
    except ValueError:
        return None
    room = pra.ShoeBox(dim, fs=SR, materials=pra.Material(ab), max_order=order)
    src = [nrng.uniform(1, dim[0] - 1), nrng.uniform(1, dim[1] - 1), 1.6]
    d, a = nrng.uniform(1, 3), nrng.uniform(0, 2 * np.pi)
    mic = np.clip([src[0] + d * np.cos(a), src[1] + d * np.sin(a), 1.2], 0.5, [dim[0] - 0.5, dim[1] - 0.5, dim[2] - 0.5])
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    return {"h": np.asarray(room.rir[0][0], dtype=np.float64), "phong_m": [round(v, 2) for v in dim], "hap_thu": round(float(ab), 4),
            "max_order": int(order), "nguon_xyz": [round(v, 2) for v in src], "mic_xyz": [round(float(v), 2) for v in mic],
            "rt60_thiet_ke_s": round(design_s, 3)}


def rir_bins(rng: random.Random, per_bin: int, max_try: int = 600) -> dict:
    """{RT60 đích: [RIR]}: sinh rồi LỌC — nhận khi T20 đo trên RIR trong ±RT60_TOL quanh đích VÀ |T30 − T20| ≤ T30_T20_TOL·T20.
    RT60 thiết kế lấy 0,6–1,0 × đích vì phòng hộp hấp thụ đều cho T20 dài hơn Sabine ~1,2–1,5× (thử 08/11). Ghi wav vào RIR_DIR."""
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    os.makedirs(RIR_DIR, exist_ok=True)
    bins: dict = {}
    for t in RT60_S:
        got, tries = [], 0
        while len(got) < per_bin and tries < max_try:
            tries += 1
            r = sim_rir(nrng, t * nrng.uniform(0.6, 1.0))
            t20 = r and rt60_schroeder(r["h"])
            t30 = r and rt60_schroeder(r["h"], lo_db=-35)
            if not (t20 and t30) or abs(t20 - t) > RT60_TOL or abs(t30 - t20) > T30_T20_TOL * t20:
                continue
            path = os.path.join(RIR_DIR, f"rt{t}_{len(got):02d}.wav")
            sf.write(path, (r["h"] / np.max(np.abs(r["h"]))).astype(np.float32), SR, subtype="FLOAT")
            got.append({**{k: v for k, v in r.items() if k != "h"}, "path": path, "rt60": round(t20, 3), "rt60_t30": round(t30, 3)})
        if len(got) < per_bin:
            raise RuntimeError(f"RT60 {t} s: chỉ được {len(got)}/{per_bin} RIR sau {max_try} lần sinh")
        print(f"RIR RT60 {t} s: {per_bin} RIR sau {tries} lần sinh", flush=True)
        bins[t] = got
    return bins


# ----------------------------------------------------------------------------- dựng bộ
def script_version() -> dict:
    me = os.path.abspath(__file__)
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = None
    return {"script": rel(me), "script_sha256": sha256(me), "git_head": head}


class Writer:
    """Ghi wav + bản ghi manifest / cases / provenance; giữ thứ tự để tái lập (cùng seed → cùng bộ)."""

    def __init__(self, out: str, ver: dict):
        self.out, self.ver = out, ver
        self.manifest, self.cases, self.prov = [], [], []

    def add(self, truc: str, ca: str, v: float, y: np.ndarray, base: dict, lang: str, nguon: list, tham_so: dict, do_lai: dict):
        y, lvl = normalize(y)
        mong_doi, ly_do, bien = expected(truc, v)
        pre = "p1x" if BO == "p1_dieukien_v1" else "p1x" + BO.rsplit("_", 1)[-1]   # id không trùng giữa các bộ (kiem_bo)
        rid = f"{pre}_{truc}_{lang}_{len(self.manifest):04d}"
        path = os.path.join(self.out, truc, rid + ".wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, y.astype(np.float32), SR, subtype="PCM_16")
        person = base.get("person") or f"fleurs:{lang}:{base['file']}"
        k1 = {"ca": ca, "mong_doi": mong_doi, "ly_do": ly_do, "ca_bien": bien}
        self.manifest.append({"id": rid, "kind": f"p1_{truc}", "cfg": ca, "src": base["id"],
                              "wav": f"../v6/{BO}/{truc}/{rid}.wav",
                              "meta": {"lang": lang, "source": base["source"], "person": person, "speaker": person,
                                       "level1": ["01_nguyen_ban"], "level2": [], "edits": [], "k1_ca": k1}})
        self.cases.append({"file": f"{truc}/{rid}.wav", "ca": ca, "mong_doi": mong_doi, "ly_do": ly_do, "ca_bien": bien, "base": base["id"]})
        self.prov.append({"id": rid, "file": rel(path), "sha256": sha256(path), "ca": ca, "mong_doi": mong_doi, "ly_do": ly_do,
                          "goc": {"id": base["id"], "path": rel(base["wav"]), "sha256": sha256(base["wav"]), "giay_phep": base["license"]},
                          "nguon_tron": nguon, "tham_so": tham_so,
                          "do_lai": {**do_lai, "muc_rms_dbfs": round(lvl, 2), "dinh": round(float(np.max(np.abs(y))), 4),
                                     "do_dai_s": round(len(y) / SR, 2)}, **self.ver})

    def flush(self, rng: random.Random) -> None:
        os.makedirs(self.out, exist_ok=True)
        dump = lambda name, obj: json.dump(obj, open(os.path.join(self.out, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        dump("manifest.json", self.manifest)
        dump("cases.json", self.cases)
        with open(os.path.join(self.out, "provenance.jsonl"), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in self.prov)
        self._kiem_nghe(rng)
        self._the_bo()

    def _the_bo(self) -> None:
        """Thẻ bộ theo DU_LIEU.md 6.5: nguồn · giấy phép · ngày · ai làm · cách làm · số tệp theo loại · số người nói · đã kiểm chưa."""
        import collections
        import datetime
        by = collections.Counter((c["ca"], c["mong_doi"]) for c in self.cases)
        src = collections.Counter(n["nguon"].split("/")[0] for p in self.prov for n in p["nguon_tron"])
        L = [f"# Thẻ bộ `{BO}`", "",
             f"- **Ngày dựng:** {datetime.date.today().isoformat()} · **Ai làm:** Huấn (P1), mã do Claude Code (Claude Opus 5.5) viết theo thiết kế Huấn duyệt",
             f"- **Cách làm:** `{self.ver['script']}` (sha256 `{self.ver['script_sha256'][:16]}…`, git `{self.ver['git_head']}`, seed {self.ver['seed']}); "
             "lời nói sạch + biến đổi có tham số biết trước; đáp án suy từ luật Constraint (`expected()`), không gán tay",
             "- **Nguồn + giấy phép:** lời nói FLEURS vi_vn / en_us dev+test (CC BY 4.0) · ồn MUSAN noise (Public Domain) + DEMAND 16k "
             f"(CC BY 4.0, bỏ {', '.join(DEMAND_BO)}) · nhạc MUSAN music không lời (CC BY / BY-SA / PD, đã bỏ ND) · sách nói MUSAN librivox "
             "tiếng Anh (PD) · RIR tự sinh bằng pyroomacoustics (phòng hộp; nhãn theo T20 đo trên RIR, chỉ nhận RIR có |T30 − T20| ≤ 20 %). "
             "Chi tiết tải: `data/p1_raw/NGUON.jsonl`; từng tệp: `provenance.jsonl`",
             f"- **Lọc gốc:** chỉ câu FLEURS có SNR sẵn có ≥ {MIN_BASE_SNR_DB:g} dB (ước lượng khung, `native_snr`); trục snr **bù ồn** sẵn có "
             "(`snr_add_for`) để SNR thật ≈ đích",
             f"- **Số tệp:** {len(self.cases)} · **số người nói (theo person):** {len({m['meta']['person'] for m in self.manifest})} "
             "· nguồn trộn: " + ", ".join(f"{k} {v}" for k, v in sorted(src.items())),
             "- **Hạn chế:** FLEURS không có mã người nói (person = từng tệp → fold chưa đúng) · FLEURS vi dev/test chỉ có giọng nam "
             "· tệp `loai` gắn lang=en · **chỉ để thử script / thử K1, không nộp**; bản nộp chạy lại trên natvi/naten",
             "- **Đã kiểm:** *(điền sau khi chạy)* `kiem_bo --moi` · `lockbox check` · kiểm bằng tai (`KIEM_NGHE.md`)", "",
             "| ca | đáp án | số tệp |", "|---|---|---|"]
        L += [f"| {ca} | {md} | {k} |" for (ca, md), k in sorted(by.items())]
        open(os.path.join(self.out, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    def _kiem_nghe(self, rng: random.Random) -> None:
        cells: dict = {}
        for c in self.cases:
            cells.setdefault(c["ca"], []).append(c)
        L = [f"# Kiểm bằng tai: {BO}", "", "2 tệp ngẫu nhiên mỗi ô. Nghe rồi ghi **đạt / không** và ghi chú (nghe đúng mức ồn / nền / vang như đáp án không;",
             "có lỗi lạ: click, cắt đỉnh, im lặng…). Người kiểm ghi tên + ngày.", "",
             "| ca | tệp | đáp án | đạt? | ghi chú |", "|---|---|---|---|---|"]
        for ca in sorted(cells):
            for c in rng.sample(cells[ca], min(2, len(cells[ca]))):
                L.append(f"| {ca} | `{c['file']}` | {c['mong_doi']} {c['ly_do'] or ''} | | |")
        L += ["", "Người kiểm: ______  Ngày: ______"]
        open(os.path.join(self.out, "KIEM_NGHE.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def pick(pool: list, rng: random.Random) -> dict:
    return pool[rng.randrange(len(pool))]


def src_entry(p: dict, **kw) -> dict:
    return {"path": rel(p["path"]), "sha256": sha256(p["path"]), "giay_phep": p["license"], "nguon": p["src"], **kw}


def build_lang(w: Writer, fl_lang: str, n: int, rng: random.Random, noise: list, music: list, rirs: dict) -> None:
    """fl_lang = mã FLEURS (vi_vn / en_us); nhãn ghi lang 2 chữ (vi / en). noise = [kho MUSAN, kho DEMAND]: luân phiên
    theo (tệp gốc + mức) để hai nguồn chia đều, không bị kho lớn (MUSAN khoảng 690 tệp) lấn kho nhỏ (DEMAND 13 môi trường)."""
    lang = fl_lang[:2]
    for bi, b in enumerate(fleurs_bases(fl_lang, n, rng)):
        base = {"id": f"fleurs_{lang}_{b['split']}_{b['file'][:-4]}", "file": b["file"], "wav": b["wav"],
                "source": "fleurs", "license": "CC BY 4.0"}
        s, s0 = load(b["wav"]), b["snr_san_db"]
        w.add("goc", "goc", 0, s, base, lang, [], {"snr_san_goc_db": s0}, {})
        for si, snr in enumerate(SNR_DB):
            p = pick(noise[(bi + si) % len(noise)], rng)
            off = rng.randrange(1 << 30)
            src = load(p["path"])
            nz = fit_len(src, len(s), off)
            add = snr_add_for(snr, s0)                      # bù ồn sẵn có của gốc: SNR thật ≈ snr
            g = gain_for_snr(s, nz, add)
            do = 10 * np.log10(energy(s) / energy(g * nz))
            w.add("snr", f"snr.{snr}dB", snr, s + g * nz, base, lang, [src_entry(p, offset=off % (len(src) - XF))],
                  {"snr_dich_db": snr, "snr_san_goc_db": s0, "snr_them_db": round(add, 3), "g": g},
                  {"snr_them_db": round(float(do), 3), "snr_that_uoc_db": round(combine_snr(do, s0), 3)})
        for pct in NEN_PCT:
            p = pick(music, rng)
            off = rng.randrange(1 << 30)
            src = load(p["path"])
            m = fit_len(src, len(s), off)
            g = gain_for_ratio(s, m, pct / 100)
            do = 100 * energy(g * m) / (energy(s) + energy(g * m))
            w.add("nen", f"nen.{pct}pct", pct, s + g * m, base, lang, [src_entry(p, offset=off % (len(src) - XF))],
                  {"ti_le_nen_dich_pct": pct, "g": g}, {"ti_le_nen_pct": round(float(do), 3)})
        for t in RT60_S:
            if not rirs[t]:
                continue
            r = pick(rirs[t], rng)
            y = reverb(s, load(r["path"]))
            phong = {k: r[k] for k in ("phong_m", "hap_thu", "max_order", "nguon_xyz", "mic_xyz", "rt60_thiet_ke_s")}
            w.add("vang", f"vang.rt60_{t}s", r["rt60"], y, base, lang,
                  [src_entry({"path": r["path"], "license": "tự sinh (pyroomacoustics, MIT)", "src": "pyroomacoustics/ShoeBox"}, **phong)],
                  {"rt60_dich_s": t}, {"rt60_rir_schroeder_s": r["rt60"], "rt60_rir_t30_s": r["rt60_t30"]})


def build_loai(w: Writer, n: int, rng: random.Random, music: list, books: list) -> None:
    """Không phải ghi âm lời nói: nhạc thuần (MUSAN music) và sách nói (MUSAN speech/librivox), mỗi loại n tệp, cắt 10 s."""
    for kind, pool in (("nhac", music), ("sach_noi", books)):
        for p in rng.sample(pool, min(n, len(pool))):
            x = load(p["path"])
            for _ in range(20):                             # tránh đoạn có bài hết / khoảng lặng số 0 dài
                off = rng.randrange(max(1, len(x) - 10 * SR))
                y = fit_len(x, 10 * SR, off)
                if longest_zero_run(y) <= LOAI_MAX_ZERO_S:
                    break
            base = {"id": "musan_" + os.path.basename(p["path"])[:-4], "wav": p["path"], "source": "musan", "license": p["license"],
                    "person": f"musan:{os.path.basename(p['path'])[:-4]}"}
            w.add("loai", f"loai.{kind}", 0, y, base, "en", [], {"loai": kind, "offset": off}, {})


def build(n: int, seed: int, bo: str = BO) -> None:
    global BO, OUT, RIR_DIR
    BO, OUT = bo, os.path.join(ROOT, "data_labels", "v6", bo)
    if bo != "p1_dieukien_v1":                       # không ghi đè RIR mà provenance của v1 đang trỏ tới
        RIR_DIR = os.path.join(RAW, "rir_pra", bo)
    if os.path.exists(os.path.join(OUT, "manifest.json")):
        raise SystemExit(f"{rel(OUT)} đã có manifest.json: không ghi đè (dữ liệu không xoá; đổi --bo)")
    rng = random.Random(seed)
    noise = [musan_pool("noise", 5.0), demand_pool()]
    music = musan_pool("music", 10.0, no_vocals=True)
    # librivox: cột 3 ANNOTATIONS là ngôn ngữ → chỉ sách nói tiếng Anh (khớp lang="en"; ngôn ngữ khác để dành trục «ngôn ngữ»)
    books = [p for p in musan_pool("speech", 10.0) if "librivox" in p["path"] and p["ann"] == "english"]
    rirs = rir_bins(rng, per_bin=12)
    print(f"ồn MUSAN {len(noise[0])} + DEMAND {len(noise[1])} · nhạc {len(music)} · sách nói {len(books)} · RIR theo RT60 {[(t, len(v)) for t, v in rirs.items()]}", flush=True)
    w = Writer(OUT, {**script_version(), "seed": seed})
    for lang in ("vi_vn", "en_us"):
        build_lang(w, lang, n, rng, noise, music, rirs)
    build_loai(w, n, rng, music, books)
    w.flush(rng)
    print(f"{len(w.manifest)} tệp → {rel(OUT)}")


# ----------------------------------------------------------------------------- selftest (hàm thuần)
def selftest() -> None:
    rng = np.random.default_rng(0)
    s, n = rng.standard_normal(SR * 4), rng.standard_normal(SR * 3)
    nz = fit_len(n, len(s), 12345)
    assert len(nz) == len(s)
    for snr in SNR_DB:
        g = gain_for_snr(s, nz, snr)
        assert abs(10 * np.log10(energy(s) / energy(g * nz)) - snr) < 1e-9
    for pct in NEN_PCT:
        g = gain_for_ratio(s, nz, pct / 100)
        assert abs(energy(g * nz) / (energy(s) + energy(g * nz)) - pct / 100) < 1e-9
    for t in (0.3, 1.0, 1.8):                                   # RIR tổng hợp: nhiễu trắng tắt dần e^(−6,91 t / T60)
        tt = np.arange(int(SR * t * 1.5)) / SR
        h = rng.standard_normal(len(tt)) * np.exp(-6.9078 * tt / t)
        assert abs(rt60_schroeder(h) - t) < 0.05 * t, (t, rt60_schroeder(h))
        assert abs(rt60_schroeder(h, lo_db=-35) - t) < 0.05 * t
    for tg, b in ((20, 25), (15, 30), (5, 25)):                 # bù ồn: gộp(ồn thêm, ồn sẵn) = đích
        assert abs(combine_snr(snr_add_for(tg, b), b) - tg) < 1e-9
    try:
        snr_add_for(20, 18)
        raise AssertionError("gốc ồn hơn đích phải báo lỗi")
    except ValueError:
        pass
    x = np.sin(np.arange(SR) / 7.3)                             # nguồn lặp: chỗ nối không được nhảy hơn bước bình thường
    lp = loopable(x)
    tiled = fit_len(x, 3 * SR, 0)
    assert len(lp) == SR - XF and np.max(np.abs(np.diff(tiled))) <= 1.05 * np.max(np.abs(np.diff(x)))
    t = np.arange(SR * 4) / SR                                  # 2 s «nói» + nền thấp hơn 30 dB → SNR sẵn có ≈ 30 dB
    yb = np.where(t < 2, np.sin(2 * np.pi * 200 * t), 0) + 10 ** (-30 / 20) * rng.standard_normal(len(t)) / np.sqrt(2)
    assert abs(native_snr(yb) - 30) < 1.5, native_snr(yb)
    assert longest_zero_run(np.r_[np.ones(10), np.zeros(SR // 2), np.ones(10)]) == 0.5
    y, lvl = normalize(s)
    assert abs(lvl - LEVEL_DBFS) < 1e-6 and np.max(np.abs(y)) <= 0.99
    y, _ = normalize(np.r_[np.zeros(SR), [1.0], np.zeros(SR)])   # xung nhọn: phải hạ để khỏi cắt
    assert np.max(np.abs(y)) <= 0.99
    assert expected("snr", 8) == ("04", "ồn quá", False) and expected("snr", 10) == ("canh_bao", "tiếng ồn cao", True)
    assert expected("snr", 12)[0] == "canh_bao" and expected("snr", 20) == ("accept", None, False)
    assert expected("nen", 10) == ("accept", None, True) and expected("nen", 15)[0] == "04"
    assert expected("vang", 1.5) == ("canh_bao", "phòng vang", False) and expected("vang", 0.5) == ("accept", None, False)
    assert expected("vang", 1.05)[0] == "canh_bao" and expected("vang", 0.95) == ("accept", None, True)
    h = np.r_[np.zeros(50), 1.0, 0.5, 0.25]
    assert np.allclose(reverb(np.r_[1.0, np.zeros(9)], h)[:3], [1.0, 0.5, 0.25])
    assert license_ok("http://creativecommons.org/licenses/by/3.0/") and not license_ok("http://creativecommons.org/licenses/by-nc/3.0/")
    for t in ("CC BY-SA 4.0", "Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)", "License:Public Domain", "CC BY 3.0",
              "Wild Heart by Bryan Mathys is licensed under a Attribution License.", "All LibriVox recordings are\nin the Public Domain."):
        assert license_text_ok(t), t
    for t in ("Attribution-NoDerivs 3.0 Unported (CC BY-ND 3.0)", "CC BY-NC-SA 3.0", "Attribution-NonCommercial 4.0", "", None, "http://x.org/a"):
        assert not license_text_ok(t), t
    demo = ("CC BY 3.0\nhttp://incompetech.com\n=====\nmusic-rfm-0058\n\"Hustle\"\nmusic-rfm-0144\n=====\n"
            "music-fma-0062\nShowtime\nCC BY-ND 3.0\nhttp://fma\n=====\n")
    per, common = parse_musan_license(demo)
    assert common == "CC BY 3.0" and per == {"music-fma-0062": "CC BY-ND 3.0"}, (per, common)
    assert per.get("music-rfm-0058", common) == "CC BY 3.0" and not license_text_ok(per["music-fma-0062"])
    print("dung_bo selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")       # Windows: stdout chuyển hướng mặc định cp1252, không in được tiếng Việt
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("lenh", choices=["build", "selftest"])
    ap.add_argument("--n", type=int, default=12, help="số tệp gốc mỗi ngôn ngữ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bo", default="p1_dieukien_v1", help="tên bộ ra (data_labels/v6/<bo>); có sẵn manifest thì dừng, không ghi đè")
    a = ap.parse_args()
    selftest() if a.lenh == "selftest" else build(a.n, a.seed, a.bo)
