# -*- coding: utf-8 -*-
"""WP7b / B2 - CHAM DIEM MOI UNG VIEN giong may (I3: <cand>/<engine>/<job_id>_c<k>.wav) va CHON ung vien tot nhat.

Doc lap voi bo do (KHONG dung XLS-R / heads2 / TCN / hand.py de chon: chon bang dac trung cua bo do = vong tron).
Diem moi ung vien (giu het trong manifest, khong loc ngam):
  G1 doc dung    ASR (vaudit.asr: wav2vec2-vi-250h | wav2vec2-base-960h) -> CER so voi loi du dinh (vi GIU dau thanh),
                 so tu, cum tu dich co mat nguyen ven trong ban ASR (target_ok)
  G2 giong nguoi resemblyzer cosine voi tam cua <= 6 ban ghi THAT khac cua chinh nguoi do (bo tep host + tep mau giong);
                 z = (cos - trung vi LOO) / (1,4826 MAD) khi co >= 3 ban ghi
  G3 chat luong  DNSMOS P.835 (models/dnsmos/sig_bak_ovr.onnx, he so da thuc khong ca nhan hoa), cat dinh, lang trong > 0,7 s
  G4 ngu dieu    toc do am tiet (bo lang >= 0,15 s), CV khoang lang, F0 trung vi / bien do p10-p90 / do lech (ban cung, pyworld)
                 -> z ben vung theo chinh nguoi do (< 3 ban ghi: theo ca bo)
  thoi luong     ty le (thoi luong noi / so am tiet x thoi luong am tiet trung vi cua nguoi do) trong [0,6 ; 1,67]
Diem tong hop (THAP = tot), trong tung nhom ung vien cua mot (muc, phan): trung binh co trong so cua HANG (Borda):
  CER x1 · (-cos giong) x1,5 · |DNSMOS OVR - OVR cua host| x1 (khop host, khong toi da hoa: sach hon host cung la dau vet)
  · ngu dieu L2 x1,5 · |log ty le thoi luong| x0,5. May giong co san (it['clone']=False: MMS, Kokoro) bo G2 khoi cong
  va khoi diem tong hop (giong co dinh, so voi host vo nghia); muc lai TTS->VC (gen_chain) la nhai -> co G2. Chon: ung vien qua MOI cong co diem tong hop nho nhat; khong ai qua ->
  van ghi ung vien diem nho nhat voi pass=False + ly do (build_sets mac dinh bo muc do; khong bao gio noi cong ngam).
  gate_mode='record' (chi kiem thu voi may gia): chon theo diem tong hop, bo qua cong.
VC (vc_sets): CER so voi ASR cua CHINH doan nguon (giu noi dung), cos voi nguoi DICH va voi nguoi NGUON, DNSMOS, ty le do dai.
Ket qua: <work>/qa/rows/<engine>/<item>.json (diem moi ung vien, GHI DE moi lan cham) · <work>/qa/<engine>.jsonl (noi lai
  tu rows/, moi ung vien mot dong — khong trung khi --redo / qacal roi qa) · <work>/qa/selection.json (kem gates_tag) ·
  <work>/qa/summary.json (ti le qua < 20 % -> co). Lan chay sau KHONG cham lai am thanh muc da co nhung XET CONG LAI tu
  rows/ theo GATES hien tai (sua nguong sau buoc qacal ap dung cho moi muc). --limit = mau chia deu may x thao tac x muc.
Viec NANG (nap ASR + resemblyzer): chay qua run_serial.sh, thuong la viec trong hang doi.
CLI: python -m v6.tts_multi.qa --set ttsmulti_vi [--engine E] [--limit N] [--root R] [--gate-mode strict|record]"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Any, Callable

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6.tts_multi import plan as P  # noqa: E402

SR = 16000
DNSMOS_ONNX = os.path.join(BASE, "models", "dnsmos", "sig_bak_ovr.onnx")
DNSMOS_N = 144160
_P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])      # DNSMOS P.835 dnsmos_local.py, is_personalized_MOS=False
_P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
_P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
EN_ASR = "facebook/wav2vec2-base-960h"
VI_ASR = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
GATES = {"cer_max": {"vi": 0.15, "en": 0.10}, "cer_max_long": {"vi": 0.20, "en": 0.15}, "spk_z_min": -1.5,
         "spk_cos_min": 0.65, "spk_cos_min_self": 0.80, "sig_min": 3.0, "clip_max": 0.001, "sil_max": 0.7, "pros_z_max": 2.5,
         "dur_ratio": (0.6, 1.67)}
VC_GATES = {"cer_max": 0.3, "sig_min": 2.5, "len_ratio": (0.92, 1.08)}
WEIGHTS = {"cer": 1.0, "spk": 1.5, "ovr_gap": 1.0, "pros_l2": 1.5, "dur_dev": 0.5}
MAD_FLOOR = {"rate": 0.3, "pause_cv": 0.15, "f0_med_st": 1.0, "f0_range_st": 1.0, "f0_sd_st": 0.5}
N_GENUINE = 6


# ---------------- tien ich ----------------
def load16(path: str) -> np.ndarray:
    """Doc wav bat ky -> mono float32 16 kHz (resample_poly, khong phu thuoc soxr)."""
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def cer(ref: str, hyp: str) -> float:
    """CER Levenshtein bo khoang trong (giong vaudit.asr.cer, numpy theo hang)."""
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    prev = np.arange(len(h) + 1)
    hc = np.frombuffer(h.encode("utf-32-le"), "<u4")
    for i in range(1, len(r) + 1):
        cur = np.empty_like(prev)
        cur[0] = i
        sub = prev[:-1] + (hc != ord(r[i - 1]))
        for j in range(1, len(h) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, sub[j - 1])
        prev = cur
    return float(prev[-1]) / len(r)


def word_map(intended: list[str], hyp: list[str]) -> dict[int, int]:
    """Chi so tu du dinh -> chi so tu ASR (SequenceMatcher tren tu da chuan hoa)."""
    sm = SequenceMatcher(a=intended, b=hyp, autojunk=False)
    out = {}
    for blk in sm.get_matching_blocks():
        for t in range(blk.size):
            out[blk.a + t] = blk.b + t
    return out


def energy_bursts(y: np.ndarray, thr_db: float = -35.0, min_gap: float = 0.08) -> list[tuple[float, float]]:
    """Cac doan co nang luong (khung 10 ms > dinh + thr_db), gop khi ho < min_gap. Dung cho ASR 'oracle' cua may gia."""
    n = SR // 100
    k = len(y) // n
    if k == 0:
        return []
    e = 10 * np.log10((y[:k * n].reshape(k, n) ** 2).mean(1) + 1e-12)
    on = e > e.max() + thr_db
    runs, s = [], None
    for i, v in enumerate(on):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append([s / 100, i / 100])
            s = None
    if s is not None:
        runs.append([s / 100, k / 100])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged]


# ---------------- bo cham (cam duoc: that / gia) ----------------
class Scorers:
    """Bo ham cham: asr(y, lang, intended) -> (text, [[w, s, e]]) · embed(y) -> vec · dnsmos(y) -> (sig, bak, ovr) ·
    f0(y) -> Hz/khung 10 ms (0 = vo thanh). Mac dinh nap mo hinh that khi goi lan dau."""

    def __init__(self, asr: Callable | None = None, embed: Callable | None = None, dnsmos: Callable | None = None,
                 f0: Callable | None = None, asr_mode: str = "real") -> None:
        self._asr, self._embed, self._dnsmos, self._f0 = asr, embed, dnsmos, f0
        self.asr_mode = asr_mode
        self._state: dict[str, Any] = {}

    # --- ASR ---
    def asr(self, y: np.ndarray, lang: str, intended: str = "") -> tuple[str, list[list]]:
        if self._asr:
            return self._asr(y, lang, intended)
        import vaudit.asr as A
        name = EN_ASR if lang == "en" else VI_ASR
        if A.ASR_NAME != name:
            A.ASR_NAME = name
            A._state.clear()
        text, words = A.transcribe_words(np.asarray(y, np.float32), SR)
        return text, [[w.text, round(w.start, 3), round(w.end, 3)] for w in words]

    # --- giong nguoi ---
    def embed(self, y: np.ndarray) -> np.ndarray:
        if self._embed:
            return np.asarray(self._embed(y), float)
        if "enc" not in self._state:
            from resemblyzer import VoiceEncoder
            self._state["enc"] = VoiceEncoder("cpu", verbose=False)
        from resemblyzer import preprocess_wav
        z = preprocess_wav(np.asarray(y, np.float32), source_sr=SR)
        z = z if len(z) >= SR // 2 else np.asarray(y, np.float32)
        return np.asarray(self._state["enc"].embed_utterance(z), float)

    # --- DNSMOS ---
    def dnsmos(self, y: np.ndarray) -> tuple[float, float, float]:
        if self._dnsmos:
            return self._dnsmos(y)
        if "dns" not in self._state:
            import onnxruntime as ort
            self._state["dns"] = ort.InferenceSession(DNSMOS_ONNX, providers=["CPUExecutionProvider"])
        sess = self._state["dns"]
        y = np.asarray(y, np.float32)
        yy = np.tile(y, int(np.ceil(DNSMOS_N / max(1, len(y))))) if len(y) < DNSMOS_N else y
        outs = [sess.run(None, {"input_1": yy[s:s + DNSMOS_N][None]})[0][0]
                for s in range(0, len(yy) - DNSMOS_N + 1, SR)]
        raw = np.mean(np.array(outs), axis=0)
        return float(_P_SIG(raw[0])), float(_P_BAK(raw[1])), float(_P_OVR(raw[2]))

    # --- F0 ---
    def f0(self, y: np.ndarray) -> np.ndarray:
        if self._f0:
            return np.asarray(self._f0(y), float)
        try:
            site = os.path.join(BASE, ".tmp_v6", "site")
            if site not in sys.path:
                sys.path.append(site)
            import pyworld as pw
            x = np.asarray(y, np.float64)
            f, t = pw.dio(x, SR, f0_floor=60.0, f0_ceil=500.0, frame_period=10.0)
            return pw.stonemask(x, f, t, SR)
        except ImportError:
            import librosa
            f = librosa.yin(np.asarray(y, float), fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=160)
            e = librosa.feature.rms(y=np.asarray(y, float), frame_length=1024, hop_length=160)[0]
            return np.where(e > e.max() * 0.05, f, 0.0)


def oracle_asr(y: np.ndarray, lang: str, intended: str) -> tuple[str, list[list]]:
    """ASR 'biet truoc' cho may gia (am don): tra dung loi du dinh, moc tu chia deu tren cac cum nang luong."""
    words = P.norm_text(intended, lang).split()
    bursts = energy_bursts(y)
    if not words or not bursts:
        return "", []
    if len(bursts) == len(words):
        spans = bursts
    else:
        a, b = bursts[0][0], bursts[-1][1]
        step = (b - a) / len(words)
        spans = [(a + i * step, a + (i + 1) * step - 0.02) for i in range(len(words))]
    return " ".join(words), [[w, round(s, 3), round(e, 3)] for w, (s, e) in zip(words, spans)]


# ---------------- ngu dieu ----------------
def prosody(words: list[list], f0: np.ndarray, lang: str) -> dict[str, float | None]:
    """Toc do am tiet, CV khoang lang (>= 0,15 s), F0 (ban cung so voi 100 Hz): trung vi, p90-p10, do lech."""
    out: dict[str, float | None] = {"rate": None, "pause_cv": None, "f0_med_st": None, "f0_range_st": None,
                                    "f0_sd_st": None}
    if words and len(words) >= 2:
        gaps = [b[1] - a[2] for a, b in zip(words, words[1:])]
        pz = [g for g in gaps if g >= P.PAUSE_MIN]
        span = words[-1][2] - words[0][1]
        sp = max(1e-3, span - sum(pz))
        out["rate"] = round(P.n_syll([w[0] for w in words], lang) / sp, 3)
        out["pause_cv"] = round(float(np.std(pz) / np.mean(pz)), 3) if len(pz) >= 2 else None
    v = np.asarray(f0, float)
    v = v[v > 0]
    if len(v) >= 10:
        st = 12 * np.log2(v / 100.0)
        out["f0_med_st"] = round(float(np.median(st)), 3)
        out["f0_range_st"] = round(float(np.percentile(st, 90) - np.percentile(st, 10)), 3)
        out["f0_sd_st"] = round(float(np.std(st)), 3)
    return out


def robust_z(x: float | None, vals: list[float], prior: list[float], floor: float) -> float | None:
    """(x - trung vi) / (1,4826 MAD); < 3 gia tri cua nguoi do -> dung phan bo ca bo (prior)."""
    if x is None:
        return None
    ref = [v for v in vals if v is not None]
    if len(ref) < 3:
        ref = [v for v in prior if v is not None]
    if len(ref) < 3:
        return None
    med = float(np.median(ref))
    mad = max(1.4826 * float(np.median(np.abs(np.asarray(ref) - med))), floor)
    return round((x - med) / mad, 3)


# ---------------- tham chieu that cua tung nguoi ----------------
class PersonRefs:
    """Ban ghi THAT cua tung nguoi (<= 6, bo tep host + tep mau giong): tam giong, cos LOO, ngu dieu, DNSMOS host."""

    def __init__(self, by_person: dict[str, list[dict]], scorers: Scorers, lang: str, cache_path: str | None = None):
        self.by_person, self.S, self.lang = by_person, scorers, lang
        self.cache_path = cache_path
        self.cache: dict[str, Any] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as fh:
                self.cache = json.load(fh)

    def save(self) -> None:
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path + ".tmp", "w", encoding="utf-8") as fh:
                json.dump(self.cache, fh)
            os.replace(self.cache_path + ".tmp", self.cache_path)

    def file_info(self, h: dict) -> dict:
        key = h["file"]
        if key not in self.cache:
            y = load16(h["wav"])
            pr = prosody(h.get("words") or [], self.S.f0(y), self.lang)
            self.cache[key] = {"emb": [round(float(v), 5) for v in self.S.embed(y)], "pros": pr,
                               "ovr": round(self.S.dnsmos(y)[2], 3)}
        return self.cache[key]

    def genuine(self, person: str, exclude: set[str]) -> list[dict]:
        hs = [h for h in self.by_person.get(person, []) if h["file"] not in exclude]
        return [self.file_info(h) for h in sorted(hs, key=lambda h: h["file"])[:N_GENUINE]]

    def speaker(self, person: str, exclude: set[str], emb: np.ndarray, fallback_exclude: set[str] | None = None) -> dict:
        """cos voi tam ban ghi that cua nguoi (bo `exclude`). Khong con ban ghi nao va co fallback_exclude (thuong chi bo
        tep host) -> so voi CHINH tep mau giong, ghi spk_basis='mau_giong' (cos bi thoi phong: cung tep da dung de nhai)."""
        g, basis = self.genuine(person, exclude), "ban_ghi_khac"
        if not g and fallback_exclude is not None:
            g, basis = self.genuine(person, fallback_exclude), "mau_giong"
        if not g:
            return {"spk_cos": None, "spk_z": None, "n_genuine": 0, "spk_basis": None}
        E = np.array([x["emb"] for x in g], float)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        c = E.mean(0)
        e = emb / (np.linalg.norm(emb) + 1e-9)
        cos = float(e @ c / (np.linalg.norm(c) + 1e-9))
        z = None
        if len(E) >= 3:
            loo = []
            for i in range(len(E)):
                ci = np.delete(E, i, 0).mean(0)
                loo.append(float(E[i] @ ci / (np.linalg.norm(ci) + 1e-9)))
            med = float(np.median(loo))
            mad = max(1.4826 * float(np.median(np.abs(np.asarray(loo) - med))), 0.02)
            z = round((cos - med) / mad, 3)
        return {"spk_cos": round(cos, 4), "spk_z": z, "n_genuine": len(E), "spk_basis": basis}


# ---------------- cham + chon ----------------
def _target_ok(intended: list[str], hyp: list[str], lo: int | None, hi: int | None) -> bool | None:
    if lo is None or hi is None or hi < lo:
        return None
    tgt = intended[lo:hi + 1]
    n = len(tgt)
    return any(hyp[i:i + n] == tgt for i in range(len(hyp) - n + 1))


def score_candidate(wav: str, intended: str, lang: str, scorers: Scorers, refs: PersonRefs, person: str,
                    exclude: set[str], syll_dur: float, host_ovr: float | None, prior: dict[str, list],
                    target: tuple | None = None, host_file: str | None = None) -> dict:
    """Moi diem cua mot ung vien TTS (xem docstring module). host_file: cho phep roi ve so voi mau giong khi nguoi do
    khong con ban ghi that nao khac (spk_basis='mau_giong')."""
    y = load16(wav)
    text, words = scorers.asr(y, lang, intended)
    iw, hw = P.norm_text(intended, lang).split(), P.norm_text(text, lang).split()
    sig, bak, ovr = scorers.dnsmos(y)
    spk = refs.speaker(person, exclude, scorers.embed(y), {host_file} if host_file else None)
    pr = prosody(words, scorers.f0(y), lang)
    g = refs.genuine(person, exclude)
    pz = {k: robust_z(pr[k], [x["pros"].get(k) for x in g], prior.get(k, []), MAD_FLOOR[k]) for k in MAD_FLOOR}
    zs = [abs(v) for v in pz.values() if v is not None]
    gaps = [b[1] - a[2] for a, b in zip(words, words[1:])] if len(words) >= 2 else []
    span = (words[-1][2] - words[0][1]) if words else len(y) / SR
    expected = P.n_syll(iw, lang) * syll_dur
    return {"wav": wav, "seconds": round(len(y) / SR, 3), "asr_text": text, "asr_words": words, "asr_mode": scorers.asr_mode,
            "cer": round(cer(" ".join(iw), " ".join(hw)), 4), "n_words": len(iw), "n_words_asr": len(hw),
            "target_ok": _target_ok(iw, hw, *(target or (None, None))), **spk,
            "dnsmos_sig": round(sig, 3), "dnsmos_bak": round(bak, 3), "dnsmos_ovr": round(ovr, 3),
            "host_ovr": host_ovr, "prosody": pr, "prosody_z": pz,
            "pros_l2": round(float(np.sqrt(np.mean(np.square(zs)))), 3) if zs else None,
            "pros_zmax": round(max(zs), 3) if zs else None,
            "dur_ratio": round(span / expected, 3) if expected > 0 else None,
            "max_sil": round(max(gaps), 3) if gaps else 0.0,
            "clip_frac": round(float(np.mean(np.abs(y) >= 0.999)), 5)}


def gates(r: dict, lang: str, clone: bool = True) -> dict[str, bool]:
    """Cong chat luong. clone=False (may giong co san needs_vc: MMS, Kokoro; plan ghi it['clone']): KHONG co cong giong
    nguoi (g2_spk) vi giong co dinh khong the giong host — chi con doc dung / chat luong / ngu dieu / thoi luong.
    spk_basis='mau_giong' (so voi chinh tep mau giong, cos thoi phong): nguong tuyet doi cao hon (spk_cos_min_self)."""
    long_ = r["n_words"] > 15
    cmax = GATES["cer_max_long" if long_ else "cer_max"][lang]
    g = {"g1_cer": r["cer"] <= cmax, "g1_target": r["target_ok"] is not False}
    if clone:
        if r.get("spk_basis") == "mau_giong":
            g["g2_spk"] = r.get("spk_cos") is not None and r["spk_cos"] >= GATES["spk_cos_min_self"]
        else:
            g["g2_spk"] = (r["spk_z"] >= GATES["spk_z_min"]) if r.get("spk_z") is not None else \
                (r.get("spk_cos") is not None and r["spk_cos"] >= GATES["spk_cos_min"])
    g["g3_quality"] = r["dnsmos_sig"] >= GATES["sig_min"] and r["clip_frac"] <= GATES["clip_max"] \
        and r["max_sil"] <= GATES["sil_max"]
    g["g4_prosody"] = r["pros_zmax"] is None or r["pros_zmax"] <= GATES["pros_z_max"]
    lo, hi = GATES["dur_ratio"]
    g["g_dur"] = r["dur_ratio"] is None or lo <= r["dur_ratio"] <= hi
    return g


def _ranks(vals: list[float | None]) -> np.ndarray:
    """Hang 0..n-1 (dong hang lay trung binh); None / nan = hang cuoi."""
    v = np.array([np.inf if (x is None or (isinstance(x, float) and math.isnan(x))) else x for x in vals], float)
    order = v.argsort(kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v))
    for x in np.unique(v):
        m = v == x
        r[m] = r[m].mean()
    return r


def composite(rows: list[dict], clone: bool = True) -> list[float]:
    """Diem tong hop (thap = tot) cho mot nhom ung vien: trung binh co trong so cua hang (WEIGHTS).
    clone=False: bo thanh phan giong nguoi (giong co san, cos voi host khong co nghia)."""
    if not rows:
        return []
    feats = {"cer": [r.get("cer") for r in rows],
             "spk": [(-r["spk_cos"]) if r.get("spk_cos") is not None else None for r in rows],
             "ovr_gap": [abs(r["dnsmos_ovr"] - r["host_ovr"]) if r.get("host_ovr") is not None else None for r in rows],
             "pros_l2": [r.get("pros_l2") for r in rows],
             "dur_dev": [abs(math.log(r["dur_ratio"])) if r.get("dur_ratio") else None for r in rows]}
    w = {k: v for k, v in WEIGHTS.items() if clone or k != "spk"}
    tot = sum(w[k] * _ranks(feats[k]) for k in w)
    return [round(float(x) / sum(w.values()), 4) for x in tot]


def select(rows: list[dict], gate_mode: str = "strict") -> dict | None:
    """Ung vien tot nhat cua mot nhom (rows da co 'gates' + 'composite')."""
    if not rows:
        return None
    ok = [r for r in rows if all(r["gates"].values())]
    pool = rows if gate_mode == "record" or not ok else ok
    best = min(pool, key=lambda r: (r["composite"], r["wav"]))
    reasons = [] if all(best["gates"].values()) else [k for k, v in best["gates"].items() if not v]
    return {**best, "pass": bool(ok) and all(best["gates"].values()), "gate_mode": gate_mode,
            "n_cand": len(rows), "n_pass": len(ok), "fail": reasons}


def candidates(cand_dir: str, engine: str, job_id: str) -> list[str]:
    return sorted(glob.glob(os.path.join(cand_dir, engine, f"{job_id}_c*.wav")))


def _host_index(items: list[dict], lang: str) -> dict[str, list[dict]]:
    """Nguoi -> ban ghi that (host cua bo + nguon cung ngon ngu tu natural_v6 / VIVOS / LibriSpeech)."""
    by: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    persons = {it["host"]["person"] for it in items}
    try:
        pool = P.load_hosts(lang) + (P.load_vivos() if lang == "vi" else P.load_librispeech())
    except FileNotFoundError:
        pool = []
    for h in pool + [it["host"] for it in items]:
        if h["person"] in persons and h["file"] not in seen:
            seen.add(h["file"])
            by[h["person"]].append(h)
    return by


GATES_TAG = f"{zlib.crc32(json.dumps([GATES, VC_GATES], sort_keys=True).encode()):08x}"   # doi nguong -> doi tag


def stratified(items: list[dict], limit: int, key: Callable[[dict], Any]) -> list[dict]:
    """<= limit muc lay XOAY VONG qua cac nhom key(it) (vd thao tac x muc): mau cham thu qacal phu moi o, khong chi
    cac o dau ke hoach (plan xep may -> muc -> thao tac). Giu thu tu trong nhom."""
    groups: dict = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    ks, out, i = sorted(groups, key=str), [], 0
    while len(out) < limit and any(groups[k] for k in ks):
        g = groups[ks[i % len(ks)]]
        if g:
            out.append(g.pop(0))
        i += 1
    return out


def rows_path(D: dict, engine: str, item_id: str) -> str:
    return os.path.join(D["qa"], "rows", P.slug(engine), f"{item_id}.json")


def save_rows(D: dict, it: dict, rows_by_part: dict[str, list[dict]]) -> None:
    """Diem MOI ung vien cua mot muc (ghi de moi lan cham -> khong trung dong khi --redo / qacal roi qa)."""
    _atomic_json(rows_path(D, it["engine"], it["item_id"]), {"item_id": it["item_id"], "engine": it["engine"],
                                                             "parts": rows_by_part})


def load_rows(D: dict, it: dict) -> dict[str, list[dict]] | None:
    p = rows_path(D, it["engine"], it["item_id"])
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)["parts"]


def rebuild_log(D: dict, engine: str) -> int:
    """qa/<engine>.jsonl = noi lai tu qa/rows/<engine>/*.json (moi (muc, job, k) dung MOT dong). Tra so dong."""
    n = 0
    with open(os.path.join(D["qa"], f"{engine}.jsonl.tmp"), "w", encoding="utf-8") as out:
        for p in sorted(glob.glob(os.path.join(D["qa"], "rows", P.slug(engine), "*.json"))):
            with open(p, encoding="utf-8") as fh:
                for rows in json.load(fh)["parts"].values():
                    for r in rows:
                        out.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n += 1
    os.replace(os.path.join(D["qa"], f"{engine}.jsonl.tmp"), os.path.join(D["qa"], f"{engine}.jsonl"))
    return n


def sel_sig(v: dict) -> tuple:
    """Dau hieu ket qua chon cua mot muc (qua cong + ung vien + ly do truot tung phan) — dem so muc doi khi xet lai."""
    return bool(v.get("pass")), tuple(sorted((k, None if p is None else (p.get("wav"), tuple(p.get("fail") or [])))
                                             for k, p in v["parts"].items()))


def select_item(it: dict, rows_by_part: dict[str, list[dict]], lang: str, gate_mode: str) -> dict:
    """Cong (GATES hien tai) + chon ung vien cho tung phan tu diem da luu (composite da tinh luc cham)."""
    clone = bool(it.get("clone", True))
    out_parts, n = {}, 0
    for part, rows in rows_by_part.items():
        for r in rows:
            r["gates"], r["clone"] = gates(r, lang, clone), clone
        out_parts[str(part)] = select(rows, gate_mode)
        n += len(rows)
    ok = all(v is not None and v["pass"] for v in out_parts.values())
    return {"item_id": it["item_id"], "engine": it["engine"], "role": it["role"], "pass": ok, "n_cand": n,
            "parts": out_parts, "gates_tag": GATES_TAG}


def run_tts_qa(set_name: str, root: str | None = None, engine: str | None = None, limit: int | None = None,
               scorers: Scorers | None = None, gate_mode: str = "strict", by_person: dict | None = None,
               redo: bool = False) -> dict:
    """Cham moi ung vien cua bo ttsmulti_<lang>. Tiep tuc duoc: muc da co trong selection.json (moi phan co ung vien)
    KHONG cham lai am thanh tru khi redo, nhung DUOC XET CONG LAI tu diem da luu (qa/rows/) -> sua qa.GATES sau buoc
    qacal ap dung cho ca cac muc da cham thu. limit: mau cham thu chia deu (stratified) theo may x thao tac x muc.
    Muc chua co ung vien (chua sinh) duoc cham lai o lan sau."""
    D = P.set_dirs(set_name, root)
    items = P.read_plan(D["plan"])
    if engine:
        items = [it for it in items if it["engine"] == engine]
    items = stratified(items, limit, lambda it: (it["engine"], it["op"], it["tier"])) if limit else items
    lang = _lang_of(set_name, items)
    S = scorers or Scorers()
    refs = PersonRefs(by_person or _host_index(items, lang), S, lang, os.path.join(D["qa"], "person_refs.json"))
    stats = P.person_stats([h for hs in refs.by_person.values() for h in hs], lang)
    sel_path = os.path.join(D["qa"], "selection.json")
    sel = json.load(open(sel_path, encoding="utf-8")) if os.path.exists(sel_path) else {}
    os.makedirs(D["qa"], exist_ok=True)
    regated, engines = 0, set()
    for it in items:
        old = sel.get(it["item_id"])
        engines.add(it["engine"])
        if old and not redo and all(v is not None for v in old["parts"].values()):
            rows = load_rows(D, it)
            if rows is not None:
                new = select_item(it, rows, lang, gate_mode)
                regated += int(sel_sig(new) != sel_sig(old) or old.get("gates_tag") != GATES_TAG)
                sel[it["item_id"]] = new
            continue
        rows = _score_item(it, D, S, refs, stats, lang)
        save_rows(D, it, rows)
        sel[it["item_id"]] = select_item(it, rows, lang, gate_mode)
        refs.save()
        _atomic_json(sel_path, sel)
    _atomic_json(sel_path, sel)
    for e in engines:
        rebuild_log(D, e)
    summ = qa_summary(sel)
    summ.update(gates_tag=GATES_TAG, regated=regated, scored_now=len(items))
    _atomic_json(os.path.join(D["qa"], "summary.json"), summ)
    return summ


def _lang_of(set_name: str, items: list[dict]) -> str:
    lang = P.SETS.get(set_name, ("vi",))[0]
    return items[0]["lang"] if lang not in ("vi", "en") and items else lang


def _prior(refs: PersonRefs) -> dict[str, list]:
    """Phan bo ngu dieu cua MOI ban ghi that da cham (ca bo) — dung khi nguoi co < 3 ban ghi."""
    return {k: [v["pros"].get(k) for v in refs.cache.values() if isinstance(v, dict) and "pros" in v] for k in MAD_FLOOR}


def _score_item(it: dict, D: dict, S: Scorers, refs: PersonRefs, stats: dict, lang: str) -> dict[str, list[dict]]:
    """Diem moi ung vien moi phan (chua xet cong): {phan: [dong]}; composite tinh trong tung nhom phan."""
    h = it["host"]
    exclude = {h["file"]} | {r["src_file"] for r in it["refs"]}
    host_ovr = refs.file_info(h)["ovr"]
    refs.genuine(h["person"], exclude)
    prior = _prior(refs)
    syll = stats.get(h["person"], {}).get("syll_dur", 0.22)
    clone = bool(it.get("clone", True))           # False = may giong co san (needs_vc), khong cham cong giong nguoi
    out: dict[str, list[dict]] = {}
    for part in it["parts"]:
        rows = []
        target = (it["edit"]["carrier_lo"], it["edit"]["carrier_hi"]) if it["op"] != "may_ca_file" else None
        for jid in part["jobs"]:
            for w in candidates(D["cand"], it["engine"], jid):
                r = score_candidate(w, part["text"], lang, S, refs, h["person"], exclude, syll, host_ovr, prior, target,
                                    host_file=h["file"])
                r.update(job_id=jid, k=int(os.path.splitext(w)[0].rsplit("_c", 1)[1]), item_id=it["item_id"],
                         part=part["part"], params=_cand_params(w))
                rows.append(r)
        for r, c in zip(rows, composite(rows, clone)):
            r["composite"] = c
        out[str(part["part"])] = rows
    return out


def _cand_params(wav: str) -> dict:
    j = os.path.splitext(wav)[0] + ".json"
    if os.path.exists(j):
        with open(j, encoding="utf-8") as fh:
            d = json.load(fh)
        return {k: d.get(k) for k in ("engine", "sr", "seconds", "params")}
    return {}


def qa_summary(sel: dict) -> dict:
    """Ti le qua theo may; nhom 'mau_giong' (cos so voi chinh tep mau giong, thoi phong) bao RIENG; dem cong truot
    (cua ung vien duoc chon) de hieu chinh nguong sau lan sinh that dau tien (build_sets queue: buoc qacal_*)."""
    per: dict[str, dict] = defaultdict(lambda: {"items": 0, "pass": 0, "no_cand": 0, "mau_giong": 0,
                                                "pass_mau_giong": 0, "fail_gate": Counter(), "spk_z": [], "cer": []})
    for v in sel.values():
        p = per[v["engine"]]
        p["items"] += 1
        p["pass"] += int(v["pass"])
        parts = [x for x in v["parts"].values() if x is not None]
        p["no_cand"] += int(len(parts) < len(v["parts"]))
        if any(x.get("spk_basis") == "mau_giong" for x in parts):
            p["mau_giong"] += 1
            p["pass_mau_giong"] += int(v["pass"])
        for x in parts:
            p["fail_gate"].update(x.get("fail") or [])
            if x.get("spk_z") is not None:
                p["spk_z"].append(x["spk_z"])
            if x.get("cer") is not None:
                p["cer"].append(x["cer"])
    for p in per.values():
        p["pass_rate"] = round(p["pass"] / max(1, p["items"]), 3)
        n_o = p["items"] - p["mau_giong"]
        p["pass_rate_ban_ghi_khac"] = round((p["pass"] - p["pass_mau_giong"]) / n_o, 3) if n_o else None
        p["co_canh_bao"] = p["pass_rate"] < 0.2
        p["fail_gate"] = dict(p["fail_gate"])
        for k in ("spk_z", "cer"):
            v = p.pop(k)
            p[f"{k}_p10_50_90"] = [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])] if v else None
    return {"engines": dict(per), "n_items": len(sel)}


def _atomic_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(path + ".tmp", path)


# ---------------- VC ----------------
def score_vc_candidate(wav: str, src_wav: str, lang: str, S: Scorers, refs: PersonRefs, target_person: str,
                       source_person: str, exclude: set[str], src_text: str | None = None,
                       target_ref_files: set[str] | None = None) -> dict:
    """VC: CER(ASR doan doi giong, ASR doan nguon) · cos voi nguoi dich / nguoi nguon · DNSMOS · ty le do dai.
    cos_target tinh tren ban ghi cua nguoi dich KHONG gom tep mau giong da dua vao may (target_ref_files); nguoi dich
    khong con tep nao khac -> so voi chinh mau giong, target_basis='mau_giong'."""
    y, x = load16(wav), load16(src_wav)
    hyp, _ = S.asr(y, lang, src_text or "")
    ref_text = src_text if src_text is not None else S.asr(x, lang, "")[0]
    e = S.embed(y)
    tgt = refs.speaker(target_person, exclude | set(target_ref_files or ()), e, exclude)
    src = refs.speaker(source_person, exclude, e)
    sig, bak, ovr = S.dnsmos(y)
    return {"wav": wav, "seconds": round(len(y) / SR, 3), "len_ratio": round(len(y) / max(1, len(x)), 4),
            "asr_text": hyp, "asr_mode": S.asr_mode,
            "cer_vs_src": round(cer(P.norm_text(ref_text, lang), P.norm_text(hyp, lang)), 4),
            "cos_target": tgt["spk_cos"], "z_target": tgt["spk_z"], "target_basis": tgt["spk_basis"],
            "cos_source": src["spk_cos"],
            "dnsmos_sig": round(sig, 3), "dnsmos_bak": round(bak, 3), "dnsmos_ovr": round(ovr, 3),
            "clip_frac": round(float(np.mean(np.abs(y) >= 0.999)), 5)}


def vc_gates(r: dict) -> dict[str, bool]:
    ct, cs = r.get("cos_target"), r.get("cos_source")
    lo, hi = VC_GATES["len_ratio"]
    return {"g1_content": r["cer_vs_src"] <= VC_GATES["cer_max"], "g2_target": ct is not None and (cs is None or ct > cs),
            "g3_quality": r["dnsmos_sig"] >= VC_GATES["sig_min"] and r["clip_frac"] <= GATES["clip_max"],
            "g_len": lo <= r["len_ratio"] <= hi}


def vc_composite(rows: list[dict]) -> list[float]:
    feats = {"cer": [r["cer_vs_src"] for r in rows], "tgt": [-(r["cos_target"] or 0) for r in rows],
             "src": [r["cos_source"] for r in rows], "len": [abs(math.log(max(r["len_ratio"], 1e-3))) for r in rows]}
    w = {"cer": 1.0, "tgt": 1.5, "src": 1.0, "len": 0.5}
    tot = sum(w[k] * _ranks(v) for k, v in feats.items())
    return [round(float(x) / sum(w.values()), 4) for x in tot]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Cham + chon ung vien giong may (WP7b)")
    ap.add_argument("--set", required=True, choices=["ttsmulti_vi", "ttsmulti_en"])
    ap.add_argument("--engine", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", default=None)
    ap.add_argument("--gate-mode", choices=["strict", "record"], default="strict")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run_tts_qa(a.set, a.root, a.engine, a.limit, gate_mode=a.gate_mode, redo=a.redo),
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
