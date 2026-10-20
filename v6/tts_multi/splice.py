# -*- coding: utf-8 -*-
"""WP7b / B3 - GHEP GIONG MAY VAO BAN GHI THAT theo 3 muc che giau -> tep + dong manifest chuan (03_chen_giong_may).

Doan chen = cac tu dich cat ra tu cau mang (carrier) cua ung vien da chon (qa.selection), moc tu = ASR cua ung vien,
cat o khe nang luong thap nhat giua hai tu (+ diem qua 0). Ban goc cat o ranh gioi tu (moc tu hosts.jsonl).
  vung        KHONG khop gi: muc am may goc, cat cung (0 ms), diem cat giua khe tu (khong tinh chinh)
  trung_binh  khop muc am (RMS khung co tieng so voi doan bi thay / +-1 s lan can) + lam muot 5-20 ms
  ky          khop LTAS (pho trung binh 1/3 quang tam, chan +-10 dB) · loc thong thap ve bang thong cua host · lay mau
              lai ve 16 kHz cua host (resample_poly) · vang theo RT60 uoc luong mu cua host (v6.ami.edit_ops.estimate_rt60
              -> phong ao pyroomacoustics) · room tone tu CAC KHOANG LANG cua host tron duoi doan chen · am luong +-1 dB
              · noi equal-power 20-50 ms · roi (xac suat reencode_p) ma hoa lai CA tep (aac 64 / mp3 96 / opus 32)
  may_ca_file  noi tung phan (tung cau) bang khoang lang lay tu phan bo khoang lang cua chinh nguoi do; vung: lang so 0;
              trung_binh: khop muc + room tone o khoang lang; ky: LTAS + bang thong + vang + room tone duoi CA tep + am luong.
Nhan: meta.level2 ['03_chen_giong_may'], edits [{op may_thay|may_chen|may_ca_file, kind segment|global, start, end,
  removed, inserted, engine, role, ref_stratum, tier, qa{...}, ref_id, seed, fade_ms, ...}], muc_che_giau, thu_thuat.
Dinh (12/10): doan chen ha muc CHI doan ve <= SEG_PEAK truoc khi noi (edits.ha_dinh_doan_db); tep cuoi vuot FILE_PEAK
  -> thao tac doan: nen mem chi trong vung sua (edits.nen_dinh_trong_doan) — ngoai vung sua TRUNG TUNG MAU voi ban goc;
  thao tac ca tep / da ma hoa lai ca tep: chuan hoa ca tep (edits.chuan_hoa_ca_tep_db).
CLI: python -m v6.tts_multi.splice --set ttsmulti_vi [--root R] [--keep-failed] [--reencode-p 0.5] [--limit N]"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6 import common as C  # noqa: E402
from v6.tts_multi import plan as P  # noqa: E402
from v6.tts_multi import qa as Q  # noqa: E402

SR = 16000
REENCODE = (("aac", 64, 44100), ("mp3", 96, 44100), ("opus", 32, 48000))
BW_FULL = 7600.0
SEG_PEAK = 0.98           # dinh toi da cua DOAN chen / doan doi giong (ha muc CHI doan, truoc khi noi)
FILE_PEAK = 0.999         # dinh tep cuoi; thao tac doan: ngoai vung sua KHONG BAO GIO bi doi (12/10)
SOFT_KNEE = 0.9


# ---------------- dinh (khong cham vao ban goc ngoai vung sua) ----------------
def limit_peak(x: np.ndarray, peak: float = SEG_PEAK) -> tuple[np.ndarray, float]:
    """Ha muc ca doan neu dinh > peak (chi doan, khong cham tep host). Tra (doan, dB da ha, 0 neu khong ha)."""
    x = np.asarray(x, float)
    m = float(np.max(np.abs(x))) if len(x) else 0.0
    if m <= peak:
        return x, 0.0
    return x * (peak / m), round(float(20 * np.log10(peak / m)), 2)


def soft_limit(x: np.ndarray, knee: float = SOFT_KNEE, ceil: float = FILE_PEAK) -> np.ndarray:
    """Nen mem: |x| <= knee giu nguyen; tren knee -> knee + (ceil-knee)*tanh((|x|-knee)/(ceil-knee)) < ceil (lien tuc,
    dao ham 1 tai knee)."""
    x = np.asarray(x, float)
    a = np.abs(x)
    hi = a > knee
    out = x.copy()
    w = ceil - knee
    out[hi] = np.sign(x[hi]) * (knee + w * np.tanh((a[hi] - knee) / w))
    return out


def finalize_peak(w: np.ndarray, region: tuple[int, int] | None, whole_file: bool) -> tuple[np.ndarray, dict]:
    """Dinh tep cuoi > FILE_PEAK: thao tac CA TEP (global / da ma hoa lai ca tep) -> chuan hoa ca tep, ghi
    'chuan_hoa_ca_tep_db'; thao tac DOAN -> nen mem CHI trong vung sua [s0, s1) (gom ca vung lam muot), ghi
    'nen_dinh_trong_doan' = so mau bi nen; ngoai vung giu nguyen tung mau (ban goc PCM16 luon <= 1)."""
    w = np.asarray(w, np.float32)
    m = float(np.max(np.abs(w))) if len(w) else 0.0
    if m <= FILE_PEAK:
        return w, {}
    if whole_file or region is None:
        return (w * (FILE_PEAK / m)).astype(np.float32), {"chuan_hoa_ca_tep_db": round(float(20 * np.log10(FILE_PEAK / m)), 2)}
    s0, s1 = max(0, int(region[0])), min(len(w), int(region[1]))
    seg = w[s0:s1].astype(float)
    n = int(np.sum(np.abs(seg) > FILE_PEAK))
    if not n:
        return w, {}
    out = w.copy()
    out[s0:s1] = soft_limit(seg).astype(np.float32)
    return out, {"nen_dinh_trong_doan": n}


def write_pcm16(path: str, w: np.ndarray) -> None:
    """Ghi PCM16 bang so nguyen tu tinh (round(x * 32768)): mau ban goc doc tu PCM16 (= n / 32768) ghi lai DUNG n —
    libsndfile tu chuyen float -> PCM16 nhan 32767 (doc chia 32768) nen lech 1 LSB o mau lon."""
    ints = np.clip(np.round(np.asarray(w, np.float64) * 32768.0), -32768, 32767).astype(np.int16)
    sf.write(path, ints, SR, subtype="PCM_16")


def edit_region(edit: dict) -> tuple[int, int] | None:
    """Vung sua theo mau (start/end + lam muot hai ben) cho finalize_peak; global -> None."""
    if edit.get("kind") != "segment":
        return None
    f = float(edit.get("fade_ms") or edit.get("blend_ms") or 0) / 1000
    return int((edit["start"] - f) * SR) - 1, int((edit["end"] + f) * SR) + 1


# ---------------- tien ich tin hieu ----------------
def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, float))) + 1e-12))


def voiced_rms(x: np.ndarray) -> float:
    """RMS cua cac khung 20 ms to nhat (>= 30 % dinh theo dB, toi da 60 % khung) = muc tieng noi, khong tinh lang."""
    n = SR // 50
    k = len(x) // n
    if k < 2:
        return rms(x)
    e = (np.asarray(x[:k * n], float).reshape(k, n) ** 2).mean(1)
    top = np.sort(e)[::-1][:max(1, int(0.6 * k))]
    top = top[top >= top[0] * 1e-3]
    return float(np.sqrt(top.mean() + 1e-12))


def eq_join(parts: list[np.ndarray], fade_ms: float) -> tuple[np.ndarray, list[int]]:
    """Noi equal-power (sin/cos) — giu nang luong khi hai ben khong tuong quan. Tra (song, vi tri tam moi noi)."""
    nf = int(SR * fade_ms / 1000)
    out = np.asarray(parts[0], float)
    joints = []
    for p in parts[1:]:
        p = np.asarray(p, float)
        n = min(nf, len(out), len(p))
        if n > 1:
            t = np.linspace(0, np.pi / 2, n)
            mix = out[-n:] * np.cos(t) + p[:n] * np.sin(t)
            joints.append(len(out) - n // 2)
            out = np.concatenate([out[:-n], mix, p[n:]])
        else:
            joints.append(len(out))
            out = np.concatenate([out, p])
    return out, joints


def quiet_point(y: np.ndarray, t0: float, t1: float) -> float:
    """Khung 10 ms im nhat trong [t0, t1] roi chinh ve diem qua 0 gan nhat (+-5 ms)."""
    s0, s1 = max(0, int(t0 * SR)), min(len(y), int(t1 * SR))
    if s1 - s0 <= 320:
        i = (s0 + s1) // 2
    else:
        e = np.convolve(np.square(y[s0:s1]), np.ones(160) / 160, "same")
        i = s0 + int(np.argmin(e))
    lo = max(0, i - 80)
    zc = np.where(np.diff(np.signbit(y[lo:i + 80])))[0]
    return (lo + int(zc[np.argmin(np.abs(zc - (i - lo)))]) if len(zc) else i) / SR


def room_tone(y: np.ndarray, words: list | None, n: int, rng: random.Random) -> tuple[np.ndarray, str]:
    """Tieng nen cua chinh host: noi cac khoang lang giua tu (>= 0,15 s, bo 30 ms hai dau), lap toi n mau.
    Qua it (< 0,3 s) -> cac khung 20 ms im nhat (15 %)."""
    pieces, src = [], "khoang_lang"
    if words:
        gaps = [(0.0, words[0][1])] + [(a[2], b[1]) for a, b in zip(words, words[1:])] + [(words[-1][2], len(y) / SR)]
        for s, e in gaps:
            if e - s >= P.PAUSE_MIN:
                seg = y[int((s + 0.03) * SR):int((e - 0.03) * SR)]
                if len(seg) > 160:
                    pieces.append(np.asarray(seg, float))
    if sum(len(p) for p in pieces) < int(0.3 * SR):
        L = SR // 50
        k = len(y) // L
        F = np.asarray(y[:k * L], float).reshape(k, L)
        idx = np.argsort((F ** 2).mean(1))[:max(4, int(0.15 * k))]
        pieces, src = [F[i] for i in sorted(idx)], "khung_im_nhat"
    rng.shuffle(pieces)
    bed, _ = eq_join(pieces, 5.0)
    while len(bed) < n + SR // 10:
        bed, _ = eq_join([bed, bed[::-1]], 10.0)
    off = rng.randint(0, max(0, len(bed) - n))
    return bed[off:off + n], src


def host_bandwidth(y: np.ndarray) -> float:
    """Tan so tren hieu dung: diem cao nhat ma LTAS (Welch) con trong 50 dB duoi dinh vung tieng noi."""
    from scipy.signal import welch
    f, p = welch(np.asarray(y, float), SR, nperseg=1024)
    db = 10 * np.log10(p + 1e-20)
    ref = db[(f > 200) & (f < 3000)].max()
    ok = np.where(db > ref - 50)[0]
    return float(f[ok[-1]]) if len(ok) else SR / 2


def lowpass(y: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    return sosfiltfilt(butter(8, min(hz, SR / 2 - 100), "lowpass", fs=SR, output="sos"), np.asarray(y, float))


def third_octave_gain(src: np.ndarray, ref: np.ndarray, max_db: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Ty le LTAS ref / src lam muot theo dai 1/3 quang tam, chan +-max_db. Tra (tan so, dB)."""
    from scipy.signal import welch
    f, ps = welch(np.asarray(src, float), SR, nperseg=512)
    _, pr = welch(np.asarray(ref, float), SR, nperseg=512)
    g = 10 * np.log10((pr + 1e-14) / (ps + 1e-14))
    out = np.zeros_like(g)
    for i, fc in enumerate(f):
        lo, hi = max(fc, 50.0) / 2 ** (1 / 6), max(fc, 50.0) * 2 ** (1 / 6)
        m = (f >= lo) & (f <= hi)
        out[i] = np.median(g[m]) if m.any() else g[i]
    return f, np.clip(out, -max_db, max_db)


def ltas_match(seg: np.ndarray, ref: np.ndarray, max_db: float = 10.0) -> tuple[np.ndarray, float]:
    """Khop pho trung binh cua seg voi ref (STFT x duong cong 1/3 quang tam). Tra (song, |dB| trung binh da chinh)."""
    from scipy.signal import istft, stft
    if len(seg) < 1024 or len(ref) < 1024:
        return np.asarray(seg, float), 0.0
    _, g = third_octave_gain(seg, ref, max_db)
    _, _, S = stft(np.asarray(seg, float), SR, nperseg=512)
    _, z = istft(S * (10 ** (g / 20))[:, None], SR, nperseg=512)
    z = np.pad(z, (0, max(0, len(seg) - len(z))))[:len(seg)]
    return np.asarray(z, float), round(float(np.mean(np.abs(g))), 2)


def reverb(seg: np.ndarray, rt60: float, seed: int, wet: float = 1.0) -> np.ndarray:
    """Vang phong ao theo RT60 (v6.ami.edit_ops._room_ir: test_simulate.make_room, pyroomacoustics)."""
    from v6.ami.edit_ops import _room_ir
    h = _room_ir(float(rt60), int(seed))
    w = np.convolve(np.asarray(seg, float), h)[:len(seg)]
    w *= rms(seg) / (rms(w) + 1e-12)
    return (1 - wet) * np.asarray(seg, float) + wet * w


def rt60_of(y: np.ndarray) -> tuple[float, str]:
    try:
        from v6.ami.edit_ops import estimate_rt60
        return estimate_rt60(np.asarray(y, float))
    except Exception as e:                       # noqa: BLE001 — uoc luong phu, khong lam hong ca buoc
        return 0.5, f"mac_dinh({type(e).__name__})"


def reencode(y: np.ndarray, rng: random.Random) -> tuple[np.ndarray, str]:
    from v6.phone.channels import codec
    name, kbps, sr = REENCODE[rng.randrange(len(REENCODE))]
    return codec(np.asarray(y, np.float32), name, kbps, sr).astype(float), f"tay_{name}_{kbps}"


def trim_silence(y: np.ndarray, margin: float = 0.03) -> np.ndarray:
    b = Q.energy_bursts(np.asarray(y, np.float32))
    if not b:
        return np.asarray(y, float)
    return np.asarray(y[max(0, int((b[0][0] - margin) * SR)):int((b[-1][1] + margin) * SR)], float)


# ---------------- cat doan dich tu ung vien ----------------
def cut_target(y: np.ndarray, words: list[list], intended: str, lo: int, hi: int, lang: str) -> tuple[np.ndarray, dict]:
    """Cat cac tu dich [lo..hi] (theo cau mang du dinh) khoi ung vien: moc tu ASR + SequenceMatcher; thieu moc ->
    chia ty le am tiet tren khoang tieng noi. Diem cat = khe im nhat giua tu dich va tu ke ben (<= 0,12 s)."""
    iw = P.norm_text(intended, lang).split()
    hw = [P.norm_text(w[0], lang) for w in words]
    mp = Q.word_map(iw, hw)
    if lo in mp and hi in mp and mp[hi] >= mp[lo]:
        i, j = mp[lo], mp[hi]
        ws, we = words[i][1], words[j][2]
        prev_end = words[i - 1][2] if i > 0 else max(0.0, ws - 0.12)
        next_start = words[j + 1][1] if j + 1 < len(words) else min(len(y) / SR, we + 0.12)
        how = "asr_khop"
    else:
        b = Q.energy_bursts(np.asarray(y, np.float32))
        a0, a1 = (b[0][0], b[-1][1]) if b else (0.0, len(y) / SR)
        syl = [max(1, P.n_syll([w], lang)) for w in iw] or [1]
        cum = np.r_[0, np.cumsum(syl)] / sum(syl)
        ws, we = a0 + cum[lo] * (a1 - a0), a0 + cum[hi + 1] * (a1 - a0)
        prev_end, next_start, how = max(0.0, ws - 0.05), min(len(y) / SR, we + 0.05), "ty_le"
    t0 = quiet_point(y, max(prev_end, ws - 0.12), ws)
    t1 = quiet_point(y, we, min(next_start, we + 0.12))
    if t1 - t0 < 0.05:
        t0, t1 = ws, we
    return np.asarray(y[int(t0 * SR):int(t1 * SR)], float), {"cut_cand": [round(t0, 3), round(t1, 3)], "cut_how": how}


# ---------------- mac ao theo muc ----------------
def dress(seg: np.ndarray, y: np.ndarray, a: float, b: float, tier: str, host_words: list | None,
          rng: random.Random, env: dict) -> tuple[np.ndarray, float, dict, list[str]]:
    """Doan chen -> (doan da mac ao, do dai lam muot ms, thong so, thu thuat)."""
    seg = np.asarray(seg, float)
    if tier == "vung":
        return seg, 0.0, {"gain_db": 0.0}, ["khong_khop", "cat_cung"]
    ref = y[int(a * SR):int(b * SR)] if b - a > 0.15 else \
        np.r_[y[max(0, int((a - 1) * SR)):int(a * SR)], y[int(b * SR):int((b + 1) * SR)]]
    if tier == "trung_binh":
        g = float(np.clip(voiced_rms(ref) / (voiced_rms(seg) + 1e-9), 0.1, 10.0))
        fade = round(rng.uniform(5, 20), 1)
        return seg * g, fade, {"gain_db": round(20 * np.log10(g), 2)}, ["khop_am_luong", f"lam_muot_{fade:.0f}ms"]
    ctx = np.r_[y[max(0, int((a - 1) * SR)):int(a * SR)], y[int(b * SR):int((b + 1) * SR)]]
    seg, eq_db = ltas_match(seg, ctx)
    info: dict[str, Any] = {"eq_db_mean": eq_db, "bw_hz": round(env["bw"])}
    tricks = ["khop_ltas"]
    if env["bw"] < BW_FULL:
        seg = lowpass(seg, env["bw"])
        tricks.append(f"loc_bang_thong_{env['bw'] / 1000:.1f}k")
    seg = reverb(seg, env["rt60"], rng.randrange(2 ** 31), env["wet"])
    info.update(rt60=round(env["rt60"], 3), rt60_method=env["rt60_how"], wet=env["wet"])
    tricks.append("vang_theo_rt60")
    bed, src = room_tone(y, host_words, len(seg), rng)
    p_ref, p_n = voiced_rms(ctx) ** 2, rms(bed) ** 2
    p_s = max(p_ref - p_n, 0.25 * p_ref)
    off = float(np.clip(rng.gauss(0, 0.5), -1, 1))
    seg = (seg * np.sqrt(p_s) / (voiced_rms(seg) + 1e-12) + bed) * 10 ** (off / 20)
    info.update(snr_db=round(10 * np.log10(p_s / max(p_n, 1e-12)), 1), noise_src=src, loud_off_db=round(off, 2))
    fade = round(rng.uniform(20, 50), 1)
    return seg, fade, info, tricks + ["room_tone_duoi_doan", "am_luong_1dB", f"noi_equal_power_{fade:.0f}ms"]


def host_env(y: np.ndarray) -> dict:
    rt, how = rt60_of(y)
    known = not how.startswith("mac_dinh")
    return {"bw": host_bandwidth(y), "rt60": rt if known else 0.3, "rt60_how": how, "wet": 1.0 if known else 0.5}


# ---------------- dung tep ----------------
def _best(sel: dict, part: int) -> dict | None:
    return sel["parts"].get(str(part)) if sel else None


def build_segment(it: dict, sel: dict, y: np.ndarray, rng: random.Random) -> tuple[np.ndarray, dict, str, list]:
    """may_thay / may_chen -> (song, edit, loi mong doi, thu thuat)."""
    e, h, tier, lang = it["edit"], it["host"], it["tier"], it["lang"]
    best = _best(sel, 0)
    cand = Q.load16(best["wav"])
    seg, cinfo = cut_target(cand, best["asr_words"], it["parts"][0]["text"], e["carrier_lo"], e["carrier_hi"], lang)
    ws = h["words"]
    if it["op"] == "may_chen":
        j = e["k"]
        a = b = (quiet_point(y, ws[j - 1][2], ws[j][1]) if tier != "vung" else (ws[j - 1][2] + ws[j][1]) / 2)
    else:
        a, b = e["a"], e["b"]
        if tier != "vung":
            k, m = e["k"], e["m"]
            a = quiet_point(y, ws[k - 1][2] if k > 0 else a - 0.05, ws[k][1])
            b = quiet_point(y, ws[m][2], ws[m + 1][1] if m + 1 < len(ws) else b + 0.05)
    env = host_env(y) if tier == "ky" else {}
    seg, fade, info, tricks = dress(seg, y, a, b, tier, ws, rng, env)
    seg, lim_db = limit_peak(seg)                     # dinh doan chen <= SEG_PEAK: khong phai chuan hoa ca tep
    if lim_db:
        info["ha_dinh_doan_db"] = lim_db
    out, joints = eq_join([y[:int(a * SR)], seg, y[int(b * SR):]], fade)
    s, t = joints[0] / SR, joints[1] / SR
    words = [w[0] for w in ws]
    k, m = e["k"], e["m"]
    expected = " ".join(words[:k] + e["inserted"].split() + words[m + 1:])
    edit = {"level2": "03_chen_giong_may", "sub": "chen_them" if it["op"] == "may_chen" else "thay_the",
            "op": it["op"], "kind": "segment", "start": round(s, 4), "end": round(t, 4), "removed": e["removed"],
            "inserted": e["inserted"], "n_words": e["n_words"], "n_syll": e["n_syll"], "fade_ms": fade,
            "cut_host": [round(a, 4), round(b, 4)], **cinfo, **info}
    return out, edit, expected, tricks


def build_whole(it: dict, sel: dict, y: np.ndarray, stats: dict, rng: random.Random) -> tuple[np.ndarray, dict, str, list]:
    """may_ca_file: noi tung phan + khoang lang theo phan bo cua nguoi do, mac ao theo muc."""
    h, tier = it["host"], it["tier"]
    ws = h.get("words")
    pauses = [p for p in stats.get(h["person"], {}).get("pauses", []) if p <= 1.5] or [0.3, 0.4, 0.5]
    segs = [trim_silence(Q.load16(_best(sel, p["part"])["wav"])) for p in it["parts"]]
    lead = ws[0][1] if ws else 0.3
    tail = (len(y) / SR - ws[-1][2]) if ws else 0.3
    gaps = [lead] + [rng.choice(pauses) if tier != "vung" else 0.3 for _ in segs[1:]] + [tail]
    env = host_env(y) if tier == "ky" else {}
    tricks, info = [], {}
    ref_v = voiced_rms(y)
    if tier == "ky":
        speech, info["eq_db_mean"] = ltas_match(np.concatenate(segs), y)
        if env["bw"] < BW_FULL:
            speech = lowpass(speech, env["bw"])
            tricks.append(f"loc_bang_thong_{env['bw'] / 1000:.1f}k")
        speech = reverb(speech, env["rt60"], rng.randrange(2 ** 31), env["wet"])
        cut = np.cumsum([0] + [len(s) for s in segs])
        segs = [speech[cut[i]:cut[i + 1]] for i in range(len(segs))]
        tricks += ["khop_ltas", "vang_theo_rt60"]
        info.update(rt60=round(env["rt60"], 3), rt60_method=env["rt60_how"], bw_hz=round(env["bw"]))
    if tier != "vung":
        segs = [s * ref_v / (voiced_rms(s) + 1e-9) for s in segs]
        tricks.append("khop_am_luong")
    pieces = []
    for i, g in enumerate(gaps):
        n = int(max(0.05, g) * SR)
        pieces.append(room_tone(y, ws, n, rng)[0] if tier == "trung_binh" else np.zeros(n))
        if i < len(segs):
            pieces.append(segs[i])
    out = np.concatenate(pieces)
    if tier == "ky":
        bed, src = room_tone(y, ws, len(out), rng)
        off = float(np.clip(rng.gauss(0, 0.5), -1, 1))
        out = (out + bed) * 10 ** (off / 20)
        tricks += ["room_tone_ca_tep", "am_luong_1dB"]
        info.update(noise_src=src, loud_off_db=round(off, 2))
    elif tier == "trung_binh":
        tricks.append("room_tone_khoang_lang")
    else:
        tricks.append("lang_so_0")
    expected = " ".join(p["text"] for p in it["parts"])
    edit = {"level2": "03_chen_giong_may", "op": "may_ca_file", "kind": "global", "start": 0.0,
            "end": round(len(out) / SR, 4), "inserted": expected, "n_parts": len(segs),
            "pauses_s": [round(g, 3) for g in gaps], **info}
    return out, edit, expected, tricks


def qa_brief(sel: dict) -> dict:
    """Diem QA gon cho manifest (moi phan: ung vien da chon)."""
    keep = ("job_id", "k", "cer", "target_ok", "spk_cos", "spk_z", "n_genuine", "spk_basis", "dnsmos_sig", "dnsmos_bak",
            "dnsmos_ovr", "host_ovr", "pros_l2", "pros_zmax", "dur_ratio", "max_sil", "composite", "pass", "n_cand",
            "n_pass", "fail", "asr_mode", "gate_mode", "params")
    return {"pass": sel["pass"], "parts": {p: {k: v.get(k) for k in keep} for p, v in sel["parts"].items() if v}}


def kind_of(op: str, tier: str) -> str:
    return op if tier == "trung_binh" else f"{op}_{tier}"


def make_entry(it: dict, wav: np.ndarray, edit: dict, expected: str, tricks: list, sel: dict, out_dir: str,
               extra_meta: dict | None = None) -> dict:
    """Ghi wav (PCM16, chan dinh) + dong manifest chuan."""
    kind = kind_of(it["op"], it["tier"])
    rid = f"{it['prefix']}_{it['item_id']}"
    d = os.path.join(out_dir, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, rid + ".wav")
    whole = edit.get("kind") != "segment" or any(str(t).startswith("tay_") for t in tricks)
    w, pk = finalize_peak(wav, edit_region(edit), whole)
    write_pcm16(path, w)
    h = it["host"]
    best0 = _best(sel, 0) or {}
    edit = {**edit, **pk, "engine": it["engine"], "role": it["role"], "ref_stratum": it["ref_stratum"], "tier": it["tier"],
            "ref_id": [r["ref_id"] for r in it["refs"]], "ref_text_src": [r["text_src"] for r in it["refs"]],
            "ref_row_gap": [r.get("row_gap") for r in it["refs"]],
            "ref_ltas_dist_db": [r.get("ltas_dist_db") for r in it["refs"]],
            "seed": it["seed"], "clone": it.get("clone", True), "qa": qa_brief(sel),
            "gen_chain": it.get("gen_chain") or [it["engine"]], "cand_params": best0.get("params")}
    meta = {"host_file": h["file"], "speaker": h["cluster"], "person": h["person"], "source": h["source"],
            "orig_text": h.get("ref") if h.get("extra") else h.get("text"), "expected_text": expected,
            "level1": ["03_ghep_chen"], "level2": ["03_chen_giong_may"], "edits": [edit], "lang": it["lang"],
            "kenh": "goc", "fold": int(it["fold"]), "person_fold": int(it["person_fold"]), "muc_che_giau": it["tier"],
            "thu_thuat": tricks, "so_thao_tac": 1, "engine": it["engine"], "engine_role": it["role"],
            "engine_family": it.get("family"),
            "ref_stratum": it["ref_stratum"], "ref_cung_phien": it["ref_stratum"] in P.SAME_STRATA,
            "paired_real": it.get("paired_real"), "wp": "WP7b", "qa_pass": bool(sel.get("pass")),
            **({"host_reuse_of": it["host_reuse_of"]} if it.get("host_reuse_of") else {}),
            **({"host_info": h["info"]} if h.get("info") else {}), **host_extra_meta(h), **(extra_meta or {})}
    try:
        meta["group"] = C.group_of(meta, it["set"])            # I2 (WP5) — tuy chon
    except Exception:                                          # noqa: BLE001 — chua co group_of thi bo qua
        pass
    mark_lockbox(rid, meta, it["set"])
    return {"id": rid, "kind": kind, "cfg": "goc", "src": rid, "wav": C.rel_v4(path), "meta": meta}


def host_extra_meta(h: dict) -> dict:
    """Truong phu cua kho host mo rong (WP5 hosts_v2: group, lockbox_v2, fold_v2, role) -> meta (role -> host_role)."""
    out = {("host_role" if k == "role" else k): h[k] for k in P.HOST_KEEP if h.get(k) is not None}
    if h.get("pool"):
        out["host_pool"] = h["pool"]
    return out


def mark_lockbox(rid: str, meta: dict, set_name: str) -> None:
    """Nguoi / cuoc hop thuoc LOCKBOX (C.lockbox_reason: vd english_speaker, ami TS3003) -> meta.lockbox=True + ly do,
    de cac buoc sau (xlsr_frames, huan luyen) loai ra dung quy uoc chung."""
    r = C.lockbox_reason({"id": rid, "meta": meta}, set_name)
    if r:
        meta["lockbox"], meta["lockbox_reason"] = True, r


def save_manifest(path: str, man: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False, indent=0)
    for attempt in range(20):                                  # Windows: tep vua ghi co the bi giu tam
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)
    raise PermissionError(f"khong ghi duoc {path}")


def load_manifest(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_splice(set_name: str, root: str | None = None, keep_failed: bool = False, reencode_p: float = 0.5,
               limit: int | None = None, seed: int = 2026) -> dict:
    """Moi muc co ung vien da chon -> tep + manifest (tiep tuc duoc: id da co bo qua). Tra tom tat."""
    D = P.set_dirs(set_name, root)
    items = P.read_plan(D["plan"])
    sel_all = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
    man = load_manifest(D["manifest"])
    have = {m["id"] for m in man}
    lang = items[0]["lang"] if items else "vi"
    hosts_all = [it["host"] for it in items]
    stats = P.person_stats(hosts_all, lang)
    done, skipped = 0, {"chua_qa": 0, "khong_qua_cong": 0, "da_co": 0, "loi": 0}
    for it in items[:limit] if limit else items:
        rid = f"{it['prefix']}_{it['item_id']}"
        sel = sel_all.get(it["item_id"])
        if rid in have:
            skipped["da_co"] += 1
            continue
        if not sel or any(v is None for v in sel["parts"].values()):
            skipped["chua_qa"] += 1
            continue
        if not sel["pass"] and not keep_failed:
            skipped["khong_qua_cong"] += 1
            continue
        rng = random.Random(f"{seed}|{it['item_id']}")
        try:
            y = Q.load16(it["host"]["wav"]).astype(float)
            if it["op"] == "may_ca_file":
                out, edit, expected, tricks = build_whole(it, sel, y, stats, rng)
            else:
                out, edit, expected, tricks = build_segment(it, sel, y, rng)
            if it["tier"] == "ky" and rng.random() < reencode_p:
                out, how = reencode(out, rng)
                tricks.append(how)
        except Exception as ex:                                # noqa: BLE001 — mot muc hong khong dung ca buoc
            print(f"  bo {it['item_id']}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
            skipped["loi"] += 1
            continue
        man.append(make_entry(it, out, edit, expected, tricks, sel, D["set"]))
        done += 1
        if done % 25 == 0:
            save_manifest(D["manifest"], man)
    save_manifest(D["manifest"], man)
    return {"set": set_name, "new": done, "skipped": skipped, "n_manifest": len(man)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Ghep giong may vao ban ghi that (WP7b)")
    ap.add_argument("--set", required=True, choices=["ttsmulti_vi", "ttsmulti_en"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--reencode-p", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    print(json.dumps(run_splice(a.set, a.root, a.keep_failed, a.reencode_p, a.limit), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
