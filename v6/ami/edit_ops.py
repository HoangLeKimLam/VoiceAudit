# -*- coding: utf-8 -*-
"""Thao tac sua tren cua so AMI (ghi am that) — dap an trong meta.edits, 'mac ao' cho khop ban ghi.

  (a) ghep_thay_that  03_ghep_giong_that: thay 1-3 tu bang CUNG cac tu do CHINH nguoi do noi o cho khac trong CUNG
      cuoc hop (copy-move, cung micro -> cung phong/nhieu), cat o ranh gioi tu, crossfade 10 ms (vaudit.edits.replace_span)
  (b) nen_cuc_bo      02_nen_cuc_bo: doan 1-3 s qua ffmpeg afftdn (nr 24-40, nf -20, tn=1 — cong thuc cua bo 'tools'),
      tron vao nhu vaudit.edits.blend_process
  (c) may_thay_the    03_chen_giong_may: 2-4 tu doc lai bang viXTTS (tieng Anh, nhai giong chinh nguoi do) — TTS sinh o
      TIEN TRINH RIENG (tts_gen.py); o day chi cat + mac ao: RIR (vaudit.channel/make_room) theo RT60 uoc luong cua
      cua so + tieng on lay tu CAC KHOANG LANG cua chinh cua so, can muc nang luong theo doan goc.
KE HOACH (chon tu, chon nguoi cho, chon doan) chi dua tren chu thich -> giong nhau cho moi micro (cap doi Array1-01 /
Mix-Headset); diem cat duoc tinh chinh +-40 ms toi khe nang luong thap nhat tren chinh audio duoc sua.
Tu AMI lien nhau (93 % khoang ho = 0) nen 'ranh gioi tu' thuong la diem tiep giap hai tu."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf

from v6.ami.annot import BASE, RES, AWord, Meeting, is_clean_word, is_content_word, speech_runs

if BASE not in sys.path:
    sys.path.insert(0, BASE)
from vaudit.edits import blend_process, replace_span  # noqa: E402

SR = 16000
MARGIN = 1.0              # sua cach mep cua so >= 1 s (K1 bo 0,3 s dau/cuoi)
CUT_PAD = 0.15            # diem cat cach tu toi da 0,15 s khi khoang ho dai
SOLO_PAD = 0.3            # khong ai khac phat am trong +-0,3 s quanh doan sua / doan cho
DONOR_GAP = 5.0           # doan cho nam ngoai cua so >= 5 s
REFINE_S = 0.04           # tinh chinh diem cat +-40 ms
GHEP_MAX_N = 3
NEN_DUR = (1.0, 3.0)
NEN_NR = (24.0, 40.0)
NEN_NF = -20
MAY_N = (2, 4)
MAY_CTX_MAX = 12
REF_DUR = (5.0, 10.0)
RT60_CLIP = (0.2, 1.0)
RT60_P10_BIAS = 1.14      # r6 pilot_sim: blind RT60 p10 lech x1,14 (MAE 0,25 s) -> chia de hieu chinh
TMP_DIR = os.path.join(BASE, ".tmp_v6", "ami")
LEVEL1 = {"02": "02_ai_hau_ky", "03": "03_ghep_chen"}      # giong vaudit.simulate.LEVEL1


@dataclass(frozen=True)
class TWord:
    """Tu co .start/.end (giay) — tu cua TTS sau khi can moc bang ASR."""
    start: float
    end: float
    text: str


# ---------------- tien ich audio ----------------
def read_slice(path: str, t0: float, dur: float) -> np.ndarray:
    """Doc [t0, t0 + dur) giay (chi phan can, khong nap ca tep). Thieu cuoi tep -> dem 0."""
    a = max(0, int(round(t0 * SR)))
    n = int(round(dur * SR))
    y, sr = sf.read(path, start=a, stop=a + n, dtype="float64", always_2d=True)
    if sr != SR:
        raise ValueError(f"{path}: can {SR} Hz, tep co {sr} Hz")
    y = y.mean(1)
    return np.pad(y, (0, max(0, n - len(y))))


def peak_gain(y: np.ndarray, target: float = 0.9) -> float:
    p = float(np.max(np.abs(y))) if len(y) else 0.0
    return target / p if p > 1e-6 else 1.0


def to_pcm16(y: np.ndarray) -> tuple[np.ndarray, int]:
    """float -> int16 (lam tron, chan bien). Tra (mang int16, so mau bi chan)."""
    z = np.round(np.asarray(y, float) * 32767.0)
    n_clip = int((np.abs(z) > 32767).sum())
    return np.clip(z, -32768, 32767).astype("<i2"), n_clip


def pcm_crc(y16: np.ndarray) -> str:
    return f"{zlib.crc32(np.ascontiguousarray(y16).tobytes()) & 0xFFFFFFFF:08x}"


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2) + 1e-12))


def frame_db(y: np.ndarray, win: float = 0.03) -> np.ndarray:
    n = int(win * SR)
    k = len(y) // n
    if k == 0:
        return np.array([-120.0])
    return 10 * np.log10((np.asarray(y[:k * n], float).reshape(k, n) ** 2).mean(1) + 1e-12)


def refine_cut(y: np.ndarray, t: float, radius: float = REFINE_S, lo: float | None = None,
               hi: float | None = None) -> float:
    """Diem cat t (giay) -> tam cua so 10 ms nang luong thap nhat trong +-radius (buoc 1 ms), gioi han [lo, hi]."""
    i, r, h = int(round(t * SR)), int(radius * SR), int(0.005 * SR)
    a = max(h, i - r, int(np.ceil((lo if lo is not None else -1.0) * SR)))
    b = min(len(y) - h, i + r, int((hi if hi is not None else 1e7) * SR))
    if b <= a:
        return float(np.clip(t, 0, len(y) / SR))
    seg = np.asarray(y[a - h:b + h], float)
    c = np.r_[0.0, np.cumsum(seg ** 2)]
    pos = np.arange(a, b + 1, SR // 1000)
    e = c[pos - a + 2 * h] - c[pos - a]
    return float(pos[int(np.argmin(e + 1e-3 * (e.max() + 1e-12) * np.abs(pos - i) / max(r, 1)))] / SR)


def cut_before(words: list, k: int, pad: float = CUT_PAD) -> float:
    """Diem cat truoc tu k: giua khoang ho (tu k-1, tu k) nhung cach tu k <= pad (nhu vaudit.edits.gap_before)."""
    s = words[k].start
    return max((words[k - 1].end + s) / 2, s - pad) if k > 0 else s - pad


def cut_after(words: list, m: int, pad: float = CUT_PAD) -> float:
    e = words[m].end
    return min((e + words[m + 1].start) / 2, e + pad) if m + 1 < len(words) else e + pad


def level_labels(edits: list[dict]) -> tuple[list[str], list[str]]:
    l2 = sorted({e["level2"] for e in edits})
    return sorted({LEVEL1[x[:2]] for x in l2}) or ["01_nguyen_ban"], l2


def time_map(a: float, b: float, s: float, e: float) -> Callable[[float], float]:
    """Truc cu -> truc moi sau khi thay [a, b] bang doan nam o [s, e] (giay tuong doi cua so)."""
    delta = (e - s) - (b - a)

    def f(t: float) -> float:
        if t <= a:
            return t
        if t >= b:
            return t + delta
        return s + (t - a) * (e - s) / max(b - a, 1e-9)
    return f


def remap_window(win: dict, f: Callable[[float], float]) -> dict:
    """Ban sao thong tin cua so voi moi moc thoi gian (tu, luot noi, cho doi nguoi, vung chong) doi qua f."""
    r = lambda t: round(f(t), 3)
    return {**win, "words": [[r(s), r(e), t, p] for s, e, t, p in win["words"]],
            "turns": [[r(s), r(e), p] for s, e, p in win["turns"]],
            "turn_changes": [r(t) for t in win["turn_changes"]],
            "overlap_regions": [[r(s), r(e)] for s, e in win["overlap_regions"]]}


def rng_for(window_id: str, what: str, seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{window_id}|{what}|{seed}".encode()))


def _inside(meeting: Meeting, agent: str, t0: float, t1: float) -> list[int]:
    return [i for i, w in enumerate(meeting.words[agent]) if t0 + MARGIN <= w.start and w.end <= t1 - MARGIN]


def _contiguous(ws: list[AWord], max_gap: float = 0.3) -> bool:
    return all(q.start - p.end <= max_gap for p, q in zip(ws, ws[1:]))


def _span_ok(ws: list[AWord]) -> bool:
    """Cum tu 'doc tron': moi tu 0,08-1,0 s, ca cum <= 0,45 s/tu + 0,5 s, khong ho > 0,3 s (tu AMI hay bi keo dai)."""
    return (all(0.08 <= w.end - w.start <= 1.0 for w in ws) and ws[-1].end - ws[0].start <= 0.45 * len(ws) + 0.5
            and _contiguous(ws))


# ---------------- (a) ghep loi that: copy-move cung nguoi, cung cuoc hop ----------------
def _ghep_groups(meeting: Meeting, agent: str, max_n: int) -> list[tuple]:
    """Moi cum 1-max_n tu 'doc tron' cua mot nguoi, khong ai khac phat am +-0,3 s (tinh 1 lan / cuoc hop).
    Cum 1 tu phai la tu noi dung; cum >= 2 tu can it nhat mot tu noi dung."""
    key = ("ghep_groups", agent, max_n)
    if key not in meeting._cache:
        ws, out = meeting.words[agent], []
        for n in range(1, max_n + 1):
            for j in range(len(ws) - n + 1):
                g = ws[j:j + n]
                if not (_span_ok(g) and all(is_clean_word(w) for w in g)):
                    continue
                if not (is_content_word(g[0]) if n == 1 else any(is_content_word(w) for w in g)):
                    continue
                if meeting.solo(agent, g[0].start, g[-1].end, SOLO_PAD):
                    out.append((j, n, tuple(w.norm for w in g), g[0].start, g[-1].end))
        meeting._cache[key] = out
    return meeting._cache[key]


def plan_ghep(meeting: Meeting, win: dict, rng: np.random.Generator, max_n: int = GHEP_MAX_N) -> dict | None:
    """Chon cum 1-3 tu cua mot nguoi trong cua so + CUNG cum tu do nguoi do noi ngoai cua so (>= 5 s).
    Uu tien cum dai nhat; ty le do dai doan cho / doan goc 0,67-1,5."""
    t0, t1 = win["t0"], win["t1"]
    opts = []
    for agent in meeting.agent_list:
        ws = meeting.words[agent]
        groups = _ghep_groups(meeting, agent, max_n)
        index: dict[tuple, list[int]] = {}
        for j, n, key, s, e in groups:
            if e < t0 - DONOR_GAP or s > t1 + DONOR_GAP:
                index.setdefault(key, []).append(j)
        for i, n, key, s, e in groups:
            if not (t0 + MARGIN <= s and e <= t1 - MARGIN) or key not in index:
                continue
            a, b = cut_before(ws, i), cut_after(ws, i + n - 1)
            for j in index[key]:
                da, db = cut_before(ws, j), cut_after(ws, j + n - 1)
                ratio = (db - da) / max(b - a, 1e-9)
                if 0.67 <= ratio <= 1.5:
                    opts.append((n, agent, i, j, a, b, da, db, ratio))
    if not opts:
        return None
    nmax = max(o[0] for o in opts)
    pool = sorted((o for o in opts if o[0] == nmax), key=lambda o: (o[1], o[2], o[3]))
    n, agent, i, j, a, b, da, db, ratio = pool[int(rng.integers(len(pool)))]
    ws = meeting.words[agent]
    return {"type": "ghep", "agent": agent, "speaker": meeting.speaker(agent), "n_words": n, "host_i": i, "donor_j": j,
            "words": " ".join(w.text for w in ws[i:i + n]), "a": round(a, 3), "b": round(b, 3),
            "donor_a": round(da, 3), "donor_b": round(db, 3), "dur_ratio": round(ratio, 3), "n_options": len(opts)}


def apply_ghep(y: np.ndarray, win: dict, plan: dict, donor_audio: np.ndarray, gain: float) -> tuple[np.ndarray, dict, tuple]:
    """y = cua so (da nhan gain); donor_audio = audio THO [donor_a - 0,05, donor_b + 0,05] cung micro -> nhan cung gain.
    Tra (song moi, dap an edit, (a, b, s, e) giay tuong doi) de doi moc cac tu/luot noi phia sau."""
    t0 = win["t0"]
    a = refine_cut(y, plan["a"] - t0, hi=plan["a"] - t0 + 0.02)
    b = refine_cut(y, plan["b"] - t0, lo=plan["b"] - t0 - 0.02)
    d = np.asarray(donor_audio, float) * gain
    da = refine_cut(d, 0.05, hi=0.07)
    db = refine_cut(d, 0.05 + plan["donor_b"] - plan["donor_a"], lo=0.03 + plan["donor_b"] - plan["donor_a"])
    seg = d[int(round(da * SR)):int(round(db * SR))]
    g2 = float(np.clip(rms(y[int(a * SR):int(b * SR)]) / rms(seg), 0.5, 2.0))    # can muc (<= +-6 dB)
    out, (s, e) = replace_span(y, a, b, seg * g2, SR)
    edit = {"level2": "03_ghep_giong_that", "sub": "thay_that", "kind": "segment", "start": round(s, 4),
            "end": round(e, 4), "words": plan["words"], "speaker": plan["speaker"], "n_words": plan["n_words"],
            "donor_start": round(plan["donor_a"] - 0.05 + da, 3), "donor_end": round(plan["donor_a"] - 0.05 + db, 3),
            "donor_offset_s": round(plan["donor_a"] - plan["a"], 2), "level_db": round(20 * np.log10(g2), 2),
            "cut_host": [round(a, 4), round(b, 4)], "tool": "copy-move cung micro, crossfade 10 ms"}
    return out, edit, (a, b, s, e)


# ---------------- (b) khu on mot doan: ffmpeg afftdn ----------------
def plan_nen(meeting: Meeting, win: dict, rng: np.random.Generator, dur: tuple[float, float] = NEN_DUR) -> dict | None:
    """Doan 1-3 s gom cac tu lien tiep cua MOT nguoi (khong ai khac phat am +-0,3 s), cat o ranh gioi tu."""
    t0, t1 = win["t0"], win["t1"]
    opts = []
    for agent in meeting.agent_list:
        ws = meeting.words[agent]
        idx = _inside(meeting, agent, t0, t1)
        pos = set(idx)
        for i in idx:
            for j in range(i, len(ws)):
                if j not in pos or (j > i and ws[j].start - ws[j - 1].end > 0.5):
                    break
                a, b = cut_before(ws, i), cut_after(ws, j)
                if b - a > dur[1]:
                    break
                if b - a >= dur[0] and meeting.solo(agent, ws[i].start, ws[j].end, SOLO_PAD) \
                        and any(is_clean_word(w) for w in ws[i:j + 1]):
                    opts.append((agent, i, j, a, b))
    if not opts:
        return None
    agent, i, j, a, b = opts[int(rng.integers(len(opts)))]
    ws = meeting.words[agent]
    return {"type": "nen", "agent": agent, "speaker": meeting.speaker(agent), "i": i, "j": j,
            "words": " ".join(w.text for w in ws[i:j + 1]), "a": round(a, 3), "b": round(b, 3),
            "nr": round(float(rng.uniform(*NEN_NR)), 1), "nf": NEN_NF, "n_options": len(opts)}


def afftdn(y: np.ndarray, nr: float, nf: float = NEN_NF) -> np.ndarray:
    """ffmpeg afftdn qua tep tam (giong gen_test_suites.ff_filter_file), giu nguyen do dai."""
    os.makedirs(TMP_DIR, exist_ok=True)
    d = tempfile.mkdtemp(dir=TMP_DIR)
    a, b = os.path.join(d, "in.wav"), os.path.join(d, "out.wav")
    try:
        sf.write(a, np.asarray(y, np.float32), SR, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a, "-af", f"afftdn=nr={nr}:nf={nf}:tn=1", "-ar", str(SR),
                        "-ac", "1", "-c:a", "pcm_f32le", b], check=True)
        z, _ = sf.read(b, dtype="float64")
    finally:
        for p in (a, b):
            if os.path.exists(p):
                os.remove(p)
        os.rmdir(d)
    return np.pad(z, (0, max(0, len(y) - len(z))))[:len(y)]


def apply_nen(y: np.ndarray, win: dict, plan: dict,
              denoise: Callable[[np.ndarray, float, float], np.ndarray] = afftdn) -> tuple[np.ndarray, dict]:
    t0 = win["t0"]
    a = refine_cut(y, plan["a"] - t0, hi=plan["a"] - t0 + 0.02)
    b = refine_cut(y, plan["b"] - t0, lo=plan["b"] - t0 - 0.02)
    ia, ib = int(round(a * SR)), int(round(b * SR))
    seg = y[ia:ib]
    proc = denoise(seg, plan["nr"], plan["nf"])
    out = blend_process(y, ia / SR, ib / SR, proc, SR)
    drop = float(np.percentile(frame_db(seg, 0.02), 20) - np.percentile(frame_db(out[ia:ib], 0.02), 20))
    edit = {"level2": "02_nen_cuc_bo", "kind": "segment", "start": round(ia / SR, 4), "end": round(ib / SR, 4),
            "words": plan["words"], "speaker": plan["speaker"], "tool": "ffmpeg afftdn", "nr": plan["nr"],
            "nf": plan["nf"], "floor_drop_db": round(drop, 1)}
    return out, edit


# ---------------- (c) chen giong may: ke hoach + mac ao (TTS sinh o tts_gen.py) ----------------
def find_ref(meeting: Meeting, agent: str, avoid: tuple[float, float], rng: np.random.Generator,
             dur: tuple[float, float] = REF_DUR) -> dict | None:
    """Doan mau giong 5-10 s cua CHINH nguoi do ngoai cua so (>= 5 s): cac tu gop khi ho <= 0,5 s, khong ai khac
    phat am +-0,3 s. Chon ngau nhien trong 5 doan co do dai gan 8 s nhat."""
    runs = speech_runs([(w.start, w.end) for w in meeting.words[agent]], 0.5)
    good = []
    for s, e in runs:
        if e - s < dur[0] or not (e < avoid[0] - DONOR_GAP or s > avoid[1] + DONOR_GAP):
            continue
        e2 = min(e, s + dur[1])
        if meeting.solo(agent, s, e2, SOLO_PAD):
            good.append((abs((e2 - s) - 8.0), s, e2))
    if not good:
        return None
    good.sort()
    _, s, e = good[int(rng.integers(min(5, len(good))))]
    return {"meeting": meeting.name, "start": round(s - 0.05, 3), "end": round(e + 0.05, 3)}


def tts_text(ws: list[AWord]) -> str:
    t = " ".join(w.text for w in ws).strip()
    return (t[:1].upper() + t[1:] + ".") if t else t


def plan_may(meeting: Meeting, win: dict, rng: np.random.Generator, ref_mic: str = "Mix-Headset") -> dict | None:
    """2-4 tu lien tiep cua mot nguoi, nam giua mot cum 5-12 tu 'doc tron' cua nguoi do (TTS doc CA cum de ngu dieu
    tu nhien, roi cat dung cac tu dich). Kem doan mau giong 5-10 s cua nguoi do trong cung cuoc hop."""
    t0, t1 = win["t0"], win["t1"]
    opts = []
    for agent in meeting.agent_list:
        ws = meeting.words[agent]
        idx = _inside(meeting, agent, t0, t1)
        pos = set(idx)
        runs, cur = [], []
        for i in idx:                                    # cum tu sach, lien tiep (ho <= 0,3 s), tu 0,08-1,0 s
            w = ws[i]
            good = is_clean_word(w) and 0.08 <= w.end - w.start <= 1.0
            if good and cur and i - 1 in pos and w.start - ws[i - 1].end <= 0.3 and i - 1 == cur[-1]:
                cur.append(i)
            else:
                if len(cur) >= 5:
                    runs.append(cur)
                cur = [i] if good else []
        if len(cur) >= 5:
            runs.append(cur)
        for run in runs:
            for c0 in range(0, max(1, len(run) - MAY_CTX_MAX + 1), 4):
                ctx = run[c0:c0 + MAY_CTX_MAX]
                for n in range(MAY_N[0], MAY_N[1] + 1):
                    for k in range(1, len(ctx) - n):     # >= 1 tu truoc va sau trong cum
                        g = [ws[q] for q in ctx[k:k + n]]
                        if not (0.4 <= g[-1].end - g[0].start <= 2.5) or not any(is_content_word(w) for w in g):
                            continue
                        if meeting.solo(agent, g[0].start, g[-1].end, SOLO_PAD):
                            opts.append((agent, tuple(ctx), k, k + n - 1))
    if not opts:
        return None
    order = list(rng.permutation(len(opts)))
    for q in order[:20]:                                 # can doan mau giong cua dung nguoi do
        agent, ctx, k, m = opts[int(q)]
        ref = find_ref(meeting, agent, (t0, t1), rng)
        if ref is None:
            continue
        ws = meeting.words[agent]
        cw = [ws[i] for i in ctx]
        return {"type": "may", "agent": agent, "speaker": meeting.speaker(agent), "k": k, "m": m,
                "context": [[round(w.start, 3), round(w.end, 3), w.text] for w in cw], "text": tts_text(cw),
                "words": " ".join(w.text for w in cw[k:m + 1]), "a": round(cut_before(ws, ctx[k]), 3),
                "b": round(cut_after(ws, ctx[m]), 3), "ref": {**ref, "mic": ref_mic}, "n_options": len(opts)}
    return None


def plan_hash(plan: dict) -> str:
    return f"{zlib.crc32(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()) & 0xFFFFFFFF:08x}"


def estimate_rt60(y: np.ndarray, max_s: float = 120.0) -> tuple[float, str]:
    """RT60 mu (Ratnam ML + chon khung tat dan, .tmp_research/r6/k1lite.blind_rt60), p10 / 1,14, chan 0,2-1,0 s.
    Khong du khung tat dan / khong nap duoc -> 0,5 s ('mac_dinh')."""
    try:
        r6 = os.path.join(BASE, ".tmp_research", "r6")
        if r6 not in sys.path:
            sys.path.insert(0, r6)
        from k1lite import blind_rt60
        est = blind_rt60(np.asarray(y[:int(max_s * SR)], float))
        v = est.get(10)
        if v:
            return float(np.clip(v / RT60_P10_BIAS, *RT60_CLIP)), f"blind_p10(n={est.get('n')})"
    except Exception as e:                   # noqa: BLE001 — uoc luong phu, khong duoc lam hong ca buoc
        return 0.5, f"mac_dinh({type(e).__name__})"
    return 0.5, "mac_dinh(it_khung)"


def noise_bed(y: np.ndarray, win: dict, meeting: Meeting, n: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Tieng on nen cua CHINH cua so: noi cac khoang lang that (chu thich: khong ai phat am, cach >= 0,15 s) bang
    crossfade 10 ms, bat dau o vi tri ngau nhien. Qua it (< 0,5 s) -> cac khung 20 ms nang luong thap nhat (10 %)."""
    t0 = win["t0"]
    pieces = []
    for s, e in meeting.pauses(t0, win["t1"]):
        a, b = int(round((s - t0) * SR)), min(len(y), int(round((e - t0) * SR)))
        if b - a >= int(0.2 * SR):
            pieces.append(np.asarray(y[a:b], float))
    src = "khoang_lang"
    if sum(len(p) for p in pieces) < int(0.5 * SR):
        f = int(0.02 * SR)
        k = len(y) // f
        F = np.asarray(y[:k * f], float).reshape(k, f)
        sel = np.argsort((F ** 2).mean(1))[:max(1, k // 10)]
        pieces, src = [F[np.sort(sel)].reshape(-1)], "khung_nang_luong_thap"
    order = [pieces[int(i)] for i in rng.permutation(len(pieces))]
    from vaudit.edits import join
    bed = join(order, SR)[0] if len(order) > 1 else order[0]
    while len(bed) < 2 * n:
        bed = join([bed, bed], SR)[0]
    off = int(rng.integers(0, max(1, len(bed) - n)))
    return bed[off:off + n], src


def _room_ir(rt60: float, seed: int) -> np.ndarray:
    """RIR phong ao (test_simulate.make_room — cung ham vaudit.channel dung), cat tre truoc dinh truc tiep."""
    from test_simulate import make_room
    random.seed(seed)
    h = np.asarray(make_room(SR, rt60), float)
    p = int(np.argmax(np.abs(h)))
    h = h[max(0, p - int(0.002 * SR)):]
    return h / (np.max(np.abs(h)) + 1e-12)


def dress_tts(seg: np.ndarray, y: np.ndarray, a: float, b: float, win: dict, meeting: Meeting, rt60: float,
              rng: np.random.Generator, attacker: str = "careful") -> tuple[np.ndarray, dict]:
    """'careful': vang theo RT60 cua so + on nen cua chinh cua so, nang luong (tieng noi + on) = doan goc [a, b].
    'careless': chi can nang luong (khong vang, khong on) — nhu vaudit.channel.attacker_dress."""
    seg = np.asarray(seg, float)
    info = {"attacker": attacker}
    if attacker == "careful":
        wet = np.convolve(seg, _room_ir(rt60, int(rng.integers(2 ** 31))))[:len(seg)]
        bed, src = noise_bed(y, win, meeting, len(seg), rng)
        info.update(rt60_used=round(rt60, 3), noise_src=src)
    else:
        wet, bed = seg, np.zeros(len(seg))
    p_orig = rms(y[int(a * SR):int(b * SR)]) ** 2
    p_n = rms(bed) ** 2 if attacker == "careful" else 0.0
    p_s = max(p_orig - p_n, 0.25 * p_orig)
    out = wet * np.sqrt(p_s) / rms(wet) + bed
    info["snr_seg_db"] = round(10 * np.log10(p_s / max(p_n, 1e-12)), 1) if attacker == "careful" else None
    return out, info


def apply_may(y: np.ndarray, win: dict, plan: dict, tts_wav: np.ndarray, tts_words: list[list], tts_info: dict,
              meeting: Meeting, rt60: tuple[float, str], rng: np.random.Generator,
              attacker: str = "careful") -> tuple[np.ndarray, dict, tuple]:
    """Cat cac tu dich (k..m) khoi cau TTS (moc tu ASR tieng Anh), mac ao, thay vao [a, b] cua cua so."""
    t0 = win["t0"]
    tw = [TWord(float(s), float(e), t) for s, e, t in tts_words]
    k, m = plan["k"], plan["m"]
    ta = refine_cut(tts_wav, cut_before(tw, k, 0.1), hi=tw[k].start + 0.02)
    tb = refine_cut(tts_wav, cut_after(tw, m, 0.1), lo=tw[m].end - 0.02)
    seg = np.asarray(tts_wav, float)[int(round(ta * SR)):int(round(tb * SR))]
    a = refine_cut(y, plan["a"] - t0, hi=plan["a"] - t0 + 0.02)
    b = refine_cut(y, plan["b"] - t0, lo=plan["b"] - t0 - 0.02)
    dressed, info = dress_tts(seg, y, a, b, win, meeting, rt60[0], rng, attacker)
    out, (s, e) = replace_span(y, a, b, dressed, SR)
    edit = {"level2": "03_chen_giong_may", "sub": "thay_the", "kind": "segment", "start": round(s, 4),
            "end": round(e, 4), "words": plan["words"], "speaker": plan["speaker"], "tts": "viXTTS (TTS_LANG=en)",
            "tts_cer": tts_info.get("cer"), "tts_seed": tts_info.get("seed"), "ref": plan["ref"],
            "rt60_est": round(rt60[0], 3), "rt60_method": rt60[1], "cut_host": [round(a, 4), round(b, 4)], **info}
    return out, edit, (a, b, s, e)
