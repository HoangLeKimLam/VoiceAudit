# -*- coding: utf-8 -*-
"""BO LOC NOI DUNG NHE cho tep goc da qua probe - filter_spec C1-C5, C7, C8 (audit1_scrape), khong torch tru dem nguoi noi.

Giai ma ffmpeg -> 16 kHz mono f32 theo khoi 60 s (RAM < 300 MB ca voi tep 6 h), dac trung khung 100 ms (v6.scrape.frames):
  C1 ty le loi noi (webrtcvad muc 2 + nang luong > nen + 6 dB): >= 0,55 giu; 0,40-0,55 cat lay doan day loi; < 0,40 bo;
     bo moi doan khong loi > 120 s (gio nghi, man hinh cho)
  C2 bang thong: tan so cao nhat con > dinh pho - 50 dB (khung loi) -> bw_p90 (roll-off 99 % cua filter_spec chi ghi
     lai la roll99_p90: pilot cho 2-5 kHz tren Opus bang rong vi nang luong tieng noi don o tan thap); >= 7 kHz 'wide', 4-7 kHz 'narrow' (K1 tra 04 duoi 6 kHz), < 4 kHz bo;
     nhay chat luong livestream (>= 2 kHz, > 30 s) -> tach tep tai do
  C3 nhac / am sac trong khoang lang (dinh pho on dinh >= 0,5 s, phang < 0,05) + am sac keo dai >= 1 s (hat):
     > 15 % tep hoac mot doan giua tep > 60 s -> bo; con lai cat bo doan nhac +- 2 s
  C4 so nguoi noi: resemblyzer d-vector tren <= 40 cua so 3 s (1 / phut) + gom cum cosine (--no-spk: bo qua, de None);
     dominant_share >= 0,7 'lecture', con lai 'qa_meeting'; > 10 nguoi hoac dominant < 0,15 -> bo
     (nguong cum SPK_DIST CHUA hieu chinh tren 20 bai giang mot nguoi nhu filter_spec yeu cau)
  C5 vet cat (chinh cho 'chua bien tap'; was_live KHONG du): a lang so trong loi noi, b buoc nen on, c nhay am luong,
     d click -> cuts_per_hour; sach: 0 vet a va <= 1 / h; 1-4 / h: tach tai vet cat; > 4 / h: bo
  C7 muc: ty le cat dinh <= 0,1 %; do to xap xi (khong K-weighting, ghi 'lufs_approx') -35..-12; LRA xap xi
  C8 cuoc goi hoi nghi: metadata nhac Zoom / Meet / Teams / truc tuyen hoac (bw_p90 7-8,2 kHz va nen < -75 dBFS) -> conf_call
Ra: meta/filtered.jsonl (do + keep_spans + pool + quyet dinh) + frames/<id>.npz (mat na loi noi / nhac cho segment)
  + frames/<id>.env8.npy (duong bao 100 ms o 8 pha, buoc 12,5 ms, cho van tay trung lap qc.dedup).

CLI: python -m v6.scrape.filter [--limit N] [--no-spk] [--max-min 400]"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import frames as FR  # noqa: E402
from v6.scrape import util as U  # noqa: E402

SR = FR.SR
BLOCK_S = 60
SPEECH_KEEP, SPEECH_TRIM = 0.55, 0.40
LONG_GAP_S = 120.0
BW_WIDE, BW_MIN = 7000.0, 4000.0
MUSIC_MAX, MUSIC_RUN_MAX_S, MUSIC_MARGIN_S = 0.15, 60.0, 2.0
CUTS_CLEAN, CUTS_SPLIT = 1.0, 4.0
CUT_MARGIN_S = 1.0
MIN_SPAN_S = 60.0
CLIP_MAX = 0.001
LUFS_RANGE = (-35.0, -12.0)
SPK_DIST = 0.25            # khoang cach cosine gom cum resemblyzer (uoc, CHUA hieu chinh)
SPK_MAX, DOM_MIN, DOM_LECTURE = 10, 0.15, 0.7
MAX_WINDOWS = 40
CONF_RE = re.compile(r"zoom|google meet|\bmeet\b|microsoft teams|\bteams\b|webex|trực tuyến|online|webinar", re.I)


def decode_blocks(path: str, block_s: int = BLOCK_S, stream: int = 0):
    """ffmpeg -> f32le 16 kHz mono, tra tung khoi numpy (khong nap ca tep)."""
    cmd = [U.ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-i", path, "-map", f"0:a:{stream}", "-ac", "1",
           "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = block_s * SR * 4
    complete = False
    try:
        while True:
            buf = p.stdout.read(nbytes)
            if not buf:
                complete = True
                break
            yield np.frombuffer(buf[:len(buf) // 4 * 4], np.float32)
    finally:
        if not complete:            # nguoi goi dung som (--max-min): giet ffmpeg, khong coi la loi
            p.kill()
        p.stdout.close()
        err = p.stderr.read().decode("utf-8", "replace")
        p.wait()
        if complete and p.returncode != 0:
            raise RuntimeError(f"ffmpeg loi ({p.returncode}): {err[:200]}")


# ---------------- dem nguoi noi ----------------
def count_speakers(windows: list[tuple[float, np.ndarray]], use_model: bool = True) -> dict[str, Any]:
    """resemblyzer (CPU) tren cac cua so 3 s; khong co / tat -> MFCC 40 chieu (proxy, kem tin cay hon)."""
    if len(windows) < 3:
        return {"n_spk_est": None, "dominant_share": None, "spk_method": None, "n_windows": len(windows)}
    sel = windows if len(windows) <= MAX_WINDOWS else [windows[i] for i in
                                                        np.linspace(0, len(windows) - 1, MAX_WINDOWS).astype(int)]
    emb, method = None, None
    if use_model:
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            enc = VoiceEncoder("cpu", verbose=False)
            emb = np.stack([enc.embed_utterance(preprocess_wav(w, SR)) for _, w in sel])
            method = "resemblyzer"
        except Exception as ex:  # noqa: BLE001  (thieu mo hinh / cua so qua ngan sau khi cat lang)
            print(f"    resemblyzer loi -> MFCC: {str(ex)[:80]}", flush=True)
    if emb is None:
        import librosa
        feats = []
        for _, w in sel:
            m = librosa.feature.mfcc(y=w.astype(np.float32), sr=SR, n_mfcc=20)
            m = (m - m.mean(1, keepdims=True))
            d = librosa.feature.delta(m)
            feats.append(np.concatenate([m.std(1), d.std(1)]))
        emb, method = np.stack(feats), "mfcc_proxy"
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=SPK_DIST if method == "resemblyzer" else 0.5,
                                  metric="cosine", linkage="average").fit_predict(emb)
    cnt = np.bincount(lab)
    min_sz = max(2, int(0.05 * len(lab)))
    return {"n_spk_est": int((cnt >= min_sz).sum()) or 1, "dominant_share": round(float(cnt.max() / len(lab)), 3),
            "spk_method": method, "n_windows": len(lab)}


# ---------------- khoang giu ----------------
def subtract(spans: list[tuple[float, float]], cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Tru cac doan cuts khoi spans (giay)."""
    out = list(spans)
    for a, b in sorted(cuts):
        nxt = []
        for s, e in out:
            if b <= s or a >= e:
                nxt.append((s, e))
                continue
            if a > s:
                nxt.append((s, a))
            if b < e:
                nxt.append((b, e))
        out = nxt
    return out


def keep_spans(dur: float, speech: np.ndarray, music: np.ndarray, cuts: list[dict], switches: list[dict],
               trim: bool, edge_s: float, min_span: float = MIN_SPAN_S) -> list[list[float]]:
    fr = FR.FR_S
    rm = [(a * fr, b * fr) for a, b in FR.runs(~speech) if (b - a) * fr > LONG_GAP_S]
    rm += [(a * fr - MUSIC_MARGIN_S, b * fr + MUSIC_MARGIN_S) for a, b in FR.runs(music)]
    rm += [(e.get("lo", e["t"]) - CUT_MARGIN_S, e.get("hi", e["t"]) + CUT_MARGIN_S) for e in cuts]
    rm += [(e["t"] - CUT_MARGIN_S, e["t"] + CUT_MARGIN_S) for e in switches]
    if trim:   # ty le loi noi 0,40-0,55: bo khoi 60 s thua loi
        w = int(60 / fr)
        for i in range(0, len(speech), w):
            if speech[i:i + w].mean() < SPEECH_KEEP:
                rm.append((i * fr, min(len(speech), i + w) * fr))
    sp = subtract([(edge_s, dur - edge_s)], rm)
    return [[round(a, 2), round(b, 2)] for a, b in sp if b - a >= min_span]


def _levels(F: dict, speech: np.ndarray) -> dict[str, float | None]:
    if not speech.any():
        return {"lufs_approx": None, "lra_approx": None}
    ms = 10 ** (F["rms_db"][speech] / 10)
    lufs = float(10 * np.log10(ms.mean() + 1e-12) - 0.691)
    k = 30
    st = [10 * np.log10(ms[i:i + k].mean() + 1e-12) for i in range(0, len(ms) - k + 1, 10)]
    lra = float(np.percentile(st, 95) - np.percentile(st, 10)) if len(st) > 3 else None
    return {"lufs_approx": round(lufs, 1), "lra_approx": round(lra, 1) if lra is not None else None}


def decide(F: dict, speech: np.ndarray, nf: np.ndarray, spk: Mapping[str, Any], meta: Mapping[str, Any],
           edge_s: float = 1.0) -> dict[str, Any]:
    """Quyet dinh tren mang khung (ham thuan, kiem thu duoc)."""
    dur = float(F["dur_s"])
    hours = max(dur / 3600, 1e-9)
    reasons, flags = [], []
    sr_ = float(speech.mean()) if len(speech) else 0.0
    trim = False
    if sr_ < SPEECH_TRIM:
        reasons.append("low_speech")
    elif sr_ < SPEECH_KEEP:
        trim = True
        flags.append("trim_speech_dense")
    edge = F.get("edge", F["roll"])
    bw = float(np.percentile(edge[speech], 90)) if speech.any() else 0.0
    roll99 = float(np.percentile(F["roll"][speech], 90)) if speech.any() else 0.0
    gated, ds_rate = FR.gated_floor(F)
    if gated:
        flags.append("gated_floor")
    if bw < BW_MIN:
        reasons.append("bw_lt_4k")
    music = FR.music_mask(F, speech)
    mratio = float(music.mean()) if len(music) else 0.0
    mruns = [(a * FR.FR_S, b * FR.FR_S) for a, b in FR.runs(music)]
    mid_long = [r for r in mruns if r[1] - r[0] > MUSIC_RUN_MAX_S and r[0] > 120 and r[1] < dur - 120]
    if mratio > MUSIC_MAX:
        reasons.append("music_ratio")
    if mid_long:
        reasons.append("music_run_mid")
    cuts = FR.detect_cuts(F, speech, edge_s=max(edge_s, 1.0))
    sw = FR.bw_switches(F, speech)
    n_a = sum(e["type"] == "a_digital_silence" for e in cuts)
    cph = len(cuts) / hours
    if n_a == 0 and cph <= CUTS_CLEAN:
        cut_mode = "clean"
    elif cph <= CUTS_SPLIT:
        cut_mode = "split"
    else:
        cut_mode = "reject"
        reasons.append("too_many_cuts")
    nspk, dom = spk.get("n_spk_est"), spk.get("dominant_share")
    if nspk is not None and (nspk > SPK_MAX or (dom is not None and dom < DOM_MIN)):
        reasons.append("speakers")
    clip = float(F["clip"].sum() / max(1, len(F["clip"]) * FR.HOP))
    if clip > CLIP_MAX:
        reasons.append("clipping")
    lv = _levels(F, speech)
    if lv["lufs_approx"] is not None and not LUFS_RANGE[0] <= lv["lufs_approx"] <= LUFS_RANGE[1]:
        reasons.append("level")
    nf_g = float(np.median(nf)) if len(nf) else None
    text = f"{meta.get('title') or ''} {meta.get('description') or ''}"
    conf = bool(CONF_RE.search(text)) or (6900 <= bw <= 8200 and (gated or (nf_g is not None and nf_g < -75)))
    spans = keep_spans(dur, speech, music, cuts, sw, trim, edge_s) if not reasons else []
    if not reasons and not spans:
        reasons.append("no_keep_span")
    pool = None
    if not reasons:
        kind = "lecture" if dom is None or dom >= DOM_LECTURE else "qa_meeting"
        pool = f"{'wide' if bw >= BW_WIDE else 'narrow'}_{kind}"
    return {"ok": not reasons, "reasons": reasons, "flags": flags, "pool": pool, "dur_s": round(dur, 1),
            "speech_ratio": round(sr_, 3), "bw_p90": round(bw), "roll99_p90": round(roll99),
            "digital_silence_per_hour": ds_rate, "music_ratio": round(mratio, 4),
            "music_runs_gt60_mid": len(mid_long), "cuts": cuts, "cuts_per_hour": round(cph, 2), "cut_mode": cut_mode,
            "n_digital_silence": n_a, "bw_switches": sw, "clip_ratio": round(clip, 6), **lv,
            "noise_floor_db": round(nf_g, 1) if nf_g is not None else None, "conf_call": conf, **spk,
            "keep_spans": spans, "keep_hours": round(sum(b - a for a, b in spans) / 3600, 3)}


def frames_path(rid: str) -> str:
    return os.path.join(U.FRAMES, U.slug(rid, 120) + ".npz")


def env8_path(rid: str) -> str:
    """Duong bao 100 ms o 8 pha (buoc 12,5 ms) cho van tay trung lap qc.dedup (review 12/10)."""
    return os.path.join(U.FRAMES, U.slug(rid, 120) + ".env8.npy")


def analyse_file(path: str, meta: Mapping[str, Any], use_spk: bool = True, max_min: float | None = None) -> dict:
    acc = FR.FrameAccumulator(use_vad=True, keep_windows=True)
    fine = FR.FineEnvelope()
    t0 = time.time()
    for k, blk in enumerate(decode_blocks(path)):
        acc.feed(blk)
        fine.feed(blk)
        if max_min and (k + 1) * BLOCK_S >= max_min * 60:
            break
    F = acc.finish()
    speech, nf = FR.speech_mask(F)
    spk = count_speakers(acc.windows, use_model=True) if use_spk else {"n_spk_est": None, "dominant_share": None,
                                                                         "spk_method": None}
    edge = 20.0 if meta.get("source") in ("ccc", "fosdem") else 1.0   # bo 20 s dau / cuoi (nhac hieu, gioi thieu)
    res = decide(F, speech, nf, spk, meta, edge_s=edge)
    os.makedirs(U.FRAMES, exist_ok=True)
    np.savez_compressed(frames_path(meta["id"]), speech=speech.astype(np.uint8),
                        music=FR.music_mask(F, speech).astype(np.uint8), rms_db=F["rms_db"].astype(np.float16),
                        roll=F["roll"].astype(np.float16))
    FR.save_env8(env8_path(meta["id"]), fine.finish())
    res["analyse_s"] = round(time.time() - t0, 1)
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--no-spk", action="store_true")
    ap.add_argument("--max-min", type=float, default=None, help="chi phan tich N phut dau (thu)")
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--redo", action="store_true", help="phan tich lai ca tep da co (dong moi ghi de)")
    a = ap.parse_args()
    probed = [r for r in U.latest_by(U.read_jsonl(U.PROBED)).values() if r.get("ok")]
    fetched = U.latest_by(U.read_jsonl(U.FETCHED))
    done = set() if a.redo else {r["id"] for r in U.read_jsonl(U.FILTERED)}
    t0, n = time.time(), 0
    for r in probed:
        if r["id"] in done or n >= a.limit:
            continue
        if time.time() - t0 > a.budget_min * 60:
            print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
            break
        meta = {**fetched.get(r["id"], {}), "id": r["id"]}
        try:
            res = analyse_file(os.path.join(U.SCRAPE, r["file"]), meta, not a.no_spk, a.max_min)
        except (RuntimeError, OSError, ValueError) as ex:
            res = {"ok": False, "reasons": ["analyse_error"], "error": str(ex)[:200]}
        row = {"id": r["id"], "lang": r.get("lang"), "channel_id": r.get("channel_id"), "file": r["file"],
               "kenh": r.get("kenh"), **res, "filtered_at": U.now_iso()}
        U.append_jsonl(U.FILTERED, [row])
        n += 1
        print(f"  {r['id']:14s} ok={row['ok']} {row.get('pool')} sp={row.get('speech_ratio')} bw={row.get('bw_p90')} "
              f"mus={row.get('music_ratio')} cut/h={row.get('cuts_per_hour')} spk={row.get('n_spk_est')}/"
              f"{row.get('dominant_share')} keep={row.get('keep_hours')} h {row['reasons']} ({row.get('analyse_s')} s)",
              flush=True)
    print(f"filter: {n} tep moi")


if __name__ == "__main__":
    main()
