# -*- coding: utf-8 -*-
"""CAT HOST tu tep goc da qua bo loc - doan chuan 5-20 s VA doan dai 30-120 s, bien dat o KHOANG LANG (>= 300 ms).

  - chi trong keep_spans cua filter (da bo nhac, vet cat +- 1 s, khoang khong loi > 120 s, doi bang thong)
  - bien doan = diem giua khoang khong loi >= 3 khung (300 ms) theo mat na loi noi trong frames/<id>.npz
  - doan chuan: do dai muc tieu ~ U(6, 18) s (gioi han 5-20 s); doan dai: ~ U(30, 120) s; ty le loi noi trong doan >= 0,6
  - cac doan chuan LIEN TIEP nhau (giu duoc thu tu de ghep lai thanh tep dai 'LONG-NAT' cung phien sau nay)
  - giai ma mot luot ffmpeg theo khoi (RAM nho), ghi 16 kHz mono PCM16 WAV:
      data/scrape/hosts_<lang>/<source>/<file>.wav        (chuan)   file = <source>_<video>_s<NNN>
      data/scrape/hosts_long_<lang>/<source>/<file>.wav   (dai)     file = <source>_<video>_l<NN>
Chi cat video qua CONG HIEN TAI (gate.current, tinh lai trong bo nho; thieu dong cong -> bo) va khong co co bi loai mac
  dinh (gate.EXCLUDE_FLAGS: religious_teaching / minor_risk / child_risk; mo bang --allow-flag khi nguoi dung xac nhan).
Ra: meta/segments.jsonl (file, tier, lang, source, id video, start, end, dur, speech_ratio, wav). Chay lai bo qua video da cat.

CLI: python -m v6.scrape.segment [--lang vi|en] [--limit N] [--max-std 80] [--max-long 12] [--seed 0] [--allow-flag F]"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from typing import Any, Mapping

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import frames as FR  # noqa: E402
from v6.scrape import gate as G  # noqa: E402
from v6.scrape import util as U  # noqa: E402

SR = FR.SR
PAUSE_FR = 3
STD = (5.0, 20.0, 6.0, 18.0)       # min, max, muc tieu thap, muc tieu cao
LONG = (30.0, 120.0, 30.0, 120.0)
MIN_SPEECH = 0.6


def pause_points(speech: np.ndarray, min_fr: int = PAUSE_FR) -> np.ndarray:
    """Diem giua (giay) cac khoang khong loi >= min_fr khung."""
    pts = [(a + b) / 2 * FR.FR_S for a, b in FR.runs(~speech.astype(bool)) if b - a >= min_fr]
    return np.array(pts, float)


def plan_windows(speech: np.ndarray, spans: list[list[float]], bounds: tuple[float, float, float, float],
                 rng: random.Random, max_n: int) -> list[tuple[float, float, float]]:
    """Cua so (start, end, speech_ratio) bien tai khoang lang, trong spans. Tham lam, lien tiep."""
    lo, hi, tlo, thi = bounds
    pts = pause_points(speech)
    out = []
    for s0, s1 in spans:
        p = pts[(pts >= s0) & (pts <= s1)]
        i = 0
        while i < len(p) - 1:
            start = p[i]
            target = start + rng.uniform(tlo, thi)
            ok = np.where((p > start + lo) & (p <= min(start + hi, s1)))[0]
            if not len(ok):
                i += 1
                continue
            j = ok[np.argmin(np.abs(p[ok] - target))]
            end = p[j]
            a, b = int(start / FR.FR_S), int(end / FR.FR_S)
            r = float(speech[a:b].mean()) if b > a else 0.0
            if r >= MIN_SPEECH:
                out.append((round(float(start), 2), round(float(end), 2), round(r, 3)))
                i = j
            else:
                i += 1
    if len(out) > max_n:   # rai deu tren ca tep, giu thu tu
        keep = sorted(rng.sample(range(len(out)), max_n))
        out = [out[k] for k in keep]
    return out


def safe_vid(rid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(rid).split(":", 1)[-1])[:60]


def cut_stream(blocks, wins: list[dict]) -> set[str]:
    """Doc cac khoi lien tiep, ghi tung cua so khi da du du lieu (bo dem <= cua so dai nhat + 1 khoi).
    Tra TAP duong dan da ghi (giai ma dung som -> chi cac cua so da du du lieu; review 12/10: truoc day tra so dem theo
    thu tu da sap xep nhung nguoi goi cat theo thu tu chuan-roi-dai -> segments.jsonl co the ghi wav khong ton tai)."""
    wins = sorted(wins, key=lambda w: w["start"])
    buf = np.zeros(0, np.float32)
    buf_t0 = 0.0
    k = 0
    written: set[str] = set()
    for blk in blocks:
        buf = np.concatenate([buf, blk])
        buf_end = buf_t0 + len(buf) / SR
        while k < len(wins) and wins[k]["end"] <= buf_end:
            w = wins[k]
            a, b = int(round((w["start"] - buf_t0) * SR)), int(round((w["end"] - buf_t0) * SR))
            os.makedirs(os.path.dirname(w["path"]), exist_ok=True)
            sf.write(w["path"], buf[max(0, a):b], SR, subtype="PCM_16")
            written.add(w["path"])
            k += 1
        nxt = wins[k]["start"] if k < len(wins) else buf_end
        drop = min(len(buf), int(max(0.0, nxt - buf_t0) * SR))
        if drop > 0:
            buf, buf_t0 = buf[drop:], buf_t0 + drop / SR
    return written


def segment_file(row: Mapping[str, Any], fetched: Mapping[str, Any], rng: random.Random, max_std: int,
                 max_long: int, write: bool = True) -> list[dict]:
    from v6.scrape.filter import decode_blocks, frames_path
    z = np.load(frames_path(row["id"]))
    speech = z["speech"].astype(bool)
    spans = row.get("keep_spans") or []
    lang = row.get("lang") or fetched.get("lang")
    src = fetched.get("source", "yt")
    vid = safe_vid(row["id"])
    wins = []
    for tier, bounds, mx, tag, nd in (("std", STD, max_std, "s", 3), ("long", LONG, max_long, "l", 2)):
        for n, (a, b, r) in enumerate(plan_windows(speech, spans, bounds, rng, mx)):
            f = f"{src}_{vid}_{tag}{n:0{nd}d}"
            path = os.path.join(U.hosts_dir(lang, tier), src, f + ".wav")
            wins.append({"file": f, "tier": tier, "lang": lang, "source": src, "id": row["id"],
                         "channel_id": row.get("channel_id"), "start": a, "end": b, "dur": round(b - a, 2),
                         "speech_ratio": r, "path": path, "wav": os.path.relpath(path, U.SCRAPE).replace("\\", "/")})
    if write and wins:
        done = cut_stream(decode_blocks(os.path.join(U.SCRAPE, row["file"])), wins)
        wins = [w for w in wins if w["path"] in done]      # dung DUNG cac tep da ghi
    for w in wins:
        w.pop("path", None)
        w["segmented_at"] = U.now_iso()
    return wins


def select_rows(filtered, gated: Mapping[str, Mapping[str, Any]], lang: str | None = None,
                allow_flags=()) -> tuple[list[dict], dict]:
    """Tep can cat: filter ok + CONG HIEN TAI ok (thieu dong cong -> bo, fail closed) + khong co co bi loai mac dinh."""
    out, why = [], {}
    for r in filtered:
        if not r.get("ok") or (lang and r.get("lang") != lang):
            continue
        key = G.hold_reason(gated.get(r["id"]), allow_flags)
        if key:
            why[key] = why.get(key, 0) + 1
            continue
        out.append(r)
    return out, why


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--max-std", type=int, default=80)
    ap.add_argument("--max-long", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lang", choices=["vi", "en"])
    ap.add_argument("--allow-flag", action="append", choices=G.EXCLUDE_FLAGS, default=[],
                    help="cat ca video co co bi loai mac dinh - CHI khi nguoi dung da xac nhan")
    a = ap.parse_args()
    gated = G.current()                     # cong tinh lai trong bo nho (khong tin gated.jsonl cu)
    rows, skipped = select_rows(U.latest_by(U.read_jsonl(U.FILTERED)).values(), gated, a.lang, a.allow_flag)
    if skipped:
        print(f"  bo qua (cong hien tai / co loai mac dinh): {skipped}", flush=True)
    fetched = U.latest_by(U.read_jsonl(U.FETCHED))
    done = {r["id"] for r in U.read_jsonl(U.SEGMENTS)}
    n = 0
    for r in rows:
        if r["id"] in done or n >= a.limit:
            continue
        rng = random.Random(f"{a.seed}:{r['id']}")
        segs = segment_file(r, fetched.get(r["id"], {}), rng, a.max_std, a.max_long)
        U.append_jsonl(U.SEGMENTS, segs or [{"id": r["id"], "file": None, "empty": True}])
        n += 1
        ns = sum(s["tier"] == "std" for s in segs)
        print(f"  {r['id']:14s} chuan {ns:3d} ({sum(s['dur'] for s in segs if s['tier'] == 'std') / 60:5.1f} ph) "
              f"dai {len(segs) - ns:2d}", flush=True)
    print(f"segment: {n} video moi")


if __name__ == "__main__":
    main()
