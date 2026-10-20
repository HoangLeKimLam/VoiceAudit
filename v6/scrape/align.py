# -*- coding: utf-8 -*-
"""MOC TU (ASR CTC + moc thoi gian tung tu) cho doan host da cat - dung lai vaudit.asr.

  vi : nguyenvulebinh/wav2vec2-base-vietnamese-250h (mac dinh cua vaudit.asr)
  en : facebook/wav2vec2-base-960h (doi vaudit.asr.ASR_NAME nhu v6.natural.build_natural._use_lang_asr), chu thuong
  Mo hinh lay tu D:/hf-cache (HF_HOME, HF_HUB_OFFLINE=1: khong goi mang). Doan > 20 s (host dai 30-120 s) chia nho tai
  cuc tieu nang luong gan moi 15 s roi cong lech thoi gian (tranh attention O(n^2) cua wav2vec2 tren 120 s).
  Khong co transcript goc -> chi kiem hop ly: so tu / giay trong [1,0; 7,5] (hosts.py loc lai).
NANG (nap torch + mo hinh ~1-1,5 GB RAM, CPU ~0,3-0,5 x thoi gian thuc): chay tron bo = hang doi; o day chi thu --limit nho.

Chi doan cua video qua CONG HIEN TAI (gate.current + gate.hold_reason, giong segment / hosts) va wav con o data/scrape.

CLI: python -m v6.scrape.align --lang vi [--tier std|long|all] [--limit 3] [--budget-min 8] [--allow-flag F]"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HF = "D:/hf-cache"
os.environ["HF_HOME"] = HF                         # truoc moi import huggingface_hub / transformers
os.environ["HF_HUB_CACHE"] = HF + "/hub"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

EN_ASR = "facebook/wav2vec2-base-960h"
CHUNK_S, SEARCH_S = 15.0, 3.0


def split_points(y: np.ndarray, sr: int, chunk_s: float = CHUNK_S, search_s: float = SEARCH_S) -> list[int]:
    """Chi so mau cat (gom 0 va len(y)): cuc tieu nang luong 50 ms trong +-search_s quanh moi chunk_s."""
    n = len(y)
    pts = [0]
    hop = int(0.05 * sr)
    t = chunk_s
    while t * sr < n - chunk_s * sr * 0.5:
        a, b = int(max(0, (t - search_s) * sr)), int(min(n, (t + search_s) * sr))
        seg = y[a:b]
        k = len(seg) // hop
        if k < 2:
            break
        e = (seg[:k * hop].reshape(k, hop) ** 2).mean(1)
        cut = a + int(np.argmin(e)) * hop + hop // 2
        if cut <= pts[-1]:
            break
        pts.append(cut)
        t = cut / sr + chunk_s
    pts.append(n)
    return pts


def transcribe_long(y: np.ndarray, sr: int, fn) -> tuple[str, list]:
    """ASR tung khuc, cong lech thoi gian. fn(y_khuc) -> (text, [Word])."""
    from vaudit.asr import Word
    pts = split_points(y, sr) if len(y) > 20 * sr else [0, len(y)]
    words = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < int(0.3 * sr):
            continue
        _, ws = fn(y[a:b])
        off = a / sr
        words += [Word(w.text, w.start + off, w.end + off) for w in ws]
    return " ".join(w.text for w in words), words


def use_lang(lang: str) -> None:
    import vaudit.asr as asr
    want = EN_ASR if lang == "en" else "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    if asr.ASR_NAME != want:
        asr.ASR_NAME = want
        asr._state.clear()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"], required=True)
    ap.add_argument("--tier", choices=["std", "long", "all"], default="all")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--allow-flag", action="append", default=[], help="nhu segment/hosts --allow-flag")
    a = ap.parse_args()
    import soundfile as sf
    from v6.scrape import gate as G
    from vaudit.asr import transcribe_words
    use_lang(a.lang)
    gated = G.current()             # khong ASR doan cua video cong hien tai bo / co loai mac dinh / wav da cach ly
    segs = [s for s in U.read_jsonl(U.SEGMENTS) if s.get("file") and s.get("lang") == a.lang
            and (a.tier == "all" or s.get("tier") == a.tier) and G.hold_reason(gated.get(s["id"]), a.allow_flag) is None
            and os.path.isfile(os.path.join(U.SCRAPE, s.get("wav") or ""))]
    done = {r["file"] for r in U.read_jsonl(U.ALIGNED)}
    t0, n = time.time(), 0
    for s in segs:
        if s["file"] in done or n >= a.limit:
            continue
        if time.time() - t0 > a.budget_min * 60:
            print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
            break
        y, sr = sf.read(os.path.join(U.SCRAPE, s["wav"]), dtype="float32")
        text, words = transcribe_long(y, sr, lambda z: transcribe_words(z, sr))
        if a.lang == "en":
            words = [type(w)(w.text.lower(), w.start, w.end) for w in words]
            text = " ".join(w.text for w in words)
        dur = len(y) / sr
        row = {"file": s["file"], "lang": a.lang, "tier": s.get("tier"), "text": text,
               "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in words],
               "n_words": len(words), "wps": round(len(words) / max(dur, 1e-6), 2),
               "asr": EN_ASR if a.lang == "en" else "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
               "aligned_at": U.now_iso()}
        U.append_jsonl(U.ALIGNED, [row])
        n += 1
        print(f"  {s['file']}: {row['n_words']} tu, {row['wps']} tu/s | {text[:80]}", flush=True)
    print(f"align: {n} doan moi ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
