# -*- coding: utf-8 -*-
"""LUAT NGON NGU CAP VIDEO cho doan cao tieng Anh - THU NGHIEM 17/10, KHONG DUNG (hosts.py khong goi).
  Ket qua calib (meta/lex_video_calib.json, ban dau: lex_video_calib_v1.json): ban ghi tieng Viet qua ASR tieng Anh cho
  video_lex trung vi 0,65 (max 0,77) chong lan video tieng Anh that (min 0,67); chi tinh tu >= 4 chu: am max 0,53 vs
  duong min 0,40 -> bien qua mong. Nguoi dung chon giu nguong Whisper en 0,5.

  Ly do: Whisper-small LID nham CA VIDEO tieng Anh FOSDEM thanh cy / sl / el / he (am hoc phong; p_en trung vi 0,03-0,2)
  trong khi ASR tieng Anh (facebook/wav2vec2-base-960h) cua chinh cac video do cho chu ro (lex_share trung vi 0,67-0,87).
  Luat (hosts.py): doan en GIU neu p_en >= lid.LID_MIN['en'] (0,5) HOAC
                   (video_lex(video) >= VIDEO_LEX_MIN va lex_share(doan) >= SEG_LEX_MIN)  -> lang_check 'lid+video_lex'.
  video_lex = trung vi lex_share (ty le tu ASR nam trong tu vung host tu nhien tieng Anh, hosts.lex_vocab) tren cac doan
  cua video (moi tang std / long) - on dinh hon tung doan vi mot video = mot ngon ngu.
calib: tren du lieu KHONG dung de dat luat Whisper:
  am  = ban ghi tieng Viet tu nhien (data/natural_v6/hosts: vss / vox / celeb; cum = ban ghi) qua ASR TIENG ANH (chay that)
        + video cao biet la ngon ngu khac (NEG_VIDEOS: ia Meyer-Cottin tieng Phap) da co ASR tieng Anh trong aligned.jsonl;
  duong = video cao en co trung vi p_en >= 0,9 (Whisper chac chan) -> phan bo video_lex va lex_share doan.
  -> meta/lex_video_calib.json. NANG VUA (nap wav2vec2 ~1 GB, ~10 phut CPU voi --n-neg 40).

CLI: python -m v6.scrape.lex_video --calib [--n-neg 40] [--per-rec 6] [--seed 0]
     python -m v6.scrape.lex_video --report      (khong nap mo hinh: video en bi Whisper loai nhung luat moi giu)"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from collections import defaultdict
from typing import Any, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

# nguong chot sau calib 17/10 (xem lex_video_calib.json); None = chua hieu chinh -> luat tat
VIDEO_LEX_MIN: float | None = None
SEG_LEX_MIN: float | None = None
VIDEO_LEX_LANGS = ("en",)
NEG_VIDEOS = ("ia:ConfrenceMeyerCottinCattchcse",)      # tien to video_id cao biet KHONG phai tieng Anh
CALIB_OUT = os.path.join(U.META, "lex_video_calib.json")
NAT_VI = os.path.join(U.BASE, "data", "natural_v6", "hosts", "hosts.jsonl")


def video_lex(segs: Iterable[Mapping[str, Any]], aligned: Mapping[str, Mapping[str, Any]], lang: str,
              vocab: set) -> dict[str, float]:
    """{video_id: trung vi lex_share cac doan (moi tang) cua video co ASR}. Ham thuan."""
    from v6.scrape.hosts import lex_share
    by_v: dict[str, list[float]] = defaultdict(list)
    for s in segs:
        if s.get("lang") != lang or not s.get("file"):
            continue
        al = aligned.get(s["file"])
        lx = lex_share(al["words"], vocab) if al else None
        if lx is not None:
            by_v[s["id"]].append(lx)
    return {v: round(statistics.median(xs), 4) for v, xs in by_v.items() if xs}


def keep_by_video_lex(lang: str, vlex: float | None, seg_lex: float | None) -> bool:
    """Luat bo sung (chi khi Whisper LID duoi nguong): True neu video va doan deu dat nguong lex. Tat neu chua hieu chinh."""
    if lang not in VIDEO_LEX_LANGS or VIDEO_LEX_MIN is None or SEG_LEX_MIN is None:
        return False
    return vlex is not None and seg_lex is not None and vlex >= VIDEO_LEX_MIN and seg_lex >= SEG_LEX_MIN


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)


def neg_vi_items(n_rec: int, per_rec: int, rng: random.Random) -> list[tuple[str, str]]:
    """[(cum ban ghi, wav)] - n_rec ban ghi tieng Viet tu nhien, <= per_rec doan 4-30 s moi ban ghi."""
    from v6.scrape.qc import wav_map
    base, wmap = os.path.dirname(NAT_VI), None
    by_c: dict[str, list[dict]] = defaultdict(list)
    for r in U.read_jsonl(NAT_VI):
        if 4.0 <= float(r.get("dur") or 0) <= 30.0:
            by_c[str(r.get("cluster"))].append(r)
    clusters = sorted(c for c, rs in by_c.items() if len(rs) >= 3)
    rng.shuffle(clusters)
    out = []
    for c in clusters[:n_rec]:
        rs = by_c[c][:]
        rng.shuffle(rs)
        for r in rs[:per_rec]:
            w = os.path.join(base, str(r.get("source")), str(r.get("file")) + ".wav")
            if not os.path.exists(w):
                wmap = wav_map(base) if wmap is None else wmap
                w = wmap.get(str(r.get("file")))
            if w and os.path.exists(w):
                out.append((c, w))
    return out


def calib(n_neg: int, per_rec: int, seed: int, out_path: str = CALIB_OUT) -> dict:
    import soundfile as sf
    from v6.scrape.align import transcribe_long, use_lang
    from v6.scrape.hosts import lex_share, lex_vocab
    from vaudit.asr import transcribe_words
    rng = random.Random(seed)
    vocab = lex_vocab("en")
    use_lang("en")
    neg_seg: dict[str, list[float]] = defaultdict(list)
    neg_words: list[dict] = []
    for c, w in neg_vi_items(n_neg, per_rec, rng):          # am that: tieng Viet qua ASR tieng Anh
        y, sr = sf.read(w, dtype="float32", always_2d=True)
        y = y.mean(1)
        _, words = transcribe_long(y, sr, lambda z: transcribe_words(z, sr))
        toks = [w_.text.lower() for w_ in words]
        neg_words.append({"rec": c, "wav": os.path.basename(w), "words": toks})
        lx = lex_share(toks, vocab)
        if lx is not None:
            neg_seg[c].append(lx)
    segs = list(U.read_jsonl(U.SEGMENTS))
    aligned = U.latest_by(U.read_jsonl(U.ALIGNED), "file")
    lid = U.latest_by(U.read_jsonl(os.path.join(U.META, "lid.jsonl")), "file")
    vlex = video_lex(segs, aligned, "en", vocab)
    p_by_v: dict[str, list[float]] = defaultdict(list)
    seg_lex_pos: list[float] = []
    for s in segs:
        r = lid.get(s.get("file") or "")
        if s.get("lang") == "en" and r and r.get("p_target") is not None:
            p_by_v[s["id"]].append(r["p_target"])
    pos_v = {v for v, ps in p_by_v.items() if statistics.median(ps) >= 0.9}
    for s in segs:
        al = aligned.get(s.get("file") or "")
        if s.get("id") in pos_v and al:
            lx = lex_share(al["words"], vocab)
            if lx is not None:
                seg_lex_pos.append(lx)
    neg_rec = [statistics.median(xs) for xs in neg_seg.values() if xs]
    neg_known = {v: x for v, x in vlex.items() if v.startswith(NEG_VIDEOS)}
    neg_all_v = neg_rec + list(neg_known.values())
    neg_all_seg = [x for xs in neg_seg.values() for x in xs]
    pos_vl = [vlex[v] for v in pos_v if v in vlex]
    res: dict[str, Any] = {
        "asr": "facebook/wav2vec2-base-960h", "seed": seed, "n_neg_rec_vi": len(neg_rec), "n_neg_seg_vi": len(neg_all_seg),
        "neg_known_videos": neg_known,
        "neg_video_max": _q(neg_all_v, 1.0), "neg_video_p95": _q(neg_all_v, 0.95), "neg_video_median": _q(neg_all_v, 0.5),
        "neg_seg_max": _q(neg_all_seg, 1.0), "neg_seg_p99": _q(neg_all_seg, 0.99),
        "n_pos_videos": len(pos_vl), "pos_video_min": _q(pos_vl, 0), "pos_video_p01": _q(pos_vl, 0.01),
        "pos_video_p05": _q(pos_vl, 0.05), "pos_video_median": _q(pos_vl, 0.5),
        "n_pos_seg": len(seg_lex_pos), "pos_seg_p01": _q(seg_lex_pos, 0.01), "pos_seg_p05": _q(seg_lex_pos, 0.05),
        "sweep_video": {str(t): {"neg_above": round(sum(x >= t for x in neg_all_v) / max(1, len(neg_all_v)), 4),
                                 "pos_below": round(sum(x < t for x in pos_vl) / max(1, len(pos_vl)), 4)}
                        for t in (0.55, 0.6, 0.62, 0.65, 0.7)},
        "neg_rec_values": sorted(round(x, 4) for x in neg_rec), "neg_words": neg_words}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    return res


def report() -> dict:
    """Video en Whisper loai (trung vi p_en < 0,5) va quyet dinh luat moi - khong nap mo hinh."""
    from v6.scrape.hosts import lex_vocab
    segs = list(U.read_jsonl(U.SEGMENTS))
    aligned = U.latest_by(U.read_jsonl(U.ALIGNED), "file")
    lid = U.latest_by(U.read_jsonl(os.path.join(U.META, "lid.jsonl")), "file")
    vlex = video_lex(segs, aligned, "en", lex_vocab("en"))
    p_by_v: dict[str, list[float]] = defaultdict(list)
    for s in segs:
        r = lid.get(s.get("file") or "")
        if s.get("lang") == "en" and r and r.get("p_target") is not None:
            p_by_v[s["id"]].append(r["p_target"])
    low = {v: round(statistics.median(ps), 3) for v, ps in p_by_v.items() if statistics.median(ps) < 0.5}
    rows = [{"video": v, "p_en_median": p, "video_lex": vlex.get(v),
             "giu": VIDEO_LEX_MIN is not None and (vlex.get(v) or 0) >= VIDEO_LEX_MIN} for v, p in sorted(low.items())]
    return {"video_lex_min": VIDEO_LEX_MIN, "seg_lex_min": SEG_LEX_MIN, "n_low_videos": len(rows),
            "n_kept": sum(r["giu"] for r in rows), "rows": rows}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--n-neg", type=int, default=40)
    ap.add_argument("--per-rec", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.calib:
        res = calib(a.n_neg, a.per_rec, a.seed)
        print(json.dumps({k: v for k, v in res.items() if k != "neg_words"}, ensure_ascii=False, indent=1))
        print("->", CALIB_OUT)
    elif a.report:
        res = report()
        for r in res["rows"]:
            print(f"  {'GIU' if r['giu'] else 'bo '} p_en={r['p_en_median']:.3f} vlex={r['video_lex']} {r['video'][:80]}")
        print({k: v for k, v in res.items() if k != "rows"})
    else:
        ap.print_help()


# ---------------- kiem thu (khong nap mo hinh) ----------------
def test_video_lex_and_rule() -> None:
    global VIDEO_LEX_MIN, SEG_LEX_MIN
    segs = [{"file": "a1", "id": "A", "lang": "en"}, {"file": "a2", "id": "A", "lang": "en"},
            {"file": "a3", "id": "A", "lang": "en"}, {"file": "b1", "id": "B", "lang": "en"},
            {"file": "c1", "id": "C", "lang": "vi"}, {"file": "d1", "id": "D", "lang": "en"}]
    w = lambda *ts: {"words": [[t, 0, 1] for t in ts]}     # noqa: E731
    aligned = {"a1": w("x", "y"), "a2": w("x", "q"), "a3": w("x", "y", "z", "q"), "b1": w("q", "q"), "c1": w("x")}
    vl = video_lex(segs, aligned, "en", {"x", "y", "z"})
    assert vl == {"A": 0.75, "B": 0.0}, vl                  # trung vi (1,0 / 0,5 / 0,75); D khong co ASR; C khac ngon ngu
    saved = (VIDEO_LEX_MIN, SEG_LEX_MIN)
    try:
        VIDEO_LEX_MIN, SEG_LEX_MIN = None, None
        assert not keep_by_video_lex("en", 0.9, 0.9)        # chua hieu chinh -> tat
        VIDEO_LEX_MIN, SEG_LEX_MIN = 0.7, 0.5
        assert keep_by_video_lex("en", 0.75, 0.5) and not keep_by_video_lex("en", 0.69, 0.9)
        assert not keep_by_video_lex("en", 0.9, 0.4) and not keep_by_video_lex("vi", 0.9, 0.9)
        assert not keep_by_video_lex("en", None, 0.9)
    finally:
        VIDEO_LEX_MIN, SEG_LEX_MIN = saved


if __name__ == "__main__":
    main()
