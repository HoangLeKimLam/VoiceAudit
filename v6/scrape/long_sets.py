# -*- coding: utf-8 -*-
"""BO TEST TEP DAI 1-10 PHUT tu ghi am cao (nguoi dung 18/10 'test het': tap cham cu chi 8 tep > 20 s).

  Nguon: video DA QUA moi cong cua hosts (co mat trong hosts_<lang> / hosts_long_<lang> hien tai -> giay phep, kenh khong
  chuyen / khong bi cach ly, LID, co loai deu da xet) va qua bo loc vet cat (filtered.ok, keep_spans = khuc LIEN MACH khong
  co vet cat). KHONG ghep cac doan roi nhau (chinh viec ghep tao vet sua): moi tep la MOT khuc lien mach cua ban goc.
  1) extract: moi video <= 1 khuc, do dai luan phien TARGETS (60 / 120 / 240 / 400 / 560 s; <= 10 phut cua K1),
     ffmpeg -> 16 kHz mono wav (data/scrape/longx_<lang>/<file>.wav), ASR + moc tu nhu align (transcribe_long),
     kiem tu / giay trong [1,0; 7,5] -> meta/longx_<lang>.jsonl (ghi tiep, --budget-min, chay lai = tiep tuc).
  2) build: thao tac cua build_natural.make_one tren tep dai (doan cho cua ghep = doan host CUNG video), PLAN moi ngon ngu
     -> data_labels/v6/long<lang>/{<kind>/<id>.wav, manifest.json} (meta giong scrapevi; bo muc khong hop le nhu build_sets).
CLI: python -m v6.scrape.long_sets extract --lang vi [--max-videos 90] [--budget-min 60]
     python -m v6.scrape.long_sets build --lang vi [--seed 0]"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

TARGETS = (60.0, 120.0, 240.0, 400.0, 560.0)
PLAN = {"nguyen_ban": 60, "xoa_loi": 25, "ghep_thay": 25, "ghep_chen": 25, "giong_cuc_bo": 25, "nen_cuc_bo": 25,
        "giong_toan_cuc": 10, "nen_toan_cuc": 10}
WPS = (1.0, 7.5)
# 18/10: doi giong / khu on CA TEP tren khuc 9 phut lam torch sap (c10 AbortHandler) -> thao tac toan cuc chi tren khuc
# <= 240 s; thao tac cuc bo van dung moi do dai.
GLOBAL_MAX_S = 240.0
V6 = os.path.join(U.BASE, "data_labels", "v6")


def longx_dir(lang: str) -> str:
    return os.path.join(U.SCRAPE, f"longx_{lang}")


def longx_jsonl(lang: str) -> str:
    return os.path.join(U.META, f"longx_{lang}.jsonl")


def host_videos(lang: str) -> dict[str, dict]:
    """{video_id: dong host dau tien} cua hosts hien tai (std + long) - chi video da qua moi cong."""
    out: dict[str, dict] = {}
    for tier in ("std", "long"):
        for r in U.read_jsonl(os.path.join(U.hosts_dir(lang, tier), "hosts.jsonl")):
            out.setdefault(r["video_id"], r)
    return out


def pick_excerpts(filtered: list[dict], allowed: set[str], max_videos: int, seed: int) -> list[dict]:
    """Ham thuan: <= 1 khuc / video; do dai luan phien TARGETS; khuc phai nam tron trong MOT keep_span."""
    rng = random.Random(seed)
    vids = sorted((r for r in filtered if r.get("ok") and r["id"] in allowed), key=lambda r: r["id"])
    rng.shuffle(vids)
    out, k = [], 0
    for r in vids:
        if len(out) >= max_videos:
            break
        spans = [(a, b) for a, b in (r.get("keep_spans") or []) if b - a >= TARGETS[0]]
        if not spans:
            continue
        want = TARGETS[k % len(TARGETS)]
        fit = [(a, b) for a, b in spans if b - a >= want] or [max(spans, key=lambda s: s[1] - s[0])]
        a, b = rng.choice(fit)
        length = min(want, b - a)
        start = round(a + rng.uniform(0, (b - a) - length), 2)
        out.append({"id": r["id"], "lang": r.get("lang"), "start": start, "dur": round(length, 2)})
        k += 1
    return out


def decode(src: str, start: float, dur: float, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run([U.ffmpeg_bin(), "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", src,
                    "-ac", "1", "-ar", str(U.SR), "-c:a", "pcm_s16le", dst], check=True, timeout=600)


def extract(lang: str, max_videos: int, budget_min: float, seed: int = 0) -> int:
    import soundfile as sf
    from v6.scrape.align import EN_ASR, transcribe_long, use_lang
    from vaudit.asr import transcribe_words
    hv = host_videos(lang)
    fetched = U.latest_by(U.read_jsonl(U.FETCHED))
    filt = [r for r in U.latest_by(U.read_jsonl(U.FILTERED)).values() if r.get("lang") == lang]
    plan = pick_excerpts(filt, set(hv), max_videos, seed)
    done = {r["video_id"] for r in U.read_jsonl(longx_jsonl(lang))}
    use_lang(lang)
    t0, n = time.time(), 0
    for ex in plan:
        if ex["id"] in done:
            continue
        if time.time() - t0 > budget_min * 60:
            print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
            break
        f = fetched.get(ex["id"]) or {}
        src = os.path.join(U.SCRAPE, f.get("file") or "")
        if not f.get("file") or not os.path.isfile(src):
            print(f"  bo {ex['id']}: thieu tep goc", flush=True)
            continue
        h = hv[ex["id"]]
        name = U.slug(f"lx_{h['source']}_{ex['id']}_{int(ex['start'])}", 90)
        wav = os.path.join(longx_dir(lang), name + ".wav")
        decode(src, ex["start"], ex["dur"], wav)
        y, sr = sf.read(wav, dtype="float32")
        text, words = transcribe_long(y, sr, lambda z: transcribe_words(z, sr))
        if lang == "en":
            words = [type(w)(w.text.lower(), w.start, w.end) for w in words]
            text = " ".join(w.text for w in words)
        dur = len(y) / sr
        wps = len(words) / max(dur, 1e-6)
        row = {**{k: h.get(k) for k in ("source", "cluster", "person", "group", "channel_id", "url", "license", "codec",
                                         "bitrate", "orig_file", "kenh", "pool", "conf_call", "flags", "title", "channel")},
               "file": name, "video_id": ex["id"], "lang": lang, "start": ex["start"], "end": round(ex["start"] + dur, 2),
               "dur": round(dur, 2), "tier": "longx", "text": text,
               "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in words], "n_words": len(words),
               "wps": round(wps, 2), "asr_ok": WPS[0] <= wps <= WPS[1],
               "asr": EN_ASR if lang == "en" else "nguyenvulebinh/wav2vec2-base-vietnamese-250h", "made_at": U.now_iso()}
        U.append_jsonl(longx_jsonl(lang), [row])
        n += 1
        print(f"  {name}: {dur:.0f} s, {len(words)} tu, {wps:.2f} tu/s {'' if row['asr_ok'] else '(BO: tu/s)'}", flush=True)
    print(f"longx {lang}: {n} khuc moi ({time.time() - t0:.0f} s)")
    return n


def build(lang: str, seed: int, out_root: str = V6) -> list[dict]:
    import soundfile as sf
    from v6.natural import build_natural as BN
    from v6.scrape import build_sets as BS
    from vaudit.asr import Word
    set_name = f"long{lang}"
    prefix = f"v6{set_name}_"
    outd = os.path.join(out_root, set_name)
    man_path = os.path.join(outd, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else []
    hv = host_videos(lang)
    longs = [r for r in U.latest_by(U.read_jsonl(longx_jsonl(lang)), "file").values()
             if r.get("asr_ok") and r["video_id"] in hv]               # video bi go sau khi trich -> khong dung
    info = {r["file"]: r for r in longs}
    mk = lambda r, path: BN.NHost(r["file"], r["cluster"], r["person"], r["source"], r["text"],   # noqa: E731
                                  [w[0] for w in r["words"]], path, [Word(*w) for w in r["words"]], {})
    hosts = [mk(r, os.path.join(longx_dir(lang), r["file"] + ".wav")) for r in longs]
    by_cluster, by_person = defaultdict(list), defaultdict(list)     # doan cho: doan host CUNG video (std + long)
    valid = set(info)
    for tier in ("std", "long"):
        hd = U.hosts_dir(lang, tier)
        for r in U.read_jsonl(os.path.join(hd, "hosts.jsonl")):
            by_cluster[r["cluster"]].append(mk(r, os.path.join(hd, r["source"], r["file"] + ".wav")))
            valid.add(r["file"])
    man = BS.prune(man, valid, outd)
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    for kind, goal in PLAN.items():
        have = [e for e in man if e["kind"] == kind]
        used = {e["meta"]["host_file"] for e in have}
        k, idx = len(have), BS.next_index(have, prefix, kind)
        pool = [h for h in hosts if h.file not in used
                and (not kind.endswith("toan_cuc") or info[h.file]["dur"] <= GLOBAL_MAX_S)]
        rng.shuffle(pool)
        for h in pool:
            if k >= goal:
                break
            try:
                r = BN.make_one(kind, h, by_person, by_cluster, rng, nrng)
            except Exception as ex:  # noqa: BLE001
                print(f"    bo {h.file}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
                continue
            if r is None:
                continue
            wav, meta = r
            vid = f"{prefix}{kind}_{idx:04d}"
            p = os.path.join(outd, kind, vid + ".wav")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            wav = np.asarray(wav, np.float32)
            peak = float(np.abs(wav).max()) if len(wav) else 0.0
            sf.write(p, wav / peak * 0.999 if peak > 0.999 else wav, BN.SR)
            m = BS._meta_fix(meta, info[h.file], lang)
            m["long_excerpt"] = {k_: info[h.file][k_] for k_ in ("video_id", "start", "end", "dur")}
            man.append({"id": vid, "kind": kind, "cfg": "goc", "src": vid,
                        "wav": os.path.relpath(p, BS.V4).replace("\\", "/"), "meta": m})
            k += 1
            idx += 1
        print(f"  {kind:16s} {k}/{goal}", flush=True)
        os.makedirs(outd, exist_ok=True)
        with open(man_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=0)
        os.replace(man_path + ".tmp", man_path)
    durs = [e["meta"]["long_excerpt"]["dur"] for e in man]
    print(f"{len(man)} tep trong manifest {set_name}; do dai khuc goc {min(durs, default=0):.0f}-{max(durs, default=0):.0f} s")
    return man


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "build"])
    ap.add_argument("--lang", choices=["vi", "en"], required=True)
    ap.add_argument("--max-videos", type=int, default=90)
    ap.add_argument("--budget-min", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.lang, a.max_videos, a.budget_min, a.seed)
    else:
        build(a.lang, a.seed)


# ---------------- kiem thu (khong ASR, khong mo hinh) ----------------
def test_pick_excerpts() -> None:
    filt = [{"id": f"v{i}", "ok": True, "keep_spans": [[0.0, 100.0], [110.0, 800.0]]} for i in range(6)]
    filt += [{"id": "short", "ok": True, "keep_spans": [[0.0, 30.0]]}, {"id": "bad", "ok": False, "keep_spans": [[0, 900]]},
             {"id": "out", "ok": True, "keep_spans": [[0, 900]]}]
    ex = pick_excerpts(filt, {f"v{i}" for i in range(6)} | {"short", "bad"}, 99, 0)
    assert {e["id"] for e in ex} == {f"v{i}" for i in range(6)}          # bo khuc ngan / video khong dat / ngoai hosts
    assert sorted(e["dur"] for e in ex) == sorted([60.0, 120.0, 240.0, 400.0, 560.0, 60.0])
    for e in ex:                                                            # nam tron trong mot keep_span
        s, t = e["start"], e["start"] + e["dur"]
        assert any(a <= s and t <= b + 1e-6 for a, b in filt[0]["keep_spans"]), e
    assert len(pick_excerpts(filt, {f"v{i}" for i in range(6)}, 2, 0)) == 2


if __name__ == "__main__":
    main()
