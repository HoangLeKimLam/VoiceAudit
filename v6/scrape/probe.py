# -*- coding: utf-8 -*-
"""CONG CONTAINER bang ffprobe (khong giai ma) cho tep goc da tai - filter_spec B1-B3.

  B1 codec_name thuoc {opus, aac, mp3, vorbis}; AAC phai profile LC (HE-AAC -> pool 'lowrate', khong vao host sach);
     sample_rate >= 16 000; <= 2 kenh; bitrate >= 64 kbps (webm/opus thuong khong ghi bit_rate -> uoc tu kich thuoc / do dai)
  B2 dung MOT luong audio (nhieu luong -> co 'multi_audio_stream', dung luong 0); do dai ffprobe khop info.duration +-2 s
  B3 luu codec / profile / bitrate / sr / encoder vao 'kenh' (yt_opus251, yt_aac140, ia_mp3, ccc_opus ...)

Ham thuan: probe_gate(ffjson, expected_dur, source, format_id) -> dict (kiem thu khong can ffprobe).
CLI: python -m v6.scrape.probe [--limit N] [--redo-failed]   (doc fetched.jsonl, ghi probed.jsonl; chay lai bo qua id da co)"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

CODECS = ("opus", "aac", "mp3", "vorbis")
MIN_SR, MAX_CH, MIN_KBPS, DUR_TOL = 16000, 2, 64.0, 2.0
MIN_KBPS_OPUS_MONO = 48.0   # Opus mono 48-64 kbps van bang rong (FOSDEM ~60 kbps); nguong 64 kbps cua filter_spec la cho stereo / AAC


def ffprobe_json(path: str) -> dict:
    cmd = [U.ffmpeg_bin("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", path]
    out = subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return json.loads(out.stdout.decode("utf-8", "replace"))


def _f(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def probe_gate(ff: Mapping[str, Any], expected_dur: float | None, source: str = "yt",
               format_id: str | None = None) -> dict[str, Any]:
    reasons, flags = [], []
    auds = [s for s in ff.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = ff.get("format", {})
    if not auds:
        return {"ok": False, "reasons": ["no_audio_stream"], "flags": [], "pool": None}
    if len(auds) > 1:
        flags.append("multi_audio_stream")
    s = auds[0]
    codec = s.get("codec_name")
    profile = s.get("profile")
    sr = int(_f(s.get("sample_rate")) or 0)
    ch = int(s.get("channels") or 0)
    dur = _f(s.get("duration")) or _f(fmt.get("duration"))
    br = _f(s.get("bit_rate")) or _f(fmt.get("bit_rate"))
    if not br and dur and _f(fmt.get("size")):
        br = _f(fmt.get("size")) * 8 / dur
    kbps = round(br / 1000, 1) if br else None
    pool = "host"
    if codec not in CODECS:
        reasons.append(f"codec_{codec}")
    if codec == "aac" and profile and "LC" not in str(profile).upper():
        flags.append("he_aac")
        pool = "lowrate"
    if sr < MIN_SR:
        reasons.append("sr_lt_16k")
    if ch > MAX_CH or ch < 1:
        reasons.append(f"channels_{ch}")
    if kbps is None:
        flags.append("bitrate_unknown")
    elif kbps < (MIN_KBPS_OPUS_MONO if codec == "opus" and ch == 1 else MIN_KBPS):
        reasons.append("bitrate_lt_64k")
    if expected_dur and dur and abs(dur - float(expected_dur)) > DUR_TOL:
        reasons.append("duration_mismatch")
    tag = (s.get("tags") or {})
    enc = tag.get("encoder") or tag.get("ENCODER") or (fmt.get("tags") or {}).get("encoder")
    fid = (format_id or "").split("-")[0]
    kenh = f"{source}_{codec}{fid}" if source == "yt" else f"{source}_{codec}"
    return {"ok": not reasons, "reasons": reasons, "flags": flags, "pool": pool if not reasons else None,
            "codec": codec, "profile": profile, "sr": sr, "channels": ch, "bitrate_kbps": kbps,
            "duration": round(dur, 3) if dur else None, "container": fmt.get("format_name"), "encoder": enc,
            "n_audio_streams": len(auds), "kenh": kenh}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--redo-failed", action="store_true", help="do lai tep da bi chan (sau khi doi nguong)")
    a = ap.parse_args()
    fetched = [r for r in U.latest_by(U.read_jsonl(U.FETCHED)).values() if r.get("ok") and r.get("file")]
    done = {r["id"] for r in U.latest_by(U.read_jsonl(U.PROBED)).values() if r.get("ok") or not a.redo_failed}
    n = 0
    for r in fetched:
        if r["id"] in done or n >= a.limit:
            continue
        path = os.path.join(U.SCRAPE, r["file"])
        try:
            res = probe_gate(ffprobe_json(path), r.get("duration"), r.get("source", "yt"), r.get("format_id"))
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as ex:
            res = {"ok": False, "reasons": ["ffprobe_error"], "error": str(ex)[:200], "flags": [], "pool": None}
        row = {"id": r["id"], "lang": r.get("lang"), "channel_id": r.get("channel_id"), "file": r["file"],
               **res, "probed_at": U.now_iso()}
        U.append_jsonl(U.PROBED, [row])
        n += 1
        print(f"  {r['id']:14s} {row.get('kenh')} {row.get('codec')}/{row.get('profile')} {row.get('sr')} Hz "
              f"ch={row.get('channels')} {row.get('bitrate_kbps')} kbps ok={row['ok']} {row['reasons']}", flush=True)
    print(f"probe: {n} tep moi")


if __name__ == "__main__":
    main()
