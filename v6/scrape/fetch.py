# -*- coding: utf-8 -*-
"""TAI tep am thanh GOC (khong ma hoa lai) cho ung vien da qua cong - buoc 3.

YouTube (yt-dlp, KHONG cookie, JS runtime = node + yt-dlp-ejs ghim, KHONG --remote-components):
  -f 'ba[acodec=opus][format_id!*=drc][format_note*=original]/ba[acodec=opus][format_id!*=drc]/
      ba[ext=m4a][format_id!*=drc][format_note*=original]/ba[ext=m4a][format_id!*=drc]'
  = Opus 251 (hoac 251-N 'original' khi co long tieng tu dong), roi AAC 140; bo ban 'Stable volume' -drc.
  Khong -x / --audio-format, --fixup never -> tep giu nguyen nhu YouTube phat; ghi .info.json; --download-archive.
  match_filter kiem LAI giay phep CC + live_status + do dai luc tai (phong thu hai lop).
  Nghi 8-25 s giua cac tep, gioi han 3 MB/s, 1 luong; gap 'confirm you're not a bot' -> dung han.
archive.org / media.ccc.de: tai tep goc qua urllib (tiep tuc bang Range), ghi <tep>.info.json tu ung vien.
FOSDEM: chi co video .av1.webm -> ffmpeg '-map 0:a:0 -c copy' (goi Opus nguyen ven, khong ma hoa lai, khong luu video).

Tran: --per-channel (TONG so tep / kenh, tinh ca lan truoc), --max-hours-channel (tong gio da tai / kenh), --max-gb (tong GB trong raw),
--budget-min (khong bat dau tep moi sau han). Chay lai = tiep tuc (bo qua id da co trong fetched.jsonl).
Ra: data/scrape/raw/<yt|ia|ccc>/<kenh>/<id>.f<fmt>.<ext> + meta/fetched.jsonl (nguon goc day du, group = kenh).

CLI: python -m v6.scrape.fetch [--lang vi] [--source yt|ia|ccc|fosdem] [--per-channel 3] [--max-gb 3] [--max-hours-channel 10]
                               [--order short|prefer] [--max-dur 5400] [--budget-min 8] [--dry-run]
                               [--religious last|skip|mix]   (mac dinh last: kenh ton giao chi lay phan tran con lai)"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

FORMAT = ("ba[acodec=opus][format_id!*=drc][format_note*=original]/ba[acodec=opus][format_id!*=drc]/"
          "ba[ext=m4a][format_id!*=drc][format_note*=original]/ba[ext=m4a][format_id!*=drc]")
MATCH = ("license*='Creative Commons' & live_status!=is_live & live_status!=is_upcoming & live_status!=post_live "
         "& duration>=600 & duration<=21600")
BOT_MARKERS = ("confirm you", "not a bot", "sign in to confirm", "http error 429")
DEFAULT_KBPS = 110.0
LATE_FLAGS = ("minor_risk", "child_risk")


class BotCheck(RuntimeError):
    """YouTube doi xac minh nguoi that / 429 -> dung han, ha toc do, thu lai sau vai gio."""


def est_bytes(cand: Mapping[str, Any], fmt: str | None) -> float:
    kbps = DEFAULT_KBPS
    for f in cand.get("audio_formats") or []:
        if f.get("format_id") == fmt and f.get("abr"):
            kbps = float(f["abr"])
    if cand.get("size") and cand.get("source") != "fosdem":   # FOSDEM: size la cua VIDEO, chi tai luong audio
        return float(cand["size"])
    return float(cand.get("duration") or 0) * kbps * 1000 / 8


def _late(g: Mapping[str, Any]) -> bool:
    return bool(set(g.get("flags") or []) & set(LATE_FLAGS))


# Kenh KHONG tai them (nguoi dung 17/10: chi kenh khong chuyen - ghi xong dang, it nguoi dang ky; kenh > 10 nghin dang ky /
# kenh chinh thuc / dai truyen hinh). Chi chan lan tai MOI; du lieu da tai va da qua cong giu nguyen (nguoi dung chon).
NO_NEW_FETCH = {
    "UCaSb4SR6fLLiBKXuKp-v0qw": "143k dk (Hoc Van Co Minh Hien)",
    "UCCo3QfNjXnHSjBJVM70OgAQ": "21,7k dk, kenh ban khoa hoc (Phan tich SPSS)",
    "UCGFYoAZldHxBUpInc-XuVUA": "28k dk, kenh chinh thuc hoi (Hoi Ho hap VN)",
    "UCJcPERudmipXzapmx0qQxww": "dai truyen hinh (LSTV)",
    "UCPXcCzV_gfUy85GxJc8ZB6Q": "15,3k dk (Quan Ngo)",
    "UCNck9rwL6uP8Xh5V2TY_Qgw": "30,7k dk (Su Hanh Tue)",
    "UCopvJPs7wemUv6WrOBYHEMQ": "195k dk (Chua Duc Vien)",
    "UCjlUGvFf27VIk0t036iT-qw": "10,2k dk (Chua Phuc Minh)",
    "UCnAsiniI4DF7vAAe7omWWFQ": "510k dk (US National Archives)",
    "UCuXy5tCgEninup9cGplbiFw": "123k dk, bai giang co dung (MIT Missing Semester)",
    "UCUp6Pd9fx8_UX7S38Ih_JqA": "52,5k dk (NHGRI)",
    "UC7_X0WkMtkWzaVUKF-PRBNQ": "25,6k dk (LF Decentralized Trust)",
}


def plan(gated: list[Mapping[str, Any]], cands: Mapping[str, Mapping[str, Any]], fetched: list[Mapping[str, Any]],
         lang: str | None, per_channel: int, max_hours_channel: float, max_gb: float, order: str,
         max_dur: float | None, retry_failed: bool = False, religious: str = "last") -> list[dict]:
    """Chon tep can tai (ham thuan, de kiem thu): ton trong tran kenh / tong GB / da tai.
    religious: 'last' (mac dinh - kenh co co 'religious_teaching' xep SAU moi kenh khac, chi dung phan GB con lai),
    'skip' (bo han), 'mix' (thu tu cu)."""
    done = {r["id"] for r in fetched if r.get("ok") or not retry_failed}   # loi cu: bo qua tru khi --retry-failed
    hrs_ch = Counter()
    gb = 0.0
    n_ch = Counter()
    for r in fetched:
        if r.get("ok"):
            hrs_ch[r.get("channel_id")] += (r.get("duration") or 0) / 3600
            n_ch[r.get("channel_id")] += 1
            gb += (r.get("bytes") or 0) / 1e9
    by_ch: dict[str, list] = defaultdict(list)
    for g in gated:
        if not g.get("ok") or g["id"] in done or (lang and g.get("lang") != lang):
            continue
        if max_dur and (g.get("duration") or 0) > max_dur:
            continue
        if religious == "skip" and "religious_teaching" in (g.get("flags") or []):
            continue
        if g.get("channel_id") in NO_NEW_FETCH:
            continue
        by_ch[g.get("channel_id")].append(g)
    relig = {c for c, gs in by_ch.items() if any("religious_teaching" in (g.get("flags") or []) for g in gs)}
    out = []
    for cid in sorted(by_ch, key=lambda c: ((religious == "last") and c in relig, str(c))):
        items = by_ch[cid]
        # video co co bi hosts loai mac dinh (minor_risk / child_risk) xep SAU trong kenh (khong ton tran vao tep se bi bo)
        if order == "short":
            items.sort(key=lambda g: (_late(g), g.get("duration") or 0, g["id"]))
        else:
            items.sort(key=lambda g: (_late(g), -g.get("prefer", 0), g.get("duration") or 0, g["id"]))
        n = n_ch[cid]                     # tran TONG so tep da tai / kenh (ca cac lan chay truoc)
        for g in items:
            if n >= per_channel:
                break
            h = (g.get("duration") or 0) / 3600
            if hrs_ch[cid] + h > max_hours_channel:
                continue
            b = est_bytes({"duration": g.get("duration"), **cands.get(g["id"], {})}, g.get("format_id")) / 1e9
            if gb + b > max_gb:
                continue
            out.append({**g, "est_gb": round(b, 4)})
            hrs_ch[cid] += h
            gb += b
            n += 1
    return out


def _row(cand: Mapping[str, Any], gate: Mapping[str, Any], path: str | None, extra: Mapping[str, Any]) -> dict:
    return {"id": cand.get("id"), "source": cand.get("source", "yt"), "lang": gate.get("lang"),
            "channel_id": cand.get("channel_id"), "channel": cand.get("channel"), "url": cand.get("url"),
            "title": cand.get("title"), "license": cand.get("license") or cand.get("licenseurl"),
            "license_effective": gate.get("license_effective"), "license_flags": gate.get("flags"),
            "upload_date": cand.get("upload_date"), "live_status": cand.get("live_status"),
            "duration": cand.get("duration"), "file": os.path.relpath(path, U.SCRAPE).replace("\\", "/") if path else None,
            "bytes": os.path.getsize(path) if path and os.path.exists(path) else 0,
            "group": cand.get("channel_id"), "retrieved_at": U.now_iso(), **extra}


def fetch_yt(items: list[Mapping[str, Any]], cands: Mapping[str, Mapping[str, Any]], budget_min: float,
             rate: str = "3M") -> None:
    yt = U.import_ytdlp()
    from yt_dlp.utils import match_filter_func
    outdir = os.path.join(U.RAW, "yt")
    opts = U.ydl_opts(
        format=FORMAT, match_filter=match_filter_func(MATCH), fixup="never", writeinfojson=True, writethumbnail=False,
        download_archive=os.path.join(outdir, "yt_done.txt"), sleep_interval=8, max_sleep_interval=25,
        ratelimit=yt.utils.parse_bytes(rate), retries=10, fragment_retries=10, concurrent_fragment_downloads=1,
        continuedl=True, overwrites=False, outtmpl=os.path.join(outdir, "%(channel_id)s", "%(id)s.f%(format_id)s.%(ext)s"),
        postprocessors=[], keepvideo=False)
    t0 = time.time()
    with yt.YoutubeDL(opts) as ydl:
        ver = yt.version.__version__
        for i, g in enumerate(items):
            if time.time() - t0 > budget_min * 60:
                print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
                return
            cand = cands.get(g["id"], {"id": g["id"], "channel_id": g.get("channel_id")})
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={g['id']}", download=True)
            except Exception as ex:  # noqa: BLE001
                msg = str(ex)
                U.append_jsonl(U.FETCHED, [_row(cand, g, None, {"ok": False, "error": msg[:300]})])
                print(f"  LOI {g['id']}: {msg[:160]}", flush=True)
                if any(m in msg.lower() for m in BOT_MARKERS):
                    raise BotCheck(msg) from ex
                continue
            if not info or not info.get("requested_downloads"):
                U.append_jsonl(U.FETCHED, [_row(cand, g, None, {"ok": False, "error": "bi match_filter / archive bo qua"})])
                continue
            rd = info["requested_downloads"][0]
            path = rd.get("filepath")
            extra = {"ok": bool(path and os.path.exists(path)), "format_id": info.get("format_id"),
                     "acodec": info.get("acodec"), "abr": info.get("abr"), "asr": info.get("asr"),
                     "license": info.get("license"), "live_status": info.get("live_status"),
                     "was_live": info.get("was_live"), "language": info.get("language"),
                     "info_json": os.path.relpath(os.path.splitext(path)[0] + ".info.json", U.SCRAPE).replace("\\", "/")
                     if path else None, "downloader": f"yt-dlp {ver}"}
            U.append_jsonl(U.FETCHED, [_row(cand, g, path, extra)])
            print(f"  [{i + 1}/{len(items)}] {g['id']} f{extra['format_id']} {extra['acodec']} "
                  f"{(os.path.getsize(path) / 1e6 if path else 0):.1f} MB", flush=True)


def _download(url: str, path: str, timeout: int = 60) -> None:
    """Tai co tiep tuc (Range) vao <path>.part roi doi ten."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    req = urllib.request.Request(url, headers={"User-Agent": U.USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "ab" if have else "wb") as fh:   # noqa: S310
        if have and resp.status != 206:      # may chu khong ho tro Range -> tai lai tu dau
            fh.seek(0)
            fh.truncate()
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            fh.write(buf)
    os.replace(part, path)


def remux_audio(url: str, path: str, timeout: int = 1800) -> None:
    """Video tu xa (FOSDEM .av1.webm) -> CHI luong audio, sao chep goi (-c copy, khong ma hoa lai) vao <path>."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part.webm"
    cmd = [U.ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-y", "-user_agent", U.USER_AGENT, "-i", url,
           "-map", "0:a:0", "-vn", "-c", "copy", part]
    subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    os.replace(part, path)


def open_target(c: Mapping[str, Any]) -> tuple[str, str]:
    """(duong dan dich, cach tai) cho ung vien nguon mo. FOSDEM: tach audio -> .webm; con lai: tep goc nguyen ven."""
    src = c["source"]
    name = c["url"].rsplit("/", 1)[-1].split("?")[0]
    ext, how = os.path.splitext(name)[1].lower(), "urllib"
    if src == "fosdem":
        ext, how = ".webm", "ffmpeg_copy_audio"
    return os.path.join(U.RAW, src, U.slug(c["channel_id"], 80), U.slug(c["id"], 80) + ext), how


def fetch_open(items: list[Mapping[str, Any]], cands: Mapping[str, Mapping[str, Any]], budget_min: float) -> None:
    t0 = time.time()
    for g in items:
        if time.time() - t0 > budget_min * 60:
            return
        c = cands[g["id"]]
        path, how = open_target(c)
        try:
            if how == "ffmpeg_copy_audio":
                remux_audio(c["url"], path)
            else:
                _download(c["url"], path)
            with open(os.path.splitext(path)[0] + ".info.json", "w", encoding="utf-8") as fh:
                json.dump(c, fh, ensure_ascii=False, indent=1)
            U.append_jsonl(U.FETCHED, [_row(c, g, path, {"ok": True, "downloader": how})])
            print(f"  {g['id']}: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
        except Exception as ex:  # noqa: BLE001
            U.append_jsonl(U.FETCHED, [_row(c, g, None, {"ok": False, "error": str(ex)[:300]})])
        time.sleep(2.0)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"])
    ap.add_argument("--source", action="append", choices=["yt", "ia", "ccc", "fosdem"])
    ap.add_argument("--per-channel", type=int, default=3)
    ap.add_argument("--max-hours-channel", type=float, default=10.0)
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=None, help="chi tai tep <= so giay nay (pilot)")
    ap.add_argument("--order", choices=["short", "prefer"], default="prefer")
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--rate", default="3M")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--religious", choices=["last", "skip", "mix"], default="last",
                    help="kenh bai giang ton giao: xep sau cung (mac dinh) / bo / tron (chi khi nguoi dung xac nhan)")
    a = ap.parse_args()
    from v6.scrape import gate as G
    cands = U.latest_by(U.read_jsonl(U.CANDIDATES))
    gated = list(G.current().values())       # cong tinh lai (khong tin gated.jsonl cu neu chinh sach da doi)
    fetched = list(U.latest_by(U.read_jsonl(U.FETCHED)).values())
    if a.source:   # vd chi nguon mo: --source ia --source ccc --source fosdem
        gated = [g for g in gated if g.get("source", "yt") in a.source]
    items = plan(gated, cands, fetched, a.lang, a.per_channel, a.max_hours_channel, a.max_gb, a.order, a.max_dur, a.retry_failed,
                 a.religious)
    print(f"ke hoach tai: {len(items)} tep, ~{sum(i['est_gb'] for i in items):.2f} GB, "
          f"{sum((i.get('duration') or 0) for i in items) / 3600:.1f} h", flush=True)
    if a.dry_run:
        for i in items:
            print(f"  {i['lang']} {i['channel_id']} {i['id']} {i.get('duration')} s ~{i['est_gb']:.3f} GB")
        return
    yt_items = [i for i in items if i.get("source", "yt") == "yt"]
    open_items = [i for i in items if i.get("source", "yt") != "yt"]
    if yt_items:
        try:
            fetch_yt(yt_items, cands, a.budget_min, a.rate)
        except BotCheck as ex:      # dung han: khong cookie, khong vuot bao ve; cho vai gio, ha --rate
            print(f"DUNG: YouTube doi xac minh / 429 ({str(ex)[:120]}). Cho vai gio roi chay lai voi --rate 1M.")
            sys.exit(3)
    if open_items:
        fetch_open(open_items, cands, a.budget_min)


if __name__ == "__main__":
    main()
