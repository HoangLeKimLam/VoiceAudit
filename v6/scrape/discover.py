# -*- coding: utf-8 -*-
"""KHAM PHA ung vien (CHI METADATA, khong tai am thanh) - buoc 1 cua goi cao du lieu.

  tabs    liet ke phang tab /videos va /streams cua kenh trong whitelist (khong co giay phep) -> meta/flat.jsonl
          (tab da liet ke bi bo qua TRU KHI lan truoc cham tran --per-tab cu va --per-tab moi lon hon, hoac --refresh)
  meta    lay metadata DAY DU (co truong license) cho mau ngau nhien k video / kenh tu flat.jsonl -> meta/candidates.jsonl
          (mau ngau nhien KHONG loc truoc theo CC -> ty le qua giay phep la uoc luong khong chech cua tung tab)
  search  quet tim YouTube bo loc CC (+ > 20 phut: sp=EgQYAjAB) theo tu khoa -> meta/search.jsonl; 'agg' gom theo kenh
          (chi de de xuat kenh moi cho whitelist; kenh ngoai whitelist bi cong chan)
  ia      archive.org advancedsearch + /metadata tung muc -> ung vien theo TUNG tep goc (licenseurl, uploader, creator,
          uploader_is_creator = gate.ia_uploader_is_creator; truy van mac dinh IA_QUERY_BY = CHI BY / BY-SA / CC0)
  ccc     media.ccc.de: hoi nghi -> su kien -> ban ghi audio/opus|mpeg dung ngon ngu goc -> ung vien (--lang: chi
          su kien co ngon ngu goc khop; vd 37c3 nhieu bai tieng Duc)
  fosdem  video.fosdem.org/<nam>/<phong>/: liet ke tep .av1.webm (nho nhat, audio Opus) -> ffprobe tu xa (chi doc dau
          tep) lay do dai + codec -> ung vien (giay phep CC BY 2.0 BE cap trang; nhom = <nam>/<phong>);
          --room all = moi phong cua nam (doc trang chi muc nam); 'fosdem-rooms --year N' chi in danh sach phong

Moi lenh ghi tiep (bo qua id da co), co ngan sach thoi gian --budget-min (mac dinh 8 phut, hop khoa run_serial).
Khong cookie, khong dang nhap, nghi giua yeu cau (yt-dlp sleep_interval_requests 1,5 s; archive.org / ccc 1 s).

CLI:
  python -m v6.scrape.discover tabs   --lang vi --max-channels 20 --per-tab 60 [--prio 3] [--refresh]
  python -m v6.scrape.discover meta   --lang vi --per-channel 6 [--seed 7] [--budget-min 8]
  python -m v6.scrape.discover search --q "buổi học trực tuyến" --q "hội thảo" [--n 200]
  python -m v6.scrape.discover agg
  python -m v6.scrape.discover ia     [--query '<advancedsearch>'] --rows 20     (mac dinh IA_QUERY_BY)
  python -m v6.scrape.discover ccc    --conf 37c3 --lang en --max-events 20
  python -m v6.scrape.discover fosdem --year 2024 --room ua2220 --room h1302 --max-per-room 3   (--room all)
  python -m v6.scrape.discover fosdem-rooms --year 2024"""
from __future__ import annotations
import os

import argparse
import html
import json
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

CC_LONG_SP = "EgQYAjAB"       # Creative Commons + > 20 phut (VERIFIED trong audit)
CC_ANY_SP = "EgIwAQ%3D%3D"     # Creative Commons, moi do dai
DESC_KEEP = 1500
AUDIO_KEYS = ("format_id", "ext", "acodec", "abr", "asr", "audio_channels", "format_note", "language")
MEDIA_EXT = (".mp3", ".m4a", ".ogg", ".opus", ".flac", ".wav", ".mp4", ".mkv", ".webm", ".mov", ".mpg", ".avi")


class Budget:
    def __init__(self, minutes: float):
        self.t0, self.lim = time.time(), minutes * 60

    def over(self) -> bool:
        return time.time() - self.t0 > self.lim


# ---------------- YouTube ----------------
def _flat_rows(info: Mapping[str, Any], cid: str, tab: str, wl_row: Mapping[str, Any]) -> list[dict]:
    rows = []
    for e in info.get("entries") or []:
        if not e or not e.get("id"):
            continue
        rows.append({"id": e["id"], "source": "yt", "channel_id": cid, "channel": info.get("channel") or info.get("uploader"),
                     "tab": tab, "lang": wl_row.get("lang"), "pool": wl_row.get("pool"), "title": e.get("title"),
                     "duration": e.get("duration"), "live_status": e.get("live_status"),
                     "availability": e.get("availability"), "retrieved_at": U.now_iso()})
    return rows


OLD_PER_TAB = 60     # dong flat.jsonl truoc 12/10 khong ghi per_tab: pilot dung --per-tab 60


def tabs_to_list(flat: list[Mapping[str, Any]], per_tab: int, refresh: bool = False) -> dict[tuple, str]:
    """(kenh, tab) da liet ke -> 'skip' (da du: lan truoc lay het tab, it hon tran) | 'deepen' (lan truoc cham tran cu
    va tran moi lon hon -> liet ke lai sau hon) | 'refresh' (--refresh). Tab chua liet ke khong co trong ket qua."""
    n_ids: dict[tuple, set] = defaultdict(set)
    cap: dict[tuple, int] = {}
    for r in flat:
        k = (r.get("channel_id"), r.get("tab"))
        if r.get("id"):
            n_ids[k].add(r["id"])
        cap[k] = max(cap.get(k, 0), int(r.get("per_tab") or OLD_PER_TAB))
    out = {}
    for k, c in cap.items():
        if refresh:
            out[k] = "refresh"
        elif len(n_ids[k]) >= c and per_tab > c:
            out[k] = "deepen"
        else:
            out[k] = "skip"
    return out


def cmd_tabs(a: argparse.Namespace) -> None:
    yt = U.import_ytdlp()
    wl = U.load_whitelist(lang=a.lang)
    chans = sorted((r for r in wl.values() if r["prio"] <= a.prio and r["owner_confidence"] in ("medium", "high")),
                   key=lambda r: (r["prio"], -float(r.get("cc_lb_hours_first100") or 0)))[:a.max_channels]
    state = tabs_to_list(U.read_jsonl(U.FLAT), a.per_tab, a.refresh)
    bud = Budget(a.budget_min)
    opts = U.ydl_opts(extract_flat="in_playlist", playlistend=a.per_tab, skip_download=True)
    with yt.YoutubeDL(opts) as ydl:
        for w in chans:
            for tab in w["tabs"]:
                if state.get((w["channel_id"], tab)) == "skip":
                    continue
                if bud.over():
                    print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
                    return
                try:
                    info = ydl.extract_info(f"https://www.youtube.com/channel/{w['channel_id']}/{tab}", download=False)
                    rows = _flat_rows(info, w["channel_id"], tab, w)
                except Exception as ex:  # noqa: BLE001  (mot kenh loi khong dung ca buoc)
                    rows = []
                    print(f"  LOI {w['channel_id']} {tab}: {str(ex)[:160]}", flush=True)
                if not rows:   # ghi dong danh dau de khong hoi lai tab rong / loi
                    rows = [{"id": None, "channel_id": w["channel_id"], "tab": tab, "lang": w["lang"], "empty": True,
                             "retrieved_at": U.now_iso()}]
                rows = [{**r, "per_tab": a.per_tab} for r in rows]   # tran da dung (de biet tab da het hay bi cat)
                U.append_jsonl(U.FLAT, rows)
                hrs = sum((r.get("duration") or 0) for r in rows) / 3600
                print(f"  {w['lang']} {w['channel_name'][:30]:30s} {tab:7s} n={len(rows) if rows[0].get('id') else 0:4d} "
                      f"{hrs:6.1f} h", flush=True)


def candidate_from_info(info: Mapping[str, Any], flat: Mapping[str, Any]) -> dict:
    """Ban ghi ung vien tu metadata day du cua yt-dlp (giu truong can cho cong + nguon goc)."""
    auds = [{k: f.get(k) for k in AUDIO_KEYS} for f in info.get("formats") or []
            if f.get("vcodec") in (None, "none") and f.get("acodec") not in (None, "none")]
    return {"id": info.get("id"), "source": "yt", "url": info.get("webpage_url") or f"https://www.youtube.com/watch?v={info.get('id')}",
            "channel_id": info.get("channel_id"), "channel": info.get("channel"), "uploader_id": info.get("uploader_id"),
            "title": info.get("title"), "description": (info.get("description") or "")[:DESC_KEEP],
            "license": info.get("license"), "live_status": info.get("live_status"), "was_live": info.get("was_live"),
            "duration": info.get("duration"), "upload_date": info.get("upload_date"),
            "categories": info.get("categories") or [], "tags": (info.get("tags") or [])[:15],
            "language": info.get("language"), "availability": info.get("availability"),
            "chapters": len(info.get("chapters") or []), "audio_formats": auds,
            "tab": flat.get("tab"), "lang": flat.get("lang"), "pool": flat.get("pool"),
            "retrieved_at": U.now_iso()}


def pick_sample(flat: list[Mapping[str, Any]], per_channel: int, seed: int, done: set[str],
                prefer: set[str] | None = None, channels: set[str] | None = None) -> list[dict]:
    """Mau ngau nhien moi kenh (gop 2 tab), chi loai do dai ngoai [600, 21600] khi da biet - KHONG loc theo CC.
    prefer: id uu tien len dau moi kenh (vd ket qua tim CC; mau khi do CHECH, khong dung uoc ty le CC).
    channels: chi cac kenh nay."""
    by_ch: dict[str, list] = defaultdict(list)
    for r in flat:
        if not r.get("id") or r["id"] in done or (channels and r.get("channel_id") not in channels):
            continue
        d = r.get("duration")
        if d is not None and not 600 <= d <= 21600:
            continue
        by_ch[r["channel_id"]].append(r)
    rng = random.Random(seed)
    out = []
    n_done = Counter(r.get("channel_id") for r in flat if r.get("id") in done)
    for cid in sorted(by_ch):
        pool = sorted(by_ch[cid], key=lambda r: r["id"])
        rng.shuffle(pool)
        if prefer:
            pool.sort(key=lambda r: r["id"] not in prefer)          # sap xep on dinh: id uu tien len dau
        out.extend(pool[:max(0, per_channel - n_done[cid])])
    return out


def cmd_meta(a: argparse.Namespace) -> None:
    yt = U.import_ytdlp()
    flat = [r for r in U.latest_by(U.read_jsonl(U.FLAT)).values() if (not a.lang or r.get("lang") == a.lang)]  # 1 dong / id
    cands = U.read_jsonl(U.CANDIDATES)
    done = {r["id"] for r in cands if r.get("id") and (not r.get("error") or not a.retry_errors)}
    prefer = {r["id"] for r in U.read_jsonl(U.SEARCH) if r.get("id")} if a.prefer_search else None
    todo = pick_sample(flat, a.per_channel, a.seed, done, prefer, set(a.channel) or None)
    print(f"can lay metadata day du: {len(todo)} video", flush=True)
    bud = Budget(a.budget_min)
    with yt.YoutubeDL(U.ydl_opts(skip_download=True)) as ydl:
        for i, fr in enumerate(todo):
            if bud.over():
                print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
                break
            try:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={fr['id']}", download=False, process=False)
                rec = candidate_from_info(info, fr)
            except Exception as ex:  # noqa: BLE001
                rec = {"id": fr["id"], "source": "yt", "channel_id": fr["channel_id"], "lang": fr.get("lang"),
                       "tab": fr.get("tab"), "title": fr.get("title"), "duration": fr.get("duration"),
                       "error": str(ex)[:300], "retrieved_at": U.now_iso()}
            U.append_jsonl(U.CANDIDATES, [rec])
            print(f"  [{i + 1}/{len(todo)}] {rec['id']} lic={str(rec.get('license'))[:40]!r} live={rec.get('live_status')} "
                  f"dur={rec.get('duration')} {'ERR ' + rec['error'][:60] if rec.get('error') else ''}", flush=True)


def search_url(q: str, any_duration: bool = False) -> tuple[str, str]:
    sp = CC_ANY_SP if any_duration else CC_LONG_SP
    return f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(q)}&sp={sp}", sp


def search_rows(info: Mapping[str, Any], q: str, sp: str) -> list[dict]:
    """Ket qua tim phang -> dong search.jsonl (bo muc khong id / kenh-playlist; KHONG co giay phep: chi goi y kenh)."""
    rows = []
    for e in info.get("entries") or []:
        if not e or not e.get("id") or e.get("ie_key") not in (None, "Youtube") or not e.get("channel_id"):
            continue
        rows.append({"id": e["id"], "channel_id": e.get("channel_id"), "channel": e.get("channel") or e.get("uploader"),
                     "duration": e.get("duration"), "title": e.get("title"), "query": q, "sp": sp,
                     "retrieved_at": U.now_iso()})
    return rows


def cmd_search(a: argparse.Namespace) -> None:
    yt = U.import_ytdlp()
    bud = Budget(a.budget_min)
    with yt.YoutubeDL(U.ydl_opts(extract_flat="in_playlist", playlistend=a.n, skip_download=True)) as ydl:
        for q in a.q:
            if bud.over():
                break
            url, sp = search_url(q, a.any_duration)
            rows = search_rows(ydl.extract_info(url, download=False), q, sp)
            U.append_jsonl(U.SEARCH, rows)
            print(f"  {q!r}: {len(rows)} ket qua", flush=True)


def cmd_agg(a: argparse.Namespace) -> None:
    wl = U.load_whitelist()
    agg: dict[str, Counter] = defaultdict(Counter)
    names = {}
    for r in U.read_jsonl(U.SEARCH):
        agg[r.get("channel_id")]["n"] += 1
        agg[r.get("channel_id")]["sec"] += r.get("duration") or 0
        names[r.get("channel_id")] = r.get("channel")
    for cid, c in sorted(agg.items(), key=lambda kv: -kv[1]["sec"])[:a.top]:
        print(f"{'WL' if cid in wl else '--'} {cid} {c['n']:4d} {c['sec'] / 3600:7.1f} h  {names.get(cid)}")


# ---------------- archive.org / media.ccc.de ----------------
def _get_json(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": U.USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (chi https co dinh)
        return json.loads(resp.read().decode("utf-8"))


def _length_s(v: Any) -> float | None:
    try:
        if ":" in str(v):
            s = 0.0
            for p in str(v).split(":"):
                s = s * 60 + float(p)
            return s
        return float(v)
    except (TypeError, ValueError):
        return None


# archive.org mac dinh (quyet dinh 1): CHI BY / BY-SA / CC0, audio, bai giang / hop / hoi thao (dau '/' phai thoat)
IA_QUERY_BY = (r"mediatype:audio AND (licenseurl:*\/licenses\/by\/* OR licenseurl:*\/licenses\/by-sa\/* OR "
               r"licenseurl:*\/publicdomain\/zero\/*) AND (subject:lecture OR subject:lectures OR subject:meeting OR "
               r"subject:seminar OR subject:talk OR subject:conference)")


def ia_candidates(ident: str, md_json: Mapping[str, Any], lang: str) -> list[dict]:
    """Ung vien theo tung tep GOC (source=original) cua mot muc archive.org (+ uploader_is_creator cho cong A1)."""
    from v6.scrape.gate import ia_uploader_is_creator
    md = md_json.get("metadata", {})
    out = []
    for f in md_json.get("files", []):
        name = str(f.get("name", ""))
        if f.get("source") != "original" or not name.lower().endswith(MEDIA_EXT):
            continue
        creator, uploader = md.get("creator"), md.get("uploader")
        cr = creator if isinstance(creator, str) else str(creator) if creator else None
        out.append({"id": f"ia:{ident}/{name}", "source": "ia", "url": f"https://archive.org/download/{ident}/{urllib.parse.quote(name)}",
                    "channel_id": f"ia:{ident}", "channel": cr, "creator": cr, "collection": md.get("collection"),
                    "uploader": uploader, "uploader_is_creator": ia_uploader_is_creator(cr, uploader),
                    "title": f"{md.get('title')} | {name}", "description": str(md.get("description") or "")[:DESC_KEEP],
                    "licenseurl": md.get("licenseurl"), "duration": _length_s(f.get("length")), "size": int(f.get("size") or 0),
                    "format": f.get("format"), "md5": f.get("md5"), "upload_date": md.get("publicdate"),
                    "language": None, "live_status": "not_live", "lang": lang, "availability": "public",
                    "retrieved_at": U.now_iso()})
    return out


def cmd_ia(a: argparse.Namespace) -> None:
    params = [("q", a.query), ("rows", str(a.rows)), ("output", "json"), ("fl[]", "identifier")]
    data = _get_json("https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params))
    idents = [d["identifier"] for d in data.get("response", {}).get("docs", [])]
    done = {r["channel_id"] for r in U.read_jsonl(U.CANDIDATES) if r.get("source") == "ia"}
    bud = Budget(a.budget_min)
    for ident in idents:
        if f"ia:{ident}" in done or bud.over():
            continue
        try:
            rows = ia_candidates(ident, _get_json(f"https://archive.org/metadata/{ident}"), a.lang)
        except Exception as ex:  # noqa: BLE001
            rows = [{"id": f"ia:{ident}", "source": "ia", "channel_id": f"ia:{ident}", "error": str(ex)[:200],
                     "lang": a.lang, "retrieved_at": U.now_iso()}]
        U.append_jsonl(U.CANDIDATES, rows)
        print(f"  ia {ident}: {len(rows)} tep goc", flush=True)
        time.sleep(1.0)


def ccc_lang(orig: Any) -> str | None:
    """'eng' -> 'en', 'deu' -> 'de'; song ngu ('deu-eng') hoac rong -> None (khong dung)."""
    o = str(orig or "")
    if not o or "-" in o:
        return None
    return {"eng": "en", "deu": "de", "vie": "vi"}.get(o, o[:2])


def ccc_candidate(ev: Mapping[str, Any], conf: str) -> dict | None:
    """Ban ghi audio dung ngon ngu goc (opus uu tien, roi mp3)."""
    orig = ev.get("original_language")
    recs = [r for r in ev.get("recordings") or [] if r.get("mime_type") in ("audio/opus", "audio/mpeg")
            and (not orig or r.get("language") == orig)]
    if not recs:
        return None
    recs.sort(key=lambda r: r.get("mime_type") != "audio/opus")
    r = recs[0]
    lang = ccc_lang(orig)
    return {"id": f"ccc:{ev.get('guid')}", "source": "ccc", "url": r.get("recording_url"), "page": ev.get("frontend_link"),
            "channel_id": f"ccc:{conf}", "channel": f"media.ccc.de {conf}", "title": ev.get("title"),
            "description": str(ev.get("description") or "")[:DESC_KEEP], "duration": ev.get("length") or r.get("length"),
            "language": orig, "lang": lang, "mime": r.get("mime_type"), "size": int((r.get("size") or 0) * 1e6),
            "persons": ev.get("persons"), "upload_date": ev.get("release_date"), "live_status": "not_live",
            "availability": "public", "retrieved_at": U.now_iso()}


def cmd_ccc(a: argparse.Namespace) -> None:
    conf = _get_json(U.OPEN_SOURCES["ccc"]["api"] + urllib.parse.quote(a.conf))
    done = {r["id"] for r in U.read_jsonl(U.CANDIDATES)}
    n = seen = 0
    bud = Budget(a.budget_min)
    events = list(conf.get("events") or [])
    if a.lang:   # loc truoc theo ngon ngu goc co san trong danh sach su kien (khong goi API tung su kien)
        events = [e for e in events if ccc_lang(e.get("original_language")) == a.lang]
    for e in events:
        if bud.over() or n >= a.max_events:
            break
        if f"ccc:{e.get('guid')}" in done:
            continue
        seen += 1
        ev = _get_json(e["url"])
        c = ccc_candidate(ev, a.conf)
        time.sleep(1.0)
        if c and (not a.lang or c["lang"] == a.lang):
            U.append_jsonl(U.CANDIDATES, [c])
            n += 1
    print(f"ccc {a.conf}: {n} ung vien moi / {seen} su kien da hoi (loc lang={a.lang})")


# ---------------- FOSDEM ----------------
_FOSDEM_ROW = re.compile(r'<td class="link"><a href="([^"]+)"[^>]*>[^<]*</a></td><td class="size">([^<]*)</td>')
_AV1 = re.compile(r"\.av1\.webm$")


def fosdem_list(index_html: str) -> list[tuple[str, str]]:
    """(ten tep, kich thuoc) cac tep .av1.webm trong trang chi muc mot phong (bo thu muc cha / thu muc con)."""
    out = []
    for href, size in _FOSDEM_ROW.findall(index_html):
        href = html.unescape(href)
        if _AV1.search(href) and "/" not in href:
            out.append((href, size.strip()))
    return out


def size_bytes(s: str) -> int:
    """'77.6 MiB' -> so byte (0 neu khong doc duoc)."""
    m = re.match(r"([\d.]+)\s*([KMG])i?B", s or "")
    if not m:
        return 0
    return int(float(m.group(1)) * {"K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30}[m.group(2)])


def ffprobe_remote(url: str, timeout: int = 60) -> dict:
    """Do dai + luong audio cua tep tu xa (ffprobe chi doc phan dau WebM)."""
    cmd = [U.ffmpeg_bin("ffprobe"), "-v", "error", "-user_agent", U.USER_AGENT, "-show_entries",
           "format=duration:stream=codec_type,codec_name,sample_rate,channels", "-of", "json", url]
    out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
    return json.loads(out.stdout.decode("utf-8", "replace"))


def fosdem_candidate(year: int, room: str, name: str, size: str, probe: Mapping[str, Any]) -> dict:
    auds = [s for s in probe.get("streams", []) if s.get("codec_type") == "audio"]
    talk = _AV1.sub("", name)
    dur = probe.get("format", {}).get("duration")
    return {"id": f"fosdem:{year}/{room}/{talk}", "source": "fosdem",
            "url": f"{U.OPEN_SOURCES['fosdem']['root']}{year}/{room}/{urllib.parse.quote(name)}",
            "page": f"https://fosdem.org/{year}/schedule/", "channel_id": f"fosdem:{year}/{room}",
            "channel": f"FOSDEM {year} {room}", "title": talk, "description": "", "duration": float(dur) if dur else None,
            "size": size_bytes(size), "acodec": auds[0].get("codec_name") if auds else None,
            "n_audio_streams": len(auds), "language": "en", "lang": "en", "live_status": "not_live",
            "availability": "public", "upload_date": str(year), "retrieved_at": U.now_iso()}


_FOSDEM_DIR = re.compile(r'<td class="link"><a href="([A-Za-z0-9_.-]+)/"')


def fosdem_rooms(index_html: str) -> list[str]:
    """Ten phong (thu muc con) trong trang chi muc mot nam video.fosdem.org/<nam>/ (bo '..')."""
    return sorted({html.unescape(d) for d in _FOSDEM_DIR.findall(index_html) if d not in ("..", ".")})


def _get_text(url: str, timeout: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": U.USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 (https co dinh)
        return resp.read().decode("utf-8", "replace")


def cmd_fosdem_rooms(a: argparse.Namespace) -> None:
    rooms = fosdem_rooms(_get_text(f"{U.OPEN_SOURCES['fosdem']['root']}{a.year}/"))
    print(f"fosdem {a.year}: {len(rooms)} phong")
    print(" ".join(rooms))


def cmd_fosdem(a: argparse.Namespace) -> None:
    root = U.OPEN_SOURCES["fosdem"]["root"]
    done = {r["id"] for r in U.read_jsonl(U.CANDIDATES)}
    bud = Budget(a.budget_min)
    rooms = a.room
    if "all" in rooms:
        rooms = fosdem_rooms(_get_text(f"{root}{a.year}/"))
        print(f"fosdem {a.year}: {len(rooms)} phong", flush=True)
    for room in rooms:
        if bud.over():
            print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
            break
        try:
            files = fosdem_list(_get_text(f"{root}{a.year}/{room}/"))
        except OSError as ex:     # phong khong co trang chi muc / loi mang -> bo phong, khong dung ca lenh
            print(f"  LOI phong {room}: {str(ex)[:120]}", flush=True)
            continue
        n = 0
        for name, size in files:
            if n >= a.max_per_room or bud.over():
                break
            talk = _AV1.sub("", name)
            if f"fosdem:{a.year}/{room}/{talk}" in done:
                continue
            url = f"{root}{a.year}/{room}/{urllib.parse.quote(name)}"
            try:
                c = fosdem_candidate(a.year, room, name, size, ffprobe_remote(url))
            except (subprocess.SubprocessError, OSError, json.JSONDecodeError) as ex:
                c = {"id": f"fosdem:{a.year}/{room}/{talk}", "source": "fosdem", "channel_id": f"fosdem:{a.year}/{room}",
                     "lang": "en", "error": str(ex)[:200], "retrieved_at": U.now_iso()}
            U.append_jsonl(U.CANDIDATES, [c])
            n += 1
            print(f"  fosdem {room} {talk[:60]} dur={c.get('duration')} {c.get('acodec')} {size}", flush=True)
            time.sleep(1.0)
        print(f"fosdem {a.year}/{room}: {n} ung vien moi ({len(files)} tep av1.webm)", flush=True)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("tabs")
    p.add_argument("--lang", choices=["vi", "en"], required=True)
    p.add_argument("--max-channels", type=int, default=20)
    p.add_argument("--per-tab", type=int, default=60)
    p.add_argument("--prio", type=int, default=3)
    p.add_argument("--refresh", action="store_true", help="liet ke lai ca tab da co (lay video moi dang)")
    p = sub.add_parser("meta")
    p.add_argument("--lang", choices=["vi", "en"])
    p.add_argument("--per-channel", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--channel", action="append", default=[], help="chi kenh nay (lap lai duoc)")
    p.add_argument("--prefer-search", action="store_true",
                   help="uu tien video co trong search.jsonl (tim CC) - mau chech, chi de lay them video CC")
    p = sub.add_parser("search")
    p.add_argument("--q", action="append", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--any-duration", action="store_true")
    p = sub.add_parser("agg")
    p.add_argument("--top", type=int, default=40)
    p = sub.add_parser("ia")
    p.add_argument("--query", default=IA_QUERY_BY)
    p.add_argument("--rows", type=int, default=20)
    p.add_argument("--lang", choices=["vi", "en"], default="en")
    p = sub.add_parser("ccc")
    p.add_argument("--conf", required=True)
    p.add_argument("--max-events", type=int, default=20)
    p.add_argument("--lang", choices=["vi", "en", "de"])
    p = sub.add_parser("fosdem")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--room", action="append", required=True)
    p.add_argument("--max-per-room", type=int, default=3)
    p = sub.add_parser("fosdem-rooms")
    p.add_argument("--year", type=int, required=True)
    for p in sub.choices.values():
        p.add_argument("--budget-min", type=float, default=8.0)
    a = ap.parse_args()
    {"tabs": cmd_tabs, "meta": cmd_meta, "search": cmd_search, "agg": cmd_agg, "ia": cmd_ia, "ccc": cmd_ccc,
     "fosdem": cmd_fosdem, "fosdem-rooms": cmd_fosdem_rooms}[a.cmd](a)


if __name__ == "__main__":
    main()
