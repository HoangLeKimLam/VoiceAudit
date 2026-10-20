# -*- coding: utf-8 -*-
"""HA TANG DUNG CHUNG cho goi cao du lieu v6.scrape (khong nap torch, khong mang).

  duong dan   SCRAPE (data/scrape) · RAW (tep tai goc) · META (jsonl tung buoc) · FRAMES (dac trung khung) · HOSTS_DIR(lang)
  danh sach   load_whitelist() -> {channel_id: dong whitelist} (data/scrape/channels_whitelist.tsv, chep tu .tmp_v6/wf)
              OPEN_SOURCES: nguon mo ngoai YouTube (archive.org / media.ccc.de / FOSDEM) + giay phep mac dinh cua trang
  yt-dlp      import_ytdlp() -> module yt_dlp (site_scrape + yt-dlp-ejs GHIM ban 0.8.0, KHONG --remote-components)
              ydl_opts(**extra) -> tuy chon chung: node lam JS runtime, nghi giua cac yeu cau, KHONG cookie (cam)
  jsonl       read_jsonl · append_jsonl · write_jsonl · latest_by(rows, key)
  ffmpeg      ffmpeg_bin('ffmpeg'|'ffprobe') -> duong dan (PATH truoc, roi thu muc WinGet)

Quy uoc chung (quyet dinh 1 cua nguoi dung, .tmp_v6/wf/DECISIONS.md): chi video YouTube co giay phep Creative Commons
(kiem TUNG video), nguon mo archive.org CC/PD, FOSDEM CC BY 2.0, media.ccc.de CC BY; chi noi bo nghien cuu, khong phat tan;
ghi url / kenh / giay phep cho tung tep; khong cookie tai khoan.

CLI: python -m v6.scrape.util paths | whitelist [--lang vi|en] | ytdlp"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Iterable, Iterator, Mapping

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

SCRAPE = os.path.join(BASE, "data", "scrape")
RAW = os.path.join(SCRAPE, "raw")
META = os.path.join(SCRAPE, "meta")
FRAMES = os.path.join(SCRAPE, "frames")
WHITELIST = os.path.join(SCRAPE, "channels_whitelist.tsv")
WHITELIST_SRC = os.path.join(BASE, ".tmp_v6", "wf", "scrape_sources", "channels_whitelist.tsv")
SITE_DIRS = (os.path.join(BASE, ".tmp_v6", "site_scrape"), os.path.join(BASE, ".tmp_v6", "cache", "scrape_site"))
FFMPEG_DIR = ("C:\\Users\\acer\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
              "\\ffmpeg-9.0.1-full_build\\bin")
SR = 16000
USER_AGENT = "VoiceAudit-research/0.1 (internal research, no redistribution)"

# tep jsonl cua tung buoc (moi buoc ghi tiep, chay lai = bo qua id da co)
FLAT = os.path.join(META, "flat.jsonl")              # discover tabs: danh sach phang (khong co giay phep)
SEARCH = os.path.join(META, "search.jsonl")          # discover search: ket qua tim CC (chi de mo rong whitelist)
CANDIDATES = os.path.join(META, "candidates.jsonl")  # discover meta/ia/ccc: metadata day du CO truong license
GATED = os.path.join(META, "gated.jsonl")            # gate: quyet dinh + ly do
FETCHED = os.path.join(META, "fetched.jsonl")        # fetch: tep goc da tai + nguon goc
PROBED = os.path.join(META, "probed.jsonl")          # probe: ffprobe + cong container
FILTERED = os.path.join(META, "filtered.jsonl")      # filter: do noi dung + keep_spans + pool
SEGMENTS = os.path.join(META, "segments.jsonl")      # segment: doan host da cat (chua co moc tu)
ALIGNED = os.path.join(META, "aligned.jsonl")        # align: moc tu ASR cua tung doan

# Nguon mo ngoai YouTube: giay phep cap TRANG (media.ccc.de / FOSDEM khong co truong license trong API)
OPEN_SOURCES = {
    "ccc": {"license": "CC BY 4.0 (media.ccc.de default, check per-file credits)", "license_level": "site_default",
            "api": "https://api.media.ccc.de/public/conferences/"},
    "fosdem": {"license": "CC BY 2.0 BE (FOSDEM site-wide)", "license_level": "site_default",
               "root": "https://video.fosdem.org/"},
    "ia": {"license": None, "license_level": "item_licenseurl", "api": "https://archive.org/advancedsearch.php"},
}


def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


QUARANTINE = os.path.join(SCRAPE, "_quarantine")   # tep bi cach ly (CHUYEN, khong xoa) - xem v6.scrape.quarantine


def seg_wav(seg: Mapping[str, Any]) -> str | None:
    """Duong dan wav cua mot dong segments.jsonl (truong 'wav' tuong doi data/scrape); da cach ly -> tim trong
    _quarantine/<cung duong dan tuong doi>; khong co -> None."""
    rel = seg.get("wav")
    if not rel:
        return None
    for base in (SCRAPE, QUARANTINE):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            return p
    return None


def hosts_dir(lang: str, tier: str = "std") -> str:
    """Thu muc host (giao dien I4): data/scrape/hosts_<lang>; host dai 30-120 s: data/scrape/hosts_long_<lang>."""
    if lang not in ("vi", "en"):
        raise ValueError(f"lang phai la vi|en, nhan {lang!r}")
    return os.path.join(SCRAPE, f"hosts_{lang}" if tier == "std" else f"hosts_long_{lang}")


def slug(s: str, n: int = 48) -> str:
    """Ten an toan cho tep (ASCII, gach duoi)."""
    import unicodedata
    t = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
    return (t or "x")[:n]


# ---------------- jsonl ----------------
def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:   # dong cuoi bi cat do ngat giua chung -> bo qua, khong lam hong ca tep
                continue
    return out


def append_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
        fh.flush()
    return n


def write_jsonl(path: str, rows: Iterable[Mapping[str, Any]]) -> int:
    """Ghi de nguyen tu (tmp + os.replace, thu lai khi Windows giu tep)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")
            n += 1
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return n
        except PermissionError:
            time.sleep(0.5 + 0.5 * attempt)
    raise PermissionError(f"khong ghi duoc {path}")


def latest_by(rows: Iterable[Mapping[str, Any]], key: str = "id") -> dict[str, dict]:
    """Dong moi nhat theo khoa (tep ghi tiep: dong sau ghi de dong truoc)."""
    out: dict[str, dict] = {}
    for r in rows:
        k = r.get(key)
        if k is not None:
            out[str(k)] = dict(r)
    return out


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"khong JSON hoa duoc {type(o).__name__}")


# ---------------- whitelist ----------------
def load_whitelist(path: str = WHITELIST, lang: str | None = None) -> dict[str, dict]:
    """channel_id -> dong whitelist (lang, pool, prio:int, tabs:list, owner_confidence, ...). Thieu ban chep -> chep tu wf."""
    if not os.path.exists(path) and os.path.exists(WHITELIST_SRC):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(WHITELIST_SRC, path)
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if not r.get("channel_id"):
                continue
            if lang and r.get("lang") != lang:
                continue
            r = dict(r)
            r["prio"] = int(r.get("prio") or 9)
            r["tabs"] = [t for t in (r.get("tab_to_scrape") or "videos").split("+") if t in ("videos", "streams")]
            out[r["channel_id"]] = r
    return out


# ---------------- yt-dlp ----------------
def import_ytdlp():
    """Nap yt_dlp tu .tmp_v6/site_scrape va yt-dlp-ejs GHIM (khong tai JS tu GitHub luc chay)."""
    for d in SITE_DIRS:
        if d not in sys.path:
            sys.path.insert(0, d)
    import yt_dlp  # noqa: WPS433
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError as e:  # khong co bo giai EJS ghim -> dung, khong roi ve remote components
        raise RuntimeError("thieu yt-dlp-ejs ghim trong .tmp_v6/cache/scrape_site") from e
    return yt_dlp


FORBIDDEN_OPTS = ("cookiefile", "cookiesfrombrowser", "username", "password", "usenetrc", "remote_components")


def ydl_opts(**extra: Any) -> dict[str, Any]:
    """Tuy chon yt-dlp chung. Cam cookie / dang nhap / remote_components (rui ro chuoi cung ung)."""
    bad = [k for k in extra if k in FORBIDDEN_OPTS]
    if bad:
        raise ValueError(f"tuy chon bi cam (quyet dinh 1 + critic): {bad}")
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "cachedir": False,
        "js_runtimes": {"node": {}},
        "sleep_interval_requests": 1.5,
        "socket_timeout": 30,
        "retries": 10,
        "extractor_retries": 3,
        "noprogress": True,
        "http_headers": {"Accept-Language": "en-US,en;q=0.8"},
    }
    opts.update(extra)
    return opts


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    p = shutil.which(name)
    if p:
        return p
    cand = os.path.join(FFMPEG_DIR, name + ".exe")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"khong thay {name} (PATH hoac {FFMPEG_DIR})")


def iter_ids_from_file(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                yield ln.split()[0]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["paths", "whitelist", "ytdlp"])
    ap.add_argument("--lang", choices=["vi", "en"])
    a = ap.parse_args()
    if a.cmd == "paths":
        for k in ("SCRAPE", "RAW", "META", "FRAMES", "WHITELIST"):
            print(f"{k:10s} {globals()[k]}")
    elif a.cmd == "whitelist":
        wl = load_whitelist(lang=a.lang)
        for cid, r in sorted(wl.items(), key=lambda kv: (kv[1]["lang"], kv[1]["prio"])):
            print(r["lang"], r["prio"], cid, r["owner_confidence"], "+".join(r["tabs"]), r["channel_name"])
        print(len(wl), "kenh")
    else:
        y = import_ytdlp()
        import yt_dlp_ejs
        print("yt-dlp", y.version.__version__, "| ejs", getattr(yt_dlp_ejs, "__version__", "?"), yt_dlp_ejs.__file__)


if __name__ == "__main__":
    main()
