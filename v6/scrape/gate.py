# -*- coding: utf-8 -*-
"""CONG METADATA (truoc khi tai) cho du lieu cao - filter_spec A1-A6 cua audit1_scrape + sua cua critic.

  A1 giay phep   (quyet dinh 1: CHI CC BY / BY-SA / CC0; NC / ND / PD-mark / dang lai = tuy chon --allow, MAC DINH TAT)
                 YouTube: truong license PHAI chua 'Creative Commons' (kiem TUNG video, ke ca kenh trong whitelist:
                 kenh tron giay phep - vd /streams 0/6 CC); kenh co dieu khoan trang NC-SA (Open Yale Courses, NPTEL)
                 -> bo ('license_nc_site_terms') tru khi --allow nc_nd.
                 archive.org: licenseurl phai la licenses/by/, licenses/by-sa/ hoac publicdomain/zero/; by-nc*, by-nd*
                 -> bo tru khi --allow nc_nd (ND cam tao ban phai sinh = moi phep cat ghep cua build_sets);
                 publicdomain/mark -> bo tru khi --allow pd_mark; nguoi dang KHONG chung minh duoc la tac gia
                 (ia_uploader_is_creator: ten mien email nguoi dang khop ten tac gia / nguoi dang da xac minh tay;
                 False -> 'ia_uploader_not_creator', khong ro -> 'ia_uploader_unknown') -> bo tru khi --allow ia_reupload
                 (giay phep do nguoi dang lai dat, khong dang tin; vd ucberkeley-webcast dang lai boi ca nhan).
                 media.ccc.de / FOSDEM: giay phep cap trang (co co 'license_site_default').
                 Kenh YouTube phai co trong whitelist voi owner_confidence >= medium (chu so huu that, khong dang lai).
  A2 live        bo is_live / is_upcoming / post_live; uu tien was_live roi not_live (was_live KHONG chung minh chua cat -
                 chi la diem uu tien; cong cat that o v6.scrape.filter).
  A3 do dai      600 - 21 600 s.
  A4 noi dung    regex tieu de (day du) + mo ta (tap manh) tu filter_spec; tre em -> bo; ton giao nhay cam (nghi le tang,
                 cau sieu, tung kinh...) -> bo; dau hieu hoc sinh vi thanh nien -> co 'minor_risk' (hosts loai mac dinh).
                 Tre em theo cach noi huong toi phu huynh / kem rieng: CHILD_HARD ('cung con', '1 thay kem 1 tro',
                 '1 kem 1') trong tieu de -> bo 'children_content'; CHILD_SOFT ('phu huynh', 'bo me', 'ba me', 'gia su',
                 'lop 1-1', 'cho con', 'N kem M') -> >= 2 cum KHAC nhau trong tieu de -> bo, 1 cum -> co 'child_risk'
                 (hop phu huynh / hoi thao cha me la loi nguoi lon: khong bo han); mo ta co cum nao -> co 'child_risk'
                 (hosts loai mac dinh). So khop tren NFKC (mo ta hay dung chu Unicode dam toan hoc, vd PANTADO).
                 Bai giang / lop hoc ton giao (danh sach kenh + regex ten kenh / ghi chu / tieu de) -> co 'religious_teaching':
                 khong bo o cong, nhung fetch xep SAU moi kenh khac va hosts loai mac dinh cho den khi nguoi dung xac nhan.
  A5 the loai    bo Music / Film & Animation; > 15 chuong -> co 'many_chapters' (co the da bien tap).
  A6 dinh dang   phai co 251 hoac 140 (hoac 251-N/140-N co 'original'), khong bao gio ban -drc.
  khac           availability != public (hoi vien, khong phat duoc) -> bo; language lech whitelist -> bo.

Ham chinh: gate_record(rec, whitelist) -> {id, ok, reasons, flags, prefer, license_effective}.
current(): cong TINH LAI trong bo nho tu candidates.jsonl (segment / hosts dung ham nay, KHONG tin gated.jsonl cu - sua
  review 12/10: gated.jsonl de lai tu truoc khi doi chinh sach van cho qua 94 ung vien NC-ND / NC site terms).
CLI: python -m v6.scrape.gate [--candidates F] [--out F] [--allow nc_nd|pd_mark|ia_reupload ...]
     (doc candidates.jsonl, ghi gated.jsonl + gate_stats.json; --allow chi bat khi nguoi dung da dong y)"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

MIN_DUR, MAX_DUR = 600, 21600
MAX_CHAPTERS = 15
CONF_OK = ("medium", "high")
BAD_LIVE = ("is_live", "is_upcoming", "post_live")
BAD_CATS = ("Music", "Film & Animation")
GOOD_AVAIL = (None, "public", "unlisted")

# A4 nguyen van filter_spec (tieu de)
TITLE_RE = re.compile(
    r"tổng hợp|compilation|highlight|trailer|teaser|recap|montage|best of|karaoke|nhạc|music|\bMV\b|cover|remix|"
    r"audio truyện|truyện audio|sách nói|đọc truyện|kể truyện|podcast|lồng tiếng|thuyết minh|vietsub|song ngữ|dubbed|"
    r"AI voice|giọng đọc|tụng kinh|chanting|niệm Phật|cầu siêu|ASMR|study with me|reaction|full tập|phim", re.I)
# mo ta: chi tap MANH (mo ta hay nhac 'podcast' / 'sach noi' o phan lien ket kenh khac -> khong dung ca regex tieu de;
# pilot 10/10: 'sách nói' trong mo ta Hoi Thanh Nha Muon Dan la lien ket, chan nham 5 buoi nhom Zoom)
DESC_RE = re.compile(
    r"karaoke|lồng tiếng|thuyết minh|AI voice|giọng đọc AI|giọng đọc máy|text to speech|\bTTS\b|voice ?over|dubbed by|"
    r"compilation of|tổng hợp các", re.I)
# mo ta: tre em chi theo tu MANH (bai giang ve giao duc / phap ly hay nhac 'children' -> khong bo)
CHILD_DESC_RE = re.compile(r"thiếu nhi|mầm non|mẫu giáo|nhà trẻ|preschool|kindergarten|for kids|cho bé", re.I)
# tre em: noi dung cho / cua tre em -> bo han ('lop 1-1' = hoc kem mot-mot, khong phai lop 1 -> CHILD_SOFT)
CHILD_RE = re.compile(
    r"thiếu nhi|mầm non|mẫu giáo|nhà trẻ|tiểu học|\blớp [1-5]\b(?!\s?-\s?1\b)|\bbé\b|các bé|cho bé|trẻ em|"
    r"\bkids?\b|children|preschool|kindergarten|toddler|nursery|elementary school", re.I)
# hoc sinh co the vi thanh nien: KHONG bo, chi co (loai khoi moi tap chia se; quyet dinh cua nguoi dung)
MINOR_RE = re.compile(r"\blớp (?:[6-9]|1[0-2])\b|THCS|THPT|học sinh|luyện thi|vào 10|scratch|\bTHT\b|"
                      r"high school|middle school|students? (?:aged|under)", re.I)
# ton giao nhay cam: nghi le / tang le (bai giang, lop hoc, hoi dap ton giao van giu - xem notes)
RELIG_SENS_RE = re.compile(r"cầu siêu|cầu an|tang lễ|đám tang|lễ tang|an táng|truy điệu|tụng kinh|niệm Phật|"
                           r"lên đồng|hầu đồng|trừ tà|funeral|memorial service|exorcism|requiem|rosary", re.I)
# archive.org licenseurl (A1, quyet dinh 1): mac dinh CHI BY / BY-SA / CC0
IA_LIC_RE = re.compile(r"creativecommons\.org/(?:licenses/(?:by|by-sa)/|publicdomain/zero/)", re.I)
IA_LIC_NCND_RE = re.compile(r"creativecommons\.org/licenses/by-(?:nc|nd|nc-sa|nc-nd)/", re.I)
IA_LIC_PDMARK_RE = re.compile(r"creativecommons\.org/publicdomain/mark/", re.I)
ALLOW_KEYS = ("nc_nd", "pd_mark", "ia_reupload")
# archive.org: nguoi dang da xac minh tay la chu so huu noi dung (email -> ghi chu); trong = chua xac minh ai
IA_VERIFIED_UPLOADERS: dict[str, str] = {}
# tu chung chung khong du chung minh chu so huu khi khop ten mien (vd 'city' trong cityofx.gov cho 'City of Y')
IA_GENERIC_TOKENS = {"city", "town", "county", "state", "university", "college", "school", "council", "media", "channel",
                     "official", "public", "library", "archive", "archives", "center", "centre", "institute", "department",
                     "office", "government", "news", "radio", "video", "audio", "lecture", "lectures", "society",
                     "association", "foundation", "club", "group", "team", "online", "webcast", "conference"}
# tre em theo cach noi huong toi phu huynh / kem rieng (review 11/10 - PANTADO lot qua CHILD_RE; review 12/10 - tach
# HARD (huong toi tre) / SOFT (nguoi lon noi ve tre: hop phu huynh, hoi thao cha me, gia su lop 12 -> chi gan co))
CHILD_HARD_RE = re.compile(r"cùng con\b(?! người)|\b1 thầy (?:kèm )?1 trò|\b1 kèm 1\b|\bkèm 1 trò|for your child|"
                           r"for kids", re.I)
CHILD_SOFT_RE = re.compile(r"cho con\b(?! người| đường)|ba mẹ|bố mẹ|phụ huynh|gia sư|\blớp 1 ?- ?1\b|\b\d kèm \d\b|"
                           r"for parents|your child|parenting", re.I)
CHILD_SOFT_REJECT = 2      # so cum SOFT KHAC nhau trong tieu de de bo han (vd 'lop hoc gia su 2 kem 4 ... ba me')
CHILD_PARENT_RE = re.compile(f"{CHILD_HARD_RE.pattern}|{CHILD_SOFT_RE.pattern}", re.I)   # HARD | SOFT (dung cho mo ta)
# ton giao: bai giang / lop hoc (KHONG bo; co 'religious_teaching' - quyet dinh 1 'tranh ton giao nhay cam neu loc duoc')
RELIG_CHANNELS = {
    "UCNck9rwL6uP8Xh5V2TY_Qgw",   # Su Hanh Tue Theravada (hoi dap Phat phap)
    "UCjlUGvFf27VIk0t036iT-qw",   # Chua Phuc Minh
    "UCOYy3RyWrnWxDX1TAniYhUA",   # Tu Vien Dao Tam
    "UCoDLpeHljBu2dWemxYLKnYw",   # Thy Lam (Lop Phat Phap)
    "UCW759e8p1Qz6QP-PKaECfGg",   # Hoi Thanh Nha Muon Dan
    "UCYv6YH8DpXIFgx51wOEuTiQ",   # Hoi Thanh Tin Lanh Ben Tre
    "UCUtN9rPOxVzQBkaTJa1d2NQ",   # Trung Tam Muc Vu Giao Phan Can Tho
    "UCopvJPs7wemUv6WrOBYHEMQ",   # Chua Duc Vien
    "UCQviXmqPa6PRBmukpy_gWeA",   # BibleTalk.tv
    "UCrt55m1zgpsOWCbhfyft4Tg",   # Nguoi Phuong Ngoai (bai giang Su Giac Nguyen; regex khong bat 'Su') - ra soat 17/10
}
RELIG_TEACH_RE = re.compile(r"phật|pháp thoại|giảng kinh|kinh thánh|hội thánh|tin lành|công giáo|giáo phận|mục vụ|"
                            r"\bchùa\b|tu viện|thiền|thượng tọa|hòa thượng|dharma|sutra|buddh|theravāda|theravada|"
                            r"\bbible\b|\bchurch\b|sermon|gospel|catechism|scripture", re.I)
# co -> loai MAC DINH o segment / hosts cho den khi nguoi dung xac nhan (--allow-flag); fetch xep sau cung
EXCLUDE_FLAGS = ("religious_teaching", "minor_risk", "child_risk")
LICENSE_REASONS = ("license_not_cc", "ia_license_not_whitelisted", "ia_license_nc_nd", "ia_license_pd_mark",
                   "license_nc_site_terms", "ia_uploader_not_creator", "ia_uploader_unknown")
# kenh NC-SA da biet du YouTube hien CC BY (legal_notes 1) -> bo tru khi --allow nc_nd
NC_SA_CHANNELS = {"UC4EY_qnSeAP1xGsh61eOoJA": "CC BY-NC-SA 3.0 (Open Yale Courses terms)",
                  "UC640y4UvDAlya_WOj5U4pfA": "CC BY-NC-SA (NPTEL terms)"}
# tieu de bi loai theo kenh (ghi chu whitelist: dang lai cua nha xuat ban, chu so huu khong ro)
CHANNEL_TITLE_REJECT = {"UCfo8NSTcRgMoo17Ya4NvGtw": re.compile(r"tập huấn|tuấn huấn", re.I)}   # 17/10: tieu de go sai 'Tuấn huấn'
# kenh bi loai ca kenh sau khi nghe / ASR pilot 11/10 (ly do ghi vao reasons)
CHANNEL_REJECT = {
    # PANTADO: lop tieng Anh 1 kem 1 cho TRE EM ('Cung con but pha Tieng Anh'), loi noi chu yeu tieng Anh tren kenh 'vi'
    # (ASR tieng Viet ra chuoi vo nghia) -> vi pham quyet dinh 1 (tre em) + lech ngon ngu
    "UC2JPCC4OKl5oQUeiLBFR--w": "channel_reject:children_english_lessons",
    # Giap bui dinh: lop truc tuyen cua giao vien co giong hoc sinh nho (ghi chu whitelist) - ra soat 17/10, quyet dinh 1
    "UCOvqfsTAKwQli9H9t5MOR8Q": "channel_reject:children_voices",
}


def _nfc(s: Any) -> str:
    return unicodedata.normalize("NFC", str(s or ""))


def _nfkc(s: Any) -> str:
    """NFKC (chu dam toan hoc -> chu thuong, vd mo ta PANTADO) roi NFC."""
    return unicodedata.normalize("NFC", unicodedata.normalize("NFKC", str(s or "")))


def ia_uploader_is_creator(creator: Any, uploader: Any) -> bool | None:
    """archive.org: nguoi dang co phai chu so huu noi dung (nguoi dat licenseurl co quyen dat)?
    True : email da xac minh tay (IA_VERIFIED_UPLOADERS) hoac ten mien email chua mot tu (>= 4 ky tu) cua ten tac gia
           (vd clerk@springfieldil.gov cho 'City of Springfield', media@berkeley.edu cho 'UC Berkeley').
    False: ten mien khong lien quan / email thu mien phi (vd david@bamsoftware.com dang lai ucberkeley-webcast).
    None : thieu nguoi dang / tac gia / email -> khong ro (cong coi nhu CHUA chung minh -> bo)."""
    up = _nfkc(uploader).strip().lower()
    if up and up in IA_VERIFIED_UPLOADERS:
        return True
    cr = U.slug(_nfkc(creator), 400) if creator not in (None, "", [], "None") else ""
    dom = up.rsplit("@", 1)[-1] if "@" in up else ""
    if not dom or not cr or cr == "x":
        return None
    toks = {t for t in cr.split("_") if len(t) >= 4 and t not in IA_GENERIC_TOKENS}
    return any(t in dom for t in toks)


def is_religious(rec: Mapping[str, Any], wl_row: Mapping[str, Any] | None = None) -> bool:
    """Kenh / video bai giang ton giao: danh sach kenh, hoac regex tren ten kenh + ghi chu whitelist + tieu de."""
    if rec.get("channel_id") in RELIG_CHANNELS:
        return True
    w = wl_row or {}
    txt = " ".join(_nfkc(x) for x in (rec.get("channel"), w.get("channel_name"), w.get("notes"), rec.get("title")))
    return bool(RELIG_TEACH_RE.search(txt))


def audio_format_ok(fmts: list[Mapping[str, Any]] | None) -> tuple[bool, str | None]:
    """A6: co 251/140 goc (khong -drc). Tra (ok, format_id uu tien)."""
    if not fmts:
        return False, None
    ids = []
    for f in fmts:
        fid = str(f.get("format_id") or "")
        note = str(f.get("format_note") or "").lower()
        if "drc" in fid or "drc" in note:
            continue
        base, _, sub = fid.partition("-")
        if base not in ("251", "140"):
            continue
        if sub and "original" not in note:    # 251-0 'English (US)' = long tieng tu dong -> bo
            continue
        ids.append(fid)
    if not ids:
        return False, None
    ids.sort(key=lambda x: (not x.startswith("251"), x))
    return True, ids[0]


def _license_yt(rec: Mapping[str, Any], wl: Mapping[str, Mapping[str, Any]], reasons: list, flags: list,
                allow: frozenset = frozenset()) -> str | None:
    lic = _nfc(rec.get("license"))
    if "creative commons" not in lic.lower():
        reasons.append("license_not_cc")
        return None
    cid = rec.get("channel_id")
    w = wl.get(cid)
    if w is None:
        reasons.append("channel_not_whitelisted")
    elif w.get("owner_confidence") not in CONF_OK:
        reasons.append("owner_confidence_low")
    if cid in NC_SA_CHANNELS:
        flags.append("license_stricter_site_terms")
        if "nc_nd" not in allow:
            reasons.append("license_nc_site_terms")
        return NC_SA_CHANNELS[cid]
    return lic


def _license_open(rec: Mapping[str, Any], reasons: list, flags: list, allow: frozenset = frozenset()) -> str | None:
    src = rec.get("source")
    if src == "ia":
        url = str(rec.get("licenseurl") or "")
        if IA_LIC_NCND_RE.search(url):
            flags.append("license_nc_nd")
            if "nc_nd" not in allow:
                reasons.append("ia_license_nc_nd")
        elif IA_LIC_PDMARK_RE.search(url):
            flags.append("license_pd_mark")
            if "pd_mark" not in allow:
                reasons.append("ia_license_pd_mark")
        elif not IA_LIC_RE.search(url):
            reasons.append("ia_license_not_whitelisted")
            return None
        uic = rec.get("uploader_is_creator", "?")
        if uic == "?":     # ung vien cu (truoc 12/10) chua co truong -> tinh tu uploader / creator da luu
            uic = ia_uploader_is_creator(rec.get("creator", rec.get("channel")), rec.get("uploader"))
        if uic is not True:   # fail closed: chi qua khi CHUNG MINH duoc nguoi dang la tac gia
            tag = "ia_uploader_not_creator" if uic is False else "ia_uploader_unknown"
            flags.append(tag)
            if "ia_reupload" not in allow:
                reasons.append(tag)
        return url
    if src in ("ccc", "fosdem"):
        flags.append("license_site_default")
        return U.OPEN_SOURCES[src]["license"]
    reasons.append(f"unknown_source:{src}")
    return None


def gate_record(rec: Mapping[str, Any], wl: Mapping[str, Mapping[str, Any]],
                allow: frozenset = frozenset()) -> dict[str, Any]:
    """Quyet dinh cong metadata cho MOT ung vien (khong mang, ham thuan). allow: tap con cua ALLOW_KEYS (mac dinh rong)."""
    reasons: list[str] = []
    flags: list[str] = []
    src = rec.get("source", "yt")
    if rec.get("error"):
        reasons.append("metadata_error")
    lic = _license_yt(rec, wl, reasons, flags, allow) if src == "yt" else _license_open(rec, reasons, flags, allow)

    live = rec.get("live_status")
    prefer = 0
    if live in BAD_LIVE:
        reasons.append(f"live_{live}")
    elif live == "was_live" or rec.get("was_live"):
        prefer = 2
    elif live == "not_live":
        prefer = 1
    dur = rec.get("duration")
    if dur is None:
        reasons.append("duration_unknown")
    elif not MIN_DUR <= float(dur) <= MAX_DUR:
        reasons.append("duration_out_of_range")

    title, desc = _nfc(rec.get("title")), _nfc(rec.get("description") or rec.get("desc_head"))[:1500]
    if TITLE_RE.search(title):
        reasons.append("title_regex:" + TITLE_RE.search(title).group(0).lower())
    if DESC_RE.search(desc):
        reasons.append("desc_regex:" + DESC_RE.search(desc).group(0).lower())
    title_k, desc_k = _nfkc(title), _nfkc(desc)
    n_soft = len({m.group(0).lower() for m in CHILD_SOFT_RE.finditer(title_k)})
    if (CHILD_RE.search(title) or CHILD_DESC_RE.search(desc[:400]) or CHILD_HARD_RE.search(title_k)
            or n_soft >= CHILD_SOFT_REJECT):
        reasons.append("children_content")
    elif n_soft or CHILD_PARENT_RE.search(desc_k[:800]):
        flags.append("child_risk")
    if is_religious(rec, wl.get(rec.get("channel_id"))):
        flags.append("religious_teaching")
    if RELIG_SENS_RE.search(title):
        reasons.append("religious_sensitive")
    if MINOR_RE.search(title) or MINOR_RE.search(desc[:400]):
        flags.append("minor_risk")
    if rec.get("channel_id") in CHANNEL_REJECT:
        reasons.append(CHANNEL_REJECT[rec["channel_id"]])
    rej = CHANNEL_TITLE_REJECT.get(rec.get("channel_id"))
    if rej and rej.search(title):
        reasons.append("channel_title_reject")

    cats = rec.get("categories") or []
    if any(c in BAD_CATS for c in cats):
        reasons.append("category_" + "_".join(c for c in cats if c in BAD_CATS).lower().replace(" & ", "_"))
    if int(rec.get("chapters") or 0) > MAX_CHAPTERS:
        flags.append("many_chapters")
    if rec.get("availability") not in GOOD_AVAIL:
        reasons.append(f"availability_{rec.get('availability')}")

    lang_wl = (wl.get(rec.get("channel_id")) or {}).get("lang") or rec.get("lang")
    lang_yt = str(rec.get("language") or "")
    if lang_wl and lang_yt and not lang_yt.lower().startswith(lang_wl):
        reasons.append(f"lang_mismatch:{lang_yt}")

    fmt = None
    if src == "yt":
        ok_fmt, fmt = audio_format_ok(rec.get("audio_formats"))
        if not ok_fmt:
            reasons.append("no_original_251_140")
    return {"id": rec.get("id"), "source": src, "channel_id": rec.get("channel_id"), "lang": lang_wl,
            "ok": not reasons, "reasons": reasons, "flags": flags, "prefer": prefer, "format_id": fmt,
            "license_effective": lic, "duration": dur, "title": title[:200], "live_status": live,
            "gated_at": U.now_iso()}


def gate_all(cands: list[Mapping[str, Any]], wl: Mapping[str, Mapping[str, Any]],
             allow: frozenset = frozenset()) -> tuple[list[dict], dict]:
    res = [gate_record(r, wl, allow) for r in cands]
    by_ch: dict[str, Counter] = defaultdict(Counter)
    why = Counter()
    lic_ok = Counter()
    for r, g in zip(cands, res):
        key = f"{g['lang']}|{g['channel_id'] or g['source']}"
        by_ch[key]["n"] += 1
        by_ch[key]["ok"] += int(g["ok"])
        by_ch[key]["hours_ok"] += (g["duration"] or 0) / 3600 if g["ok"] else 0
        is_cc = not any(x in g["reasons"] for x in LICENSE_REASONS)
        by_ch[key]["license_ok"] += int(is_cc)
        lic_ok[g["lang"]] += int(is_cc)
        for x in g["reasons"]:
            why[x.split(":")[0]] += 1
    n_lang = Counter(g["lang"] for g in res)
    flg = Counter(f for g in res if g["ok"] for f in g["flags"])
    stats = {"n": len(res), "n_ok": sum(g["ok"] for g in res),
             "license_pass_rate": {k: round(lic_ok[k] / v, 3) for k, v in n_lang.items() if v},
             "reasons": dict(why.most_common()), "allow": sorted(allow),
             "flags_among_ok": dict(flg.most_common()),
             "by_channel": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in sorted(by_ch.items())}}
    return res, stats


def hold_reason(g: Mapping[str, Any] | None, allow_flags=()) -> str | None:
    """Ly do KHONG dung video o cac buoc sau (segment / align / lid / hosts) theo cong hien tai, None = dung duoc.
    Thieu dong cong -> 'khong_co_cong' (fail closed); cong bo -> 'cong_loai'; co loai mac dinh -> 'co:<co>'."""
    if g is None:
        return "khong_co_cong"
    if not g.get("ok"):
        return "cong_loai"
    bad = sorted((set(EXCLUDE_FLAGS) - set(allow_flags)) & set(g.get("flags") or []))
    return "co:" + bad[0] if bad else None


def stored_allow(gated: str = U.GATED) -> frozenset:
    """Tap --allow cua lan chay gate gan nhat (gate_stats.json) - tinh lai cong voi CUNG lua chon cua nguoi dung."""
    try:
        with open(os.path.join(os.path.dirname(gated), "gate_stats.json"), encoding="utf-8") as fh:
            return frozenset(x for x in json.load(fh).get("allow") or [] if x in ALLOW_KEYS)
    except (OSError, ValueError):
        return frozenset()


def current(allow: frozenset | None = None, candidates: str = U.CANDIDATES, gated: str = U.GATED,
            verbose: bool = True) -> dict[str, dict]:
    """Cong tinh lai trong bo nho (re: ham thuan tren candidates.jsonl). allow None -> lay tu gate_stats.json.
    So voi gated.jsonl da luu: lech -> in canh bao (nen chay lai 'python -m v6.scrape.gate')."""
    allow = stored_allow(gated) if allow is None else frozenset(allow)
    cands = list(U.latest_by(U.read_jsonl(candidates)).values())
    res = {g["id"]: g for g in gate_all(cands, U.load_whitelist(), allow)[0] if g.get("id")}
    old = U.latest_by(U.read_jsonl(gated))
    flips = sum(1 for k, g in res.items() if k in old and bool(old[k].get("ok")) != bool(g["ok"]))
    missing = sum(1 for k in res if k not in old)
    if verbose and (flips or missing):
        print(f"  CANH BAO: gated.jsonl cu lech cong hien tai ({flips} doi ket qua, {missing} chua co) - dung cong tinh "
              f"lai trong bo nho; nen chay lai 'python -m v6.scrape.gate'", flush=True)
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", default=U.CANDIDATES)
    ap.add_argument("--out", default=U.GATED)
    ap.add_argument("--allow", action="append", choices=ALLOW_KEYS, default=[],
                    help="mo giay phep ngoai quyet dinh 1 - CHI khi nguoi dung da dong y")
    a = ap.parse_args()
    cands = list(U.latest_by(U.read_jsonl(a.candidates)).values())
    res, stats = gate_all(cands, U.load_whitelist(), frozenset(a.allow))
    U.write_jsonl(a.out, res)
    with open(os.path.join(os.path.dirname(a.out), "gate_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=1)
    print(f"ung vien {stats['n']}, qua cong {stats['n_ok']}; ty le giay phep dat: {stats['license_pass_rate']}; "
          f"allow={stats['allow']}; co trong tap dat: {stats['flags_among_ok']}")
    for k, v in stats["reasons"].items():
        print(f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
