"""Cào ghi âm TỰ NHIÊN (nhãn «01 có khả năng», PHU_DU_LIEU mục 2.2 ý 3) từ YouTube — để đo / chọn ngưỡng chặn oan K1 (≤ 2 %).

Bước 1 `discover` (CHỈ metadata, không tải âm thanh):
  tìm theo truy vấn với bộ lọc «Creative Commons» của YouTube → đọc metadata đầy đủ từng video → `gate()` theo PHU_DU_LIEU mục 1.8:
  giấy phép CC BY · kênh không chuyên · không phụ lục A · không tập khoá v2 (crc32('yt:<kênh>') % 5 == 0, v6/data/lockbox.py)
  · không tôn giáo / trẻ em / trường phổ thông · đúng ngôn ngữ · độ dài hợp lý.
  Ghi MỌI video (cả bị loại, kèm lý do) vào data/p1_scrape/ung_vien_<lang>.jsonl — không lọc ngầm. Chạy lại: bỏ qua video đã đọc.
Dùng:  .venv/Scripts/python.exe experts/P1/scrape/cao_tunhien.py discover --lang vi [--per-query 40] | stats | selftest
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(ROOT, "data", "p1_scrape")
RULES = os.path.join(ROOT, "lanes", "PHU_DU_LIEU_5_LANE.md")
CC_FILTER = "EgIwAQ%3D%3D"                     # bộ lọc «Creative Commons» của trang tìm kiếm YouTube
# «kênh không chuyên» (mục 1.8 chưa định nghĩa số) — quy ước TẠM, chờ Huấn chốt: chưa có tích xanh VÀ ≤ 100 000 người theo dõi
MAX_FOLLOWERS = 100_000
MIN_S, MAX_S = 60, 3 * 3600
BAD_CATEGORIES = {"Music", "Gaming", "Film & Animation", "Sports"}   # nhạc / game / phim: không phải ghi âm lời nói tự nhiên
BANNED = {  # mục 1.8: không lấy nội dung tôn giáo, trẻ em, trường phổ thông (khớp từ trong tiêu đề / mô tả / thẻ, không phân biệt hoa thường)
    "ton_giao": r"phật|chùa|thiền sư|kinh thánh|thánh lễ|giáo xứ|nhà thờ|công giáo|tin lành|cầu nguyện|giảng đạo|bible|church|sermon|"
                r"gospel|quran|mosque|buddhis|prayer|worship|pastor|jesus|allah",
    "tre_em": r"thiếu nhi|trẻ em|em bé|mầm non|bé học|kids|children|child|baby|toddler|nursery|kindergarten",
    "truong_pho_thong": r"học sinh|tiểu học|thcs|thpt|trung học cơ sở|trung học phổ thông|lớp [1-9]\b|lớp 1[0-2]\b|ôn thi vào 10|"
                        r"elementary school|middle school|high school|primary school|\bk-12\b|grade [1-9]\b|grade 1[0-2]\b",
}
# Luật thêm 21/11/2026 (Huấn chốt): video không phải ghi âm lời nói lọt qua thể loại «Entertainment / Travel» (vd «giao lưu guitar»,
# «giao lưu chèo», «FC Alpha vs FC …»). Chỉ soi TIÊU ĐỀ + THẺ (mô tả dài, hay nhắc nhạc nền). Đặt ra SAU khi xem K1 trên vi
# → số vi sau lọc chỉ để tham khảo; en chạy K1 SAU khi chốt luật này = phép thử sạch.
NONSPEECH = (r"guitar|acoustic|\bukulele\b|\bpiano\b|đánh đàn|chơi đàn|hát chèo|\bchèo\b|cải lương|ca nhạc|văn nghệ|bolero|karaoke|"
             r"\bconcert\b|live music|song cover|cover song|\bsinging\b|bóng đá|\bfc\b|\bfootball\b|\bsoccer\b|\bfutsal\b|gameplay")
# Luật thêm 21/11/2026 tối (Huấn duyệt _kiem/SOAT_VIDEO_2111.md nhóm A, B, D): soi TIÊU ĐỀ + THẺ + TÊN KÊNH, không soi mô tả.
# Constraint «Loại âm thanh»: ghi âm lời nói, KHÔNG podcast / phát thanh / sách nói / nhạc → Huấn chốt podcast = 04.
KHONG_TU_NHIEN = {
    "ton_giao": r"phật pháp|pháp thoại|hành thiền|đạo hữu|vipassana|mahasi|tâm linh|thượng đế|kriya|\bspiritual\b|\bmeditation\b|"
                r"\bfellowship\b|\bministry\b|missioned souls|ishwar|"
                # 22/11 (Huấn duyệt _kiem/SOAT_VIDEO_VI_BU_2211.md): biến thể dấu + tên gọi
                r"pháp thọai|hòa thượng|\bht\.|pháp luân|thiền an cư|trà đạo vấn đáp",
    "sach_noi": r"sách nói|audiobook|truyện audio|đọc truyện|truyện đêm khuya|triết lý|cổ nhân|động lực từ|jim rohn|bài học cuộc sống|chữa lành|"
                r"tiên hiệp|truyện gia đình|storytime|learn english|english conversation|english pod|"
                r"truyện ma|chuyện ma|cổ tích|tóm tắt sách|tóm tắt cuộc đời|bí mật lịch sử|hoạt hình|đọc án",
    "long_tieng": r"song ngữ|lồng tiếng|thuyết minh|vietsub|\bdubbed\b|the compiler|vietnamese version",
    "podcast": r"podcast|talk ?show|art of charm|\bradio\b",   # chỉ soi tiêu đề + tên kênh: thẻ hay gắn «podcast» cho phỏng vấn thường
    "truyen_hinh": r"\bhtv\d*\b|\bvtv\d*\b|truyền hình|bản tin|phát sóng",          # 22/11: chương trình / đài truyền hình
    "truong_pho_thong": r"public schools?",                                         # 22/11: học khu phổ thông (mục 1.8)
}
TITLE_ONLY = {"podcast", "truyen_hinh", "truong_pho_thong"}
# 22/11: danh sách gỡ tay theo nghe mù Gemini (gemini/KQ_NGHE_MU_VI_BU.jsonl) + soát chính sách; tách ra tệp vì dài (id → lý do)
LOAI_TAY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "loai_tay.json")
# Từ khoá (tiêu đề / kênh) không bắt được, Huấn duyệt gỡ tay 21/11 (_kiem/SOAT_VIDEO_2111.md; nghe: KET_QUA_NGHE_P1_2111.md)
LOAI_THEO_NGHE = {"5hnynaqVIHg": "game + nhạc + giọng TTS", "m5Vle27My1o": "hát nhạc Bolero",
                  "BO6qQSOVQ78": "podcast (kênh Hà Phi Official, thẻ podcast)", "uaO7E-FaHXM": "podcast (thẻ podcast, tập #290)",
                  # nghe mù Gemini 21/11 (gemini/KQ_NGHE_MU_2111.jsonl, KQ_NHOM_C_2111.md), Huấn duyệt gỡ
                  "9aetNeMbnZU": "review phim dựng sẵn, nhạc nền + chèn clip phim (nhóm C)", "MKrhGPTioxw": "chương trình truyền hình HTV7 (nhóm C)",
                  "eowka-cMoxM": "talkshow truyền hình «The Talk» (nhóm C)", "DhKYvJ_ZXeE": "đọc động lực, nghi TTS + nhạc lofi",
                  "MQ7ywlCPim4": "đọc kịch bản + nhạc nền", "mgNJvmQZnNA": "nghi giọng TTS", "arxZmuYMrOE": "truyện audio, giọng TTS",
                  "Pa6bHOHKaEc": "đọc quảng cáo du lịch", "bImYMj88b20": "thuyết minh + nhạc nền", "ARN6FFV2MYo": "đọc kịch bản",
                  "mmFgpMmymYc": "bài hát", "0Bw8CKdFDQ0": "chương trình podcast có nhạc (City Talk)",
                  **(json.load(open(LOAI_TAY_JSON, encoding="utf-8")) if os.path.exists(LOAI_TAY_JSON) else {})}
# Nghe mù Gemini: lời nói tự nhiên nhưng NHẠC NỀN RÕ (video hướng dẫn dựng sẵn). Constraint nền > 10 % → 04, nên đáp án chưa chắc là accept.
# Huấn chốt 21/11: giữ tệp, đáp án «khong_chac», ca riêng tunhien.<lang>.nhac_nen, không tính vào chặn oan.
# ponytail: Gemini chỉ nghe 1 đoạn / video, áp cho cả video; cần thước đo «nền» (cond_models) để tách theo đoạn.
NHAC_NEN_NGHE = {"T_XCvMIcc88", "cOyB3IMPOt8", "Kr_3r_XIKZ0", "LfbHiUAzRjM", "5NVYg2HNAdA", "zfycN6kG14s", "N4Z27dN3Wvo",
                 "F8ntmEp3pCo", "Df11SkLnekM", "So8aseCO3hY",
                 "GUjZlXA2Mj8", "O6xFbLpGoF8", "ZtHfdR5XJP8", "gdIordqkHKs"}   # 22/11 vi cào bù
QUERIES = {  # loại nội dung trong phạm vi bản nộp (mục 1.5): cuộc gọi, tin nhắn thoại, cuộc họp, buổi học, phỏng vấn, lời khai
    # 21/11: bỏ «talkshow nhỏ» (vi) và «podcast style chat» (en): trái Constraint «không podcast»; «tâm sự chia sẻ» kéo về radio đọc truyện
    "vi": ["phỏng vấn", "trò chuyện", "chia sẻ kinh nghiệm", "buổi họp", "thảo luận nhóm", "hội thảo chia sẻ",
           "bài giảng đại học", "kể chuyện đời", "nói chuyện cùng", "giao lưu",
           # 21/11 cào bù (Huấn duyệt): cuộc họp / buổi học / phỏng vấn / buổi nói chuyện thật, tránh podcast, sách nói, trường phổ thông
           "phỏng vấn đường phố", "tọa đàm", "buổi nói chuyện", "hỏi đáp trực tiếp", "họp nhóm", "buổi chia sẻ",
           "seminar tiếng việt", "workshop chia sẻ", "phỏng vấn nhân vật", "kể lại trải nghiệm", "vlog nói chuyện", "thuyết trình"],
    "en": ["interview", "conversation", "community meeting", "town hall meeting", "panel discussion", "university lecture",
           "oral history interview", "talk about my experience", "workshop talk", "board meeting", "q&a session"],
}


# ----------------------------------------------------------------------------- luật (hàm thuần)
def appendix_a(text: str) -> set[str]:
    """Kênh phụ lục A (khoá '<nguồn>:<kênh>'), đọc thẳng từ PHU_DU_LIEU để chỉ có MỘT nơi định nghĩa."""
    part = text[text.find("## Phụ lục A"):]
    return set(re.findall(r"`(yt:[\w-]+)`", part))


def is_lockbox(channel_id: str) -> bool:
    """Tập khoá v2, luật scrape: zlib.crc32('yt:<channel_id>') % 5 == 0 (v6/data/lockbox.py scrape_key)."""
    return zlib.crc32(f"yt:{channel_id}".encode()) % 5 == 0


def banned_topics(m: dict) -> list[str]:
    s = " ".join([m.get("title") or "", m.get("description") or "", " ".join(m.get("tags") or [])]).lower()
    return [k for k, rx in BANNED.items() if re.search(rx, s)]


def nonspeech(m: dict) -> list[str]:
    """Từ trong tiêu đề / thẻ cho thấy video là nhạc / thể thao / game, không phải ghi âm lời nói."""
    s = " ".join([m.get("title") or "", " ".join(m.get("tags") or [])]).lower()
    out = {x.group(0) for x in re.finditer(NONSPEECH, s)}
    tc = " ".join([m.get("title") or "", m.get("channel") or ""]).lower()
    s += " " + (m.get("channel") or "").lower()
    out |= {f"{k}:{x.group(0)}" for k, rx in KHONG_TU_NHIEN.items() for x in re.finditer(rx, tc if k in TITLE_ONLY else s)}
    if m.get("id") in LOAI_THEO_NGHE:
        out.add(f"nghe:{LOAI_THEO_NGHE[m['id']]}")
    return sorted(out)


def gate(m: dict, lang: str, appx: set[str]) -> list[str]:
    """Lý do loại (rỗng = qua). Không đoán: thiếu trường thì ghi lý do 'thiếu_<trường>'."""
    r = []
    if "Creative Commons" not in (m.get("license") or ""):
        r.append(f"giay_phep:{m.get('license')}")
    cid = m.get("channel_id")
    if not cid:
        r.append("thieu_channel_id")
    else:
        if f"yt:{cid}" in appx:
            r.append("phu_luc_A")
        if is_lockbox(cid):
            r.append("tap_khoa_v2")
    if m.get("channel_is_verified"):
        r.append("kenh_tich_xanh")
    fc = m.get("channel_follower_count")
    if fc is None:
        r.append("thieu_so_nguoi_theo_doi")
    elif fc > MAX_FOLLOWERS:
        r.append(f"kenh_lon:{fc}")
    d = m.get("duration") or 0
    if not MIN_S <= d <= MAX_S:
        r.append(f"do_dai:{d}")
    if m.get("live_status") not in (None, "not_live", "was_live"):
        r.append(f"live:{m.get('live_status')}")
    if (m.get("age_limit") or 0) > 0:
        r.append("gioi_han_tuoi")
    if set(m.get("categories") or []) & BAD_CATEGORIES:
        r.append(f"the_loai:{','.join(m['categories'])}")
    if m.get("language") and not str(m["language"]).startswith(lang):
        r.append(f"ngon_ngu:{m['language']}")
    r += [f"noi_dung:{k}" for k in banned_topics(m)]
    r += [f"khong_loi_noi:{w}" for w in nonspeech(m)]
    return r


# ----------------------------------------------------------------------------- cào metadata
KEEP = ("id", "title", "description", "tags", "categories", "license", "channel", "channel_id", "channel_follower_count",
        "channel_is_verified", "duration", "language", "live_status", "age_limit", "upload_date", "webpage_url")


def search_ids(query: str, n: int) -> list[str]:
    import yt_dlp
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}&sp={CC_FILTER}"
    with yt_dlp.YoutubeDL({"extract_flat": True, "playlistend": n, "quiet": True, "no_warnings": True}) as y:
        info = y.extract_info(url, download=False)
    return [e["id"] for e in info.get("entries") or [] if e.get("id")]


def video_meta(vid: str) -> dict:
    import yt_dlp
    with yt_dlp.YoutubeDL({"skip_download": True, "quiet": True, "no_warnings": True}) as y:
        d = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
    m = {k: d.get(k) for k in KEEP}
    m["description"] = (m.get("description") or "")[:2000]
    return m


def discover(lang: str, per_query: int) -> None:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"ung_vien_{lang}.jsonl")
    seen = {json.loads(x)["id"] for x in open(path, encoding="utf-8")} if os.path.exists(path) else set()
    appx = appendix_a(open(RULES, encoding="utf-8").read())
    print(f"phụ lục A: {len(appx)} kênh yt · đã có {len(seen)} video", flush=True)
    for q in QUERIES[lang]:
        ids = [i for i in search_ids(q, per_query) if i not in seen]
        print(f"«{q}»: {len(ids)} video mới", flush=True)
        for vid in ids:
            try:
                m = video_meta(vid)
                row = {**m, "truy_van": q, "lang_dich": lang, "ly_do_loai": gate(m, lang, appx)}
            except Exception as e:  # noqa: BLE001  (video riêng tư / bị chặn: ghi lỗi, không dừng)
                row = {"id": vid, "truy_van": q, "lang_dich": lang, "ly_do_loai": [f"loi:{type(e).__name__}"]}
            row["doc_luc"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen.add(vid)


def stats() -> None:
    import collections
    for lang in ("vi", "en"):
        path = os.path.join(OUT, f"ung_vien_{lang}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(x) for x in open(path, encoding="utf-8")]
        ok = [r for r in rows if not r["ly_do_loai"]]
        why = collections.Counter(x.split(":")[0] for r in rows for x in r["ly_do_loai"])
        ch = collections.Counter(r["channel_id"] for r in ok)
        hrs = sum(r["duration"] for r in ok) / 3600
        print(f"{lang}: {len(rows)} video đọc · {len(ok)} qua cổng · {len(ch)} kênh · {hrs:.1f} giờ · "
              f"video / kênh tối đa {max(ch.values()) if ch else 0}")
        print("   lý do loại:", dict(why.most_common()))


def selftest() -> None:
    appx = appendix_a("x\n## Phụ lục A. y\n`yt:UCabc` và `fosdem:fosdem:2024/k1` `yt:UC-d_e`")
    assert appx == {"yt:UCabc", "yt:UC-d_e"}
    assert any(is_lockbox(f"UC{i}") for i in range(50)) and not all(is_lockbox(f"UC{i}") for i in range(50))
    good = {"license": "Creative Commons Attribution license (reuse allowed)", "channel_id": "UCx", "channel_follower_count": 500,
            "channel_is_verified": False, "duration": 600, "live_status": "not_live", "age_limit": 0, "categories": ["People & Blogs"],
            "language": "vi", "title": "phỏng vấn", "description": "", "tags": []}
    cid = next(c for c in (f"UCx{i}" for i in range(50)) if not is_lockbox(c))
    assert gate({**good, "channel_id": cid}, "vi", set()) == []
    assert gate({**good, "channel_id": cid}, "vi", {f"yt:{cid}"}) == ["phu_luc_A"]
    assert "kenh_tich_xanh" in gate({**good, "channel_id": cid, "channel_is_verified": True}, "vi", set())
    assert gate({**good, "channel_id": cid, "license": None}, "vi", set())[0].startswith("giay_phep")
    assert gate({**good, "channel_id": cid, "title": "Bài giảng cho học sinh lớp 9"}, "vi", set()) == ["noi_dung:truong_pho_thong"]
    assert gate({**good, "channel_id": cid, "title": "Thánh lễ Chúa nhật"}, "vi", set()) == ["noi_dung:ton_giao"]
    assert gate({**good, "channel_id": cid, "language": "en"}, "vi", set()) == ["ngon_ngu:en"]
    assert nonspeech({"title": "Sấu Coffee Acoustic - Giao lưu Guitar 5"}) == ["acoustic", "guitar"]
    assert nonspeech({"title": "GIAO LƯU NHỮNG NGƯỜI YÊU CHÈO 2026"}) == ["chèo"]
    assert nonspeech({"title": "Giao lưu FC Alpha vs FC teamate"}) == ["fc"]
    assert nonspeech({"title": "Pháp thoại Sư X chia sẻ kinh nghiệm hành thiền"}) == ["ton_giao:hành thiền", "ton_giao:pháp thoại"]
    assert nonspeech({"title": "Trò chuyện sao khó thế | Podcast #137"}) == ["podcast:podcast"]
    assert nonspeech({"title": "Bài giảng 7: SQL | Song Ngữ"}) == ["long_tieng:song ngữ"]
    assert nonspeech({"title": "Tâm sự", "channel": "Tuấn Khang Radio"}) == ["podcast:radio"]
    assert nonspeech({"title": "Phỏng vấn Product Manager", "tags": ["podcast", "radio"]}) == []   # podcast / radio: không soi thẻ
    assert nonspeech({"id": "5hnynaqVIHg", "title": "Sống lại hồi ức"}) == ["nghe:game + nhạc + giọng TTS"]
    assert nonspeech({"title": "Pháp thọai Sư Thiện Viên"}) == ["ton_giao:pháp thọai"]
    assert nonspeech({"title": "Ám Ảnh Cuộc Gọi | Truyện Ma Đêm Khuya"}) == ["sach_noi:truyện ma"]
    assert nonspeech({"title": "((Chuyện trưa 12 giờ HTV7 Full 360) Buổi phát sóng"}) == ["truyen_hinh:htv7", "truyen_hinh:phát sóng"]
    assert nonspeech({"title": "Stanford CS224W", "channel": "Tri Thức Mở by The Compiler"}) == ["long_tieng:the compiler"]
    assert nonspeech({"title": "Chia sẻ kinh nghiệm nuôi lươn", "tags": ["truyền hình"]}) == []      # truyen_hinh: không soi thẻ
    assert nonspeech({"title": "Kể chuyện đời tôi"}) == []
    for ok in ("Kỹ năng phát biểu giao tiếp trong cuộc họp", "Diễn đàn doanh nghiệp", "Ca sĩ trả lời phỏng vấn", "discover new ideas", "How to write a cover letter"):
        assert nonspeech({"title": ok}) == [], ok
    real = appendix_a(open(RULES, encoding="utf-8").read())
    assert "yt:UCXyjy-WREt-AieU2wsIvP6g" in real and len(real) >= 20, len(real)
    print("cao_tunhien selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:] or ["stats"]
    if a[0] == "discover":
        discover(a[a.index("--lang") + 1], int(a[a.index("--per-query") + 1]) if "--per-query" in a else 40)
        stats()
    elif a[0] == "stats":
        stats()
    else:
        selftest()
