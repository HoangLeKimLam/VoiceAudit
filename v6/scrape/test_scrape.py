# -*- coding: utf-8 -*-
"""Kiem thu nhe cho goi v6.scrape (khong mang, khong nap mo hinh, < 400 MB RAM, ~1 phut). Chay qua khoa:
    bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.scrape.test_scrape [chuoi_loc]
Ham test_* thuan (tuong thich pytest nhung khong can pytest). Tep tam: data/scrape/_selftest (xoa lai sau moi lan)."""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import traceback

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import frames as FR  # noqa: E402
from v6.scrape import util as U  # noqa: E402

WORK = os.path.join(U.SCRAPE, "_selftest")
WL = {"UCa": {"channel_id": "UCa", "lang": "vi", "owner_confidence": "high", "prio": 1, "tabs": ["videos"]},
      "UClow": {"channel_id": "UClow", "lang": "vi", "owner_confidence": "low", "prio": 2, "tabs": ["videos"]},
      "UC4EY_qnSeAP1xGsh61eOoJA": {"channel_id": "UC4EY_qnSeAP1xGsh61eOoJA", "lang": "en", "owner_confidence": "high",
                                   "prio": 2, "tabs": ["videos"]}}
CC = "Creative Commons Attribution license (reuse allowed)"
F251 = [{"format_id": "251", "acodec": "opus", "abr": 100.0, "format_note": "medium"},
        {"format_id": "140", "acodec": "mp4a.40.2", "abr": 129.0, "format_note": "medium"}]


def _cand(**kw):
    base = {"id": "vid00000001", "source": "yt", "channel_id": "UCa", "title": "Buổi học số 4 - lập trình",
            "description": "", "license": CC, "live_status": "was_live", "duration": 3600, "categories": ["Education"],
            "chapters": 0, "availability": "public", "language": "vi", "audio_formats": F251}
    base.update(kw)
    return base


# ---------------- util ----------------
def test_util_jsonl_and_opts():
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, "a.jsonl")
    U.write_jsonl(p, [{"id": "x", "v": 1}])
    U.append_jsonl(p, [{"id": "x", "v": 2}, {"id": "y", "v": np.int64(3)}])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"id": "broken"')               # dong cuoi bi cat
    rows = U.read_jsonl(p)
    assert len(rows) == 3 and U.latest_by(rows)["x"]["v"] == 2 and rows[2]["v"] == 3
    o = U.ydl_opts(skip_download=True)
    assert o["js_runtimes"] == {"node": {}} and "cookiefile" not in o
    for bad in ("cookiefile", "cookiesfrombrowser", "remote_components"):
        try:
            U.ydl_opts(**{bad: "x"})
        except ValueError:
            continue
        raise AssertionError(f"{bad} phai bi cam")
    assert U.slug("Bùi Việt Hà / x") == "bui_viet_ha_x"
    assert U.hosts_dir("vi").endswith("hosts_vi") and U.hosts_dir("en", "long").endswith("hosts_long_en")


def test_util_whitelist():
    wl = U.load_whitelist()
    assert len(wl) >= 40
    vi = [r for r in wl.values() if r["lang"] == "vi"]
    en = [r for r in wl.values() if r["lang"] == "en"]
    assert len(vi) >= 15 and len(en) >= 8
    assert all(set(r["tabs"]) <= {"videos", "streams"} and r["tabs"] for r in wl.values())


# ---------------- gate ----------------
def test_gate_pass_and_license():
    from v6.scrape import gate as G
    g = G.gate_record(_cand(), WL)
    assert g["ok"] and g["prefer"] == 2 and g["format_id"] == "251", g
    g = G.gate_record(_cand(license="Standard YouTube License"), WL)
    assert not g["ok"] and "license_not_cc" in g["reasons"]
    g = G.gate_record(_cand(channel_id="UCzzz"), WL)
    assert "channel_not_whitelisted" in g["reasons"]
    g = G.gate_record(_cand(channel_id="UClow"), WL)
    assert "owner_confidence_low" in g["reasons"]
    g = G.gate_record(_cand(channel_id="UC4EY_qnSeAP1xGsh61eOoJA", language="en"), WL)   # NC-SA site terms: mac dinh bo
    assert not g["ok"] and g["reasons"] == ["license_nc_site_terms"] and "NC-SA" in g["license_effective"], g
    assert "license_stricter_site_terms" in g["flags"]
    g = G.gate_record(_cand(channel_id="UC4EY_qnSeAP1xGsh61eOoJA", language="en"), WL, frozenset({"nc_nd"}))
    assert g["ok"], g                                      # chi mo khi nguoi dung --allow nc_nd


def test_gate_ia_licenses_and_reupload():
    from v6.scrape import gate as G
    base = {"source": "ia", "duration": 1800, "title": "Public lecture", "lang": "en", "live_status": "not_live",
            "creator": "City of Springfield", "uploader": "clerk@springfieldil.gov"}
    lic = [("http://creativecommons.org/licenses/by/4.0/", True), ("https://creativecommons.org/licenses/by-sa/3.0/", True),
           ("http://creativecommons.org/publicdomain/zero/1.0/", True),
           ("http://creativecommons.org/licenses/by-nc-nd/3.0/", False), ("http://creativecommons.org/licenses/by-nc/4.0/", False),
           ("http://creativecommons.org/licenses/by-nd/4.0/", False), ("http://creativecommons.org/publicdomain/mark/1.0/", False)]
    for i, (url, ok) in enumerate(lic):
        g = G.gate_record({**base, "id": f"ia:t/{i}.mp3", "licenseurl": url}, {})
        assert g["ok"] is ok, (url, g["reasons"])
    nd = {**base, "id": "ia:t/nd.mp3", "licenseurl": "http://creativecommons.org/licenses/by-nc-nd/3.0/"}
    assert G.gate_record(nd, {})["reasons"] == ["ia_license_nc_nd"] and G.gate_record(nd, {}, frozenset({"nc_nd"}))["ok"]
    pdm = {**base, "id": "ia:t/pd.mp3", "licenseurl": "http://creativecommons.org/publicdomain/mark/1.0/"}
    assert G.gate_record(pdm, {})["reasons"] == ["ia_license_pd_mark"] and G.gate_record(pdm, {}, frozenset({"pd_mark"}))["ok"]
    by = {**base, "id": "ia:t/re.mp3", "licenseurl": "http://creativecommons.org/licenses/by/4.0/"}
    re_up = {**by, "creator": "UC Berkeley", "uploader": "david@bamsoftware.com"}     # dang lai boi ca nhan (pilot)
    g = G.gate_record(re_up, {})
    assert not g["ok"] and g["reasons"] == ["ia_uploader_not_creator"], g
    assert G.gate_record(re_up, {}, frozenset({"ia_reupload"}))["ok"]
    assert G.gate_record({**by, "uploader": None}, {})["reasons"] == ["ia_uploader_unknown"]
    assert G.gate_record({**by, "uploader_is_creator": False}, {})["reasons"] == ["ia_uploader_not_creator"]
    old = {k: v for k, v in re_up.items() if k != "creator"}                         # ung vien cu: tac gia o 'channel'
    old["channel"] = "['Mike MOSER', 'John P. Huelsenbeck']"
    assert G.gate_record(old, {})["reasons"] == ["ia_uploader_not_creator"]
    assert G.ia_uploader_is_creator("UC Berkeley", "webcast@berkeley.edu") is True
    assert G.ia_uploader_is_creator("UC Berkeley", "someone@gmail.com") is False
    assert G.ia_uploader_is_creator("City of York", "x@cityofleeds.gov") is False          # 'city' chung chung
    assert G.ia_uploader_is_creator(None, "a@b.org") is None and G.ia_uploader_is_creator("X Y", "") is None


def test_gate_children_hard_soft():
    from v6.scrape import gate as G
    hard = ["Cùng con bứt phá Tiếng Anh với lớp học 1 thầy kèm 1 trò...",           # PANTADO (khong can loai ca kenh)
            "Cùng khám phá lớp học gia sư 2 kèm 4 tại Pantado ba mẹ nhé....", "Lớp học cho bé"]
    for t in hard:
        g = G.gate_record(_cand(title=t), WL)
        assert not g["ok"] and "children_content" in g["reasons"], (t, g["reasons"])
    adult = ["Họp phụ huynh đầu năm - chia sẻ của giáo viên", "Hội thảo: Bố mẹ và con cái trong thời đại số",
             "Gia sư toán lớp 12 - luyện thi", "Lớp 1-1 với chuyên gia"]
    for t in adult:                                        # nguoi lon noi ve tre / kem: KHONG bo han, chi gan co
        g = G.gate_record(_cand(title=t), WL)
        assert g["ok"] and "child_risk" in g["flags"], (t, g)
    assert "minor_risk" in G.gate_record(_cand(title="Gia sư toán lớp 12 - luyện thi"), WL)["flags"]
    bold = "Buổi mới: \U0001d41cù\U0001d427\U0001d420 \U0001d41c\U0001d428\U0001d427 học"  # chu dam toan hoc
    g = G.gate_record(_cand(title="Hội thảo tháng 9", description=bold), WL)
    assert g["ok"] and "child_risk" in g["flags"], g      # NFKC -> 'cung con' trong mo ta -> co
    assert "child_risk" not in G.gate_record(_cand(title="Hội thảo tháng 9"), WL)["flags"]


def test_gate_current_recomputes():
    from v6.scrape import gate as G
    os.makedirs(WORK, exist_ok=True)
    cp, gp = os.path.join(WORK, "cands.jsonl"), os.path.join(WORK, "gated.jsonl")
    nd = {"id": "ia:x/a.mp3", "source": "ia", "licenseurl": "http://creativecommons.org/licenses/by-nc-nd/3.0/",
          "duration": 1800, "title": "t", "lang": "en", "live_status": "not_live", "creator": "UC Berkeley",
          "uploader": "media@berkeley.edu"}
    U.write_jsonl(cp, [nd])
    U.write_jsonl(gp, [{"id": "ia:x/a.mp3", "ok": True}])       # gated.jsonl CU (truoc khi doi chinh sach)
    res = G.current(allow=frozenset(), candidates=cp, gated=gp, verbose=False)
    assert res["ia:x/a.mp3"]["ok"] is False and "ia_license_nc_nd" in res["ia:x/a.mp3"]["reasons"]
    with open(os.path.join(WORK, "gate_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"allow": ["nc_nd", "rac"]}, fh)
    assert G.stored_allow(gp) == frozenset({"nc_nd"})
    assert G.current(candidates=cp, gated=gp, verbose=False)["ia:x/a.mp3"]["ok"] is True   # cung --allow da luu


def test_gate_rules():
    from v6.scrape import gate as G
    cases = {"live_post_live": _cand(live_status="post_live"), "duration_out_of_range": _cand(duration=300),
             "title_regex": _cand(title="Tổng hợp karaoke hay nhất"), "children_content": _cand(title="Lớp học cho bé"),
             "religious_sensitive": _cand(title="Lễ cầu siêu cho hương linh"),
             "category_music": _cand(categories=["Music"]), "availability_subscriber_only":
             _cand(availability="subscriber_only"), "lang_mismatch": _cand(language="en"),
             "no_original_251_140": _cand(audio_formats=[{"format_id": "251-drc", "format_note": "medium, DRC"},
                                                         {"format_id": "251-0", "format_note": "English (US)"}]),
             "desc_regex": _cand(description="Video đọc bằng AI voice"), "metadata_error": _cand(error="x")}
    for want, c in cases.items():
        g = G.gate_record(c, WL)
        assert not g["ok"] and any(r.startswith(want) for r in g["reasons"]), (want, g["reasons"])
    g = G.gate_record(_cand(title="Luyện thi vào 10 - buổi 2"), WL)
    assert g["ok"] and "minor_risk" in g["flags"]
    g = G.gate_record(_cand(chapters=20), WL)
    assert g["ok"] and "many_chapters" in g["flags"]
    # ra soat 17/10: kenh ton giao khong bat duoc bang regex, kenh giong tre em, tieu de 'tap huan' go sai
    assert G.is_religious(_cand(channel_id="UCrt55m1zgpsOWCbhfyft4Tg", title="Buổi nói chuyện"))
    wl2 = {**WL, **{c: {**WL["UCa"], "channel_id": c} for c in ("UCOvqfsTAKwQli9H9t5MOR8Q", "UCfo8NSTcRgMoo17Ya4NvGtw")}}
    g = G.gate_record(_cand(channel_id="UCOvqfsTAKwQli9H9t5MOR8Q"), wl2)
    assert not g["ok"] and "channel_reject:children_voices" in g["reasons"], g["reasons"]
    g = G.gate_record(_cand(channel_id="UCfo8NSTcRgMoo17Ya4NvGtw", title="Tuấn huấn 2026 - Môn Tiếng Việt 4"), wl2)
    assert not g["ok"], g["reasons"]
    assert G.gate_record(_cand(channel_id="UCfo8NSTcRgMoo17Ya4NvGtw", title="Chia sẻ Ứng dụng CNTT"), wl2)["ok"]
    ok, fid = G.audio_format_ok([{"format_id": "251-0", "format_note": "English (US)"},
                                 {"format_id": "251-1", "format_note": "Vietnamese original (default), medium"}])
    assert ok and fid == "251-1"


def test_gate_open_sources_and_stats():
    from v6.scrape import gate as G
    ia_ok = {"id": "ia:x/a.mp3", "source": "ia", "licenseurl": "http://creativecommons.org/licenses/by/4.0/",
             "duration": 1800, "title": "City council meeting", "lang": "en", "live_status": "not_live",
             "creator": "Springfield City Council", "uploader": "clerk@springfield.gov"}
    ia_bad = {**ia_ok, "id": "ia:y/b.mp3", "licenseurl": "http://example.com/all-rights"}
    ccc = {"id": "ccc:g", "source": "ccc", "duration": 2400, "title": "Talk", "lang": "en", "live_status": "not_live"}
    res, st = G.gate_all([ia_ok, ia_bad, ccc, _cand()], WL)
    assert [r["ok"] for r in res] == [True, False, True, True]
    assert "license_site_default" in res[2]["flags"]
    assert st["n"] == 4 and st["n_ok"] == 3 and st["reasons"]["ia_license_not_whitelisted"] == 1


# ---------------- discover / fetch / probe ----------------
def test_discover_helpers():
    from v6.scrape import discover as D
    info = {"id": "abc", "channel_id": "UCa", "license": CC, "live_status": "was_live", "duration": 1000,
            "formats": [{"format_id": "251", "vcodec": "none", "acodec": "opus", "abr": 90},
                        {"format_id": "18", "vcodec": "avc1", "acodec": "mp4a"}], "chapters": [{}] * 3,
            "description": "x" * 5000}
    c = D.candidate_from_info(info, {"tab": "streams", "lang": "vi"})
    assert c["chapters"] == 3 and len(c["audio_formats"]) == 1 and len(c["description"]) == D.DESC_KEEP
    flat = [{"id": f"v{i}", "channel_id": "UCa" if i < 6 else "UCb", "duration": 100 if i == 0 else 1200}
            for i in range(10)]
    s = D.pick_sample(flat, 3, 7, done={"v5"})
    assert sum(r["channel_id"] == "UCa" for r in s) == 2 and sum(r["channel_id"] == "UCb" for r in s) == 3
    assert all(r["id"] != "v0" for r in s)
    s = D.pick_sample(flat, 2, 7, done=set(), prefer={"v9", "v8"}, channels={"UCb"})   # kenh moi: uu tien id tim CC
    assert {r["id"] for r in s} == {"v8", "v9"}, s
    md = {"metadata": {"licenseurl": "https://creativecommons.org/licenses/by/4.0/", "title": "T", "creator": "C"},
          "files": [{"name": "a.mp3", "source": "original", "length": "01:00:00", "size": "1000"},
                    {"name": "a_64kb.mp3", "source": "derivative"}, {"name": "x.txt", "source": "original"}]}
    rows = D.ia_candidates("item1", md, "en")
    assert len(rows) == 1 and rows[0]["duration"] == 3600.0 and rows[0]["id"] == "ia:item1/a.mp3"
    assert rows[0]["creator"] == "C" and rows[0]["uploader_is_creator"] is None          # khong co uploader -> khong ro
    md2 = {**md, "metadata": {**md["metadata"], "creator": "Open Media Berkeley", "uploader": "x@berkeley.edu"}}
    assert D.ia_candidates("item2", md2, "en")[0]["uploader_is_creator"] is True
    ev = {"guid": "g1", "original_language": "eng", "length": 1800, "title": "t",
          "recordings": [{"mime_type": "audio/mpeg", "language": "eng", "recording_url": "u1", "size": 10},
                         {"mime_type": "audio/opus", "language": "eng", "recording_url": "u2", "size": 8},
                         {"mime_type": "audio/opus", "language": "deu", "recording_url": "u3"}]}
    c = D.ccc_candidate(ev, "37c3")
    assert c["url"] == "u2" and c["lang"] == "en"
    assert D.ccc_lang("deu") == "de" and D.ccc_lang("deu-eng") is None and D.ccc_lang(None) is None


def test_discover_fosdem_parse():
    from v6.scrape import discover as D
    page = ('<tr><td class="link"><a href="../">Parent directory/</a></td><td class="size">-</td></tr>'
            '<tr><td class="link"><a href="fosdem-2024-1734-x.av1.webm" title="t">fosdem-2024-1734-x.av1.webm</a></td>'
            '<td class="size">77.6 MiB</td><td class="date">d</td></tr>'
            '<tr><td class="link"><a href="fosdem-2024-1734-x.mp4" title="t">fosdem-2024-1734-x.mp4</a></td>'
            '<td class="size">567.6 MiB</td></tr>')
    files = D.fosdem_list(page)
    assert files == [("fosdem-2024-1734-x.av1.webm", "77.6 MiB")], files
    assert D.size_bytes("77.6 MiB") == int(77.6 * 2 ** 20) and D.size_bytes("-") == 0
    probe = {"format": {"duration": "1500.5"}, "streams": [{"codec_type": "video", "codec_name": "av1"},
                                                          {"codec_type": "audio", "codec_name": "opus"}]}
    c = D.fosdem_candidate(2024, "ua2220", *files[0], probe)
    assert c["id"] == "fosdem:2024/ua2220/fosdem-2024-1734-x" and c["duration"] == 1500.5 and c["acodec"] == "opus"
    assert c["channel_id"] == "fosdem:2024/ua2220" and c["url"].endswith("/2024/ua2220/fosdem-2024-1734-x.av1.webm")
    from v6.scrape import gate as G
    g = G.gate_record(c, {})
    assert g["ok"] and "license_site_default" in g["flags"], g
    pid = "UC2JPCC4OKl5oQUeiLBFR--w"                          # kenh bi loai ca kenh (tre em, lech ngon ngu)
    gp = G.gate_record(_cand(channel_id=pid), {pid: {**WL["UCa"], "channel_id": pid}})
    assert not gp["ok"] and "channel_reject:children_english_lessons" in gp["reasons"], gp
    from v6.scrape import fetch as Fe
    path, how = Fe.open_target(c)
    assert how == "ffmpeg_copy_audio" and path.endswith(".webm") and os.sep + "fosdem" + os.sep in path
    assert Fe.est_bytes(c, None) == 1500.5 * Fe.DEFAULT_KBPS * 1000 / 8      # khong dung size cua VIDEO
    path2, how2 = Fe.open_target({"source": "ia", "url": "https://archive.org/download/it/a%20b.MP3",
                                  "channel_id": "ia:it", "id": "ia:it/a b.MP3"})
    assert how2 == "urllib" and path2.endswith(".mp3")


def test_discover_search_tabs_rooms():
    from v6.scrape import discover as D
    url, sp = D.search_url("hội thảo")
    assert sp == D.CC_LONG_SP and "search_query=h%E1%BB%99i+th%E1%BA%A3o" in url and url.endswith("&sp=" + sp)
    info = {"entries": [{"id": "v1", "channel_id": "UCx", "channel": "X", "duration": 1500, "title": "t", "ie_key": "Youtube"},
                        {"id": "PL1", "ie_key": "YoutubeTab", "channel_id": "UCx"}, None, {"id": "v2", "ie_key": "Youtube"}]}
    rows = D.search_rows(info, "q", sp)
    assert [r["id"] for r in rows] == ["v1"] and rows[0]["channel"] == "X" and rows[0]["query"] == "q"
    flat = ([{"id": f"a{i}", "channel_id": "UCa", "tab": "videos"} for i in range(60)]          # pilot: cham tran 60
            + [{"id": "b0", "channel_id": "UCb", "tab": "videos"}]                             # tab ngan: da het
            + [{"id": None, "channel_id": "UCc", "tab": "streams", "empty": True}]
            + [{"id": f"d{i}", "channel_id": "UCd", "tab": "videos", "per_tab": 200} for i in range(150)])
    st = D.tabs_to_list(flat, 200)
    assert st == {("UCa", "videos"): "deepen", ("UCb", "videos"): "skip", ("UCc", "streams"): "skip",
                  ("UCd", "videos"): "skip"}, st
    assert D.tabs_to_list(flat, 60)[("UCa", "videos")] == "skip"
    assert set(D.tabs_to_list(flat, 60, refresh=True).values()) == {"refresh"}
    idx = ('<tr><td class="link"><a href="../">Parent directory/</a></td></tr>'
           '<tr><td class="link"><a href="ua2220/" title="ua2220">ua2220/</a></td><td class="size">-</td></tr>'
           '<tr><td class="link"><a href="h1302/" title="h1302">h1302/</a></td><td class="size">-</td></tr>')
    assert D.fosdem_rooms(idx) == ["h1302", "ua2220"]
    assert "licenses\\/by\\/" in D.IA_QUERY_BY and "by-nc" not in D.IA_QUERY_BY and "by-nd" not in D.IA_QUERY_BY


def test_qc_offgrid_audio():
    """Ban sao cat lech luoi 100 ms (850 mau = 53 ms) phai trung o CA duong host (pha tu wav) lan tep-vs-tep (env8)."""
    from v6.scrape import qc as Q
    from v6.scrape.validate_cuts import synth_speech
    y = synth_speech(150, seed=31)
    off = 16000 * 40 + 850
    cp = (y[off:off + 16000 * 60] * 0.5).astype(np.float32)      # ban cat ngan 60 s, nho hon 6 dB
    other = synth_speech(120, seed=32)
    fe = FR.FineEnvelope()
    fe.feed(y[:500001])
    fe.feed(y[500001:])                                            # bien khoi le: van dung
    ph = FR.phases_from_env8(fe.finish(), 8)
    for k in (0, 3, 7):                                            # env8 pha k == duong bao 100 ms bat dau tu mau 200 k
        ref = Q.env_from_audio(y, k * 200)
        n = min(len(ref), len(ph[k]))
        assert n > 1000 and np.abs(ref[:n] - ph[k][:n]).max() < 1e-2, k
    idx = Q.Index()
    idx.add("A", Q.phases_from_y(y)[0], 4)
    idx.add("C", Q.phases_from_y(other)[0], 4)
    idx.finish()
    m = idx.query_multi(Q.phases_from_audio(cp))                   # duong host
    assert m and m[0]["name"] == "A" and abs(m[0]["offset_s"] + 40.05) <= 0.11, m
    res = Q.dedup([("A", Q.phases_from_y(y)), ("B", Q.phases_from_y(cp)), ("C", Q.phases_from_y(other))], stride=4)
    pairs = {tuple(sorted((d["a"], d["b"]))) for d in res["dup_pairs"]}
    assert pairs == {("A", "B")} and res["n_approx"] == 0, res["dup_pairs"]
    assert abs(abs(res["dup_pairs"][0]["offset_s"]) - 40.05) <= 0.11, res["dup_pairs"]
    assert res["dup_pairs"][0]["corr"] >= 0.9 and not res["rejected_pairs"], res
    # xac minh: cung ung vien nhung do lech SAI / noi dung khac -> tuong quan thap -> bi bo
    qa = Q.phases_from_audio(cp)
    good = Q.verify({**m[0], "phase": m[0]["phase"]}, qa, Q.phases_from_y(y)[0])
    bad = Q.verify({**m[0], "offset_fr": m[0]["offset_fr"] + 37}, qa, Q.phases_from_y(y)[0])
    other_q = Q.verify(m[0], Q.phases_from_audio(other[:16000 * 60]), Q.phases_from_y(y)[0])
    assert good["verified"] and not bad["verified"] and not other_q["verified"], (good["corr"], bad["corr"], other_q["corr"])


def test_qc_fingerprint_dedup():
    from v6.scrape import qc as Q
    rng = np.random.default_rng(0)
    n = 9000                                            # 15 phut duong bao 100 ms
    a = (-30 + 12 * np.sin(np.arange(n) / 7.0) * rng.random(n) + rng.normal(0, 4, n)).astype(np.float32)
    b = a[3000:7000] + 3.0 + rng.normal(0, 0.3, 4000).astype(np.float32)   # ban cat ngan + tang ich + ma hoa lai
    c = (-30 + rng.normal(0, 6, n)).astype(np.float32)                      # noi dung khac
    h, p = Q.fp_hashes(a, 1)
    h4, p4 = Q.fp_hashes(a, 4)
    assert len(h) > 1000 and set(p4.tolist()) <= set(p.tolist()) and all(x % 4 == 0 for x in p4[:50])
    res = Q.dedup([("A", a), ("B", b), ("C", c)], stride=4, against=[("hostB", b[100:900])])
    pairs = {(d["a"], d["b"]) for d in res["dup_pairs"]}
    assert pairs == {("A", "B")}, res["dup_pairs"]
    d = res["dup_pairs"][0]
    assert abs(abs(d["offset_s"]) - 300.0) <= 0.2, d
    assert [x["scraped"] for x in res["against_hits"]][:1] in (["A"], ["B"]), res["against_hits"]
    flat = np.full(500, -60, np.float32)                # lang deu: khong sinh bam
    assert len(Q.fp_hashes(flat)[0]) == 0


def test_fetch_plan_caps():
    from v6.scrape import fetch as Fe
    gated = [{"id": f"a{i}", "ok": True, "channel_id": "UCa", "lang": "vi", "duration": 3600, "prefer": i % 2,
              "format_id": "251"} for i in range(5)]
    gated += [{"id": "b0", "ok": True, "channel_id": "UCb", "lang": "en", "duration": 1800, "format_id": "251"},
              {"id": "bad", "ok": False, "channel_id": "UCb", "lang": "vi", "duration": 600}]
    cands = {g["id"]: {"audio_formats": F251} for g in gated}
    p = Fe.plan(gated, cands, [], "vi", per_channel=3, max_hours_channel=10, max_gb=3, order="prefer", max_dur=None)
    assert [x["id"] for x in p] == ["a1", "a3", "a0"], p
    p = Fe.plan(gated, cands, [{"id": "a1", "ok": True, "channel_id": "UCa", "duration": 3600 * 9.5, "bytes": 0}],
                "vi", 3, 10, 3, "prefer", None)
    assert p == []                                    # tran gio kenh
    done2 = [{"id": f"a{i}", "ok": True, "channel_id": "UCa", "duration": 600, "bytes": 0} for i in (0, 1)]
    p = Fe.plan(gated, cands, done2, "vi", 2, 99, 9, "prefer", None)
    assert p == []                                    # tran so tep / kenh tinh CA cac lan chay truoc
    p = Fe.plan(gated, cands, [], None, 9, 99, 0.07, "short", None)
    assert len(p) == 2 and {x["id"] for x in p} == {"b0", "a0"}   # tran GB: ~45 MB / h moi tep
    assert Fe.est_bytes({"duration": 3600, "audio_formats": F251}, "251") == 3600 * 100 * 1000 / 8


def test_fetch_plan_religious_and_flags():
    from v6.scrape import fetch as Fe
    gated = [{"id": "r0", "ok": True, "channel_id": "UAr", "lang": "vi", "duration": 1200, "format_id": "251",
              "flags": ["religious_teaching"]},
             {"id": "z0", "ok": True, "channel_id": "UZz", "lang": "vi", "duration": 1200, "format_id": "251", "flags": []},
             {"id": "z1", "ok": True, "channel_id": "UZz", "lang": "vi", "duration": 600, "format_id": "251",
              "flags": ["minor_risk"]}]
    cands = {g["id"]: {"audio_formats": F251} for g in gated}
    def ids(**kw):
        return [x["id"] for x in Fe.plan(gated, cands, [], "vi", 9, 99, 9, "short", None, **kw)]
    assert ids() == ["z0", "z1", "r0"]            # kenh ton giao SAU CUNG; video minor_risk sau video thuong cung kenh
    assert ids(religious="skip") == ["z0", "z1"]
    assert ids(religious="mix") == ["r0", "z0", "z1"]
    saved = dict(Fe.NO_NEW_FETCH)
    try:
        Fe.NO_NEW_FETCH["UZz"] = "kenh lon (17/10)"          # kenh chuyen nghiep: khong tai them
        assert ids() == ["r0"]
    finally:
        Fe.NO_NEW_FETCH.clear()
        Fe.NO_NEW_FETCH.update(saved)


def test_probe_gate():
    from v6.scrape import probe as P
    ff = {"streams": [{"codec_type": "audio", "codec_name": "opus", "sample_rate": "48000", "channels": 2}],
          "format": {"duration": "3600.0", "size": str(int(45e6)), "format_name": "matroska,webm"}}
    r = P.probe_gate(ff, 3601, "yt", "251")
    assert r["ok"] and r["kenh"] == "yt_opus251" and 99 < r["bitrate_kbps"] < 101
    ff2 = {"streams": [{"codec_type": "audio", "codec_name": "aac", "profile": "HE-AAC", "sample_rate": "22050",
                        "channels": 2, "bit_rate": "48000"}], "format": {"duration": "100"}}
    r = P.probe_gate(ff2, 200, "yt", "139")
    assert not r["ok"] and {"bitrate_lt_64k", "duration_mismatch"} <= set(r["reasons"]) and "he_aac" in r["flags"]
    assert P.probe_gate({"streams": []}, None)["reasons"] == ["no_audio_stream"]
    mono = {"streams": [{"codec_type": "audio", "codec_name": "opus", "sample_rate": "48000", "channels": 1,
                         "bit_rate": "60000"}], "format": {"duration": "1500"}}
    r = P.probe_gate(mono, 1500, "fosdem")                    # Opus mono 60 kbps (FOSDEM) -> qua
    assert r["ok"] and r["kenh"] == "fosdem_opus", r
    st = {**mono, "streams": [{**mono["streams"][0], "channels": 2}]}
    assert "bitrate_lt_64k" in P.probe_gate(st, 1500, "fosdem")["reasons"]   # stereo 60 kbps -> chan


# ---------------- frames / filter ----------------
def test_frames_digital_silence_across_blocks():
    acc = FR.FrameAccumulator(use_vad=False)
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(SR_ := 16000 * 4) * 0.05).astype(np.float32)
    y[int(1.99 * 16000):int(2.06 * 16000)] = 0.0          # 70 ms, cat ngang bien khoi 2 s
    y[int(3.0 * 16000):int(3.02 * 16000)] = 0.0           # 20 ms: qua ngan
    acc.feed(y[:32000])
    acc.feed(y[32000:])
    F = acc.finish()
    ev = F["ds_events"]
    assert len(ev) == 1 and abs(ev[0][0] - 1.99) < 0.002 and abs(ev[0][1] - 0.07) < 0.002, ev
    assert len(F["rms_db"]) == 40 and F["spec16"].shape[0] == 40 and float(F["dur_s"]) == 4.0
    del SR_


def test_frames_speech_mask_and_cuts_synthetic():
    from v6.scrape.validate_cuts import synth_speech
    y = synth_speech(60, seed=1)
    F, sp, nf = FR.analyse_array(y, use_vad=False)
    assert 0.3 < sp.mean() < 0.95, sp.mean()
    assert len(FR.detect_cuts(F, sp)) <= 2                  # tin hieu goc: it bao nham
    z = y.copy()
    z[30 * 16000:] += (np.random.default_rng(3).standard_normal(len(z) - 30 * 16000) * 10 ** (-42 / 20)).astype(np.float32)
    F2, sp2, _ = FR.analyse_array(z, use_vad=False)
    ev = FR.detect_cuts(F2, sp2)
    assert any(e["type"] == "b_noise_floor_step" and e["lo"] - 1 <= 30 <= e["hi"] + 1 for e in ev), ev
    zz = np.concatenate([y[:20 * 16000], np.zeros(1600, np.float32), y[20 * 16000:]])
    F3, sp3, _ = FR.analyse_array(zz, use_vad=False)
    assert any(e["type"] == "a_digital_silence" and abs(e["t"] - 20.05) < 0.5 for e in FR.detect_cuts(F3, sp3))


def test_frames_edge_bandwidth_and_gated():
    import scipy.signal as ss
    rng = np.random.default_rng(11)
    x = rng.standard_normal(16000 * 20).astype(np.float32) * 0.1
    b, a = ss.butter(8, 3800 / 8000)
    xl = ss.lfilter(b, a, x).astype(np.float32)
    for sig, lo, hi in ((x, 7000, 8001), (xl, 3000, 5800)):
        F, _, _ = FR.analyse_array(sig, use_vad=False)
        e = float(np.percentile(F["edge"], 90))
        assert lo <= e <= hi, (lo, e, hi)
    F = {"rms_db": np.zeros(36000), "ds_events": np.zeros((100, 2))}
    assert FR.gated_floor(F) == (True, 100.0)
    F["ds_events"] = np.zeros((10, 2))
    assert FR.gated_floor(F)[0] is False


def test_filter_spans_and_decide():
    from v6.scrape import filter as Fi
    assert Fi.subtract([(0, 100)], [(10, 20), (50, 60), (95, 120)]) == [(0, 10), (20, 50), (60, 95)]
    n = 3000                                                # 300 s
    sp = np.ones(n, bool)
    sp[1000:2300] = False                                   # 130 s khong loi -> bo
    music = np.zeros(n, bool)
    spans = Fi.keep_spans(300.0, sp, music, [{"t": 50.0}], [], False, 1.0, min_span=10)
    assert spans == [[1.0, 49.0], [51.0, 100.0], [230.0, 299.0]], spans
    from v6.scrape.validate_cuts import synth_speech
    F, spm, nf = FR.analyse_array(synth_speech(120, seed=2), use_vad=False)
    res = Fi.decide(F, spm, nf, {"n_spk_est": 1, "dominant_share": 1.0}, {"title": "Lớp học Zoom"})
    assert res["conf_call"] and res["cut_mode"] in ("clean", "split") and "keep_spans" in res
    assert res["ok"] == (not res["reasons"]) and (res["pool"] or "").endswith("lecture") == res["ok"]


def test_filter_decode_blocks_and_file():
    from v6.scrape import filter as Fi
    from v6.scrape.validate_cuts import synth_speech
    os.makedirs(WORK, exist_ok=True)
    p = os.path.join(WORK, "s.wav")
    sf.write(p, synth_speech(150, seed=4), 16000)
    blocks = list(Fi.decode_blocks(p, block_s=60))
    assert [len(b) for b in blocks] == [960000, 960000, 480000]
    blocks2 = []
    for b in Fi.decode_blocks(p, block_s=60):
        blocks2.append(b)
        break                                             # dung som: khong raise
    old = U.FRAMES
    U.FRAMES = os.path.join(WORK, "frames")
    try:
        res = Fi.analyse_file(p, {"id": "selftest:s", "title": ""}, use_spk=False)
        assert os.path.exists(Fi.frames_path("selftest:s")) and res["dur_s"] == 150.0
    finally:
        U.FRAMES = old


def test_count_speakers_mfcc_proxy():
    from v6.scrape import filter as Fi
    from v6.scrape.validate_cuts import synth_speech
    a = synth_speech(30, seed=5, f0=110)
    b = synth_speech(30, seed=6, f0=230)
    wins = [(i, (a if i % 2 else b)[i * 16000:(i + 3) * 16000]) for i in range(0, 24, 3)]
    r = Fi.count_speakers(wins, use_model=False)
    assert r["spk_method"] == "mfcc_proxy" and r["n_windows"] == 8 and 0 < r["dominant_share"] <= 1
    assert Fi.count_speakers(wins[:2], use_model=False)["n_spk_est"] is None


# ---------------- segment / align / hosts ----------------
def test_segment_plan_and_cut():
    from v6.scrape import segment as Se
    sp = np.zeros(6000, bool)                               # 600 s, cau 4 s / lang 0,5 s
    for s in range(0, 6000, 45):
        sp[s:s + 40] = True
    rng = random.Random(0)
    w = Se.plan_windows(sp, [[0.0, 600.0]], Se.STD, rng, 1000)
    assert w and all(5.0 < b - a <= 20.0 and r >= 0.6 for a, b, r in w)
    assert all(w[i][1] <= w[i + 1][0] + 1e-6 for i in range(len(w) - 1))   # khong chong
    wl = Se.plan_windows(sp, [[0.0, 600.0]], Se.LONG, rng, 3)
    assert len(wl) == 3 and all(30.0 < b - a <= 120.0 for a, b, _ in wl)
    y = np.arange(16000 * 200, dtype=np.float32) / (16000 * 200)
    os.makedirs(WORK, exist_ok=True)
    wins = [{"start": 10.0, "end": 15.0, "path": os.path.join(WORK, "w1.wav")},
            {"start": 100.0, "end": 190.0, "path": os.path.join(WORK, "w2.wav")}]
    done = Se.cut_stream((y[i:i + 16000 * 60] for i in range(0, len(y), 16000 * 60)), wins)
    z, _ = sf.read(wins[1]["path"], dtype="float32")
    assert done == {w["path"] for w in wins} and len(z) == 90 * 16000 and abs(z[0] - y[100 * 16000]) < 1e-3
    assert Se.safe_vid("ia:item/a b.mp3") == "item_a_b_mp3"
    # giai ma dung som (120 s): chi cua so da du du lieu, tra DUNG duong dan da ghi (khong theo thu tu chuan-roi-dai)
    w2 = [{"start": 100.0, "end": 190.0, "path": os.path.join(WORK, "w4.wav")},
          {"start": 10.0, "end": 15.0, "path": os.path.join(WORK, "w3.wav")}]
    done2 = Se.cut_stream((y[i:i + 16000 * 60] for i in range(0, 16000 * 120, 16000 * 60)), w2)
    assert done2 == {w2[1]["path"]} and not os.path.exists(w2[0]["path"])
    filt = [{"id": "a", "ok": True, "lang": "vi"}, {"id": "b", "ok": True, "lang": "vi"},
            {"id": "c", "ok": True, "lang": "vi"}, {"id": "d", "ok": False, "lang": "vi"}, {"id": "e", "ok": True, "lang": "vi"}]
    gated = {"a": {"ok": True, "flags": []}, "c": {"ok": True, "flags": ["religious_teaching"]},
             "e": {"ok": False, "flags": []}}
    rows, why = Se.select_rows(filt, gated, "vi")
    assert [r["id"] for r in rows] == ["a"] and why == {"khong_co_cong": 1, "co:religious_teaching": 1, "cong_loai": 1}, why
    assert [r["id"] for r in Se.select_rows(filt, gated, "vi", ("religious_teaching",))[0]] == ["a", "c"]


def test_align_split_points():
    from v6.scrape import align as Al
    from vaudit.asr import Word
    sr = 16000
    y = np.ones(sr * 50, np.float32)
    y[int(14.2 * sr):int(14.4 * sr)] = 0.0
    pts = Al.split_points(y, sr)
    assert pts[0] == 0 and pts[-1] == len(y) and abs(pts[1] / sr - 14.3) < 0.1, [p / sr for p in pts]

    def fake(z):
        return "a", [Word("a", 0.1, 0.2)]
    text, words = Al.transcribe_long(y, sr, fake)
    assert len(words) == len(pts) - 1 and abs(words[1].start - (pts[1] / sr + 0.1)) < 1e-6


def test_hosts_rows_and_caps():
    from v6.scrape import hosts as H
    segs = [{"file": f"yt_v{i // 5}_s{i:03d}", "tier": "std", "lang": "vi", "source": "yt", "id": f"v{i // 5}",
             "channel_id": "UCa", "start": float(i * 10), "end": float(i * 10 + 10), "dur": 10.0} for i in range(20)]
    words = [[f"w{k}", k * 0.5, k * 0.5 + 0.3] for k in range(20)]
    aligned = {s["file"]: {"text": "x", "words": words, "n_words": 20, "wps": 2.0} for s in segs}
    aligned[segs[0]["file"]] = {"text": "x", "words": words[:3], "n_words": 3, "wps": 0.3}
    fetched = {f"v{j}": {"url": f"https://youtu.be/v{j}", "license": CC, "file": f"raw/yt/UCa/v{j}.f251.webm"}
               for j in range(4)}
    probed = {f"v{j}": {"codec": "opus", "bitrate_kbps": 100.0, "kenh": "yt_opus251"} for j in range(4)}
    filtered = {f"v{j}": {"pool": "wide_lecture", "flags": []} for j in range(4)}
    gated = {f"v{j}": {"ok": True, "flags": []} for j in range(4)}
    gated["v1"] = {"ok": True, "flags": ["minor_risk"]}
    lid = {s["file"]: {"p_target": 0.99, "top": "vi", "model": "m"} for s in segs}
    saved = H._VOCAB.get("vi")
    H._VOCAB["vi"] = {f"w{k}" for k in range(20)}             # tu vung gia: khong doc tep that
    kw = {"lid": lid, "check_wav": False}
    try:
        rows, why = H.build_rows(segs, aligned, fetched, probed, filtered, gated, "vi", "std", **kw)
        assert len(rows) == 14 and why == {"it_tu": 1, "co_loai:minor_risk": 5}, why          # minor_risk: loai mac dinh
        rows, why = H.build_rows(segs, aligned, fetched, probed, filtered, gated, "vi", "std", ("minor_risk",), **kw)
        assert len(rows) == 19 and why == {"it_tu": 1}, why                                   # --allow-flag mo lai
        assert all(r["lex_share"] == 1.0 and r["lang_check"] == "lid" and r["lid_p"] == 0.99 for r in rows)
        junk = {**aligned, segs[1]["file"]: {"text": "x", "words": [[f"zz{k}", k, k + 0.5] for k in range(20)],
                                             "n_words": 20, "wps": 2.0}}
        g2 = {k: v for k, v in gated.items() if k != "v2"}                                     # thieu dong cong
        g2["v3"] = {"ok": False, "flags": []}
        lid2 = {**lid, segs[6]["file"]: {"p_target": 0.5, "top": "en"}}
        lid2.pop(segs[7]["file"])
        _, why2 = H.build_rows(segs, junk, fetched, probed, filtered, g2, "vi", "std", ("minor_risk",), lid=lid2,
                               check_wav=False)
        assert why2 == {"it_tu": 1, "khong_co_cong": 5, "cong_loai": 5, "lid_lech": 1, "chua_lid": 1}, why2
        rows3, why3 = H.build_rows(segs, junk, fetched, probed, filtered, gated, "vi", "std", ("minor_risk",),
                                   lid=None, require_lid=False, check_wav=False)              # khong LID: proxy (phu)
        assert why3 == {"it_tu": 1, "lech_ngon_ngu": 1} and rows3[0]["lang_check"] == "lex_proxy_uncalibrated", why3
        _, why4 = H.build_rows(segs, aligned, fetched, probed, filtered, gated, "vi", "std", ("minor_risk",), lid=lid)
        assert why4 == {"thieu_wav": 20}, why4                                                # wav khong ton tai
        rs, _ = H.build_rows(segs, aligned, fetched, probed, filtered, gated, "vi", "std", ("minor_risk",), **kw,
                             spk_map={"yt:UCa": "scrape:yt:UCa+1"})
        assert {r["group"] for r in rs} == {"scrape:yt:UCa+1"} == {r["person"] for r in rs}  # kenh da hop (spk_cluster)
    finally:
        if saved is None:
            H._VOCAB.pop("vi", None)
        else:
            H._VOCAB["vi"] = saved
    assert H.lex_share([["a", 0, 1], ["b", 1, 2]], {"a"}) == 0.5 and H.lex_share([], {"a"}) is None
    r = rows[0]
    need = {"file", "source", "cluster", "person", "dur", "text", "words", "url", "channel_id", "license", "codec",
            "bitrate", "orig_file", "group"}
    assert need <= set(r) and r["cluster"].startswith("yt:UCa:") and r["group"] == r["person"]
    assert any("minor_risk" in x["flags"] for x in rows)
    capped = H.cap_groups(rows, max_hours=60 / 3600)
    assert len(capped) == 6 and len({x["video_id"] for x in capped}) == 4     # vong tron theo video


def test_hosts_calib_lex_smoke():
    from v6.scrape import hosts as H
    out = os.path.join(WORK, "lex_calib.json")
    res = H.calib_lex(max_n=150, out_path=out)
    assert os.path.exists(out) and res["lex_min"] == H.LEX_MIN and not H.LEX_CALIBRATED
    for lang in ("vi", "en"):
        d = res[lang]
        assert d["n_pos"] > 0 and 0 <= d["pos_frac_below_lex_min"] <= 1 and d["n_neg_other_lang"] > 0, (lang, d)


def test_module_selftests():
    """Kiem thu rieng cua module moi (khong nap mo hinh): quarantine, lid (ham phu), spk_cluster (gom cum / hop kenh)."""
    from v6.scrape import lex_video as LV
    from v6.scrape import lid as L
    from v6.scrape import quarantine as Qr
    from v6.scrape import spk_cluster as Sc
    Qr.test_plan_and_apply(os.path.join(WORK, "quar"))
    L.test_summarize_and_todo()
    LV.test_video_lex_and_rule()
    Sc.test_centroids_and_merge()


# ---------------- validate_cuts / build_sets ----------------
def test_validate_cuts_plant_and_match():
    from v6.scrape import validate_cuts as V
    rng = random.Random(1)
    y = V.synth_speech(200, seed=7)
    other = V.synth_speech(30, seed=8, noise_db=-45)
    z, joints = V.plant_many(y, 5, rng, other)
    assert len({j["edit"] for j in joints}) == 5 and all(0 < j["t"] < len(z) / 16000 for j in joints)
    assert all(joints[i]["t"] <= joints[i + 1]["t"] for i in range(len(joints) - 1))
    per, nh = V.match([{"t": joints[0]["t"] + 0.5}, {"t": 1.0}], [dict(j) for j in joints])
    assert sum(h for h, _ in per.values()) >= 1 and nh == 1
    r = V.run_excerpt(V.synth_speech(120, seed=9), other, 3, random.Random(2), use_vad=False)
    s = V.summarize([r])
    assert s["n_excerpts"] == 1 and s["recall_all"] is not None and 0 <= s["recall_all"] <= 1
    sp = V.summarize([r], "_prod")                      # cau hinh production (gated=None nhu filter.decide)
    n_edits = sum(t for _, t in r["per_kind"].values())
    assert sp["recall_all"] is not None and n_edits == 3 == sum(t for _, t in r["per_kind_prod"].values()), r["per_kind_prod"]
    assert {"base_dets_prod", "dets_prod", "n_det_prod", "gated_edited"} <= set(r)


def test_build_sets_smoke():
    from v6.scrape import build_sets as B
    from v6.scrape.validate_cuts import synth_speech
    root = os.path.join(WORK, "bs")
    hd = os.path.join(root, "hosts_vi")
    os.makedirs(os.path.join(hd, "yt"), exist_ok=True)
    rows = []
    for i in range(4):
        y = synth_speech(12, seed=20 + i)
        f = f"yt_vid{i // 2}_s{i:03d}"
        sf.write(os.path.join(hd, "yt", f + ".wav"), y, 16000)
        words = [[f"t{k}", 0.4 + k * 0.55, 0.4 + k * 0.55 + 0.4] for k in range(19)]
        rows.append({"file": f, "source": "yt", "cluster": f"yt:UCa:vid{i // 2}", "person": "yt:UCa", "dur": 12.0,
                     "text": " ".join(w[0] for w in words), "words": words, "group": "yt:UCa", "pool": "wide_lecture",
                     "url": "u", "channel_id": "UCa", "license": CC})
    U.write_jsonl(os.path.join(hd, "hosts.jsonl"), rows)
    old = U.hosts_dir
    U.hosts_dir = lambda lang, tier="std": hd
    try:
        man = B.build("vi", 0.005, 0, ["nguyen_ban", "xoa_loi", "ghep_thay", "ghep_chen"], os.path.join(root, "sets"))
    finally:
        U.hosts_dir = old
    kinds = {e["kind"] for e in man}
    assert {"nguyen_ban", "xoa_loi"} <= kinds, kinds
    for e in man:
        m = e["meta"]
        assert e["id"].startswith("v6scrapevi_") and m["person"] == m["speaker"] == "yt:UCa"
        assert m["recording"].startswith("yt:UCa:vid") and 0 <= m["fold"] <= 4 and m["scrape"]["license"] == CC
        assert os.path.exists(os.path.join(B.V4, e["wav"]))
        assert all(not str(x).startswith(("may_", "03_giong_may")) for x in m["level2"])
    # 17/10: host bi go (kenh cach ly) -> muc dung no bi bo khi chay lai, wav chuyen _pruned, id moi khong ghi de
    gone = {r["file"] for r in rows if "vid0" in r["file"]}
    U.write_jsonl(os.path.join(hd, "hosts.jsonl"), [r for r in rows if r["file"] not in gone])
    U.hosts_dir = lambda lang, tier="std": hd
    try:
        man2 = B.build("vi", 0.005, 0, ["nguyen_ban", "xoa_loi", "ghep_thay", "ghep_chen"], os.path.join(root, "sets"))
    finally:
        U.hosts_dir = old
    assert all(not (B.entry_files(e) & gone) for e in man2), "con muc dung host da go"
    assert len({e["id"] for e in man2}) == len(man2) and all(os.path.exists(os.path.join(B.V4, e["wav"])) for e in man2)
    n_drop = sum(bool(B.entry_files(e) & gone) for e in man)
    assert n_drop == 0 or os.path.exists(os.path.join(root, "sets", "scrapevi", "pruned.jsonl"))
    keep, drop = B.split_manifest([{"meta": {"host_file": "a", "edits": [{"donor_file": "b"}]}},
                                   {"meta": {"host_file": "a", "edits": []}}], {"a"})
    assert len(keep) == 1 and len(drop) == 1                                    # doan cho (donor) bi go cung bo
    assert B.next_index([{"id": "p_k_0000"}, {"id": "p_k_0007"}], "p_", "k") == 8
    n7 = {"speaker": "yt:UCa:vid0", "level2": ["02_nen_toan_cuc"],
          "edits": [{"level2": "02_nen_toan_cuc", "kind": "global", "family": "N7", "tool": "WebRTC NS"}]}
    assert B._meta_fix(n7, rows[0], "vi").get("app_ns") is True               # quyet dinh 4: NS ung dung -> 01 khi doc
    n3 = {**n7, "edits": [{"level2": "02_nen_toan_cuc", "kind": "global", "family": "N3"}]}
    n7seg = {**n7, "edits": [{"level2": "02_nen_cuc_bo", "kind": "segment", "family": "N7"}]}
    assert "app_ns" not in B._meta_fix(n3, rows[0], "vi") and "app_ns" not in B._meta_fix(n7seg, rows[0], "vi")


def test_report_smoke():
    from v6.scrape import report as R
    rep = R.collect()                                      # chi doc meta/*.jsonl that (co the rong)
    md = R.to_md(rep)
    assert "## vi" in md and "## en" in md and set(rep["vi"]) >= {"license", "gate", "fetch", "probe", "filter"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v) and flt in k]
    shutil.rmtree(WORK, ignore_errors=True)
    bad = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
        except Exception:  # noqa: BLE001
            bad += 1
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    shutil.rmtree(WORK, ignore_errors=True)
    print(f"{len(tests) - bad}/{len(tests)} PASS")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
