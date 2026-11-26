"""Kiểm danh sách ứng viên Gemini đề xuất (gemini/UNG_VIEN_CAO_2411.jsonl) theo luật P1_QUY_TRINH §4–5. CHỈ metadata, không tải âm thanh.

Với mỗi video: đọc metadata THẬT (yt-dlp, skip_download) → cao_tunhien.gate() (giấy phép CC BY, phụ lục A, tập khoá v2, tích xanh,
≤ 100 000 người theo dõi, thể loại, tôn giáo / trẻ em / trường phổ thông, podcast / sách nói / lồng tiếng / truyền hình …)
+ luật phiếu giao việc: 3–40 phút · kênh không có trong KENH_DA_DUNG_2411.txt · ≤ 3 video / kênh · đúng ngôn ngữ khai.
Đối chiếu lời khai Gemini (giấy phép, mã kênh, người theo dõi, độ dài) với metadata thật → ghi chỗ mâu thuẫn.
Ra: data/p1_scrape/gemini_2411_meta.jsonl (metadata, chạy lại bỏ qua video đã đọc) · data/p1_scrape/SOAT_UNG_VIEN_GEMINI_2511.md

  .venv/Scripts/python.exe experts/P1/scrape/kiem_ung_vien_gemini.py meta | soat | selftest
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cao_tunhien as T  # noqa: E402

GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
SRC, DUNG = os.path.join(GEM, "UNG_VIEN_CAO_2411.jsonl"), os.path.join(GEM, "KENH_DA_DUNG_2411.txt")
META = os.path.join(T.OUT, "gemini_2411_meta.jsonl")
SOAT = os.path.join(T.OUT, "SOAT_UNG_VIEN_GEMINI_2511.md")
MIN_PHUT, MAX_PHUT, MAX_MOI_KENH = 3, 40, 3


def ung_vien() -> list[dict]:
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    return [r for r in rows if "video_id" in r]


def da_dung() -> set[str]:
    return {l.strip() for l in open(DUNG, encoding="utf-8") if l.strip() and not l.startswith("#")}


def meta() -> None:
    done = {json.loads(l)["id"] for l in open(META, encoding="utf-8")} if os.path.exists(META) else set()
    todo = [r["video_id"] for r in ung_vien() if r["video_id"] not in done]
    print(f"{len(todo)} video cần đọc metadata ({len(done)} đã có)", flush=True)
    with open(META, "a", encoding="utf-8") as fh:
        for i, vid in enumerate(todo):
            try:
                m = T.video_meta(vid)
            except Exception as ex:                      # noqa: BLE001  video riêng tư / bị xoá: ghi lỗi, không bỏ ngầm
                m = {"id": vid, "loi": f"{type(ex).__name__}: {str(ex)[:160]}"}
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)}", flush=True)
            time.sleep(0.5)
    print("xong", META)


def ly_do(r: dict, m: dict, appx: set, dung: set) -> list[str]:
    """Lý do loại một video (rỗng = qua). Luật trên metadata thật; lời khai Gemini chỉ để đối chiếu."""
    if "loi" in m:
        return [f"khong_doc_duoc:{m['loi'][:60]}"]
    out = T.gate(m, r["ngon_ngu"], appx)
    d = (m.get("duration") or 0) / 60
    if not MIN_PHUT <= d <= MAX_PHUT:
        out.append(f"do_dai_phut:{d:.1f}")
    if m.get("channel_id") in dung:
        out.append("kenh_da_dung")
    return out


def mau_thuan(r: dict, m: dict) -> list[str]:
    if "loi" in m:
        return []
    out = []
    if r.get("kenh_id") and r["kenh_id"] != m.get("channel_id"):
        out.append(f"kenh_id Gemini {r['kenh_id']} ≠ thật {m.get('channel_id')}")
    if "Creative Commons" in (r.get("giay_phep_thay") or "") and "Creative Commons" not in (m.get("license") or ""):
        out.append(f"giấy phép Gemini CC ≠ thật {m.get('license')}")
    fc, g = m.get("channel_follower_count"), r.get("nguoi_theo_doi")
    if fc is not None and isinstance(g, (int, float)) and g > 0 and not 0.5 <= fc / g <= 2:
        out.append(f"người theo dõi Gemini {g} ≠ thật {fc}")
    if m.get("duration") and abs(m["duration"] / 60 - (r.get("do_dai_phut") or 0)) > 2:
        out.append(f"độ dài Gemini {r.get('do_dai_phut')} ≠ thật {m['duration'] / 60:.1f} phút")
    return out


def gioi_han_kenh(qua: list[dict], meta_of: dict) -> tuple[list, list]:
    """≤ 3 video / kênh (theo mã kênh THẬT); giữ theo thứ tự stt, phần dư → loại."""
    dem, giu, du = collections.Counter(), [], []
    for r in sorted(qua, key=lambda x: x["stt"]):
        c = meta_of[r["video_id"]]["channel_id"]
        dem[c] += 1
        (giu if dem[c] <= MAX_MOI_KENH else du).append(r)
    return giu, du


def soat() -> None:
    appx, dung = T.appendix_a(open(T.RULES, encoding="utf-8").read()), da_dung()
    mo = {}
    for l in open(META, encoding="utf-8"):
        m = json.loads(l)
        mo[m["id"]] = m
    rows = ung_vien()
    assert all(r["video_id"] in mo for r in rows), "thiếu metadata: chạy `meta` trước"
    trung = [v for v, k in collections.Counter(r["video_id"] for r in rows).items() if k > 1]
    loai, qua = {}, []
    seen = set()
    for r in rows:
        if r["video_id"] in seen:
            loai[r["video_id"]] = (r, ["trung_video"])
            continue
        seen.add(r["video_id"])
        ld = ly_do(r, mo[r["video_id"]], appx, dung)
        (loai.__setitem__(r["video_id"], (r, ld)) if ld else qua.append(r))
    giu, du = gioi_han_kenh(qua, mo)
    for r in du:
        loai[r["video_id"]] = (r, ["qua_3_video_moi_kenh"])
    mt = [(r, mau_thuan(r, mo[r["video_id"]])) for r in rows]
    mt = [(r, x) for r, x in mt if x]
    ld_dem = collections.Counter(x.split(":")[0] for _, l in loai.values() for x in l)
    L = ["# Soát ứng viên Gemini `UNG_VIEN_CAO_2411.jsonl` (metadata thật, chưa tải)", "",
         f"- Ứng viên: {len(rows)} (trùng video: {len(trung)}) · **qua: {len(giu)}** "
         f"(vi {sum(r['ngon_ngu'] == 'vi' for r in giu)} / en {sum(r['ngon_ngu'] == 'en' for r in giu)}; "
         f"kênh vi {len({mo[r['video_id']]['channel_id'] for r in giu if r['ngon_ngu'] == 'vi'})} / "
         f"en {len({mo[r['video_id']]['channel_id'] for r in giu if r['ngon_ngu'] == 'en'})}) · loại: {len(loai)}",
         "- Lý do loại (một video có thể nhiều lý do): " + ", ".join(f"{k} {v}" for k, v in ld_dem.most_common()),
         f"- Lời khai Gemini mâu thuẫn metadata thật: {len(mt)} video", "",
         "## Qua (soát TIÊU ĐỀ trước khi tải — luật §4)", "", "| stt | ngôn ngữ | tiêu đề | kênh | phút | người theo dõi | loại (Gemini) | môi trường (Gemini) |",
         "|---|---|---|---|---|---|---|---|"]
    for r in sorted(giu, key=lambda x: (x["ngon_ngu"], x["stt"])):
        m = mo[r["video_id"]]
        L.append(f"| {r['stt']} | {r['ngon_ngu']} | {(m.get('title') or '').replace('|', '/')[:90]} | {(m.get('channel') or '').replace('|', '/')[:30]} | "
                 f"{m['duration'] / 60:.0f} | {m.get('channel_follower_count')} | {r.get('loai_noi_dung')} | {r.get('moi_truong')} |")
    L += ["", "## Loại", "", "| stt | ngôn ngữ | tiêu đề (thật) | lý do |", "|---|---|---|---|"]
    for vid, (r, l) in sorted(loai.items(), key=lambda t: t[1][0]["stt"]):
        L.append(f"| {r['stt']} | {r['ngon_ngu']} | {(mo[vid].get('title') or '').replace('|', '/')[:80]} | {'; '.join(l)} |")
    L += ["", "## Mâu thuẫn lời khai Gemini", ""] + [f"- stt {r['stt']} `{r['video_id']}`: {'; '.join(x)}" for r, x in mt]
    open(SOAT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:6]))


def selftest() -> None:
    m = {"id": "x", "license": "Creative Commons Attribution license (reuse allowed)", "channel_id": "UCa", "channel_follower_count": 500,
         "duration": 600, "title": "phỏng vấn", "categories": ["People & Blogs"]}
    cid_ok = next(f"UC{i}" for i in range(99) if not T.is_lockbox(f"UC{i}"))
    m["channel_id"] = cid_ok
    r = {"stt": 1, "video_id": "x", "ngon_ngu": "vi", "kenh_id": cid_ok, "nguoi_theo_doi": 480, "do_dai_phut": 10, "giay_phep_thay": "Creative Commons"}
    assert ly_do(r, m, set(), set()) == [] and mau_thuan(r, m) == []
    assert "kenh_da_dung" in ly_do(r, m, set(), {cid_ok})
    assert any(x.startswith("do_dai_phut") for x in ly_do(r, {**m, "duration": 45 * 60}, set(), set()))
    assert any("giấy phép" in x for x in mau_thuan(r, {**m, "license": "Standard YouTube License"}))
    rs = [{"stt": i, "video_id": f"v{i}"} for i in range(5)]
    giu, du = gioi_han_kenh(rs, {f"v{i}": {"channel_id": "UCz"} for i in range(5)})
    assert [x["stt"] for x in giu] == [0, 1, 2] and len(du) == 2
    print("kiem_ung_vien_gemini selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"meta": meta, "soat": soat, "selftest": selftest}[sys.argv[1]]()
