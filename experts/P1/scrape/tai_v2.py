"""Tải → cắt đoạn → dựng bộ `p1_tunhien_v2` từ ứng viên Gemini đã soát (Huấn duyệt 25/11: «chỉ loại nhóm A, tải 158 video»).

Danh sách = video qua kiem_ung_vien_gemini (metadata thật + cao_tunhien.gate + 3–40 phút + kênh chưa dùng + ≤ 3 video / kênh)
trừ cao_tunhien.nonspeech (gồm loai_tay.json: nhóm A). Dùng lại hàm THUẦN của tai_cat (download, plan_segments, seg_gate) nhưng
log / bộ / tiền tố id RIÊNG (không chạm p1_tunhien_v1, không gọi tai_cat.build vì hàm đó ghi đè manifest v1).
Đáp án như v1: «accept» (01 có khả năng); video Gemini nghe có nhạc nền rõ → «khong_chac», ca tunhien.<lang>.nhac_nen (luật v1, Huấn chốt 21/11).

  .venv/Scripts/python.exe experts/P1/scrape/tai_v2.py tai | cat | dung | selftest
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cao_tunhien as T  # noqa: E402
import kiem_ung_vien_gemini as G  # noqa: E402
import tai_cat as TC  # noqa: E402

BO, PRE, SO_VIDEO = "p1_tunhien_v2", "p1n2", 158
DST = os.path.join(T.ROOT, "data_labels", "v6", BO)
LOG = {k: (lambda lang, k=k: os.path.join(T.OUT, f"{k}_gemini_{lang}.jsonl")) for k in ("tai", "doan")}


def jl(p: str) -> list[dict]:
    return [json.loads(x) for x in open(p, encoding="utf-8")] if os.path.exists(p) else []


def chon() -> list[dict]:
    """Metadata thật của các video được tải, kèm ngôn ngữ khai và ghi chú Gemini (ghi chú KHÔNG phải nhãn)."""
    appx, dung = T.appendix_a(open(T.RULES, encoding="utf-8").read()), G.da_dung()
    mo = {m["id"]: m for m in jl(G.META)}
    rows, seen, qua = G.ung_vien(), set(), []
    for r in rows:
        if r["video_id"] in seen:
            continue
        seen.add(r["video_id"])
        if not G.ly_do(r, mo[r["video_id"]], appx, dung):
            qua.append(r)
    giu, _ = G.gioi_han_kenh(qua, mo)
    return [{**mo[r["video_id"]], "lang": r["ngon_ngu"], "gemini": r} for r in giu if not T.nonspeech(mo[r["video_id"]])]


def tai() -> None:
    items = chon()
    assert len(items) == SO_VIDEO, f"danh sách {len(items)} ≠ {SO_VIDEO} đã duyệt"          # kiểm TRƯỚC khi tải
    assert max(collections.Counter(m["channel_id"] for m in items).values()) <= 3
    for lang in ("vi", "en"):
        done = {r["id"] for r in jl(LOG["tai"](lang))}
        todo = [m for m in items if m["lang"] == lang and m["id"] not in done]
        print(f"{lang}: cần tải {len(todo)} video", flush=True)
        for i, m in enumerate(todo):
            try:
                row = TC.download(m, lang)
            except Exception as e:  # noqa: BLE001  video lỗi: ghi lại, tải tiếp
                row = {"id": m["id"], "loi": f"{type(e).__name__}: {e}"[:300]}
            with open(LOG["tai"](lang), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{i + 1}/{len(todo)} {m['id']} {'LỖI' if 'loi' in row else row['acodec']}", flush=True)


def cat() -> None:
    """Như tai_cat.segment: 1 đoạn dài + 1 vừa + 2 ngắn / video, cổng CHỈ dấu vết biên tập + ngôn ngữ (không lọc điều kiện K1)."""
    from vaudit import k1x
    meta = {m["id"]: m for m in chon()}
    for lang in ("vi", "en"):
        tai_ok = {r["id"]: r for r in jl(LOG["tai"](lang)) if "loi" not in r}
        done = {r["video"] for r in jl(LOG["doan"](lang))}
        rng = random.Random(2)
        for k, vid in enumerate(sorted(tai_ok)):
            if vid in done:
                continue
            x, _ = sf.read(os.path.join(T.ROOT, tai_ok[vid]["wav16"]), dtype="float32")
            for grp, a, b in TC.plan_segments(len(x) / TC.SR, k, rng):
                y = x[int(a * TC.SR): int(b * TC.SR)]
                lp = k1x.language_probs(y)
                row = {"video": vid, "channel_id": meta[vid]["channel_id"], "nhom_do_dai": grp, "start": a, "end": b,
                       "lang_p": {kk: round(float(v), 3) for kk, v in lp.items() if kk != "top"} | {"top": lp.get("top")},
                       "so_0_dai_s": round(TC.zero_run_s(y), 3), "ly_do_loai": TC.seg_gate(y, lang, lp)}
                with open(LOG["doan"](lang), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{lang} {k + 1}/{len(tai_ok)} {vid}", flush=True)


def ban_ghi(s: dict, m: dict, t: dict, lang: str, idx: int) -> dict:
    rid, abr, g = f"{PRE}_{lang}_{s['video']}_{idx}", t.get("abr"), m["gemini"]
    return {"id": rid, "kind": "tunhien", "cfg": "goc", "src": s["video"], "wav": f"../v6/{BO}/{lang}/{rid}.wav",
            "meta": {"lang": lang, "source": "yt", "person": f"yt:{m['channel_id']}", "speaker": f"yt:{m['channel_id']}",
                     "channel_id": m["channel_id"], "recording": s["video"], "url": m["webpage_url"], "license": m["license"],
                     "level1": ["01_nguyen_ban"], "level2": [], "edits": [], "nhan_chac": "01_co_kha_nang",
                     "kenh": f"yt_{t.get('acodec')}_{round(abr) if abr else '?'}k", "nhom_do_dai": s["nhom_do_dai"],
                     "doan_trong_video_s": [s["start"], s["end"]], "cho_noi": [], "so_nguoi_noi": None, "thiet_bi": None,
                     "truy_van": "gemini_2411", "ngay_cao": datetime.date.today().isoformat(),
                     "gemini_ghi_chu": {k: g.get(k) for k in ("loai_noi_dung", "moi_truong", "co_nhac_nen", "so_nguoi_noi_uoc")}}}


def dap_an(r: dict) -> dict:
    lang, nn = r["meta"]["lang"], r["meta"]["gemini_ghi_chu"]["co_nhac_nen"] == "ro"
    return {"file": f"{lang}/{r['id']}.wav", "ca": f"tunhien.{lang}." + ("nhac_nen" if nn else r["meta"]["nhom_do_dai"]),
            "mong_doi": "khong_chac" if nn else "accept", "ly_do": None, "ca_bien": False, "base": r["meta"]["person"]}


def dung() -> None:
    if os.path.exists(os.path.join(DST, "manifest.json")):
        raise SystemExit(f"{DST} đã có manifest.json: không ghi đè")
    meta, man = {m["id"]: m for m in chon()}, []
    for lang in ("vi", "en"):
        tai_ok = {r["id"]: r for r in jl(LOG["tai"](lang)) if "loi" not in r}
        cnt: dict = {}
        for s in jl(LOG["doan"](lang)):
            if s["ly_do_loai"] or s["video"] not in meta:
                continue
            cnt[s["video"]] = cnt.get(s["video"], 0) + 1
            rec = ban_ghi(s, meta[s["video"]], tai_ok[s["video"]], lang, cnt[s["video"]])
            x, _ = sf.read(os.path.join(T.ROOT, tai_ok[s["video"]]["wav16"]), dtype="int16", start=int(s["start"] * TC.SR), stop=int(s["end"] * TC.SR))
            path = os.path.join(DST, lang, rec["id"] + ".wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, x, TC.SR, subtype="PCM_16")
            man.append(rec)
    json.dump(man, open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump([dap_an(r) for r in man], open(os.path.join(DST, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    the(man)
    print(f"{len(man)} đoạn → {os.path.relpath(DST, T.ROOT)}")


def the(man: list[dict]) -> None:
    cell = collections.defaultdict(lambda: [0, set()])
    for r in man:
        c = cell[(r["meta"]["lang"], r["meta"]["nhom_do_dai"])]
        c[0] += 1
        c[1].add(r["meta"]["person"])
    L = [f"# Thẻ bộ `{BO}`", "",
         "- **Là gì:** ghi âm tự nhiên cào từ YouTube, nhãn **«01 có khả năng»** (PHU_DU_LIEU mục 2.2 ý 3); bộ test mới cho hàng ồn / nền (NHAT_KY 2.16).",
         f"- **Ngày dựng:** {datetime.date.today().isoformat()} · **Ai làm:** Huấn (P1); Gemini Neon **đề xuất** ứng viên (gemini/UNG_VIEN_CAO_2411.jsonl), "
         "Claude Code (Claude Opus 5.5) kiểm metadata thật + tải + dựng.",
         "- **Cách làm:** `experts/P1/scrape/kiem_ung_vien_gemini.py` (metadata thật: giấy phép CC BY, kênh không chuyên = không tích xanh và "
         "≤ 100 000 người theo dõi, bỏ phụ lục A, bỏ tập khoá v2, bỏ tôn giáo / trẻ em / trường phổ thông / podcast / sách nói / lồng tiếng / truyền hình, "
         "3–40 phút, kênh chưa dùng, ≤ 3 video / kênh; soát tiêu đề: Huấn bỏ 2 video nhóm A) → `tai_v2.py` (yt-dlp bestaudio giữ tệp gốc, "
         "ffmpeg → wav 16 kHz mono; 1 đoạn dài + 1 vừa + 2 ngắn / video, bỏ 30 s đầu / cuối; cổng đoạn: số 0 ≤ 0,1 s, ngôn ngữ đích ≥ 0,5). "
         "**Không lọc theo điều kiện K1.** Ghi chú Gemini (môi trường, nhạc nền) chỉ lưu trong `meta.gemini_ghi_chu`, không phải nhãn; "
         "video Gemini nghe có nhạc nền rõ → `khong_chac` (luật v1).",
         "- **Người nói:** `person = yt:<channel_id>` (một kênh một fold).",
         "- **Hạn chế:** 10 ứng viên Gemini không xem được (có thể mã bịa / video gỡ) đã bỏ; từ khoá nội dung có thể loại nhầm; chưa nghe tay.",
         "- **Đã kiểm:** *(điền sau)* `kiem_bo --moi` · `kiem_bo --tat-ca` · `lockbox check`", "",
         "| ngôn ngữ | nhóm độ dài | số đoạn | số kênh |", "|---|---|---|---|"]
    L += [f"| {lg} | {g} | {c[0]} | {len(c[1])} |" for (lg, g), c in sorted(cell.items())]
    open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def selftest() -> None:
    items = chon()
    assert len(items) == SO_VIDEO, len(items)
    assert not {"vJaaH9aV1Hc", "zwUsFN__jtE"} & {m["id"] for m in items}
    assert max(collections.Counter(m["channel_id"] for m in items).values()) <= 3
    m = items[0]
    r = ban_ghi({"video": m["id"], "nhom_do_dai": "3-16s", "start": 40.0, "end": 50.0}, m, {"acodec": "opus", "abr": 130.2}, m["lang"], 1)
    assert r["id"].startswith(f"{PRE}_") and r["wav"].startswith(f"../v6/{BO}/")
    print("tai_v2 selftest dat ·", collections.Counter(x["lang"] for x in items))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"tai": tai, "cat": cat, "dung": dung, "selftest": selftest}[sys.argv[1]]()
