"""Chọn tệp cho Gemini SOÁT NỘI DUNG bộ p1_tunhien_v2 (Huấn duyệt 26/11): có phải lời nói, loại âm thanh, ngôn ngữ, số người nói, nhạc,
trẻ em / tôn giáo. KHÔNG hỏi mức ồn (bài học 24/11: nhãn mức ồn của Gemini không tin được).

Mỗi video 1 đoạn (ưu tiên 16–64 s → 3–16 s → 1–10 phút; cùng nhóm thì lấy đoạn đầu theo id) = 150 tệp
+ 12 ca đối chứng nhãn từ cách tạo, cắt 25 s giữa tệp nguyên liệu: 3 nhạc MUSAN (1 có hát), 3 sách nói MUSAN tiếng Đức / Pháp / Tây Ban Nha,
  2 tiếng ồn MUSAN, 2 FLEURS vi + 2 FLEURS en (nguyên tệp). Xáo (seed 2611), đổi tên trung tính v001…, SAO CHÉP (không di chuyển).
Bảng đối chiếu để trên D: (p1_tunhien_v2/_kiem/gemini_soat_v2_2611_map.json), không để trong gemini/.

  .venv/Scripts/python.exe experts/P1/scrape/chon_soat_gemini_v2.py run | selftest
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import shutil
import sys

import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BO = os.path.join(ROOT, "data_labels", "v6", "p1_tunhien_v2")
RAW = os.path.join(ROOT, "data", "p1_raw")
MUSAN = os.path.join(RAW, "musan", "musan")
GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
OUT, DS = os.path.join(GEM, "nghe_v2_2611"), os.path.join(GEM, "DS_SOAT_V2_2611.csv")
MAP = os.path.join(BO, "_kiem", "gemini_soat_v2_2611_map.json")
SEED, SR, CAT_S = 2611, 16000, 25
UU_TIEN = {"16-64s": 0, "3-16s": 1, "1-10m": 2}


def chon_that() -> list[dict]:
    man = json.load(open(os.path.join(BO, "manifest.json"), encoding="utf-8"))
    best: dict = {}
    for r in sorted(man, key=lambda r: r["id"]):
        k = (UU_TIEN[r["meta"]["nhom_do_dai"]], r["id"])
        v = r["meta"]["recording"]
        if v not in best or k < best[v][0]:
            best[v] = (k, r)
    return [{"loai": "that", "nguon": os.path.join(BO, r["meta"]["lang"], r["id"] + ".wav"), "id": r["id"], "video": v,
             "lang_khai": r["meta"]["lang"], "nhom": r["meta"]["nhom_do_dai"]} for v, (_, r) in sorted(best.items())]


def annot(path: str) -> list[list[str]]:
    return [l.split() for l in open(path, encoding="utf-8") if l.strip()]


def chon_doi_chung(rng: random.Random) -> list[dict]:
    out = []
    nhac = [a for d in sorted(glob.glob(os.path.join(MUSAN, "music", "*", "ANNOTATIONS"))) for a in annot(d)]
    hat, khong_hat = sorted(a[0] for a in nhac if a[2] == "Y"), sorted(a[0] for a in nhac if a[2] == "N")
    for ten, mong in ((rng.choice(hat), "nhac_co_hat"), *((x, "nhac_khong_loi") for x in rng.sample(khong_hat, 2))):
        out.append({"loai": "doi_chung", "nguon": glob.glob(os.path.join(MUSAN, "music", "*", ten + ".wav"))[0], "mong_doi": mong, "cat": True})
    sach = annot(os.path.join(MUSAN, "speech", "librivox", "ANNOTATIONS"))
    for ng in ("german", "french", "spanish"):
        ten = rng.choice(sorted(a[0] for a in sach if a[2] == ng))
        out.append({"loai": "doi_chung", "nguon": os.path.join(MUSAN, "speech", "librivox", ten + ".wav"), "mong_doi": f"sach_noi_{ng}", "cat": True})
    for p in rng.sample(sorted(glob.glob(os.path.join(MUSAN, "noise", "free-sound", "*.wav"))), 2):
        out.append({"loai": "doi_chung", "nguon": p, "mong_doi": "chi_tieng_on", "cat": True})
    for lg, thu_muc in (("vi", "vi_vn"), ("en", "en_us")):
        ds = sorted(glob.glob(os.path.join(RAW, "fleurs", thu_muc, "*", "*.wav")))
        ds = [p for p in ds if 8 <= sf.info(p).duration <= 30][:400]
        for p in rng.sample(ds, 2):
            out.append({"loai": "doi_chung", "nguon": p, "mong_doi": f"loi_doc_{lg}", "cat": False})
    return out


def ghi(r: dict, dst: str) -> None:
    """Tệp thật / FLEURS: chép nguyên. Đối chứng dài: cắt CAT_S giây giữa tệp, mono 16 kHz PCM16."""
    if not r.get("cat"):
        shutil.copyfile(r["nguon"], dst)
        return
    info = sf.info(r["nguon"])
    assert info.samplerate == SR, (r["nguon"], info.samplerate)
    n = int(CAT_S * SR)
    a = max(0, (info.frames - n) // 2)
    x, _ = sf.read(r["nguon"], start=a, stop=min(info.frames, a + n), dtype="int16")
    sf.write(dst, x if x.ndim == 1 else x[:, 0], SR, subtype="PCM_16")


def run() -> None:
    for p in (OUT, DS, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rng = random.Random(SEED)
    rows = chon_that() + chon_doi_chung(rng)
    # kiểm danh sách TRƯỚC khi chép (bài học 24/11)
    that = [r for r in rows if r["loai"] == "that"]
    assert len(that) == 150 and len({r["video"] for r in that}) == 150, len(that)
    assert len(rows) == 162 and len({r["nguon"] for r in rows}) == 162 and all(os.path.exists(r["nguon"]) for r in rows)
    rng.shuffle(rows)
    os.makedirs(OUT)
    for i, r in enumerate(rows, 1):
        r["ten"] = f"v{i:03d}.wav"
        ghi(r, os.path.join(OUT, r["ten"]))
        r["giay"] = round(sf.info(os.path.join(OUT, r["ten"])).duration, 1)
    with open(DS, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stt", "tep", "giay"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, f"gemini/nghe_v2_2611/{r['ten']}", r["giay"]])
    for r in rows:
        r["nguon"] = os.path.relpath(r["nguon"], ROOT).replace("\\", "/")
    json.dump({"seed": SEED, "script": "experts/P1/scrape/chon_soat_gemini_v2.py", "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    assert len(os.listdir(OUT)) == 162
    print(f"xong: {OUT} (162 tệp) · {DS} · {MAP}")
    print("nhóm độ dài tệp thật:", {g: sum(r.get("nhom") == g for r in rows) for g in UU_TIEN}, "· tổng phút:", round(sum(r["giay"] for r in rows) / 60))


def selftest() -> None:
    that = chon_that()
    assert len(that) == 150 and len({r["video"] for r in that}) == 150 and all(os.path.exists(r["nguon"]) for r in that)
    dc = chon_doi_chung(random.Random(SEED))
    assert len(dc) == 12 and all(os.path.exists(r["nguon"]) for r in dc), [r["nguon"] for r in dc if not os.path.exists(r["nguon"])]
    print("chon_soat_gemini_v2 selftest dat ·", [r["mong_doi"] for r in dc])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1]]()
