"""Bước 3 (Huấn chốt 24/11): chọn 72 tệp cho Gemini nghe mù ồn / nền / vang.

60 tệp thật p1_tunhien_v1 (accept) phân tầng theo SNR ước của bộ đo bản 2:
  A < 5 dB (chặn rõ) · B 5–10 dB (sát ngưỡng) · C > 15 dB (qua); mỗi tầng 10 vi + 10 en; mỗi kênh tối đa 1 tệp.
+ 12 ca đối chứng p1_dieukien_v2 (nhãn từ cách dựng). Xáo, đổi tên trung tính g001…, SAO CHÉP (không di chuyển).
Bảng đối chiếu chỉ để trên D: (p1_tunhien_v1/_kiem/gemini_nghe_2411_map.json), không để trong gemini/.

Chạy: .venv/Scripts/python.exe experts/P1/build_cond/chon_nghe_gemini.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, SYN, V6, estimate, fold, load  # noqa: E402

SEED = 2411
GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
OUT = os.path.join(GEM, "nghe_2411")
DS = os.path.join(GEM, "DS_NGHE_DK_2411.csv")
MAP = os.path.join(V6, NAT, "_kiem", "gemini_nghe_2411_map.json")
TANG = {"A_duoi5dB": lambda s: s < 5, "B_5_10dB": lambda s: 5 <= s < 10, "C_tren15dB": lambda s: s > 15}
DOI_CHUNG = [("goc", "vi"), ("goc", "en"), ("snr.0dB", "vi"), ("snr.0dB", "en"), ("snr.20dB", "vi"), ("snr.20dB", "en"),
             ("nen.2pct", "vi"), ("nen.2pct", "en"), ("nen.30pct", "vi"), ("nen.30pct", "en"),
             ("vang.rt60_0.3s", "vi"), ("vang.rt60_1.5s", "en")]


def chon_that(rng: random.Random) -> list:
    cases, meas = load(NAT)
    cal = json.load(open(os.path.join(V6, SYN, "_kiem", "hieu_chinh.json"), encoding="utf-8"))
    fs = sorted(f for f, c in cases.items() if c["mong_doi"] == "accept" and f in meas)
    est = {f: estimate(meas[f], cal) for f in fs}
    dung, out = set(), []
    for tang, ok in TANG.items():
        for lang in ("vi", "en"):
            g = [f for f in fs if f.startswith(lang + "/") and est[f]["snr"] is not None and ok(est[f]["snr"])]
            rng.shuffle(g)
            k = 0
            for f in g:
                if k == 10:
                    break
                if cases[f]["base"] in dung:
                    continue
                dung.add(cases[f]["base"])
                out.append({"nguon": f"{NAT}/{f}", "loai": "that", "tang": tang, "lang": lang, "fold": fold(cases[f]["base"]),
                            "uoc": {a: (round(v, 2) if v is not None else None) for a, v in est[f].items()}})
                k += 1
            if k < 10:
                raise SystemExit(f"Thiếu tệp: tầng {tang} {lang} chỉ có {k} kênh khác nhau")
    return out


def chon_doi_chung(rng: random.Random) -> list:
    P = [json.loads(l) for l in open(os.path.join(V6, SYN, "provenance.jsonl"), encoding="utf-8")]
    out = []
    for ca, lang in DOI_CHUNG:
        g = sorted(p["file"].split(f"{SYN}/", 1)[-1] for p in P if p["ca"] == ca and f"_{lang}_" in p["id"])
        if not g:
            raise SystemExit(f"Không có ca đối chứng {ca} {lang}")
        out.append({"nguon": f"{SYN}/{rng.choice(g)}", "loai": "doi_chung", "tang": ca, "lang": lang})
    return out


def run() -> None:
    for p in (OUT, DS, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rng = random.Random(SEED)
    rows = chon_that(rng) + chon_doi_chung(rng)
    rng.shuffle(rows)
    os.makedirs(OUT)
    for i, r in enumerate(rows, 1):
        r["ten"] = f"g{i:03d}.wav"
        shutil.copyfile(os.path.join(V6, r["nguon"]), os.path.join(OUT, r["ten"]))
    with open(DS, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stt", "tep"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, f"gemini/nghe_2411/{r['ten']}"])
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/chon_nghe_gemini.py", "tep": rows},
              open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # kiểm đầu ra
    assert len(rows) == 72 and len(os.listdir(OUT)) == 72
    that = [r for r in rows if r["loai"] == "that"]
    assert len({r["nguon"] for r in rows}) == 72 and len(that) == 60
    for r in rows:
        assert os.path.getsize(os.path.join(OUT, r["ten"])) == os.path.getsize(os.path.join(V6, r["nguon"]))
    print(f"xong: {OUT} (72 tệp), {DS}, {MAP}")
    print("tầng thật:", {t: sum(r["tang"] == t for r in that) for t in TANG}, "· fold test:", sum(r["fold"] >= 3 for r in that))


if __name__ == "__main__":
    run()
