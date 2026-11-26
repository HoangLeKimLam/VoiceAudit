"""Lượt Gemini nghe mù tệp C4 chặn (Huấn duyệt 24/11). Luật chọn chốt trước (NHAT_KY 2.16 «Lượt Gemini C4»):
  46 tệp C4 chặn: accept p1_tunhien_v1, fold dev 0–2, chưa có trong lượt nghe 2411, mỗi video 1 tệp (tất cả video có tệp bị chặn)
  24 tệp C4 không chặn: cùng điều kiện, mỗi video 1 tệp, 12 vi + 12 en
  10 đối chứng p1_dieukien_v2 fold dev: goc · snr.0dB · snr.20dB · nen.30pct · vang.rt60_1.5s, mỗi loại 1 vi + 1 en
Xáo, đổi tên c001…, SAO CHÉP (không di chuyển) vào gemini/nghe_c4v2_2411/. Bảng đối chiếu chỉ để trên D:.

  .venv/Scripts/python.exe experts/P1/build_cond/chon_nghe_c4.py run | selftest
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, SYN, V6, fold, jl  # noqa: E402
from cham_tram import cases_of, dap_an_syn  # noqa: E402
from cham_vang import gop, lam  # noqa: E402

SEED = 2415
GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
# Bản 2: lần 1 (nghe_c4_2411, DS_NGHE_C4_2411.csv, gemini_nghe_c4_2411_map.json) chọn trùng 4 video giữa nhóm chặn và không chặn,
# bị phép kiểm cuối bắt; Huấn duyệt chuyển vào _quarantine 24/11 (gemini/_quarantine, p1_tunhien_v1/_quarantine/_kiem).
OUT = os.path.join(GEM, "nghe_c4v2_2411")
DS = os.path.join(GEM, "DS_NGHE_C4V2_2411.csv")
MAP = os.path.join(V6, NAT, "_kiem", "gemini_nghe_c4v2_2411_map.json")
DA_NGHE = os.path.join(V6, NAT, "_kiem", "gemini_nghe_2411_map.json")
SO_CHAN, SO_QUA = 46, {"vi": 12, "en": 12}
DOI_CHUNG = ("goc", "snr.0dB", "snr.20dB", "nen.30pct", "vang.rt60_1.5s")


def vid(f: str) -> str:
    return re.match(r".*/p1n_(?:vi|en)_(.+)_\d+\.wav$", f).group(1)


def nhom(m: dict) -> str:
    """Lý do nổi bật theo số đo (thứ tự ưu tiên; mỗi tệp một nhóm)."""
    if m["bh"] is not None and m["bh"] < 5:
        return "1_bh_duoi5"
    if m["loi_bh"] < 0.5:
        return "2_it_loi"
    if m["c50"] < 10:
        return "3_vang"
    if m["bh"] is not None and m["bh"] < 10:
        return "4_bh_5_10"
    return "5_con_lai"


def moi_video(rng: random.Random, files: list) -> list:
    """Mỗi video 1 tệp ngẫu nhiên; trả danh sách theo thứ tự video đã sắp."""
    g = {}
    for f in sorted(files):
        g.setdefault(vid(f), []).append(f)
    return [rng.choice(g[v]) for v in sorted(g)]


def chon_that(rng: random.Random) -> list:
    cs, M = cases_of(SYN), gop(SYN)
    cn, MN = cases_of(NAT), gop(NAT)
    da, _ = dap_an_syn(cs)
    c4 = lam([f for f in da if f in M], M, da)["chan"]["C4"]
    heard = {r["nguon"].split("/", 1)[1] for r in json.load(open(DA_NGHE, encoding="utf-8"))["tep"]}
    pool = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and f not in heard and fold(c["base"]) <= 2]
    chan = [f for f in pool if c4(MN[f])]
    qua = [f for f in pool if not c4(MN[f])]
    out = []
    pick = moi_video(rng, chan)
    if len(pick) != SO_CHAN:
        raise SystemExit(f"Số video bị chặn ở dev là {len(pick)}, khác {SO_CHAN} đã chốt")
    out += [("chan", f) for f in pick]
    da_chon = {vid(f) for f in pick}                     # video có đoạn bị chặn thì không lấy thêm đoạn không chặn (lỗi lần 1)
    for lg, k in SO_QUA.items():
        g = moi_video(rng, [f for f in qua if f.startswith(lg + "/") and vid(f) not in da_chon])
        out += [("qua", f) for f in rng.sample(g, k)]
    return [{"nguon": f"{NAT}/{f}", "loai": "that", "c4": t, "lang": f[:2], "nhom": nhom(MN[f]), "video": vid(f),
             "fold": fold(cn[f]["base"])} for t, f in out]


def chon_doi_chung(rng: random.Random) -> list:
    cs = cases_of(SYN)
    used = {r["nguon"] for r in json.load(open(DA_NGHE, encoding="utf-8"))["tep"]}
    P = list(jl(os.path.join(V6, SYN, "provenance.jsonl")))
    out = []
    for ca in DOI_CHUNG:
        for lang in ("vi", "en"):
            g = sorted(f for f in (p["file"].split(f"{SYN}/", 1)[-1] for p in P if p["ca"] == ca and f"_{lang}_" in p["id"])
                       if fold(cs[f]["base"]) <= 2 and f"{SYN}/{f}" not in used)
            if not g:
                raise SystemExit(f"Không có ca đối chứng {ca} {lang} ở dev")
            out.append({"nguon": f"{SYN}/{rng.choice(g)}", "loai": "doi_chung", "c4": None, "lang": lang, "nhom": ca})
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
        r["ten"] = f"c{i:03d}.wav"
        shutil.copyfile(os.path.join(V6, r["nguon"]), os.path.join(OUT, r["ten"]))
    with open(DS, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stt", "tep"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, f"gemini/nghe_c4v2_2411/{r['ten']}"])
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/chon_nghe_c4.py", "tep": rows},
              open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # kiểm đầu ra
    n = SO_CHAN + sum(SO_QUA.values()) + 2 * len(DOI_CHUNG)
    that = [r for r in rows if r["loai"] == "that"]
    assert len(rows) == n == len(os.listdir(OUT)) == len({r["nguon"] for r in rows}), len(rows)
    assert all(r["fold"] <= 2 for r in that) and len({r["video"] for r in that}) == len(that)
    for r in rows:
        assert os.path.getsize(os.path.join(OUT, r["ten"])) == os.path.getsize(os.path.join(V6, r["nguon"]))
    dem = {}
    for r in that:
        k = (r["c4"], r["lang"], r["nhom"])
        dem[k] = dem.get(k, 0) + 1
    print(f"xong: {OUT} ({n} tệp), {DS}, {MAP}")
    for k in sorted(dem):
        print(" ", k, dem[k])


def selftest() -> None:
    assert vid("vi/p1n_vi_-6-CYw_gB5WE_12.wav") == "-6-CYw_gB5WE"
    base = {"bh": 20.0, "loi_bh": 0.9, "c50": 40.0}
    assert nhom(base) == "5_con_lai" and nhom({**base, "bh": 3.0, "loi_bh": 0.1}) == "1_bh_duoi5"
    assert nhom({**base, "loi_bh": 0.2, "c50": 2.0}) == "2_it_loi" and nhom({**base, "c50": 2.0, "bh": 7.0}) == "3_vang"
    assert nhom({**base, "bh": None}) == "5_con_lai"
    p = moi_video(random.Random(0), ["vi/p1n_vi_a_1.wav", "vi/p1n_vi_a_2.wav", "vi/p1n_vi_b_1.wav"])
    assert len(p) == 2 and {vid(f) for f in p} == {"a", "b"}
    print("chon_nghe_c4 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
