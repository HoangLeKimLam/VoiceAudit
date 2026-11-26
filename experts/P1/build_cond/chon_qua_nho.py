"""«Quá nhỏ» (Huấn chốt 24/11): chỉ tính theo công thức K1 — mức tiếng nói (khung đủ to ∩ ASR ra chữ) < −40 dBFS.
Claude đo độc lập + Gemini nghe mù để đánh giá mức đó có thật sự nhỏ không.

Chọn: mọi tệp p1_tunhien_v1 bị K1 tính < −40 dBFS + đối chứng cùng bộ (5 tệp sát ngưỡng −36…−40, 5 tệp thường −20…−26),
mỗi kênh tối đa 1 đối chứng; xáo, tên trung tính h001…, SAO CHÉP sang gemini/nghe_qn_2411/. Bảng đối chiếu chỉ trên D:.
Kèm số đo độc lập không dùng ASR: RMS cả tệp, đỉnh, mức trung bình của 30 % khung 30 ms to nhất.

Chạy: .venv/Scripts/python.exe experts/P1/build_cond/chon_qua_nho.py
"""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6, jl  # noqa: E402

SEED, NGUONG = 2412, -40.0
GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
OUT, DS = os.path.join(GEM, "nghe_qn_2411"), os.path.join(GEM, "DS_NGHE_QN_2411.csv")
MAP = os.path.join(V6, NAT, "_kiem", "gemini_qua_nho_2411_map.json")


def do_doc_lap(path: str) -> dict:
    y, _ = sf.read(path, dtype="float64")
    y = y.mean(1) if y.ndim > 1 else y
    db = lambda p: float(10 * np.log10(max(p, 1e-20)))  # noqa: E731
    n = 480
    fr = np.array([np.mean(y[i:i + n] ** 2) for i in range(0, len(y) - n + 1, n)])
    top = np.sort(fr)[-max(1, int(0.3 * len(fr))):]
    return {"rms_ca_tep_dbfs": round(db(np.mean(y ** 2)), 1), "dinh_dbfs": round(db(np.max(y ** 2)), 1),
            "muc_30pct_khung_to_dbfs": round(db(np.mean(top)), 1)}


def k1_level() -> dict:
    cases = json.load(open(os.path.join(V6, NAT, "cases.json"), encoding="utf-8"))
    cases = {c["file"]: c for c in (cases if isinstance(cases, list) else cases["cases"])}
    out = {}
    for r in jl(os.path.join(V6, NAT, "_kiem", "k1_ket_qua.jsonl")):
        lv = [k.get("am_luong_dbfs") for k in r["measures"].get("kenh", [])]
        if r["file"] in cases and cases[r["file"]]["mong_doi"] == "accept" and lv and lv[0] is not None:
            out[r["file"]] = (lv[0], cases[r["file"]]["base"])
    return out


def run() -> None:
    for p in (OUT, DS, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rng = random.Random(SEED)
    lv = k1_level()
    qn = sorted(f for f, (v, _) in lv.items() if v < NGUONG)
    dung = {lv[f][1] for f in qn}
    rows = [{"nguon": f, "nhom": "qua_nho", "k1_dbfs": lv[f][0]} for f in qn]
    for nhom, lo, hi in (("sat_nguong", -40.0, -36.0), ("thuong", -26.0, -20.0)):
        g = sorted(f for f, (v, b) in lv.items() if lo <= v <= hi)
        rng.shuffle(g)
        k = 0
        for f in g:
            if k == 5:
                break
            if lv[f][1] not in dung:
                dung.add(lv[f][1])
                rows.append({"nguon": f, "nhom": nhom, "k1_dbfs": lv[f][0]})
                k += 1
        assert k == 5, f"thiếu đối chứng {nhom}"
    rng.shuffle(rows)
    os.makedirs(OUT)
    for i, r in enumerate(rows, 1):
        r["ten"] = f"h{i:03d}.wav"
        src = os.path.join(V6, NAT, r["nguon"])
        shutil.copyfile(src, os.path.join(OUT, r["ten"]))
        r.update(do_doc_lap(src))
    with open(DS, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stt", "tep"])
        w.writerows([i, f"gemini/nghe_qn_2411/{r['ten']}"] for i, r in enumerate(rows, 1))
    json.dump({"seed": SEED, "nguong_k1_dbfs": NGUONG, "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    assert len(os.listdir(OUT)) == len(rows) == len(qn) + 10
    print(f"xong: {len(rows)} tệp ({len(qn)} quá nhỏ + 10 đối chứng) → {OUT}")
    for r in sorted(rows, key=lambda r: r["k1_dbfs"]):
        print(f"  {r['nhom']:10s} K1 {r['k1_dbfs']:6.1f} · RMS {r['rms_ca_tep_dbfs']:6.1f} · 30% khung to {r['muc_30pct_khung_to_dbfs']:6.1f} · "
              f"đỉnh {r['dinh_dbfs']:6.1f} · {r['nguon']}")


KQ = os.path.join(GEM, "KQ_NGHE_QN_2411.jsonl")
WSP = os.path.join(V6, NAT, "_kiem", "gemini_qua_nho_2411_whisper.jsonl")
CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_NGHE_QUA_NHO_2411.md")


def rank(v: list) -> np.ndarray:
    return np.argsort(np.argsort(v)).astype(float)


def cham() -> None:
    """Luật chốt trước (NHAT_KY 2.16): tin nếu ≥ 4/5 'thuong' → binh_thuong/to VÀ Spearman(xếp hạng Gemini, xếp hạng K1) ≥ 0,6."""
    from cham_dieu_kien import pct
    from cham_nghe_gemini import giong_chu
    kq = {r["tep"].split("/")[-1]: r for r in jl(KQ) if "stt" in r}
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    assert len(kq) == len(rows) and all(r["ten"] in kq for r in rows)
    for r in rows:
        r["g"] = kq[r["ten"]]
    rho = float(np.corrcoef(rank([r["g"]["xep_hang"] for r in rows]), rank([-r["k1_dbfs"] for r in rows]))[0, 1])
    rho_doc = float(np.corrcoef(rank([r["g"]["xep_hang"] for r in rows]), rank([-r["rms_ca_tep_dbfs"] for r in rows]))[0, 1])
    th = [r for r in rows if r["nhom"] == "thuong"]
    ok_th = sum(r["g"]["am_luong"] in ("binh_thuong", "to") for r in th)
    tin = ok_th >= 4 and rho >= 0.6
    L = ["# Gemini nghe mù âm lượng: 10 tệp «quá nhỏ» theo K1 + 10 đối chứng (luật chốt trước, NHAT_KY 2.16)", "",
         "Nguồn nhãn: **Gemini nghe sơ bộ**. «Quá nhỏ» vẫn tính theo công thức K1 (< −40 dBFS); kết quả nghe không đổi cách đếm.",
         f"Gemini ghi `tu_chuan_hoa_am_luong`: {jl(KQ)[0].get('tu_chuan_hoa_am_luong')}.", "",
         f"## 1. Độ tin cậy: tệp thường nghe binh_thuong/to {ok_th}/5 · Spearman với K1 = {rho:.2f} "
         f"(với RMS độc lập = {rho_doc:.2f}) → **{'ĐẠT' if tin else 'KHÔNG ĐẠT'}**", "",
         "| nhóm | n | qua_nho (chặt) | qua_nho/nho (rộng) | binh_thuong/to |", "|---|---|---|---|---|"]
    for nh in ("qua_nho", "sat_nguong", "thuong"):
        g = [r["g"]["am_luong"] for r in rows if r["nhom"] == nh]
        L.append(f"| {nh} | {len(g)} | {pct(g.count('qua_nho'), len(g))} | {pct(sum(v in ('qua_nho', 'nho') for v in g), len(g))} | "
                 f"{pct(sum(v in ('binh_thuong', 'to') for v in g), len(g))} |")
    if not os.path.exists(WSP):
        from transformers import pipeline
        asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30, device="cpu")
        with open(WSP, "w", encoding="utf-8") as fh:
            for r in rows:
                y, sr = sf.read(os.path.join(OUT, r["ten"]), dtype="float32")
                t = asr({"raw": y.mean(1) if y.ndim > 1 else y, "sampling_rate": sr}, return_timestamps=True)["text"]
                fh.write(json.dumps({"ten": r["ten"], "text": t.strip()}, ensure_ascii=False) + "\n")
    w = {x["ten"]: x["text"] for x in jl(WSP)}
    sc = [(giong_chu(r["g"]["bang_chung"], w[r["ten"]]), r) for r in rows]
    L += ["", "## 2. Từng tệp (sắp theo mức K1)", "",
          "| tệp | nhóm | K1 dBFS | RMS dBFS | Gemini am_luong | xếp hạng Gemini | giong_ro | khớp lời whisper |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['ten']} | {r['nhom']} | {r['k1_dbfs']} | {r['rms_ca_tep_dbfs']} | {r['g']['am_luong']} | {r['g']['xep_hang']} | "
          f"{r['g']['giong_ro']} | {s:.0%} |" for s, r in sorted(sc, key=lambda x: x[1]["k1_dbfs"])]
    L += ["", f"Kiểm chéo lời trích bằng whisper-small: khớp ≥ 50 % số từ {pct(sum(s >= 0.5 for s, _ in sc), len(sc))}."]
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": run, "cham": cham}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
