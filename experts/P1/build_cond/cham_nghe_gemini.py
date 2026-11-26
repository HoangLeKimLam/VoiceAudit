"""Chấm bước 3 theo luật chốt trước (NHAT_KY 2.16, 24/11) — Gemini nghe mù 72 tệp gemini/nghe_2411.

  run        bảng đối chứng · ồn theo tầng (chặt / rộng) · chặn oan ước · thả sai ước · vang · tu_tin thấp
  whisper    chép lời 72 tệp bằng whisper-small (chạy nền, ghi _kiem/gemini_nghe_2411_whisper.jsonl) để kiểm chéo bang_chung
"""
from __future__ import annotations

import difflib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6, jl, pct  # noqa: E402

GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
KQ = os.path.join(GEM, "KQ_NGHE_DK_2411.jsonl")
MAP = os.path.join(V6, NAT, "_kiem", "gemini_nghe_2411_map.json")
WSP = os.path.join(V6, NAT, "_kiem", "gemini_nghe_2411_whisper.jsonl")
OUT = os.path.join(V6, NAT, "_kiem", "CHAM_NGHE_GEMINI_2411.md")
KHOP = {"goc": ("khong", "nhe"), "snr.20dB": ("khong", "nhe"), "nen.2pct": ("khong", "nhe"),
        "snr.0dB": ("ro", "che_loi"), "nen.30pct": ("ro", "che_loi")}
ON = {"chat": ("che_loi",), "rong": ("ro", "che_loi")}


def khop_doi_chung(r: dict, g: dict) -> bool:
    if r["tang"].startswith("vang."):
        return g["vang"] == "nang" if r["tang"].endswith("1.5s") else g["vang"] in ("khong", "vua")
    return g["on_nen"] in KHOP[r["tang"]]


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w ]+", " ", s.lower()).split())


def giong_chu(bc: str, ts: str) -> float:
    """Tỉ lệ từ của bang_chung khớp liền mạch dài nhất trong lời whisper (0–1)."""
    a, b = norm(bc).split(), norm(ts).split()
    if not a or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(x.size for x in m.get_matching_blocks()) / len(a)


def load() -> list:
    kq = {r["tep"].split("/")[-1]: r for r in jl(KQ) if "stt" in r}
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    assert len(kq) == 72 and all(r["ten"] in kq for r in rows), "kết quả Gemini thiếu tệp"
    return [{**r, "g": kq[r["ten"]]} for r in rows]


def run() -> str:
    rows = load()
    dc = [r for r in rows if r["loai"] == "doi_chung"]
    th = [r for r in rows if r["loai"] == "that"]
    ok = [r for r in dc if khop_doi_chung(r, r["g"])]
    L = ["# Bước 3: Gemini nghe mù 72 tệp (chấm theo luật chốt trước, NHAT_KY 2.16)", "",
         "Nguồn nhãn: **Gemini nghe sơ bộ**, chưa phải người nghe tay. Wilson 95 % trong ngoặc.", "",
         f"## 1. Đối chứng: {len(ok)}/12 khớp → Gemini {'ĐẠT' if len(ok) >= 10 else 'KHÔNG ĐẠT'} ngưỡng tin cậy (≥ 10/12)", "",
         "| ca | lang | Gemini on_nen | Gemini vang | khớp |", "|---|---|---|---|---|"]
    L += [f"| {r['tang']} | {r['lang']} | {r['g']['on_nen']} | {r['g']['vang']} | {'✓' if r in ok else '✗'} |"
          for r in sorted(dc, key=lambda r: (r["tang"], r["lang"]))]
    L += ["", "## 2. Tệp thật: Gemini nói ồn, theo tầng SNR ước của bộ đo", "",
          "| tầng | lang | n | ồn (chặt: che_loi) | ồn (rộng: ro/che_loi) | vang nặng |", "|---|---|---|---|---|---|"]
    for t in ("A_duoi5dB", "B_5_10dB", "C_tren15dB"):
        for lang in ("vi", "en", "chung"):
            g = [r for r in th if r["tang"] == t and (lang == "chung" or r["lang"] == lang)]
            L.append(f"| {t} | {lang} | {len(g)} | {pct(sum(r['g']['on_nen'] in ON['chat'] for r in g), len(g))} | "
                     f"{pct(sum(r['g']['on_nen'] in ON['rong'] for r in g), len(g))} | {pct(sum(r['g']['vang'] == 'nang' for r in g), len(g))} |")
    L += ["", "## 3. Chặn oan ước (tầng A + B, bộ đo chặn, Gemini nói KHÔNG ồn) · thả sai ước (tầng C, bộ đo thả, Gemini nói ồn)", "",
          "| cách đếm | lang | chặn oan ước | thả sai ước |", "|---|---|---|---|"]
    for cach, on in ON.items():
        for lang in ("vi", "en", "chung"):
            ab = [r for r in th if r["tang"] != "C_tren15dB" and (lang == "chung" or r["lang"] == lang)]
            c = [r for r in th if r["tang"] == "C_tren15dB" and (lang == "chung" or r["lang"] == lang)]
            L.append(f"| {cach} | {lang} | {pct(sum(r['g']['on_nen'] not in on for r in ab), len(ab))} | "
                     f"{pct(sum(r['g']['on_nen'] in on for r in c), len(c))} |")
    L += ["", "Phân bố on_nen theo tầng (khong / nhe / ro / che_loi):"]
    for t in ("A_duoi5dB", "B_5_10dB", "C_tren15dB"):
        g = [r["g"]["on_nen"] for r in th if r["tang"] == t]
        L.append(f"- {t}: " + " / ".join(str(g.count(v)) for v in ("khong", "nhe", "ro", "che_loi")))
    thap = [r for r in rows if r["g"]["tu_tin"] == "thap"]
    L += ["", f"## 4. tu_tin thấp: {len(thap)} tệp" + ("" if not thap else ": " + ", ".join(f"{r['ten']} ({r['loai']}/{r['tang']})" for r in thap))]
    if os.path.exists(WSP):
        w = {r["ten"]: r for r in jl(WSP)}
        sc = sorted(((giong_chu(r["g"]["bang_chung"], w[r["ten"]]["text"]), r) for r in rows if r["ten"] in w), key=lambda x: x[0])
        L += ["", f"## 5. Kiểm chéo bang_chung bằng whisper-small ({len(sc)} tệp)", "",
              f"- Khớp ≥ 50 % số từ: {pct(sum(s >= 0.5 for s, _ in sc), len(sc))}",
              "- Khớp < 50 % (cần xem: Gemini mô tả nền thay vì trích lời, nghe mẫu khác cửa sổ, hoặc trích sai):", ""]
        L += [f"  - {r['ten']} ({r['loai']}, {r['lang']}, nghe_mau={r['g']['nghe_mau']}): {s:.0%} · Gemini «{r['g']['bang_chung']}» · "
              f"whisper «{w[r['ten']]['text'][:90]}…»" for s, r in sc if s < 0.5]
    else:
        L += ["", "## 5. Kiểm chéo whisper: chưa chạy (`cham_nghe_gemini.py whisper`)"]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return "\n".join(L)


def whisper() -> None:
    import soundfile as sf
    from transformers import pipeline
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", chunk_length_s=30, device="cpu")
    done = {r["ten"] for r in jl(WSP)} if os.path.exists(WSP) else set()
    for r in json.load(open(MAP, encoding="utf-8"))["tep"]:
        if r["ten"] in done:
            continue
        y, sr = sf.read(os.path.join(GEM, "nghe_2411", r["ten"]), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        txt = asr({"raw": y, "sampling_rate": sr}, generate_kwargs={"task": "transcribe"}, return_timestamps=True)["text"]
        with open(WSP, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ten": r["ten"], "text": txt.strip()}, ensure_ascii=False) + "\n")
        print(r["ten"], flush=True)


def selftest() -> None:
    assert giong_chu("Britain initiated a naval blockade", "so Britain initiated a naval blockade of Germany") == 1.0
    assert giong_chu("tiếng quạt và xe", "hello world") == 0.0 and giong_chu("", "x") == 0.0
    assert khop_doi_chung({"tang": "vang.rt60_1.5s"}, {"vang": "nang"}) and not khop_doi_chung({"tang": "goc"}, {"on_nen": "ro"})
    print("selftest ok")


if __name__ == "__main__":
    {"run": lambda: print(run()), "whisper": whisper, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
