"""Chấm lượt Gemini C4 theo luật chốt trước (NHAT_KY 2.16 «Lượt Gemini C4»). Nguồn nhãn: Gemini nghe sơ bộ.
Đọc gemini/KQ_NGHE_C4V2_2411.jsonl + bảng đối chiếu trên D:. Ra: p1_tunhien_v1/_kiem/CHAM_NGHE_C4_2411.md

  .venv/Scripts/python.exe experts/P1/build_cond/cham_nghe_c4.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6, pct  # noqa: E402

KQ = r"F:\CODING\Voiceaudit_ai002\gemini\KQ_NGHE_C4V2_2411.jsonl"
MAP = os.path.join(V6, NAT, "_kiem", "gemini_nghe_c4v2_2411_map.json")
CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_NGHE_C4_2411.md")
MUC = {"on_nen": ("khong", "nhe", "ro", "che_loi"), "vang": ("khong", "vua", "nang"), "co_loi": ("gan_het", "khoang_nua", "it")}
ON = {"rong": ("ro", "che_loi"), "chat": ("che_loi",)}          # nhãn coi là «ồn thật»
DC_OK = {"goc": ("on_nen", ("khong", "nhe")), "snr.20dB": ("on_nen", ("khong", "nhe")), "snr.0dB": ("on_nen", ("ro", "che_loi")),
         "nen.30pct": ("on_nen", ("ro", "che_loi")), "vang.rt60_1.5s": ("vang", ("vua", "nang"))}


def doc_kq(path: str, n: int) -> dict:
    """ten → nhãn; kiểm đủ stt, giá trị hợp lệ, dòng tổng khớp."""
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if "ghi_chu" not in rows[0] or "phiên mới" not in rows[0]["ghi_chu"]:
        raise SystemExit("Dòng đầu thiếu ghi_chu phiên mới")
    kq = {os.path.basename(r["tep"]): r for r in rows if "stt" in r}
    if sorted(r["stt"] for r in kq.values()) != list(range(1, n + 1)):
        raise SystemExit(f"stt không đủ 1..{n}")
    for r in kq.values():
        for k, v in MUC.items():
            if r.get(k) not in v:
                raise SystemExit(f"{r['tep']}: {k}={r.get(k)!r} không hợp lệ")
    tong = [r["tong"] for r in rows if "tong" in r]
    for k in MUC:
        dem = Counter(r[k] for r in kq.values())
        if tong and any(tong[0][k].get(m, 0) != dem[m] for m in MUC[k]):
            print(f"Lưu ý: dòng tổng {k} của Gemini {tong[0][k]} ≠ đếm lại {dict(dem)} (dùng số đếm lại)")
    return kq


def dem_chan(rows: list, kq: dict) -> dict:
    """rows: tệp thật (c4 = 'chan' | 'qua') → (chặn oan, thả sai) theo hai cách đếm."""
    out = {}
    for cach, on in ON.items():
        ch = [r for r in rows if r["c4"] == "chan"]
        qu = [r for r in rows if r["c4"] == "qua"]
        out[cach] = ((sum(kq[r["ten"]]["on_nen"] not in on for r in ch), len(ch)),
                     (sum(kq[r["ten"]]["on_nen"] in on for r in qu), len(qu)))
    return out


def run() -> None:
    mp = json.load(open(MAP, encoding="utf-8"))["tep"]
    kq = doc_kq(KQ, len(mp))
    that = [r for r in mp if r["loai"] == "that"]
    dc = [r for r in mp if r["loai"] == "doi_chung"]
    L = ["# Lượt Gemini C4: chấm theo luật chốt trước (NHAT_KY 2.16)", "",
         "Nguồn nhãn: **Gemini nghe sơ bộ** (phiên mới, mù). Không phải người nghe tay; chỉ dùng để chọn hướng sửa cách đo.", "",
         "## 1. Đối chứng", ""]
    sai = 0
    for r in sorted(dc, key=lambda r: (r["nhom"], r["lang"])):
        k, ok = DC_OK[r["nhom"]]
        g = kq[r["ten"]]
        sai += g[k] not in ok
        L.append(f"- {r['nhom']} ({r['lang']}, {r['ten']}): {k} = `{g[k]}` (cần {'/'.join(ok)}) → {'khớp' if g[k] in ok else '**SAI**'}"
                 + (f"; on_nen `{g['on_nen']}`" if k == "vang" else ""))
    dung_duoc = sai < 3
    L += ["", f"Sai {sai}/10 → {'dùng được' if dung_duoc else '**KHÔNG dùng để kết luận** (≥ 3/10 sai)'}.", "",
          "## 2. Chặn oan / thả sai (tệp thật)", "",
          "| phần | chặn oan rộng (khong/nhe) | chặn oan chặt (khong/nhe/ro) | thả sai rộng (ro/che_loi) | thả sai chặt (che_loi) |",
          "|---|---|---|---|---|"]
    phan = [("tất cả", that), ("vi", [r for r in that if r["lang"] == "vi"]), ("en", [r for r in that if r["lang"] == "en"])]
    phan += [(f"nhóm {n}", [r for r in that if r["nhom"] == n]) for n in sorted({r["nhom"] for r in that if r["c4"] == "chan"})]
    for ten, rs in phan:
        d = dem_chan(rs, kq)
        tha = (lambda x: pct(*x) if x[1] else "–")
        L.append(f"| {ten} | {pct(*d['rong'][0])} | {pct(*d['chat'][0])} | {tha(d['rong'][1])} | {tha(d['chat'][1])} |")
    L += ["", "## 3. Phân bố nhãn Gemini trên tệp C4 chặn, theo nhóm", "",
          "| nhóm | n | on_nen | vang | co_loi | loai_nen hay gặp | Gemini nghe ngôn ngữ khác nhãn |", "|---|---|---|---|---|---|---|"]
    for n in sorted({r["nhom"] for r in that}):
        for c4 in ("chan", "qua"):
            rs = [r for r in that if r["nhom"] == n and r["c4"] == c4]
            if not rs:
                continue
            g = [kq[r["ten"]] for r in rs]
            f = lambda k: " ".join(f"{m} {v}" for m, v in Counter(x[k] for x in g).most_common())  # noqa: E731
            ln = Counter(t for x in g for t in x["loai_nen"]).most_common(3)
            lech = sum(kq[r["ten"]]["ngon_ngu"] != r["lang"] for r in rs)
            L.append(f"| {n} ({'C4 chặn' if c4 == 'chan' else 'C4 không chặn'}) | {len(rs)} | {f('on_nen')} | {f('vang')} | {f('co_loi')} | "
                     f"{', '.join(f'{a} {b}' for a, b in ln)} | {lech} |")
    L += ["", f"- Gemini `tu_tin: thap`: {sum(x['tu_tin'] == 'thap' for x in kq.values())}/{len(kq)}.",
          "- Tệp C4 chặn mà Gemini nghe `khong`: " + ", ".join(
              f"{r['ten']} ({r['lang']}, {r['nhom']})" for r in that if r["c4"] == "chan" and kq[r["ten"]]["on_nen"] == "khong"),
          "- Mỗi video 1 tệp; fold dev 0–2. Chưa mở test. Chưa kiểm chéo lời trích dẫn bằng whisper."]
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    kq = {"a": {"on_nen": "khong"}, "b": {"on_nen": "ro"}, "c": {"on_nen": "che_loi"}, "d": {"on_nen": "nhe"}}
    rows = [{"ten": "a", "c4": "chan"}, {"ten": "b", "c4": "chan"}, {"ten": "c", "c4": "qua"}, {"ten": "d", "c4": "qua"}]
    d = dem_chan(rows, kq)
    assert d["rong"] == ((1, 2), (1, 2)) and d["chat"] == ((2, 2), (1, 2)), d
    print("cham_nghe_c4 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
