"""Huấn nghe tay 40 tệp lấy từ 80 tệp lượt Gemini C4 (Huấn chốt 24/11). Luật chọn chốt trước (NHAT_KY 2.16 «Huấn nghe tay C4»), seed 2416:
  26 tệp C4 chặn, chia theo nhóm lý do: 1_bh_duoi5 8 · 2_it_loi 3 · 3_vang 3 · 4_bh_5_10 10 · 5_con_lai 2
  8 tệp C4 không chặn: 4 vi + 4 en
  6 đối chứng: snr.0dB vi + en · nen.30pct vi + en · goc 1 · snr.20dB 1
Dùng lại tệp có sẵn trong gemini/nghe_c4v2_2411 (không chép dữ liệu mới). Trang mù: không hiện nhóm, số đo hay kết quả Gemini.

  .venv/Scripts/python.exe experts/P1/build_cond/nghe_tay_c4.py run | selftest
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6  # noqa: E402
from nghe_tay import page  # noqa: E402

SEED = 2416
GEM = r"F:\CODING\Voiceaudit_ai002\gemini"
THU_MUC = "nghe_c4v2_2411"
SRC = os.path.join(V6, NAT, "_kiem", "gemini_nghe_c4v2_2411_map.json")
OUT = os.path.join(GEM, "NGHE_TAY_C4_2411.html")
MAP = os.path.join(V6, NAT, "_kiem", "NGHE_TAY_C4_2411_map.json")
SO_CHAN = {"1_bh_duoi5": 8, "2_it_loi": 3, "3_vang": 3, "4_bh_5_10": 10, "5_con_lai": 2}
SO_QUA = {"vi": 4, "en": 4}
DC = (("snr.0dB", "vi"), ("snr.0dB", "en"), ("nen.30pct", "vi"), ("nen.30pct", "en"), ("goc", None), ("snr.20dB", None))


def chon(rows: list, rng: random.Random) -> list:
    rows = sorted(rows, key=lambda r: r["ten"])
    out = []
    for n, k in SO_CHAN.items():
        out += rng.sample([r for r in rows if r["loai"] == "that" and r["c4"] == "chan" and r["nhom"] == n], k)
    for lg, k in SO_QUA.items():
        out += rng.sample([r for r in rows if r["loai"] == "that" and r["c4"] == "qua" and r["lang"] == lg], k)
    for ca, lg in DC:
        out.append(rng.choice([r for r in rows if r["loai"] == "doi_chung" and r["nhom"] == ca and lg in (None, r["lang"])]))
    rng.shuffle(out)
    return out


def run() -> None:
    for p in (OUT, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rows = chon(json.load(open(SRC, encoding="utf-8"))["tep"], random.Random(SEED))
    n = sum(SO_CHAN.values()) + sum(SO_QUA.values()) + len(DC)
    assert len(rows) == n == len({r["ten"] for r in rows})                    # kiểm trước khi ghi
    assert all(os.path.exists(os.path.join(GEM, THU_MUC, r["ten"])) for r in rows)
    open(OUT, "w", encoding="utf-8").write(page(rows, THU_MUC, "nghe_tay_c4_2411", "KQ_NGHE_C4V2_2411.jsonl"))
    json.dump({"seed": SEED, "script": "experts/P1/build_cond/nghe_tay_c4.py", "tep": rows},
              open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"xong: {OUT} ({n} tệp), {MAP}")
    print("thành phần:", sorted((r["loai"], r.get("c4"), r["nhom"], r["lang"]) for r in rows))


KQ = os.path.join(V6, NAT, "_kiem", "nghe_tay_c4_2411.json")
KQ_GEM = os.path.join(GEM, "KQ_NGHE_C4V2_2411.jsonl")
CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_NGHE_TAY_C4_2411.md")
W_NHOM = {"1_bh_duoi5": 14, "2_it_loi": 5, "3_vang": 6, "4_bh_5_10": 18, "5_con_lai": 3}   # số video dev C4 chặn mỗi nhóm
TI_LE_CHAN = (239, 1370)
ON = {"rong": ("ro", "che_loi"), "chat": ("che_loi",)}
DC_OK = {"goc": ("khong", "nhe"), "snr.20dB": ("khong", "nhe"), "snr.0dB": ("ro", "che_loi"), "nen.30pct": ("ro", "che_loi")}


def uoc(rows: list, on_of, on: tuple) -> float:
    """Tỉ lệ chặn oan có trọng số theo nhóm (trên tệp C4 chặn)."""
    tong = sum(W_NHOM.values())
    s = 0.0
    for n, w in W_NHOM.items():
        g = [r for r in rows if r["c4"] == "chan" and r["nhom"] == n]
        s += w / tong * (sum(on_of(r) not in on for r in g) / len(g) if g else 0.0)
    return s


def cham() -> None:
    from cham_dieu_kien import jl, pct, wilson
    kq = json.load(open(KQ, encoding="utf-8"))["ket_qua"]
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    thieu = [r["ten"] for r in rows if not all(k in kq.get(r["ten"], {}) for k in ("on_nen", "vang", "giong_ro"))]
    if thieu:
        raise SystemExit(f"Thiếu nhãn {len(thieu)} tệp: {thieu}")
    gem = {os.path.basename(r["tep"]): r for r in jl(KQ_GEM) if "stt" in r}
    on = lambda r: kq[r["ten"]]["on_nen"]  # noqa: E731
    that = [r for r in rows if r["loai"] == "that"]
    dc = [r for r in rows if r["loai"] == "doi_chung"]
    L = ["# Huấn nghe tay 40 tệp lượt C4 (luật chốt trước: NHAT_KY 2.16 «Huấn nghe tay C4»)", "",
         "Nguồn nhãn: **người nghe tay (Huấn)**, trang mù `gemini/NGHE_TAY_C4_2411.html`.", "",
         f"## 1. Đối chứng: khớp {sum(on(r) in DC_OK[r['nhom']] for r in dc)}/{len(dc)}", ""]
    L += [f"- {r['nhom']} ({r['lang']}, {r['ten']}): Huấn `{on(r)}` · cần {'/'.join(DC_OK[r['nhom']])} → "
          f"{'khớp' if on(r) in DC_OK[r['nhom']] else '**lệch**'} · Gemini `{gem[r['ten']]['on_nen']}`" for r in dc]
    L += ["", "## 2. Chặn oan / thả sai", "",
          "| phần | chặn oan rộng (khong/nhe) | chặn oan chặt (+ro) | thả sai rộng (ro/che_loi) | thả sai chặt (che_loi) |", "|---|---|---|---|---|"]
    phan = [("tất cả", that), ("vi", [r for r in that if r["lang"] == "vi"]), ("en", [r for r in that if r["lang"] == "en"])]
    phan += [(f"nhóm {n}", [r for r in that if r["nhom"] == n and r["c4"] == "chan"]) for n in W_NHOM]
    for ten, rs in phan:
        ch, qu = [r for r in rs if r["c4"] == "chan"], [r for r in rs if r["c4"] == "qua"]
        f = lambda k, n: pct(k, n) if n else "–"  # noqa: E731
        L.append(f"| {ten} | {f(sum(on(r) not in ON['rong'] for r in ch), len(ch))} | {f(sum(on(r) not in ON['chat'] for r in ch), len(ch))} | "
                 f"{f(sum(on(r) in ON['rong'] for r in qu), len(qu))} | {f(sum(on(r) in ON['chat'] for r in qu), len(qu))} |")
    ch = [r for r in that if r["c4"] == "chan"]
    p = TI_LE_CHAN[0] / TI_LE_CHAN[1]
    L += ["", "## 3. Ước chặn oan trên toàn bộ tệp accept p1_tunhien_v1", ""]
    for cach in ("rong", "chat"):
        k = sum(on(r) not in ON[cach] for r in ch)
        lo, hi = wilson(k, len(ch))
        L.append(f"- {cach}: {100 * p * uoc(that, on, ON[cach]):.1f} % (Wilson k/26: {100 * p * lo:.1f}–{100 * p * hi:.1f} %)"
                 f" = {TI_LE_CHAN[0]}/{TI_LE_CHAN[1]} × tỉ lệ chặn oan có trọng số nhóm")
    L.append("- Điều kiện cài vào K1: ≤ 2 % theo cả hai cách đếm.")
    hai = lambda v: v in ("ro", "che_loi")  # noqa: E731
    L += ["", "## 4. Gemini so với Huấn (on_nen, 40 tệp)", "",
          f"- trùng 4 mức: {pct(sum(gem[r['ten']]['on_nen'] == on(r) for r in rows), len(rows))}",
          f"- trùng mức gộp (khong/nhe ↔ ro/che_loi): {pct(sum(hai(gem[r['ten']]['on_nen']) == hai(on(r)) for r in rows), len(rows))}",
          "", "## 5. Từng tệp", "", "| tệp | loại | C4 | nhóm | lang | Huấn on_nen | vang | giong_ro | Gemini on_nen | ghi chú |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["loai"], str(r.get("c4")), r["nhom"], r["ten"])):
        h = kq[r["ten"]]
        L.append(f"| {r['ten']} | {r['loai']} | {r.get('c4') or ''} | {r['nhom']} | {r['lang']} | {h['on_nen']} | {h['vang']} | "
                 f"{h['giong_ro']} | {gem[r['ten']]['on_nen']} | {h.get('ghi_chu', '')} |")
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    rr = [{"c4": "chan", "nhom": n, "ten": f"{n}{i}"} for n in W_NHOM for i in range(2)]
    lab = {r["ten"]: ("khong" if r["ten"].endswith("0") else "ro") for r in rr}
    assert abs(uoc(rr, lambda r: lab[r["ten"]], ON["rong"]) - 0.5) < 1e-12
    src = json.load(open(SRC, encoding="utf-8"))["tep"]
    a, b = chon(src, random.Random(SEED)), chon(src, random.Random(SEED))
    assert [r["ten"] for r in a] == [r["ten"] for r in b] and len(a) == 40 == len({r["ten"] for r in a})
    h = page(a[:2], THU_MUC, "nghe_tay_c4_2411", "KQ_NGHE_C4V2_2411.jsonl")
    assert f'src="{THU_MUC}/{a[0]["ten"]}"' in h and "nghe_tay_c4_2411.json" in h and "nghe_2411/" not in h and "2 tệp" in h
    print("nghe_tay_c4 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "cham": cham, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
