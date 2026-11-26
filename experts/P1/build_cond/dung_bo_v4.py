# -*- coding: utf-8 -*-
"""Dựng bộ HUẤN LUYỆN `p1_dieukien_v4` cho mô hình đo ồn / nền / vang (Huấn duyệt 25/11; luật chốt trước: NHAT_KY 2.20).

Gọi lại hàm của dung_bo_v3 / dung_bo (không sửa chúng). Mỗi ngôn ngữ: goc 150 · snr 6 loại × 350 (SNR_A đích ½ U(−5, 30) + ½ U(7, 13) dB)
· nen_nhac_loa 300 (U(1, 30) %) · vang 450 (RIR mô phỏng lọc T20 / T30 + RIR thật qua cùng phép lọc; nhãn = T20 đo trên RIR).
Gốc cắt ≤ 30 s. Đáp án từ tham số trộn; kiểm độc lập trên tệp đầu ra bằng `kiem`.

  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v4.py build [--limit N]
  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v4.py kiem
  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v4.py selftest
"""
from __future__ import annotations

import argparse
import collections
import datetime
import os
import random
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dung_bo as D  # noqa: E402
import dung_bo_v3 as V3  # noqa: E402
from cham_dieu_kien import jl  # noqa: E402

BO, SEED, MAX_S = "p1_dieukien_v4", 2511, 30.0
N_GOC, N_SNR, N_NEN, N_VANG = 150, 350, 300, 450
RT60_BINS = tuple(round(0.3 + 0.1 * i, 1) for i in range(16))      # 0,3–1,8 s
RIR_PER_BIN = 12
DAY = False                       # --day (NHAT_KY 2.21): chỉ snr U(8, 16) dB × 150 / loại + nen U(5, 15) % × 100, mỗi ngôn ngữ; không goc / vang


# ----------------------------------------------------------------------------- hàm thuần
def snr_dich(rng: random.Random, san: float) -> float:
    """½ U(−5, 30) + ½ U(7, 13) dB, cắt cụt dưới sàn gốc − 1 dB (snr_add_for cần gốc sạch hơn đích)."""
    while True:
        v = rng.uniform(-5, 30) if rng.random() < 0.5 else rng.uniform(7, 13)
        if v < san - 1:
            return round(v, 2)


def rir_ok(h: np.ndarray) -> tuple | None:
    """(T20, T30) nếu EDC thẳng (|T30 − T20| ≤ 20 % T20) và T20 trong 0,25–1,9 s (cùng phép lọc của dung_bo.rir_bins)."""
    t20, t30 = D.rt60_schroeder(h), D.rt60_schroeder(h, lo_db=-35)
    if not (t20 and t30) or abs(t30 - t20) > D.T30_T20_TOL * t20 or not 0.25 <= t20 <= 1.9:
        return None
    return round(t20, 3), round(t30, 3)


def bin_of(t: float) -> float:
    return min(RT60_BINS, key=lambda b: abs(b - t))


# ----------------------------------------------------------------------------- nguồn
def rir_pool(rng: random.Random) -> list[dict]:
    D.RT60_S = RT60_BINS
    bins = D.rir_bins(rng, RIR_PER_BIN)
    sims = [{**r, "loai_rir": "pra"} for t in RT60_BINS for r in bins[t]]
    real = []
    for p in V3.rir_that():
        ok = rir_ok(D.load(p))
        if ok:
            real.append({"path": p, "rt60": ok[0], "rt60_t30": ok[1], "loai_rir": "that"})
    print(f"RIR: mô phỏng {len(sims)} · thật qua lọc {len(real)}/{len(V3.rir_that())}", flush=True)
    return sims + real


# ----------------------------------------------------------------------------- việc
def chon(that: list, fl: list, rng: random.Random) -> dict:
    return rng.choice(that) if rng.random() < 0.5 else rng.choice(fl)


def viec_day(that: list, fl: list, rng: random.Random) -> list:
    out = []
    for loai in V3.LOAI:
        for _ in range(150):
            b = chon(that, fl, rng)
            lv = round(rng.uniform(8, min(16, b["san_a"] - 1)), 2)
            out.append(("snr", f"snr_{loai}.{lv}dB", lv, loai, b))
    for _ in range(100):
        pct = round(rng.uniform(5, 15), 2)
        out.append(("nen", f"nen_nhac_loa.{pct}pct", pct, "nhac_loa", chon(that, fl, rng)))
    return out


def viec(lang: str, that: list, fl: list, rirs: list, rng: random.Random) -> list:
    if DAY:
        return viec_day(that, fl, rng)
    out =[("goc", "goc", 0, None, chon(that, fl, rng)) for _ in range(N_GOC)]
    for loai in V3.LOAI:
        for _ in range(N_SNR):
            b = chon(that, fl, rng)
            lv = snr_dich(rng, b["san_a"])
            out.append(("snr", f"snr_{loai}.{lv}dB", lv, loai, b))
    for _ in range(N_NEN):
        pct = round(rng.uniform(1, 30), 2)
        out.append(("nen", f"nen_nhac_loa.{pct}pct", pct, "nhac_loa", chon(that, fl, rng)))
    out += [("vang", None, None, rng.choice(rirs), chon(that, fl, rng)) for _ in range(N_VANG)]
    return out


def mot_vang(w, r: dict, b: dict, lang: str) -> None:
    s = V3.nap_goc(b)
    y = D.reverb(s, D.load(r["path"]))
    lic = "tự sinh (pyroomacoustics, MIT)" if r["loai_rir"] == "pra" else "Apache 2.0 (RIRS_NOISES)"
    phong = {k: r[k] for k in ("phong_m", "hap_thu", "max_order", "nguon_xyz", "mic_xyz", "rt60_thiet_ke_s") if k in r}
    ts = {"goc_tep": b["file"], "cat_s": b["cat_s"], "san_a_db": b["san_a"], "san_full_db": b["san_full"], "loai_rir": r["loai_rir"]}
    w.add("vang", f"vang.rt60_{bin_of(r['rt60'])}s", r["rt60"], y, V3.base_w(b), lang, [V3.entry(r["path"], lic, "RIR/" + r["loai_rir"], **phong)],
          ts, {"rt60_rir_schroeder_s": r["rt60"], "rt60_rir_t30_s": r["rt60_t30"]})


# ----------------------------------------------------------------------------- dựng
def the_bo(w, bo_stat: dict, limit) -> None:
    by = collections.Counter(c["ca"].split(".")[0] for c in w.cases)
    md = collections.Counter(c["mong_doi"] for c in w.cases)
    ng = collections.Counter(m["meta"]["source"] for m in w.manifest)
    L = [f"# Thẻ bộ `{BO}` (bộ HUẤN LUYỆN mô hình đo ồn / nền / vang)", "",
         f"- **Ngày dựng:** {datetime.date.today().isoformat()} · **Ai làm:** Huấn (P1); mã do Claude Code (Claude Opus 5.5) viết theo thiết kế Huấn duyệt 25/11",
         f"- **Cách làm:** `experts/P1/build_cond/dung_bo_v4.py` (seed {SEED}{', limit ' + str(limit) if limit else ''}); luật `lanes/P1/NHAT_KY/NHAT_KY_P1.md` 2.20{" + 2.21 (--day: snr U(8, 16) dB, nen U(5, 15) %)" if DAY else ""}. "
         "Trộn bằng hàm của `dung_bo_v3` (SNR / nền trọng số A theo mức đích) + vang bằng RIR có RT60 đo trên RIR (T20, lọc |T30 − T20| ≤ 20 %). Không gán tay.",
         "- **Nguồn + giấy phép:** gốc thật `p1_tunhien_v1` (YouTube CC BY, kênh không chuyên, person = kênh) · FLEURS vi_vn / en_us (CC BY 4.0) · "
         "DEMAND (CC BY 4.0) · MUSAN music / speech / noise (CC BY / BY-SA / PD; bỏ NC / ND, bỏ thánh ca) · RIRS_NOISES RIR + ồn phòng thật (Apache 2.0) · "
         "RIR mô phỏng pyroomacoustics. Từng tệp: `provenance.jsonl`.",
         f"- **Gốc bị bỏ (sàn A < {V3.MIN_SAN_A:g} dB):** {bo_stat} · gốc cắt ≤ {MAX_S:g} s",
         f"- **Số tệp:** {len(w.cases)} · theo trục: {dict(sorted(by.items()))} · đáp án: {dict(md)} · nguồn gốc: {dict(ng)} · "
         f"person: {len({m['meta']['person'] for m in w.manifest})}",
         "- **Hạn chế:** FLEURS không có mã người nói (person = từng tệp) · loại `nguoi_noi_1`, `quan`, `tv` có giọng người khác · "
         "RT60 của gốc khô chưa đo (coi ≈ 0,3 s vì C50 ≥ 20 dB) · không có ca ghép ồn + vang · **chỉ để huấn luyện / chọn trên fold 0–2; fold 3–4 là test**.",
         "- **Đã kiểm:** *(điền sau)* `dung_bo_v4.py kiem` · `kiem_bo --moi` · `lockbox check` · Spearman(RT60, C50 Brouhaha)"]
    open(os.path.join(D.OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def build(limit: int | None) -> None:
    D.BO, D.OUT = BO, os.path.join(D.ROOT, "data_labels", "v6", BO)
    D.RIR_DIR = os.path.join(D.RAW, "rir_pra", BO)
    V3.MAX_S = MAX_S
    if os.path.exists(os.path.join(D.OUT, "manifest.json")):
        raise SystemExit(f"{D.rel(D.OUT)} đã có manifest.json: không ghi đè")
    rng = random.Random(SEED)
    P = V3.nguon_chung()
    rirs = [] if DAY else rir_pool(rng)
    jobs, bo_stat = [], {}
    for lang in ("vi", "en"):
        that, fl, bo_stat[lang] = V3.chon_goc(lang, rng)
        print(f"{lang}: gốc thật {len(that)} ({len({b['kenh'] for b in that})} kênh) · FLEURS {len(fl)} · bỏ {bo_stat[lang]}", flush=True)
        jobs += [(lang, j) for j in viec(lang, that, fl, rirs, rng)]
    if limit:
        jobs = rng.sample(jobs, limit)
    print("việc theo trục:", dict(collections.Counter(j[0] for _, j in jobs)), flush=True)
    w = D.Writer(D.OUT, {**D.script_version(), "seed": SEED})
    w.ver = {**w.ver, "script": D.rel(os.path.abspath(__file__)), "script_sha256": D.sha256(os.path.abspath(__file__))}
    for i, (lang, j) in enumerate(jobs, 1):
        if j[0] == "vang":
            mot_vang(w, j[3], j[4], lang)
        else:
            V3.mot_tep(w, j, lang, rng, P)
        if i % 200 == 0:
            print(f"  {i}/{len(jobs)}", flush=True)
    w.flush(rng)
    the_bo(w, bo_stat, limit)
    print(f"{len(w.manifest)} tệp → {D.rel(D.OUT)}")


def kiem() -> None:
    """Đo lại độc lập trên tệp đầu ra (như dung_bo_v3.kiem, bỏ qua vang: vang kiểm bằng C50 Brouhaha sau khi đo)."""
    out = os.path.join(D.ROOT, "data_labels", "v6", BO)
    rows, lech = [p for p in jl(os.path.join(out, "provenance.jsonl")) if p["ca"].startswith(("snr_", "nen_"))], []
    for p in rows:
        y, _ = sf.read(os.path.join(D.ROOT, p["file"]), dtype="float64")
        s = V3.nap_goc({"wav": os.path.join(D.ROOT, p["goc"]["path"])})[: len(y)]
        ts, dl = p["tham_so"], p["do_lai"]
        if "snr_that_A_db" in dl:
            v, ref = V3.kiem_lech(y, s, ts["san_a_db"]), dl["snr_that_A_db"]
        else:
            k = float(np.dot(y, s) / np.dot(s, s))
            v, ref = 100 * V3.ea(y - k * s) / V3.ea(y), dl["nen_A_pct"]
        if abs(v - ref) > 0.3:
            lech.append((p["id"], p["ca"], round(v, 2), ref))
    L = [f"# Kiểm đáp án `{BO}` (đo lại độc lập trên tệp đầu ra)", "", f"- Tệp snr / nen: {len(rows)} · lệch > 0,3: **{len(lech)}**"]
    L += [f"- {x}" for x in lech[:40]]
    open(os.path.join(out, "_kiem_dap_an.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def selftest() -> None:
    r = random.Random(0)
    v = [snr_dich(r, 26.0) for _ in range(4000)]
    assert max(v) < 25.0 and min(v) >= -5.0
    assert 0.4 < np.mean([7 <= x <= 13 for x in v]) < 0.7 and np.mean([15 <= x <= 20 for x in v]) > 0.05
    for t in (0.4, 1.0, 1.6):                                   # RIR mũ tổng hợp: T20 = t, EDC thẳng → nhận
        n = np.arange(int(2.5 * t * D.SR))
        h = np.random.default_rng(1).standard_normal(len(n)) * 10 ** (-3 * n / (t * D.SR))
        ok = rir_ok(h)
        assert ok and abs(ok[0] - t) < 0.05 * t, (t, ok)
    assert rir_ok(np.r_[1.0, np.zeros(100)]) is None
    assert bin_of(0.94) == 0.9 and bin_of(2.0) == 1.8
    print("dung_bo_v4 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "kiem", "selftest"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--bo", default=BO)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--day", action="store_true")
    a = ap.parse_args()
    BO, SEED, DAY = a.bo, a.seed, a.day
    {"build": lambda: build(a.limit), "kiem": kiem, "selftest": selftest}[a.cmd]()
