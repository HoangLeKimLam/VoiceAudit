# -*- coding: utf-8 -*-
"""Kiểm H-vang + H-lặng (NHAT_KY 2.16, luật chốt 24/11). Một lượt DeepFilterNet3 (c = giọng lọc được, r = y − c), trọng số A, khung 20 ms:
  khung lời: E(c) ≥ p95(E(c)) − 30 dB · khung nghỉ: E(c) < p95 − 40 dB và cách khung lời gần nhất phía trước ≥ 200 ms
  nen_a_loi = E(r) / E(y) chỉ trên khung lời (%) · snr_nghi = 10·log10((mức y khung lời − nền) / nền), nền = trung bình y khung nghỉ
  (None nếu khung nghỉ < 0,3 s) · ti_le_loi, nghi_s
Ra: data_labels/v6/<bo>/_kiem/dieu_kien_vang.jsonl (chạy lại thì bỏ qua tệp đã đo).

  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_vang.py run <bo> [--limit N]
  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_vang.py selftest"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from do_tram import LOC, ROOT, SR, clean_dfn  # noqa: E402

N = 320                     # 20 ms
LOI_DB, NGHI_DB, CACH, NGHI_MIN_S = 30.0, 40.0, 10, 0.3   # CACH = 10 khung = 200 ms


def khung(x: np.ndarray) -> np.ndarray:
    m = len(x) // N
    return np.mean(x[:m * N].reshape(m, N) ** 2, axis=1)


def do_mot(y: np.ndarray, c: np.ndarray) -> dict:
    ya, ca = LOC["a"](y), LOC["a"](c)
    ey, ec, er = khung(ya), khung(ca), khung(ya - ca)
    p95 = float(np.percentile(ec, 95))
    loi = ec >= p95 * 10 ** (-LOI_DB / 10)
    tu_loi = np.full(len(ec), 10 ** 9)                      # số khung kể từ khung lời gần nhất phía trước
    last = -10 ** 9
    for i, v in enumerate(loi):
        last = i if v else last
        tu_loi[i] = i - last
    nghi = (ec < p95 * 10 ** (-NGHI_DB / 10)) & (tu_loi >= CACH)
    out = {"ti_le_loi": round(float(loi.mean()), 3), "nghi_s": round(float(nghi.sum() * N / SR), 2),
           "nen_a_loi": round(100 * float(er[loi].sum()) / max(float(ey[loi].sum()), 1e-20), 3), "snr_nghi": None}
    if nghi.sum() * N / SR >= NGHI_MIN_S:
        nen, sp = float(ey[nghi].mean()), float(ey[loi].mean())
        out["snr_nghi"] = round(10 * np.log10(max(sp - nen, 1e-20) / max(nen, 1e-20)), 2)
    return out


def run(bo: str, limit: int | None = None) -> None:
    import soundfile as sf
    d = os.path.join(ROOT, "data_labels", "v6", bo)
    cases = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    cases = cases if isinstance(cases, list) else cases["cases"]
    raw = os.path.join(d, "_kiem", "dieu_kien_vang.jsonl")
    done = {json.loads(l)["file"] for l in open(raw, encoding="utf-8")} if os.path.exists(raw) else set()
    todo = [c for c in cases if c["file"] not in done and os.path.exists(os.path.join(d, c["file"]))][:limit]
    print(f"{bo}: {len(todo)} tệp cần đo ({len(done)} đã có)", flush=True)
    t0 = time.time()
    with open(raw, "a", encoding="utf-8") as fh:
        for i, c in enumerate(todo):
            y, sr = sf.read(os.path.join(d, c["file"]), dtype="float64")
            if sr != SR or y.ndim != 1:
                raise SystemExit(f"{c['file']}: cần mono 16 kHz, gặp sr={sr} shape={y.shape}")
            fh.write(json.dumps({"file": c["file"], **do_mot(y, clean_dfn(y))}, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", raw, flush=True)


def selftest() -> None:
    rng = np.random.default_rng(0)
    t = SR * 4
    on = np.zeros(t)
    on[SR // 2: SR * 3 // 2] = 1
    on[SR * 2: SR * 3] = 1                                  # 2 s lời, 2 s nghỉ (0–0,5 · 1,5–2 · 3–4 s)
    s = rng.standard_normal(t) * on
    n = rng.standard_normal(t) * 0.1                        # nền đều −20 dB so với giọng (trên khung lời)
    m = do_mot(s + n, s)
    assert abs(m["ti_le_loi"] - 0.5) < 0.02, m
    assert abs(m["snr_nghi"] - 20) < 1.0, m                 # SNR khung lời ≈ 20 dB
    assert abs(m["nen_a_loi"] - 100 * 0.01 / 1.01) < 0.3, m
    # vang: đuôi giọng kéo dài 150 ms sau mỗi đoạn lời, không có nền → khung nghỉ (≥ 200 ms sau lời) không chứa vang
    tail = np.zeros(t)
    for e in (SR * 3 // 2, SR * 3):
        k = np.arange(int(0.15 * SR))
        tail[e:e + len(k)] = rng.standard_normal(len(k)) * 0.5 * np.exp(-k / (0.04 * SR))
    m2 = do_mot(s + tail + n * 0.01, s)
    assert m2["snr_nghi"] > 35, m2
    print("do_vang selftest dat", m, m2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"] and len(a) >= 2:
        run(a[1], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    else:
        raise SystemExit(__doc__)
