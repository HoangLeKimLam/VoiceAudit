# -*- coding: utf-8 -*-
"""Kiểm H-trầm (NHAT_KY 2.16, luật chốt 24/11): một lượt DeepFilterNet3, phần dư r = y − giọng lọc được LỌC SAU khi tách,
nen_X = E(f(r)) / E(f(y)) với 4 cách f: toàn dải (K0, phải tái lập nen_dfn_pct cũ), thông cao 100 Hz, 300 Hz, trọng số A.
Bộ lọc tuyến tính nên f(y) − f(c) = f(r). C50 (Brouhaha) không chạy lại, lấy từ dieu_kien.jsonl cũ.
Ra: data_labels/v6/<bo>/_kiem/dieu_kien_tram.jsonl (chạy lại thì bỏ qua tệp đã đo).

  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_tram.py run <bo> [--limit N]
  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_tram.py selftest"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.signal import bilinear, butter, lfilter, sosfiltfilt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SR = 16000
_M: dict = {}


def a_weight(fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Trọng số A (IEC 61672), số hoá bằng biến đổi song tuyến tính."""
    f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
    num = [(2 * np.pi * f4) ** 2 * 10 ** (1.9997 / 20), 0, 0, 0, 0]
    den = np.polymul([1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    den = np.polymul(np.polymul(den, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2])
    return bilinear(num, den, fs)


_HP = {hz: butter(4, hz, "highpass", fs=SR, output="sos") for hz in (100, 300)}
_A = a_weight(SR)
LOC = {"full": lambda x: x, "hp100": lambda x: sosfiltfilt(_HP[100], x), "hp300": lambda x: sosfiltfilt(_HP[300], x),
       "a": lambda x: lfilter(*_A, x)}


def nen_bien_the(y: np.ndarray, clean: np.ndarray) -> dict:
    """{nen_<cách>: % năng lượng phần dư sau lọc} cho mỗi cách lọc."""
    r = y - clean
    return {f"nen_{k}": round(100 * float(np.sum(f(r) ** 2)) / max(float(np.sum(f(y) ** 2)), 1e-20), 3) for k, f in LOC.items()}


def _dfn():
    if not _M:
        import torch
        from df.enhance import init_df
        torch.set_num_threads(4)
        _M["dfn"] = init_df(log_level="ERROR")[:2]
    return _M["dfn"]


def clean_dfn(y: np.ndarray) -> np.ndarray:
    import torch
    import torchaudio.functional as AF
    from df.enhance import enhance
    model, st = _dfn()
    x48 = AF.resample(torch.from_numpy(y.astype(np.float32))[None], SR, st.sr())
    c = AF.resample(enhance(model, st, x48, pad=True), st.sr(), SR)[0].numpy()[:len(y)]
    return np.pad(c, (0, len(y) - len(c))).astype(np.float64)


def run(bo: str, limit: int | None = None) -> None:
    import soundfile as sf
    d = os.path.join(ROOT, "data_labels", "v6", bo)
    cases = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    cases = cases if isinstance(cases, list) else cases["cases"]
    raw = os.path.join(d, "_kiem", "dieu_kien_tram.jsonl")
    done = {json.loads(l)["file"] for l in open(raw, encoding="utf-8")} if os.path.exists(raw) else set()
    todo = [c for c in cases if c["file"] not in done and os.path.exists(os.path.join(d, c["file"]))][:limit]
    print(f"{bo}: {len(todo)} tệp cần đo ({len(done)} đã có)", flush=True)
    t0 = time.time()
    with open(raw, "a", encoding="utf-8") as fh:
        for i, c in enumerate(todo):
            y, sr = sf.read(os.path.join(d, c["file"]), dtype="float64")
            if sr != SR or y.ndim != 1:
                raise SystemExit(f"{c['file']}: cần mono 16 kHz, gặp sr={sr} shape={y.shape}")
            fh.write(json.dumps({"file": c["file"], **nen_bien_the(y, clean_dfn(y))}, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", raw, flush=True)


def selftest() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(SR * 2) / SR
    s = rng.standard_normal(len(t))                                  # «giọng» phổ rộng
    hum = 3 * np.sin(2 * np.pi * 50 * t)                             # ù 50 Hz: năng lượng lớn, tai nghe nhỏ
    m = nen_bien_the(s + hum, s)
    assert abs(m["nen_full"] - 100 * 4.5 / 5.5) < 2, m               # E(hum) = 4,5 · E(s) = 1 → ~82 %
    assert m["nen_hp300"] < 1 and m["nen_a"] < 1 and m["nen_hp100"] < 5, m
    w = rng.standard_normal(len(t)) * 0.3                            # nền trắng: lọc không đổi tỉ lệ nhiều
    m2 = nen_bien_the(s + w, s)
    assert all(abs(v - 100 * 0.09 / 1.09) < 1.5 for v in m2.values()), m2
    print("do_tram selftest dat", m, m2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"] and len(a) >= 2:
        run(a[1], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    else:
        raise SystemExit(__doc__)
