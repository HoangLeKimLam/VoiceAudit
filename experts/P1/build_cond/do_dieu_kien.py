# -*- coding: utf-8 -*-
"""Đo điều kiện ghi âm cho 3 dòng Constraint (ồn · nền · vang) bằng 2 mô hình dựng sẵn, ghi số đo thô (CHƯA áp ngưỡng):
  Brouhaha (pyannote/brouhaha, Lavechin et al. 2023; mã MIT, trọng số OpenRAIL): mỗi khung ~17 ms → vad, SNR (dB), C50 (dB)
      → snr_bh_db, c50_bh_db = trung vị trên khung vad > 0,5; loi_noi_bh_s
  DeepFilterNet3 (Schröter et al. 2023; MIT / Apache-2.0): lọc giọng ở 48 kHz → phần dư = tệp − giọng đã lọc
      → nen_dfn_pct = năng lượng phần dư / năng lượng tệp (dòng «âm không phải tiếng nói ≤ 10 %»); snr_dfn_db = giọng / phần dư
Chạy bằng môi trường RIÊNG D:\\venv_brouhaha (numpy < 2, pyannote.audio 3.3; không cài vào .venv của nhóm).
Ra: data_labels/v6/<bo>/_kiem/dieu_kien.jsonl (1 dòng / tệp; chạy lại thì bỏ qua tệp đã đo).

  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_dieu_kien.py run <bo> [--limit N]
  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_dieu_kien.py selftest"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SR = 16000
_M: dict = {}


def energy_split(y: np.ndarray, clean: np.ndarray) -> tuple[float, float]:
    """(nen_pct, snr_db) từ tệp y và giọng đã lọc `clean` (cùng độ dài): phần dư = y − clean."""
    r = y - clean
    ey, ec, er = float(np.sum(y ** 2)), float(np.sum(clean ** 2)), float(np.sum(r ** 2))
    return 100 * er / max(ey, 1e-20), 10 * np.log10(max(ec, 1e-20) / max(er, 1e-20))


def frame_stats(d: np.ndarray, step: float) -> dict:
    """d: (khung, 3) = vad, snr, c50 của Brouhaha → trung vị SNR / C50 trên khung có lời nói."""
    sp = d[:, 0] > 0.5
    med = lambda c: round(float(np.median(d[sp, c])), 2) if sp.any() else None
    return {"snr_bh_db": med(1), "c50_bh_db": med(2), "loi_noi_bh_s": round(float(sp.sum() * step), 2)}


def _models():
    if not _M:
        import torch
        from df.enhance import init_df
        from pyannote.audio import Inference, Model
        torch.set_num_threads(4)
        _M["bh"] = Inference(Model.from_pretrained("pyannote/brouhaha"))
        _M["dfn"] = init_df(log_level="ERROR")[:2]
    return _M


def measure(y: np.ndarray) -> dict:
    import torch
    import torchaudio.functional as AF
    from df.enhance import enhance
    m = _models()
    out = m["bh"]({"waveform": torch.from_numpy(y[None].astype(np.float32)), "sample_rate": SR})
    res = frame_stats(np.asarray(out.data), out.sliding_window.step)
    model, st = m["dfn"]
    x48 = AF.resample(torch.from_numpy(y.astype(np.float32))[None], SR, st.sr())
    c = AF.resample(enhance(model, st, x48, pad=True), st.sr(), SR)[0].numpy()[:len(y)]
    c = np.pad(c, (0, len(y) - len(c)))
    nen, snr = energy_split(y.astype(np.float64), c.astype(np.float64))
    return {**res, "nen_dfn_pct": round(nen, 3), "snr_dfn_db": round(snr, 2)}


def run(bo: str, limit: int | None = None) -> None:
    import soundfile as sf
    d = os.path.join(ROOT, "data_labels", "v6", bo)
    cases = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    cases = cases if isinstance(cases, list) else cases["cases"]
    os.makedirs(os.path.join(d, "_kiem"), exist_ok=True)
    raw = os.path.join(d, "_kiem", "dieu_kien.jsonl")
    done = {json.loads(l)["file"] for l in open(raw, encoding="utf-8")} if os.path.exists(raw) else set()
    todo = [c for c in cases if c["file"] not in done and os.path.exists(os.path.join(d, c["file"]))][:limit]
    print(f"{bo}: {len(todo)} tệp cần đo ({len(done)} đã có)", flush=True)
    t0 = time.time()
    with open(raw, "a", encoding="utf-8") as fh:
        for i, c in enumerate(todo):
            y, sr = sf.read(os.path.join(d, c["file"]), dtype="float32")
            if sr != SR or y.ndim != 1:
                raise SystemExit(f"{c['file']}: cần mono 16 kHz, gặp sr={sr} shape={y.shape}")
            t1 = time.time()
            fh.write(json.dumps({"file": c["file"], "dur_s": round(len(y) / SR, 2), **measure(y), "s": round(time.time() - t1, 1)},
                                ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", raw, flush=True)


def selftest() -> None:
    rng = np.random.default_rng(0)
    s, n = rng.standard_normal(16000), rng.standard_normal(16000)
    g = np.sqrt(0.1 / 0.9)                          # nền = 10 % năng lượng → SNR = 10·log10(9) ≈ 9,54 dB
    nen, snr = energy_split(s + g * n * np.sqrt(np.sum(s ** 2) / np.sum(n ** 2)), s)
    assert abs(nen - 10) < 0.5 and abs(snr - 9.54) < 0.3, (nen, snr)
    d = np.array([[0.9, 20, 5], [0.1, -5, 30], [0.8, 10, 7]], float)
    st = frame_stats(d, 0.02)
    assert st == {"snr_bh_db": 15.0, "c50_bh_db": 6.0, "loi_noi_bh_s": 0.04}, st
    assert frame_stats(np.zeros((3, 3)), 0.02)["snr_bh_db"] is None
    print("do_dieu_kien selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"] and len(a) >= 2:
        run(a[1], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    else:
        raise SystemExit(__doc__)
