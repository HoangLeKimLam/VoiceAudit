"""Đo độc lập (khác phương pháp K1) cho tệp K1 chặn: mặt nạ lời nói = VAD Brouhaha (khung > 0,5). Luật chốt: NHAT_KY_P1 2.31.
  loi_bh_s / loi_bh_pct : giây và tỉ lệ lời nói          → «quá ít tiếng nói»
  muc_loi_dbfs          : công suất TB trên mẫu thuộc khung lời → «quá nhỏ»
  e_6_8k_db             : năng lượng 6–8 kHz / 0–8 kHz trên khung lời → «băng hẹp»
Chạy bằng D:\\venv_brouhaha (cờ -I: không thêm thư mục script vào sys.path, nên tệp này tự đủ).

  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_doc_lap.py run
  D:\\venv_brouhaha\\Scripts\\python.exe -I experts/P1/build_cond/do_doc_lap.py selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
V6 = os.path.join(ROOT, "data_labels", "v6")
OUT = os.path.join(V6, "p1_tunhien_v2", "_kiem", "doc_lap_2611.jsonl")
SR = 16000
KIEM_PP = ("_goc", "tinhieu.bangtan_5k", "tinhieu.g711_len_16k", "tinhieu.muc_noi_-41.5dBFS", "tinhieu.tiengnoi_20pct")


def so_do(y: np.ndarray, vad: np.ndarray, step: float) -> dict:
    """vad: xác suất lời nói mỗi khung Brouhaha (bước `step` s)."""
    sp = vad > 0.5
    m = np.zeros(len(y), bool)
    for i in np.where(sp)[0]:
        m[int(i * step * SR): int((i + 1) * step * SR)] = True
    out = {"dur_s": round(len(y) / SR, 2), "loi_bh_s": round(float(sp.sum() * step), 2)}
    out["loi_bh_pct"] = round(100 * out["loi_bh_s"] / max(out["dur_s"], 1e-9), 1)
    if m.sum() < SR // 4:
        return {**out, "muc_loi_dbfs": None, "e_6_8k_db": None}
    s = y[m].astype(np.float64)
    out["muc_loi_dbfs"] = round(float(10 * np.log10(np.mean(s ** 2) + 1e-20)), 1)
    n = 1024
    k = len(s) // n
    P = (np.abs(np.fft.rfft(s[: k * n].reshape(k, n) * np.hanning(n), axis=1)) ** 2).sum(0)
    f = np.fft.rfftfreq(n, 1 / SR)
    out["e_6_8k_db"] = round(float(10 * np.log10(P[f >= 6000].sum() / P.sum() + 1e-30)), 1)
    return out


def ds() -> list:
    v2 = os.path.join(V6, "p1_tunhien_v2")
    out = [(os.path.join(v2, json.loads(l)["file"]), "v2", json.loads(l)) for l in open(os.path.join(v2, "_kiem", "k1_ket_qua_2611.jsonl"), encoding="utf-8")]
    out = [(p, b, r["file"], (r["reasons"] or [""])[0]) for p, b, r in out if r["decision"] == "04"]
    k = os.path.join(V6, "k1axes", "p1_v1")
    for f in sorted(os.listdir(k)):
        ca = "_goc" if f.startswith("_goc__") else f.split("__")[0]
        if ca in KIEM_PP:
            out.append((os.path.join(k, f), "kiem_pp", f, ca))
    assert sum(b == "v2" for _, b, _, _ in out) == 21 and sum(b == "kiem_pp" for _, b, _, _ in out) == 120, len(out)
    return out


def run() -> None:
    import soundfile as sf
    import torch
    from pyannote.audio import Inference, Model
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    torch.set_num_threads(4)
    bh = Inference(Model.from_pretrained("pyannote/brouhaha"))
    rows = []
    for i, (p, bo, f, ly) in enumerate(ds()):
        y, sr = sf.read(p, dtype="float32", always_2d=True)
        y = y.mean(1)
        if sr != SR:
            raise SystemExit(f"{f}: sr={sr}, cần 16 kHz")
        o = bh({"waveform": torch.from_numpy(y[None]), "sample_rate": SR})
        rows.append({"bo": bo, "file": f, "k1": ly, **so_do(y, np.asarray(o.data)[:, 0], o.sliding_window.step)})
        print(i + 1, bo, f, rows[-1], flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def selftest() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(4 * SR) / SR
    y = 0.01 * np.sin(2 * np.pi * 500 * t) + 0.001 * rng.standard_normal(len(t))       # 500 Hz ~ -43 dBFS
    vad = np.r_[np.ones(100), np.zeros(100)]                                            # khung 20 ms: 2 s lời / 4 s
    r = so_do(y, vad, 0.02)
    assert r["loi_bh_s"] == 2.0 and r["loi_bh_pct"] == 50.0, r
    assert -44 < r["muc_loi_dbfs"] < -42, r
    assert r["e_6_8k_db"] < -15, r
    assert so_do(y, np.zeros(200), 0.02)["muc_loi_dbfs"] is None
    print("selftest đạt", r)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1]]()
