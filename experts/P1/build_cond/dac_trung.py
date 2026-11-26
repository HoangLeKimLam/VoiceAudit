# -*- coding: utf-8 -*-
"""Đặc trưng F2 / F3 cho mô hình đo ồn / nền / vang (NHAT_KY 2.20), chạy trong .venv (CPU):
  AST  = k1x.audio_events (≤ 8 cửa sổ 10,24 s; mỗi nhóm speech / music / tvradio / event …: max + trung bình)
  w2v  = wav2vec2 VI (vaudit.asr) lớp ẩn cuối, KHÔNG dùng features.chunk_frames (lỗi hàng trống, lanes/DE_XUAT/2511_P1_chunk-frames.md):
         khúc 20 s KHÔNG chồng, nối thẳng; gộp trung bình trên khung lời / khung nghỉ (VAD năng lượng 20 ms của k1.speech_mask) → 1 536 số
Ra: data_labels/v6/<bo>/_kiem/dac_trung.jsonl (chạy lại thì bỏ qua tệp đã có). --dev: chỉ tệp fold 0–2 (tệp thật p1_tunhien_v1).

  .venv/Scripts/python.exe experts/P1/build_cond/dac_trung.py run <bo> [--dev] [--limit N] | selftest
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import fold  # noqa: E402

SR, CH, HOP = 16000, 20 * 16000, 320
DEV = "cpu"                                   # --gpu → "cuda" (D:\venv_gpu, TF32 tắt); mỗi bộ chỉ đo bằng MỘT thiết bị


def _on(m):
    """Mô hình về DEV một lần (vaudit nạp trên CPU)."""
    if next(m.parameters()).device.type != DEV:
        m.to(DEV)
    return m


def ast_events(y: np.ndarray, max_windows: int = 8) -> dict:
    """Như k1x.audio_events (cùng cửa sổ, cùng nhóm) nhưng chạy trên DEV."""
    import torch
    from vaudit import k1x
    fx, m, grp = k1x._ast()
    m = _on(m)
    W = int(10.24 * SR)
    n = len(y)
    starts = [0] if n <= W else sorted({int(s) for s in np.linspace(0, n - W, min(max_windows, max(1, int(n / W) + 1)))})
    per = {k: [] for k in grp}
    for s in starts:
        x = {k: v.to(DEV) for k, v in fx(y[s:s + W].astype(np.float32), sampling_rate=SR, return_tensors="pt").items()}
        with torch.no_grad():
            p = torch.sigmoid(m(**x).logits[0]).cpu().numpy()
        for k, idx in grp.items():
            per[k].append(float(p[idx].max()) if idx else 0.0)
    return {o: f(v) for k, v in per.items() for o, f in ((k + "_max", lambda v: float(np.max(v))), (k + "_mean", lambda v: float(np.mean(v))))}


def hidden(y: np.ndarray) -> np.ndarray:
    """(khung 20 ms × 768): khúc 20 s không chồng, mỗi khúc chuẩn hoá như _w2v_hidden; khúc cuối < 1 s gộp vào khúc trước."""
    import torch
    from vaudit.asr import _load
    _, model = _load()
    model = _on(model)
    cuts = list(range(0, len(y), CH))
    if len(cuts) > 1 and len(y) - cuts[-1] < SR:
        cuts.pop()
    out = []
    for i, s in enumerate(cuts):
        seg = y[s: cuts[i + 1] if i + 1 < len(cuts) else len(y)]
        x = torch.from_numpy(((seg - seg.mean()) / (seg.std() + 1e-7)).astype(np.float32))[None].to(DEV)
        with torch.no_grad():
            out.append(model.wav2vec2(x).last_hidden_state[0].cpu().numpy())
    return np.concatenate(out)


def pool(h: np.ndarray, y: np.ndarray) -> tuple[list, float]:
    """Trung bình lớp ẩn trên khung lời và khung nghỉ (lưới 20 ms); thiếu khung (< 5) thì lấy trung bình cả tệp."""
    from vaudit import k1
    n = len(h)
    db = 10 * np.log10(np.square(y[: n * HOP].reshape(n, HOP), dtype=np.float64).mean(1) + 1e-12) if len(y) >= n * HOP else None
    if db is None:
        db = 10 * np.log10(np.square(np.pad(y, (0, n * HOP - len(y))), dtype=np.float64).reshape(n, HOP).mean(1) + 1e-12)
    sp, _ = k1.speech_mask(db)
    allm = h.mean(0)
    a = h[sp].mean(0) if sp.sum() >= 5 else allm
    b = h[~sp].mean(0) if (~sp).sum() >= 5 else allm
    return [round(float(v), 4) for v in np.r_[a, b]], float(sp.mean())


def files_of(bo: str, dev: bool) -> list[str]:
    c = json.load(open(os.path.join(ROOT, "data_labels", "v6", bo, "cases.json"), encoding="utf-8"))
    c = c if isinstance(c, list) else c["cases"]
    return [x["file"] for x in c if not dev or fold(x["base"]) < 3]


def run(bo: str, dev: bool, limit: int | None) -> None:
    from vaudit import k1x
    d = os.path.join(ROOT, "data_labels", "v6", bo)
    out = os.path.join(d, "_kiem", "dac_trung.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["file"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [f for f in files_of(bo, dev) if f not in done][:limit]
    print(f"{bo}: {len(todo)} tệp cần trích ({len(done)} đã có)", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for i, f in enumerate(todo):
            y, sr = sf.read(os.path.join(d, f), dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if sr != SR:
                raise SystemExit(f"{f}: cần 16 kHz, gặp {sr}")
            w, share = pool(hidden(y), y)
            fh.write(json.dumps({"file": f, "ast": {k: round(v, 4) for k, v in ast_events(y).items()},
                                 "ti_le_khung_loi": round(share, 3), "wada_db": round(float(k1x.wada_snr(y)), 2),   # chỉ để XÁC NHẬN độc lập
                                 "w2v": w}, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", out, flush=True)


def selftest() -> None:
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(int(41.5 * SR)) * 0.05).astype(np.float32)
    y[: 20 * SR] *= 0.01                                    # 20 s đầu nhỏ: khung nghỉ
    h = hidden(y)
    assert abs(len(h) - 41.5 / 0.02) < 6, len(h)             # 20 s + 21,5 s (khúc cuối 1,5 s đứng riêng) — không thiếu khung
    assert np.all(np.abs(h).sum(1) > 0), "có khung lớp ẩn toàn 0"
    v, share = pool(h, y)
    assert len(v) == 1536 and 0.3 < share < 0.7, share
    print("dac_trung selftest dat")


def khop(n: int = 20) -> None:
    """So đặc trưng trên DEV với mốc CPU (p1_dieukien_v3/_kiem/dac_trung_cpu_moc.jsonl, .venv transformers 4.57) trên n tệp."""
    d = os.path.join(ROOT, "data_labels", "v6", "p1_dieukien_v3")
    moc = [json.loads(l) for l in open(os.path.join(d, "_kiem", "dac_trung_cpu_moc.jsonl"), encoding="utf-8")][:n]
    da, dw = [], []
    for r in moc:
        y, _ = sf.read(os.path.join(d, r["file"]), dtype="float32")
        a = ast_events(y)
        w, _ = pool(hidden(y), y)
        da += [abs(a[k] - r["ast"][k]) for k in r["ast"]]
        dw.append(np.max(np.abs(np.array(w) - np.array(r["w2v"]))) / (np.max(np.abs(r["w2v"])) + 1e-9))
    print(f"{len(moc)} tệp · AST lệch tuyệt đối max {max(da):.4f} · w2v lệch tương đối max {max(dw):.4f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if "--gpu" in a:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False            # P1_QUY_TRINH §6
        torch.backends.cudnn.allow_tf32 = False
        DEV = "cuda"
    if a[0] == "khop":
        khop()
    elif a[0] == "selftest":
        selftest()
    else:
        run(a[1], "--dev" in a, int(a[a.index("--limit") + 1]) if "--limit" in a else None)
