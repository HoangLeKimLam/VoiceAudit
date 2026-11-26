"""Đặc trưng ứng viên S1 / S2 (NHAT_KY 2.22, luật chốt trước), chạy GPU trong D:\\venv_gpu (TF32 tắt; mỗi bộ đo bằng MỘT thiết bị).
  S1 (`_kiem/dac_trung2.jsonl`, khoá "bh"): Brouhaha theo khung — trên khung lời (vad > 0,5) phân vị 10/25/50/75/90 + TB + độ lệch
     của SNR và C50; trung vị SNR trên khung nghỉ; tỉ lệ khung lời.
  S2 (`_kiem/w2v_lop/<file với / → __>.npy`, float16, (lớp, 3, 768)): w2v VI (vaudit.asr) mọi lớp ẩn, khúc 20 s không chồng như
     dac_trung.hidden; mỗi lớp: TB khung lời · TB khung nghỉ · độ lệch khung lời (VAD năng lượng như dac_trung.pool).
Tự kiểm (`khop`): p50 SNR khung lời ≈ snr_bh_db có sẵn; TB lời / nghỉ của lớp cuối ≈ w2v trong dac_trung.jsonl.

  D:\\venv_gpu\\Scripts\\python.exe -I experts/P1/build_cond/dac_trung2.py run <bo> [--dev] [--limit N] | khop <bo> [N]
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import dac_trung as D  # noqa: E402

SR, HOP = 16000, 320
_M: dict = {}


def nap() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("không thấy cuda")
    torch.backends.cuda.matmul.allow_tf32 = False             # P1_QUY_TRINH §6
    torch.backends.cudnn.allow_tf32 = False
    from pyannote.audio import Inference, Model
    from pyannote.audio.core.model import Introspection
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion
    from vaudit.asr import _load
    torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution, Introspection])
    _M["bh"] = Inference(Model.from_pretrained("pyannote/brouhaha"), device=torch.device("cuda"))
    _M["w2v"] = _load()[1].to("cuda")


def s1(y: np.ndarray) -> dict:
    out = _M["bh"]({"waveform": torch.from_numpy(y[None].astype(np.float32)), "sample_rate": SR})
    d = np.asarray(out.data)
    sp = d[:, 0] > 0.5
    r = {"ti_le_loi_bh": round(float(sp.mean()), 4),
         "snr_nghi_p50": round(float(np.median(d[~sp, 1])), 3) if (~sp).any() else None}
    for c, ten in ((1, "snr"), (2, "c50")):
        v = d[sp, c]
        for q in (10, 25, 50, 75, 90):
            r[f"{ten}_p{q}"] = round(float(np.percentile(v, q)), 3) if sp.any() else None
        r[f"{ten}_tb"] = round(float(v.mean()), 3) if sp.any() else None
        r[f"{ten}_dl"] = round(float(v.std()), 3) if sp.any() else None
    return r


def s2(y: np.ndarray) -> np.ndarray:
    """(lớp, 3, 768): TB lời · TB nghỉ · độ lệch lời. Khúc + chuẩn hoá giống dac_trung.hidden; mặt nạ giống dac_trung.pool."""
    from vaudit import k1
    cuts = list(range(0, len(y), D.CH))
    if len(cuts) > 1 and len(y) - cuts[-1] < SR:
        cuts.pop()
    hs = []
    for i, s in enumerate(cuts):
        seg = y[s: cuts[i + 1] if i + 1 < len(cuts) else len(y)]
        x = torch.from_numpy(((seg - seg.mean()) / (seg.std() + 1e-7)).astype(np.float32))[None].to("cuda")
        with torch.no_grad():
            hs.append(torch.stack(_M["w2v"].wav2vec2(x, output_hidden_states=True).hidden_states)[:, 0].cpu().numpy())
    h = np.concatenate(hs, axis=1)                                            # (lớp, khung, 768)
    n = h.shape[1]
    yy = y[: n * HOP] if len(y) >= n * HOP else np.pad(y, (0, n * HOP - len(y)))
    sp, _ = k1.speech_mask(10 * np.log10(np.square(yy.reshape(n, HOP), dtype=np.float64).mean(1) + 1e-12))
    allm = h.mean(1)
    a = h[:, sp].mean(1) if sp.sum() >= 5 else allm
    b = h[:, ~sp].mean(1) if (~sp).sum() >= 5 else allm
    c = h[:, sp].std(1) if sp.sum() >= 5 else h.std(1)
    return np.stack([a, b, c], axis=1)


def ten_npy(f: str) -> str:
    return f.replace("/", "__").replace("\\", "__")[:-4] + ".npy"


def run(bo: str, dev: bool, limit: int | None) -> None:
    d = os.path.join(D.ROOT, "data_labels", "v6", bo)
    out, nd = os.path.join(d, "_kiem", "dac_trung2.jsonl"), os.path.join(d, "_kiem", "w2v_lop")
    os.makedirs(nd, exist_ok=True)
    done = {json.loads(l)["file"] for l in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    todo = [f for f in D.files_of(bo, dev) if f not in done][:limit]
    nap()
    print(f"{bo}: {len(todo)} tệp cần trích ({len(done)} đã có)", flush=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as fh:
        for i, f in enumerate(todo):
            y, sr = sf.read(os.path.join(d, f), dtype="float32")
            if y.ndim > 1:
                y = y.mean(1)
            if sr != SR:
                raise SystemExit(f"{f}: cần 16 kHz, gặp {sr}")
            np.save(os.path.join(nd, ten_npy(f)), s2(y).astype(np.float16))  # npy trước, dòng jsonl sau → dòng có nghĩa là đủ cả hai
            fh.write(json.dumps({"file": f, "bh": s1(y)}, ensure_ascii=False) + "\n")
            fh.flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", out, flush=True)


def khop(bo: str, n: int = 20) -> None:
    d = os.path.join(D.ROOT, "data_labels", "v6", bo)
    dk = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(d, "_kiem", "dieu_kien.jsonl"), encoding="utf-8")}
    dt = {json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(d, "_kiem", "dac_trung.jsonl"), encoding="utf-8")}
    files = [f for f in sorted(dt) if f in dk][:n]
    nap()
    e1, e2 = [], []
    for f in files:
        y, _ = sf.read(os.path.join(d, f), dtype="float32")
        r, h = s1(y), s2(y)
        if dk[f]["snr_bh_db"] is not None:
            e1.append(abs(r["snr_p50"] - dk[f]["snr_bh_db"]))
        ref = np.array(dt[f]["w2v"])
        e2.append(float(np.max(np.abs(np.r_[h[-1, 0], h[-1, 1]] - ref)) / (np.max(np.abs(ref)) + 1e-9)))
    print(f"{len(files)} tệp · số lớp {h.shape[0]} · |p50 − snr_bh_db| max {max(e1):.3f} dB · w2v lớp cuối lệch tương đối max {max(e2):.4f}")
    assert max(e1) < 0.05 and max(e2) < 0.01, "không khớp số có sẵn"
    print("khop dat")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        run(a[1], "--dev" in a, int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    elif a[0] == "khop":
        khop(a[1], int(a[2]) if len(a) > 2 else 20)
