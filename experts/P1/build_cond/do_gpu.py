# -*- coding: utf-8 -*-
"""Đo 3 lượt (do_dieu_kien · do_tram · do_vang) trên GPU trong MỘT vòng / tệp, ghi đúng 3 tệp jsonl cùng định dạng như chạy CPU.
Không sửa 3 script gốc: nạp mô hình lên cuda vào _M của chúng (cách của scratchpad kiem_gpu.py, đã khớp CPU khi tắt TF32).
DFN của do_tram / do_vang dùng chung một lần lọc (clean_dfn) — giống hệt hai lần chạy riêng vì DFN tất định.

  D:\\venv_gpu\\Scripts\\python.exe -I experts/P1/build_cond/do_gpu.py run <bo> [--limit N]
  D:\\venv_gpu\\Scripts\\python.exe -I experts/P1/build_cond/do_gpu.py khop <bo_da_do_cpu> [N]   (so GPU với số CPU có sẵn)"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import do_dieu_kien as dk  # noqa: E402
import do_tram as dt  # noqa: E402
import do_vang as dv  # noqa: E402

SR = 16000
OUTS = ("dieu_kien.jsonl", "dieu_kien_tram.jsonl", "dieu_kien_vang.jsonl")


def nap() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("không thấy cuda")
    torch.backends.cuda.matmul.allow_tf32 = False             # P1_QUY_TRINH §6: TF32 làm lệch CPU tới 0,57 dB
    torch.backends.cudnn.allow_tf32 = False
    from df.enhance import init_df
    from pyannote.audio import Inference, Model
    from pyannote.audio.core.model import Introspection
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from torch.torch_version import TorchVersion
    torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution, Introspection])
    dk._M["bh"] = Inference(Model.from_pretrained("pyannote/brouhaha"), device=torch.device("cuda"))
    dk._M["dfn"] = init_df(log_level="ERROR")[:2]
    dt._M["dfn"] = dk._M["dfn"]
    dev = next(dk._M["dfn"][0].parameters()).device
    if dev.type != "cuda":
        raise SystemExit(f"DFN đang ở {dev}, cần cuda")


def mot(y32: np.ndarray) -> tuple:
    t = time.time()
    a = dk.measure(y32)
    y = y32.astype(np.float64)
    c = dt.clean_dfn(y)
    return {**a, "s": round(time.time() - t, 1)}, dt.nen_bien_the(y, c), dv.do_mot(y, c)


def run(bo: str, limit: int | None = None) -> None:
    d = os.path.join(dk.ROOT, "data_labels", "v6", bo)
    cases = json.load(open(os.path.join(d, "cases.json"), encoding="utf-8"))
    cases = cases if isinstance(cases, list) else cases["cases"]
    os.makedirs(os.path.join(d, "_kiem"), exist_ok=True)
    paths = [os.path.join(d, "_kiem", o) for o in OUTS]
    done = [{json.loads(l)["file"] for l in open(p, encoding="utf-8")} if os.path.exists(p) else set() for p in paths]
    todo = [c for c in cases if any(c["file"] not in s for s in done)][:limit]
    nap()
    print(f"{bo}: {len(todo)} tệp cần đo", flush=True)
    t0 = time.time()
    fh = [open(p, "a", encoding="utf-8") for p in paths]
    try:
        for i, c in enumerate(todo):
            y, sr = sf.read(os.path.join(d, c["file"]), dtype="float32")
            if sr != SR or y.ndim != 1:
                raise SystemExit(f"{c['file']}: cần mono 16 kHz, gặp sr={sr} shape={y.shape}")
            a, b, v = mot(y)
            rows = ({"file": c["file"], "dur_s": round(len(y) / SR, 2), **a}, {"file": c["file"], **b}, {"file": c["file"], **v})
            for f, s, r in zip(fh, done, rows):
                if c["file"] not in s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        for f in fh:
            f.close()
    print("xong", d, flush=True)


def khop(bo: str, n: int = 30) -> None:
    """So số GPU với số CPU đã có của bộ bo trên n tệp đầu (lệch > 0,1 thì liệt kê)."""
    d = os.path.join(dk.ROOT, "data_labels", "v6", bo, "_kiem")
    cpu = [{json.loads(l)["file"]: json.loads(l) for l in open(os.path.join(d, o), encoding="utf-8")} for o in OUTS]
    files = sorted(cpu[0])[:n]
    nap()
    lech, tong = [], 0
    for f in files:
        y, _ = sf.read(os.path.join(dk.ROOT, "data_labels", "v6", bo, f), dtype="float32")
        for got, ref in zip(mot(y), (cpu[0][f], cpu[1][f], cpu[2][f])):
            for k, v in got.items():
                if k in ("s", "file") or not isinstance(v, (int, float)) or not isinstance(ref.get(k), (int, float)):
                    continue
                tong += 1
                if abs(v - ref[k]) > 0.1:
                    lech.append((f, k, v, ref[k]))
    print(f"lệch > 0,1: {len(lech)}/{tong}", lech[:12])


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "run":
        run(a[1], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    elif a[0] == "khop":
        khop(a[1], int(a[2]) if len(a) > 2 else 30)
