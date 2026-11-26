"""Tái hiện lỗi vaudit.features.chunk_frames (đề xuất lanes/DE_XUAT/2511_P1_chunk-frames.md). Không cần mô hình, không đọc dữ liệu.

fn giả trả 1 cho mọi khung (số khung như wav2vec2: floor((n − 400) / 320) + 1) → hàng nào của đầu ra còn 0 là hàng không được gán.
  .venv/Scripts/python.exe experts/P1/coverage/tai_hien_chunk_frames.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from vaudit.features import SR, chunk_frames  # noqa: E402


def gaps(sec: float) -> tuple:
    y = np.zeros(int(sec * SR), np.float32)
    out, dt = chunk_frames(y, lambda seg: np.ones(((len(seg) - 400) // 320 + 1, 1), np.float32))
    z = np.where(out[:, 0] == 0)[0]
    return len(z), len(out), sorted({int(t // 1) for t in (z + 0.5) * dt})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    n, T, where = gaps(80.0)
    assert n > 0, "không tái hiện được"
    print(f"tệp 80 s: {n}/{T} hàng không được gán, nằm ở giây {where[0]}–{where[-1]}")
    hit = [s for s in range(21, 601) if gaps(float(s))[0] > 0]
    print(f"độ dài 21–600 s (bước 1 s): {len(hit)}/580 có hàng không được gán; ngắn nhất {hit[0]} s")
    print("số hàng trống theo độ dài:", {s: gaps(float(s))[0] for s in (30, 56, 57, 60, 80, 120, 170, 300, 600)})
