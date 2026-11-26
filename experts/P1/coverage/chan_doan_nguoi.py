"""Chẩn đoán hàng «Số người nói» trên T1.x tag p1_v1_c4 (chỉ đọc; không sửa K1, không sửa dữ liệu).
Chạy lại k1.check (events=False) trên: 24 tệp nguoi.2_nguoi, 24 tệp nguoi.6_nguoi, 27 tệp 1 người bị cảnh báo thừa «nhiều người nói», 24 tệp gốc.
Bọc k1.count_speakers để lấy đúng (y, mặt nạ tiếng nói) K1 dùng, rồi ghi mỗi cửa sổ: nhúng resemblyzer, thời điểm, phần (người) theo cách ghép
(_concat_people: mỗi người 8 s nối tiếp). Kết quả: _chan_doan/nguoi_2611.npz (nhúng) và nguoi_2611.jsonl (tóm từng tệp).

  .venv/Scripts/python.exe experts/P1/coverage/chan_doan_nguoi.py run
  .venv/Scripts/python.exe experts/P1/coverage/chan_doan_nguoi.py tom
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
DIR = os.path.join(ROOT, "data_labels", "v6", "k1axes", "p1_v1")
RES = os.path.join(ROOT, ".tmp_v6", "results", "k1_axes_p1_v1_c4.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_chan_doan")
NPZ, JL = os.path.join(OUT, "nguoi_2611.npz"), os.path.join(OUT, "nguoi_2611.jsonl")
SR, W, H = 16000, int(1.6 * 16000), int(0.8 * 16000)


def ds_tep() -> list:
    rows = json.load(open(RES, encoding="utf-8"))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    out = [(r["file"], r["ca"], r["base"], 2 if "2_nguoi" in r["ca"] else 6) for r in rows if r["ca"] in ("nguoi.2_nguoi", "nguoi.6_nguoi")]
    out += [(r["file"], r["ca"], r["base"], 1) for r in rows
            if not r["ca"].startswith("nguoi.") and "nhiều người nói" in " ".join(map(str, r["loi"]))]
    bases = sorted({r["base"] for r in rows})
    out += [(f"_goc__{b}.wav", "_goc", b, 1) for b in bases]
    assert len(out) == 24 + 24 + 27 + 24, len(out)
    return out


def run() -> None:
    from scipy.cluster.hierarchy import fcluster, linkage
    from vaudit import k1
    if os.path.exists(JL):
        raise SystemExit(f"Đã có {JL}; không ghi đè")
    os.makedirs(OUT, exist_ok=True)
    goc, bat = k1.count_speakers, {}

    def boc(y, mask, win=0.03):
        bat["y"], bat["mask"], bat["win"] = y, mask, win
        return goc(y, mask, win)

    k1.count_speakers = boc
    enc = None
    arr, rec = {}, []
    ds = ds_tep()
    for i, (f, ca, base, k) in enumerate(ds):
        bat.clear()
        r = k1.check(os.path.join(DIR, f), events=False)
        n = (r["measures"].get("kenh") or [{}])[0].get("so_nguoi_noi")
        if "y" not in bat:
            rec.append({"file": f, "ca": ca, "base": base, "k": k, "n_k1": n, "loi": "count_speakers không được gọi", "decision": r["decision"]})
            continue
        enc = enc or getattr(boc, "enc", None) or goc.enc      # count_speakers ghi .enc lên tên toàn cục (= hàm bọc)
        y, mask, fn = bat["y"], bat["mask"], int(bat["win"] * SR)
        E, t, part = [], [], []
        for s in range(0, max(1, len(y) - W), H):
            fr = mask[s // fn:(s + W) // fn]
            if len(fr) and fr.mean() >= 0.6 and np.mean(np.abs(y[s:s + W]) < 3e-5) < k1.SPK_ZERO_SHARE:
                E.append(enc.embed_utterance(y[s:s + W].astype(np.float32)))
                t.append(s / SR)
                part.append(int((s + W / 2) // (8 * SR)) if k > 1 else 0)
        E, part = np.array(E), np.array(part)
        d = {"file": f, "ca": ca, "base": base, "k": k, "n_k1": n, "decision": r["decision"], "dur_s": round(len(y) / SR, 1),
             "n_cua_so": len(E), "cua_so_moi_phan": np.bincount(part, minlength=k).tolist() if len(E) else []}
        if len(E) >= 2:
            Z = linkage(E, method="average", metric="cosine")
            lab = fcluster(Z, t=0.35, criterion="distance")
            d["co_cum"] = sorted((int((lab == c).sum()) for c in set(lab)), reverse=True)
            d["cao_gop_cuoi"] = [round(float(h), 3) for h in Z[-6:, 2]]
            if k > 1 and len(set(part.tolist())) >= 2:            # có phần (người) không còn cửa sổ nào → chỉ so các phần còn lại
                M = np.array([E[part == p].mean(0) for p in range(k) if (part == p).sum()])
                M /= np.linalg.norm(M, axis=1, keepdims=True)
                D = 1 - M @ M.T
                d["kc_giua_nguoi_min"] = round(float(D[np.triu_indices(len(M), 1)].min()), 3)
                d["kc_giua_nguoi"] = [round(float(x), 3) for x in D[np.triu_indices(len(M), 1)]]
            En = E / np.linalg.norm(E, axis=1, keepdims=True)
            d["kc_trong_nguoi_tb"] = round(float(np.mean([np.mean(1 - En[part == p] @ En[part == p].T)
                                                           for p in range(k) if (part == p).sum() > 1])), 3)
        arr[f + "|E"], arr[f + "|part"], arr[f + "|t"] = E, part, np.array(t)
        rec.append(d)
        print(f"{i + 1}/{len(ds)} {f} k={k} K1={n} cửa sổ={len(E)} {d.get('co_cum')}", flush=True)
    np.savez_compressed(NPZ, **arr)
    with open(JL, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rec)


def tom() -> None:
    rec = [json.loads(x) for x in open(JL, encoding="utf-8")]
    for x in rec:
        print(x["ca"][:16].ljust(16), x["base"], "k", x["k"], "K1", x["n_k1"], "cs", x["n_cua_so"], x.get("cua_so_moi_phan"), "cụm", x.get("co_cum"),
              "giữa_min", x.get("kc_giua_nguoi_min"), "trong", x.get("kc_trong_nguoi_tb"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "tom": tom}[sys.argv[1]]()
