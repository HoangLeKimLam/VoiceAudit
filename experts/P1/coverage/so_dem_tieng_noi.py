"""Cụm 1 (NHAT_KY 2.19): so 5 cách đếm giây tiếng nói cho dòng «quá ít tiếng nói» (≥ 2 s và ≥ 30 %), KHÔNG sửa K1.

C0 = K1 hiện tại (CTC ±0,1 s, chuẩn hoá theo khúc 16 s) · C1 = C0 ∧ khung > −60 dBFS · C2 = ép trống khúc ASR RMS < −55 dBFS
C3 = pyannote seg 3.0 (lớp > 0) nới ±0,1 s. Mọi ứng viên giữ bước hợp AST của K1 khi dưới ngưỡng.
C4 = C0 nhưng hàng chunk_frames không được gán (logits toàn 0, argmax 0 ≠ blank) coi là blank: sửa gốc trong k1.py (NHAT_KY 2.19 đính chính).

  .venv/Scripts/python.exe experts/P1/coverage/so_dem_tieng_noi.py selftest | run | tom
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from vaudit import k1  # noqa: E402

SR, WIN = 16000, 0.03
V6 = os.path.join(ROOT, "data_labels", "v6")
AX = os.path.join(V6, "k1axes", "p1_v1")
NAT = os.path.join(V6, "p1_tunhien_v1")
RES_AX = os.path.join(ROOT, ".tmp_v6", "results", "k1_axes_p1_v1.json")
OUT = os.path.join(ROOT, ".tmp_v6", "results", "so_dem_tieng_noi_v2.jsonl")     # v1 (12 tệp chạy thử, không có C4) giữ nguyên
TOM = os.path.join(AX, "SO_DEM_TIENG_NOI.md")
CANDS = ("C0", "C1", "C2", "C3", "C4")
GATE_FRAME, GATE_CHUNK, DIL = -60.0, -55.0, 0.10
SYN_CASES = ("tinhieu.tiengnoi_", "dodai.", "loai.chi_tieng_on", "loai.am_thuan")
PAD_AFTER = ("tinhieu.tiengnoi_15pct", "tinhieu.tiengnoi_50pct", "tinhieu.tiengnoi_20pct", "tinhieu.tiengnoi_40pct")
PAD_MID = ("tinhieu.tiengnoi_1.4s", "tinhieu.tiengnoi_3.0s")


def fold(base: str) -> int:
    return zlib.crc32(base.encode()) % 5


# ------------------------------------------------------------------ mặt nạ
def _dilate(tok_t: np.ndarray, n: int) -> np.ndarray:
    """Như k1.asr_speech_mask: khung 30 ms cách thời điểm có token ≤ 0,1 s."""
    tc = (np.arange(n) + 0.5) * WIN
    if not len(tok_t):
        return np.zeros(n, bool)
    j = np.searchsorted(tok_t, tc)
    d = np.minimum(np.abs(tc - tok_t[np.clip(j, 0, len(tok_t) - 1)]), np.abs(tc - tok_t[np.clip(j - 1, 0, len(tok_t) - 1)]))
    return d <= DIL


def ctc_masks(y: np.ndarray) -> tuple:
    """(C0, C2): một lượt mô hình, cache logits theo khúc; C2 ép trống khúc có RMS < GATE_CHUNK."""
    import torch
    from vaudit.asr import _load
    from vaudit.features import LONG_S, chunk_frames
    proc, model = _load()
    blank = proc.tokenizer.pad_token_id
    cache = []

    def run(seg):
        x = proc(seg.astype(np.float32), sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            lg = model(x).logits[0].numpy()
        cache.append((10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12), lg))
        return lg
    it = iter(())

    def run_gated(seg):
        r, lg = next(it)
        if r < GATE_CHUNK:
            lg = lg.copy()
            lg[:, blank] = lg.max() + 1e3
        return lg
    n = len(y) // int(WIN * SR)
    out = []
    for fn in (run, run_gated):
        if fn is run_gated:
            it = iter(cache)
        if len(y) / SR > LONG_S:
            lg, dt = chunk_frames(y, fn)
        else:
            lg = fn(y)
            dt = len(y) / SR / lg.shape[0]
        tok = lg.argmax(1) != blank
        out.append(_dilate((np.where(tok)[0] + 0.5) * dt, n))
        if fn is run:                                    # C4: hàng chunk_frames không gán (logits toàn 0) = blank
            out.append(_dilate((np.where(tok & lg.any(1))[0] + 0.5) * dt, n))
    return out[0], out[2], out[1], min(r for r, _ in cache)


def seg_mask(y: np.ndarray) -> np.ndarray:
    """pyannote seg 3.0 (cửa sổ 10 s như k1x.overlap_share), lớp > 0 = có lời; nới ±0,1 s ở lưới 30 ms."""
    from vaudit import k1x
    sess = k1x._seg()
    W = 10 * SR
    x = np.pad(y.astype(np.float32), (0, (-len(y)) % W)).reshape(-1, 1, W)
    cls = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: x[i:i + 4]})[0].argmax(-1).ravel() for i in range(0, len(x), 4)])
    dt = 10.0 / 589
    return _dilate((np.where(cls > 0)[0] + 0.5) * dt, len(y) // int(WIN * SR))


def measure(y: np.ndarray) -> dict:
    db = k1.frame_db(y)
    sp_e, _ = k1.speech_mask(db)
    m0, m2, m4, rmin = ctc_masks(y)
    k = min(len(db), len(m0))
    m1 = m0.copy()
    m1[:k] &= db[:k] > GATE_FRAME
    masks = {"C0": m0, "C1": m1, "C2": m2, "C3": seg_mask(y), "C4": m4}
    dur = len(y) / SR
    ast = None
    out = {"dur": round(dur, 2), "nang_luong_s": round(float(sp_e.sum() * WIN), 2), "rms_khuc_min": round(float(rmin), 1)}
    for c, m in masks.items():
        s = float(m.sum() * WIN)
        if s < 2 or s < 0.3 * dur:                       # bước hợp AST như k1.analyse_channel
            kk = min(len(sp_e), len(m))
            ast = k1.ast_speech_frames(y, kk) if ast is None else ast
            s = max(s, float((m[:kk] | (sp_e[:kk] & (ast[:kk] > k1.AST_SPEECH_THR))).sum() * WIN))
        out[c] = round(s, 2)
        out[c + "_mask"] = m
    return out


# ------------------------------------------------------------------ vùng đệm (nhãn từ cách dựng, kiểm bằng so khớp mẫu)
def _room(n: int, seed: int) -> np.ndarray:
    return np.round(np.random.default_rng(seed).normal(0, 10 ** (-70 / 20), n) * 32768) / 32768


def pad_region(case: str, y: np.ndarray, base: str) -> tuple | None:
    """Trả (đầu lời, cuối lời) theo mẫu; None nếu ca không có vùng đệm. Lỗi nếu so khớp không đúng cách dựng."""
    if case in PAD_AFTER:
        g, _ = sf.read(os.path.join(AX, f"_goc__{base}.wav"), dtype="float64")
        assert np.array_equal(y[:len(g)], g.astype(np.float32)), f"{case} {base}: phần đầu khác tệp gốc"
        assert len(y) == len(g) or np.max(np.abs(y[len(g):])) < 10 ** (-50 / 20), f"{case} {base}: vùng đệm không phải tiếng phòng"
        return 0, len(g)
    if case in PAD_MID:
        r = _room(len(y), 7).astype(np.float32)
        L = int(np.argmax(np.abs(y - r) > 1.5 / 32768))        # ≥ n_pad // 2 (mẫu lời đầu có thể trùng tiếng phòng)
        for n_pad in range(2 * L + 1, max(0, 2 * L - 8), -1):
            tail = _room(n_pad, 7)[n_pad // 2:].astype(np.float32)
            if np.allclose(y[len(y) - len(tail):], tail, atol=1.5 / 32768):
                return n_pad // 2, len(y) - len(tail)
        raise AssertionError(f"{case} {base}: không khớp vùng đệm hai bên")
    return None


# ------------------------------------------------------------------ chạy
def jobs() -> list:
    rows = json.load(open(RES_AX, encoding="utf-8"))
    J = [("ax", r["file"], r["ca"], r["base"]) for r in rows if r["ca"].startswith(SYN_CASES) and r["file"].endswith(".wav")]
    J += [("goc", f"_goc__{b}.wav", "_goc", b) for b in sorted({r["base"] for r in rows})]
    c = json.load(open(os.path.join(NAT, "cases.json"), encoding="utf-8"))
    J += [("nat", x["file"], x["ca"], x["base"]) for x in (c if isinstance(c, list) else c["cases"]) if fold(x["base"]) < 3]
    return J


def run() -> None:
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(l)["file"] for l in open(OUT, encoding="utf-8")}
    J = [j for j in jobs() if j[1] not in done]
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as fo:
        for i, (bo, f, ca, base) in enumerate(J):
            y, sr = sf.read(os.path.join(AX if bo != "nat" else NAT, f), dtype="float32")
            if y.ndim > 1 or sr != SR:
                raise ValueError(f"{f}: cần mono 16 kHz, có {y.shape} {sr}")
            m = measure(y)
            rec = {"bo": bo, "file": f, "ca": ca, "base": base, **{k: v for k, v in m.items() if not k.endswith("_mask")}}
            reg = pad_region(ca, y, base) if bo == "ax" else None
            if reg:
                a, b = reg[0] // int(WIN * SR), reg[1] // int(WIN * SR)
                for c in CANDS:
                    mk = m[c + "_mask"]
                    rec[c + "_dem"] = round(float((mk[:a].sum() + mk[b + 1:].sum()) * WIN), 2)   # khung biên tính là lời
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fo.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(J)} ({time.time() - t0:.0f} s)", flush=True)
    print("xong", OUT)


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def kn(k: int, n: int) -> str:
    a, b = wilson(k, n)
    return f"{k}/{n} = {100 * k / max(n, 1):.1f} % ({100 * a:.1f}–{100 * b:.1f})"


def qua_it(s: float, dur: float) -> bool:
    return s < 2 or s < 0.3 * dur


def tom() -> None:
    R = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    ax = {r["file"]: r for r in json.load(open(RES_AX, encoding="utf-8"))}
    L = ["# So 5 cách đếm tiếng nói (NHAT_KY 2.19, luật chốt trước)", "",
         "C0 = K1 hiện tại · C1 = ∧ khung > −60 dBFS · C2 = ép trống khúc RMS < −55 dBFS · C3 = pyannote seg ±0,1 s · C4 = hàng ghép khúc không gán = blank", ""]
    pad = [r for r in R if "C0_dem" in r]
    L += ["## (a) Giây «tiếng nói» trong vùng đệm (nhãn từ cách dựng)", "", "| ứng viên | tệp > 0,5 s | tổng giây | lớn nhất |", "|---|---|---|---|"]
    for c in CANDS:
        v = [r[c + "_dem"] for r in pad]
        L.append(f"| {c} | {kn(sum(x > 0.5 for x in v), len(v))} | {sum(v):.1f} | {max(v):.1f} |")
    nat = [r for r in R if r["bo"] == "nat"]
    L += ["", f"## (b) Tệp thật dev p1_tunhien_v1 ({len(nat)} tệp): bị coi «quá ít tiếng nói»", "",
          "| ứng viên | tất cả | vi | en | năng lượng cũng < ngưỡng (xác nhận độc lập) |", "|---|---|---|---|---|"]
    for c in CANDS + ("nang_luong_s",):
        b = [r for r in nat if qua_it(r[c], r["dur"])]
        conf = sum(qua_it(r["nang_luong_s"], r["dur"]) for r in b)
        lg = {g: (sum(r["file"].startswith(g) for r in b), sum(r["file"].startswith(g) for r in nat)) for g in ("vi/", "en/")}
        L.append(f"| {c.replace('nang_luong_s', 'VAD năng lượng')} | {kn(len(b), len(nat))} | {lg['vi/'][0]}/{lg['vi/'][1]} | "
                 f"{lg['en/'][0]}/{lg['en/'][1]} | {conf}/{len(b)} |")
    L += ["", "## (c) Ca p1_v1: quyết định đúng theo exp", "", "| nhóm ca | " + " | ".join(CANDS) + " | C0 khớp K1 thật |",
          "|---|" + "---|" * (len(CANDS) + 1)]
    groups = {}
    for r in (r for r in R if r["bo"] in ("ax", "goc")):
        row = ax.get(r["file"])
        dec = row["exp"]["dec"] if row else ["accept"]
        other = bool(row) and any("quá ít tiếng nói" not in x for x in row["reasons"])
        g = groups.setdefault(r["ca"] if r["ca"].startswith("tinhieu.") else r["ca"].split(".")[0] + ("" if r["ca"] == "_goc" else ".*"), [])
        res = [("04" in dec) if (other or qua_it(r[c], r["dur"])) else ("accept" in dec) for c in CANDS]
        real = None if not row else ((row["decision"] == "04") == (other or qua_it(r["C0"], r["dur"])))
        g.append((res, real))
    for name, v in sorted(groups.items()):
        n = len(v)
        cells = [f"{sum(x[0][i] for x in v)}/{n}" for i in range(len(CANDS))]
        rr = [x[1] for x in v if x[1] is not None]
        L.append(f"| {name} | " + " | ".join(cells) + f" | {sum(rr)}/{len(rr)} |")
    tot = [sum(x[0][i] for v in groups.values() for x in v) for i in range(len(CANDS))]
    nall = sum(len(v) for v in groups.values())
    L.append("| **tổng** | " + " | ".join(kn(t, nall) for t in tot) + " | |")
    open(TOM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    assert fold("x") == zlib.crc32(b"x") % 5
    t = _dilate(np.array([1.0]), 100)
    assert t.sum() == 7 and t[30] and t[36] and not t[29] and not t[37], t.sum()   # tâm khung 0,915–1,095 s
    rows = json.load(open(RES_AX, encoding="utf-8"))
    for ca in PAD_AFTER + PAD_MID:                             # nhãn vùng đệm khớp cách dựng trên mọi tệp gốc
        for b in sorted({r["base"] for r in rows}):
            y, _ = sf.read(os.path.join(AX, f"{ca}__{b}.wav"), dtype="float32")
            a, e = pad_region(ca, y, b)
            assert 0 <= a < e <= len(y)
    y, _ = sf.read(os.path.join(AX, "tinhieu.tiengnoi_15pct__p1k1vi_00.wav"), dtype="float32")
    m0, _, m4, _ = ctc_masks(y)                                # mô phỏng C0 == mã K1 thật
    assert (m4 <= m0).all() and m4.sum() < m0.sum(), "C4 phải bỏ bớt (chỉ bỏ) khung của C0 trên tệp 80 s"
    assert np.array_equal(m0, k1.asr_speech_mask(y)), "C0 lệch k1.asr_speech_mask"
    y, _ = sf.read(os.path.join(AX, "_goc__p1k1en_03.wav"), dtype="float32")
    assert np.array_equal(ctc_masks(y)[0], k1.asr_speech_mask(y)), "C0 lệch (tệp ngắn)"
    print("so_dem_tieng_noi selftest dat")


if __name__ == "__main__":
    {"run": run, "selftest": selftest, "tom": tom}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
