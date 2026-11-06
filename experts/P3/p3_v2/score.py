# -*- coding: utf-8 -*-
"""SCORE (P3 v2) — cham MOT tep theo hop dong H3 bang mo hinh da hoc tren TOAN BO du lieu (fit_final.py + train_seq.py --final -> weights/).

score(wav_path, ctx) -> {"den": (T,) f32, "voi": (T,) f32, "nen": float, "giong": float}
ctx (he thong P0 dua vao, cung luoi khung 20 ms):
  "z"      : khung XLS-R PCA-64 (T, 64)          · "pooled": vector gop XLS-R (6144,)
  "fuse1"  : diem khung nen fuse1 (T, 6)          · "h2": diem dau ca tep nen h2 (3,)  (B2 / pha tron giong en dung)
  "lang"   : "vi" / "en"                          · "ff": dac trung vat ly khung (tuy chon; khong co thi tinh tu wav)
Thieu trong so / ctx -> RuntimeError (khong tra so gia).
CLI: python experts/P3/p3_v2/score.py <wav> --id <id du lieu> [--lang vi]   (lay z / pooled / fuse1 / h2 tu cache du lieu)
     python experts/P3/p3_v2/score.py selftest"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

W = os.path.join(HERE, "weights")
_M: dict = {}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def sg(x):
    return 1 / (1 + np.exp(-np.asarray(x, np.float64)))


def _models() -> dict:
    if _M:
        return _M
    need = ["final.json", "den.txt", "voi.txt", "nenA.txt", "giongA.txt", "headsB.npz", "headsB2.npz", "seq.pt", "seq_norm.npz"]
    miss = [n for n in need if not os.path.exists(os.path.join(W, n))]
    if miss:
        raise RuntimeError(f"chua co trong so {miss}: chay fit_final.py va train_seq.py --final")
    import lightgbm as lgb
    import torch
    from train_seq import build_model
    _M["cfg"] = json.load(open(os.path.join(W, "final.json"), encoding="utf-8"))
    for k in ("den", "voi", "nenA", "giongA"):
        _M[k] = lgb.Booster(model_file=os.path.join(W, f"{k}.txt"))
    for k in ("headsB", "headsB2", "seq_norm"):
        z = np.load(os.path.join(W, f"{k}.npz"))
        _M[k] = {n: z[n] for n in z.files}
    net = build_model(int(_M["seq_norm"]["d_in"]))
    net.load_state_dict(torch.load(os.path.join(W, "seq.pt"), map_location="cpu"))
    net.eval()
    _M["seq"] = net
    return _M


def frame_raw(ff: np.ndarray, z: np.ndarray, fuse1: np.ndarray) -> tuple:
    """Xac suat khung tho (LightGBM, TCN) cho (den, voi): ((T, 2), (T, 2))."""
    import ctxfeat
    import torch
    from train_frames import keep_cols, stack_feats
    m = _models()
    X = ctxfeat.ctx_feats(ff, z)
    T = len(X)
    Xg = np.hstack([X[:, keep_cols(True, True)], stack_feats(fuse1, T)])
    g = np.stack([m["den"].predict(Xg), m["voi"].predict(Xg)], 1)
    nz = m["seq_norm"]
    xt = ((torch.tensor(X.astype(np.float16).astype(np.float32)) - torch.tensor(nz["mu"])) / torch.tensor(nz["sd"])).clamp(-8, 8).T[None]
    with torch.no_grad():
        fr, _ = m["seq"](xt, torch.ones(1, T))
    return g, torch.sigmoid(fr[0]).T.numpy()


def mix_post(g: np.ndarray, q: np.ndarray, lang: str) -> tuple:
    """Gop logit LightGBM / TCN theo FRAME_MIX roi hau xu ly theo ngon ngu (final.json)."""
    from post import post2, post_raw_mx
    c = _models()["cfg"]
    out = []
    for k, ch in enumerate(("den", "voi")):
        wgt = c["frame_mix"][ch][lang]
        s = sg(wgt * lg(g[:, k]) + (1 - wgt) * lg(q[:, k])).astype(np.float32)
        p = c["post"][ch][lang]
        out.append(post2(s, p[1], p[2], p[3], 0.5, 3) if p[0] == "post2" else post_raw_mx(s, p[1], p[2]))
    return out[0], out[1]


def heads(ff: np.ndarray, pause: np.ndarray, z: np.ndarray, pooled: np.ndarray, fuse1: np.ndarray, h2: np.ndarray, lang: str) -> tuple:
    from train_heads import file_stats
    from train_heads_b2 import zf_stats
    m = _models()
    xa = np.r_[pause, file_stats(ff)][None, :]
    A = [float(m["nenA"].predict(xa)[0]), float(m["giongA"].predict(xa)[0])]
    B_ = m["headsB"]
    xp = np.nan_to_num(np.asarray(pooled, np.float64))
    xb = (xp - B_["mu"]) / B_["sd"]
    B = [float(sg(xb @ B_[f"w{k}"] + B_[f"b{k}"])) for k in range(2)]
    if lang == "vi":
        return B[0], float(sg(0.7 * lg(B[1]) + 0.3 * lg(A[1])))
    C = m["headsB2"]
    zf = np.nan_to_num(zf_stats(z, fuse1, h2))
    nen = float(sg(((np.r_[xp, zf] - C["n_mu"]) / C["n_sd"]) @ C["n_w"] + C["n_b"]))
    xg = np.r_[(xp - C["p_mu"]) / C["p_sd"], np.clip((np.r_[zf, pause[:640]] - C["r_c"]) / C["r_s"], -8, 8) * 2.5]
    dsp = float(sg(xg @ C["d_w"] + C["d_b"]))
    return nen, 0.15 * float(h2[1]) + 0.85 * dsp


def score(wav_path: str, ctx: dict | None = None) -> dict:
    from feats import frame_feats, read_wav
    from feats_pause import pause_feats
    ctx = ctx or {}
    for k in ("z", "pooled", "fuse1", "h2"):
        if ctx.get(k) is None:
            raise RuntimeError(f"ctx thieu '{k}' (P0 dua vao tu XLS-R / fuse1 / h2)")
    z = np.asarray(ctx["z"], np.float32)
    fuse1 = np.asarray(ctx["fuse1"], np.float32)
    x = read_wav(wav_path)
    ff = ctx.get("ff")
    if ff is None:
        ff = frame_feats(x, len(z))
    lang = (ctx.get("lang") or "vi")[:2]
    g, q = frame_raw(ff, z, fuse1)
    den, voi = mix_post(g, q, lang)
    nen, giong = heads(ff, pause_feats(x, z), z, np.asarray(ctx["pooled"]), fuse1, np.asarray(ctx["h2"], np.float32), lang)
    return {"den": den.astype(np.float32), "voi": voi.astype(np.float32), "nen": nen, "giong": giong}


def selftest() -> None:
    assert abs(float(sg(lg(0.3))) - 0.3) < 1e-9
    print("score selftest dat (ham thuan; chay that can weights/)")


if __name__ == "__main__":
    import argparse
    import pickle
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        sys.exit(0)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav")
    ap.add_argument("--id", required=True)
    ap.add_argument("--lang", default="vi")
    a = ap.parse_args()
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    V4 = os.path.join(BASE, "data_labels", "realistic_v4")
    C = os.path.join(BASE, ".tmp_v6", "cache")
    f1 = pickle.load(open(os.path.join(C, "boundary_oof_fuse1.pkl"), "rb"))[a.id]
    h = pickle.load(open(os.path.join(C, "heads2_h2.pkl"), "rb"))
    r = score(a.wav, {"z": np.load(os.path.join(V4, "xlsr_frames", a.id + ".npy")), "pooled": np.load(os.path.join(V4, "xlsr_pooled", a.id + ".npy")),
                      "fuse1": f1, "h2": h["P"][list(h["ids"]).index(a.id)], "lang": a.lang})
    print({k: (round(float(v.max()), 4) if np.ndim(v) else round(v, 4)) for k, v in r.items()})
