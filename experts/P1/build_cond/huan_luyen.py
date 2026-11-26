# -*- coding: utf-8 -*-
"""Huấn luyện + chọn mô hình đo ồn / nền / vang (luật chốt trước: NHAT_KY 2.20). Chỉ fold 0–2; KHÔNG đọc nhãn fold 3–4.

Đích: snr_a (dB; nền suy ra = 100 / (1 + 10^(snr/10)), quyết định < 10 dB → 04 như cham_v3) · rt60 (s; tệp vang + goc).
Ứng viên: {ridge, gbm, mlp} × {F1 số đo có sẵn, F2 = F1 + AST, F3 = F2 + wav2vec2 (PCA 64)}. OOF 3 fold theo base trên v4 dev.
Chọn: (a*) thấp nhất với bắt rõ ≤ 5 dB ≥ 80 %; hoà trong 0,5 điểm % → đơn giản hơn. Báo kèm: v3 dev (cùng thước C3v2), tệp thật dev (c)
với xác nhận độc lập WADA-SNR (không là đầu vào), RT60 (MAE, cảnh báo oan).

  .venv/Scripts/python.exe experts/P1/build_cond/huan_luyen.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cham_v3 as C  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, jl, pct  # noqa: E402
from cham_tram import cases_of  # noqa: E402

V4, V3 = "p1_dieukien_v4", "p1_dieukien_v3"
OUT = os.path.join(V6, V4, "_kiem", "HUAN_LUYEN_2511.md")
MODELS, FEATS = ("ridge", "gbm", "mlp"), ("F1", "F2", "F3")
SIMPLE = {f"{m}_{f}": (MODELS.index(m), FEATS.index(f)) for m in MODELS for f in FEATS}
SKIP = {"file", "s", "dur_s"}


# ----------------------------------------------------------------------------- dữ liệu
def so_do(bo: str) -> dict:
    """file → {tên: số} gộp 3 tệp đo (None giữ None)."""
    out = {}
    for n in ("dieu_kien.jsonl", "dieu_kien_tram.jsonl", "dieu_kien_vang.jsonl"):
        for r in jl(os.path.join(V6, bo, "_kiem", n)):
            out.setdefault(r["file"], {}).update({k: v for k, v in r.items() if k not in SKIP and (v is None or isinstance(v, (int, float)))})
    return out


def dac_trung(bo: str) -> dict:
    p = os.path.join(V6, bo, "_kiem", "dac_trung.jsonl")
    return {r["file"]: r for r in jl(p)} if os.path.exists(p) else {}


def dap_an(bo: str) -> dict:
    """file → {snr, is04, bien, rt60, base} CHỈ fold 0–2 (giống cham_v3.dap_an_v3, thêm vang / rt60)."""
    cs = cases_of(bo)
    prov = {p["file"].split(f"{bo}/", 1)[-1]: p for p in jl(os.path.join(V6, bo, "provenance.jsonl"))}
    out = {}
    for f, c in cs.items():
        if fold(c["base"]) >= 3:
            continue
        d, t = prov[f]["do_lai"], prov[f]["tham_so"]
        if c["ca"] == "goc" or c["ca"].startswith("vang."):
            s = t["san_a_db"]
        elif "snr_that_A_db" in d:
            s = d["snr_that_A_db"]
        else:
            s = 10 * np.log10((100 - d["nen_A_pct"]) / d["nen_A_pct"])
        rt = d.get("rt60_rir_schroeder_s") or (0.3 if c["ca"] == "goc" else None)
        out[f] = {"snr": float(s), "is04": c["mong_doi"] == "04", "bien": bool(c["ca_bien"]), "rt60": rt, "base": c["base"]}
    return out


def ma_tran(files: list, M: dict, A: dict, feat: str, cols: list | None = None) -> tuple:
    """(X, tên cột). F1 = số đo; F2 += AST; F3 += w2v (1 536 cột, PCA trong pipeline)."""
    if cols is None:
        keys = sorted({k for f in files for k in M[f]})
        ast = sorted({k for f in files for k in A[f]["ast"]}) if feat != "F1" else []
        cols = [f"m:{k}" for k in keys] + [f"a:{k}" for k in ast] + ([f"w:{i}" for i in range(1536)] if feat == "F3" else [])
    X = np.full((len(files), len(cols)), np.nan)
    for i, f in enumerate(files):
        for j, c in enumerate(cols):
            t, k = c.split(":", 1)
            v = M[f].get(k) if t == "m" else A[f]["ast"].get(k) if t == "a" else A[f]["w2v"][int(k)]
            X[i, j] = np.nan if v is None else v
    return X, cols


def mo_hinh(name: str, cols: list):
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    w = [i for i, c in enumerate(cols) if c.startswith("w:")]
    s = [i for i, c in enumerate(cols) if not c.startswith("w:")]
    pre = [("s", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), s)]
    if w:
        pre.append(("w", make_pipeline(StandardScaler(), PCA(64, random_state=0)), w))
    head = {"ridge": RidgeCV(alphas=np.logspace(-2, 3, 12)),
            "gbm": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, l2_regularization=1.0, random_state=0),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=800, early_stopping=True, random_state=0)}[name]
    return make_pipeline(ColumnTransformer(pre), head)


def oof(files: list, y: np.ndarray, X: np.ndarray, cols: list, name: str, base: dict) -> np.ndarray:
    p = np.full(len(files), np.nan)
    fo = np.array([fold(base[f]) for f in files])
    assert fo.max() <= 2, "oof chỉ nhận fold 0–2 (fold 3–4 là test)"
    for k in (0, 1, 2):
        tr, te = fo != k, fo == k
        p[te] = mo_hinh(name, cols).fit(X[tr], y[tr]).predict(X[te])
    return p


# ----------------------------------------------------------------------------- chấm
def bang_snr(res: dict, da: dict, share: dict) -> tuple:
    """Cùng công thức cham_v3.bang (đổi danh sách ứng viên)."""
    C.UV = tuple(res)
    cs = {f: {"ca": ""} for f in da}
    return C.bang(res, {f: (v["snr"], v["is04"], v["bien"]) for f, v in da.items()}, cs, share)


def chon(tom: dict, order: dict | None = None) -> str | None:
    order = order or SIMPLE
    ok = [u for u in tom if tom[u]["ro"][1] and tom[u]["ro"][0] / tom[u]["ro"][1] >= 0.8]
    if not ok:
        return None
    best = min(tom[u]["a_star"] for u in ok)
    return min((u for u in ok if tom[u]["a_star"] <= best + 0.005), key=lambda u: order[u])


V4B, OUT21 = "p1_dieukien_v4b", os.path.join(V6, "p1_dieukien_v4", "_kiem", "HUAN_LUYEN_2511b.md")
UV21 = {"mlp_F1": 0, "gbm_F2": 1, "mlp_F2": 2, "tb": 3}


def run21() -> None:
    """NHAT_KY 2.21: train trên v4 + v4b fold 0–2, OOF theo base; (a*) / bắt rõ CHỈ trên tệp v4 dev (cùng thước 2.20)."""
    da = {**dap_an(V4), **dap_an(V4B)}
    M, A = {**so_do(V4), **so_do(V4B)}, {**dac_trung(V4), **dac_trung(V4B)}
    tr = sorted(f for f in da if f in M and f in A)
    ev = [f for f in tr if "p1xv4b_" not in f]
    y = {f: float(np.clip(da[f]["snr"], -10, 40)) for f in tr}
    cn, MN = cases_of(NAT), C.gop(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2]
    share = C.share_that(MN, acc)
    res, mae = {}, {}
    for u in UV21:
        p = {}
        for k in (0, 1, 2):
            trk = [f for f in tr if fold(da[f]["base"]) != k]
            tek = [f for f in ev if fold(da[f]["base"]) == k]
            p.update(zip(tek, lam_du(u, trk, np.array([y[f] for f in trk]), M, A)(tek, M, A)))
        res[u] = {f: bool(p[f] < 10) for f in ev}
        mae[u] = float(np.mean([abs(p[f] - y[f]) for f in ev]))
        print(u, f"MAE {mae[u]:.2f} dB", flush=True)
    L = ["# Cải thiện mô hình SNR trên dev (NHAT_KY 2.21, luật chốt trước)", "",
         f"- Train: v4 + v4b fold 0–2 ({len(tr)} tệp, trong đó v4b {len(tr) - len(ev)}) · chấm OOF trên v4 dev ({len(ev)} tệp, cùng thước 2.20)", "",
         "## 1. SNR / nền (OOF; < 10 dB → 04)", ""]
    Lb, tom = bang_snr(res, {f: da[f] for f in ev}, share)
    L += Lb + ["", "| ứng viên | MAE dB |", "|---|---|"] + [f"| {u} | {mae[u]:.2f} |" for u in res]
    u = chon(tom, UV21)
    L += ["", f"- **chọn {u}**" + (f" · (a*) {100 * tom[u]['a_star']:.1f} % → **{'ĐỦ' if tom[u]['a_star'] <= 0.02 else 'CHƯA ĐỦ'}** (cần ≤ 2 %)" if u else "")]
    if u:
        L += bao_kem(u, lam_du(u, tr, np.array([y[f] for f in tr]), M, A), share, acc, MN)
    L += ["", "- Chưa mở test (fold 3–4 v3 / v4)."]
    open(OUT21, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def run() -> None:
    da4, M4, A4 = dap_an(V4), so_do(V4), dac_trung(V4)
    files = sorted(f for f in da4 if f in M4 and f in A4)
    base = {f: da4[f]["base"] for f in files}
    y = np.array([np.clip(da4[f]["snr"], -10, 40) for f in files])
    cn, MN = cases_of(NAT), C.gop(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2]
    share = C.share_that(MN, acc)
    L = ["# Huấn luyện mô hình đo ồn / nền / vang (NHAT_KY 2.20, luật chốt trước)", "",
         f"- v4 dev (fold 0–2): {len(files)} tệp có đủ số đo + đặc trưng · OOF 3 fold theo base",
         "- Phân bố thật (snr_nghi, tệp ≥ 10 dB): " + " · ".join(f"{b[0]}–{b[1]}: {100 * share[b]:.1f} %" for b in C.DAI), "",
         "## 1. SNR / nền (OOF v4 dev; quyết định < 10 dB → 04)", ""]
    pred, res, mae, cols_of = {}, {}, {}, {}
    for ft in FEATS:
        X, cols = ma_tran(files, M4, A4, ft)
        cols_of[ft] = cols
        for m in MODELS:
            u = f"{m}_{ft}"
            pred[u] = oof(files, y, X, cols, m, base)
            res[u] = dict(zip(files, (pred[u] < 10).tolist()))
            mae[u] = float(np.mean(np.abs(pred[u] - y)))
            print(u, f"MAE {mae[u]:.2f} dB", flush=True)
    Lb, tom = bang_snr(res, {f: da4[f] for f in files}, share)
    L += Lb + ["", "| ứng viên | MAE dB (OOF) |", "|---|---|"] + [f"| {u} | {mae[u]:.2f} |" for u in res]
    u = chon(tom)
    L += ["", f"- Đạt bắt rõ ≥ 80 %: {', '.join(x for x in tom if tom[x]['ro'][1] and tom[x]['ro'][0] / tom[x]['ro'][1] >= 0.8) or 'không có'}"
          f" → **chọn {u}**" + (f" · (a*) {100 * tom[u]['a_star']:.1f} % → **{'ĐỦ' if tom[u]['a_star'] <= 0.02 else 'CHƯA ĐỦ'}** (cần ≤ 2 %)" if u else "")]
    if u:
        L += bao_kem(u, lam_du(u, files, y, M4, A4), share, acc, MN)
    L += rt60(files, da4, M4, A4)
    L += ["", "- Chưa mở test (fold 3–4 v3 / v4)."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def lam_du(u: str, files: list, y, M: dict, A: dict):
    """Khớp ứng viên u trên files → hàm du(files2, M2, A2) → dB dự đoán. u = '<mô hình>_<F>' hoặc 'tb' (TB mlp_F1 + gbm_F2)."""
    if u == "tb":
        d1, d2 = lam_du("mlp_F1", files, y, M, A), lam_du("gbm_F2", files, y, M, A)
        return lambda f2, M2, A2: (d1(f2, M2, A2) + d2(f2, M2, A2)) / 2
    m, ft = u.split("_")
    X, cols = ma_tran(files, M, A, ft)
    model = mo_hinh(m, cols).fit(X, y)
    return lambda f2, M2, A2: model.predict(ma_tran(f2, M2, A2, ft, cols)[0])


def bao_kem(u: str, du, share, acc, MN) -> list:
    """Mô hình đã khớp (du) → v3 dev (so C3v2 cùng thước) + tệp thật dev (c) có xác nhận WADA."""
    L = ["", f"## 2. {u} khớp trên cả tập train → v3 dev (không dùng để chọn)", ""]
    cs3 = cases_of(V3)
    prov3 = {p["file"].split(f"{V3}/", 1)[-1]: p for p in jl(os.path.join(V6, V3, "provenance.jsonl"))}
    da3 = C.dap_an_v3(cs3, prov3)
    M3g, M3, A3 = C.gop(V3), so_do(V3), dac_trung(V3)
    f3 = [f for f in da3 if f in M3g and f in A3]
    p3 = du(f3, M3, A3)
    from cham_vang import lam
    from cham_tram import dap_an_syn
    c2, M2 = cases_of(C.SYN), C.gop(C.SYN)
    d2, _ = dap_an_syn(c2)
    coef_v2 = lam([f for f in d2 if f in M2], M2, d2)["coef"]["C3"]
    c3v2 = C.lam_chan(f3, M3g, {f: 0.0 for f in f3}, coef_v2)["chan"]["C3v2"]
    res3 = {u: dict(zip(f3, (p3 < 10).tolist())), "C3v2": {f: c3v2(M3g[f]) for f in f3}}
    C.UV = tuple(res3)
    L += C.bang(res3, da3, cs3, share)[0] + ["", "Theo loại nền (SNR A thật ≥ 10 dB):", ""] \
        + C.theo_loai({**res3, "C3": res3["C3v2"]}, da3, cs3)       # theo_loai lấy danh sách loại từ res["C3"]
    AN = dac_trung(NAT)
    MNr = so_do(NAT)
    fa = [f for f in acc if f in AN and f in MNr]
    pa = du(fa, MNr, AN)
    b = [f for f, p in zip(fa, pa) if p < 10]
    xn = sum(AN[f]["wada_db"] < 10 for f in b)
    b2 = [f for f in fa if c3v2(MN[f])]
    L += ["", "## 3. Tệp thật p1_tunhien_v1 (dev, accept)", "", "| bộ đo | (c) bị chặn | vi | en | xác nhận độc lập WADA < 10 dB |", "|---|---|---|---|---|",
          f"| {u} | {pct(len(b), len(fa))} | {sum(f[:2] == 'vi' for f in b)} | {sum(f[:2] == 'en' for f in b)} | {xn}/{len(b)} |",
          f"| C3v2 | {pct(len(b2), len(fa))} | {sum(f[:2] == 'vi' for f in b2)} | {sum(f[:2] == 'en' for f in b2)} | "
          f"{sum(AN[f]['wada_db'] < 10 for f in b2)}/{len(b2)} |"]
    return L


def rt60(files: list, da4: dict, M4: dict, A4: dict) -> list:
    """Đầu RT60: OOF trên tệp vang + goc; MAE, cảnh báo oan (thật ≤ 1,0 s mà đoán > 1,0) và bỏ sót (thật > 1,0 mà đoán ≤ 1,0)."""
    fr = [f for f in files if da4[f]["rt60"] is not None]
    y = np.array([da4[f]["rt60"] for f in fr])
    base = {f: da4[f]["base"] for f in fr}
    L = ["", "## 4. RT60 (OOF v4 dev: tệp vang + goc)", "", "| ứng viên | MAE s | cảnh báo oan (thật ≤ 0,9 s) | bỏ sót (thật ≥ 1,1 s) |", "|---|---|---|---|"]
    for ft in FEATS:
        X, cols = ma_tran(fr, M4, A4, ft)
        for m in MODELS:
            p = oof(fr, y, X, cols, m, base)
            lo, hi = y <= 0.9, y >= 1.1
            L.append(f"| {m}_{ft} | {np.mean(np.abs(p - y)):.3f} | {pct(int(np.sum(p[lo] > 1.0)), int(lo.sum()))} | "
                     f"{pct(int(np.sum(p[hi] <= 1.0)), int(hi.sum()))} |")
    return L


def selftest() -> None:
    rng = np.random.default_rng(0)
    n = 300
    files = [f"snr/x_vi_{i}.wav" for i in range(n)]
    s = rng.uniform(-5, 30, n)
    M = {f: {"bh": v + rng.normal(0, 1), "c50": 30.0, "x": None if i % 7 == 0 else 1.0} for i, (f, v) in enumerate(zip(files, s))}
    A = {f: {"ast": {"music_max": 0.1}, "w2v": [0.0] * 1536} for f in files}
    dev_b = [b for b in (f"b{i}" for i in range(200)) if fold(b) <= 2][:30]
    base = {f: dev_b[i % 30] for i, f in enumerate(files)}
    X, cols = ma_tran(files, M, A, "F2")
    assert X.shape == (n, 4) and np.isnan(X[0, 2]), (X.shape, cols)
    p = oof(files, s, X, cols, "ridge", base)
    assert np.mean(np.abs(p - s)) < 1.5 and not np.isnan(p).any()
    tom = {"ridge_F1": {"ro": (9, 10), "a_star": 0.030}, "gbm_F1": {"ro": (9, 10), "a_star": 0.026}, "mlp_F3": {"ro": (5, 10), "a_star": 0.0}}
    assert chon(tom) == "ridge_F1"                       # gbm tốt hơn 0,4 điểm % (< 0,5) → lấy ridge đơn giản hơn; mlp không đạt bắt rõ
    print("huan_luyen selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "run21": run21, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
