"""NHAT_KY 2.22 (luật chốt trước): ứng viên bộ ước SNR mạnh hơn. CHỈ fold 0–2.
  mlp_F1 (mốc) · S1 = F1 + Brouhaha theo khung (MLP sklearn như mlp_F1) · S2 = w2v 13 lớp (đầu torch: softmax theo lớp → MLP)
  · S3 = S2 + F1 + S1 (cùng đầu torch). Train v4 + v4b fold 0–2; OOF 3 fold theo base, chấm CHỈ tệp v4 dev (cùng thước 2.20 / 2.21).
Chọn: (a*) thấp nhất với bắt rõ ≥ 80 %; hoà 0,5 điểm → đơn giản hơn (mlp_F1 < S1 < S2 < S3).
Cổng khả thi: σ sai số vùng 8–16 dB của ứng viên chọn > 2,0 dB → dừng hướng mô hình.
Thước thứ hai: bỏ base trong p1_dieukien_v4/_kiem/base_goc_on.json (bản gốc ồn theo tai Huấn), nếu đã có.

  .venv/Scripts/python.exe experts/P1/build_cond/huan_luyen2.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cham_v3 as C  # noqa: E402
import huan_luyen as H  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, jl, pct  # noqa: E402
from cham_tram import cases_of  # noqa: E402
from dac_trung2 import ten_npy  # noqa: E402

UV = {"mlp_F1": 0, "S1": 1, "S2": 2, "S3": 3}
OUT = os.path.join(V6, H.V4, "_kiem", "HUAN_LUYEN_2511c.md")
BO_ON = os.path.join(V6, H.V4, "_kiem", "base_goc_on.json")
DU = os.path.join(V6, H.V4, "_kiem", "du_doan_oof_2511c.jsonl")
SIGMA_MAX = 2.0


# ----------------------------------------------------------------------------- dữ liệu
def nap(bo: str) -> dict:
    """{'M': số đo, 'B': S1, 'bo': bo} — chỉ tệp có đủ số đo + dac_trung2 (dòng jsonl ⇒ npy đã ghi)."""
    M = H.so_do(bo)
    B = {r["file"]: r["bh"] for r in jl(os.path.join(V6, bo, "_kiem", "dac_trung2.jsonl"))}
    return {"M": M, "B": B, "bo": bo}


def vo_huong(files: list, D: dict, cols: list | None = None) -> tuple:
    """Ma trận F1 (m:) + S1 (b:) — cột NaN khi thiếu; dùng H.mo_hinh (impute + scale) cho phần vô hướng."""
    if cols is None:
        cols = [f"m:{k}" for k in sorted({k for f in files for k in D[f]["M"]})] + \
               [f"b:{k}" for k in sorted({k for f in files for k in D[f]["B"]})]
    X = np.full((len(files), len(cols)), np.nan)
    for i, f in enumerate(files):
        for j, c in enumerate(cols):
            t, k = c.split(":", 1)
            v = (D[f]["M"] if t == "m" else D[f]["B"]).get(k)
            X[i, j] = np.nan if v is None else v
    return X, cols


def lop(files: list, D: dict) -> np.ndarray:
    """(n, 13, 2304) float16 từ npy."""
    return np.stack([np.load(os.path.join(V6, D[f]["bo"], "_kiem", "w2v_lop", ten_npy(f))).reshape(13, -1) for f in files])


def gop(*bos: dict) -> dict:
    """file → {'M','B','bo'} cho mọi tệp có đủ M và B."""
    return {f: {"M": b["M"][f], "B": b["B"][f], "bo": b["bo"]} for b in bos for f in b["B"] if f in b["M"]}


# ----------------------------------------------------------------------------- đầu torch (S2 / S3)
def dau_torch(Xl: np.ndarray, Xs: np.ndarray | None, y: np.ndarray, groups: list, seed: int = 0):
    """Khớp đầu: chuẩn hoá theo lớp/chiều → softmax(w)·lớp → [ + vô hướng ] → MLP(256, 64), dropout 0,2. Trả hàm dự đoán."""
    import torch
    from torch import nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gs = sorted(set(groups))
    va_g = set(rng.choice(gs, max(1, len(gs) // 10), replace=False).tolist())   # early stop: 10 % train theo base
    va = np.array([g in va_g for g in groups])
    mu, sd = Xl[~va].astype(np.float32).mean(0), Xl[~va].astype(np.float32).std(0) + 1e-4
    sc = None
    if Xs is not None:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        sc = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()).fit(Xs[~va])
    ns = 0 if sc is None else sc.transform(Xs[:1]).shape[1]

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros(Xl.shape[1]))
            self.mlp = nn.Sequential(nn.Linear(Xl.shape[2] + ns, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xl, xs):
            z = (torch.softmax(self.w, 0)[None, :, None] * xl).sum(1)
            return self.mlp(z if xs is None else torch.cat([z, xs], 1))[:, 0]

    def tens(Xl2, Xs2):
        a = torch.from_numpy(((Xl2.astype(np.float32) - mu) / sd))
        return a, None if sc is None else torch.from_numpy(sc.transform(Xs2).astype(np.float32))

    m = Head()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    xl, xs = tens(Xl, Xs)
    ym, ys = float(y[~va].mean()), float(y[~va].std() + 1e-6)              # học trên đích chuẩn hoá, trả về dB
    yt = torch.from_numpy(((y - ym) / ys).astype(np.float32))
    tr_i, va_i = np.where(~va)[0], np.where(va)[0]
    best, state = np.inf, None
    for ep in range(60):
        m.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 64)):
            opt.zero_grad()
            loss = ((m(xl[b], None if xs is None else xs[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            v = float(((m(xl[va_i], None if xs is None else xs[va_i]) - yt[va_i]) ** 2).mean())
        if v < best:
            best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state)
    m.eval()

    def du(Xl2, Xs2):
        a, s = tens(Xl2, Xs2)
        with torch.no_grad():
            return m(a, s).numpy().astype(float) * ys + ym
    return du, torch.softmax(m.w, 0).detach().numpy()


# ----------------------------------------------------------------------------- ứng viên
def khop(u: str, files: list, y: np.ndarray, D: dict, base: dict):
    """Khớp ứng viên u trên files → hàm du(files2, D2) → dB."""
    if u in ("mlp_F1", "S1"):
        X, cols = vo_huong(files, D)
        cols = [c for c in cols if u == "S1" or c.startswith("m:")]
        X = vo_huong(files, D, cols)[0]
        mo = H.mo_hinh("mlp", cols).fit(X, y)
        return lambda f2, D2: mo.predict(vo_huong(f2, D2, cols)[0])
    Xs, cols = vo_huong(files, D) if u == "S3" else (None, None)
    du, w = dau_torch(lop(files, D), Xs, y, [base[f] for f in files])
    print(f"  {u} trọng số lớp: {np.round(w, 3).tolist()}", flush=True)
    return lambda f2, D2: du(lop(f2, D2), None if cols is None else vo_huong(f2, D2, cols)[0])


# ----------------------------------------------------------------------------- chấm
def oof_v4(u: str, tr: list, ev: list, y: dict, D: dict, da: dict) -> dict:
    base = {f: da[f]["base"] for f in tr}
    p = {}
    for k in (0, 1, 2):
        trk = [f for f in tr if fold(base[f]) != k]
        tek = [f for f in ev if fold(da[f]["base"]) == k]
        assert max(fold(base[f]) for f in trk) <= 2
        p.update(zip(tek, khop(u, trk, np.array([y[f] for f in trk]), D, base)(tek, D)))
    return p


def sai_so(p: dict, y: dict, ev: list, ca: dict) -> list:
    """Dòng bảng: σ / MAE vùng 8–16 dB + lệch TB theo loại nền (vùng 8–16)."""
    vung = [f for f in ev if 8 <= y[f] <= 16]
    e = np.array([p[f] - y[f] for f in vung])
    loai = sorted({ca[f].split(".")[0] for f in vung})
    lech = {lo: np.mean([p[f] - y[f] for f in vung if ca[f].split(".")[0] == lo]) for lo in loai}
    return [float(e.std()), float(np.abs(e).mean()), lech]


def bao_that(du) -> list:
    """Tệp thật p1_tunhien_v1 dev accept: (c) bị chặn theo hai cách đếm (nhóm: mọi tệp; P1: WADA không xác nhận = chặn oan)."""
    cn = cases_of(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and fold(c["base"]) <= 2]
    Dn = gop(nap(NAT))
    AN = H.dac_trung(NAT)
    fa = [f for f in acc if f in Dn and f in AN]
    pa = du(fa, Dn)
    b = [f for f, v in zip(fa, pa) if v < 10]
    xn = sum(AN[f]["wada_db"] < 10 for f in b)
    return ["", "## 3. Tệp thật p1_tunhien_v1 (dev, accept)", "",
            "| cách đếm | chặn oan |", "|---|---|",
            f"| nhóm (mọi tệp thật hợp lệ) | {pct(len(b), len(fa))} (vi {sum(f[:2] == 'vi' for f in b)}, en {sum(f[:2] == 'en' for f in b)}) |",
            f"| P1 (WADA không xác nhận < 10 dB) | {pct(len(b) - xn, len(fa))} |"]


def bao_v3(du, share: dict) -> list:
    cs3 = cases_of(H.V3)
    prov3 = {p["file"].split(f"{H.V3}/", 1)[-1]: p for p in jl(os.path.join(V6, H.V3, "provenance.jsonl"))}
    da3 = C.dap_an_v3(cs3, prov3)
    D3 = gop(nap(H.V3))
    f3 = [f for f in da3 if f in D3]
    res = {"chon": dict(zip(f3, (du(f3, D3) < 10).tolist()))}
    C.UV = tuple(res)
    return ["", "## 2. Ứng viên chọn, khớp trên cả train → v3 dev (không dùng để chọn)", ""] + C.bang(res, da3, cs3, share)[0]



def bang(res: dict, ev: list, da: dict, share: dict, tieu_de: str) -> tuple:
    Lb, tom = H.bang_snr(res, {f: da[f] for f in ev}, share)
    return ["", f"### {tieu_de} ({len(ev)} tệp)", ""] + Lb, tom


def run() -> None:
    for p in (OUT, DU):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    da = {**H.dap_an(H.V4), **H.dap_an(H.V4B)}
    D = gop(nap(H.V4), nap(H.V4B))
    tr = sorted(f for f in da if f in D)
    ev = [f for f in tr if "p1xv4b_" not in f]
    y = {f: float(np.clip(da[f]["snr"], -10, 40)) for f in tr}
    ca = {c["file"]: c["ca"] for c in json.load(open(os.path.join(V6, H.V4, "cases.json"), encoding="utf-8"))}
    cn, MN = cases_of(NAT), C.gop(NAT)
    share = C.share_that(MN, [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2])
    on = set(json.load(open(BO_ON, encoding="utf-8"))["base"]) if os.path.exists(BO_ON) else None
    ev2 = [f for f in ev if on is not None and da[f]["base"] not in on]
    res, res2, ss = {}, {}, {}
    for u in UV:
        p = oof_v4(u, tr, ev, y, D, da)
        res[u] = {f: bool(p[f] < 10) for f in ev}
        res2[u] = {f: res[u][f] for f in ev2}
        ss[u] = sai_so(p, y, ev, ca)
        with open(DU, "a", encoding="utf-8") as fh:                   # lưu OOF → thước 2 tính sau khi có kết quả nghe, không train lại
            fh.writelines(json.dumps({"uv": u, "file": f, "du": round(float(p[f]), 3)}) + "\n" for f in ev)
        print(u, f"σ vùng 8–16 {ss[u][0]:.2f} dB · MAE {ss[u][1]:.2f}", flush=True)
    L = ["# Bộ ước SNR mạnh hơn (NHAT_KY 2.22, luật chốt trước)", "",
         f"- Train v4 + v4b fold 0–2 ({len(tr)} tệp) · chấm OOF CHỈ v4 dev · bản gốc ồn (tai Huấn): "
         + ("chưa có kết quả nghe" if on is None else f"{sorted(on) or 'không có'}"), "", "## 1. SNR (OOF; < 10 dB → 04)"]
    Lb, tom = bang(res, ev, da, share, "Thước 2.20 / 2.21: mọi tệp v4 dev")
    L += Lb
    if on is not None:
        L += bang(res2, ev2, da, share, "Thước 2: bỏ base có bản gốc ồn")[0]
    loai = sorted(ss["mlp_F1"][2])
    L += ["", "| ứng viên | σ vùng 8–16 dB | MAE vùng 8–16 | " + " | ".join(f"lệch {lo}" for lo in loai) + " |", "|---|---|---|" + "---|" * len(loai)]
    L += [f"| {u} | {ss[u][0]:.2f} | {ss[u][1]:.2f} | " + " | ".join(f"{ss[u][2][lo]:+.2f}" for lo in loai) + " |" for u in UV]
    u = H.chon(tom, UV)
    L += ["", f"- **chọn {u}** · (a*) {100 * tom[u]['a_star']:.1f} % → **{'ĐỦ' if tom[u]['a_star'] <= 0.02 else 'CHƯA ĐỦ'}** (cần ≤ 2 %)" if u else "- không ứng viên nào đạt bắt rõ"]
    if u:
        qua = ss[u][0] <= SIGMA_MAX
        L += [f"- Cổng khả thi: σ {ss[u][0]:.2f} dB {'≤' if qua else '>'} {SIGMA_MAX} → " + ("**qua**" if qua else "**DỪNG hướng mô hình, báo Huấn**")]
        base = {f: da[f]["base"] for f in tr}
        du = khop(u, tr, np.array([y[f] for f in tr]), D, base)
        L += bao_v3(du, share) + bao_that(du)
    L += ["", "- Chưa mở test (fold 3–4 v3 / v4)."]
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def thuoc2() -> None:
    """Thước 2 từ OOF đã lưu: bỏ base trong base_goc_on.json (sau khi chấm nghe_goc.py)."""
    on = set(json.load(open(BO_ON, encoding="utf-8"))["base"])
    da = H.dap_an(H.V4)
    du: dict = {}
    for r in jl(DU):
        du.setdefault(r["uv"], {})[r["file"]] = r["du"]
    ev2 = [f for f in du["mlp_F1"] if da[f]["base"] not in on]
    cn, MN = cases_of(NAT), C.gop(NAT)
    share = C.share_that(MN, [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2])
    res = {u: {f: du[u][f] < 10 for f in ev2} for u in UV}
    L, tom = bang(res, ev2, da, share, f"Thước 2: bỏ base có bản gốc ồn {sorted(on) or '(không có)'}")
    L += ["", f"- chọn theo thước 2 (chỉ để báo, luật chọn dùng thước 1): {H.chon(tom, UV)}"]
    print("\n".join(L))


def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.uniform(-5, 30, n)
    Xl = rng.normal(0, 1, (n, 13, 8)).astype(np.float16)
    Xl[:, 4, 0] = (y / 10).astype(np.float16)                    # chỉ lớp 4 mang tín hiệu
    groups = [f"b{i % 40}" for i in range(n)]
    du, w = dau_torch(Xl, None, y, groups)
    e = np.abs(du(Xl, None) - y).mean()
    assert e < 3.0 and int(np.argmax(w)) == 4, (e, np.round(w, 3))
    Dt = {"a": {"M": {"x": 1.0}, "B": {"s": None}}, "b": {"M": {"x": 2.0}, "B": {"s": 3.0}}}
    X, cols = vo_huong(["a", "b"], Dt)
    assert cols == ["m:x", "b:s"] and np.isnan(X[0, 1]) and X[1, 1] == 3.0
    assert H.chon({"mlp_F1": {"ro": (9, 10), "a_star": 0.030}, "S3": {"ro": (9, 10), "a_star": 0.026}}, UV) == "mlp_F1"
    print(f"huan_luyen2 selftest dat · MAE {e:.2f} · trọng số lớp 4 {w[4]:.2f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "thuoc2": thuoc2, "selftest": selftest}[sys.argv[1]]()
