"""So ba cách đo hàng ồn trên CÙNG thước (Huấn duyệt 26/11: đồng thuận là cách P1 chọn, cách cũ giữ để đối chiếu). CHỈ dev (fold 0–2).
  C3v2       : cách đo cũ của P1 (công thức trên số đo Brouhaha / DeepFilterNet, hệ số khớp trên p1_dieukien_v2; cham_v3).
  mlp_F1     : một bộ ước, chặn khi đoán < 10 dB.
  dong_thuan : mlp_F1 VÀ S2 cùng < 10 dB.
Cách đo của nhóm (vaudit/cond.py, XLS-R) KHÔNG chạy được ở đây: thiếu vaudit/cond_models.joblib (gói nhóm trưởng).
Ba bộ: v4 dev (OOF 2.22 đã lưu) · v3 dev (mô hình khớp trên v4 + v4b fold 0–2; v3 không dùng để chọn gì) · tệp thật p1_tunhien_v1 dev.
Không dùng kết quả này để chọn lại cấu hình.

  .venv/Scripts/python.exe experts/P1/build_cond/so_ba_cach.py run
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cham_v3 as C  # noqa: E402
import de_xuat_bien as X  # noqa: E402
import huan_luyen as H  # noqa: E402
import huan_luyen2 as H2  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, jl, pct  # noqa: E402
from cham_tram import cases_of, dap_an_syn  # noqa: E402
from cham_vang import lam  # noqa: E402

K = os.path.join(V6, H.V4, "_kiem")
OUT, DU_V3 = os.path.join(K, "SO_BA_CACH_2611.md"), os.path.join(V6, H.V3, "_kiem", "du_doan_2611.jsonl")
UV = ("C3v2", "mlp_F1", "dong_thuan")


def c3v2_fn(files: list, Mg: dict):
    c2, M2 = cases_of(C.SYN), C.gop(C.SYN)
    d2, _ = dap_an_syn(c2)
    coef = lam([f for f in d2 if f in M2], M2, d2)["coef"]["C3"]
    return C.lam_chan(files, Mg, {f: 0.0 for f in files}, coef)["chan"]["C3v2"]


def ket(du: dict, c3: dict, files: list) -> dict:
    return {"C3v2": {f: bool(c3[f]) for f in files}, "mlp_F1": {f: du["mlp_F1"][f] < 10 for f in files},
            "dong_thuan": {f: max(du["mlp_F1"][f], du["S2"][f]) < 10 for f in files}}


def doc_du(p: str) -> dict:
    du: dict = {}
    for r in jl(p):
        du.setdefault(r["uv"], {})[r["file"]] = r["du"]
    return du


def du_v3(f3: list) -> dict:
    """Khớp mlp_F1 và S2 trên cả train rồi dự đoán v3 dev; lưu lại."""
    if os.path.exists(DU_V3):
        return doc_du(DU_V3)
    da = {**H.dap_an(H.V4), **H.dap_an(H.V4B)}
    D = H2.gop(H2.nap(H.V4), H2.nap(H.V4B))
    tr = sorted(f for f in da if f in D)
    y = np.array([float(np.clip(da[f]["snr"], -10, 40)) for f in tr])
    base = {f: da[f]["base"] for f in tr}
    assert max(fold(b) for b in base.values()) <= 2
    D3 = H2.gop(H2.nap(H.V3))
    du = {u: dict(zip(f3, H2.khop(u, tr, y, D, base)(f3, D3).tolist())) for u in ("mlp_F1", "S2")}
    with open(DU_V3, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"uv": u, "file": f, "du": round(v, 3)}) + "\n" for u in du for f, v in du[u].items())
    return du


def run() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    cn, MN = cases_of(NAT), C.gop(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2]
    share = C.share_that(MN, acc)
    L = ["# So ba cách đo hàng ồn trên cùng thước (26/11) — chỉ dev, test chưa mở", "",
         "- C3v2 = cách đo cũ của P1 · mlp_F1 = một bộ ước · dong_thuan = mlp_F1 và S2 cùng < 10 dB (cách P1 chọn, Huấn duyệt 26/11).",
         "- Cách đo của nhóm (`vaudit/cond.py`) không chạy được trên máy này: thiếu `vaudit/cond_models.joblib`.", ""]
    # v4 dev
    da4, M4g = H.dap_an(H.V4), C.gop(H.V4)
    du4 = doc_du(H2.DU)
    f4 = sorted(f for f in du4["S2"] if f in M4g)
    c3 = c3v2_fn(f4, M4g)
    res4 = ket(du4, {f: c3(M4g[f]) for f in f4}, f4)
    C.UV = UV
    L += [f"## 1. Dữ liệu dựng v4 dev ({len(f4)} tệp; mlp_F1 / S2 là OOF)", ""] + H.bang_snr(res4, {f: da4[f] for f in f4}, share)[0]
    # v3 dev
    cs3 = cases_of(H.V3)
    prov3 = {p["file"].split(f"{H.V3}/", 1)[-1]: p for p in jl(os.path.join(V6, H.V3, "provenance.jsonl"))}
    da3, M3g = C.dap_an_v3(cs3, prov3), C.gop(H.V3)
    D3 = H2.gop(H2.nap(H.V3))
    f3 = sorted(f for f in da3 if f in M3g and f in D3)
    c3 = c3v2_fn(f3, M3g)
    res3 = ket(du_v3(f3), {f: c3(M3g[f]) for f in f3}, f3)
    C.UV = UV
    L += ["", f"## 2. Dữ liệu dựng v3 dev ({len(f3)} tệp; chưa từng dùng để chọn mô hình)", ""] + C.bang(res3, da3, cs3, share)[0] \
        + ["", "Chặn oan theo loại nền (SNR thật ≥ 10 dB):", ""] + C.theo_loai({**res3, "C3": res3["C3v2"]}, da3, cs3)
    # tệp thật dev
    dt, _ = X.du_that()
    AN = H.dac_trung(NAT)
    fa = sorted(f for f in acc if f in dt["S2"] and f in AN)
    c3 = c3v2_fn(fa, MN)
    resn = ket(dt, {f: c3(MN[f]) for f in fa}, fa)
    L += ["", f"## 3. Tệp thật p1_tunhien_v1 dev accept ({len(fa)} tệp có đủ số đo cho cả ba cách)", "",
          "| cách đo | bị chặn (cách đếm của nhóm) | vi | en | WADA < 10 dB xác nhận | chặn oan cách P1 |", "|---|---|---|---|---|---|"]
    for u in UV:
        b = [f for f in fa if resn[u][f]]
        xn = sum(AN[f]["wada_db"] < 10 for f in b)
        L.append(f"| {u} | {pct(len(b), len(fa))} | {sum(f[:2] == 'vi' for f in b)} | {sum(f[:2] == 'en' for f in b)} | {xn}/{len(b)} | {pct(len(b) - xn, len(fa))} |")
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run}[sys.argv[1]]()
