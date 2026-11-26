"""Minh chứng cho đề xuất sửa CÁCH ĐO hàng ồn (lanes/DE_XUAT/2611_P1_cach-do-on.md). Ngưỡng hàng cứng giữ 10 dB; chỉ đổi luật quyết định
khi phép đo còn sai số. CHỈ fold 0–2 (dev). Không train lại OOF: đọc `du_doan_oof_2511c.jsonl` (2.22, train v4 + v4b).
  A  biên an toàn : chặn khi đoán < 10 − m (m = 0 … 5 dB); vùng [10 − m, 10) → cảnh báo, không chặn.
  B  đồng thuận   : chặn khi mlp_F1 VÀ S2 cùng < 10 − m (hai bộ ước khác đặc trưng: số đo có sẵn · w2v 13 lớp).
Ra: (1) v4 dev OOF: (a*), chặn oan theo dải, bắt < 10 dB theo dải; (2) như (1) nhưng bỏ base nghi ồn (ĐỘ NHẠY, chưa có tai Huấn);
(3) tệp thật dev: bị chặn + WADA xác nhận; (4) 24 tệp Huấn đã nghe (nghe_tay_snr_2511): theo nhãn tai, luật nào còn chặn.

  .venv/Scripts/python.exe experts/P1/build_cond/de_xuat_bien.py run
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cham_v3 as C  # noqa: E402
import huan_luyen as H  # noqa: E402
import huan_luyen2 as H2  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, jl, pct  # noqa: E402
from cham_tram import cases_of  # noqa: E402

K = os.path.join(V6, H.V4, "_kiem")
OUT, DU_THAT = os.path.join(K, "DE_XUAT_BIEN_2611.md"), os.path.join(K, "du_doan_that_2611.jsonl")
NGHE = os.path.join(V6, NAT, "_kiem", "nghe_tay_snr_2511.json")
BIEN = (0, 1, 2, 3, 4, 5)
NGHI = "fleurs_en_dev_5484291758150903027"            # base nghi bản gốc ồn (NHAT_KY 2.22) — chỉ dùng cho bảng ĐỘ NHẠY
LUAT = ("mlp_F1", "S2", "B")


def diem(du: dict, luat: str, f: str) -> float:
    """Số đem so với 10 − m. B = max của hai bộ ước (cả hai cùng dưới ngưỡng ⇔ max dưới ngưỡng)."""
    return max(du["mlp_F1"][f], du["S2"][f]) if luat == "B" else du[luat][f]


def cham(du: dict, luat: str, m: float, da: dict, files: list, share: dict) -> dict:
    ok = [f for f in files if not da[f]["bien"]]
    ch = {f: diem(du, luat, f) < 10 - m for f in ok}
    hl = [f for f in ok if da[f]["snr"] >= 10 and not da[f]["is04"]]
    vp = [f for f in ok if da[f]["is04"]]
    rate = {}
    for b in C.DAI:
        x = [f for f in hl if b[0] <= da[f]["snr"] < b[1]]
        rate[b] = (sum(ch[f] for f in x), len(x))
    dai = {n: [f for f in vp if lo <= da[f]["snr"] < hi] for n, lo, hi in (("≤ 5", -99, 5.0001), ("5–8", 5.0001, 8), ("8–10", 8, 10.5))}
    return {"a_star": sum(k / n * share[b] for b, (k, n) in rate.items() if n) / sum(share.values()), "rate": rate,
            "bat": (sum(ch[f] for f in vp), len(vp)), "dai": {n: (sum(ch[f] for f in x), len(x)) for n, x in dai.items()}}


def bang(du: dict, da: dict, files: list, share: dict, tieu_de: str) -> list:
    L = ["", f"### {tieu_de} ({len(files)} tệp)", "",
         "| luật | m dB | (a*) | chặn oan 10–12 | 12–15 | 15–20 | ≥ 20 | bắt < 10 dB | bắt ≤ 5 | bắt 5–8 | bắt 8–10 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lu in LUAT:
        for m in BIEN:
            r = cham(du, lu, m, da, files, share)
            L.append(f"| {lu} | {m} | **{100 * r['a_star']:.1f} %** | " + " | ".join(f"{k}/{n} = {100 * k / n:.1f} %" for k, n in r["rate"].values())
                     + f" | {pct(*r['bat'])} | " + " | ".join(f"{k}/{n} = {100 * k / max(n, 1):.0f} %" for k, n in r["dai"].values()) + " |")
    return L


def du_that() -> tuple[dict, list]:
    """Khớp mlp_F1 và S2 trên cả train (v4 + v4b fold 0–2) → dự đoán tệp thật dev accept. Lưu lại, chạy lần sau đọc tệp."""
    cn = cases_of(NAT)
    acc = sorted(f for f, c in cn.items() if c["mong_doi"] == "accept" and fold(c["base"]) <= 2)
    if os.path.exists(DU_THAT):
        du: dict = {}
        for r in jl(DU_THAT):
            du.setdefault(r["uv"], {})[r["file"]] = r["du"]
        return du, acc
    da = {**H.dap_an(H.V4), **H.dap_an(H.V4B)}
    D = H2.gop(H2.nap(H.V4), H2.nap(H.V4B))
    tr = sorted(f for f in da if f in D)
    y = np.array([float(np.clip(da[f]["snr"], -10, 40)) for f in tr])
    base = {f: da[f]["base"] for f in tr}
    Dn = H2.gop(H2.nap(NAT))
    assert all(f in Dn for f in acc), "thiếu đặc trưng tệp thật"
    du = {u: dict(zip(acc, H2.khop(u, tr, y, D, base)(acc, Dn).tolist())) for u in ("mlp_F1", "S2")}
    with open(DU_THAT, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"uv": u, "file": f, "du": round(v, 3)}) + "\n" for u in du for f, v in du[u].items())
    return du, acc


def bang_that(du: dict, acc: list) -> list:
    AN = H.dac_trung(NAT)
    L = ["", f"## 3. Tệp thật p1_tunhien_v1 dev accept ({len(acc)} tệp; mọi tệp đều «nên nhận» theo nhãn cách tạo)", "",
         "| luật | m dB | bị chặn (cách nhóm) | WADA < 10 dB xác nhận | chặn oan cách P1 (WADA không xác nhận) |", "|---|---|---|---|---|"]
    for lu in LUAT:
        for m in BIEN:
            b = [f for f in acc if diem(du, lu, f) < 10 - m]
            xn = sum(AN[f]["wada_db"] < 10 for f in b)
            L.append(f"| {lu} | {m} | {pct(len(b), len(acc))} | {xn}/{len(b)} | {pct(len(b) - xn, len(acc))} |")
    kq = json.load(open(NGHE, encoding="utf-8"))["ket_qua"]
    nhan = collections.defaultdict(list)
    for f, v in kq.items():
        if f in du["mlp_F1"]:
            nhan[v.get("on_nen")].append(f)
    cot = [x for x in ("khong", "nhe", "ro", "che_loi") if nhan[x]]
    L += ["", f"## 4. 24 tệp thật Huấn đã nghe ({sum(len(v) for v in nhan.values())} tệp có dự đoán): còn bị chặn theo nhãn tai", "",
          "Hiệu chỉnh tai (2.21): `ro` = dưới 10 dB (chặn đúng); `khong` = sạch (chặn là oan); `nhe` = 11/14 trên 10 dB (có lẽ oan).", "",
          "| luật | m dB | " + " | ".join(f"{x} (n = {len(nhan[x])})" for x in cot) + " |", "|---|---|" + "---|" * len(cot)]
    for lu in LUAT:
        for m in BIEN:
            L.append(f"| {lu} | {m} | " + " | ".join(str(sum(diem(du, lu, f) < 10 - m for f in nhan[x])) for x in cot) + " |")
    return L


def run() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    da = H.dap_an(H.V4)
    du: dict = {}
    for r in jl(H2.DU):
        du.setdefault(r["uv"], {})[r["file"]] = r["du"]
    ev = sorted(du["mlp_F1"])
    assert set(ev) == set(du["S2"]) and all(fold(da[f]["base"]) <= 2 for f in ev)
    cn, MN = cases_of(NAT), C.gop(NAT)
    share = C.share_that(MN, [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2])
    L = ["# Minh chứng đề xuất sửa cách đo hàng ồn (26/11) — chỉ dev, test chưa mở", "",
         "- A = biên an toàn trên một bộ ước (mlp_F1 hoặc S2): chặn khi đoán < 10 − m. B = đồng thuận: mlp_F1 và S2 cùng < 10 − m.",
         "- m = 0 là cách đo hiện tại. OOF 2.22 (train v4 + v4b fold 0–2).", "", "## 1–2. Dữ liệu dựng v4 dev (OOF)"]
    L += bang(du, da, ev, share, "1. Thước 1: mọi tệp v4 dev")
    L += bang(du, da, [f for f in ev if da[f]["base"] != NGHI], share, f"2. ĐỘ NHẠY (chưa có tai Huấn): bỏ base nghi ồn `{NGHI}`")
    dt, acc = du_that()
    L += bang_that(dt, acc)
    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def selftest() -> None:
    da = {f"f{i}": {"snr": s, "is04": s < 10, "bien": False} for i, s in enumerate((3.0, 7.0, 9.0, 11.0, 13.0, 17.0, 25.0))}
    du = {"mlp_F1": {f: da[f]["snr"] - 2 for f in da}, "S2": {f: da[f]["snr"] + 1 for f in da}}
    share = {b: 0.25 for b in C.DAI}
    r0, r3 = cham(du, "mlp_F1", 0, da, list(da), share), cham(du, "mlp_F1", 3, da, list(da), share)
    assert r0["rate"][(10, 12)] == (1, 1) and abs(r0["a_star"] - 0.25) < 1e-9 and r0["bat"] == (3, 3)
    assert r3["a_star"] == 0 and r3["bat"] == (2, 3) and r3["dai"]["8–10"] == (0, 1)
    assert cham(du, "B", 0, da, list(da), share)["a_star"] == 0 and diem(du, "B", "f0") == 4.0
    print("de_xuat_bien selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1]]()
