"""Hiệu chỉnh + chấm bộ đo ồn / nền trên p1_dieukien_v3 (luật chốt trước: NHAT_KY 2.16 «Luật hiệu chỉnh + chấm trên v3»).
Chỉ dev (fold 0–2); OOF 3 fold trong dev. Không đọc fold test. Ra: p1_dieukien_v3/_kiem/CHAM_V3_2411.md

  .venv/Scripts/python.exe experts/P1/build_cond/cham_v3.py run | selftest
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, SYN, V6, fold, jl, late, pct  # noqa: E402
from cham_tram import NS04, cases_of, dap_an_syn  # noqa: E402
from cham_vang import gop  # noqa: E402

V3 = "p1_dieukien_v3"
CHAM = os.path.join(V6, V3, "_kiem", "CHAM_V3_2411.md")
KEP = -20.0                                                    # snr_nghi ≤ −60 (lỗi −173 dB) ⇒ mức lời ≤ nền ⇒ kẹp −20
DAI = ((10, 12), (12, 15), (15, 20), (20, 99))
UV = ("C3v2", "C3", "NGHI", "TB", "C3_VA_NGHI")


def nghi(m: dict):
    v = m.get("snr_nghi")
    return None if v is None else max(v, KEP)


def X(u: str, m: dict):
    """Hàng đặc trưng; None = không đo được."""
    if u in ("C3", "C3v2"):
        return None if m["bh"] is None else [late(m["bh"]), 1.0]
    if u == "NGHI":
        return None if nghi(m) is None else [late(nghi(m)), 1.0]
    if u == "TB":
        return None if m["bh"] is None or nghi(m) is None else [late(m["bh"]), late(nghi(m)), 1.0]
    raise ValueError(u)


def khop(rows: list) -> np.ndarray:
    return np.linalg.lstsq(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), rcond=None)[0]


def lam_chan(files: list, M: dict, ns: dict, coef_v2) -> dict:
    """Khớp trên files → {ứng viên: hàm chặn(m) → bool}."""
    c = {u: khop([(X(u, M[f]), ns[f]) for f in files if X(u, M[f]) is not None]) for u in ("C3", "NGHI", "TB")}
    c["C3v2"] = coef_v2
    sc = lambda u, m: X(u, m) is not None and float(np.dot(c[u], X(u, m))) > NS04  # noqa: E731
    f = {u: (lambda m, u=u: sc(u, m)) for u in ("C3v2", "C3", "NGHI")}
    f["TB"] = lambda m: sc("TB", m) if X("TB", m) is not None else f["C3"](m)
    f["C3_VA_NGHI"] = lambda m: f["C3"](m) and f["NGHI"](m)
    return {"chan": f, "coef": c}


def dap_an_v3(cs: dict, prov: dict) -> dict:
    """file → (snr_A thật, là_04, ca_bien) cho dev; nen quy ra SNR tương đương."""
    out = {}
    for f, c in cs.items():
        if fold(c["base"]) >= 3:
            continue
        p = prov[f]
        d, t = p["do_lai"], p["tham_so"]
        if c["ca"] == "goc":
            s = t["san_a_db"]
        elif "snr_that_A_db" in d:
            s = d["snr_that_A_db"]
        else:
            s = 10 * np.log10((100 - d["nen_A_pct"]) / d["nen_A_pct"])
        out[f] = (float(s), c["mong_doi"] == "04", bool(c["ca_bien"]))
    return out


def oof(dev: list, M: dict, ns: dict, coef_v2, cs: dict) -> dict:
    """Quyết định OOF: fold k dùng hàm khớp trên 2 fold còn lại."""
    res = {u: {} for u in UV}
    for k in (0, 1, 2):
        tr = [f for f in dev if fold(cs[f]["base"]) != k]
        ch = lam_chan(tr, M, ns, coef_v2)["chan"]
        for f in (f for f in dev if fold(cs[f]["base"]) == k):
            for u in UV:
                res[u][f] = ch[u](M[f])
    return res


def bang(res: dict, da: dict, cs: dict, share: dict) -> list:
    L = ["| ứng viên | (a) chặn oan ≥ 10 dB | 10–12 | 12–15 | 15–20 | ≥ 20 | (a*) phân bố thật | (b) bắt < 10 dB | bắt rõ ≤ 5 dB | (a) vi | (a) en |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    tom = {}
    for u in UV:
        r = res[u]
        ok = [f for f in r if not da[f][2]]
        hl = [f for f in ok if da[f][0] >= 10 and not da[f][1]]
        vp = [f for f in ok if da[f][1]]
        ro = [f for f in vp if da[f][0] <= 5]
        bins = {b: [f for f in hl if b[0] <= da[f][0] < b[1]] for b in DAI}
        rate = {b: sum(r[f] for f in x) / len(x) if x else 0.0 for b, x in bins.items()}
        a_star = sum(rate[b] * share[b] for b in DAI) / max(sum(share.values()), 1e-9)
        lg = {g: [f for f in hl if f"_{g}_" in f] for g in ("vi", "en")}
        tom[u] = {"a": (sum(r[f] for f in hl), len(hl)), "a_star": a_star, "ro": (sum(r[f] for f in ro), len(ro)),
                  "b": (sum(r[f] for f in vp), len(vp))}
        L.append(f"| {u} | {pct(*tom[u]['a'])} | " + " | ".join(pct(sum(r[f] for f in bins[b]), len(bins[b])) for b in DAI)
                 + f" | **{100 * a_star:.1f} %** | {pct(*tom[u]['b'])} | {pct(*tom[u]['ro'])} | "
                 + " | ".join(pct(sum(r[f] for f in lg[g]), len(lg[g])) for g in ("vi", "en")) + " |")
    return L, tom


def theo_loai(res: dict, da: dict, cs: dict) -> list:
    loai = sorted({cs[f]["ca"].split(".")[0] for f in res["C3"]})
    L = ["| ứng viên | " + " | ".join(loai) + " |", "|---|" + "---|" * len(loai)]
    for u in UV:
        r = res[u]
        cells = []
        for lo in loai:
            hl = [f for f in r if cs[f]["ca"].split(".")[0] == lo and not da[f][2] and da[f][0] >= 10 and not da[f][1]]
            cells.append(pct(sum(r[f] for f in hl), len(hl)) if hl else "–")
        L.append(f"| {u} | " + " | ".join(cells) + " |")
    return L


def share_that(MN: dict, acc: list) -> dict:
    v = [nghi(MN[f]) for f in acc if nghi(MN[f]) is not None]
    return {b: sum(b[0] <= x < b[1] for x in v) / len(v) for b in DAI}


def run() -> None:
    cs, prov = cases_of(V3), {p["file"].split(f"{V3}/", 1)[-1]: p for p in jl(os.path.join(V6, V3, "provenance.jsonl"))}
    M = gop(V3)
    da = dap_an_v3(cs, prov)
    dev = [f for f in da if f in M]
    ns = {f: late(da[f][0]) for f in dev}
    # hệ số C3 cũ (khớp trên v2, đáp án A) — tái dùng cham_vang.lam
    from cham_vang import lam
    c2, M2 = cases_of(SYN), gop(SYN)
    d2, _ = dap_an_syn(c2)
    coef_v2 = lam([f for f in d2 if f in M2], M2, d2)["coef"]["C3"]
    cn, MN = cases_of(NAT), gop(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2]
    share = share_that(MN, acc)
    res = oof(dev, M, ns, coef_v2, cs)
    L = ["# Hiệu chỉnh + chấm bộ đo ồn / nền trên p1_dieukien_v3 (luật chốt trước: NHAT_KY 2.16)", "",
         f"- Dev v3: {len(dev)} tệp (bỏ ca_bien khi chấm) · OOF 3 fold theo base · tệp thật accept dev p1_tunhien_v1: {len(acc)}",
         "- Phân bố thật (snr_nghi, tệp ≥ 10 dB): " + " · ".join(f"{b[0]}–{b[1]}: {100 * share[b]:.1f} %" for b in DAI), "",
         "## 1. Chặn oan theo Constraint / bắt đúng (dev v3, OOF)", ""]
    Lb, tom = bang(res, da, cs, share)
    L += Lb + ["", "## 2. Chặn oan theo loại nền (SNR A thật ≥ 10 dB)", ""] + theo_loai(res, da, cs)
    full = lam_chan(dev, M, ns, coef_v2)
    ch = full["chan"]
    L += ["", "## 3. Hệ số khớp trên cả dev v3", ""] + [f"- {u}: {np.round(c, 3).tolist()}" for u, c in full["coef"].items()]
    L += ["", "## 4. Tệp thật p1_tunhien_v1 (dev, accept)", "", "| ứng viên | (c) bị chặn | vi | en | xác nhận độc lập (snr_nghi < 10) |", "|---|---|---|---|---|"]
    for u in UV:
        b = [f for f in acc if ch[u](MN[f])]
        xn = "–" if u in ("NGHI", "TB", "C3_VA_NGHI") else f"{sum(nghi(MN[f]) is not None and nghi(MN[f]) < 10 for f in b)}/{len(b)}"
        L.append(f"| {u} | {pct(len(b), len(acc))} | {sum(f[:2] == 'vi' for f in b)}/{sum(f[:2] == 'vi' for f in acc)} | "
                 f"{sum(f[:2] == 'en' for f in b)}/{sum(f[:2] == 'en' for f in acc)} | {xn} |")
    L += ["", "## 5. 50 tệp Huấn nghe (chỉ mô tả)", ""] + nghe_huan(ch, MN)
    ok = [u for u in UV if tom[u]["ro"][1] and tom[u]["ro"][0] / tom[u]["ro"][1] >= 0.8]
    chon = min(ok, key=lambda u: (tom[u]["a"][0] / max(tom[u]["a"][1], 1),
                                  sum(ch[u](MN[f]) for f in acc))) if ok else None
    du = chon and tom[chon]["a_star"] <= 0.02
    L += ["", f"- Đạt bắt rõ ≥ 80 %: {', '.join(ok) or 'không có'} → **chọn {chon}**",
          f"- (a*) của {chon}: {100 * tom[chon]['a_star']:.1f} % → **{'ĐỦ' if du else 'CHƯA ĐỦ'}** để cài K1 (cần ≤ 2 %)" if chon else "",
          "- Chưa mở test (fold 3–4 v3)."]
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


def nghe_huan(ch: dict, MN: dict) -> list:
    lab = {}
    for mp, kq in (("NGHE_TAY_2411_map.json", "nghe_tay_2411.json"), ("NGHE_TAY_C4_2411_map.json", "nghe_tay_c4_2411.json")):
        k = json.load(open(os.path.join(V6, NAT, "_kiem", kq), encoding="utf-8"))["ket_qua"]
        for r in json.load(open(os.path.join(V6, NAT, "_kiem", mp), encoding="utf-8"))["tep"]:
            if r["loai"] == "that" and r["nguon"].split("/", 1)[1] in MN:
                lab[r["nguon"].split("/", 1)[1]] = k[r["ten"]]["on_nen"]
    L = ["| ứng viên | chặn oan rộng | chặn oan chặt | thả sai rộng | thả sai chặt |", "|---|---|---|---|---|"]
    for u in UV:
        b = [f for f in lab if ch[u](MN[f])]
        q = [f for f in lab if not ch[u](MN[f])]
        L.append(f"| {u} | {sum(lab[f] in ('khong', 'nhe') for f in b)}/{len(b)} | {sum(lab[f] != 'che_loi' for f in b)}/{len(b)} | "
                 f"{sum(lab[f] in ('ro', 'che_loi') for f in q)}/{len(q)} | {sum(lab[f] == 'che_loi' for f in q)}/{len(q)} |")
    return L


def selftest() -> None:
    assert nghi({"snr_nghi": -173.2}) == KEP and nghi({"snr_nghi": None}) is None
    M = {f"f{i}": {"bh": s + 1, "snr_nghi": s - 1} for i, s in enumerate(range(1, 30, 2))}   # mức lẻ: không ca nào đúng ngưỡng 10 dB
    ns = {f: late(M[f]["bh"] - 1) for f in M}
    ch = lam_chan(list(M), M, ns, np.array([1.0, 0.0]))["chan"]
    got = [ch["TB"](M[f]) for f in M]
    assert got == [late(M[f]["bh"] - 1) > NS04 for f in M], got
    assert ch["NGHI"]({"bh": 5, "snr_nghi": None}) is False and ch["TB"]({"bh": 0.0, "snr_nghi": None}) is True
    print("cham_v3 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "selftest"]()
