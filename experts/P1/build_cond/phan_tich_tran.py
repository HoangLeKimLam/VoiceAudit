"""NHAT_KY 2.22 (khám phá, CHỈ v4 dev + p1_tunhien_v1 dev cho trọng số): trần đạt được của (a*) và các ca chặn oan ≥ 20 dB.
1) Bộ ước giả định đoán = SNR thật + N(0, σ) (không lệch): (a*) theo σ, cùng công thức cham_v3.bang (bins DAI, trọng số share thật).
2) mlp_F1 (OOF, train v4 dev): các tệp SNR thật ≥ 20 dB bị đoán < 10 dB → loại nền, base, SNR Brouhaha / WADA của goc cùng base.

  .venv/Scripts/python.exe experts/P1/build_cond/phan_tich_tran.py
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
from cham_dieu_kien import NAT, V6, fold  # noqa: E402
from cham_tram import cases_of  # noqa: E402


def a_star(pred: np.ndarray, y: np.ndarray, ok: np.ndarray, share: dict) -> tuple[float, dict]:
    rate = {}
    for b in C.DAI:
        m = ok & (y >= b[0]) & (y < b[1])
        rate[b] = float((pred[m] < 10).mean()) if m.any() else 0.0
    return sum(rate[b] * share[b] for b in C.DAI) / sum(share.values()), rate


def main() -> None:
    cn, MN = cases_of(NAT), C.gop(NAT)
    acc = [f for f, c in cn.items() if c["mong_doi"] == "accept" and f in MN and fold(c["base"]) <= 2]
    share = C.share_that(MN, acc)
    da, M, A = H.dap_an(H.V4), H.so_do(H.V4), H.dac_trung(H.V4)
    files = sorted(f for f in da if f in M)
    y = np.array([da[f]["snr"] for f in files])
    ok = np.array([not da[f]["is04"] and not da[f]["bien"] for f in files])
    print("trọng số thật:", {f"{b[0]}–{b[1]}": round(100 * v, 1) for b, v in share.items()}, f"(tổng {100 * sum(share.values()):.1f} %)")

    print("\n## 1) Bộ ước không lệch, sai số N(0, σ) — TB 20 lần gieo")
    rng = np.random.default_rng(0)
    for s in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        r = [a_star(y + rng.normal(0, s, len(y)), y, ok, share) for _ in range(20)]
        rate = {b: np.mean([x[1][b] for x in r]) for b in C.DAI}
        print(f"σ={s:4.2f} dB · (a*) {100 * np.mean([x[0] for x in r]):4.1f} % · " + " · ".join(f"{b[0]}–{b[1]}: {100 * v:4.1f} %" for b, v in rate.items()))

    X, cols = H.ma_tran(files, M, {}, "F1")
    p = H.oof(files, y, X, cols, "mlp", {f: da[f]["base"] for f in files})
    a, rate = a_star(p, y, ok, share)
    print(f"\nmlp_F1 OOF (train v4 dev): (a*) {100 * a:.1f} % · " + " · ".join(f"{b[0]}–{b[1]}: {100 * v:.1f} %" for b, v in rate.items()))
    sd = np.std((p - y)[(y >= 8) & (y <= 16)])
    print(f"độ lệch chuẩn sai số vùng 8–16 dB: {sd:.2f} dB")

    ca = {c["file"]: c["ca"] for c in json.load(open(os.path.join(V6, H.V4, "cases.json"), encoding="utf-8"))}
    goc = collections.defaultdict(set)
    for f in files:
        if f.startswith("goc/"):
            goc[da[f]["base"]].add((M[f].get("snr_bh_db"), A.get(f, {}).get("wada_db")))
    xa = [i for i, f in enumerate(files) if ok[i] and y[i] >= 20 and p[i] < 10]
    print(f"\n## 2) Chặn oan ≥ 20 dB: {len(xa)} tệp")
    for i in xa:
        f = files[i]
        print(f"  thật {y[i]:5.1f} đoán {p[i]:5.1f} · {ca[f]:22s} · {da[f]['base'][:40]:40s} · goc {sorted(goc.get(da[f]['base'], [])) or '—'}"
              f" · bh {M[f].get('snr_bh_db')} wada {A.get(f, {}).get('wada_db')}")
    print("theo loại:", collections.Counter(ca[files[i]].split(".")[0] for i in xa))
    print("theo base:", collections.Counter(da[files[i]]["base"] for i in xa).most_common(6))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
