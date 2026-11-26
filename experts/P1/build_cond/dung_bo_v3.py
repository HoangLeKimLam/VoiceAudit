# -*- coding: utf-8 -*-
"""Dựng bộ `p1_dieukien_v3` (Huấn duyệt 24/11; luật chốt trước: NHAT_KY 2.16 «Bộ p1_dieukien_v3»).

Mục đích: hiệu chỉnh / kiểm bộ đo ồn–nền trên các loại nền bộ v2 thiếu (người khác nói, quán, nhạc qua loa, TV, tiếng động ngắt
quãng, ồn phòng thật), SNR trọng số A dày quanh 10 dB. Gốc = ghi âm thật rất sạch (p1_tunhien_v1, person = kênh) + FLEURS.
Đáp án từ cách trộn (expected() của dung_bo, theo mức đích trọng số A); ghi cả SNR thật trọng số A và toàn dải. Không gán tay.

  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v3.py build [--limit N] [--bo p1_dieukien_v3]
  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v3.py kiem [--bo …]      (kiểm độc lập đáp án trên tệp đầu ra)
  .venv/Scripts/python.exe experts/P1/build_cond/dung_bo_v3.py selftest
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dung_bo as D  # noqa: E402
from cham_dieu_kien import NAT, V6, fold, jl  # noqa: E402
from do_tram import LOC  # noqa: E402

SEED = 2417
SNR_A = (0, 3, 5, 7, 9, 10, 11, 13, 15, 20)
NEN_A = (5, 8, 10, 12, 15, 20)
LOAI = ("nguoi_noi_1", "quan", "nhac_loa", "tv", "ngat_quang", "moi_truong")
NGOAI_TRUC = {"nguoi_noi_1", "quan", "tv"}         # có giọng người khác → hàng số người nói / nói chồng có thể bật
MOI_O, MOI_O_NEN, GOC_MOI_LANG = 10, 12, 20
MAX_S, MIN_SAN_A = 300.0, 25.0
DEMAND_QUAN = ("PCAFETER", "PRESTO", "SPSQUARE", "OMEETING")
RIRS = os.path.join(D.RAW, "rirs_noises", "RIRS_NOISES", "real_rirs_isotropic_noises")
A = LOC["a"]


# ----------------------------------------------------------------------------- hàm thuần
def ea(x: np.ndarray) -> float:
    return D.energy(A(x))


def gain_a(s: np.ndarray, n: np.ndarray, snr_db: float) -> float:
    """g sao cho 10·log10(E_A(s) / E_A(g·n)) = snr_db."""
    return float(np.sqrt(ea(s) / (ea(n) * 10 ** (snr_db / 10))))


def gain_ratio_a(s: np.ndarray, m: np.ndarray, r: float) -> float:
    """g sao cho E_A(g·m) / (E_A(s) + E_A(g·m)) = r."""
    return float(np.sqrt(r / (1 - r) * ea(s) / ea(m)))


def san_a(s: np.ndarray) -> float | None:
    """SNR sẵn có của gốc trên tín hiệu trọng số A (cùng công thức native_snr)."""
    return D.native_snr(A(s))


def snr_that(s: np.ndarray, n: np.ndarray, san: float | None, f) -> float:
    do = 10 * np.log10(D.energy(f(s)) / max(D.energy(f(n)), 1e-30))
    return float(D.combine_snr(do, san)) if san is not None else float(do)


def ngat_quang(clips: list, n: int, rng: random.Random) -> np.ndarray:
    """Rải đoạn 0,3–2 s lấy từ clips, xen khoảng lặng 0,5–3 s; ít nhất 2 đoạn (tệp ngắn thì rút ngắn khoảng lặng)."""
    out, i, k = np.zeros(n), int(rng.uniform(0.2, 1.0) * D.SR), 0
    while i < n or k < 2:
        c = clips[rng.randrange(len(clips))]
        L = min(int(rng.uniform(0.3, 2.0) * D.SR), len(c))
        if i >= n:                                          # chưa đủ 2 đoạn: đặt lại vào đầu
            i = int(rng.uniform(0, max(1, n - L)))
        st = rng.randrange(max(1, len(c) - L))
        seg = c[st:st + L] * np.hanning(L) ** 0.1           # vuốt mép rất nhẹ, tránh click
        e = min(n, i + L)
        out[i:e] += seg[:e - i]
        i, k = e + int(rng.uniform(0.5, 3.0) * D.SR), k + 1
    return out


def kiem_lech(y: np.ndarray, s: np.ndarray, san: float | None) -> float:
    """Kiểm độc lập trên tệp đầu ra: y = k·s + n (bình phương tối thiểu) → SNR A có bù sàn."""
    k = float(np.dot(y, s) / np.dot(s, s))
    return snr_that(k * s, y - k * s, san, A)


# ----------------------------------------------------------------------------- nguồn
def goc_that(lang: str) -> list[dict]:
    """Gốc thật rất sạch từ p1_tunhien_v1 theo luật chốt (NHAT_KY 2.16)."""
    from cham_tram import cases_of
    from cham_vang import gop
    cn, mn = cases_of(NAT), gop(NAT)
    k1 = {r["file"]: r["decision"] for r in jl(os.path.join(V6, NAT, "_kiem", "k1_ket_qua.jsonl"))}
    nhac = set(re.findall(r"\| (\S+) \| \d+ (?:vi|en) \|", open(os.path.join(V6, NAT, "_kiem", "QUET_NHAC_2411.md"), encoding="utf-8").read()))
    man = {f"{r['meta']['lang']}/{r['id']}.wav": r for r in json.load(open(os.path.join(V6, NAT, "manifest.json"), encoding="utf-8"))}
    out = []
    for f, c in sorted(cn.items()):
        m = mn.get(f)
        if not f.startswith(lang + "/") or c["mong_doi"] != "accept" or m is None or k1.get(f) == "04" or m["bh"] is None:
            continue
        if man[f]["meta"]["recording"] in nhac or m["snr_nghi"] is None:
            continue
        if m["bh"] >= 25 and m["snr_nghi"] >= 25 and m["loi_bh"] >= 0.7 and m["c50"] >= 20:
            out.append({"id": f"p1n:{f}", "wav": os.path.join(V6, NAT, f), "source": "yt", "license": man[f]["meta"]["license"],
                        "person": c["base"], "kenh": c["base"], "file": f})
    return out


def goc_fleurs(lang: str, n: int, rng: random.Random) -> list[dict]:
    return [{"id": f"fleurs_{lang}_{b['split']}_{b['file'][:-4]}", "file": b["file"], "wav": b["wav"], "source": "fleurs",
             "license": "CC BY 4.0", "kenh": None} for b in D.fleurs_bases(f"{lang}_{'vn' if lang == 'vi' else 'us'}", n, rng)]


def rir_that() -> list[str]:
    """RIR thật (REVERB 2014 type1, AIR, RWCP có vang); bỏ phòng câm (ane) và RIR mô phỏng (type2 simroom)."""
    out = []
    for ln in open(os.path.join(RIRS, "rir_list"), encoding="utf-8"):
        p = ln.split()[-1].split("/")[-1]
        if "_ane" in p or "simroom" in p:
            continue
        q = os.path.join(RIRS, p)
        if os.path.exists(q):
            out.append(q)
    return out


def on_phong_that() -> list[str]:
    return [os.path.join(RIRS, ln.split()[-1].split("/")[-1]) for ln in open(os.path.join(RIRS, "noise_list"), encoding="utf-8")
            if os.path.exists(os.path.join(RIRS, ln.split()[-1].split("/")[-1]))]


def fleurs_cau(lang: str) -> list[str]:
    d = os.path.join(D.RAW, "fleurs", f"{lang}_{'vn' if lang == 'vi' else 'us'}")
    return sorted(os.path.join(d, sp, f) for sp in ("dev", "test") if os.path.isdir(os.path.join(d, sp))
                  for f in os.listdir(os.path.join(d, sp)) if f.endswith(".wav"))


def entry(path: str, lic: str, src: str, **kw) -> dict:
    return {"path": D.rel(path), "sha256": D.sha256(path), "giay_phep": lic, "nguon": src, **kw}


# ----------------------------------------------------------------------------- dựng nền
_CACHE: dict = {}


def nap(path: str) -> np.ndarray:
    """D.load có nhớ đệm nhỏ (nguồn ồn / nhạc / RIR dùng lại nhiều lần)."""
    if path not in _CACHE:
        if len(_CACHE) > 48:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[path] = D.load(path)
    return _CACHE[path]


def _loop(path: str, n: int, rng: random.Random) -> tuple[np.ndarray, int]:
    off = rng.randrange(1 << 30)
    return D.fit_len(nap(path), n, off), off


def _qua_rir(x: np.ndarray, rir: str) -> np.ndarray:
    return D.reverb(x, nap(rir))


def lam_nen(loai: str, lang: str, n: int, rng: random.Random, P: dict, tru: str) -> tuple[np.ndarray, list]:
    """Tín hiệu nền dài n mẫu + danh sách nguồn (provenance). tru = tệp gốc (không lấy chính nó làm người nói khác)."""
    if loai == "nguoi_noi_1":
        p = rng.choice([q for q in P["fleurs"][lang] if q != tru])
        r = None
        while r is None:
            r = D.sim_rir(np.random.default_rng(rng.randrange(1 << 30)), rng.uniform(0.3, 0.6))
        x, off = _loop(p, n, rng)
        return D.reverb(x, r["h"]), [entry(p, "CC BY 4.0", "FLEURS", offset=off),
                                      {"nguon": "pyroomacoustics/ShoeBox", **{k: v for k, v in r.items() if k != "h"}}]
    if loai == "quan":
        if rng.random() < 0.5:
            p = os.path.join(D.RAW, "demand", rng.choice(DEMAND_QUAN), "ch01.wav")
            x, off = _loop(p, n, rng)
            return x, [entry(p, "CC BY 4.0", "DEMAND/" + p.split(os.sep)[-2], offset=off)]
        ps = rng.sample([q for q in P["fleurs"][lang] if q != tru], 5)
        parts = [(_loop(q, n, rng), 10 ** (rng.uniform(-3, 3) / 20)) for q in ps]
        return sum(g * x for (x, _), g in parts), [entry(q, "CC BY 4.0", "FLEURS/babble", offset=o) for q, ((_, o), _) in zip(ps, parts)]
    if loai in ("nhac_loa", "tv"):
        rir = rng.choice(P["rir"])
        mp = rng.choice(P["music"])
        mu, om = _loop(mp["path"], n, rng)
        src = [D.src_entry(mp, offset=om)]
        if loai == "tv":
            sp = rng.choice(P["speech"])
            x, osp = _loop(sp["path"], n, rng)
            mu = mu * np.sqrt(D.energy(x) / max(D.energy(mu), 1e-30) / 10)      # nhạc thấp hơn lời 10 dB
            mu, src = x + mu, [D.src_entry(sp, offset=osp)] + src
        return _qua_rir(mu, rir), src + [entry(rir, "Apache 2.0 (RIRS_NOISES)", "RIRS_NOISES/real_rir")]
    if loai == "ngat_quang":
        ps = rng.sample(P["noise"], 6)
        return ngat_quang([nap(p["path"]) for p in ps], n, rng), [D.src_entry(p) for p in ps]
    if loai == "moi_truong":
        if rng.random() < 0.5:
            p = rng.choice(P["demand"])
            x, off = _loop(p["path"], n, rng)
            return x, [D.src_entry(p, offset=off)]
        p = rng.choice(P["phong"])
        x, off = _loop(p, n, rng)
        return x, [entry(p, "Apache 2.0 (RIRS_NOISES)", "RIRS_NOISES/isotropic_noise", offset=off)]
    raise ValueError(loai)


# ----------------------------------------------------------------------------- dựng bộ
def nap_goc(b: dict) -> np.ndarray:
    s = D.load(b["wav"])
    return s[: int(MAX_S * D.SR)]


def chon_goc(lang: str, rng: random.Random) -> tuple[list, list, dict]:
    """(gốc thật, gốc FLEURS, thống kê bỏ). Đo sàn trọng số A một lần / gốc; bỏ gốc < MIN_SAN_A."""
    that, fl, bo = [], [], {"that": 0, "fleurs": 0}
    for ten, ds, out in (("that", goc_that(lang), that), ("fleurs", goc_fleurs(lang, 40, rng), fl)):
        for b in ds:
            s = nap_goc(b)
            sa = san_a(s)
            if sa is None or sa < MIN_SAN_A:
                bo[ten] += 1
                continue
            out.append({**b, "san_a": round(sa, 2), "san_full": D.native_snr(s), "cat_s": round(len(s) / D.SR, 2)})
    return that, fl, bo


def lay(that: list, fl: list, k_that: int, k_fl: int, rng: random.Random) -> list:
    """k_that gốc thật ở k_that kênh khác nhau + k_fl gốc FLEURS."""
    kenh: dict = {}
    for b in that:
        kenh.setdefault(b["kenh"], []).append(b)
    ks = rng.sample(sorted(kenh), k_that)
    return [rng.choice(kenh[k]) for k in ks] + rng.sample(fl, k_fl)


def base_w(b: dict) -> dict:
    """Bản ghi gốc cho Writer: gốc thật id = kênh (fold theo người nói); FLEURS id = mã tệp."""
    if b["source"] == "yt":
        return {"id": b["kenh"], "wav": b["wav"], "source": "yt", "license": b["license"], "person": b["kenh"], "file": b["file"]}
    return {"id": b["id"], "wav": b["wav"], "source": "fleurs", "license": "CC BY 4.0", "file": b["file"]}


def viec(lang: str, that: list, fl: list, rng: random.Random) -> list:
    """Danh sách việc (truc, ca, mức, loại nền, gốc) cho một ngôn ngữ."""
    out = [("goc", "goc", 0, None, b) for b in lay(that, fl, GOC_MOI_LANG // 2, GOC_MOI_LANG // 2, rng)]
    for loai in LOAI:
        for lv in SNR_A:
            out += [("snr", f"snr_{loai}.{lv}dB", lv, loai, b) for b in lay(that, fl, MOI_O // 2, MOI_O // 2, rng)]
    for pct in NEN_A:
        out += [("nen", f"nen_nhac_loa.{pct}pct", pct, "nhac_loa", b) for b in lay(that, fl, MOI_O_NEN // 2, MOI_O_NEN // 2, rng)]
    return out


def mot_tep(w, job: tuple, lang: str, rng: random.Random, P: dict) -> None:
    truc, ca, lv, loai, b = job
    s = nap_goc(b)
    ts = {"goc_tep": b["file"], "cat_s": b["cat_s"], "san_a_db": b["san_a"], "san_full_db": b["san_full"]}
    if truc == "goc":
        w.add("goc", "goc", 0, s, base_w(b), lang, [], ts, {})
        return
    n, nguon = lam_nen(loai, lang, len(s), rng, P, b["wav"])
    ts = {**ts, "loai_nen": loai, "ngoai_truc": loai in NGOAI_TRUC}
    if truc == "snr":
        add = D.snr_add_for(lv, b["san_a"])
        g = gain_a(s, n, add)
        do = {"snr_that_A_db": round(snr_that(s, g * n, b["san_a"], A), 3),
              "snr_that_full_db": round(snr_that(s, g * n, b["san_full"], lambda x: x), 3)}
        ts = {**ts, "snr_dich_A_db": lv, "snr_them_A_db": round(add, 3), "g": g}
    else:
        g = gain_ratio_a(s, n, lv / 100)
        do = {"nen_A_pct": round(100 * ea(g * n) / (ea(s) + ea(g * n)), 3),
              "nen_full_pct": round(100 * D.energy(g * n) / (D.energy(s) + D.energy(g * n)), 3)}
        ts = {**ts, "nen_dich_A_pct": lv, "g": g}
    w.add(truc, ca, lv, s + g * n, base_w(b), lang, nguon, ts, do)


def nguon_chung() -> dict:
    sp = [p for p in D.musan_pool("speech", 10.0) if "us-gov" in p["path"] or ("librivox" in p["path"] and p["ann"] == "english")]
    return {"music": D.musan_pool("music", 10.0), "speech": sp, "noise": D.musan_pool("noise", 1.0), "demand": D.demand_pool(),
            "rir": rir_that(), "phong": on_phong_that(), "fleurs": {lg: fleurs_cau(lg) for lg in ("vi", "en")}}


def the_bo(w, bo: dict, seed: int, limit) -> None:
    import collections
    import datetime
    by = collections.Counter((c["ca"].split(".")[0], c["mong_doi"]) for c in w.cases)
    ng = collections.Counter(m["meta"]["source"] for m in w.manifest)
    L = [f"# Thẻ bộ `{D.BO}`", "",
         f"- **Ngày dựng:** {datetime.date.today().isoformat()} · **Ai làm:** Huấn (P1); mã do Claude Code (Claude Opus 5.5) viết theo thiết kế Huấn duyệt 24/11",
         f"- **Cách làm:** `experts/P1/build_cond/dung_bo_v3.py` (seed {seed}{', limit ' + str(limit) if limit else ''}). Luật: `lanes/P1/NHAT_KY/NHAT_KY_P1.md` 2.16 «Bộ p1_dieukien_v3». "
         "Gốc sạch + nền có tham số biết trước; đáp án từ `dung_bo.expected()` theo **mức đích SNR / nền trọng số A**; ghi kèm SNR / nền thật trọng số A và toàn dải. Không gán tay.",
         "- **Nguồn + giấy phép:** gốc thật `p1_tunhien_v1` (YouTube CC BY, kênh không chuyên, person = kênh) · FLEURS vi_vn / en_us (CC BY 4.0) · "
         "DEMAND (CC BY 4.0) · MUSAN music / speech / noise (CC BY / BY-SA / PD; bỏ NC / ND, bỏ thánh ca) · RIRS_NOISES RIR thật + ồn phòng thật (Apache 2.0) · RIR mô phỏng pyroomacoustics. Từng tệp: `provenance.jsonl`.",
         f"- **Gốc bị bỏ vì sàn trọng số A < {MIN_SAN_A:g} dB:** {bo}",
         f"- **Số tệp:** {len(w.cases)} · nguồn gốc: " + ", ".join(f"{k} {v}" for k, v in sorted(ng.items()))
         + f" · số person: {len({m['meta']['person'] for m in w.manifest})}",
         "- **Hạn chế:** FLEURS không có mã người nói (person = từng tệp) · loại `nguoi_noi_1`, `quan`, `tv` có giọng người khác: hàng số người nói / "
         "nói chồng có thể bật (`tham_so.ngoai_truc`), chỉ chấm hàng ồn / nền · gốc thật dài cắt còn ≤ 300 s.",
         "- **Đã kiểm:** *(điền sau)* `kiem_bo --moi` · `lockbox check` · `dung_bo_v3.py kiem` (đáp án đo lại độc lập)", "",
         "| nhóm ca | đáp án | số tệp |", "|---|---|---|"]
    L += [f"| {ca} | {md} | {k} |" for (ca, md), k in sorted(by.items())]
    open(os.path.join(D.OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def build(bo: str, limit: int | None) -> None:
    D.BO, D.OUT = bo, os.path.join(D.ROOT, "data_labels", "v6", bo)
    if os.path.exists(os.path.join(D.OUT, "manifest.json")):
        raise SystemExit(f"{D.rel(D.OUT)} đã có manifest.json: không ghi đè")
    rng = random.Random(SEED)
    P = nguon_chung()
    print({k: (len(v) if isinstance(v, list) else {a: len(b) for a, b in v.items()}) for k, v in P.items()}, flush=True)
    jobs, bo_stat = [], {}
    for lang in ("vi", "en"):
        that, fl, bo_stat[lang] = chon_goc(lang, rng)
        print(f"{lang}: gốc thật {len(that)} ({len({b['kenh'] for b in that})} kênh) · FLEURS {len(fl)} · bỏ {bo_stat[lang]}", flush=True)
        jobs += [(lang, j) for j in viec(lang, that, fl, rng)]
    if limit:
        jobs = rng.sample(jobs, limit)
    w = D.Writer(D.OUT, {**D.script_version(), "seed": SEED})
    w.ver = {**w.ver, "script": D.rel(os.path.abspath(__file__)), "script_sha256": D.sha256(os.path.abspath(__file__))}
    for i, (lang, j) in enumerate(jobs, 1):
        mot_tep(w, j, lang, rng, P)
        if i % 50 == 0:
            print(f"  {i}/{len(jobs)}", flush=True)
    w.flush(rng)
    the_bo(w, bo_stat, SEED, limit)
    print(f"{len(w.manifest)} tệp → {D.rel(D.OUT)}")


def kiem(bo: str) -> None:
    """Kiểm độc lập đáp án trên tệp đầu ra (luật: lệch > 0,3 dB / 0,3 điểm % thì báo)."""
    out = os.path.join(D.ROOT, "data_labels", "v6", bo)
    rows, lech = list(jl(os.path.join(out, "provenance.jsonl"))), []
    for p in rows:
        if p["ca"] == "goc":
            continue
        y, _ = sf.read(os.path.join(D.ROOT, p["file"]), dtype="float64")
        s = D.load(os.path.join(D.ROOT, p["goc"]["path"]))[: len(y)]
        ts, dl = p["tham_so"], p["do_lai"]
        if "snr_that_A_db" in dl:
            v, ref = kiem_lech(y, s, ts["san_a_db"]), dl["snr_that_A_db"]
        else:
            k = float(np.dot(y, s) / np.dot(s, s))
            v, ref = 100 * ea(y - k * s) / ea(y), dl["nen_A_pct"]
        if abs(v - ref) > 0.3:
            lech.append((p["id"], p["ca"], round(v, 2), ref))
    L = [f"# Kiểm đáp án `{bo}` (đo lại độc lập trên tệp đầu ra)", "",
         f"- Tệp snr / nen: {sum(p['ca'] != 'goc' for p in rows)} · lệch > 0,3: **{len(lech)}**"] + [f"- {x}" for x in lech[:40]]
    open(os.path.join(out, "_kiem_dap_an.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))


def selftest() -> None:
    rng = np.random.default_rng(0)
    s = rng.standard_normal(D.SR * 4)
    n = rng.standard_normal(D.SR * 4) * 3
    g = gain_a(s, n, 10.0)
    assert abs(10 * np.log10(ea(s) / ea(g * n)) - 10.0) < 1e-6
    g2 = gain_ratio_a(s, n, 0.1)
    assert abs(ea(g2 * n) / (ea(s) + ea(g2 * n)) - 0.1) < 1e-9
    assert abs(kiem_lech(0.5 * s + g * n, s, None) - (10.0 + 20 * np.log10(0.5))) < 0.3   # giọng × 0,5 → SNR giảm 6 dB
    assert abs(snr_that(s, g * n, 25.0, A) - D.combine_snr(10.0, 25.0)) < 1e-6
    r = random.Random(1)
    for L_s in (3.0, 5.0, 20.0):                         # gốc ngắn nhất 3 s (tệp < 1 s thì 2 đoạn phủ kín, ngoài phạm vi)
        t = ngat_quang([rng.standard_normal(D.SR)], int(L_s * D.SR), r)
        assert len(t) == int(L_s * D.SR) and (t != 0).mean() < 0.95 and (t != 0).any()
    print("dung_bo_v3 selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("build", "kiem", "selftest"))
    ap.add_argument("--bo", default="p1_dieukien_v3")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"build": lambda: build(a.bo, a.limit), "kiem": lambda: kiem(a.bo), "selftest": selftest}[a.cmd]()
