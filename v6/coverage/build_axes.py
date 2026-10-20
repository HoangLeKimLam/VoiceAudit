# -*- coding: utf-8 -*-
"""v6e / C1 - BO PHU DIEU KIEN BIEN cua Constraint, ap len tep goc TU NHIEN co nhan day du (khong doi nhan, chi doi dieu kien).
Co so: tep goc natvi / naten (cfg goc, trong pham vi, khong lockbox): 18 that + 12 tep moi nhom [nen_toan_cuc · giong_toan_cuc ·
nen_cuc_bo · giong_cuc_bo · ghep (thay/chen, vung-trung binh-ky) · xoa_loi (vung-trung binh-ky) · da_thao_tac] chia deu Viet / Anh.
Moi tep co so qua MOI dieu kien (cfg 'goc', meta.cond = ten dieu kien, meta.axis = {truc, gia tri}, meta.base = id co so):
  level     muc tieng noi trung binh -38 / -32 / -26 dBFS                    (Constraint >= -40 dBFS)
  snr       tieng dong that (free-sound) cong them o SNR 10 / 12 / 15 / 25 dB (>= 10 dB)
  nhac      nhac nen tong hop (hoa am + nhip) 5 % / 10 % nang luong          (am khong phai tieng noi <= 10 %)
  tv        giong nguoi KHAC (lam TV) 8 % nang luong
  vang      RT60 0,3 / 0,6 / 1,0 s                                             (<= 1,0 s)
  catdinh   0,5 % / 1,0 % mau cham muc toi da                                  (<= 1 %)
  matTH     2 doan 0 dai 0,05 / 0,10 s ngoai vung sua                          (<= 0,1 s)
  bangtan   cat thong thap 6,0 / 6,5 / 7,0 kHz                                 (nang luong toi >= 6 kHz)
  tocdo     phat nhanh / cham -12 / -5 / +5 / +12 % (doi ca cao do; moc nhan doi theo)   (+-5 %, 5-12 % khong canh bao)
  matdo     them nen phong lam tieng noi chi con 30 % / 40 % do dai            (>= 30 %)
Moc nhan (start / end / joint) bien doi dung theo tung dieu kien. Cac truc chi anh huong K1 (dinh dang, tan so lay mau, do sau bit,
hai kenh, do dai 5-10 phut) lam o build_k1_axes.py. Ghi vao data_labels/v6/cover (NAT_LANG=cover), id 'v6cover_'.
  NAT_LANG=cover python -m v6.coverage.build_axes [--per-group 12] [--seed 21]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from fractions import Fraction

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.environ.setdefault("NAT_LANG", "cover")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import butter, resample_poly, sosfiltfilt  # noqa: E402

from v6.natural import build_natural as B  # noqa: E402

SR = 16000
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
GROUPS = {"that": ["nguyen_ban"], "nen_toan": ["nen_toan_cuc"], "giong_toan": ["giong_toan_cuc"], "nen_cuc_bo": ["nen_cuc_bo", "nen_cuc_bo_ky"],
          "giong_cuc_bo": ["giong_cuc_bo", "giong_cuc_bo_ky"],
          "ghep": ["ghep_thay", "ghep_chen", "ghep_thay_vung", "ghep_chen_vung", "ghep_thay_ky", "ghep_chen_ky"],
          "xoa": ["xoa_loi", "xoa_loi_vung", "xoa_loi_ky"], "da": ["da_ghep_nen", "da_ghep_giong", "da_hai_cho"]}
CONDS = ([("level", v) for v in (-38, -32, -26)] + [("snr", v) for v in (10, 12, 15, 25)] + [("nhac", v) for v in (5, 10)] +
         [("tv", 8)] + [("vang", v) for v in (0.3, 0.6, 1.0)] + [("catdinh", v) for v in (0.5, 1.0)] +
         [("matTH", v) for v in (0.05, 0.10)] + [("bangtan", v) for v in (6000, 6500, 7000)] +
         [("tocdo", v) for v in (-12, -5, 5, 12)] + [("matdo", v) for v in (30, 40)])


def db(x: float) -> float:
    return 20 * np.log10(max(float(x), 1e-9))


def speech_frames(y: np.ndarray) -> np.ndarray:
    fr = np.sqrt(np.convolve(y ** 2, np.ones(320) / 320, "valid")[::160] + 1e-12)
    return fr >= fr.max() * 10 ** (-30 / 20)


def speech_rms(y: np.ndarray) -> float:
    m = speech_frames(y)
    fr = np.sqrt(np.convolve(y ** 2, np.ones(320) / 320, "valid")[::160] + 1e-12)
    return float(np.sqrt(np.mean(fr[m] ** 2))) if m.any() else float(np.sqrt(np.mean(y ** 2)))


def mix(y: np.ndarray, n: np.ndarray, rel_db: float) -> np.ndarray:
    """Cong n vao y sao cho nang luong n (tren do dai y) thap hon tieng noi rel_db dB."""
    n = np.resize(n, len(y))
    return y + n * (speech_rms(y) / (np.sqrt(np.mean(n ** 2)) + 1e-9)) * 10 ** (-rel_db / 20)


def music(n: int, rng: np.random.Generator) -> np.ndarray:
    """Nhac nen tong hop: hoa am 3 am thay moi 1,5 s + xung nhip 2 Hz; chi la gia tri dai dien (khong co nhac that)."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    roots = [220.0, 261.6, 196.0, 246.9]
    for k in range(int(np.ceil(n / SR / 1.5))):
        a, b = int(k * 1.5 * SR), min(n, int((k + 1) * 1.5 * SR))
        r = roots[int(rng.integers(len(roots)))]
        for m_ in (1.0, 1.26, 1.5):
            for h in (1, 2, 3):
                out[a:b] += np.sin(2 * np.pi * r * m_ * h * t[a:b]) / h
    out *= 0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 2 * t)) ** 4
    return out


def room_tone(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    fr = np.sqrt(np.convolve(y ** 2, np.ones(320) / 320, "valid")[::160] + 1e-12)
    idx = np.argsort(fr)[:max(8, int(0.15 * len(fr)))]
    pieces = [y[i * 160:i * 160 + 320] for i in rng.permutation(idx)]
    bed = np.concatenate(pieces * (n // (320 * len(pieces)) + 2))
    return bed[:n]


def shift_edits(edits: list, f) -> list:
    out = []
    for e in edits:
        x = dict(e)
        for k in ("start", "end", "joint"):
            if x.get(k) is not None:
                x[k] = round(float(f(x[k])), 4)
        out.append(x)
    return out


def edit_spans(edits: list) -> list:
    return [(x.get("joint", x["start"]) - 0.5, x.get("joint", x["end"]) + 0.5) if x.get("joint") is not None else (x["start"] - 0.5, x["end"] + 0.5)
            for x in edits if x.get("kind") == "segment"]


def apply(name: str, v, y: np.ndarray, edits: list, pool: list, rng: np.random.Generator):
    """Tra (y moi, edits moi, tham so thuc te) hoac None neu dieu kien khong ap duoc."""
    f = lambda t: t  # noqa: E731
    info = {}
    if name == "level":
        y = y * (10 ** (v / 20) / (speech_rms(y) + 1e-9))
    elif name == "snr":
        y = mix(y, B_noise(len(y), rng), v)
    elif name == "nhac":
        # v % nang luong so voi tong: ty so nang luong n / (y + n) = v / 100
        n = music(len(y), rng)
        n = n * np.sqrt(v / 100 / (1 - v / 100) * np.mean(y ** 2) / (np.mean(n ** 2) + 1e-12))
        y = y + n
    elif name == "tv":
        o = np.resize(pool[int(rng.integers(len(pool)))], len(y))
        o = o * np.sqrt(v / 100 / (1 - v / 100) * np.mean(y ** 2) / (np.mean(o ** 2) + 1e-12))
        y = y + o
    elif name == "vang":
        random.seed(int(rng.integers(1 << 30)))
        from test_simulate import make_room
        rir = np.asarray(make_room(SR, v), float)
        k = int(np.argmax(np.abs(rir)))
        z = np.convolve(y, rir)[k:k + len(y)]
        y = z * (np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(z ** 2)) + 1e-9))
        info["rt60"] = v
    elif name == "catdinh":
        a = np.abs(y)
        t = float(np.quantile(a, 1 - v / 100))
        y = np.clip(y, -t, t)
        info["muc_cat"] = round(t, 4)
    elif name == "matTH":
        L = int(round(v * SR))
        bad = edit_spans(edits)
        tries, pos = 0, []
        while len(pos) < 2 and tries < 200:
            tries += 1
            s = int(rng.uniform(1.0, len(y) / SR - 1.0 - v) * SR) if len(y) / SR > 2.5 + v else -1
            if s < 0:
                return None
            if all(not (s / SR < b and s / SR + v > a) for a, b in bad) and all(abs(s - p) > SR for p in pos):
                pos.append(s)
        if len(pos) < 2:
            return None
        y = y.copy()
        for s in pos:
            y[s:s + L] = 0.0
        info["vi_tri_s"] = [round(p / SR, 3) for p in pos]
    elif name == "bangtan":
        y = sosfiltfilt(butter(8, v, "low", fs=SR, output="sos"), y)
    elif name == "tocdo":
        s = 1 + v / 100
        fr = Fraction(s).limit_denominator(100)
        y = resample_poly(y, fr.denominator, fr.numerator)
        f = lambda t, s=s: t / s  # noqa: E731
        info["he_so"] = s
    elif name == "matdo":
        sp = float(speech_frames(y).sum() * 0.01)
        total = sp / (v / 100)
        pad = total - len(y) / SR
        if pad <= 0.5:
            return None
        a = int(pad / 2 * SR)
        bed = room_tone(y, int(pad * SR), rng)
        y = np.r_[bed[:a], y, bed[a:]]
        f = lambda t, a=a: t + a / SR  # noqa: E731
        info["them_s"] = round(pad, 2)
    else:
        raise ValueError(name)
    return y, shift_edits(edits, f), info


def B_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    random.seed(int(rng.integers(1 << 30)))
    np.random.seed(int(rng.integers(1 << 30)))
    from test_simulate import load_noise
    return np.asarray(load_noise(SR, n), float)


def pick_base(per_group: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for lang in ("natvi", "naten"):
        man = [e for e in json.load(open(os.path.join(BASE, "data_labels", "v6", lang, "manifest.json"), encoding="utf-8"))
               if e["cfg"] == "goc" and not e["meta"].get("ngoai_pham_vi") and not e["meta"].get("lockbox")
               and e["meta"].get("k1", {}).get("decision") not in ("04", "tu_choi", "loi_k1")]
        for g, kinds in GROUPS.items():
            pool = [e for e in man if e["kind"] in kinds]
            n = (18 if g == "that" else per_group) // 2
            out += rng.sample(pool, min(n, len(pool)))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=12)
    ap.add_argument("--seed", type=int, default=21)
    a = ap.parse_args()
    os.makedirs(B.OUTD, exist_ok=True)
    base = pick_base(a.per_group, a.seed)
    print(f"{len(base)} tep co so x {len(CONDS)} dieu kien", flush=True)
    pool = [sf.read(os.path.join(V4, e["wav"]), dtype="float64")[0] for e in base[:30]]
    man = B._load_man()
    have = {e["meta"]["base"] + "|" + e["meta"]["cond"] for e in man}
    counter = {}
    for bi, e in enumerate(base):
        y0, _ = sf.read(os.path.join(V4, e["wav"]), dtype="float64")
        for name, v in CONDS:
            cond = f"{name}_{v}"
            if f"{e['id']}|{cond}" in have:
                continue
            rng = np.random.default_rng(abs(hash((e["id"], cond))) % (2 ** 32))
            r = apply(name, v, y0.copy(), e["meta"].get("edits", []), pool, rng)
            if r is None:
                continue
            y, edits, info = r
            if not np.isfinite(y).all():
                continue
            meta = {**e["meta"], "edits": edits, "cond": cond, "axis": {"truc": name, "gia_tri": v, **info}, "base": e["id"],
                    "base_kind": e["kind"], "kenh": "goc", "tier": e["meta"].get("muc_che_giau", "trung_binh")}
            for k in ("k1", "fold"):
                meta.pop(k, None)
            counter[cond] = counter.get(cond, 0) + 1
            vid = f"{B.ID}{e['kind']}__{cond}__{bi:03d}".replace(".", "p").replace("-", "m")
            man.append(B._write(vid, e["kind"], "goc", np.asarray(y, np.float32), meta, [meta["speaker"]], vid))
        if (bi + 1) % 10 == 0:
            B._save_man(man)
            print(f"  {bi + 1}/{len(base)} tep co so, tong {len(man)}", flush=True)
    B._save_man(man)
    print(f"xong: {len(man)} tep", {k: v for k, v in sorted(counter.items())}, flush=True)


if __name__ == "__main__":
    main()
