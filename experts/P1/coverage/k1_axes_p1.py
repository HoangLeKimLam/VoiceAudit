# -*- coding: utf-8 -*-
"""P1 · Test nhóm A (các hàng tệp / tín hiệu của Constraint) = v6.coverage.k1_axes chạy trên tệp gốc P1 tự chọn (Huấn duyệt 25/11).

Không sửa k1_axes.py. Tệp này gán lại OUT / pick / _bases_by_id / _musan_noise / CASES của module, rồi gọi build / run gốc.

Tệp gốc: ghi âm thật p1_tunhien_v1, chỉ kênh fold dev (crc32 % 5 < 3).
  - Đạt bộ lọc sạch dung_bo_v3.goc_that: Brouhaha SNR ≥ 25, snr lúc nghỉ ≥ 25, tỉ lệ lời ≥ 0,7, C50 ≥ 20, không nhạc nền.
  - Thêm lọc ĐỘC LẬP: nền DFN trọng số A (nen_a) ≤ 3 %.
  - Lọc như k1_axes.pick: K1 nhận / cảnh báo, ≥ 8 s, 1 người nói, băng thông ≥ 7 kHz, âm lượng ≥ −35 dBFS, cắt đỉnh < 0,5 %.
  - Mỗi kênh một tệp; 12 vi + 12 en; xếp xen kẽ vi / en để ca dài (10 gốc đầu) có 5 vi + 5 en.
Ồn MUSAN (ca loai.chi_tieng_on): chỉ giấy phép CC BY / BY-SA / CC0 / PD (dung_bo.musan_pool).
Ca thêm (lượng tiếng nói, ngay NGOÀI vùng đệm 1,6–2,6 s / 24–36 % của k1_axes.expect): 1,4 s / 3,0 s lời; 20 % / 40 % lời.

  python experts/P1/coverage/k1_axes_p1.py selftest
  python experts/P1/coverage/k1_axes_p1.py build [--bases 24] [--long-bases 10] [--only a,b]
  python experts/P1/coverage/k1_axes_p1.py run [--tag p1_v1] [các cờ khác của k1_axes run]
  python experts/P1/coverage/k1_axes_p1.py tom [--tag p1_v1]     bảng k/n + Wilson 95 % theo hàng × ngôn ngữ
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "experts", "P1", "build_cond")]
from v6.coverage import k1_axes as K  # noqa: E402

SEED = 2418
N_BASE, N_LONG = 24, 10
NEN_A_MAX = 3.0
OUT = os.path.join(ROOT, "data_labels", "v6", "k1axes", "p1_v1")
GOC_JSON = os.path.join(OUT, "goc.json")
NAT_K1 = os.path.join(ROOT, "data_labels", "v6", "p1_tunhien_v1", "_kiem", "k1_ket_qua.jsonl")


# ------------------------------------------------------------------ tệp gốc
def _k1_ok(r: dict) -> bool:
    m = r.get("measures", {})
    ch = (m.get("kenh") or [{}])[0]
    return (r["decision"] in ("nhan", "canh_bao") and m.get("do_dai_s", 0) >= 8 and ch.get("so_nguoi_noi") in (None, 1)
            and (ch.get("bang_thong_hz") or 0) >= 7000 and (ch.get("am_luong_dbfs") or -99) >= -35 and (ch.get("cat_dinh_pct") or 0) < 0.5)


def pick(n: int = N_BASE) -> list:
    import dung_bo_v3 as V
    from cham_dieu_kien import NAT, fold
    from cham_vang import gop
    mn = gop(NAT)
    k1 = {r["file"]: r for r in map(json.loads, open(NAT_K1, encoding="utf-8"))}
    rng = random.Random(SEED)
    per_lang, used = {}, set()                     # một kênh có thể nói cả vi lẫn en → mỗi kênh chỉ một tệp trên cả bộ
    for lang, k in (("vi", (n + 1) // 2), ("en", n // 2)):
        by_ch = defaultdict(list)
        for b in V.goc_that(lang):
            f = b["file"]
            if fold(b["kenh"]) < 3 and (mn[f].get("nen_a") is not None and mn[f]["nen_a"] <= NEN_A_MAX) and f in k1 and _k1_ok(k1[f]):
                by_ch[b["kenh"]].append(b)
        chans = sorted(set(by_ch) - used)
        if len(chans) < k:
            raise RuntimeError(f"{lang}: chỉ {len(chans)} kênh đạt lọc, cần {k}")
        got = []
        for i, ch in enumerate(rng.sample(chans, k)):
            b = rng.choice(sorted(by_ch[ch], key=lambda x: x["file"]))
            got.append({"id": f"p1k1{lang}_{i:02d}", "wav": b["wav"],
                        "meta": {"lang": lang, "person": ch, "source": "yt", "file": b["file"], "license": b["license"],
                                 "nen_a": mn[b["file"]]["nen_a"]}})
        per_lang[lang] = got
        used |= {e["meta"]["person"] for e in got}
    out = []
    for i in range(max(len(v) for v in per_lang.values())):       # xen kẽ vi / en
        out += [per_lang[l][i] for l in ("vi", "en") if i < len(per_lang[l])]
    return out


def _bases_by_id(ids: list) -> list:
    got = {e["id"]: e for e in json.load(open(GOC_JSON, encoding="utf-8"))}
    return [got[i] for i in ids if i in got]


# ------------------------------------------------------------------ ồn MUSAN có lọc giấy phép
_POOL = None


def _musan_noise(y, w, d, o):
    """Như k1_axes._musan_noise nhưng chỉ chọn trong tệp MUSAN noise có giấy phép cho phép."""
    import soundfile as sf
    from scipy.signal import resample_poly
    global _POOL
    if _POOL is None:
        import dung_bo
        _POOL = [p["path"] for p in dung_bo.musan_pool("noise", 3.0)]
        if not _POOL:
            raise FileNotFoundError("musan_pool('noise') rỗng")
    z, sr = sf.read(_POOL[len(y) % len(_POOL)], dtype="float64", always_2d=True)
    z = z.mean(1)
    if sr != K.SR:
        z = resample_poly(z, K.SR, sr)
    z = K.tile(z, 10.0)
    K.wr(d + ".wav", z * 10 ** (-26 / 20) / (np.sqrt(np.mean(z ** 2)) + 1e-9), subtype="FLOAT")


# ------------------------------------------------------------------ ca thêm: lượng tiếng nói ngoài vùng đệm
def _speech_s(seg: np.ndarray, floor: float) -> float:
    """Thời lượng lời theo ĐÚNG cách của k1_axes.oracle: khung 30 ms > sàn tệp gốc + 10 dB và > −60 dBFS."""
    db = K.energy_db(seg)
    return float(((db > floor + 10) & (db > -60)).sum() * 0.03)


def _room(n: int) -> np.ndarray:
    return np.random.default_rng(7).normal(0, 10 ** (-70 / 20), n)


def _mid_speech(sp_s: float, total_s: float):
    """Đoạn lời dày nhất có ĐO ĐƯỢC ≥ sp_s giây lời (≤ sp_s khi sp_s < 2: lấy đúng sp_s giây) đặt GIỮA tệp dài total_s;
    hai bên là tiếng phòng −70 dBFS (ngoài vùng 0,3 s đầu / cuối)."""
    def f(y, w, d, o):
        floor = float(np.percentile(K.energy_db(y), 10))
        L = sp_s
        s = K.dense(y, L)
        while sp_s >= 2 and _speech_s(s, floor) < sp_s and L < total_s - 1.0:   # khoảng nghỉ trong đoạn → nới đoạn tới khi đủ lời
            L += 0.1
            s = K.dense(y, L)
        n_pad = int(total_s * K.SR) - len(s)
        pad = _room(n_pad)
        K.wr(d + ".wav", np.r_[pad[:n_pad // 2], s, pad[n_pad // 2:]])
    return f


def _pad_share(share: float):
    """Như k1_axes._pad_speech nhưng đo lời THẬT của tệp gốc (cách của oracle) thay vì ước 60 % độ dài → tỉ lệ lời đúng share."""
    def f(y, w, d, o):
        sp = _speech_s(y, float(np.percentile(K.energy_db(y), 10)))
        n = max(0, int(sp / share * K.SR) - len(y))
        K.wr(d + ".wav", np.r_[y, _room(n)])
    return f


def extra_cases() -> list:
    return [("tinhieu.tiengnoi_1.4s", _mid_speech(1.4, 3.5), K.E("spd")),
            ("tinhieu.tiengnoi_3.0s", _mid_speech(3.0, 7.5), K.E("spd")),
            ("tinhieu.tiengnoi_20pct", _pad_share(0.20), K.E("spd")),
            ("tinhieu.tiengnoi_40pct", _pad_share(0.40), K.E("spd"))]


_ORIG_CASES = K.CASES


def install() -> None:
    K.OUT, K.V4 = OUT, ""                       # wav là đường dẫn tuyệt đối → os.path.join("", p) = p
    K.pick, K._bases_by_id, K._musan_noise = pick, _bases_by_id, _musan_noise
    K.CASES = lambda: _ORIG_CASES() + extra_cases()
    K.CASE_MAP = None


# ------------------------------------------------------------------ tóm tắt k/n + Wilson
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def tom(tag: str) -> str:
    rows = json.load(open(os.path.join(K.RES, f"k1_axes_{tag}.json"), encoding="utf-8"))
    rows = rows["rows"] if isinstance(rows, dict) else rows
    g = defaultdict(lambda: [0, 0])
    for r in rows:
        for key in ((r["ca"].split(".")[0], r["lang"]), (r["ca"].split(".")[0], "ca2")):
            g[key][0] += r["ok"]
            g[key][1] += 1
    lines = [f"# Test nhóm A · k1_axes trên tệp gốc P1 (tag {tag})", "", "| hàng | vi k/n | en k/n | cả hai k/n (Wilson 95 %) |", "|---|---|---|---|"]
    for h in sorted({k[0] for k in g}):
        cells = []
        for l in ("vi", "en", "ca2"):
            k, n = g.get((h, l), [0, 0])
            lo, hi = wilson(k, n)
            cells.append(f"{k}/{n}" + (f" ({lo * 100:.1f}–{hi * 100:.1f} %)" if l == "ca2" and n else ""))
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    bad = [r for r in rows if not r["ok"]]
    lines += ["", f"Ca sai: {len(bad)}/{len(rows)}" + (" (chỉ in 80 ca đầu; so trước / sau bằng experts/P1/coverage/so_c4.py)" if len(bad) > 80 else ""), ""] \
        + [f"- {r['ca']} · {r['base']} ({r['lang']}): {'; '.join(map(str, r['loi']))[:160]}" for r in bad[:80]]
    return "\n".join(lines)


# ------------------------------------------------------------------ selftest (không chạy K1, không ghi dữ liệu)
def selftest() -> None:
    import tempfile
    import soundfile as sf
    install()
    names = [n for n, _, _ in K.CASES()]
    assert len(names) == len(set(names)), "trùng tên ca"
    assert all(n in names for n, _, _ in extra_cases())
    # ca thêm: độ dài đúng, lời nằm giữa, tiếng phòng −70 dBFS
    sr = K.SR
    t = np.arange(12 * sr) / sr
    y = 0.1 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.5 * t) > -0.6)   # «lời» bật / tắt, ~70 % thời gian
    y = y + np.random.default_rng(1).normal(0, 1e-4, len(y))                        # sàn −80 dBFS như tệp thật (không có số 0 tuyệt đối)
    fl = float(np.percentile(K.energy_db(y), 10))
    with tempfile.TemporaryDirectory() as td:
        for name, fn, ex in extra_cases():
            stem = os.path.join(td, name)
            fn(y, None, stem, None)
            z, zr = sf.read(stem + ".wav")
            assert zr == sr
            if "s" == name[-1]:
                sp, tot = float(name.split("_")[1][:-1]), {"1.4": 3.5, "3.0": 7.5}[name.split("_")[1][:-1]]
                assert abs(len(z) / sr - tot) < 1e-3, (name, len(z) / sr)
                loud = np.flatnonzero(np.abs(z) > 1e-2)
                assert loud[0] / sr >= 0.3 and (len(z) - loud[-1]) / sr >= 0.3, name     # lời không rơi vào 0,3 s đầu / cuối
                if sp < 2:
                    assert (loud[-1] - loud[0]) / sr <= sp + 1e-3, name
                else:
                    assert _speech_s(z, fl) >= sp, (name, _speech_s(z, fl))
            else:
                share = float(name.split("_")[1][:-3]) / 100
                assert abs(_speech_s(z, fl) / (len(z) / sr) - share) < 0.01, (name, _speech_s(z, fl) / (len(z) / sr))
            assert np.sqrt(np.mean(z[:int(0.2 * sr)] ** 2)) < 10 ** (-65 / 20) or name.endswith("pct"), name
    # Wilson: giá trị tay (k=8, n=10 → 0,490–0,943)
    lo, hi = wilson(8, 10)
    assert abs(lo - 0.4902) < 1e-3 and abs(hi - 0.9433) < 1e-3, (lo, hi)
    # tệp gốc (đọc số đo có sẵn, không ghi gì)
    from cham_dieu_kien import fold
    b = pick()
    assert len(b) == N_BASE and sum(e["meta"]["lang"] == "vi" for e in b) == 12, len(b)
    assert len({e["meta"]["person"] for e in b}) == N_BASE, "trùng kênh"
    assert all(fold(e["meta"]["person"]) < 3 and e["meta"]["nen_a"] <= NEN_A_MAX for e in b)
    assert sum(e["meta"]["lang"] == "vi" for e in b[:N_LONG]) == N_LONG // 2, "ca dài không chia đều vi / en"
    assert all(re.fullmatch(r"p1k1(vi|en)_\d\d", e["id"]) and os.path.exists(e["wav"]) for e in b)
    assert pick() == b, "pick không tất định"
    print("selftest ĐẠT ·", len(names), "ca ·", len(b), "tệp gốc:", ", ".join(f"{e['id']}={e['meta']['file']}" for e in b[:4]), "…")


def main(a: list) -> None:
    cmd = a[0] if a else "selftest"
    if cmd == "selftest":
        return selftest()
    install()
    if cmd == "build":
        os.makedirs(OUT, exist_ok=True)
        if "--only" not in a:
            bases = pick(K._arg(a, "--bases", int, N_BASE))
            json.dump(bases, open(GOC_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        K.build(K._arg(a, "--bases", int, N_BASE), K._arg(a, "--long-bases", int, N_LONG), K._arg(a, "--only", str))
    elif cmd == "run":                                   # cùng cách đọc cờ như CLI k1_axes; tag mặc định p1_v1 (không ghi đè báo cáo gốc)
        ids = K._arg(a, "--base-ids", str)
        K.run(K._arg(a, "--limit"), K._arg(a, "--bases"), K._arg(a, "--only", str), False if "--no-events" in a else None, "--long" in a,
              ids.split(",") if ids else None, K._arg(a, "--tag", str, "p1_v1"), "--extended" in a)
    elif cmd == "tom":
        tag = K._arg(a, "--tag", str, "p1_v1")
        md = tom(tag)
        p = os.path.join(OUT, f"TOM_{tag}.md")
        open(p, "w", encoding="utf-8").write(md)
        print(md[:3000])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
