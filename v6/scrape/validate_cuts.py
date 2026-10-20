# -*- coding: utf-8 -*-
"""KIEM DINH BO LOC VET CAT (critic: 'was_live != chua bien tap', filter_spec C5 chua tung duoc do).

Cay vet sua DA BIET vao doan am thanh sach roi chay frames.detect_cuts, do recall theo loai + precision + bao nham / gio:
  splice_other   thay 2-6 s bang am thanh tu TEP KHAC (nen on / kenh khac)          -> 2 moi noi
  deletion       cat 0,5-3 s cung ban ghi, noi lai (crossfade 0 / 5 / 20 ms)        -> 1 moi noi (ca kho nhat)
  jump_silence   cat 0,5-3 s + chen lang so 50-200 ms (cat-dan kieu trinh sua)       -> 1 moi noi
  gain_step      tu moi noi tro di nhan +-10 dB (ghep doan da chinh muc khac)       -> 1 moi noi
  noise_step     tu moi noi tro di cong on trang -50..-40 dBFS (doan thu o phong khac) -> 1 moi noi
Khop: phat hien trong +-TOL_S (1 s) quanh moi noi. Bao nham do tren CHINH doan goc chua sua (truoc khi cay).
Bao cao HAI cau hinh bo do: 'orig_gated' (trang thai cong cua doan goc) va 'production' (gated=None nhu filter.decide).
Nguon doan sach: tep da qua filter (cut_mode 'clean'), moi tep lay N doan 5 phut trong keep_spans; hoac --synthetic
(loi noi gia lap, chi kiem tra duong ong, KHONG phai so do tren du lieu that).
Ra: data/scrape/meta/validate_cuts.json + bang in ra.

CLI: python -m v6.scrape.validate_cuts [--synthetic] [--n-files 6] [--per-file 2] [--excerpt-s 300] [--seed 0]"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import frames as FR  # noqa: E402
from v6.scrape import util as U  # noqa: E402

SR = FR.SR
TOL_S = 1.0
KINDS = ("splice_other", "deletion", "jump_silence", "gain_step", "noise_step")
OUT = os.path.join(U.META, "validate_cuts.json")


# ---------------- loi noi gia lap (kiem thu / --synthetic) ----------------
def synth_speech(dur_s: float, seed: int = 0, noise_db: float = -62.0, level_db: float = -24.0,
                 f0: float | None = None) -> np.ndarray:
    """Am tiet huu thanh (F0 truot, 3 formant) + am xat, khoang lang tu / cau; nen on hong."""
    rng = np.random.default_rng(seed)
    n = int(dur_s * SR)
    y = np.zeros(n, np.float64)
    base_f0 = f0 or rng.uniform(100, 220)
    t = 0.2
    while t < dur_s - 0.5:
        for _ in range(rng.integers(2, 9)):              # tu trong cau
            for _ in range(rng.integers(1, 4)):          # am tiet trong tu
                L = rng.uniform(0.12, 0.3)
                m = int(L * SR)
                a = int(t * SR)
                if a + m >= n:
                    break
                tt = np.arange(m) / SR
                f = base_f0 * (1 + rng.uniform(-0.15, 0.15)) * (1 + rng.uniform(-0.1, 0.1) * tt / L)
                ph = 2 * np.pi * np.cumsum(f) / SR
                fm = [rng.uniform(300, 800), rng.uniform(900, 2200), rng.uniform(2400, 3200)]
                s = np.zeros(m)
                for h in range(1, 30):
                    fh = h * f.mean()
                    if fh > 7000:
                        break
                    g = sum(np.exp(-0.5 * ((fh - F) / 120) ** 2) for F in fm) + 0.02
                    s += g * np.sin(h * ph)
                if rng.random() < 0.3:                   # am xat
                    s[: m // 3] += 0.3 * rng.standard_normal(m // 3) * np.linspace(1, 0, m // 3)
                y[a:a + m] += s * np.hanning(m)
                t += L + rng.uniform(0.02, 0.06)
            t += rng.uniform(0.06, 0.18)
        t += rng.uniform(0.35, 0.9)
    y *= 10 ** (level_db / 20) / (np.sqrt(np.mean(y[y != 0] ** 2)) + 1e-12)
    pink = np.cumsum(rng.standard_normal(n)) * 0.02
    pink = pink - np.convolve(pink, np.ones(801) / 801, mode="same")
    pink = pink / (np.std(pink) + 1e-12) * 10 ** (noise_db / 20)
    return (y + pink).astype(np.float32)


# ---------------- cay vet sua ----------------
def _xfade_join(a: np.ndarray, b: np.ndarray, fade_ms: float) -> np.ndarray:
    k = int(fade_ms / 1000 * SR)
    if k <= 0 or len(a) < k or len(b) < k:
        return np.concatenate([a, b])
    w = np.linspace(0, 1, k, dtype=np.float32)
    return np.concatenate([a[:-k], a[-k:] * (1 - w) + b[:k] * w, b[k:]])


def plant(y: np.ndarray, kind: str, t: float, rng: random.Random, other: np.ndarray | None) -> tuple[np.ndarray, list[float]]:
    """Cay MOT vet sua tai t (giay). Tra (tin hieu moi, cac thoi diem moi noi trong tin hieu moi)."""
    a = int(t * SR)
    fade = rng.choice([0.0, 5.0, 20.0])
    if kind == "splice_other":
        L = int(rng.uniform(2, 6) * SR)
        src = other if other is not None and len(other) > L + SR else y
        o = rng.randint(0, len(src) - L - 1)
        seg = src[o:o + L] * (np.sqrt(np.mean(y[a:a + L] ** 2) + 1e-12) / (np.sqrt(np.mean(src[o:o + L] ** 2)) + 1e-12))
        z = _xfade_join(_xfade_join(y[:a], seg.astype(np.float32), fade), y[a + L:], fade)
        return z, [t, t + L / SR - fade / 1000]
    if kind in ("deletion", "jump_silence"):
        L = int(rng.uniform(0.5, 3.0) * SR)
        head = y[:a]
        if kind == "jump_silence":
            head = np.concatenate([head, np.zeros(int(rng.uniform(0.05, 0.2) * SR), np.float32)])
            fade = 0.0
        return _xfade_join(head, y[a + L:], fade), [t]
    if kind == "gain_step":
        g = 10 ** (rng.choice([-10.0, 10.0]) / 20)
        z = y.copy()
        z[a:] *= g
        return np.clip(z, -1, 1), [t]
    if kind == "noise_step":
        z = y.copy()
        lvl = 10 ** (rng.uniform(-50, -40) / 20)
        z[a:] += (np.random.default_rng(rng.randint(0, 10 ** 6)).standard_normal(len(z) - a) * lvl).astype(np.float32)
        return z, [t]
    raise ValueError(kind)


def plant_many(y: np.ndarray, n_edits: int, rng: random.Random, other: np.ndarray | None,
               min_gap_s: float = 25.0) -> tuple[np.ndarray, list[dict]]:
    """Cay n vet sua cach nhau >= min_gap_s, tu cuoi ve dau (thoi diem truoc vet chua bi doi)."""
    dur = len(y) / SR
    n_slots = int((dur - 30) / min_gap_s)
    k = min(n_edits, max(0, n_slots))
    if k == 0:
        return y, []
    slots = sorted(rng.sample(range(n_slots), k))
    plan = [(15 + (s + 0.5) * min_gap_s + rng.uniform(-3, 3), rng.choice(KINDS)) for s in slots]
    joints: list[dict] = []
    z = y
    for e, (t, kind) in enumerate(sorted(plan, reverse=True)):
        n_old = len(z)
        z, js = plant(z, kind, t, rng, other)
        delta = (len(z) - n_old) / SR            # moi moi noi da cay deu nam SAU t -> dich theo do dai thay doi
        joints = [{**j, "t": round(j["t"] + delta, 2)} for j in joints]
        joints += [{"t": round(x, 2), "kind": kind, "edit": e} for x in js]
    return z, sorted(joints, key=lambda j: j["t"])


def match(dets: list[dict], joints: list[dict], tol: float = TOL_S) -> tuple[dict, int]:
    """Recall theo loai (moi vet sua = it nhat MOT moi noi cua no duoc bat) + so phat hien khop."""
    hit_det = set()
    per = defaultdict(lambda: [0, 0])
    by_edit: dict[tuple, list] = defaultdict(list)
    for j in joints:
        by_edit[(j["kind"], j.get("edit", j["t"]))].append(j)
    for j in joints:
        ok = [i for i, d in enumerate(dets) if d.get("lo", d["t"]) - tol <= j["t"] <= d.get("hi", d["t"]) + tol]
        j["hit"] = bool(ok)
        hit_det.update(ok)
    for (kind, _), js in by_edit.items():
        per[kind][1] += 1
        per[kind][0] += int(any(j["hit"] for j in js))
    return {k: v for k, v in per.items()}, len(hit_det)


def run_excerpt(y: np.ndarray, other: np.ndarray | None, n_edits: int, rng: random.Random, use_vad: bool = True) -> dict:
    """Hai cau hinh bo do (review 12/10):
      ''      'orig_gated': trang thai 'chuoi co cong' lay tu doan GOC chua sua (vet lang so cay day dac khong tu bat luat cong)
      '_prod' 'production': gated=None nhu filter.decide (tu do tren chinh tep da sua) - cau hinh that su chay o filter."""
    F0, sp0, _ = FR.analyse_array(y, use_vad=use_vad)
    gated0, _ = FR.gated_floor(F0)
    base = FR.detect_cuts(F0, sp0, gated=gated0)
    base_prod = FR.detect_cuts(F0, sp0, gated=None)
    z, joints = plant_many(y, n_edits, rng, other)
    F1, sp1, _ = FR.analyse_array(z, use_vad=use_vad)
    dets = FR.detect_cuts(F1, sp1, gated=gated0)
    dets_prod = FR.detect_cuts(F1, sp1, gated=None)
    joints_prod = [dict(j) for j in joints]
    per, n_hit = match(dets, joints)
    per_p, n_hit_p = match(dets_prod, joints_prod)
    return {"dur_s": round(len(y) / SR, 1), "gated": bool(gated0), "gated_edited": bool(FR.gated_floor(F1)[0]),
            "base_dets": base, "dets": dets, "joints": joints, "per_kind": per, "n_det": len(dets), "n_det_hit": n_hit,
            "base_dets_prod": base_prod, "dets_prod": dets_prod, "per_kind_prod": per_p, "n_det_prod": len(dets_prod),
            "n_det_hit_prod": n_hit_p}


def summarize(results: list[dict], sfx: str = "") -> dict:
    """sfx '' = cau hinh orig_gated; '_prod' = cau hinh production (gated=None)."""
    per = defaultdict(lambda: [0, 0])
    nd = nh = 0
    base_n, base_h = 0, 0.0
    base_types: dict[str, int] = defaultdict(int)
    for r in results:
        bd = r.get("base_dets" + sfx, r["base_dets"])
        for d in bd:
            base_types[d["type"]] += 1
        for k, (h, t) in r.get("per_kind" + sfx, r["per_kind"]).items():
            per[k][0] += h
            per[k][1] += t
        nd += r.get("n_det" + sfx, r["n_det"])
        nh += r.get("n_det_hit" + sfx, r["n_det_hit"])
        base_n += len(bd)
        base_h += r["dur_s"] / 3600
    tot_h = sum(v[0] for v in per.values())
    tot = sum(v[1] for v in per.values())
    return {"recall_by_kind": {k: {"hit": h, "n": t, "recall": round(h / t, 3) if t else None} for k, (h, t) in sorted(per.items())},
            "recall_all": round(tot_h / tot, 3) if tot else None, "precision": round(nh / nd, 3) if nd else None,
            "false_alarms_per_hour_clean": round(base_n / base_h, 2) if base_h else None,
            "base_types": dict(sorted(base_types.items())), "n_excerpts": len(results), "hours": round(base_h, 3)}


def _decode_excerpt(path: str, start: float, dur: float) -> np.ndarray:
    cmd = [U.ffmpeg_bin("ffmpeg"), "-nostdin", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", path,
           "-map", "0:a:0", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=300, check=True).stdout
    return np.frombuffer(out[:len(out) // 4 * 4], np.float32).copy()


def real_excerpts(n_files: int, per_file: int, ex_s: float, rng: random.Random) -> list[tuple[str, np.ndarray]]:
    rows = [r for r in U.latest_by(U.read_jsonl(U.FILTERED)).values() if r.get("ok") and r.get("cut_mode") == "clean"]
    rng.shuffle(rows)
    out = []
    for r in rows[:n_files]:
        spans = [s for s in r.get("keep_spans") or [] if s[1] - s[0] >= ex_s]
        for _ in range(per_file):
            if not spans:
                break
            s = rng.choice(spans)
            st = rng.uniform(s[0], s[1] - ex_s)
            out.append((r["id"], _decode_excerpt(os.path.join(U.SCRAPE, r["file"]), st, ex_s)))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-files", type=int, default=6)
    ap.add_argument("--per-file", type=int, default=2)
    ap.add_argument("--excerpt-s", type=float, default=300.0)
    ap.add_argument("--edits", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.synthetic:
        ex = [(f"synth{i}", synth_speech(a.excerpt_s, seed=a.seed * 100 + i)) for i in range(a.n_files * a.per_file)]
    else:
        ex = real_excerpts(a.n_files, a.per_file, a.excerpt_s, rng)
    if len(ex) < 2:
        print("khong du doan sach (can tep filter ok + cut_mode clean); dung --synthetic de thu duong ong")
        return
    results = []
    for i, (rid, y) in enumerate(ex):
        other = ex[(i + 1) % len(ex)][1]
        r = run_excerpt(y, other, a.edits, rng)
        r["id"] = rid
        results.append(r)
        print(f"  {rid}: {r['dur_s']} s, phat hien {r['n_det']} (khop {r['n_det_hit']}), goc {len(r['base_dets'])} | "
              f"{ {k: f'{h}/{t}' for k, (h, t) in r['per_kind'].items()} }", flush=True)
    summ = summarize(results)
    summ["detector_config"] = "orig_gated (trang thai cong tu doan goc chua sua)"
    summ["production"] = {**summarize(results, "_prod"), "detector_config": "production (gated=None, nhu filter.decide)"}
    summ["n_gated_flip_after_edit"] = sum(r["gated"] != r["gated_edited"] for r in results)
    summ["mode"] = "synthetic" if a.synthetic else "real"
    os.makedirs(U.META, exist_ok=True)
    with open(OUT if not a.synthetic else OUT.replace(".json", "_synthetic.json"), "w", encoding="utf-8") as fh:
        json.dump({"summary": summ, "results": results}, fh, ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
