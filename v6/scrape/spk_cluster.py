# -*- coding: utf-8 -*-
"""GOM CUM GIONG NOI giua cac kenh cao -> nhom chong ro ri (thay dan 'nhom = kenh', giao dien I2 'until clustered').

  1. resemblyzer d-vector (CPU, pretrained DONG BANG - quyet dinh 6) cho <= --max-per-channel doan chuan / kenh
     (meta/segments.jsonl, rai deu theo video; bo nho dem meta/spk_emb.npz theo ten tep).
  2. Trong moi kenh: gom cum agglomerative cosine (khoang cach < WITHIN_DIST) -> cum >= MIN_CLUSTER doan = 'nguoi noi
     chinh' cua kenh -> tam cum (chuan hoa).
  3. Giua cac kenh: hai tam cum cosine >= CROSS_SIM -> HOP NHAT hai kenh (union-find). CHI HOP, khong tach kenh: cung mot
     nguoi o hai kenh (dang lai, khach moi, kenh phu) -> cung nhom = cung fold; tach kenh theo nguoi can nghe xac nhan.
  Ra: meta/spk_groups.json {map: {'<nguon>:<kenh>': 'scrape:<nhom>'} (CHI kenh bi hop), pairs: cap kenh co do giong cao
  nhat, params}. hosts.group_for doc tep nay (meta['group'] -> v6.common.group_of giu nguyen).
  DO TREN PILOT 12/10 (resemblyzer, tam cum theo video): cung kenh khac video 0,71-0,94 (n = 7); khac kenh p50 0,69 /
  p99 0,88 / max 0,90 (n = 203; cac kenh hop hoi dong / Zoom khac nhau giong nhau vi cung kieu phong / kenh truyen) ->
  HAI phan bo CHONG NHAU: resemblyzer KHONG tach duoc. Nguong 0,85 hop nham 4 kenh khong lien quan (City of Independence,
  Marion Township, LF Decentralized Trust, PANTADO) -> CROSS_SIM mac dinh 0,95 (tren moi cap khac kenh da thay: khong hop
  tu dong) va REVIEW_SIM 0,85: cap >= 0,85 ghi vao 'review_pairs' de NGHE xac nhan. Can mo hinh nguoi noi tot hon
  (WavLM-SV / ECAPA, pretrained dong bang) truoc khi dua vao hop tu dong - xem hang doi.
NANG VUA (resemblyzer ~50x thoi gian thuc tren CPU): pilot chay duoc; tron bo = hang doi.

CLI: python -m v6.scrape.spk_cluster [--lang vi|en] [--max-per-channel 60] [--budget-min 8] [--cross-sim 0.95]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

WITHIN_DIST = 0.30
MIN_CLUSTER = 3
CROSS_SIM = 0.95       # hop tu dong (xem docstring: resemblyzer khong tach duoc o 0,85)
REVIEW_SIM = 0.85      # cap kenh can nghe xac nhan
OUT = os.path.join(U.META, "spk_groups.json")
EMB = os.path.join(U.META, "spk_emb.npz")


def channel_key(seg: Mapping[str, Any]) -> str:
    return f"{seg.get('source', 'yt')}:{seg.get('channel_id')}"


def centroids(emb: np.ndarray, within_dist: float = WITHIN_DIST, min_cluster: int = MIN_CLUSTER) -> np.ndarray:
    """Tam (chuan hoa) cac cum >= min_cluster doan trong MOT kenh; it doan -> 1 tam trung binh."""
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    if len(e) < 2 * min_cluster:
        c = e.mean(0, keepdims=True)
        return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    from sklearn.cluster import AgglomerativeClustering
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=within_dist, metric="cosine",
                                  linkage="average").fit_predict(e)
    cs = [e[lab == k].mean(0) for k in np.unique(lab) if (lab == k).sum() >= min_cluster]
    if not cs:
        cs = [e.mean(0)]
    c = np.stack(cs)
    return c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)


def merge_channels(cents: Mapping[str, np.ndarray], cross_sim: float = CROSS_SIM) -> tuple[dict[str, str], list[dict]]:
    """Union-find tren kenh: tam cum hai kenh cosine >= cross_sim -> cung nhom. Tra ({kenh: nhom} CHI kenh bi hop,
    [cap kenh: do giong lon nhat] sap giam)."""
    keys = sorted(cents)
    parent = {k: k for k in keys}

    def find(k: str) -> str:
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            s = float((cents[a] @ cents[b].T).max())
            pairs.append({"a": a, "b": b, "max_sim": round(s, 4)})
            if s >= cross_sim:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    comp: dict[str, list] = defaultdict(list)
    for k in keys:
        comp[find(k)].append(k)
    out = {}
    for members in comp.values():
        if len(members) > 1:
            g = f"scrape:{sorted(members)[0]}+{len(members) - 1}"
            out.update({m: g for m in members})
    return out, sorted(pairs, key=lambda p: -p["max_sim"])


def pick_segments(segs: Iterable[Mapping[str, Any]], lang: str | None, per_channel: int, seed: int = 0) -> dict[str, list]:
    by: dict[str, list] = defaultdict(list)
    for s in segs:
        if s.get("file") and s.get("tier") == "std" and (not lang or s.get("lang") == lang) and U.seg_wav(s):
            by[channel_key(s)].append(s)
    rng = random.Random(seed)
    out = {}
    for k, xs in by.items():
        xs = sorted(xs, key=lambda s: s["file"])
        rng.shuffle(xs)
        out[k] = xs[:per_channel]
    return out


def embed_all(chosen: Mapping[str, list], budget_min: float) -> dict[str, np.ndarray]:
    """{tep: d-vector}; dung lai bo nho dem spk_emb.npz."""
    cache: dict[str, np.ndarray] = {}
    if os.path.exists(EMB):
        z = np.load(EMB)
        cache = {k: z[k] for k in z.files}
    todo = [s for xs in chosen.values() for s in xs if s["file"] not in cache]
    if todo:
        from resemblyzer import VoiceEncoder, preprocess_wav
        from v6.scrape.qc import read16k
        enc = VoiceEncoder("cpu", verbose=False)
        t0 = time.time()
        for i, s in enumerate(todo):
            if time.time() - t0 > budget_min * 60:
                print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
                break
            try:
                cache[s["file"]] = enc.embed_utterance(preprocess_wav(read16k(U.seg_wav(s)), U.SR)).astype(np.float32)
            except Exception as ex:  # noqa: BLE001  (doan qua ngan sau khi cat lang...)
                print(f"  bo {s['file']}: {str(ex)[:80]}", flush=True)
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(todo)}] {time.time() - t0:.0f} s", flush=True)
        np.savez(EMB, **cache)
    return cache


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"])
    ap.add_argument("--max-per-channel", type=int, default=60)
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--cross-sim", type=float, default=CROSS_SIM)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    chosen = pick_segments(U.read_jsonl(U.SEGMENTS), a.lang, a.max_per_channel, a.seed)
    emb = embed_all(chosen, a.budget_min)
    cents = {}
    for k, xs in chosen.items():
        e = [emb[s["file"]] for s in xs if s["file"] in emb]
        if e:
            cents[k] = centroids(np.stack(e))
    mapping, pairs = merge_channels(cents, a.cross_sim)
    res = {"generated_at": U.now_iso(), "map": mapping, "n_channels": len(cents),
           "n_merged_channels": len(mapping), "pairs_top": pairs[:30],
           "review_pairs": [p for p in pairs if REVIEW_SIM <= p["max_sim"] < a.cross_sim],
           "params": {"within_dist": WITHIN_DIST, "min_cluster": MIN_CLUSTER, "cross_sim": a.cross_sim, "review_sim": REVIEW_SIM,
                      "max_per_channel": a.max_per_channel, "model": "resemblyzer", "calibrated": False}}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(f"spk_cluster: {len(cents)} kenh, hop {len(mapping)} kenh vao {len(set(mapping.values()))} nhom -> {OUT}")
    for p in pairs[:8]:
        print("  ", p)


def load_map(path: str = OUT) -> dict[str, str]:
    try:
        with open(path, encoding="utf-8") as fh:
            return dict(json.load(fh).get("map") or {})
    except (OSError, ValueError):
        return {}


# ---------------- kiem thu (khong nap mo hinh) ----------------
def test_centroids_and_merge() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 64))                       # 3 nguoi
    def utt(p: int, n: int) -> np.ndarray:                     # noqa: E306
        return base[p] + 0.15 * rng.standard_normal((n, 64))
    cents = {"yt:A": centroids(utt(0, 12)), "yt:B": centroids(np.concatenate([utt(1, 10), utt(0, 4)])),
             "yt:C": centroids(utt(2, 12))}
    mp, pairs = merge_channels(cents, 0.85)
    assert mp == {"yt:A": "scrape:yt:A+1", "yt:B": "scrape:yt:A+1"}, (mp, pairs[:3])   # B co khach la nguoi cua A
    assert pairs[0]["max_sim"] >= 0.85 and {pairs[0]["a"], pairs[0]["b"]} == {"yt:A", "yt:B"}
    assert merge_channels(cents, 0.999)[0] == {}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        test_centroids_and_merge()
        print("PASS test_centroids_and_merge")
    else:
        main()
