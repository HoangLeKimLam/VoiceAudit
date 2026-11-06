# -*- coding: utf-8 -*-
"""Ghep goi nop P3 v2 tu cac nguon OOF, chon nguon RIENG theo ngon ngu (nguong he thong da dat rieng vi / en; chon tren OOF — luat cung 2).

Nguon khung (den / voi):  p3v1                    = experts/P3/p3_v1/oof_p3.pkl (cot 0 / 1)
                          raw:<tag>[:km,kc,ko]    = .tmp_v6/cache/p3v2_raw_<tag>.pkl qua post.post(km, kc, ko) (mac dinh 9, 9, 21)
                          rawp:<tag>[:km,kc,ko]   = nhu raw + nang bac co tre post.plateau (alpha 0,5) — chi dung khi ko > 1
                          rawmx:<tag>[:km,kc,ko]  = post.post_raw_mx (nguong tren diem tho, bien doan noi theo nang bac)
                          seq:<tag>[:km,kc,ko]    = .tmp_v6/cache/p3v2_seq_<tag>.pkl["frames"] qua post.post
                          fuse1[:km,kc,ko]        = cot den / voi cua fuse1 qua post.post
Nguon dau (nen / giong):  p3v1 · h2 · file:<duong_dan.pkl> ({id: (2,)}) · seqh:<tag>
Spec: "vi=<nguon>;en=<nguon>"  (mot nguon cho ca hai: "<nguon>").
Tep khong co diem trong nguon (tep khong co .wav: chi kieu cua P4 / P5, nhan 03) -> 0 (khong ra dong 02; khong thoi phong Req 2 / 3 bang
diem khac thang do nguong) — bao so tep.
CLI (ban nop p3_v2): python experts/P3/p3_v2/compose.py --den "rawp:ens55:9,9,21" --voi "vi=rawmx:ens55:9,9,1;en=rawp:ens73:9,9,21"
       --nen "vi=file:experts/P3/p3_v2/heads_vi_mix.pkl;en=file:experts/P3/p3_v2/heads_b2.pkl"
       --giong "vi=file:experts/P3/p3_v2/heads_vi_mix.pkl;en=file:experts/P3/p3_v2/heads_b2.pkl" --name p3v2 --out experts/P3/p3_v2 [--eval]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
_C = {}


def _load(path: str):
    if path not in _C:
        _C[path] = pickle.load(open(path, "rb"))
    return _C[path]


def parse_spec(spec: str) -> dict:
    """'vi=a;en=b' -> {'vi': 'a', 'en': 'b'}; 'a' -> {'vi': 'a', 'en': 'a'} (ham thuan)."""
    if "=" not in spec.split(":")[0]:
        return {"vi": spec, "en": spec}
    out = dict(p.split("=", 1) for p in spec.split(";"))
    assert set(out) == {"vi", "en"}, out
    return out


def _pp(src: str) -> tuple:
    parts = src.split(":")
    kw = (9, 9, 21)
    if len(parts) >= 3 or (parts[0] == "fuse1" and len(parts) == 2):
        kw = tuple(int(x) for x in parts[-1].split(","))
        parts = parts[:-1]
    return parts, kw


def frame_source(src: str, col: int):
    from post import post, post2
    parts, (km, kc, ko) = _pp(src)
    if parts[0] == "rawmx":
        from post import post_raw_mx
        d = _load(os.path.join(CACHE, f"p3v2_raw_{parts[1]}.pkl"))
        return lambda i: post_raw_mx(np.asarray(d[i], np.float32)[:, col], km, kc) if i in d else None
    if parts[0] == "rawp":
        d = _load(os.path.join(CACHE, f"p3v2_raw_{parts[1]}.pkl"))
        return lambda i: post2(np.asarray(d[i], np.float32)[:, col], km, kc, ko, 0.5, 3) if i in d else None
    if parts[0] == "p3v1":
        d = _load(os.path.join(HERE, "..", "p3_v1", "oof_p3.pkl"))
        return lambda i: np.asarray(d[i], np.float32)[:, col] if i in d else None
    if parts[0] == "fuse1":
        d = _load(os.path.join(CACHE, "boundary_oof_fuse1.pkl"))
        return lambda i: post(np.asarray(d[i], np.float32)[:, 2 + col], km, kc, ko) if i in d else None
    d = _load(os.path.join(CACHE, f"p3v2_{'raw' if parts[0] == 'raw' else 'seq'}_{parts[1]}.pkl"))
    d = d.get("frames", d)
    return lambda i: post(np.asarray(d[i], np.float32)[:, col], km, kc, ko) if i in d else None


def head_source(src: str, col: int):
    parts = src.split(":", 1)
    if parts[0] == "p3v1":
        d = _load(os.path.join(HERE, "..", "p3_v1", "heads_p3.pkl"))
    elif parts[0] == "h2":
        h = _load(os.path.join(CACHE, "heads2_h2.pkl"))
        d = {i: h["P"][n] for n, i in enumerate(h["ids"])}
    elif parts[0] == "seqh":
        d = _load(os.path.join(CACHE, f"p3v2_seq_{parts[1]}.pkl"))["heads"]
    else:
        d = _load(os.path.join(BASE, parts[1]) if not os.path.isabs(parts[1]) else parts[1])
    return lambda i: float(np.ravel(d[i])[col]) if i in d else None


def compose(specs: dict, langs: dict, base_f: dict, base_h: dict) -> tuple:
    """specs {kenh: {lang: nguon}} -> (oof {id: (T, 2) f16}, heads {id: (2,) f32}, so tep thieu theo kenh)."""
    miss = {k: 0 for k in specs}
    fs = {k: {lg: frame_source(s, c) for lg, s in specs[k].items()} for k, c in (("den", 0), ("voi", 1))}
    hs = {k: {lg: head_source(s, c) for lg, s in specs[k].items()} for k, c in (("nen", 0), ("giong", 1))}
    oof, heads = {}, {}
    for i, P in base_f.items():
        lg = langs.get(i, "vi")
        cols = []
        for k, c in (("den", 2), ("voi", 3)):
            v = fs[k][lg](i)
            if v is None:
                miss[k] += 1
                v = np.zeros(len(P), np.float32)
            n = len(P)
            v = np.r_[v[:n], np.full(max(0, n - len(v)), v[-1] if len(v) else 0.0)]
            cols.append(v)
        oof[i] = np.stack(cols, 1).astype(np.float16)
    for i, h in base_h.items():
        lg = langs.get(i, "vi")
        out = []
        for k, c in (("nen", 0), ("giong", 1)):
            v = hs[k][lg](i)
            if v is None:
                miss[k] += 1
                v = 0.0
            out.append(v)
        heads[i] = np.array(out, np.float32)
    return oof, heads, miss


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("den", "voi", "nen", "giong"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--name", default="p3v2")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    from v6.boundary import detect as D
    items = D.load_items(keep=("lang",))
    langs = {i: ("en" if it["set"] == "v6naten" else str(it["meta"].get("lang") or "vi")[:2]) for i, it in items.items()}
    bf = _load(os.path.join(CACHE, "boundary_oof_fuse1.pkl"))
    h2 = _load(os.path.join(CACHE, "heads2_h2.pkl"))
    bh = {i: h2["P"][n] for n, i in enumerate(h2["ids"])}
    specs = {k: parse_spec(getattr(a, k)) for k in ("den", "voi", "nen", "giong")}
    oof, heads, miss = compose(specs, langs, bf, bh)
    po, ph = os.path.join(a.out, f"oof_{a.name}.pkl"), os.path.join(a.out, f"heads_{a.name}.pkl")
    pickle.dump(oof, open(po, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(heads, open(ph, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("->", po, len(oof), "·", ph, len(heads), "· tep lay diem nen:", miss)
    if a.eval:
        from evalp3 import P3Eval, load_frames, load_heads
        E = P3Eval()
        base = E.run()
        d, v = load_frames(po)
        n, g = load_heads(ph)
        res = E.run(d, v, n, g)
        txt = f"# evalp3 `{a.name}`\n\nspec: {specs}\n\n" + E.table(res, base)
        open(os.path.join(BASE, ".tmp_v6", "results", f"evalp3_{a.name}.md"), "w", encoding="utf-8").write(txt + "\n")
        print(E.summary(res))


def selftest() -> None:
    assert parse_spec("p3v1") == {"vi": "p3v1", "en": "p3v1"}
    assert parse_spec("vi=p3v1;en=raw:st2") == {"vi": "p3v1", "en": "raw:st2"}
    assert _pp("raw:st2:9,9,11") == (["raw", "st2"], (9, 9, 11)) and _pp("raw:st2") == (["raw", "st2"], (9, 9, 21))
    assert _pp("fuse1:1,1,21") == (["fuse1"], (1, 1, 21))
    print("compose selftest dat")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        main()
