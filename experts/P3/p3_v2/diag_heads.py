# -*- coding: utf-8 -*-
"""Chan doan dau ca tep: tep that diem cao nhat (theo kieu / nguon) va TPR theo cong cu o nguong ngan sach (ham doc OOF). CLI: diag_heads.py heads.pkl [k]"""
import sys, os, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE))); sys.path.insert(0, BASE)
from evalp3 import load_heads
from v6.boundary import fuse9 as F9, fusion as F
sys.stdout.reconfigure(encoding="utf-8")
k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
h = load_heads(sys.argv[1])[k]
ctx = F9.Ctx9("fuse1", "fuse1", "h2", F.NAT_SETS)
for lg in ("vi", "en"):
    real = [i for i in ctx.real if ctx.lang[i] == lg]
    s = np.array([h[i] for i in real])
    order = np.argsort(-s)[:12]
    print(f"== {lg}: tep that diem cao nhat (nguong ~ phan vi 99.75 = {np.quantile(s, 0.9975):.3f})")
    for o in order:
        it = ctx.items[real[o]]
        print(f"   {s[o]:.3f} {it['kind']:16s} {it['meta'].get('source')} {real[o][:44]} f{it['fold']} ns={[e.get('ns_level') for e in it['meta'].get('edits_khong_tinh', [])]}")
    thr = np.quantile(s, 0.9975)
    tl = collections.defaultdict(list)
    for i in ctx.ids:
        it = ctx.items[i]
        if ctx.lang[i] != lg or it["real"]:
            continue
        for e in it["meta"].get("edits", []):
            if e["level2"] == ("02_nen_toan_cuc" if k == 0 else "02_giong_toan_cuc"):
                tl[(it["kind"], e.get("tool") or e.get("op") or e.get("family"), e.get("ns_level"))].append(h[i] >= thr)
    for t, v in sorted(tl.items(), key=str):
        print(f"   TPR {np.mean(v):.2f} n={len(v):3d} {t}")
