# -*- coding: utf-8 -*-
"""P3 v2 — mo hinh CHUOI nho (TCN gian no, hoc tu dau tren dac trung — luat cung 1: dau nho dat tren dac trung) cho kenh khung den / voi
va dau ca tep nen / giong, 5 fold THEO NGUOI NOI (mo hinh fold f chi hoc tep fold != f; chuan hoa dac trung cung hoc tren fold != f).

Dau vao moi khung: ctxfeat.ctx_feats (vat ly .wav + XLS-R PCA-64, ~207 chieu). Dau ra: logit khung (den, voi) + logit tep (nen, giong).
Ham mat mat = BCE khung (trong so: tep that x3, doan WebRTC NS <= 2 x10, bien doan bo qua) + BCE tep
  + MIL dung tieu chi giai ma: r = trung binh truot 21 khung cua logit (diem "doan >= 0,4 s");
    tep that: phat softplus(max_t r) (dinh doan cao nhat cua tep that dat nguong); doan sua cuc bo: phat softplus(-max_{t trong doan} r).
Ra: .tmp_v6/cache/p3v2_seq_<tag>.pkl {"frames": {id: (T, 2) f16 xac suat}, "heads": {id: (2,)}} cho tep goc + benign.
CLI: python experts/P3/p3_v2/train_seq.py [--tag s1] [--epochs 14] [--threads 20] [--seed 0] [--no-app]"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, BASE)
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

CACHE = os.path.join(BASE, ".tmp_v6", "cache")
CROP = 700
RUN = 21


def file_targets(meta: dict, T: int, real: bool, cfg: str) -> tuple:
    """(y (T, 2), w (T, 2), yh (2,), seg [(a, b, kenh)], neg_file) (ham thuan)."""
    from train_frames import frame_labels
    yd, yv, ign, ky = frame_labels(meta, T)
    y = np.stack([yd, yv], 1).astype(np.float32)
    w = np.ones((T, 2), np.float32)
    l2 = set(meta.get("level2", []))
    yh = np.array([float("02_nen_toan_cuc" in l2), float("02_giong_toan_cuc" in l2)], np.float32)
    for k in range(2):
        if y[:, k].all():
            w[:, k] = 0.3                                   # tep sua ca tep: nhieu khung duong
    if real:
        w *= 3.0
        w[ky] = 10.0
    w[ign] = 0.0
    if cfg != "goc":
        w *= 0.5
    segs = []
    t = (np.arange(T) + 0.5) * 0.02
    for e in meta.get("edits", []):
        if e.get("level2", "").startswith("02") and e.get("kind") == "segment":
            a, b = np.searchsorted(t, float(e["start"])), np.searchsorted(t, float(e["end"]))
            if b - a >= 8:
                segs.append((int(a), int(b), 0 if "_nen_" in e["level2"] else 1))
    return y, w, yh, segs, bool(real)


def build_model(d_in: int, ch: int = 96):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def __init__(self, c, dil):
            super().__init__()
            self.c1 = nn.Conv1d(c, c, 5, padding=2 * dil, dilation=dil)
            self.c2 = nn.Conv1d(c, c, 1)
            self.n = nn.GroupNorm(1, c)

        def forward(self, x):
            return x + self.c2(torch.nn.functional.gelu(self.n(self.c1(x))))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Conv1d(d_in, ch, 1), nn.GELU(), nn.Dropout(0.1))
            self.blocks = nn.Sequential(*[Block(ch, d) for d in (1, 2, 4, 8, 16, 1, 4, 16)])
            self.fr = nn.Conv1d(ch, 2, 1)
            self.hd = nn.Sequential(nn.Linear(2 * ch, 64), nn.GELU(), nn.Linear(64, 2))

        def forward(self, x, m):                            # x (B, D, T), m (B, T) 0/1
            h = self.blocks(self.inp(x))
            fr = self.fr(h)                                 # (B, 2, T)
            mm = m[:, None, :]
            mean = (h * mm).sum(2) / mm.sum(2).clamp(min=1)
            mx = (h - 1e4 * (1 - mm)).max(2).values
            return fr, self.hd(torch.cat([mean, mx], 1))

    return Net()


def loss_fn(fr, hd, Y, W, YH, M, SEG, NEG, lam: float = 1.0):
    import torch
    import torch.nn.functional as Fn
    bce = Fn.binary_cross_entropy_with_logits(fr, Y, reduction="none") * W
    lf = bce.sum() / W.sum().clamp(min=1)
    lh = Fn.binary_cross_entropy_with_logits(hd, YH)
    r = Fn.avg_pool1d(fr, RUN, stride=1, padding=RUN // 2, count_include_pad=False)        # (B, 2, T)
    r = r - 1e4 * (1 - M[:, None, :])
    lm, n = fr.new_zeros(()), 0
    for b in range(fr.shape[0]):
        if NEG[b]:
            lm = lm + Fn.softplus(r[b].max(1).values).sum() * 2.0
            n += 2
        for a, e, k in SEG[b]:
            lm = lm + Fn.softplus(-r[b, k, a:e].max())
            n += 1
    return lf + 0.5 * lh + lam * lm / max(n, 1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="s1")
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--no-app", action="store_true")
    ap.add_argument("--folds", default="01234")
    ap.add_argument("--final", action="store_true", help="hoc tren TOAN BO tep (khong giu fold) -> weights/seq.pt + seq_norm.npz (cho score.py)")
    a = ap.parse_args()
    import torch
    from multiprocessing import Pool
    from train_frames import file_job, pool_ids
    from v6.boundary import detect as D
    torch.set_num_threads(a.threads)
    t0 = time.time()
    items = D.load_items()
    ids = pool_ids(items)
    if a.no_app:
        ids = [i for i in ids if items[i]["cfg"] == "goc"]
    X = {}
    with Pool(min(a.threads, 22)) as pool:
        for i, x in pool.imap_unordered(file_job, [(i, None, None, None, "full") for i in ids], chunksize=16):
            X[i] = x.astype(np.float16)
    TG = {i: file_targets(items[i]["meta"], len(X[i]), items[i]["real"], items[i]["cfg"]) for i in ids}
    d_in = next(iter(X.values())).shape[1]
    print(f"{len(ids)} tep · {d_in} chieu · {time.time() - t0:.0f}s", flush=True)
    fold = {i: items[i]["fold"] for i in ids}
    ev = [i for i in ids if items[i]["cfg"] == "goc"]
    out_f, out_h = {}, {}
    for f in ([-1] if a.final else map(int, a.folds)):
        sd0 = a.seed * 10 + (f if f >= 0 else 9)
        torch.manual_seed(sd0)
        rng = np.random.default_rng(sd0)
        tr = [i for i in ids if fold[i] != f]
        samp = np.vstack([X[i][::7].astype(np.float32) for i in tr])
        mu, sd = samp.mean(0), samp.std(0) + 1e-3
        del samp
        mu_t, sd_t = torch.tensor(mu), torch.tensor(sd)
        net = build_model(d_in)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3)
        steps = a.epochs * (len(tr) // 32 + 1)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.15)
        lens = np.array([min(len(X[i]), CROP) for i in tr])
        for ep in range(a.epochs):
            net.train()
            order = np.argsort(lens + rng.uniform(0, 60, len(lens)))           # gom theo do dai
            batches = [order[k:k + 32] for k in range(0, len(order), 32)]
            rng.shuffle(batches)
            tot = 0.0
            for bi in batches:
                L = int(lens[bi].max())
                B = len(bi)
                xb = np.zeros((B, L, d_in), np.float32)
                yb, wb, mb = np.zeros((B, L, 2), np.float32), np.zeros((B, L, 2), np.float32), np.zeros((B, L), np.float32)
                yh, seg, neg = np.zeros((B, 2), np.float32), [], []
                for n, k in enumerate(bi):
                    i = tr[k]
                    x = X[i]
                    y, w, h, sg, ng = TG[i]
                    s0 = 0
                    if len(x) > L:
                        s0 = int(rng.integers(0, len(x) - L + 1))
                        if sg:                               # giu doan sua trong khung cat
                            a0, b0, _ = sg[int(rng.integers(len(sg)))]
                            lo, hi = max(0, b0 - L), min(a0, len(x) - L)
                            s0 = int(rng.integers(lo, hi + 1)) if hi >= lo else int(np.clip(a0 - 50, 0, len(x) - L))
                    e0 = min(len(x), s0 + L)
                    ln = e0 - s0
                    xb[n, :ln] = x[s0:e0]
                    yb[n, :ln], wb[n, :ln], mb[n, :ln] = y[s0:e0], w[s0:e0], 1
                    yh[n] = h
                    seg.append([(max(0, p - s0), min(ln, q - s0), c) for p, q, c in sg if q - s0 > 8 and p - s0 < ln - 8])
                    neg.append(ng)
                xt = (torch.tensor(xb) - mu_t) / sd_t
                xt = xt.clamp(-8, 8).transpose(1, 2)
                M = torch.tensor(mb)
                fr, hd = net(xt * M[:, None, :], M)
                loss = loss_fn(fr, hd, torch.tensor(yb).transpose(1, 2), torch.tensor(wb).transpose(1, 2), torch.tensor(yh), M, seg, neg, a.lam)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                sch.step()
                tot += float(loss) * B
            print(f"fold {f} ep {ep} loss {tot / len(tr):.4f} · {time.time() - t0:.0f}s", flush=True)
        net.eval()
        if a.final:
            W = os.path.join(HERE, "weights")
            os.makedirs(W, exist_ok=True)
            torch.save(net.state_dict(), os.path.join(W, "seq.pt"))
            np.savez(os.path.join(W, "seq_norm.npz"), mu=mu, sd=sd, d_in=d_in)
            print("-> weights/seq.pt", f"{time.time() - t0:.0f}s")
            return
        with torch.no_grad():
            for i in ev:
                if fold[i] != f:
                    continue
                xt = ((torch.tensor(X[i].astype(np.float32)) - mu_t) / sd_t).clamp(-8, 8).T[None]
                M = torch.ones(1, xt.shape[2])
                fr, hd = net(xt, M)
                out_f[i] = torch.sigmoid(fr[0]).T.numpy().astype(np.float16)
                out_h[i] = torch.sigmoid(hd[0]).numpy().astype(np.float32)
        print(f"fold {f} xong · {time.time() - t0:.0f}s", flush=True)
    p = os.path.join(CACHE, f"p3v2_seq_{a.tag}.pkl")
    pickle.dump({"frames": out_f, "heads": out_h}, open(p, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    print("->", p, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
