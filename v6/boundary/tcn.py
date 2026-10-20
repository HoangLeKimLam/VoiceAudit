# -*- coding: utf-8 -*-
"""v6f / N5.1b - DAU TCN (mang tich chap gian cach 1D) tren khung XLS-R L12 (PCA 64) thay cho HGB tung khung.
Vi sao: HGB nhin tung khung voi 192 dac trung tu che (z, buoc nhay chuan hoa, tuong phan cuc bo +-5 khung = chi +-0,1 s); mot mang gian cach
(truong nhin ~ +-1,3 s) co the so SANH hai ben moi noi (cung phong / cung duong truyen / cung xu ly?) va hoc dac trung lien tiep thay vi tu che.
Backbone XLS-R chong gia mao van DONG BANG (khung da trich san); chi dau nho hoc moi (khoang 0,2 trieu tham so). Cung nhan khung, cung fold NGUOI NOI
(D.person_fold), cung dau ra OOF dang (T, 3) float16 [ghep, xoa, 02 cuc bo] da lam tron 3 khung -> dung tiep decode_file / sweep / fuse8.
  python -m v6.boundary.tcn train --tag tcn [--with-chain --with-benign] [--limit 600] [--epochs 12]
  python -m v6.boundary.tcn train --tag tcn_reg --regions   (nhan VUNG: dau ra (ghep, xoa, trong doan khu on, trong doan doi giong))
  python -m v6.boundary.tcn infer --tag tcn          (cham them tep moi trich khung bang mo hinh fold da luu)"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, ".tmp_v6", "torch_cu"))
sys.path.insert(1, BASE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from v6.boundary import detect as D  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CDIR = os.path.join(BASE, ".tmp_v6", "cache")
WIN, BATCH, C_HID, DILS = 800, 16, 96, (1, 2, 4, 8, 16, 1)
N_CLS = 4
REGIONS = 0            # --regions: nhan VUNG (trong doan khu on / doi giong mot doan) thay cho nhan RANH GIOI 02: 5 lop, dau ra (ghep, xoa, vung khu on, vung doi giong)
                       # --regions2: them VUNG doan ghep loi that (lop 5) va VUNG doan giong may (lop 6): 7 lop, dau ra (ghep, xoa, khu on, doi giong, doan ghep, giong may)


class Block(nn.Module):
    def __init__(self, c: int, d: int):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 5, padding=2 * d, dilation=d)
        self.norm = nn.GroupNorm(1, c)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.norm(self.conv(x)))


class TCN(nn.Module):
    def __init__(self, cin: int = 128, n_cls: int | None = None):
        super().__init__()
        self.inp = nn.Conv1d(cin, C_HID, 1)
        self.blocks = nn.Sequential(*[Block(C_HID, d) for d in DILS])
        self.out = nn.Conv1d(C_HID, n_cls or N_CLS, 1)

    def forward(self, x):                                  # (B, cin, T) -> (B, N_CLS, T)
        return self.out(self.blocks(self.inp(x)))


def make_labels(meta: dict, T: int) -> np.ndarray:
    """Nhan khung: che do mac dinh = D.labels (0 nen, 1 ranh gioi ghep, 2 moi noi xoa, 3 ranh gioi 02); --regions: lop 3 / 4 = TRONG doan khu on / doi giong mot doan."""
    y = D.labels(meta, T)
    if not REGIONS:
        return y
    y[y == 3] = 0
    t = (np.arange(T) + 0.5) * D.DT
    for e in meta.get("edits", []):
        if e.get("kind") != "segment" or e.get("start") is None:
            continue
        c = {"02_nen_cuc_bo": 3, "02_giong_cuc_bo": 4}.get(e.get("level2"))
        if REGIONS >= 2 and c is None and e.get("sub") != "cat_bo":
            c = {"03_ghep_giong_that": 5, "03_chen_giong_may": 6}.get(e.get("level2"))
        if c is not None:
            y[(t >= e["start"]) & (t <= e["end"]) & (y == 0)] = c
    return y


def file_inputs(z: np.ndarray, med: np.ndarray) -> np.ndarray:
    """(T, 128): z va log1p(|buoc nhay| / trung vi tep) - cung y nghia voi detect.feats nhung de mang tu hoc tuong phan cuc bo."""
    z = z.astype(np.float32)
    d = np.abs(np.diff(z, axis=0, prepend=z[:1]))
    return np.c_[z, np.log1p(d / (med + 1e-6))].astype(np.float32)


def load_all(ids: list, items: dict) -> tuple:
    """Khung (T, 64) float16 cua moi tep nap qua MOT tep memmap tren D (khong tinh vao bo nho cam ket; khoa theo danh sach id) + trung vi buoc nhay + nhan khung (int8)."""
    import hashlib
    key = hashlib.sha1("|".join(ids).encode()).hexdigest()[:16]
    path, idx = os.path.join(CDIR, f"tcn_frames_{key}.f16"), os.path.join(CDIR, f"tcn_frames_{key}.npz")
    if not (os.path.exists(path) and os.path.exists(idx)):
        offs, meds = [0], []
        with open(path + ".tmp", "wb") as fh:
            for n, i in enumerate(ids):
                z = np.load(os.path.join(D.FR, i + ".npy"))
                fh.write(np.ascontiguousarray(z, np.float16).tobytes())
                offs.append(offs[-1] + len(z))
                d = np.abs(np.diff(z.astype(np.float32), axis=0, prepend=z[:1].astype(np.float32)))
                meds.append(np.median(d, axis=0))
                if (n + 1) % 5000 == 0:
                    print(f"  nap {n + 1}/{len(ids)}", flush=True)
        os.replace(path + ".tmp", path)
        np.savez(idx, offs=np.array(offs, np.int64), med=np.array(meds, np.float32))
    z = np.load(idx)
    offs, med = z["offs"], z["med"]
    mm = np.memmap(path, np.float16, "r", shape=(int(offs[-1]), 64))
    Z = {i: mm[offs[k]:offs[k + 1]] for k, i in enumerate(ids)}
    MED = {i: med[k] for k, i in enumerate(ids)}
    Y = {i: make_labels(items[i]["meta"], len(Z[i])).astype(np.int8) for i in ids}
    return Z, MED, Y


def batches(idx: list, Z: dict, MED: dict, Y: dict, rng: np.random.Generator):
    idx = list(idx)
    rng.shuffle(idx)
    for b in range(0, len(idx), BATCH):
        sel = idx[b:b + BATCH]
        T = max(min(len(Z[i]), WIN) for i in sel)
        x = np.zeros((len(sel), T, 2 * Z[sel[0]].shape[1]), np.float32)
        y = -np.ones((len(sel), T), np.int64)
        for k, i in enumerate(sel):
            n = len(Z[i])
            s = int(rng.integers(0, n - WIN + 1)) if n > WIN else 0
            e = min(n, s + WIN)
            x[k, :e - s] = file_inputs(Z[i], MED[i])[s:e]
            y[k, :e - s] = Y[i][s:e].astype(np.int64)
        yield torch.from_numpy(x).transpose(1, 2).to(DEV), torch.from_numpy(y).to(DEV)


def fit(train_ids: list, Z: dict, MED: dict, Y: dict, epochs: int, seed: int) -> TCN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TCN().to(DEV)
    cnt = np.bincount(np.concatenate([Y[i] for i in train_ids]).astype(np.int64), minlength=N_CLS) + 1
    w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32, device=DEV)
    lossf = nn.CrossEntropyLoss(weight=w / w.mean(), ignore_index=-1)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    steps = epochs * ((len(train_ids) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in batches(train_ids, Z, MED, Y, rng):
            loss = lossf(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        print(f"    epoch {ep + 1}/{epochs} loss {tot / max(1, len(train_ids) / BATCH):.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def score(model: TCN, ids: list, Z: dict, MED: dict) -> dict:
    out = {}
    for i in ids:
        x = torch.from_numpy(file_inputs(Z[i], MED[i])).T[None].to(DEV)
        p = torch.softmax(model(x)[0], 0).cpu().numpy()
        out[i] = np.stack([D.smooth(p[c]) for c in range(1, N_CLS)], 1).astype(np.float16)
    return out


def paths(tag: str) -> tuple:
    return os.path.join(CDIR, f"boundary_oof_{tag}.pkl"), [os.path.join(CDIR, f"tcn_{tag}_f{f}.pt") for f in range(5)]


def set_regions(level: int) -> None:
    global REGIONS, N_CLS
    REGIONS, N_CLS = level, {0: 4, 1: 5, 2: 7}[level]


def train(tag: str, with_chain: bool, with_benign: bool, limit: int, epochs: int) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    ids = sorted(items)
    if limit:
        ids = ids[::max(1, len(ids) // limit)]
    held = {s for s, w in (("v6benign", with_benign), ("v6chain", with_chain)) if not w} | {"v6amilong"}
    print(f"{len(ids)} tep ({DEV}); hoc tren cac bo ngoai {sorted(held)}", flush=True)
    Z, MED, Y = load_all(ids, items)
    oof_p, wpaths = paths(tag)
    oof = {}
    t0 = time.time()
    for f in range(5):
        tr = [i for i in ids if items[i]["fold"] != f and items[i]["set"] not in held]
        te = [i for i in ids if items[i]["fold"] == f]
        print(f"  fold {f + 1}/5: hoc {len(tr)} tep, cham {len(te)} tep", flush=True)
        model = fit(tr, Z, MED, Y, epochs, 100 + f)
        torch.save(model.state_dict(), wpaths[f])
        oof.update(score(model, te, Z, MED))
        pickle.dump(oof, open(oof_p + ".tmp", "wb"))
        os.replace(oof_p + ".tmp", oof_p)
        print(f"  fold {f + 1}/5 xong, {(time.time() - t0) / 60:.1f} phut", flush=True)
        del model
        torch.cuda.empty_cache()
    print("tcn train xong", flush=True)


def final(tag: str, with_chain: bool, with_benign: bool, epochs: int) -> None:
    """Mo hinh CUOI cho san pham: hoc MOT lan tren TOAN BO tep hoc (khong chua fold nao); luu tcn_<tag>_final.pt. Diem nguong van lay tu OOF (train)."""
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    ids = sorted(items)
    held = {s for s, w in (("v6benign", with_benign), ("v6chain", with_chain)) if not w} | {"v6amilong"}
    tr = [i for i in ids if items[i]["set"] not in held]
    print(f"{len(tr)} tep hoc ({DEV}); khong hoc tren {sorted(held)}", flush=True)
    Z, MED, Y = load_all(ids, items)                     # cung tep memmap voi 'train' (khoa theo danh sach id day du): khong them ~2 GB tren D
    model = fit(tr, Z, MED, Y, epochs, 777)
    torch.save(model.state_dict(), os.path.join(CDIR, f"tcn_{tag}_final.pt"))
    print(f"da luu tcn_{tag}_final.pt", flush=True)


def infer(tag: str) -> None:
    """Cham them cac tep da co khung nhung chua co trong OOF (vd bo chain moi) bang mo hinh fold da luu."""
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    oof_p, wpaths = paths(tag)
    oof = pickle.load(open(oof_p, "rb"))
    todo = [i for i in sorted(items) if i not in oof]
    print(f"{len(todo)} tep moi", flush=True)
    Z, MED, _ = {}, {}, None
    for i in todo:
        z = np.load(os.path.join(D.FR, i + ".npy"))
        Z[i] = z
        d = np.abs(np.diff(z.astype(np.float32), axis=0, prepend=z[:1].astype(np.float32)))
        MED[i] = np.median(d, axis=0)
    for f in range(5):
        model = TCN().to(DEV)
        model.load_state_dict(torch.load(wpaths[f], weights_only=True))
        model.eval()
        oof.update(score(model, [i for i in todo if items[i]["fold"] == f], Z, MED))
    pickle.dump(oof, open(oof_p + ".tmp", "wb"))
    os.replace(oof_p + ".tmp", oof_p)
    print("xong", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "infer", "final"])
    ap.add_argument("--tag", default="tcn")
    ap.add_argument("--with-chain", action="store_true")
    ap.add_argument("--with-benign", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="chay thu: lay khoang N tep")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--regions", action="store_true", help="nhan vung (trong doan 02 mot doan) thay nhan ranh gioi 02; luu y: phai dung cung co o 'infer'")
    ap.add_argument("--regions2", action="store_true", help="nhu --regions + vung doan ghep loi that + vung doan giong may (7 lop)")
    a = ap.parse_args()
    set_regions(2 if a.regions2 else 1 if a.regions else 0)
    {"train": lambda: train(a.tag, a.with_chain, a.with_benign, a.limit, a.epochs), "infer": lambda: infer(a.tag),
     "final": lambda: final(a.tag, a.with_chain, a.with_benign, a.epochs)}[a.step]()
