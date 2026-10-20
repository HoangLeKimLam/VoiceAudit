# -*- coding: utf-8 -*-
"""FT-1 - TINH CHINH MOT PHAN XLS-R (backbone chong gia mao DA HUAN LUYEN, khong hoc tu dau) cho bo do moi noi / vung.
Vi sao: khung XLS-R dong bang (L12, PCA 64) duoc hoc cho bai toan 'tep gia hay that', khong phai cho 'o dau co moi noi'. TCN tren khung dong bang bao hoa o ~0,5 AP
(fd_probe: them chieu hay doi tang deu khong hon) va gan nhu mu truoc sua KY (khop kenh + nen + cross-fade). Tinh chinh vai tang cuoi (7-12) CUNG DAU TCN cho phep backbone
rut dac trung nhay cam voi moi noi. Mac dinh LoRA hang 16 tren 6 tang cuoi (--rank 0: hoc ca tang, ton ~0,6 GB VRAM hon): may nay chi con ~4,5 GB bo nho cam ket cua Windows va VRAM dat truoc
tinh 1:1 vao do (da do: moi tien trinh CUDA ~2,3 GB may chu + VRAM) -> lan hoc ca 6 tang het cam ket va sap MemoryError; cau hinh nay dung ~2,7 GB. Cung nhan khung (tcn.make_labels, 7 lop), cung fold NGUOI NOI, cung dau ra OOF (T, 6) float16 [ghep, xoa, khu on, doi giong, doan ghep, giong may]
-> dung tiep decode / fuse9. KHONG dua tag ft vao cmp_oof (no doc cot 0 cho moi viec va can du 5 fold): dung `ft compare`.
Thu nghiem MOT CHIA: hoc tren cac fold != F, cham fold F + cac bo ngoai tap hoc (chain, benign, amilong; TCN dong bang cung KHONG hoc), so voi OOF cua TCN dong bang tren CUNG tep.
LUU Y khi doc ket qua (so sanh KHONG hoan toan cung dieu kien): du lieu giong TCN dong bang 'tcn' tru bo v4 (630 tep khong co wav); lich hoc khac - TCN dong bang 12 luot cua so <= 16 s
tren dac trung co dinh, FT chi 8 luot cua so 4 s -> neu FT <= TCN thi CHUA ket luan duoc, chi khi FT > TCN moi la bang chung (FT it luot hoc hon, it ngu canh hon).
Dau ra duong dong bang tru trung vi theo tep truoc khi chieu (nhu pipeline: (h - median) @ W) de khong hoc dac trung kenh / nguoi noi tuyet doi.
  python -m v6.boundary.ft train   --fold 0 --tag ft1 [--layers 6 --rank 16 --epochs 8 --bs 4 --crop 4] [--limit N]
  python -m v6.boundary.ft eval    --fold 0 --tag ft1 [--limit N]    -> boundary_oof_<tag>_f<F>.pkl (tep fold F + bo ngoai tap hoc)
  python -m v6.boundary.ft compare --fold 0 --tag ft1                -> .tmp_v6/results/ft_cmp_<tag>_f<F>.md      (--limit N: chay thu nho, vai phut)"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, ".tmp_v6", "torch_cu"))
sys.path.insert(1, BASE)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as Fn  # noqa: E402

from v6.boundary import detect as D  # noqa: E402
from v6.boundary import tcn as T  # noqa: E402

NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
SR, HOP, RF = 16000, 320, 400                       # moi khung 20 ms, truong nhin 25 ms
N_LAYERS = 12
CDIR = os.path.join(BASE, ".tmp_v6", "cache")
HELD = {"v6benign", "v6chain", "v6amilong"}          # nhu TCN dong bang 'tcn' (khong hoc chain / benign / tep dai) -> cham de so sanh tong quat
EVENT_BIAS = 0.6                                     # ti le cua so chua mot su kien (moi noi / vung)
EVAL_CHUNK_S = 16.0                                  # cham tep dai theo khuc 16 s (gan cua so hoc 8 s hon 24 s)
DEV = torch.device("cpu" if os.environ.get("FT_CPU") else "cuda")      # FT_CPU=1: chi de kiem thu ong dan (mo hinh nho)
AMP = DEV.type == "cuda"
MEM_FRAC = float(os.environ.get("FT_MEM_FRAC", "0.22"))      # tran bo nho GPU 1,8 GB (VRAM dat truoc tinh vao bo nho CAM KET cua Windows; lan chay dau het cam ket o ~5 GB -> sap MemoryError)


def cap_gpu_memory() -> None:
    if AMP:
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)


def n_frames(n: int) -> int:
    return (n - RF) // HOP + 1


KEEP_META = ("edits", "level2", "person", "speaker", "lang")
TAGS = (("natvi", "v6natvi"), ("naten", "v6naten"), ("amilong", "v6amilong"), ("cover", "v6cover"), ("cover2", "v6cover2"), ("benign", "v6benign"), ("chain", "v6chain"),
        ("ttsvi", "v6ttsvi"), ("ttsen", "v6ttsen"))


def light_items() -> dict:
    """id -> {set, kind, cfg, fold, real, wav, meta (thu gon)}: cung bo loc nhu detect.load_items (cfg trong CFGS, trong pham vi, khong lockbox - tru amilong) nhung doc tung manifest roi bo meta lon
    (K1, canh bao ...) va khong kiem tra khung: tiet kiem ~1,5 GB bo nho cam ket (VRAM + may chu cua tep dau tien da vuot 5 GB va sap)."""
    items = {}
    for s, tag in TAGS:
        mp = os.path.join(BASE, "data_labels", "v6", s, "manifest.json")
        if not os.path.exists(mp):
            continue
        for e in json.load(open(mp, encoding="utf-8")):
            m = e["meta"]
            if e.get("wav") and e["cfg"] in D.CFGS and not m.get("ngoai_pham_vi") and (not m.get("lockbox") or tag == "v6amilong"):
                items[e["id"]] = {"set": tag, "kind": e["kind"], "cfg": e["cfg"], "fold": D.person_fold(m), "real": not m.get("level2"),
                                  "wav": os.path.normpath(os.path.join(D.V4, e["wav"])), "meta": {k: m[k] for k in KEEP_META if k in m}}
    return items


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / np.sqrt(x.var() + 1e-7)


def oof_path(tag: str, fold: int) -> str:
    return D.tagged(D.OOFP, f"{tag}_f{fold}")


class LoRALinear(nn.Module):
    """Linear DONG BANG + nhanh LoRA: y = base(x) + (x A) B * (alpha / r); A, B fp32 hoc duoc, B khoi tao 0 (luc dau dung bang backbone goc). Chi ~2 trieu tham so cho 6 tang: khong can
    gradient / trang thai Adam cho ca ma tran 1024 x 4096 -> tiet kiem ~0,6 GB VRAM (VRAM dat truoc tinh vao bo nho cam ket cua Windows)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float = 32.0):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.randn(base.in_features, r, device=dev) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(r, base.out_features, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype) * self.scale


LORA_TARGETS = (("attention", ("q_proj", "k_proj", "v_proj", "out_proj")), ("feed_forward", ("intermediate_dense", "output_dense")))


class FT(torch.nn.Module):
    """XLS-R (12 tang) voi vai tang cuoi hoc duoc (rank > 0: chi nhanh LoRA tren cac Linear; rank = 0: hoc ca tang) + tru trung vi theo tep + chieu 64 + dac trung buoc nhay
    chuan hoa + TCN (cung kien truc dau dong bang)."""

    def __init__(self, n_train: int = 6, n_cls: int = 7, rank: int = 0, small: bool = False):
        super().__init__()
        if small:                                      # kiem thu: mo hinh nho khoi tao ngau nhien (KHONG dung de hoc)
            from transformers import Wav2Vec2Config, Wav2Vec2Model
            self.w = Wav2Vec2Model(Wav2Vec2Config(hidden_size=64, num_hidden_layers=N_LAYERS, num_attention_heads=4, intermediate_size=128, conv_dim=(32,) * 7,
                                                  do_stable_layer_norm=True, feat_extract_norm="layer", num_feat_extract_layers=7, layerdrop=0.0))
        else:
            from transformers import AutoModelForAudioClassification
            full = AutoModelForAudioClassification.from_pretrained(NAME, low_cpu_mem_usage=True, torch_dtype=torch.float16)      # nap fp16: nua bo nho may chu + VRAM cho phan dong bang
            self.w = full.wav2vec2
            del full
        self.w.encoder.layers = self.w.encoder.layers[:N_LAYERS]
        self.w.config.layerdrop = 0.0
        for p in self.w.parameters():
            p.requires_grad_(False)
        tl = list(self.w.encoder.layers[N_LAYERS - n_train:]) if n_train else []
        for layer in tl:
            if rank:
                for owner_name, names in LORA_TARGETS:
                    owner = getattr(layer, owner_name)
                    for nm in names:
                        setattr(owner, nm, LoRALinear(getattr(owner, nm), rank))        # tham so moi (A, B) hoc duoc mac dinh
            else:
                layer.float()                                   # hoc ca tang: o fp32 (phan dong bang giu fp16: cung so lieu voi autocast)
                for p in layer.parameters():
                    p.requires_grad_(True)
        if n_train:
            self.w.encoder.layer_norm.float()
            for p in self.w.encoder.layer_norm.parameters():
                p.requires_grad_(True)
        hid = self.w.config.hidden_size
        self.proj = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 64))
        self.tcn = T.TCN(cin=128, n_cls=n_cls)

    def train(self, mode: bool = True):
        super().train(mode)
        self.w.eval()                                # backbone luon o che do eval: khong dropout / che thoi gian (cau hinh goc: layerdrop 0,1, SpecAugment)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x (B, N) -> (B, n_cls, T)
        h = self.w(x).last_hidden_state.float()
        h = h - h.median(dim=1, keepdim=True).values              # bo lech kenh / nguoi noi theo tep, nhu duong dong bang
        p = self.proj(h).float()
        d = Fn.pad((p[:, 1:] - p[:, :-1]).abs(), (0, 0, 1, 0))
        j = torch.log1p(d / (d.median(dim=1, keepdim=True).values + 1e-6))
        return self.tcn(torch.cat([p, j], -1).transpose(1, 2))

    def trainable_state(self) -> dict:
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}


def rank_of(sd: dict) -> int:
    """Hang LoRA suy tu khoa ...A trong tep luu (0 = hoc ca tang)."""
    return next((int(v.shape[1]) for k, v in sd.items() if k.endswith(".A")), 0)


def n_train_of(sd: dict) -> int:
    """So tang cuoi hoc duoc suy tu cac khoa trong tep luu (w.encoder.layers.<chi so>...)."""
    idx = [int(k.split(".")[3]) for k in sd if k.startswith("w.encoder.layers.")]
    return N_LAYERS - min(idx) if idx else 0


def setup(fold: int, limit: int = 0) -> tuple:
    """(items, wp, tr, te, extra): tr = hoc (fold != F, tru bo ngoai tap hoc); te = cham (fold F); extra = bo ngoai tap hoc (chain, benign, amilong). limit > 0: lay mau cho chay thu."""
    T.set_regions(2)
    items = light_items()
    wp = {i: it["wav"] for i, it in items.items()}
    ok = lambda i: os.path.exists(wp[i])                                                      # noqa: E731
    ids = [i for i in sorted(items) if items[i]["set"] not in HELD and ok(i)]
    tr = [i for i in ids if items[i]["fold"] != fold]
    te = [i for i in ids if items[i]["fold"] == fold]
    extra = [i for i in sorted(items) if items[i]["set"] in HELD and ok(i)]
    if limit:
        sub = lambda v: v[::max(1, len(v) // limit)]                                           # noqa: E731
        tr, te, extra = sub(tr), sub(te), sub(extra)
    return items, wp, tr, te, extra


def make_batches(tr: list, items: dict, wp: dict, N: dict, Y: dict, bs: int, crop: int, rng: np.random.Generator):
    order = sorted(tr, key=lambda i: N[i] * rng.uniform(0.85, 1.15))      # xep theo do dai (cua so cat theo tep ngan nhat trong lo) co nhieu -> lo doi moi epoch
    chunks = [order[k:k + bs] for k in range(0, len(order), bs)]
    rng.shuffle(chunks)
    for ch in chunks:
        n = min(crop * SR, min(N[i] for i in ch))
        tc = n_frames(n)
        xs, ys = [], []
        for i in ch:
            y = Y[i]
            t0 = 0
            if len(y) > tc:
                ev = np.flatnonzero(y > 0)
                if len(ev) and rng.random() < EVENT_BIAS:
                    c = int(rng.choice(ev))
                    t0 = int(np.clip(c - tc // 2 + rng.integers(-tc // 4, tc // 4 + 1), 0, len(y) - tc))
                else:
                    t0 = int(rng.integers(0, len(y) - tc + 1))
            wav, _ = sf.read(wp[i], start=t0 * HOP, frames=n, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(1)
            if len(wav) < n:
                wav = np.pad(wav, (0, n - len(wav)))
            xs.append(norm(wav))
            ys.append(y[t0:t0 + tc])
        m = min(len(v) for v in ys)
        yield torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack([v[:m] for v in ys]).astype(np.int64))


def train(fold: int, tag: str, n_layers: int, epochs: int, bs: int, crop: int, lr: float, limit: int = 0, rank: int = 0) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(100 + fold)
    cap_gpu_memory()
    items, wp, tr, te, _ = setup(fold, limit)
    N = {i: sf.info(wp[i]).frames for i in tr}
    tr = [i for i in tr if N[i] >= 3 * SR]
    Y = {i: T.make_labels(items[i]["meta"], n_frames(N[i])).astype(np.int8) for i in tr}
    items = None                                                    # meta lon (K1, canh bao ...): giai phong bo nho cam ket truoc khi nap GPU
    print(f"hoc {len(tr)} tep (fold != {fold}), cham {len(te)} tep (fold {fold}); {n_layers} tang cuoi hoc duoc; cua so {crop} s, lo {bs}", flush=True)
    cnt = np.bincount(np.concatenate([Y[i] for i in tr]).astype(np.int64), minlength=T.N_CLS) + 1
    w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32).to(DEV)
    lossf = nn.CrossEntropyLoss(weight=w / w.mean(), ignore_index=-1)
    model = FT(n_layers, T.N_CLS, rank, small=bool(os.environ.get("FT_CPU"))).to(DEV).train()
    bb = [p for p in model.w.parameters() if p.requires_grad]
    hd = [*model.proj.parameters(), *model.tcn.parameters()]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr, "weight_decay": 1e-2}, {"params": hd, "lr": 1.5e-3, "weight_decay": 1e-3}], fused=AMP)      # fused: khong tao ban sao tam thoi
    steps = epochs * ((len(tr) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, 1.5e-3], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    rng = np.random.default_rng(100 + fold)
    path = os.path.join(CDIR, f"ft_{tag}_f{fold}.pt")
    t0, step, skipped = time.time(), 0, 0
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for x, y in make_batches(tr, None, wp, N, Y, bs, crop, rng):
            x, y = x.to(DEV), y.to(DEV)
            try:
                with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
                    out = model(x)
                m = min(out.shape[-1], y.shape[-1])
                loss = lossf(out[..., :m].float(), y[:, :m])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([*bb, *hd], 1.0)
                scaler.step(opt)
                scaler.update()
            except torch.cuda.OutOfMemoryError:                           # lo qua lon cho bo nho GPU con trong: bo lo nay, tiep tuc
                out = loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                if skipped > 50:
                    raise
                continue
            sched.step()
            tot, nb, step = tot + float(loss), nb + 1, step + 1
            if step % 200 == 0:
                print(f"  epoch {ep + 1}/{epochs} buoc {step}/{steps} loss {tot / nb:.4f} ({(time.time() - t0) / 60:.1f} phut)", flush=True)
        torch.save(model.trainable_state(), path + ".tmp")
        os.replace(path + ".tmp", path)
        print(f"epoch {ep + 1}/{epochs} xong: loss {tot / max(nb, 1):.4f}, {(time.time() - t0) / 60:.1f} phut, bo qua {skipped} lo -> {path}", flush=True)


@torch.no_grad()
def infer_file(model: FT, wav: np.ndarray, chunk_s: float = EVAL_CHUNK_S, ovl_s: float = 2.0) -> np.ndarray | None:
    """(T, n_cls) xac suat, dung n_frames(len(wav)) khung; tep dai > chunk_s thi chia khuc chong ovl_s, bo nua phan chong moi dau (khop moc khung, khong mat khung); tep qua ngan -> None."""
    n, L, ov = len(wav), int(chunk_s * SR), int(ovl_s * SR)
    assert (L - ov) % HOP == 0 and ov % (2 * HOP) == 0, "buoc khuc va nua chong phai la boi cua 320 mau"
    starts = (list(range(0, max(1, n - ov), L - ov)) or [0]) if n > L else [0]
    half, step = ov // 2 // HOP, (L - ov) // HOP
    parts = []
    for ci, s0 in enumerate(starts):
        seg = wav[s0:s0 + L]
        if len(seg) < RF + HOP:
            break
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            out = model(torch.from_numpy(norm(seg))[None].to(DEV))
        p = torch.softmax(out[0].float(), 0).T.cpu().numpy()
        a = 0 if ci == 0 else half
        b = len(p) if ci == len(starts) - 1 or s0 + L >= n else step + half
        parts.append(p[a:b])
    return np.concatenate(parts) if parts else None


def evaluate(fold: int, tag: str, limit: int = 0) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cap_gpu_memory()
    items, wp, tr, te, extra = setup(fold, limit)
    te, extra = [i for i in te if items[i]["cfg"] == "goc"], [i for i in extra if items[i]["cfg"] == "goc"]      # so sanh chi can tep goc (kenh dien thoai: bo, nhanh 2 lan)
    items = None
    sd = torch.load(os.path.join(CDIR, f"ft_{tag}_f{fold}.pt"), map_location="cpu", weights_only=True)
    model = FT(n_train_of(sd), T.N_CLS, rank_of(sd), small=bool(os.environ.get("FT_CPU"))).to(DEV).eval()
    res = model.load_state_dict(sd, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys
    oof, bad, t0 = {}, [], time.time()
    for k, i in enumerate(te + extra):
        try:
            wav, _ = sf.read(wp[i], dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            p = infer_file(model, wav)
            if p is None or not np.isfinite(p).all():
                bad.append(i)
                continue
            oof[i] = np.stack([D.smooth(p[:, c]) for c in range(1, T.N_CLS)], 1).astype(np.float16)
        except Exception as ex:                                       # noqa: BLE001  mot tep hong khong lam hong ca luot cham
            bad.append(i)
            print(f"  bo {i}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(te) + len(extra)} ({(time.time() - t0) / 60:.1f} phut)", flush=True)
    outp = oof_path(tag, fold)
    pickle.dump(oof, open(outp + ".tmp", "wb"))
    os.replace(outp + ".tmp", outp)
    print(f"{len(oof)} tep -> {outp}; bo {len(bad)} tep", flush=True)


def compare(fold: int, tag: str) -> None:
    """Cung tep: AUC (tep co thao tac vs tep that) va TPR o vu oan 3 / 5 / 10 % (nguong = phan vi cua tep that CUNG nhom, CHUNG cho hai ben) cua FT-1 va TCN dong bang."""
    from v6.natural import req_check as RC
    sys.stdout.reconfigure(encoding="utf-8")
    items = D.load_items(keep=("edits", "level2"))
    ft = pickle.load(open(oof_path(tag, fold), "rb"))
    base = pickle.load(open(D.tagged(D.OOFP, "tcn"), "rb"))
    reg = pickle.load(open(D.tagged(D.OOFP, "tcn_reg"), "rb"))
    tasks = [("03 ghep / chen", ("thay_that", "chen_that"), "sub", 0, 0, 0), ("03 xoa loi", ("cat_bo",), "sub", 1, 1, 1),
             ("02 khu on mot doan", ("02_nen_cuc_bo",), "level2", 2, 2, 2), ("02 doi giong mot doan", ("02_giong_cuc_bo",), "level2", 3, 3, 3)]
    L = [f"# FT-1 ({tag}, fold {fold}) so voi TCN dong bang - tep goc, cung tep, nguong = phan vi tep that cung nhom (chung hai ben)\n",
         "| Bo | Chuyen gia | n sua / that | TCN dong bang: AUC · TPR 3/5/10 % | FT-1: AUC · TPR 3/5/10 % |", "|---|---|---|---|---|"]
    groups = (("tieng Viet (fold %d)" % fold, ("v6natvi",), True), ("tieng Anh (fold %d)" % fold, ("v6naten",), True),
              ("chain (dieu kien chong; khong hoc)", ("v6chain",), False), ("amilong (hop dai; khong hoc)", ("v6amilong",), False))
    for sname, sets, by_fold in groups:
        ids = [i for i, it in items.items() if it["set"] in sets and it["cfg"] == "goc" and (it["fold"] == fold or not by_fold) and i in ft and i in base and i in reg]
        real = [i for i in ids if items[i]["real"]]
        if len(real) < 5:
            continue
        for name, keys, field, cb, cr, cf in tasks:
            pos = [i for i in ids if any(e.get(field) in keys for e in items[i]["meta"].get("edits", []))]
            if len(pos) < 5:
                continue
            cells = []
            for src, col in ((base if cb < 2 else reg, cb if cb < 2 else cr), (ft, cf)):
                sc = {i: float(src[i].astype(float)[:, col].max()) for i in ids}
                a = RC.auc([sc[i] for i in pos], [sc[i] for i in real])
                tp = [np.mean([sc[i] > np.quantile([sc[j] for j in real], 1 - q) for i in pos]) for q in (0.03, 0.05, 0.10)]
                cells.append(f"{a:.3f} · " + "/".join(f"{100 * x:.0f}" for x in tp) + " %")
            L.append(f"| {sname} | {name} | {len(pos)} / {len(real)} | {cells[0]} | {cells[1]} |")
    open(os.path.join(D.RES, f"ft_cmp_{tag}_f{fold}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "eval", "compare"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="ft1")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--rank", type=int, default=16, help="hang LoRA (0 = hoc ca cac tang cuoi, ton ~0,6 GB VRAM hon)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--crop", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.0, help="toc do hoc backbone; 0 = tu chon (LoRA 2e-4, hoc ca tang 3e-5)")
    ap.add_argument("--limit", type=int, default=0, help="chay thu: lay khoang N tep moi nhom")
    a = ap.parse_args()
    lr = a.lr or (2e-4 if a.rank else 3e-5)
    {"train": lambda: train(a.fold, a.tag, a.layers, a.epochs, a.bs, a.crop, lr, a.limit, a.rank), "eval": lambda: evaluate(a.fold, a.tag, a.limit),
     "compare": lambda: compare(a.fold, a.tag)}[a.step]()
