# -*- coding: utf-8 -*-
"""SAN PHAM: dau TINH CHINH (FT-2 / FT-3) chay tren CPU. Lay dau ra tang 6 cua XLS-R dong bang (da chay san trong Product.encode, int8), chay them 6 tang cuoi DA GOP LoRA
(W' = W + (alpha / r) (A B)^T, nen khong ton chi phi them cho nhanh LoRA) o int8, tang chuan hoa cuoi, tru trung vi theo khuc, chieu 64 + dac trung buoc nhay + TCN -> xac suat khung
(T, 6) [ghep, xoa, khu on, doi giong, doan ghep, giong may] dung bo cuc cot nhu OOF cua ft.py (nguong thresholds_v9<op>.json lay tu OOF cua CUNG cach chay: khuc 16 s chong 2 s).
  Khoi tao: FTHead(ckpt, layers6_fp32)  - layers6_fp32: cac tang 6..11 cua XLS-R dang fp32 (sao chep TRUOC khi luong tu hoa)."""
from __future__ import annotations

import copy
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as Fn  # noqa: E402

from v6.boundary import detect as D  # noqa: E402
from v6.boundary import tcn as T  # noqa: E402

N_LAYERS = 12
ALPHA = 32.0                                                      # = ft.LoRALinear alpha mac dinh
LORA_TARGETS = (("attention", ("q_proj", "k_proj", "v_proj", "out_proj")), ("feed_forward", ("intermediate_dense", "output_dense")))


def merge_lora(layers: list, sd: dict, first_index: int) -> list:
    """Gop nhanh LoRA vao Linear cua cac tang (sao chep, fp32): W' = W + (alpha / r) (A B)^T. sd: trang thai luu cua ft.FT.trainable_state(); first_index = chi so cua tang dau."""
    out = []
    for k, layer in enumerate(layers):
        layer = copy.deepcopy(layer).float()
        idx = first_index + k
        for owner_name, names in LORA_TARGETS:
            owner = getattr(layer, owner_name)
            for nm in names:
                a, b = sd.get(f"w.encoder.layers.{idx}.{owner_name}.{nm}.A"), sd.get(f"w.encoder.layers.{idx}.{owner_name}.{nm}.B")
                if a is None:
                    continue
                lin = getattr(owner, nm)
                with torch.no_grad():
                    lin.weight += (ALPHA / a.shape[1]) * (a.float() @ b.float()).T
        out.append(layer.eval())
    return out


class FTHead:
    def __init__(self, ckpt: str, layers_fp32: list, final_ln: nn.LayerNorm, int8: bool = True):
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        idx = sorted({int(k.split(".")[3]) for k in sd if k.startswith("w.encoder.layers.")})
        first = idx[0] if idx else N_LAYERS - len(layers_fp32)
        self.first = first
        src = layers_fp32[first - (N_LAYERS - len(layers_fp32)):]
        layers = []
        for k, layer in enumerate(src):                                  # gop + luong tu hoa TUNG tang: dinh bo nho ~1 tang fp32 thay vi 6 (luong tu hoa dong theo tung Linear -> ket qua nhu ca khoi)
            m = merge_lora([layer], sd, first + k)[0]
            layers.append(torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8) if int8 else m)
        self.layers = nn.ModuleList(layers)
        self.ln = copy.deepcopy(final_ln).float()
        if "w.encoder.layer_norm.weight" in sd:
            self.ln.weight.data.copy_(sd["w.encoder.layer_norm.weight"].float())
            self.ln.bias.data.copy_(sd["w.encoder.layer_norm.bias"].float())
        self.ln.eval()
        hid = self.ln.weight.shape[0]
        self.proj = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, 64)).eval()
        self.proj.load_state_dict({k[len("proj."):]: v.float() for k, v in sd.items() if k.startswith("proj.")})
        n_cls = sd["tcn.out.weight"].shape[0]
        self.tcn = T.TCN(cin=128, n_cls=n_cls).eval()
        self.tcn.load_state_dict({k[len("tcn."):]: v.float() for k, v in sd.items() if k.startswith("tcn.")})
        self.n_cls = n_cls

    @torch.no_grad()
    def chunk_probs(self, hs_first: torch.Tensor) -> np.ndarray:
        """hs_first: (1, T, 1024) dau ra cua tang truoc tang hoc dau tien (hidden_states[first]) -> (T, n_cls) xac suat softmax (cot 0 = nen)."""
        h = hs_first
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        h = self.ln(h).float()
        h = h - h.median(dim=1, keepdim=True).values
        p = self.proj(h).float()
        d = Fn.pad((p[:, 1:] - p[:, :-1]).abs(), (0, 0, 1, 0))
        j = torch.log1p(d / (d.median(dim=1, keepdim=True).values + 1e-6))
        out = self.tcn(torch.cat([p, j], -1).transpose(1, 2))
        return torch.softmax(out[0].float(), 0).T.cpu().numpy()

    @staticmethod
    def smooth_cols(p: np.ndarray) -> np.ndarray:
        """(T, n_cls) -> (T, n_cls - 1) float32: cot 1.. lam tron 3 khung (nhu OOF)."""
        return np.stack([D.smooth(p[:, c]) for c in range(1, p.shape[1])], 1).astype(np.float32)
