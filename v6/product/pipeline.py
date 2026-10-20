# -*- coding: utf-8 -*-
"""SAN PHAM v9 (chay tren CPU, chi dung XLS-R dong bang + dau nho): tep am thanh -> K1 -> XLS-R (17 tang, khuc 32 s chong 2 s, int8 tuy chon) ->
  khung L12 (PCA 64) + vector gop tho (tang 8 / 12 / 16)  ->  TCN ranh gioi (ghep / xoa)  ·  TCN vung (khu on / doi giong / doan ghep / giong may)
  ->  dau ca file (khu on / doi giong / giong may ca file) -> giai ma + nguong theo ngon ngu + <= 2 dong / khuc 14 s -> nhan tep 04 > 03 > 02 > 01.
Do thoi gian tung tang (RTF = giay xu ly / giay am thanh; Req 15: <= 0,6 tren CPU 4 nhan).
  python -m v6.product.pipeline <tep.wav> --op A2 [--lang vi|en] [--fp32] [--threads 4] [--bound tcn --reg tcn_reg --heads h2 --thr thresholds_v9A.json] [--fold k] [--strict]
TRUY VET mo hinh <-> nguong: tep nguong (fuse9 thr) ghi params bound / reg / heads (+ sha1 tung checkpoint). Product TU CHOI (ValueError) neu --bound / --reg / --heads khac
params cua tep nguong, hoac sha1 checkpoint nap khac sha1 ghi trong tep (mo hinh da hoc lai sau khi dat nguong). Bo trong (None) = lay theo tep nguong. Tep nguong cu khong co
bound / reg / heads (thresholds_v9A.json, 03/10) = tcn / tcn_reg / h2. Ket qua analyse() ghi out["models"] (ten + sha1 checkpoint, tep nguong, fold, hieu chinh).
Nhan (tag) -> mo hinh:  tcn -> tcn_tcn_*.pt · tcn_reg -> tcn_tcn_reg_*.pt · tcn_r2 -> tcn_tcn_r2_*.pt · fuse1 / fuse1x = FT-2 LoRA (ft_ft2_*.pt, fthead.py) + tcn_reg, trung binh logit (nhu ens6).
Che do mac dinh (nguoi dung that): mo hinh CUOI (*_final.pt / ft_ft2_f-1.pt; tcn khong co _final -> trung binh 5 fold), dau ca file heads_final_<heads>.joblib.
Che do DINH TUYEN FOLD (fold=k, chi de danh gia NGOAI MAU): tcn_<tag>_f{k}.pt, ft_ft2_f{k}.pt, heads_final_<heads>_f{k}.joblib (heads2 final --per-fold) = mo hinh KHONG thay
fold nguoi noi k (v6.boundary.detect.person_fold) + nguong thr_fold[k] (nguong tu tep that cua cac fold KHAC, nhu nghien cuu) neu tep nguong co. set_fold(k) doi fold khong nap lai XLS-R.
Xac suat ca tep: det["head_prob"] = {nen, giong, ttsf: p} tu v6.product.calib (calib_<heads>.json) hoac None neu chua hieu chinh; det["head"] van la diem tho;
det["head_prob_method"] = {khoa: isotonic / platt [(gop all)]} ghi bo hieu chinh da dung; out["models"]["calib_scope"]: bo hieu chinh hoc tren OOF ca 5 fold.
12/10: bo hieu chinh chi dung khi calib.source.sha1 == params.oof_sha1.heads cua tep nguong (khac -> head_prob None + out["models"]["calib_check"] ghi ly do);
bo hieu chinh KHONG TIN CAY (calib.issue_of) -> xac suat khoa do None; strict=True (--strict; e2e_test / check_decode mac dinh): checkpoint chua co trong tep nguong,
tep nguong khong ghi sha1, nguon hieu chinh khong kiem duoc -> ValueError. load_models / set_fold nguyen tu: loi thi giu nguyen fold cu."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

SR = 16000
NAME = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
FR = os.path.join(BASE, "data_labels", "realistic_v4", "xlsr_frames")
CACHE = os.path.join(BASE, ".tmp_v6", "cache")
RES = os.path.join(BASE, ".tmp_v6", "results")
LAYERS = (8, 12, 16)
CHUNK_S, OVL_S, DT = float(os.environ.get("PRODUCT_CHUNK_S", "32")), 2.0, 0.02      # khuc XLS-R: khung huan luyen duoc trich MOT luot ca tep (~<= 32 s) -> khuc 32 s giu nguyen
                                                                                       # ngu canh voi hau het tep tu nhien (khuc 16 s cua rang buoc 'toi da 2 dong' la viec khac, o ticket.py)
                                                                                       # LUU Y fuse1: OOF cua FT-2 cham theo khuc 16 s (ft.EVAL_CHUNK_S) -> tep > 16 s lech ngu canh (check_decode do)
LEGACY = {"bound": "tcn", "reg": "tcn_reg", "heads": "h2"}         # tep nguong cu khong ghi bound / reg / heads (fuse9 thr mac dinh)
ALIAS = {"fuse1x": "fuse1", "tcn_regx": "tcn_reg", "ft2x": "ft2"}   # OOF mo rong (ft_infer_extra) = CUNG mo hinh -> cung checkpoint
FUSE = {"fuse1": ("ft2", "tcn_reg")}                                # nhan hop nhat -> (nhan FT, nhan TCN dong bang)
EPS = 1e-4
_SHA = {}


def canon(tag: str | None) -> str | None:
    return ALIAS.get(tag, tag) if tag else tag


def sha1_file(path: str) -> str:
    st = os.stat(path)
    k = (os.path.abspath(path), st.st_size, st.st_mtime)
    if k not in _SHA:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        _SHA[k] = h.hexdigest()
    return _SHA[k]


def tcn_paths(tag: str, fold: int | None) -> list:
    """Checkpoint TCN cho nhan: fold k -> [tcn_<tag>_f{k}.pt]; mac dinh -> [_final.pt] neu co, khong thi 5 fold (trung binh)."""
    if fold is not None:
        return [os.path.join(CACHE, f"tcn_{tag}_f{fold}.pt")]
    return sorted(glob.glob(os.path.join(CACHE, f"tcn_{tag}_final.pt"))) or [os.path.join(CACHE, f"tcn_{tag}_f{f}.pt") for f in range(5)]


def ft_path(tag: str, fold: int | None) -> str:
    return os.path.join(CACHE, f"ft_{tag}_f{-1 if fold is None else fold}.pt")


def heads_path(heads: str, fold: int | None) -> str:
    return os.path.join(CACHE, f"heads_final_{heads}.joblib" if fold is None else f"heads_final_{heads}_f{fold}.joblib")


def expected_ckpts(bound: str, reg: str, heads: str | None, fold: int | None) -> list:
    """[(vai tro, duong dan)] moi checkpoint Product se nap cho (bound, reg, heads, fold) - fuse9 thr ghi sha1 cua chinh danh sach nay."""
    out, seen = [], set()
    for role, tag in (("bound", canon(bound)), ("reg", canon(reg))):
        if not tag:
            continue
        ft_tag, tcn_tag = FUSE.get(tag, (None, tag))
        cand = ([("ft", ft_path(ft_tag, fold))] if ft_tag else []) + [("tcn", p) for p in tcn_paths(tcn_tag, fold)]
        for kind, p in cand:
            if p not in seen:
                seen.add(p)
                out.append((f"{role}:{kind}", p))
    if heads:
        out.append(("heads", heads_path(heads, fold)))
    return out


def resolve_params(prm: dict, bound: str | None, reg: str | None, heads: str | None, thr_name: str = "") -> tuple:
    """(bound, reg, heads, legacy): lay theo tep nguong; tham so truyen vao (khac None) phai KHOP (sau ALIAS) neu khong -> ValueError."""
    legacy = not all(k in prm for k in ("bound", "reg", "heads"))
    want = {k: prm.get(k, LEGACY[k]) for k in ("bound", "reg", "heads")}
    for k, given in (("bound", bound), ("reg", reg), ("heads", heads)):
        if given is not None and canon(given) != canon(want[k]):
            raise ValueError(f"tep nguong {thr_name} dat cho {k}={want[k]!r}{' (mac dinh tep cu)' if legacy and k not in prm else ''} nhung mo hinh yeu cau {k}={given!r}: "
                             "dung tep nguong cua dung mo hinh (fuse9 thr --bound/--reg/--heads) hoac bo tham so de lay theo tep")
    return want["bound"], want["reg"], want["heads"], legacy


def check_sha(prm: dict, loaded: list, thr_name: str = "", strict: bool = False) -> str:
    """So sha1 checkpoint da nap voi params['ckpt_sha1'] cua tep nguong. Tra 'verified' / 'unverified (tep cu)'; khac -> ValueError.
    strict (e2e_test / check_decode): tep nguong khong ghi sha1, hoac checkpoint nap KHONG co trong tep nguong (vd heads_final_h3_f{k} lam sau khi dat nguong) -> ValueError."""
    ref = prm.get("ckpt_sha1")
    if not ref:
        if strict:
            raise ValueError(f"tep nguong {thr_name} khong ghi sha1 checkpoint (tep cu): che do strict khong chap nhan - chay fuse9 thr de tao tep truy vet duoc")
        return "unverified (tep nguong khong ghi sha1 checkpoint)"
    bad = [(m["file"], m["sha1"][:10], ref[m["file"]][:10]) for m in loaded if m["file"] in ref and ref[m["file"]] != m["sha1"]]
    if bad:
        raise ValueError(f"checkpoint khong khop tep nguong {thr_name} (ten, sha1 nap, sha1 trong tep): {bad} - mo hinh da doi sau khi dat nguong: chay lai fuse9 thr")
    unknown = [m["file"] for m in loaded if m["file"] not in ref]
    if unknown and strict:
        raise ValueError(f"checkpoint {unknown} chua co trong tep nguong {thr_name} (lam sau khi dat nguong): che do strict tu choi - chay lai fuse9 thr")
    return "verified" if not unknown else f"verified tru {len(unknown)} tep chua co khi dat nguong: {unknown}"


def check_calib(prm: dict, cal: dict, heads: str, strict: bool = False) -> tuple:
    """(dung bo hieu chinh?, trang thai): bo hieu chinh calib_<heads>.json phai hoc tren CUNG OOF dau ca tep ma tep nguong ghi (params.oof_sha1.heads).
    Khac -> khong dung (ticket ghi 'chua hieu chinh'), strict -> ValueError; tep nguong / tep hieu chinh khong ghi sha1 -> dung nhung 'unverified' (strict -> ValueError)."""
    want = (prm.get("oof_sha1") or {}).get("heads")
    src = cal.get("source") or {}
    got = src.get("sha1")
    if want and got and want != got:
        msg = (f"calib_{heads}.json hoc tren {src.get('oof')} sha1 {got[:12]}, khac OOF dau ca tep cua tep nguong (sha1 {want[:12]}): "
               "dau ca tep da hoc lai ma chua hieu chinh lai (chay calib fit)")
        if strict:
            raise ValueError(msg)
        return False, "TU CHOI bo hieu chinh: " + msg + " -> xac suat ca tep CHUA hieu chinh"
    if not (want and got):
        msg = "unverified (" + ("tep nguong khong ghi sha1 OOF dau ca tep" if not want else "tep hieu chinh khong ghi sha1 nguon") + ")"
        if strict:
            raise ValueError(f"bo hieu chinh calib_{heads}.json: {msg} - che do strict tu choi")
        return True, msg
    return True, "verified"


def load_lencal(thr_path: str, enabled: bool, strict: bool) -> tuple:
    """Nguong theo do dai (v6.product.lencal, 18/10): lencal_<ten>.json canh tep nguong. Tra (bo hieu chinh | None, {file, sha1}, ly do).
    Tep hieu chinh phai sinh tu CUNG tep nguong (truong thr_file); khac -> khong dung (strict: ValueError)."""
    if not enabled:
        return None, None, "tat (len_adj=False)"
    p = os.path.join(os.path.dirname(thr_path), os.path.basename(thr_path).replace("thresholds_", "lencal_"))
    if not os.path.exists(p):
        return None, None, f"chua co {os.path.basename(p)}: nguong KHONG dieu chinh theo do dai"
    cal = json.load(open(p, encoding="utf-8"))
    if cal.get("thr_file") != os.path.basename(thr_path):
        msg = f"{os.path.basename(p)} sinh tu {cal.get('thr_file')}, khong phai {os.path.basename(thr_path)}"
        if strict:
            raise ValueError(msg + " - che do strict tu choi")
        return None, None, msg + ": khong dung"
    return cal, {"file": os.path.basename(p), "sha1": sha1_file(p), "d_ref": cal.get("d_ref"), "tail": cal.get("tail")}, "dung"


class Product:
    def __init__(self, bound: str | None = None, reg: str | None = None, heads: str | None = None, thr: str = "thresholds_v9A.json", int8: bool = True, threads: int = 4,
                 fold: int | None = None, switchable: bool | None = None, use_calib: bool = True, strict: bool = False, len_adj: bool = True):
        torch.set_num_threads(threads)
        self.strict = strict                                           # True: moi mat xich (sha1 checkpoint, sha1 nguon hieu chinh) phai kiem duoc, khong thi ValueError
        self.thr_name = thr
        thr_path = thr if os.path.isabs(thr) else os.path.join(RES, thr)
        j = json.load(open(thr_path, encoding="utf-8"))
        self.thr_file = {"file": os.path.basename(thr_path), "sha1": sha1_file(thr_path), "name": j.get("name")}
        self.lencal, self.lencal_file, self.lencal_check = load_lencal(thr_path, len_adj, strict)
        self.params = j.get("params", {})
        self.bound, self.reg, self.heads_tag, self.legacy = resolve_params(self.params, bound, reg, heads, thr)
        self.thr_all, self.thr_fold = j["thr"], {int(k): v for k, v in j.get("thr_fold", {}).items()}
        self.thr = self.thr_all
        self.mode = canon(self.bound)
        need_ft = any(canon(t) in FUSE for t in (self.bound, self.reg))
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        self.fx = AutoFeatureExtractor.from_pretrained(NAME)
        import gc
        full = AutoModelForAudioClassification.from_pretrained(NAME, low_cpu_mem_usage=True)
        m = full.wav2vec2.eval()
        del full                                                       # bo dau phan loai + tang tren 17: dinh bo nho nap thap hon (may chi con ~3 GB trong)
        m.encoder.layers = m.encoder.layers[:max(LAYERS)]
        self._l6, self._ln = None, None
        if need_ft:
            import copy
            keep = torch.float16 if (int8 and (fold is not None if switchable is None else switchable)) else torch.float32
            # sao chep TRUOC khi luong tu hoa. Doi fold + int8: giu fp16 (~150 MB thay vi ~300 MB; sai so lam tron fp16 ~5e-4 << sai so int8) - 11/10: e2e dinh tuyen fold
            # + Whisper cua K1x vuot bo nho cam ket (access violation khi nap Whisper) khi giu ban fp32. merge_lora doi lai fp32 truoc khi gop LoRA.
            self._l6 = [copy.deepcopy(m.encoder.layers[i]).to(keep) for i in range(6, 12)]
            self._ln = copy.deepcopy(m.encoder.layer_norm).float()
        gc.collect()
        self.xlsr = torch.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8, inplace=True) if int8 else m
        gc.collect()
        self.int8 = int8
        self.W = np.load(os.path.join(FR, "_pca64.npz"))["W"]
        self.switchable = (fold is not None) if switchable is None else switchable
        self.fold = None
        self.ft = None
        self.ft_cache_size = 1
        self.load_models(fold)
        if need_ft and not self.switchable:
            self._l6 = None                                            # khong doi fold -> bo ban fp32 (~300 MB)
        self.calib, self.calib_file = None, None
        self.calib_check = "tat (use_calib=False)" if not use_calib else "khong co dau ca tep" if not self.heads_tag else None
        if use_calib and self.heads_tag:
            from v6.product import calib
            try:
                cal = calib.load(self.heads_tag)
            except FileNotFoundError:
                cal = None
                self.calib_check = f"chua co tep hieu chinh calib_{self.heads_tag}.json"
            if cal is not None:
                self.calib_file = {"file": os.path.basename(calib.path_of(self.heads_tag)), "sha1": sha1_file(calib.path_of(self.heads_tag)),
                                   "source_sha1": (cal.get("source") or {}).get("sha1")}
                ok, self.calib_check = check_calib(self.params, cal, self.heads_tag, strict)
                self.calib = cal if ok else None

    # ---------------------------------------------------------------- nap mo hinh nho (TCN, FT, dau ca tep) theo fold
    def load_models(self, fold: int | None) -> None:
        """Nap TCN / FT / dau ca tep cho fold (None = mo hinh cuoi) + chon nguong; kiem sha1 voi tep nguong.
        NGUYEN TU: kiem ton tai + sha1 moi tep TRUOC, nap vao bien cuc bo, chi gan vao self khi moi buoc thanh cong -> loi (thieu tep, sha1 khac) de Product
        nguyen o fold cu (khong con tinh trang mo hinh fold k + nguong / fold cua fold truoc)."""
        import joblib
        if fold is not None and not (0 <= int(fold) < 5):
            raise ValueError(f"fold {fold} ngoai 0..4")
        tcn_tags = sorted({FUSE.get(canon(t), (None, canon(t)))[1] for t in (self.bound, self.reg) if t})
        ft_tags = sorted({FUSE[canon(t)][0] for t in (self.bound, self.reg) if t and canon(t) in FUSE})
        if ft_tags and self._l6 is None:
            raise RuntimeError("doi fold cho FT can Product(..., switchable=True) (giu ban fp32 cac tang 6..11)")
        hp = heads_path(self.heads_tag, fold) if self.heads_tag else None
        plan = [(f"tcn:{tag}", p) for tag in tcn_tags for p in tcn_paths(tag, fold)] + [(f"ft:{tag}", ft_path(tag, fold)) for tag in ft_tags]
        plan += [("heads", hp)] if hp else []
        for role, p in plan:                                            # 1) ton tai (truoc khi doi gi)
            if not os.path.exists(p):
                what = {"tcn": "mo hinh TCN", "ft": "mo hinh FT", "heads": "dau ca tep"}[role.split(":")[0]]
                hint = ": nhanh khu on / doi giong ca tep se tat am tham" + (" (chay heads2 final --per-fold)" if fold is not None else "") if role == "heads" else ""
                raise FileNotFoundError(f"thieu {what} {p}{hint}")
        loaded = [{"role": role, "file": os.path.basename(p), "sha1": sha1_file(p)} for role, p in plan]
        sha_status = check_sha(self.params, loaded, self.thr_name, getattr(self, "strict", False))      # 2) sha1 (ValueError -> khong doi trang thai)
        tcn = {tag: self._load_tcn_paths(tcn_paths(tag, fold)) for tag in tcn_tags}                    # 3) nap vao bien cuc bo
        ft_new = None
        for tag in ft_tags:
            from v6.product.fthead import FTHead
            ckpt = ft_path(tag, fold)
            cache = self.__dict__.setdefault("_ft_cache", {})               # giu toi da ft_cache_size dau FT (mac dinh 1; check_decode dat 2: fold dang xet + mo hinh cuoi)
            key = (ckpt, sha1_file(ckpt))
            if key not in cache:
                while len(cache) >= max(1, self.ft_cache_size):
                    cache.pop(next(iter(cache)))                            # self.ft (neu la dau bi bo) van giu tham chieu toi khi gan dau moi
                import gc
                gc.collect()                                                # tra bo nho dau cu TRUOC khi dung dau moi (bo nho cam ket may con ~5 GB)
                cache[key] = FTHead(ckpt, self._l6, self._ln, int8=self.int8)
            ft_new = cache[key]
        heads = joblib.load(hp)["models"] if hp else {}
        if fold is None:
            thr, scope = self.thr_all, "toan bo fold (mo hinh cuoi)"
        elif fold in self.thr_fold:
            thr, scope = self.thr_fold[fold], f"thr_fold[{fold}]: tep that cua cac fold khac"
        else:
            thr, scope = self.thr_all, "toan bo fold (tep nguong khong co thr_fold: nguong co thay fold nay)"
        self.tcn, self.ft, self.heads, self.sha_status, self.thr, self.thr_scope = tcn, ft_new, heads, sha_status, thr, scope      # 4) gan mot lan
        self.loaded, self.fold = loaded, fold

    def set_fold(self, fold: int | None) -> None:
        if fold != self.fold:
            self.load_models(fold)

    def model_record(self) -> dict:
        return {"thresholds": self.thr_file, "params": {k: self.params.get(k) for k in ("bound", "reg", "heads", "f03", "freg", "fh", "ftts", "fspl", "snap", "typed")},
                "resolved": {"bound": self.bound, "reg": self.reg, "heads": self.heads_tag}, "legacy_params": self.legacy, "fold": self.fold, "thr_scope": self.thr_scope,
                "ckpt": list(self.loaded), "sha_check": self.sha_status, "calib": self.calib_file, "calib_check": getattr(self, "calib_check", None),
                "calib_scope": self.calib_scope(), "strict": getattr(self, "strict", False), "int8": self.int8, "chunk_s": CHUNK_S,
                "lencal": getattr(self, "lencal_file", None), "lencal_check": getattr(self, "lencal_check", None)}

    def len_thr(self, T: dict, lang: str, dur: float) -> tuple:
        """(nguong dung cho tep dai dur giay, ghi chu | None). Khong co bo hieu chinh / tep dung d_ref / tep ngan hon d_ref khi khong hai chieu
        -> CHINH doi tuong nguong cu (khong sao chep). Hai chieu (lencal 18/10): tep ngan hon d_ref HA nguong (vu oan / tep deu theo do dai)."""
        cal = getattr(self, "lencal", None)
        two = bool(cal and cal.get("two_sided"))
        if cal is None or dur == cal["d_ref"] or (dur < cal["d_ref"] and not two):
            return T, None
        from v6.product import lencal as LC
        Ta = LC.adjust(T, dur, LC.sigma_for(cal, lang, self.fold), cal["d_ref"], two)
        return Ta, {"dur_s": round(dur, 2), "d_ref_s": cal["d_ref"], "fold": self.fold,
                    "doi": {k: [round(T[k], 4), round(Ta[k], 4)] for k in Ta if Ta[k] != T.get(k)}}

    def calib_scope(self, generic: bool = False) -> str | None:
        """Bo hieu chinh hoc tren OOF CUA CA 5 fold (diem cua mo hinh fold) -> che do fold=k: xac suat cua tep fold k hoi 'trong mau' o buoc hieu chinh (khong o mo hinh).
        generic=True: cau khong gan voi mot fold (bao cao e2e chay qua ca 5 fold)."""
        if getattr(self, "calib", None) is None:
            return None
        if generic and getattr(self, "switchable", False):
            return "OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do)"
        return "OOF 5 fold (mo hinh cuoi: ngoai mau)" if self.fold is None else f"OOF 5 fold ke ca fold {self.fold} (buoc hieu chinh trong mau cho fold nay)"

    @property
    def tcn_b(self) -> list:
        """Giu cho ma cu (check_oof.py): danh sach TCN cua nhan ranh gioi (fuse1 -> TCN vung)."""
        return self.tcn[FUSE.get(canon(self.bound), (None, canon(self.bound)))[1]]

    @property
    def tcn_r(self) -> list:
        return self.tcn[FUSE.get(canon(self.reg), (None, canon(self.reg)))[1]]

    @staticmethod
    def _load_tcn_paths(paths: list) -> list:
        from v6.boundary import tcn as T
        models = []
        for p in paths:
            sd = torch.load(p, map_location="cpu", weights_only=True)
            net = T.TCN(n_cls=sd["out.weight"].shape[0])
            net.load_state_dict(sd)
            models.append(net.eval())
        return models

    @staticmethod
    def _load_tcn(tag: str, n_cls: int = 0) -> tuple:
        """Giu cho ma cu (check_decode cu): (mo hinh, so lop) cua nhan theo duong dan mac dinh."""
        models = Product._load_tcn_paths(tcn_paths(tag, None))
        return models, models[0].out.out_channels

    # ---------------------------------------------------------------- XLS-R theo khuc
    def encode(self, y: np.ndarray) -> tuple:
        """(khung L12 (T, 1024) float32, vector gop tho (6144,)) - khuc CHUNK_S chong 2 s, ghep bo nua phan chong moi dau."""
        n, L, ov = len(y), int(CHUNK_S * SR), int(OVL_S * SR)
        starts = list(range(0, max(1, n - ov), L - ov)) or [0]
        frames, sums, sq, cnt = [], {k: 0.0 for k in LAYERS}, {k: 0.0 for k in LAYERS}, 0
        self._ftp = [] if getattr(self, "ft", None) is not None else None
        for ci, s0 in enumerate(starts):
            seg = y[s0:s0 + L]
            if len(seg) < SR // 2:
                break
            x = self.fx(seg.astype(np.float32), sampling_rate=SR, return_tensors="pt").input_values
            with torch.no_grad():
                hs = self.xlsr(x, output_hidden_states=True).hidden_states
            T = hs[12].shape[1]
            a = 0 if ci == 0 else int(round(ov / 2 / SR / DT))
            b = T if ci == len(starts) - 1 or s0 + L >= n else (L - ov) // (SR // 50) + int(round(ov / 2 / SR / DT))      # = buoc khuc (khung) + a: khong mat khung o cho noi
            frames.append(hs[12][0, a:b].numpy())
            if self._ftp is not None:
                self._ftp.append(self.ft.chunk_probs(hs[self.ft.first])[a:b])
            for k in LAYERS:
                h = hs[k][0, a:b].numpy().astype(np.float64)
                sums[k] = sums[k] + h.sum(0)
                sq[k] = sq[k] + (h * h).sum(0)
            cnt += b - a
        pooled = []
        for k in LAYERS:
            mu = sums[k] / cnt
            pooled += [mu, np.sqrt(np.maximum(sq[k] / cnt - mu * mu, 0.0))]
        return np.concatenate(frames).astype(np.float32), np.concatenate(pooled).astype(np.float32)

    def tcn_probs(self, models: list, z: np.ndarray) -> np.ndarray:
        from v6.boundary import detect as D
        from v6.boundary import tcn as T
        d = np.abs(np.diff(z.astype(np.float32), axis=0, prepend=z[:1].astype(np.float32)))
        x = torch.from_numpy(T.file_inputs(z, np.median(d, axis=0))).T[None]
        with torch.no_grad():
            p = torch.stack([torch.softmax(m(x)[0], 0) for m in models]).mean(0).numpy()
        return np.stack([D.smooth(p[c]) for c in range(1, p.shape[0])], 1)

    def frame_probs(self, tag: str, z: np.ndarray, cache: dict) -> np.ndarray:
        """Diem khung (T, c) cua mot nhan: TCN -> c = 3 / 4 / 6 cot; fuse1 -> 6 cot = trung binh logit (FT-2, TCN vung) tren cot chung, cot chi FT co -> FT (nhu ens6.fuse)."""
        tag = canon(tag)
        if tag in cache:
            return cache[tag]
        if tag in FUSE:
            from v6.boundary.ens6 import combine
            from v6.product.fthead import FTHead
            pf = FTHead.smooth_cols(np.concatenate(self._ftp))                   # (T, 6) ghep, xoa, khu on, doi giong, doan ghep, giong may
            preg = self.frame_probs(FUSE[tag][1], z, cache)                      # (T, 4) ghep, xoa, khu on, doi giong
            n = min(len(pf), len(preg))
            cols = [combine([pf[:n, c], preg[:n, c]], [0.5, 0.5], "logit") if c < preg.shape[1] else np.clip(pf[:n, c], EPS, 1 - EPS) for c in range(pf.shape[1])]
            p = np.stack(cols, 1).astype(np.float32)
        else:
            p = self.tcn_probs(self.tcn[tag], z)
        cache[tag] = p
        return p

    def head_probs(self, head: dict, lang: str) -> dict | None:
        """{khoa: xac suat da hieu chinh} (I1) hoac None neu khong co tep hieu chinh; khoa khong co bo hieu chinh, hoac bo hieu chinh KHONG TIN CAY
        (calib.issue_of: bang do tin cay lech > 0,2 lam doi vung quyet dinh, vd ttsf) -> None (ticket giu luat nguong diem va ghi 'chua hieu chinh')."""
        if self.calib is None:
            return None
        from v6.product import calib
        out = {}
        for k, v in head.items():
            try:
                out[k] = None if calib.issue_of(self.calib, k, lang) else calib.apply(self.calib, k, v, lang=lang)
            except KeyError:
                out[k] = None
        return out

    def head_prob_methods(self, head: dict, lang: str) -> dict | None:
        """{khoa: 'isotonic' / 'platt' [+ ' (gop all)' khi ngon ngu nay thieu du lieu, vd ttsf tieng Anh khong co tep duong] / None} - ghi vao ho so de biet xac suat tu dau."""
        if self.calib is None:
            return None
        from v6.product import calib
        out = {}
        for k in head:
            try:
                bad = calib.issue_of(self.calib, k, lang)
                out[k] = calib.method_of(self.calib, k, lang) + (f" KHONG TIN CAY ({bad}) -> khong dung" if bad else "")
            except KeyError:
                out[k] = None
        return out

    # ---------------------------------------------------------------- mot kenh mono
    def analyse_channel(self, y: np.ndarray, lang: str, tm: dict) -> dict:
        """Tra {rows, pb, pr, head, head_prob, T, dropped, dur}: rows chua qua luat #bind (ticket.py ap luat), pb / pr = diem khung 20 ms."""
        t0 = time.time()
        h, pooled = self.encode(y)
        tm["xlsr"] = tm.get("xlsr", 0) + time.time() - t0
        t0 = time.time()
        z = ((h - np.median(h, axis=0, keepdims=True)) @ self.W).astype(np.float16)
        cache = {}
        pb, pr = self.frame_probs(self.bound, z, cache), self.frame_probs(self.reg, z, cache)
        tm["tcn"] = tm.get("tcn", 0) + time.time() - t0
        t0 = time.time()
        T0 = self.thr[lang] if lang in self.thr else next(iter(self.thr.values()))
        T, len_note = self.len_thr(T0, lang, len(y) / SR)                 # 18/10: nguong kenh doan tang theo do dai tep (lencal)
        det = self.decode(pb, pr, pooled[2048:4096], T)
        det["T_base"], det["len_adj"] = T0, len_note
        det["head_prob"] = self.head_probs(det["head"], lang)
        det["head_prob_method"] = self.head_prob_methods(det["head"], lang)
        tm["decode"] = tm.get("decode", 0) + time.time() - t0
        from vaudit import cond
        cm, cw, cr = cond.assess(pooled)
        return {**det, "pb": pb, "pr": pr, "T": T, "pooled": pooled, "cond": {"measures": cm, "warnings": cw, "reasons": cr}}

    def decode(self, pb: np.ndarray, pr: np.ndarray, v12: np.ndarray, T: dict) -> dict:
        """Giai ma nhu nghien cuu (fuse9.Ctx9.records): dong ranh gioi + dong vung (typed / snap / merge theo params) + dau ca tep (nguong T) -> cap_rows."""
        from v6.boundary import detect as D
        from v6.boundary import fusion as F
        from v6.boundary.heads2 import predict_final
        from v6.boundary.rowmerge import merge, snap_merge, typed_merge
        dur = len(pb) * DT
        rows = D.decode_file(pb, (T["ghep"], T["xoa"]), dur)
        for r in rows:                                                 # nm = diem tuyet doi (bien + nguong) nhu nghien cuu (fuse9.rows03): cap_rows xep theo nm giua cac kenh
            r.update(kind="segment", nm=r["margin"] + (T["ghep"] if r["via"] == "ranh_gioi_ghep" else T["xoa"]))
        names = [("den", 0, "02_nen_cuc_bo"), ("voi", 1, "02_giong_cuc_bo"), ("spl", 2, "03_ghep_giong_that"), ("ttsr", 3, "03_chen_giong_may")]
        use = [(k, c, lv) for k, c, lv in names if k in T and 2 + c < pr.shape[1]]
        regs = F.decode_regions([pr[:, 2 + c] for _, c, _ in use], [T[k] for k, _, _ in use], tuple(lv for _, _, lv in use)) if use else []
        rows = typed_merge(rows, regs) if self.params.get("typed") else snap_merge(rows, regs) if self.params.get("snap") else merge(rows, regs)
        head = {}
        for key, level, idx in (("nen", "02_nen_toan_cuc", "nen_toan"), ("giong", "02_giong_toan_cuc", "giong_toan"), ("ttsf", "03_chen_giong_may", "tts_file")):
            if idx in self.heads:
                head[key] = float(predict_final(self.heads[idx], v12)[0])
                if key in T and head[key] >= T[key]:
                    rows = merge(rows, [{"level2": level, "kind": "global", "p": head[key], "nm": head[key] - T[key]}])
        rows, dropped = F.cap_rows(rows)
        return {"rows": rows, "head": head, "dropped": bool(dropped), "dur": dur}

    # ---------------------------------------------------------------- tep
    def analyse(self, path: str, lang: str | None = None, extended_k1: bool = False) -> dict:
        from vaudit import k1
        tm = {}
        t0 = time.time()
        ck = k1.check(path, extended=extended_k1)
        tm["k1"] = time.time() - t0
        out = {"k1": ck, "rows": [], "channels": [], "timing_s": tm, "models": self.model_record()}
        if ck["decision"] in ("04", "tu_choi"):
            out["label"] = "04"
            return out
        out["dur_s"] = max(len(c) for c in ck["channels"]) / SR
        lid = ck["measures"].get("kenh", [{}])[0].get("ngon_ngu", {}).get("top") if extended_k1 else None
        lg_det = lang or lid or "vi"
        out["lang_source"] = "goi_y" if lang else "lid" if lid else "mac_dinh_vi"
        lg_det = lg_det if lg_det in self.thr else "vi"                      # ngon ngu nhan dang chon bo nguong (truoc day luon la vi)
        out["lang"], out["lid"] = lg_det, lid
        for ch, y in enumerate(ck["channels"]):
            det = self.analyse_channel(np.asarray(y, np.float32), lg_det, tm)
            det["kenh"], det["y"] = ch, np.asarray(y, np.float32)
            out["channels"].append(det)
            out["rows"] += [{**r, "kenh": ch} for r in det["rows"]]
        cr = [c["cond"]["reasons"] for c in out["channels"]]
        if cr and all(cr):                              # mọi kênh trượt hàng Constraint SNR / nền (mô hình học) -> 04 như K1
            ck["decision"], ck["reasons"] = "04", list(dict.fromkeys(ck["reasons"] + [x for r in cr for x in r]))
            out["rows"], out["label"] = [], "04"
        else:
            ck["warnings"] = list(dict.fromkeys(ck["warnings"] + [w for c in out["channels"] for w in c["cond"]["warnings"]]))
            out["label"] = "03" if any(r["level2"].startswith("03") for r in out["rows"]) else "02" if out["rows"] else "01"
        tm["total"] = sum(v for k, v in tm.items() if k != "total")
        out["rtf"] = tm["total"] / out["dur_s"]
        return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--op", default="A2", help="A2 = nguong cua A (03/10, trung tung gia tri) lam lai 12/10 co params + sha1 checkpoint + thr_fold (truy vet duoc)")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bound", default=None, help="bo trong = theo tep nguong; khac tep nguong -> tu choi")
    ap.add_argument("--reg", default=None)
    ap.add_argument("--heads", default=None)
    ap.add_argument("--thr", default=None)
    ap.add_argument("--fold", type=int, default=None, help="danh gia ngoai mau: mo hinh khong thay fold nguoi noi nay")
    ap.add_argument("--strict", action="store_true", help="tu choi checkpoint / bo hieu chinh khong kiem duoc sha1 voi tep nguong")
    a = ap.parse_args()
    t0 = time.time()
    P = Product(a.bound, a.reg, a.heads, a.thr or f"thresholds_v9{a.op}.json", not a.fp32, a.threads, fold=a.fold, strict=a.strict)
    print(f"nap mo hinh: {time.time() - t0:.1f} s", flush=True)
    r = P.analyse(a.wav, a.lang)
    print(json.dumps({"label": r["label"], "k1": r["k1"]["decision"], "timing_s": r["timing_s"], "rtf": r.get("rtf"), "dur_s": r.get("dur_s"), "models": r["models"]},
                     ensure_ascii=False, indent=1, default=float))
    for x in r["rows"]:
        print(f"  {x['level2']:<22} {x.get('start', ''):>8} {x.get('end', ''):>8}  p={x.get('p', 0):.2f}")


if __name__ == "__main__":
    main()
