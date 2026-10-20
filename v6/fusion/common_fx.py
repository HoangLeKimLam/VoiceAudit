# -*- coding: utf-8 -*-
"""Ham DUNG CHUNG cho thi nghiem ket hop (WP8a; WP8b chi import, khong sua):
  items_fx      nap NHE cac manifest v6 (chi giu khoa can thiet) -> id -> {set, cfg, wav, lang, source, person, fold (fold NGUOI NOI), group, real,
                real_raw, role, engine, y, fam, edits}; tap WP7b ttsmulti_vi / ttsmulti_en / vcmulti tu nap khi co manifest
  in_train / is_heldout / is_pos_scope   tap hoc (TRAIN_TAGS, bo may 'heldout') · pham vi tep duong (tu nhien + WP7b goc)
  auc_tie       AUC co xu ly hoa (scipy rankdata, hang trung binh) - req_check.auc khong xu ly hoa
  tpr_fold      TPR tai ti le bao nham q, nguong = phan vi 1 - q cua tep THAT o cac fold KHAC (nhu cmp_oof)
  fpr_by        ti le bao nham theo nhom (nguon) voi nguong cung kieu fold-khac
  boot_delta    khoang tin cay bootstrap CAP theo cum (nhom group_of: chuoi AMI, hoi thoai EdAcc, kenh VSS, ... ; khong co I2 -> nguoi noi) cho AUC(B) - AUC(A)
  apply_app_ns  QUYET DINH 4: NS mac dinh ung dung (N7 WebRTC NS, benign ns2 / ns3) -> '01' (mac dinh) / 'drop' / '02' (cau chu hop dong)
  prep_block / fit_score / fit_arm   khop mo hinh khoi: dien trung vi + chuan hoa (+ PCA trong fold) -> LR + HGB, diem = trung binh hai thu hang (nhu heads2)
  save_heads    ghi pickle tuong thich heads2 ({ids, P (n, 3)}) -> fuse9 --heads <tag> doc duoc khong can sua
Fold: detect.person_fold (crc32(person) % 5) - giong moi OOF v9; KHONG dung meta.fold (crc32(speaker))."""
from __future__ import annotations

import json
import math
import os
import pickle
import sys
from typing import Sequence

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

V4 = os.path.join(BASE, "data_labels", "realistic_v4")
V6 = os.path.join(BASE, "data_labels", "v6")
CACHE = os.path.join(BASE, ".tmp_v6", "cache")
RES = os.path.join(BASE, ".tmp_v6", "results")
POOL = os.path.join(V4, "xlsr_pooled")
SET_TAG = {"natvi": "v6natvi", "naten": "v6naten", "cover": "v6cover", "cover2": "v6cover2", "benign": "v6benign", "chain": "v6chain",
           "amilong": "v6amilong", "ttsvi": "v6ttsvi", "ttsen": "v6ttsen",
           "ttsmulti_vi": "v6ttsmulti_vi", "ttsmulti_en": "v6ttsmulti_en", "vcmulti": "v6vcmulti"}   # 3 bo cuoi: WP7b (nhieu may TTS / VC)
ALL_SETS = tuple(SET_TAG)
SCORE_SETS = ALL_SETS                                # tap cham diem cua fx_file / fx_tts / report_fx (manifest chua co -> bo qua)
SET_TAG_AUG = {"aug": "v6aug"}                       # aug: chi co khung / vector gop XLS-R (khong wav) -> chi khoi E dung duoc
NAT_TAGS = ("v6natvi", "v6naten")
MULTI_TAGS = ("v6ttsmulti_vi", "v6ttsmulti_en", "v6vcmulti")   # WP7b: tep chu tu nhien (vox / celeb / edacc / ami / voxp) + giong may / doi giong AI
POS_TAGS = NAT_TAGS + MULTI_TAGS                     # tep DUONG duoc danh gia so voi tep that tu nhien goc (fold khac)
TRAIN_TAGS = ("v6natvi", "v6naten", "v6cover", "v6cover2", "v6ttsvi", "v6ttsen") + MULTI_TAGS   # = fusion.TRAIN_SETS bo aug (+ WP7b, tru may 'heldout')
EN_SETS = ("naten", "ttsen", "ttsmulti_en")
CFGS = ("goc", "app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi")   # = detect.CFGS
TARGETS = ("nen_toan", "giong_toan", "tts_file")
EDIT_KEYS = ("level2", "kind", "start", "end", "joint", "sub", "op", "family", "tool", "tts", "vo_hai", "engine", "role")
QS = (0.01, 0.03, 0.05)
APP_NS_MODES = ("01", "drop", "02")
APP_NS_DEFAULT = os.environ.get("FX_APP_NS", "01")       # QUYET DINH 4: NS mac dinh ung dung = 01


def person_fold(meta: dict) -> int:
    """Fold NGUOI NOI cua v9 (detect.person_fold): crc32(person hoac speaker) % 5 qua common.assign_fold."""
    from v6.boundary import detect as D
    return D.person_fold(meta)


def group_fx(meta: dict, set_name: str = "") -> str:
    """Nhom cho bootstrap: v6.common.group_of neu WP5 da them (I2), neu khong thi person (hoac speaker)."""
    try:
        from v6.common import group_of
        g = group_of(meta, set_name)
        if g:
            return str(g)
    except Exception:  # noqa: BLE001 - I2 chua co hoac loi: quay ve nguoi noi
        pass
    return str(meta.get("person") or meta.get("speaker"))


def is_app_ns_edit(e: dict) -> bool:
    """Dau khu on ca tep la NS MAC DINH cua ung dung (ho N7 'WebRTC NS'; benign ghi family 'WebRTC NS')."""
    if str(e.get("level2", "")) != "02_nen_toan_cuc":
        return False
    return str(e.get("family")) == "N7" or "webrtc" in f"{e.get('tool', '')} {e.get('family', '')}".lower()


def apply_app_ns(meta: dict, mode: str = "") -> dict | None:
    """QUYET DINH 4 (08/10): khu on mac dinh cua ung dung (Zoom / Meet / Zalo / WebRTC NS) = nhan 01. Tra meta MOI (khong sua meta goc):
      '01'   bo dau N7 khoi edits / level2 (tep chi co N7 thanh tep that 01; tep N7 + sua khac giu cac sua khac) - mac dinh
      'drop' tep co / tung co NS ung dung -> None (bo khoi hoc VA danh gia)
      '02'   theo cau chu hop dong ('khu on ca tep = 02 vo hai'): giu N7 la duong, tra benign ns2 / ns3 (meta.app_ns_relabel) ve nhan cu 02
    Them khoa 'app_ns' = True cho moi tep co / tung co NS ung dung."""
    mode = mode or APP_NS_DEFAULT
    if mode not in APP_NS_MODES:
        raise ValueError(f"app_ns '{mode}' khong hop le ({'/'.join(APP_NS_MODES)})")
    edits = list(meta.get("edits") or [])
    rel = meta.get("app_ns_relabel")
    had = any(is_app_ns_edit(e) for e in edits) or bool(rel)
    if not had:
        return meta
    if mode == "drop":
        return None
    out = dict(meta, app_ns=True)
    if mode == "02":
        if rel and "02_nen_toan_cuc" in (rel.get("level2_cu") or []):
            out["level2"] = list(rel.get("level2_cu") or [])
            out["edits"] = [dict(e) for e in rel.get("edits_cu") or []]
        return out
    keep = [e for e in edits if not is_app_ns_edit(e)]
    still = any(str(e.get("level2", "")) == "02_nen_toan_cuc" for e in keep)
    out["edits"] = keep
    out["level2"] = [x for x in (meta.get("level2") or []) if x != "02_nen_toan_cuc" or still]
    return out


def labels3(meta: dict) -> list:
    """[khu on ca tep, doi giong ca tep, giong may ca tep] (= heads2.labels)."""
    l2 = set(meta.get("level2") or [])
    tts = any(e.get("op") == "may_ca_file" for e in meta.get("edits", []))
    return [int("02_nen_toan_cuc" in l2), int("02_giong_toan_cuc" in l2), int(tts)]


def families(meta: dict) -> list:
    """Ho cong cu cua tung dau ca tep (de bo-mot-ho-ra): khu on N1..N7, doi giong V1..V6, may: dong co TTS; None neu khong co."""
    out = [None, None, None]
    for e in meta.get("edits", []):
        l2 = str(e.get("level2", ""))
        if l2 == "02_nen_toan_cuc":
            out[0] = str(e.get("family") or e.get("tool") or "?")
        elif l2 == "02_giong_toan_cuc":
            out[1] = str(e.get("family") or e.get("tool") or "?")
        elif e.get("op") == "may_ca_file":
            out[2] = str(e.get("tts") or e.get("engine") or e.get("tool") or e.get("model") or "?")     # WP7b ghi edit.engine
    return out


def wav_path(rel: str) -> str:
    """Duong dan wav cua manifest: tuong doi realistic_v4 (quy uoc common.rel_v4; moi manifest hien co dang '../v6/...'), tuyet doi giu nguyen."""
    return os.path.normpath(os.path.join(V4, rel))


def is_heldout(it: dict) -> bool:
    """Tep cua may TTS / VC vai 'heldout' (engines.json, meta.engine_role): KHONG bao gio vao tap hoc - chi de do tong quat hoa."""
    return it.get("role") == "heldout"


def in_train(it: dict, train_cfg: str = "all") -> bool:
    """Tep duoc vao tap hoc dau ca tep: tap trong TRAIN_TAGS (+ aug), khong phai may heldout, (train_cfg 'goc': chi ban goc)."""
    return (it["set"] in TRAIN_TAGS or it["set"] == "v6aug") and not is_heldout(it) and (train_cfg == "all" or it["cfg"] == "goc")


def is_pos_scope(it: dict) -> bool:
    """Tep goc thuoc tap tu nhien hoac WP7b: tep duong cua no duoc so voi tep that tu nhien goc."""
    return it["set"] in POS_TAGS and it["cfg"] == "goc"


def items_fx(sets: Sequence[str] = ALL_SETS, cfgs: Sequence[str] = CFGS, need_pool: bool = False, with_edits: bool = True,
             app_ns: str = "") -> dict:
    """id -> dict nhe cho moi tep trong pham vi (cung loc voi detect.load_items: cfg, ngoai_pham_vi, lockbox tru amilong).
    Khong can khung XLS-R (khac load_items) - can wav (tru aug). need_pool: chi giu tep co vector gop tho.
    app_ns: cach gan nhan NS mac dinh ung dung (apply_app_ns; '' = APP_NS_DEFAULT = '01' theo quyet dinh 4).
    real = tep that SAU app_ns; real_raw = level2 manifest THO rong (= detect.load_items / fuse9.Ctx9: nguong ung vien rows_fx phai dung tap nay).
    role / engine: meta.engine_role ('train' | 'heldout') va meta.engine cua bo WP7b (None voi tap khac)."""
    out = {}
    for s in sets:
        mp = os.path.join(V6, s, "manifest.json")
        if not os.path.exists(mp):
            continue
        tag = SET_TAG.get(s, "v6" + s)
        for e in read_json(mp):
            m = e["meta"]
            if e["cfg"] not in cfgs or m.get("ngoai_pham_vi") or (m.get("lockbox") and s != "amilong"):
                continue
            real_raw = not m.get("level2")
            m = apply_app_ns(m, app_ns)
            if m is None:
                continue
            if need_pool and not os.path.exists(os.path.join(POOL, e["id"] + ".npy")):
                continue
            lang = str(m.get("lang") or ("en" if s in EN_SETS else "vi"))[:2]
            out[e["id"]] = {"set": tag, "kind": e["kind"], "cfg": e["cfg"], "wav": wav_path(e["wav"]), "lang": lang,
                            "source": str(m.get("source")), "person": str(m.get("person") or m.get("speaker")),
                            "fold": person_fold(m), "group": group_fx(m, s), "real": not m.get("level2"), "real_raw": real_raw,
                            "role": m.get("engine_role"), "engine": m.get("engine"),
                            "y": labels3(m), "fam": families(m), "level2": list(m.get("level2") or []),
                            "cond": m.get("cond"), "tier": m.get("tier"), "app_ns": bool(m.get("app_ns")),
                            "edits": [{k: x[k] for k in EDIT_KEYS if k in x} for x in m.get("edits", [])] if with_edits else []}
    return out


def is_vss(it: dict) -> bool:
    return it.get("source") == "vss"


# ------------------------------------------------------------------ do luong
def auc_tie(pos: Sequence[float], neg: Sequence[float], min_n: int = 5) -> float:
    """AUC Mann-Whitney voi hang TRUNG BINH cho diem trung (diem bao hoa 1.0 cua float16 khong lam lech). < min_n moi ben -> NaN."""
    from scipy.stats import rankdata
    p, n = np.asarray(pos, float), np.asarray(neg, float)
    p, n = p[np.isfinite(p)], n[np.isfinite(n)]
    if len(p) < min_n or len(n) < min_n:
        return float("nan")
    r = rankdata(np.r_[p, n])
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def tpr_fold(s: np.ndarray, pos: np.ndarray, neg: np.ndarray, fold: np.ndarray, qs: Sequence[float] = QS) -> list:
    """TPR tai moi q: nguong cua fold f = phan vi 1 - q cua diem tep THAT (neg) o cac fold != f; dem tep duong o fold f vuot nguong (>)."""
    out = []
    for q in qs:
        hit = n = 0
        for f in np.unique(fold[pos]) if len(pos) else []:
            ref = s[neg[fold[neg] != f]]
            pf = pos[fold[pos] == f]
            if len(ref) < 5 or not len(pf):
                continue
            th = np.quantile(ref, 1 - q)
            hit += int(np.sum(s[pf] > th))
            n += len(pf)
        out.append(hit / n if n else float("nan"))
    return out


def fpr_by(s: np.ndarray, ev: np.ndarray, ref: np.ndarray, fold: np.ndarray, key: np.ndarray, q: float) -> dict:
    """Ti le bao nham theo nhom key cua cac tep ev (tep that hoac tep vo hai), nguong = phan vi 1 - q cua ref o fold KHAC.
    Tra {nhom: (so bi bao, n)}."""
    th = {}
    out: dict = {}
    for k in ev:
        f = fold[k]
        if f not in th:
            r = s[ref[fold[ref] != f]]
            th[f] = np.quantile(r, 1 - q) if len(r) >= 5 else np.inf
        a, n = out.get(key[k], (0, 0))
        out[key[k]] = (a + int(s[k] > th[f]), n + 1)
    return out


def boot_delta(sa: np.ndarray, sb: np.ndarray, pos: np.ndarray, neg: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 0) -> tuple:
    """AUC(sb) - AUC(sa) tren cung tep duong / that; khoang 95 % bootstrap boc lai CUM (nguoi noi / nhom) co hoan lai (v6.stats)."""
    from v6.stats import bootstrap_ci_by_group
    idx = np.r_[pos, neg]
    v = np.c_[np.r_[np.ones(len(pos)), np.zeros(len(neg))], sa[idx], sb[idx]]

    def fn(x: np.ndarray) -> float:
        p = x[:, 0] == 1
        return auc_tie(x[p, 2], x[~p, 2]) - auc_tie(x[p, 1], x[~p, 1])
    return bootstrap_ci_by_group(v, np.asarray(groups)[idx], fn, n=n, seed=seed)


def fmt_ci(t: tuple) -> str:
    est, lo, hi = t
    if not np.isfinite(est):
        return "-"
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]" if np.isfinite(lo) else f"{est:+.3f}"


# ------------------------------------------------------------------ mo hinh khoi
def rank01(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Xep hang v theo phan bo ref -> [0, 1] (= heads2.rank01)."""
    return np.searchsorted(np.sort(ref), v) / max(1, len(ref))


def prep_block(Xtr: np.ndarray, Xte: np.ndarray, n_pca: int = 0, clip: float = 10.0) -> tuple:
    """Dien NaN bang trung vi tap hoc, chuan hoa, cat +-clip, PCA n_pca (fit TRONG fold hoc) neu so cot lon hon."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Xtr = np.asarray(Xtr, np.float32)
    Xte = np.asarray(Xte, np.float32)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # cot toan NaN (vd khoi praat thieu) -> dien 0 ngay duoi
        med = np.nanmedian(Xtr, 0) if len(Xtr) else np.zeros(Xtr.shape[1], np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    fill = lambda a: np.where(np.isfinite(a), a, med)  # noqa: E731
    A, B = fill(Xtr), fill(Xte)
    sc = StandardScaler().fit(A)
    A, B = np.clip(sc.transform(A), -clip, clip), np.clip(sc.transform(B), -clip, clip)
    if n_pca and A.shape[1] > n_pca:
        p = PCA(n_components=min(n_pca, A.shape[0] - 1), random_state=0).fit(A)
        A, B = p.transform(A), p.transform(B)
    return A.astype(np.float32), B.astype(np.float32)


def fit_score(A: np.ndarray, y: np.ndarray, B: np.ndarray, lr_on: bool = True, hgb_iter: int = 150, hgb_lr: float = 0.07) -> np.ndarray:
    """LR (C=0.05, can bang) + HGB (150 vong, lr 0.07, can bang); diem = trung binh thu hang theo tap hoc (cung thang voi heads2)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    gb = HistGradientBoostingClassifier(max_iter=hgb_iter, learning_rate=hgb_lr, class_weight="balanced", random_state=0).fit(A, y)
    r = rank01(gb.predict_proba(B)[:, 1], gb.predict_proba(A)[:, 1])
    if not lr_on:
        return r
    lr = LogisticRegression(C=0.05, max_iter=400, class_weight="balanced").fit(A, y)
    return (rank01(lr.predict_proba(B)[:, 1], lr.predict_proba(A)[:, 1]) + r) / 2


def fit_arm(blocks: dict, keys: Sequence[str], tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ket hop SOM: moi khoa trong keys la (X (n, d), n_pca); xu ly tung khoi trong fold hoc roi noi lai -> fit_score."""
    pa, pb = [], []
    for k in keys:
        X, n_pca = blocks[k]
        A, B = prep_block(X[tr], X[te], n_pca)
        pa.append(A)
        pb.append(B)
    return fit_score(np.hstack(pa), y[tr], np.hstack(pb))


def stack_features(S: np.ndarray, C: np.ndarray | None) -> np.ndarray:
    """Dau vao bo xep chong: diem tung chu S (n, k) [+ boi canh dieu kien C (n, c) va tich S x C de nguong / trong so doi theo dieu kien]."""
    if C is None or not C.shape[1]:
        return S
    inter = (S[:, :, None] * C[:, None, :]).reshape(len(S), -1)
    return np.hstack([S, C, inter]).astype(np.float32)


def fit_late(blocks: dict, letters: dict, arm_letters: str, tr: np.ndarray, te: np.ndarray, y: np.ndarray, fold: np.ndarray,
             ctx_letters: str = "") -> np.ndarray:
    """Ket hop MUON: diem OOF tung chu (E, H, ...) trong tap hoc (fold long nhau theo nguoi noi) -> LR xep chong; ap len te.
    ctx_letters (vd 'C'): cac chu KHONG lam mo hinh co so ma vao bo xep chong lam boi canh (bo xep chong theo dieu kien).
    Hang cua fold trong bi bo qua (qua it duong) la NaN va KHONG vao fit bo xep chong (truoc 11/10: giu 0 - lap day im lang)."""
    from sklearn.linear_model import LogisticRegression
    base = [L for L in arm_letters if L not in ctx_letters]
    inner = np.full((len(tr), len(base)), np.nan, np.float32)
    outer = np.zeros((len(te), len(base)), np.float32)
    ftr = fold[tr]
    skipped = 0
    for j, L in enumerate(base):
        keys = letters[L]
        for g in np.unique(ftr):
            itr, ite = tr[ftr != g], tr[ftr == g]
            if y[itr].sum() < 3 or y[itr].sum() == len(itr):
                skipped += 1
                continue
            inner[ftr == g, j] = fit_arm(blocks, keys, itr, ite, y)
        outer[:, j] = fit_arm(blocks, keys, tr, te, y)
    Ctr = Cte = None
    ck = [k for L in ctx_letters if L in arm_letters for k in letters[L]]
    if ck:
        pa, pb = zip(*[prep_block(blocks[k][0][tr], blocks[k][0][te], blocks[k][1]) for k in ck])
        Ctr, Cte = np.hstack(pa), np.hstack(pb)
    ok = np.isfinite(inner).all(1)
    if skipped:
        print(f"  canh bao fit_late: {skipped} fold trong bo qua (qua it duong), {int((~ok).sum())} hang khong vao bo xep chong", flush=True)
    Xin = stack_features(inner[ok], None if Ctr is None else Ctr[ok])
    yin = y[tr][ok]
    if len(np.unique(yin)) < 2:
        return np.full(len(te), np.nan, np.float32)
    st = LogisticRegression(C=1.0 if Ctr is None else 0.3, max_iter=600, class_weight="balanced").fit(Xin, yin)
    return rank01(st.predict_proba(stack_features(outer, Cte))[:, 1], st.predict_proba(Xin)[:, 1])


def save_heads(tag: str, ids: list, P: np.ndarray, extra: dict | None = None) -> str:
    """Pickle tuong thich heads2: .tmp_v6/cache/heads2_<tag>.pkl = {ids, P (n, 3) float32, ...}.
    O chua cham (NaN: fold thieu duong) ghi 0 de fuse9 doc duoc (0 = khong bao) va danh dau trong 'unscored' (n, 3) cho report_fx."""
    path = os.path.join(CACHE, f"heads2_{tag}.pkl")
    P = np.asarray(P, np.float32)
    un = ~np.isfinite(P)
    d = {"ids": list(ids), "P": np.where(un, 0.0, P).astype(np.float32), "layers": (12,), "unscored": un}
    d.update(extra or {})
    dump_atomic(d, path)
    return path


def load_pickle(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "rb") as fh:
        return pickle.load(fh)


def dump_atomic(obj, path: str) -> None:
    """Ghi pickle qua tep .tmp roi doi ten (ngat giua chung khong lam hong ban cu); tep dong TRUOC khi doi ten (Windows)."""
    with open(path + ".tmp", "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(path + ".tmp", path)


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def read_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_wav16(path: str, max_s: float = 0.0) -> np.ndarray:
    """Doc wav -> mono float32 16 kHz (resample neu can), cat max_s giay neu > 0."""
    import soundfile as sf
    frames = -1
    if max_s:
        frames = int(max_s * sf.info(path).samplerate) + 1           # tep dai (amilong): chi doc phan can dung
    y, sr = sf.read(path, frames=frames, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    if max_s:
        y = y[: int(max_s * 16000)]
    return np.ascontiguousarray(y, np.float32)


def nanq(x: Sequence[float], q: float) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if len(a) else math.nan
