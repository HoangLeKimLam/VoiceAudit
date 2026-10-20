# -*- coding: utf-8 -*-
"""Kiem thu cac CHE DO cua san pham (pipeline.Product) bang mo hinh GIA nho (khong nap XLS-R, khong doc .tmp_v6/cache): thu muc tam chua checkpoint FT-2 LoRA nho,
TCN vung, dau ca tep gia, tep nguong (co / khong params, co sha1, co thr_fold), PCA; transformers.from_pretrained bi thay bang wav2vec2 nho 24 tang (hidden 64).
Phu: (1) truy vet - tu choi tep nguong khac mo hinh (bound / reg / heads), tu choi sha1 checkpoint khac, tep nguong cu = tcn / tcn_reg / h2; (2) duong fuse1 them 07/10:
cot ghep / xoa / khu on / doi giong = trung binh logit FT-2 + TCN vung, cot doan ghep / giong may = FT-2, pb == pr, giai ma ra dong hop le; (3) dinh tuyen fold (fold=k, set_fold):
dung ft_ft2_f{k} / tcn_tcn_reg_f{k} / heads_final_h2_f{k} + thr_fold[k]; (4) det["head_prob"] tu calib (None khi chua hieu chinh); (5) 12/10: bo hieu chinh phai hoc
tren cung OOF dau ca tep voi tep nguong (lech -> khong dung, strict -> loi), bo hieu chinh khong tin cay -> None, strict tu choi checkpoint chua co trong tep nguong,
set_fold nguyen tu khi loi, Product.analyse chon bo nguong theo LID (lid / goi_y / mac_dinh_vi) va dieu kien hoc 04 -> quyet dinh K1 = 04.
  python -m v6.product.test_pipeline_modes"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace

os.environ["FT_CPU"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from v6.boundary import ft  # noqa: E402
from v6.boundary import tcn as T  # noqa: E402
from v6.boundary.ens6 import combine  # noqa: E402
from v6.product import calib as CAL  # noqa: E402
from v6.product import pipeline as P  # noqa: E402

HID = 64
LANG_THR = {"ghep": 0.55, "xoa": 0.55, "den": 0.5, "voi": 0.5, "ttsr": 0.5, "nen": 0.6, "giong": 0.6, "ttsf": 0.6}


class Ident:
    def transform(self, X):
        X = np.atleast_2d(X)
        return np.zeros((X.shape[0], 1)) + 0.0


class ConstProb:
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, B):
        return np.tile([1 - self.p, self.p], (len(B), 1))


def fake_heads(p_lr: float, p_gb: float) -> dict:
    m = {"sc": Ident(), "pca": Ident(), "lr": ConstProb(p_lr), "gb": ConstProb(p_gb), "ref_lr": np.linspace(0, 1, 11), "ref_gb": np.linspace(0, 1, 11)}
    return {"models": {"nen_toan": m, "giong_toan": m, "tts_file": m}, "layers": (12,)}


class FakeFx:
    def __call__(self, seg, sampling_rate, return_tensors):
        x = np.asarray(seg, np.float32)
        return SimpleNamespace(input_values=torch.from_numpy((x - x.mean()) / (x.std() + 1e-7))[None])


def tiny_w2v():
    from transformers import Wav2Vec2Config, Wav2Vec2Model
    torch.manual_seed(7)
    return Wav2Vec2Model(Wav2Vec2Config(hidden_size=HID, num_hidden_layers=24, num_attention_heads=4, intermediate_size=128, conv_dim=(32,) * 7,
                                        do_stable_layer_norm=True, feat_extract_norm="layer", num_feat_extract_layers=7, layerdrop=0.0)).eval()


@contextlib.contextmanager
def patched_env(tmp: str):
    """CACHE / RES / FR -> thu muc tam; transformers.from_pretrained -> mo hinh nho."""
    import transformers
    old = (P.CACHE, P.RES, P.FR, CAL.RES, vars(transformers.AutoFeatureExtractor).get("from_pretrained"), vars(transformers.AutoModelForAudioClassification).get("from_pretrained"))
    from vaudit import cond
    old_assess = cond.assess
    P.CACHE, P.RES, P.FR, CAL.RES = os.path.join(tmp, "cache"), os.path.join(tmp, "res"), os.path.join(tmp, "fr"), os.path.join(tmp, "res")
    transformers.AutoFeatureExtractor.from_pretrained = staticmethod(lambda *a, **k: FakeFx())
    transformers.AutoModelForAudioClassification.from_pretrained = staticmethod(lambda *a, **k: SimpleNamespace(wav2vec2=tiny_w2v()))
    cond.assess = lambda pooled: ({}, [], [])                          # vector gop gia (384 chieu) khong hop bo kiem dieu kien that
    try:
        yield
    finally:
        P.CACHE, P.RES, P.FR, CAL.RES, fa, fm = old
        for cls, f in ((transformers.AutoFeatureExtractor, fa), (transformers.AutoModelForAudioClassification, fm)):
            if f is None:
                delattr(cls, "from_pretrained")                        # lop goc khong tu dinh nghia -> tro lai ham ke thua
            else:
                setattr(cls, "from_pretrained", f)
        cond.assess = old_assess


def build(tmp: str, params: dict | None = None, with_sha: bool = True, with_fold_thr: bool = True) -> str:
    """Tao checkpoint gia + tep nguong; tra ten tep nguong."""
    cache, res, fr = (os.path.join(tmp, d) for d in ("cache", "res", "fr"))
    for d in (cache, res, fr):
        os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(fr, "_pca64.npz"), W=np.random.default_rng(0).normal(0, 0.2, (HID, 64)).astype(np.float32))
    for k in (-1, 0, 1, 2, 3, 4):
        torch.manual_seed(100 + k)
        m = ft.FT(n_train=6, n_cls=7, rank=4, small=True).eval()
        with torch.no_grad():
            for n_, v in m.state_dict().items():
                if n_.endswith(".B"):
                    v.normal_(0, 0.05)
        torch.save(m.trainable_state(), os.path.join(cache, f"ft_ft2_f{k}.pt"))
    for tag, n_cls in (("tcn_reg", 5), ("tcn", 4)):
        for k in range(5):
            torch.manual_seed(200 + k + n_cls)
            torch.save(T.TCN(n_cls=n_cls).state_dict(), os.path.join(cache, f"tcn_{tag}_f{k}.pt"))
    joblib.dump(fake_heads(0.7, 0.4), os.path.join(cache, "heads_final_h2.joblib"))
    for k in range(5):
        joblib.dump(fake_heads(0.1 * k, 0.9), os.path.join(cache, f"heads_final_h2_f{k}.joblib"))
    prm = dict(params if params is not None else {"bound": "fuse1", "reg": "fuse1", "heads": "h2", "f03": 0.01, "freg": 0.03, "fh": 0.005, "ftts": 0.01,
                                                   "snap": True, "typed": True})
    if with_sha and "bound" in prm:
        old = P.CACHE
        P.CACHE = cache
        try:
            prm["ckpt_sha1"] = {os.path.basename(p): P.sha1_file(p) for fold in (None, 0, 1, 2, 3, 4)
                                for _, p in P.expected_ckpts(prm["bound"], prm["reg"], prm["heads"], fold) if os.path.exists(p)}
        finally:
            P.CACHE = old
    j = {"name": "T", "params": prm, "thr": {"vi": LANG_THR, "en": {**LANG_THR, "ghep": 0.6}}}
    if with_fold_thr:
        j["thr_fold"] = {str(k): {"vi": {**LANG_THR, "ghep": 0.50 + 0.01 * k}, "en": LANG_THR} for k in range(5)}
    name = "thresholds_v9T.json"
    json.dump(j, open(os.path.join(res, name), "w", encoding="utf-8"))
    return name


def wave(sec: float = 3.0, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.1, int(sec * P.SR)).astype(np.float32)


# ------------------------------------------------------------------ truy vet (khong can mo hinh)
def test_resolve_params_legacy_alias_and_refusal() -> None:
    assert P.resolve_params({}, "tcn", "tcn_reg", "h2")[:3] == ("tcn", "tcn_reg", "h2")
    assert P.resolve_params({}, None, None, None)[3] is True
    full = {"bound": "fuse1x", "reg": "fuse1x", "heads": "h2"}
    assert P.resolve_params(full, "fuse1", None, None)[:3] == ("fuse1x", "fuse1x", "h2")       # fuse1x = cung mo hinh voi fuse1
    for args in (("fuse1", None, None), (None, "tcn_reg", None), (None, None, "h3")):
        try:
            P.resolve_params({} if args[0] else full, *args)
        except ValueError:
            continue
        raise AssertionError(f"phai tu choi {args}")


def test_expected_ckpts_final_and_fold() -> None:
    with tempfile.TemporaryDirectory() as d:
        old = P.CACHE
        P.CACHE = d
        try:
            fin = [os.path.basename(p) for _, p in P.expected_ckpts("fuse1", "fuse1", "h2", None)]
            assert fin == ["ft_ft2_f-1.pt"] + [f"tcn_tcn_reg_f{k}.pt" for k in range(5)] + ["heads_final_h2.joblib"], fin
            open(os.path.join(d, "tcn_tcn_reg_final.pt"), "wb").close()
            fin = [os.path.basename(p) for _, p in P.expected_ckpts("fuse1", "fuse1", "h2", None)]
            assert fin == ["ft_ft2_f-1.pt", "tcn_tcn_reg_final.pt", "heads_final_h2.joblib"], fin
            f3 = [os.path.basename(p) for _, p in P.expected_ckpts("fuse1x", "fuse1x", "h3", 3)]
            assert f3 == ["ft_ft2_f3.pt", "tcn_tcn_reg_f3.pt", "heads_final_h3_f3.joblib"], f3
            tc = [os.path.basename(p) for _, p in P.expected_ckpts("tcn", "tcn_reg", "h2", 1)]
            assert tc == ["tcn_tcn_f1.pt", "tcn_tcn_reg_f1.pt", "heads_final_h2_f1.joblib"], tc
        finally:
            P.CACHE = old


def test_check_sha() -> None:
    loaded = [{"file": "a.pt", "sha1": "1" * 40}, {"file": "b.pt", "sha1": "2" * 40}]
    assert P.check_sha({}, loaded).startswith("unverified")
    assert P.check_sha({"ckpt_sha1": {"a.pt": "1" * 40, "b.pt": "2" * 40}}, loaded) == "verified"
    assert "b.pt" in P.check_sha({"ckpt_sha1": {"a.pt": "1" * 40}}, loaded)
    try:
        P.check_sha({"ckpt_sha1": {"a.pt": "9" * 40}}, loaded)
    except ValueError:
        return
    raise AssertionError("phai tu choi sha1 khac")


# ------------------------------------------------------------------ Product voi mo hinh nho
def test_fuse1_default_final_models() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1)
            assert prod.mode == "fuse1" and prod.ft is not None and prod._l6 is None and prod.fold is None
            rec = prod.model_record()
            names = [m["file"] for m in rec["ckpt"]]
            assert names[0] == "tcn_tcn_reg_f0.pt" and "ft_ft2_f-1.pt" in names and "heads_final_h2.joblib" in names and rec["sha_check"] == "verified", rec
            assert len(prod.tcn["tcn_reg"]) == 5 and rec["resolved"]["bound"] == "fuse1" and rec["calib"] is None
            tm = {}
            det = prod.analyse_channel(wave(3.0), "vi", tm)
            n = ft.n_frames(3 * P.SR)
            assert det["pb"].shape == (n, 6) and det["pb"] is det["pr"], det["pb"].shape
            assert det["head_prob"] is None and det["head_prob_method"] is None and set(det["head"]) == {"nen", "giong", "ttsf"} and rec["calib_scope"] is None
            # cong thuc fuse1: tinh lai tu hai nguon
            h, _ = prod.encode(wave(3.0))
            from v6.product.fthead import FTHead
            pf = FTHead.smooth_cols(np.concatenate(prod._ftp))
            z = ((h - np.median(h, axis=0, keepdims=True)) @ prod.W).astype(np.float16)
            preg = prod.tcn_probs(prod.tcn["tcn_reg"], z)
            assert preg.shape == (n, 4)
            assert np.allclose(det["pb"][:, 1], combine([pf[:, 1], preg[:, 1]], [0.5, 0.5], "logit"), atol=1e-5)
            assert np.allclose(det["pb"][:, 5], np.clip(pf[:, 5], 1e-4, 1 - 1e-4), atol=1e-6)
            for r in det["rows"]:
                assert r["level2"][:2] in ("02", "03") and r["kind"] in ("segment", "global")
            assert {"xlsr", "tcn", "decode"} <= set(tm)


def test_fold_routing_and_switch() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        with patched_env(d):
            prod = P.Product(None, None, None, thr, int8=True, threads=1, fold=2)
            names = [m["file"] for m in prod.model_record()["ckpt"]]
            assert names == ["tcn_tcn_reg_f2.pt", "ft_ft2_f2.pt", "heads_final_h2_f2.joblib"], names
            assert prod.thr["vi"]["ghep"] == 0.52 and prod.thr_scope.startswith("thr_fold[2]") and prod._l6 is not None
            d2 = prod.analyse_channel(wave(2.5, 1), "vi", {})
            prod.set_fold(4)
            assert [m["file"] for m in prod.loaded][1] == "ft_ft2_f4.pt" and prod.thr["vi"]["ghep"] == 0.54
            d4 = prod.analyse_channel(wave(2.5, 1), "vi", {})
            assert not np.allclose(d2["pb"], d4["pb"])                    # mo hinh khac -> diem khac
            assert d2["head"]["nen"] != d4["head"]["nen"]                 # dau ca tep cua fold khac
            prod.set_fold(None)
            assert prod.thr is prod.thr_all and prod.model_record()["ckpt"][-1]["file"] == "heads_final_h2.joblib"
            try:
                prod.set_fold(7)
            except ValueError:
                pass
            else:
                raise AssertionError("fold 7 phai bi tu choi")


def test_fold_without_thr_fold_records_scope() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d, with_fold_thr=False)
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1, fold=1)
            assert prod.thr is prod.thr_all and "khong co thr_fold" in prod.thr_scope


def test_refuses_mismatched_models_and_changed_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        with patched_env(d):
            for args in (("tcn", None, None), (None, "tcn_reg", None), (None, None, "h3")):
                try:
                    P.Product(*args, thr=thr, int8=False, threads=1)
                except ValueError:
                    continue
                raise AssertionError(f"phai tu choi {args}")
            torch.manual_seed(999)
            torch.save(T.TCN(n_cls=5).state_dict(), os.path.join(d, "cache", "tcn_tcn_reg_f3.pt"))      # hoc lai sau khi dat nguong
            P.Product(thr=thr, int8=False, threads=1, fold=1)              # fold 1 khong dung tep do -> van nap
            for fold in (3, None):
                try:
                    P.Product(thr=thr, int8=False, threads=1, fold=fold)
                except ValueError as ex:
                    assert "tcn_tcn_reg_f3.pt" in str(ex)
                    continue
                raise AssertionError(f"fold {fold}: phai tu choi checkpoint da doi")


def test_legacy_threshold_file_runs_tcn_mode() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d, params={"f03": 0.01, "freg": 0.03, "fh": 0.005, "ftts": 0.0, "fspl": 0.0}, with_sha=False)
        with patched_env(d):
            prod = P.Product("tcn", "tcn_reg", "h2", thr, int8=False, threads=1)
            rec = prod.model_record()
            assert prod.mode == "tcn" and prod.ft is None and rec["legacy_params"] and rec["sha_check"].startswith("unverified")
            det = prod.analyse_channel(wave(2.0, 3), "en", {})
            assert det["pb"].shape[1] == 3 and det["pr"].shape[1] == 4 and len(prod.tcn_b) == 5 and len(prod.tcn_r) == 5
            try:
                P.Product("fuse1", None, None, thr, int8=False, threads=1)
            except ValueError:
                return
            raise AssertionError("tep nguong cu (tcn) phai bi tu choi cho fuse1")


def test_head_prob_from_calibration_file() -> None:
    S, Y, lang, fold, groups, real = CAL._fake(n=800, seed=11)
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        cal = {"heads": "h2", "cal": CAL.calibrate(S, Y, lang, fold, groups, real)}
        json.dump(cal, open(os.path.join(d, "res", "calib_h2.json"), "w", encoding="utf-8"))
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1)
            det = prod.analyse_channel(wave(2.0, 4), "vi", {})
            hp = det["head_prob"]
            assert set(hp) == {"nen", "giong", "ttsf"} and all(v is None or 0.0 <= v <= 1.0 for v in hp.values()), hp
            assert abs(hp["nen"] - CAL.apply(cal, "nen", det["head"]["nen"], "vi")) < 1e-12
            assert prod.model_record()["calib"]["file"] == "calib_h2.json" and prod.model_record()["calib_scope"].startswith("OOF 5 fold (mo hinh cuoi")
            assert det["head_prob_method"]["ttsf"] == CAL.method_of(cal, "ttsf", "vi") and set(det["head_prob_method"]) == set(hp)
            den = prod.analyse_channel(wave(2.0, 4), "en", {})              # _fake: khong co giong may tieng Anh -> bo hieu chinh gop 'all', phai ghi ro
            assert den["head_prob_method"]["ttsf"].endswith("(gop all)"), den["head_prob_method"]


def _edit_thr(d: str, name: str, fn) -> None:
    p = os.path.join(d, "res", name)
    j = json.load(open(p, encoding="utf-8"))
    fn(j)
    json.dump(j, open(p, "w", encoding="utf-8"))


def _platt(rel: list | None = None) -> dict:
    return {"method": "platt", "a": 1.0, "b": 0.0, "rel": rel or []}


def test_calib_source_must_match_threshold_heads_oof() -> None:
    """Bo hieu chinh hoc tren OOF dau ca tep KHAC OOF ma tep nguong ghi (dau da hoc lai, chua hieu chinh lai) -> khong dung (head_prob None, ghi ly do); strict -> ValueError."""
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        _edit_thr(d, thr, lambda j: j["params"].update(oof_sha1={"heads": "a" * 40}))
        cal = {"heads": "h2", "source": {"oof": "heads2_h2.pkl", "sha1": "b" * 40}, "cal": {"all": {"nen": _platt(), "giong": _platt(), "ttsf": _platt()}}}
        json.dump(cal, open(os.path.join(d, "res", "calib_h2.json"), "w", encoding="utf-8"))
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1)
            rec = prod.model_record()
            assert prod.calib is None and rec["calib_check"].startswith("TU CHOI") and rec["calib"]["source_sha1"] == "b" * 40, rec["calib_check"]
            assert prod.analyse_channel(wave(2.0, 5), "vi", {})["head_prob"] is None and rec["calib_scope"] is None
            try:
                P.Product(thr=thr, int8=False, threads=1, strict=True)
            except ValueError as ex:
                assert "calib_h2.json" in str(ex)
            else:
                raise AssertionError("strict phai tu choi bo hieu chinh lech nguon")
            cal["source"]["sha1"] = "a" * 40
            json.dump(cal, open(os.path.join(d, "res", "calib_h2.json"), "w", encoding="utf-8"))
            ok = P.Product(thr=thr, int8=False, threads=1, strict=True)
            assert ok.calib is not None and ok.model_record()["calib_check"] == "verified"
            assert ok.analyse_channel(wave(2.0, 5), "vi", {})["head_prob"]["nen"] is not None


def test_unreliable_calibrator_gives_no_probability() -> None:
    """calib.issue_of: bo hieu chinh lech doi vung quyet dinh (nhu vi / ttsf cua calib_h2) -> head_prob[khoa] = None, head_prob_method ghi KHONG TIN CAY."""
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        bad = _platt([[0.6, 0.7, 32, 0.67, 0.19, 0.09, 0.35]])
        json.dump({"heads": "h2", "cal": {"all": {"nen": _platt(), "giong": _platt(), "ttsf": bad}}}, open(os.path.join(d, "res", "calib_h2.json"), "w", encoding="utf-8"))
        with patched_env(d):
            det = P.Product(thr=thr, int8=False, threads=1).analyse_channel(wave(2.0, 6), "vi", {})
            assert det["head_prob"]["ttsf"] is None and det["head_prob"]["nen"] is not None, det["head_prob"]
            assert "KHONG TIN CAY" in det["head_prob_method"]["ttsf"] and "KHONG" not in det["head_prob_method"]["nen"]


def test_strict_refuses_unknown_or_unverified_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        _edit_thr(d, thr, lambda j: j["params"]["ckpt_sha1"].pop("heads_final_h2_f3.joblib"))         # dau fold 3 lam SAU khi dat nguong
        with patched_env(d):
            assert P.Product(thr=thr, int8=False, threads=1, fold=3).sha_status.startswith("verified tru 1")
            try:
                P.Product(thr=thr, int8=False, threads=1, strict=True, fold=3)
            except ValueError as ex:
                assert "heads_final_h2_f3.joblib" in str(ex)
            else:
                raise AssertionError("strict phai tu choi checkpoint khong co trong tep nguong")
            P.Product(thr=thr, int8=False, threads=1, fold=2, strict=True)                          # fold 2: moi tep deu co sha1 -> nhan
        legacy = build(d, params={"f03": 0.01, "freg": 0.03, "fh": 0.005, "ftts": 0.0, "fspl": 0.0}, with_sha=False)
        with patched_env(d):
            try:
                P.Product("tcn", "tcn_reg", "h2", legacy, int8=False, threads=1, strict=True)
            except ValueError:
                pass
            else:
                raise AssertionError("strict phai tu choi tep nguong cu khong ghi sha1")


def test_set_fold_is_atomic_on_error() -> None:
    """Loi khi doi fold (thieu dau ca tep, sha1 khac) -> Product giu NGUYEN fold cu: mo hinh, nguong, fold, danh sach checkpoint deu cua fold cu."""
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1, fold=1)
            ft1, tcn1, heads1, names1 = prod.ft, prod.tcn, prod.heads, [m["file"] for m in prod.loaded]
            os.remove(os.path.join(d, "cache", "heads_final_h2_f3.joblib"))
            torch.manual_seed(321)
            torch.save(T.TCN(n_cls=5).state_dict(), os.path.join(d, "cache", "tcn_tcn_reg_f4.pt"))      # hoc lai sau khi dat nguong
            for fold, err in ((3, FileNotFoundError), (4, ValueError)):
                try:
                    prod.set_fold(fold)
                except err:
                    pass
                else:
                    raise AssertionError(f"fold {fold} phai loi {err.__name__}")
                assert prod.fold == 1 and prod.ft is ft1 and prod.tcn is tcn1 and prod.heads is heads1 and [m["file"] for m in prod.loaded] == names1
                assert prod.thr["vi"]["ghep"] == 0.51 and prod.thr_scope.startswith("thr_fold[1]")
            assert prod.analyse_channel(wave(2.0, 7), "vi", {})["pb"].shape[1] == 6


def test_analyse_language_routing_and_constraint_04() -> None:
    """Product.analyse: LID cua K1x chon bo nguong (lang_source lid / goi_y / mac_dinh_vi); dieu kien hoc 04 (cond) viet de quyet dinh K1 = 04 (e2e: Constraint 04)."""
    from vaudit import cond, k1
    with tempfile.TemporaryDirectory() as d:
        thr = build(d)
        y = wave(2.0, 8)

        def fake_check(top, decision="canh_bao"):
            return lambda path, extended=False: {"decision": decision, "reasons": [] if decision != "04" else ["04 · “băng hẹp”"], "warnings": [],
                                                 "channels": [] if decision == "04" else [y], "measures": {"kenh": [{"ngon_ngu": {"top": top}}]}}
        old = k1.check
        with patched_env(d):
            prod = P.Product(thr=thr, int8=False, threads=1)
            try:
                cases = (("en", None, True, "en", "lid"), (None, None, True, "vi", "mac_dinh_vi"), ("fr", None, True, "vi", "lid"),
                         ("vi", "en", True, "en", "goi_y"), ("en", None, False, "vi", "mac_dinh_vi"))
                for top, given, ext, want_lang, want_src in cases:
                    k1.check = fake_check(top)
                    out = prod.analyse("x.wav", given, extended_k1=ext)
                    assert (out["lang"], out["lang_source"]) == (want_lang, want_src), (top, given, ext, out["lang"], out["lang_source"])
                    assert out["lid"] == (top if ext else None) and out["channels"][0]["T"] is prod.thr[want_lang] and out["label"] in ("01", "02", "03")
                cond.assess = lambda pooled: ({}, [], ["04 · “ồn quá” — mô hình"])                        # moi kenh truot dieu kien hoc -> 04 nhu K1
                k1.check = fake_check("vi")
                out = prod.analyse("x.wav", None, extended_k1=True)
                assert out["label"] == "04" and out["k1"]["decision"] == "04" and out["rows"] == [] and "04 · “ồn quá” — mô hình" in out["k1"]["reasons"]
                k1.check = fake_check("vi", "04")
                out = prod.analyse("x.wav", None, extended_k1=True)
                assert out["label"] == "04" and out["channels"] == [] and out["models"]["thresholds"]["file"] == thr
            finally:
                k1.check = old


def test_len_thr_length_aware_thresholds() -> None:
    """18/10 lencal: tep <= d_ref giu CHINH doi tuong nguong; tep dai nang nguong kenh doan theo logit + sigma * ln(d / d_ref);
    dau ca tep (nen / giong) khong doi; khong co bo hieu chinh -> khong doi; load_lencal tu choi tep sinh tu tep nguong khac."""
    import math
    from v6.product import lencal as LC
    prod = P.Product.__new__(P.Product)
    prod.fold = 1
    T = {"ghep": 0.9, "xoa": 0.85, "den": 0.4, "voi": 0.3, "nen": 0.92, "giong": 0.9}
    prod.lencal = {"d_ref": 10.0, "sigma": {"vi": {"all": {k: {"sigma": 1.0, "xi": 0.0, "u": 0.0} for k in ("ghep", "xoa", "den", "voi")},
                                                 "1": {k: {"sigma": 2.0, "xi": 0.0, "u": 0.0} for k in ("ghep", "xoa", "den", "voi")}}}}
    Ts, note = prod.len_thr(T, "vi", 8.0)
    assert Ts is T and note is None
    Tl, note = prod.len_thr(T, "vi", 100.0)
    assert abs(LC.logit(Tl["den"]) - (LC.logit(0.4) + 2.0 * math.log(10))) < 1e-9          # sigma cua fold 1
    assert Tl["nen"] == 0.92 and Tl["giong"] == 0.9 and set(note["doi"]) == {"ghep", "xoa", "den", "voi"}
    prod.lencal["two_sided"] = True                                               # hai chieu: tep ngan HA nguong, dau ca tep van giu
    Tsh, note = prod.len_thr(T, "vi", 5.0)
    assert Tsh["den"] < 0.4 and Tsh["nen"] == 0.92 and note is not None and prod.len_thr(T, "vi", 10.0)[0] is T
    prod.lencal = None
    assert prod.len_thr(T, "vi", 600.0) == (T, None)
    with tempfile.TemporaryDirectory() as d:
        tp = os.path.join(d, "thresholds_vX.json")
        json.dump({"thr": {}}, open(tp, "w"))
        assert P.load_lencal(tp, True, False)[0] is None                                    # chua co tep
        json.dump({"thr_file": "thresholds_KHAC.json", "d_ref": 10.0, "sigma": {}}, open(os.path.join(d, "lencal_vX.json"), "w"))
        cal, rec, why = P.load_lencal(tp, True, False)
        assert cal is None and "khong dung" in why
        try:
            P.load_lencal(tp, True, True)
            raise AssertionError("strict phai tu choi")
        except ValueError:
            pass
        json.dump({"thr_file": "thresholds_vX.json", "d_ref": 10.0, "sigma": {}, "tail": "gpd"}, open(os.path.join(d, "lencal_vX.json"), "w"))
        cal, rec, why = P.load_lencal(tp, True, True)
        assert cal is not None and rec["file"] == "lencal_vX.json" and why == "dung"
        assert P.load_lencal(tp, False, True) == (None, None, "tat (len_adj=False)")


if __name__ == "__main__":
    n = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
            print("ok", name, flush=True)
    print(f"{n} test dat")
