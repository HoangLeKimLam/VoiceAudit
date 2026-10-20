# -*- coding: utf-8 -*-
"""VoiceAudit v9 — bo chay NHANH mot tep (chi XLS-R dong bang + dau nho, CPU): input -> day du output theo VoiceAudit_6Muc (muc Output, Scope).

  python voiceaudit_v9.py <tep ghi am> -o <thu muc ra> [--op A|F] [--lang vi|en] [--no-k1x] [--no-pdf] [--threads 4] [--fp32]
--op mac dinh A (run() va CLI giong nhau); F = fuse1 (thresholds_v9F.json, ban nghien cuu tot nhat) - doi mac dinh sang F la quyet dinh rieng, cho ket qua e2e --op F.
Ca A va F deu KHONG co nguong giong may (ttsr / ttsf): ho so ghi 'giong may mot doan / ca file' la thao tac KHONG soi (diem_nguong.thao_tac_soi + gioi han).
Ra: ho_so.json · anh/*.png · bao_cao.html/.pdf · bang_moc.csv · bang_so_do.csv · log.txt. Ma thoat: 0 xong (01/02/03) · 2 tu choi nhan · 3 nhan 04 · 4 loi.
K1 (vaudit.k1, mac dinh MO RONG k1x: ngon ngu, noi chong) -> XLS-R + TCN + dau ca file (v6.product.pipeline) -> ho so (v6.product.ticket: luat Scope,
12 canh bao, gioi han) -> anh / HTML / PDF (v6.io.render, report). Ho so ghi phien ban: ten tep nguong + tham so + ten tung tep mo hinh THUC nap
(Product.model_record luc phan tich, ke ca dinh tuyen fold) + sha1. --bound / --reg / --heads bo trong (mac dinh) = theo params cua tep nguong.
Ngon ngu chon bo nguong: --lang > K1 mo rong (Whisper, mac dinh) > nhan dang ngon ngu rieng (khi --no-k1x) ; run() va CLI cung mac dinh k1x=True.
Kiem khong can mo hinh: python -m v6.product.golden (vi du 6 s cua hop dong + ma thoat 0/2/3/4 + PDF)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import voiceaudit_v6 as V6  # noqa: E402  (dung lai khung K1 / khung xuong / Steps)
from v6.product import ticket as TK  # noqa: E402  (patch hang so TRUOC khi nap schema / render / report)
from v6.io import common  # noqa: E402
from v6.io.common import EXIT_04, EXIT_ERR, EXIT_OK, EXIT_REJECT, L1_NAME  # noqa: E402

LOG = logging.getLogger("voiceaudit.v9")
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, ".tmp_v6", "results")
CACHE = os.path.join(BASE, ".tmp_v6", "cache")
XLSR = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
OPTIONAL_ROLES = {"ngưỡng K1 mở rộng"}                  # tep tuy chon: thieu thi ma dung gia tri mac dinh (khong phai loi)
_PRODUCT = {}
_SHA1 = {}


def product(op: str, int8: bool, threads: int, bound: str | None = None, reg: str | None = None, heads: str | None = None):
    """Product cho diem hoat dong op. bound / reg / heads = None -> theo params cua tep nguong (Product.resolve_params; khac tep nguong -> ValueError).
    Dung lai Product da nap (e2e_test / robust_test dat san trong _PRODUCT) neu op / int8 / threads khop va cac tham so da cho (khac None) khop."""
    want = (bound, reg, heads)
    for key, P in _PRODUCT.items():
        if key[:3] == (op, int8, threads) and all(w is None or w == g for w, g in zip(want, key[3:])):
            return P
    from v6.product.pipeline import Product
    P = Product(bound, reg, heads, f"thresholds_v9{op}.json", int8, threads)
    _PRODUCT[(op, int8, threads, P.bound, P.reg, P.heads_tag)] = P
    return P


# ---------------------------------------------------------------- phien ban (O-3)
def sha1_file(path: str) -> str | None:
    """sha1 tep (nho dem theo duong dan + kich thuoc + mtime)."""
    if not os.path.isfile(path):
        return None
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime)
    if key not in _SHA1:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        _SHA1[key] = h.hexdigest()
    return _SHA1[key]


EXTRA_FILES = (("PCA khung XLS-R", os.path.join(BASE, "data_labels", "realistic_v4", "xlsr_frames", "_pca64.npz")),
               ("điều kiện ghi (SNR / nền / vang)", os.path.join(BASE, "vaudit", "cond_models.joblib")),
               ("ngưỡng K1 mở rộng", os.path.join(BASE, "vaudit", "k1x_thresholds.json")))


def role_vn(role: str) -> str:
    """Vai tro trong Product.loaded ('bound:tcn', 'tcn:tcn_reg', 'ft:ft2', 'heads') -> ten tieng Viet."""
    kind, _, tag = role.partition(":")
    return {"tcn": f"TCN ({tag})", "ft": f"FT-2 LoRA ({tag})", "heads": "đầu cả tệp"}.get(kind, role)


def model_files(P, bound: str, reg: str, heads: str) -> list:
    """DU PHONG khi khong co Product.loaded (bo gia lap): [(vai tro, duong dan)] tai dung bang pipeline.expected_ckpts (mot quy tac dat ten) + tep phu."""
    if hasattr(P, "model_files"):
        return list(P.model_files)
    from v6.product.pipeline import expected_ckpts
    names = {"bound": "ranh giới", "reg": "vùng"}
    out = []
    for r, p in expected_ckpts(bound, reg, heads, None):                 # vai tro: bound:tcn / bound:ft / reg:tcn / heads
        use, _, kind = r.partition(":")
        out.append((f"{'FT-2 LoRA' if kind == 'ft' else 'TCN'} ({names.get(use, use)})" if kind else "đầu cả tệp", p))
    return out + list(EXTRA_FILES)


def loaded_files(P, rec: dict | None) -> list | None:
    """Tep mo hinh Product THUC nap luc phan tich: rec = res['models'] (Product.model_record) hoac P.loaded. [(vai tro, ten, sha1)]; None neu khong co."""
    ck = (rec or {}).get("ckpt")
    if ck is None:
        ck = getattr(P, "loaded", None)
    if ck is None:
        return None
    return [(role_vn(m["role"]), m["file"], m.get("sha1")) for m in ck] + [(r, os.path.basename(p), sha1_file(p)) for r, p in EXTRA_FILES]


def _xlsr_version() -> dict:
    v = {"ten": XLSR, "ban": None, "ghi_chu": "XLS-R chống giả mạo, đóng băng, 12 tầng đầu"}
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(XLSR, "config.json")
        if isinstance(p, str):
            v["ban"] = os.path.basename(os.path.dirname(p))
            v["sha1"] = sha1_file(p)
    except Exception as ex:                                     # khong co cache HF: ghi ro, khong chan
        v["ghi_chu"] += f" (không đọc được bản trong cache: {type(ex).__name__})"
    return v


def versions(P, thr_name: str, params: dict, op: str, bound: str | None = None, reg: str | None = None, heads: str | None = None,
             rec: dict | None = None) -> dict:
    """Khoi phien ban (O-3). rec = res['models'] (Product.model_record luc phan tich): ten + sha1 checkpoint THUC nap (ke ca dinh tuyen fold),
    fold, pham vi nguong, ket qua so sha1 voi tep nguong. Khong co rec / P.loaded (bo gia lap) -> tai dung danh sach tu tham so (ghi ro nguon)."""
    rec = rec if rec is not None else (P.model_record() if hasattr(P, "model_record") else None)
    src = loaded_files(P, rec)
    nguon = "Product.loaded (tệp thực nạp lúc phân tích)"
    if src is None:
        nguon = "tái dựng từ tham số (không có Product.loaded — bộ giả lập)"
        src = [(r, os.path.basename(p), sha1_file(p)) for r, p in model_files(P, bound or "tcn", reg or "tcn_reg", heads or "h2")]
    mods, opt = [], []
    for role, name, s in src:
        if s is None and role in OPTIONAL_ROLES:
            opt.append(f"{name}: không có tệp — dùng giá trị mặc định trong mã")
            continue
        mods.append({"vai_tro": role, "ten": name, "sha1": s, **({} if s else {"ghi_chu": "không thấy tệp"})})
    rec = rec or {}
    hd = (rec.get("resolved") or {}).get("heads") or getattr(P, "heads_tag", None) or heads or "h2"
    cal = os.path.join(RES, f"calib_{hd}.json")
    calrec = rec.get("calib") or ({"file": os.path.basename(cal), "sha1": sha1_file(cal)} if os.path.exists(cal) else None)
    pv = {"nguong": {"tep": thr_name, "sha1": sha1_file(os.path.join(RES, thr_name)), "params": params, "diem_hoat_dong": op,
                     "che_do": getattr(P, "mode", bound), "giai_ra": rec.get("resolved"), "fold": rec.get("fold"), "pham_vi_nguong": rec.get("thr_scope"),
                     "kiem_sha1_voi_tep_nguong": rec.get("sha_check")},
          "mo_hinh": [m for m in mods if m["sha1"]], "thieu": [m["ten"] for m in mods if not m["sha1"]], "mac_dinh": opt, "nguon_danh_sach": nguon,
          "xlsr": _xlsr_version(), "kho_tham_chieu": {"ten": "refbank_signals.json", "sha1": sha1_file(os.path.join(RES, "refbank_signals.json"))},
          "hieu_chinh": {"ten": calrec["file"], "sha1": calrec["sha1"]} if calrec else None,
          "ma_nguon": {os.path.basename(p): sha1_file(p) for p in (os.path.join(BASE, "v6", "product", "pipeline.py"), os.path.join(BASE, "v6", "product", "ticket.py"),
                                                                    os.path.abspath(__file__))}}
    return pv


def manifest_sha(pv: dict) -> str:
    return hashlib.sha256(json.dumps(pv, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- ngon ngu + hieu chinh
def detect_lang(path: str) -> tuple:
    """Nhan dang ngon ngu rieng (Whisper, vaudit.k1x) khi khong bat K1 mo rong. Tra (vi|en|None, xac suat)."""
    from vaudit import k1, k1x
    x = k1.decode(path, 1)
    if x is None or not len(x):
        return None, None
    lp = k1x.language_probs(x[:, 0])
    return ("en" if lp["en"] > lp["vi"] else "vi"), lp


def lang_used(P, res: dict, fallback: str) -> str:
    """Bo nguong ma pipeline THUC dung (so khop det['T'] voi P.thr) - ticket dung cung kho tham chieu cua ngon ngu do."""
    thr = getattr(P, "thr", {}) or {}
    for det in res.get("channels", []):
        for k, v in thr.items():
            if v is det["T"] or v == det["T"]:
                return k
    return fallback


def head_probs(det: dict, heads: str, lang: str | None = None):
    """Giao dien I1: det['head_prob'] do pipeline dat; neu pipeline chua dat khoa thi tu ap v6.product.calib (neu co). None = chua hieu chinh."""
    if "head_prob" in det:
        return det["head_prob"]
    try:
        from v6.product import calib
        cal = calib.load(heads)
    except Exception as ex:                                     # chua co module / tep hieu chinh -> ghi ro 'chua hieu chinh'
        LOG.info("khong co hieu chinh dau ca tep (%s): %s", heads, ex)
        return None
    if not cal:
        return None
    try:
        return {k: float(calib.apply(cal, k, float(v), lang=lang)) for k, v in det["head"].items() if v is not None}
    except Exception as ex:                                     # giao dien I1 khac ky vong: khong bia xac suat, ghi 'chua hieu chinh'
        LOG.error("ap hieu chinh dau ca tep loi (%s): %s", heads, ex)
        return None


# ---------------------------------------------------------------- ghi ket qua
def finish9(ho: dict, out_dir: str, steps, no_pdf: bool) -> int:
    """Kiem luoc do + #bind + luat hop dong TRUOC khi ghi; ghi ho so, CSV, HTML, PDF. Tra ma thoat."""
    from v6.io import report, schema
    err = schema.validate(ho) or schema.check_bind(ho) or schema.check_contract(ho)
    ho["kiem_tra_dong_nhat"] = {"dat": not err, "vi_pham": err}
    if err:
        with open(os.path.join(out_dir, "ho_so_khong_hop_le.json"), "w", encoding="utf-8") as fh:
            json.dump(ho, fh, ensure_ascii=False, indent=1, default=str)
        LOG.error("ho so KHONG hop le, khong ghi ho_so.json: %s", err)
        return EXIT_ERR
    report.write_csvs(ho, out_dir)
    ok = True
    with steps("K9 báo cáo HTML"):
        hp = report.write_html(ho, out_dir)
    if not no_pdf:
        with steps("K9 PDF (Chrome)"):
            ok, note, _ = report.to_pdf(hp, os.path.join(out_dir, "bao_cao.pdf"))
        if not ok:
            LOG.error("PDF: %s", note)
    ho["thoi_gian_s"] = {**ho.get("thoi_gian_s", {}), **dict(steps.t)}
    with open(os.path.join(out_dir, "ho_so.json"), "w", encoding="utf-8") as fh:
        json.dump(ho, fh, ensure_ascii=False, indent=1, default=float)
    if not ok:
        return EXIT_ERR
    lab = ho["output_chinh"]["nhan"]["ma"]
    return EXIT_REJECT if ho["trang_thai"] == "tu_choi" else EXIT_04 if lab == "04" else EXIT_OK


def k1_fail9(ho: dict, ck: dict, out_dir: str, steps, no_pdf: bool, dur: float, kho: str) -> int:
    from v6.io.explain import ADVICE_04, reasons_k1
    from v6.io.render import render_overview
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    oc["nhan"] = {"ma": "04", "ten": L1_NAME["04"]}
    oc["ly_do_04"] = reasons_k1(ck)
    oc["de_nghi"] = ADVICE_04
    w, notes = TK.k1_warn_block(ck, [], analysed=False)          # K1 11/10: warn_detail -> chi_tiet, measures.ghi_chu_k1 -> ghi chu ngoai hop dong
    op_["canh_bao"] = w
    op_["diem_nguong"]["K1"]["ghi_chu_khong_thuoc_hop_dong"] = notes
    op_["bang_so_do"] = TK.k1_measures9(ck)
    op_["tuyen_bo_gioi_han"] = TK.limitations9(ck, [], None, kho, ho["he_thong"].get("canh_bao_he_thong"))
    if ck.get("channels"):
        with steps("K9 ảnh"):
            n = len(ck["channels"])
            for i, y in enumerate(ck["channels"]):
                kt = TK.kenh_ten(i, n)
                name = f"anh/tong_quan{'_' + ('trai' if kt == 'trái' else 'phai') if kt else ''}.png"
                render_overview(y, dur, [], None, None, os.path.join(out_dir, name), "Tổng quan · nhãn 04 · trượt K1")
                op_["anh_tong_quan"].append(name)
    return finish9(ho, out_dir, steps, no_pdf)


def _bank_note(sig_meta: dict) -> str:
    return (f"phân vị điểm mô hình và số đo tín hiệu trên tệp thật tự nhiên cùng ngôn ngữ (refbank_signals.json: "
            f"{sig_meta.get('nguon', 'không rõ nguồn')}; điểm chuyên gia: mục 'refs' của tệp ngưỡng)")


def _clean_out(out_dir: str) -> None:
    os.makedirs(os.path.join(out_dir, "anh"), exist_ok=True)
    for f in V6.OUT_FILES:
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    for f in os.listdir(os.path.join(out_dir, "anh")):
        if f.endswith(".png"):
            os.remove(os.path.join(out_dir, "anh", f))


def lang_source(lang_cli: str | None, k1x: bool, detected: tuple | None, res: dict, lg: str) -> str:
    """Nguon chon bo nguong ngon ngu, lay tu ket qua THUC (res['lang_source'] cua Product: goi_y / lid / mac_dinh_vi), khong suy doan."""
    if lang_cli:
        return "tham số --lang"
    if detected is not None:
        g, lp = detected
        if g is None:
            return "mặc định vi (tắt K1 mở rộng; nhận dạng ngôn ngữ riêng không đo được)"
        return f"nhận dạng ngôn ngữ riêng vì tắt K1 mở rộng ({json.dumps(lp, ensure_ascii=False)})"
    ls, lid = res.get("lang_source"), res.get("lid")
    if ls == "lid" and lid == lg:
        return "K1 mở rộng (Whisper)"
    if ls == "lid":
        return f"mặc định {lg} (K1 mở rộng nhận ra {lid!r}, không có bộ ngưỡng cho ngôn ngữ đó)"
    if ls == "mac_dinh_vi":
        return f"mặc định {lg} (K1 mở rộng không nhận ra ngôn ngữ)" if k1x else f"mặc định {lg}"
    return "K1 mở rộng (Whisper)" if k1x and lid is None and lg in TK.LANGS else f"mặc định {lg}"


def run(inp: str, out_dir: str, *, op: str = "A", lang: str | None = None, k1x: bool = True, no_pdf: bool = False, int8: bool = True, threads: int = 4,
        bound: str | None = None, reg: str | None = None, heads: str | None = None, analyser=None) -> int:
    """analyser: doi tuong co .analyse(path, lang, extended_k1) va .thr (mac dinh Product that); golden.py truyen bo gia lap khong can mo hinh.
    bound / reg / heads = None (mac dinh) -> theo params cua tep nguong (doi tep nguong sang fuse1 khong lam moi lan chay mac dinh loi)."""
    _clean_out(out_dir)
    steps, t_all = V6.Steps(), time.time()
    ho = V6.skeleton(inp, None)
    ho["he_thong"]["ten"] = "VoiceAudit v9 (thử nghiệm)"
    sig_meta = json.load(open(os.path.join(RES, "refbank_signals.json"), encoding="utf-8"))
    kho = _bank_note(sig_meta)
    ho["output_phu"]["tuyen_bo_gioi_han"] = TK.limitations9(None, [], None, kho)
    if os.path.isdir(inp):
        V6._reject(ho, "từ chối nhận · số tệp mỗi lần phải đúng 1 (nhận được một thư mục)")
        return finish9(ho, out_dir, steps, no_pdf)
    if not os.path.isfile(inp):
        raise FileNotFoundError(inp)
    ho["tep"].update(sha256=V6._sha256(inp), kich_thuoc_byte=os.path.getsize(inp))
    ho["tep"]["id"] = f"v9io_{ho['tep']['sha256'][:16]}"
    if os.path.getsize(inp) > 250 * 1024 * 1024:
        V6._reject(ho, "từ chối nhận · dung lượng > 250 MB")
        return finish9(ho, out_dir, steps, no_pdf)
    P = analyser if analyser is not None else product(op, int8, threads, bound, reg, heads)
    if getattr(P, "stub", False):
        ho["he_thong"]["mo_hinh"] = "gia_lap"
    thr_name = getattr(P, "thr_name", f"thresholds_v9{op}.json")
    meta = json.load(open(os.path.join(RES, thr_name), encoding="utf-8"))
    refs, params = meta["refs"], meta["params"]
    sig = sig_meta["tables"]
    detected = None
    if lang is None and not k1x:
        with steps("nhận dạng ngôn ngữ (Whisper)"):
            detected = detect_lang(inp)
    with steps("K1 + XLS-R + TCN + dau ca file"):
        res = P.analyse(inp, lang or (detected or (None,))[0], extended_k1=k1x)
    pv = versions(P, thr_name, params, op, bound, reg, heads, rec=res.get("models"))      # SAU khi phan tich: dung tep thuc nap (ke ca fold)
    ho["he_thong"]["phien_ban"] = pv
    ho["he_thong"]["manifest_sha256"] = manifest_sha(pv)
    if pv["thieu"]:
        ho["he_thong"]["canh_bao_he_thong"].append("thiếu tệp mô hình khi ghi phiên bản: " + ", ".join(pv["thieu"]))
    ck = res["k1"]
    ho["output_phu"]["diem_nguong"]["K1"] = V6._k1_block(ck)
    if ck["decision"] == "tu_choi":
        V6._reject(ho, "; ".join(ck["reasons"]))
        ho["output_phu"]["tuyen_bo_gioi_han"] = TK.limitations9(ck, [], None, kho, ho["he_thong"]["canh_bao_he_thong"])
        return finish9(ho, out_dir, steps, no_pdf)
    chans = ck["channels"]
    dur = round(len(chans[0]) / 16000, 3) if chans else float(ck["measures"].get("do_dai_s") or 0.0)
    ho["tep"].update(do_dai_s=dur, so_kenh_phan_tich=len(chans))
    if ck["decision"] == "04":
        return k1_fail9(ho, ck, out_dir, steps, no_pdf, dur, kho)
    k1_lang = res.get("lid") if "lid" in res else ((ck["measures"].get("kenh", [{}])[0].get("ngon_ngu", {}) or {}).get("top") if k1x else None)
    want = lang or (detected or (None,))[0] or res.get("lang") or (k1_lang if k1_lang in TK.LANGS else "vi")
    lg = lang_used(P, res, want)
    ho["he_thong"]["ngon_ngu_nguong"] = {"ma": lg, "nguon": lang_source(lang, k1x, detected, res, lg), "k1_nhan_ra": k1_lang}
    hd = (pv["nguong"].get("giai_ra") or {}).get("heads") or heads or "h2"
    for det in res["channels"]:
        det["head_prob"] = head_probs(det, hd, lg)
    with steps("K9 phiếu + giải thích"):
        decisions = TK.assemble9(ho, res, lg, refs, sig, params)
        TK.finalize9(ho, ck, decisions, res["channels"], kho)
    ho["he_thong"]["che_do_khuc"] = "XLS-R khúc 32 s chồng 2 s + TCN (CPU)"
    adj = [d.get("len_adj") for d in res["channels"] if d.get("len_adj")]
    ho["he_thong"]["nguong_theo_do_dai"] = {                                   # 18/10: tep dai -> nguong kenh doan nang de giu vu oan / tep
        "tep_hieu_chinh": (res.get("models", {}).get("lencal") or {}).get("file"), "trang_thai": res.get("models", {}).get("lencal_check"),
        "da_nang": bool(adj), "chi_tiet": adj}
    with steps("K9 ảnh"):
        TK.render_all(ho, decisions, out_dir)
    ho["thoi_gian_s"] = {**{k: round(v, 3) for k, v in res.get("timing_s", {}).items()}, "tong_s": round(time.time() - t_all, 3)}
    code = finish9(ho, out_dir, steps, no_pdf)
    LOG.info("TONG %.2f s cho %.2f s am thanh | ma thoat %d", time.time() - t_all, dur, code)
    return code


def setup_log(out_dir: str, verbose: bool = False) -> None:
    """log.txt trong thu muc ra: buoc do thoi gian (voiceaudit.v6) + thong bao / loi cua v9."""
    V6._setup_log(out_dir, verbose)
    LOG.handlers[:] = V6.LOG.handlers
    LOG.setLevel(logging.INFO)


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="VoiceAudit v9 — kiểm định bản ghi âm (thử nghiệm, chỉ nghiên cứu)")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--op", default="A", help="diem hoat dong = tep .tmp_v6/results/thresholds_v9<op>.json; A (mac dinh, cau hinh da chay that qua san pham) "
                                             "| F (fuse1 = ban nghien cuu tot nhat, mac dinh cua e2e_test; CHUA chay that qua voiceaudit_v9)")
    ap.add_argument("--lang", default=None, choices=[None, "vi", "en"])
    ap.add_argument("--k1x", action="store_true", default=True, help="K1 mo rong (ngon ngu, noi chong): mac dinh BAT (giong run())")
    ap.add_argument("--no-k1x", dest="k1x", action="store_false")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bound", default=None, help="bo trong = theo tep nguong (khac tep nguong -> loi)")
    ap.add_argument("--reg", default=None)
    ap.add_argument("--heads", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    setup_log(a.out, a.verbose)
    try:
        code = run(a.input, a.out, op=a.op, lang=a.lang, k1x=a.k1x, no_pdf=a.no_pdf, int8=not a.fp32, threads=a.threads, bound=a.bound, reg=a.reg, heads=a.heads)
    except Exception:
        LOG.error("LOI:\n%s", traceback.format_exc())
        code = EXIT_ERR
    print(f"ma thoat {code} · {os.path.join(a.out, 'ho_so.json')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
