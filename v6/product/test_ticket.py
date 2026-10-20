# -*- coding: utf-8 -*-
"""Kiem thu KHONG CAN mo hinh cho lop ho so v9 (v6.product.ticket + v6.io.schema.check_contract + v6.io.report):
chieu dau hieu cao / thap hon, 12 canh bao hop dong, luat ca file xac suat >= 0,6 va vung xam 0,4-0,6 (04 / canh bao nghi 03), chua hieu chinh,
moi doan mot thao tac theo bien vuot nguong, chong lan 29 / 31 %, <= 2 dong / khuc 16 s + canh bao, cat mep 0,3 s, xoa loi <= 1 s,
kenh trai / phai, du cau gioi han, HTML co ma bam du + phien ban, CSV kenh, khong lo kieu con / cong cu; luat (4) muc tep o tep hai kenh,
nhanh ca file tat (khong co nguong) bo qua xac suat hieu chinh, khoi phien ban theo tep THUC nap (fold), nguon ngon ngu, dung lai Product;
bo nguong THAT (khong ttsr / ttsf): giong may khong soi phai ke ten trong gioi han + cau nhan 01 dung so thao tac; K1 ban 11/10 (warn_detail,
ghi_chu_k1, ca duong K1 04; dong 'kiem mot phan' tu not_impl); diem tho + nguong tho khi da hieu chinh; cua so xoa loi 1 s quanh dinh moi noi;
anh tong quan: lan cua lop khong soi de trong + ten 'không soi'; cau nhan 01 du phong (v6: 6 thao tac; v9 cu: khong neu so);
'python -m v6.io.schema' chay rieng (tu nap hang so v9) va --scan.
  python -m v6.product.test_ticket"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

import voiceaudit_v6 as V6  # noqa: E402
from v6.product import ticket as TK  # noqa: E402
from v6.io import report, schema  # noqa: E402
from v6.io.common import L04, OP  # noqa: E402

RES = os.path.join(BASE, ".tmp_v6", "results")
_META = {}
T_FAKE = {"ghep": 0.9, "xoa": 0.6, "den": 0.85, "voi": 0.5, "ttsr": 0.5, "nen": 0.9, "giong": 0.9}
PV = {"nguong": {"tep": "thresholds_v9A.json", "sha1": "a" * 40, "params": {"f03": 0.01}, "diem_hoat_dong": "A"},
      "mo_hinh": [{"vai_tro": "TCN vùng", "ten": "tcn_tcn_reg_f0.pt", "sha1": "b" * 40}]}


def meta() -> tuple:
    if not _META:
        _META["thr"] = json.load(open(os.path.join(RES, "thresholds_v9A.json"), encoding="utf-8"))
        _META["sig"] = json.load(open(os.path.join(RES, "refbank_signals.json"), encoding="utf-8"))["tables"]
    return _META["thr"], _META["sig"]


def channel(dur: float, rows: list, kenh: int = 0, head: dict | None = None, hp=None, seed: int = 0, peaks: list | None = None) -> dict:
    """Kenh gia: am thanh ngau nhien co dieu bien, diem khung thap (0,2) + dinh tai cac doan cho truoc [(cot pr, a, b, gia tri)]."""
    rng = np.random.default_rng(seed + kenh)
    n = int(dur * 16000)
    y = (rng.normal(0, 0.02, n) * (1 + np.sin(np.arange(n) / 4000))).astype(np.float32)
    F = int(dur / 0.02)
    pb = np.full((F, 2), 0.05)
    pr = np.full((F, 6), 0.05)
    for col, a, b, v in peaks or []:
        tgt = pb if col < 2 else pr
        tgt[int(a / 0.02):int(b / 0.02), col] = v
    det = {"rows": rows, "pb": pb, "pr": pr, "head": head or {"nen": 0.1, "giong": 0.1, "ttsf": 0.1}, "T": dict(T_FAKE), "dropped": False,
           "dur": dur, "kenh": kenh, "y": y, "cond": {"measures": {"diem_dieu_kien": {"snr": -1.0, "nen": -1.0, "vang": -1.0}}}}
    if hp is not False:
        det["head_prob"] = hp
    return det


def real_T(lang: str = "vi") -> dict:
    """Bo nguong THAT cua thresholds_v9A (khong co ttsr / spl / ttsf)."""
    return dict(meta()[0]["thr"][lang])


def build(chans: list, dur: float, warnings: list | None = None, ck_extra: dict | None = None) -> tuple:
    """assemble9 + finalize9 tren cac kenh gia -> (ho so, decisions, loi luoc do + #bind + hop dong). ck_extra: truong K1 them (vd warn_detail)."""
    thr, sig = meta()
    ho = V6.skeleton("t.wav", None)
    ho["he_thong"].update(ten="VoiceAudit v9 (thử nghiệm)", phien_ban=PV, manifest_sha256="c" * 64, mo_hinh="gia_lap")
    ho["tep"].update(sha256="d" * 64, kich_thuoc_byte=1, id="v9io_t", do_dai_s=dur, so_kenh_phan_tich=len(chans))
    per = [{"ngon_ngu": {"vi": 0.9, "en": 0.05, "top": "vi", "p_top": 0.9}, "noi_chong_pct": 1.0} for _ in chans]
    ck = {"decision": "canh_bao", "reasons": [], "warnings": list(warnings or []) + ["0,3 s đầu và cuối không được soi"],
          "measures": {"do_dai_s": dur, "kenh": per}, "not_impl": [], "channels": [c["y"] for c in chans]}
    for k, v in (ck_extra or {}).items():
        ck[k] = {**ck[k], **v} if isinstance(v, dict) and isinstance(ck.get(k), dict) else v
    ho["output_phu"]["diem_nguong"]["K1"] = V6._k1_block(ck)
    res = {"channels": chans, "k1": ck}
    dec = TK.assemble9(ho, res, "vi", thr["refs"], sig, thr["params"])
    TK.finalize9(ho, ck, dec, chans, "kho thử")
    err = schema.validate(ho) or schema.check_bind(ho) or schema.check_contract(ho)
    return ho, dec, err


def warns(ho: dict) -> list:
    return [w["noi_dung"] for w in ho["output_phu"]["canh_bao"]]


# ---------------------------------------------------------------- tests
def test_phrase_direction() -> None:
    lo = {"ten": "tiếng nền cả file", "gia_tri": -60.0, "don_vi": "dB", "trung_vi_that": -40.0, "cao_hon_pct": 1.0, "thap_hon_pct": 99.0}
    hi = {**lo, "gia_tri": -20.0, "cao_hon_pct": 98.0, "thap_hon_pct": 2.0}
    assert "thấp hơn 99 %" in TK.phrase9(lo), TK.phrase9(lo)
    assert "cao hơn 98 %" in TK.phrase9(hi), TK.phrase9(hi)
    assert "cao hơn" not in TK.phrase9(lo)


def test_explain_directions_in_records() -> None:
    rows = [{"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.95},
            {"level2": "02_giong_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "nm": 0.95, "via": "tcn_vung"}]
    ho, _, err = build([channel(20.0, rows, head={"nen": 0.95, "giong": 0.1, "ttsf": 0.1}, hp=None, peaks=[(3, 6.0, 8.0, 0.95)])], 20.0)
    assert not err, err
    for o in ho["output_chinh"]["thao_tac"]:
        V = []
        schema._explain_dirs(o, V)
        assert not V, V
        assert sum(1 for s in o["dau_hieu"] if s["loai"] == "tin_hieu") >= 2, o["dau_hieu"]


def test_explain_direction_violation_detected() -> None:
    o = {"id": "T1", "giai_thich": "x tiếng nền cả file −60,00 dB: cao hơn 1 % tệp; độ nghiêng 3,00: cao hơn 90 %",
         "dau_hieu": [{"ten": "tiếng nền cả file", "gia_tri": -60.0, "cao_hon_pct": 1.0, "thap_hon_pct": 99.0},
                      {"ten": "độ nghiêng", "gia_tri": 3.0, "cao_hon_pct": 90.0, "thap_hon_pct": 10.0}]}
    V = []
    schema._explain_dirs(o, V)
    assert len(V) == 1 and "sai chiều" in V[0], V


def test_warnings_contract_vocab() -> None:
    k1w = ["nén mạnh (< 24 kbps)", "tiếng ồn rất cao — SNR ước lượng < 10 dB (không chắc khi phòng vang)", "không đo được tiếng ồn (quá ít khoảng lặng)",
           "nhiều hơn 1 giọng (3)", "hai kênh khác nhau — phân tích từng kênh", "nói chồng ước lượng 15 % thời lượng có tiếng nói", "phòng vang — RT60 ước lượng trên 1,0 s",
           "có đoạn mất tín hiệu 1.2–1.5 s", "cắt đỉnh 1–5 %", "ngôn ngữ khác (nhận ra “fr”, không phải tiếng Việt / Anh)", "kiểu nói ngoài phạm vi (hát, thì thầm hoặc la hét)"]
    w, notes = TK.warnings9(k1w, [{"cham_gioi_han": True, "nghi_03": True, "sus03": ["x"]}])
    txt = [x["noi_dung"] for x in w]
    assert txt == schema.WARN_VOCAB, txt                                  # du 12, dung thu tu bang Output
    assert notes == ["không đo được tiếng ồn (quá ít khoảng lặng)", "hai kênh khác nhau — phân tích từng kênh"], notes
    mt = next(x for x in w if x["ma"] == "mat_tin_hieu")
    assert "1.2–1.5" in mt["chi_tiet"]


def test_whole_file_calibrated_rule() -> None:
    hp = {"nen": 0.7, "giong": 0.5, "ttsf": 0.3}
    ho, dec, err = build([channel(20.0, [{"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.99}], head={"nen": 0.99, "giong": 0.95, "ttsf": 0.2}, hp=hp)], 20.0)
    assert not err, err
    ops = [(o["nhan_cap2"], o["diem"]["p_hieu_chinh"]) for o in ho["output_chinh"]["thao_tac"]]
    assert ops == [("02_nen_toan_cuc", 0.7)], ops                       # giong 0,5: vung xam, khong bao du diem tho 0,95 >= nguong
    k3 = ho["output_phu"]["diem_nguong"]["kenh"][0]["K3"]
    assert k3["02_giong_toan_cuc"]["vung_xam"] == [0.4, 0.6] and k3["02_giong_toan_cuc"]["p_hieu_chinh"] == 0.5
    assert ho["output_chinh"]["nhan"]["ma"] == "02" and schema.WARN_03 not in warns(ho)
    assert not any("chưa hiệu chỉnh" in x for x in ho["output_phu"]["tuyen_bo_gioi_han"])


def test_grey_only_gives_04_with_reason() -> None:
    ho, dec, err = build([channel(20.0, [], hp={"nen": 0.45, "giong": 0.1, "ttsf": 0.1})], 20.0)
    assert not err, err
    assert ho["output_chinh"]["nhan"]["ma"] == "04"
    duong = [r["duong"] for r in ho["output_chinh"]["ly_do_04"]]
    assert duong == ["K3_vung_xam", "K5_khong_con_dong"], duong
    assert "0,45" in ho["output_chinh"]["ly_do_04"][0]["noi_dung"]
    assert [x["nhan_cap2"] for x in ho["phieu"]] == [L04] and ho["output_chinh"]["de_nghi"]
    assert schema.WARN_EDGE_C in warns(ho)


def test_grey_ttsf_with_02_warns_suspected_03() -> None:
    det = channel(20.0, [], hp={"nen": 0.8, "giong": 0.1, "ttsf": 0.5})
    det["T"]["ttsf"] = 0.9                                              # nhanh giong may ca file BAT (co nguong)
    ho, dec, err = build([det], 20.0)
    assert not err, err
    assert ho["output_chinh"]["nhan"]["ma"] == "02"
    assert schema.WARN_03 in warns(ho), warns(ho)


def test_suspected_03_only_with_label_02() -> None:
    rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 5.0, "end": 6.0, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"}]
    det = channel(20.0, rows, hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.5}, peaks=[(0, 5.0, 6.0, 0.99)])
    det["T"]["ttsf"] = 0.9
    ho, dec, err = build([det], 20.0)
    assert not err, err
    assert ho["output_chinh"]["nhan"]["ma"] == "03" and schema.WARN_03 not in warns(ho)
    assert ho["output_phu"]["diem_nguong"]["kenh"][0]["K8"]["nghi_03"], "K8 giu danh sach nghi 03 tho cua kenh"


def _stereo_grey_ttsf(right_rows: list, peaks: list) -> tuple:
    left = channel(20.0, [], kenh=0, hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.5})
    left["T"]["ttsf"] = 0.9
    right = channel(20.0, right_rows, kenh=1, hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.1}, peaks=peaks)
    right["T"]["ttsf"] = 0.9
    return build([left, right], 20.0)


def test_stereo_suspected_03_decided_at_file_level() -> None:
    # luat (4) o muc TEP: kenh trai vung xam giong may (nhan kenh 04), kenh phai co dong 02 -> tep 02 + canh bao; truoc day thieu canh bao va ma thoat 4
    r1 = [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "nm": 0.9, "via": "tcn_vung"}]
    ho, dec, err = _stereo_grey_ttsf(r1, [(3, 6.0, 8.0, 0.9)])
    assert err == [], err
    assert [d["label"] for d in dec] == ["04", "02"] and ho["output_chinh"]["nhan"]["ma"] == "02"
    w = next(x for x in ho["output_phu"]["canh_bao"] if x["noi_dung"] == schema.WARN_03)
    assert "kênh trái" in w["chi_tiet"], w
    assert not ho["output_chinh"]["ly_do_04"]


def test_stereo_suspected_03_not_with_label_03() -> None:
    r1 = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 10.0, "end": 11.0, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"}]
    ho, dec, err = _stereo_grey_ttsf(r1, [(0, 10.0, 11.0, 0.99)])
    assert err == [], err
    assert ho["output_chinh"]["nhan"]["ma"] == "03" and schema.WARN_03 not in warns(ho)


def test_branch_off_ignores_calibrated_probability() -> None:
    # tep nguong khong co 'ttsf' (nhanh giong may ca file tat): xac suat hieu chinh 0,9 KHONG duoc bat lai nhanh
    ho, dec, err = build([channel(20.0, [], head={"nen": 0.1, "giong": 0.1, "ttsf": 0.95}, hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.9})], 20.0)
    assert err == [], err
    assert ho["output_chinh"]["nhan"]["ma"] == "01" and not ho["output_chinh"]["thao_tac"]
    k3 = ho["output_phu"]["diem_nguong"]["kenh"][0]["K3"]["03_giong_may_toan_cuc"]
    assert k3["nhanh_tat"] is True and k3["p_hieu_chinh_bo_qua"] == 0.9 and "nhánh tắt" in k3["ghi_chu"]
    lim = " ".join(ho["output_phu"]["tuyen_bo_gioi_han"])
    assert "Nhánh đang tắt" in lim and "giọng máy cả file" in lim and "chưa hiệu chỉnh" not in lim
    ho2, _, err2 = build([channel(20.0, [], hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.5})], 20.0)
    assert err2 == [] and ho2["output_chinh"]["nhan"]["ma"] == "01", (err2, ho2["output_chinh"]["nhan"])     # vung xam cua nhanh tat khong ra 04


def test_branch_off_with_op_is_violation() -> None:
    det = channel(20.0, [], head={"nen": 0.1, "giong": 0.1, "ttsf": 0.95}, hp={"nen": 0.1, "giong": 0.1, "ttsf": 0.9})
    det["T"]["ttsf"] = 0.9
    ho, _, err = build([det], 20.0)
    assert err == [] and [o["nhan_cap2"] for o in ho["output_chinh"]["thao_tac"]] == ["03_giong_may_toan_cuc"]
    ho["output_phu"]["diem_nguong"]["kenh"][0]["K3"]["03_giong_may_toan_cuc"]["nhanh_tat"] = True
    assert any("nhánh tắt" in v for v in schema.check_contract(ho))


def test_versions_follow_loaded_checkpoints() -> None:
    import voiceaudit_v9 as V9
    rec = {"ckpt": [{"role": "tcn:tcn", "file": "tcn_tcn_f2.pt", "sha1": "e" * 40}, {"role": "heads", "file": "heads_final_h2_f2.joblib", "sha1": "f" * 40}],
           "fold": 2, "thr_scope": "thr_fold[2]: tep that cua cac fold khac", "sha_check": "verified", "resolved": {"bound": "tcn", "reg": "tcn_reg", "heads": "h2"}}
    thr, _ = meta()
    pv = V9.versions(None, "thresholds_v9A.json", thr["params"], "A", rec=rec)
    names = [m["ten"] for m in pv["mo_hinh"]]
    assert names[:2] == ["tcn_tcn_f2.pt", "heads_final_h2_f2.joblib"], names
    assert not any("final.pt" in n or n == "heads_final_h2.joblib" for n in names), names
    assert pv["nguong"]["fold"] == 2 and pv["nguong"]["kiem_sha1_voi_tep_nguong"] == "verified" and pv["nguon_danh_sach"].startswith("Product.loaded")
    assert pv["mo_hinh"][0]["vai_tro"] == "TCN (tcn)" and pv["mo_hinh"][1]["vai_tro"] == "đầu cả tệp"


def test_product_cache_reuse_with_default_params() -> None:
    import voiceaudit_v9 as V9
    key, sentinel = ("Z", True, 4, "fuse1", "fuse1", "h2"), object()
    V9._PRODUCT[key] = sentinel
    try:
        assert V9.product("Z", True, 4) is sentinel                    # mac dinh None: dung lai Product da nap (e2e_test / robust_test)
        assert V9.product("Z", True, 4, "fuse1", None, "h2") is sentinel
    finally:
        V9._PRODUCT.pop(key, None)


def test_lang_source_from_pipeline_result() -> None:
    import voiceaudit_v9 as V9
    f = V9.lang_source
    assert f("en", True, None, {}, "en") == "tham số --lang"
    assert f(None, True, None, {"lang_source": "lid", "lid": "en"}, "en") == "K1 mở rộng (Whisper)"
    assert f(None, True, None, {"lang_source": "mac_dinh_vi", "lid": None}, "vi").startswith("mặc định vi (K1 mở rộng không nhận ra")
    assert "không có bộ ngưỡng" in f(None, True, None, {"lang_source": "lid", "lid": "fr"}, "vi")
    assert f(None, False, ("en", {"en": 0.9, "vi": 0.1}), {"lang_source": "goi_y"}, "en").startswith("nhận dạng ngôn ngữ riêng")
    assert f(None, False, (None, None), {"lang_source": "mac_dinh_vi"}, "vi").startswith("mặc định vi (tắt K1 mở rộng")


def test_uncalibrated_keeps_threshold_rule_and_says_so() -> None:
    rows = [{"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.95}]
    ho, dec, err = build([channel(20.0, rows, head={"nen": 0.95, "giong": 0.5, "ttsf": 0.1}, hp=False)], 20.0)
    assert not err, err
    assert [o["nhan_cap2"] for o in ho["output_chinh"]["thao_tac"]] == ["02_nen_toan_cuc"]
    k3 = ho["output_phu"]["diem_nguong"]["kenh"][0]["K3"]["02_giong_toan_cuc"]
    assert k3["vung_xam"] is None and "chưa hiệu chỉnh" in k3["ghi_chu"]
    assert any("chưa hiệu chỉnh" in x for x in ho["output_phu"]["tuyen_bo_gioi_han"])
    assert "chưa hiệu chỉnh" in ho["output_chinh"]["thao_tac"][0]["giai_thich"]


def test_partial_calibration_key_none() -> None:
    # pipeline I1: khoa khong co bo hieu chinh -> None; khoa do giu luat nguong diem va ho so ghi 'chua hieu chinh'
    rows = [{"level2": "03_chen_giong_may", "kind": "global", "p": 0.95}]
    det = channel(20.0, rows, head={"nen": 0.99, "giong": 0.1, "ttsf": 0.95}, hp={"nen": 0.7, "giong": 0.05, "ttsf": None})
    det["T"]["ttsf"] = 0.9
    ho, dec, err = build([det], 20.0)
    assert not err, err
    ops = sorted((o["nhan_cap2"], o["diem"]["p_hieu_chinh"]) for o in ho["output_chinh"]["thao_tac"])
    assert ops == [("02_nen_toan_cuc", 0.7), ("03_giong_may_toan_cuc", None)], ops
    k3 = ho["output_phu"]["diem_nguong"]["kenh"][0]["K3"]
    assert k3["03_giong_may_toan_cuc"]["hieu_chinh"] is False and k3["02_nen_toan_cuc"]["hieu_chinh"] is True
    assert any("chưa hiệu chỉnh" in x for x in ho["output_phu"]["tuyen_bo_gioi_han"])
    assert ho["output_chinh"]["nhan"]["ma"] == "03"


def test_one_op_per_segment_by_margin_not_raw_p() -> None:
    # den: diem 0,90 / nguong 0,85 (vuot it) · voi: diem 0,80 / nguong 0,50 (vuot xa) - chong lan 100 % -> giu doi giong
    rows = [{"level2": "02_nen_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "nm": 0.9, "via": "tcn_vung"},
            {"level2": "02_giong_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.8, "nm": 0.8, "via": "tcn_vung"}]
    ho, dec, err = build([channel(20.0, rows, hp=None, peaks=[(2, 6.0, 8.0, 0.9), (3, 6.0, 8.0, 0.8)])], 20.0)
    assert not err, err
    assert [o["nhan_cap2"] for o in ho["output_chinh"]["thao_tac"]] == ["02_giong_cuc_bo"]


def test_overlap_29_vs_31_percent() -> None:
    def run(b2: float) -> int:
        rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 5.0, "end": 7.0, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"},
                {"level2": "02_giong_cuc_bo", "kind": "segment", "start": b2, "end": b2 + 2.0, "p": 0.8, "nm": 0.8, "via": "tcn_vung"}]
        ho, _, err = build([channel(30.0, rows, hp=None, peaks=[(0, 5.0, 7.0, 0.99), (3, b2, b2 + 2.0, 0.8)])], 30.0)
        assert not err, err
        return len([o for o in ho["output_chinh"]["thao_tac"] if o["pham_vi"] == "một đoạn"])
    assert run(6.42) == 2                                               # chong 0,58 / 2 = 29 %
    assert run(6.38) == 1                                               # chong 0,62 / 2 = 31 %


def test_chunk_cap_and_warning() -> None:
    rows = [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": a, "end": a + 1.0, "p": 0.8, "nm": v, "via": "tcn_vung"}
            for a, v in ((2.0, 0.7), (6.0, 0.9), (10.0, 0.8))]
    ho, dec, err = build([channel(30.0, rows, hp=None, peaks=[(3, 2.0, 3.0, 0.7), (3, 6.0, 7.0, 0.9), (3, 10.0, 11.0, 0.8)])], 30.0)
    assert not err, err
    kept = [(o["bat_dau_s"]) for o in ho["output_chinh"]["thao_tac"]]
    assert kept == [6.0, 10.0], kept                                    # bo dong vuot it nhat
    assert schema.WARN_MORE_C in warns(ho)


def test_edge_clamp_and_xoa_max_1s() -> None:
    rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 0.0, "end": 0.35, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"},
            {"level2": "03_ghep_giong_that", "kind": "segment", "start": 0.1, "end": 2.0, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"},
            {"level2": "03_ghep_giong_that", "kind": "segment", "start": 20.0, "end": 22.0, "p": 1.0, "nm": 0.9, "via": "ranh_gioi_xoa"}]
    ho, dec, err = build([channel(40.0, rows, hp=None, peaks=[(0, 0.0, 2.0, 0.99), (1, 20.0, 22.0, 0.9)])], 40.0)
    assert not err, err
    ops = [(o["nhan_cap2"], o["bat_dau_s"], o["ket_thuc_s"]) for o in ho["output_chinh"]["thao_tac"]]
    assert ops == [("03_ghep_giong_that", 0.3, 2.0), ("03_xoa_loi", 20.5, 21.5)], ops


def test_two_channels_left_right_and_report() -> None:
    r0 = [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "nm": 0.9, "via": "tcn_vung"}]
    c0 = channel(20.0, r0, kenh=0, hp=None, peaks=[(3, 6.0, 8.0, 0.9)])
    c1 = channel(20.0, [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 10.0, "end": 11.0, "p": 1.0, "nm": 0.99, "via": "ranh_gioi_ghep"}],
                 kenh=1, hp=None, peaks=[(0, 10.0, 11.0, 0.99)])
    ho, dec, err = build([c0, c1], 20.0)
    assert not err, err
    assert [o["kenh_ten"] for o in ho["output_chinh"]["thao_tac"]] == ["trái", "phải"]
    assert [m["kenh_ten"] for m in ho["output_phu"]["bang_moc"]] == ["trái", "phải"]
    h = report.build_html(ho)
    assert "kênh trái" in h and "kênh phải" in h and "kênh 1" not in h
    with tempfile.TemporaryDirectory(dir=os.path.join(BASE, ".tmp_v6")) as d:
        report.write_csvs(ho, d)
        rows = list(csv.DictReader(open(os.path.join(d, "bang_moc.csv"), encoding="utf-8-sig")))
    assert [r["kenh"] for r in rows] == ["trái", "phải"], rows
    bad = json.loads(json.dumps(ho))
    bad["output_chinh"]["thao_tac"][0]["kenh_ten"] = None
    assert any(v.startswith("O5") for v in schema.check_contract(bad))


def test_mono_has_no_channel_name() -> None:
    rows = [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "nm": 0.9, "via": "tcn_vung"}]
    ho, dec, err = build([channel(20.0, rows, hp=None, peaks=[(3, 6.0, 8.0, 0.9)])], 20.0)
    assert not err, err
    assert ho["output_chinh"]["thao_tac"][0]["kenh_ten"] is None
    assert "kênh" not in report._main_section(ho).split("<article")[1].split("</h3>")[0]


def test_limitations_complete() -> None:
    lim = " ".join(TK.limitations9(None, [], False, "kho thử"))
    miss = [m for m in schema.LIMIT_MARKERS if m not in lim]
    assert not miss, miss
    assert "chưa hiệu chỉnh" in lim and "160 tệp VIVOS" in lim and "Chưa có cổng sàng lọc" in lim


def test_k1_unchecked_follows_actual_checks() -> None:
    ck = {"measures": {"kenh": [{"ngon_ngu": {"top": "vi"}, "noi_chong_pct": 2.0}]}, "not_impl": []}
    chans = [{"cond": {"measures": {"diem_dieu_kien": {"snr": 0.1, "nen": 0.1, "vang": 0.1}}}}]
    un = TK.k1_unchecked(ck, chans)
    assert un == ["loại âm thanh (podcast, phát thanh, sách nói, nhạc)", "kiểu nói (hát, thì thầm, la hét)"], un
    un0 = TK.k1_unchecked({"measures": {"kenh": [{}]}, "not_impl": ["x"]}, [])
    assert any("ngôn ngữ" in x for x in un0) and any("≤ 10 %" in x for x in un0)


def test_report_has_full_hash_versions_and_bank_footnote() -> None:
    rows = [{"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.95}]
    ho, dec, err = build([channel(20.0, rows, head={"nen": 0.95, "giong": 0.1, "ttsf": 0.1}, hp=None)], 20.0)
    assert not err, err
    h = report.build_html(ho)
    assert "d" * 64 in h and "thresholds_v9A.json" in h and "a" * 40 in h and "tcn_tcn_reg_f0.pt" in h and "b" * 40 in h
    assert "Kho tham chiếu" in h and "không phải xác suất tệp bị sửa" in h
    assert "VoiceAudit v9" in h and "VoiceAudit v6" not in h


def test_contract_names_and_no_subtype_words() -> None:
    assert OP["03_chen_giong_may"][0] == "giọng máy" and OP["03_giong_may_toan_cuc"][0] == "giọng máy" and OP["03_xoa_loi"][0] == "xoá lời"
    assert {v[0] for v in OP.values()} == set(schema.CONTRACT_TYPES)
    for k, txt in TK.WHY.items():
        assert not any(w in txt for w in schema.FORBIDDEN_IN_EXPLAIN), (k, txt)


def test_file_label_order() -> None:
    f = lambda *labs: TK.file_label([{"label": x} for x in labs])     # noqa: E731
    assert f("01", "04") == "04" and f("04", "02") == "02" and f("02", "03") == "03" and f("01", "01") == "01"


def test_missing_versions_is_violation() -> None:
    rows = [{"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.95}]
    ho, dec, err = build([channel(20.0, rows, head={"nen": 0.95, "giong": 0.1, "ttsf": 0.1}, hp=None)], 20.0)
    ho["he_thong"]["phien_ban"] = None
    assert any(v.startswith("O3") for v in schema.check_contract(ho))


def test_real_thresholds_machine_voice_not_examined() -> None:
    # thresholds_v9A that: khong co ttsr / ttsf -> giong may (doan + ca file) KHONG soi: khong bao, gioi han ke ten, cau 01 dung so thao tac
    rows = [{"level2": "03_chen_giong_may", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "via": "tcn_vung"},
            {"level2": "03_chen_giong_may", "kind": "global", "p": 0.97}]
    det = channel(20.0, rows, head={"nen": 0.1, "giong": 0.1, "ttsf": 0.97}, hp=None, peaks=[(5, 6.0, 8.0, 0.9)])
    det["T"] = real_T("vi")
    ho, dec, err = build([det], 20.0)
    assert not err, err
    assert ho["output_chinh"]["nhan"]["ma"] == "01" and not ho["output_chinh"]["thao_tac"], ho["output_chinh"]["thao_tac"]
    soi = ho["output_phu"]["diem_nguong"]["thao_tac_soi"]
    assert soi["so_duoc_soi"] == 6 and soi["khong_soi"] == ["giọng máy một đoạn", "giọng máy cả file"], soi
    off_line = next(x for x in ho["output_phu"]["tuyen_bo_gioi_han"] if x.startswith("Nhánh đang tắt"))
    assert "giọng máy một đoạn" in off_line and "giọng máy cả file" in off_line and "6/8" in off_line, off_line
    assert TK.MV_TESTED not in ho["output_phu"]["tuyen_bo_gioi_han"]       # khong noi 'giong may moi thu...' khi giong may khong soi
    h = report.build_html(ho)
    assert "Không thấy dấu vết của 6 thao tác được soi; không soi: giọng máy một đoạn, giọng máy cả file" in h
    assert "Không thấy dấu vết của 8 thao tác" not in h
    assert TK.ops_off({"T": T_FAKE}) == ["03_giong_may_toan_cuc"]          # nguong gia lap co ttsr: chi ca file tat


def test_coverage_claim_violation_detected() -> None:
    det = channel(20.0, [], hp=None)
    det["T"] = real_T("vi")
    ho, _, err = build([det], 20.0)
    assert not err, err
    bad = json.loads(json.dumps(ho))
    bad["output_phu"]["tuyen_bo_gioi_han"] = [x for x in bad["output_phu"]["tuyen_bo_gioi_han"] if not x.startswith("Nhánh đang tắt")]
    assert any(v.startswith("S3") and "giọng máy một đoạn" in v for v in schema.check_contract(bad)), schema.check_contract(bad)
    bad2 = json.loads(json.dumps(ho))
    del bad2["output_phu"]["diem_nguong"]["thao_tac_soi"]
    assert any(v.startswith("S3 thiếu") for v in schema.check_contract(bad2))


def test_k1_new_output_shape_warn_detail_and_notes() -> None:
    # K1 11/10: warnings da chuan hoa + warn_detail (moc) + measures.ghi_chu_k1 (chu ngoai hop dong) + not_impl luon co
    extra = {"warnings": ["có đoạn mất tín hiệu"], "warn_detail": {"có đoạn mất tín hiệu": ["3.9–4.3 s"]},
             "measures": {"ghi_chu_k1": ["không đo được tiếng ồn (quá ít khoảng lặng)"]}, "not_impl": ["Tốc độ phát lệch ≤ ± 5 % (không tự kiểm được)"]}
    ho, _, err = build([channel(20.0, [], hp=None)], 20.0, ck_extra=extra)
    assert not err, err
    mt = next(w for w in ho["output_phu"]["canh_bao"] if w["noi_dung"] == "có đoạn mất tín hiệu")
    assert mt.get("chi_tiet") == "3.9–4.3 s", mt
    assert ho["output_phu"]["diem_nguong"]["K1"]["ghi_chu_khong_thuoc_hop_dong"] == ["không đo được tiếng ồn (quá ít khoảng lặng)"]
    assert not any(r["ma"] == "ghi_chu_k1" for r in ho["output_phu"]["bang_so_do"])
    assert warns(ho) == ["có đoạn mất tín hiệu", "0,3 s đầu và cuối không được soi"], warns(ho)


def test_calibrated_whole_file_keeps_raw_threshold() -> None:
    ho, _, err = build([channel(20.0, [], head={"nen": 0.5, "giong": 0.1, "ttsf": 0.1}, hp={"nen": 0.65, "giong": 0.05, "ttsf": 0.02})], 20.0)
    assert not err, err
    d = ho["output_chinh"]["thao_tac"][0]["diem"]
    assert d["diem"] == 0.5 and d["nguong"] == T_FAKE["nen"] and d["p_hieu_chinh"] == 0.65 and d["nguong_xac_suat"] == 0.6, d


def test_xoa_window_centred_on_joint() -> None:
    rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 20.0, "end": 23.0, "p": 1.0, "nm": 0.9, "via": "ranh_gioi_xoa"}]
    ho, _, err = build([channel(40.0, rows, hp=None, peaks=[(1, 21.0, 21.1, 0.9)])], 40.0)
    assert not err, err
    o = ho["output_chinh"]["thao_tac"][0]
    assert (o["nhan_cap2"], o["bat_dau_s"], o["ket_thuc_s"]) == ("03_xoa_loi", 20.55, 21.55), o["bat_dau_s"]
    assert TK.joint_of(20.0, 23.0, None) == 21.5


def test_k1_events_partial_check_uses_k1_not_impl() -> None:
    # lop su kien am CO chay (measures.kenh[*].su_kien_am): khong ghi 'khong kiem loai am thanh' nhung PHAI ghi phan K1 chua kiem duoc (not_impl)
    per = [{"ngon_ngu": {"vi": 0.9, "en": 0.05, "top": "vi", "p_top": 0.9}, "noi_chong_pct": 1.0, "su_kien_am": {"Music_mean": 0.01}}]
    ni = ["Loại âm thanh: podcast / phát thanh / sách nói không tách được bằng lớp AudioSet — chỉ chặn nhạc / bài hát",
          "Kiểu nói: người lớn (trẻ em chỉ ghi số đo, không chặn)", "Tốc độ phát lệch ≤ ± 5 % (không tự kiểm được)"]
    ho, _, err = build([channel(20.0, [], hp=None)], 20.0, ck_extra={"measures": {"kenh": per}, "not_impl": ni})
    assert not err, err
    lim = " ".join(ho["output_phu"]["tuyen_bo_gioi_han"])
    assert "loại âm thanh: podcast / phát thanh / sách nói không tách được" in lim and "kiểu nói: người lớn (trẻ em" in lim, lim
    assert "loại âm thanh (podcast, phát thanh, sách nói, nhạc)" not in lim
    ho2, _, _ = build([channel(20.0, [], hp=None)], 20.0)                  # khong co su_kien_am -> ca hai dong 'khong tu kiem'
    assert "loại âm thanh (podcast, phát thanh, sách nói, nhạc)" in " ".join(ho2["output_phu"]["tuyen_bo_gioi_han"])


def test_overview_lane_of_off_branch_blank_and_named() -> None:
    # anh tong quan: lop cuc bo khong soi (giong may doan voi nguong that) -> cot diem khung de trong + ten lan 'không soi'
    det = channel(20.0, [], hp=None, peaks=[(5, 6.0, 8.0, 0.9), (3, 10.0, 11.0, 0.7)])
    det["T"] = real_T("vi")
    _, dec, err = build([det], 20.0)
    assert not err, err
    P, _ = TK.frame_matrix(dec[0])
    j = TK.C.CLS.index("03_chen_giong_may")
    assert P[:, j].max() == 0.0 and P[:, TK.C.CLS.index("02_giong_cuc_bo")].max() == 0.7
    names = TK.lane_names(dec[0])
    assert names[j - 1] == "giọng máy đoạn" + TK.OFF_LANE and all(TK.OFF_LANE not in n for i, n in enumerate(names) if i != j - 1), names
    _, dec2, _ = build([channel(20.0, [], hp=None, peaks=[(5, 6.0, 8.0, 0.9)])], 20.0)    # nguong gia lap co ttsr -> lan van ve
    assert TK.frame_matrix(dec2[0])[0][:, j].max() == 0.9 and TK.OFF_LANE not in TK.lane_names(dec2[0])[j - 1]


def test_k1_04_path_reads_warn_detail() -> None:
    # duong K1 ra 04 (voiceaudit_v9.k1_fail9) dung chung k1_warn_block: warn_detail -> chi_tiet, measures.ghi_chu_k1 -> ghi chu ngoai hop dong
    ck = {"warnings": ["có đoạn mất tín hiệu"], "warn_detail": {"có đoạn mất tín hiệu": ["1.0–1.2 s"]}, "measures": {"ghi_chu_k1": ["ghi chú lạ"]}}
    w, notes = TK.k1_warn_block(ck, [], analysed=False)
    assert w == [{"ma": "mat_tin_hieu", "noi_dung": "có đoạn mất tín hiệu", "chi_tiet": "1.0–1.2 s"}] and notes == ["ghi chú lạ"], (w, notes)


def test_label01_sentence_fallbacks() -> None:
    # ho so khong co diem_nguong.thao_tac_soi: v6 -> 6 thao tac (he v6); v9 cu -> khong neu so (khong biet nhanh nao tat)
    v6 = {"he_thong": {"ten": "VoiceAudit v6"}, "output_phu": {"diem_nguong": {}}}
    v9 = {"he_thong": {"ten": "VoiceAudit v9 (thử nghiệm)"}, "output_phu": {"diem_nguong": {}}}
    assert report.label01_text(v6).startswith("Không thấy dấu vết của 6 thao tác được soi")
    t9 = report.label01_text(v9)
    assert "các thao tác được soi" in t9 and not any(ch.isdigit() for ch in t9), t9


def test_schema_cli_and_scan_on_v9_record() -> None:
    # 'python -m v6.io.schema' chay RIENG (khong nap ticket truoc): ho so v9 hop le phai DAT (truoc day bao sai S3 loai / C3 gioi han); --scan dem dung
    import subprocess
    rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 20.0, "end": 23.0, "p": 1.0, "nm": 0.9, "via": "ranh_gioi_xoa"},
            {"level2": "03_chen_giong_may", "kind": "segment", "start": 6.0, "end": 8.0, "p": 0.9, "via": "tcn_vung"}]
    ho, _, err = build([channel(40.0, rows, hp=None, peaks=[(1, 21.0, 21.1, 0.9), (5, 6.0, 8.0, 0.9)])], 40.0)
    assert not err and {o["nhan_cap2"] for o in ho["output_chinh"]["thao_tac"]} == {"03_xoa_loi", "03_chen_giong_may"}, err
    ho["kiem_tra_dong_nhat"] = {"dat": True, "vi_pham": []}
    with tempfile.TemporaryDirectory(dir="D:\\tmp_va") as td:
        os.makedirs(os.path.join(td, "a"))
        p = os.path.join(td, "a", "ho_so.json")
        json.dump(ho, open(p, "w", encoding="utf-8"), ensure_ascii=False, default=float)
        bad = json.loads(json.dumps(ho, default=float))
        bad["output_phu"]["canh_bao"].append({"ma": "k1", "noi_dung": "không đo được tiếng ồn"})          # canh bao ngoai hop dong
        os.makedirs(os.path.join(td, "b"))
        json.dump(bad, open(os.path.join(td, "b", "ho_so.json"), "w", encoding="utf-8"), ensure_ascii=False)
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, "-m", "v6.io.schema", p], cwd=BASE, capture_output=True, text=True, encoding="utf-8", env=env)
        assert r.returncode == 0 and r.stdout.strip() == "ĐẠT", (r.returncode, r.stdout[-800:], r.stderr[-800:])
        s = subprocess.run([sys.executable, "-m", "v6.io.schema", "--scan", td], cwd=BASE, capture_output=True, text=True, encoding="utf-8", env=env)
        assert s.returncode == 1 and "đạt 1 · không đạt 1" in s.stdout and os.path.join(td, "b") in s.stdout, (s.stdout[-800:], s.stderr[-800:])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    bad = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            bad += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
    print(f"{len(tests) - bad}/{len(tests)} dat")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
