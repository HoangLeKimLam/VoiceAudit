# -*- coding: utf-8 -*-
"""VI DU VANG (Output, O-8) cho san pham v9 - KHONG CAN mo hinh: tep tong hop 6 s co 2 thao tac biet truoc, giong vi du trong VoiceAudit_6Muc
(giong may · mot doan · 00:01,13 – 00:02,48 · sai lech nang  +  khu on · ca file · 00:00,00 – 00:06,00 · vo hai; canh bao 0,3 s dau va cuoi).
Chay TOAN BO voiceaudit_v9.run (K1 gia lap + diem khung gia lap thay Product; ticket, anh, HTML, CSV, PDF that) va so sanh voi BAN GHI KY VONG co dinh.
LUU Y: nhanh giong may mot doan dung nguong GIA LAP ttsr = 0,5 (bo nguong that thresholds_v9A / v9F KHONG co ttsr / spl / ttsf); ca 'golden_real' chay
cung tep voi nguong that va kiem ho so ghi trung thuc 'giong may mot doan / ca file: khong soi' (6/8 thao tac duoc soi), nhan 02.
Ca 'golden_F' = cau hinh fuse1 (--op F, thresholds_v9F.json): ban ghi mo hinh dung NHU Product.model_record (pipeline.expected_ckpts + sha1 tep that tren dia
+ pipeline.check_sha) -> kiem khoi phien ban: tep nguong F + sha1, checkpoint FT-2 / TCN / dau ca tep + sha1 khop params.ckpt_sha1 cua tep nguong.
KHONG thay cho chay Product that (nap mo hinh) - viec do la hang doi (e2e_test --op F).
Kiem them ma thoat: 0 (xong) · 2 (tu choi: thu muc) · 3 (K1 ra 04; vung xam 0,4-0,6 -> 04) · 4 (loi: tep khong ton tai).
PDF: mo bang pypdf (.tmp_v6/site_pdf) va kiem co ma bam SHA-256 du, ten tep nguong + sha1, ten tep mo hinh + sha1, thoi diem chay.
S-9 (cat dau-cuoi / doi am luong khong la chinh sua): s9_compare / s9_variants co kiem o day; chay that tren tep tu nhien la viec NANG (queue):
  python -m v6.product.golden [--no-pdf]
  python -m v6.product.golden --s9 [--limit 8] [--out D:/tmp_va/wp3/s9] [--op A|F]   (nap Product that; ghi .tmp_v6/results/s9_trim_gain_invariance[_F].md)"""
from __future__ import annotations

import json
import os
import re
import shutil
import sys
import traceback

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import voiceaudit_v9 as V9  # noqa: E402
from v6.io import report  # noqa: E402

SR = 16000
DUR = 6.0
SEG = (1.13, 2.48)
OUT = os.path.join(BASE, ".tmp_v6", "golden9")
RES = os.path.join(BASE, ".tmp_v6", "results")
EXPECTED = {
    "nhan": {"ma": "03", "ten": "Ghép hoặc chèn đoạn"},
    "thao_tac": [
        {"id": "T1", "loai": "khử ồn", "pham_vi": "cả file", "bat_dau": "00:00,00", "ket_thuc": "00:06,00", "muc_do": "vô hại", "kenh_ten": None,
         "anh": "anh/T1.png", "mo_dau": "Khử ồn cả file, mức vô hại. Xác suất khử ồn cả file (đã hiệu chỉnh) 0,93 ≥ 0,60 (điểm thô 0,97)."},
        {"id": "T2", "loai": "giọng máy", "pham_vi": "một đoạn", "bat_dau": "00:01,13", "ket_thuc": "00:02,48", "muc_do": "sai lệch nặng", "kenh_ten": None,
         "anh": "anh/T2.png", "mo_dau": "Giọng máy ở 00:01,13 – 00:02,48 (1,35 s), mức sai lệch nặng. Điểm vùng giọng máy (TCN) 0,97 ≥ ngưỡng 0,50"},
    ],
    "bang_moc": [["T1", "00:00,00", "00:06,00", 0.0, 6.0], ["T2", "00:01,13", "00:02,48", 1.13, 2.48]],
    "canh_bao": ["0,3 s đầu và cuối không được soi"],
    "ly_do_04": [],
    "trang_thai": "xong",
    "so_kenh_phan_tich": 1,
    "ngon_ngu_nguong": "vi",
}


def make_wav(path: str) -> str:
    """Tieng noi gia (hai hai am co dieu bien am tiet 4 Hz) + on nen; doan 1,13-2,48 s doi am sac (giong may gia lap). Co dinh (seed 9)."""
    rng = np.random.default_rng(9)
    t = np.arange(int(DUR * SR)) / SR
    f0 = 130 + 15 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(k * ph) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t) ** 2)
    m = (t >= SEG[0]) & (t < SEG[1])
    voice[m] = sum(np.sin(k * ph[m]) / k ** 0.6 for k in range(1, 20)) * 0.6
    y = 0.08 * voice / np.abs(voice).max() + rng.normal(0, 0.002, len(t))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, y.astype(np.float32), SR, subtype="PCM_16")
    return path


def model_record_like(thr_name: str) -> dict:
    """Ban ghi mo hinh GIONG Product.model_record (mo hinh cuoi, fold None) KHONG nap mo hinh: danh sach checkpoint theo pipeline.expected_ckpts
    (mot quy tac dat ten voi Product.load_models), sha1 tep THAT tren dia, so voi params.ckpt_sha1 bang pipeline.check_sha."""
    from v6.product import pipeline as PL
    j = json.load(open(os.path.join(RES, thr_name), encoding="utf-8"))
    prm = j["params"]
    bound, reg, heads, legacy = PL.resolve_params(prm, None, None, None, thr_name)
    loaded = []
    for role, path in PL.expected_ckpts(bound, reg, heads, None):
        kind = role.partition(":")[2] or "heads"
        name = os.path.basename(path)
        tag = re.match(r"(?:tcn|ft)_(.+?)_(?:final|f-?\d+)\.pt$", name)
        loaded.append({"role": f"{kind}:{tag.group(1)}" if tag else "heads", "file": name, "sha1": PL.sha1_file(path)})
    return {"thresholds": {"file": thr_name, "sha1": PL.sha1_file(os.path.join(RES, thr_name)), "name": j.get("name")}, "legacy_params": legacy,
            "resolved": {"bound": bound, "reg": reg, "heads": heads}, "fold": None, "thr_scope": "toan bo fold (mo hinh cuoi)", "ckpt": loaded,
            "sha_check": PL.check_sha(prm, loaded, thr_name), "calib": None}


class FakeProduct:
    """Thay Product (khong nap mo hinh): K1 gia lap + diem khung gia lap dung 2 thao tac cua vi du; mode='k1_04' / 'grey' / 'raise' cho cac ma thoat;
    'golden_real' = nguong that A; 'golden_F' = nguong that F (fuse1) + ban ghi mo hinh nhu Product.model_record."""
    stub = True

    def __init__(self, scenario: str = "golden"):
        self.scenario = scenario
        self.thr_name = "thresholds_v9F.json" if scenario == "golden_F" else "thresholds_v9A.json"
        thr = json.load(open(os.path.join(RES, self.thr_name), encoding="utf-8"))["thr"]
        # vi du hop dong can nhanh giong may doan: nguong GIA LAP ttsr 0,5 (KHONG co trong nguong that). Ca 'golden_real' / 'golden_F' dung nguong that:
        # dong giong may bi bo (nhanh tat) va ho so phai ghi 'giong may mot doan' la thao tac khong soi.
        self.thr = thr if scenario in ("golden_real", "golden_F") else {k: {**v, "ttsr": 0.5} for k, v in thr.items()}
        if scenario == "golden_F":
            self.mode, self.heads_tag = "fuse1", "h2"
        else:
            self.model_files = V9.model_files(None, "tcn", "tcn_reg", "h2")

    def analyse(self, path: str, lang: str | None = None, extended_k1: bool = True) -> dict:
        if self.scenario == "raise":
            raise RuntimeError("loi gia lap")
        y, _ = sf.read(path, dtype="float32")
        ck = {"decision": "canh_bao", "reasons": [], "warnings": ["0,3 s đầu và cuối không được soi"], "not_impl": [], "channels": [y],
              "measures": {"do_dai_s": DUR, "so_kenh": 1, "kenh": [{"ngon_ngu": {"vi": 0.93, "en": 0.04, "top": "vi", "p_top": 0.93}, "noi_chong_pct": 0.0}]}}
        if self.scenario == "k1_04":                                    # K1 ban 11/10: canh bao chuan + warn_detail (moc) + measures.ghi_chu_k1
            ck.update(decision="04", reasons=["04 · “băng hẹp” — dải tần thực < 6 kHz"], warnings=["có đoạn mất tín hiệu"],
                      warn_detail={"có đoạn mất tín hiệu": ["2.0–2.4 s"]})
            ck["measures"]["ghi_chu_k1"] = ["không đo được tiếng ồn (quá ít khoảng lặng)"]
            return {"k1": ck, "rows": [], "channels": [], "timing_s": {}, "label": "04"}
        F = int(DUR / 0.02)
        pb = np.full((F, 2), 0.02)
        pr = np.full((F, 6), 0.03)
        T = self.thr[lang or "vi"]
        if self.scenario == "grey":
            rows, head, hp = [], {"nen": 0.8, "giong": 0.1, "ttsf": 0.05}, {"nen": 0.5, "giong": 0.05, "ttsf": 0.02}
        else:
            pr[int(SEG[0] / 0.02):int(SEG[1] / 0.02), 5] = 0.97
            rows = [{"level2": "03_chen_giong_may", "kind": "segment", "start": SEG[0], "end": SEG[1], "p": 0.93, "nm": 0.97, "via": "tcn_vung"},
                    {"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.97, "nm": 0.97 - T["nen"]}]
            head, hp = {"nen": 0.97, "giong": 0.1, "ttsf": 0.05}, {"nen": 0.93, "giong": 0.05, "ttsf": 0.02}
        det = {"rows": rows, "pb": pb, "pr": pr, "head": head, "head_prob": hp, "T": T, "dropped": False, "dur": DUR, "kenh": 0, "y": y,
               "cond": {"measures": {"diem_dieu_kien": {"snr": -1.0, "nen": -1.0, "vang": -1.0}}}}
        out = {"k1": ck, "rows": rows, "channels": [det], "timing_s": {"k1": 0.0}, "label": "03"}
        if self.scenario == "golden_F":
            out["models"] = model_record_like(self.thr_name)
        return out


def project(ho: dict) -> dict:
    """Phan co dinh cua ho so de so voi EXPECTED (bo so do phu thuoc kho, thoi gian, bam)."""
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    return {"nhan": oc["nhan"],
            "thao_tac": [{"id": o["id"], "loai": o["loai"], "pham_vi": o["pham_vi"], "bat_dau": o["bat_dau"], "ket_thuc": o["ket_thuc"], "muc_do": o["muc_do"],
                          "kenh_ten": o["kenh_ten"], "anh": o["anh"], "mo_dau": next((m["mo_dau"] for m in EXPECTED["thao_tac"] if m["id"] == o["id"] and
                                                                                      o["giai_thich"].startswith(m["mo_dau"])), o["giai_thich"][:160])}
                         for o in oc["thao_tac"]],
            "bang_moc": [[m["thao_tac"], m["bat_dau"], m["ket_thuc"], m["bat_dau_s"], m["ket_thuc_s"]] for m in op_["bang_moc"]],
            "canh_bao": [w["noi_dung"] for w in op_["canh_bao"]], "ly_do_04": oc["ly_do_04"], "trang_thai": ho["trang_thai"],
            "so_kenh_phan_tich": ho["tep"]["so_kenh_phan_tich"], "ngon_ngu_nguong": ho["he_thong"].get("ngon_ngu_nguong", {}).get("ma")}


def _out(name: str) -> str:
    d = os.path.join(OUT, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------- tests
def test_golden_record(no_pdf: bool = False) -> None:
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    d = _out("golden")
    V9.setup_log(d, False)
    code = V9.run(wav, d, no_pdf=no_pdf, analyser=FakeProduct())
    assert code == 0, f"ma thoat {code}: {open(os.path.join(d, 'log.txt'), encoding='utf-8').read()[-2000:]}"
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    got = project(ho)
    json.dump(got, open(os.path.join(OUT, "golden_projection.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    assert got == EXPECTED, json.dumps(got, ensure_ascii=False, indent=1)
    assert ho["kiem_tra_dong_nhat"]["dat"] and not ho["kiem_tra_dong_nhat"]["vi_pham"]
    for o in ho["output_chinh"]["thao_tac"]:
        assert os.path.getsize(os.path.join(d, o["anh"])) > 10000, o["anh"]
        assert sum(1 for s in o["dau_hieu"] if s.get("loai") == "tin_hieu") >= 2, o["dau_hieu"]
    for f in ("bao_cao.html", "bang_moc.csv", "bang_so_do.csv", "anh/tong_quan.png"):
        assert os.path.exists(os.path.join(d, f)), f
    if no_pdf:
        return
    pdf = os.path.join(d, "bao_cao.pdf")
    assert open(pdf, "rb").read(5) == b"%PDF-", "PDF khong mo duoc"
    txt = report.pdf_text(pdf)
    assert txt is not None, "khong co pypdf de doc PDF (.tmp_v6/site_pdf)"
    flat = re.sub(r"\s+", "", txt)
    pv = ho["he_thong"]["phien_ban"]
    need = [ho["tep"]["sha256"], pv["nguong"]["tep"], pv["nguong"]["sha1"], pv["mo_hinh"][0]["ten"], pv["mo_hinh"][0]["sha1"], ho["tao_luc"]]
    miss = [x for x in need if re.sub(r"\s+", "", x) not in flat]
    assert not miss, f"PDF thieu {miss}"


def test_golden_real_thresholds_honest_coverage() -> None:
    """Cung tep, bo nguong THAT (khong ttsr / ttsf): khong duoc bao giong may; nhan 02 (chi khu on ca file); gioi han ke ten 2 thao tac khong soi."""
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    d = _out("golden_real")
    V9.setup_log(d, False)
    code = V9.run(wav, d, no_pdf=True, analyser=FakeProduct("golden_real"))
    assert code == 0, code
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    assert ho["kiem_tra_dong_nhat"]["dat"], ho["kiem_tra_dong_nhat"]
    assert ho["output_chinh"]["nhan"]["ma"] == "02" and [o["loai"] for o in ho["output_chinh"]["thao_tac"]] == ["khử ồn"]
    soi = ho["output_phu"]["diem_nguong"]["thao_tac_soi"]
    assert soi["khong_soi"] == ["giọng máy một đoạn", "giọng máy cả file"] and soi["so_duoc_soi"] == 6, soi
    assert any(x.startswith("Nhánh đang tắt") and "giọng máy một đoạn" in x for x in ho["output_phu"]["tuyen_bo_gioi_han"])


def test_golden_F_versions_and_coverage() -> None:
    """--op F (fuse1): khoi phien ban ghi tep nguong F + sha1, checkpoint FT-2 / TCN vung / dau ca tep + sha1 KHOP params.ckpt_sha1, che do fuse1,
    kiem sha1 'verified'; giong may khong soi (F khong co ttsr / ttsf); ho so qua validate + #bind + check_contract."""
    from v6.product import pipeline as PL
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    d = _out("golden_F")
    V9.setup_log(d, False)
    code = V9.run(wav, d, op="F", no_pdf=True, analyser=FakeProduct("golden_F"))
    assert code == 0, f"ma thoat {code}: {open(os.path.join(d, 'log.txt'), encoding='utf-8').read()[-2000:]}"
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    assert ho["kiem_tra_dong_nhat"]["dat"], ho["kiem_tra_dong_nhat"]
    prm = json.load(open(os.path.join(RES, "thresholds_v9F.json"), encoding="utf-8"))["params"]
    ng = ho["he_thong"]["phien_ban"]["nguong"]
    assert (ng["tep"], ng["diem_hoat_dong"], ng["che_do"]) == ("thresholds_v9F.json", "F", "fuse1"), ng
    assert ng["sha1"] == PL.sha1_file(os.path.join(RES, "thresholds_v9F.json")) and ng["kiem_sha1_voi_tep_nguong"] == "verified", ng
    assert ng["giai_ra"] == {"bound": "fuse1", "reg": "fuse1", "heads": "h2"} and ng["params"]["bound"] == "fuse1", ng
    mods = {m["ten"]: m["sha1"] for m in ho["he_thong"]["phien_ban"]["mo_hinh"]}
    want = ["ft_ft2_f-1.pt", "heads_final_h2.joblib"] + [os.path.basename(p) for p in PL.tcn_paths("tcn_reg", None)]
    assert all(mods.get(f) == prm["ckpt_sha1"][f] for f in want), {f: (mods.get(f), prm["ckpt_sha1"].get(f)) for f in want}
    assert ho["output_chinh"]["nhan"]["ma"] == "02" and [o["loai"] for o in ho["output_chinh"]["thao_tac"]] == ["khử ồn"]
    soi = ho["output_phu"]["diem_nguong"]["thao_tac_soi"]
    assert soi["khong_soi"] == ["giọng máy một đoạn", "giọng máy cả file"] and soi["so_duoc_soi"] == 6, soi


def test_exit_reject_dir() -> None:
    d = _out("reject")
    code = V9.run(os.path.join(OUT, "in"), d, no_pdf=True, analyser=FakeProduct())
    assert code == 2, code
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    assert ho["trang_thai"] == "tu_choi" and ho["output_chinh"]["nhan"]["ma"] is None


def test_exit_k1_04() -> None:
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    d = _out("k1_04")
    code = V9.run(wav, d, no_pdf=True, analyser=FakeProduct("k1_04"))
    assert code == 3, code
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    assert ho["output_chinh"]["nhan"]["ma"] == "04" and ho["output_chinh"]["ly_do_04"][0]["duong"] == "K1"
    assert "băng hẹp" in ho["output_chinh"]["ly_do_04"][0]["noi_dung"] and not ho["phieu"] and not ho["output_chinh"]["thao_tac"]
    mt = [w for w in ho["output_phu"]["canh_bao"] if w["noi_dung"] == "có đoạn mất tín hiệu"]          # duong K1 04 cung doc warn_detail / ghi_chu_k1
    assert mt and mt[0].get("chi_tiet") == "2.0–2.4 s", ho["output_phu"]["canh_bao"]
    assert ho["output_phu"]["diem_nguong"]["K1"]["ghi_chu_khong_thuoc_hop_dong"] == ["không đo được tiếng ồn (quá ít khoảng lặng)"]


def test_exit_grey_zone_04() -> None:
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    d = _out("grey")
    code = V9.run(wav, d, no_pdf=True, analyser=FakeProduct("grey"))
    assert code == 3, code
    ho = json.load(open(os.path.join(d, "ho_so.json"), encoding="utf-8"))
    assert [r["duong"] for r in ho["output_chinh"]["ly_do_04"]] == ["K3_vung_xam", "K5_khong_con_dong"]


def test_exit_error() -> None:
    d = _out("error")
    code = V9.main([os.path.join(OUT, "in", "khong_co.wav"), "-o", d, "--no-pdf"])
    assert code == 4, code
    try:
        V9.run(make_wav(os.path.join(OUT, "in", "golden_6s.wav")), _out("error2"), no_pdf=True, analyser=FakeProduct("raise"))
        raise AssertionError("loi phan tich phai noi len de main() tra 4")
    except RuntimeError:
        pass


# ---------------------------------------------------------------- S-9: cat dau-cuoi / doi am luong ca tep KHONG la chinh sua (bat bien nhan + moc)
S9_TRIM = (1.0, 0.5)              # cat 1,0 s dau va 0,5 s cuoi
S9_GAIN_DB = -6.0
S9_TOL_S = 0.5                    # sai moc cho phep (moc la mot vung, khong cam ket tung am tiet)


def s9_variants(wav: str, out_dir: str) -> dict:
    """Tao 2 bien the KHONG phai chinh sua: 'trim' (cat dau / cuoi) va 'gain' (-6 dB ca tep). Tra {ten: (duong dan, do lech moc s, cua so giu (a, b) theo tep goc)}."""
    y, sr = sf.read(wav, dtype="float32", always_2d=True)
    dur = len(y) / sr
    a, b = S9_TRIM
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(wav))[0]
    p_trim, p_gain = os.path.join(out_dir, f"{stem}__trim.wav"), os.path.join(out_dir, f"{stem}__gain.wav")
    sf.write(p_trim, y[int(a * sr):len(y) - int(b * sr)], sr, subtype="PCM_16")
    sf.write(p_gain, np.clip(y * 10 ** (S9_GAIN_DB / 20), -1, 1), sr, subtype="PCM_16")
    return {"orig": (wav, 0.0, (0.0, dur)), "trim": (p_trim, a, (a, dur - b)), "gain": (p_gain, 0.0, (0.0, dur))}


def s9_compare(ho_a: dict, ho_b: dict, shift: float = 0.0, keep: tuple | None = None, tol: float = S9_TOL_S) -> dict:
    """So ho so goc (a) voi bien the (b): moc b + shift = moc goc. Thao tac doan cua a nam ngoai cua so giu (+ 0,3 s mep) khong tinh (da bi cat that).
    Tra {nhan_goc, nhan_bien_the, cung_nhan, thieu, thua, dat}; thao tac ca file so theo loai."""
    def ops(ho: dict, sh: float) -> list:
        return [(o["nhan_cap2"], o["kenh"], None if o["pham_vi"] == "cả file" else (o["bat_dau_s"] + sh, o["ket_thuc_s"] + sh)) for o in ho["output_chinh"]["thao_tac"]]
    oa, ob = ops(ho_a, 0.0), ops(ho_b, shift)
    if keep is not None:
        lo, hi = keep[0] + 0.3, keep[1] - 0.3
        oa = [o for o in oa if o[2] is None or (o[2][0] >= lo and o[2][1] <= hi)]
    left = list(ob)
    miss = []
    for o in oa:
        hit = next((x for x in left if x[0] == o[0] and x[1] == o[1] and (o[2] is None) == (x[2] is None)
                    and (o[2] is None or (abs(o[2][0] - x[2][0]) <= tol and abs(o[2][1] - x[2][1]) <= tol))), None)
        if hit is None:
            miss.append(o)
        else:
            left.remove(hit)
    la, lb = ho_a["output_chinh"]["nhan"]["ma"], ho_b["output_chinh"]["nhan"]["ma"]
    return {"nhan_goc": la, "nhan_bien_the": lb, "cung_nhan": la == lb, "thieu": miss, "thua": left, "dat": la == lb and not miss and not left}


def s9_pick(limit: int) -> list:
    """Tep that tu nhien (natvi / naten): nua tep goc, nua tep da ghep chen (de co moc doan de so). Bo ban qua ung dung (ten co '__')."""
    import glob
    out = []
    for s in ("natvi", "naten"):
        for sub in ("nguyen_ban", "ghep_chen"):
            fs = sorted(f for f in glob.glob(os.path.join(BASE, "data_labels", "v6", s, sub, "*.wav")) if "__" not in os.path.basename(f))
            out += fs[::max(1, len(fs) // max(1, limit // 4))][:max(1, limit // 4)]
    return out[:limit]


def s9_run(limit: int, out_root: str, op: str = "A") -> int:
    """NANG (nap Product that, ~4 GB, ~1 phut / tep x 3): chay voiceaudit_v9.run (diem hoat dong op) tren goc / trim / gain, so sanh,
    ghi .tmp_v6/results/s9_trim_gain_invariance{_<op> neu khac A}.{json,md}."""
    rows = []
    sfx = "" if op == "A" else f"_{op}"
    for wav in s9_pick(limit):
        stem = os.path.splitext(os.path.basename(wav))[0]
        var = s9_variants(wav, os.path.join(out_root, stem, "in"))
        ho = {}
        for name, (p, _, _) in var.items():
            d = os.path.join(out_root, stem, name)
            os.makedirs(d, exist_ok=True)
            V9.setup_log(d, False)
            code = V9.run(p, d, op=op, no_pdf=True)
            f = os.path.join(d, "ho_so.json")
            ho[name] = json.load(open(f, encoding="utf-8")) if code in (0, 3) and os.path.exists(f) else None
        for name in ("trim", "gain"):
            r = {"tep": os.path.relpath(wav, BASE), "bien_the": name}
            if ho["orig"] is None or ho[name] is None:
                r.update(dat=False, loi="không có hồ sơ (mã thoát 2 / 4)")
            else:
                _, sh, keep = var[name]
                r.update(s9_compare(ho["orig"], ho[name], sh, keep))
            rows.append(r)
            print(json.dumps(r, ensure_ascii=False, default=str), flush=True)
    jp = os.path.join(RES, f"s9_trim_gain_invariance{sfx}.json")
    json.dump(rows, open(jp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    L = ["# S-9: cắt đầu–cuối / đổi âm lượng cả tệp không tính là chỉnh sửa", "",
         f"Cắt {S9_TRIM[0]} s đầu + {S9_TRIM[1]} s cuối; âm lượng {S9_GAIN_DB} dB; sai mốc cho phép {S9_TOL_S} s. Sinh bởi `python -m v6.product.golden --s9`.", "",
         "| biến thể | số tệp | cùng nhãn | đạt (nhãn + mốc) |", "|---|---|---|---|"]
    for name in ("trim", "gain"):
        rs = [r for r in rows if r["bien_the"] == name]
        L.append(f"| {name} | {len(rs)} | {sum(bool(r.get('cung_nhan')) for r in rs)} | {sum(bool(r.get('dat')) for r in rs)} |")
    L += ["", "Tệp không đạt:"] + [f"- {r['tep']} · {r['bien_the']} · {r.get('nhan_goc')}→{r.get('nhan_bien_the')} · thiếu {r.get('thieu')} · thừa {r.get('thua')}"
                                  f"{' · ' + r['loi'] if r.get('loi') else ''}" for r in rows if not r.get("dat")]
    open(os.path.join(RES, f"s9_trim_gain_invariance{sfx}.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def _ho_ops(lab: str, ops: list) -> dict:
    return {"output_chinh": {"nhan": {"ma": lab}, "thao_tac": [{"nhan_cap2": c, "kenh": 0, "pham_vi": "cả file" if a is None else "một đoạn",
                                                              "bat_dau_s": a, "ket_thuc_s": b} for c, a, b in ops]}}


def test_s9_compare() -> None:
    a = _ho_ops("03", [("02_nen_toan_cuc", None, None), ("03_ghep_giong_that", 5.0, 6.0), ("03_xoa_loi", 0.6, 1.2)])
    b = _ho_ops("03", [("02_nen_toan_cuc", None, None), ("03_ghep_giong_that", 4.1, 5.2)])           # trim 1 s: doan 5-6 s ve 4-5 s; xoa loi 0,6-1,2 s bi cat that
    r = s9_compare(a, b, 1.0, (1.0, 19.5))
    assert r["dat"] and r["cung_nhan"], r
    r2 = s9_compare(a, _ho_ops("02", [("02_nen_toan_cuc", None, None)]), 0.0, (0.0, 20.0))
    assert not r2["dat"] and not r2["cung_nhan"] and len(r2["thieu"]) == 2, r2
    r3 = s9_compare(a, _ho_ops("03", [("02_nen_toan_cuc", None, None), ("03_ghep_giong_that", 5.0, 6.0), ("03_xoa_loi", 0.6, 1.2),
                                      ("02_giong_cuc_bo", 9.0, 10.0)]), 0.0, (0.0, 20.0))
    assert not r3["dat"] and r3["thua"] == [("02_giong_cuc_bo", 0, (9.0, 10.0))], r3


def test_s9_variants() -> None:
    wav = make_wav(os.path.join(OUT, "in", "golden_6s.wav"))
    v = s9_variants(wav, os.path.join(OUT, "s9"))
    yt, _ = sf.read(v["trim"][0])
    yg, _ = sf.read(v["gain"][0])
    y0, _ = sf.read(wav)
    assert abs(len(yt) / SR - (DUR - sum(S9_TRIM))) < 0.01 and v["trim"][1] == S9_TRIM[0]
    assert len(yg) == len(y0) and abs(np.abs(yg).max() / np.abs(y0).max() - 10 ** (S9_GAIN_DB / 20)) < 0.01


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--s9" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 8
        op = sys.argv[sys.argv.index("--op") + 1] if "--op" in sys.argv else "A"
        return s9_run(lim, sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else "D:/tmp_va/wp3/s9", op)
    no_pdf = "--no-pdf" in sys.argv
    tests = [("test_golden_record", lambda: test_golden_record(no_pdf)), ("test_golden_real_thresholds_honest_coverage", test_golden_real_thresholds_honest_coverage),
             ("test_golden_F_versions_and_coverage", test_golden_F_versions_and_coverage),
             ("test_exit_reject_dir", test_exit_reject_dir), ("test_exit_k1_04", test_exit_k1_04),
             ("test_exit_grey_zone_04", test_exit_grey_zone_04), ("test_exit_error", test_exit_error), ("test_s9_compare", test_s9_compare),
             ("test_s9_variants", test_s9_variants)]
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
