# -*- coding: utf-8 -*-
"""VoiceAudit v6 — bo chay THU NGHIEM mot tep: input -> day du output theo VoiceAudit.html (#io, #ev, #bind, #bounds).

  python voiceaudit_v6.py <tep ghi am> -o <thu muc ra> [--no-pdf] [--models models/v6] [--cache DIR] [--threads 4]
Ra: ho_so.json · anh/*.png · bao_cao.html/.pdf · bang_moc.csv · bang_so_do.csv · log.txt (thoi gian tung buoc).
Ma thoat: 0 xong (01/02/03) · 2 tu choi nhan · 3 nhan 04 · 4 loi.
K1 = vaudit.k1.check (04/tu choi kem ly do, canh bao, phan tich tung kenh khi hai kenh khac nhau) ->
K2 v6.io.extract -> K3..K8 v6.io.decide -> K9 v6.io.explain/render/report; schema.assert_valid TRUOC khi ghi ho so.
--stub <kich ban>: mo hinh GIA LAP + K2 'lite' + K1 khong nap ASR — CHI de tu kiem ma (v6.io.conformance)."""
import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # chay cuc bo: khong goi mang khi nap mo hinh
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from v6.io import common  # noqa: E402

common.setup_env()
from v6.io.common import (EXIT_04, EXIT_ERR, EXIT_OK, EXIT_REJECT, GLOBAL, L1_NAME, L04, LIMITS, LOCAL, OP,  # noqa: E402
                          SCHEMA_ID, SEV, WARN_EDGE, WARN_MORE, mmss)

LOG = logging.getLogger("voiceaudit.v6")
OUT_FILES = ("ho_so.json", "ho_so_khong_hop_le.json", "bao_cao.html", "bao_cao.pdf", "bang_moc.csv", "bang_so_do.csv")
K1_NAMES = {"dinh_dang": "định dạng", "ma_hoa": "mã hoá", "so_kenh": "số kênh", "tan_so_lay_mau": "tần số lấy mẫu (Hz)",
            "do_dai_s": "độ dài (s)", "toc_do_bit_kbps": "tốc độ bit (kbps)", "hai_kenh_giong_nhau": "hai kênh giống nhau",
            "bang_thong_hz": "dải tần thực (Hz)", "am_luong_dbfs": "âm lượng tiếng nói (dBFS)", "cat_dinh_pct": "cắt đỉnh (%)",
            "mat_tin_hieu": "đoạn mất tín hiệu (s)", "snr_db": "tiếng nói so với ồn, ước lượng (dB)",
            "tieng_noi_s": "thời lượng tiếng nói (s)", "so_nguoi_noi": "số người nói"}


class Steps:
    """Do thoi gian tung buoc, ghi vao log.txt."""

    def __init__(self):
        self.t = {}

    @contextlib.contextmanager
    def __call__(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.t[name] = round(time.time() - t0, 3)
            LOG.info("%-28s %8.2f s", name, self.t[name])


def _setup_log(out_dir: str, verbose: bool) -> None:
    LOG.handlers.clear()
    LOG.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out_dir, "log.txt"), mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(fh)
    if verbose:
        LOG.addHandler(logging.StreamHandler(sys.stderr))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def skeleton(path: str, stub: str | None) -> dict:
    return {"schema": SCHEMA_ID, "tao_luc": _dt.datetime.now().isoformat(timespec="seconds"),
            "he_thong": {"ten": "VoiceAudit v6 (thử nghiệm)", "mo_hinh": "gia_lap" if stub else "that",
                         "manifest_sha256": None, "che_do_khuc": "—", "canh_bao_he_thong": []},
            "tep": {"ten": os.path.basename(path.rstrip("\\/")), "sha256": None, "id": None, "kich_thuoc_byte": None,
                    "do_dai_s": None, "so_kenh_phan_tich": 0},
            "trang_thai": "xong",
            "output_chinh": {"nhan": {"ma": None, "ten": None}, "ly_do_04": [], "thao_tac": [], "de_nghi": None},
            "output_phu": {"canh_bao": [], "bang_moc": [], "diem_nguong": {"K1": {}, "kenh": []}, "bang_so_do": [],
                           "tuyen_bo_gioi_han": list(LIMITS), "anh_tong_quan": [], "xac_suat_khung": []},
            "phieu": [], "kiem_tra_dong_nhat": {"dat": False, "vi_pham": []}, "thoi_gian_s": {}}


def _k1_block(k1r: dict) -> dict:
    return {"quyet_dinh": k1r["decision"], "ly_do": k1r["reasons"], "canh_bao": k1r["warnings"],
            "so_do": {k: v for k, v in k1r["measures"].items() if k != "kenh"}, "kenh": k1r["measures"].get("kenh", []),
            "chua_tu_kiem": k1r.get("not_impl", [])}


def _k1_measures(k1r: dict) -> list:
    out = []
    for k, v in k1r["measures"].items():
        if k == "kenh":
            for i, m in enumerate(v):
                out += [{"thao_tac": "K1", "kenh": i, "ma": f"kenh{i + 1}.{kk}", "ten": K1_NAMES.get(kk, kk),
                         "gia_tri": vv if isinstance(vv, (int, float, str, type(None))) else json.dumps(vv, ensure_ascii=False)}
                        for kk, vv in m.items()]
        else:
            out.append({"thao_tac": "K1", "kenh": None, "ma": k, "ten": K1_NAMES.get(k, k), "gia_tri": v})
    return out


def _warnings(k1r: dict, decisions: list) -> list:
    out = [{"ma": "vung_khong_soi" if w == WARN_EDGE else "k1", "noi_dung": w} for w in k1r["warnings"]]
    if k1r["decision"] != "04" and not any(w["noi_dung"] == WARN_EDGE for w in out):
        out.append({"ma": "vung_khong_soi", "noi_dung": WARN_EDGE})
    if any(d["cham_gioi_han"] for d in decisions):
        out.append({"ma": "con_cho_sua", "noi_dung": WARN_MORE})
    return out


def _k8_text(d: dict) -> str:
    if not d["passed"]:
        return "Dưới ngưỡng cổng 01 ⟹ 01, dừng (phiếu rỗng)."
    return {"03": "Có dòng 03 ⟹ nhãn 03.", "02": "Không có dòng 03, có dòng 02 ⟹ nhãn 02.",
            "04": "Đã qua cổng 01 nhưng chỉ còn dòng 04 ⟹ nhãn 04 kèm lý do."}[d["label"]]


def _proposals(d: dict, models, row_op: dict) -> list:
    from v6.io.explain import verifier_text
    out = []
    for x in [x for x in d["rows"] if x.get("kind") == "segment"] + d["dropped"]:
        c = x.get("demoted_from", x["level2"])
        if x["level2"] in LOCAL:
            fate = f"giữ ⟹ {row_op.get(id(x), '?')}"
        elif x["level2"] == L04:
            fate = "bác, điểm ≥ 0,80 ⟹ dòng 04"
        else:
            fate = "bác, điểm yếu ⟹ bỏ (không vào phiếu)"
        out.append({"bat_dau_s": round(float(x["start"]), 3), "ket_thuc_s": round(float(x["end"]), 3), "lop": c,
                    "p": round(float(x["p"]), 4), "kiem_chung": verifier_text(x, models.keep_thr), "ket_cuc": fate})
    return out


def _channel_block(d: dict, models, row_op: dict) -> dict:
    k3 = {c: {"p": round(d["pg"][j], 4), "nguong_dong": common.G_HI, "vung_xam": [common.G_LO, common.G_HI]}
          for j, c in enumerate(GLOBAL)}
    k7 = {}
    if d["passed"]:
        k7 = {"lop": {c: {"diem": round(d["s2"][k], 4), "nguong": round(d["thr_eff"][k], 4)} for k, c in enumerate(LOCAL)},
              "de_xuat": _proposals(d, models, row_op), "ghi_chu_luat": d["ghi_chu"]}
    return {"kenh": d["kenh"], "K3": k3, "K5": {"diem": round(d["gate"], 4), "nguong": round(d["gate_thr"], 4),
                                                "qua_cong": d["passed"]},
            "K7": k7, "K8": {"nhan_kenh": d["label"], "giai_thich": _k8_text(d)}, "che_do_khuc": d["che_do"],
            "khuc": d["khuc"], "cham_gioi_han": d["cham_gioi_han"]}


def _make_op(x: dict, d: dict, models, op_id: str, stt: int, dur: float) -> dict:
    from v6.io.explain import explain_op, op_scores, sign_rows
    lab = x["level2"]
    loai, pv, md = OP[lab]
    local = lab in LOCAL
    a, b = (float(x["start"]), float(x["end"])) if local else (0.0, dur)
    op = {"id": op_id, "kenh": d["kenh"], "nhan_cap2": lab, "loai": loai, "pham_vi": pv, "muc_do": md, "bac": SEV[lab],
          "bat_dau_s": round(a, 3), "ket_thuc_s": round(b, 3), "bat_dau": mmss(a), "ket_thuc": mmss(b),
          "anh": f"anh/{op_id}.png", "dong_phieu": stt}
    vals = d["sf"].get(d["rec"], x["start"], x["end"]) if local else d["rec"]["global"]
    op["diem"] = op_scores(x, d, models)
    op["dau_hieu"] = sign_rows(models.bank, lab, vals, "segment" if local else "global")
    nen_tc = any(y["level2"] == "02_nen_toan_cuc" for y in d["rows"])
    op["giai_thich"] = explain_op(op, op["dau_hieu"], op["diem"], nen_tc)
    return op


def assemble(ho: dict, decisions: list, models, k1r: dict, dur: float) -> None:
    """Dung phieu + thao tac + bang phu tu cac quyet dinh (K9 chi trinh bay)."""
    from v6.io.explain import ADVICE_04, file_measures, reason_for_row
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    n_ch = len(decisions)
    for d in decisions:
        pre = f"Kênh {d['kenh'] + 1}: " if n_ch > 1 else ""
        row_op = {}
        order = [x for x in d["rows"] if x["level2"] in GLOBAL] + \
                sorted([x for x in d["rows"] if x["level2"] in LOCAL], key=lambda x: x["start"]) + \
                [x for x in d["rows"] if x["level2"] == L04]
        for x in order:
            stt = len(ho["phieu"]) + 1
            seg = x.get("start") is not None and x["level2"] not in GLOBAL
            row = {"stt": stt, "kenh": d["kenh"], "nhan_cap2": x["level2"], "kieu": x["kind"],
                   "p": None if x.get("p") is None else round(float(x["p"]), 4),
                   "bat_dau_s": round(float(x["start"]), 3) if seg else None,
                   "ket_thuc_s": round(float(x["end"]), 3) if seg else None, "thao_tac": None, "ly_do": None}
            if x["level2"] in OP:
                op = _make_op(x, d, models, f"T{len(oc['thao_tac']) + 1}", stt, dur)
                oc["thao_tac"].append(op)
                row["thao_tac"] = row_op[id(x)] = op["id"]
                if "kiem_chung" in op["diem"]:
                    row["kiem_chung"] = op["diem"]["kiem_chung"]
            else:
                rs = reason_for_row(x, d, pre)
                row["ly_do"], row["duong"] = rs["noi_dung"], rs["duong"]
                if x.get("demoted_from"):
                    row["ha_tu"] = x["demoted_from"]
                if ho["output_chinh"]["nhan"]["ma"] == "04":
                    oc["ly_do_04"].append(rs)
            ho["phieu"].append(row)
        op_["diem_nguong"]["kenh"].append(_channel_block(d, models, row_op))
        op_["bang_so_do"] += file_measures(models.bank, d["rec"], d["kenh"])
        op_["xac_suat_khung"].append({"kenh": d["kenh"], "t": [round(float(v), 3) for v in d["grid_t"]],
                                      "lop": common.CLS, "P": [[round(float(v), 3) for v in p] for p in d["P"]]})
    for o in oc["thao_tac"]:
        op_["bang_moc"].append({"thao_tac": o["id"], "kenh": o["kenh"], "loai": o["loai"], "pham_vi": o["pham_vi"],
                                "muc_do": o["muc_do"], "bat_dau": o["bat_dau"], "ket_thuc": o["ket_thuc"],
                                "bat_dau_s": o["bat_dau_s"], "ket_thuc_s": o["ket_thuc_s"]})
        op_["bang_so_do"] += [{"thao_tac": o["id"], "kenh": o["kenh"], **s} for s in o["dau_hieu"]]
    op_["bang_so_do"] += _k1_measures(k1r)
    op_["canh_bao"] = _warnings(k1r, decisions)
    if oc["nhan"]["ma"] == "04":
        oc["de_nghi"] = ADVICE_04


def _limits(ho: dict, k1r: dict | None, bank_stub: bool) -> None:
    lim = ho["output_phu"]["tuyen_bo_gioi_han"]
    if k1r and k1r.get("not_impl"):
        lim.append("K1 chưa tự kiểm các điều kiện: " + "; ".join(k1r["not_impl"]) + ".")
    lim += ho["he_thong"]["canh_bao_he_thong"]
    if bank_stub:
        lim.append("HỒ SƠ TỰ KIỂM: mô hình và kho tham chiếu là GIẢ LẬP — mọi con số không có ý nghĩa đo lường.")


def _render_all(ho: dict, decisions: list, out_dir: str, dur: float) -> None:
    from v6.io.render import render_op, render_overview
    n_ch = len(decisions)
    for d in decisions:
        ops = [o for o in ho["output_chinh"]["thao_tac"] if o["kenh"] == d["kenh"]]
        kl = f" · kênh {d['kenh'] + 1}" if n_ch > 1 else ""
        for o in ops:
            render_op(d["y"], dur, o, d["P"], d["grid_t"], os.path.join(out_dir, o["anh"]), kl)
        name = f"anh/tong_quan{'_k' + str(d['kenh'] + 1) if n_ch > 1 else ''}.png"
        lab = ho["output_chinh"]["nhan"]["ma"]
        render_overview(d["y"], dur, ops, d["P"], d["grid_t"], os.path.join(out_dir, name),
                        f"Tổng quan{kl} · nhãn tệp {lab} · {L1_NAME.get(lab, '')}")
        ho["output_phu"]["anh_tong_quan"].append(name)


def _finish(ho: dict, out_dir: str, steps: Steps, no_pdf: bool) -> int:
    """Kiem luoc do + #bind TRUOC khi ghi; ghi ho so, CSV, HTML, PDF. Tra ma thoat."""
    from v6.io import report, schema
    err = schema.validate(ho) or schema.check_bind(ho)
    ho["kiem_tra_dong_nhat"] = {"dat": not err, "vi_pham": err}
    if err:
        with open(os.path.join(out_dir, "ho_so_khong_hop_le.json"), "w", encoding="utf-8") as fh:
            json.dump(ho, fh, ensure_ascii=False, indent=1, default=str)
        LOG.error("ho so KHONG hop le, khong ghi ho_so.json: %s", err)
        return EXIT_ERR
    report.write_csvs(ho, out_dir)
    with steps("K9 báo cáo HTML"):
        hp = report.write_html(ho, out_dir)
    if not no_pdf:
        with steps("K9 PDF (Chrome)"):
            ok, note, _ = report.to_pdf(hp, os.path.join(out_dir, "bao_cao.pdf"))
        if not ok:
            LOG.error("PDF: %s", note)
    ho["thoi_gian_s"] = dict(steps.t)
    with open(os.path.join(out_dir, "ho_so.json"), "w", encoding="utf-8") as fh:
        json.dump(ho, fh, ensure_ascii=False, indent=1)
    if not no_pdf and not ok:
        return EXIT_ERR
    lab = ho["output_chinh"]["nhan"]["ma"]
    return EXIT_REJECT if ho["trang_thai"] == "tu_choi" else EXIT_04 if lab == "04" else EXIT_OK


def _reject(ho: dict, reason: str) -> None:
    ho["trang_thai"] = "tu_choi"
    ho["tu_choi_ly_do"] = [{"duong": "K1", "noi_dung": reason}]
    ho["output_phu"]["diem_nguong"]["K1"] = {"quyet_dinh": "tu_choi", "ly_do": [reason]}


def run(inp: str, out_dir: str, *, no_pdf: bool = False, models_dir: str | None = None, stub: str | None = None,
        cache_dir: str | None = None, threads: int = 4) -> int:
    os.makedirs(os.path.join(out_dir, "anh"), exist_ok=True)
    for f in OUT_FILES:
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    for f in os.listdir(os.path.join(out_dir, "anh")):
        if f.endswith(".png"):
            os.remove(os.path.join(out_dir, "anh", f))
    steps, t_all = Steps(), time.time()
    ho = skeleton(inp, stub)
    LOG.info("VoiceAudit v6 (thu nghiem) | vao %s | stub=%s", inp, stub)
    if os.path.isdir(inp):
        _reject(ho, "từ chối nhận · số tệp mỗi lần phải đúng 1 (nhận được một thư mục)")
        return _finish(ho, out_dir, steps, no_pdf)
    if not os.path.isfile(inp):
        raise FileNotFoundError(inp)
    ho["tep"].update(sha256=_sha256(inp), kich_thuoc_byte=os.path.getsize(inp))
    sha = ho["tep"]["sha256"]
    ho["tep"]["id"] = f"{common.ID_PREFIX}{sha[:16]}"
    import vaudit.k1 as k1
    from v6.io.stub import light_k1
    with steps("K1 kiểm đầu vào"), (light_k1() if stub else contextlib.nullcontext()):
        k1r = k1.check(inp)
    ho["output_phu"]["diem_nguong"]["K1"] = _k1_block(k1r)
    if k1r["decision"] == "tu_choi":
        _reject(ho, "; ".join(k1r["reasons"]))
        return _finish(ho, out_dir, steps, no_pdf)
    chans = k1r["channels"]
    dur = round(len(chans[0]) / 16000, 3) if chans else float(k1r["measures"].get("do_dai_s") or 0.0)
    ho["tep"].update(do_dai_s=dur, so_kenh_phan_tich=len(chans))
    if k1r["decision"] == "04":
        return _k1_fail(ho, k1r, out_dir, steps, no_pdf, stub, dur)
    return _analyse(ho, k1r, out_dir, steps, no_pdf, stub, dur, models_dir, cache_dir, threads, sha, t_all)


def _k1_fail(ho, k1r, out_dir, steps, no_pdf, stub, dur) -> int:
    from v6.io.explain import ADVICE_04, reasons_k1
    oc = ho["output_chinh"]
    oc["nhan"] = {"ma": "04", "ten": L1_NAME["04"]}
    oc["ly_do_04"] = reasons_k1(k1r)
    oc["de_nghi"] = ADVICE_04
    ho["output_phu"]["canh_bao"] = _warnings(k1r, [])
    ho["output_phu"]["bang_so_do"] = _k1_measures(k1r)
    _limits(ho, k1r, bool(stub))
    if k1r["channels"]:
        from v6.io.render import render_overview
        with steps("K9 ảnh"):
            for i, y in enumerate(k1r["channels"]):
                name = f"anh/tong_quan{'_k' + str(i + 1) if len(k1r['channels']) > 1 else ''}.png"
                render_overview(y, dur, [], None, None, os.path.join(out_dir, name), "Tổng quan · nhãn 04 · trượt K1")
                ho["output_phu"]["anh_tong_quan"].append(name)
    return _finish(ho, out_dir, steps, no_pdf)


def _analyse(ho, k1r, out_dir, steps, no_pdf, stub, dur, models_dir, cache_dir, threads, sha, t_all) -> int:
    with steps("nạp thư viện (torch, sklearn, librosa)"):
        from v6.io.decide import combine, decide_channel, load_models
        from v6.io.extract import extract_channel
    with steps("nạp mô hình"):
        models = load_models(models_dir, stub)
    ho["he_thong"]["manifest_sha256"] = models.manifest_sha
    decisions = []
    for c, y in enumerate(k1r["channels"]):
        with steps(f"K2 trích đặc trưng k{c + 1}"):
            k2 = extract_channel(y, f"{common.ID_PREFIX}{sha[:16]}_k{c}", lite=bool(stub), cache_dir=cache_dir,
                                 cache_key=f"{sha[:24]}_k{c}", threads=threads, log=LOG)
        with steps(f"K3–K8 quyết định k{c + 1}"):
            d = decide_channel(models, k2, c)
        d["y"] = y
        decisions.append(d)
        LOG.info("kenh %d: nhan %s | cong %.3f/%.3f | che do %s | K2 %s", c + 1, d["label"], d["gate"], d["gate_thr"],
                 d["che_do"], k2.timing)
    lab = combine(decisions)
    ho["output_chinh"]["nhan"] = {"ma": lab, "ten": L1_NAME[lab]}
    ho["he_thong"]["che_do_khuc"] = " · ".join(sorted({d["che_do"] for d in decisions}))
    ho["he_thong"]["canh_bao_he_thong"] = sorted({w for d in decisions for w in d["canh_bao_he_thong"]})
    with steps("K9 phiếu + giải thích"):
        assemble(ho, decisions, models, k1r, dur)
    _limits(ho, k1r, getattr(models.bank, "stub", False))
    with steps("K9 ảnh"):
        _render_all(ho, decisions, out_dir, dur)
    code = _finish(ho, out_dir, steps, no_pdf)
    total = time.time() - t_all
    LOG.info("TONG %.2f s cho %.2f s am thanh | RTF %.2f | ma thoat %d", total, dur, total / max(dur, 1e-6), code)
    return code


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="VoiceAudit v6 — kiểm định bản ghi âm (thử nghiệm, chỉ nghiên cứu)")
    ap.add_argument("input", help="một tệp ghi âm .wav .flac .mp3 .m4a .ogg")
    ap.add_argument("-o", "--out", required=True, help="thư mục ra")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--models", default=None, help=f"thư mục mô hình cuối (mặc định {common.MODELS_DIR})")
    ap.add_argument("--cache", default=None, help="thư mục cache K2 (tuỳ chọn, theo sha256 tệp)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "4")))
    ap.add_argument("--stub", default=None, help="kịch bản mô hình GIẢ LẬP (chỉ để tự kiểm mã): 03|02|01|04xam|04bac|dai")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    _setup_log(a.out, a.verbose)
    try:
        code = run(a.input, a.out, no_pdf=a.no_pdf, models_dir=a.models, stub=a.stub, cache_dir=a.cache, threads=a.threads)
    except Exception:
        LOG.error("LOI:\n%s", traceback.format_exc())
        code = EXIT_ERR
    LOG.info("ma thoat %d", code)
    print(f"ma thoat {code} · {os.path.join(a.out, 'ho_so.json')}")
    return code


if __name__ == "__main__":
    sys.exit(main())
