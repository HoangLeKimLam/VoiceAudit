# -*- coding: utf-8 -*-
"""Giai thich tieng Viet cho tung thao tac (moi dau hieu co so + thu hang so voi kho sach, ghi DUNG nguon kho)
va ly do cho tung duong ra 04 (K1 truot · K3 vung xam · qua cong 01 ma khong con dong · dong manh bi bac).
K9 chi trinh bay: khong them/bot quyet dinh nao."""
import numpy as np

from v6.io.common import G_HI, G_LO, GLOBAL, L04, LOCAL, OP, STRONG, mmss, vn

# dau hieu cho moi loai: (khoa, ten tieng Viet, don vi) — khoa trung ten cot trong global / SegFeats
SIGNS = {
    "02_nen_toan_cuc": [("G_noisefloor_db", "tiếng nền cả file", "dB"), ("G_lfcc0", "năng lượng LFCC-0 cả file", ""),
                        ("G_quiet_specvar", "độ biến thiên phổ trong khoảng lặng", "")],
    "02_giong_toan_cuc": [("P_spoof_med", "điểm chống giả mạo cả file", ""), ("G_dmfcc_var", "phương sai ΔMFCC cả file", ""),
                          ("IMG_modhi", "điều biến phổ nhanh cả file", "")],
    "02_nen_cuc_bo": [("in:I_noisefloor_d", "tiếng nền trong đoạn so với ngoài", "dB"),
                      ("in:I_dmfcc_var_ratio", "ΔMFCC trong/ngoài (log tỉ số)", ""),
                      ("edge:B_lfcc_dist", "bước nhảy LFCC ở mép", ""), ("edge:B_noisefloor_step", "bậc sàn nhiễu ở mép", "dB")],
    "02_giong_cuc_bo": [("in:P_spoof_in", "điểm chống giả mạo trong đoạn", ""),
                        ("in:I_dmfcc_var_ratio", "ΔMFCC trong/ngoài (log tỉ số)", ""),
                        ("in:I_f0_shift_abs", "độ lệch cao độ trong đoạn", "nửa cung"), ("edge:B_lfcc_dist", "bước nhảy LFCC ở mép", "")],
    "03_ghep_giong_that": [("edge:B_cflux_peak", "đỉnh flux phức ở mép", ""), ("edge:B_noisefloor_step", "bậc sàn nhiễu ở mép", "dB"),
                           ("edge:P_spk_dist_LR", "độ đổi giọng hai bên mép", ""),
                           ("in:I_dmfcc_var_ratio", "ΔMFCC trong/ngoài (log tỉ số)", "")],
    "03_chen_giong_may": [("in:I_dmfcc_var_ratio", "ΔMFCC trong/ngoài (log tỉ số)", ""), ("edge:B_lfcc_dist", "bước nhảy LFCC ở mép", ""),
                          ("edge:B_noisefloor_step", "bậc sàn nhiễu ở mép", "dB"), ("in:P_spoof_in", "điểm chống giả mạo trong đoạn", "")],
}
FALLBACK = {"global": [("G_dmfcc_var", "phương sai ΔMFCC cả file", ""), ("IMG_glcm_energy", "năng lượng kết cấu ảnh phổ cả file", ""),
                       ("G_noisefloor_db", "tiếng nền cả file", "dB"), ("P_spoof_max", "điểm chống giả mạo lớn nhất", "")],
            "segment": [("in:I_contrast_d", "tương phản phổ trong − ngoài", ""), ("in:I_lfcc_dist", "khoảng cách LFCC trong/ngoài", ""),
                        ("edge:IMG_ssim_LR", "độ giống ảnh phổ hai bên mép", ""), ("edge:P_spk_dist_LR", "độ đổi giọng hai bên mép", ""),
                        ("in:IMG_modhi_d", "điều biến phổ nhanh trong − ngoài", ""), ("edge:B_mel_dist", "khoảng cách phổ mel hai bên mép", "")]}
FILE_MEASURES = [("G_noisefloor_db", "tiếng nền cả file", "dB"), ("G_lfcc0", "năng lượng LFCC-0 cả file", ""),
                 ("G_dmfcc_var", "phương sai ΔMFCC cả file", ""), ("P_spoof_med", "điểm chống giả mạo cả file (trung vị)", ""),
                 ("G_quiet_specvar", "độ biến thiên phổ trong khoảng lặng", "")]
WHY_SEV = {
    "02_nen_toan_cuc": "Khử ồn áp đều lên cả file (thường do bấm nút lọc tiếng ồn) — chỉ chạm tiếng nền, không chạm giọng hay nội dung.",
    "02_nen_cuc_bo": "Khử ồn chỉ một đoạn: bấm nút lọc thì thao tác áp cả file, nên khử đúng một đoạn là dấu hiệu che vết.",
    "02_giong_toan_cuc": "Tính chất giọng bị đổi đều trên cả file — lời và nhịp nói còn, nhưng giọng không còn như bản ghi gốc.",
    "02_giong_cuc_bo": "Tính chất giọng chỉ bị đổi trong đoạn này, phần còn lại của file thì không.",
    "03_ghep_giong_that": "Đoạn này mang dấu vết mối nối ở mép: phần đặt vào có thể là lời thật lấy từ chỗ khác, không phải của lần nói này.",
    "03_chen_giong_may": "Đoạn này mang dấu hiệu giọng tổng hợp chèn vào: động học phổ bên trong lệch so với phần còn lại của file.",
}
ADVICE_04 = "Đề nghị đưa tệp cho người nghe lại (chuyên viên) — hệ thống không đủ cơ sở để kết luận."


def _pct(x: float) -> str:
    return vn(100 * x, 0) + " %"


def sign_rows(bank, lab: str, vals: dict, kind: str) -> list:
    """Dau hieu co so cho mot thao tac: gia tri + vi tri so voi kho sach. Luon co >= 2 dau hieu neu du lieu cho phep."""
    out, seen = [], set()
    for key, name, unit in SIGNS[lab] + FALLBACK[kind]:
        if key in seen or (len(out) >= 2 and (key, name, unit) not in SIGNS[lab]):
            continue
        seen.add(key)
        v = vals.get(key)
        rk = bank.rank(key, None if v is None else float(v))
        if rk is None:
            continue
        hi, lo = rk["duoi"], rk["tren"]
        out.append({"ma": key, "ten": name, "gia_tri": round(float(v), 4), "don_vi": unit,
                    "trung_vi_that": round(rk["trung_vi"], 4), "phan_vi_that": round(100 * hi, 1),
                    "cao_hon_pct": round(100 * hi, 1), "thap_hon_pct": round(100 * lo, 1),
                    "bat_thuong": bool(hi >= 0.95 or lo >= 0.95), "kho": rk["nguon"], "n_kho": rk["n"]})
    return out


def phrase(s: dict, with_src: bool = True) -> str:
    """'tiếng nền cả file −44,50 dB — thấp hơn 99 % so với 160 tệp mô phỏng sạch (trung vị −20,30; bất thường)'."""
    hi, lo = s["cao_hon_pct"] / 100, s["thap_hon_pct"] / 100
    if hi >= lo:
        rel = "cao hơn toàn bộ" if hi >= 1.0 else f"cao hơn {_pct(hi)}"
    else:
        rel = "thấp hơn toàn bộ" if lo >= 1.0 else f"thấp hơn {_pct(lo)}"
    tag = "bất thường" if s["bat_thuong"] else ("hơi lệch" if max(hi, lo) >= 0.90 else "trong khoảng thường gặp")
    u = f" {s['don_vi']}" if s["don_vi"] else ""
    src = f" so với {s['kho']}" if with_src else ""
    return f"{s['ten']} {vn(s['gia_tri'])}{u} — {rel}{src} (trung vị {vn(s['trung_vi_that'])}; {tag})"


def op_scores(row: dict, dec: dict, models) -> dict:
    """Diem va nguong lien quan toi mot dong (lay tu quyet dinh, khong tinh lai)."""
    lab = row["level2"]
    if lab in GLOBAL:
        return {"p_toan_cuc": round(float(row["p"]), 4), "nguong_dong": G_HI, "vung_xam": [G_LO, G_HI],
                "diem_cong": round(dec["gate"], 4), "nguong_cong": round(dec["gate_thr"], 4)}
    k = LOCAL.index(lab)
    j = k + 1
    calib = row.get("pm") is not None
    return {"p_tang2": round(float(row["p"]), 4), "nguong_tang2": round(float(dec["thr_eff"][k]), 4),
            "diem_cong": round(dec["gate"], 4), "nguong_cong": round(dec["gate_thr"], 4),
            "kiem_chung": {"che_do": "hieu_chuan" if calib else "khop_nhan", "nhan": row.get("verified_as"),
                           "p_dung_loai": None if row.get("pv") is None else round(float(row["pv"]), 4),
                           "p_giu": None if not calib else round(float(row["pm"]), 4),
                           "nguong_giu": None if not calib else round(float(models.keep_thr.get(j, np.nan)), 4)}}


def _vc_name(vc: str | None) -> str:
    return "không sửa" if vc in (None, "none") else OP.get(vc, (vc,))[0]


def explain_op(op: dict, signs: list, sc: dict, nen_tc: bool) -> str:
    loai, pv, md = OP[op["nhan_cap2"]]
    if pv == "cả file":
        head = f"{loai.capitalize()} cả file, mức {md}. "
        why = (f"Xác suất nhánh toàn cục {vn(sc['p_toan_cuc'])} ≥ ngưỡng {vn(G_HI)}; "
               f"cổng 01 {vn(sc['diem_cong'])} ≥ {vn(sc['nguong_cong'])}. ")
    else:
        head = (f"{loai.capitalize()} ở {op['bat_dau']} – {op['ket_thuc']} "
                f"({vn(op['ket_thuc_s'] - op['bat_dau_s'])} s), mức {md}. ")
        extra = " (đã có khử ồn cả file nên ngưỡng nâng lên 0,90)" if op["nhan_cap2"] == "02_nen_cuc_bo" and nen_tc else ""
        kc = sc["kiem_chung"]
        if kc["che_do"] == "hieu_chuan":
            ver = (f"bộ kiểm chứng độc lập đo lại đúng đoạn này: xác suất đúng loại {vn(kc['p_dung_loai'])}, "
                   f"xác suất giữ sau hiệu chuẩn {vn(kc['p_giu'])} ≥ ngưỡng {vn(kc['nguong_giu'])} "
                   f"(lớp cao nhất của bộ kiểm chứng: “{_vc_name(kc['nhan'])}”). ")
        else:
            ver = f"bộ kiểm chứng độc lập đo lại đúng đoạn này và trả “{_vc_name(kc['nhan'])}” — khớp. "
        why = f"Điểm lớp {vn(sc['p_tang2'])} ≥ ngưỡng {vn(sc['nguong_tang2'])}{extra}; " + ver
    body = sorted(signs, key=lambda s: -max(s["cao_hon_pct"], s["thap_hon_pct"]))[:3]
    srcs = sorted({s["kho"] for s in body})
    return (head + why + WHY_SEV[op["nhan_cap2"]] + f" Dấu hiệu, so với {' / '.join(srcs) or 'kho sạch'}: " +
            "; ".join(phrase(s, with_src=False) for s in body) + ".")


def reason_for_row(x: dict, dec: dict, prefix: str = "") -> dict | None:
    """Ly do cua MOT dong 04 (K3 vung xam · K7 bi bac · K5 qua cong ma khong con dong)."""
    if x["level2"] != L04:
        return None
    if x["kind"] == "gray":
        c, p = x["why"].split(" p=")
        return {"duong": "K3_vung_xam", "noi_dung": prefix +
                f"Tín hiệu cả file lưng chừng: xác suất {OP[c][0]} cả file {vn(float(p))} nằm trong vùng xám "
                f"{vn(G_LO)}–{vn(G_HI)} — chưa đủ chắc để kết luận."}
    if x["kind"] == "segment":
        return {"duong": "K7_bi_bac", "noi_dung": prefix +
                f"Phát hiện {OP[x['demoted_from']][0]} mạnh ở {mmss(x['start'])} – {mmss(x['end'])} "
                f"(điểm {vn(x['p'])} ≥ {vn(STRONG)}) nhưng bộ kiểm chứng độc lập đo lại đúng đoạn này không xác nhận "
                f"(lớp cao nhất: “{_vc_name(x.get('verified_as'))}”).",
                "bat_dau_s": round(float(x["start"]), 3), "ket_thuc_s": round(float(x["end"]), 3)}
    if x["kind"] == "gate":
        return {"duong": "K5_khong_con_dong", "noi_dung": prefix +
                f"Qua cổng 01 (điểm {vn(dec['gate'])} ≥ ngưỡng {vn(dec['gate_thr'])}: có dấu hiệu sửa) "
                f"nhưng không còn thao tác nào qua được kiểm chứng."}
    return {"duong": "K5_khong_con_dong", "noi_dung": prefix + f"Dòng 04 không rõ đường ({x.get('why', '')})."}


def verifier_text(x: dict, keep_thr: dict) -> str:
    """Tom tat ket qua bo kiem chung cho bang 'diem va nguong'."""
    s = f"lớp cao nhất “{_vc_name(x.get('verified_as'))}”"
    if x.get("pv") is not None:
        s += f" · p đúng loại {vn(x['pv'])}"
    if x.get("pm") is not None:
        c = x.get("demoted_from", x["level2"])
        s += f" · p giữ {vn(x['pm'])} / ngưỡng {vn(keep_thr.get(LOCAL.index(c) + 1, float('nan')))}"
    return s


def reasons_k1(k1: dict) -> list:
    return [{"duong": "K1", "noi_dung": f"Tệp không đạt điều kiện đầu vào: {x}"} for x in k1.get("reasons", [])]


def file_measures(bank, rec: dict, kenh: str) -> list:
    """So do ca tep (K2) cho bang so do — co ca khi nhan 01/04."""
    out = []
    for key, name, unit in FILE_MEASURES:
        v = rec["global"].get(key)
        rk = bank.rank(key, None if v is None else float(v))
        if rk is None:
            continue
        out.append({"thao_tac": "K2", "kenh": kenh, "ma": key, "ten": name, "gia_tri": round(float(v), 4), "don_vi": unit,
                    "trung_vi_that": round(rk["trung_vi"], 4), "phan_vi_that": round(100 * rk["duoi"], 1),
                    "bat_thuong": bool(rk["duoi"] >= 0.95 or rk["tren"] >= 0.95), "kho": rk["nguon"]})
    return out
