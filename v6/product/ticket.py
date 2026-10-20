# -*- coding: utf-8 -*-
"""HO SO (ticket) cho san pham v9: dong + thao tac + dau hieu + giai thich + anh, dung luoc do 'voiceaudit.v6.ho_so/1' va luat #bind cua v6.io (da mo rong 8 thao tac, common9).
Nhan: ket qua Product.analyse (v6.product.pipeline). Chi BUOC TRINH BAY + ep LUAT HOP DONG (VoiceAudit_6Muc, muc Output va Scope):
  - thao tac ca file: bao khi xac suat DA HIEU CHINH >= 0,6 (det["head_prob"], giao dien I1); vung xam 0,4-0,6 -> 04 kem ly do (luat 5) hoac canh bao
    'co doan nghi 03 chua du chac' (luat 4). Chua co tep hieu chinh -> giu luat nguong diem cu VA ghi ro 'xac suat ca tep chua hieu chinh'.
  - luat (2) cong sang loc: CHUA XAY - ghi trung thuc 'chua co cong sang loc'; qua cong thay bang 'co dong thao tac hoac co xac suat ca tep trong vung xam'.
  - moi doan mot thao tac: xep theo BIEN VUOT NGUONG (logit diem - logit nguong cua chinh loai do); chong lan > 30 % giu dong vuot xa hon;
    <= 2 dong cuc bo moi khuc 16 s (buoc 14 s) + canh bao 'co the con cho sua khac'; doan ve 0,3 s dau / cuoi bi cat; xoa loi: vung <= 1 s chua moi noi.
  - canh bao chi dung 12 cum tu cua hop dong (cac chuoi K1 khac -> ghi chu K1); kenh 'trai / phai' khi hai kenh phan tich rieng;
    tuyen bo gioi han du: cac dong 'khong tu kiem duoc' + MOI Assumption + 'doan doi giong co the chua loi moi'.
  - khong nhan vo pham vi soi: thao tac khong co khoa nguong trong tep nguong (ops_off; vd giong may doan / ca file voi thresholds_v9A, v9F) bi bo,
    ke ten trong gioi han 'Nhanh dang tat' va o diem_nguong.thao_tac_soi (bao cao nhan 01 noi dung so thao tac duoc soi); anh tong quan de trong lan
    diem khung cua lop khong soi va ghi ten 'không soi'.
  - K1 (ban 11/10): canh bao da chuan + warn_detail -> 'chi_tiet', measures.ghi_chu_k1 -> ghi chu ngoai hop dong, o CA HAI duong ra (k1_warn_block);
    dong Constraint chi kiem mot phan (loai am thanh / kieu noi khi lop su kien am co chay) lay nguyen chu tu ck['not_impl'].
Moi thao tac >= 2 dau hieu co so (diem chuyen gia + so do tin hieu, kem phan vi / trung vi cua TEP THAT tu nhien), loi giai thich ghi dung chieu cao hon / thap hon.
Kiem khong can mo hinh: python -m v6.product.test_ticket · python -m v6.product.selftest · python -m v6.product.golden"""
from __future__ import annotations

import math
import os

import numpy as np

from v6.product import common9

common9.patch()                                   # PHAI truoc khi nap v6.io.schema / render / report
from v6.io import common as C  # noqa: E402

C.OP["03_chen_giong_may"] = ("giọng máy", "một đoạn", "sai lệch nặng")      # ten loai dung bang Scope (khong noi kieu con 'chen')
from v6.io.common import (CHUNK_S, EDGE_S, G_HI, G_LO, GLOBAL, L04, LOCAL, MAX_LOCAL_PER_CHUNK, NEN_CB_WITH_TC, OP, OVERLAP_MAX, SCOPE_FRAC, SEV,  # noqa: E402
                          chunk_starts, mmss, overlap, vn)
from v6.io.schema import (WARN_03, WARN_EDGE_C, WARN_MORE_C, WARN_VOCAB)  # noqa: E402
from v6.product import signals as S  # noqa: E402
from v6.product.refbank import rank  # noqa: E402

DT = 0.02
LANGS = ("vi", "en")
KENH_TEN = ("trái", "phải")
XOA_MAX_S = 1.0                                   # Scope: xoa loi = vung <= 1 s chua moi noi
EXPERT_OF = {"02_nen_cuc_bo": "den", "02_giong_cuc_bo": "voi", "03_ghep_giong_that": "ghep", "03_xoa_loi": "xoa", "03_chen_giong_may": "ttsr",
             "02_nen_toan_cuc": "nen", "02_giong_toan_cuc": "giong", "03_giong_may_toan_cuc": "ttsf"}
EXPERT_NAME = {"den": "điểm vùng khử ồn (TCN)", "voi": "điểm vùng đổi giọng (TCN)", "ghep": "điểm ranh giới ghép (TCN)", "xoa": "điểm mối nối xoá lời (TCN)",
               "ttsr": "điểm vùng giọng máy (TCN)", "spl": "điểm vùng đoạn ghép (TCN)", "nen": "điểm khử ồn cả file", "giong": "điểm đổi giọng cả file", "ttsf": "điểm giọng máy cả file"}
HEAD_LEVEL = {"nen": "02_nen_toan_cuc", "giong": "02_giong_toan_cuc", "ttsf": "03_giong_may_toan_cuc"}
LEVEL_HEAD = {v: k for k, v in HEAD_LEVEL.items()}
OP_KEYS = {"02_nen_toan_cuc": ("nen",), "02_nen_cuc_bo": ("den",), "02_giong_toan_cuc": ("giong",), "02_giong_cuc_bo": ("voi",),   # khoa nguong SINH ra moi
           "03_ghep_giong_that": ("ghep", "spl"), "03_xoa_loi": ("xoa",), "03_chen_giong_may": ("ttsr",), "03_giong_may_toan_cuc": ("ttsf",)}  # thao tac (pipeline.decode)
PR_COL = {"den": 2, "voi": 3, "spl": 4, "ttsr": 5}
REGION_KEY = {"02_nen_cuc_bo": "den", "02_giong_cuc_bo": "voi", "03_ghep_giong_that": "spl", "03_chen_giong_may": "ttsr"}
WHY = {"02_nen_toan_cuc": "Tiếng nền cả file có đặc tính của khử ồn áp đều (vector đặc trưng XLS-R lệch khỏi tệp thật).",
       "02_nen_cuc_bo": "Khử ồn chỉ một đoạn: đoạn này có đặc tính nền khác phần còn lại của file.",
       "02_giong_toan_cuc": "Tính chất giọng bị đổi đều trên cả file — lời và nhịp nói còn, nhưng giọng không còn như bản ghi gốc.",
       "02_giong_cuc_bo": "Tính chất giọng chỉ bị đổi trong đoạn này, phần còn lại của file thì không; đoạn đổi giọng có thể chứa lời mới — cần nghe lại đoạn được khoanh.",
       "03_ghep_giong_that": "Đoạn này mang dấu vết mối nối ở hai mép: phần lời trong đoạn có thể lấy từ chỗ khác.",
       "03_xoa_loi": "Mối nối này có dấu hiệu một đoạn lời đã bị cắt bỏ rồi nối lại.",
       "03_chen_giong_may": "Đoạn này mang dấu hiệu giọng tổng hợp (giọng máy): đặc trưng bên trong lệch so với phần còn lại của file.",
       "03_giong_may_toan_cuc": "Cả file mang dấu hiệu giọng tổng hợp (vector đặc trưng XLS-R gần với giọng máy hơn lời nói thật)."}
RULE_TEXT = {"03": "Luật (3): có ít nhất một thao tác nhãn 03 được báo ⟹ 03.",
             "02": "Luật (4): không có thao tác 03, có thao tác 02 được báo ⟹ 02.",
             "04": "Luật (5): qua bước thay cổng sàng lọc nhưng không báo được thao tác nào (xác suất cả tệp ở vùng 0,4–0,6) ⟹ 04 kèm lý do.",
             "01": "Không có dòng thao tác nào và không có xác suất cả tệp trong vùng xám ⟹ 01 (thay cho luật (2) vì chưa có cổng sàng lọc)."}
GATE_NOTE = ("Chưa có cổng sàng lọc (luật 2 của Scope: mô hình hỏi “tệp có dấu vết sửa nào không”). Thay tạm: qua cổng ⟺ có ít nhất một dòng thao tác "
             "sau luật ràng buộc, hoặc có xác suất cả tệp trong vùng xám 0,4–0,6; cột điểm là tỉ số điểm / ngưỡng lớn nhất, chỉ để tham khảo.")
NO_CALIB = "xác suất cả tệp chưa hiệu chỉnh"
BRANCH_OFF = ("nhánh tắt: tệp ngưỡng không đặt ngưỡng cho loại này — không báo thao tác cả file loại này, kể cả khi có xác suất đã hiệu chỉnh "
              "(bật lại nhánh là quyết định riêng, ghi trong tệp ngưỡng)")
K1_WARN_MAP = (("nén mạnh", "nen_manh", "nén mạnh"), ("tiếng ồn rất cao", "on_cao", "tiếng ồn cao"), ("tiếng ồn cao", "on_cao", "tiếng ồn cao"),
               ("cắt đỉnh", "cat_dinh", "cắt đỉnh"), ("có đoạn mất tín hiệu", "mat_tin_hieu", "có đoạn mất tín hiệu"), ("phòng vang", "phong_vang", "phòng vang"),
               ("kiểu nói ngoài phạm vi", "kieu_noi", "kiểu nói ngoài phạm vi"), ("ngôn ngữ khác", "ngon_ngu_khac", "ngôn ngữ khác"),
               ("nhiều hơn 1 giọng", "nhieu_nguoi_noi", "nhiều người nói"), ("nhiều người nói", "nhieu_nguoi_noi", "nhiều người nói"),
               ("nói chồng", "noi_chong", "nói chồng"), (WARN_EDGE_C, "vung_khong_soi", WARN_EDGE_C))


# ---------------------------------------------------------------- tien ich
def logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def kenh_ten(kenh: int, n_ch: int) -> str | None:
    """Hop dong: kenh 'trai / phai' CHI khi hai kenh duoc phan tich rieng (K1 giu ca hai, thu tu [trai, phai])."""
    return KENH_TEN[kenh] if n_ch > 1 and 0 <= kenh < 2 else None


def phrase9(s: dict) -> str:
    """'tiếng nền cả file −44,50 dB: thấp hơn 99 % tệp trong kho tham chiếu (trung vị −30,20)' - DUNG CHIEU (cao_hon = ti le kho <= gia tri)."""
    hi, lo = float(s["cao_hon_pct"]), float(s["thap_hon_pct"])
    rel = f"cao hơn {vn(hi, 0)} %" if hi >= lo else f"thấp hơn {vn(lo, 0)} %"
    u = f" {s['don_vi']}" if s.get("don_vi") else ""
    return f"{s['ten']} {vn(s['gia_tri'])}{u}: {rel} tệp trong kho tham chiếu (trung vị {vn(s['trung_vi_that'])})"


def sign_direction(s: dict) -> str:
    return "cao hơn" if float(s["cao_hon_pct"]) >= float(s["thap_hon_pct"]) else "thấp hơn"


# ---------------------------------------------------------------- bien vuot nguong
def row_expert(level2: str, row: dict) -> str:
    """Chuyen gia da sinh ra dong: doan ghep do TCN VUNG giai ma (via 'tcn_vung') la 'spl', con lai theo loai."""
    return "spl" if level2 == "03_ghep_giong_that" and row.get("via") == "tcn_vung" else EXPERT_OF[level2]


def expert_value(level2: str, row: dict, det: dict) -> float | None:
    """Gia tri diem cua chuyen gia ung voi dong: lon nhat trong doan +- 0,3 s (cuc bo) hoac diem dau ca file (diem tho)."""
    ex = row_expert(level2, row)
    if ex in HEAD_LEVEL:
        return det["head"].get(ex)
    a, b = int(max(0.0, row["start"] - 0.3) / DT), int((row["end"] + 0.3) / DT) + 1
    pb, pr = det["pb"], det["pr"]
    if ex in ("ghep", "xoa"):
        col = 0 if ex == "ghep" else 1
        return float(pb[a:b, col].max()) if len(pb[a:b]) else None
    col = PR_COL[ex]
    return float(pr[a:b, col].max()) if col < pr.shape[1] and len(pr[a:b]) else None


def margin_of(r: dict, det: dict, calibrated: bool) -> float:
    """Bien vuot nguong (logit diem - logit nguong) cua CHINH loai do: 'moi doan mot thao tac: thao tac co diem vuot nguong xa nhat'."""
    T = det["T"]
    if r["level2"] in GLOBAL:
        if calibrated and r.get("p_cal") is not None:
            return logit(r["p_cal"]) - logit(G_HI)
        k = LEVEL_HEAD.get(r["level2"])
        return logit(r.get("p", det["head"].get(k, 0.5))) - logit(T.get(k, 0.5))
    cands = []
    ex = row_expert(r["level2"], r)
    v = expert_value(r["level2"], r, det)
    if v is not None and ex in T:
        cands.append(logit(v) - logit(T[ex]))
    via = r.get("via", "")
    key = {"ranh_gioi_ghep": "ghep", "ranh_gioi_xoa": "xoa", "ranh_gioi+vung": "ghep"}.get(via) or (REGION_KEY.get(r["level2"]) if via == "tcn_vung" else None)
    if key in T and r.get("nm") is not None:
        cands.append(logit(r["nm"]) - logit(T[key]))
    return max(cands) if cands else logit(float(r.get("p", 0.5))) - logit(0.5)


# ---------------------------------------------------------------- thao tac ca file (luat xac suat >= 0,6)
def branch_off(det: dict) -> list:
    """Khoa dau ca tep dang TAT: tep nguong khong dat nguong cho khoa do (vd thresholds_v9A.json khong co 'ttsf')."""
    return [k for k in HEAD_LEVEL if k not in det["T"]]


def ops_off(det: dict) -> list:
    """Thao tac (ma cap 2, thu tu OP_KEYS) KHONG duoc soi o lan chay nay: khong khoa nguong nao sinh ra no trong tep nguong
    (vd thresholds_v9A / v9F khong co 'ttsr', 'spl', 'ttsf' -> giong may mot doan va giong may ca file deu khong soi)."""
    return [lvl for lvl, keys in OP_KEYS.items() if not any(k in det["T"] for k in keys)]


def op_name(lvl: str) -> str:
    return f"{OP[lvl][0]} {OP[lvl][1]}"


def whole_file(det: dict, hp: dict | None, rows: list) -> tuple:
    """(dong ca file, vung xam [(level2, p)], da_hieu_chinh_het). hp = det['head_prob'] (I1): None -> giu dong ca file cua pipeline (nguong diem);
    khoa nao khong co xac suat hieu chinh (None) thi khoa do giu luat nguong diem cu. Nhanh TAT (khong co nguong trong tep nguong) bo qua xac suat
    hieu chinh: hieu chinh khong duoc tu bat lai nhanh ma nghien cuu da tat (bat nhanh = quyet dinh rieng, ghi trong tep nguong)."""
    hp = hp or {}
    off = branch_off(det)
    out, grey, uncal = [], [], []
    for k, lvl in HEAD_LEVEL.items():
        if k in off:                                                   # dong cua nhanh tat da bi bo o to_ticket_rows_full (ops_off)
            continue
        p = hp.get(k)
        if p is None:
            out += [r for r in rows if r["level2"] == lvl]
            if k in det["head"]:
                uncal.append(k)
            continue
        p = float(p)
        if p >= G_HI:
            out.append({"level2": lvl, "kind": "global", "p": p, "p_cal": p, "raw": det["head"].get(k)})
        elif p >= G_LO:
            grey.append((lvl, p))
    return out, grey, not uncal and any(hp.get(k) is not None for k in HEAD_LEVEL if k not in off)


# ---------------------------------------------------------------- luat #bind tren dong cuc bo
def joint_of(a: float, b: float, det: dict | None) -> float:
    """Tam moi noi xoa loi: dinh diem moi noi xoa (pb[:, 1]) trong dong; khong co diem khung thi lay giua dong."""
    pb = None if det is None else det.get("pb")
    i0, i1 = int(a / DT), int(b / DT) + 1
    if pb is None or len(pb) == 0 or pb.shape[1] < 2 or i0 >= min(i1, len(pb)):
        return (a + b) / 2
    seg = np.asarray(pb[i0:i1, 1], dtype=float)
    top = np.flatnonzero(seg >= seg.max() - 1e-9)                       # dinh phang -> giua dai dinh
    return (i0 + float(top.mean()) + 0.5) * DT


def clamp_local(rows: list, dur: float, det: dict | None = None) -> tuple:
    """Cat 0,3 s dau / cuoi; xoa loi <= 1 s quanh MOI NOI (dinh pb[:, 1]); bo doan > 50 % tep. Tra (dong, ghi chu)."""
    out, notes = [], []
    for r in rows:
        if r["level2"] in LOCAL and r.get("start") is not None:
            a, b = float(r["start"]), float(r["end"])
            if r["level2"] == "03_xoa_loi" and b - a > XOA_MAX_S:
                m = min(max(joint_of(a, b, det), a + XOA_MAX_S / 2), b - XOA_MAX_S / 2)     # cua so 1 s chua moi noi, nam trong dong
                a, b = m - XOA_MAX_S / 2, m + XOA_MAX_S / 2
            a, b = max(a, EDGE_S), min(b, dur - EDGE_S)
            if b - a < 0.1:
                notes.append(f"bỏ {r['level2']} {r['start']:.2f}–{r['end']:.2f} s: nằm trong vùng 0,3 s không soi")
                continue
            if b - a > SCOPE_FRAC * dur:
                notes.append(f"bỏ {r['level2']} {a:.2f}–{b:.2f} s: dài hơn 50 % tệp")
                continue
            r = {**r, "start": a, "end": b}
        out.append(r)
    return out, notes


def _mk(x: dict) -> float:
    return float(x.get("margin", x.get("p", 0.0)))


def enforce(rows: list, dur: float) -> tuple:
    """Chong lan > 30 % giu dong VUOT NGUONG XA HON; <= 2 dong cuc bo moi khuc 16 s (buoc 14 s). Tra (dong, ghi chu, cham_gioi_han, dong_bo)."""
    notes, capped, dropped = [], False, []
    loc = [x for x in rows if x["level2"] in LOCAL]
    other = [x for x in rows if x["level2"] not in LOCAL]
    kept = []
    has_nen_tc = any(x["level2"] == "02_nen_toan_cuc" for x in rows)
    for x in sorted(loc, key=lambda x: -_mk(x)):
        if x["level2"] == "02_nen_cuc_bo" and has_nen_tc and float(x.get("p", 0)) < NEN_CB_WITH_TC:
            notes.append(f"bỏ nền cục bộ p={x.get('p', 0):.2f} < 0,9 khi đã có nền cả tệp")
            dropped.append((x, "nen_tc"))
            continue
        if any(overlap((x["start"], x["end"]), (y["start"], y["end"])) > OVERLAP_MAX for y in kept):
            notes.append(f"bỏ {x['level2']} {x['start']:.2f}–{x['end']:.2f} s: chồng > 30 % với dòng vượt ngưỡng xa hơn")
            dropped.append((x, "chong_lan"))
            continue
        kept.append(x)
    for s0 in chunk_starts(dur):
        inside = [x for x in kept if s0 <= (x["start"] + x["end"]) / 2 < s0 + CHUNK_S]
        if len(inside) >= MAX_LOCAL_PER_CHUNK:
            capped = True                         # cham gioi han: cac buoc giai ma truoc (detect.decode_file) cung da cat <= 2 dong / khuc ma khong bao
        for x in sorted(inside, key=lambda x: -_mk(x))[MAX_LOCAL_PER_CHUNK:]:
            kept.remove(x)
            notes.append(f"bỏ {x['level2']} {x['start']:.2f}–{x['end']:.2f} s: quá 2 dòng trong khúc {s0:.0f}–{s0 + CHUNK_S:.0f} s")
            dropped.append((x, "qua_2_dong"))
    return other + sorted(kept, key=lambda x: x["start"]), notes, capped, dropped


def to_ticket_rows(rows: list, dur: float, det: dict | None = None, hp: dict | None = None) -> tuple:
    """Dong pipeline -> dong ho so: doi ten loai, luat ca file (I1), bien vuot nguong, cat mep, ep luat.
    Tra (rows, ghi_chu, cham_gioi_han) nhu truoc; ban day du: to_ticket_rows_full."""
    r = to_ticket_rows_full(rows, dur, det, hp)
    return r["rows"], r["notes"], r["capped"]


def to_ticket_rows_full(rows: list, dur: float, det: dict | None = None, hp: dict | None = None) -> dict:
    mapped = []
    for r in rows:
        r = dict(r)
        if r["level2"] == "03_ghep_giong_that" and r.get("via") == "ranh_gioi_xoa":
            r["level2"] = "03_xoa_loi"
        if r["level2"] == "03_chen_giong_may" and r["kind"] == "global":
            r["level2"] = "03_giong_may_toan_cuc"
        r["p"] = float(r.get("nm", r.get("p", 0.0))) if r.get("via", "").startswith("ranh_gioi") else float(r.get("p", 0.0))
        mapped.append(r)
    grey, calibrated, n0 = [], False, []
    if det is not None:
        off = set(ops_off(det))                    # nhanh tat: khong bao (pipeline von khong giai ma; dong la do dau vao khac thi bo + ghi chu)
        n0 = [f"bỏ {r['level2']}: nhánh tắt (tệp ngưỡng không đặt ngưỡng)" for r in mapped if r["level2"] in off]
        mapped = [r for r in mapped if r["level2"] not in off]
        glob, grey, calibrated = whole_file(det, hp, mapped)
        mapped = glob + [r for r in mapped if r["level2"] not in GLOBAL]
        for r in mapped:
            r["margin"] = margin_of(r, det, calibrated)
    mapped, n1 = clamp_local(mapped, dur, det)
    n1 = n0 + n1
    kept, n2, capped, dropped = enforce(mapped, dur)
    sus03 = [f"{x['level2']} {x['start']:.2f}–{x['end']:.2f} s ({why})" for x, why in dropped if x["level2"].startswith("03")]
    sus03 += [f"{lvl} p={p:.2f} (vùng xám)" for lvl, p in grey if lvl.startswith("03")]
    return {"rows": kept, "notes": n1 + n2, "capped": capped, "grey": grey, "calibrated": calibrated, "sus03": sus03}


# ---------------------------------------------------------------- dau hieu (>= 2 so moi thao tac)
def _sign(ma: str, ten: str, value: float, unit: str, table: list, nguon: str, loai: str = "tin_hieu") -> dict | None:
    if value is None or not np.isfinite(value):
        return None
    rk = rank(table, float(value))
    hi, lo = rk["duoi"], rk["tren"]
    return {"ma": ma, "ten": ten, "gia_tri": round(float(value), 4), "don_vi": unit, "trung_vi_that": round(rk["trung_vi"], 4), "phan_vi_that": round(100 * hi, 1),
            "cao_hon_pct": round(100 * hi, 1), "thap_hon_pct": round(100 * lo, 1), "bat_thuong": bool(hi >= 0.95 or lo >= 0.95), "kho": nguon, "loai": loai}


def signs_for(level2: str, row: dict, det: dict, lang: str, refs: dict, sig: dict) -> list:
    """>= 2 dau hieu: diem chuyen gia + so do tin hieu (so voi tep that tu nhien cung ngon ngu); thieu thi bu bang so do ca file."""
    y = det["y"]
    out = []
    ex = row_expert(level2, row)
    v = expert_value(level2, row, det)
    tab = refs.get(lang, refs.get("vi", {})).get(ex)
    nguon = f"tệp thật tự nhiên ({lang})"
    if tab and v is not None:
        out.append(_sign(f"E_{ex}", EXPERT_NAME[ex], v, "", tab, f"điểm mô hình trên {nguon}", "mo_hinh"))
    st = sig.get(lang, sig.get("vi", {}))
    if level2 in LOCAL:
        a, b = float(row["start"]), float(row["end"])
        for ma, ten, val, unit, key in (("B_noisefloor_step", "bậc sàn nhiễu hai bên đoạn", S.noise_step(y, a, b), "dB", "noise_step"),
                                        ("I_noisefloor_d", "tiếng nền trong đoạn so với ngoài", S.inout_nf(y, a, b), "dB", "inout_nf"),
                                        ("I_tilt_d", "chênh độ nghiêng phổ trong / ngoài đoạn", S.inout_tilt(y, a, b), "dB", "inout_tilt")):
            if key in st:
                out.append(_sign(ma, ten, val, unit, st[key], nguon))
    out = [s for s in out if s]
    if sum(1 for s in out if s["loai"] == "tin_hieu") < 2:          # doan qua ngan / thao tac ca file: so do ca file
        for ma, ten, val, unit, key in (("G_noisefloor_db", "tiếng nền cả file", S.nf_db(y), "dB", "nf_file"), ("G_tilt_db", "độ nghiêng phổ cả file", S.tilt_db(y), "dB", "tilt_file")):
            if key in st:
                s = _sign(ma, ten, val, unit, st[key], nguon)
                if s and all(s["ma"] != t["ma"] for t in out):
                    out.append(s)
    return out


def explain_signs(signs: list) -> list:
    """Cac dau hieu lech nhieu nhat (>= 2, toi da 3), uu tien co >= 2 so do tin hieu (khong chi diem mo hinh)."""
    order = sorted(signs, key=lambda s: -max(s["cao_hon_pct"], s["thap_hon_pct"]))
    body = order[:3]
    if sum(1 for s in body if s.get("loai") != "mo_hinh") < 2:
        body += [s for s in order[3:] if s.get("loai") != "mo_hinh"][:2 - sum(1 for s in body if s.get("loai") != "mo_hinh")]
    return body


def cap(s: str) -> str:
    """Viet hoa chu dau, GIU nguyen phan con lai (str.capitalize() ha 'TCN' thanh 'tcn')."""
    return s[:1].upper() + s[1:]


def explain9(op: dict, signs: list, lang: str) -> str:
    loai, pv, md = OP[op["nhan_cap2"]]
    head = f"{cap(loai)} cả file, mức {md}. " if pv == "cả file" else f"{cap(loai)} ở {op['bat_dau']} – {op['ket_thuc']} ({vn(op['ket_thuc_s'] - op['bat_dau_s'])} s), mức {md}. "
    sc = op["diem"]
    if sc.get("p_hieu_chinh") is not None:
        why = f"Xác suất {loai} cả file (đã hiệu chỉnh) {vn(sc['p_hieu_chinh'])} ≥ {vn(G_HI)} (điểm thô {vn(sc['diem'])}). "
    else:
        name = cap(EXPERT_NAME[sc["chuyen_gia"]])
        cmp_ = f"{vn(sc['diem'])} ≥ ngưỡng {'điểm ' if pv == 'cả file' else ''}{vn(sc['nguong'])}" if sc["diem"] >= sc["nguong"] else \
            f"{vn(sc['diem'])} (ngưỡng {vn(sc['nguong'])}; dòng do bộ giải mã hợp nhất ranh giới + vùng đưa ra)"
        extra = f"{NO_CALIB}: chưa áp luật “xác suất ≥ 0,6”; " if pv == "cả file" else ""
        tail = "" if pv == "cả file" else "; chưa có bộ kiểm chứng độc lập đo lại"
        why = f"{name} {cmp_} ({extra}ngưỡng đặt ở tỉ lệ báo nhầm {vn(100 * sc['ti_le_bao_nham'], 1)} % trên tệp thật {lang}{tail}). "
    txt = "; ".join(phrase9(s) for s in explain_signs(signs))
    return head + why + WHY[op["nhan_cap2"]] + " Dấu hiệu lệch nhiều nhất: " + txt + "."


# ---------------------------------------------------------------- lap ho so
def _k3_block(det: dict, hp: dict | None) -> dict:
    T, out = det["T"], {}
    for c in GLOBAL:
        k = LEVEL_HEAD[c]
        raw = det["head"].get(k)
        if k not in T:
            pc = (hp or {}).get(k)
            out[c] = {"p": None if raw is None else round(float(raw), 4), "p_hieu_chinh": None, "nguong_dong": None, "vung_xam": None, "hieu_chinh": False,
                      "nhanh_tat": True, "p_hieu_chinh_bo_qua": None if pc is None else round(float(pc), 4), "ghi_chu": BRANCH_OFF}
        elif hp and hp.get(k) is not None:
            p = hp.get(k)
            out[c] = {"p": None if raw is None else round(float(raw), 4), "p_hieu_chinh": None if p is None else round(float(p), 4), "nguong_dong": G_HI,
                      "vung_xam": [G_LO, G_HI], "hieu_chinh": True}
        else:
            out[c] = {"p": None if raw is None else round(float(raw), 4), "p_hieu_chinh": None, "nguong_dong": None if k not in T else round(float(T[k]), 4),
                      "vung_xam": None, "hieu_chinh": False, "ghi_chu": NO_CALIB}
    return out


def _ratio(det: dict) -> float:
    T, r = det["T"], []
    r += [float(det["head"].get(k, 0)) / max(float(T[k]), 1e-6) for k in HEAD_LEVEL if k in T]
    r += [float(det["pb"][:, c].max()) / max(float(T[k]), 1e-6) for c, k in ((0, "ghep"), (1, "xoa")) if k in T and len(det["pb"])]
    r += [float(det["pr"][:, PR_COL[k]].max()) / max(float(T[k]), 1e-6) for k in ("den", "voi", "spl", "ttsr") if k in T and PR_COL[k] < det["pr"].shape[1] and len(det["pr"])]
    return max(r) if r else 0.0


def channel_label(rows: list, grey: list) -> str:
    """Luat Scope (3)(4)(5) cho mot kenh (luat (1) o K1; luat (2) chua co cong -> thay bang: khong dong va khong vung xam = 01)."""
    if any(x["level2"].startswith("03") for x in rows):
        return "03"
    if rows:
        return "02"
    return "04" if grey else "01"


def file_label(decisions: list) -> str:
    """Nhan tep tu cac kenh: 03 > 02 > 04 > 01 (04 o day chi la vung xam; K1 04 da dung truoc)."""
    labs = {d["label"] for d in decisions}
    return next((x for x in ("03", "02", "04") if x in labs), "01")


def grey_reason(lvl: str, p: float, pre: str = "") -> str:
    return (pre + f"Tín hiệu cả file lưng chừng: xác suất {OP[lvl][0]} cả file {vn(p)} nằm trong vùng {vn(G_LO)}–{vn(G_HI)} "
            "— chưa đủ chắc để kết luận.")


Q_PARAM = {"ghep": "f03", "xoa": "f03", "den": "freg", "voi": "freg", "spl": "fspl", "ttsr": "ftts", "nen": "fh", "giong": "fh", "ttsf": "ftts"}


def _op_and_row(ho: dict, x: dict, det: dict, kt: str | None, ctx: dict) -> None:
    """Mot dong phieu + mot thao tac (diem, dau hieu, giai thich) cho dong da qua luat."""
    oc, dur, kenh, T = ho["output_chinh"], ctx["dur"], det["kenh"], det["T"]
    stt = len(ho["phieu"]) + 1
    seg = x["level2"] in LOCAL
    ex = row_expert(x["level2"], x)
    cal = x.get("p_cal") is not None
    row = {"stt": stt, "kenh": kenh, "kenh_ten": kt, "nhan_cap2": x["level2"], "kieu": "segment" if seg else "global", "p": round(float(x["p"]), 4),
           "bien_vuot_nguong": round(float(x["margin"]), 4),
           "bat_dau_s": round(float(x["start"]), 3) if seg else None, "ket_thuc_s": round(float(x["end"]), 3) if seg else None, "thao_tac": None, "ly_do": None}
    loai, pv, md = OP[x["level2"]]
    a, b = (float(x["start"]), float(x["end"])) if seg else (0.0, dur)
    op_id = f"T{len(oc['thao_tac']) + 1}"
    val = expert_value(x["level2"], x, det)
    diem = {"chuyen_gia": ex, "diem": float(val if val is not None else x["p"]), "nguong": float(T.get(ex, 0.0)),        # ca hai deu la DIEM THO
            "ti_le_bao_nham": float(ctx["params"].get(Q_PARAM[ex], 0.0)), "bien_vuot_nguong": round(float(x["margin"]), 4)}
    if seg:
        diem["kiem_chung"] = {"che_do": "chua_co", "nhan": None, "p_dung_loai": None, "p_giu": None, "nguong_giu": None}
        row["kiem_chung"] = diem["kiem_chung"]
    else:
        diem["p_hieu_chinh"] = round(float(x["p_cal"]), 4) if cal else None
        diem["nguong_xac_suat"] = G_HI if cal else None                 # nguong cua xac suat DA HIEU CHINH (luat bao), khac nguong diem tho
        diem["luat"] = "xác suất đã hiệu chỉnh ≥ 0,6" if cal else f"điểm ≥ ngưỡng riêng ({NO_CALIB})"
    op = {"id": op_id, "kenh": kenh, "kenh_ten": kt, "nhan_cap2": x["level2"], "loai": loai, "pham_vi": pv, "muc_do": md, "bac": SEV[x["level2"]],
          "bat_dau_s": round(a, 3), "ket_thuc_s": round(b, 3), "bat_dau": mmss(a), "ket_thuc": mmss(b), "anh": f"anh/{op_id}.png", "dong_phieu": stt, "diem": diem}
    op["dau_hieu"] = signs_for(x["level2"], x, det, ctx["lang"], ctx["refs"], ctx["sig"])
    op["giai_thich"] = explain9(op, op["dau_hieu"], ctx["lang"])
    oc["thao_tac"].append(op)
    row["thao_tac"] = op_id
    ho["phieu"].append(row)


def _grey_rows(ho: dict, det: dict, kt: str | None, grey: list) -> list:
    """Luat (5): dong 04 'vung xam' trong phieu + ly do 04 cua kenh."""
    pre = f"Kênh {kt}: " if kt else ""
    reasons = []
    for lvl, p in grey:
        txt = grey_reason(lvl, p, pre)
        ho["phieu"].append({"stt": len(ho["phieu"]) + 1, "kenh": det["kenh"], "kenh_ten": kt, "nhan_cap2": L04, "kieu": "gray", "p": round(float(p), 4),
                            "bat_dau_s": None, "ket_thuc_s": None, "thao_tac": None, "ly_do": txt, "duong": "K3_vung_xam", "ha_tu": lvl})
        reasons.append({"duong": "K3_vung_xam", "noi_dung": txt})
    reasons.append({"duong": "K5_khong_con_dong", "noi_dung": pre + "Không có đoạn nào vượt ngưỡng riêng của loại đó (sau luật vùng soi và tối đa 2 đoạn mỗi khúc)."})
    return reasons


def _k7_block(det: dict, rows: list, notes: list) -> dict:
    T, lop = det["T"], {}
    for c in LOCAL:
        ex = EXPERT_OF[c]
        if ex in ("ghep", "xoa"):
            v = det["pb"][:, 0 if ex == "ghep" else 1].max() if len(det["pb"]) else None
        else:
            v = det["pr"][:, PR_COL[ex]].max() if PR_COL[ex] < det["pr"].shape[1] and len(det["pr"]) else None
        lop[c] = {"diem": None if v is None else round(float(v), 4), "nguong": None if ex not in T else round(float(T[ex]), 4), "bat": ex in T}
    return {"lop": lop, "de_xuat": [{"bat_dau_s": round(float(x["start"]), 3), "ket_thuc_s": round(float(x["end"]), 3), "lop": x["level2"], "p": round(float(x["p"]), 4),
                                     "bien_vuot_nguong": round(float(x["margin"]), 4), "kiem_chung": "chưa có bộ kiểm chứng độc lập (dòng do một chuyên gia chấm)",
                                     "ket_cuc": "giữ, chưa qua kiểm chứng"} for x in rows if x["level2"] in LOCAL], "ghi_chu_luat": notes}


def _tables(ho: dict) -> None:
    """Bang moc + bang so do tu danh sach thao tac."""
    op_ = ho["output_phu"]
    for o in ho["output_chinh"]["thao_tac"]:
        op_["bang_moc"].append({"thao_tac": o["id"], "kenh": o["kenh"], "kenh_ten": o["kenh_ten"], "loai": o["loai"], "pham_vi": o["pham_vi"], "muc_do": o["muc_do"],
                                "bat_dau": o["bat_dau"], "ket_thuc": o["ket_thuc"], "bat_dau_s": o["bat_dau_s"], "ket_thuc_s": o["ket_thuc_s"]})
        for s in o["dau_hieu"]:
            d = sign_direction(s)
            op_["bang_so_do"].append({"thao_tac": o["id"], "kenh": o["kenh"], "kenh_ten": o["kenh_ten"],
                                      **{k: v for k, v in s.items() if k in ("ma", "ten", "gia_tri", "don_vi", "trung_vi_that", "phan_vi_that", "bat_thuong", "kho", "loai")},
                                      "so_voi_kho": f"{d} {vn(s['cao_hon_pct'] if d == 'cao hơn' else s['thap_hon_pct'], 0)} %"})


def assemble9(ho: dict, res: dict, lang: str, refs: dict, sig: dict, thr_params: dict) -> list:
    """Dung phieu / thao tac / khoi diem va nguong cho moi kenh; tra danh sach 'decisions' (nhan kenh, dong, vung xam, nghi 03) cho nhan tep, canh bao, anh."""
    ctx = {"dur": float(ho["tep"]["do_dai_s"]), "lang": lang, "refs": refs, "sig": sig, "params": thr_params}
    n_ch = len(res["channels"])
    decisions = []
    for det in res["channels"]:
        kt = kenh_ten(det["kenh"], n_ch)
        hp = det.get("head_prob")
        tr = to_ticket_rows_full(det["rows"], ctx["dur"], det, hp)
        rows = tr["rows"]
        lab = channel_label(rows, tr["grey"])
        for x in sorted([x for x in rows if x["level2"] in GLOBAL], key=lambda x: -x["margin"]) + sorted([x for x in rows if x["level2"] in LOCAL], key=lambda x: x["start"]):
            _op_and_row(ho, x, det, kt, ctx)
        reasons = _grey_rows(ho, det, kt, tr["grey"]) if lab == "04" else []
        passed = bool(rows) or bool(tr["grey"])
        capped = bool((tr["capped"] or det.get("dropped")) and rows)
        ho["output_phu"]["diem_nguong"]["kenh"].append(
            {"kenh": det["kenh"], "kenh_ten": kt, "ngon_ngu_nguong": lang, "K3": _k3_block(det, hp),
             "K5": {"che_do": "chua_co_cong_sang_loc", "diem": round(_ratio(det), 4), "nguong": 1.0, "qua_cong": passed, "ghi_chu": GATE_NOTE},
             "K7": _k7_block(det, rows, tr["notes"]) if passed else {}, "K8": {"nhan_kenh": lab, "giai_thich": RULE_TEXT[lab], "nghi_03": tr["sus03"]},
             "che_do_khuc": "XLS-R khúc 32 s chồng 2 s + TCN", "khuc": [], "cham_gioi_han": capped})
        decisions.append({"kenh": det["kenh"], "kenh_ten": kt, "rows": rows, "label": lab, "cham_gioi_han": capped, "grey": tr["grey"],
                          "nghi_03": bool(tr["sus03"]), "sus03": tr["sus03"], "reasons": reasons, "calibrated": tr["calibrated"], "off": branch_off(det), "ops_off": ops_off(det),
                          "y": det["y"], "pb": det["pb"], "pr": det["pr"], "dur": ctx["dur"]})
    _tables(ho)
    return decisions


# ---------------------------------------------------------------- nhan tep, canh bao, gioi han
def warnings9(k1_warnings: list, decisions: list, analysed: bool = True, warn_detail: dict | None = None) -> tuple:
    """Canh bao CHI theo 12 cum tu hop dong (thu tu bang Output). Tra (canh_bao, ghi_chu_k1 khong thuoc hop dong).
    warn_detail = ck['warn_detail'] cua K1 (ban 11/10: canh bao da chuan hoa + chi tiet rieng, vd moc 'co doan mat tin hieu') -> 'chi_tiet'."""
    found, notes = {}, []
    for w in k1_warnings:
        hit = next(((ma, txt) for pre, ma, txt in K1_WARN_MAP if w.startswith(pre)), None)
        if hit is None:
            notes.append(w)
            continue
        ma, txt = hit
        found.setdefault(txt, {"ma": ma, "noi_dung": txt, "chi_tiet": []})
        if w != txt:
            found[txt]["chi_tiet"].append(w)
    for w, det in (warn_detail or {}).items():
        hit = next(((ma, txt) for pre, ma, txt in K1_WARN_MAP if str(w).startswith(pre)), None)
        if hit is None:
            notes += [f"{w}: {d}" for d in (det if isinstance(det, list) else [det])]
            continue
        ma, txt = hit
        found.setdefault(txt, {"ma": ma, "noi_dung": txt, "chi_tiet": []})
        found[txt]["chi_tiet"] += [str(d) for d in (det if isinstance(det, list) else [det]) if str(d) not in found[txt]["chi_tiet"]]
    if any(d.get("nghi_03") for d in decisions):
        det03 = [(f"kênh {d['kenh_ten']}: " if d.get("kenh_ten") else "") + s for d in decisions for s in d.get("sus03", [])]
        found[WARN_03] = {"ma": "nghi_03", "noi_dung": WARN_03, "chi_tiet": list(dict.fromkeys(det03))}
    if any(d["cham_gioi_han"] for d in decisions):
        found[WARN_MORE_C] = {"ma": "con_cho_sua", "noi_dung": WARN_MORE_C, "chi_tiet": ["tối đa 2 đoạn mỗi khúc 16 s, cộng các thao tác cả file"]}
    if analysed:
        found.setdefault(WARN_EDGE_C, {"ma": "vung_khong_soi", "noi_dung": WARN_EDGE_C, "chi_tiet": []})
    out = []
    for txt in WARN_VOCAB:
        if txt in found:
            w = found[txt]
            if not w["chi_tiet"]:
                w = {k: v for k, v in w.items() if k != "chi_tiet"}
            else:
                w = {**w, "chi_tiet": "; ".join(w["chi_tiet"])}
            out.append(w)
    return out, notes


def k1_warn_block(ck: dict, decisions: list, analysed: bool) -> tuple:
    """(canh_bao, ghi chu K1 ngoai hop dong) cho MOI duong ra (phan tich xong / K1 ra 04): K1 ban 11/10 tach chi tiet canh bao (warn_detail, vd moc
    'co doan mat tin hieu') va chu ngoai hop dong (measures.ghi_chu_k1) - ca hai duong deu phai doc, khong chi finalize9."""
    w, notes = warnings9(ck.get("warnings", []), decisions, analysed, ck.get("warn_detail"))
    return w, list(dict.fromkeys(notes + list((ck.get("measures") or {}).get("ghi_chu_k1") or [])))


K1_PARTIAL = ("Loại âm thanh:", "Kiểu nói:")        # dong not_impl cua K1 khi lop su kien am CO chay: chi kiem mot phan (vd chi chan nhac / bai hat)


def _lower1(s: str) -> str:
    return s[:1].lower() + s[1:]


def k1_unchecked(ck: dict | None, chans: list) -> list:
    """Cac dong Constraint ma LAN CHAY NAY khong tu kiem (tinh tu so do thuc co; dong kiem MOT PHAN lay nguyen chu tu ck['not_impl'] cua K1)."""
    if not ck:
        return []
    per = ck.get("measures", {}).get("kenh", []) or [{}]
    cond = [c.get("cond", {}).get("measures", {}).get("diem_dieu_kien", {}) for c in chans] or [{}]
    has = lambda key: any(key in m for m in per)                      # noqa: E731
    cnd = lambda key: any(c.get(key) is not None for c in cond)        # noqa: E731
    out = []
    if not cnd("snr"):                              # K1 chi con SNR mu (snr_db, tham khao); dong SNR do vaudit.cond trong pipeline
        out.append("tiếng nói so với tiếng ồn < 10 dB ⟹ 04 (chỉ cảnh báo bằng ước lượng mù)")
    if not cnd("nen") and not has("diem_nen"):
        out.append("âm không phải tiếng nói ≤ 10 % năng lượng")
    if not has("su_kien_am"):
        out += ["loại âm thanh (podcast, phát thanh, sách nói, nhạc)", "kiểu nói (hát, thì thầm, la hét)"]
    else:                                           # lop su kien am co chay: K1 tu ghi phan CHUA kiem duoc (podcast / phat thanh / sach noi; tre em)
        out += [_lower1(x) for x in ck.get("not_impl") or [] if str(x).startswith(K1_PARTIAL)]
    if not has("ngon_ngu"):
        out.append("ngôn ngữ (tiếng Việt / tiếng Anh)")
    if not has("noi_chong_pct"):
        out.append("nói chồng ≤ 10 %")
    if not cnd("vang") and not has("rt60_s"):
        out.append("phòng RT60 ≤ 1,0 s")
    return out


LIMITS_EXTRA = [
    "Đoạn đổi giọng có thể chứa lời mới do người khác nói — hệ thống không kiểm nội dung; người dùng cần nghe lại đoạn được khoanh.",
    "Giả định về tệp: không phải bản phát lại qua loa rồi ghi lại; micro không bị che hay cọ xát; nén khi ghi và khi gửi qua ứng dụng là biến đổi đồng đều "
    "trên cả tệp — trừ đoạn im lặng khi bộ mã bật DTX (thay im lặng bằng tiếng ồn giả); ứng dụng tự khử ồn (kể cả khử ồn AI) thì khử đều trên cả tệp.",
    "Giả định về người sửa: sửa cục bộ tổng cộng ≤ 50 % độ dài tệp; mỗi đoạn thay hoặc chèn dài ≥ 0,1 s, phần bị xoá ≥ 1 âm tiết; lời sau khi sửa mạch lạc, "
    "hợp ngữ pháp; dùng công cụ phổ thông, không cố tình phá dấu vết (không phủ nhiễu che, không cố ý nén lại nhiều lần để tẩy vết; chuyển tiếp qua ứng dụng "
    "vài lần vẫn trong phạm vi); giọng máy có thể nhái rất giống người thật.",
    "Giả định về người dùng: chấp nhận nhãn 04 và đưa tệp cho người nghe lại khi nhận 04; chấp nhận chờ tới 3 phút cho mỗi 5 phút âm thanh.",
    "Không chép lời, tóm tắt hay dịch; không nói số chỗ sửa chính xác, kiểu con (thay hay chèn), công cụ đã dùng, nội dung đúng hay sai, hay người nói là ai. "
    "Mốc thời gian là một vùng, không cam kết đúng từng âm tiết; 0,3 s đầu và cuối tệp không được soi.",
    "Cắt bỏ đầu–cuối tệp và đổi âm lượng cả tệp không tính là chỉnh sửa; thao tác khác ngoài 8 thao tác không có nhãn riêng và không được cam kết phát hiện.",
    "Khử ồn mặc định của ứng dụng gọi / họp (Zoom, Meet, Zalo, WebRTC) được xếp là điều kiện ghi bình thường (nhãn 01) khi học, không phải thao tác 02; "
    "khử ồn cả tệp bằng công cụ khác vẫn báo 02 · vô hại.",
    "Số đo độ chính xác lấy từ lời nói tự nhiên tiếng Việt và tiếng Anh trộn thao tác mô phỏng; thao tác che giấu kỹ (khớp âm sắc, nền âm, nén lại) và phòng vang "
    "(RT60 ≥ 0,3 s) làm tỉ lệ tìm thấy giảm mạnh.",
    "VoiceAudit v9 là bản THỬ NGHIỆM: chưa thử trên ghi âm thiết bị thật có chỉnh sửa thật.",
    "Chưa có cổng sàng lọc (luật 2) và chưa có bộ kiểm chứng độc lập: mỗi dòng một đoạn do một chuyên gia chấm, chưa được đo lại xác nhận.",
]
LIMIT_BANK = ("Kho tham chiếu của các con số phần trăm: {kho} — khác kho 160 tệp VIVOS mô phỏng ghi ở định nghĩa bài toán; con số phần trăm là mức bất thường "
              "so với kho này, không phải xác suất tệp bị sửa.")


OFF_HEAD = "Nhánh đang tắt (tệp ngưỡng không đặt ngưỡng)"
MV_TESTED = "Giọng máy mới thử trên một bộ tổng hợp."


def limitations9(ck: dict | None, chans: list, calibrated: bool | None, kho: str, sys_warn: list | None = None, off: list | None = None) -> list:
    """off = ma cap 2 cac thao tac KHONG soi (ops_off; nhan ca khoa dau cu 'nen' / 'giong' / 'ttsf' de tuong thich)."""
    out = list(C.LIMITS_REQUIRED) + LIMITS_EXTRA
    known = off is not None                       # None = chua phan tich (K1 04 / tu choi): khong noi gi ve nhanh
    off = [HEAD_LEVEL.get(k, k) for k in (off or [])]
    if calibrated is False:
        out.append(f"Thao tác cả file: {NO_CALIB} — báo theo ngưỡng điểm riêng của từng loại thay cho luật “xác suất ≥ 0,6”; chưa áp vùng xám 0,4–0,6 (luật 5).")
    if off:
        names = ", ".join(op_name(lvl) for lvl in off)
        out.append(f"{OFF_HEAD}: {names} — lần chạy này KHÔNG soi các thao tác đó ({len(OP_KEYS) - len(off)}/{len(OP_KEYS)} thao tác được soi); "
                   "nhãn 01 không nói gì về các thao tác này.")
    if known and any(lvl not in off for lvl in ("03_chen_giong_may", "03_giong_may_toan_cuc")):
        out.append(MV_TESTED)
    un = k1_unchecked(ck, chans)
    if un:
        out.append("Lần chạy này không tự kiểm các điều kiện: " + "; ".join(un) + ".")
    if kho:
        out.append(LIMIT_BANK.format(kho=kho))
    out += list(sys_warn or [])
    return out


def k1_measures9(ck: dict) -> list:
    """Bang so do K1 (kenh 'trai / phai' khi hai kenh)."""
    from voiceaudit_v6 import K1_NAMES
    import json
    out = []
    per = ck["measures"].get("kenh", [])
    for k, v in ck["measures"].items():
        if k == "ghi_chu_k1":                                         # chu ngoai hop dong -> diem_nguong.K1.ghi_chu_khong_thuoc_hop_dong
            continue
        if k == "kenh":
            for i, m in enumerate(v):
                kt = kenh_ten(i, len(per))
                for kk, vv in m.items():
                    out.append({"thao_tac": "K1", "kenh": i if kt else None, "kenh_ten": kt, "ma": f"{'trai' if kt == 'trái' else 'phai'}.{kk}" if kt else kk,
                                "ten": K1_NAMES.get(kk, kk), "gia_tri": vv if isinstance(vv, (int, float, str, type(None))) else json.dumps(vv, ensure_ascii=False)})
        else:
            out.append({"thao_tac": "K1", "kenh": None, "kenh_ten": None, "ma": k, "ten": K1_NAMES.get(k, k), "gia_tri": v})
    return out


def soi_block(off: list) -> dict:
    """Thao tac duoc soi / khong soi o lan chay nay (report dung cho cau nhan 01; check_contract doi chieu voi gioi han)."""
    on = [lvl for lvl in OP_KEYS if lvl not in off]
    return {"so_duoc_soi": len(on), "tong": len(OP_KEYS), "duoc_soi": [op_name(x) for x in on], "khong_soi": [op_name(x) for x in off],
            "ma_khong_soi": list(off), "can_cu": "khóa ngưỡng có trong tệp ngưỡng (pipeline chỉ giải mã loại có ngưỡng)"}


def finalize9(ho: dict, ck: dict, decisions: list, chans: list, kho: str) -> str:
    """Nhan tep + ly do 04 + canh bao + gioi han + bang so do K1. Tra ma nhan."""
    from v6.io.explain import ADVICE_04
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    lab = file_label(decisions)
    oc["nhan"] = {"ma": lab, "ten": C.L1_NAME[lab]}
    if lab == "04":
        oc["ly_do_04"] = [r for d in decisions for r in d["reasons"]]
        oc["de_nghi"] = ADVICE_04
    # luat (4) xet o MUC TEP: nhan tep 02 va CO kenh nao co doan nghi 03 (kenh do co the la 04 vung xam) -> canh bao; K8.nghi_03 giu danh sach tho moi kenh
    decisions = [{**d, "nghi_03": lab == "02" and bool(d["sus03"])} for d in decisions]
    w, notes = k1_warn_block(ck, decisions, True)
    op_["canh_bao"] = w
    op_["diem_nguong"]["K1"]["ghi_chu_khong_thuoc_hop_dong"] = notes
    cal = all(d["calibrated"] for d in decisions) if decisions else None              # mot khoa / mot kenh chua hieu chinh -> ghi ro trong gioi han
    off = [lvl for lvl in OP_KEYS if any(lvl in d.get("ops_off", []) for d in decisions)]
    op_["diem_nguong"]["thao_tac_soi"] = soi_block(off)
    op_["tuyen_bo_gioi_han"] = limitations9(ck, chans, cal, kho, ho["he_thong"].get("canh_bao_he_thong"), off=off)
    op_["bang_so_do"] += k1_measures9(ck)
    return lab


OFF_LANE = " · không soi"


def frame_matrix(d: dict) -> tuple:
    """Ma tran diem khung (T, len(CLS)) + truc thoi gian cho render (cot theo C.CLS). Lop cuc bo KHONG soi (d['ops_off']) de trong (0):
    anh khong duoc ve diem khung cua nhanh tat nhu the da soi (ten lan ghi 'không soi', xem lane_names)."""
    pb, pr = d["pb"], d["pr"]
    off = set(d.get("ops_off") or [])
    T = len(pb)
    P = np.zeros((T, len(C.CLS)))
    col = {"02_nen_cuc_bo": pr[:, 2] if pr.shape[1] > 2 else None, "02_giong_cuc_bo": pr[:, 3] if pr.shape[1] > 3 else None,
           "03_ghep_giong_that": pr[:, 4] if pr.shape[1] > 4 else pb[:, 0], "03_chen_giong_may": pr[:, 5] if pr.shape[1] > 5 else None, "03_xoa_loi": pb[:, 1]}
    for c, v in col.items():
        if v is not None and c in C.CLS and c not in off:
            P[:min(T, len(v)), C.CLS.index(c)] = np.asarray(v)[:T]
    P[:, 0] = 1 - P[:, 1:].max(1)
    return P, (np.arange(T) + 0.5) * DT


def lane_names(d: dict) -> list:
    """Ten lan K4 cua anh tong quan (theo C.CLS[1:]); lop khong soi -> them ' · không soi'."""
    from v6.io.render import LANE_NAMES
    off = set(d.get("ops_off") or [])
    return [n + (OFF_LANE if c in off else "") for c, n in zip(C.CLS[1:], LANE_NAMES)]


def render_all(ho: dict, decisions: list, out_dir: str) -> None:
    from v6.io.render import render_op, render_overview
    dur = float(ho["tep"]["do_dai_s"])
    for d in decisions:
        P, gt = frame_matrix(d)
        kt = d.get("kenh_ten")
        kl = f" · kênh {kt}" if kt else ""
        ops = [o for o in ho["output_chinh"]["thao_tac"] if o["kenh"] == d["kenh"]]
        for o in ops:
            render_op(d["y"], dur, o, P, gt, os.path.join(out_dir, o["anh"]), kl)
        name = f"anh/tong_quan{'_' + ('trai' if kt == 'trái' else 'phai') if kt else ''}.png"
        lab = ho["output_chinh"]["nhan"]["ma"]
        render_overview(d["y"], dur, ops, P, gt, os.path.join(out_dir, name), f"Tổng quan{kl} · nhãn tệp {lab} · {C.L1_NAME.get(lab, '')}",
                        lanes=lane_names(d))
        ho["output_phu"]["anh_tong_quan"].append(name)
