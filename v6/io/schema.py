# -*- coding: utf-8 -*-
"""Luoc do JSON 'voiceaudit.v6.ho_so/1' + bo kiem (khong can thu vien jsonschema) + LUAT #bind / Scope.

validate(ho)     -> danh sach loi khuon (kieu, truong bat buoc, enum, mau chuoi)
check_bind(ho)   -> danh sach vi pham rang buoc gan ket (#bind, Scope) — vi pham la loi LOGIC
assert_valid(ho) -> nem ValueError neu co loi; goi TRUOC khi ghi ho so
check_contract(ho) -> vi pham muc Output / Scope cua hop dong cho ho so v9 (12 canh bao, kenh trai/phai, phien ban, luat 0,6, chieu dau hieu, gioi han)
Dung: python -m v6.io.schema --dump  (in luoc do)  |  python -m v6.io.schema <ho_so.json> [--contract]  (kiem mot tep)
  python -m v6.io.schema --scan .tmp_v6/e2e9 [--newer v6/product/ticket.py]  (kiem moi ho so v9 ben duoi; ho so v9 tu nap hang so v9)"""
import os
import argparse
import json
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.io.common import (CHUNK_S, EDGE_S, GLOBAL, L1_NAME, L04, LIMITS_REQUIRED, LOCAL, MAX_LOCAL_PER_CHUNK,
                          NEN_CB_WITH_TC, OP, OVERLAP_MAX, SCHEMA_ID, SCOPE_FRAC, SEV, STRONG, WARN_EDGE, WARN_MORE,
                          chunk_starts, overlap, vn)

TOL = 1e-3
# 12 canh bao cua hop dong (VoiceAudit_6Muc, Output phu), DUNG chu va thu tu; ho so v9 chi duoc dung cac cum nay (chi tiet do -> truong 'chi_tiet')
WARN_EDGE_C = WARN_EDGE                                  # "0,3 s đầu và cuối không được soi"
WARN_MORE_C = "có thể còn chỗ sửa khác"
WARN_03 = "có đoạn nghi 03 chưa đủ chắc"
WARN_VOCAB = ["nén mạnh", "tiếng ồn cao", "cắt đỉnh", "có đoạn mất tín hiệu", "phòng vang", "kiểu nói ngoài phạm vi", "ngôn ngữ khác",
              "nhiều người nói", "nói chồng", WARN_03, WARN_MORE_C, WARN_EDGE_C]
CONTRACT_TYPES = ["khử ồn", "đổi giọng", "ghép lời thật", "xoá lời", "giọng máy"]
EXTRA_LEVELS = ["03_xoa_loi", "03_giong_may_toan_cuc"]  # common9 them; liet ke san de enum dung du thu tu nap module
# moc chu bat buoc trong tuyen bo gioi han (hop dong: '**' - dieu kien khong tu kiem duoc, MOI Assumption, doi giong co the chua loi moi)
LIMIT_MARKERS = ["giám định tư pháp", "không phải chứng nhận", "micro", "tốc độ phát", "lời mới", "phát lại qua loa", "cọ xát", "DTX",
                 "kể cả khử ồn AI", "≤ 50 %", "≥ 0,1 s", "≥ 1 âm tiết", "mạch lạc", "phá dấu vết", "nén lại nhiều lần", "nhái rất giống",
                 "chấp nhận nhãn 04", "3 phút"]
FORBIDDEN_IN_EXPLAIN = ["chèn vào", "thay thế", "viXTTS", "XTTS", "RVC", "ElevenLabs", "F5-TTS", "công cụ"]
NUM = {"type": "number"}
STR = {"type": "string"}
NUM_OR_NULL = {"type": ["number", "null"]}
TIME = {"type": "string", "pattern": r"^\d{2,}:\d{2},\d{2}$"}
SIGN = {"type": "object", "required": ["ma", "ten", "gia_tri", "don_vi", "trung_vi_that", "phan_vi_that", "bat_thuong", "kho"],
        "properties": {"ma": STR, "ten": STR, "gia_tri": NUM, "don_vi": STR, "trung_vi_that": NUM,
                       "phan_vi_that": {"type": "number", "minimum": 0, "maximum": 100}, "bat_thuong": {"type": "boolean"},
                       "kho": STR}}
OPERATION = {"type": "object",
             "required": ["id", "kenh", "nhan_cap2", "loai", "pham_vi", "muc_do", "bac", "bat_dau_s", "ket_thuc_s",
                          "bat_dau", "ket_thuc", "anh", "diem", "dau_hieu", "giai_thich", "dong_phieu"],
             "properties": {"id": {"type": "string", "pattern": r"^T\d+$"}, "kenh": {"type": "integer", "minimum": 0},
                            "nhan_cap2": {"enum": sorted(set(GLOBAL + LOCAL + EXTRA_LEVELS))},
                            "loai": {"enum": sorted({v[0] for v in OP.values()} | set(CONTRACT_TYPES))},
                            "pham_vi": {"enum": ["cả file", "một đoạn"]},
                            "muc_do": {"enum": ["vô hại", "đáng ngờ", "sai lệch", "sai lệch nặng"]},
                            "bac": {"type": "integer", "minimum": 1, "maximum": 5},
                            "bat_dau_s": {"type": "number", "minimum": 0}, "ket_thuc_s": {"type": "number", "minimum": 0},
                            "bat_dau": TIME, "ket_thuc": TIME, "anh": {"type": "string", "pattern": r"^anh/[^/\\]+\.png$"},
                            "diem": {"type": "object"}, "dau_hieu": {"type": "array", "items": SIGN},
                            "giai_thich": {"type": "string", "minLength": 20}, "dong_phieu": {"type": "integer", "minimum": 1}}}
ROW = {"type": "object", "required": ["stt", "kenh", "nhan_cap2", "kieu"],
       "properties": {"stt": {"type": "integer", "minimum": 1}, "kenh": {"type": "integer", "minimum": 0},
                      "nhan_cap2": {"enum": sorted(set(GLOBAL + LOCAL + EXTRA_LEVELS)) + [L04]}, "kieu": {"enum": ["global", "segment", "gray", "gate"]},
                      "p": NUM_OR_NULL, "bat_dau_s": NUM_OR_NULL, "ket_thuc_s": NUM_OR_NULL,
                      "thao_tac": {"type": ["string", "null"]}, "ly_do": {"type": ["string", "null"]}}}
REASON = {"type": "object", "required": ["duong", "noi_dung"],
          "properties": {"duong": {"enum": ["K1", "K3_vung_xam", "K5_khong_con_dong", "K7_bi_bac"]},
                         "noi_dung": {"type": "string", "minLength": 10}}}
HO_SO = {
    "type": "object",
    "required": ["schema", "tao_luc", "he_thong", "tep", "trang_thai", "output_chinh", "output_phu", "phieu",
                 "kiem_tra_dong_nhat", "thoi_gian_s"],
    "properties": {
        "schema": {"const": SCHEMA_ID}, "tao_luc": STR,
        "he_thong": {"type": "object", "required": ["ten", "mo_hinh", "che_do_khuc", "canh_bao_he_thong"],
                     "properties": {"mo_hinh": {"enum": ["that", "gia_lap"]}, "canh_bao_he_thong": {"type": "array", "items": STR}}},
        "tep": {"type": "object", "required": ["ten", "sha256", "id", "do_dai_s", "so_kenh_phan_tich"],
                "properties": {"sha256": {"type": ["string", "null"]}, "do_dai_s": NUM_OR_NULL,
                               "so_kenh_phan_tich": {"type": "integer", "minimum": 0}}},
        "trang_thai": {"enum": ["xong", "tu_choi", "loi"]},
        "output_chinh": {"type": "object", "required": ["nhan", "ly_do_04", "thao_tac"],
                         "properties": {"nhan": {"type": "object", "required": ["ma", "ten"],
                                                 "properties": {"ma": {"enum": ["01", "02", "03", "04", None]},
                                                                "ten": {"type": ["string", "null"]}}},
                                        "ly_do_04": {"type": "array", "items": REASON},
                                        "thao_tac": {"type": "array", "items": OPERATION}}},
        "output_phu": {"type": "object",
                       "required": ["canh_bao", "bang_moc", "diem_nguong", "bang_so_do", "tuyen_bo_gioi_han", "anh_tong_quan"],
                       "properties": {"canh_bao": {"type": "array", "items": {"type": "object", "required": ["ma", "noi_dung"]}},
                                      "bang_moc": {"type": "array", "items": {"type": "object",
                                                   "required": ["thao_tac", "kenh", "loai", "pham_vi", "muc_do", "bat_dau",
                                                                "ket_thuc", "bat_dau_s", "ket_thuc_s"]}},
                                      "diem_nguong": {"type": "object", "required": ["K1", "kenh"]},
                                      "bang_so_do": {"type": "array", "items": {"type": "object", "required": ["thao_tac", "ma", "gia_tri"]}},
                                      "tuyen_bo_gioi_han": {"type": "array", "minItems": 3, "items": STR},
                                      "anh_tong_quan": {"type": "array", "items": STR}}},
        "phieu": {"type": "array", "items": ROW},
        "kiem_tra_dong_nhat": {"type": "object", "required": ["dat", "vi_pham"]},
        "thoi_gian_s": {"type": "object"}}}

_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(v, t: str) -> bool:
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(obj, sch: dict = HO_SO, path: str = "$") -> list:
    """Kiem khuon theo tap con JSON Schema: type, required, properties, items, enum, const, pattern, min/max, minLength, minItems."""
    err = []
    if "const" in sch and obj != sch["const"]:
        return [f"{path}: phải là {sch['const']!r}"]
    if "enum" in sch and obj not in sch["enum"]:
        return [f"{path}: {obj!r} không thuộc {sch['enum']}"]
    if "type" in sch:
        ts = sch["type"] if isinstance(sch["type"], list) else [sch["type"]]
        if not any(_is(obj, t) for t in ts):
            return [f"{path}: kiểu {type(obj).__name__}, cần {ts}"]
    if isinstance(obj, dict):
        err += [f"{path}: thiếu '{k}'" for k in sch.get("required", []) if k not in obj]
        for k, s in sch.get("properties", {}).items():
            if k in obj:
                err += validate(obj[k], s, f"{path}.{k}")
    if isinstance(obj, list):
        if len(obj) < sch.get("minItems", 0):
            err.append(f"{path}: cần ≥ {sch['minItems']} phần tử")
        if "items" in sch:
            for i, x in enumerate(obj):
                err += validate(x, sch["items"], f"{path}[{i}]")
    if isinstance(obj, str):
        if len(obj) < sch.get("minLength", 0):
            err.append(f"{path}: chuỗi quá ngắn")
        if "pattern" in sch and not re.search(sch["pattern"], obj):
            err.append(f"{path}: {obj!r} không khớp mẫu {sch['pattern']}")
    if _is(obj, "number"):
        if "minimum" in sch and obj < sch["minimum"]:
            err.append(f"{path}: {obj} < {sch['minimum']}")
        if "maximum" in sch and obj > sch["maximum"]:
            err.append(f"{path}: {obj} > {sch['maximum']}")
    return err


def _expected_label(rows: list) -> str | None:
    s = {x["nhan_cap2"][:2] for x in rows}
    return "03" if "03" in s else "02" if "02" in s else None


def _check_rows(ho: dict, dur: float, V: list) -> None:
    rows = ho["phieu"]
    chans = {x["kenh"] for x in rows}
    for c in chans:
        loc = [x for x in rows if x["kenh"] == c and x["nhan_cap2"] in LOCAL]
        has_nen_tc = any(x["kenh"] == c and x["nhan_cap2"] == "02_nen_toan_cuc" for x in rows)
        for x in [x for x in rows if x["kenh"] == c]:
            seg = x.get("bat_dau_s") is not None and x.get("ket_thuc_s") is not None
            if x["nhan_cap2"] in GLOBAL and seg:
                V.append(f"B5 dòng toàn cục {x['stt']} lại có đoạn")
            if x["nhan_cap2"] == L04 and not x.get("ly_do"):
                V.append(f"B3 dòng 04 số {x['stt']} không có lý do")
            if x["nhan_cap2"] == L04 and x["kieu"] == "segment" and float(x.get("p") or 0) < STRONG - TOL:
                V.append(f"B3 dòng bị bác số {x['stt']} điểm < 0,80 mà vẫn thành 04")
        for x in loc:
            a, b = x.get("bat_dau_s"), x.get("ket_thuc_s")
            if a is None or b is None or not b > a:
                V.append(f"B2 dòng cục bộ {x['stt']} không có đoạn")
                continue
            if a < EDGE_S - TOL or b > dur - EDGE_S + TOL:
                V.append(f"S2 đoạn {x['stt']} nằm trong vùng 0,3 s không soi")
            if b - a > SCOPE_FRAC * dur + TOL:
                V.append(f"S1 đoạn {x['stt']} dài hơn 50 % tệp")
            kc = x.get("kiem_chung") or {}
            if kc.get("che_do") == "hieu_chuan":
                if kc.get("p_giu") is None or kc.get("nguong_giu") is None or kc["p_giu"] < kc["nguong_giu"] - TOL:
                    V.append(f"B3 dòng {x['stt']} giữ mà xác suất giữ dưới ngưỡng")
            elif kc.get("che_do") == "chua_co":
                pass                                    # trung thuc: chua co bo kiem chung doc lap, khong ghi nhan gia
            elif kc.get("nhan") != x["nhan_cap2"]:
                V.append(f"B3 dòng {x['stt']} giữ mà bộ kiểm chứng không trả cùng nhãn")
            if x["nhan_cap2"] == "02_nen_cuc_bo" and has_nen_tc and float(x.get("p") or 0) < NEN_CB_WITH_TC - TOL:
                V.append(f"B6 nền cb {x['stt']} < 0,9 khi đã có nền tc")
        for i, u in enumerate(loc):
            for w in loc[i + 1:]:
                if None not in (u.get("bat_dau_s"), w.get("bat_dau_s")) and \
                        overlap((u["bat_dau_s"], u["ket_thuc_s"]), (w["bat_dau_s"], w["ket_thuc_s"])) > OVERLAP_MAX + TOL:
                    V.append(f"B6 dòng {u['stt']} và {w['stt']} chồng > 30 %")
        for s0 in chunk_starts(dur):
            n = sum(1 for x in loc if x.get("bat_dau_s") is not None and
                    s0 <= (x["bat_dau_s"] + x["ket_thuc_s"]) / 2 < s0 + CHUNK_S)
            if n > MAX_LOCAL_PER_CHUNK:
                V.append(f"B6 kênh {c}: {n} dòng cục bộ trong khúc {s0:.0f}–{s0 + CHUNK_S:.0f} s")


def _check_ops(ho: dict, dur: float, V: list) -> None:
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    ops, rows = oc["thao_tac"], ho["phieu"]
    by_stt = {x["stt"]: x for x in rows}
    ids = [o["id"] for o in ops]
    if len(set(ids)) != len(ids):
        V.append("B9 trùng mã thao tác")
    for o in ops:
        loai, pv, md = OP[o["nhan_cap2"]]
        if (o["loai"], o["pham_vi"], o["muc_do"], o["bac"]) != (loai, pv, md, SEV[o["nhan_cap2"]]):
            V.append(f"S3 {o['id']}: loại/phạm vi/mức độ/bậc không cố định theo thao tác")
        x = by_stt.get(o["dong_phieu"])
        if x is None or x.get("thao_tac") != o["id"] or x["nhan_cap2"] != o["nhan_cap2"] or x["kenh"] != o["kenh"]:
            V.append(f"B9 {o['id']} không khớp dòng phiếu")
        elif pv == "một đoạn" and (abs(x["bat_dau_s"] - o["bat_dau_s"]) > TOL or abs(x["ket_thuc_s"] - o["ket_thuc_s"]) > TOL):
            V.append(f"B9 {o['id']} mốc khác dòng phiếu")
        if pv == "cả file" and not (abs(o["bat_dau_s"]) <= TOL and abs(o["ket_thuc_s"] - dur) <= 2 * TOL):
            V.append(f"B5 {o['id']} toàn cục không khoanh cả file")
        good = [s for s in o["dau_hieu"] if s.get("gia_tri") is not None]
        if len(good) < 2:
            V.append(f"C1 {o['id']} có < 2 dấu hiệu có số")
        if not re.search(r"\d", o["giai_thich"]):
            V.append(f"C1 {o['id']} giải thích không có số")
    for x in rows:
        if x["nhan_cap2"] in GLOBAL + LOCAL and not any(o["dong_phieu"] == x["stt"] for o in ops):
            V.append(f"B9 dòng phiếu {x['stt']} ({x['nhan_cap2']}) không có thao tác")
    moc = {m["thao_tac"]: m for m in op_["bang_moc"]}
    if set(moc) != set(ids):
        V.append("B7 bảng mốc khác danh sách thao tác")
    for o in ops:
        m = moc.get(o["id"])
        if m and (abs(m["bat_dau_s"] - o["bat_dau_s"]) > TOL or abs(m["ket_thuc_s"] - o["ket_thuc_s"]) > TOL):
            V.append(f"B7 bảng mốc {o['id']} lệch mốc thao tác")
    allowed = set(ids) | {"K1", "K2"}
    if any(s["thao_tac"] not in allowed for s in op_["bang_so_do"]):
        V.append("B7 bảng số đo có mục không thuộc phiếu")
    if len({o["anh"] for o in ops}) != len(ops):
        V.append("C1 hai thao tác dùng chung ảnh")


def check_bind(ho: dict) -> list:
    """Luat gan ket (#bind + Scope). Tra danh sach vi pham (rong = dat)."""
    V = []
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    lab, ops, rows = oc["nhan"]["ma"], oc["thao_tac"], ho["phieu"]
    if ho["trang_thai"] in ("tu_choi", "loi"):
        if lab is not None or ops or rows:
            V.append("B0 từ chối/lỗi mà vẫn có nhãn hoặc phiếu")
        return V
    if lab is None:
        return ["B1 tệp đã xử lý mà không có nhãn"]
    if oc["nhan"]["ten"] != L1_NAME[lab]:
        V.append("B1 tên nhãn không khớp mã")
    dur = float(ho["tep"]["do_dai_s"] or 0.0)
    k1_fail = ho["output_phu"]["diem_nguong"]["K1"].get("quyet_dinh") == "04"
    kenh = op_["diem_nguong"]["kenh"]
    passed = {k["kenh"]: bool(k["K5"]["qua_cong"]) for k in kenh if k.get("K5")}
    exp = _expected_label(rows)
    if lab == "01":
        if rows or ops:
            V.append("B2 nhãn 01 mà phiếu không rỗng")
        if k1_fail or not passed or any(passed.values()):
            V.append("B1 nhãn 01 không đến từ cổng 01 (phải mọi kênh dưới ngưỡng)")
    elif lab in ("02", "03"):
        if exp != lab:
            V.append("B4 nhãn chính khác dòng có bậc cao nhất")
    elif lab == "04":
        if exp is not None:
            V.append("B4 nhãn 04 mà phiếu có dòng 02/03")
        if not k1_fail and not any(x["nhan_cap2"] == L04 for x in rows):
            V.append("B1 nhãn 04 không có dòng 04 và không trượt K1")
    if any(passed.get(x["kenh"]) is False for x in rows):
        V.append("B1 có dòng phiếu ở kênh dưới cổng 01")
    if k1_fail and (rows or ops):
        V.append("B1 trượt K1 mà vẫn có phiếu")
    if (lab == "04") != bool(oc["ly_do_04"]):
        V.append("B3 nhãn 04 phải kèm lý do (và chỉ nhãn 04 mới có lý do 04)")
    if lab not in ("02", "03") and ops:
        V.append("B4 nhãn 01/04 mà có thao tác")
    _check_rows(ho, dur, V)
    _check_ops(ho, dur, V)
    warns = [w["noi_dung"] for w in op_["canh_bao"]]
    if not k1_fail and WARN_EDGE not in warns:
        V.append("C2 thiếu cảnh báo vùng 0,3 s không soi")
    if any(k.get("cham_gioi_han") for k in kenh) and not any(w.startswith(WARN_MORE_C) for w in warns):
        V.append("C2 chạm giới hạn 2 đoạn/khúc mà thiếu cảnh báo 'có thể còn chỗ sửa khác'")
    if any(s not in op_["tuyen_bo_gioi_han"] for s in LIMITS_REQUIRED):
        V.append("C3 thiếu tuyên bố giới hạn bắt buộc")
    return V


def _explain_dirs(o: dict, V: list) -> None:
    """Moi dau hieu nhac trong giai thich phai dung chieu: cao hon khi ti le kho <= gia tri lon hon, nguoc lai thap hon."""
    txt, n = o["giai_thich"], 0
    for s in o["dau_hieu"]:
        if "cao_hon_pct" not in s:
            continue
        key = f"{s['ten']} {vn(s['gia_tri'])}"
        i = txt.find(key)
        if i < 0:
            continue
        n += 1
        tail = txt[i + len(key):i + len(key) + 40]
        want = "cao hơn" if s["cao_hon_pct"] >= s["thap_hon_pct"] else "thấp hơn"
        if want not in tail:
            V.append(f"O1 {o['id']}: dấu hiệu “{s['ten']}” ghi sai chiều (cần “{want}”)")
    if n < 2:
        V.append(f"O1 {o['id']}: giải thích nêu < 2 dấu hiệu có số")


def check_contract(ho: dict) -> list:
    """Kiem THEM cho ho so v9 theo muc Output / Scope cua hop dong (ngoai #bind): tu vung canh bao, kenh trai/phai, phien ban + bam,
    luat xac suat ca tep 0,6 / vung xam, chieu dau hieu, xoa loi <= 1 s, du cau gioi han. Tra danh sach vi pham."""
    V = []
    oc, op_ = ho["output_chinh"], ho["output_phu"]
    warns = [w["noi_dung"] for w in op_["canh_bao"]]
    V += [f"O2 cảnh báo ngoài hợp đồng: {w!r}" for w in warns if w not in WARN_VOCAB]
    if len(set(warns)) != len(warns):
        V.append("O2 cảnh báo trùng")
    lim = " ".join(op_["tuyen_bo_gioi_han"])
    V += [f"O4 tuyên bố giới hạn thiếu mục “{m}”" for m in LIMIT_MARKERS if m not in lim]
    if ho["trang_thai"] != "xong":
        return V
    t = ho["tep"]
    if not (isinstance(t.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", t["sha256"])):
        V.append("O3 thiếu mã băm SHA-256 đủ 64 ký tự")
    pv = ho["he_thong"].get("phien_ban") or {}
    ng = pv.get("nguong") or {}
    if not (ng.get("tep") and re.fullmatch(r"[0-9a-f]{40}", str(ng.get("sha1"))) and isinstance(ng.get("params"), dict)):
        V.append("O3 thiếu phiên bản ngưỡng (tên tệp + sha1 + tham số)")
    mods = pv.get("mo_hinh") or []
    if not mods or any(not (m.get("ten") and re.fullmatch(r"[0-9a-f]{40}", str(m.get("sha1")))) for m in mods):
        V.append("O3 thiếu tên / sha1 của tệp mô hình")
    lab = oc["nhan"]["ma"]
    k1_fail = op_["diem_nguong"]["K1"].get("quyet_dinh") == "04"
    if not k1_fail and WARN_EDGE_C not in warns:
        V.append("C23 thiếu cảnh báo 0,3 s đầu và cuối")
    kenh = op_["diem_nguong"]["kenh"]
    nghi = any(k.get("K8", {}).get("nghi_03") for k in kenh)
    if (WARN_03 in warns) != (lab == "02" and nghi):
        V.append("S5 cảnh báo “có đoạn nghi 03 chưa đủ chắc” phải đi kèm đúng nhãn 02 có đoạn nghi 03")
    n_ch = t.get("so_kenh_phan_tich") or 0
    for o in oc["thao_tac"] + op_["bang_moc"]:
        want = (["trái", "phải"][o["kenh"]] if o["kenh"] in (0, 1) else "?") if n_ch > 1 else None
        if o.get("kenh_ten", "thiếu") != want:
            V.append(f"O5 {o.get('id', o.get('thao_tac'))}: kênh phải ghi {want!r} (đang {o.get('kenh_ten', 'thiếu')!r})")
    for o in oc["thao_tac"]:
        if o["loai"] not in CONTRACT_TYPES:
            V.append(f"S1 {o['id']}: loại {o['loai']!r} không thuộc 5 loại của hợp đồng")
        if o["nhan_cap2"] == "03_xoa_loi" and o["ket_thuc_s"] - o["bat_dau_s"] > 1.0 + TOL:
            V.append(f"O6 {o['id']}: vùng xoá lời dài hơn 1 s")
        _explain_dirs(o, V)
        V += [f"S11 {o['id']}: giải thích có “{w}”" for w in FORBIDDEN_IN_EXPLAIN if w in o["giai_thich"]]
    for k in kenh:
        for c, v in (k.get("K3") or {}).items():
            have = any(o["kenh"] == k["kenh"] and o["nhan_cap2"] == c for o in oc["thao_tac"])
            if v.get("nhanh_tat"):
                if have:
                    V.append(f"S2 kênh {k['kenh']} {c}: nhánh tắt (không có ngưỡng) mà vẫn báo thao tác cả file")
            elif v.get("hieu_chinh"):
                if v.get("vung_xam") != [0.4, 0.6]:
                    V.append(f"S2 kênh {k['kenh']} {c}: vùng xám phải là [0,4; 0,6]")
                p = v.get("p_hieu_chinh")
                if (p is not None and p >= 0.6) != have:
                    V.append(f"S2 kênh {k['kenh']} {c}: thao tác cả file phải có ⟺ xác suất hiệu chỉnh ≥ 0,6")
            elif v.get("vung_xam") is not None or "chưa hiệu chỉnh" not in str(v.get("ghi_chu", "")):
                V.append(f"S2 kênh {k['kenh']} {c}: chưa hiệu chỉnh mà không ghi rõ (vùng xám phải để trống)")
        if (k.get("K5") or {}).get("che_do") != "chua_co_cong_sang_loc" and not (k.get("K5") or {}).get("mo_hinh"):
            V.append(f"S4 kênh {k['kenh']}: cổng sàng lọc không ghi rõ chế độ")
    if any(not v.get("hieu_chinh") and not v.get("nhanh_tat") for k in kenh for v in (k.get("K3") or {}).values()) and "chưa hiệu chỉnh" not in lim:
        V.append("S2 tuyên bố giới hạn phải ghi “xác suất cả tệp chưa hiệu chỉnh”")
    _check_coverage(ho, kenh, V)
    return V


def _check_coverage(ho: dict, kenh: list, V: list) -> None:
    """Khong nhan vo pham vi soi: thao tac KHONG soi (nhanh tat: tep nguong khong co khoa sinh ra no) phai duoc ke ten trong gioi han
    (dong 'Nhanh dang tat'), khong duoc co trong danh sach thao tac, va phai khop K3.nhanh_tat / K7.lop.bat."""
    if not kenh:
        return
    soi = ho["output_phu"]["diem_nguong"].get("thao_tac_soi")
    if not isinstance(soi, dict):
        V.append("S3 thiếu danh sách thao tác được soi / không soi (diem_nguong.thao_tac_soi)")
        return
    off = list(soi.get("ma_khong_soi") or [])
    off_line = next((x for x in ho["output_phu"]["tuyen_bo_gioi_han"] if x.startswith("Nhánh đang tắt")), "")
    for lvl, name in zip(off, soi.get("khong_soi") or []):
        if name not in off_line:
            V.append(f"S3 thao tác không soi “{name}” chưa được ghi trong tuyên bố giới hạn")
    V += [f"S3 {o['id']}: báo thao tác {o['nhan_cap2']} của nhánh tắt" for o in ho["output_chinh"]["thao_tac"] if o["nhan_cap2"] in off]
    if soi.get("so_duoc_soi") != soi.get("tong", 8) - len(off):
        V.append("S3 số thao tác được soi không khớp danh sách không soi")
    for k in kenh:
        V += [f"S3 kênh {k['kenh']} {c}: K3 ghi nhánh tắt nhưng không có trong danh sách không soi"
              for c, v in (k.get("K3") or {}).items() if v.get("nhanh_tat") and c not in off]
        V += [f"S3 kênh {k['kenh']} {c}: K7 ghi tắt nhưng không có trong danh sách không soi"
              for c, v in ((k.get("K7") or {}).get("lop") or {}).items() if v.get("bat") is False and c != "03_ghep_giong_that" and c not in off]


def assert_valid(ho: dict) -> None:
    err = validate(ho)
    if not err:
        err = check_bind(ho)
    if err:
        raise ValueError("hồ sơ không hợp lệ: " + " | ".join(err[:20]))


def is_v9(ho: dict) -> bool:
    return str(ho.get("he_thong", {}).get("ten", "")).startswith("VoiceAudit v9")


def v9_constants() -> None:
    """Ho so v9 can hang so DA MO RONG (common9: 8 thao tac + gioi han v9; ticket: ten loai 'giọng máy'). Trong san pham ticket da nap truoc;
    khi chay 'python -m v6.io.schema' rieng thi PHAI nap o day, neu khong ho so v9 hop le bi bao sai (S3 loai / C3 gioi han)."""
    from v6.product import ticket  # noqa: F401  (nap = patch tai cho cac danh sach / tu dien dung chung)


def check_file(path: str, contract: bool = False) -> list:
    """validate + #bind (+ hop dong: --contract hoac ho so v9). Ho so v9: nap hang so v9 truoc (v9_constants)."""
    with open(path, encoding="utf-8") as fh:
        ho = json.load(fh)
    if is_v9(ho):
        v9_constants()
    err = validate(ho) or check_bind(ho)
    if not err and (contract or is_v9(ho)):
        err = check_contract(ho)
    return err


def scan(root: str, newer: str | None = None) -> dict:
    """Kiem MOI ho so v9 duoi root (vd .tmp_v6/e2e9): ho_so.json -> check_file; ho_so_khong_hop_le.json (finish9 da bat loi, ma thoat 4) = khong dat.
    newer = tep moc: chi xet ho so moi hon tep do (vd v6/product/ticket.py). Ho so khong phai v9 bo qua (hang so v6 khac v9)."""
    import os
    t0 = os.path.getmtime(newer) if newer else None
    out = {"dat": 0, "khong_dat": [], "bo_qua": []}
    v9_constants()
    for d, _, files in os.walk(root):
        for f in ("ho_so.json", "ho_so_khong_hop_le.json"):
            if f not in files or (t0 is not None and os.path.getmtime(os.path.join(d, f)) <= t0):
                continue
            p = os.path.join(d, f)
            with open(p, encoding="utf-8") as fh:
                ho = json.load(fh)
            if not is_v9(ho):
                out["bo_qua"].append(p)
            elif f == "ho_so_khong_hop_le.json":
                out["khong_dat"].append((p, list((ho.get("kiem_tra_dong_nhat") or {}).get("vi_pham") or ["finish9 ghi ho_so_khong_hop_le.json"])))
            else:
                err = check_file(p, True)
                if err:
                    out["khong_dat"].append((p, err))
                else:
                    out["dat"] += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Lược đồ + kiểm #bind hồ sơ VoiceAudit v6 / v9")
    ap.add_argument("ho_so", nargs="?")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--contract", action="store_true", help="kiem them luat hop dong (tu bat voi ho so v9)")
    ap.add_argument("--scan", default=None, help="thu muc: kiem moi ho so v9 ben duoi (ho_so.json + ho_so_khong_hop_le.json)")
    ap.add_argument("--newer", default=None, help="voi --scan: chi xet ho so moi hon tep nay")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.scan:
        r = scan(a.scan, a.newer)
        for p, err in r["khong_dat"]:
            print(f"KHÔNG ĐẠT {p}:\n  " + "\n  ".join(err))
        print(f"đạt {r['dat']} · không đạt {len(r['khong_dat'])} · bỏ qua (không phải v9) {len(r['bo_qua'])}")
        sys.exit(0 if not r["khong_dat"] else 1)
    if a.dump or not a.ho_so:
        print(json.dumps({"$id": SCHEMA_ID, **HO_SO}, ensure_ascii=False, indent=1))
        return
    err = check_file(a.ho_so, a.contract)
    print("ĐẠT" if not err else "KHÔNG ĐẠT:\n  " + "\n  ".join(err))
    sys.exit(0 if not err else 1)


if __name__ == "__main__":
    main()
