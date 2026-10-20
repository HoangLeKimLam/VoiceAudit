# -*- coding: utf-8 -*-
"""BO NGHE MU + BO CHAM Req 13 (WP5, 10/10) cho nguoi dung / nguoi cham ngoai nhom.

  nghe   (REQ-4 kiem nhan 01): moi dong bao nham cua engine_v9A tren tep THAT goc VSS (49 tep) + EdAcc (21 tep), cung
         70 tep THAT KHONG bi bao (doi chung, cung nguon: 49 VSS + 21 EdAcc, hat giong co dinh). Moi dong -> mot doan wav
         +-3 s quanh dong bi khoanh (thao tac ca file -> ca tep); moi doi chung cap voi mot tep bao nham cung nguon va
         SAO CHEP cau truc cua no (so dong, ca tep / doan, do rong, vi tri tuong doi) de nguoi nghe KHONG biet doan nao
         bi bao. Thu tu xao tron, ten an danh N001... Xuat lai: bo cu duoc CHUYEN sang .tmp_v6/codebak/listen_kit_cu/ (khong xoa)
         -> <root>/listen_kit/nghe/am_thanh/*.wav + phieu_nghe.csv (cot tra loi de trong) ; khoa: <root>/listen_kit_khoa/khoa_nghe.csv
  req13  anh bang chung tu .tmp_v6/e2e9/*/ (ho_so.json: dong thao tac co anh, loai cua dong TRUNG loai that cua tep) can
         bang 5 loai (khu on / doi giong / ghep loi that / xoa loi / giong may), toi da --per-class anh moi loai; CHE dai tieu
         de (chua ten loai) -> listen_kit/req13/anh/A001.png + phieu_req13.csv ; khoa: listen_kit_khoa/khoa_req13.csv
  score  cham phieu Req 13 da dien: dung / tong theo nguoi cham + gop, Wilson 95 %, chi tieu >= 70 % (>= 3 nguoi cham)
  scrape (B-12) mau nghe mu ~5 % ban goc cao (I4) moi kenh, toi da 30 s / doan -> listen_kit/cao/ + phieu_cao.csv;
         khoa (url / kenh / giay phep / codec) o listen_kit_khoa/khoa_cao.csv; dem ban goc thieu giay phep

  readme viet lai listen_kit/README.md theo tep dang co (muc 1 nghe, 2 Req 13 + quy tac chon anh / loai thieu, 3 cao neu co)
         -> export va scrape khong ghi de muc cua nhau; lenh trong README deu dang cd + run_serial

CLI: python -m v6.data.listen_kit export [--engine .tmp_v6/results/engine_v9A.json] [--per-class 8] [--seed 2026]
     | score <phieu_req13.csv> [--key <khoa_req13.csv>] | scrape [--share 0.05] | readme | selftest
(bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.listen_kit ...)"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402

KIT = os.path.join(C.BASE, "listen_kit")
KEY_DIR = os.path.join(C.BASE, "listen_kit_khoa")
ENGINE = os.path.join(C.RESULTS, "engine_v9A.json")
E2E = os.path.join(C.TMP, "e2e9")
ARCHIVE = os.path.join(C.TMP, "codebak", "listen_kit_cu")
PAD_S, PSEUDO_W, EDGE_S = 3.0, 0.5, 0.5
FLAG_SOURCES = {"v6natvi": "vss", "v6naten": "edacc"}
TITLE_FRAC = 0.07                               # dai tieu de anh ticket (~7 % chieu cao, chua ten loai + muc do) -> che trang
CLASSES = ("khu_on", "doi_giong", "ghep_loi_that", "xoa_loi", "giong_may")
CLASS_VI = {"khu_on": "Khử ồn", "doi_giong": "Đổi giọng", "ghep_loi_that": "Ghép lời thật", "xoa_loi": "Xoá lời",
            "giong_may": "Giọng máy"}
FORM_COLS = ["ma_doan", "tep", "pham_vi", "bat_dau_s", "ket_thuc_s", "co_vet_cat", "co_nhac_nen", "doi_nguoi_noi",
             "nghe_nhu_khu_on", "muc_tu_tin_1_3", "ghi_chu"]


# ---------------- loai thao tac ----------------
def class_of_level2(level2: str) -> str | None:
    """nhan cap 2 cua dong ticket -> 1 trong 5 loai Scope."""
    l2 = str(level2 or "")
    if l2.startswith("02_nen"):
        return "khu_on"
    if l2.startswith("02_giong"):
        return "doi_giong"
    if l2 == "03_xoa_loi":
        return "xoa_loi"
    if "giong_may" in l2:
        return "giong_may"
    if l2.startswith("03_ghep") or l2 == "03_ghep_chen":
        return "ghep_loi_that"
    return None


def classes_of_kind(kind: str) -> set[str]:
    """loai tep sinh (ten thu muc e2e: nen_toan_cuc, xoa_loi_ky, may_thay, da_ghep_nen...) -> tap loai THAT trong tep."""
    k = str(kind)
    out = set()
    if k.startswith("nen_") or k.endswith("_nen"):
        out.add("khu_on")
    if k.startswith("giong_"):
        out.add("doi_giong")
    if k.startswith("ghep_") or k == "da_ghep_nen":
        out.add("ghep_loi_that")
    if k.startswith("xoa_loi"):
        out.add("xoa_loi")
    if k.startswith("may_"):
        out.add("giong_may")
    return out


def kind_of_case(case_id: str) -> str:
    """'v6naten_xoa_loi_ky_0016' -> 'xoa_loi_ky'."""
    s = re.sub(r"^v6[a-z0-9]*_", "", case_id)
    return re.sub(r"_\d+$", "", s)


# ---------------- doan nghe ----------------
def clip_window(start: float | None, end: float | None, dur: float, pad: float = PAD_S) -> tuple[float, float]:
    """(t0, t1) cua doan nghe: +-pad quanh [start, end], cat theo [0, dur]; start None (ca tep) -> ca tep."""
    if start is None or end is None:
        return 0.0, float(dur)
    return max(0.0, float(start) - pad), min(float(dur), float(end) + pad)


def pseudo_row(dur: float, rng: random.Random, width: float = PSEUDO_W) -> tuple[float, float]:
    """Dong gia cho tep doi chung: rong `width`, tam ngau nhien trong [EDGE_S, dur - EDGE_S]."""
    lo, hi = EDGE_S + width / 2, max(EDGE_S + width / 2, dur - EDGE_S - width / 2)
    c = rng.uniform(lo, hi)
    return round(c - width / 2, 3), round(c + width / 2, 3)


def mirror_row(dur: float, start: float, end: float, dur_ref: float, rng: random.Random) -> tuple[float, float]:
    """Dong gia cho doi chung SAO CHEP dong bao nham cap doi: cung DO RONG, cung VI TRI TUONG DOI (tam / thoi luong),
    cat vao [EDGE_S, dur - EDGE_S]; tep qua ngan -> vi tri ngau nhien. Rong + vi tri khong con lo doan nao bi bao."""
    w = max(0.02, float(end) - float(start))
    if dur_ref <= 0 or dur < w + 2 * EDGE_S:
        return pseudo_row(dur, rng, min(w, max(0.02, dur - 2 * EDGE_S)))
    c = (float(start) + float(end)) / 2 / dur_ref * dur
    c = min(max(c, EDGE_S + w / 2), dur - EDGE_S - w / 2)
    return round(c - w / 2, 3), round(c - w / 2, 3) + round(w, 3)


def _manifest_index(sets: Iterable[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for s in sets:
        for e in C.read_manifest(os.path.join(C.V6DATA, s, "manifest.json")):
            if e.get("cfg") == "goc":
                m = e.get("meta") or {}
                out[e["id"]] = {"wav": C.abs_wav(e["wav"]), "source": m.get("source"), "person": m.get("person"),
                                "host_file": m.get("host_file"), "npv": bool(m.get("ngoai_pham_vi")), "lang": m.get("lang")}
    return out


def select_listen(records: Mapping[str, Sequence[Mapping[str, Any]]], index: Mapping[str, Mapping[str, Any]],
                  seed: int = 2026) -> tuple[list[dict], list[dict]]:
    """(bao_nham, doi_chung): tep THAT goc nguon VSS (v6natvi) / EdAcc (v6naten), final 02|03 vs final 01 (ngau nhien, cung so
    luong moi nguon)."""
    flagged, pool = [], defaultdict(list)
    for s, src in FLAG_SOURCES.items():
        for r in records.get(s, []):
            ix = index.get(r["id"])
            if not ix or r.get("cfg") != "goc" or r.get("expected") != "01" or ix["npv"] or ix["source"] != src:
                continue
            if r.get("final") in ("02", "03"):
                flagged.append({**r, "source": src})
            elif r.get("final") == "01":
                pool[src].append({**r, "source": src})
    rng = random.Random(seed)
    need = Counter(r["source"] for r in flagged)
    controls = []
    for src in sorted(need):
        cand = sorted(pool[src], key=lambda r: r["id"])
        controls += rng.sample(cand, min(need[src], len(cand)))
    return flagged, controls


def _rows_of(r: Mapping[str, Any]) -> list[tuple[Mapping[str, Any], float | None, float | None]]:
    """(dong, start, end) cua mot tep bao nham; dong ca tep -> (dong, None, None)."""
    out = []
    for row in (r.get("rows") or [{"kind": "global"}]):
        is_glob = row.get("kind") == "global" or row.get("start") is None
        out.append((row, None, None) if is_glob else (row, float(row["start"]), float(row["end"])))
    return out


def build_clips(flagged: Sequence[Mapping], controls: Sequence[Mapping], durs: Mapping[str, float],
                seed: int = 2026) -> list[dict]:
    """Moi dong bao nham -> 1 doan. Moi doi chung duoc CAP voi mot tep bao nham cung nguon (xao tat dinh) va SAO CHEP cau
    truc cua no: cung so dong, cung kieu (ca tep / doan), cung do rong + vi tri tuong doi tung dong (mirror_row)
    -> phan bo so doan / tep, do rong va vi tri cua doi chung = cua bao nham (nguoi nghe khong phan biet duoc)."""
    rng = random.Random(seed + 1)
    clips = []
    for r in flagged:
        for j, (row, s, e) in enumerate(_rows_of(r)):
            t0, t1 = clip_window(s, e, durs[r["id"]])
            clips.append({"id": r["id"], "source": r["source"], "nhom": f"bao_nham_{r['source']}", "final": r.get("final"),
                          "row": j, "level2": row.get("level2"), "via": row.get("via"), "p": row.get("p"),
                          "row_s": s, "row_e": e, "t0": t0, "t1": t1, "cap_doi": None})
    by_src_f: dict[str, list] = defaultdict(list)
    for r in sorted(flagged, key=lambda x: x["id"]):
        by_src_f[r["source"]].append(r)
    for src in sorted(by_src_f):
        rng.shuffle(by_src_f[src])
    used: Counter = Counter()
    for r in controls:
        src = r["source"]
        pool = by_src_f.get(src) or []
        if not pool:
            continue
        ref = pool[used[src] % len(pool)]
        used[src] += 1
        for j, (_, s0, e0) in enumerate(_rows_of(ref)):
            s, e = (None, None) if s0 is None else mirror_row(durs[r["id"]], s0, e0, durs[ref["id"]], rng)
            t0, t1 = clip_window(s, e, durs[r["id"]])
            clips.append({"id": r["id"], "source": src, "nhom": "doi_chung", "final": r.get("final"), "row": -1 - j,
                          "level2": None, "via": None, "p": None, "row_s": s, "row_e": e, "t0": t0, "t1": t1,
                          "cap_doi": ref["id"]})
    rng.shuffle(clips)
    for k, c in enumerate(clips, 1):
        c["ma_doan"] = f"N{k:03d}"
    return clips


def _write_csv(path: str, cols: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _archive_old(d: str, archive: str) -> str | None:
    """Thu muc kit cu -> chuyen (khong xoa) sang archive/<thoi diem>/ de khong con doan cu lan voi doan moi."""
    if not os.path.isdir(d) or not os.listdir(d):
        return None
    import shutil
    import time
    dst = os.path.join(archive, time.strftime("%Y%m%d_%H%M%S"), os.path.basename(d))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(d, dst)
    return dst


def export_listen(engine: str = ENGINE, kit: str = KIT, key_dir: str = KEY_DIR, seed: int = 2026,
                  records: Mapping | None = None, index: Mapping | None = None, archive: str = ARCHIVE) -> dict[str, Any]:
    import soundfile as sf
    if records is None:
        with open(engine, encoding="utf-8") as fh:
            records = json.load(fh)["records"]
    index = index if index is not None else _manifest_index(("natvi", "naten"))
    flagged, controls = select_listen(records, index, seed)
    durs = {r["id"]: sf.info(index[r["id"]]["wav"]).duration for r in list(flagged) + list(controls)}
    clips = build_clips(flagged, controls, durs, seed)
    _archive_old(os.path.join(kit, "nghe"), archive)
    out_dir = os.path.join(kit, "nghe", "am_thanh")
    os.makedirs(out_dir, exist_ok=True)
    form = []
    for c in clips:
        y, sr = sf.read(index[c["id"]]["wav"], dtype="float32", always_2d=True)
        y = y.mean(1)
        a, b = int(round(c["t0"] * sr)), int(round(c["t1"] * sr))
        sf.write(os.path.join(out_dir, c["ma_doan"] + ".wav"), y[a:b], sr, subtype="PCM_16")
        glob_ = c["row_s"] is None
        c.update({"clip_t0_s": round(c["t0"], 3), "clip_t1_s": round(c["t1"], 3),
                  "host_file": index[c["id"]]["host_file"], "person": index[c["id"]]["person"]})
        form.append({"ma_doan": c["ma_doan"], "tep": f"am_thanh/{c['ma_doan']}.wav", "pham_vi": "ca_tep" if glob_ else "doan",
                     "bat_dau_s": 0.0 if glob_ else round(c["row_s"] - c["t0"], 2),
                     "ket_thuc_s": round(c["t1"] - c["t0"], 2) if glob_ else round(c["row_e"] - c["t0"], 2)})
    form.sort(key=lambda r: r["ma_doan"])
    _write_csv(os.path.join(kit, "nghe", "phieu_nghe.csv"), FORM_COLS, form)
    key_cols = ["ma_doan", "id", "source", "nhom", "final", "row", "level2", "via", "p", "row_s", "row_e", "cap_doi", "clip_t0_s",
                "clip_t1_s", "host_file", "person"]
    _write_csv(os.path.join(key_dir, "khoa_nghe.csv"), key_cols, sorted(clips, key=lambda c: c["ma_doan"]))
    return {"clips": len(clips), "flagged_files": len(flagged), "control_files": len(controls),
            "by_group": dict(Counter(c["nhom"] for c in clips)),
            "flagged_by_source": dict(Counter(r["source"] for r in flagged)),
            "global_clips": sum(c["row_s"] is None for c in clips), "dir": os.path.join(kit, "nghe")}


# ---------------- Req 13 ----------------
def req13_candidates(e2e: str = E2E) -> list[dict]:
    """Dong thao tac co anh, loai dong (nhan_cap2) thuoc loai THAT cua tep (theo ten tep sinh)."""
    out = []
    for hp in sorted(glob.glob(os.path.join(e2e, "*", "ho_so.json"))):
        case = os.path.basename(os.path.dirname(hp))
        truth = classes_of_kind(kind_of_case(case))
        if not truth:
            continue
        with open(hp, encoding="utf-8") as fh:
            hs = json.load(fh)
        for row in (hs.get("output_chinh") or {}).get("thao_tac", []):
            cls = class_of_level2(row.get("nhan_cap2"))
            img = os.path.join(os.path.dirname(hp), str(row.get("anh") or ""))
            if cls and cls in truth and row.get("anh") and os.path.isfile(img):
                out.append({"case": case, "row_id": row.get("id"), "class": cls, "nhan_cap2": row.get("nhan_cap2"),
                            "pham_vi": row.get("pham_vi"), "img": img, "lang": "en" if case.startswith("v6naten") else "vi"})
    return out


def pick_balanced(cands: Sequence[Mapping], per_class: int, seed: int = 2026) -> list[dict]:
    """<= per_class anh moi loai, xen vi/en; ngau nhien tat dinh; tra danh sach da xao tron."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for c in cands:
        by[c["class"]].append(c)
    out = []
    for cls in CLASSES:
        xs = sorted(by[cls], key=lambda c: (c["case"], str(c["row_id"])))
        rng.shuffle(xs)
        vi = [c for c in xs if c["lang"] == "vi"]
        en = [c for c in xs if c["lang"] == "en"]
        mix = [c for pair in zip(vi, en) for c in pair] + vi[len(en):] + en[len(vi):]
        out += [dict(c) for c in mix[:per_class]]
    rng.shuffle(out)
    return out


def mask_title(src: str, dst: str, frac: float = TITLE_FRAC) -> None:
    """Chep anh, to trang dai tieu de tren cung (ten loai, pham vi, muc do) -> nguoi cham khong thay nhan."""
    from PIL import Image, ImageDraw
    with Image.open(src) as im:
        im = im.convert("RGB")
        px = int(round(im.height * frac))
        ImageDraw.Draw(im).rectangle([0, 0, im.width, px], fill=(255, 255, 255))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        im.save(dst)


def export_req13(e2e: str = E2E, kit: str = KIT, key_dir: str = KEY_DIR, per_class: int = 8, seed: int = 2026,
                 n_raters: int = 3) -> dict[str, Any]:
    cands = req13_candidates(e2e)
    picked = pick_balanced(cands, per_class, seed)
    rows, keys = [], []
    for k, c in enumerate(picked, 1):
        code = f"A{k:03d}"
        mask_title(c["img"], os.path.join(kit, "req13", "anh", code + ".png"))
        rows.append({"ma_anh": code, "anh": f"anh/{code}.png", **{f"nguoi_cham_{i}": "" for i in range(1, n_raters + 1)}})
        keys.append({"ma_anh": code, "dap_an_so": CLASSES.index(c["class"]) + 1, "dap_an": c["class"], "case": c["case"],
                     "row_id": c["row_id"], "nhan_cap2": c["nhan_cap2"], "pham_vi": c["pham_vi"], "lang": c["lang"]})
    _write_csv(os.path.join(kit, "req13", "phieu_req13.csv"), ["ma_anh", "anh"] + [f"nguoi_cham_{i}" for i in range(1, n_raters + 1)], rows)
    _write_csv(os.path.join(key_dir, "khoa_req13.csv"), list(keys[0].keys()) if keys else ["ma_anh"], keys)
    return {"candidates": len(cands), "cand_by_class": dict(Counter(c["class"] for c in cands)), "picked": len(picked),
            "picked_by_class": dict(Counter(c["class"] for c in picked)), "dir": os.path.join(kit, "req13")}


def score_req13(sheet: str, key: str = os.path.join(KEY_DIR, "khoa_req13.csv"), target: float = 0.70) -> dict[str, Any]:
    """Phieu da dien (o 1-5 hoac ten loai) vs khoa -> ti le dung theo nguoi cham + gop (Wilson 95 %)."""
    def read(p: str) -> list[dict]:
        with open(p, encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    ans = {r["ma_anh"]: int(r["dap_an_so"]) for r in read(key)}
    sheet_rows = read(sheet)
    raters = [c for c in (sheet_rows[0].keys() if sheet_rows else []) if c.startswith("nguoi_cham")]
    khu_on = CLASSES.index("khu_on") + 1

    def tally(skip_khu_on: bool) -> tuple[dict, int, int]:
        per, tot_k, tot_n = {}, 0, 0
        for rc in raters:
            k = n = 0
            for r in sheet_rows:
                v = str(r.get(rc) or "").strip().lower()
                if not v or r["ma_anh"] not in ans or (skip_khu_on and ans[r["ma_anh"]] == khu_on):
                    continue
                g = int(v) if v.isdigit() else (CLASSES.index(v) + 1 if v in CLASSES else None)
                if g is None:
                    continue
                n += 1
                k += int(g == ans[r["ma_anh"]])
            if n:
                per[rc] = {"dung": k, "tong": n, "ti_le": round(k / n, 3)}
                tot_k, tot_n = tot_k + k, tot_n + n
        return per, tot_k, tot_n

    def pooled(k: int, n: int) -> dict:
        lo, hi = C.wilson(k, n) if n else (0.0, 1.0)
        return {"dung": k, "tong": n, "ti_le": round(k / n, 3) if n else None, "wilson95": [round(lo, 3), round(hi, 3)]}

    per, tot_k, tot_n = tally(False)
    _, xk, xn = tally(True)            # anh khu on ca file lo loai qua mau xanh + 'thao tac ca file' -> bao cao ca hai
    return {"per_rater": per, "pooled": pooled(tot_k, tot_n), "pooled_excl_khu_on": pooled(xk, xn),
            "n_raters": len(per), "dat": bool(tot_n and tot_k / tot_n >= target and len(per) >= 3),
            "dat_excl_khu_on": bool(xn and xk / xn >= target and len(per) >= 3)}


README = """# Bộ nghe mù + bộ chấm Req 13 (VoiceAudit, cập nhật {date})

Thư mục này đưa cho NGƯỜI NGHE / NGƯỜI CHẤM. **Không mở thư mục `listen_kit_khoa/`** (đáp án) trước khi điền xong phiếu.
Âm thanh chỉ dùng nội bộ nghiên cứu, không phát tán lại.
Các lệnh dưới đây chạy trong Git-Bash từ thư mục gốc dự án, luôn qua khoá chạy chung:
`bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.listen_kit <lệnh>`.

## 1. Nghe mù (`nghe/`) — kiểm nhãn "tệp thật" (Req 4)
- `nghe/am_thanh/N001.wav …`: {n_clips} đoạn ngắn (thường 6–7 s; đoạn "cả tệp" là nguyên tệp 4–20 s).
  Một phần là chỗ hệ thống báo nhầm trên tệp được coi là thật, một phần là đoạn đối chứng chọn ngẫu nhiên — bạn KHÔNG biết đoạn nào là đoạn nào.
- `nghe/phieu_nghe.csv` (mở bằng Excel): mỗi dòng một đoạn. `pham_vi = doan`: chú ý khoảng `bat_dau_s – ket_thuc_s`
  (giây, tính từ đầu đoạn nghe, thường 0,5–2 s). `pham_vi = ca_tep`: nghe cả đoạn.
- Điền các cột (để trống = chưa nghe):
  - `co_vet_cat`: nghe thấy chỗ cắt / nối / nhảy cảnh không? `co` / `khong` / `khong_ro`
  - `co_nhac_nen`: có nhạc nền hoặc hiệu ứng âm thanh không? `co` / `khong` / `khong_ro`
  - `doi_nguoi_noi`: có đổi người nói (hai giọng khác nhau) trong/quanh chỗ đánh dấu không? `co` / `khong` / `khong_ro`
  - `nghe_nhu_khu_on`: tiếng nghe "bị lọc" (khử ồn, méo, ù, rè kiểu ứng dụng gọi) không? `co` / `khong` / `khong_ro`
  - `muc_tu_tin_1_3`: 1 = đoán, 2 = khá chắc, 3 = chắc chắn. `ghi_chu`: tự do.
- Nên nghe bằng tai nghe, mỗi đoạn tối đa 3 lần, không xem phổ. Hai người nghe độc lập thì tốt hơn (mỗi người một bản phiếu).

## 2. Bộ chấm Req 13 (`req13/`) — "Bằng chứng dễ hiểu"
- `req13/anh/A001.png …`: {n_img} ảnh bằng chứng do hệ thống vẽ (dải tiêu đề chứa tên loại đã bị che trắng).
- `req13/phieu_req13.csv`: mỗi người chấm (≥ 3 người NGOÀI nhóm) điền một cột `nguoi_cham_k` bằng một số 1–5:
  **1 = Khử ồn · 2 = Đổi giọng · 3 = Ghép lời thật · 4 = Xoá lời · 5 = Giọng máy**.
  Chỉ nhìn ảnh, không nghe, không trao đổi với nhau.
- Chấm: `bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.listen_kit score
  listen_kit/req13/phieu_req13.csv` (chỉ tiêu: tỉ lệ đúng gộp ≥ 70 % với ≥ 3 người chấm; lệnh báo cả tỉ lệ khi BỎ các ảnh
  khử ồn, vì màu xanh + "thao tác cả file" gần như lộ loại đó).
- Cách chọn ảnh: CHỈ lấy dòng thao tác mà hệ thống nhận ĐÚNG loại thật của tệp (bộ e2e9), tối đa 8 ảnh mỗi loại.
  Hiện có {img_classes}; chưa có ảnh "Giọng máy" (hệ thống chưa nhận đúng ca giọng máy nào trong e2e9)
  nên bộ này CHƯA đo đúng chỉ số 1-trong-5 của hợp đồng; phiếu vẫn cho chọn 1 trong 5. Mọi con số Req 13 từ bộ này phải
  ghi kèm hai giới hạn đó (chỉ ca nhận đúng, thiếu loại Giọng máy).
- Lưu ý: màu vùng tô mã hoá mức độ (xanh = vô hại = chỉ có khử ồn cả file; cam = sai lệch) và dòng chữ "thao tác cả file"
  nói phạm vi — đó là một phần thiết kế bằng chứng, không phải nhãn; ghi lại khi báo cáo kết quả.
{cao}"""

README_CAO = """
## 3. Nghe mù mẫu dữ liệu cào (`cao/`) — kiểm 5 % nguồn cào (B-12)
- `cao/am_thanh/S001.wav …`: {n_cao} đoạn (tối đa 30 s) lấy ngẫu nhiên khoảng 5 % bản gốc mỗi kênh cào (ít nhất 1 đoạn
  mỗi kênh). Nguồn, URL, giấy phép chỉ có trong khoá (`listen_kit_khoa/khoa_cao.csv`), không có trong phiếu.
- `cao/phieu_cao.csv`: điền `co` / `khong` / `khong_ro` cho các cột
  - `co_vet_cat_an`: có chỗ cắt / ghép ẩn (nhảy tiếng, đổi nền đột ngột) không?
  - `co_nhac_nen`: có nhạc nền / hiệu ứng không?
  - `nhieu_nguoi_noi`: có hơn một người nói không?
  - `giong_may_hay_doc_ban`: nghe như giọng máy (TTS) hoặc đọc bản soạn sẵn không?
  - `chat_luong_kem`: méo, rè, nén nặng, quá nhỏ không? `ghi_chu`: tự do.
- Đoạn nào có `co` ở `co_vet_cat_an` hoặc `giong_may_hay_doc_ban` thì bản gốc đó KHÔNG dùng được làm "tệp thật" (nhãn 01).
- Bộ này đang là bản tạm (dữ liệu cào của WP6 chưa xong); xuất lại bằng lệnh `scrape` khi WP6 xong (bộ cũ được chuyển sang
  `.tmp_v6/codebak/listen_kit_cu/`, không xoá).
"""


def write_readme(kit: str = KIT, key_dir: str = KEY_DIR) -> str:
    """README theo TRANG THAI THAT cua bo (dem tep tren dia) -> export va scrape khong ghi de muc cua nhau."""
    import time
    n_clips = len(glob.glob(os.path.join(kit, "nghe", "am_thanh", "N*.wav")))
    n_img = len(glob.glob(os.path.join(kit, "req13", "anh", "A*.png")))
    n_cao = len(glob.glob(os.path.join(kit, "cao", "am_thanh", "S*.wav")))
    classes: Counter = Counter()
    key = os.path.join(key_dir, "khoa_req13.csv")
    if os.path.exists(key):
        with open(key, encoding="utf-8-sig", newline="") as fh:
            classes = Counter(r.get("dap_an") or "?" for r in csv.DictReader(fh))
    img_classes = (" · ".join(f"{CLASS_VI.get(k, k)} {v}" for k, v in sorted(classes.items())) if classes
                   else f"{n_img} ảnh")
    text = README.format(date=time.strftime("%d/%m/%Y"), n_clips=n_clips, n_img=n_img, img_classes=img_classes,
                         cao=README_CAO.format(n_cao=n_cao) if n_cao else "")
    path = os.path.join(kit, "README.md")
    os.makedirs(kit, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ---------------- B-12: mau nghe 5 % du lieu cao (I4) ----------------
SCRAPE_HOSTS = {os.path.basename(os.path.dirname(p))[len("hosts_"):]: p        # I4 hosts_vi|en + hosts_long_* cua WP6
                for p in sorted(glob.glob(os.path.join(C.BASE, "data", "scrape", "hosts_*", "hosts.jsonl")))}
SCRAPE_COLS = ["ma_doan", "tep", "co_vet_cat_an", "co_nhac_nen", "nhieu_nguoi_noi", "giong_may_hay_doc_ban", "chat_luong_kem",
               "ghi_chu"]
SCRAPE_MAX_S = 30.0


def sample_scrape(recs: Sequence[Mapping[str, Any]], share: float = 0.05, min_n: int = 20, seed: int = 2026) -> list[dict]:
    """Mau tat dinh ~share moi kenh (it nhat 1 tep / kenh, tong >= min_n neu du): thu tu crc32(seed|file) trong kenh."""
    import zlib
    by: dict[str, list] = defaultdict(list)
    for r in recs:
        by[str(r.get("group") or r.get("channel_id") or r.get("source") or "?")].append(r)
    key = lambda r: (zlib.crc32(f"{seed}|{r['file']}".encode("utf-8")), str(r["file"]))  # noqa: E731
    out = []
    for g in sorted(by):
        rs = sorted(by[g], key=key)
        out += rs[:max(1, int(round(share * len(rs))))]
    if len(out) < min(min_n, len(recs)):
        have = {r["file"] for r in out}
        out += [r for r in sorted(recs, key=key) if r["file"] not in have][:min(min_n, len(recs)) - len(out)]
    return out


def export_scrape(kit: str = KIT, key_dir: str = KEY_DIR, share: float = 0.05, seed: int = 2026,
                  hosts: Mapping[str, str] = SCRAPE_HOSTS, archive: str = ARCHIVE) -> dict[str, Any]:
    """B-12: mau nghe mu ~5 % ban goc cao moi kenh (kiem vet cat an / nhac / nhieu nguoi / giong may) -> listen_kit/cao/;
    khoa (url, kenh, giay phep, codec) o listen_kit_khoa/khoa_cao.csv. Chua co hosts.jsonl cua WP6 -> bao, khong loi."""
    import soundfile as sf
    recs = []
    for lang, p in hosts.items():
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                recs += [{**json.loads(l), "lang": lang, "_root": os.path.dirname(p)} for l in fh if l.strip()]
    if not recs:
        return {"status": "chua co du lieu cao (I4: data/scrape/hosts_vi|en/hosts.jsonl)", "clips": 0}
    picked = sample_scrape(recs, share, seed=seed)
    random.Random(seed).shuffle(picked)
    _archive_old(os.path.join(kit, "cao"), archive)
    out_dir = os.path.join(kit, "cao", "am_thanh")
    os.makedirs(out_dir, exist_ok=True)
    form, key, missing = [], [], 0
    for k, r in enumerate(picked, 1):
        wav = os.path.join(r["_root"], str(r.get("source") or ""), f"{r['file']}.wav")
        if not os.path.exists(wav):
            missing += 1
            continue
        y, sr = sf.read(wav, dtype="float32", always_2d=True)
        ma = f"S{k:03d}"
        sf.write(os.path.join(out_dir, ma + ".wav"), y.mean(1)[: int(SCRAPE_MAX_S * sr)], sr, subtype="PCM_16")
        form.append({"ma_doan": ma, "tep": f"am_thanh/{ma}.wav"})
        key.append({"ma_doan": ma, "file": r["file"], "lang": r["lang"], "group": r.get("group"), "url": r.get("url"),
                    "channel_id": r.get("channel_id"), "license": r.get("license"), "codec": r.get("codec"),
                    "dur": r.get("dur")})
    _write_csv(os.path.join(kit, "cao", "phieu_cao.csv"), SCRAPE_COLS, form)
    _write_csv(os.path.join(key_dir, "khoa_cao.csv"), ["ma_doan", "file", "lang", "group", "url", "channel_id", "license",
                                                       "codec", "dur"], key)
    no_lic = sum(1 for r in recs if not r.get("license"))
    readme = write_readme(kit, key_dir)                  # README them muc 3 (cao), giu muc 1-2
    return {"hosts": len(recs), "groups": len({r.get("group") for r in recs}), "clips": len(form), "missing_wav": missing,
            "hosts_without_license": no_lic, "dir": os.path.join(kit, "cao"), "readme": readme}


def export(engine: str = ENGINE, kit: str = KIT, key_dir: str = KEY_DIR, per_class: int = 8, seed: int = 2026) -> dict[str, Any]:
    a = export_listen(engine, kit, key_dir, seed)
    b = export_req13(E2E, kit, key_dir, per_class, seed)
    return {"nghe": a, "req13": b, "readme": write_readme(kit, key_dir), "key_dir": key_dir}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Bo nghe mu (REQ-4) + bo cham Req 13")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--engine", default=ENGINE)
    p.add_argument("--per-class", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    p = sub.add_parser("score")
    p.add_argument("sheet")
    p.add_argument("--key", default=os.path.join(KEY_DIR, "khoa_req13.csv"))
    p = sub.add_parser("scrape", help="B-12: mau nghe mu ~5 %% du lieu cao moi kenh")
    p.add_argument("--share", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2026)
    sub.add_parser("readme", help="viet lai README theo tep dang co (khong xuat lai am thanh / anh)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "export":
        out: Any = export(a.engine, KIT, KEY_DIR, a.per_class, a.seed)
    elif a.cmd == "scrape":
        out = export_scrape(KIT, KEY_DIR, a.share, a.seed)
    elif a.cmd == "readme":
        out = {"readme": write_readme(KIT, KEY_DIR)}
    elif a.cmd == "score":
        out = score_req13(a.sheet, a.key)
    else:
        from v6.data import test_groups
        sys.exit(test_groups.run_all("listen"))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
