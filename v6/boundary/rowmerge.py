# -*- coding: utf-8 -*-
"""Gop dong thao tac tu nhieu nguon (ranh gioi TCN, vung TCN, dau ca file): dung chung cho nghien cuu (fuse8 / fuse9) va san pham (v6/product)."""
from __future__ import annotations

from v6.boundary import detect as D

REAL_SPLICE, TTS_SEG = "03_ghep_giong_that", "03_chen_giong_may"
OVERLAP = 0.3


def merge(rows: list, extra: list) -> list:
    """Them dong `extra` vao `rows` neu chua co dong cung loai chong > 30 % (dong toan cuc: chi bi bo neu DA CO dong TOAN CUC cung loai - dong doan khong chan dong ca file)."""
    out = list(rows)
    for r in extra:
        same = [x for x in out if x["level2"] == r["level2"]]
        if r["kind"] == "global" and any(x["kind"] == "global" for x in same):
            continue
        if r["kind"] == "segment" and any(D._ov((r["start"], r["end"]), (x["start"], x["end"])) > OVERLAP for x in same if x["kind"] == "segment"):
            continue
        out.append(r)
    return out


def snap_merge(rows: list, regions: list) -> list:
    """Dong ranh gioi GHEP (khong phai moi noi xoa loi) chong len mot dong VUNG cung loai (> 30 %) lay ranh gioi cua VUNG co diem cao nhat; dong VUNG con lai them vao nhu merge()."""
    out = []
    for r in rows:
        g = [x for x in regions if x["level2"] == r["level2"] and D._ov((r["start"], r["end"]), (x["start"], x["end"])) > OVERLAP] \
            if r["kind"] == "segment" and r.get("via") == "ranh_gioi_ghep" else []
        x = max(g, key=lambda v: v["nm"]) if g else None
        out.append({**r, "start": x["start"], "end": x["end"], "nm": max(r["nm"], x["nm"]), "via": "ranh_gioi+vung"} if x else r)
    return merge(out, regions)


def resolve_types(regions: list) -> list:
    """Hai dong VUNG 03 khac loai (doan ghep that / giong may) chong > 30 % la CUNG mot doan: chi giu loai co diem cao hon."""
    keep = []
    for x in sorted(regions, key=lambda v: -v["nm"]):
        pair = x["level2"] in (REAL_SPLICE, TTS_SEG)
        if pair and any(k["level2"] in (REAL_SPLICE, TTS_SEG) and k["level2"] != x["level2"] and D._ov((k["start"], k["end"]), (x["start"], x["end"])) > OVERLAP for k in keep):
            continue
        keep.append(x)
    return sorted(keep, key=lambda v: v["start"])


def typed_merge(rows: list, regions: list) -> list:
    """Nhu snap_merge nhung con PHAN LOAI: dong ranh gioi GHEP (khong phai xoa loi) chong > 30 % len mot dong VUNG doan ghep that / giong may lay LOAI va RANH GIOI
    cua vung (hai loai cung chong thi lay vung co diem cao hon). Giong may chen vao tao hai mep nhu ghep that, nen neu khong phan loai thi cung mot thao tac thanh hai dong."""
    regions, out = resolve_types(regions), []
    for r in rows:
        g = [x for x in regions if x["level2"] in (REAL_SPLICE, TTS_SEG) and D._ov((r["start"], r["end"]), (x["start"], x["end"])) > OVERLAP] \
            if r["kind"] == "segment" and r.get("via") == "ranh_gioi_ghep" else []
        x = max(g, key=lambda v: v["nm"]) if g else None
        out.append({**r, "level2": x["level2"], "start": x["start"], "end": x["end"], "nm": max(r["nm"], x["nm"]), "via": "ranh_gioi+vung"} if x else r)
    return snap_merge(out, regions)
