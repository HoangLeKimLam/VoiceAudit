# -*- coding: utf-8 -*-
"""Kiem mot bo du lieu (manifest H1) TRUOC KHI NOP / GHEP — chi kiem nhung gi can de cac bo ghep duoc va nhan khong lan:
cau truc · id khong trung giua cac bo · meta.person (fold do he thong tu tinh: person_fold = crc32(person) % 5) · ma nhan va tinh nhat quan level1 / level2 / edits · truong bat buoc
theo loai thao tac · ranh gioi nhan giua cac lane · chinh sach nhai giong · tap khoa. KHONG kiem cach lam (mo hinh, cong cu sinh).

Quy uoc (CHUNG.md §4 H1 H2, DU_LIEU.md §5, quyet dinh 18):
  ban ghi   {id, kind, cfg, src, wav, meta{lang vi|en, source, person, speaker, fold, level1[], level2[], edits[]}}
  level2    02_nen_toan_cuc 02_nen_cuc_bo 02_giong_toan_cuc 02_giong_cuc_bo 03_ghep_giong_that 03_chen_giong_may
  level1    [] level2 -> ["01_nguyen_ban"]; co 02_* -> "02_ai_hau_ky"; co 03_* -> "03_ghep_chen"
  edit      {level2, kind segment|global, start/end (segment)}; op -> sub: ghep_thay thay_that · ghep_chen chen_that · xoa_loi cat_bo ·
            may_thay thay_the · may_chen chen_them; may_ca_file = global; xoa loi = 03_ghep_giong_that + sub "cat_bo";
            giong may = 03_chen_giong_may + engine (tts / engine); chuyen giong AI = 02_giong_* (khong phai 03);
            WebRTC NS phai ghi ns_level; muc <= 2 khong phai thao tac (khu on don so cua ung dung -> 01)

CLI: python -m v6.data.kiem_bo <manifest.json> [--moi] [--khong-lockbox]   (--moi: bo moi, kiem chat; mac dinh: bo cu, mot so loi -> canh bao)
     python -m v6.data.kiem_bo --tat-ca                                       (moi manifest data_labels/v6: id trung giua cac bo, nhan)
     python -m v6.data.kiem_bo selftest
Dung trong ma hoc: from v6.data.kiem_bo import dung_de_hoc  ->  ban ghi nao duoc dung de hoc / cham."""
from __future__ import annotations

import collections
import glob
import io
import json
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
L2 = {"02_nen_toan_cuc", "02_nen_cuc_bo", "02_giong_toan_cuc", "02_giong_cuc_bo", "03_ghep_giong_that", "03_chen_giong_may"}
NO_CLONE = {"celeb", "vss", "vsasv", "vivoice", "yt", "ia", "ccc", "fosdem", "scrape"}
GEN_KEYS = ("tts", "tts_engine", "engine", "vc", "vc_engine", "gen_engine")
OP_SUB = {"ghep_thay": "thay_that", "ghep_chen": "chen_that", "xoa_loi": "cat_bo", "may_thay": "thay_the", "may_chen": "chen_them"}   # DU_LIEU.md §2.3
ERR, WARN = "LOI", "CANH BAO"


def dung_de_hoc(rec: dict) -> bool:
    """Ban ghi duoc dung de hoc / cham (khong tinh tap khoa — kiem rieng bang v6.data.lockbox): bo tam_loai, chinh_sach, ngoai_pham_vi."""
    m = rec.get("meta") or {}
    return not (m.get("tam_loai") or m.get("chinh_sach") or m.get("ngoai_pham_vi"))


def _is_webrtc(e: dict) -> bool:
    return e.get("tool") == "WebRTC NS" or e.get("family") == "N7" or e.get("tool_family") == "N7"


def kiem_ban_ghi(r: dict, moi: bool) -> list[tuple[str, str]]:
    """[(muc do, thong bao)] cho mot ban ghi (ham thuan)."""
    out: list[tuple[str, str]] = []
    strict = ERR if moi else WARN
    for k in ("id", "kind", "cfg", "wav", "meta"):
        if k not in r:
            out.append((ERR, f"thieu truong {k}"))
    m = r.get("meta") or {}
    if m.get("lang") not in ("vi", "en"):
        out.append((ERR, "meta.lang phai la vi hoac en"))
    for k in ("source", "person", "speaker"):
        if not m.get(k):
            out.append((ERR, f"thieu meta.{k} (can de chia fold theo nguoi noi va kiem nhai giong)"))
    eds = [e for e in (m.get("edits") or []) if isinstance(e, dict)]
    l2 = list(m.get("level2") or [])
    for x in l2:
        if x == "03_xoa_loi":
            out.append((ERR, "ma 03_xoa_loi khong dung trong du lieu — xoa loi = 03_ghep_giong_that + edits[].sub = \"cat_bo\""))
        elif x not in L2:
            out.append((ERR, f"ma level2 la: {x}"))
    if set(l2) != {e.get("level2") for e in eds}:
        out.append((ERR, f"level2 {sorted(set(l2))} khong khop level2 cua edits {sorted({str(e.get('level2')) for e in eds})}"))
    want_l1 = (["02_ai_hau_ky"] if any(str(x).startswith("02") for x in l2) else []) + (["03_ghep_chen"] if any(str(x).startswith("03") for x in l2) else [])
    if sorted(m.get("level1") or []) != sorted(want_l1 or ["01_nguyen_ban"]):
        out.append((strict, f"level1 {m.get('level1')} nen la {want_l1 or ['01_nguyen_ban']}"))
    dur = ((m.get("k1") or {}).get("measures") or {}).get("do_dai_s")
    if isinstance(dur, (int, float)) and not 3 <= dur <= 600:
        out.append((WARN, f"do dai {dur:.1f} s ngoai 3 giay - 10 phut (se ra 04, khong cham)"))
    for e in eds:
        lv, kd = str(e.get("level2")), e.get("kind")
        if kd not in ("segment", "global"):
            out.append((ERR, f"edit kind phai la segment / global: {kd}"))
        if kd == "segment":
            s, t = e.get("start"), e.get("end")
            if not (isinstance(s, (int, float)) and isinstance(t, (int, float)) and 0 <= s < t):
                out.append((ERR, "edit segment can 0 <= start < end (giay)"))
            elif isinstance(dur, (int, float)) and t > dur + 0.05:
                out.append((ERR, f"edit ket thuc {t:.2f} s sau cuoi tep {dur:.2f} s"))
        if lv == "03_ghep_giong_that" and kd != "segment":
            out.append((ERR, "ghep / xoa loi that luon la mot doan (segment)"))
        if lv == "03_ghep_giong_that" and (e.get("op") == "xoa_loi" or e.get("sub") == "cat_bo") and e.get("sub") != "cat_bo":
            out.append((ERR, "xoa loi phai co sub = \"cat_bo\""))
        op = e.get("op")
        if op in OP_SUB and e.get("sub") != OP_SUB[op]:
            out.append((strict, f"op {op} can sub = \"{OP_SUB[op]}\" (DU_LIEU.md §2.3)"))
        if op == "may_ca_file" and kd != "global":
            out.append((ERR, "giong may ca tep (may_ca_file) phai la kind = global"))
        if lv == "03_chen_giong_may" and not any(e.get(k) for k in GEN_KEYS):
            out.append((strict, "giong may can ghi cong cu sinh (tts / engine) de chia tap cong cu chua thay"))
        if (e.get("op") == "doi_giong_ai" or e.get("vc") or e.get("vc_engine")) and lv.startswith("03"):
            out.append((ERR, "chuyen giong AI la 02_giong_* (loi van la cua nguoi noi), khong phai 03"))
        if _is_webrtc(e):
            nl = e.get("ns_level")
            if nl is None:
                out.append((strict, "WebRTC NS phai ghi ns_level (muc <= 2 = 01, muc 3 = 02)"))
            elif isinstance(nl, int) and nl <= 2:
                out.append((ERR, "WebRTC NS muc <= 2 la khu on don so cua ung dung -> khong phai thao tac (bo khoi edits, nhan 01)"))
        if any(e.get(k) for k in GEN_KEYS) and str(m.get("source", "")).lower() in NO_CLONE and not m.get("chinh_sach"):
            out.append((ERR, f"nhai giong nguoi thuoc nguon {m.get('source')} — trai chinh sach (chi nguoi trong kho nghien cuu)"))
    return out


def all_ids(skip: str | None = None) -> collections.Counter:
    c = collections.Counter()
    for mf in glob.glob(os.path.join(BASE, "data_labels", "v6", "*", "manifest.json")):
        if skip and os.path.normcase(os.path.abspath(mf)) == os.path.normcase(os.path.abspath(skip)):
            continue
        try:
            c.update(str(r.get("id")) for r in json.load(io.open(mf, encoding="utf-8")))
        except (OSError, ValueError):
            pass
    return c


def kiem_bo(path: str, moi: bool = False, lockbox: bool = True, fold_fn=None, ids_khac: collections.Counter | None = None) -> dict:
    recs = json.load(io.open(path, encoding="utf-8"))
    if not isinstance(recs, list):
        return {"loi": [("*", ERR, "manifest phai la danh sach ban ghi")], "n": 0}
    out = []
    seen = collections.Counter(str(r.get("id")) for r in recs if isinstance(r, dict))
    other = all_ids(skip=path) if ids_khac is None else ids_khac
    guard = None
    if lockbox:
        sys.path.insert(0, BASE)
        from v6.data.lockbox import Guard  # noqa: E402
        guard = Guard(set_name=os.path.basename(os.path.dirname(path)), allow_unknown_ids=True)
    flags = collections.Counter()
    root = os.path.join(BASE, "data_labels", "realistic_v4")
    for r in recs:
        if not isinstance(r, dict):
            out.append(("?", ERR, "ban ghi khong phai dict"))
            continue
        rid = str(r.get("id"))
        if seen[rid] > 1:
            all_tam = all((x.get("meta") or {}).get("tam_loai") for x in recs if isinstance(x, dict) and str(x.get("id")) == rid)
            out.append((rid, WARN if all_tam else ERR, "id trung trong bo" + (" (da gan tam_loai, khong dung)" if all_tam else "")))
        if other.get(rid):
            out.append((rid, ERR, "id trung voi bo khac — dung tien to id rieng"))
        for lvl, msg in kiem_ban_ghi(r, moi):
            out.append((rid, lvl, msg))
        m = r.get("meta") or {}
        p = str(m.get("person") or "")
        if p and ":" not in p:
            out.append((rid, WARN, "meta.person nen co dang <nguon>:<ma nguoi> (cung mot nguoi phai cung chuoi o moi bo — fold tinh tu chuoi nay)"))
        if r.get("wav") and not os.path.exists(os.path.normpath(os.path.join(root, r["wav"]))):
            out.append((rid, WARN, "khong thay tep wav (duong dan tuong doi data_labels/realistic_v4)"))
        for f in ("tam_loai", "chinh_sach", "ngoai_pham_vi"):
            if m.get(f):
                flags[f] += 1
        if guard is not None and guard.why(r):
            flags["tap_khoa"] += 1
    return {"n": len(recs), "loi": out, "co": dict(flags)}


def report(path: str, res: dict) -> str:
    L = [f"== {path}: {res['n']} ban ghi"]
    by = collections.Counter((lvl, msg) for _, lvl, msg in res["loi"])
    for (lvl, msg), c in sorted(by.items(), key=lambda t: (t[0][0] != ERR, -t[1])):
        ex = next(rid for rid, l2, m2 in res["loi"] if l2 == lvl and m2 == msg)
        L.append(f"  [{lvl}] {c:6d} x {msg}   (vd {ex})")
    if res.get("co"):
        L.append("  co dac biet (khong dung de hoc): " + ", ".join(f"{k} {v}" for k, v in res["co"].items())
                 + ("  — bo co muc TAP KHOA: chi dung de cham cuoi, khong hoc / chon nguong" if res["co"].get("tap_khoa") else ""))
    n_err = sum(1 for _, lvl, _ in res["loi"] if lvl == ERR)
    L.append(f"  => {'DAT' if n_err == 0 else f'CHUA DAT ({n_err} loi)'}")
    return "\n".join(L)


def selftest() -> None:
    ok = {"id": "p3x_1", "kind": "nen_toan_cuc", "cfg": "goc", "src": "p3x_1", "wav": "../v6/x/a.wav",
          "meta": {"lang": "vi", "source": "vox", "person": "vox:id1", "speaker": "vox:id1", "fold": 0,
                   "level1": ["02_ai_hau_ky"], "level2": ["02_nen_toan_cuc"],
                   "edits": [{"level2": "02_nen_toan_cuc", "kind": "global", "tool": "DeepFilterNet3"}]}}
    assert not [x for x in kiem_ban_ghi(ok, True) if x[0] == ERR]
    bad = json.loads(json.dumps(ok))
    bad["meta"]["edits"] = [{"level2": "02_nen_toan_cuc", "kind": "global", "tool": "WebRTC NS", "ns_level": 2}]
    assert any("WebRTC NS muc <= 2" in m for _, m in kiem_ban_ghi(bad, True))
    xo = json.loads(json.dumps(ok))
    xo["meta"].update(level1=["03_ghep_chen"], level2=["03_xoa_loi"], edits=[{"level2": "03_xoa_loi", "kind": "segment", "start": 1, "end": 1.5}])
    assert any("03_xoa_loi" in m for _, m in kiem_ban_ghi(xo, True))
    vc = json.loads(json.dumps(ok))
    vc["meta"].update(level1=["03_ghep_chen"], level2=["03_chen_giong_may"], edits=[{"level2": "03_chen_giong_may", "kind": "global", "op": "doi_giong_ai", "vc": "freevc24"}])
    assert any("chuyen giong AI" in m for _, m in kiem_ban_ghi(vc, True))
    cl = json.loads(json.dumps(ok))
    cl["meta"].update(source="celeb", level1=["03_ghep_chen"], level2=["03_chen_giong_may"],
                      edits=[{"level2": "03_chen_giong_may", "kind": "segment", "start": 1, "end": 2, "tts": "viXTTS"}])
    assert any("trai chinh sach" in m for _, m in kiem_ban_ghi(cl, True))
    ms = json.loads(json.dumps(ok))
    ms["meta"].update(level1=["03_ghep_chen"], level2=["03_chen_giong_may"],
                      edits=[{"level2": "03_chen_giong_may", "kind": "segment", "start": 1, "end": 2, "op": "may_chen", "tts": "kokoro"}])
    assert any(lvl == ERR and "sub" in m for lvl, m in kiem_ban_ghi(ms, True))
    assert not [x for x in kiem_ban_ghi(ms, False) if x[0] == ERR]                       # bo cu: chi canh bao
    ms["meta"]["edits"][0]["sub"] = "chen_them"
    assert not [x for x in kiem_ban_ghi(ms, True) if x[0] == ERR]
    mf = json.loads(json.dumps(ms)); mf["meta"]["edits"][0].update(op="may_ca_file", sub=None)
    assert any("may_ca_file" in m for _, m in kiem_ban_ghi(mf, True))
    l1 = json.loads(json.dumps(ok)); l1["meta"]["level1"] = ["01_nguyen_ban"]
    assert any("level1" in m for lvl, m in kiem_ban_ghi(l1, True) if lvl == ERR)
    assert dung_de_hoc(ok) and not dung_de_hoc({"meta": {"tam_loai": "x"}})
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "manifest.json")
        two = [ok, json.loads(json.dumps(ok))]
        io.open(p, "w", encoding="utf-8").write(json.dumps(two))
        res = kiem_bo(p, moi=True, lockbox=False, fold_fn=lambda m: 0, ids_khac=collections.Counter({"p3x_1": 1}))
        msgs = {m for _, _, m in res["loi"]}
        assert "id trung trong bo" in msgs and any("bo khac" in m for m in msgs)
        two[1]["id"] = "p3x_2"; two[1]["meta"]["person"] = "khongcohaicham"
        io.open(p, "w", encoding="utf-8").write(json.dumps(two))
        res2 = kiem_bo(p, moi=True, lockbox=False, ids_khac=collections.Counter())
        assert any("meta.person" in m for _, _, m in res2["loi"]) and not any("trung" in m for _, _, m in res2["loi"])
    print("kiem_bo selftest dat")


def main(argv: list[str] | None = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        selftest(); return 0
    moi, lockbox = "--moi" in a, "--khong-lockbox" not in a
    paths = sorted(glob.glob(os.path.join(BASE, "data_labels", "v6", "*", "manifest.json"))) if "--tat-ca" in a else [x for x in a if not x.startswith("--")]
    if not paths:
        print(__doc__); return 2
    n_err = 0
    for p in paths:
        res = kiem_bo(p, moi=moi, lockbox=lockbox)
        print(report(p, res))
        n_err += sum(1 for _, lvl, _ in res["loi"] if lvl == ERR)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
