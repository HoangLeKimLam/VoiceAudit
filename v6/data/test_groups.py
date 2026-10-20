# -*- coding: utf-8 -*-
"""Kiem thu nhe cho v6.data (groups / lockbox / expand_hosts / listen_kit) + C.group_of / C.fold_of_group.
Khong nap model, chi du lieu gia trong .tmp_v6/selftest_wp5 (+ doc hosts*.jsonl that cho tra cuu AMI). Chay:
    bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.test_groups [chuoi_loc]
Ham test_* tuong thich pytest nhung khong can pytest."""
from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sys
import time
import traceback
import zlib
from collections import Counter
from typing import Any

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402
from v6.data import expand_hosts as X  # noqa: E402
from v6.data import groups as G  # noqa: E402
from v6.data import listen_kit as K  # noqa: E402
from v6.data import lockbox as L  # noqa: E402

WORK = os.path.join(C.TMP, "selftest_wp5")


def _fresh(name: str) -> str:
    d = os.path.join(WORK, name)
    if os.path.isdir(d):
        shutil.rmtree(d)                    # chi thu muc kiem thu cua chinh module nay
    os.makedirs(d)
    return d


def _wav(path: str, dur: float = 2.0, f0: float = 220.0, seed: int = 0) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.default_rng(seed)
    t = np.arange(int(C.SR * dur)) / C.SR
    y = 0.1 * np.sin(2 * np.pi * f0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.7 * t + seed)) + 0.003 * rng.standard_normal(len(t))
    sf.write(path, y.astype(np.float32), C.SR, subtype="PCM_16")
    return path


def _raises(exc: type, fn, *a, **k) -> BaseException:
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} khong raise {exc.__name__}")


# ================= common.group_of / fold_of_group =================
def test_group_of_sources() -> None:
    g = C.group_of
    assert g({"source": "vss", "person": "vss:nguyenkhangofficial", "speaker": "vss:nguyenkhangofficial:ep1"}) == "vss:nguyenkhangofficial"
    assert g({"source": "edacc", "person": "edacc:EDACC-C08-A", "speaker": "edacc:EDACC-C08"}) == "edacc:EDACC-C08"
    assert g({"source": "edacc", "person": "edacc:EDACC-C31-B"}) == "edacc:EDACC-C31"            # chi co nguoi
    assert g({"source": "ami", "speaker": "ami:ES2004c", "person": "ami:MEE014", "host_file": "ami_test-00002_00329"}) == "ami:ES2004"
    assert g({"source": "ami", "speaker": "ami:TS3003a", "person": "ami:TS3003a", "host_file": "TS3003a_1"}) == "ami:TS3003"
    assert g({"source": "ami", "speaker": "ami:EN2002c", "person": "ami:FEO065"}) == "ami:EN2002"
    assert g({"source": "voxp", "person": "voxp:None", "speaker": "voxp:None", "host_file": "voxp_test-00000_00007"}) == \
        "voxp:None:voxp_test-00000_00007"
    assert g({"source": "voxp", "person": "voxp:124961"}) == "voxp:124961"
    assert g({"source": "vox", "person": "vox:id00083", "speaker": "vox:id00083"}) == "vox:id00083"
    assert g({"source": "celeb", "person": "celeb:id00963"}) == "celeb:id00963"
    assert g({"speaker": "VIVOSSPK12"}) == "VIVOSSPK12"                                          # v4
    assert g({"group": "scrape:UCabc", "person": "UCabc", "source": "yt"}) == "scrape:UCabc"
    assert g({"channel_id": "UCxyz", "source": "yt"}) == "scrape:UCxyz"
    assert g({"meta": {"source": "vox", "person": "vox:id1"}}) == "vox:id1"                       # nhan ca dong manifest
    assert g({"source": "vss", "person": "vss:kenh_TS3003a"}) == "vss:kenh_TS3003a"             # khong nham thanh AMI
    _raises(ValueError, g, {})
    _raises(ValueError, g, {"source": "x"})


def test_group_of_ami_person_lookup_from_hosts() -> None:
    """Nguoi AMI khong kem ma hop -> tra series qua hosts_en (du lieu that). Bo qua neu khong co hosts_en."""
    tab = C._ami_person_series()
    if not tab:
        print("    (bo qua: khong co hosts_en)")
        return
    person, series = sorted(tab.items())[0]
    assert C.group_of({"source": "ami", "person": person, "host_file": "khong_co"}) == f"ami:{series}"
    # moi nguoi AMI chi thuoc mot series (tien de cua nhom series = hop nhat nhom nguoi)
    multi = {}
    for src, cl, p in C._host_table().values():
        if src == "ami":
            multi.setdefault(p, set()).add(C.ami_series(cl))
    assert all(len(v) == 1 for v in multi.values()), {k: v for k, v in multi.items() if len(v) > 1}


def test_fold_of_group_crc_and_table() -> None:
    old = C._frozen_group_folds
    try:
        C._frozen_group_folds = lambda: {}                                        # type: ignore[assignment]
        for grp in ("vox:id1", "ami:ES2004", "voxp:None:x", "scrape:UC1"):
            assert C.fold_of_group(grp) == zlib.crc32(("fold:" + grp).encode("utf-8")) % C.N_FOLDS == C.crc_fold_of_group(grp)
        C._frozen_group_folds = lambda: {"vox:id1": 4}                           # type: ignore[assignment]
        assert C.fold_of_group("vox:id1") == 4 == C.fold_of_group("vox:id1", strict=True)
        assert C.fold_of_group("vox:id2") == zlib.crc32(b"fold:vox:id2") % C.N_FOLDS
        _raises(C.V6Error, C.fold_of_group, "vox:id2", strict=True)            # code hoc: chi fold tu bang
        _raises(ValueError, C.fold_of_group, "")
        # fold du phong doc lap voi chon lockbox cao (crc32 % 5 == 0): nhom lockbox phai roi vao NHIEU fold
        lock = [g for g in (f"yt:UC{i}" for i in range(2000)) if zlib.crc32(g.encode()) % 5 == 0]
        folds = Counter(C.fold_of_group(g) for g in lock)
        assert len(folds) == 5 and min(folds.values()) > len(lock) / 10, folds
    finally:
        C._frozen_group_folds = old


def test_person_fold_unchanged() -> None:
    """detect.person_fold (mo hinh v9) khong doi: van la assign_fold([person])."""
    from v6.boundary import detect as D
    m = {"person": "vss:nguyenkhangofficial", "speaker": "vss:nguyenkhangofficial:x"}
    assert D.person_fold(m) == C.assign_fold(["vss:nguyenkhangofficial"]) == zlib.crc32(b"vss:nguyenkhangofficial") % 5


# ================= groups =================
def _synthetic_counts(seed: int = 0) -> dict:
    rng = random.Random(seed)
    counts = {}
    for i in range(60):
        src = ["vi:vox", "vi:celeb", "en:voxp", "en:edacc"][i % 4]
        counts[f"{src}:{i}"] = Counter({src: rng.randint(5, 40)})
    counts["vi:vss:kenh"] = Counter({"vi:vss": 900})                       # nhom khong lo
    return counts


def test_assign_stratified_balanced_and_deterministic() -> None:
    counts = _synthetic_counts()
    a = G.assign_stratified(counts)
    b = G.assign_stratified(dict(reversed(list(counts.items()))))           # thu tu dau vao khong anh huong
    assert a == b
    assert set(a.values()) <= set(range(5)) and len(a) == len(counts)
    for s in ("vi:vox", "vi:celeb", "en:voxp", "en:edacc"):
        per = [0] * 5
        for g, c in counts.items():
            per[a[g]] += c.get(s, 0)
        assert min(per) > 0 and max(per) / min(per) < 1.6, (s, per)
    # fold chua nhom VSS khong lo nhan it nhom khac hon (can bang cot tong)
    fv = a["vi:vss:kenh"]
    n_in = sum(1 for g, f in a.items() if f == fv and g != "vi:vss:kenh")
    n_other = [sum(1 for g, f in a.items() if f == k) for k in range(5) if k != fv]
    assert n_in <= min(n_other), (n_in, n_other)


def test_assign_stratified_pins() -> None:
    counts = _synthetic_counts(1)
    pins = {"vi:vox:0": 3, "en:voxp:2": 0, "khong_co": 1}
    a = G.assign_stratified(counts, pinned=pins)
    assert a["vi:vox:0"] == 3 and a["en:voxp:2"] == 0 and "khong_co" not in a
    _raises(ValueError, G.assign_stratified, counts, 5, {"vi:vox:0": 7})


def test_balance_reports_person_split_and_moves() -> None:
    items = [{"id": "a", "set": "natvi", "cfg": "goc", "group": "g1", "person": "p1", "stratum": "vi:vox", "lang": "vi",
              "real": True, "pf": 0},
             {"id": "b", "set": "natvi", "cfg": "goc", "group": "g2", "person": "p1", "stratum": "vi:vox", "lang": "vi",
              "real": False, "pf": 1},
             {"id": "c", "set": "naten", "cfg": "goc", "group": "g3", "person": "voxp:None", "stratum": "en:voxp", "lang": "en",
              "real": True, "pf": 2}]
    bal = G.balance(items, {"g1": 0, "g2": 4, "g3": 2})
    assert bal["persons_in_several_folds"] == 1 and bal["persons_in_several_folds_non_voxp_none"] == ["p1"]
    assert bal["moved_vs_person_fold"] == 1 and bal["real_goc_v2"]["vi"] == [1, 0, 0, 0, 0]
    assert bal["files"][G.ALL] == [1, 0, 1, 0, 1]


def test_build_writes_files_and_respects_freeze() -> None:
    d = _fresh("groups_build")
    items = []
    for i in range(30):
        src = ["vox", "celeb", "voxp"][i % 3]
        lang = "en" if src == "voxp" else "vi"
        for j in range(3):
            items.append({"id": f"id{i}_{j}", "set": "natvi", "cfg": "goc", "group": f"{src}:{i}", "person": f"{src}:{i}",
                          "source": src, "lang": lang, "stratum": f"{lang}:{src}", "real": j == 0, "pf": i % 5})
    items.append(dict(items[0], group="vox:dup"))                           # id trung, nhom khac
    old = G.iter_items
    old_frozen = C.FOLDS_V2_GROUPS
    try:
        G.iter_items = lambda include_v4=True, rules=None: iter(items)      # type: ignore[assignment]
        ids_p, grp_p, md = os.path.join(d, "f.json"), os.path.join(d, "g.json"), os.path.join(d, "r.md")
        r = G.build(out_ids=ids_p, out_groups=grp_p, report=md, include_v4=False)
        assert r["n_ids"] == 90 and r["dups"]["n_dup_ids"] == 1 and r["dups"]["group_conflict"] == ["id0_0"]
        ids = json.load(open(ids_p, encoding="utf-8"))
        grp = json.load(open(grp_p, encoding="utf-8"))["groups"]
        assert ids["id0_1"] == grp["vox:0"] and ids["id0_0"] == grp["vox:dup"]   # dong sau cung thang
        assert os.path.exists(md) and "folds_v2" in open(md, encoding="utf-8").read()
        _raises(C.V6Error, G.build, out_ids=ids_p, out_groups=grp_p, report=md)
        # --force: nhom cu giu fold (doc tu bang) -> tro C.FOLDS_V2_GROUPS ve bang gia
        C.FOLDS_V2_GROUPS = grp_p
        items.append({**items[1], "id": "moi_0", "group": "vox:moi", "person": "vox:moi"})
        G.build(force=True, out_ids=ids_p, out_groups=grp_p, report=md, include_v4=False)
        grp2 = json.load(open(grp_p, encoding="utf-8"))["groups"]
        assert all(grp2[k] == v for k, v in grp.items()) and "vox:moi" in grp2
        assert grp2["vox:moi"] == C.crc_fold_of_group("vox:moi")               # nhom moi = fold da phat (on dinh)
    finally:
        G.iter_items = old
        C.FOLDS_V2_GROUPS = old_frozen
        C._frozen_group_folds.cache_clear()


def _it(rid: str, group: str, stratum: str = "en:voxp", donors: tuple = (), lock: str | None = None) -> dict:
    lang = stratum.split(":")[0]
    return {"id": rid, "set": "naten", "cfg": "goc", "group": group, "person": "voxp:None", "source": stratum.split(":")[1],
            "lang": lang, "stratum": stratum, "real": True, "pf": 0, "donors": list(donors), "bad_donors": [],
            "lockbox_v2": lock}


def test_donor_components_same_fold() -> None:
    """Ghep loi that voxp:None lay doan tu ban goc voxp:None KHAC -> hai nhom phai cung fold (truoc sua: 41 dong lech)."""
    items = []
    for i in range(40):
        items.append(_it(f"h{i}", f"voxp:None:h{i}"))
    for i in range(0, 40, 2):                                               # dong ghep: host h{i}, donor h{i+1}
        items.append(_it(f"g{i}", f"voxp:None:h{i}", donors=(f"voxp:None:h{i + 1}",)))
    items.append(_it("g_chain", "voxp:None:h1", donors=("voxp:None:h2",)))   # chuoi h0-h1-h2-h3
    comp = G.components(items)
    assert comp["voxp:None:h0"] == comp["voxp:None:h1"] == comp["voxp:None:h2"] == comp["voxp:None:h3"]
    assert comp["voxp:None:h4"] == comp["voxp:None:h5"] != comp["voxp:None:h0"]
    gf, conf = G.assign_components(items, comp, {})
    assert G.donor_cross_fold(items, gf)["n"] == 0 and not conf
    naive = G.assign_stratified(G.group_counts(items))                      # khong gop -> co lech (chung minh test co y nghia)
    assert G.donor_cross_fold(items, naive)["n"] > 0
    # ghim mau thuan trong mot thanh phan -> ghi lai, van cung fold
    gf2, conf2 = G.assign_components(items, comp, {"voxp:None:h0": 1, "voxp:None:h1": 3})
    assert gf2["voxp:None:h0"] == gf2["voxp:None:h1"] and len(conf2) == 1
    assert G.donor_groups({"edits": [{"donor_file": "khong_co_trong_hosts"}, {"tts": "x"}]}) == ([], ["khong_co_trong_hosts"])


def test_build_excludes_lockbox_and_links_donors() -> None:
    d = _fresh("groups_build2")
    items = [_it(f"a{i}", f"vox:p{i}", "vi:vox") for i in range(20)]
    items += [_it("lk0", "vox:moi", "vi:vox", lock="new_person:vox:moi"), _it("lk1", "vox:moi", "vi:vox", lock="new_person:vox:moi")]
    items += [_it("n0", "voxp:None:x"), _it("n1", "voxp:None:y", donors=("voxp:None:x",))]
    old_it, old_r = G.iter_items, G._lockbox_rules
    try:
        G.iter_items = lambda include_v4=True, rules=None: iter(items)      # type: ignore[assignment]
        G._lockbox_rules = lambda: None                                    # type: ignore[assignment]
        p = [os.path.join(d, x) for x in ("f.json", "g.json", "r.md")]
        r = G.build(out_ids=p[0], out_groups=p[1], report=p[2], include_v4=False)
        g = json.load(open(p[1], encoding="utf-8"))
        assert r["donor_cross_fold"] == 0 and g["groups"]["voxp:None:x"] == g["groups"]["voxp:None:y"]
        assert g["lockbox_v2_groups"] == ["vox:moi"] and g["n_lockbox_v2_items"] == 2
        assert g["groups"]["vox:moi"] == zlib.crc32(b"fold:vox:moi") % 5
        assert sum(g["balance"]["files"][G.ALL]) == len(items) - 2                    # lockbox ngoai can bang
        assert "Lockbox v2" in open(p[2], encoding="utf-8").read()
    finally:
        G.iter_items, G._lockbox_rules = old_it, old_r


def test_k1axes_items_inherit_base() -> None:
    d = _fresh("k1")
    cp = os.path.join(d, "cases.json")
    json.dump([{"file": "dinhdang.mp3__b1.mp3", "base": "b1"}, {"file": "x__khong.wav", "base": "khong"}], open(cp, "w"))
    base = _it("b1", "vox:p1", "vi:vox")
    out, miss = G.k1axes_items({"b1": base}, cp)
    assert [o["id"] for o in out] == ["dinhdang.mp3__b1.mp3"] and out[0]["group"] == "vox:p1" and miss == ["x__khong.wav"]


def test_envelope_match_and_chain_dupes() -> None:
    d = _fresh("dupes")
    h1, h2 = _wav(os.path.join(d, "h1.wav"), 3.0, 200, seed=1), _wav(os.path.join(d, "h2.wav"), 3.0, 330, seed=7)
    y1 = sf.read(h1, dtype="float32")[0]
    y2 = sf.read(h2, dtype="float32")[0]
    assert G.envelope_match(y1, y1 + 0.001 * np.random.default_rng(0).standard_normal(len(y1)), C.SR) > 0.95
    assert G.envelope_match(y1, y2, C.SR) < G.envelope_match(y1, y1, C.SR)
    disk = os.path.join(d, "x.wav")
    shutil.copy(h2, disk)                                                   # dong sau (h2) da ghi de wav
    man = [{"id": "x", "wav": disk, "kind": "nguyen_ban", "meta": {"host_file": "h1", "person": "p1"}},
           {"id": "y", "wav": disk, "kind": "nguyen_ban", "meta": {"host_file": "h1", "person": "p1"}},
           {"id": "x", "wav": disk, "kind": "nguyen_ban", "meta": {"host_file": "h2", "person": "p2"}}]
    mp = os.path.join(d, "manifest.json")
    json.dump(man, open(mp, "w", encoding="utf-8"))
    old = G._host_wav
    try:
        G._host_wav = lambda hf: {"h1": h1, "h2": h2}.get(hf)               # type: ignore[assignment]
        r = G.chain_dupes(mp, out=os.path.join(d, "plan.json"))
    finally:
        G._host_wav = old
    assert r["n_rows"] == 3 and r["n_unique"] == 2 and len(r["duplicates"]) == 1
    dup = r["duplicates"][0]
    assert dup["keep_row"] == 2 == dup["last_row_wins_now"]
    acts = {c["row"]: (c["action"], c["new_id"]) for c in dup["rows"]}
    assert acts[2] == ("giu", "x") and acts[0] == ("doi_ten_sinh_lai", "x_d1")


# ================= lockbox =================
def _scrape_group(target_zero: bool) -> str:
    for i in range(200):
        g = f"scrape:UC{i}"
        if (zlib.crc32(g.encode()) % L.SCRAPE_MOD == 0) == target_zero:
            return g
    raise AssertionError


def test_lockbox_rules() -> None:
    R = L.Rules(["vox:a", "ami:ES2004", "edacc:EDACC-C08"], ["celeb:dev1"], ["engineX"])
    assert R.reason({"source": "vox", "person": "vox:a"}) is None
    assert R.reason({"source": "vox", "person": "vox:b"}) == "new_person:vox:b"
    assert R.reason({"source": "celeb", "person": "celeb:dev1"}) is None                   # nha cho dev
    assert R.reason({"source": "ami", "speaker": "ami:ES2004b", "person": "ami:NEW1"}) is None   # series da dung
    # pham vi new_person = dac ta (vox / celeb); nguon tieng Anh moi KHONG tu dong khoa (quyet dinh ghi trong bao cao)
    assert R.reason({"source": "ami", "speaker": "ami:IB4001a", "person": "ami:X"}) is None
    assert R.reason({"source": "librispeech", "person": "librispeech:19"}) is None
    assert R.reason({"source": "edacc", "person": "edacc:EDACC-C08-B"}) is None
    assert R.reason({"source": "vss", "person": "vss:moi"}) is None                         # khong phai kho cong khai
    g0, g1 = _scrape_group(True), _scrape_group(False)
    assert R.reason({"group": g0, "url": "https://x", "license": "cc-by"}) == f"scrape:{g0}"
    assert R.reason({"group": g1, "url": "https://x", "license": "cc-by"}) is None
    # dang WP6 (v6.scrape.build_sets): nhom 'yt:<kenh>', nguon goc long trong meta.scrape, khong co url cap tren
    y0 = next(g for g in (f"yt:UC{i}" for i in range(200)) if zlib.crc32(g.encode()) % 5 == 0)
    ch0 = y0.split(":", 1)[1]
    assert R.reason({"group": y0, "person": y0, "source": "yt", "scrape": {"url": "u", "channel_id": ch0}}) == f"scrape:{y0}"
    assert R.reason({"group": y0, "person": y0}, "scrapevi") == f"scrape:{y0}"            # nhan theo ten tap
    # 'license' KHONG lam ban ghi kho cong khai thanh 'cao' -> new_person van ap
    assert R.reason({"source": "vox", "person": "vox:b", "license": "CC BY-NC 4.0"}) == "new_person:vox:b"
    # engine giu lai: cap tren (engine / tts) va quy uoc hien co edits[].tts; lockbox_v2=False KHONG nha
    assert R.reason({"source": "vox", "person": "vox:a", "engine": "engineX"}) == "tts_heldout:engineX"
    assert R.reason({"source": "vox", "person": "vox:a", "tts": "engineX"}) == "tts_heldout:engineX"
    assert R.reason({"source": "vox", "person": "vox:a", "edits": [{"op": "chen", "tts": "engineX"}]}) == "tts_heldout:engineX"
    assert R.reason({"source": "vox", "person": "vox:a", "edits": [{"vc": "engineX"}]}) == "tts_heldout:engineX"
    assert R.reason({"source": "vox", "person": "vox:a", "engine": "engineX", "lockbox_v2": False}) == "tts_heldout:engineX"
    assert R.reason({"source": "vox", "person": "vox:a", "edits": [{"tts": "viXTTS"}]}) is None      # engine hoc duoc
    assert R.reason({"source": "vox", "person": "vox:a", "lockbox_v2": True}) == "explicit"
    assert R.reason({"source": "vox", "person": "vox:zz", "lockbox_v2": False}) == "new_person:vox:zz"   # dong khong nha duoc
    assert R.reason({"meta": {"source": "vox", "person": "vox:b"}}) == "new_person:vox:b"


def test_lockbox_freeze_append_only_and_guard() -> None:
    d = _fresh("lockbox")
    path = os.path.join(d, "lockbox_v2.json")
    w_new = _wav(os.path.join(d, "a", "new.wav"), seed=3)
    w_old = _wav(os.path.join(d, "a", "old.wav"), seed=4)
    w_rel = _wav(os.path.join(d, "a", "rel.wav"), seed=5)
    w_sneak = _wav(os.path.join(d, "a", "sneak.wav"), seed=6)
    cands = [("vox_new_0", w_new, {"source": "vox", "person": "vox:new", "host_file": "vox_new_0"}, "hosts_v2"),
             ("vox_old_0", w_old, {"source": "vox", "person": "vox:old", "host_file": "vox_old_0"}, "hosts"),
             ("vox_rel_0", w_rel, {"source": "vox", "person": "vox:rel", "lockbox_v2": False}, "hosts_v2"),
             # dong sua bat ky ghi lockbox_v2=False cho nguoi lockbox -> KHONG duoc nha ca nhom (lo hong cu)
             ("v6x_sneak", w_sneak, {"source": "vox", "person": "vox:new", "lockbox_v2": False}, "natvi"),
             ("vox_miss", os.path.join(d, "khong_co.wav"), {"source": "vox", "person": "vox:new2"}, "hosts_v2")]
    plan = os.path.join(d, "plan.json")                                      # nha cho dev CHI qua ke hoach expand_hosts
    json.dump({"persons": [{"person": "vox:rel", "group": "vox:rel", "lockbox_v2": False},
                           {"person": "vox:new", "group": "vox:new", "lockbox_v2": True}]}, open(plan, "w"))
    old = L._candidates
    try:
        L._candidates = lambda extra=(): iter(cands)                          # type: ignore[assignment]
        s = L.freeze(dry_run=True, path=path, used_groups=["vox:old"], plan_path=plan)
        assert not os.path.exists(path) and s["added"] == 2 and s["n_missing_wav"] == 1
        s = L.freeze(path=path, used_groups=["vox:old"], plan_path=plan, release=["celeb:tay"])
        st = json.load(open(path, encoding="utf-8"))
        assert s["first_freeze"] and sorted(st["items"]) == ["v6x_sneak", "vox_new_0"]
        assert st["released_groups"] == ["celeb:tay", "vox:rel"]
        assert st["items"]["vox_new_0"]["sha1"] == L.sha1_file(w_new) and st["consumed_v1"]["status"] == "consumed"
        assert st["used_groups"] == ["vox:old"] and st["rules"]["new_person"] == ["celeb", "vox"]
        s2 = L.freeze(path=path, plan_path="")                               # lan 2: khong them gi, anh chup giu nguyen
        assert s2["added"] == 0 and not s2["first_freeze"]
        assert json.load(open(path, encoding="utf-8"))["released_groups"] == ["celeb:tay", "vox:rel"]   # khong mat
        # chot chan
        L.assert_not_lockbox(["vox_old_0", {"id": "v6nat_x", "meta": {"source": "vox", "person": "vox:old"}}], path=path)
        _raises(C.LeakageError, L.assert_not_lockbox, ["VOX_NEW_0"], path=path)               # khong phan biet hoa/thuong
        _raises(C.LeakageError, L.assert_not_lockbox, [{"id": "v6nat_9", "meta": {"host_file": "vox_new_0"}}], path=path)
        _raises(C.LeakageError, L.assert_not_lockbox, [{"id": "v6nat_8", "meta": {"source": "celeb", "person": "celeb:q"}}],
                path=path)                                                   # luat new_person
        L.assert_not_lockbox([{"id": "v6nat_7", "meta": {"source": "vox", "person": "vox:rel"}}], path=path)
        cp = os.path.join(d, "a", "copy.wav")
        shutil.copy(w_new, cp)
        _raises(C.LeakageError, L.assert_not_lockbox, [{"id": "doi_ten", "wav": cp, "meta": {"source": "vss", "person": "vss:k"}}],
                path=path, check_sha1=True)
        keep, lock = L.split_not_lockbox([{"id": "vox_new_0"}, {"id": "khac", "meta": {"source": "vss", "person": "vss:k"}}], path=path)
        assert [r["id"] for r in keep] == ["khac"] and [r["id"] for r in lock] == ["vox_new_0"]
        # audio doi -> loi, khong ghi
        time.sleep(0.01)
        _wav(w_new, seed=99)
        _raises(C.V6Error, L.freeze, path=path, plan_path="")
    finally:
        L._candidates = old


def test_lockbox_rules_need_freeze() -> None:
    _raises(C.V6Error, L.Rules.from_state, {})


def test_lockbox_guard_fails_loud_when_unfrozen() -> None:
    """Chot chan KHONG im lang cho qua khi thieu tep / chua dong bang (truoc sua: moi ban ghi deu qua)."""
    d = _fresh("lockbox_unfrozen")
    missing = os.path.join(d, "khong_co.json")
    rec = {"id": "x", "meta": {"source": "vox", "person": "vox:moi"}}
    e = _raises(C.V6Error, L.assert_not_lockbox, [rec], path=missing)
    assert not isinstance(e, C.LeakageError)                                 # split_not_lockbox khong nuot loi nay
    _raises(C.V6Error, L.split_not_lockbox, [rec], path=missing)
    empty = os.path.join(d, "rong.json")
    json.dump({"version": 2, "items": {}}, open(empty, "w"))
    _raises(C.V6Error, L.assert_not_lockbox, ["x"], path=empty)
    # allow_unfrozen: new_person khong kiem duoc (khong anh chup) nhung cao / explicit van bi chan
    L.assert_not_lockbox([rec], path=missing, allow_unfrozen=True)
    y0 = next(g for g in (f"yt:UC{i}" for i in range(200)) if zlib.crc32(g.encode()) % 5 == 0)
    _raises(C.LeakageError, L.assert_not_lockbox, [{"id": "s", "meta": {"group": y0, "source": "yt"}}], path=missing,
            allow_unfrozen=True)
    _raises(C.LeakageError, L.assert_not_lockbox, [{"id": "e", "meta": {"lockbox_v2": True}}], path=missing,
            allow_unfrozen=True)


def test_lockbox_engines_of() -> None:
    m = {"engine": "a", "tts": "b", "edits": [{"tts": "c", "donor_file": "h"}, "khong_phai_dict", {"vc_engine": "d"}]}
    assert L.engines_of(m) == ["a", "b", "c", "d"] and L.engines_of({}) == []
    hy = {"engine": "xtts+knnvc", "edits": [{"gen_chain": ["xtts", "knnvc"], "engine": "xtts+knnvc"}]}
    assert set(L.engines_of(hy)) == {"xtts", "knnvc"}


def test_lockbox_hybrid_and_engine_role() -> None:
    """WP7b: engine '<tts>+<vc>', edits[].gen_chain, meta.engine_role -> van bi khoa (review 10/10, muc high)."""
    r = L.Rules(["vox:x"], heldout=["knnvc", "vieneu"])
    assert r.reason({"engine": "knnvc"}) == "tts_heldout:knnvc"
    assert r.reason({"engine": "xtts+knnvc", "edits": [{"gen_chain": ["xtts", "knnvc"]}]}) == "tts_heldout:knnvc"
    assert r.reason({"edits": [{"gen_chain": ["xtts", "knnvc"]}]}) == "tts_heldout:knnvc"
    assert r.reason({"meta": {"gen_chain": ["vieneu"]}}) == "tts_heldout:vieneu"
    assert r.reason({"engine": "xtts+openvoice"}) is None                     # ca hai deu train -> dung duoc
    assert r.reason({"engine": "moi_chua_dang_ky", "engine_role": "heldout"}) == "tts_heldout:engine_role"
    assert r.reason({"edits": [{"engine": "z", "role": "heldout"}]}) == "tts_heldout:engine_role"
    assert r.reason({"engine": "xtts", "engine_role": "train", "edits": [{"role": "train"}]}) is None
    assert r.reason({"engine": "xtts+knnvc", "lockbox_v2": False}) == "tts_heldout:knnvc"   # khong nha duoc


def test_groups_vivos_donor_resolves() -> None:
    meta = {"edits": [{"donor_file": "train_VIVOSSPK12_R001.wav"}, {"donor_file": "test_VIVOSDEV03_00012.wav"},
                      {"donor_file": "la_khong_ro.wav"}]}
    out, bad = G.donor_groups(meta)
    assert "VIVOSSPK12" in out and "VIVOSDEV03" in out and bad == ["la_khong_ro.wav"]


def _scratch_groups_env(d: str, items: list) -> tuple[list[str], Any]:
    """Thay iter_items / luat lockbox / bang fold that bang ban gia trong d; tra (duong dan, ham khoi phuc)."""
    old = (G.iter_items, G._lockbox_rules, C.FOLDS_V2_GROUPS)
    paths = [os.path.join(d, x) for x in ("f.json", "g.json", "r.md")]
    G.iter_items = lambda include_v4=True, rules=None: iter(items)          # type: ignore[assignment]
    G._lockbox_rules = lambda: None                                        # type: ignore[assignment]
    C.FOLDS_V2_GROUPS = paths[1]
    C._frozen_group_folds.cache_clear()

    def restore() -> None:
        G.iter_items, G._lockbox_rules, C.FOLDS_V2_GROUPS = old
        C._frozen_group_folds.cache_clear()
    return paths, restore


def test_build_force_keeps_issued_folds() -> None:
    """Review 12/10 (high): fold C.fold_of_group da phat cho nhom MOI (WP6 meta.fold, expand_hosts fold_v2) phai giu
    nguyen sau build --force (truoc sua: 13/20 nhom moi doi fold)."""
    d = _fresh("groups_stable")
    items = [_it(f"a{i}_{j}", f"vox:p{i}", "vi:vox") for i in range(25) for j in range(3)]
    paths, restore = _scratch_groups_env(d, items)
    try:
        G.build(out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        old_tbl = json.load(open(paths[1], encoding="utf-8"))["groups"]
        new_groups = [f"yt:UCmoi{i}" for i in range(20)] + [f"celeb:moi{i}" for i in range(20)]
        issued = {g: C.fold_of_group(g) for g in new_groups}                # phat ra TRUOC khi bang biet nhom
        items += [_it(f"n{i}_{j}", g, "vi:yt" if g.startswith("yt") else "vi:celeb") for i, g in enumerate(new_groups)
                  for j in range(1 + i % 4)]
        r = G.build(force=True, out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        C._frozen_group_folds.cache_clear()
        tbl = json.load(open(paths[1], encoding="utf-8"))["groups"]
        assert r["mode"] == "pinned" and r["n_new_groups_pinned_crc"] == 40
        assert all(tbl[g] == f for g, f in old_tbl.items())                  # nhom cu giu fold
        assert {g: tbl[g] for g in new_groups} == issued                      # nhom moi = fold da phat
        assert {g: C.fold_of_group(g, strict=True) for g in new_groups} == issued
        ids = json.load(open(paths[0], encoding="utf-8"))
        assert all(ids[f"n{i}_0"] == issued[g] for i, g in enumerate(new_groups))
    finally:
        restore()


def test_build_force_donor_only_group_joins_component() -> None:
    """Nhom CHI lam donor (khong co dong) duoc ghi vao bang theo fold host, va o lan --force khong bi ghim crc (truoc sua:
    4 nhom voxp:None donor khong co trong bang -> lan lam moi se loai 9 dong naten/aug khoi CV)."""
    d = _fresh("groups_donor_only")
    items = [_it(f"h{i}", f"voxp:None:h{i}") for i in range(10)] + [_it("g0", "voxp:None:h1", donors=("voxp:None:solo",))]
    paths, restore = _scratch_groups_env(d, items)
    try:
        G.build(out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        tbl = json.load(open(paths[1], encoding="utf-8"))["groups"]
        assert tbl["voxp:None:solo"] == tbl["voxp:None:h1"]                     # nhom donor le duoc ghi vao bang
        items.append(_it("g_new", "voxp:None:h0", donors=("voxp:None:donor_moi",)))
        r = G.build(force=True, out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        g2 = json.load(open(paths[1], encoding="utf-8"))["groups"]
        assert r["cv_excluded"] == 0 and r["n_new_link_only_groups"] == 1 and r["n_new_groups_pinned_crc"] == 0
        assert g2["voxp:None:donor_moi"] == tbl["voxp:None:h0"] and all(g2[k] == v for k, v in tbl.items())
    finally:
        restore()


def test_groups_target_groups_parse() -> None:
    vc = {"person": "celeb:id00916", "source": "celeb", "edits": [{"op": "doi_giong_ai", "target_person": "celeb:id00881",
                                                                    "target_source": "celeb"}]}
    assert G.target_groups(vc) == ["celeb:id00881"]
    assert G.target_groups({"target": {"person": "voxp:12", "source": "voxp"}, "ref_person": "vox:a"}) == ["vox:a", "voxp:12"]
    assert G.target_groups({"edits": [{"tts": "x"}]}) == []
    assert G.person_group("edacc:EDACC-C07-B", "edacc") == "edacc:EDACC-C07"   # nguoi EdAcc -> cuoc hoi thoai
    it = G._item("v6vcm_x", "vcmulti", "goc", vc)
    assert it["group"] == "celeb:id00916" and it["targets"] == ["celeb:id00881"]


def test_groups_target_link_excluded_not_merged() -> None:
    """Review 12/10 (high): VC host A noi bang giong B. B o fold khac -> dong LOAI khoi CV (khong gop A-B: do thi VC that
    noi 338 nguoi thanh 16 thanh phan). B cung fold -> giu."""
    d = _fresh("groups_target")
    items = [_it(f"r{i}_{j}", f"vox:p{i}", "vi:vox") for i in range(30) for j in range(2)]
    paths, restore = _scratch_groups_env(d, items)
    try:
        G.build(out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        tbl = json.load(open(paths[1], encoding="utf-8"))["groups"]
        same = next(g for g in tbl if g != "vox:p0" and tbl[g] == tbl["vox:p0"])
        diff = [g for g in tbl if tbl[g] != tbl["vox:p0"]]
        vc_items = [dict(_it(f"vc{k}", "vox:p0", "vi:vox"), targets=[t]) for k, t in enumerate(diff)]
        vc_items.append(dict(_it("vc_ok", "vox:p0", "vi:vox"), targets=[same]))
        items += vc_items
        r = G.build(force=True, out_ids=paths[0], out_groups=paths[1], report=paths[2], include_v4=False)
        g2 = json.load(open(paths[1], encoding="utf-8"))
        ids = json.load(open(paths[0], encoding="utf-8"))
        assert r["cv_excluded"] == len(diff) and r["link_cross_fold_in_cv"] == 0
        assert all(f"vc{k}" not in ids for k in range(len(diff))) and ids["vc_ok"] == tbl["vox:p0"]
        assert set(g2["cv_excluded"]) == {f"vc{k}" for k in range(len(diff))} and g2["cv_excluded"]["vc0"]["kind"] == "target"
        assert g2["groups"] == {**tbl, **{g: g2["groups"][g] for g in g2["groups"] if g not in tbl}}   # khong nhom nao doi fold
        assert not g2["components"]                                             # giong dich KHONG gop thanh phan
        assert "LOAI KHOI CV" in open(paths[2], encoding="utf-8").read()
        C._frozen_group_folds.cache_clear()
        assert G.target_fold_ok("vox:p0", same) and not G.target_fold_ok("vox:p0", diff[0])
    finally:
        restore()


def test_groups_pinned_donor_conflict_excluded() -> None:
    """Hai nhom da ghim khac fold noi qua donor_file moi -> khong gop, khong doi fold nhom ghim, dong noi loai khoi CV."""
    items = [_it("h0", "vox:a"), _it("h1", "vox:b"), _it("x", "vox:a", donors=("vox:b",)), _it("y", "vox:c", donors=("vox:d",))]
    comp, skipped = G.link_components(items, {"vox:a": 0, "vox:b": 3, "vox:c": 2})
    assert comp["vox:a"] != comp["vox:b"] and comp["vox:c"] == comp["vox:d"]      # c-d gop (d chua ghim)
    assert [s["id"] for s in skipped] == ["x"] and skipped[0]["folds"] == [0, 3]
    gf, conf = G.assign_components(items, comp, {"vox:a": 0, "vox:b": 3, "vox:c": 2})
    assert gf["vox:a"] == 0 and gf["vox:b"] == 3 and gf["vox:c"] == 2 and not conf
    ex = G._cv_excluded(items, lambda g: {**gf, "vox:d": 2}.get(g, 9))
    assert set(ex) == {"x"} and ex["x"]["kind"] == "donor"
    # trung id: dong sau cung quyet dinh (dong sau sach -> id giu trong CV)
    assert G._cv_excluded(items + [_it("x", "vox:a")], lambda g: {**gf, "vox:d": 2}.get(g, 9)) == {}


def test_lockbox_string_ids_apply_rules() -> None:
    """Review 12/10 (high): id dang chuoi duoc tra nguoc ra meta va ap LUAT -> ban goc cao lockbox tao SAU lan dong bang
    cuoi khong lot qua (truoc sua: 17 ban goc cao lockbox qua duoc khi truyen id)."""
    d = _fresh("lockbox_ids")
    path = os.path.join(d, "lockbox_v2.json")
    json.dump({"version": 2, "used_groups": ["vox:old"], "released_groups": [], "items": {}}, open(path, "w"))
    y0 = next(g for g in (f"yt:UC{i}" for i in range(500)) if zlib.crc32(g.encode()) % 5 == 0)
    y1 = next(g for g in (f"yt:UC{i}" for i in range(500)) if zlib.crc32(g.encode()) % 5 != 0)
    cands = [("yt_lock_s000", "", {"file": "yt_lock_s000", "source": "yt", "group": y0, "person": y0,
                                   "channel_id": y0.split(":")[1], "url": "u", "host_file": "yt_lock_s000"}, "hosts_vi"),
             ("yt_ok_s000", "", {"file": "yt_ok_s000", "source": "yt", "group": y1, "person": y1,
                                 "channel_id": y1.split(":")[1], "url": "u", "host_file": "yt_ok_s000"}, "hosts_vi"),
             ("v6nat_new", "", {"source": "vox", "person": "vox:moi"}, "natvi"),
             ("v6nat_old", "", {"source": "vox", "person": "vox:old"}, "natvi")]
    old = L._candidates
    try:
        L._candidates = lambda extra=(): iter(cands)                          # type: ignore[assignment]
        e = _raises(C.LeakageError, L.assert_not_lockbox, ["yt_lock_s000"], path=path)
        assert "scrape:" + y0 in str(e)
        _raises(C.LeakageError, L.assert_not_lockbox, ["V6NAT_NEW"], path=path)        # new_person qua id
        L.assert_not_lockbox(["yt_ok_s000", "v6nat_old"], path=path)
        _raises(C.V6Error, L.assert_not_lockbox, ["khong_ai_biet"], path=path)        # id la -> khong im lang
        L.assert_not_lockbox(["khong_ai_biet"], path=path, allow_unknown_ids=True)
        keep, lock = L.split_not_lockbox(["yt_ok_s000", "yt_lock_s000", {"id": "v6nat_old", "meta": {"person": "vox:old",
                                                                                                    "source": "vox"}}], path=path)
        assert keep == ["yt_ok_s000", {"id": "v6nat_old", "meta": {"person": "vox:old", "source": "vox"}}]
        assert lock == ["yt_lock_s000"]
    finally:
        L._candidates = old
        L.reset_caches()


def test_lockbox_split_loads_state_once() -> None:
    """Review 12/10 (low): split_not_lockbox nap trang thai MOT lan cho ca danh sach (truoc: moi dong mot lan)."""
    d = _fresh("lockbox_once")
    path = os.path.join(d, "lockbox_v2.json")
    json.dump({"version": 2, "used_groups": ["vox:old"], "items": {}}, open(path, "w"))
    calls = Counter()
    old = L.load_state

    def counting(p: str = L.LOCKBOX_V2) -> dict:
        calls["n"] += 1
        return old(p)
    try:
        L.load_state = counting                                              # type: ignore[assignment]
        recs = [{"id": f"r{i}", "meta": {"source": "vox", "person": "vox:old" if i % 3 else "vox:moi"}} for i in range(300)]
        keep, lock = L.split_not_lockbox(recs, path=path)
        assert len(lock) == 100 and len(keep) == 200 and calls["n"] <= 2, calls
    finally:
        L.load_state = old


def test_lockbox_scrape_key_stable_under_clustering() -> None:
    """Review 12/10 (medium): luat cao theo KENH (nguon:channel_id), khong theo meta.group (doi khi phan cum)."""
    R = L.Rules(["vox:a"])
    y0 = next(g for g in (f"yt:UC{i}" for i in range(500)) if zlib.crc32(g.encode()) % 5 == 0)
    y1 = next(g for g in (f"yt:UC{i}" for i in range(500)) if zlib.crc32(g.encode()) % 5 != 0)
    for y in (y0, y1):
        ch = y.split(":")[1]
        before = R.reason({"group": y, "source": "yt", "channel_id": ch, "url": "u"})
        after = [R.reason({"group": f"{y}:spk{k}", "source": "yt", "scrape": {"channel_id": ch}}) for k in range(30)]
        assert set(after) == {before} and (before == f"scrape:{y}") == (y == y0)
    assert L.scrape_key({"group": "yt:UCz:spk1", "source": "yt", "channel_id": "UCz"}) == "yt:UCz"
    assert L.scrape_key({"group": "scrape:x"}) == "scrape:x"                 # khong channel_id -> nhom (cach cu)


def test_lockbox_linked_target_and_donor() -> None:
    """Review 12/10 (high): tep VC / ghep co giong / audio cua nguoi LOCKBOX (new_person, cao crc%5==0) -> khoa ca dong."""
    R = L.Rules(["vox:a", "vox:b", "celeb:x"])
    vc = {"source": "vox", "person": "vox:a", "edits": [{"op": "doi_giong_ai", "target_person": "vox:moi",
                                                         "target_source": "vox"}]}
    assert R.reason(vc) == "linked_new_person:vox:moi"
    assert R.reason({**vc, "edits": [{"target_person": "vox:b", "target_source": "vox"}]}) is None
    assert R.reason({"source": "vox", "person": "vox:a", "target": {"person": "celeb:moi", "source": "celeb"}}) \
        == "linked_new_person:celeb:moi"
    y0 = next(g for g in (f"yt:UC{i}" for i in range(500)) if zlib.crc32(g.encode()) % 5 == 0)
    fake = {"yt_x_s001": {"group": y0, "source": "yt", "channel_id": y0.split(":")[1], "kind": "scrape"},
            "vox_test-1_00001": {"group": "vox:id9", "source": "vox", "lockbox_v2": True, "kind": "new"},
            "vox_test-1_00002": {"group": "vox:dev1", "source": "vox", "lockbox_v2": False, "role": "dev", "kind": "new"}}
    old = L.host_index
    try:
        L.host_index = lambda: fake                                          # type: ignore[assignment]
        ed = lambda f: {"source": "vox", "person": "vox:a", "edits": [{"donor_file": f}]}   # noqa: E731
        assert R.reason(ed("yt_x_s001")) == f"linked_scrape:{y0}"
        assert R.reason(ed("vox_test-1_00001")) == "linked_new_person:vox:id9"
        assert R.reason(ed("vox_test-1_00002")) == "linked_new_person:vox:dev1"      # chua nha (released) -> van khoa
        assert L.Rules(["vox:a"], ["vox:dev1"]).reason(ed("vox_test-1_00002")) is None
        assert G.donor_groups({"edits": [{"donor_file": "vox_test-1_00001"}]}) == (["vox:id9"], [])
    finally:
        L.host_index = old


def test_lockbox_curate_records_decision() -> None:
    d = _fresh("lockbox_curate")
    path = os.path.join(d, "lockbox_v2.json")
    json.dump({"version": 2, "used_groups": ["voxp:1"], "items": {}}, open(path, "w"))
    _raises(C.V6Error, L.curate, ["voxp"], path=path)                        # thieu --by
    _raises(C.V6Error, L.curate, ["vss"], by="u", path=path)                 # khong phai kho cong khai
    r = L.curate(["voxp"], by="nguoi dung 12/10", note="EN lockbox", path=path)
    st = json.load(open(path, encoding="utf-8"))
    assert r["changed"] and st["curated_sources"] == ["celeb", "vox", "voxp"] and st["curated_history"][0]["by"]
    rules = L.Rules.from_state(st)
    assert rules.reason({"source": "voxp", "person": "voxp:2"}) == "new_person:voxp:2"
    assert rules.reason({"source": "voxp", "person": "voxp:1"}) is None
    assert "voxp" in L.en_lockbox_status(rules.curated) and "CHUA" in L.en_lockbox_status(L.CURATED_SOURCES)


def test_expand_has_long() -> None:
    durs = {"t.parquet": [5.0, 25.0, 3.0]}
    assert X._has_long([("D:/x/t.parquet", 1, None)], durs) and not X._has_long([("D:/x/t.parquet", 0, None)], durs)


def test_listen_readme_sections() -> None:
    d = _fresh("listen_readme")
    kit, key = os.path.join(d, "kit"), os.path.join(d, "khoa")
    for sub, name in (("nghe/am_thanh", "N001.wav"), ("req13/anh", "A001.png"), ("cao/am_thanh", "S001.wav")):
        os.makedirs(os.path.join(kit, sub), exist_ok=True)
        open(os.path.join(kit, sub, name), "wb").close()
    os.makedirs(key)
    with open(os.path.join(key, "khoa_req13.csv"), "w", encoding="utf-8-sig", newline="") as fh:
        fh.write("ma_anh,dap_an_so,dap_an\nA001,4,xoa_loi\n")
    text = open(K.write_readme(kit, key), encoding="utf-8").read()
    assert "## 3." in text and "run_serial.sh" in text and 'cd "D:/T' in text and "Xoá lời 1" in text
    assert "CHỈ lấy dòng thao tác mà hệ thống nhận ĐÚNG" in text
    os.remove(os.path.join(kit, "cao", "am_thanh", "S001.wav"))
    assert "## 3." not in open(K.write_readme(kit, key), encoding="utf-8").read()


def test_listen_scrape_sample_and_export() -> None:
    recs = [{"file": f"yt_a_{i:03d}", "group": "yt:A"} for i in range(100)] + [{"file": "yt_b_000", "group": "yt:B"}]
    s = K.sample_scrape(recs, 0.05, min_n=3)
    assert Counter(r["group"] for r in s) == {"yt:A": 5, "yt:B": 1} and s == K.sample_scrape(recs, 0.05, min_n=3)
    assert len(K.sample_scrape(recs[:10], 0.05, min_n=4)) == 4                         # bu den min_n
    d = _fresh("listen_scrape")
    hroot = os.path.join(d, "hosts_vi")
    os.makedirs(hroot)
    with open(os.path.join(hroot, "hosts.jsonl"), "w", encoding="utf-8") as fh:
        for i in range(3):
            _wav(os.path.join(hroot, "yt", f"f{i}.wav"), 1.0, seed=i)
            fh.write(json.dumps({"file": f"f{i}", "source": "yt", "group": "yt:C", "url": "u", "license": "cc-by" if i else ""})
                     + "\n")
    out = K.export_scrape(os.path.join(d, "kit"), os.path.join(d, "khoa"), 0.05, hosts={"vi": os.path.join(hroot, "hosts.jsonl")},
                          archive=os.path.join(d, "arch"))
    assert out["clips"] == 3 and out["hosts_without_license"] == 1 and out["missing_wav"] == 0
    with open(os.path.join(d, "kit", "cao", "phieu_cao.csv"), encoding="utf-8-sig") as fh:
        head = fh.readline()
    assert "url" not in head and "license" not in head                                  # phieu mu: khong lo nguon
    assert K.export_scrape(os.path.join(d, "kit2"), os.path.join(d, "khoa2"), hosts={"vi": os.path.join(d, "khong_co.jsonl")})["clips"] == 0


def test_expand_n_eligible() -> None:
    durs = {"t.parquet": [5.0, 2.0, 25.0, 19.9]}
    rows = [("D:/x/t.parquet", 0, None), ("D:/x/t.parquet", 1, None), ("D:/x/t.parquet", 3, None), ("D:/x/u.parquet", 0, None)]
    assert X.n_eligible(rows, durs) == 2 and X.n_eligible(rows, {}) == 0


def test_expand_priority_window_and_passes() -> None:
    c = {"person": "vox:a", "cap": 2, "host_rows": [["p", 9, None], ["p", 1, None], ["p", 5, None], ["p", 3, None],
                                                       ["p", 7, None], ["p", 0, None]]}
    durs = {"p": [5.0] * 10}
    durs["p"][1] = 30.0                                                      # biet truoc ngoai 4-20 s -> bo qua
    assert [r[1] for r in X.priority_window(c, 3, durs, set())] == [9, 5, 3]  # theo thu tu uu tien, khong theo chi so
    assert [r[1] for r in X.priority_window(c, 3, durs, {("p", 9)})] == [5, 3, 7]   # dong da thu khong chiem cho
    p1 = X.plan_passes([c], {}, durs, set(), 1)
    assert [r[1] for r in p1[0]["host_rows"]] == [9, 5]                      # dung so con thieu (cap 2)
    assert [r[1] for r in X.plan_passes([c], {"vox:a": 1}, durs, {("p", 9)}, 2)[0]["host_rows"]] == [5]
    assert X.plan_passes([c], {"vox:a": 2}, durs, set(), 1) == []            # du cap -> khong doc lai
    p3 = X.plan_passes([c], {"vox:a": 1}, durs, {("p", 9), ("p", 5), ("p", 3)}, X.MAX_PASSES)
    assert [r[1] for r in p3[0]["host_rows"]] == [7, 0]                      # vong cuoi: moi dong con lai
    assert X.plan_passes([c], {}, durs, set(), 1, max_per_person=1)[0]["cap"] == 1


def _tiny_parquet(path: str, durs: list[float], person: str = "a") -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq
    arr = pa.array([{"array": (0.05 * np.sin(np.arange(int(C.SR * d)) / 7.0)).tolist(), "sampling_rate": C.SR}
                    for d in durs])
    pq.write_table(pa.table({"audio": arr, "speaker": [person] * len(durs)}), path, row_group_size=len(durs))
    return path


def test_expand_durs_streams_and_hosts_priority() -> None:
    d = _fresh("expand_stream")
    pqp = _tiny_parquet(os.path.join(d, "test-00000-of-00001.parquet"), [5.0, 2.0, 6.0, 7.0, 8.0, 4.5])
    old = X.parquets
    X.parquets = lambda src: [pqp] if src == "vox" else []
    try:
        res = X.step_durs(out=os.path.join(d, "durs.json"), batch_size=2)
        got = X.load_durs(os.path.join(d, "durs.json"))["test-00000-of-00001.parquet"]
        assert got == [5.0, 2.0, 6.0, 7.0, 8.0, 4.5] and res["this_run"] == 1
        assert X.step_durs(out=os.path.join(d, "durs.json"))["this_run"] == 0              # ghi tiep, khong doc lai
        plan = {"persons": [{"person": "vox:a", "group": "vox:a", "source": "vox", "cap": 2, "fold_v2": 0,
                             "lockbox_v2": True, "ref_rows": [],
                             "host_rows": [["test-00000-of-00001.parquet", i, None] for i in (4, 1, 3, 0, 2, 5)]}]}
        pp = os.path.join(d, "plan.json")
        with open(pp, "w", encoding="utf-8") as fh:
            json.dump(plan, fh)
        out = X.step_hosts(pp, os.path.join(d, "hosts"), no_asr=True, durs_path=os.path.join(d, "durs.json"))
        with open(out["out"], encoding="utf-8") as fh:
            rows = sorted(json.loads(l)["row"] for l in fh if l.strip())
        assert rows == [3, 4], rows                       # 2 dong uu tien dau qua loc (4, 3), khong phai chi so nho (0, 2)
        assert out["host_files_total"] == 2 and out["vong1_nguoi"] == 1
    finally:
        X.parquets = old


# ================= expand_hosts =================
def test_expand_naming_and_session() -> None:
    assert X.fid_of("vox", "D:/x/test-00003-of-00038.parquet", 7) == "vox_test-00003-of-00038_00007"
    assert X.session_of("celeb", "id00001/abcDEF123/00001.wav") == "abcDEF123"
    assert X.session_of("celeb", "id00880/00014.wav") is None                                 # dang that tren dia
    assert X.session_of("vox", None) is None


def test_expand_choose() -> None:
    persons = {"vox:a": [("p1", i, None) for i in range(20)],
               "vox:b": [("p1", 100 + i, None) for i in range(3)],               # it dong -> bo
               "vox:used": [("p2", i, None) for i in range(30)],
               "celeb:c": [("p3", i, "s1") for i in range(10)] + [("p3", 50 + i, "s2") for i in range(3)]}
    ch = X.choose(persons, {"vox:used"}, target=10, cap=8, refs=2, min_rows=4)
    names = [c["person"] for c in ch]
    assert set(names) == {"vox:a", "celeb:c"} and ch == X.choose(persons, {"vox:used"}, 10, 8, 2, 4)
    c = next(c for c in ch if c["person"] == "celeb:c")
    assert {r[2] for r in c["ref_rows"]} == {"s2"} and all(r[2] == "s1" for r in c["host_rows"]) and not c["ref_same_session_risk"]
    a = next(c for c in ch if c["person"] == "vox:a")
    assert [r[1] for r in a["ref_rows"]] == [18, 19] and a["ref_same_session_risk"]
    assert len(a["host_rows"]) == 18                                         # MOI dong khong phai mau (khong cat cap*2)
    assert not {tuple(r) for r in a["ref_rows"]} & {tuple(r) for r in a["host_rows"]}       # mau khac ban goc
    assert all(c["lockbox_v2"] for c in ch)
    many = {f"vox:{i}": [("p", j, None) for j in range(5)] for i in range(200)}
    dev = X.choose(many, set(), 200, 8, 2, 4, dev_share=0.5)
    share = sum(not c["lockbox_v2"] for c in dev) / len(dev)
    assert 0.35 < share < 0.65, share
    assert X.choose(persons, set(), 1, 8, 2, 4)[0]["person"] in persons


def test_expand_eligibility_and_audio_lengths() -> None:
    import pyarrow as pa
    persons = [{"person": "vox:a", "cap": 3, "host_rows": [["p1", 0, None], ["p1", 1, None], ["p1", 2, None], ["p1", 3, None]]},
               {"person": "vox:b", "cap": 3, "host_rows": [["p1", 4, None], ["p2", 0, None]]}]
    e = X.eligibility(persons, {"p1": [5.0, 2.0, 19.0, 7.0, 30.0]})["eligibility"]
    assert e["persons_with_ge1_4_20s"] == 1 and e["est_host_files_4_20s"] == 3 and e["persons_fully_known"] == 1
    assert isinstance(X.eligibility(persons, {})["eligibility"], str)
    arr = pa.array([{"array": [0.0] * 32000, "sampling_rate": 16000}, {"array": [0.0] * 8000, "sampling_rate": 8000}])
    assert X._audio_lengths(pa.table({"audio": arr}), "audio") == [2.0, 1.0]
    buf = os.path.join(_fresh("expand_len"), "x.wav")
    _wav(buf, 1.5)
    arr2 = pa.array([{"bytes": open(buf, "rb").read(), "path": "x.wav"}])
    assert X._audio_lengths(pa.table({"audio": arr2}), "audio") == [1.5]


# ================= listen_kit =================
def test_listen_class_maps() -> None:
    assert K.class_of_level2("02_nen_toan_cuc") == "khu_on" and K.class_of_level2("02_giong_cuc_bo") == "doi_giong"
    assert K.class_of_level2("03_ghep_giong_that") == "ghep_loi_that" and K.class_of_level2("03_xoa_loi") == "xoa_loi"
    assert K.class_of_level2("03_chen_giong_may") == "giong_may" and K.class_of_level2("03_giong_may_toan_cuc") == "giong_may"
    assert K.class_of_level2("01") is None
    assert K.kind_of_case("v6naten_xoa_loi_ky_0016") == "xoa_loi_ky" and K.kind_of_case("v6nat_may_ca_file_0003") == "may_ca_file"
    assert K.classes_of_kind("da_ghep_nen") == {"ghep_loi_that", "khu_on"} and K.classes_of_kind("nguyen_ban") == set()
    assert K.classes_of_kind("ghep_thay_ky") == {"ghep_loi_that"} and K.classes_of_kind("giong_toan_cuc") == {"doi_giong"}


def test_listen_windows() -> None:
    assert K.clip_window(1.0, 1.5, 10.0) == (0.0, 4.5)
    assert K.clip_window(8.0, 8.5, 9.0) == (5.0, 9.0)
    assert K.clip_window(None, None, 7.3) == (0.0, 7.3)
    rng = random.Random(0)
    for _ in range(200):
        a, b = K.pseudo_row(6.0, rng)
        assert 0.5 - 1e-9 <= a < b <= 5.5 + 1e-9 and abs(b - a - 0.5) < 1e-6


def _fake_engine(d: str) -> tuple[dict, dict]:
    recs = {"v6natvi": [], "v6naten": []}
    index = {}
    k = 0
    for s, src in (("v6natvi", "vss"), ("v6naten", "edacc")):
        for i in range(12):
            rid = f"{s}_nb_{i:03d}"
            dur = 5.0 + (i % 3)
            index[rid] = {"wav": _wav(os.path.join(d, "src", rid + ".wav"), dur, 150 + 10 * i, seed=k), "source": src,
                          "person": f"{src}:p{i}", "host_file": f"h{k}", "npv": False, "lang": "vi" if s == "v6natvi" else "en"}
            k += 1
            if i < 3:
                rows = [{"level2": "03_ghep_giong_that", "kind": "segment", "start": 1.0 + i, "end": 1.5 + i, "p": 1.0,
                         "via": "ranh_gioi_ghep"}]
                if i == 0:
                    rows.append({"level2": "02_nen_toan_cuc", "kind": "global", "p": 0.9})
                recs[s].append({"id": rid, "cfg": "goc", "expected": "01", "final": "03", "rows": rows})
            else:
                recs[s].append({"id": rid, "cfg": "goc", "expected": "01", "final": "01", "rows": []})
        recs[s].append({"id": "khac_nguon", "cfg": "goc", "expected": "01", "final": "03", "rows": []})
    index["khac_nguon"] = {"wav": "", "source": "celeb", "npv": False}
    return recs, index


def test_listen_select_and_export() -> None:
    d = _fresh("listen")
    recs, index = _fake_engine(d)
    fl, ctl = K.select_listen(recs, index, seed=1)
    assert len(fl) == 6 and Counter(r["source"] for r in ctl) == Counter(r["source"] for r in fl)
    assert not {r["id"] for r in fl} & {r["id"] for r in ctl} and all(r["final"] == "01" for r in ctl)
    kit, key = os.path.join(d, "kit"), os.path.join(d, "khoa")
    arch = os.path.join(d, "cu")
    os.makedirs(os.path.join(kit, "nghe", "am_thanh"))
    open(os.path.join(kit, "nghe", "am_thanh", "N999.wav"), "wb").close()  # doan cu cua lan xuat truoc
    r = K.export_listen(kit=kit, key_dir=key, seed=1, records=recs, index=index, archive=arch)
    assert r["clips"] == 8 + 8 and r["by_group"]["doi_chung"] == 8         # doi chung sao chep cau truc (2 dong o tep 0)
    assert not os.path.exists(os.path.join(kit, "nghe", "am_thanh", "N999.wav"))
    assert len(glob_files(arch, "N999.wav")) == 1                           # chuyen, khong xoa
    with open(os.path.join(kit, "nghe", "phieu_nghe.csv"), encoding="utf-8-sig") as fh:
        form = list(csv.DictReader(fh))
    assert [f["ma_doan"] for f in form] == [f"N{i:03d}" for i in range(1, 17)]
    assert all(f[c] == "" for f in form for c in ("co_vet_cat", "co_nhac_nen", "doi_nguoi_noi", "nghe_nhu_khu_on"))
    assert "nhom" not in form[0] and "id" not in form[0]                                    # phieu khong lo dap an
    with open(os.path.join(key, "khoa_nghe.csv"), encoding="utf-8-sig") as fh:
        keyrows = {k["ma_doan"]: k for k in csv.DictReader(fh)}
    for f in form:
        info = sf.info(os.path.join(kit, "nghe", f["tep"]))
        kr = keyrows[f["ma_doan"]]
        assert abs(info.duration - (float(kr["clip_t1_s"]) - float(kr["clip_t0_s"]))) < 0.01
        if f["pham_vi"] == "doan":
            assert 0 <= float(f["bat_dau_s"]) < float(f["ket_thuc_s"]) <= info.duration + 0.01
    assert not os.path.commonpath([kit, key]) == kit                                         # khoa nam ngoai bo nghe
    # MU: theo nguon, phan bo (pham vi, do rong) cua doi chung == cua bao nham (truoc sua: doi chung luon 0,50 s)
    for src in ("vss", "edacc"):
        sig = {"bao": Counter(), "dc": Counter()}
        for f in form:
            kr = keyrows[f["ma_doan"]]
            if kr["source"] != src:
                continue
            w = round(float(f["ket_thuc_s"]) - float(f["bat_dau_s"]), 1) if f["pham_vi"] == "doan" else None
            sig["dc" if kr["nhom"] == "doi_chung" else "bao"][(f["pham_vi"], w)] += 1
        assert sig["bao"] == sig["dc"], (src, sig)


def glob_files(root: str, name: str) -> list[str]:
    return [os.path.join(a, name) for a, _, fs in os.walk(root) if name in fs]


def test_listen_mirror_row() -> None:
    rng = random.Random(0)
    a, b = K.mirror_row(10.0, 2.0, 3.2, 5.0, rng)                   # tam 2,6/5 = 0,52 -> 5,2 tren tep 10 s, rong 1,2
    assert abs((b - a) - 1.2) < 1e-6 and abs((a + b) / 2 - 5.2) < 1e-6
    a, b = K.mirror_row(4.0, 4.5, 5.0, 5.0, rng)                    # sat cuoi -> cat vao [0,5 ; 3,5]
    assert b <= 3.5 + 1e-6 and abs((b - a) - 0.5) < 1e-6
    a, b = K.mirror_row(1.2, 1.0, 2.0, 5.0, rng)                    # tep qua ngan -> van hop le
    assert 0 <= a < b <= 1.2


def test_req13_mask_pick_and_score() -> None:
    d = _fresh("req13")
    from PIL import Image
    src = os.path.join(d, "e2e", "v6nat_xoa_loi_0001", "anh", "T1.png")
    os.makedirs(os.path.dirname(src))
    im = Image.new("RGB", (200, 100), (200, 30, 30))
    im.save(src)
    dst = os.path.join(d, "m.png")
    K.mask_title(src, dst)
    with Image.open(dst) as m:
        assert m.getpixel((100, 2)) == (255, 255, 255) and m.getpixel((100, 50)) == (200, 30, 30)
    # e2e gia: 5 loai, dong dung loai + mot dong sai loai (bi bo)
    cases = {"v6nat_xoa_loi_0001": "03_xoa_loi", "v6naten_nen_toan_cuc_0002": "02_nen_toan_cuc",
             "v6nat_giong_cuc_bo_0003": "02_giong_cuc_bo", "v6naten_ghep_thay_0004": "03_ghep_giong_that",
             "v6nat_may_thay_0005": "03_chen_giong_may", "v6nat_may_thay_0006": "03_ghep_giong_that"}
    for case, l2 in cases.items():
        p = os.path.join(d, "e2e", case, "anh", "T1.png")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        im.save(p)
        json.dump({"output_chinh": {"thao_tac": [{"id": "T1", "nhan_cap2": l2, "anh": "anh/T1.png", "pham_vi": "mot doan"}]}},
                  open(os.path.join(d, "e2e", case, "ho_so.json"), "w", encoding="utf-8"))
    cands = K.req13_candidates(os.path.join(d, "e2e"))
    assert len(cands) == 5 and {c["class"] for c in cands} == set(K.CLASSES)
    kit, key = os.path.join(d, "kit"), os.path.join(d, "khoa")
    r = K.export_req13(os.path.join(d, "e2e"), kit, key, per_class=8, seed=0)
    assert r["picked"] == 5
    with open(os.path.join(key, "khoa_req13.csv"), encoding="utf-8-sig") as fh:
        keys = list(csv.DictReader(fh))
    sheet = os.path.join(kit, "req13", "phieu_req13.csv")
    with open(sheet, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    assert "dap_an" not in rows[0]
    ans = {k["ma_anh"]: k["dap_an_so"] for k in keys}
    for i, row in enumerate(rows):
        row["nguoi_cham_1"] = ans[row["ma_anh"]]                                # dung het
        row["nguoi_cham_2"] = ans[row["ma_anh"]] if i < 4 else "1" if ans[row["ma_anh"]] != "1" else "2"
        row["nguoi_cham_3"] = K.CLASSES[int(ans[row["ma_anh"]]) - 1]               # ghi bang ten loai
    with open(sheet, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = K.score_req13(sheet, os.path.join(key, "khoa_req13.csv"))
    assert s["per_rater"]["nguoi_cham_1"]["ti_le"] == 1.0 and s["per_rater"]["nguoi_cham_2"]["dung"] == 4
    assert s["pooled"]["dung"] == 14 and s["pooled"]["tong"] == 15 and s["n_raters"] == 3 and s["dat"]
    assert s["pooled_excl_khu_on"]["tong"] == 12 and "dat_excl_khu_on" in s                  # bo 1 anh khu on x 3 nguoi


# ================= chay =================
GROUPS = {"lockbox": ("lockbox",), "expand": ("expand",), "listen": ("listen", "req13"), "groups": ("group", "assign", "balance",
          "build", "envelope", "fold", "person", "donor", "k1axes")}


def run_all(filt: str = "") -> int:
    keys = GROUPS.get(filt, (filt,) if filt else ("",))
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)
             and any(k in n for k in keys)]
    bad = 0
    for n, f in tests:
        t = time.time()
        try:
            f()
            print(f"OK   {n} ({time.time() - t:.2f}s)")
        except Exception:
            bad += 1
            print(f"FAIL {n}\n{traceback.format_exc()}")
    print(f"{len(tests) - bad}/{len(tests)} OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(run_all(sys.argv[1] if len(sys.argv) > 1 else ""))
