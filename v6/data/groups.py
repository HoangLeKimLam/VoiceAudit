# -*- coding: utf-8 -*-
"""NHOM + FOLD v2 (WP5, 10/10): chia 5 fold theo NHOM (v6.common.group_of), can bang theo nguon (kieu StratifiedGroupKFold,
tat dinh, hoa giai bang crc32) -> .tmp_v6/cache/folds_v2.json {id: fold} + folds_v2_groups.json {nhom: fold} (bang dong bang
ma C.fold_of_group doc). KHONG doi detect.person_fold / mo hinh v9 hien tai: folds_v2 chi dung cho lan hoc lai sau.

  nhom      AMI = series · EdAcc = cuoc hoi thoai · VSS = kenh · voxp:None = tung ban goc · du lieu cao = kenh (toi khi phan cum)
            · con lai = nguoi (VIVOS v4: ma nguoi noi, GIU fold cu cua speaker_fold.json = nhom ghim)
  lien ket  (a) donor: nhom NOI nhau qua meta.edits[*].donor_file (ghep loi that lay doan tu ban goc khac: voxp:None lay tu
            ban goc voxp:None KHAC vi build_natural coi chung la mot 'nguoi') -> hop nhat (union-find) thanh mot THANH PHAN,
            ca thanh phan vao cung fold (khong thi cung doan audio nam o ca hoc lan kiem)
            (b) giong dich: edits[*].target_person (+target_source) / ref_person, meta.target.person (VC WP7b: host nguoi A
            noi bang giong nguoi B) -> KHONG gop: do thi VC cua WP7b noi 338 nguoi thanh 16 thanh phan (lon nhat 176 nhom)
            -> gop = mot fold khong lo. Thay vao do dong co nguoi lien ket o fold KHAC bi LOAI KHOI CV (khong co trong
            folds_v2.json, ghi o folds_v2_groups.json["cv_excluded"]); WP7b nen chon giong dich CUNG fold (target_fold_ok)
            Hai thanh phan da ghim KHAC fold cung khong gop (dong noi bi loai khoi CV) -> khong bao gio doi fold nhom ghim
  lockbox   nhom lockbox v2 cap NHOM (new_person / scrape) KHONG dua vao can bang; dong lockbox cap DONG (tts_heldout /
            explicit / linked_*) chi bo dong do khoi can bang, nhom host van can bang. fold = bang cu hoac C.crc_fold_of_group,
            danh dau trong folds_v2_groups.json["lockbox_v2_groups"]. folds_v2 KHONG thay assert_not_lockbox
  k1axes    data_labels/v6/k1axes/cases.json (khong co manifest.json): moi ca nhan fold cua ban goc ('base')
  can bang  tang = "<lang>:<nguon>" (+ cot tong); dat thanh phan lon truoc, chon fold lam tong do lech chuan ti le moi tang nho
            nhat; hoa: fold it tep hon, roi crc32("<nhom>|<k>"). CHI khi chua co bang (lan dau) hoac --reset.
  on dinh   da co bang (--force): MOI nhom deu ghim: nhom trong bang giu fold, nhom MOI ghim vao C.crc_fold_of_group = dung
            gia tri C.fold_of_group da phat cho WP6 (meta.fold) / expand_hosts (fold_v2) -> khong bao gio co 'hai dinh nghia
            fold'. Nhom moi vi vay khong duoc can bang lai (doi lai: on dinh). --reset = chia lai tu dau (lam lech moi fold da ghi)
  bao cao   .tmp_v6/results/folds_v2.md: tep / nhom / tep that goc theo fold x tang, so tep doi fold so voi person_fold,
            nguoi nam o > 1 fold (phai = 0 tru voxp:None), dong co lien ket (donor / giong dich) o fold KHAC trong CV (phai = 0)
  trung id  dupes: 3 id trung trong chain/manifest.json -> doi chieu audio tren dia voi ban goc, de xuat giu / doi ten
            (doi ten = sinh lai wav + trich lai khung XLS-R = viec xep hang) -> .tmp_v6/cache/chain_dupes_plan.json

CLI: python -m v6.data.groups build [--force] [--reset] [--no-v4] | report | dupes | selftest
     (bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.groups ...)"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402

N = C.N_FOLDS
CACHE = os.path.join(C.TMP, "cache")
REPORT_MD = os.path.join(C.RESULTS, "folds_v2.md")
DUPES_PLAN = os.path.join(CACHE, "chain_dupes_plan.json")
ALL = "__tong__"
DONOR_KEYS = ("donor_file",)                 # khoa trong meta.edits[*] tro toi ban goc KHAC co audio bi cat sang
TARGET_KEYS = (("target_person", "target_source"), ("ref_person", "ref_source"))   # nguoi co GIONG trong tep (VC / TTS)
K1AXES_CASES = os.path.join(C.V6DATA, "k1axes", "cases.json")
VIVOS_DONOR = re.compile(r"(VIVOS(?:SPK|DEV)\d+)_")   # ten tep VIVOS -> ma nguoi = nhom v4 (nhu speaker_fold.json)
BIG_COMPONENT_SHARE = 0.10                   # thanh phan gop (donor) > 10 % so dong -> canh bao trong bao cao


# ---------------- doc tap ----------------
def manifest_paths() -> list[str]:
    return sorted(glob.glob(os.path.join(C.V6DATA, "*", "manifest.json")))


def _lang_of(meta: Mapping[str, Any], set_name: str) -> str:
    lang = str(meta.get("lang") or "").lower()
    if lang:
        return lang
    return "en" if set_name in ("naten", "amilong", "ttsen") else "vi"


def donor_groups(meta: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """(nhom cua moi ban goc cho doan audio qua meta.edits[*].donor_file, donor_file khong tra duoc nhom)."""
    out, bad = [], []
    for ed in meta.get("edits") or ():
        if not isinstance(ed, Mapping):
            continue
        for k in DONOR_KEYS:
            df = ed.get(k)
            if not isinstance(df, str) or not df:
                continue
            try:
                out.append(C.group_of({"host_file": df}))
            except ValueError:
                hrow = _extra_hosts().get(df)                    # ban goc cao (data/scrape/hosts_*) / hosts_v2
                vm = VIVOS_DONOR.search(os.path.basename(df))   # v4: 'train_VIVOSSPK12_R001.wav' -> nhom 'VIVOSSPK12'
                if hrow:
                    out.append(hrow["group"])
                elif vm:
                    out.append(vm.group(1))
                else:
                    bad.append(df)
    return out, bad


def _extra_hosts() -> Mapping[str, Mapping[str, Any]]:
    from v6.data import lockbox as LB
    return LB.host_index()


def person_group(person: str, source: Any = None) -> str:
    """Nguoi (+ nguon) -> nhom I2 (AMI nguoi -> series qua hosts_en, EdAcc -> cuoc hoi thoai...); khong ro -> 'unknown:<nguoi>'."""
    src = str(source or (person.split(":", 1)[0] if ":" in person else "")).lower()
    try:
        return C.group_of({"person": person, "source": src})
    except ValueError:
        return f"unknown:{person}"


def target_groups(meta: Mapping[str, Any]) -> list[str]:
    """Nhom cua NGUOI KHAC co giong nam trong tep: edits[*].target_person|ref_person (+ _source), meta.target_person|
    ref_person, meta.target.person (+ .source) va edits[*].target.person. Thu tu giu, khong lap."""
    out: list[str] = []
    for d in [meta] + [ed for ed in meta.get("edits") or () if isinstance(ed, Mapping)]:
        for pk, sk in TARGET_KEYS:
            p = d.get(pk)
            if isinstance(p, str) and p.strip():
                out.append(person_group(p.strip(), d.get(sk)))
        t = d.get("target")
        if isinstance(t, Mapping) and isinstance(t.get("person"), str) and t["person"].strip():
            out.append(person_group(t["person"].strip(), t.get("source")))
    return list(dict.fromkeys(out))


def target_fold_ok(host_group: str, target_person: str, target_source: Any = None) -> bool:
    """Cho WP7b khi chon giong dich VC: True neu nguoi dich CUNG fold v2 voi host (C.fold_of_group) -> dong khong bi loai
    khoi CV. Fold tra ve la cuoi cung (build --force ghim nhom moi vao dung gia tri nay)."""
    return C.fold_of_group(host_group) == C.fold_of_group(person_group(target_person, target_source))


def _item(rid: str, set_name: str, cfg: str, meta: Mapping[str, Any], rules: Any = None) -> dict[str, Any]:
    try:
        group = C.group_of(meta, set_name)
    except ValueError:                      # khong co thong tin nguoi -> nhom rieng tung tep (dem trong bao cao)
        group = f"unknown:{rid}"
    donors, bad_donors = donor_groups(meta)
    targets = [t for t in target_groups(meta) if t != group]
    lock = None
    if rules is not None:
        try:
            lock = rules.reason(meta, set_name)
        except ValueError:
            lock = None
    person = str(meta.get("person") or meta.get("speaker") or "")
    source = str(meta.get("source") or ("vivos" if set_name == "v4" else "?")).lower()
    lang = _lang_of(meta, set_name)
    try:
        pf = C.assign_fold([person]) if person else -1
    except C.LeakageError:
        pf = -1
    return {"id": rid, "set": set_name, "cfg": cfg, "group": group, "person": person, "source": source, "lang": lang,
            "stratum": f"{lang}:{source}", "real": not meta.get("level2"), "pf": pf, "donors": donors,
            "bad_donors": bad_donors, "targets": targets, "lockbox_v2": lock}


def iter_items(include_v4: bool = True, rules: Any = None) -> Iterator[dict[str, Any]]:
    """Moi dong manifest v6 (+ features_v4.pkl) -> dict nhe (id, set, nhom, tang, that/sua, person_fold cu, nhom cho,
    ly do lockbox v2 neu co `rules`)."""
    for mp in manifest_paths():
        s = os.path.basename(os.path.dirname(mp))
        for e in C.read_manifest(mp):
            yield _item(str(e["id"]), s, str(e.get("cfg") or "goc"), e.get("meta") or {}, rules)
    p4 = os.path.join(C.V4, "features_v4.pkl")
    if include_v4 and os.path.exists(p4):
        with open(p4, "rb") as fh:
            v4 = pickle.load(fh)
        for r in (v4.values() if isinstance(v4, dict) else v4):
            yield _item(str(r["id"]), "v4", "goc", r.get("meta") or {}, rules)
        del v4


def k1axes_items(by_id: Mapping[str, Mapping[str, Any]], path: str = K1AXES_CASES) -> tuple[list[dict], list[str]]:
    """Ca k1axes (cases.json: file + base) -> item mang nhom / tang cua ban goc 'base'; base khong thay -> danh sach rieng."""
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as fh:
        cases = json.load(fh)
    out, miss = [], []
    for c in cases:
        b = by_id.get(str(c.get("base") or ""))
        if b is None:
            miss.append(str(c.get("file")))
            continue
        out.append({**b, "id": str(c["file"]), "set": "k1axes", "cfg": "k1axes", "real": False, "donors": [],
                    "bad_donors": [], "targets": []})
    return out, miss


# ---------------- lien ket nhom (union-find) ----------------
class _UF:
    def __init__(self) -> None:
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        r = x
        while self.p[r] != r:
            r = self.p[r]
        while self.p[x] != r:
            self.p[x], x = r, self.p[x]
        return r

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))                    # goc = ten nho nhat -> tat dinh
            self.p[hi] = lo


def link_components(items: Iterable[Mapping[str, Any]],
                    pins: Mapping[str, int] | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """({nhom: dai dien thanh phan}, canh khong gop duoc). Chi gop qua donor_file (KHONG qua giong dich). Hai thanh phan da
    ghim KHAC fold -> khong gop, canh ghi vao danh sach (dong do se bi loai khoi CV thay vi doi fold nhom ghim)."""
    pins = pins or {}
    uf = _UF()
    items = list(items)
    for it in items:
        uf.find(it["group"])
        for d in it.get("donors") or ():
            uf.find(d)
    pin_of = {g: int(f) for g, f in pins.items() if g in uf.p}     # moi nhom luc dau la goc cua chinh no
    skipped = []
    for it in items:
        for d in it.get("donors") or ():
            ra, rb = uf.find(it["group"]), uf.find(d)
            if ra == rb:
                continue
            pa, pb = pin_of.get(ra), pin_of.get(rb)
            if pa is not None and pb is not None and pa != pb:
                skipped.append({"id": it["id"], "set": it["set"], "group": it["group"], "linked": d, "kind": "donor",
                                "folds": [pa, pb]})
                continue
            uf.union(ra, rb)
            r = uf.find(ra)
            pin = pa if pa is not None else pb
            pin_of.pop(ra, None)
            pin_of.pop(rb, None)
            if pin is not None:
                pin_of[r] = pin
    return {g: uf.find(g) for g in list(uf.p)}, skipped


def components(items: Iterable[Mapping[str, Any]], pins: Mapping[str, int] | None = None) -> dict[str, str]:
    """{nhom: dai dien thanh phan} — nhom noi qua donor_file gop lai; nhom khong lien ket = chinh no."""
    return link_components(items, pins)[0]


def link_cross_fold(items: Iterable[Mapping[str, Any]], fold_of: Any, kinds: Iterable[str] = ("donors", "targets"),
                    skip: Iterable[str] = ()) -> dict[str, Any]:
    """Dong co nhom lien ket (donor / giong dich) o fold KHAC fold cua chinh no. fold_of: dict hoac ham nhom -> fold
    (ham: nhom nao cung tra duoc). skip: id da loai khoi CV. Trong CV phai = 0."""
    look = fold_of if callable(fold_of) else (lambda g: fold_of.get(g))
    skip = set(skip)
    bad: Counter = Counter()
    by_kind: Counter = Counter()
    ex = []
    for it in items:
        if it["id"] in skip:
            continue
        f = look(it["group"])
        for k in kinds:
            hit = next((d for d in it.get(k) or () if look(d) is not None and look(d) != f), None)
            if hit is not None:
                bad[it["set"]] += 1
                by_kind[k] += 1
                if len(ex) < 5:
                    ex.append((it["id"], it["group"], hit, k))
                break
    return {"n": sum(bad.values()), "by_set": dict(bad), "by_kind": dict(by_kind), "examples": ex}


def donor_cross_fold(items: Iterable[Mapping[str, Any]], fold_of: Mapping[str, int]) -> dict[str, Any]:
    """Dong co doan cho (donor) nam o fold KHAC fold cua chinh no -> phai = 0 (nhom donor khong co trong fold_of: bo qua)."""
    return link_cross_fold(items, fold_of, ("donors",))


# ---------------- chia can bang ----------------
def assign_stratified(counts: Mapping[str, Mapping[str, int]], n_folds: int = N,
                      pinned: Mapping[str, int] | None = None) -> dict[str, int]:
    """counts: {nhom: {tang: so tep}} -> {nhom: fold}. Tham lam kieu StratifiedGroupKFold, tat dinh.
    pinned: nhom da co fold (giu nguyen, tinh vao tai cua fold truoc khi xep nhom con lai)."""
    pinned = {g: int(f) for g, f in (pinned or {}).items() if g in counts}
    bad = {g: f for g, f in pinned.items() if not 0 <= f < n_folds}
    if bad:
        raise ValueError(f"fold ghim ngoai 0..{n_folds - 1}: {list(bad.items())[:5]}")
    strata = sorted({s for c in counts.values() for s in c})
    col = {s: i for i, s in enumerate(strata)}
    tot = np.zeros(len(strata) + 1)

    def vec(g: str) -> np.ndarray:
        v = np.zeros(len(strata) + 1)
        for s, n in counts[g].items():
            v[col[s]] += n
        v[-1] = v[:-1].sum()
        return v

    vecs = {g: vec(g) for g in counts}
    for v in vecs.values():
        tot += v
    tot[tot == 0] = 1.0
    F = np.zeros((n_folds, len(strata) + 1))
    out = dict(pinned)
    for g, f in pinned.items():
        F[f] += vecs[g]
    order = sorted((g for g in counts if g not in pinned), key=lambda g: (-vecs[g][-1], zlib.crc32(g.encode("utf-8"))))
    for g in order:
        v = vecs[g]
        best = None
        for k in range(n_folds):
            F[k] += v
            cost = float((F / tot).std(axis=0).sum())
            F[k] -= v
            key = (round(cost, 12), F[k][-1], zlib.crc32(f"{g}|{k}".encode("utf-8")))
            if best is None or key < best[0]:
                best = (key, k)
        out[g] = best[1]
        F[best[1]] += v
    return out


def group_counts(items: Iterable[Mapping[str, Any]]) -> dict[str, Counter]:
    c: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        c[it["group"]][it["stratum"]] += 1
    return dict(c)


def v4_pins(items: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Nhom VIVOS (tap v4) -> fold cu trong speaker_fold.json (giu GroupKFold cua features_v4)."""
    m = C.speaker_folds()
    return {it["group"]: m[it["group"]] for it in items if it["set"] == "v4" and it["group"] in m}


# ---------------- bao cao ----------------
def balance(items: list[Mapping[str, Any]], fold_of: Mapping[str, int]) -> dict[str, Any]:
    """Can bang theo fold x tang: so tep, so nhom, so tep THAT goc tu nhien (natvi/naten), doi fold so voi person_fold."""
    files: dict[str, list[int]] = defaultdict(lambda: [0] * N)
    groups: dict[str, list[set]] = defaultdict(lambda: [set() for _ in range(N)])
    real_goc: dict[str, list[int]] = defaultdict(lambda: [0] * N)
    old_pf: dict[str, list[int]] = defaultdict(lambda: [0] * N)
    person_folds: dict[str, set] = defaultdict(set)
    moved = 0
    for it in items:
        f = fold_of[it["group"]]
        s = it["stratum"]
        for key in (s, ALL):
            files[key][f] += 1
            groups[key][f].add(it["group"])
        if it["set"] in ("natvi", "naten") and it["cfg"] == "goc" and it["real"]:
            real_goc[it["lang"]][f] += 1
            if 0 <= it["pf"] < N:
                old_pf[it["lang"]][it["pf"]] += 1
        if it["person"]:
            person_folds[it["person"]].add(f)
        moved += int(0 <= it["pf"] < N and it["pf"] != f)
    split = sorted(p for p, fs in person_folds.items() if len(fs) > 1)
    return {"files": dict(files), "groups": {k: [len(x) for x in v] for k, v in groups.items()},
            "real_goc_v2": dict(real_goc), "real_goc_person_fold": dict(old_pf), "moved_vs_person_fold": moved,
            "persons_in_several_folds": len(split),
            "persons_in_several_folds_non_voxp_none": [p for p in split if not C._person_missing(p)][:50]}


def _md_table(title: str, rows: Mapping[str, list[int]]) -> list[str]:
    out = [f"### {title}", "", "| tang | " + " | ".join(f"f{k}" for k in range(N)) + " | max/min |", "|---" * (N + 2) + "|"]
    for k in sorted(rows, key=lambda x: (x == ALL, x)):
        v = rows[k]
        ratio = (max(v) / min(v)) if min(v) else float("inf")
        out.append(f"| {k} | " + " | ".join(str(x) for x in v) + f" | {ratio:.2f} |")
    return out + [""]


def write_report(bal: Mapping[str, Any], dups: Mapping[str, Any], n_items: int, n_groups: int, path: str = REPORT_MD,
                 extra: Mapping[str, Any] | None = None) -> str:
    ex = dict(extra or {})
    lk = ex.get("links") or {}
    cross = lk.get("donor_cross_fold") or {}
    L = ["# folds_v2 — chia 5 fold theo NHOM, can bang theo nguon (v6.data.groups)", "",
         f"Viet luc {time.strftime('%Y-%m-%d %H:%M')}. {n_items} dong, {n_groups} nhom. "
         "Nhom: AMI = series, EdAcc = cuoc hoi thoai, VSS = kenh, voxp:None = tung ban goc, du lieu cao = kenh, con lai = nguoi.",
         "Chua dung o mo hinh nao: v9 van dung detect.person_fold. Dung cho lan hoc lai sau (H6).", "",
         f"- Nhom noi qua edits[].donor_file (gop cung fold): {lk.get('n_linked_groups', 0)} nhom trong "
         f"{lk.get('n_components_multi', 0)} thanh phan (lon nhat {lk.get('largest_component', 0)} nhom); "
         f"donor_file khong tra duoc nhom: {lk.get('n_bad_donor_files', len(lk.get('bad_donor_files') or []))} tep / "
         f"{lk.get('n_bad_donor_rows', '?')} dong (ten VIVOS da tra ve ma nguoi)",
         f"- Che do: {ex.get('mode', '?')} ('pinned' = da co bang: MOI nhom ghim, nhom moi = crc32('fold:'+nhom) % 5 = fold "
         f"C.fold_of_group da phat; 'balanced' = lan dau / --reset). Nhom moi co dong ghim lan nay: "
         f"{ex.get('n_new_groups_pinned_crc', 0)}; nhom moi chi la donor / giong dich (ghi vao bang theo thanh phan / fold du "
         f"phong): {ex.get('n_new_link_only_groups', 0)}",
         f"- Dong co doan cho (donor) o fold KHAC trong CV (phai = 0): {cross.get('n', 0)} {cross.get('by_set', {})}",
         f"- Dong co GIONG nguoi khac (VC / TTS: target_person / ref_person): {lk.get('n_target_rows', 0)}; lien ket "
         f"(donor + giong dich) o fold KHAC trong CV (phai = 0): {(lk.get('link_cross_fold_in_cv') or {}).get('n', 0)}",
         f"- LOAI KHOI CV (nguoi lien ket o fold khac; khong co trong folds_v2.json, xem folds_v2_groups.json['cv_excluded']): "
         f"{sum((lk.get('cv_excluded_by_kind') or {}).values())} dong {lk.get('cv_excluded_by_kind', {})} "
         f"theo tap {lk.get('cv_excluded_by_set', {})}. Giong dich: WP7b nen chon cung fold (v6.data.groups.target_fold_ok)",
         f"- Hai thanh phan ghim khac fold khong gop (dong noi -> loai khoi CV): {lk.get('n_skipped_unions', 0)}; "
         f"ghim mau thuan trong mot thanh phan: {len(lk.get('pin_conflicts') or [])}",
         f"- Thanh phan gop (donor) lon nhat: {lk.get('largest_multi_component_rows', 0)} dong"
         + (f" > {int(100 * BIG_COMPONENT_SHARE)} % so dong -> CANH BAO can bang" if lk.get("big_multi_component") else ""),
         f"- Lockbox v2: {len(ex.get('lockbox_v2_groups') or [])} nhom (cap nhom: new_person / scrape) / "
         f"{ex.get('n_lockbox_v2_items', 0)} dong (ke ca dong cap dong: tts_heldout / explicit / linked_*), KHONG tinh "
         "vao can bang (fold = bang cu hoac crc32('fold:'+nhom) % 5, chi de day du). Code hoc van phai goi "
         "v6.data.lockbox.assert_not_lockbox.",
         f"- k1axes (cases.json, khong co manifest): {(ex.get('k1axes') or {}).get('n', 0)} ca nhan fold cua ban goc 'base' "
         f"(base khong thay: {len((ex.get('k1axes') or {}).get('base_missing') or [])}); co trong folds_v2.json, ngoai bang can bang",
         f"- So dong doi fold so voi person_fold: {bal['moved_vs_person_fold']}",
         f"- Nguoi nam o > 1 fold: {bal['persons_in_several_folds']} "
         f"(khong tinh voxp:None: {len(bal['persons_in_several_folds_non_voxp_none'])} -> "
         f"{bal['persons_in_several_folds_non_voxp_none'][:10]})",
         f"- Id trung trong mot manifest: {dups.get('n_dup_ids', 0)} ({dups.get('examples', [])})", ""]
    L += _md_table("So dong theo fold x tang", bal["files"])
    L += _md_table("So nhom theo fold x tang", bal["groups"])
    L += _md_table("Tep THAT goc tu nhien (natvi/naten) theo fold v2", bal["real_goc_v2"])
    L += _md_table("Doi chieu: tep THAT goc theo person_fold cu (v9)", bal["real_goc_person_fold"])
    L += ["Ghi chu: VSS = 1 kenh lon (vss:nguyenkhangofficial) -> nam tron trong mot fold la dung luat (nguoi cham khong o tap hoc);",
          "can bang chi lam duoc o cac nguon con lai + tong. Nguong / FA phai bao cao THEO NGUON."]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


# ---------------- build ----------------
def _load_frozen(path: str | None = None) -> dict[str, int]:
    p = path or C.FOLDS_V2_GROUPS
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return {str(k): int(v) for k, v in json.load(fh).get("groups", {}).items()}


GROUP_LEVEL_LOCK = ("new_person:", "scrape:")     # ly do lockbox ca NHOM (con lai: chi dong do)


def _row_exclusion(it: Mapping[str, Any], fold_of: Any) -> dict[str, Any] | None:
    """Ly do loai MOT dong khoi CV: nhom lien ket (donor khong gop duoc / giong dich) o fold KHAC host; None = giu."""
    f = fold_of(it["group"])
    for k in ("donors", "targets"):
        hit = next((d for d in it.get(k) or () if fold_of(d) != f), None)
        if hit is not None:
            return {"set": it["set"], "group": it["group"], "linked": hit, "kind": k[:-1], "folds": [f, fold_of(hit)]}
    return None


def _cv_excluded(items: Iterable[Mapping[str, Any]], fold_of: Any) -> dict[str, dict[str, Any]]:
    """{id: ly do} cho id bi loai khoi CV; trung id: dong SAU CUNG quyet dinh (giong detect.load_items)."""
    out: dict[str, dict[str, Any]] = {}
    for it in items:
        why = _row_exclusion(it, fold_of)
        if why is None:
            out.pop(it["id"], None)
        else:
            out[it["id"]] = why
    return out


def build(force: bool = False, reset: bool = False, include_v4: bool = True, out_ids: str = C.FOLDS_V2,
          out_groups: str = C.FOLDS_V2_GROUPS, report: str = REPORT_MD) -> dict[str, Any]:
    """Tinh folds_v2. Da co bang -> chi ghi lai khi --force: MOI nhom ghim (bang cu, nhom moi = C.crc_fold_of_group ->
    khop gia tri C.fold_of_group da phat), khong can bang lai. Lan dau / --reset: can bang theo nguon."""
    if os.path.exists(out_groups) and not force:
        raise C.V6Error(f"{out_groups} da co (bang dong bang). Dung --force de them nhom moi (moi fold da phat giu nguyen), "
                        "--force --reset de chia lai tu dau (lam lech moi so lieu / meta.fold da dung folds_v2)")
    rules = _lockbox_rules()
    items = list(iter_items(include_v4, rules))
    by_id = {it["id"]: it for it in items}
    k1, k1_miss = k1axes_items(by_id)     # ca k1axes: chi nhan fold cua base, khong tinh vao can bang / dem trung id
    per_id: dict[str, set] = defaultdict(set)
    for it in items:
        per_id[it["id"]].add(it["group"])
    dup_ids = sorted(i for i, c in Counter(it["id"] for it in items).items() if c > 1)
    conflict = sorted(i for i in dup_ids if len(per_id[i]) > 1)
    lock_groups = sorted({it["group"] for it in items if str(it.get("lockbox_v2") or "").startswith(GROUP_LEVEL_LOCK)})
    lock_set = set(lock_groups)
    train = [it for it in items if it["group"] not in lock_set and not it.get("lockbox_v2")]
    frozen = {} if reset else _load_frozen(out_groups)
    linked = {d for it in items for k in ("donors", "targets") for d in it.get(k) or ()}
    all_groups = {it["group"] for it in items} | linked
    pins = {**v4_pins(items), **frozen}    # bang cu thang (on dinh); lan dau: chi v4 ghim speaker_fold.json
    row_groups = {it["group"] for it in items}
    if frozen:                             # da phat fold -> nhom moi CO DONG ghim vao fold du phong DA PHAT (khong can bang lai)
        # nhom CHI lam donor (khong co dong nao mang nhom do -> chua ai ghi fold cua no) KHONG ghim: theo thanh phan cua
        # host (ban 10/10 khong ghi cac nhom nay vao bang -> 4 nhom voxp:None donor, sua: nay ghi vao bang)
        pins.update({g: C.crc_fold_of_group(g) for g in row_groups if g not in pins})
    comp, skipped = link_components(train, pins)
    gfold, pin_conflicts = assign_components(train, comp, pins)
    root_fold = {comp[g]: f for g, f in gfold.items() if g in comp}
    for g, r in comp.items():              # nhom chi xuat hien lam donor: theo thanh phan cua no
        if g not in gfold and r in root_fold:
            gfold[g] = root_fold[r]
    for g in sorted(all_groups - set(gfold)):   # lockbox / giong dich / donor le: bang cu hoac fold du phong (ngoai can bang)
        gfold[g] = pins.get(g, C.crc_fold_of_group(g))
    excluded = _cv_excluded(items, lambda g: gfold.get(g, C.crc_fold_of_group(g)))
    ids: dict[str, int] = {}
    for it in items + k1:                  # trung id: dong sau cung thang (giong detect.load_items)
        if it["id"] in excluded:
            ids.pop(it["id"], None)
        else:
            ids[it["id"]] = gfold[it["group"]]
    bal = balance(train, gfold)
    cross = link_cross_fold(items, gfold, ("donors",), skip=excluded)
    cross_all = link_cross_fold(items, lambda g: gfold.get(g, C.crc_fold_of_group(g)), skip=excluded)
    links = _links_summary(items, train, comp, skipped, pin_conflicts, cross, cross_all, excluded)
    dups = {"n_dup_ids": len(dup_ids), "group_conflict": conflict, "examples": dup_ids[:5]}
    n_lock = sum(1 for it in items if it["group"] in lock_set or it.get("lockbox_v2"))
    extra = {"lockbox_v2_groups": lock_groups, "n_lockbox_v2_items": n_lock, "links": links,
             "k1axes": {"n": len(k1), "base_missing": k1_miss[:20]}, "mode": "pinned" if frozen else "balanced",
             "n_new_groups_pinned_crc": sum(1 for g in row_groups if frozen and g not in frozen),
             "n_new_link_only_groups": sum(1 for g in all_groups - row_groups if g not in frozen)}
    C._atomic_write_json(out_ids, ids)
    C._atomic_write_json(out_groups, {"version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_folds": N,
                                      "rule": "group_of (AMI series / EdAcc hoi thoai / VSS kenh / voxp:None tung ban goc / "
                                              "cao = kenh / con lai nguoi); nhom noi qua edits[].donor_file gop cung fold; "
                                              "dong co donor/giong dich o fold khac -> cv_excluded (khong co trong folds_v2.json); "
                                              "can bang theo lang:nguon CHI lan dau / --reset (khong tinh lockbox v2); "
                                              "--force ghim moi nhom (nhom moi = crc32('fold:'+nhom) % 5 = fold da phat); "
                                              "v4 ghim speaker_fold.json; k1axes theo base",
                                      "n_items": len(items), "duplicate_ids": dups, "balance": bal, **extra,
                                      "cv_excluded": excluded,
                                      "components": {g: r for g, r in comp.items() if g != r}, "groups": gfold})
    C._frozen_group_folds.cache_clear()
    write_report(bal, dups, len(items), len(gfold), report, extra)
    return {"n_items": len(items), "n_ids": len(ids), "n_groups": len(gfold), "dups": dups, "mode": extra["mode"],
            "moved_vs_person_fold": bal["moved_vs_person_fold"], "donor_cross_fold": cross["n"],
            "link_cross_fold_in_cv": cross_all["n"], "cv_excluded": len(excluded),
            "cv_excluded_by_kind": dict(Counter(v["kind"] for v in excluded.values())),
            "n_components_multi": links["n_components_multi"], "lockbox_v2_groups": len(lock_groups),
            "n_new_groups_pinned_crc": extra["n_new_groups_pinned_crc"], "n_new_link_only_groups": extra["n_new_link_only_groups"],
            "k1axes": len(k1), "pin_conflicts": len(pin_conflicts), "report": report}


def _links_summary(items: list[Mapping[str, Any]], train: list[Mapping[str, Any]], comp: Mapping[str, str],
                   skipped: list, pin_conflicts: list, cross: Mapping, cross_all: Mapping,
                   excluded: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    csize = Counter(comp.values())
    rows = Counter(comp.get(it["group"], it["group"]) for it in train)
    big = max(rows.values()) if rows else 0
    big_multi = max((n for r, n in rows.items() if csize[r] > 1), default=0)
    return {"n_linked_groups": sum(1 for r in comp.values() if csize[r] > 1),
            "n_components_multi": sum(1 for n in csize.values() if n > 1),
            "largest_component": max(csize.values()) if csize else 0,
            "largest_multi_component_rows": big_multi, "largest_rows": big,
            "big_multi_component": bool(train) and big_multi > BIG_COMPONENT_SHARE * len(train),
            "n_bad_donor_files": len({d for it in items for d in it.get("bad_donors") or ()}),
            "n_bad_donor_rows": sum(1 for it in items if it.get("bad_donors")),
            "bad_donor_files": sorted({d for it in items for d in it.get("bad_donors") or ()})[:20],
            "n_target_rows": sum(1 for it in items if it.get("targets")),
            "donor_cross_fold": dict(cross), "link_cross_fold_in_cv": dict(cross_all),
            "skipped_unions": skipped[:20], "n_skipped_unions": len(skipped), "pin_conflicts": pin_conflicts[:20],
            "cv_excluded_by_set": dict(Counter(v["set"] for v in excluded.values())),
            "cv_excluded_by_kind": dict(Counter(v["kind"] for v in excluded.values()))}


def _lockbox_rules() -> Any:
    """Luat lockbox v2 neu da dong bang (khong thi None: chua co du lieu moi nao de loai)."""
    try:
        from v6.data import lockbox as LB
        return LB.Rules.from_state()
    except C.V6Error:
        return None


def assign_components(items: list[Mapping[str, Any]], comp: Mapping[str, str],
                      pins: Mapping[str, int]) -> tuple[dict[str, int], list[dict]]:
    """Can bang tren THANH PHAN (nhom noi qua donor) roi tra fold cho tung nhom. Ghim: thanh phan nhan fold ghim cua nhom
    ghim nhieu tep nhat; ghim mau thuan trong mot thanh phan -> ghi lai (pin_conflicts)."""
    gc = group_counts(items)
    ccounts: dict[str, Counter] = defaultdict(Counter)
    for g, c in gc.items():
        ccounts[comp.get(g, g)].update(c)
    members: dict[str, list[str]] = defaultdict(list)
    for g in gc:
        members[comp.get(g, g)].append(g)
    cpins, conflicts = {}, []
    for r, gs in members.items():
        ps = {g: int(pins[g]) for g in gs if g in pins}
        if not ps:
            continue
        best = max(ps, key=lambda g: (sum(gc[g].values()), g))
        cpins[r] = ps[best]
        if len(set(ps.values())) > 1:
            conflicts.append({"component": r, "pins": ps, "chosen": ps[best]})
    cfold = assign_stratified(ccounts, N, cpins)
    return {g: cfold[comp.get(g, g)] for g in gc}, conflicts


# ---------------- trung id chain ----------------
def _env(y: np.ndarray, sr: int, hop_s: float = 0.02) -> np.ndarray:
    h = max(1, int(sr * hop_s))
    n = len(y) // h
    e = np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1) + 1e-10)
    return np.log(e)


def envelope_match(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Tuong quan (lon nhat theo do lech +-0,5 s) giua duong bao log-nang luong cua hai tin hieu (chiu duoc ma hoa)."""
    ea, eb = _env(a, sr), _env(b, sr)
    n = min(len(ea), len(eb))
    if n < 10:
        return 0.0
    best = -1.0
    for lag in range(-25, 26):
        x = ea[max(0, lag): n + min(0, lag)]
        y = eb[max(0, -lag): n - max(0, lag)]
        m = min(len(x), len(y))
        if m < 10 or x[:m].std() == 0 or y[:m].std() == 0:
            continue
        best = max(best, float(np.corrcoef(x[:m], y[:m])[0, 1]))
    return best


def _host_wav(host_file: str) -> str | None:
    for d in ("hosts", "hosts_en"):
        for p in glob.glob(os.path.join(C.BASE, "data", "natural_v6", d, "*", host_file + ".wav")):
            return p
    return None


def chain_dupes(manifest: str | None = None, out: str = DUPES_PLAN, write: bool = True) -> dict[str, Any]:
    """Id trung trong chain/manifest.json: doi chieu wav tren dia (dong sau ghi de dong truoc) voi tung ban goc.
    De xuat: dong khop audio giu id; dong con lai -> id moi '<id>_d<j>' (wav bi ghi de -> phai SINH LAI + trich lai khung)."""
    import soundfile as sf
    mp = manifest or os.path.join(C.V6DATA, "chain", "manifest.json")
    man = C.read_manifest(mp)
    rows = defaultdict(list)
    for i, e in enumerate(man):
        rows[e["id"]].append((i, e))
    plan = []
    for rid, rs in sorted(rows.items()):
        if len(rs) < 2:
            continue
        wav = C.abs_wav(rs[0][1]["wav"])
        y, sr = sf.read(wav, dtype="float32", always_2d=True) if os.path.exists(wav) else (np.zeros((0, 1)), C.SR)
        y = y.mean(1)
        cands = []
        for i, e in rs:
            m = e.get("meta") or {}
            hp = _host_wav(str(m.get("host_file") or ""))
            score = None
            if hp and len(y):
                h, hsr = sf.read(hp, dtype="float32", always_2d=True)
                score = round(envelope_match(y, h.mean(1), sr), 4) if hsr == sr else None
            cands.append({"row": i, "host_file": m.get("host_file"), "person": m.get("person"), "lang": m.get("lang"),
                          "kind": e.get("kind"), "cond": m.get("cond"), "host_wav": hp, "match": score})
        scored = [c for c in cands if c["match"] is not None]
        keep = max(scored, key=lambda c: c["match"])["row"] if scored else rs[-1][0]
        j = 0
        for c in cands:
            if c["row"] == keep:
                c["action"], c["new_id"] = "giu", rid
            else:
                j += 1
                c["action"], c["new_id"] = "doi_ten_sinh_lai", f"{rid}_d{j}"
        plan.append({"id": rid, "wav": wav, "wav_dur_s": round(len(y) / sr, 3) if len(y) else None, "rows": cands,
                     "keep_row": keep, "last_row_wins_now": rs[-1][0]})
    res = {"manifest": mp, "n_rows": len(man), "n_unique": len(rows), "duplicates": plan,
           "note": "KHONG sua manifest (khong thuoc WP5). Doi ten -> sinh lai wav dong bi ghi de + trich xlsr_frames/xlsr_pooled "
                   "cho id moi (viec xep hang). Trong luc cho: detect.load_items lay dong SAU CUNG -> kiem keep_row == last_row_wins_now."}
    if write:
        C._atomic_write_json(out, res)
    return res


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Nhom + fold v2 can bang theo nguon; id trung chain")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", help="tinh folds_v2.json + folds_v2_groups.json + results/folds_v2.md")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--no-v4", action="store_true")
    sub.add_parser("report", help="in can bang tu folds_v2_groups.json")
    sub.add_parser("dupes", help="ke hoach go 3 id trung chain -> cache/chain_dupes_plan.json")
    sub.add_parser("selftest", help="chay kiem thu cua v6.data")
    a = ap.parse_args()
    if a.cmd == "build":
        out: Any = build(a.force, a.reset, not a.no_v4)
    elif a.cmd == "report":
        with open(C.FOLDS_V2_GROUPS, encoding="utf-8") as fh:
            d = json.load(fh)
        out = {k: d[k] for k in ("created", "n_items", "duplicate_ids")} | {"files": d["balance"]["files"]}
    elif a.cmd == "dupes":
        out = chain_dupes()
    else:
        from v6.data import test_groups
        sys.exit(test_groups.run_all())
    print(json.dumps(out, ensure_ascii=False, indent=1, default=C._json_default)[:6000])


if __name__ == "__main__":
    main()
