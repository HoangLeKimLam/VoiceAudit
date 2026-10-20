# -*- coding: utf-8 -*-
"""LOCKBOX v2 (WP5, 10/10): tap khoa MOI cho du lieu MOI + chot chan cho code hoc / hieu chinh goi sau nay.

LOCKBOX v1 DA DUNG HET (CONSUMED_V1): v6.common.lockbox_reason danh dau nguoi tieng Anh crc32(speaker) % 2 == 1 + AMI TS3003
nhung KHONG module v9 nao goi is_lockbox / split_lockbox (detect.load_items, ft.light_items, xlsr_frames, aug_stream, heads2
chi doc meta.lockbox, True cho 30 cua so amilong) -> nhung nguoi do da vao hoc TCN / FT-2 / heads2 va kho nguong fuse9;
TS3003 con nam trong naten / cover / aug / chain qua ma host 'ami_test-*'. Khong con so nao tren v1 la uoc luong tap khoa.

LUAT v2 (ap cho du lieu MOI; thu tu kiem: explicit -> tts_heldout -> scrape -> new_person -> linked):
  explicit      meta['lockbox_v2'] is True
  tts_heldout   giong may / doi giong sinh boi engine role = 'heldout' trong v6/tts_multi/engines.json; ten engine lay o
                meta.{engine, tts, tts_engine, vc, vc_engine, gen_engine} VA moi meta.edits[*].{tts, engine, vc, vc_engine,
                tts_engine} (quy uoc hien co: edits[i].tts = 'viXTTS'), them gen_chain (list) o ca hai cap; chuoi lai
                WP7b '<tts>+<vc>' tach theo '+'; hoac meta.engine_role / edits[i].role|engine_role == 'heldout'.
                KHONG nha duoc bang lockbox_v2=False
  scrape        ban ghi cao (url / channel_id / meta.scrape / meta.scraped, nhom 'scrape:*', nguon yt|ia|ccc|fosdem|scrape*,
                tap scrape*, hoac host_file thuoc data/scrape/hosts_*/hosts.jsonl) co zlib.crc32(KHOA KENH) % 5 == 0.
                Khoa kenh = '<nguon>:<channel_id>' (meta.channel_id | meta.scrape.channel_id | dong host cao) -> ON DINH khi
                WP6 phan cum nguoi trong kenh (meta.group doi, kenh khong doi); khong co channel_id -> nhom. Du lieu 12/10:
                khoa kenh == nhom tren ca 45 ban goc cao -> khong dong nao doi trang thai.
                'license' KHONG dung de nhan dien (ban ghi kho cong khai cung co the ghi giay phep)
  new_person    nguon tuyen chon (curated_sources trong lockbox_v2.json; mac dinh theo dac ta: vox + celeb) co NHOM
                (C.group_of) KHONG nam trong anh chup nhom da dung (used_groups: moi nhom trong data_labels/v6/*/manifest.json +
                hosts*.jsonl luc dong bang dau tien) va KHONG nam trong released_groups. released_groups CHI lay tu ke hoach
                expand_hosts (nguoi lockbox_v2=False do --dev-share), hosts_v2 role=dev va --release ghi ro; mot dong manifest
                bat ky co lockbox_v2=False KHONG nha duoc gi. Mo rong nguon (vd tieng Anh) = quyet dinh nguoi dung:
                python -m v6.data.lockbox curate --add edacc ami voxp librispeech --by "<ai, ngay>" (ghi vao tep, co lich su)
  linked        tep chua audio / GIONG cua nguoi khac: edits[*].donor_file (ban goc cho doan) va giong dich VC / TTS
                (edits[*].target_person|ref_person, meta.target.person) -> nguoi lien ket la new_person ->
                'linked_new_person:<nhom>', ban goc cho la cao lockbox -> 'linked_scrape:<kenh>' (cap DONG, nhom host van dung)
Tep dong bang .tmp_v6/cache/lockbox_v2.json: luat, anh chup used_groups, released_groups, curated_sources, items {id: sha1, wav,
ly do, nhom, host_file}. Chi THEM, khong bao gio xoa / doi: sha1 cu khac -> loi.
CHOT CHAN assert_not_lockbox(ids|recs): id (str) duoc TRA NGUOC ra meta qua chi muc song (manifest v6 + hosts tu nhien + hosts
cao + hosts_v2, doc lai khi tep doi) roi ap LUAT -> du lieu tao SAU lan dong bang cuoi van bi chan (ban 11/10: chi so voi muc da
dong bang -> 17 ban goc cao lockbox lot qua). Id khong tra duoc -> C.V6Error (truyen ban ghi day du, hoac allow_unknown_ids=True).
Thieu tep / chua dong bang -> C.V6Error (khong im lang cho qua), tru khi allow_unfrozen=True.
Luu y fold: v6.data.groups bo nhom lockbox v2 khoi can bang fold; folds_v2 KHONG thay duoc assert_not_lockbox.

CLI: python -m v6.data.lockbox freeze [--dry-run] [--jsonl P ...] [--release NHOM ...] | status | check <manifest>
     | curate [--add NGUON ...] [--remove NGUON ...] --by "<ai, ngay>" [--note ...] [--dry-run] | selftest
     (bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.lockbox ...)"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import zlib
from collections import Counter
from typing import Any, Iterable, Iterator, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402

LOCKBOX_V2 = os.path.join(C.TMP, "cache", "lockbox_v2.json")
ENGINES_JSON = os.path.join(C.BASE, "v6", "tts_multi", "engines.json")
CURATED_SOURCES = frozenset({"vox", "celeb"})        # dac ta WP5: nguoi VoxVietnam / Vietnam-Celeb moi (mo rong = curate)
KNOWN_SOURCES = frozenset({"vox", "celeb", "vivos", "edacc", "ami", "voxp", "librispeech"})   # quyet dinh 2: kho cong khai
SCRAPE_SOURCES = frozenset({"yt", "ia", "ccc", "fosdem", "scrape"})
SCRAPE_MOD = 5
ENGINE_KEYS = ("engine", "tts", "tts_engine", "vc", "vc_engine", "gen_engine")
EDIT_ENGINE_KEYS = ("tts", "engine", "vc", "vc_engine", "tts_engine")
NEW_HOSTS = (os.path.join(C.BASE, "data", "natural_v6", "hosts_v2", "hosts.jsonl"),)
EXPAND_PLAN = os.path.join(C.TMP, "cache", "expand_hosts_plan.json")
# truong giu lai trong chi muc tra nguoc id -> meta (du cho luat; bo words / text de chi muc nho)
META_KEYS = ENGINE_KEYS + ("lockbox_v2", "engine_role", "gen_chain", "host_file", "group", "person", "speaker", "cluster",
                           "source", "url", "channel_id", "scraped", "meeting", "series", "src", "base", "target_person",
                           "target_source", "ref_person", "ref_source", "target", "role")
EDIT_KEYS = EDIT_ENGINE_KEYS + ("gen_chain", "role", "engine_role", "donor_file", "target_person", "target_source",
                                "ref_person", "ref_source", "target")


def scrape_hosts_paths() -> tuple[str, ...]:
    """I4 (hosts_vi / hosts_en) + moi tap cao khac WP6 tao them (vd hosts_long_vi / hosts_long_en) — glob LUC GOI."""
    base = os.path.join(C.BASE, "data", "scrape")
    return tuple(sorted({os.path.join(base, f"hosts_{l}", "hosts.jsonl") for l in ("vi", "en")}
                        | {os.path.normpath(p) for p in glob.glob(os.path.join(base, "hosts_*", "hosts.jsonl"))}))


SCRAPE_HOSTS = scrape_hosts_paths()                  # anh chup luc nap (tuong thich); code ben duoi goi lai ham

CONSUMED_V1 = {
    "status": "consumed",
    "declared": "2026-10-10",
    "what": "lockbox v1 = v6.common.lockbox_reason: nguoi tieng Anh crc32(speaker)%2==1, AMI TS3003, bo hard, MIST, cau hinh nen la",
    "why": ["detect.load_items / ft.light_items / xlsr_frames / aug_stream / heads2 chi loc meta.lockbox (True cho 30 cua so amilong)",
            "4 283 / 9 064 dong naten (1 661 / 3 904 goc) co C.lockbox_reason != None (moi luat v1) da vao hoc TCN, FT-2, "
            "heads2 va kho nguong fuse9",
            "TS3003 nam trong naten (175 goc + 249 bien the), cover 78, cover2 13, aug 164, chain 66, benign 9 (ma host khong chua ma hop)",
            "e2e_v9 va req_check_v9A da cham ca nguoi lockbox v1"],
    "consequence": "Moi so lieu tieng Anh / amilong hien co la so dev, khong phai tap khoa. Lay lai v1 = hoc lai khong co nhung nguoi do.",
}
EN_LOCKBOX_STATUS = (   # 12/10, do tren dia (D:/tmp_va/wp5/probe_en.py): ghi vao lockbox_v2.json de nguoi doc thay
    "Tieng Anh CHUA co tap khoa nguoi moi: EdAcc test 30/30 cuoc hoi thoai va AMI test 4/4 series da dung het; VoxPopuli "
    "con ~17 nguoi chua dung (~173 dong); LibriSpeech chua co tren dia. Chi co cao EN kenh crc32 % 5 == 0. Can quyet dinh: "
    "(a) curate --add voxp (17 nguoi), (b) tai them (LibriSpeech test-clean/other, AMI / EdAcc dev) roi curate, "
    "(c) giu kenh cao EN lam tap khoa duy nhat.")


# ---------------- tien ich ----------------
def sha1_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_state(path: str = LOCKBOX_V2) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def heldout_engines(path: str = ENGINES_JSON) -> frozenset[str]:
    """Ten engine role=heldout trong registry I3 (chua co tep -> rong)."""
    if not os.path.exists(path):
        return frozenset()
    try:
        with open(path, encoding="utf-8") as fh:
            reg = json.load(fh)
    except (OSError, ValueError) as e:
        raise C.V6Error(f"khong doc duoc {path}: {e}") from e
    return frozenset(str(k) for k, v in reg.items() if isinstance(v, Mapping) and str(v.get("role", "")).lower() == "heldout")


def _read_jsonl(path: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _sig(paths: Iterable[str]) -> tuple:
    """Chu ky (duong dan, mtime, co) cua cac tep -> khoa bo nho dem (tep doi -> doc lai)."""
    out = []
    for p in paths:
        try:
            st = os.stat(p)
            out.append((p, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((p, None, None))
    return tuple(out)


HOST_RECHECK_S = 2.0                                  # kiem lai mtime toi da moi 2 s (luat goi host_index cho tung dong)
_HOST_CACHE: dict[str, Any] = {"t": -1e9, "key": None, "val": {}}


def reset_caches() -> None:
    """Xoa bo nho dem (kiem thu / sau khi ghi tep host trong cung tien trinh)."""
    _HOST_CACHE.update({"t": -1e9, "key": None, "val": {}})
    _INDEX_CACHE.clear()
    _ITEMS_CACHE.clear()


def host_index() -> dict[str, dict[str, Any]]:
    """host_file -> {group, source, channel_id, person, lockbox_v2, role, kind} cho ban goc cao (data/scrape/hosts_*) +
    hosts_v2 (nguoi moi). Doc lai khi tep doi (mtime / co, kiem toi da moi HOST_RECHECK_S) -> WP6 ghi lai hosts.jsonl
    khong lam bang cu."""
    now = time.monotonic()
    if _HOST_CACHE["key"] is not None and now - _HOST_CACHE["t"] < HOST_RECHECK_S:
        return _HOST_CACHE["val"]
    paths = scrape_hosts_paths() + tuple(NEW_HOSTS)
    key = _sig(paths)
    if key != _HOST_CACHE["key"]:
        out: dict[str, dict[str, Any]] = {}
        for p in paths:
            if not os.path.exists(p):
                continue
            kind = "new" if p in NEW_HOSTS else "scrape"
            for r in _read_jsonl(p):
                ch = r.get("channel_id")
                g = r.get("group") or (f"scrape:{ch}" if ch and kind == "scrape" else r.get("person") or f"scrape:{r['file']}")
                out[str(r["file"])] = {"group": str(g), "source": str(r.get("source") or ""), "channel_id": ch,
                                       "person": r.get("person"), "lockbox_v2": r.get("lockbox_v2"), "role": r.get("role"),
                                       "kind": kind}
        _HOST_CACHE.update({"key": key, "val": out})
    _HOST_CACHE["t"] = now
    return _HOST_CACHE["val"]


def _scrape_host_groups() -> dict[str, str]:
    """host_file -> nhom cho ban goc cao (I4: data/scrape/hosts_*/hosts.jsonl) -> nhan ra ca dong SUA tu ban goc cao."""
    return {f: v["group"] for f, v in host_index().items() if v["kind"] == "scrape"}


def scrape_key(meta: Mapping[str, Any]) -> str:
    """Khoa ON DINH cho luat cao: '<nguon>:<channel_id>' (meta.channel_id | meta.scrape.channel_id | dong host cao);
    khong co channel_id -> meta.group / nhom host cao (cach cu). Rong = khong xac dinh."""
    sc = meta.get("scrape") if isinstance(meta.get("scrape"), Mapping) else {}
    host = host_index().get(str(meta.get("host_file") or ""), {}) if meta.get("host_file") else {}
    ch = meta.get("channel_id") or sc.get("channel_id") or (host.get("channel_id") if host.get("kind") == "scrape" else None)
    src = str(meta.get("source") or host.get("source") or "").lower()
    if isinstance(ch, str) and ch.strip():
        return f"{src}:{ch.strip()}" if src and src not in ("scrape",) else f"scrape:{ch.strip()}"
    return str(meta.get("group") or (host.get("group") if host.get("kind") == "scrape" else "") or "")


def is_scraped(meta: Mapping[str, Any], set_name: str = "") -> bool:
    """Ban ghi cao? Theo nguon goc (url / channel_id / meta.scrape / scraped), nhom 'scrape:*', nguon cao, tap scrape*,
    hoac host_file la ban goc cao. KHONG dung 'license' (kho cong khai cung ghi giay phep)."""
    g = str(meta.get("group") or "")
    src = str(meta.get("source") or "").lower()
    if meta.get("url") or meta.get("channel_id") or isinstance(meta.get("scrape"), Mapping) or meta.get("scraped") is True:
        return True
    if g.startswith("scrape:") or src in SCRAPE_SOURCES or src.startswith("scrape") or set_name.startswith("scrape"):
        return True
    host = str(meta.get("host_file") or "")
    return bool(host) and host_index().get(host, {}).get("kind") == "scrape"


def _engine_names(d: Mapping[str, Any], keys: Iterable[str]) -> list[str]:
    """Ten engine trong d[keys] + d.gen_chain (list); chuoi lai '<tts>+<vc>' (WP7b hybrid) tach theo '+'."""
    raw = [str(d[k]) for k in keys if isinstance(d.get(k), (str, int)) and str(d[k])]
    gc = d.get("gen_chain")
    if isinstance(gc, (list, tuple)):
        raw += [str(x) for x in gc if isinstance(x, (str, int)) and str(x)]
    elif isinstance(gc, str) and gc:
        raw.append(gc)
    out: list[str] = []
    for s in raw:
        out += [p.strip() for p in s.split("+") if p.strip()]
    return out


def engines_of(meta: Mapping[str, Any]) -> list[str]:
    """Moi ten engine TTS / VC trong meta: khoa cap tren + gen_chain + meta.edits[*] (quy uoc hien co: edits[i].tts =
    'viXTTS'; WP7b: engine '<tts>+<vc>', edits[i].gen_chain [tts, vc]). Chuoi lai tach theo '+'."""
    out = _engine_names(meta, ENGINE_KEYS)
    for ed in meta.get("edits") or ():
        if isinstance(ed, Mapping):
            out += _engine_names(ed, EDIT_ENGINE_KEYS)
    return out


def heldout_role(meta: Mapping[str, Any]) -> bool:
    """WP7b ghi meta.engine_role / edits[i].role|engine_role = 'heldout' -> khoa ke ca khi ten engine chua co trong registry."""
    if str(meta.get("engine_role") or "").lower() == "heldout":
        return True
    for ed in meta.get("edits") or ():
        if isinstance(ed, Mapping) and any(str(ed.get(k) or "").lower() == "heldout" for k in ("engine_role", "role")):
            return True
    return False


LINK_KEYS = ("donor_file", "target_person", "ref_person", "target")


def _has_links(meta: Mapping[str, Any]) -> bool:
    if any(k in meta for k in LINK_KEYS[1:]):
        return True
    return any(isinstance(ed, Mapping) and any(k in ed for k in LINK_KEYS) for ed in meta.get("edits") or ())


def linked_groups(meta: Mapping[str, Any]) -> list[tuple[str, str, Mapping[str, Any] | None]]:
    """[(kind, nhom, dong host)] cua NGUOI KHAC co audio / giong trong tep: donor_file (tra qua hosts tu nhien, host_index,
    ten VIVOS) va giong dich (v6.data.groups.target_groups). Donor khong tra duoc -> bo qua (groups bao cao rieng)."""
    if not _has_links(meta):
        return []
    from v6.data import groups as G
    out: list[tuple[str, str, Mapping[str, Any] | None]] = []
    hi = host_index()
    for ed in meta.get("edits") or ():
        if not isinstance(ed, Mapping):
            continue
        df = ed.get("donor_file")
        if isinstance(df, str) and df:
            if df in hi:
                out.append(("donor", hi[df]["group"], hi[df]))
            else:
                gs, _ = G.donor_groups({"edits": [{"donor_file": df}]})
                out += [("donor", g, None) for g in gs]
    out += [("target", g, None) for g in G.target_groups(meta)]
    return out


def current_groups() -> set[str]:
    """Moi nhom dang co trong manifest v6 + hosts tu nhien (anh chup used_groups)."""
    from v6.data.groups import iter_items
    out = {it["group"] for it in iter_items(include_v4=False)}
    for p in C.HOSTS_JSONL:
        if os.path.exists(p):
            for r in _read_jsonl(p):
                out.add(C.group_of({"host_file": r["file"], "person": r.get("person"), "source": r.get("source"),
                                    "cluster": r.get("cluster")}))
    return out


# ---------------- luat ----------------
class Rules:
    """Luat v2 tren mot anh chup co dinh (used / released / heldout / curated) -> tat dinh, kiem thu duoc khong can dia."""

    def __init__(self, used_groups: Iterable[str], released_groups: Iterable[str] = (), heldout: Iterable[str] = (),
                 curated: Iterable[str] = CURATED_SOURCES):
        self.used = frozenset(used_groups)
        self.released = frozenset(released_groups)
        self.heldout = frozenset(heldout)
        self.curated = frozenset(str(s).lower() for s in curated)

    @classmethod
    def from_state(cls, state: Mapping[str, Any] | None = None) -> "Rules":
        st = load_state() if state is None else state
        if not st.get("used_groups"):
            raise C.V6Error("lockbox_v2 chua dong bang (chay: python -m v6.data.lockbox freeze) -> chua co anh chup used_groups")
        return cls(st["used_groups"], st.get("released_groups", []), heldout_engines(), curated_of(st))

    def _new_person(self, g: str, source: str = "") -> bool:
        src = (source or g.split(":", 1)[0]).lower()
        return bool(g) and src in self.curated and g not in self.used and g not in self.released

    def reason(self, meta: Mapping[str, Any], set_name: str = "") -> str | None:
        """Ly do khoa (None = dung duoc). lockbox_v2=False tren mot dong KHONG nha gi: chi released_groups nha new_person;
        tts_heldout / scrape khong bao gio nha duoc. Nguoi lien ket (donor / giong dich) cung duoc kiem (cap dong)."""
        m = meta["meta"] if isinstance(meta.get("meta"), Mapping) else meta
        if m.get("lockbox_v2") is True:
            return "explicit"
        for e in engines_of(m):
            if e in self.heldout:
                return f"tts_heldout:{e}"
        if heldout_role(m):
            return "tts_heldout:engine_role"
        host = str(m.get("host_file") or "")
        try:
            g = C.group_of(m, set_name)
        except ValueError:
            g = host_index().get(host, {}).get("group", "") if host else ""
        if is_scraped(m, set_name):
            k = scrape_key(m) or g
            if k and zlib.crc32(k.encode("utf-8")) % SCRAPE_MOD == 0:
                return f"scrape:{k}"
        if g and self._new_person(g, str(m.get("source") or "")):
            return f"new_person:{g}"
        return self._linked_reason(m)

    def _linked_reason(self, m: Mapping[str, Any]) -> str | None:
        for kind, lg, hrow in linked_groups(m):
            if hrow is not None and hrow.get("kind") == "scrape":
                k = scrape_key({"channel_id": hrow.get("channel_id"), "source": hrow.get("source"), "group": lg})
                if k and zlib.crc32(k.encode("utf-8")) % SCRAPE_MOD == 0:
                    return f"linked_scrape:{k}"
                continue
            if hrow is not None and hrow.get("lockbox_v2") is True:
                return f"linked_new_person:{lg}"
            if self._new_person(lg):
                return f"linked_new_person:{lg}"
        return None


def curated_of(state: Mapping[str, Any]) -> frozenset[str]:
    """Nguon new_person trong tep (quyet dinh da ghi), mac dinh CURATED_SOURCES."""
    cs = state.get("curated_sources")
    return frozenset(str(s).lower() for s in cs) if cs else CURATED_SOURCES


EN_SOURCES = frozenset({"edacc", "ami", "voxp", "librispeech"})


def en_lockbox_status(curated: Iterable[str]) -> str:
    en = sorted(set(curated) & EN_SOURCES)
    return f"Nguon new_person tieng Anh da chon: {en} (lockbox v2 tieng Anh = nguoi moi cua cac nguon nay + cao EN)" if en \
        else EN_LOCKBOX_STATUS


# ---------------- dong bang ----------------
def _candidates(extra_jsonl: Iterable[str] = ()) -> Iterator[tuple[str, str, dict[str, Any], str]]:
    """(id, wav tuyet doi, meta, tap) cho moi ban ghi co the thuoc lockbox v2: manifest v6 + hosts cao + hosts_v2 + them."""
    for mp in sorted(glob.glob(os.path.join(C.V6DATA, "*", "manifest.json"))):
        s = os.path.basename(os.path.dirname(mp))
        for e in C.read_manifest(mp):
            yield str(e["id"]), C.abs_wav(str(e["wav"])) if e.get("wav") else "", dict(e.get("meta") or {}), s
    for p in scrape_hosts_paths() + tuple(NEW_HOSTS) + tuple(extra_jsonl):
        if not os.path.exists(p):
            continue
        root = os.path.dirname(p)
        for r in _read_jsonl(p):
            meta = {k: v for k, v in r.items() if k not in ("words", "text")}
            meta.setdefault("host_file", r["file"])
            yield str(r["file"]), os.path.join(root, str(r.get("source") or ""), r["file"] + ".wav"), meta, os.path.basename(root)


def rules_doc(curated: Iterable[str] = CURATED_SOURCES) -> dict[str, Any]:
    """Mo ta luat hien hanh (ghi vao lockbox_v2.json de nguoi doc tep biet luat nao da ap)."""
    return {"order": ["explicit", "tts_heldout", "scrape", "new_person", "linked"], "new_person": sorted(curated),
            "scrape_mod": SCRAPE_MOD, "scrape_key": "crc32('<nguon>:<channel_id>') (channel_id | scrape.channel_id | host cao); "
            "khong co channel_id -> nhom",
            "scrape_detect": "url|channel_id|meta.scrape|scraped|group scrape:*|source "
            + "|".join(sorted(SCRAPE_SOURCES)) + "|scrape*|set scrape*|host_file in data/scrape/hosts_*",
            "tts_heldout": "engines.json role=heldout; keys " + ",".join(ENGINE_KEYS) + " + edits[*]." + ",".join(EDIT_ENGINE_KEYS)
            + " + gen_chain; '+' split; engine_role|edits[*].role == heldout",
            "linked": "edits[*].donor_file (host cao / hosts_v2 / tu nhien) + giong dich edits[*].target_person|ref_person, "
            "meta.target.person -> new_person / scrape cua nguoi lien ket (cap dong)",
            "explicit": "meta.lockbox_v2 is True", "released_groups": "expand_hosts plan lockbox_v2=False | hosts_v2 role=dev | --release",
            "row_lockbox_v2_false": "khong nha gi", "string_ids": "tra nguoc id -> meta (chi muc song) roi ap luat"}


def dev_groups(plan_path: str = EXPAND_PLAN, hosts_v2: Iterable[str] = NEW_HOSTS) -> set[str]:
    """Nhom duoc NHA cho dev: nguoi lockbox_v2=False trong ke hoach expand_hosts (--dev-share) + hosts_v2 role=dev.
    Day la nguon DUY NHAT (cung --release ghi ro) cua released_groups."""
    out: set[str] = set()
    if plan_path and os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as fh:
            pl = json.load(fh)
        out |= {str(c.get("group") or c["person"]) for c in pl.get("persons", []) if c.get("lockbox_v2") is False}
    for p in hosts_v2:
        if os.path.exists(p):
            out |= {str(r.get("group") or r["person"]) for r in _read_jsonl(p) if r.get("role") == "dev"}
    return out


def _new_state(used_groups: Iterable[str] | None) -> dict[str, Any]:
    return {"version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "consumed_v1": CONSUMED_V1,
            "rules": rules_doc(), "curated_sources": sorted(CURATED_SOURCES), "en_lockbox": EN_LOCKBOX_STATUS,
            "used_groups": sorted(set(used_groups) if used_groups is not None else current_groups()),
            "released_groups": [], "items": {}}


def freeze(dry_run: bool = False, extra_jsonl: Iterable[str] = (), path: str = LOCKBOX_V2,
           used_groups: Iterable[str] | None = None, plan_path: str = EXPAND_PLAN,
           release: Iterable[str] = ()) -> dict[str, Any]:
    """Lan dau: chup used_groups (moi nhom hien co) roi phan loai. Lan sau: chi THEM muc moi; sha1 muc cu doi -> V6Error.
    released_groups = cu | dev_groups(plan_path) | release (KHONG bao gio lay tu dong manifest co lockbox_v2=False)."""
    st = load_state(path)
    first = not st
    if first:
        st = _new_state(used_groups)
    cur = curated_of(st)
    if st.get("rules") != rules_doc(cur):          # luat trong ma doi -> ghi lai lich su, anh chup used_groups giu nguyen
        st["rules_history"] = list(st.get("rules_history", [])) + [{"until": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                                    "rules": st.get("rules")}]
        st["rules"] = rules_doc(cur)
        st["consumed_v1"] = CONSUMED_V1
    st.setdefault("curated_sources", sorted(cur))
    st["en_lockbox"] = en_lockbox_status(cur)
    items: dict[str, Any] = dict(st.get("items", {}))
    released = set(st.get("released_groups", [])) | dev_groups(plan_path) | {str(g) for g in release}
    cands = list(_candidates(extra_jsonl))
    rules = Rules(st["used_groups"], released, heldout_engines(), cur)
    added, changed, missing = [], [], []
    reasons: Counter = Counter()
    for rid, wav, meta, s in cands:
        why = rules.reason(meta, s)
        if not why:
            continue
        reasons[why.split(":")[0]] += 1
        if not wav or not os.path.exists(wav):
            missing.append(rid)
            continue
        sha = sha1_file(wav)
        old = items.get(rid)
        if old:
            if old["sha1"] != sha:
                changed.append(rid)
            continue
        try:
            grp = C.group_of(meta, s)
        except ValueError:
            grp = why.split(":", 1)[-1]
        items[rid] = {"sha1": sha, "wav": wav.replace("\\", "/"), "reason": why, "group": grp,
                      "host_file": meta.get("host_file"), "set": s}
        added.append(rid)
    if changed:
        raise C.V6Error(f"{len(changed)} muc lockbox_v2 da dong bang co AUDIO KHAC (vd {changed[:3]}) -> khong ghi")
    summary = {"first_freeze": first, "n_candidates": len(cands), "n_used_groups": len(st["used_groups"]),
               "released_groups": len(released), "added": len(added), "total_items": len(items), "by_reason": dict(reasons),
               "curated_sources": sorted(cur), "missing_wav": missing[:20], "n_missing_wav": len(missing), "path": path,
               "dry_run": dry_run}
    if not dry_run:
        st.update({"items": items, "released_groups": sorted(released), "updated": time.strftime("%Y-%m-%d %H:%M:%S")})
        C._atomic_write_json(path, st)
    return summary


def curate(add: Iterable[str] = (), remove: Iterable[str] = (), by: str = "", note: str = "", path: str = LOCKBOX_V2,
           dry_run: bool = False) -> dict[str, Any]:
    """Ghi QUYET DINH doi nguon new_person vao lockbox_v2.json (co lich su). Can 'by' (ai, ngay). Bo nguon = nha nguoi moi
    cua nguon do -> chi khi nguoi dung chu y; nguon phai thuoc kho cong khai cua quyet dinh 2."""
    st = load_state(path)
    if not st.get("used_groups"):
        raise C.V6Error("lockbox_v2 chua dong bang -> chay freeze truoc")
    if not by.strip():
        raise C.V6Error("curate can --by '<ai, ngay>' (quyet dinh nguoi dung)")
    add_s, rem_s = {str(s).lower() for s in add}, {str(s).lower() for s in remove}
    bad = sorted((add_s | rem_s) - KNOWN_SOURCES)
    if bad:
        raise C.V6Error(f"nguon khong thuoc kho cong khai (quyet dinh 2): {bad}")
    old = set(curated_of(st))
    new = (old | add_s) - rem_s
    entry = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "by": by, "note": note, "from": sorted(old), "to": sorted(new)}
    if not dry_run and new != old:
        st["curated_sources"] = sorted(new)
        st["curated_history"] = list(st.get("curated_history", [])) + [entry]
        st["updated"] = entry["at"]
        C._atomic_write_json(path, st)
    return {"changed": new != old, **entry, "dry_run": dry_run,
            "note_after": "chay lai freeze de dong bang sha1 nguoi moi cua nguon vua them"}


# ---------------- chot chan ----------------
_ITEMS_CACHE: dict[tuple, tuple[frozenset[str], frozenset[str], frozenset[str]]] = {}


def _items(path: str = LOCKBOX_V2) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """(id, host_file, sha1) da dong bang — doc lai khi tep doi."""
    if not os.path.exists(path):
        return frozenset(), frozenset(), frozenset()
    key = _sig([path])
    if key not in _ITEMS_CACHE:
        it = load_state(path).get("items", {})
        _ITEMS_CACHE.clear()
        _ITEMS_CACHE[key] = (frozenset(k.lower() for k in it),
                             frozenset(str(v.get("host_file")).lower() for v in it.values() if v.get("host_file")),
                             frozenset(v["sha1"] for v in it.values() if v.get("sha1")))
    return _ITEMS_CACHE[key]


def _rule_meta(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Ban rut gon cua meta du cho luat (chi muc id -> meta nho)."""
    out = {k: meta[k] for k in META_KEYS if k in meta}
    if isinstance(meta.get("scrape"), Mapping):
        out["scrape"] = {"channel_id": meta["scrape"].get("channel_id")}
    eds = [{k: ed[k] for k in EDIT_KEYS if k in ed} for ed in meta.get("edits") or () if isinstance(ed, Mapping)]
    if eds:
        out["edits"] = eds
    return out


_INDEX_CACHE: dict[tuple, dict[str, tuple[dict[str, Any], str]]] = {}


def _index_paths() -> list[str]:
    return (sorted(glob.glob(os.path.join(C.V6DATA, "*", "manifest.json"))) + list(scrape_hosts_paths())
            + list(NEW_HOSTS) + list(C.HOSTS_JSONL))


def id_index() -> dict[str, tuple[dict[str, Any], str]]:
    """id (chu thuong) -> (meta rut gon, tap) cho MOI ban ghi da biet: _candidates (manifest v6 + hosts cao + hosts_v2) +
    hosts tu nhien. Doc lai khi bat ky tep nguon nao doi -> du lieu tao sau lan dong bang van tra duoc."""
    key = (_candidates, _sig(_index_paths()))         # giu chinh ham (khong id()) -> ham thay the trong kiem thu khong trung khoa
    if key not in _INDEX_CACHE:
        idx: dict[str, tuple[dict[str, Any], str]] = {}
        for p in C.HOSTS_JSONL:
            if os.path.exists(p):
                for r in _read_jsonl(p):
                    meta = {k: v for k, v in r.items() if k not in ("words", "text")}
                    meta.setdefault("host_file", r["file"])
                    idx[str(r["file"]).lower()] = (_rule_meta(meta), os.path.basename(os.path.dirname(p)))
        for rid, _wav, meta, s in _candidates():
            idx[rid.lower()] = (_rule_meta(meta), s)        # trung id: dong sau cung thang (giong detect.load_items)
        _INDEX_CACHE.clear()
        _INDEX_CACHE[key] = idx
    return _INDEX_CACHE[key]


class _NoSnapshotRules:
    """Luat khi CHUA co anh chup used_groups (chi voi allow_unfrozen=True): bo new_person, giu explicit / tts_heldout / scrape."""

    def __init__(self, rules: Rules):
        self.rules = rules

    def reason(self, meta: Mapping[str, Any], set_name: str = "") -> str | None:
        why = self.rules.reason(meta, set_name)
        return None if why and (why.startswith("new_person:") or why.startswith("linked_new_person:")) else why


class Guard:
    """Chot chan nap MOT lan (trang thai, muc dong bang, luat, chi muc id) roi kiem nhieu ban ghi -> split_not_lockbox
    khong doc lai tep cho tung dong."""

    def __init__(self, path: str = LOCKBOX_V2, check_rules: bool = True, allow_unfrozen: bool = False,
                 set_name: str = "", allow_unknown_ids: bool = False):
        st = load_state(path)
        if not st.get("used_groups") and not allow_unfrozen:
            raise C.V6Error(f"lockbox_v2 chua dong bang hoac khong thay tep ({path}) -> chot chan khong bao ve duoc gi. "
                            "Chay: python -m v6.data.lockbox freeze (hoac allow_unfrozen=True neu co chu y)")
        self.ids, self.hosts, self.shas = _items(path)
        self.set_name = set_name
        self.allow_unknown = allow_unknown_ids
        self.rules: Rules | _NoSnapshotRules | None = None
        if check_rules:
            r = Rules(st.get("used_groups") or (), st.get("released_groups", []), heldout_engines(), curated_of(st))
            self.rules = r if st.get("used_groups") else _NoSnapshotRules(r)
        self._index: dict[str, tuple[dict[str, Any], str]] | None = None
        self.unknown: list[str] = []

    def _lookup(self, rid: str) -> tuple[dict[str, Any], str] | None:
        if self._index is None:
            self._index = id_index()
        return self._index.get(rid.lower())

    def why(self, x: Any, check_sha1: bool = False) -> tuple[str, str] | None:
        """(ten, ly do) neu x la lockbox v2, None neu dung duoc. Id khong tra duoc -> ghi vao self.unknown."""
        if isinstance(x, str):
            if x.lower() in self.ids or x.lower() in self.hosts:
                return x, "id_dong_bang"
            if self.rules is None:
                return None
            hit = self._lookup(x)
            if hit is None:
                self.unknown.append(x)
                return None
            why = self.rules.reason(hit[0], self.set_name or hit[1])
            return (x, why) if why else None
        rid = str(x.get("id") or "")
        meta = x["meta"] if isinstance(x.get("meta"), Mapping) else x
        host = str(meta.get("host_file") or "").lower()
        if rid.lower() in self.ids or (host and host in self.hosts):
            return rid or host, "id_dong_bang"
        why = self.rules.reason(meta, self.set_name) if self.rules else None
        if why:
            return rid or host, why
        if check_sha1 and x.get("wav"):
            w = C.abs_wav(str(x["wav"]))
            if os.path.exists(w) and sha1_file(w) in self.shas:
                return rid, "sha1_trung"
        return None

    def raise_unknown(self) -> None:
        if self.unknown and not self.allow_unknown:
            raise C.V6Error(f"{len(self.unknown)} id khong tra duoc ra meta (khong co trong manifest v6 / hosts): "
                            f"{self.unknown[:5]} -> truyen ban ghi day du (dict co meta) hoac allow_unknown_ids=True")


def assert_not_lockbox(ids_or_recs: Iterable[Any], purpose: str = "hoc", path: str = LOCKBOX_V2,
                       check_rules: bool = True, check_sha1: bool = False, allow_unfrozen: bool = False,
                       set_name: str = "", allow_unknown_ids: bool = False) -> None:
    """Raise C.LeakageError neu co ban ghi lockbox v2 trong tap dung cho `purpose` (hoc / hieu chinh / chon nguong).
    Nhan id (str: tra nguoc ra meta qua id_index roi ap luat) hoac dong manifest / meta (dict): id hoac host_file trong
    lockbox_v2.json, hoac (check_rules) luat v2 khop. check_sha1: dong co 'wav' -> so sha1 voi muc da dong bang.
    Thieu tep / chua co used_groups -> C.V6Error (KHONG phai LeakageError -> split_not_lockbox khong nuot), tru khi
    allow_unfrozen=True (khi do chi kiem duoc luat khong can anh chup: explicit / tts_heldout / scrape).
    Id khong tra duoc -> C.V6Error, tru khi allow_unknown_ids=True."""
    g = Guard(path, check_rules, allow_unfrozen, set_name, allow_unknown_ids)
    bad = [b for b in (g.why(x, check_sha1) for x in ids_or_recs) if b]
    if bad:
        raise C.LeakageError(f"{len(bad)} ban ghi LOCKBOX v2 trong tap {purpose}: {bad[:5]}")
    g.raise_unknown()


def split_not_lockbox(recs: Iterable[Any], path: str = LOCKBOX_V2, set_name: str = "",
                      allow_unfrozen: bool = False, allow_unknown_ids: bool = False) -> tuple[list, list]:
    """(dung_duoc, lockbox_v2) — giu thu tu; cung tieu chi voi assert_not_lockbox (khong sha1). Nap trang thai MOT lan.
    Chua dong bang / id khong tra duoc -> V6Error."""
    g = Guard(path, True, allow_unfrozen, set_name, allow_unknown_ids)
    keep, lock = [], []
    for r in recs:
        (lock if g.why(r) else keep).append(r)
    g.raise_unknown()
    return keep, lock


def stale_sources(path: str = LOCKBOX_V2) -> list[str]:
    """Tep nguon (manifest / hosts cao / hosts_v2 / engines.json) moi hon lan dong bang cuoi -> nen chay lai freeze
    (chot chan van ap luat qua id_index; chi sha1 cua du lieu moi la chua co)."""
    st = load_state(path)
    t = st.get("updated") or st.get("created")
    if not t:
        return []
    ts = time.mktime(time.strptime(t, "%Y-%m-%d %H:%M:%S"))
    paths = [p for p in _index_paths() + [ENGINES_JSON] if p not in C.HOSTS_JSONL]
    return [p for p in paths if os.path.exists(p) and os.path.getmtime(p) > ts]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Lockbox v2: dong bang + kiem")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze", help="chup used_groups (lan dau) + them muc lockbox v2 moi (sha1)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--jsonl", nargs="*", default=[], help="hosts.jsonl them (cung dinh dang hosts)")
    p.add_argument("--release", nargs="*", default=[], help="nhom NHA cho dev ghi ro (ngoai ke hoach expand_hosts)")
    sub.add_parser("status", help="tom tat lockbox_v2.json")
    p = sub.add_parser("check", help="manifest co dong lockbox v2 khong (thoat 3 neu co)")
    p.add_argument("manifest")
    p = sub.add_parser("curate", help="QUYET DINH nguoi dung: them / bot nguon new_person (ghi lich su)")
    p.add_argument("--add", nargs="*", default=[])
    p.add_argument("--remove", nargs="*", default=[])
    p.add_argument("--by", default="")
    p.add_argument("--note", default="")
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "freeze":
        out: Any = freeze(a.dry_run, a.jsonl, release=a.release)
    elif a.cmd == "status":
        st = load_state()
        out = {"exists": bool(st), "created": st.get("created"), "updated": st.get("updated"),
               "n_used_groups": len(st.get("used_groups", [])), "n_items": len(st.get("items", {})),
               "by_reason": dict(Counter(v["reason"].split(":")[0] for v in st.get("items", {}).values())),
               "curated_sources": sorted(curated_of(st)), "en_lockbox": st.get("en_lockbox", EN_LOCKBOX_STATUS),
               "consumed_v1": st.get("consumed_v1", CONSUMED_V1)["status"],
               "stale_sources": [os.path.relpath(p, C.BASE) for p in stale_sources()]}
    elif a.cmd == "check":
        try:
            assert_not_lockbox(C.read_manifest(a.manifest), purpose=os.path.basename(a.manifest),
                               set_name=os.path.basename(os.path.dirname(os.path.abspath(a.manifest))))
            out = {"ok": True}
        except C.LeakageError as e:
            print(json.dumps({"ok": False, "error": str(e)[:2000]}, ensure_ascii=False))
            sys.exit(3)
    elif a.cmd == "curate":
        out = curate(a.add, a.remove, a.by, a.note, dry_run=a.dry_run)
    else:
        from v6.data import test_groups
        sys.exit(test_groups.run_all("lockbox"))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])


if __name__ == "__main__":
    main()
