# -*- coding: utf-8 -*-
"""BO TEST THAT tu host cao: 'scrapevi' / 'scrapeen' - ban goc 01 + sua bang GIONG THAT (dung lai v6.natural.build_natural).

Thao tac (ham make_one cua build_natural, import - khong chep code):
  nguyen_ban                          -> 01 (bao nham tren ban ghi that ngoai doi: muc tieu chinh cua bo nay)
  nen_toan_cuc / nen_cuc_bo           khu on N1..N5 + N7 WebRTC NS                                 -> 02
                                      NGOAI TRU N7 CA TEP: quyet dinh 4 (NS mac dinh cua ung dung = 01) -> manifest giu
                                      level2 goc nhung gan meta.app_ns = True; nguoi doc PHAI qua
                                      v6.fusion.common_fx.apply_app_ns (mac dinh '01') - dem '02' tho se tinh sai phan N7
  giong_toan_cuc / giong_cuc_bo       doi cao do V1..V6                                             -> 02
  ghep_thay / ghep_chen               doan GIONG THAT cung kenh (cung video uu tien 60 %; video khac CHI tu kenh pool
                                      'lecture' - mot nguoi noi chinh - de 'cung nguoi' co nghia)       -> 03
  xoa_loi                             cat 1-12 tu roi noi                                           -> 03
KHONG co thao tac TTS / doi giong AI: cam nhai giong nguoi trong am thanh cao (quyet dinh 2).
Fold: meta.person = meta.speaker = nhom (kenh, giao dien I2 neu co) -> ca person_fold (crc32(person)) lan meta.fold deu
  giu nguyen kenh trong MOT fold; meta.recording = video. fold = v6.common.fold_of_group(nhom) neu co, khong thi crc32 % 5.
Ra: data_labels/v6/<set>/{<kind>/<id>.wav, manifest.json}, id '<v6scrapevi|v6scrapeen>_<kind>_NNNN'.
NANG vua (doc ~2 000 host, cong cu khu on / doi giong CPU): chay tron bo = hang doi; thu nho bang --scale 0.01 --out-root.

CLI: python -m v6.scrape.build_sets --lang vi [--scale 1] [--seed 0] [--kinds nguyen_ban,xoa_loi] [--out-root DIR]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

PLAN = {"nguyen_ban": 600, "nen_toan_cuc": 150, "nen_cuc_bo": 150, "giong_toan_cuc": 150, "giong_cuc_bo": 150,
        "ghep_thay": 200, "ghep_chen": 150, "xoa_loi": 200}
PER_CLUSTER = 6            # tran tep goc moi video moi loai
V6 = os.path.join(U.BASE, "data_labels", "v6")
V4 = os.path.join(U.BASE, "data_labels", "realistic_v4")


def fold_for(group: str) -> int:
    try:
        from v6.common import fold_of_group   # giao dien I2 (khong bat buoc)
        return int(fold_of_group(group))
    except Exception:  # noqa: BLE001
        return zlib.crc32(group.encode("utf-8")) % 5


def load_scrape_hosts(lang: str):
    """Host I4 -> NHost cua build_natural (speaker = cum = video, person = nhom)."""
    from v6.natural import build_natural as BN
    from vaudit.asr import Word
    hdir = U.hosts_dir(lang)
    out, info = [], {}
    for r in U.read_jsonl(os.path.join(hdir, "hosts.jsonl")):
        words = [Word(t, a, b) for t, a, b in r["words"]]
        h = BN.NHost(r["file"], r["cluster"], r["person"], r["source"], r["text"], [w.text for w in words],
                     os.path.join(hdir, r["source"], r["file"] + ".wav"), words, {})
        out.append(h)
        info[r["file"]] = r
    return out, info


def _meta_fix(meta: dict, host_row: dict, lang: str) -> dict:
    grp = host_row["group"]
    m = {**meta, "lang": lang, "recording": meta.get("speaker"), "speaker": grp, "person": grp, "group": grp,
         "kenh": "goc", "scrape": {k: host_row.get(k) for k in ("url", "channel_id", "license", "codec", "bitrate",
                                                               "orig_file", "video_id", "start", "end", "kenh",
                                                               "pool", "conf_call", "flags")}}
    m["fold"] = fold_for(grp)
    if any(is_app_ns(e) for e in m.get("edits") or []):
        m["app_ns"] = True        # quyet dinh 4: nhan hieu luc 01 sau apply_app_ns (xem docstring)
    return m


def is_app_ns(edit: dict) -> bool:
    """Khu on CA TEP bang NS mac dinh cua ung dung (ho N7 WebRTC) - dung lai v6.fusion.common_fx neu co."""
    try:
        from v6.fusion.common_fx import is_app_ns_edit   # WP8a (giao dien I5) - khong bat buoc
        return bool(is_app_ns_edit(edit))
    except Exception:  # noqa: BLE001
        return str(edit.get("level2", "")) == "02_nen_toan_cuc" and (
            str(edit.get("family")) == "N7" or "webrtc" in f"{edit.get('tool', '')} {edit.get('family', '')}".lower())


def entry_files(e: dict) -> set[str]:
    """Moi host ma muc manifest dung: host goc + doan cho (donor_file) cua cac thao tac ghep."""
    return {e["meta"]["host_file"]} | {str(x["donor_file"]) for x in e["meta"].get("edits") or [] if x.get("donor_file")}


def split_manifest(man: list[dict], valid: set[str]) -> tuple[list[dict], list[dict]]:
    """(giu, bo): bo muc dung host KHONG con trong hosts hien tai (vd kenh bi cach ly / cong loai sau khi da dung bo).
    Ham thuan - 17/10: ban cu chi cong them, giu ca muc cua kenh da go."""
    keep, drop = [], []
    for e in man:
        (keep if entry_files(e) <= valid else drop).append(e)
    return keep, drop


def next_index(entries: list[dict], prefix: str, kind: str) -> int:
    """So thu tu ke tiep cho id '<prefix><kind>_NNNN' (lon nhat + 1, KHONG phai so luong: tranh ghi de tep khi da bo muc)."""
    tag = f"{prefix}{kind}_"
    nums = [int(e["id"][len(tag):]) for e in entries if e["id"].startswith(tag) and e["id"][len(tag):].isdigit()]
    return max(nums) + 1 if nums else 0


def prune(man: list[dict], valid: set[str], outd: str) -> list[dict]:
    """Bo muc khong hop le: chuyen wav sang <outd>/_pruned/<kind>/ (khong xoa) + ghi <outd>/pruned.jsonl."""
    import shutil
    keep, drop = split_manifest(man, valid)
    if drop:
        log = os.path.join(outd, "pruned.jsonl")
        with open(log, "a", encoding="utf-8") as fh:
            for e in drop:
                src = os.path.normpath(os.path.join(V4, e["wav"]))
                if os.path.exists(src):
                    dst = os.path.join(outd, "_pruned", e["kind"], os.path.basename(src))
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.move(src, dst)
                fh.write(json.dumps({"id": e["id"], "kind": e["kind"], "host_file": e["meta"]["host_file"],
                                     "missing": sorted(entry_files(e) - valid), "at": U.now_iso()}, ensure_ascii=False) + "\n")
        print(f"  bo {len(drop)} muc dung host khong con hop le -> {log}", flush=True)
    return keep


def build(lang: str, scale: float, seed: int, kinds: list[str], out_root: str) -> list[dict]:
    import soundfile as sf
    from v6.natural import build_natural as BN
    set_name = f"scrape{lang}"
    prefix = f"v6{set_name}_"
    outd = os.path.join(out_root, set_name)
    man_path = os.path.join(outd, "manifest.json")
    man = json.load(open(man_path, encoding="utf-8")) if os.path.exists(man_path) else []
    hosts, info = load_scrape_hosts(lang)
    man = prune(man, set(info), outd)
    by_cluster, by_person = defaultdict(list), defaultdict(list)
    for h in hosts:
        by_cluster[h.speaker].append(h)
        if str(info[h.file].get("pool") or "").endswith("lecture"):
            by_person[h.person].append(h)
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    for kind in kinds:
        n_goal = max(1, int(round(PLAN[kind] * scale)))
        have = [e for e in man if e["kind"] == kind]
        used = {e["meta"]["host_file"] for e in have}
        per_c = defaultdict(int)
        for e in have:
            per_c[e["meta"].get("recording")] += 1
        k = len(have)
        idx = next_index(have, prefix, kind)
        pool = [h for h in hosts if h.file not in used]
        rng.shuffle(pool)
        for h in pool:
            if k >= n_goal:
                break
            if per_c[h.speaker] >= PER_CLUSTER or len(h.words) < BN.MIN_WORDS:
                continue
            try:
                r = BN.make_one(kind, h, by_person, by_cluster, rng, nrng)
            except Exception as ex:  # noqa: BLE001  (mot host hong khong dung ca buoc)
                print(f"    bo {h.file}: {type(ex).__name__} {str(ex)[:80]}", flush=True)
                continue
            if r is None:
                continue
            wav, meta = r
            vid = f"{prefix}{kind}_{idx:04d}"
            d = os.path.join(outd, kind)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, vid + ".wav")
            wav = np.asarray(wav, np.float32)
            peak = float(np.abs(wav).max()) if len(wav) else 0.0
            if peak > 0.999:
                wav = wav / peak * 0.999
            sf.write(p, wav, BN.SR)
            man.append({"id": vid, "kind": kind, "cfg": "goc", "src": vid,
                        "wav": os.path.relpath(p, V4).replace("\\", "/"), "meta": _meta_fix(meta, info[h.file], lang)})
            per_c[h.speaker] += 1
            k += 1
            idx += 1
        print(f"  {kind:16s} {k}/{n_goal}", flush=True)
        os.makedirs(outd, exist_ok=True)
        with open(man_path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=0)
        os.replace(man_path + ".tmp", man_path)
    return man


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"], required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kinds", default=",".join(PLAN))
    ap.add_argument("--out-root", default=V6)
    a = ap.parse_args()
    kinds = [k for k in a.kinds.split(",") if k]
    bad = [k for k in kinds if k not in PLAN]
    if bad:
        raise SystemExit(f"loai khong ho tro (khong co TTS cho nguoi cao): {bad}")
    man = build(a.lang, a.scale, a.seed, kinds, a.out_root)
    print(f"{len(man)} tep trong manifest scrape{a.lang} ({a.out_root})")


if __name__ == "__main__":
    main()
