# -*- coding: utf-8 -*-
"""WP7b / B4 - DOI GIONG BANG AI (VC) tren loi THAT -> 02 · va chuoi TTS -> VC -> 03 (giong may lai ve giong host).

Nhan theo hop dong (audit1_tts muc 0): VC ap len LOI THAT = 02 ('doi giong' ke ca khi giong moi la cua nguoi khac);
TTS roi VC = 03 (noi dung do may sinh, VC chi doi am sac) -> tang 'hybrid' nam trong bo ttsmulti_<lang>, gen_chain [tts, vc].
  vc_toan     ca tep host (nguoi A) doi sang giong nguoi B (khac nguoi, cung ngon ngu, uu tien cung gioi neu biet)
              -> 02_giong_toan_cuc, edits [{op doi_giong_ai, kind global, engine, role, target_person ...}]
  vc_cuc_bo   mot doan 1-5 s cat o ranh gioi tu (+0,25 s dem hai ben) doi giong roi tron lai (blend 20-30 ms, do dai giu
              nguyen) -> 02_giong_cuc_bo, kind segment, start/end = doan da doi
  muc         vung: dau ra may VC nguyen (khong khop muc, cat cung 0 ms) · trung_binh: khop muc am, tron 20 ms ·
              ky: khop LTAS + loc ve bang thong host + am luong +-1 dB, tron 30 ms (+ tai ma hoa ca tep)
Nguoi dich (target) CHI tu bo nghien cuu duoc phep (vox, VIVOS | edacc, ami, voxp, LibriSpeech; celeb chi khi
  --include-sources celeb, mac dinh bo - DECISIONS muc 2 cam 'nguoi noi tieng'); KHONG vss/vsasv/
  tep cao. Chon nguoi dich (target_rank, 12/10): 1 CUNG phan LOCKBOX voi host, 2 CUNG person_fold, 3 cung kho (chong
  'loi tat theo kho', X-10: host la chinh cac tep that cua natvi/naten), 4 cung gioi, 5 nguoi dich it dung nhat; ghi
  target.same_lockbox / same_fold / same_source / same_gender. Nguoi dich vao meta.speakers (+ target_person,
  target_group, target_person_fold) -> C.lockbox_reason / record_speakers thay ca giong dich.
  May VC co ref_seconds lon (kNN-VC ~300 s) -> noi nhieu tep cua nguoi dich thanh mau giong dai.
Chia: vong tron (plan.round_robin) qua ngon ngu x may x muc x thao tac, fold it muc truoc; o rong -> PlanError.
Dinh: doan doi giong ha muc CHI doan (<= splice.SEG_PEAK) truoc blend (to hop loi) -> ngoai doan TRUNG TUNG MAU voi
  host; vc_toan / da ma hoa lai ca tep: chuan hoa ca tep neu can (edits.chuan_hoa_ca_tep_db). Gain khop muc chan 0,1-10.
Job VC (I3): {job_id, src_wav, target_ref_wav, seed} -> <cand>/<engine>/<job_id>_c<k>.wav (+ .json).
CLI: python -m v6.tts_multi.vc_sets plan [--per-cell 20] [--langs vi,en] [--root R] [--registry F] [--host-dirs ..]
     [--person-cap-frac F] [--allow-empty-cells] [--force-replan]
     python -m v6.tts_multi.vc_sets qa [--limit N] | splice [--keep-failed]
     python -m v6.tts_multi.vc_sets hybrid --tts-set ttsmulti_vi --vc seedvc [--n 40]   (them muc 03 vao bo tts)
  Tang lai dung LAI tep host cua muc goc (host_reuse_of): cung nguoi / fold / nhom nen khong ro CV, nhung mot host co the
  xuat hien hai lan trong bo ttsmulti (muc goc + muc lai) - ngoai le co chu dich cua quy tac 'moi host mot lan'."""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6 import common as C  # noqa: E402
from v6.tts_multi import plan as P  # noqa: E402
from v6.tts_multi import qa as Q  # noqa: E402
from v6.tts_multi import splice as S  # noqa: E402

SR = 16000
SET = "vcmulti"
VC_OPS = ("vc_toan", "vc_cuc_bo")
VC_TIERS = ("vung", "trung_binh", "ky")
OP_CODE = {"vc_toan": "vt", "vc_cuc_bo": "vb"}
SEG_PAD = 0.25
SEG_DUR = (1.0, 5.0)
MAX_SRC_S = 28.0           # Seed-VC: nguon <= 30 s (engines.json notes)


# ---------------- ke hoach ----------------
def gender_of(h: dict, ls_gender: dict) -> str:
    g = (h.get("info") or {}).get("gender") or ls_gender.get(h["person"], "")
    return str(g).lower()


@functools.lru_cache(maxsize=None)
def lock_key(person: str, lang: str, cluster: str | None = None, host_file: str | None = None) -> bool:
    """Nguoi / ban goc co thuoc LOCKBOX khong (C.lockbox_reason voi cung truong ma vc_entry ghi: speaker = cum)."""
    meta = {"speaker": cluster or person, "lang": lang, **({"host_file": host_file} if host_file else {})}
    return C.lockbox_reason({"id": "x", "meta": meta}) is not None


def target_rank(host: dict, p: str, pool_by_person: dict[str, list[dict]], ls_gender: dict, lang: str,
                uses: Counter | None = None) -> tuple:
    """Khoa xep nguoi dich (nho = tot), uu tien lan luot:
      1 CUNG PHAN LOCKBOX voi host (vc_entry ghi nguoi dich vao meta.speakers -> C.lockbox_reason doc ca hai: muc mo
        khong duoc mang giong nguoi lockbox va nguoc lai)
      2 CUNG person_fold voi host (giong dich chi xuat hien trong mot fold CV)
      3 cung kho (chong loi tat theo kho, X-10) · 4 cung gioi (neu biet) · 5 nguoi dich it dung nhat."""
    h0 = pool_by_person[p][0]
    g = gender_of(host, ls_gender)
    return (int(lock_key(p, lang) != lock_key(host["person"], lang, host["cluster"], host["file"])),
            int(P.person_fold(p) != P.person_fold(host["person"])), int(h0["source"] != host["source"]),
            int(not (g and gender_of(h0, ls_gender) == g)), (uses or Counter())[p])


def pick_target(host: dict, pool_by_person: dict[str, list[dict]], ls_gender: dict, rng: random.Random,
                lang: str = "", uses: Counter | None = None) -> str | None:
    """Nguoi dich khac nguoi host, cung ngon ngu, theo target_rank (hoa -> ngau nhien)."""
    persons = sorted(p for p in pool_by_person if p != host["person"])
    if not persons:
        return None
    lang = lang or host.get("lang", "")
    ranks = {p: target_rank(host, p, pool_by_person, ls_gender, lang, uses) for p in persons}
    best = min(ranks.values())
    return rng.choice([p for p in persons if ranks[p] == best])


def target_ref(person: str, files: list[dict], seconds: float, ref_dir: str, rng: random.Random) -> dict | None:
    """Mau giong nguoi dich: <= 10 s -> mot doan (plan.ref_window); lon hon -> noi nhieu tep toi 'seconds' giay."""
    files = sorted(files, key=lambda h: h["file"])
    if seconds <= P.REF_DUR[1]:
        for h in rng.sample(files, len(files)):
            w = P.ref_window(h)
            if w:
                return {**P.write_ref(h, w, ref_dir), "person": person}
        return None
    rid = f"long_{P.slug(person.replace(':', '_'))}_{int(seconds)}"
    path = os.path.join(ref_dir, rid + ".wav")
    used, tot = [], 0.0                      # danh sach tep dung luon tinh lai (ke ca khi tep noi da co) -> loai khoi QA
    for h in files:
        used.append(h)
        tot += float(h.get("dur") or sf.info(h["wav"]).duration)
        if tot >= seconds:
            break
    if not used:
        return None
    if not os.path.exists(path):
        os.makedirs(ref_dir, exist_ok=True)
        sf.write(path, np.concatenate([Q.load16(h["wav"]) for h in used])[:int(seconds * SR)], SR)
    return {"ref_id": rid, "wav": path, "text": "", "text_src": "none", "src_file": ",".join(h["file"] for h in used),
            "src_cluster": person, "dur": round(min(tot, seconds), 2), "person": person}


def pick_segment(h: dict, rng: random.Random) -> dict | None:
    """Doan 1-5 s cat o ranh gioi tu, cach mep >= 0,5 s + dem."""
    ws, dur = h.get("words"), float(h["dur"])
    if not ws or len(ws) < 4:
        return None
    for _ in range(30):
        k = rng.randint(1, len(ws) - 2)
        m = k
        while m + 1 < len(ws) - 1 and ws[m + 1][2] - ws[k][1] <= rng.uniform(*SEG_DUR):
            m += 1
        a, b = P._gap_mid(ws, k, dur, True), P._gap_mid(ws, m, dur, False)
        if SEG_DUR[0] <= b - a <= SEG_DUR[1] and a - SEG_PAD >= P.EDGE_S and b + SEG_PAD <= dur - P.EDGE_S:
            return {"k": k, "m": m, "a": round(a, 4), "b": round(b, 4), "words": " ".join(w[0] for w in ws[k:m + 1])}
    return None


def make_vc_plan(per_cell: int = 20, langs: tuple = ("vi", "en"), seed: int = 2026, root: str | None = None,
                 registry: str | None = None, engines: list[str] | None = None, hosts_by_lang: dict | None = None,
                 extra_by_lang: dict | None = None, ops: tuple = VC_OPS, tiers: tuple = VC_TIERS,
                 exclude_sources: tuple = P.DEFAULT_EXCLUDE, person_cap_frac: float = P.PERSON_CAP_FRAC,
                 allow_empty_cells: bool = False, force: bool = False, host_dirs: tuple | None = None) -> dict:
    """Ke hoach bo vcmulti: ghi plan.jsonl + jobs/<engine>.jsonl (giao dien VC cua I3) + plan_summary.json.
    Chia VONG TRON (P.round_robin) qua o ngon ngu x may x muc x thao tac, fold it muc nhat truoc, nguoi it dung truoc;
    host chi danh dau da dung SAU khi tao xong muc. O rong -> P.PlanError (tru allow_empty_cells).
    exclude_sources: bo han nguon ca vai host lan vai nguoi dich (mac dinh P.DEFAULT_EXCLUDE = ('celeb',): cho quyet
    dinh 'nguoi noi tieng', xem plan.make_plan). host_dirs: kho host them cho tieng Viet (vd ('hosts', 'hosts_v2'))."""
    rng = random.Random(seed)
    D = P.set_dirs(SET, root)
    stage = os.path.join(D["work"], "vcsrc_new")         # doan nguon ghi tam -> doi thanh vcsrc sau khi luu tru ban cu
    if os.path.isdir(stage):
        shutil.move(stage, os.path.join(D["work"], "_old", time.strftime("%m%d_%H%M%S") + "_vcsrc_new"))
    reg, reg_src = P.load_registry(registry)
    ls_gender = P.librispeech_gender()
    items, jobs, ctx, cells_all, supply, caps = [], defaultdict(list), {"counter": 0}, [], {}, {}
    os.makedirs(D["refs"], exist_ok=True)
    for lang in langs:
        engs = engines or P.engines_for(reg, "vc", lang)
        hosts = (hosts_by_lang or {}).get(lang) or P.load_hosts(lang, dirs=host_dirs if lang == "vi" else None)
        extra = (extra_by_lang or {}).get(lang)
        extra = extra if extra is not None else (P.load_vivos() if lang == "vi" else P.load_librispeech())
        hosts = [h for h in hosts if h["source"] not in exclude_sources]
        extra = [h for h in extra if h["source"] not in exclude_sources]
        pool: dict[str, list[dict]] = defaultdict(list)
        for h in hosts + extra:
            if not P._allowed(h["source"], h["person"]):
                raise ValueError(f"nguon cam: {h['file']}")
            pool[h["person"]].append(h)
        src = [h for h in hosts if h.get("words") and float(h["dur"]) <= MAX_SRC_S]
        pools = P._fold_pools(src)
        supply[lang] = {"hosts_by_fold": {f: len(v) for f, v in sorted(pools.items())},
                        "persons": len({h["person"] for h in src}), "target_persons": len(pool)}
        cells = [(lang, e, t, o) for e in engs for t in tiers for o in ops]
        cells_all += cells
        quota = {c: per_cell for c in cells}
        caps[lang] = cap = max(2, math.ceil(person_cap_frac * sum(quota.values())))
        A, uses = P._Alloc(cap, seed), Counter()

        def fill(cell, want, rnd, A=A, uses=uses, pools=pools, pool=pool, lang=lang):
            _, eng, tier, op = cell
            for f in A.folds(cell, eng, rnd):
                for h in A.free(pools.get(f, []), lambda x: (x["file"], op) not in A.bad):
                    it = _vc_item(h, op, tier, eng, reg.get(eng, {}), lang, pool, ls_gender, rng, D, stage, ctx, seed,
                                  uses)
                    if it is None:
                        A.bad.add((h["file"], op))
                        continue
                    A.take(h, cell, eng, f)
                    jobs[eng].append(it.pop("_job"))
                    items.append(it)
                    return it
            return None
        P.round_robin(cells, quota, fill)
    summ = vc_summary(items, cells_all, reg_src, caps, per_cell, exclude_sources, supply)
    if summ["empty_cells"] and not allow_empty_cells:
        raise P.PlanError(f"vcmulti: {len(summ['empty_cells'])} o rong (vd {summ['empty_cells'][:3]}); nguon {supply}",
                          summ)
    summ["archived_to"] = P.archive_work(D, force, extra_dirs=[os.path.join(D["work"], "vcsrc")])
    if os.path.isdir(stage):
        shutil.move(stage, os.path.join(D["work"], "vcsrc"))
    os.makedirs(D["jobs"], exist_ok=True)
    P.write_plan(D["plan"], items)
    for eng, js in jobs.items():
        with open(os.path.join(D["jobs"], f"{eng}.jsonl"), "w", encoding="utf-8") as fh:
            for j in js:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    with open(D["summary"], "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    return summ


def vc_summary(items: list[dict], cells: list, reg_src: str, caps: dict, per_cell: int, exclude_sources: tuple,
               supply: dict) -> dict:
    cnt = Counter(f"{i['lang']}|{i['engine']}|{i['tier']}|{i['op']}" for i in items)
    key = ["|".join(c) for c in cells]
    per_p = Counter(i["host"]["person"] for i in items)
    t = [i["target"] for i in items]
    return {"n_items": len(items), "n_target": per_cell * len(cells), "registry": reg_src, "person_cap": caps,
            "cells": dict(sorted(cnt.items())), "empty_cells": [k for k in key if not cnt.get(k)],
            "cell_min": min((cnt.get(k, 0) for k in key), default=0),
            "cell_max": max((cnt.get(k, 0) for k in key), default=0), "underfilled": len(items) < per_cell * len(cells),
            "folds": dict(sorted(Counter(f"{i['lang']}|{i['person_fold']}" for i in items).items())),
            "fold_x_engine": dict(sorted(Counter(f"{i['lang']}|{i['engine']}|{i['person_fold']}" for i in items).items())),
            "same_gender": sum(1 for x in t if x.get("same_gender")),
            "same_source": sum(1 for x in t if x.get("same_source")),
            "same_fold": sum(1 for x in t if x.get("same_fold")),
            "same_lockbox": sum(1 for x in t if x.get("same_lockbox")),
            "lockbox_items": sum(1 for i in items if i.get("lockbox_any")),
            "roles": dict(Counter(i["role"] for i in items)), "exclude_sources": list(exclude_sources),
            "sources": dict(Counter(f"{i['lang']}|{i['host']['source']}" for i in items)),
            "target_sources": dict(Counter(f"{i['lang']}|{x['source']}" for i, x in zip(items, t))),
            "persons": len(per_p), "max_per_person": max(per_p.values(), default=0),
            "target_persons": len({x["person"] for x in t}), "supply": supply}


def _target_group(person: str, h0: dict) -> str:
    try:
        return C.group_of({"person": person, "source": h0["source"], "speaker": h0["cluster"]}, SET)
    except Exception:                                          # noqa: BLE001 — I2 tuy chon
        return person


def _vc_item(h, op, tier, eng, ent, lang, pool, ls_gender, rng, D, stage, ctx, seed, uses) -> dict | None:
    """Mot muc VC (nguoi dich + mau giong + doan nguon) hoac None. Doan nguon ghi vao stage, duong dan cuoi = vcsrc/."""
    tp = pick_target(h, pool, ls_gender, rng, lang, uses)
    if tp is None:
        return None
    ref_s = float(ent.get("ref_seconds") or (300 if ent.get("family") == "knnvc" else P.REF_DUR[1]))
    ref = target_ref(tp, pool[tp], ref_s, D["refs"], rng)
    if ref is None:
        return None
    seg = pick_segment(h, rng) if op == "vc_cuc_bo" else None
    if op == "vc_cuc_bo" and seg is None:
        return None
    iid = f"{OP_CODE[op]}{P.TIER_CODE[tier]}_{P.slug(eng)}_{ctx['counter']:05d}"
    ctx["counter"] += 1
    uses[tp] += 1
    jid = f"{iid}_p0_r0"
    src_wav = h["wav"]
    if seg is not None or sf.info(h["wav"]).samplerate != SR:
        y = Q.load16(h["wav"])
        if seg is not None:
            y = y[int((seg["a"] - SEG_PAD) * SR):int((seg["b"] + SEG_PAD) * SR)]
        os.makedirs(stage, exist_ok=True)
        sf.write(os.path.join(stage, jid + ".wav"), y, SR)
        src_wav = os.path.join(D["work"], "vcsrc", jid + ".wav")
    t0 = pool[tp][0]
    g_h, g_t = gender_of(h, ls_gender), gender_of(t0, ls_gender)
    rank = target_rank(h, tp, pool, ls_gender, lang)
    host = {k: h.get(k) for k in ("file", "source", "cluster", "person", "dur", "text", "words", "wav", "ref", "info",
                                  "extra")}
    host.update({k: h[k] for k in ("pool",) + P.HOST_KEEP if k in h})
    sd = zlib.crc32(f"{seed}|{jid}".encode()) % 100000
    t_lock = lock_key(tp, lang)
    return {"item_id": iid, "set": SET, "prefix": P.SETS[SET][1], "lang": lang, "op": op, "tier": tier, "engine": eng,
            "role": ent.get("role", "train"), "host": host, "seg": seg, "src_wav": src_wav, "seg_pad": SEG_PAD,
            "target": {"person": tp, "source": t0["source"], "ref": ref, "gender": g_t,
                       "same_gender": bool(g_h and g_h == g_t), "same_source": t0["source"] == h["source"],
                       "same_lockbox": rank[0] == 0, "same_fold": rank[1] == 0, "lockbox": t_lock,
                       "person_fold": P.person_fold(tp), "group": _target_group(tp, t0)},
            "lockbox_any": t_lock or lock_key(h["person"], lang, h["cluster"], h["file"]),
            "parts": [{"part": 0, "text": h.get("text", ""), "jobs": [jid]}], "refs": [ref],
            "ref_stratum": "nguoi_khac", "person_fold": P.person_fold(h["person"]), "fold": C.assign_fold([h["cluster"]]),
            "seed": sd, "_job": {"job_id": jid, "src_wav": src_wav, "target_ref_wav": ref["wav"], "seed": sd}}


# ---------------- cham ----------------
def run_vc_qa(root: str | None = None, scorers: Q.Scorers | None = None, gate_mode: str = "strict",
              by_person: dict | None = None, src_text_oracle: bool = False, engine: str | None = None,
              limit: int | None = None) -> dict:
    """Cham moi ung vien VC; src_text_oracle (chi kiem thu) -> loi tham chieu = loi host thay vi ASR doan nguon.
    engine: chi muc cua may nay; limit: cham thu (buoc qacal) toi da limit muc, chia deu (Q.stratified) theo ngon ngu x
    may x thao tac x muc. Muc da cham: khong cham lai am thanh, XET CONG LAI tu qa/rows/ (Q.vc_gates hien tai)."""
    D = P.set_dirs(SET, root)
    items = [it for it in P.read_plan(D["plan"]) if not engine or it["engine"] == engine]
    if limit:
        items = Q.stratified(items, limit, lambda it: (it["lang"], it["engine"], it["op"], it["tier"]))
    Sc = scorers or Q.Scorers()
    sel_path = os.path.join(D["qa"], "selection.json")
    sel = json.load(open(sel_path, encoding="utf-8")) if os.path.exists(sel_path) else {}
    regated, engines = 0, set()
    for lang in sorted({it["lang"] for it in items}):
        its = [it for it in items if it["lang"] == lang]
        idx = by_person or _vc_index(its, lang)
        refs = Q.PersonRefs(idx, Sc, lang, os.path.join(D["qa"], f"person_refs_{lang}.json"))
        for it in its:
            engines.add(it["engine"])
            old = sel.get(it["item_id"])
            if old and old["parts"].get("0"):
                rows = Q.load_rows(D, it)
                if rows is not None:
                    new = vc_select(it, rows["0"], gate_mode)
                    regated += int(Q.sel_sig(new) != Q.sel_sig(old) or old.get("gates_tag") != Q.GATES_TAG)
                    sel[it["item_id"]] = new
                continue
            rows = _vc_score(it, D, Sc, refs, lang, src_text_oracle)
            Q.save_rows(D, it, {"0": rows})
            sel[it["item_id"]] = vc_select(it, rows, gate_mode)
            refs.save()
            Q._atomic_json(sel_path, sel)
    Q._atomic_json(sel_path, sel)
    for e in engines:
        Q.rebuild_log(D, e)
    summ = Q.qa_summary(sel)
    summ.update(gates_tag=Q.GATES_TAG, regated=regated, scored_now=len(items))
    Q._atomic_json(os.path.join(D["qa"], "summary.json"), summ)
    return summ


def _vc_score(it: dict, D: dict, Sc: Q.Scorers, refs: Q.PersonRefs, lang: str, src_text_oracle: bool) -> list[dict]:
    h = it["host"]
    rows = []
    for jid in it["parts"][0]["jobs"]:
        for w in Q.candidates(D["cand"], it["engine"], jid):
            txt = (it["seg"]["words"] if it["seg"] else h.get("text", "")) if src_text_oracle else None
            tref = {f for f in str(it["target"]["ref"].get("src_file", "")).split(",") if f}
            r = Q.score_vc_candidate(w, it["src_wav"], lang, Sc, refs, it["target"]["person"], h["person"],
                                     {h["file"]}, txt, target_ref_files=tref)
            r.update(job_id=jid, k=int(os.path.splitext(w)[0].rsplit("_c", 1)[1]), item_id=it["item_id"])
            rows.append(r)
    for r, c in zip(rows, Q.vc_composite(rows) if rows else []):
        r["composite"] = c
    return rows


def vc_select(it: dict, rows: list[dict], gate_mode: str) -> dict:
    for r in rows:
        r["gates"] = Q.vc_gates(r)
    best = Q.select(rows, gate_mode)
    return {"item_id": it["item_id"], "engine": it["engine"], "role": it["role"], "pass": bool(best and best["pass"]),
            "n_cand": len(rows), "parts": {"0": best}, "gates_tag": Q.GATES_TAG}


def _vc_index(items: list[dict], lang: str) -> dict[str, list[dict]]:
    persons = {it["host"]["person"] for it in items} | {it["target"]["person"] for it in items}
    pool = P.load_hosts(lang) + (P.load_vivos() if lang == "vi" else P.load_librispeech())
    by: dict[str, list[dict]] = defaultdict(list)
    for h in pool:
        if h["person"] in persons:
            by[h["person"]].append(h)
    return by


# ---------------- ghep ----------------
def fit_len(z: np.ndarray, n: int) -> np.ndarray:
    return np.pad(z, (0, max(0, n - len(z))))[:n]


def _dress_vc(z: np.ndarray, ref: np.ndarray, tier: str, rng: random.Random) -> tuple[np.ndarray, list[str], dict]:
    if tier == "vung":
        return z, ["khong_khop"], {"gain_db": 0.0}
    if tier == "ky":
        z, eq = S.ltas_match(z, ref)
        info = {"eq_db_mean": eq}
        tricks = ["khop_ltas"]
        bw = S.host_bandwidth(ref)
        if bw < S.BW_FULL:
            z = S.lowpass(z, bw)
            tricks.append(f"loc_bang_thong_{bw / 1000:.1f}k")
            info["bw_hz"] = round(bw)
        off = float(np.clip(rng.gauss(0, 0.5), -1, 1))
        g = float(np.clip(S.voiced_rms(ref) / (S.voiced_rms(z) + 1e-9), 0.1, 10.0))   # chan nhu splice.dress
        z = z * g * 10 ** (off / 20)
        return z, tricks + ["am_luong_1dB"], {**info, "loud_off_db": round(off, 2),
                                              "gain_db": round(float(20 * np.log10(g)), 2)}
    g = float(np.clip(S.voiced_rms(ref) / (S.voiced_rms(z) + 1e-9), 0.1, 10.0))
    return z * g, ["khop_am_luong"], {"gain_db": round(float(20 * np.log10(g)), 2)}


def build_vc(it: dict, sel: dict, rng: random.Random, reencode_p: float) -> tuple[np.ndarray, dict, list]:
    from vaudit.edits import blend_process
    y = Q.load16(it["host"]["wav"]).astype(float)
    z = Q.load16(sel["parts"]["0"]["wav"]).astype(float)
    base = {"level2": "02_giong_toan_cuc" if it["op"] == "vc_toan" else "02_giong_cuc_bo", "op": "doi_giong_ai",
            "engine": it["engine"], "role": it["role"], "tier": it["tier"], "target_person": it["target"]["person"],
            "target_source": it["target"]["source"], "target_ref_id": it["target"]["ref"]["ref_id"],
            "same_gender": it["target"]["same_gender"], "seed": it["seed"], "family": f"VC:{it['engine']}"}
    if it["op"] == "vc_toan":
        z, tricks, info = _dress_vc(fit_len(z, len(y)), y, it["tier"], rng)
        edit = {**base, "kind": "global", "start": 0.0, "end": round(len(y) / SR, 4), **info}
        out = z
    else:
        a, b, pad = it["seg"]["a"], it["seg"]["b"], it["seg_pad"]
        n = int((b - a + 2 * pad) * SR)
        z = fit_len(z, n)[int(pad * SR):int(pad * SR) + int(b * SR) - int(a * SR)]
        z, tricks, info = _dress_vc(z, y[int(a * SR):int(b * SR)], it["tier"], rng)
        z, lim_db = S.limit_peak(z)               # dinh doan <= SEG_PEAK; blend_process la to hop loi -> ngoai doan giu nguyen
        if lim_db:
            info["ha_dinh_doan_db"] = lim_db
        fade = {"ky": 30, "trung_binh": 20}.get(it["tier"], 0)
        out = blend_process(y, a, b, z, SR, fade_ms=fade)
        edit = {**base, "kind": "segment", "start": round(int(a * SR) / SR, 4), "end": round(int(b * SR) / SR, 4),
                "words": it["seg"]["words"], "blend_ms": fade, **info}
        tricks.append(f"tron_{fade}ms" if fade else "cat_cung")
    if it["tier"] == "ky" and rng.random() < reencode_p:
        out, how = S.reencode(out, rng)
        tricks.append(how)
    return out, edit, tricks


def vc_entry(it: dict, wav: np.ndarray, edit: dict, tricks: list, sel: dict, out_dir: str) -> dict:
    kind = ("vc_toan_cuc" if it["op"] == "vc_toan" else "vc_cuc_bo") + {"ky": "_ky", "vung": "_vung"}.get(it["tier"], "")
    rid = f"{it['prefix']}_{it['item_id']}"
    d = os.path.join(out_dir, kind)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, rid + ".wav")
    whole = edit.get("kind") != "segment" or any(str(t).startswith("tay_") for t in tricks)
    w, pk = S.finalize_peak(wav, S.edit_region(edit), whole)    # doan: ngoai vung sua khong bao gio doi
    S.write_pcm16(path, w)
    best = sel["parts"]["0"]
    keep = ("job_id", "k", "cer_vs_src", "cos_target", "z_target", "target_basis", "cos_source", "dnsmos_sig", "dnsmos_bak",
            "dnsmos_ovr", "len_ratio", "composite", "pass", "n_cand", "n_pass", "fail", "asr_mode", "gate_mode")
    edit = {**edit, **pk, "qa": {"pass": sel["pass"], "parts": {"0": {k: best.get(k) for k in keep}}}}
    h, tg = it["host"], it["target"]
    meta = {"host_file": h["file"], "speaker": h["cluster"], "person": h["person"], "source": h["source"],
            # giong DICH cung la nguoi trong tep -> C.record_speakers / lockbox_reason doc ca hai (khong ro lockbox)
            "speakers": [tg["person"]], "target_person": tg["person"], "target_group": tg.get("group"),
            "target_person_fold": tg.get("person_fold"), "target_same_fold": tg.get("same_fold"),
            "orig_text": h.get("text"), "expected_text": h.get("text"), "level1": ["02_ai_hau_ky"],
            "level2": [edit["level2"]], "edits": [edit], "lang": it["lang"], "kenh": "goc", "fold": int(it["fold"]),
            "person_fold": int(it["person_fold"]), "muc_che_giau": it["tier"], "thu_thuat": tricks, "so_thao_tac": 1,
            "engine": it["engine"], "engine_role": it["role"], "wp": "WP7b", "qa_pass": bool(sel.get("pass")),
            **({"host_info": h["info"]} if h.get("info") else {}), **S.host_extra_meta(h)}
    try:
        meta["group"] = C.group_of(meta, SET)
    except Exception:                                          # noqa: BLE001 — I2 tuy chon
        pass
    S.mark_lockbox(rid, meta, SET)
    return {"id": rid, "kind": kind, "cfg": "goc", "src": rid, "wav": C.rel_v4(path), "meta": meta}


def run_vc_splice(root: str | None = None, keep_failed: bool = False, reencode_p: float = 0.5, seed: int = 2026) -> dict:
    D = P.set_dirs(SET, root)
    items = P.read_plan(D["plan"])
    sel_all = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
    man = S.load_manifest(D["manifest"])
    have = {m["id"] for m in man}
    new, skip = 0, defaultdict(int)
    for it in items:
        rid = f"{it['prefix']}_{it['item_id']}"
        sel = sel_all.get(it["item_id"])
        if rid in have:
            skip["da_co"] += 1
        elif not sel or not sel["parts"].get("0"):
            skip["chua_qa"] += 1
        elif not sel["pass"] and not keep_failed:
            skip["khong_qua_cong"] += 1
        else:
            try:
                out, edit, tricks = build_vc(it, sel, random.Random(f"{seed}|{it['item_id']}"), reencode_p)
            except Exception as ex:                            # noqa: BLE001 — mot muc hong khong dung ca buoc
                print(f"  bo {it['item_id']}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
                skip["loi"] += 1
                continue
            man.append(vc_entry(it, out, edit, tricks, sel, D["set"]))
            new += 1
    S.save_manifest(D["manifest"], man)
    return {"set": SET, "new": new, "skipped": dict(skip), "n_manifest": len(man)}


# ---------------- tang lai TTS -> VC (03) ----------------
def base_ok(sel_item: dict, needs_vc: bool = False) -> bool:
    """Muc tts co dung lam goc cho tang lai? Binh thuong = pass. May giong co san (needs_vc): bo qua cong giong nguoi
    g2_spk (VC se doi am sac sau; selection cu truoc ban sua 11/10 van con g2_spk) — cac cong con lai phai qua."""
    if not needs_vc:
        return bool(sel_item.get("pass"))
    parts = list(sel_item.get("parts", {}).values())
    return bool(parts) and all(p is not None and all(v for k, v in p.get("gates", {}).items() if k != "g2_spk")
                               for p in parts)


def plan_hybrid(tts_set: str, vc: str, n: int = 40, root: str | None = None, registry: str | None = None,
                seed: int = 2026, keep_failed: bool = False, tts_engines: list[str] | None = None) -> dict:
    """Chon n muc da co ung vien tot cua bo tts, doi am sac ung vien sang giong CHINH host (mau giong ban ghi khac)
    bang may VC -> muc moi '<item>_h<vc>' (engine '<tts>+<vc>', gen_chain [tts, vc]) them vao plan.jsonl cua bo tts.
    Job VC ghi o jobs/hybrid_<tts>+<vc>.jsonl; ung vien vao cand/<tts>+<vc>/ (build_sets gen --hybrid)."""
    D = P.set_dirs(tts_set, root)
    items = P.read_plan(D["plan"])
    sel = json.load(open(os.path.join(D["qa"], "selection.json"), encoding="utf-8"))
    reg, _ = P.load_registry(registry)
    have = {it["item_id"] for it in items}
    rng = random.Random(f"{seed}|{vc}")
    base = [it for it in items if not it.get("gen_chain") and it["item_id"] in sel
            and (not tts_engines or it["engine"] in tts_engines)
            and all(v is not None for v in sel[it["item_id"]]["parts"].values())
            and (keep_failed or base_ok(sel[it["item_id"]], needs_vc=bool(reg.get(it["engine"], {}).get("needs_vc"))
                                        or not it.get("clone", True)))]
    rng.shuffle(base)
    new, jobs = [], defaultdict(list)
    for it in base:
        if len(new) >= n:
            break
        nid = f"{it['item_id']}_h{P.slug(vc)}"
        if nid in have:
            continue
        eng = f"{it['engine']}+{vc}"
        parts = []
        for p in it["parts"]:
            best = sel[it["item_id"]]["parts"][str(p["part"])]
            jid = f"{nid}_p{p['part']}_r0"
            jobs[eng].append({"job_id": jid, "src_wav": best["wav"], "target_ref_wav": it["refs"][0]["wav"],
                              "seed": zlib.crc32(f"{seed}|{jid}".encode()) % 100000})
            parts.append({"part": p["part"], "text": p["text"], "jobs": [jid]})
        roles = {it["role"], reg.get(vc, {}).get("role", "train")}
        new.append({**it, "item_id": nid, "engine": eng, "role": "heldout" if "heldout" in roles else "train",
                    "parts": parts, "gen_chain": [it["engine"], vc], "hybrid_of": it["item_id"],
                    "clone": True,                  # VC doi am sac ve giong host -> qa cham lai cong giong nguoi (G2)
                    "host_reuse_of": it["item_id"]})  # cung tep host voi muc goc (cung nguoi/fold -> khong ro CV)
    P.write_plan(D["plan"], items + new)
    for eng, js in jobs.items():
        with open(os.path.join(D["jobs"], f"hybrid_{eng}.jsonl"), "w", encoding="utf-8") as fh:
            for j in js:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    return {"tts_set": tts_set, "vc": vc, "new_items": len(new), "jobs_files": sorted(f"hybrid_{e}.jsonl" for e in jobs)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Doi giong AI (02) + TTS->VC (03) (WP7b)")
    ap.add_argument("step", choices=["plan", "qa", "splice", "hybrid"])
    ap.add_argument("--engine", default=None, help="qa: chi muc cua may nay")
    ap.add_argument("--limit", type=int, default=None, help="qa: cham thu toi da N muc")
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--langs", default="vi,en")
    ap.add_argument("--root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--keep-failed", action="store_true")
    ap.add_argument("--reencode-p", type=float, default=0.5)
    ap.add_argument("--tts-set", default="ttsmulti_vi")
    ap.add_argument("--vc", default="seedvc")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--from-engines", default=None, help="hybrid: chi lay muc cua cac may tts nay (vd mms_vie)")
    P.add_plan_args(ap)
    a = ap.parse_args()
    if a.step == "plan":
        try:
            out = make_vc_plan(a.per_cell, tuple(a.langs.split(",")), root=a.root, registry=a.registry,
                               **P.plan_kwargs(a))
        except P.PlanError as ex:
            print(json.dumps({"loi": str(ex), "summary": ex.summary}, ensure_ascii=False, indent=1, default=str))
            sys.exit(2)
    elif a.step == "qa":
        out = run_vc_qa(a.root, engine=a.engine, limit=a.limit)
    elif a.step == "splice":
        out = run_vc_splice(a.root, a.keep_failed, a.reencode_p)
    else:
        out = plan_hybrid(a.tts_set, a.vc, a.n, a.root, a.registry, keep_failed=a.keep_failed,
                          tts_engines=a.from_engines.split(",") if a.from_engines else None)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
