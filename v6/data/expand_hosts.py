# -*- coding: utf-8 -*-
"""MO RONG BAN GOC TIENG VIET (WP5, 10/10): them >= 150 nguoi VoxVietnam / Vietnam-Celeb MOI (chua co trong hosts.jsonl).

Hien trang: hosts.jsonl lay 900 ban goc vox tu 17 nguoi + 900 celeb tu 83 nguoi vi build_natural.step_hosts doc parquet
THEO THU TU dong (dong xep theo nguoi) va tieng Viet khong co tran / nguoi. Tren dia: VoxVietnam test 38 parquet,
Vietnam-Celeb test 10 parquet (giay phep CC BY-NC 4.0, chi nghien cuu noi bo).

  scan   chi doc cot nguoi (speaker / label [+ file]) cua moi parquet (khong giai ma audio, nhe) -> dem dong / nguoi,
         so phien (celeb: thu muc video trong cot 'file'), danh dau nguoi da dung
  plan   chon nguoi moi tat dinh (crc32), moi nguoi <= --cap ban goc (mac dinh 8) + giu --refs dong khac lam MAU NHAI
         GIONG (quyet dinh 2: mau lay tu ban ghi khac cua cung nguoi; celeb uu tien phien khac) -> .tmp_v6/cache/
         expand_hosts_plan.json. Nguoi moi mac dinh vao LOCKBOX v2 (meta.lockbox_v2=True); --dev-share F nha mot phan
         (crc32 % 100 < 100F) cho dev voi fold moi C.fold_of_group (ghi lockbox_v2=False; fold da ghi la CUOI CUNG:
         v6.data.groups build --force ghim nhom moi vao dung gia tri do). Thieu nguoi so voi --target -> truong 'deviation'
         (tran tren dia, so nguoi chi dung duoc neu cat cua so dong > 20 s); n_parquets / n_ref_other_parquet_possible
         cho quyet dinh lay mau giong tu parquet khac
  hosts  (VIEC NANG: ASR wav2vec2 tieng Viet, xep hang) cat 4-20 s + moc tu (vaudit.asr.transcribe_words, nhu
         build_natural.step_hosts) -> data/natural_v6/hosts_v2/<nguon>/<file>.wav + hosts.jsonl (cung truong hosts +
         group, fold_v2, lockbox_v2, role, row, parquet). Ghi tiep duoc; KHONG dong vao hosts.jsonl cu (mo hinh v9 giu nguyen).
         --no-asr = thu duong I/O (words rong), --limit-persons / --out cho chay thu nho.

  durs   (xep hang, ~I/O, doc THEO LO 16 dong: parquet chi 1 row group, doc ca nhom = cap phat 4 GB) doc cot audio
         -> thoi luong moi dong -> .tmp_v6/cache/expand_hosts_durs.json; plan doc tep nay (neu co) de bao so nguoi /
         ban goc QUA loc 4-20 s TRUOC viec ASR nang; hosts bo qua dong ngoai 4-20 s khong can giai ma.
         Ke hoach giu MOI dong khong phai mau theo thu tu uu tien crc32; hosts lay theo uu tien (moi vong: dung so dong con
         thieu, tot nhat chua thu; vong 4: con lai) -> ban goc tron tu nhieu vi tri, khong chi dong chi so nho.
         Chay lai sau ngat: dong da bi loai (it tu / ngoai 4-20 s) se duoc thu lai (chi ton thoi gian).

CLI: python -m v6.data.expand_hosts scan | durs [--limit-parquets N] | plan [--dry-run] [--target 150] [--cap 8]
     [--refs 2] [--min-rows 4] [--dev-share 0] [--min-eligible 0] | hosts [--limit-persons N] [--max-per-person K] [--out DIR] [--no-asr]
     | selftest
(bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.expand_hosts ...)"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zlib
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6 import common as C  # noqa: E402

NAT = os.path.join(C.BASE, "data", "natural_v6")
SOURCES = {"vox": ("voxvietnam", "speaker"), "celeb": ("vietnam_celeb", "label")}
HOSTS_VI = os.path.join(NAT, "hosts", "hosts.jsonl")
OUT_DIR = os.path.join(NAT, "hosts_v2")
PLAN = os.path.join(C.TMP, "cache", "expand_hosts_plan.json")
DURS = os.path.join(C.TMP, "cache", "expand_hosts_durs.json")
MIN_DUR, MAX_DUR, MIN_WORDS = 4.0, 20.0, 8          # nhu build_natural.step_hosts


def parquets(src: str) -> list[str]:
    sub, _ = SOURCES[src]
    return sorted(glob.glob(os.path.join(NAT, sub, "data", "test*.parquet")))


def fid_of(src: str, parquet: str, row: int) -> str:
    """Ten ban goc GIONG build_natural._iter_parquet: '<nguon>_<parquet bo .parquet>_<dong:05d>' (chong trung voi hosts cu)."""
    return f"{src}_{os.path.basename(parquet)[:-8]}_{row:05d}"


def session_of(src: str, file_col: Any) -> str | None:
    """Cot file dang 'id00001/<video>/<doan>.wav' -> ma phien (video). Ban HF tren dia chi co 'id00880/00014.wav'
    (khong co video) -> None; vox khong co cot file -> None."""
    if not file_col:
        return None
    parts = str(file_col).replace("\\", "/").split("/")
    return parts[-2] if len(parts) >= 3 else None


def used_persons(path: str = HOSTS_VI) -> Counter:
    c: Counter = Counter()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    c[json.loads(line)["person"]] += 1
    return c


def scan(src: str) -> dict[str, Any]:
    """Doc cot nguoi (+ file) tung parquet -> {person: {rows: [(parquet, dong, phien)], ...}} + thong tin row group."""
    import pyarrow.parquet as pq
    _, col = SOURCES[src]
    persons: dict[str, list] = defaultdict(list)
    files = []
    for p in parquets(src):
        pf = pq.ParquetFile(p)
        names = pf.schema_arrow.names
        cols = [col] + (["file"] if "file" in names else [])
        t = pf.read(columns=cols)
        spk = t.column(col).to_pylist()
        fl = t.column("file").to_pylist() if "file" in cols else [None] * len(spk)
        for i, (s, f) in enumerate(zip(spk, fl)):
            persons[f"{src}:{s}"].append((p, i, session_of(src, f)))
        files.append({"parquet": os.path.basename(p), "rows": len(spk), "row_groups": pf.num_row_groups})
        del t
    return {"persons": dict(persons), "files": files}


def choose(persons: Mapping[str, list], used: Iterable[str], target: int, cap: int, refs: int, min_rows: int,
           dev_share: float = 0.0) -> list[dict[str, Any]]:
    """Chon nguoi moi tat dinh: du >= min_rows dong, thu tu crc32(nguoi); moi nguoi <= cap dong ban goc + refs dong mau
    (celeb: mau uu tien phien KHAC phien cua ban goc). Tra danh sach ke hoach theo nguoi."""
    used = set(used)
    cands = sorted((p for p, rows in persons.items() if p not in used and len(rows) >= min_rows),
                   key=lambda p: (zlib.crc32(p.encode("utf-8")), p))
    out = []
    for p in cands[:target]:
        rows = sorted(persons[p], key=lambda r: (zlib.crc32(f"{p}|{r[0]}|{r[1]}".encode("utf-8")), r[1]))
        sess = Counter(r[2] for r in rows if r[2])
        host_rows, ref_rows = rows, []
        if len(sess) >= 2:                      # ban goc tu cac phien tru phien it nhat; mau tu phien it nhat
            ref_sess = min(sess, key=lambda s: (sess[s], s))
            ref_rows = [r for r in rows if r[2] == ref_sess][:refs]
            host_rows = [r for r in rows if r[2] != ref_sess]
        else:                                   # khong co ma phien: mau = cac dong CUOI theo chi so (xa ban goc nhat ->
            far = sorted(rows, key=lambda r: (r[0], r[1]))[-refs:] if refs else []   # de khac video hon), ban goc = phan con lai
            ref_rows = far
            host_rows = [r for r in rows if r not in far]
        role_dev = (zlib.crc32(f"lockbox_v2:{p}".encode("utf-8")) % 100) < round(100 * dev_share)
        out.append({"person": p, "group": p, "n_rows": len(rows), "n_sessions": len(sess),
                    "n_parquets": len({os.path.basename(a) for a, _, _ in rows}),
                    # MOI dong khong phai mau, theo thu tu uu tien: buoc hosts dung khi du `cap` ban goc qua loc 4-20 s / ASR
                    "host_rows": [[os.path.basename(a), b, c] for a, b, c in host_rows],
                    "cap": cap, "ref_rows": [[os.path.basename(a), b, c] for a, b, c in ref_rows],
                    "ref_same_session_risk": len(sess) < 2, "lockbox_v2": not role_dev,
                    "fold_v2": C.fold_of_group(p)})
    return out


def n_eligible(rows: Iterable, durs: Mapping[str, list[float]]) -> int:
    """So dong 4-20 s theo durs (dong chua biet thoi luong khong tinh)."""
    n = 0
    for pq_path, row, _ in rows:
        d = durs.get(os.path.basename(pq_path))
        n += int(d is not None and int(row) < len(d) and MIN_DUR <= d[int(row)] <= MAX_DUR)
    return n


def plan(target: int = 150, cap: int = 8, refs: int = 2, min_rows: int = 4, dev_share: float = 0.0,
         dry_run: bool = True, out: str = PLAN, durs_path: str = DURS, min_eligible: int = 0) -> dict[str, Any]:
    """min_eligible > 0 (can durs): chi chon nguoi co >= min_eligible dong 4-20 s (tran tren dia bao trong sources.*)."""
    used = used_persons()
    durs = load_durs(durs_path)
    res: dict[str, Any] = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dry_run": dry_run,
                           "params": {"target": target, "cap": cap, "refs": refs, "min_rows": min_rows, "dev_share": dev_share,
                                      "min_eligible": min_eligible},
                           "sources": {}, "persons": []}
    per_src_target = {"vox": int(np.ceil(target * 0.6)), "celeb": target - int(np.ceil(target * 0.6))}
    for src in SOURCES:
        sc = scan(src)
        P = sc["persons"]
        used_src = {p for p in used if p.startswith(src + ":")}
        if durs:                                  # tran tren dia: nguoi moi co >= 1 / >= 4 dong 4-20 s
            ne = {p: n_eligible(v, durs) for p, v in P.items() if p not in used_src}
            res.setdefault("eligible_ceiling", {})[src] = {"new_ge1": sum(n >= 1 for n in ne.values()),
                                                           "new_ge4": sum(n >= 4 for n in ne.values()),
                                                           # chi co dong > 20 s: cat cua so moi dung duoc (lua chon (a))
                                                           "new_long_only": sum(1 for p in ne if not ne[p] and _has_long(P[p], durs))}
            if min_eligible:
                P = {p: v for p, v in P.items() if p in used_src or ne.get(p, 0) >= min_eligible}
        avail = [p for p in P if p not in used_src and len(P[p]) >= min_rows]
        chosen = choose(P, used_src, per_src_target[src], cap, refs, min_rows, dev_share)
        res["sources"][src] = {"parquets": len(sc["files"]), "rows": sum(f["rows"] for f in sc["files"]),
                               "persons_on_disk": len(P), "persons_used_in_hosts": len(used_src),
                               "used_not_found_on_disk": sorted(used_src - set(P))[:10],
                               "new_persons_with_min_rows": len(avail), "chosen": len(chosen),
                               "rows_per_person_q": [int(x) for x in np.quantile([len(v) for v in P.values()], [0, .25, .5, .75, 1])]
                               if P else [], "row_groups": sorted({f["row_groups"] for f in sc["files"]})}
        if len(chosen) < per_src_target[src]:      # nguon nay thieu -> bu bang nguon kia o vong sau
            res["sources"][src]["shortfall"] = per_src_target[src] - len(chosen)
        res["persons"] += [{**c, "source": src} for c in chosen]
    short = sum(v.get("shortfall", 0) for v in res["sources"].values())
    if short:                                    # bu: lay them tu nguon con du
        for src in SOURCES:
            have = {c["person"] for c in res["persons"]}
            P = scan(src)["persons"]
            if durs and min_eligible:
                P = {p: v for p, v in P.items() if n_eligible(v, durs) >= min_eligible}
            extra = choose(P, {p for p in used if p.startswith(src + ":")} | have, short, cap, refs, min_rows, dev_share)
            res["persons"] += [{**c, "source": src} for c in extra]
            short -= len(extra)
            if short <= 0:
                break
    res["n_persons"] = len(res["persons"])
    res["n_lockbox"] = sum(c["lockbox_v2"] for c in res["persons"])
    res["est_host_files_upper"] = sum(min(c["cap"], len(c["host_rows"])) for c in res["persons"])
    res["n_ref_other_parquet_possible"] = sum(1 for c in res["persons"] if c.get("n_parquets", 1) >= 2)
    res.update(eligibility(res["persons"], durs))
    if res["n_persons"] < target:                # LECH so voi dac ta (>= target nguoi moi) -> ghi ro trong ke hoach
        ceil = res.get("eligible_ceiling") or {}
        res["deviation"] = (f"Chi {res['n_persons']} / {target} nguoi moi: tran tren dia (test split, dong 4-20 s) = "
                            f"{sum(v.get('new_ge1', 0) for v in ceil.values())} nguoi; cat cua so dong > 20 s chi them "
                            f"{sum(v.get('new_long_only', 0) for v in ceil.values())}. Dat >= {target} can tai them "
                            "(vd VoxVietnam train, CC BY-NC 4.0, ~100 GB) hoac nguoi dung chap nhan con so nay.")
    C._atomic_write_json(out, res)
    return res


def _has_long(rows: Iterable, durs: Mapping[str, list[float]]) -> bool:
    """Nguoi co dong > MAX_DUR (dung duoc neu cho cat cua so 4-20 s)."""
    for pq_path, row, _ in rows:
        d = durs.get(os.path.basename(pq_path))
        if d is not None and int(row) < len(d) and d[int(row)] > MAX_DUR:
            return True
    return False


# ---------------- thoi luong (doc cot audio, xep hang khi chay du) ----------------
def load_durs(path: str = DURS) -> dict[str, list[float]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("durs", {})


def eligibility(persons: list[Mapping[str, Any]], durs: Mapping[str, list[float]]) -> dict[str, Any]:
    """Uoc luong TRUOC viec ASR: moi nguoi co bao nhieu dong ban goc 4-20 s (theo durs). Thieu durs cua parquet -> 'unknown'.
    ASR (>= 8 tu) con loc them -> day la CAN TREN."""
    if not durs:
        return {"eligibility": "chua co durs (chay: bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.expand_hosts durs) -> chua biet so ban goc qua loc 4-20 s"}
    known = ok = 0
    per = []
    for c in persons:
        n_ok = n_unknown = 0
        for pq_name, row, _ in c["host_rows"]:
            d = durs.get(pq_name)
            if d is None or int(row) >= len(d):
                n_unknown += 1
            elif MIN_DUR <= d[int(row)] <= MAX_DUR:
                n_ok += 1
        if n_unknown == 0:
            known += 1
        ok += int(n_ok > 0)
        per.append(min(c["cap"], n_ok))
    return {"eligibility": {"persons_fully_known": known, "persons_with_ge1_4_20s": ok,
                            "persons_with_ge4_4_20s": sum(1 for x in per if x >= 4),
                            "est_host_files_4_20s": int(sum(per)), "target": len(persons),
                            "note": "can tren truoc loc ASR >= 8 tu"}}


def _audio_lengths(table: Any, ac: str) -> list[float]:
    """Thoi luong (giay) tung dong tu cot audio: struct{array: list<double>, sampling_rate} -> do dai list / sr (pyarrow,
    khong doi sang python); struct{bytes} -> soundfile.info tren bytes."""
    import pyarrow as pa
    import pyarrow.compute as pc
    col = table.column(ac).combine_chunks()
    t = col.type
    if pa.types.is_struct(t) and t.get_field_index("array") >= 0:
        n = pc.list_value_length(col.field("array")).to_numpy(zero_copy_only=False)
        sr = (col.field("sampling_rate").to_numpy(zero_copy_only=False) if t.get_field_index("sampling_rate") >= 0
              else np.full(len(n), C.SR))
        return [round(float(a) / float(b or C.SR), 3) for a, b in zip(n, sr)]
    if pa.types.is_struct(t) and t.get_field_index("bytes") >= 0:
        import soundfile as sf
        out = []
        for b in col.field("bytes").to_pylist():
            try:
                out.append(round(sf.info(io.BytesIO(b)).duration, 3) if b else -1.0)
            except RuntimeError:
                out.append(-1.0)
        return out
    raise C.V6Error(f"khong biet doc thoi luong tu cot {ac} kieu {t}")


def step_durs(out: str = DURS, limit_parquets: int = 0, batch_size: int = 16) -> dict[str, Any]:
    """Doc cot audio THEO LO (iter_batches, pre_buffer=False) -> thoi luong moi dong. Ghi tiep duoc.
    KHONG dung read_row_group: moi parquet chi 1 row group list<double> -> doc ca nhom = cap phat ~4 GB (ArrowMemoryError
    da gap 10/10). Theo lo 16 dong: dinh nho (do bang peak_mem_gb)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    res = {"durs": load_durs(out), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    todo = [p for src in SOURCES for p in parquets(src) if os.path.basename(p) not in res["durs"]]
    if limit_parquets:
        todo = todo[:limit_parquets]
    t0 = time.time()
    for p in todo:
        pf = pq.ParquetFile(p, pre_buffer=False, buffer_size=1 << 23)
        ac = next(n for n in pf.schema_arrow.names if "audio" in n)
        d: list[float] = []
        for batch in pf.iter_batches(batch_size=batch_size, columns=[ac]):
            d += _audio_lengths(pa.Table.from_batches([batch]), ac)
            del batch
        res["durs"][os.path.basename(p)] = d
        C._atomic_write_json(out, res)
    allv = [x for v in res["durs"].values() for x in v]
    return {"out": out, "parquets_done": len(res["durs"]), "this_run": len(todo), "seconds": round(time.time() - t0, 1),
            "peak_mem_gb": _peak_gb(),
            "rows": len(allv), "share_4_20s": round(float(np.mean([MIN_DUR <= x <= MAX_DUR for x in allv])), 3) if allv else None,
            "dur_q": [round(float(x), 2) for x in np.quantile(allv, [0, .1, .5, .9, 1])] if allv else []}


def _peak_gb() -> float | None:
    """Dinh bo nho tien trinh (Windows: peak_wset) de uoc luong viec xep hang."""
    try:
        import psutil
    except ImportError:
        return None
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)


# ---------------- buoc hosts (NANG) ----------------
MAX_PASSES = 4       # vong 1..3: dung so dong con thieu (uu tien cao nhat chua thu); vong 4: moi dong con lai


def _decode(v: Any) -> tuple[np.ndarray, int] | None:
    import soundfile as sf
    if isinstance(v, dict) and v.get("bytes") is not None:
        y, sr = sf.read(io.BytesIO(v["bytes"]), dtype="float32", always_2d=True)
        return y.mean(1), int(sr)
    if isinstance(v, dict) and v.get("array") is not None:
        return np.asarray(v["array"], np.float32), int(v.get("sampling_rate") or C.SR)
    return None


def _rows_by_parquet(persons: list[Mapping[str, Any]]) -> dict[str, dict[int, Mapping[str, Any]]]:
    out: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for c in persons:
        for pq_name, row, _ in c["host_rows"]:
            out[pq_name][int(row)] = c
    return out


def priority_window(c: Mapping[str, Any], k: int, durs: Mapping[str, list[float]], tried: set) -> list[list]:
    """k dong DAU theo thu tu uu tien cua ke hoach (crc32 tron, khong theo chi so dong) ma durs khong loai (chua biet = giu),
    tru dong da thu o vong truoc. Giu da dang trong nguoi: khong lay toan dong chi so nho (de cung video)."""
    out = []
    for r in c["host_rows"]:
        d = durs.get(r[0])
        if (r[0], int(r[1])) in tried or (d is not None and int(r[1]) < len(d) and not MIN_DUR <= d[int(r[1])] <= MAX_DUR):
            continue
        out.append(r)
        if len(out) >= k:
            break
    return out


def plan_passes(persons: list[Mapping[str, Any]], kept: Mapping[str, int], durs: Mapping[str, list[float]], tried: set,
                pass_no: int, max_per_person: int = 0) -> list[dict[str, Any]]:
    """Nguoi con thieu + dong cua vong pass_no: dung (cap - da giu) dong CHUA THU tot nhat theo uu tien -> trong vong khong
    con canh tranh giua dong (thu tu chi so khong lam lech); vong cuoi = moi dong con lai."""
    out = []
    for c in persons:
        cap = min(c["cap"], max_per_person) if max_per_person else c["cap"]
        need = cap - kept.get(c["person"], 0)
        if need <= 0:
            continue
        k = len(c["host_rows"]) if pass_no >= MAX_PASSES else need
        rows = priority_window(c, k, durs, tried)
        if rows:
            out.append({**c, "host_rows": rows, "cap": cap})
    return out


def _process_row(c: Mapping[str, Any], row: Mapping[str, Any], src: str, col: str, ac: str, p: str, i: int, out_dir: str,
                 no_asr: bool, stats: Counter) -> dict[str, Any] | None:
    """Mot dong parquet -> ban ghi host (da ghi wav) hoac None (ly do vao stats)."""
    import librosa
    import soundfile as sf
    if f"{src}:{row[col]}" != c["person"]:
        stats["sai_nguoi"] += 1          # parquet doi thu tu so voi luc scan -> bo
        return None
    dec = _decode(row[ac])
    if dec is None:
        stats["khong_giai_ma"] += 1
        return None
    y, sr = dec
    if sr != C.SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SR)
    dur = len(y) / C.SR
    if not MIN_DUR <= dur <= MAX_DUR:
        stats["ngoai_4_20s"] += 1
        return None
    text, words = "", []
    if not no_asr:
        from vaudit.asr import transcribe_words
        text, words = transcribe_words(y)
        if len(words) < MIN_WORDS:
            stats["it_tu"] += 1
            return None
    fid = fid_of(src, p, i)
    d = os.path.join(out_dir, src)
    os.makedirs(d, exist_ok=True)
    sf.write(os.path.join(d, fid + ".wav"), y, C.SR)
    return {"file": fid, "source": src, "cluster": c["person"], "person": c["person"], "dur": round(dur, 3),
            "text": text, "words": [[w.text, round(w.start, 3), round(w.end, 3)] for w in words],
            "group": c["group"], "fold_v2": c["fold_v2"], "lockbox_v2": c["lockbox_v2"],
            "role": "lockbox_v2" if c["lockbox_v2"] else "dev", "parquet": os.path.basename(p), "row": i,
            "ref_rows": c["ref_rows"], "asr": not no_asr}


def _stream_source(src: str, cs: list[Mapping[str, Any]], out_dir: str, no_asr: bool, done: set, kept: Counter,
                   tried: set, stats: Counter, out: Any) -> None:
    """Duyet parquet cua mot nguon theo lo 8 dong, chi xu ly dong trong cua so cua vong nay."""
    import pyarrow.parquet as pq
    _, col = SOURCES[src]
    want = _rows_by_parquet(cs)
    for p in parquets(src):
        rows = want.get(os.path.basename(p))
        if not rows:
            continue
        pf = pq.ParquetFile(p, pre_buffer=False, buffer_size=1 << 23)
        ac = next(n for n in pf.schema_arrow.names if "audio" in n)
        starts = np.cumsum([0] + [pf.metadata.row_group(g).num_rows for g in range(pf.num_row_groups)])
        for g in sorted({int(np.searchsorted(starts, r, side="right") - 1) for r in rows}):
            i = int(starts[g]) - 1
            for batch in pf.iter_batches(batch_size=8, row_groups=[g], columns=[ac, col]):
                for row in batch.to_pylist():
                    i += 1
                    c = rows.get(i)
                    if c is None:
                        continue
                    fid = fid_of(src, p, i)
                    if fid in done or kept[c["person"]] >= c["cap"]:
                        continue
                    tried.add((os.path.basename(p), i))
                    rec = _process_row(c, row, src, col, ac, p, i, out_dir, no_asr, stats)
                    if rec is None:
                        continue
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                    done.add(fid)
                    kept[c["person"]] += 1
                    stats["ghi"] += 1
                del batch


def step_hosts(plan_path: str = PLAN, out_dir: str = OUT_DIR, limit_persons: int = 0, max_per_person: int = 0,
               no_asr: bool = False, durs_path: str = DURS) -> dict[str, Any]:
    """Cat ban goc cho nguoi trong ke hoach THEO UU TIEN: moi vong chi xet (cap - da giu) dong chua thu tot nhat (thu tu
    crc32 cua ke hoach, bo dong durs biet ngoai 4-20 s); vong 1..3 nhu vay, vong 4 moi dong con lai; chi nguoi con thieu
    (bi loc ASR / thoi luong) moi doc lai parquet. Giai ma theo lo 8 dong (pyarrow), ghi tiep duoc."""
    with open(plan_path, encoding="utf-8") as fh:
        pl = json.load(fh)
    persons = pl["persons"][:limit_persons] if limit_persons else pl["persons"]
    durs = load_durs(durs_path)
    os.makedirs(out_dir, exist_ok=True)
    idx = os.path.join(out_dir, "hosts.jsonl")
    done: set = set()
    kept: Counter = Counter()
    tried: set = set()
    if os.path.exists(idx):
        with open(idx, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add(r["file"])
                    kept[r["person"]] += 1
                    tried.add((r.get("parquet"), r.get("row")))
    stats: Counter = Counter()
    with open(idx, "a", encoding="utf-8") as out:
        for pass_no in range(1, MAX_PASSES + 1):
            todo = plan_passes(persons, kept, durs, tried, pass_no, max_per_person)
            stats[f"vong{pass_no}_nguoi"] = len(todo)
            by_src: dict[str, list] = defaultdict(list)
            for c in todo:
                by_src[c["source"]].append(c)
            for src, cs in by_src.items():
                _stream_source(src, cs, out_dir, no_asr, done, kept, tried, stats, out)
    with_hosts = sum(1 for c in persons if kept[c["person"]])
    return {"out": idx, "persons": len(persons), "persons_with_hosts": with_hosts, "target_persons": len(persons),
            "shortfall_persons": len(persons) - with_hosts, "persons_ge4_hosts": sum(1 for c in persons if kept[c["person"]] >= 4),
            "host_files_total": int(sum(kept[c["person"]] for c in persons)), "peak_mem_gb": _peak_gb(),
            **dict(stats)}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Them nguoi VoxVietnam / Vietnam-Celeb moi lam ban goc (lockbox v2 / fold moi)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="dem nguoi tren dia vs nguoi da dung (chi doc cot nguoi)")
    p = sub.add_parser("durs", help="XEP HANG (I/O): thoi luong moi dong parquet -> cache/expand_hosts_durs.json")
    p.add_argument("--limit-parquets", type=int, default=0)
    p.add_argument("--out", default=DURS)
    p = sub.add_parser("plan", help="chon nguoi moi -> cache/expand_hosts_plan.json")
    p.add_argument("--dry-run", action="store_true", help="danh dau ke hoach la ban thu (van ghi tep ke hoach, khong cat audio)")
    p.add_argument("--target", type=int, default=150)
    p.add_argument("--cap", type=int, default=8)
    p.add_argument("--refs", type=int, default=2)
    p.add_argument("--min-rows", type=int, default=4)
    p.add_argument("--dev-share", type=float, default=0.0)
    p.add_argument("--min-eligible", type=int, default=0, help="chi nguoi co >= K dong 4-20 s theo durs")
    p = sub.add_parser("hosts", help="VIEC NANG: cat + ASR ban goc theo ke hoach")
    p.add_argument("--plan", default=PLAN)
    p.add_argument("--out", default=OUT_DIR)
    p.add_argument("--limit-persons", type=int, default=0)
    p.add_argument("--max-per-person", type=int, default=0)
    p.add_argument("--no-asr", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "scan":
        used = used_persons()
        out: Any = {}
        for src in SOURCES:
            P = scan(src)["persons"]
            u = {p for p in used if p.startswith(src + ":")}
            out[src] = {"persons_on_disk": len(P), "rows": sum(len(v) for v in P.values()), "used": len(u),
                        "new": len(set(P) - u), "used_hosts": sum(used[p] for p in u)}
    elif a.cmd == "plan":
        r = plan(a.target, a.cap, a.refs, a.min_rows, a.dev_share, a.dry_run, min_eligible=a.min_eligible)
        out = {k: r.get(k) for k in ("dry_run", "params", "sources", "n_persons", "n_lockbox", "est_host_files_upper",
                                     "eligible_ceiling", "eligibility", "n_ref_other_parquet_possible", "deviation")}
        out["first_persons"] = [(c["person"], c["n_rows"], c["n_sessions"], c["fold_v2"]) for c in r["persons"][:25]]
    elif a.cmd == "durs":
        out = step_durs(a.out, a.limit_parquets)
    elif a.cmd == "hosts":
        out = step_hosts(a.plan, a.out, a.limit_persons, a.max_per_person, a.no_asr)
    else:
        from v6.data import test_groups
        sys.exit(test_groups.run_all("expand"))
    print(json.dumps(out, ensure_ascii=False, indent=1)[:8000])


if __name__ == "__main__":
    main()
