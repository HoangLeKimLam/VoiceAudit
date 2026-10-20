# -*- coding: utf-8 -*-
"""HA TANG DUNG CHUNG v6 — moi script v6 import tu day (from v6 import common as C).

  duong dan   BASE · V4 · SUITES · V6DATA · TMP (.tmp_v6) · RESULTS
  moi truong  setup_env(): HF_HOME/TORCH_HOME/TEMP/TMP/MPLCONFIGDIR/PIP_CACHE_DIR... ve D: (goi san khi import)
  RAM         ram_free_gb() · require_ram(gb) · heavy_job() (khoa: mot viec nang mot luc tren may)
  fold        speaker_folds() · assign_fold(speakers) · validate_fold(meta) · folds_to_score(fold)
  id moi      check_new_ids(ids) · existing_ids() — cache dung chung (ff/deepfense/ssl/pretrained, v5_segfeat_cache)
              khoa theo id: trung id = dung lai dac trung CU mot cach im lang -> cam
  dac trung   extract_features(manifest, feat_path) boc run_codec_variants.features + sidecar <feat>.ids.json
              (id · sha256 tung wav · fold · sha256 pkl) · load_feats(name) · feat_path_for(name)
  thong ke    wilson · clopper_pearson · mcnemar (chinh xac) · bootstrap_ci_by_group (bootstrap theo cum)
  LOCKBOX     is_lockbox(rec, set_name) · lockbox_reason · split_lockbox · assert_no_lockbox
  AMI         ami_series(text) · check_disjoint_groups(calib, test) (chia theo SERIES)
  NHOM (I2)   group_of(meta, set_name) · fold_of_group(group[, strict]) (bang folds_v2_groups.json, nhom moi = crc32("fold:"+nhom) % 5,
              gia tri phat ra la cuoi cung: build --force ghim nhom moi vao dung gia tri do)
  ket qua     write_result(name, dict) · read_result(name) -> .tmp_v6/results/<name>.json

CLI: python -m v6.common ram | folds | ids <manifest> [--feat F] | extract <manifest> <feat.pkl> [--threads 4]
     [--min-ram 3] [--dry-run] | lockbox <ten features>"""
from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

# thong ke nam o v6/stats.py, xuat lai tai day (C.wilson, C.mcnemar...)
from v6.stats import bootstrap_ci_by_group, clopper_pearson, mcnemar, wilson  # noqa: E402,F401

V4 = os.path.join(BASE, "data_labels", "realistic_v4")
SUITES = os.path.join(BASE, "data_labels", "suites")
V6DATA = os.path.join(BASE, "data_labels", "v6")
TMP = os.path.join(BASE, ".tmp_v6")
RESULTS = os.path.join(TMP, "results")
INDEX = os.path.join(TMP, "id_index.json")          # cache danh sach id cua tung pkl (khoa: kich thuoc + mtime)
LOCK = os.path.join(TMP, "heavy.lock")
SR = 16000
N_FOLDS = 5
DEFAULT_THREADS = 4
SHARED_CACHES = ("ff_cache", "deepfense_cache", "ssl_cache", "pretrained_cache")   # thu muc <id>.pkl/.npz
ID_TUPLE_PKLS = ("v5_segfeat_cache.pkl",)                                            # pkl khoa (id, a, b)
FEAT_CACHES = (("ff_cache", ".pkl"), ("deepfense_cache", ".npz"), ("ssl_cache", ".npz"))  # features() ghi 3 cai nay
# 8 cau hinh nen da co (run_codec_variants.CONFIGS); bitrate/cau hinh khac = LOCKBOX
CODEC_CONFIGS = frozenset({"mp3_32k", "mp3_128k", "aac_32k", "aac_24k", "opus_16k_voip", "opus_32k_48k",
                           "amrwb_12k65", "tandem_aac32_opus16"})
AMI_SERIES = ("ES2004", "IS1009", "TS3003")        # 3 nhom: cung nguoi + cung phong trong mot series
AMI_LOCKBOX_SERIES = "TS3003"

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+\-]*$")
V6_PREFIX_RE = re.compile(r"^(v6[a-z0-9]+)_")
_V6_STRIP_RE = re.compile(r"^v6[a-z0-9]*_", re.I)
# ten bo ma o dau mot "tu" cua cfg (mp3_64k_44k, m4a_aac_96k, tandem_aac32_opus16, amrwb_12k65...)
_CODEC_RE = re.compile(r"(?:^|[_\-.])(?:mp3|aac|m4a|opus|vorbis|ogg|amr|gsm|g7\d\d|speex|ilbc|wma|ac3|silk|evs|lc3|"
                       r"encodec|lyra)", re.I)
_AMI_RE = re.compile(r"(?<![A-Za-z0-9])((?:ES|IS|TS|EN|IB|IN)\d{4})(?:[a-e])?(?![0-9])")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class V6Error(RuntimeError):
    """Loi chung cua ha tang v6."""


class IdCollisionError(V6Error):
    """Id moi trung id da dung (cache dung chung / features_*.pkl / sidecar) hoac sai quy tac dat ten."""


class LeakageError(V6Error):
    """Vi pham quy tac chong ro ri (fold nguoi noi, series AMI, LOCKBOX)."""


class InsufficientRamError(V6Error):
    """RAM trong khong du cho viec nang."""


# ---------------- moi truong ----------------
ENV_DIRS = {
    "HF_HOME": os.path.join(BASE, ".hf_cache"),                        # cung thu muc vaudit / run_codec_variants dung
    "TORCH_HOME": os.path.join(BASE, ".tmp_research", "torch_home"),   # da co model torch hub tai san
    "TEMP": os.path.join(TMP, "tmp"),
    "TMP": os.path.join(TMP, "tmp"),
    "TMPDIR": os.path.join(TMP, "tmp"),
    "MPLCONFIGDIR": os.path.join(TMP, "mpl"),
    "PIP_CACHE_DIR": os.path.join(TMP, "pip_cache"),
    "NUMBA_CACHE_DIR": os.path.join(TMP, "numba"),                     # librosa -> numba
    "JOBLIB_TEMP_FOLDER": os.path.join(TMP, "joblib"),                 # sklearn n_jobs memmap
    "XDG_CACHE_HOME": os.path.join(TMP, "xdg_cache"),
}


def setup_env(threads: int | None = None) -> dict[str, str]:
    """Tro moi thu muc cache/tam ve D: (lam lai bao nhieu lan cung duoc). threads -> THREADS/OMP/MKL/OPENBLAS."""
    for k, v in ENV_DIRS.items():
        os.makedirs(v, exist_ok=True)
        os.environ[k] = v
    tempfile.tempdir = None               # tempfile doc lai TEMP moi
    if threads:
        for k in ("THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[k] = str(int(threads))
    return dict(ENV_DIRS)


setup_env()


# ---------------- RAM ----------------
_PS = ["powershell", "-NoProfile", "-NonInteractive", "-Command"]
_RAM_CMDS = ((_PS + ["(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory).AvailableMBytes"], 2 ** 20),
             (_PS + ["(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"], 2 ** 10),
             (["wmic", "OS", "get", "FreePhysicalMemory", "/Value"], 2 ** 10))


def ram_free_gb() -> float:
    """RAM con dung duoc (GiB): psutil neu co, neu khong thi PowerShell / wmic / /proc/meminfo."""
    try:
        import psutil
        return psutil.virtual_memory().available / 2 ** 30
    except Exception:                     # khong co / hong psutil -> do cach khac
        return _ram_free_fallback()


def _ram_free_fallback() -> float:
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 2 ** 10 / 2 ** 30
    for cmd, unit in _RAM_CMDS:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        m = re.search(r"(\d+)", out or "")
        if m:
            return int(m.group(1)) * unit / 2 ** 30
    raise V6Error("khong do duoc RAM trong (khong co psutil, PowerShell, wmic)")


def require_ram(gb: float, what: str = "") -> float:
    """Raise InsufficientRamError neu RAM trong < gb. Tra ve so GB dang trong."""
    free = ram_free_gb()
    if free < gb:
        raise InsufficientRamError(f"can >= {gb:.1f} GB RAM trong{' cho ' + what if what else ''}, "
                                   f"hien co {free:.2f} GB")
    return free


def _pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        if os.name != "nt":
            try:
                os.kill(pid, 0)           # chi an toan tren POSIX (tren Windows os.kill giet tien trinh)
            except OSError:
                return False
            return True
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in (out or "")


def _clear_stale_lock() -> None:
    try:
        with open(LOCK, encoding="utf-8") as fh:
            pid = int(json.load(fh).get("pid", -1))
    except FileNotFoundError:
        return
    except (ValueError, OSError):
        pid = -1                          # khoa hong -> coi nhu cu
    if pid < 0 or not _pid_alive(pid):
        with contextlib.suppress(FileNotFoundError):
            os.remove(LOCK)


@contextlib.contextmanager
def heavy_job(what: str, min_ram_gb: float = 3.0) -> Iterator[float]:
    """Mot viec nang mot luc (may ~3,5 GB trong): khoa TMP/heavy.lock + require_ram.
    Khoa cua tien trinh da chet bi bo. Chi dieu phoi cac script v6 dung heavy_job."""
    os.makedirs(TMP, exist_ok=True)
    _clear_stale_lock()
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(LOCK, encoding="utf-8") as fh:
            raise V6Error(f"dang co viec nang khac chay: {fh.read()}") from None
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "what": what, "t": time.strftime("%Y-%m-%d %H:%M:%S")}).encode())
        os.close(fd)
        yield require_ram(min_ram_gb, what)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.remove(LOCK)


# ---------------- fold nguoi noi ----------------
@functools.lru_cache(maxsize=1)
def _fold_map() -> tuple[tuple[str, int], ...]:
    with open(os.path.join(V4, "speaker_fold.json"), encoding="utf-8") as fh:
        return tuple((str(k), int(v)) for k, v in json.load(fh).items())


def speaker_folds() -> dict[str, int]:
    """Nguoi noi VIVOS -> fold 0-4 (speaker_fold.json, trung GroupKFold cua features_v4.pkl). Tra ban sao."""
    return dict(_fold_map())


def record_speakers(rec_or_meta: Mapping[str, Any]) -> list[str]:
    """Danh sach nguoi noi cua ban ghi (nhan rec hoac meta): meta['speakers'] (hoi thoai) + meta['speaker']."""
    meta = rec_or_meta["meta"] if isinstance(rec_or_meta.get("meta"), Mapping) else rec_or_meta
    spks = meta.get("speakers") or []
    spks = [spks] if isinstance(spks, str) else list(spks)
    sp = meta.get("speaker")
    if isinstance(sp, str) and sp:
        spks.append(sp)
    elif isinstance(sp, (list, tuple)):
        spks.extend(sp)
    return sorted({str(s) for s in spks if s})


def assign_fold(speakers: str | Sequence[str], unknown: str = "crc") -> int:
    """Fold cho MOT ban ghi theo (cac) nguoi noi cua no.
    - co nguoi noi trong speaker_fold.json -> fold do; nguoi noi da biet thuoc >1 fold -> LeakageError
      (khong mo hinh fold nao chua thay tat ca ho);
    - toan nguoi noi ngoai: unknown="crc" -> zlib.crc32(nguoi noi dau theo abc) % 5 (giong gen_test_suites.fold_of),
      unknown="all" -> -1 = cham bang ca 5 mo hinh fold roi lay trung binh."""
    spks = [speakers] if isinstance(speakers, str) else [str(s) for s in speakers]
    if not spks or not all(spks):
        raise ValueError("can it nhat mot ten nguoi noi khac rong")
    m = speaker_folds()
    known = {m[s] for s in spks if s in m}
    if len(known) > 1:
        raise LeakageError(f"nguoi noi {sorted(spks)} thuoc nhieu fold {sorted(known)}: khong cham sach duoc")
    if known:
        return known.pop()
    if unknown == "crc":
        return zlib.crc32(sorted(spks)[0].encode("utf-8")) % N_FOLDS
    if unknown == "all":
        return -1
    raise ValueError(f"unknown phai la 'crc' hoac 'all', nhan {unknown!r}")


def validate_fold(meta: Mapping[str, Any]) -> int:
    """meta['fold'] phai la so nguyen trong {-1, 0..4}; co nguoi noi VIVOS da biet -> phai dung fold cua ho (cam -1)."""
    if "fold" not in meta:
        raise LeakageError("meta thieu 'fold' (dung assign_fold / fold cua tep nguon)")
    f = meta["fold"]
    if isinstance(f, bool) or not isinstance(f, (int, np.integer)) or not -1 <= int(f) < N_FOLDS:
        raise LeakageError(f"fold {f!r} khong hop le (can -1 hoac 0..{N_FOLDS - 1})")
    m = speaker_folds()
    known = {m[s] for s in record_speakers(meta) if s in m}
    if known and (len(known) > 1 or int(f) not in known):
        raise LeakageError(f"fold {f} khac fold nguoi noi {sorted(known)} ({record_speakers(meta)})")
    return int(f)


def folds_to_score(fold: int) -> list[int]:
    """Fold -> danh sach mo hinh fold dung de cham (-1 = ca 5, lay trung binh)."""
    return list(range(N_FOLDS)) if int(fold) == -1 else [int(fold)]


# ---------------- id moi (chong trung cache) ----------------
def id_prefix(rid: str) -> str | None:
    """'v6ami_ES2004a_w3' -> 'v6ami' (None neu khong theo quy tac v6<ten>_)."""
    m = V6_PREFIX_RE.match(rid)
    return m.group(1) if m else None


def sidecar_path(feat_path: str) -> str:
    return os.path.abspath(feat_path) + ".ids.json"


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_default)
    os.replace(tmp, path)


def _feature_pkls() -> list[str]:
    ps = glob.glob(os.path.join(V4, "features_*.pkl")) + \
        glob.glob(os.path.join(V6DATA, "**", "features_*.pkl"), recursive=True)
    return sorted({os.path.abspath(p) for p in ps})


def _sidecars() -> list[str]:
    ps = glob.glob(os.path.join(V4, "*.ids.json")) + glob.glob(os.path.join(V6DATA, "**", "*.ids.json"), recursive=True)
    return sorted({os.path.abspath(p) for p in ps})


def _pkl_ids(path: str, index: Mapping[str, Any]) -> dict[str, Any]:
    """Muc chi muc {key, ids} cua mot pkl; chi mo pkl khi kich thuoc/mtime doi."""
    st = os.stat(path)
    key = f"{st.st_size}:{st.st_mtime_ns}"
    ent = index.get(path)
    if ent and ent.get("key") == key:
        return ent
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    except Exception as e:                # dang ghi do / hong -> KHONG bo qua (bo qua = co the trung id)
        raise V6Error(f"khong doc duoc id trong {path}: {e}") from e
    keys = obj.keys() if isinstance(obj, dict) else (r["id"] for r in obj)
    ids = sorted({str(k[0]) if isinstance(k, tuple) else str(k) for k in keys})
    return {"key": key, "ids": ids}


def existing_ids(exclude: Iterable[str] = ()) -> dict[str, list[str]]:
    """Moi id da dung (chu thuong: NTFS khong phan biet hoa/thuong) -> noi xuat hien.
    Nguon: V4/<SHARED_CACHES>/<id>.*, v5_segfeat_cache.pkl, features_*.pkl (V4 + V6DATA), sidecar *.ids.json.
    exclude: duong dan pkl/sidecar bo qua (tap cua chinh minh khi chay tiep)."""
    excl = {os.path.normcase(os.path.abspath(p)) for p in exclude if p}
    out: dict[str, list[str]] = {}

    def add(ids: Iterable[str], where: str) -> None:
        for i in ids:
            out.setdefault(i.lower(), []).append(where)

    for d in SHARED_CACHES:
        p = os.path.join(V4, d)
        if os.path.isdir(p):
            add((os.path.splitext(f)[0] for f in os.listdir(p)), d)
    old = _read_json(INDEX) if os.path.exists(INDEX) else {}
    pkls = _feature_pkls() + [os.path.join(V4, n) for n in ID_TUPLE_PKLS if os.path.exists(os.path.join(V4, n))]
    new = dict(old)
    for p in pkls:
        if os.path.normcase(p) in excl:
            continue
        new[p] = _pkl_ids(p, old)
        add(new[p]["ids"], os.path.basename(p))
    if new != old:
        _atomic_write_json(INDEX, new)
    for sc in _sidecars():
        if os.path.normcase(sc) in excl:
            continue
        add((it["id"] for it in _read_json(sc).get("items", [])), "sidecar:" + os.path.basename(sc))
    return out


def _name_errors(ids: Sequence[str], require_prefix: bool) -> list[str]:
    if not ids:
        return ["danh sach id rong"]
    errs = [f"id sai ky tu: {i!r}" for i in ids if not isinstance(i, str) or not ID_RE.match(i)]
    dups = sorted(i for i, c in Counter(str(i).lower() for i in ids).items() if c > 1)
    if dups:
        errs.append(f"id trung nhau trong chinh tap (khong phan biet hoa/thuong): {dups[:10]}")
    if require_prefix:
        pref = {id_prefix(str(i)) for i in ids}
        if None in pref:
            bad = [i for i in ids if not id_prefix(str(i))][:5]
            errs.append(f"id phai bat dau bang tien to v6<ten>_ (vd v6ami_): {bad}")
        elif len(pref) > 1:
            errs.append(f"mot tap chi mot tien to, nhan {sorted(pref)}")
    return errs


def check_new_ids(ids: Iterable[str], own_feat_path: str | None = None, own_ids: Iterable[str] = (),
                  require_prefix: bool = True) -> None:
    """Raise IdCollisionError neu id moi da dung o bat ky dau (existing_ids) hoac sai quy tac dat ten.
    own_feat_path: pkl cua CHINH tap nay khi chay tiep (bo qua pkl + sidecar cua no).
    own_ids: id sidecar cua chinh tap da giu voi sha256 wav KHOP -> duoc phep co san trong cache dung chung.
    Tien to (v6ami...) da xuat hien trong features_*.pkl / sidecar cua tap KHAC -> cung loi."""
    ids = list(ids)
    errs = _name_errors(ids, require_prefix)
    if errs:
        raise IdCollisionError("\n".join(errs))
    excl = [own_feat_path, sidecar_path(own_feat_path)] if own_feat_path else []
    used = existing_ids(exclude=excl)
    own = {i.lower() for i in own_ids}
    ok_for_own = set(SHARED_CACHES) | set(ID_TUPLE_PKLS)
    hits = []
    for i in ids:
        where = used.get(i.lower(), [])
        if i.lower() in own:
            where = [w for w in where if w not in ok_for_own]
        if where:
            hits.append(f"{i} -> {', '.join(sorted(set(where)))}")
    if require_prefix:
        pre = id_prefix(ids[0]) + "_"
        mine = {i.lower() for i in ids}
        foreign = sorted({w for k, ws in used.items() if k.startswith(pre) and k not in mine
                          for w in ws if w not in ok_for_own})
        if foreign:
            hits.append(f"tien to {pre} da thuoc tap khac: {foreign[:5]}")
    if hits:
        more = f"\n... +{len(hits) - 20} id nua" if len(hits) > 20 else ""
        raise IdCollisionError(f"{len(hits)} id/tien to da dung (cache dung chung khoa theo id):\n"
                               + "\n".join(hits[:20]) + more)


# ---------------- trich dac trung ----------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def abs_wav(wav: str) -> str:
    """Duong dan wav trong manifest (tuong doi V4 nhu run_codec_variants, hoac tuyet doi) -> tuyet doi."""
    return os.path.normpath(wav if os.path.isabs(wav) else os.path.join(V4, wav))


def rel_v4(path: str) -> str:
    """Duong dan -> tuong doi V4 dang '/' (quy uoc manifest hien co); khac o dia -> giu tuyet doi."""
    try:
        return os.path.relpath(os.path.abspath(path), V4).replace("\\", "/")
    except ValueError:
        return os.path.abspath(path).replace("\\", "/")


def _strip_feat_name(name: str) -> str:
    n = os.path.basename(str(name))
    n = n[:-4] if n.endswith(".pkl") else n
    return n[len("features_"):] if n.startswith("features_") else n


def feat_path_for(name: str, root: str = V4) -> str:
    """'v6ami' -> V4/features_v6ami.pkl (run_pipeline_v5b EXT=v6ami doc duoc)."""
    return os.path.join(root, f"features_{_strip_feat_name(name)}.pkl")


def _check_feat_path(feat_path: str) -> str:
    p = os.path.abspath(feat_path)
    name = _strip_feat_name(p)
    if not re.fullmatch(r"features_v6[A-Za-z0-9_]*\.pkl", os.path.basename(p)):
        raise V6Error(f"ten pkl v6 phai la features_v6<ten>.pkl, nhan {os.path.basename(p)}")
    if not os.path.normcase(p).startswith(os.path.normcase(BASE)):
        raise V6Error(f"pkl phai nam trong {BASE}: {p}")
    if os.path.exists(p) and not os.path.exists(sidecar_path(p)):
        raise V6Error(f"{p} da co ma khong co sidecar v6 -> khong ghi vao pkl cu ({name})")
    return p


def _normalize_entry(m: Mapping[str, Any]) -> dict[str, Any]:
    """Ban sao dong manifest dung dang features() can; kiem fold + wav 16 kHz mono (features() mac dinh nhu vay)."""
    import soundfile as sf
    missing = [k for k in ("id", "kind", "wav", "meta") if k not in m]
    if missing:
        raise ValueError(f"{m.get('id', '?')}: dong manifest thieu {missing}")
    meta = dict(m["meta"])
    try:
        validate_fold(meta)
    except LeakageError as e:
        raise LeakageError(f"{m['id']}: {e}") from None
    wav = abs_wav(str(m["wav"]))
    if not os.path.isfile(wav):
        raise FileNotFoundError(f"{m['id']}: khong thay {wav}")
    info = sf.info(wav)
    if info.samplerate != SR or info.channels != 1:
        raise ValueError(f"{m['id']}: features() can 16 kHz mono, tep co {info.samplerate} Hz {info.channels} kenh")
    return {**m, "id": str(m["id"]), "kind": str(m["kind"]), "meta": meta, "src": m.get("src") or str(m["id"]),
            "cfg": m.get("cfg") or "orig", "wav": rel_v4(wav)}


def _resume_ids(feat_path: str, items: Sequence[Mapping[str, Any]]) -> set[str]:
    """Id da duoc sidecar cua chinh pkl nay giu. Cung id ma audio/fold doi -> loi (dac trung cu se bi dung lai)."""
    sc = sidecar_path(feat_path)
    if not os.path.exists(sc):
        return set()
    old = {it["id"]: it for it in _read_json(sc).get("items", [])}
    bad = [it["id"] for it in items if it["id"] in old
           and (old[it["id"]].get("sha256") != it["sha256"] or old[it["id"]].get("fold") != it["fold"])]
    if bad:
        raise IdCollisionError(f"{len(bad)} id da trich voi AUDIO/FOLD KHAC (vd {bad[:3]}) -> dung tien to moi")
    return {it["id"] for it in items if it["id"] in old}


def _write_sidecar(feat_path: str, items: Sequence[Mapping[str, Any]], status: str,
                   extra: Mapping[str, Any] | None = None) -> None:
    sc = sidecar_path(feat_path)
    created = _read_json(sc).get("created") if os.path.exists(sc) else None
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    _atomic_write_json(sc, {"version": 1, "feat_path": os.path.abspath(feat_path), "prefix": id_prefix(items[0]["id"]),
                            "status": status, "created": created or now, "updated": now, "n": len(items),
                            **(extra or {}), "items": list(items)})


def _finish(feat_path: str, entries: Sequence[Mapping[str, Any]], items: Sequence[Mapping[str, Any]],
            summary: Mapping[str, Any]) -> dict[str, Any]:
    """Sau features(): dong bo nhan tu manifest (features() bo qua id da co -> meta cu), kiem cache, sha256 pkl."""
    with open(feat_path, "rb") as fh:
        cache = pickle.load(fh)
    emap = {e["id"]: e for e in entries}
    keep = ("kind", "meta", "src", "cfg", "wav")
    upd = {k: {**r, **{f: emap[k][f] for f in keep}} for k, r in cache.items()
           if k in emap and any(r.get(f) != emap[k][f] for f in keep)}
    if upd:
        tmp = f"{feat_path}.tmp{os.getpid()}"
        with open(tmp, "wb") as fh:
            pickle.dump({k: upd.get(k, r) for k, r in cache.items()}, fh)
        os.replace(tmp, feat_path)
    missing = [i for i in emap if i not in cache]
    no_cache = [i for i in emap if not all(os.path.exists(os.path.join(V4, d, i + ext)) for d, ext in FEAT_CACHES)]
    extra_ids = sorted(k for k in cache if k not in emap)
    del cache
    info = {"pkl_sha256": sha256_file(feat_path), "missing": missing, "missing_cache": no_cache,
            "extra_in_pkl": extra_ids, "meta_synced": len(upd)}
    status = "done" if not missing and not no_cache else "partial"
    _write_sidecar(feat_path, items, status, info)
    return {**summary, "status": status, **{k: (len(v) if isinstance(v, list) else v) for k, v in info.items()}}


def extract_features(manifest: Sequence[Mapping[str, Any]], feat_path: str, threads: int = DEFAULT_THREADS,
                     min_ram_gb: float = 3.0, dry_run: bool = False) -> dict[str, Any]:
    """Trich dac trung Y HET v4 (run_codec_variants.features) cho tap moi, co chot chan ro ri:
    1) chuan hoa manifest (ban sao: src=id, cfg='orig' neu thieu; wav tuong doi V4) + kiem wav 16 kHz mono;
    2) validate_fold(meta); 3) check_new_ids (chay tiep: id sidecar da giu voi sha256 wav + fold khop duoc bo qua);
    4) heavy_job: khoa + require_ram(min_ram_gb); 5) ghi sidecar <feat>.ids.json TRUOC khi chay (giu cho id);
    6) features(); 7) dong bo nhan tu manifest vao pkl, sidecar: sha256 pkl, id thieu pkl/cache, status done|partial.
    dry_run=True: chi 1-3, khong ghi gi, khong nap model. feat_path: features_v6<ten>.pkl (nen dat o V4 cho v5b EXT).
    Luu y: rec['gt'] / rec['neg'] = [] nhu moi tap ngoai (danh gia doc meta['edits'])."""
    setup_env(threads)
    feat_path = _check_feat_path(feat_path)
    entries = [_normalize_entry(m) for m in manifest]
    items = [{"id": e["id"], "wav": e["wav"], "sha256": sha256_file(abs_wav(e["wav"])), "fold": int(e["meta"]["fold"])}
             for e in entries]
    own = _resume_ids(feat_path, items)
    check_new_ids([e["id"] for e in entries], own_feat_path=feat_path, own_ids=own)
    summary = {"feat_path": feat_path, "n": len(entries), "resume": len(own), "prefix": id_prefix(entries[0]["id"]),
               "folds": {str(f): sum(it["fold"] == f for it in items) for f in sorted({it["fold"] for it in items})}}
    if dry_run:
        return {**summary, "dry_run": True}
    with heavy_job(f"extract_features {os.path.basename(feat_path)}", min_ram_gb):
        _write_sidecar(feat_path, items, "running")
        from run_codec_variants import features
        features(entries, feat_path)
    return _finish(feat_path, entries, items, summary)


def load_feats(name: str) -> dict[str, dict[str, Any]]:
    """features_<name>.pkl (tim o V4 roi V6DATA; nhan 'codec', 'features_codec.pkl' hoac duong dan) -> {id: ban ghi}."""
    if (os.path.isabs(name) or "/" in name or "\\" in name) and os.path.isfile(name):
        paths = [name]
    else:
        n = _strip_feat_name(name)
        paths = [feat_path_for(n, V4), feat_path_for(n, V6DATA)]
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                d = pickle.load(fh)
            return d if isinstance(d, dict) else {r["id"]: r for r in d}
    raise FileNotFoundError(f"khong thay features cho {name!r}: {paths}")


def read_manifest(path: str) -> list[dict[str, Any]]:
    """manifest .json (list) hoac .jsonl -> list dong."""
    with open(path, encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in fh if line.strip()]
        return list(json.load(fh))


# ---------------- LOCKBOX ----------------
def _norm_set(set_name: Any) -> str:
    s = os.path.basename(str(set_name or "")).lower()
    s = s[:-4] if s.endswith(".pkl") else s
    return s[len("features_"):] if s.startswith("features_") else s


def _base_ids(rec: Mapping[str, Any]) -> list[str]:
    """id, src, meta.src/parent va ban da bo tien to v6*_ (tap v6 dan xuat tu tap cu giu tien to goc phia sau)."""
    meta = rec.get("meta") or {}
    out = []
    for x in (rec.get("id"), rec.get("src"), meta.get("src"), meta.get("parent")):
        x = str(x) if x else ""
        while x:
            out.append(x)
            m = _V6_STRIP_RE.match(x)
            x = x[m.end():] if m else ""
    return out


def ami_series(text: str) -> str | None:
    """'v6ami_ES2004a_w3' / 'ES2004a.Mix-Headset.wav' -> 'ES2004' (None neu khong co ma cuoc hop AMI)."""
    m = _AMI_RE.search(str(text or ""))
    return m.group(1) if m else None


def _is_english(rec: Mapping[str, Any], sets: set[str], bases: Sequence[str]) -> bool:
    meta = rec.get("meta") or {}
    if any("english" in s or s.startswith("v6en") for s in sets):
        return True
    if str(meta.get("lang", "")).lower() in ("en", "eng", "english"):
        return True
    return any(b.startswith("en_") for b in bases) or any(s.startswith("LS") for s in record_speakers(rec))


def lockbox_reason(rec: Mapping[str, Any], set_name: str = "") -> str | None:
    """Ly do ban ghi thuoc LOCKBOX (chi dung cho bao cao cuoi, cam moi quyet dinh chon/bo), None neu khong:
      meta['lockbox']=True · bo hard (T4: tap suite_hard / id hard_*)
      · MIST gia (features_mist.pkl: tap mist / id mist_*)
      · nguoi noi tieng Anh co zlib.crc32(speaker) % 2 == 1 · AMI series TS3003
      · cau hinh nen (cfg / meta.codec_cfg, tach '+' hoac '__') khong thuoc 8 cau hinh CODEC_CONFIGS.
    set_name: ten tap/pkl ('suite_hard', 'features_mist.pkl'...); meta['src_set'] = tap goc cua ban dan xuat."""
    meta = rec.get("meta") or {}
    if meta.get("lockbox") is True:
        return "meta_lockbox"
    sets = {s for s in (_norm_set(set_name), _norm_set(meta.get("src_set"))) if s}
    bases = _base_ids(rec)
    if sets & {"suite_hard", "hard"} or any(b.startswith("hard_") for b in bases):
        return "hard_suite"
    if "mist" in sets or any(b.startswith("mist_") for b in bases):
        return "mist_fake"
    if _is_english(rec, sets, bases):
        odd = [s for s in record_speakers(rec) if zlib.crc32(s.encode("utf-8")) % 2 == 1]
        if odd:
            return f"english_speaker:{odd[0]}"
    texts = bases + [str(meta.get(k) or "") for k in ("series", "meeting", "host_file")] + [str(rec.get("wav") or "")]
    if any(ami_series(t) == AMI_LOCKBOX_SERIES for t in texts):
        return f"ami_{AMI_LOCKBOX_SERIES}"
    for cfg in (meta.get("codec_cfg"), rec.get("cfg"), meta.get("cfg")):
        for part in re.split(r"\+|__", str(cfg or "")):
            if part and _CODEC_RE.search(part) and part not in CODEC_CONFIGS:
                return f"codec_unseen:{part}"
    return None


def is_lockbox(rec: Mapping[str, Any], set_name: str = "") -> bool:
    """True neu ban ghi thuoc LOCKBOX (xem lockbox_reason)."""
    return lockbox_reason(rec, set_name) is not None


def split_lockbox(recs: Iterable[Mapping[str, Any]], set_name: str = "") -> tuple[list, list]:
    """(dung_duoc_de_quyet_dinh, lockbox) — giu nguyen thu tu."""
    recs = list(recs)
    lock = [is_lockbox(r, set_name) for r in recs]
    return [r for r, x in zip(recs, lock) if not x], [r for r, x in zip(recs, lock) if x]


def assert_no_lockbox(recs: Iterable[Mapping[str, Any]], set_name: str = "", purpose: str = "hieu chinh") -> None:
    """Raise LeakageError neu co ban ghi LOCKBOX lot vao tap dung cho `purpose` (hoc / hieu chinh / chon nguong)."""
    bad = [(r.get("id"), lockbox_reason(r, set_name)) for r in recs if is_lockbox(r, set_name)]
    if bad:
        raise LeakageError(f"{len(bad)} ban ghi LOCKBOX trong tap {purpose} ({set_name}): {bad[:5]}")


# ---------------- NHOM chia fold (giao dien I2, 10/10) ----------------
# Mot ham nhom cho moi tap: AMI theo SERIES (cung 4 nguoi + cung phong), EdAcc theo CUOC HOI THOAI (2 nguoi chung
# mot cuoc goi), VSS theo KENH (= nguoi dan chuong trinh), voxp:None tach rieng tung ban goc, du lieu cao theo kenh
# (cho toi khi phan cum nguoi noi xong -> meta.group), con lai theo nguoi. KHONG thay detect.person_fold (mo hinh v9
# giu nguyen); fold moi = folds_v2 (v6.data.groups) cho lan hoc lai sau.
FOLDS_V2 = os.path.join(TMP, "cache", "folds_v2.json")              # {id: fold}
FOLDS_V2_GROUPS = os.path.join(TMP, "cache", "folds_v2_groups.json")  # {"groups": {nhom: fold}, ...} (bang dong bang)
_EDACC_RE = re.compile(r"EDACC-C\d+", re.I)
_NONE_PERSONS = {"", "none", "nan", "null", "unknown"}
HOSTS_JSONL = (os.path.join(BASE, "data", "natural_v6", "hosts", "hosts.jsonl"),
               os.path.join(BASE, "data", "natural_v6", "hosts_en", "hosts.jsonl"))


@functools.lru_cache(maxsize=1)
def _host_table() -> dict[str, tuple[str, str, str]]:
    """host_file -> (source, cluster, person) tu hosts*.jsonl (doc mot lan; chi 3 truong, bo words/text)."""
    out: dict[str, tuple[str, str, str]] = {}
    for p in HOSTS_JSONL:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[str(r["file"])] = (str(r.get("source") or ""), str(r.get("cluster") or ""), str(r.get("person") or ""))
    return out


@functools.lru_cache(maxsize=1)
def _ami_person_series() -> dict[str, str]:
    """nguoi AMI ('ami:FEE005') -> series ('ES2004') theo cum (cuoc hop) cua ban goc trong hosts_en."""
    out: dict[str, str] = {}
    for src, cluster, person in _host_table().values():
        s = ami_series(cluster) if src == "ami" else None
        if s:
            out.setdefault(person, s)
    return out


def _meta_of(rec_or_meta: Mapping[str, Any]) -> Mapping[str, Any]:
    return rec_or_meta["meta"] if isinstance(rec_or_meta.get("meta"), Mapping) else rec_or_meta


def _person_missing(person: str) -> bool:
    return person.split(":", 1)[-1].strip().lower() in _NONE_PERSONS


def group_of(meta: Mapping[str, Any], set_name: str = "") -> str:
    """Khoa NHOM chong ro ri cho mot ban ghi (nhan meta hoac ca dong manifest).
      meta['group'] co san (du lieu cao: kenh / cum nguoi da phan) -> giu nguyen
      AMI    -> 'ami:<series>'  (ES2004 / IS1009 / TS3003...: tu meeting/speaker/cluster/host_file, hoac nguoi -> series)
      EdAcc  -> 'edacc:EDACC-Cxx' (cuoc hoi thoai)
      VSS    -> 'vss:<kenh>'    (meta.person cua VSS da la kenh)
      voxp:None -> 'voxp:None:<host_file>' (khong biet nguoi -> moi ban goc mot nhom rieng)
      du lieu cao chua phan cum -> 'scrape:<channel_id>'
      con lai -> meta.person, khong co thi meta.speaker (VIVOS v4: ma nguoi noi).
    set_name chi de bao loi. Thieu het thong tin -> ValueError."""
    m = _meta_of(meta)
    g = m.get("group")
    if isinstance(g, str) and g.strip():
        return g.strip()
    host = str(m.get("host_file") or "")
    hsrc, hcluster, hperson = _host_table().get(host, ("", "", "")) if host else ("", "", "")
    person = str(m.get("person") or hperson or "")
    speaker = str(m.get("speaker") or "")
    cluster = str(m.get("cluster") or hcluster or "")
    source = str(m.get("source") or hsrc or "").lower()
    if not source:
        for x in (person, speaker, host):
            if x and (":" in x or "_" in x):
                source = re.split(r"[:_]", x, maxsplit=1)[0].lower()
                break
    if source == "ami" or (not source and (ami_series(speaker) or ami_series(cluster) or m.get("meeting"))):
        for t in (m.get("series"), m.get("meeting"), speaker, cluster, host, person):
            s = ami_series(str(t or ""))
            if s:
                return f"ami:{s}"
        s = _ami_person_series().get(person)
        if s:
            return f"ami:{s}"
        if source == "ami" and person:
            return person if person.startswith("ami:") else f"ami:{person}"
    if source == "edacc":
        for t in (cluster, speaker, person, host):
            mm = _EDACC_RE.search(str(t))
            if mm:
                return f"edacc:{mm.group(0).upper()}"
    if source == "vss":
        ch = person or (cluster.split(":")[1] if cluster.count(":") >= 1 else "")
        if ch:
            return ch if ch.startswith("vss:") else f"vss:{ch}"
    if source == "voxp" and _person_missing(person or speaker):
        key = host or str(m.get("src") or m.get("base") or "")
        return f"voxp:None:{key}" if key else "voxp:None"
    chan = m.get("channel_id")
    if isinstance(chan, str) and chan.strip() and not person:
        return f"scrape:{chan.strip()}"
    if person and not _person_missing(person):
        return person
    if speaker and not _person_missing(speaker):
        return speaker
    if isinstance(chan, str) and chan.strip():
        return f"scrape:{chan.strip()}"
    raise ValueError(f"khong xac dinh duoc nhom (thieu group/person/speaker/host_file){' trong tap ' + set_name if set_name else ''}")


@functools.lru_cache(maxsize=2)
def _frozen_table(path: str, mtime: float) -> Mapping[str, int]:
    with open(path, encoding="utf-8") as fh:
        return MappingProxyType({str(k): int(v) for k, v in json.load(fh).get("groups", {}).items()})


def _frozen_group_folds() -> Mapping[str, int]:
    """Bang dong bang {nhom: fold} (doc mot lan theo mtime: tep ghi lai -> tu doc lai, khong dung bang cu)."""
    if not os.path.exists(FOLDS_V2_GROUPS):
        return MappingProxyType({})
    return _frozen_table(FOLDS_V2_GROUPS, os.path.getmtime(FOLDS_V2_GROUPS))


_frozen_group_folds.cache_clear = _frozen_table.cache_clear  # type: ignore[attr-defined]  # giu API cu (groups.build goi)

FOLD_SALT = "fold:"   # crc32 tran (khong muoi) = luat lockbox v2 cho du lieu cao (crc32 % 5 == 0) -> phai KHAC de fold doc lap


def crc_fold_of_group(group: str) -> int:
    """Fold du phong cua nhom CHUA co trong bang: zlib.crc32("fold:" + nhom) % 5. v6.data.groups.build --force GHIM moi
    nhom moi vao dung gia tri nay -> fold da phat ra (WP6 meta.fold, expand_hosts fold_v2) khong bao gio doi."""
    return zlib.crc32((FOLD_SALT + group).encode("utf-8")) % N_FOLDS


def fold_of_group(group: str, strict: bool = False) -> int:
    """Nhom -> fold 0..4: theo bang dong bang folds_v2_groups.json (chia can bang theo nguon, v6.data.groups),
    nhom moi chua co trong bang -> crc_fold_of_group (tat dinh; co muoi de fold KHONG trung voi chon lockbox v2 cua du
    lieu cao = crc32(kenh) % 5 == 0). Gia tri tra ve la CUOI CUNG: build --force ghim nhom moi vao dung gia tri nay.
    strict=True: nhom chua co trong bang -> V6Error (code hoc muon chac chan fold lay tu bang).
    Nhom lockbox v2 van phai loai bang v6.data.lockbox.assert_not_lockbox."""
    if not isinstance(group, str) or not group:
        raise ValueError("nhom phai la chuoi khac rong")
    table = _frozen_group_folds()
    if group in table:
        return int(table[group])
    if strict:
        raise V6Error(f"nhom {group!r} chua co trong {FOLDS_V2_GROUPS} (chay: python -m v6.data.groups build --force)")
    return crc_fold_of_group(group)


def check_disjoint_groups(calib_groups: Iterable[Any], test_groups: Iterable[Any], what: str = "series") -> None:
    """Raise LeakageError neu nhom (series AMI / nguoi noi) vua dung hieu chinh vua dung kiem tra."""
    both = sorted({str(x) for x in calib_groups} & {str(x) for x in test_groups})
    if both:
        raise LeakageError(f"{what} vua hieu chinh vua kiem tra: {both}")


# ---------------- ket qua ----------------
def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=str)
    raise TypeError(f"khong ghi JSON duoc kieu {type(o).__name__}")


def write_result(name: str, data: Mapping[str, Any], root: str = RESULTS) -> str:
    """Ghi ket qua (dict) vao <root>/<name>.json (ghi nguyen tu, them _meta: ten, gio, script). Tra duong dan."""
    if not _NAME_RE.match(name):
        raise ValueError(f"ten ket qua chi gom [A-Za-z0-9_.-], nhan {name!r}")
    payload = {**data, "_meta": {"name": name, "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                                 "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""}}
    path = os.path.join(root, name + ".json")
    _atomic_write_json(path, payload)
    return path


def read_result(name: str, root: str = RESULTS) -> dict[str, Any]:
    return _read_json(os.path.join(root, name + ".json"))


# ---------------- CLI ----------------
def _cli_lockbox(name: str) -> dict[str, Any]:
    recs = load_feats(name)
    reasons: dict[str, int] = {}
    for r in recs.values():
        k = (lockbox_reason(r, name) or "mo").split(":")[0]
        reasons[k] = reasons.get(k, 0) + 1
    return {"set": name, "n": len(recs), "reasons": reasons}


def main() -> None:
    ap = argparse.ArgumentParser(description="Ha tang v6: RAM, fold, id moi, trich dac trung, LOCKBOX")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ram", help="RAM dang trong (GB)")
    sub.add_parser("folds", help="so nguoi noi moi fold")
    p = sub.add_parser("ids", help="kiem id cua manifest (khong ghi gi)")
    p.add_argument("manifest")
    p.add_argument("--feat", default=None, help="pkl dich (kiem ca chay tiep + sha256)")
    p = sub.add_parser("extract", help="trich dac trung (viec NANG: WavLM + DeepFense)")
    p.add_argument("manifest")
    p.add_argument("feat")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    p.add_argument("--min-ram", type=float, default=3.0)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("lockbox", help="dem ban ghi LOCKBOX theo ly do trong features_<ten>.pkl")
    p.add_argument("name")
    a = ap.parse_args()
    with contextlib.suppress(AttributeError):
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd == "ram":
        out: Any = {"ram_free_gb": round(ram_free_gb(), 2)}
    elif a.cmd == "folds":
        m = speaker_folds()
        out = {str(f): sum(v == f for v in m.values()) for f in range(N_FOLDS)}
    elif a.cmd == "ids":
        man = read_manifest(a.manifest)
        if a.feat:
            out = extract_features(man, a.feat, dry_run=True)
        else:
            check_new_ids([m["id"] for m in man])
            out = {"ok": len(man)}
    elif a.cmd == "extract":
        out = extract_features(read_manifest(a.manifest), a.feat, a.threads, a.min_ram, a.dry_run)
    else:
        out = _cli_lockbox(a.name)
    print(json.dumps(out, ensure_ascii=False, default=_json_default))


if __name__ == "__main__":
    main()
