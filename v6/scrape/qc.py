# -*- coding: utf-8 -*-
"""KIEM SOAT CHAT LUONG du lieu cao (coverage B-12): trung lap + mau nghe tay. Khong torch, khong mang.

  dedup   van tay DUONG BAO NANG LUONG (rms_db khung 100 ms, da luu trong frames/<id>.npz cua filter):
          bit_i = [rms(i + LAG) > rms(i)] (bat bien voi tang ich, ben voi ma hoa lai Opus/AAC/MP3), bam 24 bit lien tiep,
          bo bam trong khoang lang (bien do duong bao < MIN_SWING dB) va bam qua pho bien (> MAX_DF tep).
          Doi chieu kieu Shazam: cung bam o hai tep -> phieu cho do lech (pos_a - pos_b); dinh phieu >= MIN_VOTES va
          >= MIN_VOTE_FRAC so bam cua phan chung -> TRUNG (cung noi dung, vd kenh dang lai / ban cat ngan / ban ghi doi).
          Chi muc chia buoc --stride ben 'kho', truy van moi vi tri -> do lech bat ky van khop.
          PHA KHUNG (sua review 11/10 + 12/10): duong bao 100 ms phu thuoc diem bat dau; doan cat lech 50 ms so voi luoi
          khung cua tep goc cho bam 24 bit khac han -> truy van o NHIEU PHA va gop phieu (lay pha tot nhat):
            - truy van tu wav (host): duong bao chinh xac o pha 0, 1/P, ..., (P-1)/P khung (P = --phases, mac dinh 8
              = buoc 12,5 ms);
            - tep cao vs tep cao: duong bao CHINH XAC o 8 pha tu frames/<id>.env8.npy (cua so 100 ms, buoc 12,5 ms; filter
              ghi cho tep moi, 'qc env8' giai ma lai tep cu - nhe, ~1 000 x thoi gian thuc). Tep chua co env8 -> pha xap
              xi (noi suy cong suat p'(i) = (1-a) p(i) + a p(i+1)) va duoc DEM rieng (n_approx) - review 12/10 do duong
              xap xi chi bat 66/136 doan cat lech luoi.
          XAC MINH (review 12/10): moi ung vien trung (ca cap tep-tep lan host) phai co tuong quan duong bao dB tai do lech
          da bau >= VERIFY_CORR (0,8) -> 'dup_pairs' / 'against_hits'; ung vien bi bo ghi 'rejected_*' (truoc day 1 trung
          gia: hop hoi dong Durham NC 'trung' doan Vietnam-Celeb 5,6 s voi dung 5 phieu toi thieu).
          --against <hosts.jsonl ...>: doi chieu them voi host co san (vd data/natural_v6/hosts/hosts.jsonl, VSS cung la
          YouTube) - doc wav tung host (nang vua: hang doi khi chay tron bo).
  env8    tinh frames/<id>.env8.npy cho tep cao chua co (giai ma lai bang ffmpeg, khong luu am thanh).
  selfcheck  kiem tra that, HAI duong -> meta/qc_selfcheck.json (muc tieu >= 95 % moi duong, B-12):
          host: moi doan da cat (meta/segments.jsonl, wav trong hosts_*) truy van (pha tu wav) vao chi muc tep goc
                -> phai trung DUNG tep va |offset + start| <= 0,25 s;
          file: moi doan coi nhu MOT TEP CAO KHAC (env8 cua no) va chay dedup() that tren {tep goc} + {doan} -> phai ra cap
                (doan, tep goc) voi do lech dung (duong tep-vs-tep: ban dang lai bi cat ngan / lech luoi).
  listen  mau nghe tay ~5 % host (hoac keep_spans khi chua co host) -> meta/listen_sample.csv (url + moc thoi gian) de
          nguoi nghe xac nhan 'chua bien tap' (B-12: >= 95 % mau xac nhan).
Ra: meta/qc_dedup.json, meta/listen_sample.csv.

CLI: python -m v6.scrape.qc env8 [--budget-min 8]
     python -m v6.scrape.qc dedup [--stride 4] [--phases 8] [--against data/natural_v6/hosts/hosts.jsonl] [--limit-against 500]
     python -m v6.scrape.qc selfcheck [--stride 4] [--phases 8] [--limit 0]
     python -m v6.scrape.qc listen [--frac 0.05] [--seed 0]"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import frames as FR  # noqa: E402
from v6.scrape import util as U  # noqa: E402

LAG = 2               # so sanh khung cach 200 ms (lon hon dao dong do ma hoa lai)
NBITS = 24
MIN_SWING = 6.0       # dB: cua so 24 bit phai co bien do duong bao >= 6 dB (bo khoang lang / on nen)
MAX_DF = 8            # bam xuat hien o > 8 tep -> bo (mau pho bien: nhac hieu, tieng vo tay...)
MIN_VOTES = 12
MIN_VOTES_SHORT = 5   # san cho truy van ngan (xem Index.need_votes)
VOTE_SHARE = 0.35
MIN_VOTE_FRAC = 0.02
PHASES = 8
VERIFY_CORR = 0.8     # tuong quan duong bao toi thieu de XAC NHAN mot ung vien trung (xem env_corr; selfcheck ghi phan bo)
QC_JSON =os.path.join(U.META, "qc_dedup.json")
SELFCHECK_JSON = os.path.join(U.META, "qc_selfcheck.json")
LISTEN_CSV = os.path.join(U.META, "listen_sample.csv")


# ---------------- van tay ----------------
def fp_hashes(rms_db: np.ndarray, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """(bam uint32, vi tri khung) cua duong bao rms_db (100 ms). stride > 1: chi lay vi tri chia het (chi muc)."""
    e = np.asarray(rms_db, np.float32)
    n = len(e) - LAG - NBITS + 1
    if n <= 0:
        return np.zeros(0, np.uint32), np.zeros(0, np.int32)
    bits = (e[LAG:] > e[:-LAG]).astype(np.uint32)
    h = np.zeros(n, np.uint32)
    for k in range(NBITS):
        h |= bits[k:k + n] << np.uint32(k)
    win = NBITS + LAG
    # bien do duong bao trong cua so (max - min) bang tich luy cua so truot
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(e, win)[:n]
    swing = sw.max(axis=1) - sw.min(axis=1)
    pos = np.arange(n, dtype=np.int32)
    keep = (swing >= MIN_SWING) & (pos % stride == 0)
    return h[keep], pos[keep]


class Index:
    """Chi muc bam -> (tep, vi tri) bang mang numpy da sap xep (RAM ~ 12 byte / muc)."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self._h: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self._p: list[np.ndarray] = []
        self.h = self.f = self.p = None
        self.stride = 1

    def add(self, name: str, rms_db: np.ndarray, stride: int) -> None:
        h, p = fp_hashes(rms_db, stride)
        self.stride = max(self.stride, stride)
        self._h.append(h)
        self._p.append(p)
        self._f.append(np.full(len(h), len(self.names), np.int32))
        self.names.append(name)

    def finish(self, max_df: int = MAX_DF) -> None:
        h = np.concatenate(self._h) if self._h else np.zeros(0, np.uint32)
        f = np.concatenate(self._f) if self._f else np.zeros(0, np.int32)
        p = np.concatenate(self._p) if self._p else np.zeros(0, np.int32)
        o = np.argsort(h, kind="stable")
        h, f, p = h[o], f[o], p[o]
        if len(h):   # so tep khac nhau co bam -> bo bam pho bien
            uh, start, cnt = np.unique(h, return_index=True, return_counts=True)
            df = np.zeros(len(uh), np.int32)
            for i, (s, c) in enumerate(zip(start, cnt)):
                df[i] = len(np.unique(f[s:s + c])) if c > 1 else 1
            bad = set(uh[df > max_df].tolist())
            if bad:
                m = ~np.isin(h, list(bad))
                h, f, p = h[m], f[m], p[m]
        self.h, self.f, self.p = h, f, p
        self.n_per = np.bincount(f, minlength=len(self.names)) if len(f) else np.zeros(len(self.names), np.int64)
        self._h = self._f = self._p = []

    def query(self, rms_db: np.ndarray, skip: int | None = None) -> list[dict]:
        """Doi chieu MOI vi tri cua truy van -> [{name, votes, offset_s, frac}] cac tep vuot nguong."""
        qh, qp = fp_hashes(rms_db, 1)
        if not len(qh) or self.h is None or not len(self.h):
            return []
        lo = np.searchsorted(self.h, qh, "left")
        hi = np.searchsorted(self.h, qh, "right")
        votes: dict[int, Counter] = defaultdict(Counter)
        for i in np.where(hi > lo)[0]:
            for j in range(lo[i], hi[i]):
                fi = int(self.f[j])
                if fi != skip:
                    votes[fi][int(qp[i] - self.p[j])] += 1
        out = []
        for fi, c in votes.items():
            off, v = c.most_common(1)[0]
            v += c.get(off - 1, 0) + c.get(off + 1, 0)         # dung sai +-1 khung
            n_ref = int(self.n_per[fi])
            frac = v / max(1, min(n_ref, len(qh)))
            if v >= self.need_votes(len(qh)) and frac >= MIN_VOTE_FRAC:
                out.append({"name": self.names[fi], "votes": int(v), "offset_s": round(off * 0.1, 1),
                            "offset_fr": int(off), "frac": round(float(frac), 4)})
        return sorted(out, key=lambda r: -r["votes"])

    def need_votes(self, n_query: int) -> int:
        """Nguong phieu theo do dai truy van: doan ngan (5-9 s ~ 25-65 bam, chi 1/stride trung chi muc) khong the dat
        MIN_VOTES -> ha xuong VOTE_SHARE * n_query / stride, khong duoi MIN_VOTES_SHORT (phieu phai CUNG do lech +-1 khung,
        trung ngau nhien 24 bit o cung do lech rat hiem; do bang so khop sai trong selfcheck)."""
        return int(max(MIN_VOTES_SHORT, min(MIN_VOTES, VOTE_SHARE * n_query / max(1, self.stride))))

    def query_multi(self, envs: list[np.ndarray], skip: int | None = None) -> list[dict]:
        """Truy van nhieu pha cua CUNG mot doan; moi tep giu pha co nhieu phieu nhat (them truong 'phase')."""
        best: dict[str, dict] = {}
        for k, e in enumerate(envs):
            for m in self.query(e, skip):
                if m["name"] not in best or m["votes"] > best[m["name"]]["votes"]:
                    best[m["name"]] = {**m, "phase": k}
        return sorted(best.values(), key=lambda r: -r["votes"])


def env_corr(q_env: np.ndarray, ref_env: np.ndarray, off_fr: int, max_shift: int = 1, floor_db: float = -70.0) -> float:
    """XAC MINH mot ung vien trung (review 12/10: phieu toi thieu 5 cho truy van ngan sinh trung gia, vd hop hoi dong Durham
    'trung' phong van nguoi noi tieng Viet): tuong quan Pearson duong bao dB cua truy van q[j] voi ref[j - off_fr]
    (do lech cua Index.query), lay max trong +-max_shift khung; bo khung lang so ca hai phia. Ban sao that (ke ca doi
    tang ich / ma hoa lai) ~0,9-1,0; ngau nhien ~0-0,5. Tra -1 neu phan chung < 20 khung."""
    q, r = np.asarray(q_env, np.float64), np.asarray(ref_env, np.float64)
    best = -1.0
    for d in range(-max_shift, max_shift + 1):
        o = off_fr + d
        j0, j1 = max(0, o), min(len(q), len(r) + o)
        if j1 - j0 < 20:
            continue
        a, b = q[j0:j1], r[j0 - o:j1 - o]
        m = (a > floor_db) | (b > floor_db)
        if m.sum() < 20 or a[m].std() < 1e-6 or b[m].std() < 1e-6:
            continue
        best = max(best, float(np.corrcoef(a[m], b[m])[0, 1]))
    return round(best, 4)


def verify(m: Mapping[str, Any], q_phases: list[np.ndarray], ref_phase0: np.ndarray) -> dict:
    """Them 'corr' va 'verified' (corr >= VERIFY_CORR) cho mot ket qua query_multi."""
    c = env_corr(q_phases[int(m.get("phase", 0))], ref_phase0, int(m.get("offset_fr", round(m["offset_s"] * 10))))
    return {**m, "corr": c, "verified": c >= VERIFY_CORR}


def env_from_audio(y: np.ndarray, phase: int = 0, hop: int = 1600) -> np.ndarray:
    """rms_db khung hop mau bat dau tu mau 'phase' - cung dinh nghia voi frames.FrameAccumulator (16 kHz, 100 ms)."""
    y = y[phase:]
    k = len(y) // hop
    fr = y[:k * hop].reshape(k, hop).astype(np.float64)
    return (20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-10)).astype(np.float32)


def read16k(path: str) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != 16000:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y


def rms_db_wav(path: str) -> np.ndarray:
    """rms_db khung 100 ms cua tep wav (host co san), pha 0."""
    return env_from_audio(read16k(path))


def phases_from_audio(y: np.ndarray, n_phase: int = PHASES, hop: int = 1600) -> list[np.ndarray]:
    """Duong bao chinh xac o n_phase pha deu nhau trong mot khung."""
    return [env_from_audio(y, int(round(k * hop / n_phase)), hop) for k in range(max(1, n_phase))]


def phases_from_env(rms_db: np.ndarray, n_phase: int = PHASES) -> list[np.ndarray]:
    """Pha XAP XI tu duong bao 100 ms co san (khong co audio / env8): noi suy cong suat giua hai khung lien tiep.
    Chi la du phong: review 12/10 do duong nay chi bat 66/136 doan cat lech luoi -> dung env8 (pha chinh xac)."""
    e = np.asarray(rms_db, np.float64)
    pw = 10 ** (e / 10)
    out = [e.astype(np.float32)]
    for k in range(1, max(1, n_phase)):
        a = k / n_phase
        q = (1 - a) * pw[:-1] + a * pw[1:]
        out.append((10 * np.log10(q + 1e-20)).astype(np.float32))
    return out


def phases_from_y(y: np.ndarray, n_phase: int = PHASES) -> list[np.ndarray]:
    """Pha CHINH XAC cua mot tep / doan am thanh qua CUNG duong env8 cua tep cao (FineEnvelope -> phases_from_env8)."""
    fe = FR.FineEnvelope()
    fe.feed(y)
    return FR.phases_from_env8(fe.finish(), n_phase)


def scraped_envelopes(n_phase: int = PHASES) -> tuple[list[tuple[str, list[np.ndarray]]], int]:
    """[(id, [duong bao 8 pha])] cua tep cao: env8 (chinh xac) neu co, khong thi pha xap xi tu rms_db. Tra them so tep xap xi."""
    from v6.scrape.filter import env8_path, frames_path
    out, n_approx = [], 0
    for r in U.latest_by(U.read_jsonl(U.FILTERED)).values():
        p8 = env8_path(r["id"])
        if os.path.exists(p8):
            out.append((r["id"], FR.phases_from_env8(np.load(p8).astype(np.float32), n_phase)))
            continue
        p = frames_path(r["id"])
        if os.path.exists(p):
            out.append((r["id"], phases_from_env(np.load(p)["rms_db"].astype(np.float32), n_phase)))
            n_approx += 1
    return out, n_approx


def _as_phases(e: np.ndarray | list, n_phase: int) -> tuple[list[np.ndarray], bool]:
    """Danh sach pha (pha 0 dau tien) -> giu nguyen; mang 1 chieu (duong bao 100 ms) -> pha xap xi (co 'xap xi')."""
    if isinstance(e, list):
        return e, False
    return phases_from_env(e, n_phase), True


def dedup(items: list[tuple[str, np.ndarray | list]], stride: int,
          against: Iterable[tuple[str, list[np.ndarray]]] = (), n_phase: int = PHASES) -> dict:
    """Cap trung trong tap cao (moi cap mot lan) + trung voi host co san (against: (ten, [duong bao/pha])).
    items: (ten, [pha 0..P-1] chinh xac) hoac (ten, duong bao 100 ms) -> pha xap xi (dem n_approx).
    Chi muc = pha 0 (buoc --stride); truy van = moi pha -> do lech bat ky (ca phan le khung) van khop."""
    idx = Index()
    conv = []
    for name, e in items:
        ph, approx = _as_phases(e, n_phase)
        conv.append((name, ph, approx))
        idx.add(name, ph[0], stride)
    idx.finish()
    ph0 = {name: ph[0] for name, ph, _ in conv}
    keys = ("votes", "offset_s", "frac", "phase", "corr")
    pairs, rej, seen = [], [], set()
    for i, (name, ph, _) in enumerate(conv):
        for m in idx.query_multi(ph, skip=i):
            key = tuple(sorted((name, m["name"])))
            if key in seen:
                continue
            seen.add(key)
            v = verify(m, ph, ph0[m["name"]])             # xac minh bang tuong quan duong bao (chong trung gia)
            (pairs if v["verified"] else rej).append({"a": name, "b": m["name"], **{k: v[k] for k in keys}})
    ext, ext_rej = [], []
    for hname, envs in against:
        envs = envs if isinstance(envs, list) else [envs]
        for m in idx.query_multi(envs):
            v = verify(m, envs, ph0[m["name"]])
            (ext if v["verified"] else ext_rej).append({"host": hname, "scraped": m["name"], **{k: v[k] for k in keys}})
    return {"n_files": len(items), "n_approx": sum(a for _, _, a in conv), "dup_pairs": pairs, "n_against": None,
            "against_hits": ext, "rejected_pairs": rej, "rejected_against": ext_rej, "verify_corr": VERIFY_CORR}


def build_env8(budget_min: float = 8.0, limit: int = 10 ** 9) -> dict:
    """frames/<id>.env8.npy cho tep cao chua co (giai ma lai bang ffmpeg, khong luu am thanh). Chay lai = tiep tuc."""
    from v6.scrape.filter import decode_blocks, env8_path
    t0, n_new, n_have, n_err = time.time(), 0, 0, 0
    for r in U.latest_by(U.read_jsonl(U.FILTERED)).values():
        if not r.get("file"):
            continue
        p8 = env8_path(r["id"])
        if os.path.exists(p8):
            n_have += 1
            continue
        if n_new >= limit or time.time() - t0 > budget_min * 60:
            print("het ngan sach / gioi han; chay lai de tiep tuc", flush=True)
            break
        fe = FR.FineEnvelope()
        try:
            for blk in decode_blocks(os.path.join(U.SCRAPE, r["file"])):
                fe.feed(blk)
        except (RuntimeError, OSError) as ex:
            n_err += 1
            print(f"  LOI {r['id']}: {str(ex)[:120]}", flush=True)
            continue
        FR.save_env8(p8, fe.finish())
        n_new += 1
        print(f"  {r['id']}: env8 {os.path.getsize(p8) / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    return {"new": n_new, "had": n_have, "errors": n_err, "seconds": round(time.time() - t0, 1)}


def wav_map(base: str) -> dict[str, str]:
    """Mot luot os.walk: ten tep (khong .wav) -> duong dan (thay glob de quy cho tung host)."""
    out: dict[str, str] = {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".wav"):
                out.setdefault(f[:-4], os.path.join(root, f))
    return out


def iter_hosts(paths: list[str], limit: int, n_phase: int = PHASES) -> Iterable[tuple[str, list[np.ndarray]]]:
    n = 0
    for p in paths:
        base = os.path.dirname(p)
        wmap = None
        for r in U.read_jsonl(p):
            if n >= limit:
                return
            wav = os.path.join(base, r["source"], r["file"] + ".wav")
            if not os.path.exists(wav):
                if wmap is None:
                    wmap = wav_map(base)
                wav = wmap.get(r["file"])
                if not wav:
                    continue
            n += 1
            yield r["file"], phases_from_audio(read16k(wav), n_phase)


def selfcheck(items: list[tuple[str, np.ndarray | list]], segs: list[dict], stride: int, n_phase: int = PHASES,
              tol_s: float = 0.25) -> dict:
    """Duong HOST: doan cat (biet tep goc + start) truy van (pha tu wav) vao chi muc tep goc -> recall dung tep + do lech."""
    idx = Index()
    ph0 = {}
    for name, e in items:
        ph0[name] = _as_phases(e, n_phase)[0][0]
        idx.add(name, ph0[name], stride)
    idx.finish()
    rows, n_hit, n_q, n_false, n_false_unver = [], 0, 0, 0, 0
    for sg in segs:
        if sg.get("id") not in ph0 or not os.path.exists(sg["path"]):
            continue
        q = phases_from_audio(read16k(sg["path"]), n_phase)
        ms = [verify(m, q, ph0[m["name"]]) for m in idx.query_multi(q)]
        top = ms[0] if ms else None
        ok = bool(top and top["name"] == sg["id"] and abs(top["offset_s"] + sg["start"]) <= tol_s and top["verified"])
        n_q += 1
        n_hit += ok
        n_false += sum(1 for m in ms if m["name"] != sg["id"] and m["verified"])      # trung SAI tep sau xac minh
        n_false_unver += sum(1 for m in ms if m["name"] != sg["id"])
        frac_ms = round((sg["start"] * 1000) % 100, 1)
        rows.append({"file": sg["file"], "id": sg["id"], "tier": sg.get("tier"), "start": sg["start"], "dur": sg.get("dur"),
                     "start_ms_mod_100": frac_ms, "hit": ok, "top": top})
    off = [r for r in rows if r["start_ms_mod_100"] not in (0.0, 100.0)]
    cs = sorted(r["top"]["corr"] for r in rows if r["top"] and r["top"]["name"] == r["id"])
    return {"n_query": n_q, "n_hit": n_hit, "recall": round(n_hit / max(1, n_q), 4),
            "n_offgrid": len(off), "recall_offgrid": round(sum(r["hit"] for r in off) / max(1, len(off)), 4),
            "n_false_matches": n_false, "n_false_before_verify": n_false_unver,
            "corr_true_min": cs[0] if cs else None, "corr_true_p01": cs[int(0.01 * len(cs))] if cs else None,
            "verify_corr": VERIFY_CORR, "stride": stride, "phases": n_phase, "tol_s": tol_s,
            "min_votes": MIN_VOTES, "min_votes_short": MIN_VOTES_SHORT, "vote_share": VOTE_SHARE, "rows": rows}


def selfcheck_files(items: list[tuple[str, np.ndarray | list]], segs: list[dict], stride: int, n_phase: int = PHASES,
                    tol_s: float = 0.25) -> dict:
    """Duong TEP-vs-TEP: moi doan cat coi nhu MOT TEP CAO KHAC (pha qua FineEnvelope, nhu env8 cua tep cao) va chay
    dedup() THAT tren {tep goc} + {doan}. Trung = co cap (doan, tep goc cua no) voi |do lech + start| <= tol_s."""
    have = {n for n, _ in items}
    q = []
    for sg in segs:
        if sg.get("id") in have and os.path.exists(sg["path"]):
            q.append((f"seg:{sg['file']}", phases_from_y(read16k(sg["path"]), n_phase), sg))
    res = dedup(list(items) + [(n, ph) for n, ph, _ in q], stride, n_phase=n_phase)
    by: dict[str, list] = defaultdict(list)
    for d in res["dup_pairs"]:
        by[d["a"]].append(d)
        by[d["b"]].append(d)
    rows, n_hit, n_false = [], 0, 0
    for name, _, sg in q:
        ok, top = False, None
        for d in by.get(name, []):
            other = d["b"] if d["a"] == name else d["a"]
            if other.startswith("seg:"):          # doan chong lan doan khac cung tep goc: khong tinh
                continue
            if other != sg["id"]:
                n_false += 1
                continue
            off = d["offset_s"] if d["a"] == name else -d["offset_s"]     # do lech nhin tu doan: ~ -start
            top = {**d, "offset_from_seg": off}
            ok = ok or abs(off + sg["start"]) <= tol_s
        n_hit += ok
        rows.append({"file": sg["file"], "id": sg["id"], "tier": sg.get("tier"), "start": sg["start"],
                     "start_ms_mod_100": round((sg["start"] * 1000) % 100, 1), "hit": ok, "pair": top})
    off = [r for r in rows if r["start_ms_mod_100"] not in (0.0, 100.0)]
    lg = [r for r in rows if r["tier"] == "long"]
    cs = sorted(r["pair"]["corr"] for r in rows if r["pair"])
    rej_src = [d for d in res["rejected_pairs"] if d["a"].startswith("seg:") != d["b"].startswith("seg:")]
    return {"n_query": len(rows), "n_hit": n_hit, "recall": round(n_hit / max(1, len(rows)), 4),
            "n_offgrid": len(off), "recall_offgrid": round(sum(r["hit"] for r in off) / max(1, len(off)), 4),
            "n_long": len(lg), "recall_long": round(sum(r["hit"] for r in lg) / max(1, len(lg)), 4),
            "n_false_matches": n_false, "n_rejected_seg_source_pairs": len(rej_src),
            "corr_true_min": cs[0] if cs else None, "verify_corr": VERIFY_CORR,
            "n_source_approx": res["n_approx"], "stride": stride, "phases": n_phase, "tol_s": tol_s, "rows": rows}


# ---------------- mau nghe ----------------
def listen_rows(frac: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    fetched = U.latest_by(U.read_jsonl(U.FETCHED))
    rows = []
    for lang in ("vi", "en"):
        hp = os.path.join(U.hosts_dir(lang), "hosts.jsonl")
        hosts = U.read_jsonl(hp)
        if hosts:
            k = max(1, int(round(frac * len(hosts))))
            for h in rng.sample(hosts, min(k, len(hosts))):
                rows.append({"lang": lang, "kind": "host", "file": h["file"], "url": h.get("url"),
                             "start": h.get("start"), "end": h.get("end"), "channel_id": h.get("channel_id"),
                             "verdict": "", "note": ""})
            continue
        for r in U.latest_by(U.read_jsonl(U.FILTERED)).values():   # chua co host: lay 1 doan 60 s / tep giu
            if r.get("lang") != lang or not r.get("ok") or not r.get("keep_spans"):
                continue
            s = rng.choice(r["keep_spans"])
            st = rng.uniform(s[0], max(s[0], s[1] - 60))
            rows.append({"lang": lang, "kind": "keep_span", "file": r["file"], "url": fetched.get(r["id"], {}).get("url"),
                         "start": round(st, 1), "end": round(min(s[1], st + 60), 1), "channel_id": r.get("channel_id"),
                         "verdict": "", "note": ""})
    return rows


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("env8")
    p.add_argument("--budget-min", type=float, default=8.0)
    p.add_argument("--limit", type=int, default=10 ** 9)
    p = sub.add_parser("dedup")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--against", action="append", default=[])
    p.add_argument("--limit-against", type=int, default=500)
    p.add_argument("--phases", type=int, default=PHASES)
    p = sub.add_parser("selfcheck")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--phases", type=int, default=PHASES)
    p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("listen")
    p.add_argument("--frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "env8":
        print("env8:", build_env8(a.budget_min, a.limit))
    elif a.cmd == "dedup":
        items, n_approx = scraped_envelopes(a.phases)
        if n_approx:
            print(f"  CANH BAO: {n_approx} tep chua co env8 -> pha xap xi (chay 'qc env8' truoc)", flush=True)
        against = [os.path.join(U.BASE, x) if not os.path.isabs(x) else x for x in a.against]
        res = dedup(items, a.stride, iter_hosts(against, a.limit_against, a.phases) if against else (), a.phases)
        res["n_against"] = min(a.limit_against, sum(len(U.read_jsonl(x)) for x in against)) if against else 0
        res["params"] = {"lag": LAG, "nbits": NBITS, "min_swing": MIN_SWING, "max_df": MAX_DF, "min_votes": MIN_VOTES, "min_votes_short": MIN_VOTES_SHORT,
                         "vote_share": VOTE_SHARE,
                         "min_vote_frac": MIN_VOTE_FRAC, "stride": a.stride, "phases": a.phases, "against": a.against}
        with open(QC_JSON, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        print(f"dedup: {res['n_files']} tep cao ({res['n_approx']} pha xap xi), {len(res['dup_pairs'])} cap trung "
              f"(+{len(res['rejected_pairs'])} ung vien bi xac minh bo); doi chieu {res['n_against']} host -> "
              f"{len(res['against_hits'])} trung (+{len(res['rejected_against'])} bi bo, corr < {VERIFY_CORR}) -> {QC_JSON}")
        for d in res["dup_pairs"][:20] + res["against_hits"][:20] + res["rejected_against"][:5]:
            print("  ", d)
    elif a.cmd == "selfcheck":
        # ke ca doan da chuyen vao _quarantine (van la doan cat that cua tep goc)
        segs = [{**sg, "path": U.seg_wav(sg) or ""} for sg in U.read_jsonl(U.SEGMENTS) if sg.get("wav")]
        if a.limit:
            segs = segs[:a.limit]
        items, n_approx = scraped_envelopes(a.phases)
        res_h = selfcheck(items, segs, a.stride, a.phases)
        res_f = selfcheck_files(items, segs, a.stride, a.phases)
        with open(SELFCHECK_JSON, "w", encoding="utf-8") as fh:
            json.dump({"host": res_h, "file": res_f, "n_source_approx": n_approx, "target_recall": 0.95}, fh,
                      ensure_ascii=False, indent=1)
        for tag, res in (("host", res_h), ("file", res_f)):
            print(f"selfcheck[{tag}] stride={a.stride} phases={a.phases}: {res['n_hit']}/{res['n_query']} = {res['recall']} "
                  f"(lech luoi 100 ms: {res['recall_offgrid']} tren {res['n_offgrid']}); so khop SAI tep: "
                  f"{res['n_false_matches']} (truoc xac minh {res.get('n_false_before_verify', '-')}); corr ban that min "
                  f"{res.get('corr_true_min')}", flush=True)
            for r in res["rows"]:
                if not r["hit"]:
                    print("   TRUOT", r["file"], r["start"], r.get("top") or r.get("pair"))
        print(f"tep goc pha xap xi: {n_approx} -> {SELFCHECK_JSON}")
    else:
        rows = listen_rows(a.frac, a.seed)
        os.makedirs(U.META, exist_ok=True)
        with open(LISTEN_CSV, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["file"])
            w.writeheader()
            w.writerows(rows)
        print(f"listen: {len(rows)} mau -> {LISTEN_CSV}")


if __name__ == "__main__":
    main()
