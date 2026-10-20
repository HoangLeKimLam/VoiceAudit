# -*- coding: utf-8 -*-
"""CACH LY du lieu cao (CHUYEN, KHONG XOA - quy tac 'never delete data'): tep cua mot kenh bi loai SAU khi da tai
(vd PANTADO: lop tieng Anh 1-1 cho TRE EM, co giong tre em) ra khoi moi cho co the bi doc nham.

  Chuyen data/scrape/<rel> -> data/scrape/_quarantine/<rel> (giu nguyen duong dan tuong doi):
    raw/<nguon>/<kenh>/ (ca thu muc: tep goc + .info.json) · hosts*_<lang>/<nguon>/<file>.wav (doan da cat cua video kenh)
    · frames/<id>.npz + frames/<id>.env8.npy
  --extra P (lap lai): chuyen them tep / thu muc ngoai data/scrape (vd manifest smoke cu) -> <thu muc cha>/_quarantine/<ten>
  Dong meta (segments / aligned / fetched ...) GIU NGUYEN (lich su, nguon goc); segment / lid / hosts tu loai qua cong
  hien tai + 'thieu_wav'; util.seg_wav van tim duoc wav trong _quarantine cho kiem tra (qc selfcheck, lid --calib).
  Nhat ky: data/scrape/_quarantine/moved.jsonl {src, dst, reason, channel_id, at}. Chay lai = bo qua tep da chuyen.

CLI: python -m v6.scrape.quarantine --channel UC2JPCC4OKl5oQUeiLBFR--w --reason children_english_lessons
                                    [--extra .tmp_v6/cache/scrape_smoke_sets/scrapevi] [--dry-run]"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from typing import Any, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402


def plan_moves(channel_id: str, fetched: Iterable[Mapping[str, Any]], segments: Iterable[Mapping[str, Any]],
               scrape_root: str = U.SCRAPE, frames_of=None) -> list[tuple[str, str]]:
    """(nguon, dich) cho moi tep cua kenh (chi tep DANG ton tai). frames_of(id) -> [duong dan frames] (mac dinh filter)."""
    if frames_of is None:
        from v6.scrape.filter import env8_path, frames_path

        def frames_of(rid: str) -> list[str]:
            return [frames_path(rid), env8_path(rid)]
    qroot = os.path.join(scrape_root, "_quarantine")
    ids, dirs, moves = set(), set(), []
    for r in fetched:
        if r.get("channel_id") != channel_id:
            continue
        ids.add(r.get("id"))
        if r.get("file"):
            dirs.add(os.path.dirname(r["file"]))           # raw/<nguon>/<kenh>
    rels = sorted(d for d in dirs if d) + sorted(s["wav"] for s in segments
                                                 if s.get("wav") and (s.get("id") in ids or s.get("channel_id") == channel_id))
    for rel in rels:
        src = os.path.join(scrape_root, rel)
        if os.path.exists(src):
            moves.append((src, os.path.join(qroot, rel)))
    for rid in sorted(i for i in ids if i):
        for p in frames_of(rid):
            if os.path.exists(p):
                moves.append((p, os.path.join(qroot, os.path.relpath(p, scrape_root))))
    return moves


def extra_moves(paths: Iterable[str]) -> list[tuple[str, str]]:
    out = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.exists(p):
            out.append((p, os.path.join(os.path.dirname(p), "_quarantine", os.path.basename(p))))
    return out


def apply_moves(moves: list[tuple[str, str]], reason: str, channel_id: str, log_path: str) -> int:
    """Chuyen tung muc (dich da ton tai -> bo qua, khong ghi de); ghi nhat ky jsonl. Tra so muc da chuyen."""
    n = 0
    for src, dst in moves:
        if os.path.exists(dst):
            print(f"  bo qua (dich da co): {dst}", flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        U.append_jsonl(log_path, [{"src": src, "dst": dst, "reason": reason, "channel_id": channel_id, "at": U.now_iso()}])
        n += 1
    return n


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--channel", required=True)
    ap.add_argument("--reason", required=True)
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    fetched = list(U.latest_by(U.read_jsonl(U.FETCHED)).values())
    moves = plan_moves(a.channel, fetched, U.read_jsonl(U.SEGMENTS))
    moves += extra_moves(os.path.join(U.BASE, x) if not os.path.isabs(x) else x for x in a.extra)
    for s, d in moves:
        print(f"  {os.path.relpath(s, U.BASE)} -> {os.path.relpath(d, U.BASE)}")
    if a.dry_run:
        print(f"(dry-run) {len(moves)} muc")
        return
    n = apply_moves(moves, a.reason, a.channel, os.path.join(U.QUARANTINE, "moved.jsonl"))
    print(f"quarantine: chuyen {n}/{len(moves)} muc -> {U.QUARANTINE} (nhat ky moved.jsonl)")


# ---------------- kiem thu ----------------
def test_plan_and_apply(tmp: str | None = None) -> None:
    root = tmp or os.path.join(U.SCRAPE, "_selftest_quar")
    shutil.rmtree(root, ignore_errors=True)
    try:
        for rel in ("raw/yt/UCbad/v1.f251.webm", "raw/yt/UCbad/v1.info.json", "raw/yt/UCok/v2.f251.webm",
                    "hosts_vi/yt/yt_v1_s000.wav", "hosts_vi/yt/yt_v2_s000.wav", "frames/v1.npz"):
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            open(os.path.join(root, rel), "w").close()
        fetched = [{"id": "v1", "channel_id": "UCbad", "file": "raw/yt/UCbad/v1.f251.webm"},
                   {"id": "v2", "channel_id": "UCok", "file": "raw/yt/UCok/v2.f251.webm"}]
        segs = [{"id": "v1", "channel_id": "UCbad", "wav": "hosts_vi/yt/yt_v1_s000.wav"},
                {"id": "v2", "channel_id": "UCok", "wav": "hosts_vi/yt/yt_v2_s000.wav"}]
        mv = plan_moves("UCbad", fetched, segs, root, frames_of=lambda rid: [os.path.join(root, "frames", rid + ".npz")])
        rels = sorted(os.path.relpath(s, root).replace("\\", "/") for s, _ in mv)
        assert rels == ["frames/v1.npz", "hosts_vi/yt/yt_v1_s000.wav", "raw/yt/UCbad"], rels
        n = apply_moves(mv, "test", "UCbad", os.path.join(root, "_quarantine", "moved.jsonl"))
        assert n == 3 and not os.path.exists(os.path.join(root, "raw/yt/UCbad"))
        assert os.path.exists(os.path.join(root, "_quarantine/raw/yt/UCbad/v1.info.json"))
        assert os.path.exists(os.path.join(root, "raw/yt/UCok/v2.f251.webm"))        # kenh khac khong dong toi
        assert len(U.read_jsonl(os.path.join(root, "_quarantine", "moved.jsonl"))) == 3
        assert plan_moves("UCbad", fetched, segs, root, frames_of=lambda rid: []) == []   # chay lai: khong con gi
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        test_plan_and_apply()
        print("PASS test_plan_and_apply")
    else:
        main()
