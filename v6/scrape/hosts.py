# -*- coding: utf-8 -*-
"""GHI HOSTS cao (giao dien I4) - gop segments + aligned + fetched + probed + filtered thanh hosts.jsonl.

  data/scrape/hosts_<lang>/hosts.jsonl        doan chuan 5-20 s   (wav: data/scrape/hosts_<lang>/<source>/<file>.wav)
  data/scrape/hosts_long_<lang>/hosts.jsonl   doan dai 30-120 s
Moi dong: cac truong cua data/natural_v6/hosts*/hosts.jsonl (file, source, cluster, person, dur, text, words)
  + nguon goc {url, channel_id, license, codec, bitrate, orig_file, group} + {video_id, start, end, tier, lang, kenh,
  pool, conf_call, flags (minor_risk...), title, channel, upload_date, retrieved_at}.
  cluster = '<source>:<channel_id>:<video>' (ban ghi) · person = group = nhom chong ro ri:
  v6.common.group_of (giao dien I2, neu da co) roi '<source>:<channel_id>' (kenh = nguoi cho den khi gom cum giong noi).
Loc (FAIL CLOSED - review 12/10):
  - video phai qua CONG HIEN TAI (v6.scrape.gate.current: tinh lai trong bo nho tu candidates.jsonl, khong tin gated.jsonl
    cu); doan khong co dong cong -> bo ('khong_co_cong'); wav phai ton tai ('thieu_wav');
  - >= 8 tu (doan chuan; doan dai >= 40 tu), 1,0 <= tu/giay <= 7,5 (ASR hop ly);
  - NGON NGU (B-12 'LID >= 0,9'): p(ngon ngu dich) cua Whisper LID dong bang (v6.scrape.lid -> meta/lid.jsonl)
    >= lid.LID_MIN[lang] (vi 0,9 / en 0,5: tieng Anh giong la VoxPopuli / EdAcc cho p_en thap, lid_calib 12/10);
    (17/10: thu luat bo sung cap video bang lex_share cho en - KHONG dung, am tieng Viet qua ASR tieng Anh chong lan
    duong; xem v6.scrape.lex_video va meta/lex_video_calib.json; nguoi dung chon giu 0,5);
    doan chua co LID -> bo ('chua_lid') tru khi --no-require-lid (khi do dung proxy lex_share >= LEX_MIN, CHUA hieu chinh:
    review 12/10 do 32 % loi tieng Anh qua ASR vi van >= 0,75 va 31 % host tieng Anh that < 0,75 -> chi la tin hieu phu,
    luon ghi lex_share vao dong).
  Tran: <= --max-hours-group gio / nhom (mac dinh 3 h, filter_spec 7) - chon rai deu theo video.
Loai mac dinh (cho den khi nguoi dung xac nhan, quyet dinh 1): video co co gate.EXCLUDE_FLAGS ('religious_teaching',
  'minor_risk', 'child_risk'); mo lai tung co bang --allow-flag <co>.
Ghi lai toan bo (nguyen tu) moi lan chay: hosts.jsonl la ham cua cac tep meta.
calib-lex: hieu chinh nguong proxy ngon ngu tren du lieu NGOAI tap loc: duong = nua giu lai cua host tu nhien cung ngon ngu
  (tu vung dung tu nua kia), am = host tu nhien ngon ngu kia + doan ASR cua kenh da biet lech ngon ngu (PANTADO, ASR vi
  tren tieng Anh) -> meta/lex_calib.json (min/p01/p05 duong, max/p99 am, bien).

CLI: python -m v6.scrape.hosts [--lang vi|en] [--max-hours-group 3] [--allow-flag religious_teaching ...] [--no-require-lid]
     python -m v6.scrape.hosts --calib-lex [--calib-out F]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Mapping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import gate as G  # noqa: E402
from v6.scrape import util as U  # noqa: E402

MIN_WORDS = {"std": 8, "long": 40}
WPS = (1.0, 7.5)
# Proxy nhan dang ngon ngu (tin hieu PHU): ty le tu ASR nam trong tu vung >= LEX_MIN_COUNT lan cua host tu nhien cung ngon
# ngu (data/natural_v6/hosts*/hosts.jsonl). CHUA hieu chinh (calib-lex 12/10: khong dat B-12) -> chi dung khi khong co LID.
LEX_MIN, LEX_MIN_COUNT = 0.75, 3
LEX_CALIBRATED = False
from v6.scrape.lid import LID_MIN, lid_min  # noqa: E402  (nguong theo ngon ngu: vi 0,9 / en 0,5, xem lid_calib.json)
LEX_SRC = {"vi": os.path.join(U.BASE, "data", "natural_v6", "hosts", "hosts.jsonl"),
           "en": os.path.join(U.BASE, "data", "natural_v6", "hosts_en", "hosts.jsonl")}
_VOCAB: dict[str, set] = {}
EXCLUDE_FLAGS = G.EXCLUDE_FLAGS
LID_JSONL = os.path.join(U.META, "lid.jsonl")
OFFLANG_CHANNELS = {"vi": {"UC2JPCC4OKl5oQUeiLBFR--w"}}   # am that: kenh 'vi' noi tieng Anh (PANTADO)
LEX_CALIB = os.path.join(U.META, "lex_calib.json")


def vocab_from_texts(texts: list[str], min_count: int = LEX_MIN_COUNT) -> set:
    from collections import Counter
    c: Counter = Counter()
    for t in texts:
        c.update(str(t or "").lower().split())
    return {w for w, n in c.items() if n >= min_count}


def lex_vocab(lang: str, path: str | None = None) -> set:
    """Tu vung tham chieu (chu thuong, tu xuat hien >= LEX_MIN_COUNT lan). Khong co tep -> tap rong (bo qua loc)."""
    if lang not in _VOCAB:
        _VOCAB[lang] = vocab_from_texts([r.get("text") for r in U.read_jsonl(path or LEX_SRC.get(lang, ""))])
    return _VOCAB[lang]


def lex_share(words: list, vocab: set) -> float | None:
    toks = [str(w[0] if isinstance(w, (list, tuple)) else w).lower() for w in words]
    if not toks or not vocab:
        return None
    return sum(t in vocab for t in toks) / len(toks)


def group_for(source: str, channel_id: str, set_name: str, spk_map: Mapping[str, str] | None = None) -> str:
    """Nhom chong ro ri: kenh da hop qua gom cum giong noi (spk_map tu v6.scrape.spk_cluster, meta['group']) ->
    I2 (v6.common.group_of) neu co, khong thi '<source>:<channel_id>'."""
    meta = {"source": f"scrape_{source}", "person": f"{source}:{channel_id}", "channel_id": channel_id,
            "channel": channel_id, "scraped": True}
    g0 = (spk_map or {}).get(f"{source}:{channel_id}")
    if g0:
        meta["group"] = g0
    try:
        from v6.common import group_of   # giao dien I2 (WP5 viet song song) - khong bat buoc
        g = group_of(meta, set_name)
        if g:
            return str(g)
    except Exception:  # noqa: BLE001  (chua co / loi -> mac dinh kenh)
        pass
    return g0 or f"{source}:{channel_id}"


def build_rows(segs: list[Mapping[str, Any]], aligned: Mapping[str, Mapping[str, Any]],
               fetched: Mapping[str, Mapping[str, Any]], probed: Mapping[str, Mapping[str, Any]],
               filtered: Mapping[str, Mapping[str, Any]], gated: Mapping[str, Mapping[str, Any]],
               lang: str, tier: str, allow_flags: tuple = (), lid: Mapping[str, Mapping[str, Any]] | None = None,
               require_lid: bool = True, check_wav: bool = True,
               spk_map: Mapping[str, str] | None = None) -> tuple[list[dict], dict]:
    """Ham thuan (tru kiem wav ton tai). gated = cong HIEN TAI {id: ket qua gate_record}; thieu dong -> bo (fail closed)."""
    rows, why = [], defaultdict(int)
    excl = set(EXCLUDE_FLAGS) - set(allow_flags)
    vocab = lex_vocab(lang)
    lid = lid or {}
    for s in segs:
        if not s.get("file") or s.get("lang") != lang or s.get("tier") != tier:
            continue
        gt = gated.get(s["id"])
        if gt is None:                                   # khong co dong cong -> khong chung minh duoc giay phep
            why["khong_co_cong"] += 1
            continue
        if not gt.get("ok"):                             # cong hien tai bo (vd chinh sach giay phep / kenh doi)
            why["cong_loai"] += 1
            continue
        bad = excl & set(gt.get("flags") or []) | (excl & set((filtered.get(s["id"]) or {}).get("flags") or []))
        if bad:
            why["co_loai:" + sorted(bad)[0]] += 1
            continue
        if check_wav and not (s.get("wav") and os.path.isfile(os.path.join(U.SCRAPE, s["wav"]))):
            why["thieu_wav"] += 1                        # vd da chuyen vao _quarantine
            continue
        al = aligned.get(s["file"])
        if al is None:
            why["chua_align"] += 1
            continue
        if al["n_words"] < MIN_WORDS[tier]:
            why["it_tu"] += 1
            continue
        if not WPS[0] <= al["wps"] <= WPS[1]:
            why["tu_giay_bat_thuong"] += 1
            continue
        lx = lex_share(al["words"], vocab)
        lr = lid.get(s["file"])
        if lr is not None:
            if lr.get("p_target") is None or lr["p_target"] < lid_min(lang):
                why["lid_lech"] += 1
                continue
            check = "lid"
        elif require_lid:
            why["chua_lid"] += 1
            continue
        else:
            if lx is not None and lx < LEX_MIN:
                why["lech_ngon_ngu"] += 1
                continue
            check = "lex_proxy_uncalibrated"
        f, p, fl, g = (d.get(s["id"], {}) for d in (fetched, probed, filtered, gated))
        src, cid = s.get("source", "yt"), s.get("channel_id") or f.get("channel_id")
        grp = group_for(src, cid, f"scrape{lang}", spk_map)
        rows.append({
            "file": s["file"], "source": src, "cluster": f"{src}:{cid}:{s['id']}", "person": grp, "dur": s["dur"],
            "text": al["text"], "words": al["words"],
            "url": f.get("url"), "channel_id": cid,
            "license": g.get("license_effective") or f.get("license_effective") or f.get("license"),
            "codec": p.get("codec"), "bitrate": p.get("bitrate_kbps"), "orig_file": f.get("file"), "group": grp,
            "video_id": s["id"], "start": s["start"], "end": s["end"], "tier": tier, "lang": lang,
            "kenh": p.get("kenh"), "pool": fl.get("pool"), "conf_call": fl.get("conf_call"),
            "flags": sorted(set((g.get("flags") or []) + (fl.get("flags") or []))), "title": f.get("title"),
            "channel": f.get("channel"), "upload_date": f.get("upload_date"), "retrieved_at": f.get("retrieved_at"),
            "asr": al.get("asr"), "lex_share": round(lx, 3) if lx is not None else None,
            "lang_check": check, "lid_p": (lr or {}).get("p_target"), "lid_top": (lr or {}).get("top"),
            "lid_model": (lr or {}).get("model")})
    return rows, dict(why)


def cap_groups(rows: list[dict], max_hours: float, seed: int = 0) -> list[dict]:
    """<= max_hours gio moi nhom; lay vong tron theo video de rai deu."""
    by_g: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_g[r["group"]][r["video_id"]].append(r)
    rng = random.Random(seed)
    out = []
    for g in sorted(by_g):
        vids = {v: sorted(rs, key=lambda r: r["start"]) for v, rs in by_g[g].items()}
        order = sorted(vids)
        rng.shuffle(order)
        tot, k = 0.0, 0
        while any(vids.values()) and tot < max_hours * 3600:
            v = order[k % len(order)]
            k += 1
            if not vids[v]:
                continue
            r = vids[v].pop(0)
            if tot + r["dur"] > max_hours * 3600:
                break
            out.append(r)
            tot += r["dur"]
    return sorted(out, key=lambda r: (r["group"], r["video_id"], r["start"]))


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3)


def calib_lex(seed: int = 0, max_n: int = 4000, out_path: str = LEX_CALIB) -> dict:
    """Hieu chinh LEX_MIN tren du lieu khong dung de chon nguong (xem docstring module)."""
    rng = random.Random(seed)
    nat = {}
    for lang, p in LEX_SRC.items():
        rows = [r for r in U.read_jsonl(p) if len(str(r.get("text") or "").split()) >= MIN_WORDS["std"]]
        rng.shuffle(rows)
        nat[lang] = rows[:2 * max_n]
    segs = {s["file"]: s for s in U.read_jsonl(U.SEGMENTS)}
    out = {"lex_min": LEX_MIN, "min_count": LEX_MIN_COUNT}
    for lang, other in (("vi", "en"), ("en", "vi")):
        rows = nat[lang]
        half = len(rows) // 2
        voc = vocab_from_texts([r.get("text") for r in rows[:half]])
        pos = [lex_share(str(r["text"]).split(), voc) for r in rows[half:]]
        neg_other = [lex_share(str(r["text"]).split(), voc) for r in nat[other][:max_n]]
        neg_real = []
        for al in U.latest_by(U.read_jsonl(U.ALIGNED), "file").values():
            sg = segs.get(al["file"]) or {}
            if al.get("lang", sg.get("lang")) == lang and sg.get("channel_id") in OFFLANG_CHANNELS.get(lang, ()):
                neg_real.append(lex_share(al["words"], voc))
        pos = [x for x in pos if x is not None]
        neg_other = [x for x in neg_other if x is not None]
        neg_real = [x for x in neg_real if x is not None]
        scraped = [r["lex_share"] for r in U.read_jsonl(os.path.join(U.hosts_dir(lang), "hosts.jsonl"))
                   if r.get("lex_share") is not None]
        out[lang] = {"n_pos": len(pos), "pos_min": _q(pos, 0), "pos_p01": _q(pos, 0.01), "pos_p05": _q(pos, 0.05),
                     "pos_frac_below_lex_min": round(sum(x < LEX_MIN for x in pos) / max(1, len(pos)), 4),
                     "n_neg_other_lang": len(neg_other), "neg_other_max": _q(neg_other, 1.0),
                     "neg_other_p99": _q(neg_other, 0.99),
                     "neg_other_frac_above_lex_min": round(sum(x >= LEX_MIN for x in neg_other) / max(1, len(neg_other)), 4),
                     "n_neg_offlang_asr": len(neg_real), "neg_offlang_asr_max": _q(neg_real, 1.0),
                     "neg_offlang_asr_median": _q(neg_real, 0.5),
                     "neg_offlang_frac_above_lex_min": round(sum(x >= LEX_MIN for x in neg_real) / max(1, len(neg_real)), 4),
                     "scraped_hosts_min": _q(scraped, 0), "n_scraped_hosts": len(scraped)}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


def load_lid(path: str = LID_JSONL) -> dict[str, dict]:
    return U.latest_by(U.read_jsonl(path), "file")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"], action="append")
    ap.add_argument("--max-hours-group", type=float, default=3.0)
    ap.add_argument("--allow-flag", action="append", choices=EXCLUDE_FLAGS, default=[],
                    help="mo lai co bi loai mac dinh - CHI khi nguoi dung da xac nhan")
    ap.add_argument("--calib-lex", action="store_true")
    ap.add_argument("--calib-out", default=LEX_CALIB)
    ap.add_argument("--no-require-lid", action="store_true",
                    help="doan chua co LID dung proxy lex_share (CHUA hieu chinh) thay vi bi bo")
    a = ap.parse_args()
    if a.calib_lex:
        res = calib_lex(out_path=a.calib_out)
        for k in ("vi", "en"):
            print(k, res[k])
        print("->", a.calib_out)
        return
    segs = U.read_jsonl(U.SEGMENTS)
    aligned = U.latest_by(U.read_jsonl(U.ALIGNED), "file")
    tabs = [U.latest_by(U.read_jsonl(p)) for p in (U.FETCHED, U.PROBED, U.FILTERED)] + [G.current()]
    lid = load_lid()
    from v6.scrape.spk_cluster import load_map
    spk_map = load_map()                     # kenh da hop do cung nguoi noi (rong neu chua chay spk_cluster)
    stats = {"n_spk_merged_channels": len(spk_map)}
    for lang in a.lang or ["vi", "en"]:
        for tier in ("std", "long"):
            rows, why = build_rows(segs, aligned, *tabs, lang=lang, tier=tier, allow_flags=tuple(a.allow_flag), lid=lid,
                                   require_lid=not a.no_require_lid, spk_map=spk_map)
            rows = cap_groups(rows, a.max_hours_group)
            out = os.path.join(U.hosts_dir(lang, tier), "hosts.jsonl")
            U.write_jsonl(out, rows)
            hrs = sum(r["dur"] for r in rows) / 3600
            stats[f"{lang}_{tier}"] = {"n": len(rows), "hours": round(hrs, 3), "groups": len({r["group"] for r in rows}),
                                       "videos": len({r["video_id"] for r in rows}), "dropped": why}
            print(f"{lang} {tier:4s}: {len(rows)} host, {hrs:.2f} h, {len({r['group'] for r in rows})} nhom, "
                  f"{len({r['video_id'] for r in rows})} video | bo: {why} -> {out}")
    with open(os.path.join(U.META, "hosts_stats.json"), "w", encoding="utf-8") as fh:
        json.dump({"generated_at": U.now_iso(), "allow_flags": a.allow_flag, "require_lid": not a.no_require_lid,
                   "lid_min": LID_MIN, "lex_min": LEX_MIN, "lex_calibrated": LEX_CALIBRATED, **stats}, fh,
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
