# -*- coding: utf-8 -*-
"""BAO CAO tong hop cac buoc cao (doc meta/*.jsonl, khong mang, khong mo hinh).

  kham pha   so kenh / tab / video phang theo ngon ngu
  giay phep  ty le qua 'Creative Commons' tren MAU NGAU NHIEN tung kenh (khong loc truoc) + ly do bi cong chan
  tai        so tep / gio / GB theo ngon ngu, theo kenh; loi tai
  probe      codec / bitrate / sr; ly do bi chan
  filter     ty le giu, pool, ly do loai, gio giu (keep_hours), vet cat / gio, so nguoi noi uoc luong
  dung duoc  THEO CHINH SACH HIEN TAI (gate.current tinh lai + filter ok + khong co co loai mac dinh): tep / gio / kenh,
             va phan dang cho nguoi dung quyet (ton giao, NC-ND, minor / child)
  hosts / lid / qc / validate_cuts / quarantine: doc hosts_stats.json, lid_calib.json, lex_calib.json (LEX_MIN CHUA hieu
             chinh), qc_selfcheck.json (2 duong), validate_cuts*.json (2 cau hinh bo do), _quarantine/moved.jsonl
Cong trong bao cao = cong TINH LAI trong bo nho (khong doc gated.jsonl cu).
Ra: data/scrape/meta/pilot_report.json + .tmp_v6/cache/scrape_pilot_report.md

CLI: python -m v6.scrape.report"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

MD = os.path.join(U.BASE, ".tmp_v6", "cache", "scrape_pilot_report.md")


def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f} %" if b else "-"


def _json(name: str) -> dict | None:
    p = os.path.join(U.META, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def usable(fetched: list[dict], gated: dict[str, dict], filt: dict[str, dict], lang: str) -> dict:
    """Tep dung duoc THEO CHINH SACH HIEN TAI + phan bi giu lai cho den khi nguoi dung quyet."""
    from v6.scrape import gate as G
    ok, held = [], defaultdict(list)
    for r in fetched:
        if r.get("lang") != lang or not r.get("ok"):
            continue
        g, f = gated.get(r["id"]) or {}, filt.get(r["id"]) or {}
        if not f.get("ok"):
            continue
        if not g.get("ok"):
            held["gate:" + ",".join(sorted({x.split(":")[0] for x in g.get("reasons") or ["no_gate_row"]}))].append(f)
            continue
        bad = sorted(set(G.EXCLUDE_FLAGS) & set(g.get("flags") or []))
        if bad:
            held["flag:" + bad[0]].append(f)
            continue
        ok.append(f)
    return {"files": len(ok), "keep_hours": _keep_h(ok), "channels": len({x.get("channel_id") for x in ok}),
            "held_back": {k: {"files": len(v), "keep_hours": _keep_h(v)} for k, v in sorted(held.items())}}


def _keep_h(rows: list[dict]) -> float:
    return round(sum(x.get("keep_hours") or 0 for x in rows), 2)


def collect() -> dict:
    from v6.scrape import gate as G
    flat = [r for r in U.read_jsonl(U.FLAT) if r.get("id")]
    cands = list(U.latest_by(U.read_jsonl(U.CANDIDATES)).values())
    gmap = G.current(verbose=False)
    gated = list(gmap.values())
    fetched = [r for r in U.latest_by(U.read_jsonl(U.FETCHED)).values()]
    probed = list(U.latest_by(U.read_jsonl(U.PROBED)).values())
    filt = list(U.latest_by(U.read_jsonl(U.FILTERED)).values())
    wl = U.load_whitelist()
    rep: dict = {"generated_at": U.now_iso()}
    for lang in ("vi", "en"):
        L: dict = {}
        fl = [r for r in flat if r.get("lang") == lang]
        L["discover"] = {"channels": len({r["channel_id"] for r in fl}), "flat_videos": len(fl),
                         "flat_hours": round(sum(r.get("duration") or 0 for r in fl) / 3600, 1)}
        cs = [c for c in cands if c.get("lang") == lang]
        err = [c for c in cs if c.get("error")]
        okm = [c for c in cs if not c.get("error")]
        gl = {g["id"]: g for g in gated}
        # YouTube: truong license tung video; nguon mo (ia / ccc / fosdem): giay phep hieu luc cua cong (licenseurl / trang)
        cc = [c for c in okm if ("creative commons" in str(c.get("license") or "").lower() if c.get("source", "yt") == "yt"
                                 else (gl.get(c["id"]) or {}).get("license_effective") is not None)]
        by_ch = defaultdict(lambda: [0, 0])
        for c in okm:
            by_ch[c["channel_id"]][1] += 1
            by_ch[c["channel_id"]][0] += int(c in cc)
        pol = defaultdict(lambda: [0, 0])     # qua A1 THEO CHINH SACH (khong ly do giay phep nao) theo nguon
        for c in okm:
            g = gl.get(c["id"]) or {}
            pol[c.get("source", "yt")][1] += 1
            pol[c.get("source", "yt")][0] += int(not set(x.split(":")[0] for x in g.get("reasons") or [])
                                                 & set(G.LICENSE_REASONS))
        L["license"] = {"sampled": len(cs), "metadata_errors": len(err), "cc": len(cc),
                        "cc_rate": round(len(cc) / len(okm), 3) if okm else None,
                        "policy_pass_by_source": {s: f"{a}/{b}" for s, (a, b) in sorted(pol.items())},
                        "error_examples": Counter(str(c["error"])[:70] for c in err).most_common(3),
                        "by_channel": {f"{(wl.get(k) or {}).get('channel_name', k)} [{k}]": f"{v[0]}/{v[1]}"
                                       for k, v in sorted(by_ch.items())}}
        gs = [g for g in gated if g.get("lang") == lang]
        why = Counter(x.split(":")[0] for g in gs for x in g["reasons"])
        L["gate"] = {"n": len(gs), "ok": sum(g["ok"] for g in gs), "ok_hours": round(sum((g.get("duration") or 0)
                     for g in gs if g["ok"]) / 3600, 1), "reasons": dict(why.most_common()),
                     "flags": dict(Counter(f for g in gs for f in g.get("flags") or []))}
        fe = [r for r in fetched if r.get("lang") == lang]
        fok = [r for r in fe if r.get("ok")]
        L["fetch"] = {"ok": len(fok), "failed": len(fe) - len(fok), "channels": len({r["channel_id"] for r in fok}),
                      "hours": round(sum(r.get("duration") or 0 for r in fok) / 3600, 2),
                      "gb": round(sum(r.get("bytes") or 0 for r in fok) / 1e9, 3),
                      "formats": dict(Counter(str(r.get("format_id")) for r in fok)),
                      "sources": dict(Counter(str(r.get("source", "yt")) for r in fok)),
                      "hours_by_source": {s: round(sum(r.get("duration") or 0 for r in fok if r.get("source", "yt") == s)
                                                   / 3600, 2) for s in {r.get("source", "yt") for r in fok}},
                      "errors": Counter(str(r.get("error"))[:80] for r in fe if not r.get("ok")).most_common(3)}
        pr = [r for r in probed if r.get("lang") == lang]
        kb = [r["bitrate_kbps"] for r in pr if r.get("bitrate_kbps")]
        L["probe"] = {"n": len(pr), "ok": sum(r["ok"] for r in pr), "kenh": dict(Counter(r.get("kenh") for r in pr)),
                      "sr": dict(Counter(r.get("sr") for r in pr)),
                      "kbps_min_med_max": [round(float(np.min(kb))), round(float(np.median(kb))), round(float(np.max(kb)))]
                      if kb else None, "reasons": dict(Counter(x for r in pr for x in r["reasons"]))}
        ft = [r for r in filt if r.get("lang") == lang]
        fdur = sum(r.get("dur_s") or 0 for r in ft) / 3600
        L["filter"] = {"n": len(ft), "ok": sum(r["ok"] for r in ft), "hours_in": round(fdur, 2),
                       "keep_hours": round(sum(r.get("keep_hours") or 0 for r in ft), 2),
                       "pools": dict(Counter(r.get("pool") for r in ft if r["ok"])),
                       "reasons": dict(Counter(x for r in ft for x in r.get("reasons") or [])),
                       "cut_mode": dict(Counter(r.get("cut_mode") for r in ft)),
                       "cut_types": dict(Counter(e["type"] for r in ft for e in r.get("cuts") or [])),
                       "cuts_per_hour_median": float(np.median([r["cuts_per_hour"] for r in ft if "cuts_per_hour" in r]))
                       if ft else None,
                       "speech_ratio_median": float(np.median([r["speech_ratio"] for r in ft if "speech_ratio" in r]))
                       if ft else None,
                       "bw_p90_median": float(np.median([r["bw_p90"] for r in ft if "bw_p90" in r])) if ft else None,
                       "n_spk": dict(Counter(r.get("n_spk_est") for r in ft)),
                       "conf_call": sum(bool(r.get("conf_call")) for r in ft)}
        L["usable_current_policy"] = usable(fetched, gmap, {r["id"]: r for r in filt}, lang)
        rep[lang] = L
    old = U.latest_by(U.read_jsonl(U.GATED))
    rep["gate_file_stale"] = sum(1 for k, g in gmap.items() if k in old and bool(old[k].get("ok")) != bool(g["ok"]))
    rep["hosts"] = _json("hosts_stats.json")
    rep["lid_calib"] = {k: v for k, v in (_json("lid_calib.json") or {}).items() if k != "rows"} or None
    lex = _json("lex_calib.json")
    rep["lex_calib"] = {**(lex or {}), "calibrated": False,
                        "note": "LEX_MIN 0,75 dat bang mat; khong dat B-12 -> chi tin hieu phu khi chua co LID"}
    lid_rows = list(U.latest_by(U.read_jsonl(os.path.join(U.META, "lid.jsonl")), "file").values())
    from v6.scrape.lid import LID_MIN, lid_min
    rep["lid"] = {"n": len(lid_rows), "thr": LID_MIN,
                  "below_thr": dict(Counter(r.get("lang") for r in lid_rows
                                            if (r.get("p_target") or 0) < lid_min(r.get("lang")))),
                  "below_0.9": dict(Counter(r.get("lang") for r in lid_rows if (r.get("p_target") or 0) < 0.9)),
                  "top": dict(Counter(f"{r.get('lang')}->{r.get('top')}" for r in lid_rows))}
    sc = _json("qc_selfcheck.json") or {}
    rep["qc_selfcheck"] = {k: {kk: v.get(kk) for kk in ("n_query", "n_hit", "recall", "recall_offgrid", "recall_long",
                                                         "n_false_matches")}
                           for k, v in sc.items() if isinstance(v, dict) and "recall" in v}
    spk = _json("spk_groups.json")
    rep["spk_groups"] = {k: spk.get(k) for k in ("n_channels", "n_merged_channels", "map", "review_pairs", "params")} if spk else None
    rep["quarantine_moved"] = len(U.read_jsonl(os.path.join(U.QUARANTINE, "moved.jsonl")))
    for key, name in (("validate_cuts", "validate_cuts.json"), ("validate_cuts_synthetic", "validate_cuts_synthetic.json"),
                      ("qc_dedup", "qc_dedup.json")):
        p = os.path.join(U.META, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
            rep[key] = d.get("summary") if "summary" in d else {"n_files": d.get("n_files"),
                                                                 "dup_pairs": d.get("dup_pairs"),
                                                                 "n_against": d.get("n_against"),
                                                                 "against_hits": len(d.get("against_hits") or [])}
    return rep


def to_md(rep: dict) -> str:
    out = [f"# Pilot cao du lieu (v6.scrape) - {rep['generated_at']}", ""]
    for lang in ("vi", "en"):
        L = rep[lang]
        d, li, g, fe, pr, ft = (L[k] for k in ("discover", "license", "gate", "fetch", "probe", "filter"))
        out += [f"## {lang}", "",
                f"- kham pha: {d['channels']} kenh, {d['flat_videos']} video phang, {d['flat_hours']} h (moi tab <= 60 muc)",
                f"- giay phep (mau ngau nhien khong loc truoc): {li['cc']}/{li['sampled'] - li['metadata_errors']} co truong "
                f"CC / giay phep nhan biet = {li['cc_rate']}; loi metadata {li['metadata_errors']}; qua A1 theo chinh sach "
                f"hien tai (BY / BY-SA / CC0, khong NC / ND / dang lai) theo nguon: {li['policy_pass_by_source']}",
                f"- cong metadata: {g['ok']}/{g['n']} qua ({g['ok_hours']} h); ly do: {g['reasons']}; co: {g['flags']}",
                f"- tai: {fe['ok']} tep, {fe['channels']} kenh, {fe['hours']} h, {fe['gb']} GB, dinh dang {fe['formats']}; "
                f"loi {fe['failed']} {fe['errors']}",
                f"- probe: {pr['ok']}/{pr['n']} qua; kenh {pr['kenh']}; sr {pr['sr']}; kbps min/trung vi/max "
                f"{pr['kbps_min_med_max']}; ly do {pr['reasons']}",
                f"- filter: {ft['ok']}/{ft['n']} giu; vao {ft['hours_in']} h -> giu {ft['keep_hours']} h; pool {ft['pools']}",
                f"  - ly do loai: {ft['reasons']}; che do cat {ft['cut_mode']}; loai vet {ft['cut_types']}",
                f"  - trung vi: vet/gio {ft['cuts_per_hour_median']}, ty le loi {ft['speech_ratio_median']}, "
                f"bw_p90 {ft['bw_p90_median']} Hz; so nguoi uoc {ft['n_spk']}; conf_call {ft['conf_call']}",
                "", "| kenh | CC / mau |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in li["by_channel"].items()]
        us = L.get("usable_current_policy") or {}
        out += ["", f"- nguon da tai: {fe.get('sources')} gio {fe.get('hours_by_source')}",
                f"- DUNG DUOC theo chinh sach hien tai: {us.get('files')} tep, {us.get('keep_hours')} h giu, "
                f"{us.get('channels')} kenh; giu lai cho nguoi dung quyet: {us.get('held_back')}", ""]
    out += [f"- gated.jsonl lech cong hien tai: {rep.get('gate_file_stale')} ung vien (0 = da chay lai gate)",
            f"- tep da cach ly (_quarantine): {rep.get('quarantine_moved')}", ""]
    for key in ("hosts", "lid", "lid_calib", "lex_calib", "qc_selfcheck", "spk_groups", "validate_cuts",
                "validate_cuts_synthetic", "qc_dedup"):
        if rep.get(key) is not None:
            out += [f"## {key}", "", "```", json.dumps(rep[key], ensure_ascii=False, indent=1)[:3500], "```", ""]
    return "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rep = collect()
    with open(os.path.join(U.META, "pilot_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    md = to_md(rep)
    os.makedirs(os.path.dirname(MD), exist_ok=True)
    with open(MD, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
