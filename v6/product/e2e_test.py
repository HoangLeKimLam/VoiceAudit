# -*- coding: utf-8 -*-
"""KIEM THU DAU-CUOI san pham v9 o CAU HINH PHAT HANH va NGOAI MAU: chay voiceaudit_v9.run tren mau tep tu nhien (natvi / naten, khong lockbox).
  - K1 mo rong (k1x) BAT nhu CLI; ngon ngu do LID cua K1x chon (KHONG bao truoc); --oracle-lang: dua ngon ngu dung (de so sanh, ra tep _oracle)
  - diem van hanh mac dinh F (fuse1, thresholds_v9F.json); bound / reg / heads lay theo params cua tep nguong (Product tu choi neu khong khop);
    strict (mac dinh): checkpoint / nguon hieu chinh khong kiem duoc sha1 -> dung han (--allow-unverified de bo qua, ghi trong bao cao)
  - DINH TUYEN FOLD (mac dinh): moi tep cham bang mo hinh KHONG thay fold nguoi noi cua no (detect.person_fold -> Product fold=k: tcn_<reg>_f{k}, ft_ft2_f{k},
    heads_final_<heads>_f{k} + nguong thr_fold[k]); --final: mo hinh cuoi (TRONG MAU: cac tep mau nam trong tap hoc cua mo hinh cuoi - chi de doi chieu)
  - MAU PHAN TANG theo (bo, kieu, nguon), chia vong tron giua cac nguon:
      tang 'goc'     : tep goc; toi da --n tep moi kieu co thao tac (gom giong may may_thay / may_chen / may_ca_file), toi da --n-real tep moi kieu tep THAT
      tang 'app_that': tep THAT da qua kenh ung dung (Zalo / WhatsApp-Telegram / goi ghi lai / VoLTE; khu on mac dinh cua ung dung = 01, quyet dinh 4),
                       toi da --n-app tep moi (bo, kenh) - bao RIENG, khong vao chi tieu chinh
    KHONG loc mau theo quyet dinh K1 da luu trong manifest: K1x CHAY THAT quyet dinh (--k1-stored-filter = cach cu, loc truoc)
Do theo dung hop dong (6Muc): chi cham tep KHONG bi Constraint chuyen 04 / tu choi (K1 04, dieu kien hoc 04 cua Product, tu_choi); 04 do PHAN TICH (luat Scope 5,
vung xam 0,4-0,6) VAN duoc cham: Req 1 macro-F1 01/02/03 (04 = bo sot nhan dung), Req 2 (tep sua ra 04 = khong bat), Req 3, Req 4 (tep that ra 04 van o mau so),
Req 9 = 04 phan tich / tep duoc cham; so va ti le tep bi loai theo TUNG ly do K1; Wilson 95 % (macro-F1: bootstrap theo nguoi noi); tach theo ngon ngu, fold,
do dai tep; Req 14, Req 12, ma tran nham lan, theo kieu / nguon, RTF, head_prob co / khong hieu chinh. Ghi dan .json (--resume tiep tuc sau khi bi ngat).
  python -m v6.product.e2e_test [--n 20] [--n-real 150] [--n-app 20] [--op F] [--oracle-lang] [--final] [--no-k1x] [--no-pdf] [--limit 0] [--sets natvi naten]
                                [--resume] [--allow-unverified] [--k1-stored-filter]
  python -m v6.product.e2e_test selftest
  -> .tmp_v6/results/e2e_v9<op>[_final][_oracle][_smoke].md + .json   (khong ghi de e2e_v9.md cua 07/10)"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import numpy as np  # noqa: E402

V4 = os.path.join(BASE, "data_labels", "realistic_v4")
RES = os.path.join(BASE, ".tmp_v6", "results")
OK_K1 = ("nhan", "canh_bao", None)                       # chi dung khi --k1-stored-filter (cach cu)
CONSTRAINT_04 = ("04", "tu_choi")                         # quyet dinh K1 THAT (sau khi Product.analyse viet de dieu kien hoc 04) -> loai khoi mau so
ED = ("02", "03")
APP_CFGS = ("app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi")
LONG_S = 20.0                                             # hop dong: bao rieng tep <= 20 s / dai hon
N_BOOT = 200


def is_lockbox(e: dict, set_name: str) -> bool:
    if e["meta"].get("lockbox"):
        return True
    try:
        from v6 import common as C
        return C.is_lockbox(e, set_name)
    except Exception:                                    # noqa: BLE001  common.py dang duoc goi khac sua song song: khong bat buoc
        return False


def expected(meta: dict) -> str:
    l2 = meta.get("level2", [])
    return "03" if any(x.startswith("03") for x in l2) else "02" if any(x.startswith("02") for x in l2) else "01"


def eligible(entries: list, set_name: str, cfgs: tuple = ("goc",), k1_filter: bool = False, real_only: bool = False) -> tuple:
    """(tep dung duoc, Counter ly do loai khoi MAU). Khong loc theo K1 da luu tru khi k1_filter (K1 that cua lan chay quyet dinh, xem summarise)."""
    keep, why = [], Counter()
    for e in entries:
        m = e["meta"]
        if e["cfg"] not in cfgs or (real_only and expected(m) != "01"):
            continue
        if m.get("ngoai_pham_vi"):
            why["ngoai_pham_vi"] += 1
        elif is_lockbox(e, set_name):
            why["lockbox"] += 1
        elif k1_filter and m.get("k1", {}).get("decision") not in OK_K1:
            why["k1_04_da_luu"] += 1
        elif not e.get("wav") or not os.path.exists(os.path.join(V4, e["wav"])):
            why["thieu_wav"] += 1
        else:
            keep.append(e)
    return keep, why


def stratified(entries: list, n_per_kind: int, rng: random.Random, key=lambda e: e["kind"]) -> list:
    """Moi nhom (mac dinh kieu): toi da n tep, lay vong tron qua cac nguon (meta.source) da xao tron -> nguon it tep van co mat."""
    out = []
    by_kind = defaultdict(lambda: defaultdict(list))
    for e in entries:
        by_kind[key(e)][str(e["meta"].get("source"))].append(e)
    for k in sorted(by_kind):
        pools = {s: rng.sample(v, len(v)) for s, v in sorted(by_kind[k].items())}
        got = []
        while len(got) < n_per_kind and any(pools.values()):
            for s in list(pools):
                if pools[s] and len(got) < n_per_kind:
                    got.append(pools[s].pop())
        out += got
    return out


def set_lang(short: str) -> str:
    """Ngon ngu dung cua bo theo hau to ten (natvi / scrapevi / longvi -> vi; naten / scrapeen / longen -> en).
    18/10: ban cu chi coi 'natvi' la tieng Viet -> bao cao scrapevi xep nham vao nhom en (Product van chon dung)."""
    return "vi" if short.endswith("vi") else "en"


def app_ns_relabel(man: list) -> list:
    """Quyet dinh 4: khu on mac dinh cua ung dung (N7 ca tep, meta.app_ns trong scrapevi / scrapeen) = 01 -> meta qua
    v6.fusion.common_fx.apply_app_ns (mac dinh '01'); muc 'drop' (None) bi bo. Bo khong co dau N7: khong doi."""
    from v6.fusion.common_fx import apply_app_ns
    out = []
    for e in man:
        m = apply_app_ns(e["meta"])
        if m is not None:
            out.append(e if m is e["meta"] else {**e, "meta": m})
    return out


def samples(sets: tuple, n_per_kind: int, seed: int = 8, n_real: int | None = None, n_app: int = 0, k1_filter: bool = False) -> tuple:
    """[(bo, tep, tang)], {bo: {tang: ly do loai}}. tang goc: kieu co thao tac n_per_kind, kieu tep that n_real; tang app_that: n_app moi (bo, kenh ung dung)."""
    n_real = n_per_kind if n_real is None else n_real
    out, why = [], {}
    for short in sets:
        man = json.load(open(os.path.join(BASE, "data_labels", "v6", short, "manifest.json"), encoding="utf-8"))
        man = app_ns_relabel(man)
        keep, w = eligible(man, "v6" + short, k1_filter=k1_filter)
        why[short] = {"goc": dict(w)}
        real = [e for e in keep if expected(e["meta"]) == "01"]
        edit = [e for e in keep if expected(e["meta"]) != "01"]
        out += [(short, e, "goc") for e in stratified(edit, n_per_kind, random.Random(f"{seed}-{short}-sua"))]
        out += [(short, e, "goc") for e in stratified(real, n_real, random.Random(f"{seed}-{short}-that"))]
        if n_app:
            ka, wa = eligible(man, "v6" + short, cfgs=APP_CFGS, k1_filter=k1_filter, real_only=True)
            why[short]["app_that"] = dict(wa)
            out += [(short, e, "app_that") for e in stratified(ka, n_app, random.Random(f"{seed}-{short}-app"), key=lambda e: e["cfg"])]
    return out, why


def wilson(k: int, n: int) -> tuple:
    from v6.product.calib import wilson as W
    return W(k, n)


# ------------------------------------------------------------------ chi tieu hop dong (ham thuan, kiem thu duoc)
def is_c04(r: dict) -> bool:
    """Constraint 04 / tu choi: tep KHONG duoc cham (hop dong: Dieu kien cham). 04 do phan tich (luat Scope 5) khong thuoc day."""
    return r.get("k1") in CONSTRAINT_04


def reason_code(s: str) -> str:
    """'04 · “ồn quá” — chi tiet' -> 'ồn quá' (cum tu hop dong); chu khac (tu choi) giu nguyen phan truoc ' — '."""
    m = re.search(r"“([^”]+)”", s or "")
    return m.group(1) if m else (s or "khong ro").split(" — ")[0][:80]


def macro_f1(rows: list) -> tuple:
    """(macro-F1 tren cac nhan 01 / 02 / 03 co tep dung, {nhan: F1}); du doan 04 khong thuoc nhan nao -> la bo sot nhan dung (FN), khong phai FP."""
    per = {}
    for c in ("01", "02", "03"):
        tp = sum(r["exp"] == c and r["got"] == c for r in rows)
        fp = sum(r["exp"] != c and r["got"] == c for r in rows)
        fn = sum(r["exp"] == c and r["got"] != c for r in rows)
        if tp + fn:
            per[c] = 2 * tp / (2 * tp + fp + fn)
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def boot_f1(rows: list, n_boot: int = N_BOOT, seed: int = 0) -> tuple:
    """Khoang 95 % cua macro-F1: bootstrap lay mau lai theo NGUOI NOI (cum)."""
    if len(rows) < 2:
        return (float("nan"), float("nan"))
    g = defaultdict(list)
    for r in rows:
        g[str(r.get("person") or r["id"])].append(r)
    keys = sorted(g)
    rng = np.random.default_rng(seed)
    vals = [macro_f1([r for k in rng.choice(keys, len(keys)) for r in g[k]])[0] for _ in range(n_boot)]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))) if vals else (float("nan"), float("nan"))


def contract_metrics(rows: list, boot: bool = True) -> dict:
    """Chi tieu hop dong tren cac dong da chay. Moi ty le = (k, n); mau so chi gom tep DUOC CHAM (khong Constraint 04 / tu choi)."""
    sc = [r for r in rows if not is_c04(r)]
    ed, real = [r for r in sc if r["exp"] in ED], [r for r in sc if r["exp"] == "01"]
    fl = [r for r in sc if r["got"] in ED]
    f1, per = macro_f1(sc)
    return {"n_run": len(rows), "n_c04": len(rows) - len(sc), "n_scored": len(sc), "req1": f1, "req1_per": per, "req1_ci": boot_f1(sc) if boot else None,
            "req2": (sum(r["got"] in ED for r in ed), len(ed)), "req3": (sum(r["exp"] in ED for r in fl), len(fl)),
            "req4": (sum(r["got"] in ED for r in real), len(real)), "req9": (sum(r["got"] == "04" for r in sc), len(sc)),
            "real01": (sum(r["got"] == "01" for r in real), len(real)), "e03": (sum(r["got"] == "03" for r in sc if r["exp"] == "03"), sum(r["exp"] == "03" for r in sc))}


def fmt_rate(kn: tuple) -> str:
    k, n = kn
    if not n:
        return "- (n=0)"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}-{hi:.3f}] ({k}/{n})"


def exclusion_table(rows: list) -> list:
    """Hop dong: 'bao so va ti le tep bi loai theo tung ly do' - theo (tang, bo), ly do = cum tu K1 (mot tep nhieu ly do -> dem moi ly do)."""
    L = ["| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |", "|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("stratum", "goc"), r["set"])].append(r)
    for (st, s), v in sorted(by.items()):
        bad = [r for r in v if is_c04(r)]
        cnt = Counter(reason_code(x) for r in bad for x in (r.get("k1_reasons") or ["khong ro"]))
        if not cnt:
            L.append(f"| {st} | {s} | {len(v)} | 0 | - | 0 | 0.000 |")
        for why, c in cnt.most_common():
            L.append(f"| {st} | {s} | {len(v)} | {len(bad)} | {why} | {c} | {c / len(v):.3f} |")
    return L


def group_table(rows: list, keyfn, title: str) -> list:
    L = [f"\n**{title}** (tep duoc cham; Wilson 95 %)\n", "| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |", "|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[keyfn(r)].append(r)
    for k, v in sorted(by.items(), key=lambda t: str(t[0])):
        m = contract_metrics(v, boot=False)
        L.append(f"| {k} | {m['n_run']} | {m['n_c04']} | {m['n_scored']} | {m['req1']:.3f} | {fmt_rate(m['req2'])} | {fmt_rate(m['req4'])} | {fmt_rate(m['req9'])} |")
    return L


def headline(rows: list, title: str) -> list:
    m = contract_metrics(rows)
    lo, hi = m["req1_ci"]
    per = " · ".join(f"{c} {v:.3f}" for c, v in m["req1_per"].items())
    return [f"\n**{title}**: chay {m['n_run']} tep, bi loai (Constraint 04 / tu choi) {m['n_c04']}, DUOC CHAM {m['n_scored']}\n",
            f"- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = {m['req1']:.3f} [bootstrap theo nguoi noi {lo:.3f}-{hi:.3f}] ({per}) · chi tieu >= 0,86",
            f"- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = {fmt_rate(m['req2'])} · >= 0,90",
            f"- **Req 3 dang tin** (02/03 ma that bi sua) = {fmt_rate(m['req3'])} · >= 0,97",
            f"- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = {fmt_rate(m['req4'])} · <= 0,03",
            f"- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = {fmt_rate(m['req9'])} · <= 0,08",
            f"- tep that ra 01 = {fmt_rate(m['real01'])} · tep co thao tac 03 gan dung 03 = {fmt_rate(m['e03'])}"]


def summarise(rows: list, meta: dict) -> list:
    n = len(rows)
    goc = [r for r in rows if r.get("stratum", "goc") == "goc"]
    app = [r for r in rows if r.get("stratum") == "app_that"]
    L = [f"# Kiem thu dau-cuoi v9 - diem van hanh {meta['op']} ({meta['mode']}), {n} tep tu nhien ({', '.join(meta['sets'])}), {meta['made']}\n",
         f"- Cau hinh: k1x {'BAT' if meta['k1x'] else 'tat'} · ngon ngu {'BAO TRUOC (oracle)' if meta['oracle_lang'] else 'LID (khong bao truoc)'} · PDF {'bat' if meta['pdf'] else 'tat'} · "
         f"bound/reg/heads = {meta['resolved']} · tep nguong {meta['thr_file']} (sha1 {meta['thr_sha1'][:10]}) · kiem sha1 checkpoint: {meta['sha_check']}"
         f" · strict {'BAT' if meta.get('strict', True) else 'TAT (--allow-unverified)'} · hieu chinh: {meta.get('calib_check', '-')}",
         f"- Mau: toi da {meta.get('n_per_kind')} tep / kieu co thao tac, {meta.get('n_real')} tep / kieu tep that (goc), {meta.get('n_app')} tep that / (bo, kenh ung dung); "
         f"{'LOC truoc theo K1 da luu (cach cu)' if meta.get('k1_filter') else 'KHONG loc theo K1 da luu: K1x chay that quyet dinh'}",
         f"- Loai khoi MAU khi lap mau (khong phai Constraint): {meta['excluded']}",
         f"- **Req 14 (ho so khong tu mau thuan)**: {sum(r['ok14'] for r in rows)}/{n}" + (f"; vi pham: {[r['id'] for r in rows if not r['ok14']][:5]}" if any(not r['ok14'] for r in rows) else ""),
         f"- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: {sum(r['n_ops'] - r['img_bad'] - r['sign_bad'] for r in rows)}/{sum(r['n_ops'] for r in rows)} thao tac",
         f"- PDF: {sum(r['pdf'] for r in rows)}/{n} tep co bao_cao.pdf" if meta["pdf"] else "- PDF: tat (--no-pdf)",
         f"- Ngon ngu dung: {sum(r['lang_used'] == r['lang_true'] for r in rows if not is_c04(r))}/{sum(not is_c04(r) for r in rows)} tep duoc cham "
         f"(nguon: {dict(Counter(r['lang_source'] for r in rows if not is_c04(r)))})",
         f"- Xac suat ca tep: {'co hieu chinh' if any(r['head_prob'] for r in rows) else 'CHUA hieu chinh (head_prob None) - luat p >= 0,6 cua hop dong chua ap duoc'}"
         + (f" ({meta['calib_scope']})" if meta.get("calib_scope") else ""),
         f"- Dau ca tep theo fold: {meta.get('heads_note', '-')}",
         f"- 04 do phan tich (luat 5) theo duong: {dict(Counter(d for r in rows if not is_c04(r) and r['got'] == '04' for d in (r.get('why04') or ['?'])))}"]
    st = [r for r in rows if r.get("k1_stored") is not None]
    if st:
        s04 = [r for r in st if r["k1_stored"] in CONSTRAINT_04]
        L.append(f"- K1 da luu vs K1x that: luu 04 {len(s04)} tep -> that 04 {sum(is_c04(r) for r in s04)}; luu khong 04 {len(st) - len(s04)} tep -> that 04 "
                 f"{sum(is_c04(r) for r in st if r['k1_stored'] not in CONSTRAINT_04)}")
    L += headline(goc, "CHI TIEU CHINH - tang goc (tep goc)")
    L += group_table(goc, lambda r: set_lang(r["set"]), "Theo ngon ngu (tang goc)")
    if len({r["set"] for r in goc}) > len({set_lang(r["set"]) for r in goc}):
        L += group_table(goc, lambda r: r["set"], "Theo bo (tang goc)")
    L += group_table(goc, lambda r: f"fold {r['fold']}", "Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)")
    L += group_table(goc, lambda r: "<= 20 s" if r["dur"] <= LONG_S else "> 20 s", "Theo do dai tep (tang goc)")
    if app:
        L += headline(app, "BAO RIENG - tep THAT qua kenh ung dung (khu on mac dinh = 01, quyet dinh 4)")
        L += group_table(app, lambda r: f"{r['set']} · {r['cfg']}", "Tep that qua kenh ung dung, theo (bo, kenh)")
    L += ["\n**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)\n"] + exclusion_table(rows)
    cm = Counter((r["exp"], "K1 04" if is_c04(r) else r["got"]) for r in goc)
    L += ["\n**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)\n", "| | 01 | 02 | 03 | 04 | K1 04 |", "|---|---|---|---|---|---|"]
    for e_ in ("01", "02", "03"):
        L.append(f"| **{e_}** | " + " | ".join(str(cm[(e_, g)]) for g in ("01", "02", "03", "04", "K1 04")) + " |")
    L += ["\n**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %\n", "| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        by[(r.get("stratum", "goc"), r["set"], r["kind"])].append(r)
    for (st_, s, k), v in sorted(by.items()):
        sc = [r for r in v if not is_c04(r)]
        c = sum(r["got"] == r["exp"] for r in sc)
        lo, hi = wilson(c, len(sc))
        L.append(f"| {st_} | {s} | {k} | {len(v)} | {len(v) - len(sc)} | {len(sc)} | {c / len(sc) if sc else float('nan'):.0%} | {lo:.0%}-{hi:.0%} | "
                 f"{sum(r['got'] == '01' for r in sc)} | {sum(r['got'] == '04' for r in sc)} | {dict(Counter(r['source'] for r in v))} |")
    L += ["\n**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)\n", "| Bo | Nguon | that n | vu oan | sua n | dung |", "|---|---|---|---|---|---|"]
    bs = defaultdict(list)
    for r in goc:
        if not is_c04(r):
            bs[(r["set"], r["source"])].append(r)
    for (s, src), v in sorted(bs.items()):
        rl, ee = [r for r in v if r["exp"] == "01"], [r for r in v if r["exp"] != "01"]
        L.append(f"| {s} | {src} | {len(rl)} | {np.mean([r['got'] in ED for r in rl]) if rl else float('nan'):.2f} | {len(ee)} | "
                 f"{np.mean([r['got'] == r['exp'] for r in ee]) if ee else float('nan'):.2f} |")
    stt = defaultdict(list)
    for r in rows:
        for kk, v in r["timing"].items():
            stt[kk].append((v, r["dur"]))
    L += ["\n**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)\n", "| Tang | giay trung vi | RTF trung vi |", "|---|---|---|"]
    for kk, v in stt.items():
        L.append(f"| {kk} | {np.median([x for x, _ in v]):.2f} | {np.median([x / max(d, 1e-3) for x, d in v]):.2f} |")
    L.append(f"\nFold nguoi noi cua mau: {dict(Counter(r['fold'] for r in rows))} · loi chay (khong co ho so, khong tinh): {meta.get('errors', 0)}")
    return L


def run_one(V9, P, short: str, e: dict, out_dir: str, a, last: dict, stratum: str = "goc") -> dict:
    lang_true = set_lang(short)
    t0 = time.time()
    last.clear()
    try:
        code = V9.run(os.path.join(V4, e["wav"]), out_dir, op=a.op, lang=lang_true if a.oracle_lang else None, k1x=a.k1x, no_pdf=not a.pdf, threads=a.threads,
                      bound=P.bound, reg=P.reg, heads=P.heads_tag)
    except Exception as ex:                               # noqa: BLE001
        print(f"  LOI {e['id']}: {type(ex).__name__} {str(ex)[:120]}", flush=True)
        code = 4
    p = os.path.join(out_dir, "ho_so.json")
    alt = os.path.join(out_dir, "ho_so_khong_hop_le.json")
    ho = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else json.load(open(alt, encoding="utf-8")) if os.path.exists(alt) else None
    if ho is None:
        return {"id": e["id"], "error": f"khong co ho so (ma {code})"}
    ops = ho["output_chinh"].get("thao_tac", [])
    res = last.get("r", {})
    ch0 = (res.get("channels") or [{}])[0]
    ck = res.get("k1") or {}
    mrec = res.get("models") or {}
    save_scores(out_dir, res)
    return {"id": e["id"], "set": short, "kind": e["kind"], "cfg": e["cfg"], "stratum": stratum, "source": str(e["meta"].get("source")), "person": e["meta"].get("person"),
            "fold": P.fold, "exp": expected(e["meta"]), "got": (ho["output_chinh"].get("nhan") or {}).get("ma"), "code": code,
            "ok14": bool(ho.get("kiem_tra_dong_nhat", {}).get("dat")), "n_ops": len(ops),
            "img_bad": sum(not os.path.exists(os.path.join(out_dir, o.get("anh", "") or "_")) for o in ops),
            "sign_bad": sum(len([s for s in o.get("dau_hieu", []) if s.get("gia_tri") is not None]) < 2 or len(o.get("giai_thich", "")) < 20 for o in ops),
            "pdf": os.path.exists(os.path.join(out_dir, "bao_cao.pdf")), "dur": float(ho["tep"].get("do_dai_s") or 0.0), "wall": time.time() - t0,
            "timing": ho.get("thoi_gian_s", {}), "lang_true": lang_true, "lang_used": res.get("lang"), "lang_source": res.get("lang_source"), "lid": res.get("lid"),
            "k1": ck.get("decision"), "k1_reasons": list(ck.get("reasons") or []), "k1_stored": (e["meta"].get("k1") or {}).get("decision"),
            "why04": [x.get("duong") for x in ho["output_chinh"].get("ly_do_04") or []], "head": ch0.get("head"), "head_prob": ch0.get("head_prob"),
            "head_prob_method": ch0.get("head_prob_method"), "calib_scope": mrec.get("calib_scope"), "sha_check": mrec.get("sha_check"),
            "ckpt": [m["file"] for m in mrec.get("ckpt", [])], "len_adj": ch0.get("len_adj"), "lencal": (mrec.get("lencal") or {}).get("file")}


def save_scores(out_dir: str, res: dict) -> None:
    """18/10: luu diem khung (pb / pr float16), dau ca tep, nguong goc, do dai moi kenh -> <out_dir>/scores.npz de GIAI MA LAI ngoai tuyen
    (thu luat quyet dinh / nguong moi khong phai chay lai XLS-R). K1 04 (khong co kenh) -> khong ghi."""
    chs = res.get("channels") or []
    if not chs:
        return
    d = {"n_ch": np.array(len(chs)), "lang": np.array(str(res.get("lang"))), "fold": np.array(-1 if res.get("models", {}).get("fold") is None
                                                                                            else res["models"]["fold"])}
    for k, c in enumerate(chs):
        d[f"pb{k}"], d[f"pr{k}"] = np.asarray(c["pb"], np.float16), np.asarray(c["pr"], np.float16)
        d[f"head{k}"] = np.array(json.dumps(c.get("head") or {}))
        d[f"T{k}"] = np.array(json.dumps(c.get("T_base") or c.get("T") or {}))
        d[f"dur{k}"] = np.array(float(len(c.get("y", [])) / 16000.0))
    np.savez_compressed(os.path.join(out_dir, "scores.npz"), **d)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="toi da so tep moi (bo, kieu co thao tac)")
    ap.add_argument("--n-real", type=int, default=150, help="toi da so tep moi (bo, kieu tep THAT) o tang goc (Req 4 can n lon: muc tieu 3 %)")
    ap.add_argument("--n-app", type=int, default=20, help="toi da so tep THAT moi (bo, kenh ung dung); 0 = bo tang app_that")
    ap.add_argument("--op", default="F")
    ap.add_argument("--sets", nargs="+", default=["natvi", "naten"])
    ap.add_argument("--oracle-lang", action="store_true", help="dua ngon ngu dung thay vi LID (so sanh)")
    ap.add_argument("--final", action="store_true", help="mo hinh cuoi thay vi dinh tuyen fold (TRONG MAU)")
    ap.add_argument("--k1x", action="store_true", default=True)
    ap.add_argument("--no-k1x", dest="k1x", action="store_false")
    ap.add_argument("--pdf", action="store_true", default=True)
    ap.add_argument("--no-pdf", dest="pdf", action="store_false")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="chay thu: chi N tep (rai deu sau khi xep theo fold), ghi tep _smoke")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--allow-unverified", action="store_true", help="tat strict: chap nhan checkpoint / bo hieu chinh khong kiem duoc sha1 (ghi trong bao cao)")
    ap.add_argument("--k1-stored-filter", action="store_true", help="cach cu: bo tep co K1 da luu = 04 truoc khi lap mau")
    ap.add_argument("--tag", default="", help="hau to ten tep ket qua (vd '_scrape'): khong ghi de bao cao cu")
    ap.add_argument("--no-lenadj", action="store_true", help="tat nguong theo do dai (lencal) - doi chieu voi ban truoc 18/10")
    a = ap.parse_args()
    from v6.boundary import detect as D
    from v6.product.pipeline import Product
    import voiceaudit_v9 as V9
    S, why = samples(tuple(a.sets), a.n, a.seed, n_real=a.n_real, n_app=a.n_app, k1_filter=a.k1_stored_filter)
    S = sorted(S, key=lambda t: (D.person_fold(t[1]["meta"]), t[1]["id"]))      # xep theo fold: doi mo hinh it lan
    if a.limit:
        S = S[::max(1, len(S) // a.limit)][:a.limit]
    tag = f"e2e_v9{a.op}" + ("_final" if a.final else "") + ("_oracle" if a.oracle_lang else "") + ("_smoke" if a.limit else "") + a.tag
    out_root = os.path.join(BASE, ".tmp_v6", "e2e9", tag)
    jp = os.path.join(RES, tag + ".json")
    done = {}
    if a.resume and os.path.exists(jp):
        done = {r["id"]: r for r in json.load(open(jp, encoding="utf-8"))["rows"]}
    thr = f"thresholds_v9{a.op}.json"
    first = None if a.final else D.person_fold(S[0][1]["meta"])
    P = Product(None, None, None, thr, True, a.threads, fold=first, switchable=not a.final, strict=not a.allow_unverified, len_adj=not a.no_lenadj)   # bound / reg / heads THEO tep nguong
    V9._PRODUCT[(a.op, True, a.threads, P.bound, P.reg, P.heads_tag)] = P                        # voiceaudit_v9.run dung lai Product nay (khong nap lai XLS-R)
    last = {}
    orig = P.analyse

    def capture(*args, **kw):
        r = orig(*args, **kw)
        last["r"] = r
        return r
    P.analyse = capture
    print(f"{len(S)} tep mau ({'mo hinh cuoi' if a.final else 'dinh tuyen fold'}; {dict(Counter(t[2] for t in S))}); loai khi lap mau: {why}", flush=True)
    rec = P.model_record()
    meta = {"op": a.op, "mode": "mo hinh cuoi (trong mau)" if a.final else "dinh tuyen fold (ngoai mau)", "sets": a.sets, "k1x": a.k1x, "oracle_lang": a.oracle_lang, "pdf": a.pdf,
            "resolved": "/".join(str(v) for v in rec["resolved"].values()), "thr_file": rec["thresholds"]["file"], "thr_sha1": rec["thresholds"]["sha1"],
            "sha_check": rec["sha_check"], "strict": not a.allow_unverified, "calib_check": rec.get("calib_check"), "excluded": why, "made": time.strftime("%Y-%m-%d %H:%M"),
            "n_per_kind": a.n, "n_real": a.n_real, "n_app": a.n_app, "k1_filter": a.k1_stored_filter, "seed": a.seed,
            "calib_scope": P.calib_scope(generic=True), "lencal": rec.get("lencal"), "lencal_check": rec.get("lencal_check"),
            "heads_note": "mo hinh cuoi" if a.final else f"heads_final_{P.heads_tag}_f{{k}} hoc lai 10/10, khong trung het OOF (xem heads2_{P.heads_tag}_perfold.md)"}
    rows, errors = [r for r in done.values() if "error" not in r], 0
    for k, (short, e, stratum) in enumerate(S):
        if e["id"] in done:
            continue
        if not a.final:
            P.set_fold(D.person_fold(e["meta"]))
        r = run_one(V9, P, short, e, os.path.join(out_root, e["id"]), a, last, stratum)
        if "error" in r:
            errors += 1
            print(f"  {r['id']}: {r['error']}", flush=True)
        else:
            rows.append(r)
        if (k + 1) % 20 == 0 or k + 1 == len(S):
            meta["errors"] = errors
            json.dump({"meta": meta, "rows": rows}, open(jp + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, default=float)
            os.replace(jp + ".tmp", jp)
            print(f"  {k + 1}/{len(S)}", flush=True)
    meta["errors"] = errors
    json.dump({"meta": meta, "rows": rows}, open(jp, "w", encoding="utf-8"), ensure_ascii=False, default=float)
    L = summarise(rows, meta) if rows else ["# khong co tep nao chay duoc"]
    open(os.path.join(RES, tag + ".md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ kiem thu (khong chay mo hinh)
def _row(**kw) -> dict:
    r = {"id": "x", "set": "natvi", "kind": "ghep_thay", "cfg": "goc", "stratum": "goc", "source": "vox", "person": "p1", "fold": 1, "exp": "03", "got": "03", "k1": "nhan",
         "k1_reasons": [], "ok14": True, "n_ops": 0, "img_bad": 0, "sign_bad": 0, "pdf": False, "dur": 5.0, "timing": {"k1": 0.5}, "lang_true": "vi", "lang_used": "vi",
         "lang_source": "lid", "head_prob": None, "why04": []}
    return {**r, **kw}


def _meta() -> dict:
    return {"op": "F", "mode": "m", "sets": ["natvi"], "k1x": True, "oracle_lang": False, "pdf": False, "resolved": "fuse1/fuse1/h2", "thr_file": "t", "thr_sha1": "0" * 40,
            "sha_check": "verified", "excluded": {}, "made": "-"}


def test_stratified_round_robin_sources() -> None:
    es = [{"id": f"{s}{k}", "kind": "ghep_thay", "meta": {"source": s}} for s, n in (("vss", 30), ("vox", 3), ("celeb", 10)) for k in range(n)]
    got = stratified(es, 9, random.Random(0))
    c = Counter(e["meta"]["source"] for e in got)
    assert len(got) == 9 and c["vox"] == 3 and c["vss"] == 3 and c["celeb"] == 3, c
    assert len(stratified(es, 100, random.Random(0))) == len(es)
    app = [{"id": f"a{k}", "kind": "nguyen_ban", "cfg": APP_CFGS[k % 4], "meta": {"source": "vox"}} for k in range(40)]
    assert Counter(e["cfg"] for e in stratified(app, 3, random.Random(0), key=lambda e: e["cfg"])) == {c_: 3 for c_ in APP_CFGS}


def test_eligible_filters() -> None:
    wav = os.path.relpath(__file__, V4)
    base = {"cfg": "goc", "kind": "x", "wav": wav}
    es = [{**base, "id": "a", "meta": {"k1": {"decision": "nhan"}}}, {**base, "id": "b", "meta": {"ngoai_pham_vi": True}},
          {**base, "id": "c", "meta": {"lockbox": True}}, {**base, "id": "d", "meta": {"k1": {"decision": "04"}}},
          {**base, "id": "e", "meta": {}, "wav": "khong_co.wav"}, {**base, "id": "f", "cfg": "app_zalo_tin_nhan", "meta": {}},
          {**base, "id": "g", "cfg": "app_zalo_tin_nhan", "meta": {"level2": ["02_nen_toan_cuc"]}}]
    keep, why = eligible(es, "v6natvi")                                  # mac dinh: KHONG loc theo K1 da luu (K1x that quyet dinh)
    assert [e["id"] for e in keep] == ["a", "d"] and why == {"ngoai_pham_vi": 1, "lockbox": 1, "thieu_wav": 1}, why
    keep, why = eligible(es, "v6natvi", k1_filter=True)
    assert [e["id"] for e in keep] == ["a"] and why["k1_04_da_luu"] == 1
    keep, _ = eligible(es, "v6natvi", cfgs=APP_CFGS, real_only=True)       # tang app_that: chi tep that qua kenh ung dung
    assert [e["id"] for e in keep] == ["f"]


def test_app_ns_relabel_is_01() -> None:
    """Quyet dinh 4: tep chi co khu on N7 (NS ung dung) -> 01; N7 + sua khac giu sua khac; tep khong N7 khong doi."""
    n7 = {"level2": "02_nen_toan_cuc", "kind": "global", "family": "N7", "tool": "WebRTC NS"}
    cut = {"level2": "03_ghep_giong_that", "kind": "segment", "op": "xoa_loi"}
    man = [{"id": "a", "meta": {"level2": ["02_nen_toan_cuc"], "edits": [n7], "app_ns": True}},
           {"id": "b", "meta": {"level2": ["02_nen_toan_cuc", "03_ghep_giong_that"], "edits": [n7, cut]}},
           {"id": "c", "meta": {"level2": [], "edits": []}}]
    out = {e["id"]: e for e in app_ns_relabel(man)}
    assert expected(out["a"]["meta"]) == "01" and expected(out["b"]["meta"]) == "03" and out["c"] is man[2]


def test_set_lang_suffix() -> None:
    assert [set_lang(s) for s in ("natvi", "scrapevi", "longvi", "naten", "scrapeen", "longen")] == ["vi"] * 3 + ["en"] * 3


def test_expected_label_rule() -> None:
    assert expected({"level2": ["02_nen_cuc_bo", "03_ghep_giong_that"]}) == "03"
    assert expected({"level2": ["02_giong_toan_cuc"]}) == "02" and expected({}) == "01"


def test_req2_counts_analysis_04_as_miss_and_drops_constraint_04() -> None:
    """Loi cu (review 11/10): bo moi tep ra 04 -> 'Req 2 = 1.000 (n=1)'. Dung hop dong: 04 phan tich = khong bat -> 0,5; Constraint 04 moi bi loai."""
    rows = [_row(id="a", got="03"), _row(id="b", got="04", why04=["K3_vung_xam"]),
            _row(id="c", got="04", k1="04", k1_reasons=["04 · “ồn quá” — tiếng nói so với tiếng ồn dưới 10 dB"])]
    m = contract_metrics(rows)
    assert m["req2"] == (1, 2) and m["n_c04"] == 1 and m["n_scored"] == 2, m
    assert m["req9"] == (1, 2) and m["req3"] == (1, 1)
    L = summarise(rows, _meta())
    assert any("Req 2" in x and "0.500" in x and "(1/2)" in x for x in L), [x for x in L if "Req 2" in x]
    assert any("| ồn quá | 1 | 0.333 |" in x for x in L), [x for x in L if "ồn quá" in x]


def test_req4_keeps_analysis_04_real_files_in_denominator() -> None:
    rows = [_row(id="r1", exp="01", got="01"), _row(id="r2", exp="01", got="04"), _row(id="r3", exp="01", got="02"), _row(id="r4", exp="01", got="02", k1="tu_choi")]
    m = contract_metrics(rows)
    assert m["req4"] == (1, 3) and m["real01"] == (1, 3) and m["req9"] == (1, 3) and m["n_c04"] == 1, m


def test_macro_f1_counts_04_as_miss() -> None:
    rows = [_row(id="1", exp="01", got="01"), _row(id="2", exp="02", got="02"), _row(id="3", exp="03", got="03"), _row(id="4", exp="03", got="04")]
    f1, per = macro_f1(rows)
    assert per == {"01": 1.0, "02": 1.0, "03": 2 / 3} and abs(f1 - (2 + 2 / 3) / 3) < 1e-12, per      # 04: FN cua 03, khong phai FP cua nhan nao
    f1b, perb = macro_f1(rows + [_row(id="5", exp="01", got="03")])
    assert perb["03"] == 2 * 1 / (2 * 1 + 1 + 1) and perb["01"] == 2 / 3, perb
    lo, hi = boot_f1(rows * 3, n_boot=50)
    assert 0.0 <= lo <= hi <= 1.0


def test_reason_code_and_exclusion_rates() -> None:
    assert reason_code("04 · “quá ít tiếng nói” — 1.2 s tiếng nói / 9.0 s") == "quá ít tiếng nói"
    assert reason_code("từ chối nhận · dung lượng > 250 MB") == "từ chối nhận · dung lượng > 250 MB"
    rows = [_row(id=str(k)) for k in range(8)] + [_row(id="x1", k1="04", k1_reasons=["04 · “băng hẹp”", "04 · “quá nhỏ”"]), _row(id="x2", k1="04", k1_reasons=["04 · “băng hẹp”"])]
    L = exclusion_table(rows)
    assert any("| băng hẹp | 2 | 0.200 |" in x for x in L) and any("| quá nhỏ | 1 | 0.100 |" in x for x in L), L


def test_summary_renders_strata_and_groups() -> None:
    rows = [_row(), _row(id="y", exp="01", got="01"), _row(id="z", exp="01", got="02", stratum="app_that", cfg="app_zalo_tin_nhan", kind="nguyen_ban", fold=0, dur=30.0)]
    L = summarise(rows, _meta())
    txt = "\n".join(L)
    assert "Req 14" in txt and "3/3" in txt and "CHUA hieu chinh" in txt and "BAO RIENG" in txt and "app_zalo_tin_nhan" in txt, txt
    assert "Theo fold" in txt and "Theo do dai" in txt and "Req 9" in txt and "Req 1 macro-F1" in txt


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        n = 0
        for name, fn in list(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                n += 1
                print("ok", name, flush=True)
        print(f"{n} test dat")
    else:
        main()
