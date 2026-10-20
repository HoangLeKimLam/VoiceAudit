# -*- coding: utf-8 -*-
"""M8 - BAO CAO THONG KE bo test loi noi tu nhien (v6natvi tieng Viet / v6naten tieng Anh) sau khi v6.engine cham:
    python -m v6.engine --ext v6natvi --tag <tag>
    python -m v6.natural.report_natural <tag> [v6natvi|v6naten]
Moi ti le kem khoang tin cay 95 % bootstrap theo CUM (video / nguoi noi, 2 000 lan): cac tep cung cum khong doc lap nen
day moi la khoang dung de ket luan (Wilson luu trong json). Tach theo: loai thao tac, kenh, nguon, do dai cho sua,
do muot moi noi, cho lay doan ghep, ho cong cu, K1. Ngoai pham vi (phat lai, doi khang) bao rieng.
Ra: .tmp_v6/results/<natvi|naten>_report_<tag>.{md,json}"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.stats import bootstrap_ci_by_group, wilson  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, ".tmp_v6", "results")
SETS = {"v6natvi": ("natvi", "tieng Viet"), "v6naten": ("naten", "tieng Anh")}   # ten tap engine -> (thu muc, ngon ngu)
LABELS = ("01", "02", "03")
MAIN = ["app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi"]
KIND_VN = {"nguyen_ban": "tep that (tu nhien)", "nguyen_ban_vsasv": "tep that (VSASV)", "nen_toan_cuc": "khu on ca file",
           "nen_cuc_bo": "khu on mot doan", "giong_toan_cuc": "doi giong ca file (dich cao do)",
           "giong_cuc_bo": "doi giong mot doan", "doi_giong_ai": "doi giong ca file bang AI (VSASV)",
           "ghep_thay": "ghep loi that - thay", "ghep_chen": "ghep loi that - chen", "xoa_loi": "xoa loi",
           "may_thay": "giong may - thay", "may_chen": "giong may - chen", "may_ca_file": "giong may ca file",
           "ngoai_pham_vi_replay": "ngoai pham vi - phat lai qua loa", "ngoai_pham_vi_doi_khang": "ngoai pham vi - doi khang",
           "ngoai_pham_vi_bang_hep": "ngoai pham vi - cuoc goi bang hep that (ky vong 04)",
           "ghep_thay_vung": "ghep thay - VUNG", "ghep_thay_ky": "ghep thay - KY (che giau)",
           "ghep_chen_vung": "ghep chen - VUNG", "ghep_chen_ky": "ghep chen - KY (che giau)",
           "xoa_loi_vung": "xoa loi - VUNG", "xoa_loi_ky": "xoa loi - KY (che giau)",
           "nen_cuc_bo_ky": "khu on mot doan - KY", "giong_cuc_bo_ky": "doi giong mot doan - KY",
           "da_ghep_nen": "DA THAO TAC: ghep/xoa + khu on ca file", "da_ghep_giong": "DA THAO TAC: ghep/xoa + doi giong doan khac",
           "da_hai_cho": "DA THAO TAC: hai cho ghep/xoa"}


def ci(flags, groups):
    """ti le + khoang Wilson + khoang bootstrap theo cum."""
    f = np.asarray(flags, float)
    if len(f) == 0:
        return None
    k, n = int(f.sum()), len(f)
    est, bl, bh = bootstrap_ci_by_group(f, list(groups), np.mean, n=2000)
    return {"p": est, "wilson": wilson(k, n), "boot": (bl, bh), "n": n, "k": k, "cum": len(set(groups))}


def fmt(c):
    if not c:
        return "-"
    b = c["boot"]
    bs = f"{100 * b[0]:.1f}-{100 * b[1]:.1f}" if np.isfinite(b[0]) else "-"
    return f"{100 * c['p']:.1f} % [{bs}] (n={c['n']}, {c['cum']} cum)"


def iou(a, b):
    i = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    u = max(a[1], b[1]) - min(a[0], b[0])
    return i / u if u > 0 else 0.0


def edit_hit(rec, meta):
    """Moi thao tac: (bat dung loai [+ trung vung neu cuc bo], moc <= 500 ms, IoU)."""
    rows = rec["rows"] if rec["final"] != "04" else []
    out = []
    for e in meta.get("edits", []):
        c = e["level2"]
        if e.get("kind") == "global":
            out.append((any(x["level2"] == c for x in rows), None, None))
            continue
        seg = (e["start"], e["end"])
        same = [x for x in rows if x["level2"] == c and x.get("start") is not None and iou((x["start"], x["end"]), seg) > 0]
        if not same:
            out.append((False, False, 0.0))
            continue
        b = max(same, key=lambda x: iou((x["start"], x["end"]), seg))
        m500 = max(abs(b["start"] - seg[0]), abs(b["end"] - seg[1])) <= 0.5
        if e.get("sub") == "cat_bo":                 # xoa loi: moi noi nam trong vung khoanh +- 0,5 s
            m500 = b["start"] - 0.5 <= e.get("joint", sum(seg) / 2) <= b["end"] + 0.5
        out.append((True, m500, iou((b["start"], b["end"]), seg)))
    return out


def macro_f1(pairs):
    f1s = []
    for lab in LABELS:
        tp = sum(1 for e, f in pairs if e == lab and f == lab)
        fp = sum(1 for e, f in pairs if e != lab and f == lab)
        fn = sum(1 for e, f in pairs if e == lab and f != lab)
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def nbucket(n):
    return "-" if n is None else "1-3 tu" if n <= 3 else "4-12 tu"


class Report:
    def __init__(self, tag, set_name="v6natvi"):
        res = json.load(open(os.path.join(RES, f"engine_{tag}.json"), encoding="utf-8"))
        self.short, self.lang = SETS[set_name]
        man_p = os.path.join(BASE, "data_labels", "v6", self.short, "manifest.json")
        man = {e["id"]: e for e in json.load(open(man_p, encoding="utf-8"))}
        self.recs = [r for r in res["records"][set_name] if r["id"] in man]
        for r in self.recs:
            r["_m"] = man[r["id"]]["meta"]
            r["_g"] = r["_m"]["speaker"]
        self.ins = [r for r in self.recs if not r["_m"].get("ngoai_pham_vi")]
        self.L, self.out, self.tag = [], {"tag": tag}, tag

    def block(self, rs, title):
        g = [r["_g"] for r in rs]
        real = [r for r in rs if r["expected"] == "01"]
        edited = [r for r in rs if r["expected"] in ("02", "03")]
        flagged = [r for r in rs if r["final"] in ("02", "03")]
        r04 = [r for r in rs if r["final"] == "04"]
        pairs = np.empty((len(rs), 2), dtype=object)
        pairs[:, 0] = [r["expected"] for r in rs]
        pairs[:, 1] = [r["final"] for r in rs]
        mf = bootstrap_ci_by_group(pairs, g, lambda v: macro_f1(list(map(tuple, v))), n=1000)
        d = {"macro_f1": mf,
             "dung_nhan": ci([r["correct"] for r in rs], g),
             "that_ra_01": ci([r["final"] == "01" for r in real], [r["_g"] for r in real]),
             "vu_oan": ci([r["final"] in ("02", "03") for r in real], [r["_g"] for r in real]),
             "khong_bo_sot": ci([r["final"] in ("02", "03") for r in edited], [r["_g"] for r in edited]),
             "bao_sua_dang_tin": ci([r["expected"] in ("02", "03") for r in flagged], [r["_g"] for r in flagged]),
             "ti_le_04": ci([r["final"] == "04" for r in rs], g),
             "04_do_k1": ci([r["_m"].get("k1", {}).get("decision") in ("04", "tu_choi") for r in r04], [r["_g"] for r in r04])}
        L = self.L
        L.append(f"\n## {title}\n\n| Chi so | Gia tri [khoang tin cay 95 %] |\n|---|---|")
        L.append(f"| Macro-F1 (01/02/03; 04 tinh la sai) | {mf[0]:.3f} [{mf[1]:.3f}-{mf[2]:.3f}] |")
        for k, name in [("dung_nhan", "Tep ra dung nhan"), ("that_ra_01", "Tep that ra 01"),
                        ("vu_oan", "Vu oan tep that (ra 02/03)"), ("khong_bo_sot", "Khong bo sot tep bi sua (ra 02/03)"),
                        ("bao_sua_dang_tin", "Bao co sua dang tin"), ("ti_le_04", "Ti le 04"),
                        ("04_do_k1", "Trong so 04: do kiem dau vao K1")]:
            L.append(f"| {name} | {fmt(d[k])} |")
        cm = Counter((r["expected"], r["final"]) for r in rs)
        L.append("\nMa tran nham lan (hang: nhan dung, cot: nhan he thong)\n\n| | 01 | 02 | 03 | 04 |\n|---|---|---|---|---|")
        for e in LABELS:
            L.append(f"| **{e}** | " + " | ".join(str(cm[(e, f)]) for f in ("01", "02", "03", "04")) + " |")
        d["ma_tran"] = {f"{e}->{f}": v for (e, f), v in cm.items()}
        return d

    def by_kind_channel(self):
        cfgs = ["goc"] + MAIN
        self.L.append("\n## 3 - Theo loai thao tac x kenh: ti le ra dung nhan (so tep)\n")
        self.L.append("| Loai | " + " | ".join(cfgs) + " |\n|---|" + "---|" * len(cfgs))
        tab = defaultdict(dict)
        for kind, vn in KIND_VN.items():
            if kind.startswith("ngoai"):
                continue
            cells = []
            for c in cfgs:
                rs = [r for r in self.ins if r["kind"] == kind and r["cfg"] == c]
                v = ci([r["correct"] for r in rs], [r["_g"] for r in rs]) if rs else None
                tab[kind][c] = v
                cells.append("-" if not v else f"{100 * v['p']:.0f} % ({v['n']})")
            if any(tab[kind].values()):
                self.L.append(f"| {vn} | " + " | ".join(cells) + " |")
        return tab

    def by_op(self):
        self.L.append("\n## 4 - Muc thao tac (tep goc): bat dung loai, moc <= 500 ms, IoU\n")
        self.L.append("| Loai | Bat dung loai | Moc <= 500 ms | IoU trung binh |\n|---|---|---|---|")
        st = {}
        for kind, vn in KIND_VN.items():
            rs = [r for r in self.ins if r["kind"] == kind and r["cfg"] == "goc" and r["_m"].get("edits")]
            if not rs:
                continue
            hits = [(h, r["_g"]) for r in rs for h in edit_hit(r, r["_m"])]
            det = ci([h[0][0] for h in hits], [h[1] for h in hits])
            loc = [(h[0][1], h[1]) for h in hits if h[0][1] is not None]
            m5 = ci([x[0] for x in loc], [x[1] for x in loc]) if loc else None
            ious = [h[0][2] for h in hits if h[0][2]]
            st[kind] = {"bat": det, "moc500": m5, "iou": float(np.mean(ious)) if ious else None}
            self.L.append(f"| {vn} | {fmt(det)} | {fmt(m5)} | " + (f"{np.mean(ious):.2f} |" if ious else "- |"))
        return st

    def split(self, title, key, rs):
        self.L.append(f"\n### {title}\n\n| Nhom | Ra dung nhan | Ti le 04 |\n|---|---|---|")
        grp = defaultdict(list)
        for r in rs:
            grp[str(key(r))].append(r)
        d = {}
        for k in sorted(grp):
            g = grp[k]
            d[k] = (ci([r["correct"] for r in g], [r["_g"] for r in g]), ci([r["final"] == "04" for r in g], [r["_g"] for r in g]))
            self.L.append(f"| {k} | {fmt(d[k][0])} | {fmt(d[k][1])} |")
        return d

    def by_concealment(self):
        """Muc che giau (vung / trung binh / ky) cho cac thao tac cuc bo 03 + 02, tep goc va qua 4 kenh."""
        base = {"ghep_thay", "ghep_chen", "xoa_loi", "nen_cuc_bo", "giong_cuc_bo"}
        self.L.append("\n## 8 - Theo MUC CHE GIAU (tep goc | 4 kenh chinh): ra dung nhan · bat dung thao tac\n\n"
                      "| Thao tac | Muc | Ra dung nhan (goc) | Bat dung thao tac (goc) | Ra dung nhan (4 kenh) |\n|---|---|---|---|---|")
        d = {}
        for op in sorted(base):
            for tier in ("vung", "trung_binh", "ky"):
                kind = op if tier == "trung_binh" else f"{op}_{tier}"
                g = [r for r in self.ins if r["kind"] == kind and r["cfg"] == "goc"]
                m4 = [r for r in self.ins if r["kind"] == kind and r["cfg"] in MAIN]
                if not g:
                    continue
                hits = [(h[0], r["_g"]) for r in g for h in edit_hit(r, r["_m"])]
                c1 = ci([r["correct"] for r in g], [r["_g"] for r in g])
                c2 = ci([x[0] for x in hits], [x[1] for x in hits])
                c3 = ci([r["correct"] for r in m4], [r["_g"] for r in m4]) if m4 else None
                d[kind] = (c1, c2, c3)
                self.L.append(f"| {op} | {tier} | {fmt(c1)} | {fmt(c2)} | {fmt(c3)} |")
        return d

    def multi_op(self):
        """Tep >= 2 thao tac: nhan tep dung (nang nhat) · bat >= 1 thao tac · bat DU moi thao tac · theo tung loai thao tac."""
        rs = [r for r in self.ins if r["cfg"] == "goc" and len(r["_m"].get("edits", [])) >= 2]
        if not rs:
            return None
        self.L.append("\n## 9 - DA THAO TAC (tep goc): nhan tep lay muc nang nhat, phan Thao tac phai co du cac dong\n\n"
                      "| Loai | n | Nhan tep dung | Bat >= 1 thao tac | Bat DU moi thao tac | Bat thao tac 02 di kem |\n|---|---|---|---|---|---|")
        d = {}
        for kind in sorted({r["kind"] for r in rs}):
            g = [r for r in rs if r["kind"] == kind]
            H = [(edit_hit(r, r["_m"]), r) for r in g]
            any1 = ci([any(x[0] for x in h) for h, _ in H], [r["_g"] for _, r in H])
            allx = ci([all(x[0] for x in h) for h, _ in H], [r["_g"] for _, r in H])
            h02 = [(x[0], r["_g"]) for h, r in H for x, e in zip(h, r["_m"]["edits"]) if e["level2"].startswith("02")]
            c02 = ci([x[0] for x in h02], [x[1] for x in h02]) if h02 else None
            lab = ci([r["correct"] for r in g], [r["_g"] for r in g])
            d[kind] = {"nhan": lab, "bat_1": any1, "bat_du": allx, "bat_02": c02}
            self.L.append(f"| {KIND_VN.get(kind, kind)} | {len(g)} | {fmt(lab)} | {fmt(any1)} | {fmt(allx)} | {fmt(c02)} |")
        return d

    def by_channel(self):
        self.L.append("\n## 6 - Moi cau hinh kenh (mau nho, chi de thay xu huong)\n\n"
                      "| Kenh | Tep that ra 01 | Tep sua ra 02/03 | Ti le 04 |\n|---|---|---|---|")
        ch = defaultdict(list)
        for r in self.ins:
            ch[r["cfg"]].append(r)
        d = {}
        for c, rs in sorted(ch.items()):
            real = [r for r in rs if r["expected"] == "01"]
            edd = [r for r in rs if r["expected"] != "01"]
            d[c] = (ci([r["final"] == "01" for r in real], [r["_g"] for r in real]),
                    ci([r["final"] in ("02", "03") for r in edd], [r["_g"] for r in edd]),
                    ci([r["final"] == "04" for r in rs], [r["_g"] for r in rs]))
            self.L.append(f"| {c} | {fmt(d[c][0])} | {fmt(d[c][1])} | {fmt(d[c][2])} |")
        return d

    def out_of_scope(self):
        oos = [r for r in self.recs if r["_m"].get("ngoai_pham_vi")]
        if not oos:
            return
        self.L.append("\n## 7 - Ngoai pham vi (khong tinh vao chi tieu)\n\n| Loai | n | 01 | 02 | 03 | 04 |\n|---|---|---|---|---|---|")
        for kind in ("ngoai_pham_vi_replay", "ngoai_pham_vi_doi_khang", "ngoai_pham_vi_bang_hep"):
            rs = [r for r in oos if r["kind"] == kind]
            if rs:
                c = Counter(r["final"] for r in rs)
                self.L.append(f"| {KIND_VN[kind]} | {len(rs)} | " +
                              " | ".join(f"{100 * c[x] / len(rs):.0f} %" for x in ("01", "02", "03", "04")) + " |")

    def run(self):
        ins, L, o = self.ins, self.L, self.out
        L.append(f"# Bo test loi noi tu nhien {self.lang} - he thong {self.tag}\n")
        L.append(f"{len(ins)} tep trong pham vi, {len(self.recs) - len(ins)} tep ngoai pham vi. Khoang trong ngoac: "
                 "khoang tin cay 95 % bootstrap theo cum (video / nguoi noi).\n")
        o["tong_goc"] = self.block([r for r in ins if r["cfg"] == "goc"], "1 - Tong: tep goc (khong qua kenh)")
        o["tong_4kenh"] = self.block([r for r in ins if r["cfg"] in MAIN], "2 - Tong: qua 4 kenh dien thoai chinh")
        o["theo_loai_kenh"] = self.by_kind_channel()
        o["muc_thao_tac"] = self.by_op()
        L.append("\n## 5 - Tach theo yeu to (tep goc)\n")
        goc = [r for r in ins if r["cfg"] == "goc"]
        ed = [r for r in goc if r["_m"].get("edits")]
        loc03 = [r for r in ed if r["kind"] in ("ghep_thay", "ghep_chen", "xoa_loi", "may_thay", "may_chen")]
        o["theo_nguon"] = self.split("Nguon ban goc", lambda r: r["_m"].get("source"), goc)
        o["theo_do_dai"] = self.split("Do dai cho sua (ghep, xoa, giong may)",
                                      lambda r: nbucket(r["_m"]["edits"][0].get("n_words")), loc03)
        o["theo_fade"] = self.split("Do muot moi noi", lambda r: f"{r['_m']['edits'][0].get('fade_ms')} ms", loc03)
        o["theo_cho_lay"] = self.split("Doan ghep lay tu", lambda r: r["_m"]["edits"][0].get("donor_where"),
                                       [r for r in ed if r["kind"] in ("ghep_thay", "ghep_chen")])
        o["theo_ho"] = self.split("Ho cong cu khu on / doi giong", lambda r: r["_m"]["edits"][0].get("family"),
                                  [r for r in ed if r["kind"] in ("nen_toan_cuc", "nen_cuc_bo", "giong_toan_cuc", "giong_cuc_bo")])
        o["theo_k1"] = self.split("Ket qua kiem dau vao K1", lambda r: r["_m"].get("k1", {}).get("decision"), goc)
        if any(r["_m"].get("host_info", {}).get("accent") for r in goc):
            o["theo_giong"] = self.split("Giong / nhom L1 nguoi noi (tieng Anh)",
                                         lambda r: r["_m"].get("host_info", {}).get("accent") or "-", goc)
        o["theo_kenh"] = self.by_channel()
        self.out_of_scope()
        o["theo_muc_che_giau"] = self.by_concealment()
        o["da_thao_tac"] = self.multi_op()
        md = "\n".join(L) + "\n"
        os.makedirs(RES, exist_ok=True)
        open(os.path.join(RES, f"{self.short}_report_{self.tag}.md"), "w", encoding="utf-8").write(md)
        json.dump(o, open(os.path.join(RES, f"{self.short}_report_{self.tag}.json"), "w", encoding="utf-8"), ensure_ascii=False,
                  indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
        return md


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(Report(sys.argv[1] if len(sys.argv) > 1 else "natvi_v5", sys.argv[2] if len(sys.argv) > 2 else "v6natvi").run())