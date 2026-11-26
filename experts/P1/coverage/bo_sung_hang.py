"""Bộ kiểm bổ sung cho 3 hàng Constraint chưa có bài kiểm chủ động trên dữ liệu P1 (Huấn duyệt 26/11, NHAT_KY 2.29):
  Ngôn ngữ vi / en      : `ngonngu.khac_<tiếng>` sách nói MUSAN librivox 6 thứ tiếng × 3 (PHẢI cảnh báo «ngôn ngữ khác»);
                          `ngonngu.vi` / `ngonngu.en` FLEURS × 12 và `ngonngu.en_sach_noi` librivox Anh × 6 (KHÔNG được cảnh báo).
  Nói chồng ≤ 10 %      : `noichong.<0|5|20|40>pct` — trộn hai câu FLEURS cùng ngôn ngữ, cùng mức RMS; độ lệch chọn sao cho tỉ lệ
                          khung cả hai cùng nói / khung có tiếng nói (đo trên TỪNG NGUỒN bằng VAD năng lượng k1.speech_mask, khác pyannote
                          đang được thử) gần đích nhất. Đáp án theo số ĐO ĐƯỢC: ≤ 7 % không được cảnh báo · ≥ 15 % phải cảnh báo · giữa = ca biên.
  Loại âm thanh         : `loai.nhac_khong_loi` / `loai.nhac_co_hat` MUSAN × 12 (PHẢI 04 «không phải ghi âm lời nói» / «quá ít tiếng nói»).
Mọi bản ghi gắn `ngoai_pham_vi` (chỉ để thử K1, không học / chấm hệ thống). meta.lang của tệp không phải vi / en đặt «en» (kiem_bo chỉ nhận vi | en),
ngôn ngữ thật ở meta.ngon_ngu_that. Nhãn ngôn ngữ / có hát lấy từ ANNOTATIONS của MUSAN (chưa có phép đo độc lập → ghi trong thẻ).

  .venv/Scripts/python.exe experts/P1/coverage/bo_sung_hang.py build | kiem | run | tom | selftest
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import random
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "build_cond"))
import dung_bo as D  # noqa: E402

sys.path.insert(0, D.ROOT)

BO, PRE, SEED, SR = "p1_k1bosung_v1", "p1kb", 2611, 16000
OUT = os.path.join(D.ROOT, "data_labels", "v6", BO)
KIEM = os.path.join(OUT, "_kiem")
RAW_KQ, BAO = os.path.join(KIEM, "k1_ext.jsonl"), os.path.join(KIEM, "KET_QUA_BO_SUNG.md")
CAT_S, HOP = 20, 320
TIENG = ("german", "spanish", "french", "russian", "portuguese", "italian")
MUC = (0, 5, 20, 40)
KHONG_CB, PHAI_CB = 0.07, 0.15
W_NN, W_NC = "ngôn ngữ khác", "nói chồng"
LY_DO_LOAI = ("không phải ghi âm lời nói", "quá ít tiếng nói")


# ----------------------------------------------------------------------------- hàm thuần
def mat_na(x: np.ndarray) -> np.ndarray:
    """VAD năng lượng 20 ms của K1 (k1.speech_mask) — độc lập với pyannote đang được thử."""
    from vaudit import k1
    n = len(x) // HOP
    db = 10 * np.log10(np.square(x[: n * HOP].reshape(n, HOP)).mean(1) + 1e-12)
    return k1.speech_mask(db)[0]


def ti_le_chong(ma: np.ndarray, mb: np.ndarray, lech: int) -> float:
    """Tỉ lệ khung cả hai cùng nói / khung có ít nhất một người nói, khi B bắt đầu trễ `lech` khung so với A."""
    n = max(len(ma), lech + len(mb))
    a, b = np.zeros(n, bool), np.zeros(n, bool)
    a[: len(ma)] = ma
    b[lech: lech + len(mb)] = mb
    return float((a & b).sum() / max(1, (a | b).sum()))


def tim_lech(ma: np.ndarray, mb: np.ndarray, dich: float) -> tuple[int, float]:
    """Độ lệch (khung) cho tỉ lệ chồng gần đích nhất. Đích 0 → B bắt đầu sau khi A kết thúc 0,3 s."""
    if dich == 0:
        lech = len(ma) + int(0.3 * SR / HOP)
        return lech, ti_le_chong(ma, mb, lech)
    best = min(range(0, len(ma)), key=lambda k: abs(ti_le_chong(ma, mb, k) - dich))
    return best, ti_le_chong(ma, mb, best)


def tron(a: np.ndarray, b: np.ndarray, lech: int) -> np.ndarray:
    y = np.zeros(max(len(a), lech * HOP + len(b)) + int(0.3 * SR))
    y[: len(a)] += a
    y[lech * HOP: lech * HOP + len(b)] += b
    return y


def dap_an_chong(do: float) -> tuple[str, bool]:
    return ("accept", False) if do <= KHONG_CB else ("canh_bao", False) if do >= PHAI_CB else ("accept", True)


def cat_giua(path: str) -> np.ndarray:
    x = D.load(path)
    n = CAT_S * SR
    a = max(0, (len(x) - n) // 2)
    return x[a: a + n]


def cham(c: dict, r: dict) -> bool | None:
    """True / False = đúng / sai theo hàng đang thử; None = ca biên (không chấm)."""
    if c["ca_bien"]:
        return None
    w = r.get("warnings", [])
    if c["hang"] == "loai":
        return r.get("decision") == "04" and any(k in x for x in r.get("reasons", []) for k in LY_DO_LOAI)
    co = any(x.startswith(W_NN if c["hang"] == "ngonngu" else W_NC) for x in w)
    return co if c["mong_doi"] == "canh_bao" else not co


def wilson(k: int, n: int, z: float = 1.96) -> str:
    if n == 0:
        return "–"
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return f"{k}/{n} = {100 * p:.0f} % ({100 * max(0, c - h):.0f}–{100 * min(1, c + h):.0f})"


# ----------------------------------------------------------------------------- dựng
class Ghi:
    def __init__(self) -> None:
        self.man, self.cases, self.prov = [], [], []

    def them(self, hang: str, ca: str, lang: str, y: np.ndarray, base: dict, mong_doi: str, bien: bool, nguon: list, ts: dict, do_lai: dict,
             that: str | None = None) -> None:
        y, lvl = D.normalize(y)
        rid = f"{PRE}_{hang}_{lang}_{len(self.man):04d}"
        path = os.path.join(OUT, hang, rid + ".wav")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sf.write(path, y.astype(np.float32), SR, subtype="PCM_16")
        ly_do = {"canh_bao": W_NN if hang == "ngonngu" else W_NC, "04": LY_DO_LOAI[0]}.get(mong_doi)
        k1 = {"ca": ca, "mong_doi": mong_doi, "ly_do": ly_do, "ca_bien": bien}
        meta = {"lang": lang, "source": base["source"], "person": base["person"], "speaker": base["person"], "level1": ["01_nguyen_ban"],
                "level2": [], "edits": [], "ngoai_pham_vi": "bộ kiểm K1 (ngôn ngữ / nói chồng / loại âm thanh), không dùng học / chấm hệ thống",
                "k1_ca": k1}
        if that:
            meta["ngon_ngu_that"] = that
        self.man.append({"id": rid, "kind": f"p1_{hang}", "cfg": ca, "src": base["id"], "wav": f"../v6/{BO}/{hang}/{rid}.wav", "meta": meta})
        self.cases.append({"file": f"{hang}/{rid}.wav", "hang": hang, **k1, "base": base["id"]})
        self.prov.append({"id": rid, "file": D.rel(path), "sha256": D.sha256(path), **k1, "nguon": nguon, "tham_so": ts,
                          "do_lai": {**do_lai, "muc_rms_dbfs": round(lvl, 2), "dinh": round(float(np.max(np.abs(y))), 4), "do_dai_s": round(len(y) / SR, 2)}})


def ent(path: str, lic: str, src: str) -> dict:
    return {"path": D.rel(path), "sha256": D.sha256(path), "giay_phep": lic, "nguon": src}


def fleurs(lang: str, n: int, rng: random.Random) -> list[dict]:
    """n câu FLEURS sạch. fleurs_bases xen kẽ nam / nữ; FLEURS vi chỉ có giọng nam nên chỉ trả một nửa → xin gấp đôi rồi cắt."""
    bs = D.fleurs_bases(f"{lang}_{'vn' if lang == 'vi' else 'us'}", 2 * n, rng)[:n]
    return [{"id": f"fleurs_{lang}_{b['split']}_{b['file'][:-4]}", "wav": b["wav"], "source": "fleurs", "license": "CC BY 4.0",
             "person": f"fleurs:{lang}:{b['file']}"} for b in bs]


def dung_ngonngu(g: Ghi, fl: dict, rng: random.Random) -> None:
    sp = {p: m for p, m in D.musan_meta("speech").items() if os.sep + "librivox" + os.sep in p and D.license_text_ok(m["license"])
          and sf.info(p).duration >= CAT_S + 5}
    for ng, n in (*((t, 3) for t in TIENG), ("english", 6)):
        ds = sorted(p for p, m in sp.items() if m["vocals"] == ng)
        assert len(ds) >= n, (ng, len(ds))
        for p in rng.sample(ds, n):
            ten = os.path.basename(p)[:-4]
            base = {"id": ten, "source": "musan_librivox", "person": f"musan:{ten}"}
            khac = ng != "english"
            g.them("ngonngu", f"ngonngu.khac_{ng}" if khac else "ngonngu.en_sach_noi", "en", cat_giua(p), base, "canh_bao" if khac else "accept", False,
                   [ent(p, sp[p]["license"], "MUSAN/speech/librivox")], {"cat_giua_s": CAT_S, "ngon_ngu_annotations": ng}, {}, that=ng)
    for lang in ("vi", "en"):
        for b in fl[lang][:12]:
            g.them("ngonngu", f"ngonngu.{lang}", lang, D.load(b["wav"]), b, "accept", False, [ent(b["wav"], b["license"], "FLEURS")], {}, {}, that=lang)


def dung_noichong(g: Ghi, fl: dict) -> None:
    for lang in ("vi", "en"):
        cap = fl[lang][12:]                                          # 12 cặp (A, B); MỖI cặp trộn ở cả 4 mức → chỉ khác mức chồng
        assert len(cap) >= 24, (lang, len(cap))
        for muc in MUC:
            for i in range(12):
                A, B = cap[2 * i], cap[2 * i + 1]
                a, b = D.normalize(D.load(A["wav"]))[0], D.normalize(D.load(B["wav"]))[0]
                ma, mb = mat_na(a), mat_na(b)
                lech, do = tim_lech(ma, mb, muc / 100)
                md, bien = dap_an_chong(do)
                g.them("noichong", f"noichong.{muc}pct", lang, tron(a, b, lech), A, md, bien,
                       [ent(A["wav"], A["license"], "FLEURS"), ent(B["wav"], B["license"], "FLEURS")],
                       {"dich_pct": muc, "lech_b_s": round(lech * HOP / SR, 2), "muc_rms_tung_nguon_dbfs": D.LEVEL_DBFS, "nguoi_b": B["person"]},
                       {"noi_chong_pct_do": round(100 * do, 2), "loi_a_s": round(float(ma.sum()) * HOP / SR, 2), "loi_b_s": round(float(mb.sum()) * HOP / SR, 2)})


def dung_loai(g: Ghi, rng: random.Random) -> None:
    mm = D.musan_meta("music")
    for ca, co_hat in (("loai.nhac_khong_loi", "N"), ("loai.nhac_co_hat", "Y")):
        ds = sorted(p for p, m in mm.items() if m["vocals"] == co_hat and D.license_text_ok(m["license"])
                    and os.path.basename(p)[:-4] not in D.THANH_CA and sf.info(p).duration >= CAT_S + 5)
        rng.shuffle(ds)
        got, nhom = 0, collections.Counter()
        for p in ds:
            thu_muc = os.path.basename(os.path.dirname(p))
            y = cat_giua(p)
            if nhom[thu_muc] >= 4 or D.longest_zero_run(y) > D.LOAI_MAX_ZERO_S or D.rms(y) < 1e-3:   # ≤ 4 tệp / kho nhạc; không lấy đoạn im
                continue
            ten = os.path.basename(p)[:-4]
            g.them("loai", ca, "en", y, {"id": ten, "source": "musan_music", "person": f"musan:{ten}"}, "04", False,
                   [ent(p, mm[p]["license"], f"MUSAN/music/{thu_muc}")], {"cat_giua_s": CAT_S, "co_hat_annotations": co_hat}, {})
            nhom[thu_muc] += 1
            got += 1
            if got == 12:
                break
        assert got == 12, (ca, got)


def the_bo(g: Ghi) -> None:
    by = collections.Counter((c["ca"], c["mong_doi"], c["ca_bien"]) for c in g.cases)
    L = [f"# Thẻ bộ `{BO}`", "",
         "- **Là gì:** bộ kiểm chủ động cho 3 hàng Constraint chưa có bài kiểm trên dữ liệu P1: Ngôn ngữ vi / en, Nói chồng ≤ 10 %, Loại âm thanh (nhạc).",
         f"- **Ngày dựng:** {datetime.date.today().isoformat()} · **Ai làm:** Huấn (P1) duyệt kế hoạch 26/11; mã do Claude Code (Claude Fable 5.2) viết.",
         f"- **Cách làm:** `experts/P1/coverage/bo_sung_hang.py build` (seed {SEED}); mọi tệp chuẩn hoá RMS {D.LEVEL_DBFS:g} dBFS, wav 16 kHz mono 16 bit.",
         "- **Nguồn + giấy phép:** FLEURS vi_vn / en_us dev + test (CC BY 4.0; chỉ câu có SNR sẵn ≥ 25 dB) · MUSAN speech / librivox (Public Domain) "
         "· MUSAN music (CC BY / BY-SA / PD; bỏ ND / NC và bỏ thánh ca). Từng tệp: `provenance.jsonl`.",
         "- **Nguồn nhãn:** ngôn ngữ và có hát / không lời lấy từ `ANNOTATIONS` của MUSAN; ngôn ngữ FLEURS từ tên tập. Tỉ lệ nói chồng là **số đo** "
         "trên từng nguồn trước khi trộn (VAD năng lượng `k1.speech_mask`, khác phương pháp với pyannote được thử), ghi ở `do_lai.noi_chong_pct_do`.",
         f"- **Đáp án nói chồng:** đo ≤ {100 * KHONG_CB:g} % → không được cảnh báo · ≥ {100 * PHAI_CB:g} % → phải cảnh báo · ở giữa → ca biên, không chấm.",
         "- **Cờ:** mọi bản ghi `ngoai_pham_vi` (chỉ thử K1). `meta.lang` của tệp không phải vi / en đặt `en` vì `kiem_bo` chỉ nhận vi | en; "
         "ngôn ngữ thật ở `meta.ngon_ngu_that`.",
         "- **Hạn chế:** FLEURS là lời đọc, một câu / tệp; FLEURS vi chỉ có giọng nam nên cặp nói chồng vi là hai giọng nam · hai người nói cùng mức RMS "
         "· nhãn ngôn ngữ / có hát của MUSAN chưa có phép đo độc lập · không có podcast / phát thanh / lồng tiếng · không có thì thầm / la hét.",
         f"- **Số tệp:** {len(g.cases)} · **số người nói (person):** {len({m['meta']['person'] for m in g.man})}",
         "- **Đã kiểm:** *(điền sau)* `bo_sung_hang.py kiem` · `kiem_bo --moi` · `kiem_bo --tat-ca` · `lockbox check`", "",
         "| ca | đáp án | ca biên | số tệp |", "|---|---|---|---|"]
    L += [f"| {ca} | {md} | {'có' if b else ''} | {k} |" for (ca, md, b), k in sorted(by.items())]
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def build() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    rng = random.Random(SEED)
    can = 12 + 2 * 12                                                # 12 câu cho ngonngu + 12 cặp cho noichong (FLEURS vi chỉ có 59 câu sạch)
    fl = {lang: fleurs(lang, can, rng) for lang in ("vi", "en")}
    for lang in fl:                                                   # kiểm nguyên liệu TRƯỚC khi ghi bất kỳ tệp nào
        assert len(fl[lang]) == can and len({b["id"] for b in fl[lang]}) == can, (lang, len(fl[lang]), can)
    g = Ghi()
    dung_ngonngu(g, fl, rng)
    dung_noichong(g, fl)
    dung_loai(g, rng)
    assert len({m["id"] for m in g.man}) == len(g.man)
    dump = lambda name, obj: json.dump(obj, open(os.path.join(OUT, name), "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # noqa: E731
    dump("manifest.json", g.man)
    dump("cases.json", g.cases)
    with open(os.path.join(OUT, "provenance.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in g.prov)
    the_bo(g)
    print(f"{len(g.man)} tệp → {D.rel(OUT)}")
    print(collections.Counter((c["ca"], c["mong_doi"], c["ca_bien"]) for c in g.cases))


# ----------------------------------------------------------------------------- kiểm đầu ra
def kiem() -> None:
    P = [json.loads(l) for l in open(os.path.join(OUT, "provenance.jsonl"), encoding="utf-8")]
    loi, do = [], collections.defaultdict(list)
    for p in P:
        path = os.path.join(D.ROOT, p["file"])
        y, sr = sf.read(path, dtype="float64")
        if sr != SR or y.ndim != 1 or D.sha256(path) != p["sha256"] or np.max(np.abs(y)) >= 1.0:
            loi.append((p["id"], "định dạng / sha256 / cắt đỉnh"))
        if p["ca"].startswith("noichong."):
            a, b = (D.normalize(D.load(os.path.join(D.ROOT, s["path"])))[0] for s in p["nguon"])
            lech = int(round(p["tham_so"]["lech_b_s"] * SR / HOP))
            z = D.normalize(tron(a, b, lech))[0]
            if len(z) != len(y) or np.corrcoef(z, y)[0, 1] < 0.9999:                       # tệp đầu ra đúng là A + B lệch như ghi
                loi.append((p["id"], "tệp không khớp phép trộn ghi trong provenance"))
            tl = 100 * ti_le_chong(mat_na(a), mat_na(b), lech)
            if abs(tl - p["do_lai"]["noi_chong_pct_do"]) > 0.05:
                loi.append((p["id"], f"tỉ lệ chồng đo lại {tl:.2f} ≠ ghi {p['do_lai']['noi_chong_pct_do']}"))
            do[p["ca"]].append(p["do_lai"]["noi_chong_pct_do"])
        elif p["ca"].startswith("loai.") and D.longest_zero_run(y) > D.LOAI_MAX_ZERO_S:
            loi.append((p["id"], "đoạn im dài"))
    print(f"{len(P)} tệp · lỗi {len(loi)}", loi[:8])
    for ca in sorted(do):
        v = np.array(do[ca])
        print(f"  {ca}: tỉ lệ chồng đo {v.min():.1f}–{v.max():.1f} % (trung vị {np.median(v):.1f}) · n = {len(v)}")
    if loi:
        raise SystemExit(1)
    print("kiem dat")


# ----------------------------------------------------------------------------- chạy K1 mở rộng + tóm
def run() -> None:
    from vaudit import k1
    cases = json.load(open(os.path.join(OUT, "cases.json"), encoding="utf-8"))
    os.makedirs(KIEM, exist_ok=True)
    done = {json.loads(x)["file"] for x in open(RAW_KQ, encoding="utf-8")} if os.path.exists(RAW_KQ) else set()
    todo = [c for c in cases if c["file"] not in done]
    for i, c in enumerate(todo):
        try:
            r = k1.check(os.path.join(OUT, c["file"]), extended=True, events=None)
            kenh = (r.get("measures", {}).get("kenh") or [{}])[0]
            row = {"file": c["file"], "decision": r["decision"], "reasons": r["reasons"], "warnings": r["warnings"], "not_impl": r["not_impl"],
                   "noi_chong_pct": kenh.get("noi_chong_pct"), "ngon_ngu": kenh.get("ngon_ngu")}
        except Exception as e:  # noqa: BLE001  ghi lỗi từng tệp, chạy tiếp
            row = {"file": c["file"], "loi": f"{type(e).__name__}: {e}"[:400]}
        with open(RAW_KQ, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"{i + 1}/{len(todo)} {c['file']} → {row.get('decision', row.get('loi'))}", flush=True)
    print(tom())


def tom() -> str:
    cases = {c["file"]: c for c in json.load(open(os.path.join(OUT, "cases.json"), encoding="utf-8"))}
    rows = {json.loads(x)["file"]: json.loads(x) for x in open(RAW_KQ, encoding="utf-8")}
    g = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))        # ca → lang → [đúng, chấm, biên]
    for f, c in cases.items():
        r = rows.get(f)
        if r is None or "loi" in r:
            continue
        lang = "vi" if "_vi_" in f else "en"
        k = cham(c, r)
        t = g[(c["hang"], c["ca"], c["mong_doi"])][lang]
        t[2] += k is None
        t[1] += k is not None
        t[0] += bool(k)
    ten = {"ngonngu": "Ngôn ngữ vi / en", "noichong": "Nói chồng ≤ 10 %", "loai": "Loại âm thanh"}
    L = [f"# K1 chế độ mở rộng trên `{BO}`", "", f"{len(rows)}/{len(cases)} tệp đã chạy · `vaudit.k1.check(extended=True)` · lỗi chạy: "
         f"{sum('loi' in r for r in rows.values())}", "", "| hàng | ca | đáp án | vi | en | cả hai (Wilson 95 %) | ca biên |", "|---|---|---|---|---|---|---|"]
    for (h, ca, md), d in sorted(g.items()):
        k, n = sum(v[0] for v in d.values()), sum(v[1] for v in d.values())
        o = lambda lg: f"{d[lg][0]}/{d[lg][1]}" if d[lg][1] else "–"  # noqa: E731
        L.append(f"| {ten[h]} | {ca} | {md} | {o('vi')} | {o('en')} | {wilson(k, n)} | {sum(v[2] for v in d.values()) or ''} |")
    kh = collections.Counter(w.split(" (")[0].split(" ước")[0] for f, r in rows.items() for w in r.get("warnings", []))
    L += ["", "Cảnh báo xuất hiện (mọi tệp): " + ", ".join(f"{k} ×{v}" for k, v in kh.most_common())]
    out = "\n".join(L) + "\n"
    open(BAO, "w", encoding="utf-8").write(out)
    return out


def selftest() -> None:
    ma = np.array([1] * 100 + [0] * 20, bool)
    mb = np.array([1] * 100, bool)
    assert ti_le_chong(ma, mb, 0) == 1.0 and ti_le_chong(ma, mb, 200) == 0.0
    assert abs(ti_le_chong(ma, mb, 50) - 50 / 150) < 1e-9
    lech, do = tim_lech(ma, mb, 0.2)
    assert abs(do - 0.2) < 0.02 and tim_lech(ma, mb, 0)[1] == 0.0, (lech, do)
    assert dap_an_chong(0.05) == ("accept", False) and dap_an_chong(0.20) == ("canh_bao", False) and dap_an_chong(0.10) == ("accept", True)
    y = tron(np.ones(HOP * 4), np.ones(HOP * 2), 3)
    assert y[HOP * 3] == 2.0 and y[0] == 1.0 and y[HOP * 4] == 1.0 and len(y) == HOP * 5 + int(0.3 * SR)
    c = {"hang": "noichong", "mong_doi": "canh_bao", "ca_bien": False}
    assert cham(c, {"warnings": ["nói chồng ước lượng 23 % thời lượng có tiếng nói"]}) and not cham(c, {"warnings": ["nhiều người nói"]})
    assert cham({**c, "mong_doi": "accept"}, {"warnings": []}) and cham({**c, "ca_bien": True}, {"warnings": []}) is None
    n = {"hang": "ngonngu", "mong_doi": "canh_bao", "ca_bien": False}
    assert cham(n, {"warnings": ["ngôn ngữ khác (nhận ra “de”, không phải tiếng Việt / Anh)"]})
    m = {"hang": "loai", "mong_doi": "04", "ca_bien": False}
    assert cham(m, {"decision": "04", "reasons": ["04 · “không phải ghi âm lời nói”"], "warnings": []}) and not cham(m, {"decision": "nhan", "reasons": []})
    print("bo_sung_hang selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"build": build, "kiem": kiem, "run": run, "tom": lambda: print(tom()), "selftest": selftest}[sys.argv[1]]()
