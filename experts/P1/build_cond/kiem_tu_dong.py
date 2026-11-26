"""Kiểm tự động bộ p1_dieukien_v1 + trang nghe HTML (thay cho việc tự dò tệp trong KIEM_NGHE.md).

Đo trên CHÍNH tệp wav đã ghi (không dùng hệ số g lúc trộn), so với đáp án:
  - chiếu lên lời sạch gốc:  a = <y,s>/<s,s>, phần dư r = y − a·s  →  SNR đo = 10·log10(E[a·s]/E[r]); nền % = 100·E[r]/E[y];
    vang: % năng lượng dư (phản xạ + đuôi vang) phải TĂNG theo RT60 đích.
  - WADA-SNR (vaudit.k1x, đo mù, không cần lời sạch): chỉ xét THỨ TỰ giảm dần theo SNR đích trong cùng tệp gốc.
  - lỗi lạ: cắt đỉnh (k1.clip_share), đỉnh, chuỗi số 0 dài, % khung có tiếng nói (k1.speech_mask), bước nhảy tại chỗ nối vòng
    nguồn ồn / nhạc (fit_len lặp nguồn ngắn → có thể «tách»).
Ghi: <bộ>/_kiem/kiem_tu_dong.json · KIEM_TU_DONG.md · pho_*.png (ảnh phổ) · <bộ>/KIEM_NGHE.html (nghe + chấm, lưu trình duyệt, xuất JSON).

Dùng:  .venv/Scripts/python.exe experts/P1/build_cond/kiem_tu_dong.py [run | selftest]
"""
from __future__ import annotations

import html
import json
import os
import re
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dung_bo import (LOAI_MAX_ZERO_S, OUT, ROOT, SR, T30_T20_TOL, XF, combine_snr, energy, expected, load, longest_zero_run,  # noqa: E402
                     native_snr, rms, rt60_schroeder)
from vaudit import k1, k1x  # noqa: E402

KIEM = os.path.join(OUT, "_kiem")
TOL_SNR_DB, TOL_NEN_PP, SEAM_RATIO, MIN_SPEECH_PCT, ZERO_RUN_S, EDGE_S = 1.0, 1.0, 10.0, 30.0, 0.1, 0.3


# ----------------------------------------------------------------------------- đo (hàm thuần)
def project(y: np.ndarray, s: np.ndarray) -> tuple[float, np.ndarray]:
    """Hệ số chiếu a và phần dư r = y − a·s (phần không phải lời sạch: ồn / nhạc / vang)."""
    a = float(np.dot(y, s) / (np.dot(s, s) + 1e-20))
    return a, y - a * s


def seam_ratio(r: np.ndarray, t: int, w: int = SR // 10) -> float:
    """|r[t] − r[t−1]| so với trung vị |Δr| quanh t (±0,1 s); > SEAM_RATIO → nghi có tiếng «tách» ở chỗ nối."""
    if not 1 <= t < len(r):
        return 0.0
    d = np.abs(np.diff(r[max(0, t - w): t + w]))
    return float(abs(r[t] - r[t - 1]) / (np.median(d) + 1e-12))


def eyring(L: float, W: float, H: float, a: float) -> float:
    """RT60 lý thuyết của phòng hộp (Eyring): 0,161·V / (−S·ln(1−α)) — tham khảo, độc lập với tín hiệu RIR."""
    V, S = L * W * H, 2 * (L * W + L * H + W * H)
    return float(0.161 * V / (-S * np.log(1 - a)))


def target_of(ca: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)(?:dB|pct|s)$", ca)
    return float(m.group(1)) if m else None


# ----------------------------------------------------------------------------- đo từng tệp
def measure(p: dict) -> dict:
    y = load(os.path.join(ROOT, p["file"]))
    truc, tgt = p["ca"].split(".")[0], target_of(p["ca"])
    db = k1.frame_db(y)
    sp, _ = k1.speech_mask(db)
    m = {"id": p["id"], "file": p["file"].split(f"{os.path.basename(OUT)}/")[1], "ca": p["ca"], "truc": truc, "dich": tgt,
         "mong_doi": p["mong_doi"], "ly_do": p["ly_do"], "base": p["goc"]["id"],
         "dinh": round(float(np.max(np.abs(y))), 4), "muc_dbfs": round(20 * np.log10(rms(y)), 2),
         "cat_dinh_pct": round(100 * k1.clip_share(y), 3), "so0_dai_s": round(longest_zero_run(y[int(EDGE_S * SR): -int(EDGE_S * SR)]), 3),
         "loi_noi_pct": round(100 * float(sp.mean()), 1), "wada_db": round(k1x.wada_snr(y), 2)}
    if truc == "goc":
        s0 = native_snr(y)
        m["snr_san_db"] = None if s0 is None else round(s0, 1)
    if truc == "vang":                                  # đo LẠI trên tệp RIR đã ghi (không đọc số trong provenance)
        src = p["nguon_tron"][0]
        h = load(os.path.join(ROOT, src["path"]))
        t20, t30 = rt60_schroeder(h), rt60_schroeder(h, lo_db=-35)
        m.update(rt60_t20_s=None if t20 is None else round(t20, 3), rt60_t30_s=None if t30 is None else round(t30, 3),
                 nhan_theo_t20=None if t20 is None else expected("vang", t20)[0],
                 rt60_eyring_s=round(eyring(*src["phong_m"], src["hap_thu"]), 3) if "phong_m" in src else None)
    if truc == "snr":
        m["snr_them_db"] = p["tham_so"].get("snr_them_db", tgt)   # phép chiếu lên gốc chỉ thấy ồn THÊM vào
    if truc in ("snr", "nen", "vang"):
        s = load(os.path.join(ROOT, p["goc"]["path"]))[: len(y)]
        a, r = project(y, s)
        m["chieu_snr_db"] = round(10 * np.log10(energy(a * s) / (energy(r) + 1e-20)), 2)
        m["du_pct"] = round(100 * energy(r) / energy(y), 2)
        m["noi_vong"] = [round(seam_ratio(r, t), 1) for t in seams(p, len(y))]
        if m["noi_vong"]:                               # mốc so sánh: cùng tỉ số tại 1 000 vị trí ngẫu nhiên của chính tệp
            pos = np.random.default_rng(0).integers(SR // 10, len(r) - SR // 10, 1000)
            m["noi_vong_p999"] = round(float(np.percentile([seam_ratio(r, int(t)) for t in pos], 99.9)), 1)
    return m


def seams(p: dict, n: int) -> list[int]:
    """Vị trí (mẫu) mà fit_len quay vòng nguồn ồn / nhạc trong tệp này."""
    out = []
    for src in p["nguon_tron"]:
        if "offset" not in src:
            continue
        L = sf.info(os.path.join(ROOT, src["path"])).frames - XF       # fit_len lặp bản loopable (ngắn hơn XF mẫu)
        out += list(range(L - src["offset"], n, L))
    return out


def flags(m: dict) -> list[str]:
    f = []
    if m["cat_dinh_pct"] > 0.1 or m["dinh"] > 0.99:
        f.append(f"cắt đỉnh {m['cat_dinh_pct']} % / đỉnh {m['dinh']}")
    if m["so0_dai_s"] > (LOAI_MAX_ZERO_S if m["truc"] == "loai" else ZERO_RUN_S):   # loai: cùng ngưỡng bộ dựng (ngắt câu sách nói)
        f.append(f"im lặng số 0 dài {m['so0_dai_s']} s")
    if m["truc"] == "goc" and m["loi_noi_pct"] < MIN_SPEECH_PCT:   # tệp biến đổi: ồn làm VAD năng lượng đếm thiếu → chỉ xét gốc
        f.append(f"gốc ít khung có tiếng nói ({m['loi_noi_pct']} %)")
    if m["truc"] == "vang":
        t20, t30 = m["rt60_t20_s"], m["rt60_t30_s"]
        if m["nhan_theo_t20"] != m["mong_doi"]:
            f.append(f"RT60 T20 đo lại {t20} s → nhãn {m['nhan_theo_t20']} ≠ {m['mong_doi']}")
        if t20 and t30 and abs(t30 - t20) > T30_T20_TOL * t20:
            f.append(f"EDC không thẳng: T20 {t20} s, T30 {t30} s → RT60 không tin được")
    if m["truc"] == "snr" and abs(m["chieu_snr_db"] - m["snr_them_db"]) > TOL_SNR_DB:
        f.append(f"SNR ồn thêm đo {m['chieu_snr_db']} dB ≠ {m['snr_them_db']:g}")
    if m["truc"] == "nen" and abs(m["du_pct"] - m["dich"]) > TOL_NEN_PP:
        f.append(f"nền đo {m['du_pct']} % ≠ đích {m['dich']:g}")
    if max(m.get("noi_vong") or [0]) > max(SEAM_RATIO, m.get("noi_vong_p999", 0)):
        f.append(f"bước nhảy chỗ nối vòng ×{max(m['noi_vong'])} (vị trí thường p99,9 ×{m['noi_vong_p999']})")
    return f


def base_noise_risk(ms: list[dict]) -> list[dict]:
    """Gốc FLEURS không sạch hẳn: SNR thật ≈ gộp(SNR ồn thêm đo bằng phép chiếu, SNR sẵn có của gốc). Gắn cờ tệp snr / goc mà nhãn
    suy theo SNR thật khác nhãn đang ghi (bản mới, không sửa ms)."""
    s0 = {m["base"]: m.get("snr_san_db") for m in ms if m["truc"] == "goc"}
    out = []
    for m in ms:
        b = s0.get(m["base"])
        if b is None or m["truc"] not in ("snr", "goc"):
            out.append(m)
            continue
        eff = b if m["truc"] == "goc" else combine_snr(m["chieu_snr_db"], b)
        lab = expected("snr", eff)[0]
        far = m["truc"] == "goc" or (m["dich"] - eff) > TOL_SNR_DB          # lệch < 1 dB ở đúng ngưỡng 10 / 15: đã là ca_bien
        extra = [f"gốc sẵn ồn (SNR ≈ {b} dB) → SNR thật ≈ {eff:.1f} dB → nhãn có thể là {lab}"] if lab != m["mong_doi"] and far else []
        out.append({**m, "snr_that_db": round(eff, 1), "co": m["co"] + extra})
    return out


# ----------------------------------------------------------------------------- ảnh phổ
def spec_sheet(items: list[dict], path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = 3
    rows = int(np.ceil(len(items) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(15, 2.3 * rows), squeeze=False)
    for ax in axs.flat:
        ax.axis("off")
    for ax, m in zip(axs.flat, items):
        y = load(os.path.join(OUT, m["file"]))
        ax.specgram(y, NFFT=512, Fs=SR, noverlap=384, cmap="magma", vmin=-130, vmax=-30)
        ax.axis("on")
        ax.set_ylim(0, 8000)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{m['ca']} · {m['id']} · đáp án {m['mong_doi']}" + (" ⚑" if m["co"] else ""), fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def sheets(ms: list[dict]) -> list[str]:
    order = {"goc": 0, "snr": 1, "nen": 2, "vang": 3}
    out = []
    for lang in ("vi", "en"):
        base = next(m["base"] for m in ms if m["truc"] == "goc" and f"_{lang}_" in m["id"])
        g = sorted((m for m in ms if m["base"] == base), key=lambda m: (order[m["truc"]], m["dich"] or 0))
        out.append(os.path.join(KIEM, f"pho_{lang}_mot_tep_goc.png"))
        spec_sheet(g, out[-1], f"Một tệp gốc ({base}) qua mọi điều kiện: ồn tăng dần ↓ SNR, nhạc nền ↑ %, vang ↑ RT60")
    loai = [m for m in ms if m["ca"] == "loai.nhac"][:6] + [m for m in ms if m["ca"] == "loai.sach_noi"][:6]
    out.append(os.path.join(KIEM, "pho_loai.png"))
    spec_sheet(loai, out[-1], "loai: nhạc không lời (trên) · sách nói (dưới)")
    bad = [m for m in ms if m["co"]][:15]
    if bad:
        out.append(os.path.join(KIEM, "pho_bi_co.png"))
        spec_sheet(bad, out[-1], "Các tệp bị cờ")
    return out


# ----------------------------------------------------------------------------- báo cáo + trang nghe
def report(ms: list[dict], imgs: list[str]) -> str:
    bad = [m for m in ms if m["co"]]
    L = ["# Kiểm tự động: p1_dieukien_v1", "", f"Script `experts/P1/build_cond/kiem_tu_dong.py` · {len(ms)} tệp · **{len(bad)} tệp bị cờ**", "",
         "## Sai lệch đo trên tệp đã ghi so với đích (chiếu lên lời sạch gốc)", "", "| ca | n | đo: trung vị | đo: min–max | đích |", "|---|---|---|---|---|"]
    cells: dict = {}
    for m in ms:
        if m["truc"] in ("snr", "nen", "vang"):
            cells.setdefault(m["ca"], []).append(m.get("snr_that_db", m["chieu_snr_db"]) if m["truc"] == "snr" else m["du_pct"])
    for ca in sorted(cells, key=lambda c: (c.split(".")[0], target_of(c))):
        v = np.array(cells[ca])
        unit = " dB" if ca.startswith("snr") else " %"
        tgt = "(% vang, tham khảo)" if ca.startswith("vang") else f"{target_of(ca):g}{unit}" + (" (SNR thật = gộp ồn thêm + ồn sẵn)" if ca.startswith("snr") else "")
        L.append(f"| {ca} | {len(v)} | {np.median(v):.2f}{unit} | {v.min():.2f}–{v.max():.2f} | {tgt} |")
    g = sorted((m for m in ms if m["truc"] == "goc"), key=lambda m: m["snr_san_db"] if m["snr_san_db"] is not None else 99)
    L += ["", "## SNR sẵn có của tệp gốc FLEURS (ước lượng thô, mặt nạ năng lượng)", "",
          "| tệp gốc | SNR sẵn có | WADA | % khung tiếng nói |", "|---|---|---|---|"]
    L += [f"| `{m['file']}` | {m['snr_san_db']} dB | {m['wada_db']} dB | {m['loi_noi_pct']} % |" for m in g]
    v = [m for m in ms if m["truc"] == "vang"]
    agree = sum(m["nhan_theo_t20"] == m["mong_doi"] for m in v)
    lin = sum(bool(m["rt60_t20_s"] and m["rt60_t30_s"]) and abs(m["rt60_t30_s"] - m["rt60_t20_s"]) <= T30_T20_TOL * m["rt60_t20_s"] for m in v)
    L += ["", f"## Vang: T20 đo lại trên tệp RIR cho cùng nhãn **{agree}/{len(v)}** · EDC thẳng (|T30 − T20| ≤ {T30_T20_TOL:.0%}) **{lin}/{len(v)}**", "",
          "| ca | T20 trung vị (min–max) | T30 trung vị | Eyring trung vị (tham khảo) |", "|---|---|---|---|"]
    med = lambda xs: f"{np.median(xs):.2f} s" if len(xs) else "—"
    for ca in sorted({m["ca"] for m in v}):
        t20 = np.array([m["rt60_t20_s"] for m in v if m["ca"] == ca and m["rt60_t20_s"]])
        L.append(f"| {ca} | {med(t20)} ({t20.min():.2f}–{t20.max():.2f}) | {med([m['rt60_t30_s'] for m in v if m['ca'] == ca and m['rt60_t30_s']])} "
                 f"| {med([m['rt60_eyring_s'] for m in v if m['ca'] == ca and m['rt60_eyring_s']])} |")
    L += ["", "## Tệp bị cờ", ""] + ([f"- `{m['file']}` ({m['ca']}): " + "; ".join(m["co"]) for m in bad] or ["(không có)"])
    L += ["", "## Ảnh phổ", ""] + [f"- `_kiem/{os.path.basename(i)}`" for i in imgs]
    L += ["", "## Giới hạn", "- Đo chiếu dùng chính lời sạch gốc → kiểm **tệp ghi ra đúng mức trộn**, không kiểm cảm nhận của tai.",
          "- SNR sẵn có: ước lượng thô theo khung, không phải bộ đo SNR của K1 (`vaudit.cond`, cần mô hình XLS-R) → chỉ để gắn cờ.",
          "- WADA chỉ ghi tham khảo: lệch mạnh theo loại ồn (MUSAN / DEMAND xen kẽ) nên không dùng để so thứ tự.",
          "- Không kiểm được bằng số: nhạc có lời lẫn vào `loai.nhac`, giọng đọc sách có ồn nền lạ → cần mô hình (AST) hoặc tai."]
    return "\n".join(L) + "\n"


def sample_files() -> list[str]:
    t = open(os.path.join(OUT, "KIEM_NGHE.md"), encoding="utf-8").read()
    return re.findall(r"\| `([^`]+\.wav)` \|", t)


def row_html(i: int, m: dict, ref: str | None) -> str:
    meas = f"WADA {m['wada_db']} dB" + (f" · SNR chiếu {m['chieu_snr_db']} dB · dư {m['du_pct']} %" if "du_pct" in m else "")
    co = f"<br><span class=co>⚑ {html.escape('; '.join(m['co']))}</span>" if m["co"] else ""
    ref_td = f'<audio controls preload="none" src="{ref}"></audio>' if ref else "—"
    return (f'<tr data-f="{html.escape(m["file"])}"><td>{i + 1}</td><td>{m["ca"]}<br><small>{m["file"]}</small></td>'
            f'<td><b>{m["mong_doi"]}</b> {html.escape(m["ly_do"] or "")}</td><td><small>{meas}</small>{co}</td>'
            f'<td><audio controls preload="none" src="{m["file"]}"></audio></td><td>{ref_td}</td>'
            f'<td><label><input type=radio name=k{i} value=dat>đạt</label> <label><input type=radio name=k{i} value=khong>không</label></td>'
            f'<td><input class=gc size=24></td></tr>')


def page(ms: list[dict]) -> str:
    by_file = {m["file"]: m for m in ms}
    goc = {m["base"]: m["file"] for m in ms if m["truc"] == "goc"}
    pick = sample_files()
    pick += [m["file"] for m in ms if m["co"] and m["file"] not in pick]
    rows = [row_html(i, by_file[f], goc.get(by_file[f]["base"]) if by_file[f]["truc"] != "goc" else None) for i, f in enumerate(pick)]
    return PAGE.replace("{ROWS}", "\n".join(rows)).replace("{N}", str(len(pick)))


PAGE = """<!DOCTYPE html><html lang="vi"><meta charset="utf-8"><title>Kiểm nghe p1_dieukien_v1</title>
<style>body{font:14px system-ui;margin:16px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px;vertical-align:top}
tr.done{background:#eef8ee}tr.fail{background:#fdeaea}.co{color:#b00}audio{width:230px;height:32px}small{color:#555}</style>
<h2>Kiểm nghe: p1_dieukien_v1 ({N} tệp)</h2>
<p>Nghe cột «tệp», so với cột «gốc sạch» (cùng câu, chưa trộn). Đạt = nghe đúng điều kiện như đáp án (mức ồn / có nhạc nền / vang) và không có lỗi lạ
(click, méo, im lặng). Kết quả tự lưu trong trình duyệt; xong bấm <b>Tải kết quả</b> → lưu <code>ket_qua_nghe.json</code> vào thư mục <code>_kiem/</code>.</p>
<p>Người kiểm: <input id=who> <button onclick=dl()>Tải kết quả</button> <span id=cnt></span></p>
<table><tr><th>#</th><th>ca / tệp</th><th>đáp án</th><th>đo tự động</th><th>tệp</th><th>gốc sạch</th><th>chấm</th><th>ghi chú</th></tr>
{ROWS}</table>
<script>
const K='kiem_nghe_p1_dieukien_v1',S=JSON.parse(localStorage[K]||'{}');
const rows=[...document.querySelectorAll('tr[data-f]')];
function paint(){let n=0;rows.forEach(r=>{const v=(S[r.dataset.f]||{}).k;r.className=v=='dat'?'done':v=='khong'?'fail':'';if(v)n++});
document.getElementById('cnt').textContent=n+'/'+rows.length+' đã chấm'}
rows.forEach(r=>{const f=r.dataset.f,s=S[f]||{};
 r.querySelectorAll('input[type=radio]').forEach(x=>{x.checked=x.value==s.k;x.onchange=()=>{S[f]={...S[f],k:x.value};save()}});
 const g=r.querySelector('.gc');g.value=s.g||'';g.oninput=()=>{S[f]={...S[f],g:g.value};save()}});
const w=document.getElementById('who');w.value=localStorage[K+'_who']||'';w.oninput=()=>localStorage[K+'_who']=w.value;
function save(){localStorage[K]=JSON.stringify(S);paint()}
function dl(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({nguoi_kiem:w.value,ngay:new Date().toISOString(),ket_qua:S},null,1)],{type:'application/json'}));a.download='ket_qua_nghe.json';a.click()}
paint();
</script></html>"""


def run() -> None:
    os.makedirs(KIEM, exist_ok=True)
    prov = [json.loads(x) for x in open(os.path.join(OUT, "provenance.jsonl"), encoding="utf-8")]
    ms = []
    for i, p in enumerate(prov):
        m = measure(p)
        ms.append({**m, "co": flags(m)})
        if i % 64 == 0:
            print(f"{i}/{len(prov)}", flush=True)
    ms = base_noise_risk(ms)
    imgs = sheets(ms)
    json.dump({"tep": ms}, open(os.path.join(KIEM, "kiem_tu_dong.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(os.path.join(KIEM, "KIEM_TU_DONG.md"), "w", encoding="utf-8").write(report(ms, imgs))
    open(os.path.join(OUT, "KIEM_NGHE.html"), "w", encoding="utf-8").write(page(ms))
    print(f"{sum(bool(m['co']) for m in ms)} tệp bị cờ → {os.path.relpath(KIEM, ROOT)}")


def selftest() -> None:
    rng = np.random.default_rng(0)
    s, n = rng.standard_normal(SR * 3), rng.standard_normal(SR * 3)
    g = np.sqrt(energy(s) / (energy(n) * 10 ** (12 / 10)))
    a, r = project(0.5 * (s + g * n), s)
    assert abs(10 * np.log10(energy(a * s) / energy(r)) - 12) < 0.1
    x = np.sin(np.arange(SR) / 20)
    assert seam_ratio(x, SR // 2) < 3 and seam_ratio(np.r_[x[:SR // 2], x[:SR // 2] + 1], SR // 2) > 2 * SEAM_RATIO
    assert abs(combine_snr(20, 20) - (20 - 10 * np.log10(2))) < 1e-9 and combine_snr(10, 200) > 9.99
    assert abs(eyring(10, 10, 3, 0.2) - 0.161 * 300 / (-320 * np.log(0.8))) < 1e-9
    assert target_of("snr.12dB") == 12 and target_of("vang.rt60_1.5s") == 1.5 and target_of("goc") is None
    print("selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"run": run, "selftest": selftest}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
