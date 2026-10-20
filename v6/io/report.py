# -*- coding: utf-8 -*-
"""Bao cao (K9): HTML tieng Viet tu CHINH ho so (khong them/bot so nao) -> PDF qua Chrome headless (hoac Edge);
kem bang_moc.csv va bang_so_do.csv. Output chinh: nhan 01–04 (04 + ly do), moi thao tac: loai · pham vi · moc
bat dau–ket thuc · muc do · kenh trai / phai (chi khi hai kenh phan tich rieng) · anh · giai thich. Output phu: canh bao · bang moc ·
diem va nguong tung buoc · bang so do (gia tri, trung vi, vi tri phan tram trong kho tham chieu) · tuyen bo gioi han.
PDF gop moi phan tren + ma bam tep (SHA-256 du 64 ky tu) + phien ban mo hinh va nguong (ten tep + sha1) + thoi diem chay.
  python -m v6.io.report <ho_so.json> [--pdf]   (dung lai bao_cao.html / .pdf canh ho so)"""
import argparse
import csv
import html
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.io.common import CHROME, OP, SEV_COLOR, TMP_DIR, mmss, vn

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
LABEL_COLOR = {"01": "#2f7d4f", "02": "#b7791f", "03": "#b3261e", "04": "#5b6470", None: "#5b6470"}
CSS = """
:root{--ink:#16181d;--ink-2:#3d434d;--ink-3:#6b7280;--line:#d9dde3;--bg:#ffffff;--panel:#f5f6f8;--accent:#1f4f8f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#e8eaee;--ink-2:#c3c8d0;--ink-3:#9aa1ab;--line:#3a3f47;--bg:#15171b;--panel:#1d2026;--accent:#8fb4ea}}
@media print{:root{--ink:#16181d;--ink-2:#3d434d;--ink-3:#6b7280;--line:#d9dde3;--bg:#ffffff;--panel:#f5f6f8;--accent:#1f4f8f}}
@page{size:A4;margin:14mm 12mm}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 "Segoe UI","Noto Sans",Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:20px 16px 40px}
header.top{border-bottom:2px solid var(--ink);padding-bottom:10px;margin-bottom:18px}
.kicker{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
h1{font-size:22px;margin:4px 0 6px}h2{font-size:16px;margin:26px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:14px;margin:0 0 6px}
.meta{display:flex;flex-wrap:wrap;gap:4px 18px;color:var(--ink-2);font-size:12px}
.label{display:inline-flex;align-items:baseline;gap:10px;padding:10px 16px;border-radius:6px;color:#fff;font-size:20px;font-weight:700}
.label small{font-size:12px;font-weight:500;opacity:.9}
.reasons{margin:10px 0 0;padding-left:18px}.reasons li{margin:3px 0}
.advice{color:var(--ink-2);font-style:italic}
article.op{border:1px solid var(--line);border-left:5px solid var(--c);border-radius:6px;padding:12px 14px;margin:14px 0;
  background:var(--panel);break-inside:avoid;page-break-inside:avoid}
article.op img,figure img{width:100%;height:auto;border:1px solid var(--line);border-radius:4px;background:#fff}
.sev{color:var(--c);font-weight:700}
p.explain{margin:8px 0 6px}
table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-size:12px}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left;vertical-align:top}
th{color:var(--ink-3);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.hot{color:#b3261e;font-weight:600}.muted{color:var(--ink-3)}
ul.warn li{margin:2px 0}
figure{margin:12px 0}figcaption{font-size:11px;color:var(--ink-3)}
footer{margin-top:26px;font-size:11px;color:var(--ink-3);border-top:1px solid var(--line);padding-top:8px}
code.hash{font-size:11px;word-break:break-all}
.pill{display:inline-block;padding:0 6px;border-radius:9px;border:1px solid var(--line);font-size:11px;color:var(--ink-2)}
"""


def e(x) -> str:
    return html.escape("" if x is None else str(x))


def _num(x, d: int = 3) -> str:
    if x is None:
        return "—"
    return vn(float(x), d) if isinstance(x, (int, float)) and not isinstance(x, bool) else e(x)


def _dir(s: dict) -> str:
    """'cao hơn 97 %' / 'thấp hơn 99 %' (vi tri trong kho tham chieu, dung chieu)."""
    if s.get("so_voi_kho"):
        return s["so_voi_kho"]
    if "cao_hon_pct" in s and "thap_hon_pct" in s:
        hi, lo = float(s["cao_hon_pct"]), float(s["thap_hon_pct"])
        return f"cao hơn {vn(hi, 0)} %" if hi >= lo else f"thấp hơn {vn(lo, 0)} %"
    return ""


def _kenh(x: dict, ho: dict) -> str:
    """Ten kenh theo hop dong: 'trai / phai' chi khi hai kenh phan tich rieng (ho so cu: 'kenh 1/2')."""
    if "kenh_ten" in x:
        return f"kênh {x['kenh_ten']}" if x["kenh_ten"] else ""
    return f"kênh {x['kenh'] + 1}" if (ho["tep"].get("so_kenh_phan_tich") or 0) > 1 and isinstance(x.get("kenh"), int) else ""


def _signs_table(signs: list) -> str:
    if not signs:
        return ""
    rows = "".join(
        f"<tr><td>{e(s['ten'])}<br><span class='muted'>{e(s['ma'])}</span></td>"
        f"<td class='num'>{_num(s['gia_tri'])}{(' ' + e(s['don_vi'])) if s['don_vi'] else ''}</td>"
        f"<td class='num'>{_num(s['trung_vi_that'])}</td>"
        f"<td class='num{' hot' if s['bat_thuong'] else ''}'>{vn(s['phan_vi_that'], 1)} %</td><td>{e(_dir(s))}</td><td>{e(s['kho'])}</td></tr>"
        for s in signs)
    return ("<table><tr><th>Dấu hiệu</th><th>Giá trị</th><th>Trung vị kho tham chiếu</th><th>Phân vị</th><th>So với kho</th><th>Kho</th></tr>"
            f"{rows}</table>")


def label01_text(ho: dict) -> str:
    """Cau nhan 01 theo SO thao tac THUC duoc soi (diem_nguong.thao_tac_soi cua ho so v9). Khong co khoi nay: ho so v6 -> 6 thao tac (he v6);
    ho so v9 cu -> KHONG neu so (khong biet nhanh nao da tat - truoc 11/10 bo nguong that khong soi giong may ma van ghi '8')."""
    soi = ho["output_phu"]["diem_nguong"].get("thao_tac_soi")
    if not isinstance(soi, dict):
        if not str(ho.get("he_thong", {}).get("ten", "")).startswith("VoiceAudit v9"):
            return "Không thấy dấu vết của 6 thao tác được soi — đây không phải chứng nhận bản thật."
        return "Không thấy dấu vết của các thao tác được soi (hồ sơ không ghi danh sách thao tác được soi) — đây không phải chứng nhận bản thật."
    off = soi.get("khong_soi") or []
    tail = f"; không soi: {', '.join(off)}" if off else ""
    return f"Không thấy dấu vết của {soi['so_duoc_soi']} thao tác được soi{tail} — đây không phải chứng nhận bản thật."


def _main_section(ho: dict) -> str:
    oc = ho["output_chinh"]
    lab = oc["nhan"]["ma"]
    if ho["trang_thai"] == "tu_choi":
        head = f"<div class='label' style='background:{LABEL_COLOR[None]}'>Từ chối nhận</div>"
    elif ho["trang_thai"] == "loi":
        head = f"<div class='label' style='background:{LABEL_COLOR[None]}'>Lỗi xử lý</div>"
    else:
        head = (f"<div class='label' data-nhan='{e(lab)}' style='background:{LABEL_COLOR[lab]}'>{e(lab)} · "
                f"{e(oc['nhan']['ten'])}</div>")
    parts = ["<h2>Output chính</h2>", head]
    reasons = oc["ly_do_04"] or ho.get("tu_choi_ly_do", [])
    if reasons:
        parts.append("<ul class='reasons'>" + "".join(f"<li data-duong='{e(r.get('duong'))}'>{e(r['noi_dung'])}</li>"
                                                     for r in reasons) + "</ul>")
    if lab == "04":
        parts.append(f"<p class='advice'>{e(oc.get('de_nghi'))}</p>")
    if lab == "01":
        parts.append(f"<p>{e(label01_text(ho))}</p>")
    for o in oc["thao_tac"]:
        c = SEV_COLOR[o["muc_do"]]
        kenh = _kenh(o, ho)
        kenh = f" · {kenh}" if kenh else ""
        parts.append(
            f"<article class='op' data-op='{e(o['id'])}' style='--c:{c}'>"
            f"<h3>{e(o['id'])}{e(kenh)} · {e(o['loai'])} · {e(o['pham_vi'])} · "
            f"<span data-moc>{e(o['bat_dau'])} – {e(o['ket_thuc'])}</span> · <span class='sev'>{e(o['muc_do'])}</span></h3>"
            f"<img src='{e(o['anh'])}' alt='Ảnh dãy âm {e(o['id'])}: {e(o['loai'])}, {e(o['pham_vi'])}'>"
            f"<p class='explain'>{e(o['giai_thich'])}</p>{_signs_table(o['dau_hieu'])}</article>")
    for p in ho["output_phu"]["anh_tong_quan"]:
        parts.append(f"<figure><img src='{e(p)}' alt='Tổng quan cả tệp'><figcaption>Tổng quan cả tệp: dạng sóng, phổ mel, "
                     f"điểm khung của các lớp cục bộ và các thao tác trong phiếu; vạch chéo = 0,3 s đầu/cuối không soi."
                     f"</figcaption></figure>")
    return "\n".join(parts)


def _k3_table(k3: dict, pre: str) -> str:
    rows = []
    for c, v in k3.items():
        vx = v.get("vung_xam")
        rows.append(f"<tr><td>{e(OP[c][0]) if c in OP else e(c)} cả file</td><td class='num'>{_num(v.get('p'))}</td>"
                    f"<td class='num'>{_num(v.get('p_hieu_chinh'))}</td><td class='num'>{_num(v.get('nguong_dong'), 2)}</td>"
                    f"<td class='num'>{(_num(vx[0], 2) + '–' + _num(vx[1], 2)) if vx else '—'}</td><td>{e(v.get('ghi_chu', ''))}</td></tr>")
    return (f"<h3>{e(pre)}K3 · Thao tác cả file (luật: xác suất ≥ 0,6; vùng xám 0,4–0,6)</h3><table><tr><th>Lớp</th><th>Điểm thô</th>"
            "<th>Xác suất hiệu chỉnh</th><th>Ngưỡng dòng</th><th>Vùng xám</th><th>Ghi chú</th></tr>" + "".join(rows) + "</table>")


def _steps(ho: dict) -> str:
    dn = ho["output_phu"]["diem_nguong"]
    k1 = dn["K1"]
    out = ["<h3>K1 · Kiểm điều kiện đầu vào</h3>",
           f"<p>Quyết định: <b>{e(k1.get('quyet_dinh'))}</b>" +
           (" · lý do: " + e("; ".join(k1.get("ly_do", []))) if k1.get("ly_do") else "") + "</p>"]
    if k1.get("ghi_chu_khong_thuoc_hop_dong"):
        out.append("<p class='muted'>Ghi chú đo K1 (không phải cảnh báo của hợp đồng): " + e("; ".join(k1["ghi_chu_khong_thuoc_hop_dong"])) + "</p>")
    for k in dn["kenh"]:
        kt = _kenh(k, ho)
        pre = f"{kt[:1].upper() + kt[1:]} · " if kt else ""
        k3, k5, k7 = k["K3"], k["K5"], k["K7"]
        if k.get("ngon_ngu_nguong"):
            out.append(f"<p>{e(pre)}Bộ ngưỡng theo ngôn ngữ: <b>{e(k['ngon_ngu_nguong'])}</b></p>")
        out.append(_k3_table(k3, pre))
        out.append(f"<h3>{e(pre)}K5 · Cổng sàng lọc</h3><p>Điểm {_num(k5.get('diem'))} · ngưỡng {_num(k5.get('nguong'))} ⟹ "
                   f"<b>{'qua cổng (có dấu hiệu sửa)' if k5['qua_cong'] else 'dưới ngưỡng ⟹ 01, dừng'}</b> · "
                   f"chế độ: {e(k['che_do_khuc'])}</p>" + (f"<p class='muted'>{e(k5['ghi_chu'])}</p>" if k5.get("ghi_chu") else ""))
        if k.get("khuc"):
            out.append("<table><tr><th>Khúc 16 s</th><th>Điểm cổng</th><th>Qua cổng</th><th>Số lớp vượt ngưỡng</th></tr>" +
                       "".join(f"<tr><td>{mmss(c['bat_dau_s'])} – {mmss(c['ket_thuc_s'])}</td><td class='num'>"
                               f"{_num(c['diem_cong'])}</td><td>{'có' if c['qua_cong'] else 'không'}</td>"
                               f"<td class='num'>{c['so_lop_vuot_nguong']}</td></tr>" for c in k["khuc"]) + "</table>")
        if k7:
            out.append(f"<h3>{e(pre)}K7 · Quyết định một đoạn và kiểm chứng</h3><table><tr><th>Lớp một đoạn</th><th>Điểm lớn nhất</th>"
                       "<th>Ngưỡng</th></tr>" +
                       "".join(f"<tr><td>{e(OP[c][0]) if c in OP else e(c)} một đoạn</td><td class='num'>{_num(v.get('diem'))}</td>"
                               f"<td class='num'>{_num(v.get('nguong')) if v.get('bat', True) else 'tắt'}</td></tr>" for c, v in k7["lop"].items()) + "</table>")
            if k7["de_xuat"]:
                out.append("<table><tr><th>Đoạn đề xuất</th><th>Lớp</th><th>Điểm</th><th>Vượt ngưỡng (logit)</th><th>Kiểm chứng</th><th>Kết cục</th></tr>" +
                           "".join(f"<tr><td>{mmss(d['bat_dau_s'])} – {mmss(d['ket_thuc_s'])}</td><td>{e(OP[d['lop']][0]) if d['lop'] in OP else e(d['lop'])}</td>"
                                   f"<td class='num'>{_num(d['p'])}</td><td class='num'>{_num(d.get('bien_vuot_nguong'))}</td>"
                                   f"<td>{e(d['kiem_chung'])}</td><td>{e(d['ket_cuc'])}</td></tr>"
                                   for d in k7["de_xuat"]) + "</table>")
            if k7.get("ghi_chu_luat"):
                out.append("<p class='muted'>Luật ràng buộc đã áp: " + e("; ".join(k7["ghi_chu_luat"])) + "</p>")
        k8 = k["K8"]
        out.append(f"<h3>{e(pre)}K8 · Nhãn</h3><p>{e(k8['giai_thich'])}</p>" +
                   (f"<p class='muted'>Đoạn nghi 03 chưa đủ chắc: {e('; '.join(k8['nghi_03']))}</p>" if k8.get("nghi_03") else ""))
    return "\n".join(out)


def _versions(ho: dict) -> str:
    pv = ho["he_thong"].get("phien_ban")
    if not pv:
        return "<p class='muted'>Không có thông tin phiên bản (tệp bị từ chối trước khi nạp mô hình).</p>"
    ng = pv.get("nguong", {})
    rows = [f"<tr><td>ngưỡng</td><td>{e(ng.get('tep'))}</td><td><code class='hash'>{e(ng.get('sha1'))}</code></td>"
            f"<td>điểm hoạt động {e(ng.get('diem_hoat_dong'))} · tham số {e(json.dumps(ng.get('params', {}), ensure_ascii=False))}</td></tr>"]
    for m in pv.get("mo_hinh", []):
        rows.append(f"<tr><td>{e(m.get('vai_tro', 'mô hình'))}</td><td>{e(m.get('ten'))}</td><td><code class='hash'>{e(m.get('sha1'))}</code></td>"
                    f"<td>{e(m.get('ghi_chu', ''))}</td></tr>")
    for k in ("xlsr", "kho_tham_chieu", "hieu_chinh"):
        v = pv.get(k)
        if v:
            rows.append(f"<tr><td>{e(k)}</td><td>{e(v.get('ten'))}</td><td><code class='hash'>{e(v.get('sha1') or v.get('ban') or '—')}</code></td>"
                        f"<td>{e(v.get('ghi_chu', ''))}</td></tr>")
    extra = "".join(f"<p class='muted'>{e(x)}</p>" for x in (pv.get("mac_dinh") or []))
    extra += f"<p class='muted'>Thiếu tệp: {e(', '.join(pv['thieu']))}</p>" if pv.get("thieu") else ""
    return ("<table><tr><th>Phần</th><th>Tệp / tên</th><th>sha1 / bản</th><th>Ghi chú</th></tr>" + "".join(rows) + "</table>" + extra +
            f"<p class='muted'>Mã gộp phiên bản (SHA-256 của danh sách trên): <code class='hash'>{e(ho['he_thong'].get('manifest_sha256'))}</code></p>")


def _aux_section(ho: dict) -> str:
    op_ = ho["output_phu"]
    parts = ["<h2>Output phụ</h2>", "<h3>Cảnh báo</h3>"]
    parts.append("<ul class='warn'>" + "".join(f"<li>{e(w['noi_dung'])}" + (f" <span class='muted'>({e(w['chi_tiet'])})</span>" if w.get("chi_tiet") else "")
                                               + "</li>" for w in op_["canh_bao"]) + "</ul>"
                 if op_["canh_bao"] else "<p class='muted'>Không có.</p>")
    parts.append("<h3>Bảng mốc thời gian</h3>")
    two = (ho["tep"].get("so_kenh_phan_tich") or 0) > 1
    if op_["bang_moc"]:
        parts.append("<table><tr><th>Thao tác</th>" + ("<th>Kênh</th>" if two else "") + "<th>Loại</th><th>Phạm vi</th><th>Bắt đầu</th><th>Kết thúc</th>"
                     "<th>Bắt đầu (s)</th><th>Kết thúc (s)</th><th>Mức độ</th></tr>" +
                     "".join(f"<tr data-moc-row='{e(m['thao_tac'])}'><td>{e(m['thao_tac'])}</td>" + (f"<td>{e(_kenh(m, ho))}</td>" if two else "") +
                             f"<td>{e(m['loai'])}</td><td>{e(m['pham_vi'])}</td><td class='num'>{e(m['bat_dau'])}</td><td class='num'>{e(m['ket_thuc'])}</td>"
                             f"<td class='num'>{_num(m['bat_dau_s'])}</td><td class='num'>{_num(m['ket_thuc_s'])}</td>"
                             f"<td>{e(m['muc_do'])}</td></tr>" for m in op_["bang_moc"]) + "</table>")
    else:
        parts.append("<p class='muted'>Không có thao tác nào trong phiếu.</p>")
    parts += ["<h2>Điểm và ngưỡng từng bước</h2>", _steps(ho), "<h2>Bảng số đo</h2>"]
    if op_["bang_so_do"]:
        parts.append("<table><tr><th>Mục</th><th>Số đo</th><th>Giá trị</th><th>Trung vị kho tham chiếu</th><th>Phân vị</th><th>So với kho</th><th>Kho</th></tr>" +
                     "".join(f"<tr><td>{e(s['thao_tac'])}</td><td>{e(s.get('ten', s['ma']))}</td><td class='num'>"
                             f"{_num(s['gia_tri'])}{(' ' + e(s.get('don_vi'))) if s.get('don_vi') else ''}</td>"
                             f"<td class='num'>{_num(s.get('trung_vi_that')) if s.get('trung_vi_that') is not None else ''}</td>"
                             f"<td class='num'>{'' if s.get('phan_vi_that') is None else vn(s['phan_vi_that'], 1) + ' %'}</td>"
                             f"<td>{e(_dir(s))}</td><td>{e(s.get('kho', ''))}</td></tr>" for s in op_["bang_so_do"]) + "</table>")
        kho = sorted({s["kho"] for s in op_["bang_so_do"] if s.get("kho")})
        if kho:
            parts.append(f"<p class='muted'>* Kho tham chiếu: {e(' · '.join(kho))}. Con số phần trăm là mức bất thường so với kho này, "
                         "không phải xác suất tệp bị sửa.</p>")
    parts.append("<h2>Phiên bản mô hình và ngưỡng</h2>" + _versions(ho))
    parts.append("<h2>Tuyên bố giới hạn</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in op_["tuyen_bo_gioi_han"]) + "</ul>")
    return "\n".join(parts)


def build_html(ho: dict) -> str:
    t = ho["tep"]
    hs = ho["he_thong"]
    meta = [f"Tệp: <b>{e(t['ten'])}</b>", f"Độ dài phân tích: {vn(t['do_dai_s'] or 0, 2)} s",
            f"Số kênh phân tích: {t['so_kenh_phan_tich']}", f"Lập lúc: {e(ho['tao_luc'])}",
            f"SHA-256: <code class='hash' data-sha256>{e(t['sha256'] or '—')}</code>"]
    stub = "<span class='pill'>MÔ HÌNH GIẢ LẬP — chỉ tự kiểm mã</span>" if hs["mo_hinh"] == "gia_lap" else ""
    return (f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>Hồ sơ kiểm định</title><style>{CSS}</style></head><body><main>"
            f"<header class='top'><div class='kicker'>{e(hs.get('ten', 'VoiceAudit'))} · sàng lọc, không phải giám định tư pháp</div>"
            f"<h1>Hồ sơ kiểm định bản ghi âm {stub}</h1><div class='meta'>" + "".join(f"<span>{m}</span>" for m in meta) +
            "</div></header>" + _main_section(ho) + _aux_section(ho) +
            f"<footer>Lược đồ {e(ho['schema'])} · mô hình {e(hs['mo_hinh'])} · mã gộp phiên bản {e((hs.get('manifest_sha256') or '—')[:16])} · "
            f"kiểm đồng nhất: {'ĐẠT' if ho['kiem_tra_dong_nhat']['dat'] else 'KHÔNG ĐẠT'} · báo cáo chỉ chứa các dòng, đoạn và "
            f"số đo có trong hồ sơ.</footer></main></body></html>")


def write_html(ho: dict, out_dir: str) -> str:
    p = os.path.join(out_dir, "bao_cao.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(build_html(ho))
    return p


def to_pdf(html_path: str, pdf_path: str, chrome: str = CHROME, timeout: int = 120) -> tuple:
    """In PDF bang Chrome headless (thieu Chrome thi Edge); ho so nguoi dung tam dat duoi .tmp_v6 (khong dung o C:). Tra (dat, ghi chu, giay)."""
    t0 = time.time()
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
    if not os.path.exists(chrome):
        chrome = EDGE if os.path.exists(EDGE) else chrome
    if not os.path.exists(chrome):
        return False, f"không thấy Chrome / Edge: {chrome}", 0.0
    prof = os.path.join(TMP_DIR, "chrome_profile", f"p{os.getpid()}_{int(t0 * 1000) % 100000}")
    os.makedirs(prof, exist_ok=True)
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
           "--disable-extensions", f"--user-data-dir={prof}", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", pathlib.Path(html_path).resolve().as_uri()]
    try:
        subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"Chrome quá {timeout} s", time.time() - t0
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    ok = os.path.exists(pdf_path) and open(pdf_path, "rb").read(5) == b"%PDF-"
    return ok, "" if ok else "Chrome không tạo được PDF", round(time.time() - t0, 2)


MOC_COLS = ["thao_tac", "kenh", "loai", "pham_vi", "muc_do", "bat_dau", "ket_thuc", "bat_dau_s", "ket_thuc_s"]
SODO_COLS = ["thao_tac", "kenh", "ma", "ten", "gia_tri", "don_vi", "trung_vi_that", "phan_vi_that", "so_voi_kho", "bat_thuong", "kho"]


def _cell(r: dict, k: str):
    if k == "kenh" and "kenh_ten" in r:                     # hop dong: 'trai / phai' khi hai kenh, de trong khi mot kenh
        return r["kenh_ten"] or ""
    return r.get(k, "")


def write_csvs(ho: dict, out_dir: str) -> tuple:
    """bang_moc.csv + bang_so_do.csv (utf-8-sig de mo bang Excel khong vo dau)."""
    pm, ps = os.path.join(out_dir, "bang_moc.csv"), os.path.join(out_dir, "bang_so_do.csv")
    for path, cols, rows in ((pm, MOC_COLS, ho["output_phu"]["bang_moc"]), (ps, SODO_COLS, ho["output_phu"]["bang_so_do"])):
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: _cell(r, k) for k in cols})
    return pm, ps


def pdf_text(pdf_path: str) -> str | None:
    """Chu trong PDF (pypdf o .tmp_v6/site_pdf neu co); None neu khong co thu vien doc PDF."""
    site = os.path.join(TMP_DIR, "site_pdf")
    if os.path.isdir(site) and site not in sys.path:
        sys.path.append(site)
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf_path).pages)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dung lai bao cao HTML / PDF tu ho_so.json")
    ap.add_argument("ho_so")
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ho = json.load(open(a.ho_so, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(a.ho_so))
    hp = write_html(ho, out)
    print(hp)
    if a.pdf:
        print(to_pdf(hp, os.path.join(out, "bao_cao.pdf")))


if __name__ == "__main__":
    main()
