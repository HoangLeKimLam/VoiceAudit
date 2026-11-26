"""Trang nghe thử thang mẫu SNR (24/11) để Huấn làm quen tai trước vòng nghe tay 2: «nền ồn hơn hay ít hơn mẫu 10 dB».
Chỉ trỏ tới tệp có sẵn trong p1_dieukien_v2 (không chép, không sinh). Seed 2414.
  Phần 1 (có nhãn): 2 giọng / ngôn ngữ × {goc, 20, 15, 12, 10, 8, 5, 0 dB}; mỗi mức một loại nền khác nhau (ghi nguồn).
  Phần 2 (tự kiểm, xáo): giọng KHÁC phần 1, mức {0, 5, 8, 12, 15, 20} × vi/en = 12 ô, có nút nghe mẫu 10 dB cùng ngôn ngữ;
  chấm ngay trong trang: 0/5/20/15 dB là ca rõ; 8/12 dB là ca sát, báo riêng.
Ra: p1_dieukien_v2/_kiem/THANG_MAU_2411.html (src tương đối ../snr, ../goc).

Chạy: .venv/Scripts/python.exe experts/P1/build_cond/thang_mau.py
"""
from __future__ import annotations

import html
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import V6  # noqa: E402

SEED = 2414
BO = os.path.join(V6, "p1_dieukien_v2")
OUT = os.path.join(BO, "_kiem", "THANG_MAU_2411.html")
THANG = ("goc", "snr.20dB", "snr.15dB", "snr.12dB", "snr.10dB", "snr.8dB", "snr.5dB", "snr.0dB")
KIEM = ("snr.0dB", "snr.5dB", "snr.8dB", "snr.12dB", "snr.15dB", "snr.20dB")


def load() -> dict:
    """{(lang, base): {ca: (file, nguồn nền)}} cho các ca trong THANG."""
    prov = {p["file"].split("p1_dieukien_v2/")[1]: p for p in map(json.loads, open(os.path.join(BO, "provenance.jsonl"), encoding="utf-8"))}
    cases = json.load(open(os.path.join(BO, "cases.json"), encoding="utf-8"))
    out: dict = {}
    for c in (cases if isinstance(cases, list) else cases["cases"]):
        if c["ca"] in THANG:
            src = prov[c["file"]]["nguon_tron"]
            nen = os.path.splitext("/".join(src[0]["path"].split("/")[-2:]))[0] if src else "—"
            out.setdefault((c["file"].split("_")[2], c["base"]), {})[c["ca"]] = (c["file"], nen)
    return {k: v for k, v in out.items() if all(t in v for t in THANG)}


def audio(f: str) -> str:
    return f'<audio controls preload="none" src="../{html.escape(f)}"></audio>'


def page(thang: list, kiem: list, mau10: dict) -> str:
    hdr = "".join(f"<th>{'sạch' if t == 'goc' else t[4:]}</th>" for t in THANG)
    p1 = "\n".join(f"<tr><td>{lang} · giọng {i}</td>" + "".join(f"<td>{audio(v[t][0])}<br><small>{html.escape(v[t][1])}</small></td>" for t in THANG) + "</tr>"
                   for i, (lang, v) in enumerate(thang, 1))
    opts = (("on_hon", "ồn hơn mẫu 10 dB"), ("kho", "khó phân biệt"), ("sach_hon", "sạch hơn mẫu 10 dB"))
    p2 = "\n".join(f'<tr data-i="{i}" data-db="{db}"><td>{i}</td><td>{audio(f)}</td><td>{audio(mau10[lang])}<br><small>mẫu 10 dB ({lang})</small></td><td>'
                   + " ".join(f'<label><input type=radio name="q{i}" value="{k}">{t}</label>' for k, t in opts) + '</td><td class=dap></td></tr>'
                   for i, (lang, f, db) in enumerate(kiem, 1))
    return f"""<!DOCTYPE html><html lang="vi"><meta charset="utf-8"><title>Thang mẫu SNR</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse;margin-bottom:16px}}td,th{{border:1px solid #ccc;padding:4px;vertical-align:top}}
audio{{width:180px;height:30px}}label{{white-space:nowrap;margin-right:8px;display:block}}small{{color:#666}}.ok{{background:#e6f6e6}}.sai{{background:#fbe3e3}}</style>
<h2>Thang mẫu SNR: làm quen tai</h2>
<p>Đặt âm lượng loa <b>một lần</b>, giữ nguyên cả trang. Mọi tệp đã cùng âm lượng (−23 dBFS); chỉ khác <b>độ to của nền so với giọng</b>.
SNR 10 dB = giọng mạnh gấp 10 lần nền = ranh giới hàng cứng «ồn quá». Số càng nhỏ càng ồn.
Mỗi ô dùng một loại nền khác nhau (dòng chữ nhỏ: DEMAND = âm thanh môi trường thật, MUSAN = tiếng ồn tự do), nên hãy so <b>độ lấn</b> của nền, không so loại tiếng.</p>
<h3>Phần 1: thang có nhãn</h3>
<p>Nghe dọc một hàng từ trái sang phải, để ý cột <b>10</b>.</p>
<table><tr><th></th>{hdr}</tr>
{p1}
</table>
<h3>Phần 2: tự kiểm (giọng khác phần 1, đã xáo)</h3>
<p>Với mỗi tệp, nghe rồi so với mẫu 10 dB bên cạnh. Chọn xong hết thì bấm <b>Chấm</b>.</p>
<table><tr><th>#</th><th>tệp</th><th>mẫu 10 dB</th><th>so với mẫu</th><th>đáp án</th></tr>
{p2}
</table>
<button onclick=cham()>Chấm</button> <span id=kq></span>
<p><small>Ca rõ: 0, 5 dB (đáp án «ồn hơn») và 15, 20 dB («sạch hơn»). Ca sát: 8, 12 dB, khó là bình thường, báo riêng.</small></p>
<script>
function cham(){{let ro=0,nro=0,sat=0,nsat=0;
 document.querySelectorAll('tr[data-i]').forEach(r=>{{const db=+r.dataset.db,x=r.querySelector('input:checked'),v=x?x.value:'';
  const dung=db<10?'on_hon':'sach_hon',ok=v===dung,clear=db<=5||db>=15;
  if(clear){{nro++;if(ok)ro++}}else{{nsat++;if(ok)sat++}}
  r.className=v?(ok?'ok':'sai'):'';r.querySelector('.dap').textContent=db+' dB → '+(db<10?'ồn hơn':'sạch hơn')}});
 document.getElementById('kq').textContent='Ca rõ đúng '+ro+'/'+nro+' · ca sát đúng '+sat+'/'+nsat}}
</script></html>
"""


def run() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    rng = random.Random(SEED)
    bo = load()
    thang, kiem, mau10 = [], [], {}
    for lang in ("vi", "en"):
        bases = sorted(b for (l, b) in bo if l == lang)
        assert len(bases) >= 4, (lang, len(bases))
        rng.shuffle(bases)
        thang += [(lang, bo[(lang, b)]) for b in bases[:2]]
        mau10[lang] = bo[(lang, bases[0])]["snr.10dB"][0]
        kiem += [(lang, bo[(lang, bases[2 + i % 2])][t][0], int(t[4:-2])) for i, t in enumerate(KIEM)]
    rng.shuffle(kiem)
    assert all(os.path.exists(os.path.join(BO, f)) for _, f, _ in kiem)
    open(OUT, "w", encoding="utf-8").write(page(thang, kiem, mau10))
    print(f"xong: {OUT} · thang {len(thang)} hàng · tự kiểm {len(kiem)} ô")


if __name__ == "__main__":
    run()
