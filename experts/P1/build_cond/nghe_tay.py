"""Huấn nghe tay 20 tệp (24/11), lấy từ 72 tệp Gemini đã nghe (gemini/nghe_2411) để vừa làm đáp án cho bộ đo ồn / nền,
vừa chấm lại Gemini. Chọn chốt trước, seed 2413, chỉ tệp ≤ 64 s (đỡ tải nghe):
  16 tệp thật: tầng A < 5 dB 6 · B 5–10 dB 6 · C > 15 dB 4, mỗi tầng nửa vi nửa en
  4 đối chứng: goc · snr.20dB · snr.0dB · nen.30pct (ngôn ngữ ngẫu nhiên)
Trang mù: không hiện tầng, số đo hay kết quả Gemini. Ra: p1_tunhien_v1/_kiem/NGHE_TAY_2411.html (+ _map.json).

Chạy: .venv/Scripts/python.exe experts/P1/build_cond/nghe_tay.py
"""
from __future__ import annotations

import html
import json
import os
import random
import sys

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cham_dieu_kien import NAT, V6  # noqa: E402

SEED = 2413
WAV = r"F:\CODING\Voiceaudit_ai002\gemini\nghe_2411"
SRC = os.path.join(V6, NAT, "_kiem", "gemini_nghe_2411_map.json")
# Trang đặt cạnh nghe_2411/ và dùng src tương đối: trang ở D: trỏ file:///F:/… bị trình duyệt chặn (Huấn không nghe được, 24/11).
OUT = os.path.join(os.path.dirname(WAV), "NGHE_TAY_2411.html")
MAP = os.path.join(V6, NAT, "_kiem", "NGHE_TAY_2411_map.json")
SO = {("A_duoi5dB", "vi"): 3, ("A_duoi5dB", "en"): 3, ("B_5_10dB", "vi"): 3, ("B_5_10dB", "en"): 3,
      ("C_tren15dB", "vi"): 2, ("C_tren15dB", "en"): 2}
DC = ("goc", "snr.20dB", "snr.0dB", "nen.30pct")
NHAN = {"on_nen": ("khong", "nhe", "ro", "che_loi"), "vang": ("khong", "vua", "nang"), "giong_ro": ("co", "khong")}


def chon(rng: random.Random) -> list:
    rows = [r for r in json.load(open(SRC, encoding="utf-8"))["tep"] if sf.info(os.path.join(WAV, r["ten"])).duration <= 64]
    out = []
    for (t, lang), k in SO.items():
        g = sorted((r for r in rows if r["loai"] == "that" and r["tang"] == t and r["lang"] == lang), key=lambda r: r["ten"])
        assert len(g) >= k, (t, lang, len(g))
        out += rng.sample(g, k)
    for t in DC:
        g = sorted((r for r in rows if r["loai"] == "doi_chung" and r["tang"] == t), key=lambda r: r["ten"])
        out.append(rng.choice(g))
    rng.shuffle(out)
    return out


def page(rows: list, thu_muc: str = "nghe_2411", key_ls: str = "nghe_tay_2411", gem_kq: str = "KQ_NGHE_DK_2411.jsonl") -> str:
    n = len(rows)
    def radios(i: int, key: str) -> str:
        return " ".join(f'<label><input type=radio name="{key}{i}" data-k="{key}" value="{v}">{v}</label>' for v in NHAN[key])
    tr = "\n".join(
        f'<tr data-f="{r["ten"]}"><td>{i}</td><td>{r["ten"]}<br><audio controls preload="none" '
        f'src="{thu_muc}/{html.escape(r["ten"])}"></audio></td>'
        f'<td>{radios(i, "on_nen")}</td><td>{radios(i, "vang")}</td><td>{radios(i, "giong_ro")}</td>'
        f'<td><input class=gc size=28></td></tr>' for i, r in enumerate(rows, 1))
    return f"""<!DOCTYPE html><html lang="vi"><meta charset="utf-8"><title>Nghe tay {key_ls.replace("nghe_tay_", "")}</title>
<style>body{{font:14px system-ui;margin:16px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:4px;vertical-align:top}}
tr.done{{background:#eef8ee}}audio{{width:230px;height:32px}}label{{white-space:nowrap;margin-right:6px}}</style>
<h2>Nghe tay: {n} tệp (mù)</h2>
<p>Nghe ở <b>một mức âm lượng</b> cho cả {n} tệp. Chỉ xét phần có người nói. Trang không hiện số đo; <b>đừng mở</b>
<code>gemini/{gem_kq}</code> hay tệp <code>_map</code> trước khi nghe xong.</p>
<ul><li><b>on_nen</b>: khong = gần như chỉ có giọng · nhe = nghe thấy nền nhưng nhỏ hơn hẳn giọng · ro = nền rõ, lời vẫn hiểu dễ ·
che_loi = nền gần bằng / to hơn giọng, phải cố hoặc mất từ</li>
<li><b>vang</b>: khong = gần mic, khô · vua = có tiếng phòng, đuôi ngắn · nang = giọng xa, đuôi vang dài rõ</li>
<li><b>giong_ro</b>: hiểu được phần lớn lời nói không</li></ul>
<p>Kết quả tự lưu trong trình duyệt. Xong bấm <b>Tải kết quả</b> → lưu <code>{key_ls}.json</code> vào <code>p1_tunhien_v1/_kiem/</code>.</p>
<p>Người nghe: <input id=who> <button onclick=dl()>Tải kết quả</button> <span id=cnt></span></p>
<table><tr><th>#</th><th>tệp</th><th>on_nen</th><th>vang</th><th>giong_ro</th><th>ghi chú</th></tr>
{tr}
</table>
<script>
const K='{key_ls}';let S={{}};try{{S=JSON.parse(localStorage[K]||'{{}}')}}catch(e){{}}
const rows=[...document.querySelectorAll('tr[data-f]')];
function full(s){{return s&&s.on_nen&&s.vang&&s.giong_ro}}
function paint(){{let n=0;rows.forEach(r=>{{const ok=full(S[r.dataset.f]);r.className=ok?'done':'';if(ok)n++}});
 document.getElementById('cnt').textContent=n+'/'+rows.length+' đã chấm đủ'}}
function save(){{try{{localStorage[K]=JSON.stringify(S)}}catch(e){{}};paint()}}
rows.forEach(r=>{{const f=r.dataset.f,s=S[f]||{{}};
 r.querySelectorAll('input[type=radio]').forEach(x=>{{x.checked=s[x.dataset.k]==x.value;x.onchange=()=>{{S[f]={{...S[f],[x.dataset.k]:x.value}};save()}}}});
 const g=r.querySelector('.gc');g.value=s.ghi_chu||'';g.oninput=()=>{{S[f]={{...S[f],ghi_chu:g.value}};save()}}}});
const w=document.getElementById('who');try{{w.value=localStorage[K+'_who']||''}}catch(e){{}};w.oninput=()=>{{try{{localStorage[K+'_who']=w.value}}catch(e){{}}}};
function dl(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({{nguoi_nghe:w.value,ngay:new Date().toISOString(),ket_qua:S}},null,1)],{{type:'application/json'}}));
 a.download='{key_ls}.json';a.click()}}
paint();
</script></html>
"""


def run() -> None:
    for p in (OUT, MAP):
        if os.path.exists(p):
            raise SystemExit(f"Đã có {p}; không ghi đè")
    rows = chon(random.Random(SEED))
    assert len(rows) == 20 and len({r["ten"] for r in rows}) == 20
    open(OUT, "w", encoding="utf-8").write(page(rows))
    json.dump({"seed": SEED, "tep": rows}, open(MAP, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"xong: {OUT}")
    print("thành phần:", sorted((r["loai"], r["tang"], r["lang"]) for r in rows))


def trang() -> None:
    """Dựng lại trang từ _map đã chốt (không chọn lại tệp)."""
    if os.path.exists(OUT):
        raise SystemExit(f"Đã có {OUT}; không ghi đè")
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    open(OUT, "w", encoding="utf-8").write(page(rows))
    assert all(os.path.exists(os.path.join(WAV, r["ten"])) for r in rows)
    print(f"xong: {OUT} ({len(rows)} tệp)")


KQ = os.path.join(V6, NAT, "_kiem", "nghe_tay_2411.json")
GEM = r"F:\CODING\Voiceaudit_ai002\gemini\KQ_NGHE_DK_2411.jsonl"
CHAM = os.path.join(V6, NAT, "_kiem", "CHAM_NGHE_TAY_2411.md")
DC_OK = {"goc": ("khong", "nhe"), "snr.20dB": ("khong", "nhe"), "snr.0dB": ("ro", "che_loi"), "nen.30pct": ("ro", "che_loi")}
TI_LE_CHAN = (483, 1372)  # bước 2: bản 2 chặn 483/1372 tệp accept p1_tunhien_v1 (NHAT_KY 2.16)


def cham() -> None:
    """Luật chốt trước: NHAT_KY 2.16 «Huấn nghe tay» 1–5."""
    from cham_dieu_kien import jl, pct, wilson
    kq = json.load(open(KQ, encoding="utf-8"))["ket_qua"]
    gem = {r["tep"].split("/")[-1]: r for r in jl(GEM) if "stt" in r}
    rows = json.load(open(MAP, encoding="utf-8"))["tep"]
    assert all(r["ten"] in kq and all(k in kq[r["ten"]] for k in NHAN) for r in rows), "thiếu nhãn"
    on = lambda r: kq[r["ten"]]["on_nen"]  # noqa: E731
    dc = [r for r in rows if r["loai"] == "doi_chung"]
    ab = [r for r in rows if r["loai"] == "that" and r["tang"] != "C_tren15dB"]
    c = [r for r in rows if r["tang"] == "C_tren15dB"]
    ab_hai = {"chặt (che_loi = ồn)": ("che_loi",), "rộng (ro/che_loi = ồn)": ("ro", "che_loi")}
    L = ["# Huấn nghe tay 20 tệp: chấm bộ đo ồn / nền bản 2 và Gemini (luật chốt trước, NHAT_KY 2.16)", "",
         "Nguồn nhãn: **người nghe tay (Huấn)**, trang mù `gemini/NGHE_TAY_2411.html`.", "",
         f"## 1. Đối chứng: {sum(on(r) in DC_OK[r['tang']] for r in dc)}/4", ""]
    L += [f"- {r['tang']} ({r['lang']}, {r['ten']}): Huấn `{on(r)}` · đáp án dựng {'/'.join(DC_OK[r['tang']])} → "
          f"{'khớp' if on(r) in DC_OK[r['tang']] else '**lệch**'}" for r in dc]
    L += ["", "## 2. Bộ đo (bản 2, ngưỡng hợp đồng)", "", "| cách đếm | chặn oan A+B (k/12) | thả sai C (k/4) | ước chặn oan toàn bộ p1_tunhien_v1 |",
          "|---|---|---|---|"]
    p_chan = TI_LE_CHAN[0] / TI_LE_CHAN[1]
    for ten, ON in ab_hai.items():
        k_ab, k_c = sum(on(r) not in ON for r in ab), sum(on(r) in ON for r in c)
        lo, hi = wilson(k_ab, len(ab))
        L.append(f"| {ten} | {pct(k_ab, len(ab))} | {pct(k_c, len(c))} | "
                 f"{100 * p_chan * k_ab / len(ab):.1f} % ({100 * p_chan * lo:.1f}–{100 * p_chan * hi:.1f}) |")
    L += ["", f"Ước toàn bộ = tỉ lệ bị chặn ở bước 2 ({TI_LE_CHAN[0]}/{TI_LE_CHAN[1]}) × tỉ lệ chặn oan A+B; khoảng = Wilson của k/12 "
          "(chưa tính sai số của tỉ lệ bị chặn). Điều kiện cài vào K1: ≤ 2 % theo **cả hai** cách đếm.", "",
          "## 3. Gemini so với Huấn (on_nen, 20 tệp)", ""]
    hai = lambda v: v in ("ro", "che_loi")  # noqa: E731
    L += [f"- trùng đúng 4 mức: {pct(sum(gem[r['ten']]['on_nen'] == on(r) for r in rows), 20)}",
          f"- trùng sau khi gộp 2 mức (khong/nhe ↔ ro/che_loi): {pct(sum(hai(gem[r['ten']]['on_nen']) == hai(on(r)) for r in rows), 20)}",
          "", "## 4. Từng tệp", "", "| tệp | loại / tầng | lang | SNR ước | nền ước % | RT60 ước | Huấn on_nen | vang | giong_ro | Gemini on_nen | vang | ghi chú |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["loai"], r["tang"], r["ten"])):
        u, h, g = r.get("uoc") or {}, kq[r["ten"]], gem[r["ten"]]
        L.append(f"| {r['ten']} | {r['loai']} / {r['tang']} | {r['lang']} | {u.get('snr', '')} | {u.get('nen', '')} | {u.get('vang', '')} | "
                 f"{h['on_nen']} | {h['vang']} | {h['giong_ro']} | {g['on_nen']} | {g['vang']} | {h.get('ghi_chu', '')} |")
    open(CHAM, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": run, "trang": trang, "cham": cham}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
