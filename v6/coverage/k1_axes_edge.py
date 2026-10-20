# -*- coding: utf-8 -*-
"""Ca BIEN / LUOI cua v6/coverage/k1_axes (12/10, audit C-04..C-13) + so do doc lap them cho oracle (tuong quan hai kenh, toc do bit khai,
muc tieng noi). Tach khoi k1_axes.py de file < 800 dong; k1_axes nap muon module nay (CASES, oracle, '--only @bien').
r5 (review 12/10): cat dinh tep nen (catdinh_nen_*), bang hep + san on (g711_on_* / amrnb_on_*, doi chung rong_on_-60dBFS), kenh2_1.9s / 2.1s
(speech_s_own: tieng noi do tren CHINH kenh), noi chong hai nguoi (nguoi.noi_chong_2, chi bat buoc khi --extended), lang_real() (ca duong ngon ngu
khac: doan LibriVox PD / CC0, data_labels/v6/k1axes/lang_real) - goi qua  python -m v6.coverage.k1_axes lang.
Khong co CLI rieng: dung qua  python -m v6.coverage.k1_axes build --only @bien  /  run --only @bien ; test: python -m v6.coverage.test_k1_axes"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from v6.coverage.k1_axes import DROP_WARN, LAG_S, SR, E, _clip, _drop, _enc, _lowpass, dense, energy_db, ff, sf, tile, wr  # noqa: E501


def _two_step(ext: str, first: tuple, second: tuple):
    """Ma hoa hai buoc qua tep tam (vd ha 8 kHz G.711 roi nang lai 16 kHz PCM; ha 8 kHz roi ma hoa Opus)."""
    def f(y, w, d, o):
        tmp = d + "__tmp" + first[0]
        try:
            ff(w, tmp, *first[1:])
            ff(tmp, d + ext, *second)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return f


def _dense_enc(sec: float, ext: str, *args: str):
    """Doan sec giay nhieu tieng noi nhat, ma hoa sang ext (vd 3,0 s MP3: kiem do dai khai sau dem / mo dau cua bo ma hoa)."""
    def f(y, w, d, o):
        tmp = d + "__tmp.wav"
        wr(tmp, dense(y, sec))
        try:
            ff(tmp, d + ext, *args)
        finally:
            os.remove(tmp)
    return f


def _stereo_mix(kind: str, v: float):
    """Hai kenh quanh nguong Constraint 'Hai kenh': rho (tron loi noi kenh trai voi nguoi khac de tuong quan ~ v), lech (kenh phai = kenh
    trai tre v giay), gan_giong (kenh phai = kenh trai + nhieu trang de tuong quan Pearson ~ v). Dap an tinh tu tuong quan DO LAI (oracle)."""
    def f(y, w, d, o):
        n = len(y)
        rng = np.random.default_rng(7)
        sy = np.std(y) + 1e-12
        if kind == "rho":
            oo = tile(o, n / SR)[:n]
            r_ = v * y / sy + np.sqrt(1 - v * v) * oo / (np.std(oo) + 1e-12)
            R = 0.8 * np.max(np.abs(y)) * r_ / (np.max(np.abs(r_)) + 1e-12)
        elif kind == "lech":
            k = int(round(v * SR))
            R = np.r_[np.zeros(k), 0.7 * y[:-k]] + rng.normal(0, 0.003, n)
        else:
            R = y + rng.normal(0, sy * np.sqrt(1 / v ** 2 - 1), n)
        x = np.c_[y, R]
        wr(d + ".wav", x / max(1.0, float(np.max(np.abs(x))) / 0.99), subtype="FLOAT")
    return f


def _clip_enc(pct: float, ext: str, *args: str):
    """Cat dinh pct % (nhu _clip, dinh 0,999; pct = 0: chi chuan dinh) roi MA HOA NEN (review 12/10: binh nguyen bi vo khi giai ma)."""
    def f(y, w, d, o):
        tmp = d + "__tmp.wav"
        if pct:
            _clip(pct)(y, w, d + "__tmp", o)
        else:
            wr(tmp, 0.999 * y / (np.max(np.abs(y)) + 1e-12), subtype="FLOAT")
        try:
            ff(tmp, d + ext, "-ar", "16000", "-ac", "1", *args)
        finally:
            os.remove(tmp)
    return f


def _noisy(build, noise_dbfs: float, seed: int = 11):
    """Dung tep bang build (ham ca) vao tep tam, doc lai, CONG on trang noise_dbfs (RMS) -> WAV float (bang hep + san on: audit C-10)."""
    def f(y, w, d, o):
        build(y, w, d + "__tmp", o)
        tmp = d + "__tmp.wav"
        try:
            z, sr = sf.read(tmp, dtype="float64")
        finally:
            os.remove(tmp)
        wr(d + ".wav", z + np.random.default_rng(seed).normal(0, 10 ** (noise_dbfs / 20), len(z)), sr, subtype="FLOAT")
    return f


def speech_s_own(y: np.ndarray, sr: int = SR) -> float:
    """Thoi luong tieng noi (s) cua MOT kenh theo nang luong voi san nhieu CUA CHINH kenh do (khung > phan vi 10 + 10 dB va > -60 dBFS) -
    dinh nghia dong 'kenh co duoi 2 s tieng noi' (cai doc lap voi vaudit.k1)."""
    db = energy_db(y, sr)
    return float(((db > np.percentile(db, 10) + 10) & (db > -60)).sum() * 0.03)


def _kenh2(target_s: float):
    """Hai micro cung canh (kenh phai = kenh trai tre 12 ms, x0,7) nhung kenh phai CHI co tieng noi trong mot cua so dai sao cho thoi luong tieng
    noi do doc lap = target_s (ngoai cua so: on -70 dBFS). Nguong Constraint 2 s: 1,9 s -> bo kenh phai; 2,1 s -> phan tich 2 kenh."""
    def f(y, w, d, o):
        x1 = dense(y, 8.0)
        k = int(0.012 * SR)
        r = np.r_[np.zeros(k), 0.7 * x1[:-k]]
        rng = np.random.default_rng(13)
        bg = rng.normal(0, 10 ** (-70 / 20), len(x1))
        a = 0
        for L in np.arange(0.3, 8.0, 0.015):
            ch2 = bg.copy()
            n = min(len(x1), int(L * SR))
            ch2[a:a + n] += r[a:a + n]
            if speech_s_own(ch2) >= target_s:
                break
        wr(d + ".wav", np.c_[x1, ch2], subtype="FLOAT")
    return f


def _overlap(y, w, d, o, others=()):
    """Hai nguoi noi CUNG LUC (tep goc + tep goc khac cung ngon ngu, 10 s, cung muc tieng noi do doc lap) -> dong 'Noi chong <= 10 %' (k1x)."""
    a, b = dense(y, 10.0), dense(o, 10.0)
    la, lb = speech_level_db(a), speech_level_db(b)
    b = b * 10 ** (((la if la is not None else -25.0) - (lb if lb is not None else -25.0)) / 20)
    z = a + b
    wr(d + ".wav", 0.9 * z / (np.max(np.abs(z)) + 1e-12), subtype="FLOAT")


def speech_level_db(y: np.ndarray, sr: int = SR) -> float | None:
    """Muc tieng noi trung binh (dBFS) DO DOC LAP: cong suat trung binh tren khung > (phan vi 10 + 10 dB) va > -60 dBFS."""
    db = energy_db(y, sr)
    sel = (db > np.percentile(db, 10) + 10) & (db > -60)
    return float(10 * np.log10(np.mean(10 ** (db[sel] / 10)))) if sel.any() else None


def _speech_level(target: float):
    """Doi muc ca tep sao cho muc tieng noi (do doc lap) = target dBFS (ca bien dong 'Am luong >= -40 dBFS')."""
    def f(y, w, d, o):
        lv = speech_level_db(y)
        wr(d + ".wav", y * 10 ** ((target - (lv if lv is not None else -20.0)) / 20), subtype="FLOAT")
    return f


def stereo_rho(x: np.ndarray, sr: int) -> tuple:
    """(tuong quan Pearson, tuong quan lon nhat khi can lech <= 50 ms tren 60 s dau) - cai DOC LAP voi vaudit.k1 (scipy.signal.correlate)."""
    from scipy.signal import correlate
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    r0 = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 1.0
    n = min(len(a), 60 * sr)
    a0, b0 = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    c = correlate(b0, a0, mode="full", method="fft")
    k = int(LAG_S * sr)
    rl = float(np.max(np.abs(c[n - 1 - k:n + k])) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))
    return r0, rl


def declared_kbps(path: str) -> float | None:
    """Toc do bit KHAI trong tep (ffprobe: luong, khong co thi dinh dang) - dinh nghia cua dong 'Toc do bit nen'."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], capture_output=True, text=True,
                           timeout=120, encoding="utf-8", errors="replace")
        info = json.loads(r.stdout or "{}")
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError):
        return None
    st = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), {})
    br = int(st.get("bit_rate") or 0) or int(info.get("format", {}).get("bit_rate") or 0)
    return br / 1000 if br else None


def EDGE_CASES() -> list:
    """Ca bien / luoi them 12/10 (audit C-04..C-13). Ten khong trung tien to voi ca cu (de 'build --only' them rieng)."""
    CLIP, MANY = ("cắt đỉnh",), ("nhiều người nói",)
    C = [("dosau.flac8", lambda y, w, d, o: sf.write(d + ".flac", y, SR, format="FLAC", subtype="PCM_S8"), E("04", "độ sâu bit dưới 16"))]
    # --- luoi toc do bit: toc do KHAI (ffprobe) quyet dinh canh bao; bang tan do doc lap quyet dinh 04 ---
    for kb in (20, 28):
        C.append((f"bitrate.opus_{kb}k", _enc(".opus", "-c:a", "libopus", "-b:a", f"{kb}k"), E("bw", kbps="probe")))
        C.append((f"bitrate.aac_adts_{kb}k", _enc(".aac", "-c:a", "aac", "-b:a", f"{kb}k"), E("bw", kbps="probe")))
        C.append((f"bitrate.vorbis_{kb}k", _enc(".ogg", "-c:a", "libvorbis", "-b:a", f"{kb}k"), E("bw", kbps="probe")))
    C.append(("bitrate.mp3_vbr_q9", _enc(".mp3", "-q:a", "9"), E("bw", kbps="probe")))
    C.append(("bitrate.mp3_32k_44k", lambda y, w, d, o: ff(w, d + ".mp3", "-ar", "44100", "-ac", "1", "-b:a", "32k"), E("bw", kbps="probe")))
    C.append(("bitrate.aac_24k_48k", lambda y, w, d, o: ff(w, d + ".m4a", "-ar", "48000", "-ac", "1", "-c:a", "aac", "-b:a", "24k"),
              E("bw", kbps="probe")))
    # --- tan so: Opus tu nguon 8 kHz (khai 48 kHz) -> chi dong 'bang hep' bat duoc; 192 kHz -> ha ve 16 kHz ---
    C.append(("tanso.opus_tu_8k", _two_step(".opus", (".wav", "-ar", "8000", "-ac", "1"), ("-c:a", "libopus", "-b:a", "32k")), E("bw", kbps="probe")))
    C.append(("tanso.192000", lambda y, w, d, o: ff(w, d + ".wav", "-ar", "192000", "-ac", "1"), E()))
    # --- do dai sat bien 3 s / 10 phut ---
    C.append(("dodai.2.9s", lambda y, w, d, o: wr(d + ".wav", dense(y, 2.9)), E("04", "độ dài ngoài 3 giây – 10 phút")))
    C.append(("dodai.3.0s_wav", lambda y, w, d, o: wr(d + ".wav", dense(y, 3.0)), E("spd")))
    C.append(("dodai.3.0s_mp3", _dense_enc(3.0, ".mp3", "-ar", "16000", "-ac", "1", "-b:a", "128k"), E("spd", kbps=128)))
    C.append(("dodai.599.5s", lambda y, w, d, o: wr(d + ".wav", tile(y, 599.5)), E(long=True)))
    C.append(("dodai.600.5s", lambda y, w, d, o: wr(d + ".wav", tile(y, 600.5)), E("04", "độ dài ngoài 3 giây – 10 phút", long=True)))
    # --- hai kenh quanh nguong (dap an theo tuong quan do lai; kenh tron hai giong -> 'nhieu nguoi noi' khong tinh la loi) ---
    for v in (0.25, 0.35):
        C.append((f"kenh.rho_{v}", _stereo_mix("rho", v), E("st", allow=MANY + ("nói chồng",))))      # kenh phai tron 2 giong
    for ms in (40, 60):
        C.append((f"kenh.lech_{ms}ms", _stereo_mix("lech", ms / 1000), E("st", allow=MANY)))
    for v in (0.998, 0.9995):
        C.append((f"kenh.gan_giong_{v}", _stereo_mix("gan_giong", v), E("st", allow=MANY)))
    # --- kenh co duoi 2 s tieng noi -> bo kenh (r5, review 12/10): 1,9 / 2,1 s do doc lap tren chinh kenh do, vung dem 0,06 s ---
    for t in (1.9, 2.1):
        C.append((f"kenh.kenh2_{t}s", _kenh2(t), E("st", allow=MANY, ch_sp=True)))
    # --- bang hep that: thoai 8 kHz nang len 16 kHz (khai 16 kHz / 16 bit, chi pho lo ra). Dap an THEO CACH DUNG (r5): luon 04 'bang hep';
    #     them san on trang -80 / -60 dBFS (audit C-10: on deu lap 4-8 kHz tung lam K1 doc 8000 Hz) ---
    BH = E("04", "băng hẹp")
    g711 = _two_step(".wav", (".wav", "-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"), ("-ar", "16000", "-c:a", "pcm_s16le"))
    amr = _two_step(".wav", (".amr", "-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb", "-b:a", "12.2k"), ("-ar", "16000", "-c:a", "pcm_s16le"))
    C += [("tinhieu.g711_len_16k", g711, BH), ("tinhieu.amrnb_len_16k", amr, BH), ("tinhieu.g711_on_-80dBFS", _noisy(g711, -80), BH),
          ("tinhieu.g711_on_-60dBFS", _noisy(g711, -60), BH), ("tinhieu.amrnb_on_-60dBFS", _noisy(amr, -60), BH),
          ("tinhieu.rong_on_-60dBFS", _noisy(lambda y, w, d, o: wr(d + ".wav", y, subtype="FLOAT"), -60), E())]
    C.append(("tinhieu.bangtan_5.5k", _lowpass(5500), E("bw")))
    # --- cat dinh trong tep NEN (r5, review 12/10: ban cu 0 % tren MP3 / AAC / Opus cat 8 %) - K1 uoc luong theo duoi bien do, sai so ~ +-1 diem %:
    #     0 % -> khong canh bao, 1,4 / 3 % -> 'cat dinh', 8 % -> 04 (5,6 % sat nguong sau ma hoa -> khong dung) ---
    for cname, ext, args, kb in (("mp3", ".mp3", ("-b:a", "128k"), 128), ("aac", ".m4a", ("-c:a", "aac", "-b:a", "128k"), 128),
                                 ("opus", ".opus", ("-c:a", "libopus", "-b:a", "48k"), 48)):
        C += [(f"tinhieu.catdinh_nen_0pct_{cname}", _clip_enc(0, ext, *args), E("bw", kbps=kb)),
              (f"tinhieu.catdinh_nen_1.4pct_{cname}", _clip_enc(1.4, ext, *args), E("bw", must=CLIP, kbps=kb)),
              (f"tinhieu.catdinh_nen_3pct_{cname}", _clip_enc(3.0, ext, *args), E("bw", must=CLIP, kbps=kb)),
              (f"tinhieu.catdinh_nen_8pct_{cname}", _clip_enc(8.0, ext, *args), E("04", "cắt đỉnh nặng", kbps=kb))]
    # --- noi chong (r5): hai nguoi noi cung luc cung muc -> 'noi chong' khi chay --extended (k1x); che do thuong: khong yeu cau ---
    C.append(("nguoi.noi_chong_2", _overlap, E(allow=MANY, ext_must=("nói chồng",))))
    # --- muc tieng noi sat nguong -40 dBFS (muc do doc lap, vung dem 1 dB) ---
    for t in (-41.5, -38.5):
        C.append((f"tinhieu.muc_noi_{t}dBFS", _speech_level(t), E("lvl")))
    # --- cat dinh sat nguong 1 % / 5 % (cach nguong >= 0,3 diem %) ---
    C += [("tinhieu.catdinh_0.7pct", _clip(0.7), E()), ("tinhieu.catdinh_1.4pct", _clip(1.4), E(must=CLIP)),
          ("tinhieu.catdinh_4.6pct", _clip(4.6), E(must=CLIP)), ("tinhieu.catdinh_5.6pct", _clip(5.6), E("04", "cắt đỉnh nặng")),
          ("tinhieu.catdinh_5.6pct_48k", _clip(5.6, 48000), E("04", "cắt đỉnh nặng"))]
    # --- mat tin hieu: nguong 0,1 s, nhieu bo ma hoa, kiem MOC ---
    DROP = (DROP_WARN,)
    C += [("tinhieu.matTH_0.09s", _drop(0.09), E()), ("tinhieu.matTH_0.12s", _drop(0.12), E(must=DROP, drop=0.12)),
          ("tinhieu.matTH_1.0s", _drop(1.0), E(must=DROP, drop=1.0)),
          ("tinhieu.matTH_0.3s_aac32k", _drop(0.3, ".m4a", "-c:a", "aac", "-b:a", "32k"), E("bw", must=DROP, kbps="probe", drop=0.3, drop_tol=0.04)),
          ("tinhieu.matTH_0.3s_opus24k", _drop(0.3, ".opus", "-c:a", "libopus", "-b:a", "24k"),
           E("bw", must=DROP, kbps="probe", drop=0.3, drop_tol=0.04))]
    return C


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LANG_REAL = os.path.join(ROOT, "data_labels", "v6", "k1axes", "lang_real")
LANG_WARN = "ngôn ngữ khác"


def lang_real(root: str = LANG_REAL, out_md: str | None = None) -> dict:
    """Ca DUONG dong 'Ngon ngu: tieng Viet / tieng Anh' (C-19): doan 30 s sach noi LibriVox (PD / CC0, ngon ngu khac vi / en; nguon + giay
    phep: <root>/sources.json) qua K1 nhu SAN PHAM (extended=True) -> phai co canh bao 'ngon ngu khac' (sach noi: AST khong chan, xem events).
    Tra {n, hit, rows}; ghi .tmp_v6/results/k1_lang_real.md."""
    from vaudit import k1
    doc = json.load(open(os.path.join(root, "sources.json"), encoding="utf-8"))
    rows = []
    for s in doc["files"]:
        r = k1.check(os.path.join(root, s["file"]), extended=True)
        per = r["measures"].get("kenh", [{}])
        rows.append({"file": s["file"], "lang": s.get("lang"), "license": s.get("license"), "decision": r["decision"], "reasons": r["reasons"][:2],
                     "warnings": r["warnings"], "top": [(q.get("ngon_ngu") or {}).get("top") for q in per],
                     "p_vi_en": [round((q.get("ngon_ngu") or {}).get("vi", 0) + (q.get("ngon_ngu") or {}).get("en", 0), 3) for q in per],
                     "ok": LANG_WARN in r["warnings"]})
    hit = sum(r["ok"] for r in rows)
    L = [f"# K1 --extended tren doan THAT ngon ngu khac (LibriVox PD / CC0): {hit}/{len(rows)} co canh bao '{LANG_WARN}'\n",
         "| tep | ngon ngu | K1 | LID top | p(vi)+p(en) | canh bao / ly do |", "|---|---|---|---|---|---|"]
    L += [f"| {r['file']} | {r['lang']} | {r['decision']} | {r['top']} | {r['p_vi_en']} | {'; '.join(r['warnings'] or r['reasons'])[:90]} |" for r in rows]
    out_md = out_md or os.path.join(ROOT, ".tmp_v6", "results", "k1_lang_real.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return {"n": len(rows), "hit": hit, "rows": rows}
