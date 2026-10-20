# -*- coding: utf-8 -*-
"""v6e / C2 - TEST K1 THEO TUNG DONG CONSTRAINT bang TEP THAT. Ban 10/10 (WP2, audit B-5 / B-6 / C-03..C-16):
  - tep goc: >= 20 tep that tu nhien (natvi + naten, kind nguyen_ban, cfg goc), moi nguoi mot tep, chia deu theo nguon;
  - DAP AN TINH LAI LUC CHAY tu dinh nghia ca (CASES) + so do DOC LAP tren tep da dung (bang tan Welch, khong dung vaudit.k1) +
    so do cua tep goc (canh bao 'thua huong', vd nhieu nguoi noi): run() KHONG doc dap an tu cases.json (cases.json chi ghi tep -> ca, tham so);
  - kiem QUYET DINH, cum tu LY DO 04 (dung chu hop dong) va CANH BAO: canh bao mong doi phai co, canh bao truc kiem khong duoc co.
Dap an suy tu bang Constraint (6muc.txt). Diem lech cua chinh hop dong: 'Toc do bit nen < 24 kbps -> chi canh bao nen manh' nhung bo ma hoa
o 16-24 kbps (MP3 / AAC) cat dai tren < 6 kHz -> dong 'Dai tan thuc >= 6 kHz' bien tep thanh 04 'bang hep'. Dap an: canh bao 'nen manh' theo
toc do bit khai; quyet dinh theo bang tan do doc lap (vung dem +-8 dB quanh nguong -> chap nhan ca hai). Opus 16 kbps giu bang rong -> nhan + 'nen manh'.
PHU DONG CONSTRAINT (ROW_COVERAGE ben duoi, in ca vao k1_axes.md): moi dong ghi noi kiem - 'tep that o day' / 'unit test vaudit.k1' /
  'vaudit.cond trong pipeline' / 'khong tu kiem duoc'. Han che da biet (KNOWN_FAIL): nguoi.6_nguoi (dem nguoi chi ra 3-4 tren 6 nguoi noi tiep;
  luat 5 % cum lam 04 sai 6-16 % tren AMI that -> khong doi; D:/tmp_va/wp2/spk_ami.log).
Ca BIEN / LUOI (12/10, audit C-04..C-13): FLAC 8 bit; luoi toc do bit (Opus / AAC ADTS / Vorbis 20 & 28 kbps, MP3 VBR, MP3 44,1 kHz,
  AAC 48 kHz) voi toc do KHAI doc bang ffprobe (kbps='probe'); Opus tu nguon 8 kHz (Opus luon khai 48 kHz -> chi dong bang tan chan);
  do dai 2,9 / 3,0 s (+ MP3) va 599,5 / 600,5 s (--long); hai kenh quanh nguong (tuong quan 0,25 / 0,35, lech 40 / 60 ms, gan giong 0,998 /
  0,9995: dap an tu tuong quan DO DOC LAP + vung dem); bang hep that (G.711 / AMR-NB nang len 16 kHz, loc 5,5 kHz); muc tieng noi -41,5 /
  -38,5 dBFS (muc do doc lap, vung dem 1 dB); cat dinh 0,7 / 1,4 / 4,6 / 5,6 % (+ 48 kHz); mat tin hieu 0,09 / 0,12 / 1,0 s + AAC 32k /
  Opus 24k, KIEM CA MOC (warn_detail) so voi vi tri chen (+-20 ms WAV, +-40 ms tep nen).
Them r5 (review 12/10): cat dinh trong tep NEN (MP3 / AAC / Opus 0 / 1,4 / 3 / 8 %); bang hep + san on trang (G.711 -80 / -60 dBFS, AMR-NB -60) va
  doi chung bang rong + on -60 dBFS; G.711 / AMR-NB: dap an THEO CACH DUNG (luon 04 'bang hep'); kenh co 1,9 / 2,1 s tieng noi (do doc lap tren
  chinh kenh); hai nguoi noi cung luc (nguoi.noi_chong_2: 'noi chong' khi --extended). --extended = K1 nhu SAN PHAM (voiceaudit_v9 k1x=True):
  them ngon ngu + noi chong (vaudit.k1x), must_not 'ngon ngu khac' / 'noi chong' tren moi ca khong thua huong tu tep goc.
DO DOC LAP hay KIEM NHAT QUAN: oracle bang tan (Welch 6-7 kHz) va thoi luong tieng noi la cach do KHAC K1; con oracle tuong quan hai kenh
  (st: corrcoef + tuong quan lon nhat khi lech <= 50 ms), muc tieng noi (lvl) va toc do bit khai (kbps='probe', ffprobe) la CUNG dinh nghia voi K1,
  cai lai doc lap -> cac ca do chi kiem K1 AP NGUONG dung (kiem nhat quan), khong kiem phep do co dung voi su that hay khong.
  python -m v6.coverage.k1_axes build [--bases 20] [--long-bases 3] [--only a,b,...]   (--only: chi dung / them cac ca co ten bat dau bang
                                                                        mot trong cac tien to, tren CUNG tep goc cua cases.json cu, giu phan con lai;
                                                                        '@bien' = moi ca bien / luoi)
                                                                        dung tep -> data_labels/v6/k1axes/v2/<ca>__<goc>.<duoi> + cases.json
  python -m v6.coverage.k1_axes run [--limit N] [--bases K] [--base-ids id1,id2] [--only a,b] [--no-events] [--long] [--extended] [--tag ten]
                                                                        (--long: them ca 120-615 s, NANG; --tag: ghi k1_axes_<ten>.md / .json,
                                                                        khong ghi de bao cao chay day du)
                                                                        chay vaudit.k1.check, so dap an -> .tmp_v6/results/k1_axes[_<tag>].md / .json
                                                                        (+ k1_axes[_<tag>]_base.json: so do tep goc, de gop)
  python -m v6.coverage.k1_axes merge --tags t1,t2,... [--tag ten]     gop cac lan chay tung phan (moi phan < 30 phut) -> k1_axes[_<ten>].md / .json
  python -m v6.coverage.k1_axes lang                                   ca duong ngon ngu khac (LibriVox, --extended) -> k1_lang_real.md
  python -m v6.coverage.k1_axes test                                   test oracle + dinh nghia ca (khong can tep that, khong nap mo hinh)
                                                                        = python -m v6.coverage.test_k1_axes; ca bien: v6/coverage/k1_axes_edge.py"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
import soundfile as sf  # noqa: E402

SR = 16000
OUT = os.path.join(BASE, "data_labels", "v6", "k1axes", "v2")
V4 = os.path.join(BASE, "data_labels", "realistic_v4")
RES = os.path.join(BASE, ".tmp_v6", "results")
N_BASE = 20
N_LONG = 3                       # tep goc dung cho ca do dai 120-615 s (ton dia / thoi gian)
CUT_S = 20.0                     # cat tep goc <= 20 s (dia)
BW_GUARD_DB = 8.0                # vung dem quanh nguong bang tan doc lap
BW_THR_DB = -50.0                # mat do pho 6-7 kHz so voi 1-3 kHz (dB): duoi nguong = khong co nang luong toi 6 kHz
KBPS_GUARD = (23.0, 25.0)        # toc do bit khai trong vung nay: canh bao 'nen manh' co / khong deu chap nhan
SAME_RHO, RHO0_GUARD = 0.999, 0.0002     # 'giong het nhau (tuong quan > 0,999) -> gop mot'; vung dem quanh nguong
SCENE_RHO, RHO_GUARD = 0.3, 0.03         # 'hai micro cung canh (tuong quan >= 0,3 sau khi can lech <= 50 ms)'
LAG_S = 0.05
CH_SP_S, CH_SP_GUARD = 2.0, 0.06         # 'kenh co duoi 2 s tieng noi -> bo kenh do'; vung dem 2 khung 30 ms
LVL_THR_DB, LVL_GUARD_DB = -40.0, 1.0    # 'muc tieng noi trung binh >= -40 dBFS'
DROP_FRAC = 0.4                          # _drop chen doan so 0 tai 40 % do dai tep goc
AXIS_WARN = ("nén mạnh", "cắt đỉnh", "có đoạn mất tín hiệu")      # canh bao do truc tep / tin hieu sinh ra (loi 'canh bao thua')
EDGE_WARN = "0,3 s đầu và cuối không được soi"                   # Constraint 'Dau va cuoi tep': LUON co khi tep duoc nhan
DROP_WARN = "có đoạn mất tín hiệu"
# 11/10: must_not = MOI canh bao hop dong khong mong doi, khong 'chua chac', khong thua huong tu tep goc (ca canh bao mo hinh: nhieu nguoi
# noi, kieu noi ...) - phep bien doi tep (ma hoa, doi muc, loc) lam xuat hien canh bao mo hinh moi la mat on dinh can bao (loi
# 'canh bao moi khong co o tep goc'), tach rieng voi loi truc (AXIS_WARN) trong bang ket qua.
MUSAN = os.path.join(BASE, "data", "musan_noise", "noise")
# Moi dong Constraint (6muc.txt) -> noi kiem (review 11/10: phai ghi ro dong nao KHONG kiem o day va vi sao)
ROW_COVERAGE = [
    ("Số tệp: đúng 1", "unit test vaudit.k1 (thu muc / danh sach 2 tep / chuoi rong / tep khong ton tai -> tu_choi); khong co 'tep' de dung ca; "
     "ma thoat 2 cua CLI san pham thuoc WP3"),
    ("Dung lượng ≤ 250 MB", "tep that o day (dungluong.vuot_nguong: ha MAX_BYTES = kich thuoc - 1 luc chay) + unit test test_size_limit_before_decode "
     "(MB theo SI = 250 000 000 byte; dung bang nguong -> nhan)"),
    ("Định dạng / mã hoá", "tep that o day (dinhdang.*: 8 duoi, ALAC trong .m4a, duoi viet hoa, .wma, noi dung hong, 0 byte) + robust_inputs "
     "(cut cut, MP3 mang duoi .wav); DRM: KHONG kiem - khong co tep DRM mau"),
    ("Tốc độ bit nén < 24 kbps -> cảnh báo", "tep that o day (bitrate.*: MP3 16/24/32/64 + VBR + 44,1 kHz, AAC m4a 16/32 + 48 kHz, AAC ADTS / Opus / "
     "Vorbis 20 & 28 kbps; toc do KHAI doc bang ffprobe; xem diem lech bang tan o tren)"),
    ("Tần số lấy mẫu ≥ 16 kHz", "tep that o day (tanso.* 8 k-192 kHz; tanso.opus_tu_8k: Opus luon khai 48 kHz -> chi dong bang tan bat duoc)"),
    ("Độ sâu bit ≥ 16", "tep that o day (dosau.*: PCM 8 bit / A-law / mu-law, FLAC 8 bit theo bits_per_raw_sample, 16-32 bit, float)"),
    ("Số kênh / hai kênh", "tep that o day (kenh.*: 3/4/6 kenh; hai kenh giong / gan giong 0,998-0,9995 / hai mic / lech 40-60 ms / tuong quan "
     "0,25-0,35 / mot kenh im / kenh phai 1,9 & 2,1 s tieng noi / hai nguon) + unit test test_two_channel_correlation_rule; 0 kenh: khong tao duoc tep"),
    ("Độ dài 3 giây – 10 phút", "tep that o day (dodai.*: 2,5 / 2,9 / 3,0 / 3,0 MP3 / 3,2 s; 120-615 s gom 599,5 / 600,5 s chi khi --long)"),
    ("Dải tần thực ≥ 6 kHz", "tep that o day (tinhieu.bangtan_* 5 / 5,5 / 7 kHz, G.711 va AMR-NB nang len 16 kHz - co / khong san on trang -80 / "
     "-60 dBFS, doi chung bang rong + on -60 dBFS, bitrate.*, tanso.opus_tu_8k); han che: bang rong + on trang phang rat to (-40 dBFS) co the bi goi bang hep"),
    ("Âm lượng tiếng nói ≥ −40 dBFS", "tep that o day (tinhieu.quanho / amluong / muc_noi -41,5 & -38,5 dBFS theo muc tieng noi do doc lap)"),
    ("Cắt đỉnh ≤ 1 % / > 5 % -> 04", "tep that o day (tinhieu.catdinh_* 0,5 / 0,7 / 1,4 / 3 / 4,6 / 5,6 / 8 %, 3 % o 44,1 kHz, 5,6 % o 48 kHz); "
     "tep NEN: tinhieu.catdinh_nen_* (MP3 128k / AAC 128k / Opus 48k x 0 / 1,4 / 3 / 8 %) - K1 chi UOC LUONG theo duoi bien do (binh nguyen vo "
     "khi giai ma; sai so ~ +-1 diem %, Opus vot dinh lon do thieu); not_impl cua tep nen ghi gioi han nay"),
    ("Mất tín hiệu", "tep that o day (tinhieu.matTH_* 0,05 / 0,09 / 0,12 / 0,3 / 1,0 s, 0,3 s trong MP3 64k / AAC 32k / Opus 24k; kiem ca moc)"),
    ("Lượng tiếng nói ≥ 2 s và ≥ 30 %", "tep that o day (tinhieu.tiengnoi_*, dodai.3.2s, loai.chi_tieng_on, loai.am_thuan)"),
    ("Số người nói ≤ 4", "tep that o day (nguoi.*); nguoi.6_nguoi la KNOWN_FAIL"),
    ("Loại âm thanh: lời nói ghi âm (không nhạc / podcast / phát thanh)", "mot phan o day (loai.chi_tieng_on, loai.am_thuan: on / am thuan -> 04); "
     "doan THAT (python -m vaudit.events real, 28 doan archive.org CC BY / BY-SA / PD dung duoc, k1_events_real.md; 16 doan NC / ND cu 'khong_dung'): "
     "bai hat co loi 12/12 -> 04, nhac KHONG loi chi 3/6 (ambient / piano: music TB 0,29-0,60 < 0,6); sach noi 0/5, "
     "phat thanh cu 0/4, podcast 0/1 -> KHONG chan (AST nghe la loi noi; ghi trong not_impl); tieng noi hop le natvi / naten 0/40 bi 04"),
    ("Kiểu nói: không hát / thì thầm / la hét; người lớn", "KHONG o day: chua co du lieu that thi tham / la het; ca tong hop WORLD (thi tham / hat / la het) "
     "AST van nghe la 'Speech' -> khong dung lam ca duong (events.py docstring); bai hat THAT co loi: nhom 'sing' max trung vi 0,07 (12 doan CC) "
     "/ 0,10 (10 doan NC cu) < 0,5 -> phan 'hat' KHONG kich hoat; tre em chi ghi so do"),
    ("Ngôn ngữ vi / en; Nói chồng ≤ 10 %", "tep that o day KHI run --extended (che do mac dinh cua san pham; vaudit/k1x.py whisper-small LID + pyannote "
     "chong lan): must_not 'ngôn ngữ khác' / 'nói chồng' tren moi ca vi / en, ca duong nguoi.noi_chong_2; ca duong ngon ngu khac: 7 doan sach noi "
     "LibriVox PD (de / fr / es / it / ja / zh, data_labels/v6/k1axes/lang_real, python -m v6.coverage.k1_axes lang)"),
    ("SNR ≥ 10 dB; nền ≤ 10 %; RT60 ≤ 1,0 s", "vaudit.cond trong pipeline (XLS-R), KHONG o K1 -> khong kiem o day; so lieu: cond_calib.md"),
    ("Cách ghi (một lần, một thiết bị, micro ≤ 2 m); tốc độ phát ± 5 %", "khong tu kiem duoc (ghi trong not_impl)"),
    ("Đầu và cuối tệp 0,3 s", "tep that o day: canh bao '0,3 s đầu và cuối không được soi' bat buoc tren moi tep duoc nhan"),
]
KNOWN_FAIL = {"nguoi.6_nguoi": "dem nguoi resemblyzer chi ra > 4 o ~55 % tep ghep 6 nguoi (12/10, spk_diag: 11/20); luat nhay hon lam doi so dem "
                                "tren bien the cung noi dung / tren AMI that (.tmp_v6/results/wp2_diag/spk_diag.py eval)",
              "tinhieu.matTH_0.3s_opus24k": "Opus 24k lap doan so 0 bang nhieu ~ -85 dBFS (dinh toi 1e-3) -> nguong |y| < 3e-5 chi bat ~55 %; "
                                             "noi nguong (RMS 10 ms < -85 dBFS) bat 100 % nhung bao sai > 0,1 s tren 2,3 % tep nen khong co "
                                             "mat tin hieu (0,2 % voi nguong hien tai) - .tmp_v6/results/wp2_diag/diag3.py; giu nguong. He qua phu: "
                                             "khi doan do KHONG bi bat, cua so dem nguoi chua no khong bi bo (nhieu Opus khong phai so 0) -> nhung "
                                             "lech -> 'nhieu nguoi noi' gia (tep goc vi 0009, 12/10)"}


def ff(src: str, dst: str, *args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, *args, dst], check=True, timeout=600)


# ------------------------------------------------------------------ chon tep goc
def pick(n: int = N_BASE) -> list:
    """n tep goc hop le (k1 da luu: nhan / canh_bao, >= 8 s, 1 nguoi noi, bang tan >= 7 kHz, am luong >= -35 dBFS), moi nguoi 1 tep,
    xoay vong theo nguon trong moi ngon ngu (vi: vss / celeb / vox; en: voxp / edacc / ami), nua vi nua en."""
    out = []
    for lang, k in (("natvi", (n + 1) // 2), ("naten", n // 2)):
        man = json.load(open(os.path.join(BASE, "data_labels", "v6", lang, "manifest.json"), encoding="utf-8"))
        by_src: dict = {}
        seen = set()
        for e in sorted(man, key=lambda e: e["id"]):
            k1 = e["meta"].get("k1", {})
            ch = (k1.get("kenh") or k1.get("measures", {}).get("kenh") or [{}])[0]
            ok = (e["kind"] == "nguyen_ban" and e["cfg"] == "goc" and k1.get("decision") in ("nhan", "canh_bao") and not e["meta"].get("lockbox")
                  and not e["meta"].get("ngoai_pham_vi") and k1.get("measures", {}).get("do_dai_s", 0) >= 8 and (ch.get("so_nguoi_noi") in (None, 1))
                  and (ch.get("bang_thong_hz") or 0) >= 7000 and (ch.get("am_luong_dbfs") or -99) >= -35 and (ch.get("cat_dinh_pct") or 0) < 0.5)
            person = str(e["meta"].get("person") or e["meta"].get("speaker"))
            if ok and person not in seen:
                seen.add(person)
                by_src.setdefault(e["meta"].get("source", "?"), []).append(e)
        srcs = sorted(by_src)
        got, i = [], 0
        while len(got) < k and any(by_src.values()):
            s = srcs[i % len(srcs)]
            if by_src[s]:
                lst = by_src[s]
                got.append(lst.pop(len(lst) // 2) if len(got) % 2 else lst.pop(0))
            i += 1
        out += got
    return out


def tile(y: np.ndarray, sec: float) -> np.ndarray:
    n = int(sec * SR)
    return np.resize(y, n) if len(y) < n else y[:n]


def wr(path: str, y: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> None:
    sf.write(path, np.asarray(y, np.float32) if y.dtype != np.float64 else y, sr, subtype=subtype)


# ------------------------------------------------------------------ dinh nghia ca: ten -> (ham dung, tham so, dap an)
# Dap an (exp): dec = accept | 04 | tu_choi | bw (theo bang tan doc lap) | spd (theo ti le tieng noi);
#   reason = cum tu ly do 04 cua hop dong; must = canh bao phai co; kbps = toc do bit yeu cau (-> 'nen manh' theo KBPS_GUARD)
def E(dec: str = "accept", reason: str | None = None, must: tuple = (), kbps: float | None = None, long: bool = False, **kw) -> dict:
    return {"dec": dec, "reason": reason, "must": list(must), "kbps": kbps, "long": long, **kw}


def _enc(ext: str, *args: str):
    return lambda y, w, d, o: ff(w, d + ext, "-ar", "16000", "-ac", "1", *args)


def _stereo(kind: str):
    def f(y, w, d, o):
        n = len(y)
        rng = np.random.default_rng(3)
        lag = int(0.012 * SR)
        if kind == "giong_nhau":
            L = R = y
        elif kind == "hai_mic_cung_canh":                      # cung loi noi, lech 12 ms, nho hon, them nhieu rieng
            L, R = y, np.r_[np.zeros(lag), 0.7 * y[:-lag]] + rng.normal(0, 0.003, n)
        elif kind == "mot_kenh_im":
            L, R = y, rng.normal(0, 1e-4, n)
        else:                                                  # hai nguoi khac nhau, moi nguoi mot kenh (tep goc khac)
            L, R = y, tile(o, n / SR)[:n] * 0.8
        wr(d + ".wav", np.c_[L, R])
    return f


def _clip(pct: float, sr: int = SR):
    def f(y, w, d, o):
        t = float(np.quantile(np.abs(y), 1 - pct / 100))
        z = np.clip(y, -t, t) / t * 0.999
        if sr != SR:                                           # cat dinh o tan so goc 44,1 / 48 kHz (nang tan so truoc khi cat)
            from math import gcd

            from scipy.signal import resample_poly
            g = gcd(sr, SR)
            u = resample_poly(y, sr // g, SR // g)
            t = float(np.quantile(np.abs(u), 1 - pct / 100))
            z = np.clip(u, -t, t) / t * 0.999
        wr(d + ".wav", z, sr)
    return f


def _drop(sec: float, ext: str = ".wav", *args: str):
    def f(y, w, d, o):
        z = y.copy()
        a = int(len(y) * 0.4)
        z[a:a + int(sec * SR)] = 0.0
        if ext == ".wav":
            wr(d + ".wav", z)
        else:
            tmp = d + "__tmp.wav"
            wr(tmp, z)
            ff(tmp, d + ext, "-ar", "16000", "-ac", "1", *args)
            os.remove(tmp)
    return f


def _level(dbfs: float):
    return lambda y, w, d, o: wr(d + ".wav", y * 10 ** (dbfs / 20) / (np.sqrt(np.mean(y ** 2)) + 1e-9), subtype="FLOAT")


def _pad_speech(share: float):
    """Them tieng phong (-70 dBFS) sau loi noi de ti le tieng noi ~ share (theo tieng_noi_s cua tep goc)."""
    def f(y, w, d, o, sp_s=None):
        sp_s = sp_s or 0.6 * len(y) / SR
        total = max(len(y) / SR, sp_s / share)
        pad = np.random.default_rng(5).normal(0, 10 ** (-70 / 20), int(total * SR) - len(y))
        wr(d + ".wav", np.r_[y, pad])
    return f


def _concat_people(k: int):
    def f(y, w, d, o, others=()):
        parts = [y] + [z for z in others[:k - 1]]
        wr(d + ".wav", np.concatenate([p[:int(8 * SR)] for p in parts]))
    return f


def _lowpass(fc: float):
    def f(y, w, d, o):
        from scipy.signal import butter, sosfiltfilt
        wr(d + ".wav", sosfiltfilt(butter(8, fc, "low", fs=SR, output="sos"), y), subtype="FLOAT")
    return f


def _bits(codec: str, ext: str = ".wav", sr: int = 16000):
    return lambda y, w, d, o: ff(w, d + ext, "-c:a", codec, "-ar", str(sr), "-ac", "1")


def CASES() -> list:
    C = []
    # --- DINH DANG ---
    for ext, args, kb in ((".wav", [], None), (".flac", [], None), (".mp3", ["-b:a", "128k"], 128), (".m4a", ["-c:a", "aac", "-b:a", "128k"], 128),
                          (".ogg", ["-c:a", "libvorbis", "-q:a", "5"], None), (".opus", ["-c:a", "libopus", "-b:a", "48k"], 48),
                          (".aac", ["-c:a", "aac", "-b:a", "96k"], 96)):
        C.append((f"dinhdang{ext}", _enc(ext, *args), E("bw", kbps=kb)))
    C.append(("dinhdang.amr_wb", lambda y, w, d, o: ff(w, d + ".amr", "-c:a", "libvo_amrwbenc", "-ar", "16000", "-ac", "1", "-b:a", "23.85k"),
              E("bw", kbps=23.85)))
    C.append(("dinhdang.m4a_alac", _bits("alac", ".m4a"), E()))
    C.append(("dinhdang.WAV_hoa", lambda y, w, d, o: wr(d + ".WAV", y), E()))
    C.append(("dinhdang.amr_nb_8k", lambda y, w, d, o: ff(w, d + ".amr", "-c:a", "libopencore_amrnb", "-ar", "8000", "-ac", "1", "-b:a", "12.2k"),
              E("04", "tần số quá thấp", kbps=12.2)))
    C.append(("dinhdang.wma_ngoai_danh_sach", lambda y, w, d, o: ff(w, d + ".wma", "-ar", "16000", "-ac", "1"), E("04", "không đọc được tệp")))
    C.append(("dinhdang.hong_noi_dung", lambda y, w, d, o: open(d + ".wav", "wb").write(b"khong phai am thanh" * 100), E("04", "không đọc được tệp")))
    C.append(("dinhdang.rong_0_byte", lambda y, w, d, o: open(d + ".mp3", "wb").close(), E("04", "không đọc được tệp")))
    # --- TOC DO BIT (canh bao theo toc do khai, quyet dinh theo bang tan do doc lap) ---
    for kb in (16, 24, 32, 64):
        C.append((f"bitrate.mp3_{kb}k", _enc(".mp3", "-b:a", f"{kb}k"), E("bw", kbps=kb)))
    for kb in (16, 32):
        C.append((f"bitrate.aac_{kb}k", _enc(".m4a", "-c:a", "aac", "-b:a", f"{kb}k"), E("bw", kbps=kb)))
        C.append((f"bitrate.opus_{kb}k", _enc(".opus", "-c:a", "libopus", "-b:a", f"{kb}k"), E("bw", kbps=kb)))
    # --- TAN SO LAY MAU ---
    for sr in (16000, 22050, 32000, 44100, 48000, 96000):
        C.append((f"tanso.{sr}", lambda y, w, d, o, sr=sr: ff(w, d + ".wav", "-ar", str(sr), "-ac", "1"), E()))
    for sr in (11025, 8000):
        C.append((f"tanso.{sr}", lambda y, w, d, o, sr=sr: ff(w, d + ".wav", "-ar", str(sr), "-ac", "1"), E("04", "tần số quá thấp")))
    # --- DO SAU BIT ---
    for name, codec in (("16bit", "pcm_s16le"), ("24bit", "pcm_s24le"), ("32bit", "pcm_s32le"), ("float32", "pcm_f32le"), ("float64", "pcm_f64le")):
        C.append((f"dosau.{name}", _bits(codec), E()))
    C.append(("dosau.flac24", lambda y, w, d, o: ff(w, d + ".flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", "-ar", "16000", "-ac", "1"), E()))
    for name, codec in (("8bit", "pcm_u8"), ("alaw", "pcm_alaw"), ("mulaw", "pcm_mulaw")):
        C.append((f"dosau.{name}", _bits(codec), E("04", "độ sâu bit dưới 16")))
    # --- SO KENH ---
    C += [("kenh.stereo_giong_nhau", _stereo("giong_nhau"), E(n_ch=1)), ("kenh.stereo_hai_mic_cung_canh", _stereo("hai_mic_cung_canh"), E(n_ch=2)),
          ("kenh.stereo_mot_kenh_im", _stereo("mot_kenh_im"), E(n_ch=1)),
          ("kenh.stereo_hai_nguon_rieng", _stereo("hai_nguon_rieng"), E("04", "hai kênh là hai nguồn riêng — gửi bản một kênh"))]
    for k in (3, 4, 6):
        C.append((f"kenh.{k}_kenh", lambda y, w, d, o, k=k: wr(d + ".wav", np.tile(y[:, None], (1, k))), E("04", "quá 2 kênh")))
    # --- DO DAI ---
    C.append(("dodai.2.5s", lambda y, w, d, o: wr(d + ".wav", tile(y, 2.5)), E("04", "độ dài ngoài 3 giây – 10 phút")))
    C.append(("dodai.3.2s", lambda y, w, d, o: wr(d + ".wav", dense(y, 3.2)), E("spd")))
    for sec, exp in ((120, E(long=True)), (300, E(long=True)), (590, E(long=True)), (615, E("04", "độ dài ngoài 3 giây – 10 phút", long=True))):
        C.append((f"dodai.{sec}s", lambda y, w, d, o, sec=sec: wr(d + ".wav", tile(y, sec)), exp))
    # --- TIN HIEU ---
    C.append(("tinhieu.bangtan_5k", _lowpass(5000), E("bw")))
    C.append(("tinhieu.bangtan_7k", _lowpass(7000), E("bw")))
    C.append(("tinhieu.quanho_-45dBFS", _level(-45), E("04", "quá nhỏ")))
    C.append(("tinhieu.amluong_-30dBFS", _level(-30), E()))
    C += [("tinhieu.catdinh_0.5pct", _clip(0.5), E()), ("tinhieu.catdinh_3pct", _clip(3.0), E(must=("cắt đỉnh",))),
          ("tinhieu.catdinh_8pct", _clip(8.0), E("04", "cắt đỉnh nặng")), ("tinhieu.catdinh_3pct_44k", _clip(3.0, 44100), E(must=("cắt đỉnh",)))]
    C += [("tinhieu.matTH_0.05s", _drop(0.05), E()), ("tinhieu.matTH_0.3s", _drop(0.3), E(must=("có đoạn mất tín hiệu",), drop=0.3)),
          ("tinhieu.matTH_0.3s_mp3", _drop(0.3, ".mp3", "-b:a", "64k"), E("bw", must=("có đoạn mất tín hiệu",), kbps=64, drop=0.3, drop_tol=0.04))]
    C += [("tinhieu.tiengnoi_15pct", _pad_speech(0.15), E("spd")), ("tinhieu.tiengnoi_50pct", _pad_speech(0.50), E("spd"))]
    # --- NGUOI NOI (ghep cac tep goc cung ngon ngu, nguoi khac nhau) ---
    C += [("nguoi.2_nguoi", _concat_people(2), E(must=("nhiều người nói",))), ("nguoi.6_nguoi", _concat_people(6), E("04", "quá 4 người nói"))]
    # --- DUNG LUONG (11/10): ban sao tep goc; run() ha k1.MAX_BYTES = kich thuoc - 1 -> phai tu_choi truoc khi giai ma (khong ton 250 MB dia)
    C.append(("dungluong.vuot_nguong", lambda y, w, d, o: wr(d + ".wav", y), E("tu_choi", max_bytes_below=True)))
    # --- LOAI AM THANH / LUONG TIENG NOI (11/10): khong co loi noi -> 04 'qua it tieng noi' HOAC 'khong phai ghi am loi noi' (AST bat)
    NOSPEECH = ("quá ít tiếng nói", "không phải ghi âm lời nói")
    C.append(("loai.chi_tieng_on", _musan_noise, E("04", NOSPEECH, forbid=("cắt đỉnh nặng",))))
    C.append(("loai.am_thuan", lambda y, w, d, o: wr(d + ".wav", 0.3 * np.sin(2 * np.pi * 1000 * np.arange(10 * SR) / SR)),
              E("04", NOSPEECH, forbid=("cắt đỉnh nặng",))))
    return C + _edge().EDGE_CASES()


def _edge():
    """Ca bien / luoi + so do doc lap them (v6/coverage/k1_axes_edge.py; nap muon de tranh vong import)."""
    from v6.coverage import k1_axes_edge
    return k1_axes_edge


def _musan_noise(y, w, d, o):
    """10 s tieng on MUSAN (khong loi noi), chon tep theo do dai tep goc (on dinh), muc -26 dBFS."""
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(MUSAN) for f in fs if f.endswith(".wav"))
    if not files:
        raise FileNotFoundError("khong co MUSAN noise o " + MUSAN)
    z, sr = sf.read(files[len(y) % len(files)], dtype="float64", always_2d=True)
    z = z.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        z = resample_poly(z, SR, sr)
    z = tile(z, 10.0)
    wr(d + ".wav", z * 10 ** (-26 / 20) / (np.sqrt(np.mean(z ** 2)) + 1e-9), subtype="FLOAT")


CASE_MAP = None


def case_map() -> dict:
    global CASE_MAP
    if CASE_MAP is None:
        CASE_MAP = {n: (f, e) for n, f, e in CASES()}
    return CASE_MAP


# ------------------------------------------------------------------ do DOC LAP (khong dung vaudit.k1)
def _read_ch(path: str) -> tuple:
    """(x (n, kenh) float64, sr goc) - soundfile cho .wav / .flac; con lai (mp3 / m4a / ogg / opus / amr) ffmpeg -> mono 16 kHz
    (soundfile doc Opus theo 'tan so dau vao' trong header, vd 8 kHz -> do bang tan khong duoc; sua 12/10)."""
    try:
        if os.path.splitext(path)[1].lower() not in (".wav", ".flac"):
            raise ValueError("de ffmpeg giai ma")
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        return x, sr
    except Exception:          # noqa: BLE001  mp3 / m4a / opus / hong: giai ma bang ffmpeg
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f64le", "-ac", "1", "-ar", str(SR), "-"], capture_output=True, timeout=600)
        if r.returncode != 0 or not r.stdout:
            return None, None
        return np.frombuffer(r.stdout, np.float64)[:, None], SR


def _read_any(path: str) -> tuple:
    """(y mono float64, sr goc)."""
    x, sr = _read_ch(path)
    return (None, None) if x is None else (x.mean(1), sr)


def band_ratio_db(y: np.ndarray, sr: int) -> float | None:
    """Mat do pho trung binh 6-7 kHz so voi trung vi 1-3 kHz (Welch, Hann 2048) - so do doc lap cho dong 'Dai tan thuc >= 6 kHz'."""
    from scipy.signal import welch
    if y is None or len(y) < 4096 or sr < 14000:
        return None
    f, p = welch(y, fs=sr, nperseg=2048)
    ref = np.median(p[(f >= 1000) & (f <= 3000)])
    hi = np.mean(p[(f >= 6000) & (f <= 7000)])
    return float(10 * np.log10((hi + 1e-30) / (ref + 1e-30)))


def energy_db(y: np.ndarray, sr: int = SR, win: float = 0.03) -> np.ndarray:
    n = int(win * sr)
    k = len(y) // n
    return 10 * np.log10((np.asarray(y[:k * n], np.float64).reshape(k, n) ** 2).mean(1) + 1e-12) if k else np.array([-120.0])


def dense(y: np.ndarray, sec: float) -> np.ndarray:
    """Doan dai sec giay co nhieu khung du to nhat (de ca 3,2 s khong roi vao khoang lang dau tep)."""
    db = energy_db(y)
    act = (db > np.percentile(db, 10) + 10).astype(float)
    k = int(sec / 0.03)
    if len(act) <= k:
        return tile(y, sec)
    i = int(np.argmax(np.convolve(act, np.ones(k), "valid")))
    return y[int(i * 0.03 * SR):int(i * 0.03 * SR) + int(sec * SR)]


def oracle(path: str, base_floor: float | None = None, ex: dict | None = None) -> dict:
    """So do doc lap: bang tan (Welch) + thoi luong tieng noi theo nang luong voi SAN NHIEU CUA TEP GOC (tieng phong them vao khong tinh);
    them theo ca (ex): tuong quan hai kenh (st), muc tieng noi (lvl), toc do bit khai (kbps='probe')."""
    ex = ex or {}
    x, sr = _read_ch(path)
    y = None if x is None else x.mean(1)
    out = {"band_6_7k_db": None if y is None else band_ratio_db(y, sr), "dur_s": None if y is None else len(y) / sr}
    if y is not None and base_floor is not None and sr:
        db = energy_db(y, sr)
        out["speech_s"] = round(float(((db > base_floor + 10) & (db > -60)).sum() * 0.03), 2)
    if ex.get("dec") == "st" and x is not None and x.shape[1] == 2:
        out["rho0"], out["rho_lag"] = (round(v, 5) for v in _edge().stereo_rho(x, sr))
        if ex.get("ch_sp"):
            out["speech_ch_s"] = [round(_edge().speech_s_own(x[:, i], sr), 3) for i in range(2)]
    if ex.get("dec") == "lvl" and y is not None:
        lv = _edge().speech_level_db(y, sr)
        out["speech_dbfs"] = None if lv is None else round(lv, 2)
    if ex.get("kbps") == "probe":
        out["kbps_declared"] = _edge().declared_kbps(path)
    return out


def expect(ex: dict, orc: dict, base: dict, extended: bool = False) -> dict:
    """Dap an cuoi cho mot tep: {dec: tap quyet dinh chap nhan, reason, must: canh bao phai co, must_not: canh bao truc khong duoc co,
    n_ch: so kenh phan tich cho phep, drop: [t0, t1, sai so] moc doan mat tin hieu}. extended: K1 chay nhu SAN PHAM (k1x: ngon ngu + noi
    chong) -> them canh bao 'ext_must' cua ca (vd 'noi chong' cho hai nguoi noi cung luc)."""
    dec = ex["dec"]
    must = set(ex["must"]) | (set(ex.get("ext_must", ())) if extended else set())
    kb = orc.get("kbps_declared") if ex["kbps"] == "probe" else ex["kbps"]
    if kb is not None and kb < KBPS_GUARD[0]:
        must.add("nén mạnh")
    unsure = {"nén mạnh"} if (kb is not None and KBPS_GUARD[0] <= kb <= KBPS_GUARD[1]) or (ex["kbps"] == "probe" and kb is None) else set()
    unsure |= set(ex.get("allow", ()))
    reason = ex["reason"]
    n_ch = None if ex.get("n_ch") is None else [ex["n_ch"]]
    if dec == "bw":
        r = orc.get("band_6_7k_db")
        if r is None or abs(r - BW_THR_DB) <= BW_GUARD_DB:
            decs = {"accept", "04"}
        else:
            decs = {"04"} if r < BW_THR_DB else {"accept"}
        reason = "băng hẹp" if decs == {"04"} else None
    elif dec == "spd":                                   # >= 2 s va >= 30 %; vung dem do VAD nang luong khac mat na ASR cua K1
        sp, dur = orc.get("speech_s"), orc.get("dur_s") or 0
        if sp is None or not dur:
            decs = {"accept", "04"}
        else:
            share = sp / dur
            decs = {"04"} if (sp < 1.6 or share < 0.24) else {"accept"} if (sp > 2.6 and share > 0.36) else {"accept", "04"}
        reason = "quá ít tiếng nói" if decs == {"04"} else None
    elif dec == "st":                                    # Constraint 'Hai kenh' xet lan luot tren tuong quan DO LAI (vung dem 2 phia)
        r0, rl = orc.get("rho0"), orc.get("rho_lag")
        if r0 is None or rl is None:
            decs, n_ch = {"accept", "04"}, None
        elif r0 > SAME_RHO + RHO0_GUARD:
            decs, n_ch = {"accept"}, [1]
        elif abs(r0 - SAME_RHO) <= RHO0_GUARD:
            decs, n_ch = {"accept"}, [1, 2]
        elif rl >= SCENE_RHO + RHO_GUARD:
            decs, n_ch = {"accept"}, [2]
        elif rl < SCENE_RHO - RHO_GUARD:
            decs, n_ch = {"04"}, None
        else:
            decs, n_ch = {"accept", "04"}, [2]
        sp = orc.get("speech_ch_s") if ex.get("ch_sp") else None
        if sp:                                           # buoc (1) 'kenh co duoi 2 s tieng noi -> bo kenh do' xet TRUOC tuong quan
            lo, hi = min(sp), max(sp)
            if hi >= CH_SP_S + CH_SP_GUARD and lo < CH_SP_S - CH_SP_GUARD:
                decs, n_ch = {"accept"}, [1]
            elif abs(lo - CH_SP_S) <= CH_SP_GUARD:
                decs, n_ch = decs | {"accept"}, sorted(set(n_ch or [2]) | {1})
        reason = "hai kênh là hai nguồn riêng — gửi bản một kênh" if decs == {"04"} else None
    elif dec == "lvl":                                   # 'muc tieng noi trung binh >= -40 dBFS', muc do doc lap, vung dem 1 dB
        lv = orc.get("speech_dbfs")
        if lv is None or abs(lv - LVL_THR_DB) <= LVL_GUARD_DB:
            decs = {"accept", "04"}
        else:
            decs = {"04"} if lv < LVL_THR_DB else {"accept"}
        reason = "quá nhỏ" if decs == {"04"} else None
    else:
        decs = {dec}
    from vaudit.k1 import CONTRACT_WARNINGS
    inherited = set(base.get("warnings", []))
    must_not = {w for w in CONTRACT_WARNINGS if w not in must and w not in unsure and w not in inherited and w != EDGE_WARN}
    drop = None
    if ex.get("drop") and base.get("n"):
        t0 = int(base["n"] * DROP_FRAC) / SR
        drop = [round(t0, 3), round(t0 + ex["drop"], 3), ex.get("drop_tol", 0.02)]
    return {"dec": sorted(decs), "reason": reason, "must": sorted(must | {EDGE_WARN}), "must_not": sorted(must_not), "n_ch": n_ch,
            "forbid": list(ex.get("forbid", ())), "drop": drop}


def drop_runs(r: dict) -> list:
    """Cac doan mat tin hieu K1 bao (tu warn_detail 'a–b s, ...')."""
    det = " ".join(r.get("warn_detail", {}).get(DROP_WARN, []))
    return [(float(a), float(b)) for a, b in re.findall(r"(\d+(?:\.\d+)?)\s*[–-]\s*(\d+(?:\.\d+)?)", det)]


def _reasons(reason) -> tuple:
    """reason: None | cum tu | tuple cum tu (chap nhan MOT trong so do)."""
    return () if not reason else (reason,) if isinstance(reason, str) else tuple(reason)


def judge(r: dict, exp: dict) -> list:
    """Danh sach loi (rong = dung) cua ket qua K1 r so voi dap an exp."""
    errs = []
    got = "accept" if r["decision"] in ("nhan", "canh_bao") else r["decision"]
    if got not in exp["dec"]:
        errs.append(f"quyet dinh {r['decision']} (mong doi {'/'.join(exp['dec'])})")
    want = _reasons(exp["reason"])
    if got == "04" and want and not any(f"“{x}”" in s for x in want for s in r["reasons"]):
        errs.append(f"thieu ly do “{' / '.join(want)}” (co: {r['reasons'][:2]})")
    bad_r = [x for x in exp.get("forbid", []) if any(f"“{x}”" in s for s in r["reasons"])]
    if bad_r:
        errs.append(f"ly do sai {bad_r}")
    if got == "accept":
        miss = [w for w in exp["must"] if w not in r["warnings"]]
        extra = [w for w in exp["must_not"] if w in r["warnings"]]
        if miss:
            errs.append(f"thieu canh bao {miss}")
        if [w for w in extra if w in AXIS_WARN]:
            errs.append(f"canh bao thua {[w for w in extra if w in AXIS_WARN]}")
        if [w for w in extra if w not in AXIS_WARN]:
            errs.append(f"canh bao moi khong co o tep goc {[w for w in extra if w not in AXIS_WARN]}")
        allowed = exp.get("n_ch")
        allowed = [allowed] if isinstance(allowed, int) else allowed
        if allowed and len(r.get("channels", [])) not in allowed:
            errs.append(f"so kenh phan tich {len(r.get('channels', []))} (mong doi {'/'.join(map(str, allowed))})")
        if exp.get("drop") and DROP_WARN in r["warnings"]:
            t0, t1, tol = exp["drop"]
            runs = drop_runs(r)
            if not any(abs(a - t0) <= tol and abs(b - t1) <= tol for a, b in runs):
                errs.append(f"moc mat tin hieu {runs[:3]} (mong doi {t0:.2f}–{t1:.2f} ± {tol})")
        from vaudit.k1 import CONTRACT_WARNINGS
        bad = [w for w in r["warnings"] if w not in CONTRACT_WARNINGS]
        if bad:
            errs.append(f"canh bao ngoai hop dong {bad}")
    return errs


# ------------------------------------------------------------------ dung tep
def _bases_by_id(ids: list) -> list:
    """Muc manifest natvi / naten theo id (giu thu tu) - de 'build --only' dung lai DUNG cac tep goc cua cases.json cu."""
    want, got = set(ids), {}
    for lang in ("natvi", "naten"):
        for e in json.load(open(os.path.join(BASE, "data_labels", "v6", lang, "manifest.json"), encoding="utf-8")):
            if e["id"] in want:
                got[e["id"]] = e
    return [got[i] for i in ids if i in got]


def _prefixes(only) -> tuple | None:
    """'a,b' | ['a', 'b'] | None -> tuple tien to (str.startswith nhan tuple). '@bien' = moi ca bien / luoi (EDGE_CASES)."""
    if not only:
        return None
    items = only.split(",") if isinstance(only, str) else list(only)
    out = []
    for s in (s.strip() for s in items):
        out += [n for n, _, _ in _edge().EDGE_CASES()] if s == "@bien" else [s] if s else []
    return tuple(out) or None


def build(n_base: int = N_BASE, n_long: int = N_LONG, only: str | None = None) -> None:
    """only: chi dung cac ca co ten bat dau bang MOT trong cac tien to ('a,b') tren CUNG tep goc cua cases.json hien co, them / thay muc
    cua chung, giu phan con lai."""
    os.makedirs(OUT, exist_ok=True)
    only = _prefixes(only)
    old = json.load(open(os.path.join(OUT, "cases.json"), encoding="utf-8")) if only else None
    bases = _bases_by_id(old["bases"]) if only else pick(n_base)
    ys = {}
    for e in bases:
        y, sr = sf.read(os.path.join(V4, e["wav"]), dtype="float64")
        ys[e["id"]] = (y.mean(1) if y.ndim > 1 else y)[:int(CUT_S * SR)]
    meta, skipped = [], []
    t0 = time.time()
    for bi, e in enumerate(bases):
        y = ys[e["id"]]
        lang = e["meta"].get("lang") or ("vi" if e["id"].startswith("v6nat_") else "en")
        others = [ys[x["id"]] for x in bases if x["id"] != e["id"] and (x["meta"].get("lang") or "") == (e["meta"].get("lang") or "")]
        o = others[0] if others else y[::-1].copy()
        w = os.path.join(OUT, f"_goc__{e['id']}.wav")
        if not only:                                   # --only: tep goc da co, khong ghi lai
            wr(w, y)
            meta.append({"file": os.path.basename(w), "ca": "_goc", "base": e["id"], "lang": lang, "person": e["meta"].get("person")})
        for name, fn, ex in CASES():
            if (ex["long"] and bi >= n_long) or (only and not name.startswith(only)):
                continue
            stem = os.path.join(OUT, f"{name}__{e['id']}")
            try:
                if name.startswith("nguoi."):
                    fn(y, w, stem, o, others=others)
                else:
                    fn(y, w, stem, o)
            except Exception as ex_:                 # noqa: BLE001  ffmpeg thieu bo ma hoa -> ghi lai, bo ca
                skipped.append((name, e["id"], f"{type(ex_).__name__} {str(ex_)[:80]}"))
                continue
            f = next((p for p in os.listdir(OUT) if p.startswith(os.path.basename(stem) + ".") and "__tmp" not in p), None)
            if f:
                meta.append({"file": f, "ca": name, "base": e["id"], "lang": lang})
        print(f"  {bi + 1}/{len(bases)} tep goc ({time.time() - t0:.0f} s)", flush=True)
    doc = {"ngay": time.strftime("%Y-%m-%d %H:%M"), "bases": [e["id"] for e in bases], "files": meta, "bo_qua": skipped}
    if only:                                           # gop: giu muc cu cua cac ca khac, thay muc cua ca 'only'
        doc["files"] = [m for m in old["files"] if not m["ca"].startswith(only)] + meta
        doc["bo_qua"] = [b for b in old.get("bo_qua", []) if not str(b[0]).startswith(only)] + skipped
        doc["them"] = old.get("them", []) + [{"ngay": doc["ngay"], "only": ",".join(only), "n": len(meta)}]
    json.dump(doc, open(os.path.join(OUT, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(len(meta), "tep;", len(skipped), "ca bo qua", skipped[:5], flush=True)


# ------------------------------------------------------------------ chay
def run(limit: int | None = None, n_bases: int | None = None, only: str | None = None, events: bool | None = None, long: bool = False,
        base_ids: list | None = None, tag: str | None = None, extended: bool = False) -> dict:
    """extended=True: K1 nhu SAN PHAM (voiceaudit_v9 k1x=True -> pipeline extended_k1): them ngon ngu + noi chong (vaudit.k1x); cham ca
    'ext_must' va must_not 'ngon ngu khac' / 'noi chong' tren moi ca. Cham hon nhieu (~6-18 s / tep): chia bang --only / --base-ids + --tag."""
    from vaudit import k1
    cj = json.load(open(os.path.join(OUT, "cases.json"), encoding="utf-8"))
    skip = int(os.environ.get("K1AXES_SKIP_BASES", "0"))                 # bo qua K tep goc dau (chay tung phan)
    bases = cj["bases"][skip:skip + n_bases] if n_bases else cj["bases"][skip:]
    if base_ids:
        bases = [b for b in cj["bases"] if b in set(base_ids)]
    files = [m for m in cj["files"] if m["base"] in bases]
    cm = case_map()
    only = _prefixes(only)
    base_info = {}
    for m in files:                                      # tep goc truoc: canh bao thua huong + san nhieu (so do doc lap)
        if m["ca"] == "_goc":
            p = os.path.join(OUT, m["file"])
            r = k1.check(p, extended=extended, events=events)
            per = r["measures"].get("kenh", [{}])
            base_info[m["base"]] = {"decision": r["decision"], "warnings": r["warnings"], "reasons": r["reasons"],
                                    "tieng_noi_s": max((q.get("tieng_noi_s", 0) for q in per), default=0), "n": sf.info(p).frames,
                                    "floor_db": float(np.percentile(energy_db(sf.read(p, dtype="float64")[0]), 10)),
                                    "noi_chong_pct": [q.get("noi_chong_pct") for q in per], "ngon_ngu": [(q.get("ngon_ngu") or {}).get("top") for q in per]}
    todo = [m for m in files if m["ca"] != "_goc" and m["ca"] in cm and (only is None or m["ca"].startswith(only))
            and base_info.get(m["base"], {}).get("decision") in ("nhan", "canh_bao") and (long or not cm[m["ca"]][1]["long"])]
    if limit:
        todo = todo[:limit]
    rows = []
    t0 = time.time()
    for i, m in enumerate(todo):
        p = os.path.join(OUT, m["file"])
        orc = oracle(p, base_info[m["base"]]["floor_db"], cm[m["ca"]][1])
        exp = expect(cm[m["ca"]][1], orc, base_info[m["base"]], extended)
        t1 = time.time()
        saved = k1.MAX_BYTES
        try:
            if cm[m["ca"]][1].get("max_bytes_below"):
                k1.MAX_BYTES = os.path.getsize(p) - 1          # thay cho tep > 250 MB that (dong 'Dung luong')
            r = k1.check(p, extended=extended, events=events)
        finally:
            k1.MAX_BYTES = saved
        errs = judge(r, exp)
        M = r["measures"]
        rows.append({"file": m["file"], "ca": m["ca"], "base": m["base"], "lang": m["lang"], "ok": not errs, "loi": errs, "decision": r["decision"],
                     "extended": extended, "noi_chong_pct": [c.get("noi_chong_pct") for c in M.get("kenh", [])],
                     "ngon_ngu": [(c.get("ngon_ngu") or {}).get("top") for c in M.get("kenh", [])],
                     "reasons": r["reasons"], "warnings": r["warnings"], "warn_detail": r.get("warn_detail", {}), "exp": exp, "oracle": orc,
                     "s": round(time.time() - t1, 1), "bang_thong_k1": [c.get("bang_thong_hz") for c in M.get("kenh", [])],
                     "k1": {"toc_do_bit_kbps": M.get("toc_do_bit_kbps"), "tuong_quan_hai_kenh": M.get("tuong_quan_hai_kenh"),
                            "hai_kenh_giong_nhau": M.get("hai_kenh_giong_nhau"), "do_dai_s": M.get("do_dai_s"),
                            "am_luong_dbfs": [c.get("am_luong_dbfs") for c in M.get("kenh", [])],
                            "cat_dinh_pct": [c.get("cat_dinh_pct") for c in M.get("kenh", [])]}})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f} s)", flush=True)
    bad_base = {b: v for b, v in base_info.items() if v["decision"] not in ("nhan", "canh_bao")}
    report(rows, base_info, bad_base, tag)
    return {"rows": rows, "base": base_info}


def report(rows: list, base_info: dict, bad_base: dict, tag: str | None = None) -> None:
    by: dict = {}
    for r in rows:
        by.setdefault(r["ca"], []).append(r)
    L = ["# Test K1 theo Constraint (tep that, dap an tinh lai luc chay tu bang Constraint + so do doc lap)\n",
         f"Tep goc: {len(base_info)} ({sum(1 for b in base_info if b.startswith('v6nat_'))} vi, {sum(1 for b in base_info if b.startswith('v6naten_'))} en);"
         f" tep goc bi K1 chan (bo ca cua no): {len(bad_base)} {list(bad_base)[:5]}",
         f"Che do K1: {'--extended (nhu san pham: them ngon ngu + noi chong, vaudit.k1x)' if any(r.get('extended') for r in rows) else 'thuong (khong k1x)'}",
         f"Canh bao thua huong tu tep goc: {sorted({w for v in base_info.values() for w in v['warnings']})}\n",
         "| Ca | Dap an (quyet dinh / ly do / canh bao phai co) | Dung | Loi thuong gap |", "|---|---|---|---|"]
    for ca, v in by.items():
        ex = v[0]["exp"]
        errs = [e for r in v for e in r["loi"]]
        common = max(set(errs), key=errs.count) if errs else ""
        kf = " [KNOWN_FAIL]" if ca in KNOWN_FAIL else ""
        L.append(f"| {ca}{kf} | {'/'.join(ex['dec'])} · {' / '.join(_reasons(ex['reason'])) or '-'} · {', '.join(w for w in ex['must'] if w != EDGE_WARN) or '-'} | {sum(r['ok'] for r in v)}/{len(v)} | {common[:90]} |")
    tot, n = sum(r["ok"] for r in rows), len(rows)
    L.append(f"\n**Tong: {tot}/{n} tep dung** ({100 * tot / max(n, 1):.0f} %). Ca lech: " + ", ".join(sorted({r['ca'] for r in rows if not r['ok']})))
    L += ["\nHan che da biet (KNOWN_FAIL, tinh vao 'lech' o tren, khong an di): " + "; ".join(f"{k}: {v}" for k, v in KNOWN_FAIL.items()),
          "\n## Phu dong Constraint (dong nao kiem o dau)\n", "| Dong Constraint | Kiem o dau |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in ROW_COVERAGE]
    os.makedirs(RES, exist_ok=True)
    stem = os.path.join(RES, "k1_axes" + (f"_{tag}" if tag else ""))
    open(stem + ".md", "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(rows, open(stem + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=str)
    json.dump(base_info, open(stem + "_base.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=str)
    print("\n".join(L))


def merge(tags: list, out_tag: str | None = None) -> int:
    """Gop ket qua cac lan chay tung phan (run --base-ids ... --tag t) thanh mot bao cao (mac dinh k1_axes.md / .json). Tra so dong."""
    rows, base = [], {}
    for t in tags:
        stem = os.path.join(RES, f"k1_axes_{t}")
        part = json.load(open(stem + ".json", encoding="utf-8"))
        rows += part
        if os.path.exists(stem + "_base.json"):
            base.update(json.load(open(stem + "_base.json", encoding="utf-8")))
        else:                                            # lan chay truoc 12/10 chua ghi _base.json: chi biet id tep goc (ca cua no da chay -> da nhan)
            base.update({r["base"]: {"decision": "nhan", "warnings": [], "ghi_chu": "khong co so do tep goc"} for r in part if r["base"] not in base})
    seen, uniq = set(), []
    for r in reversed(rows):                             # cung tep o hai phan -> giu lan chay sau
        if r["file"] not in seen:
            seen.add(r["file"])
            uniq.append(r)
    uniq.reverse()
    report(uniq, base, {b: v for b, v in base.items() if v.get("decision") not in ("nhan", "canh_bao")}, out_tag)
    return len(uniq)


def _arg(a: list, k: str, cast=int, default=None):
    return cast(a[a.index(k) + 1]) if k in a else default


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[0] == "build":
        build(_arg(a, "--bases", int, N_BASE), _arg(a, "--long-bases", int, N_LONG), _arg(a, "--only", str))
    elif a[0] == "run":
        ids = _arg(a, "--base-ids", str)
        run(_arg(a, "--limit"), _arg(a, "--bases"), _arg(a, "--only", str), False if "--no-events" in a else None, "--long" in a,
            ids.split(",") if ids else None, _arg(a, "--tag", str), "--extended" in a)
    elif a[0] == "lang":                                 # ca duong ngon ngu khac (extended, NANG vua: ~7 x 15 s)
        _edge().lang_real()
    elif a[0] == "merge":
        print(merge(_arg(a, "--tags", str).split(","), _arg(a, "--tag", str)), "dong")
    else:
        from v6.coverage import test_k1_axes
        test_k1_axes.run_tests()
