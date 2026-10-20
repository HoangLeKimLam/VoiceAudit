# -*- coding: utf-8 -*-
"""K1 · Kiem dieu kien dau vao — cai theo bang Constraint cua VoiceAudit_6Muc (ban trich: .tmp_v6/wf/6muc.txt).

check(path, extended=False, events=None) -> dict:
  decision    : "nhan" | "canh_bao" | "04" | "tu_choi"
  reasons     : ly do 04 / tu choi, dang  04 · “<cum tu hop dong>” — <chi tiet>   (REASON: dung chu cot 'Khong dat thi')
  warnings    : CHI cac cum tu trong 12 canh bao cua hop dong, NGUYEN VAN (CONTRACT_WARNINGS); chi tiet (moc, so do) -> warn_detail
  warn_detail : {canh bao: [chi tiet ...]}  (vd moc mat tin hieu, so giong uoc luong) - ho so (ticket) dua vao 'chi_tiet'
  measures    : so do cua tung dieu kien (kenh: so do tung kenh); ghi_chu_k1: chu khong thuoc hop dong (khong dua vao canh bao)
  channels    : song mono 16 kHz can phan tich (1 kenh, hoac 2 kenh neu hai micro cung canh)
  not_impl    : dong Constraint LAN CHAY NAY khong tu kiem (tinh theo co bat; nhom 'khong tu kiem duoc' luon co) + ghi chu theo tep
                (NOTE_LOSSY_CLIP: tep nen, cat dinh chi uoc luong; NOTE_GATED_SPK: tep cong on, dem nguoi khong tin cay)
Phan cong dong Constraint: SNR, nen che tieng noi, phong vang do vaudit.cond (mo hinh hoc tren vector XLS-R) trong v6/product/pipeline.py
(can vector XLS-R, nap sau K1). SNR mu cua K1 chi con la so do (snr_db): tren tep tu nhien hop le trung vi 5,8 dB -> khong dung de canh bao.
Loai am thanh / kieu noi: vaudit/events.py (AST AudioSet, dong bang) - bat mac dinh khi co mo hinh cuc bo (events=None).
Ngon ngu / noi chong: vaudit/k1x.py khi extended=True.
  python -m vaudit.k1 <tep ...> [--extended] [--no-events]     in ket qua
  python -m vaudit.k1 test                                     test (= python -m vaudit.test_k1; tep tong hop, ASR / resemblyzer thay bang ban
                                                               nhe, khong nap mo hinh)"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

SR = 16000
MAX_BYTES = 250 * 10 ** 6            # 'Dung luong <= 250 MB': MB theo SI (10^6 byte) - ban cu dung MiB (262 144 000 byte), xem ghi chu WP2
FORMATS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".amr"}      # Constraint: 8 duoi
CODECS = {"pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_s32le", "pcm_s32be", "pcm_f32le", "pcm_f32be", "pcm_f64le", "pcm_f64be",
          "flac", "alac", "mp3", "aac", "opus", "vorbis", "amr_wb", "amr_nb"}      # amr_nb lay mau 8 kHz -> bi chan boi 'tan so qua thap'
PCM_LOW_BITS = {"pcm_u8", "pcm_s8", "pcm_alaw", "pcm_mulaw"}
LOSSLESS = {"flac", "alac"}
COMPRESSED = {"mp3", "aac", "opus", "vorbis", "amr_wb", "amr_nb"}
LOW_KBPS = 24000
TIMEOUT_S = 180
# 12 canh bao cua hop dong (Output phu), dung chu va thu tu
WARN = {"nen_manh": "nén mạnh", "on_cao": "tiếng ồn cao", "cat_dinh": "cắt đỉnh", "mat_tin_hieu": "có đoạn mất tín hiệu", "phong_vang": "phòng vang",
        "kieu_noi": "kiểu nói ngoài phạm vi", "ngon_ngu": "ngôn ngữ khác", "nhieu_nguoi": "nhiều người nói", "noi_chong": "nói chồng",
        "nghi_03": "có đoạn nghi 03 chưa đủ chắc", "con_cho_sua": "có thể còn chỗ sửa khác", "dau_cuoi": "0,3 s đầu và cuối không được soi"}
CONTRACT_WARNINGS = tuple(WARN.values())
# cot 'Khong dat thi' cua bang Constraint (nhan 04)
REASON = {"khong_doc": "không đọc được tệp", "do_sau": "độ sâu bit dưới 16", "tan_so": "tần số quá thấp", "do_dai": "độ dài ngoài 3 giây – 10 phút",
          "qua_2_kenh": "quá 2 kênh", "hai_nguon": "hai kênh là hai nguồn riêng — gửi bản một kênh", "bang_hep": "băng hẹp", "qua_nho": "quá nhỏ",
          "cat_dinh_nang": "cắt đỉnh nặng", "on_qua": "ồn quá", "nen_che": "nền che tiếng nói", "it_tieng_noi": "quá ít tiếng nói",
          "khong_phai_loi_noi": "không phải ghi âm lời nói", "qua_4_nguoi": "quá 4 người nói"}
REJECT = {"so_tep": "từ chối nhận · đúng 1 tệp mỗi lần", "dung_luong": "từ chối nhận · dung lượng > 250 MB"}
# chu cu (k1x, ban K1 truoc 10/10) -> canh bao hop dong; tien to khong khop -> ghi_chu_k1
LEGACY = (("nén mạnh", "nen_manh"), ("tiếng ồn", "on_cao"), ("cắt đỉnh", "cat_dinh"), ("có đoạn mất tín hiệu", "mat_tin_hieu"),
          ("phòng vang", "phong_vang"), ("kiểu nói", "kieu_noi"), ("ngôn ngữ khác", "ngon_ngu"), ("nhiều hơn 1 giọng", "nhieu_nguoi"),
          ("nhiều người nói", "nhieu_nguoi"), ("nói chồng", "noi_chong"), ("có đoạn nghi 03", "nghi_03"), ("có thể còn chỗ sửa", "con_cho_sua"),
          ("0,3 s đầu", "dau_cuoi"))
ALWAYS_UNCHECKED = ["Cách ghi: một lần ghi liên tục, một thiết bị, micro ≤ 2 m (không tự kiểm được)",
                    "Tốc độ phát lệch ≤ ± 5 % (không tự kiểm được)"]
NOTE_LOSSY_CLIP = ("Cắt đỉnh trên tệp nén: chỉ ước lượng theo đuôi biên độ (sai số khoảng ± 1 điểm %; Opus cắt nặng có thể đo thiếu) — "
                   "không đo trực tiếp được số mẫu chạm mức tối đa")
NOTE_GATED_SPK = "Số người nói: tệp có cổng ồn / nhiều đoạn số 0 — phần lớn cửa sổ bị bỏ, không đếm được tin cậy"
COND_ROWS = ["Tiếng nói so với tiếng ồn ≥ 10 dB — kiểm bằng vaudit.cond (XLS-R) trong pipeline, không ở K1",
             "Âm không phải tiếng nói ≤ 10 % năng lượng — kiểm bằng vaudit.cond (XLS-R) trong pipeline, không ở K1",
             "Phòng RT60 ≤ 1,0 s — kiểm bằng vaudit.cond (XLS-R) trong pipeline, không ở K1"]


def reason(key: str, detail: str = "") -> str:
    """Ly do 04 theo dung cum tu hop dong: 04 · “<cum tu>” — <chi tiet>."""
    return f"04 · “{REASON[key]}”" + (f" — {detail}" if detail else "")


def normalize_warnings(ws: list) -> tuple:
    """Moi canh bao -> (canh bao hop dong nguyen van, chi tiet, chu ngoai hop dong). Tra (warnings, warn_detail, ghi_chu)."""
    out, detail, notes = [], {}, []
    for w in ws:
        if w in CONTRACT_WARNINGS:
            key = w
        else:
            key = next((WARN[k] for pre, k in LEGACY if w.startswith(pre)), None)
        if key is None:
            notes.append(w)
            continue
        if key not in out:
            out.append(key)
        if w != key:
            rest = w[len(key):].strip(" —–-:()") if w.startswith(key) else w
            if rest:
                detail.setdefault(key, []).append(rest)
    order = {w: i for i, w in enumerate(CONTRACT_WARNINGS)}
    return sorted(out, key=order.get), detail, notes


def _run(cmd: list, text: bool) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=text, timeout=TIMEOUT_S, **({"encoding": "utf-8", "errors": "replace"} if text else {}))
    except (subprocess.TimeoutExpired, OSError):
        return None


def probe(path):
    r = _run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path], True)
    if r is None or r.returncode != 0:
        return None
    try:
        info = json.loads(r.stdout or "{}")
    except json.JSONDecodeError:
        return None
    aud = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    return (aud[0], info.get("format", {})) if aud else None


def decode(path, channels, sr=SR):
    """Giai ma ve float32 (mac dinh 16 kHz; sr=None giu tan so goc), giu so kenh. Tra (n x kenh) hoac None."""
    r = _run(["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-f", "f32le", "-acodec", "pcm_f32le", *(["-ar", str(sr)] if sr else []),
              "-ac", str(channels), "-"], False)
    if r is None or r.returncode != 0 or not r.stdout:
        return None
    a = np.frombuffer(r.stdout, np.float32)
    return a[:len(a) // channels * channels].reshape(-1, channels)


def frame_db(y, win=0.03):
    n = int(win * SR)
    k = len(y) // n
    if k == 0:
        return np.array([-120.0])
    f = y[:k * n].reshape(k, n)
    return 10 * np.log10((f.astype(np.float64) ** 2).mean(1) + 1e-12)


ZERO_DB = -100.0                     # khung 30 ms duoi muc nay = so 0 ky thuat so (frame_db cua toan so 0 = -120 dB)
BW_MIN_HZ = 6000                     # 'Dai tan thuc: co nang luong toi >= 6 kHz'
BW_DYN_DB, BW_DYN_Q = 6.0, (99, 20)  # dai 250 Hz phai co do dong khung (p99 - p20) >= 6 dB: nang luong THEO tieng noi, khong phai san on deu
SPK_ZERO_SHARE = 0.02                # cua so dem nguoi co >= 2 % mau |y| < 3e-5 (doan so 0 / mat tin hieu) -> bo
SPK_UNRELIABLE = 0.5                 # > 50 % cua so du tieng noi bi bo vi so 0 (cong on / mat tin hieu) -> so nguoi noi khong tin cay
LOSSY_CLIP_FRAC, LOSSY_CLIP_Q, LOSSY_CLIP_BIAS = 0.9, 99.8, 0.004   # cat dinh tep nen: P(|y| >= 0,9 * p99,8) - 0,4 % (xem clip_share_lossy)


def speech_mask(db):
    """VAD nang luong: khung cao hon san nhieu (phan vi 10) it nhat 10 dB va > -60 dBFS. San nhieu tinh tren khung KHONG phai so 0 ky thuat so
    (sua 12/10: doan mat tin hieu 1 s trong tep 10 s keo phan vi 10 xuong -120 dB -> moi khung thanh 'tieng noi', doi so cua so dem nguoi);
    tep (gan nhu) toan so 0: giu cach cu."""
    ok = db > ZERO_DB
    floor = np.percentile(db[ok], 10) if ok.mean() >= 0.1 else np.percentile(db, 10)
    return (db > floor + 10) & (db > -60), floor


def narrowband(bw: float) -> bool:
    """True -> 04 'bang hep'. bandwidth_hz tra BIEN TREN cua dai 250 Hz cao nhat con nang luong: bw = 6000 nghia la nang luong chi toi dai
    [5750, 6000) -> CHUA toi 6 kHz (sua 12/10: ban cu 'bw < 6000' cho qua loc 5,5 kHz; tren du lieu luu chi 10 / 24 753 kenh co bw = 6000)."""
    return bool(np.isfinite(bw) and bw <= BW_MIN_HZ)


def bandwidth_hz(y):
    """Tan so cao nhat con nang luong: pho trung binh dai han theo dai 250 Hz, dai cao nhat con tren (muc 1-3 kHz) - 55 dB VA co do dong
    khung p99 - p20 >= 6 dB (nang luong thay doi theo tieng noi); tra BIEN TREN cua dai do (xem narrowband). Hieu chinh: tep that 6-7 kHz thap
    hon muc 1-3 kHz 19-33 dB; tep bang hep (8 kHz nang len 16 kHz) thap hon 78 dB. Do dong (sua 12/10 r5, .tmp_v6/results/wp2_diag/clipbw_bw2.txt):
    G.711 / AMR-NB nang len 16 kHz + on trang -80..-30 dBFS: ban cu 8000 Hz (qua), nay 20/20 tep goc -> 04 'bang hep'; 200 tep tu nhien hop le:
    0 tep lat sang bang hep; han che: loi noi bang rong + on trang PHANG rat to (-40 dBFS, SNR 6-18 dB) -> 4/20 bi goi bang hep."""
    n = 1024
    k = len(y) // n
    if k < 4:
        return float("nan")
    S = np.abs(np.fft.rfft(y[:k * n].reshape(k, n) * np.hanning(n), axis=1)) ** 2
    ltas = 10 * np.log10(S.mean(0) + 1e-20)
    f = np.fft.rfftfreq(n, 1 / SR)
    ref = np.median(ltas[(f >= 1000) & (f <= 3000)])
    edges = np.arange(0, SR / 2 + 1, 250)
    top = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (f >= a) & (f < b)
        if not sel.any() or np.median(ltas[sel]) <= ref - 55:
            continue
        e = 10 * np.log10(S[:, sel].mean(1) + 1e-20)
        if np.percentile(e, BW_DYN_Q[0]) - np.percentile(e, BW_DYN_Q[1]) >= BW_DYN_DB:
            top = b
    return float(top)


def asr_speech_mask(y, win=0.03):
    """Mat na tieng noi theo mo hinh ASR tieng Viet (CTC): khung co ky tu khac 'trong' ± 0,1 s.
    Tieng on deu (u may, quat) co tinh tuan hoan nhung ASR khong nhan ra chu -> khong bi tinh la tieng noi."""
    import torch

    from .asr import _load
    from .features import LONG_S, chunk_frames
    proc, model = _load()
    blank = proc.tokenizer.pad_token_id

    def run(seg):
        x = proc(seg.astype(np.float32), sampling_rate=SR, return_tensors="pt").input_values
        with torch.no_grad():
            return model(x).logits[0].numpy()
    if len(y) / SR > LONG_S:
        lg, dt = chunk_frames(y, run)
    else:
        lg = run(y)
        dt = len(y) / SR / lg.shape[0]
    tok = lg.argmax(1) != blank
    t_tok = (np.where(tok)[0] + 0.5) * dt
    n = len(y) // int(win * SR)
    tc = (np.arange(n) + 0.5) * win
    if not len(t_tok):
        return np.zeros(n, bool)
    j = np.searchsorted(t_tok, tc)
    d = np.minimum(np.abs(tc - t_tok[np.clip(j, 0, len(t_tok) - 1)]), np.abs(tc - t_tok[np.clip(j - 1, 0, len(t_tok) - 1)]))
    return d <= 0.10


def count_speakers(y, mask, win=0.03):
    """Dem nguoi noi: nhung resemblyzer tren cua so 1,6 s co tieng noi, gom cum trung binh, cat tai khoang cach cosine 0,35,
    chi dem cum chiem >= 10 % so cua so. Hieu chinh tren 72 tep biet so nguoi (42 tep dai/hoi thoai + 30 tep v4):
    dung 86 %, tep 1 nguoi dem = 1 o 94 % (nguong cu 0,30 / 5 %: dung 58 %).
    Bo cua so co >= 2 % mau gan 0 (doan mat tin hieu: nhung lech -> cum gia). Kiem 12/10 (.tmp_v6/results/wp2_diag/spk_diag.py, mat na nang luong + ASR
    nhu K1 that): k1axes v2 20 tep goc: tep goc 1 nguoi dem 1 = 20/20; 9 bien the cung noi dung (ma hoa, mat tin hieu, cat dinh, doi muc) dem
    khac tep goc 4/180 (truoc sua san nhieu + bo cua so so 0: 11/180); ghep 2 nguoi -> 2-4: 70 %; ghep 6 nguoi -> > 4: 55 % (11/20);
    32 cua so AMI 64 s (<= 4 nguoi) -> > 4: 0/32 (11/10 voi mat na chi nang luong: 1/32). Luat 5 % cum / t = 0,32 / 0,38 / cum nho phai cach xa:
    xem spk_diag eval - khong luat nao tot hon ca ba mat; giu t = 0,35, 10 %.
    Han che da biet: 6 nguoi noi noi tiep (k1_axes nguoi.6_nguoi) chi ra > 4 o ~55 % -> con lai khong ra 04 'qua 4 nguoi noi'.
    Tep CONG ON (ung dung tat tieng giua cac tu, benign 'cong_on'): gan nhu moi cua so co so 0 -> bi bo, dem ep = 1 (review 12/10: 44/44 cua so bo,
    ghep 2 nguoi + cong on dem dung 0/12; tat luat bo cua so cung chi 1/12) -> count_speakers.last ghi so cua so bo; analyse_channel danh dau
    'so_nguoi_noi_tin_cay' = False va check() them dong not_impl."""
    try:
        from resemblyzer import VoiceEncoder
        from scipy.cluster.hierarchy import fcluster, linkage
    except Exception:
        return None
    enc = count_speakers.enc = getattr(count_speakers, "enc", None) or VoiceEncoder("cpu", verbose=False)
    W, H, fn = int(1.6 * SR), int(0.8 * SR), int(win * SR)
    embs, n_zero = [], 0
    for s in range(0, max(1, len(y) - W), H):
        fr = mask[s // fn:(s + W) // fn]
        if len(fr) and fr.mean() >= 0.6:
            if np.mean(np.abs(y[s:s + W]) < 3e-5) < SPK_ZERO_SHARE:
                embs.append(enc.embed_utterance(y[s:s + W].astype(np.float32)))
            else:
                n_zero += 1
    count_speakers.last = {"cua_so": len(embs) + n_zero, "bo_vi_so_0": n_zero}      # analyse_channel doc (tep cong on: dem khong tin cay)
    if len(embs) < 2:
        return 1
    E = np.array(embs)
    Z = linkage(E, method="average", metric="cosine")
    lab = fcluster(Z, t=0.35, criterion="distance")
    big = [c for c in set(lab) if (lab == c).sum() >= max(2, 0.10 * len(lab))]
    return max(1, len(big))


def _plateau(mask) -> float:
    """Ti le mau thuoc chuoi >= 2 mau lien tiep sat dinh: cat dinh that tao 'binh nguyen'; dinh don le (am thuan 1 kHz = 2/16 mau o dinh,
    xung nhon) khong phai cat dinh (sua 11/10: am thuan 1 kHz bi 04 'cat dinh nang' 12,5 %)."""
    m = np.asarray(mask, bool)
    if len(m) < 2:
        return 0.0
    nb = np.zeros(len(m), bool)
    nb[1:] |= m[:-1]
    nb[:-1] |= m[1:]
    return float(np.mean(m & nb))


def clip_share(y) -> float:
    """Ti le mau 'cham muc toi da' theo CHINH tep: sau khi chuan hoa lai bien do, mau bi cat van nam sat dinh (khong nhat thiet = 0,999).
    Xet RIENG cuc duong / cuc am (cong lai): sau khi bo lech DC (check) hoac khi cat khong doi xung, hai muc cat lech nhau -> do theo
    max|y| chi thay mot phia (sua 11/10: ca cat 8 % do ra ~4 % -> chi canh bao thay vi 04 'cat dinh nang')."""
    if not len(y):
        return 0.0
    hi, lo = float(np.max(y)), float(np.min(y))
    s = _plateau(y >= 0.998 * hi) if hi > 0.05 else 0.0
    s += _plateau(y <= 0.998 * lo) if lo < -0.05 else 0.0
    return s


def clip_share_lossy(y) -> float:
    """Cat dinh trong tep NEN (MP3 / AAC / Opus ...): giai ma lam vo 'binh nguyen' (clip_share = 0 tren 9/9 tep cat 8 %, review 12/10) nhung
    mau bi cat van don sat muc cat -> do 'duoi day': moi cuc, ti le mau >= 0,9 * phan vi 99,8 cua cuc do, cong lai, tru 0,4 % (trung vi tren tep
    khong cat). Hieu chinh (.tmp_v6/results/wp2_diag/clipbw_clip2.txt, 20 tep goc x MP3 64k / AAC 128k / Opus 48k): cat 0,5 % -> <= 0,8;
    1,4 % -> 1,3-1,9; 3 % -> 2,8-4,1; 5,6 % -> 4,3-7,0; 8 % -> 4,7-9,6 (Opus vot dinh lon do thieu); 300 tep tu nhien hop le: trung vi 0,
    > 1 %: ~1 % tep. Opus 24k cat nang: do thieu nhieu (han che, ghi trong not_impl)."""
    s = 0.0
    for v in (y[y > 0], -y[y < 0]):
        if len(v) >= 100 and float(v.max()) > 0.05:
            s += float(np.count_nonzero(v >= LOSSY_CLIP_FRAC * np.percentile(v, LOSSY_CLIP_Q))) / len(y)
    return max(0.0, s - LOSSY_CLIP_BIAS)


def dropouts(y, sp, lossy: bool, win=0.03) -> list:
    """Doan toan so 0 dai > 0,1 s GIUA loi noi (co tieng noi truoc va sau), cach dau / cuoi > 0,3 s. Tep nen: giai ma so 0 ra nhieu rat nho
    -> nguong |y| < 3e-5 (-90 dBFS) thay vi 1e-7."""
    z = np.abs(y) < (3e-5 if lossy else 1e-7)
    if not z.any():
        return []
    d = np.diff(np.r_[0, z.astype(np.int8), 0])
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    fn = int(win * SR)
    runs = []
    for a, b in zip(starts, ends):
        if b - a > 0.1 * SR and 0.3 * SR < a and b < len(y) - 0.3 * SR and sp[:a // fn].any() and sp[b // fn + 1:].any():
            runs.append((round(a / SR, 2), round(b / SR, 2)))
    return runs


def analyse_channel(y, dur, extended=False, events=False, clip=None, lossy=False):
    """So do + canh bao + ly do 04 cua MOT kenh mono 16 kHz. Dong 'qua it tieng noi' xet o check() (tep hai kenh: kenh nhieu tieng noi hon).
    extended=True: them ngon ngu + noi chong (vaudit/k1x.py); events=True: loai am thanh + kieu noi (vaudit/events.py). clip: ti le cat dinh
    do o tan so goc (tep khong nen); None -> do tren song 16 kHz."""
    m, w, r = {}, [], []
    db = frame_db(y)
    sp_e, floor = speech_mask(db)
    sp_a = asr_speech_mask(y)
    k = min(len(sp_e), len(sp_a))
    sp = np.zeros(len(db), bool)
    sp[:k] = sp_e[:k] & sp_a[:k]           # tieng noi = du to VA mo hinh ASR nhan ra chu
    bw = bandwidth_hz(y)
    m["bang_thong_hz"] = None if not np.isfinite(bw) else round(bw)
    if narrowband(bw):
        r.append(reason("bang_hep", "dải tần thực < 6 kHz"))
    # muc tieng noi: khung du to VA co chu; khong co -> khung ASR nghe ra chu (tieng rat nho < -60 dBFS); khong co chu nao -> khong do
    # (de dong 'qua it tieng noi' tra loi: im lang / am thuan khong phai 'qua nho')
    sel = sp if sp.any() else np.r_[sp_a[:k], np.zeros(len(db) - k, bool)]
    lvl = float(10 * np.log10(np.mean(10 ** (db[sel] / 10)))) if sel.any() else None
    m["am_luong_dbfs"] = None if lvl is None else round(lvl, 1)
    if lvl is not None and lvl < -40:
        r.append(reason("qua_nho", "âm lượng tiếng nói < −40 dBFS"))
    c = clip_share(y) if clip is None else clip
    m["cat_dinh_do_o"] = "16 kHz" if clip is None else "tần số gốc"
    if lossy:                                # tep nen: binh nguyen bi vo -> uoc luong theo duoi bien do (sai so ~ +-1 diem %)
        c = max(c, clip_share_lossy(y))
        m["cat_dinh_do_o"] = "ước lượng đuôi biên độ (tệp nén)"
    m["cat_dinh_pct"] = round(100 * c, 2)
    if c > 0.05:
        r.append(reason("cat_dinh_nang", f"{100 * c:.1f} % số mẫu chạm mức tối đa"))
    elif c > 0.01:
        w.append(WARN["cat_dinh"] + f" — {100 * c:.1f} % số mẫu")
    runs = dropouts(y, sp, lossy)
    m["mat_tin_hieu"] = runs
    if runs:
        w.append(WARN["mat_tin_hieu"] + " — " + ", ".join(f"{a}–{b} s" for a, b in runs[:5]))
    # SNR mu theo ASR: chi la so do. Tren tep tu nhien hop le trung vi 5,8 dB (135/150 ho so e2e mang canh bao on) -> khong canh bao.
    pw = 10 ** (db / 10)
    nz = np.zeros(len(db), bool)
    nz[:k] = ~sp_a[:k]
    if sp.any() and nz.sum() * 0.03 >= 1.0:
        pn = pw[nz].mean()
        m["snr_db"] = round(float(10 * np.log10(max(pw[sp].mean() - pn, 1e-12) / (pn + 1e-12))), 1)
    else:
        m["snr_db"] = None
    m["snr_ghi_chu"] = "ước lượng mù theo ASR, chỉ tham khảo; dòng SNR kiểm bằng vaudit.cond trong pipeline"
    sp_s = float(sp_a.sum() * 0.03)         # thoi luong tieng noi: theo ASR (mat na nang luong bo sot tieng nho)
    m["tieng_noi_s"] = round(sp_s, 1)
    count_speakers.last = None
    n_spk = count_speakers(y, sp) if not r else None
    m["so_nguoi_noi"] = n_spk
    st = getattr(count_speakers, "last", None)          # ban nhe (test) khong co .last
    if n_spk is not None and st and st["cua_so"]:
        m["so_nguoi_noi_cua_so"] = dict(st)
        m["so_nguoi_noi_tin_cay"] = st["bo_vi_so_0"] <= SPK_UNRELIABLE * st["cua_so"]
    if n_spk is not None and n_spk > 4:
        r.append(reason("qua_4_nguoi", f"ước lượng {n_spk} người"))
    elif n_spk is not None and n_spk > 1:
        w.append(WARN["nhieu_nguoi"] + f" — ước lượng {n_spk} giọng")
    if events and not r:
        from . import events as ev
        try:
            me, we, re_ = ev.assess(y)
        except Exception as ex:             # noqa: BLE001  mo hinh AST hong / thieu: K1 van chay, dong nay ghi la chua kiem
            me, we, re_ = {"su_kien_am_loi": f"{type(ex).__name__}: {str(ex)[:120]}"}, [], []
        m.update(me)
        w.extend(we)
        r.extend(re_)
    if extended and not r:
        from . import k1x
        mx, wx, rx = k1x.assess(y)
        m.update(mx)
        w.extend(wx)
        r.extend(rx)
    return m, w, r


def _native_clip(path: str, ch: int, sr: int, max_bytes: int = 300 * 2 ** 20) -> list | None:
    """Ti le cat dinh tung kenh do o tan so GOC (tep khong nen, sr != 16 kHz): ha tan so lam nhoe dinh bi cat (Gibbs) -> do thieu."""
    if sr == SR:
        return None
    x = decode(path, ch, sr=None)
    if x is None or x.nbytes > max_bytes:
        return None
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return [clip_share(x[:, i]) for i in range(ch)]


def _empty() -> dict:
    return {"decision": "nhan", "reasons": [], "warnings": [], "warn_detail": {}, "measures": {}, "channels": [], "not_impl": []}


def check(path, extended=False, events=None):
    """K1 cho MOT tep. events=None: bat dong loai am thanh / kieu noi neu co mo hinh AST cuc bo (vaudit.events.available())."""
    out = _empty()
    R, W, M = out["reasons"], [], out["measures"]
    if events is None:
        from . import events as ev
        # ASR da bi thay bang ban nhe (v6/io/stub.light_k1, test khong nap mo hinh) -> khong tu bat AST
        events = ev.available() and getattr(asr_speech_mask, "__name__", "") == "asr_speech_mask"
    out["not_impl"] = not_impl(extended, events)
    if not isinstance(path, (str, os.PathLike)) or not os.path.isfile(path):
        out.update(decision="tu_choi"); R.append(REJECT["so_tep"])
        return out
    size = os.path.getsize(path)
    M["dung_luong_byte"] = size
    if size > MAX_BYTES:
        out.update(decision="tu_choi"); R.append(REJECT["dung_luong"])
        return out
    ext = os.path.splitext(path)[1].lower()
    pr = probe(path) if size > 0 else None
    if ext not in FORMATS or pr is None:
        out.update(decision="04"); R.append(reason("khong_doc", "định dạng ngoài .wav .flac .mp3 .m4a .aac .ogg .opus .amr" if ext not in FORMATS
                                                     else "không có luồng âm thanh giải mã được"))
        return out
    st, fm = pr
    codec, ch, sr = st.get("codec_name", ""), int(st.get("channels", 0) or 0), int(st.get("sample_rate", 0) or 0)
    try:
        dur = float(fm.get("duration") or st.get("duration") or 0)
    except (TypeError, ValueError):
        dur = 0.0
    br = int(st.get("bit_rate") or 0) or int(fm.get("bit_rate") or 0)
    if not br and codec in COMPRESSED and dur > 0:
        br = int(size * 8 / dur)
        M["toc_do_bit_uoc_tu_dung_luong"] = True
    bits = int(st.get("bits_per_raw_sample") or 0) or int(st.get("bits_per_sample") or 0)
    M.update(dinh_dang=ext, ma_hoa=codec, so_kenh=ch, tan_so_lay_mau=sr, do_dai_s=round(dur, 2), toc_do_bit_kbps=br // 1000, so_bit=bits or None)
    lossless_pcm = codec.startswith("pcm_") or codec in LOSSLESS
    if codec in PCM_LOW_BITS or (lossless_pcm and 0 < bits < 16):
        R.append(reason("do_sau", f"{bits or 8} bit ({codec})"))
    elif codec not in CODECS:
        R.append(reason("khong_doc", f"mã hoá {codec or 'không rõ'} không hỗ trợ"))
    if codec in COMPRESSED and 0 < br < LOW_KBPS:
        W.append(WARN["nen_manh"] + f" — khai {br / 1000:.1f} kbps < 24 kbps")
    if 0 < sr < SR:
        R.append(reason("tan_so", f"lấy mẫu {sr} Hz < 16 kHz"))
    if ch < 1 or sr <= 0:
        R.append(reason("khong_doc", "thông tin luồng âm thanh không hợp lệ"))
    elif ch > 2:
        R.append(reason("qua_2_kenh", f"{ch} kênh"))
    if dur and (dur < 3 or dur > 600):
        R.append(reason("do_dai", f"{dur:.2f} s"))
    if R:
        return _finish(out, W, "04")
    x = decode(path, ch)
    if x is None or len(x) == 0:
        R.append(reason("khong_doc", "giải mã lỗi"))
        return _finish(out, W, "04")
    bad = ~np.isfinite(x)
    if bad.any():
        M["mau_khong_hop_le_pct"] = round(100 * float(bad.mean()), 3)
        if bad.mean() > 0.01:
            R.append(reason("khong_doc", "giải mã ra giá trị không hợp lệ (NaN / vô cực)"))
            return _finish(out, W, "04")
        x = np.where(bad, 0.0, x).astype(np.float32)
    x = np.clip(x, -1e4, 1e4)
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk > 1.0:                                            # WAV float vuot [-1, 1]: chuan ve dinh 0,999 truoc khi do (resemblyzer / VAD
        M["bien_do_dinh_goc"] = round(pk, 3)                # nhay bien do tuyet doi: x1000 -> 'nhieu nguoi noi' sai); cat dinh do tuong doi
        x = (x * (0.999 / pk)).astype(np.float32)
    dc = x.mean(0)                                          # lech DC (khong thuoc Constraint) lam hong VAD nang luong -> bo truoc khi do
    M["lech_dc"] = [round(float(v), 4) for v in dc]
    if np.any(np.abs(dc) > 1e-3):
        x = (x - dc).astype(np.float32)
    dur_dec = len(x) / SR
    M["do_dai_giai_ma_s"] = round(dur_dec, 2)
    if not dur or abs(dur_dec - dur) > 0.5:                 # thieu khai bao / tep cut cut: tin do dai giai ma duoc
        dur = dur_dec
        if dur < 3 or dur > 600:
            R.append(reason("do_dai", f"{dur:.2f} s (giải mã được)"))
            return _finish(out, W, "04")
    lossy = codec in COMPRESSED
    nclip = _native_clip(path, ch, sr) if lossless_pcm else None
    chans, ci = _channels(x, M, W, R)
    if R:
        return _finish(out, W, "04")
    per = []
    for y, i in zip(chans, ci):
        m, w, r = analyse_channel(np.asarray(y, np.float32), dur, extended, events, None if nclip is None else nclip[i], lossy)
        m["kenh_goc"] = i
        per.append(m); W.extend(w); R.extend(r)
    M["kenh"] = per
    if events and any("su_kien_am_loi" in m for m in per):
        out["not_impl"] = not_impl(extended, False)
    if lossy:
        out["not_impl"].append(NOTE_LOSSY_CLIP)
    if any(m.get("so_nguoi_noi_tin_cay") is False for m in per):
        out["not_impl"].append(NOTE_GATED_SPK)
    best = max(m["tieng_noi_s"] for m in per)               # Constraint: tep hai kenh tinh tren kenh nhieu tieng noi hon
    if best < 2 or best < 0.3 * dur:
        R.append(reason("it_tieng_noi", f"{best:.1f} s tiếng nói / {dur:.1f} s (cần ≥ 2 s và ≥ 30 %)"))
    if R:
        return _finish(out, W, "04")
    W.append(WARN["dau_cuoi"])
    out["channels"] = [np.asarray(y, np.float32) for y in chans]
    return _finish(out, W, None)


def _channels(x, M, W, R) -> tuple:
    """Constraint 'Hai kenh', xet lan luot: (1) kenh < 2 s tieng noi -> bo kenh do · (2) giong het nhau (tuong quan > 0,999) -> gop mot ·
    (3) hai micro cung canh (tuong quan >= 0,3 sau khi can lech <= 50 ms) -> phan tich tung kenh · (4) con lai -> 04. Tra (kenh, chi so goc)."""
    if x.shape[1] == 1:
        return [x[:, 0]], [0]
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    sp_len = [float(speech_mask(frame_db(c.astype(np.float32)))[0].sum() * 0.03) for c in (a, b)]
    M["tieng_noi_nang_luong_s"] = [round(v, 1) for v in sp_len]
    if min(sp_len) < 2.0 <= max(sp_len):
        keep = int(np.argmax(sp_len))
        M["hai_kenh"] = f"bỏ kênh {'phải' if keep == 0 else 'trái'} (< 2 s tiếng nói) — chỉ phân tích kênh còn lại"
        return [x[:, keep]], [keep]
    same = np.corrcoef(a, b)[0, 1] > 0.999 if a.std() > 0 and b.std() > 0 else np.allclose(a, b)
    M["hai_kenh_giong_nhau"] = bool(same)
    if same:
        M["hai_kenh"] = "hai kênh giống hệt nhau — gộp một"
        return [x.mean(1)], [0]
    from scipy.signal import fftconvolve
    n = min(len(a), 60 * SR)
    a0, b0 = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    c = fftconvolve(b0, a0[::-1], mode="full")
    lag = int(0.05 * SR)
    rho = float(np.max(np.abs(c[n - 1 - lag:n + lag])) / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-12))
    M["tuong_quan_hai_kenh"] = round(rho, 3)
    if rho < 0.3:
        R.append(reason("hai_nguon", f"tương quan {rho:.2f} < 0,3"))
        return [], []
    M["hai_kenh"] = "hai micro cùng ghi một cảnh — phân tích từng kênh"
    return [x[:, 0], x[:, 1]], [0, 1]


def _finish(out: dict, W: list, decision: str | None) -> dict:
    ws, detail, notes = normalize_warnings(W)
    out["warnings"], out["warn_detail"] = ws, detail
    if notes:
        out["measures"]["ghi_chu_k1"] = notes
    out["reasons"] = list(dict.fromkeys(out["reasons"]))
    if decision is None:
        decision = "04" if out["reasons"] else ("canh_bao" if any(w != WARN["dau_cuoi"] for w in ws) else "nhan")
    out["decision"] = decision
    if decision in ("04", "tu_choi"):
        out["channels"] = []
    return out


def not_impl(extended: bool, events: bool) -> list:
    out = list(COND_ROWS)
    if not events:
        out += ["Loại âm thanh (không phải podcast, phát thanh, sách nói, nhạc)", "Kiểu nói (không hát, thì thầm, la hét)"]
    else:
        out.append("Loại âm thanh: podcast / phát thanh / sách nói không tách được bằng lớp AudioSet — chỉ chặn nhạc / bài hát "
                   "(nhạc không lời nhẹ, ambient có thể lọt: 3/6 đoạn thật bị chặn)")
        out.append("Kiểu nói: người lớn (trẻ em chỉ ghi số đo, không chặn)")
    if not extended:
        out += ["Ngôn ngữ (tiếng Việt / tiếng Anh)", "Nói chồng ≤ 10 %"]
    return out + ALWAYS_UNCHECKED


# ------------------------------------------------------------------ test (khong nap mo hinh: mat na ASR = nang luong, dem nguoi = 1)
def _light():
    import contextlib

    @contextlib.contextmanager
    def cm():
        g = globals()
        orig = (g["asr_speech_mask"], g["count_speakers"])
        g["asr_speech_mask"] = lambda y, win=0.03: speech_mask(frame_db(y, win))[0]
        g["count_speakers"] = lambda y, mask, win=0.03: 1
        try:
            yield
        finally:
            g["asr_speech_mask"], g["count_speakers"] = orig
    return cm()


def _speechlike(sec: float, seed: int = 0) -> np.ndarray:
    """Tin hieu gia loi noi: hai hoa 120-6000 Hz dieu bien am tiet 4 Hz, co khoang nghi, them nhieu rat nho."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sec * SR)) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    ph = 2 * np.pi * np.cumsum(f0) / SR
    y = sum(np.sin(k * ph) / k for k in range(1, 50) if k * 140 < 7600)
    env = (np.sin(2 * np.pi * 4 * t) > -0.2) * (np.sin(2 * np.pi * 0.25 * t) > -0.6)
    y = 0.3 * y / np.max(np.abs(y)) * env + rng.normal(0, 3e-3, len(t))
    return y.astype(np.float32)


# noi sinh DUY NHAT cua tung canh bao hop dong (O-2; vaudit/test_k1.test_warning_sources kiem du 12)
WARN_SOURCE = {"nén mạnh": "k1 (toc do bit khai)", "cắt đỉnh": "k1 (do tin hieu)", "có đoạn mất tín hiệu": "k1 (do tin hieu)",
               "nhiều người nói": "k1 (resemblyzer)", "kiểu nói ngoài phạm vi": "vaudit.events (AST)", "ngôn ngữ khác": "vaudit.k1x (--extended)",
               "nói chồng": "vaudit.k1x (--extended)", "0,3 s đầu và cuối không được soi": "k1 (luon co khi nhan)",
               "tiếng ồn cao": "vaudit.cond (pipeline)", "phòng vang": "vaudit.cond (pipeline)",
               "có đoạn nghi 03 chưa đủ chắc": "ticket (WP3)", "có thể còn chỗ sửa khác": "ticket (WP3)"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    if args and args[0] == "test":
        from vaudit import test_k1                # test tach rieng (k1.py < 800 dong)
        test_k1.run_tests()
    else:
        for f in [a for a in args if not a.startswith("--")]:
            r = check(f, extended="--extended" in args, events=False if "--no-events" in args else None)
            print(json.dumps({k: v for k, v in r.items() if k != "channels"}, ensure_ascii=False, indent=1, default=str))
