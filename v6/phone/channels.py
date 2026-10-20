# -*- coding: utf-8 -*-
"""M7 - MO PHONG KENH DIEN THOAI / APP (khong can dien thoai that). Moi cau hinh ghi ro thong so lay tu dau.

  Nen don        MP3 32/64/128 · AAC-LC 24/32/64/128 · Opus 16 (VoIP, 16 kHz)/24/32/64 · AMR-WB 12,65/23,85 · G.722 · Vorbis
  Chuoi          goi roi luu (nen an) · chuyen tiep nhieu lan · nen hai lan
  Kich ban app   WhatsApp/Telegram tin nhan thoai (Opus 16 kHz 16 kbps - co nguon) · Zalo, Messenger tin nhan thoai (GIA DINH:
                 khong co tai lieu cong khai) · goi qua app roi ghi lai (WebRTC HPF+NS+AGC -> Opus -> mat goi -> AAC) ·
                 ghi cuoc goi VoLTE (AMR-WB 12,65 -> AAC 128) · ghi hinh Zoom (AAC 32 kHz ~55 kbps - co nguon)
Xu ly WebRTC dung thu vien that (aec_audio_processing = WebRTC APM). Mat goi: bo 20 ms theo dot 1-3 khung, che bang lap
khung truoc suy giam dan (XAP XI bo che mat goi cua Opus/NetEQ). Moi ham tra (song 16 kHz float32, meta)."""
from __future__ import annotations

import os
import subprocess
import tempfile

import numpy as np

SR = 16000

# ten -> (chuoi buoc, mo ta, nguon thong so)
# buoc: ("codec", ten_codec, kbps, sr_ma_hoa) | ("webrtc", ns_level, agc) | ("hpf", hz) | ("loss", ti_le)
CONFIGS: dict[str, tuple[list, str, str]] = {
    "goc": ([], "khong doi", "-"),
    "mp3_32": ([("codec", "mp3", 32, 16000)], "MP3 32 kbps", "chuan"),
    "mp3_64": ([("codec", "mp3", 64, 44100)], "MP3 64 kbps", "chuan"),
    "mp3_128": ([("codec", "mp3", 128, 44100)], "MP3 128 kbps", "chuan"),
    "aac_24": ([("codec", "aac", 24, 16000)], "AAC-LC 24 kbps", "chuan"),
    "aac_32": ([("codec", "aac", 32, 16000)], "AAC-LC 32 kbps", "chuan"),
    "aac_64": ([("codec", "aac", 64, 44100)], "AAC-LC 64 kbps", "chuan"),
    "aac_128": ([("codec", "aac", 128, 44100)], "AAC-LC 128 kbps", "chuan"),
    "opus_16": ([("codec", "opus_voip", 16, 16000)], "Opus 16 kbps che do goi", "chuan"),
    "opus_24": ([("codec", "opus", 24, 48000)], "Opus 24 kbps", "chuan"),
    "opus_32": ([("codec", "opus", 32, 48000)], "Opus 32 kbps", "chuan"),
    "opus_64": ([("codec", "opus", 64, 48000)], "Opus 64 kbps", "chuan"),
    "amrwb_12": ([("codec", "amrwb", 12.65, 16000)], "AMR-WB 12,65 kbps (VoLTE)", "3GPP"),
    "amrwb_23": ([("codec", "amrwb", 23.85, 16000)], "AMR-WB 23,85 kbps", "3GPP"),
    "g722": ([("codec", "g722", 64, 16000)], "G.722 64 kbps (tong dai)", "ITU"),
    "vorbis": ([("codec", "vorbis", 48, 44100)], "Vorbis ~48 kbps", "chuan"),
    "chuoi_opus16_aac128": ([("codec", "opus_voip", 16, 16000), ("codec", "aac", 128, 44100)],
                            "goi Opus 16 roi luu AAC 128 (nen an)", "suy luan"),
    "chuoi_amrwb_aac128": ([("codec", "amrwb", 12.65, 16000), ("codec", "aac", 128, 16000)],
                           "VoLTE AMR-WB roi luu AAC 128 (nen an)", "suy luan"),
    "chuoi_aac32_opus16": ([("codec", "aac", 32, 16000), ("codec", "opus_voip", 16, 16000)], "chuyen tiep AAC -> Opus", "suy luan"),
    "chuoi_mp3_128_64": ([("codec", "mp3", 128, 44100), ("codec", "mp3", 64, 44100)], "nen MP3 hai lan", "suy luan"),
    "chuoi_3lan": ([("codec", "aac", 32, 16000), ("codec", "opus_voip", 16, 16000), ("codec", "aac", 32, 16000)],
                   "chuyen tiep 3 lan", "suy luan"),
    "app_whatsapp_telegram": ([("hpf", 80), ("codec", "opus_voip", 16, 16000)],
                              "tin nhan thoai WhatsApp/Telegram: Opus 16 kHz 16 kbps", "co nguon"),
    "app_zalo_tin_nhan": ([("webrtc", 2, True), ("codec", "aac", 32, 16000)],
                          "tin nhan thoai Zalo: AAC 16 kHz 32 kbps + khu on", "GIA DINH"),
    "app_messenger_tin_nhan": ([("codec", "aac", 48, 44100)], "tin nhan thoai Messenger: AAC 44,1 kHz 48 kbps", "GIA DINH"),
    "app_goi_ghi_lai": ([("webrtc", 2, True), ("codec", "opus_voip", 24, 16000), ("loss", 0.02), ("codec", "aac", 64, 44100)],
                        "goi qua app (WebRTC + Opus 24 + mat goi 2 %) roi ghi AAC 64", "co nguon (chuoi WebRTC)"),
    "app_volte_ghi_cuoc_goi": ([("webrtc", 1, False), ("codec", "amrwb", 12.65, 16000), ("codec", "aac", 128, 16000)],
                               "cuoc goi VoLTE ghi lai: AMR-WB 12,65 -> AAC 128", "3GPP + suy luan"),
    "app_zoom_ghi_hinh": ([("webrtc", 2, True), ("codec", "opus", 32, 48000), ("codec", "aac", 55, 32000)],
                          "ghi hinh Zoom: Opus 32 -> AAC 32 kHz ~55 kbps", "co nguon"),
}
MAIN_PHONE = ["app_zalo_tin_nhan", "app_whatsapp_telegram", "app_goi_ghi_lai", "app_volte_ghi_cuoc_goi"]

_ENC = {  # ten -> (duoi tep, tham so ffmpeg theo kbps)
    "mp3": (".mp3", lambda k: ["-c:a", "libmp3lame", "-b:a", f"{k}k"]),
    "aac": (".m4a", lambda k: ["-c:a", "aac", "-b:a", f"{k}k"]),
    "opus": (".ogg", lambda k: ["-c:a", "libopus", "-b:a", f"{k}k", "-application", "audio"]),
    "opus_voip": (".ogg", lambda k: ["-c:a", "libopus", "-b:a", f"{k}k", "-application", "voip"]),
    "amrwb": (".amr", lambda k: ["-c:a", "libvo_amrwbenc", "-b:a", f"{k}k"]),
    "g722": (".wav", lambda k: ["-c:a", "g722"]),
    "vorbis": (".ogg", lambda k: ["-c:a", "libvorbis", "-q:a", "3"]),
}


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    r = subprocess.run(cmd, input=data, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "ignore")[-400:])
    return r.stdout


def codec(y: np.ndarray, name: str, kbps: float, enc_sr: int) -> np.ndarray:
    """Ma hoa -> giai ma ve 16 kHz mono float32 (qua tep tam, dung container that cua codec)."""
    ext, args = _ENC[name]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in.f32")
        enc = os.path.join(d, "enc" + ext)
        np.asarray(y, np.float32).tofile(src)
        _run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", src,
              "-ar", str(enc_sr), "-ac", "1", *args(kbps), enc])
        out = _run(["ffmpeg", "-v", "error", "-i", enc, "-f", "f32le", "-ar", str(SR), "-ac", "1", "-"])
    z = np.frombuffer(out, np.float32).copy()
    return align(z, y)


MAX_LAG = int(0.15 * SR)


def lag_of(z: np.ndarray, y: np.ndarray) -> int:
    """Do tre (mau) cua z so voi y: dinh tuong quan cheo trong +-150 ms tren 4 s dau."""
    from scipy.signal import fftconvolve
    n = min(len(y), len(z), 4 * SR)
    a, b = y[:n] - y[:n].mean(), z[:n] - z[:n].mean()
    c = fftconvolve(b, a[::-1], mode="full")
    mid = n - 1
    seg = c[mid - MAX_LAG:mid + MAX_LAG + 1]
    return int(np.argmax(seg)) - MAX_LAG


def align(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bu tre bo ma (AMR-WB, G.722, Opus... them mau dau) de moc thoi gian cua nhan khong lech, roi dua ve dung do dai goc."""
    lag = lag_of(z, y)
    z = z[lag:] if lag > 0 else np.pad(z, (-lag, 0))
    return (np.pad(z, (0, max(0, len(y) - len(z))))[:len(y)]).astype(np.float32)


def webrtc(y: np.ndarray, ns_level: int, agc: bool) -> np.ndarray:
    import aec_audio_processing as apm
    p = apm.AudioProcessor(enable_aec=False, enable_ns=True, ns_level=ns_level, enable_agc=agc, agc_mode=1, enable_vad=False)
    p.set_stream_format(SR, 1, SR, 1)
    n = SR // 100
    x = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    pad = (-len(x)) % n
    x = np.pad(x, (0, pad))
    out = [np.frombuffer(p.process_stream(x[i:i + n].tobytes()), np.int16) for i in range(0, len(x), n)]
    return align(np.concatenate(out).astype(np.float32) / 32767, np.asarray(y, np.float32))


def hpf(y: np.ndarray, hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    return sosfilt(butter(2, hz, "highpass", fs=SR, output="sos"), y).astype(np.float32)


def packet_loss(y: np.ndarray, rate: float, rng: np.random.Generator) -> tuple[np.ndarray, list]:
    """Bo khung 20 ms theo dot 1-3 khung; che bang lap khung truoc, suy giam 6 dB moi khung, noi mem 2 ms."""
    f = SR // 50
    z = y.copy()
    lost = []
    i = f
    while i + f < len(z):
        if rng.random() < rate / 2:
            burst = int(rng.integers(1, 4))
            prev = z[i - f:i].copy()
            for b in range(burst):
                a = i + b * f
                if a + f > len(z):
                    break
                seg = prev * (10 ** (-6 * (b + 1) / 20))
                fade = np.linspace(0, 1, 32)
                seg[:32] = z[a:a + 32] * (1 - fade) + seg[:32] * fade
                z[a:a + f] = seg
            lost.append(round(i / SR, 3))
            i += burst * f
        i += f
    return z, lost


def apply(y: np.ndarray, cfg: str, seed: int = 0) -> tuple[np.ndarray, dict]:
    steps, desc, src = CONFIGS[cfg]
    rng = np.random.default_rng(seed)
    z = np.asarray(y, np.float32)
    meta = {"kenh": cfg, "mo_ta": desc, "nguon_thong_so": src}
    for st in steps:
        if st[0] == "codec":
            z = codec(z, st[1], st[2], st[3])
        elif st[0] == "webrtc":
            z = webrtc(z, st[1], st[2])
        elif st[0] == "hpf":
            z = hpf(z, st[1])
        elif st[0] == "loss":
            z, lost = packet_loss(z, st[1], rng)
            meta["mat_goi_s"] = lost
    peak = float(np.abs(z).max()) if len(z) else 0.0
    if peak > 0.999:
        z = z / peak * 0.999
    return z.astype(np.float32), meta


if __name__ == "__main__":
    import sys
    import soundfile as sf
    sys.stdout.reconfigure(encoding="utf-8")
    y, sr = sf.read(sys.argv[1], dtype="float32")
    assert sr == SR and y.ndim == 1, "can tep 16 kHz mono"
    for c in CONFIGS:
        z, m = apply(y, c)
        g = float(np.dot(z, y) / (np.dot(y, y) + 1e-12))       # bo khac biet do AGC doi muc
        err = 10 * np.log10(np.mean((z - g * y) ** 2) / (np.mean((g * y) ** 2) + 1e-12) + 1e-12)
        print(f"{c:26s} dai khop {len(z) == len(y)}  tre con lai {lag_of(z, y):+4d} mau  sai khac {err:6.1f} dB"
              f"  | {m['nguon_thong_so']}")
