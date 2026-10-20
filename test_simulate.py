# -*- coding: utf-8 -*-
"""Test mo phong ghi am: RIR (pyroomacoustics, khong tai) + nhieu (MUSAN noise/)
sau do do RT60/SNR/G2 tren ban goc va ban mo phong de so sanh."""
import os
import sys
import glob
import random

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "data_sim")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import soundfile as sf
import pyroomacoustics as pra

# ---------- 1. Sinh phong ao (khong can tai RIR that) ----------
def make_room(sr, rt60_target=0.4):
    """Tao phong ngau nhien co RT60 xap xi rt60_target giay.
    Kich thuoc phong ti le theo RT60 muc tieu -- phong nho khong the
    vang lau du bao nhieu bac phan xa, dung vat ly thuc."""
    scale = 1.0 + (rt60_target - 0.25) / (0.9 - 0.25) * 2.0  # 1.0x - 3.0x
    room_dim = [
        random.uniform(3, 6) * scale,
        random.uniform(3, 6) * scale,
        random.uniform(2.5, 3.5) * (1 + (scale - 1) * 0.3),  # tran thap hon it
    ]
    e_absorption, max_order = pra.inverse_sabine(rt60_target, room_dim)
    max_order = min(max_order, 25)
    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption),
                        max_order=max_order)
    src = [room_dim[0] * 0.5, room_dim[1] * 0.5, 1.5]
    mic = [room_dim[0] * 0.3, room_dim[1] * 0.3, 1.2]
    room.add_source(src)
    room.add_microphone(mic)
    room.compute_rir()
    return room.rir[0][0]

# ---------- 2. Do RT60 tu chinh tin hieu (Schroeder backward integration) ----------
def estimate_rt60(rir):
    energy = rir[::-1] ** 2
    energy = np.cumsum(energy)[::-1]
    energy_db = 10 * np.log10(energy / (energy[0] + 1e-12) + 1e-12)
    try:
        i5 = np.where(energy_db <= -5)[0][0]
        i35 = np.where(energy_db <= -35)[0][0]
        slope = (energy_db[i35] - energy_db[i5]) / (i35 - i5)
        rt60 = -60 / slope / 1.0  # so mau -> giay xu ly ben ngoai bang sr
        return rt60
    except IndexError:
        return None

def estimate_snr(y, sr, frame=1024):
    """Uoc luong SNR tho: nang luong 10% khung yeu nhat = nhieu nen,
    90% khung manh nhat = tin hieu."""
    n = len(y) // frame
    if n < 4:
        return None
    energies = np.array([np.sum(y[i*frame:(i+1)*frame] ** 2) for i in range(n)])
    energies = energies[energies > 0]
    if len(energies) < 4:
        return None
    sorted_e = np.sort(energies)
    noise_e = np.mean(sorted_e[:max(1, len(sorted_e)//10)])
    signal_e = np.mean(sorted_e[-max(1, len(sorted_e)//10):])
    if noise_e <= 0:
        return None
    return 10 * np.log10(signal_e / noise_e)

def spectral_variance_silence_ratio(y, sr, frame=1024):
    """G2: ti so phuong sai pho tren doan im lang / doan co tieng."""
    n = len(y) // frame
    if n < 8:
        return None
    frames = np.array([y[i*frame:(i+1)*frame] for i in range(n)])
    energies = np.sum(frames ** 2, axis=1)
    thresh = np.percentile(energies, 30)
    silent = frames[energies <= thresh]
    loud = frames[energies > thresh]
    if len(silent) < 2 or len(loud) < 2:
        return None
    var_silent = np.mean(np.var(np.abs(np.fft.rfft(silent, axis=1)), axis=0))
    var_loud = np.mean(np.var(np.abs(np.fft.rfft(loud, axis=1)), axis=0))
    if var_loud <= 0:
        return None
    return var_silent / var_loud

# ---------- 3. Tim file nguon sach da tai (vivos, vlsp) ----------
def find_clean_sources(n=20):
    cands = []
    for sub in ("vivos", "vlsp2020_vinai_100h"):
        cands += glob.glob(os.path.join(DATA, sub, "*.wav"))
    random.shuffle(cands)
    return cands[:n]

# ---------- 4. Tim file nhieu tu MUSAN (neu da co) hoac dung nhieu trang ----------
def load_noise(sr, length):
    noise_dir = os.path.join(DATA, "musan_noise")
    files = glob.glob(os.path.join(noise_dir, "**", "*.wav"), recursive=True)
    if files:
        f = random.choice(files)
        y, nsr = sf.read(f)
        if nsr != sr:
            import librosa
            y = librosa.resample(y.astype(float), orig_sr=nsr, target_sr=sr)
        if len(y) < length:
            y = np.tile(y, int(np.ceil(length / len(y))))
        start = random.randint(0, max(0, len(y) - length))
        return y[start:start+length]
    # Chua co MUSAN -> dung nhieu trang Gaussian tam thoi de vao test duoc ngay
    return np.random.normal(0, 0.01, length)

# ---------- 5. Mo phong 1 file: tich chap RIR + tron nhieu ----------
def simulate(y, sr, rt60_target):
    rir = make_room(sr, rt60_target)
    wet = np.convolve(y, rir)[:len(y)]
    wet = wet / (np.max(np.abs(wet)) + 1e-9) * 0.9
    noise = load_noise(sr, len(wet))
    snr_db = random.uniform(15, 30)
    sig_power = np.mean(wet ** 2)
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(sig_power / (10 ** (snr_db / 10)) / noise_power)
    out = wet + noise * scale
    out = out / (np.max(np.abs(out)) + 1e-9) * 0.9
    return out, rir

# ---------- MAIN ----------
if __name__ == "__main__":
    sources = find_clean_sources(20)
    print(f"So file nguon sach tim duoc: {len(sources)}")
    if not sources:
        print("KHONG TIM THAY file .wav trong data/vivos hoac data/vlsp2020_vinai_100h")
        sys.exit(1)

    results = []
    for i, f in enumerate(sources):
        y, sr = sf.read(f)
        if y.ndim > 1:
            y = y.mean(axis=1)
        rt60_target = random.uniform(0.25, 0.7)
        sim, rir = simulate(y, sr, rt60_target)

        rt60_est = estimate_rt60(rir)
        rt60_est_sec = rt60_est / sr if rt60_est else None
        snr_orig = estimate_snr(y, sr)
        snr_sim = estimate_snr(sim, sr)
        g2_orig = spectral_variance_silence_ratio(y, sr)
        g2_sim = spectral_variance_silence_ratio(sim, sr)

        out_path = os.path.join(OUT, f"sim_{i:02d}.wav")
        sf.write(out_path, sim, sr)

        results.append({
            "file": os.path.basename(f),
            "rt60_target": rt60_target,
            "rt60_measured": rt60_est_sec,
            "snr_orig": snr_orig,
            "snr_sim": snr_sim,
            "g2_orig": g2_orig,
            "g2_sim": g2_sim,
        })

    print(f"\nDa mo phong {len(results)} file -> {OUT}\n")
    print(f"{'File':<30} {'RT60 dat':>9} {'RT60 do':>9} {'SNR goc':>9} {'SNR mp':>9} {'G2 goc':>9} {'G2 mp':>9}")
    for r in results:
        def fmt(v, nd=2):
            return f"{v:.{nd}f}" if v is not None else "?"
        print(f"{r['file'][:30]:<30} {fmt(r['rt60_target']):>9} {fmt(r['rt60_measured']):>9} "
              f"{fmt(r['snr_orig']):>9} {fmt(r['snr_sim']):>9} {fmt(r['g2_orig'],3):>9} {fmt(r['g2_sim'],3):>9}")

    valid_rt60 = [r["rt60_target"] for r in results if r["rt60_measured"] is not None]
    valid_snr = [r["snr_sim"] for r in results if r["snr_sim"] is not None]
    print(f"\nRT60 dat trung binh: {np.mean(valid_rt60):.2f}s (muc tieu 0.25-0.9s)")
    if valid_snr:
        print(f"SNR ban mo phong trung binh: {np.mean(valid_snr):.1f} dB (muc tieu 12-38dB de la 'ghi am')")

    # ---------- 6. "Chong may": so SNR & G2 giua ban mo phong va ghi-am-that (AMI) ----------
    # Luu y: RT60 chi do truc tiep duoc tu RIR da biet (chi co o ban mo phong).
    # Voi ghi am that khong co RIR de doi chieu, nen phep so sanh cong bang
    # chi dung SNR va G2 -- ca hai deu tinh thang tu song am, ap dung duoc ca hai nhom.
    ami_files = glob.glob(os.path.join(DATA, "ami_real", "**", "*.wav"), recursive=True)
    print(f"\n--- Doi chieu voi {len(ami_files)} file ghi-am-that (AMI) ---")
    ami_snr, ami_g2 = [], []
    for f in ami_files:
        try:
            y, sr = sf.read(f)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if len(y) < sr:  # bo file qua ngan (<1s), khong du de uoc luong
                continue
            s = estimate_snr(y, sr)
            g = spectral_variance_silence_ratio(y, sr)
            if s is not None:
                ami_snr.append(s)
            if g is not None:
                ami_g2.append(g)
        except Exception:
            continue

    sim_snr = [r["snr_sim"] for r in results if r["snr_sim"] is not None]
    sim_g2 = [r["g2_sim"] for r in results if r["g2_sim"] is not None]

    def stats(name, arr):
        if not arr:
            return f"{name}: khong co du lieu"
        return f"{name}: min={min(arr):.3f} max={max(arr):.3f} trung binh={np.mean(arr):.3f} (n={len(arr)})"

    print(stats("SNR   mo phong ", sim_snr))
    print(stats("SNR   AMI that  ", ami_snr))
    print(stats("G2    mo phong ", sim_g2))
    print(stats("G2    AMI that  ", ami_g2))

    if sim_snr and ami_snr:
        overlap_snr = (max(min(sim_snr), min(ami_snr)), min(max(sim_snr), max(ami_snr)))
        ok = overlap_snr[0] < overlap_snr[1]
        print(f"\nSNR: vung chong lap = {overlap_snr}  ->  {'CO CHONG LAP' if ok else 'KHONG CHONG LAP -- can dieu chinh'}")
    if sim_g2 and ami_g2:
        overlap_g2 = (max(min(sim_g2), min(ami_g2)), min(max(sim_g2), max(ami_g2)))
        ok = overlap_g2[0] < overlap_g2[1]
        print(f"G2:  vung chong lap = {overlap_g2}  ->  {'CO CHONG LAP' if ok else 'KHONG CHONG LAP -- can dieu chinh'}")
