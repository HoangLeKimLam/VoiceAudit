# -*- coding: utf-8 -*-
"""Cac thao tac chinh sua co ban, CAT/NOI TAI RANH GIOI TU (giua khoang ho giua 2 tu).

Moi ham tra ve (song_moi, danh_sach_moc) theo TRUC THOI GIAN CUA FILE MOI,
de ground-truth dinh vi luon dung sau khi file bi co/gian.
"""
import numpy as np

FADE_MS = 10


def gap_before(words, k: int) -> float:
    """Diem cat truoc tu k: giua khoang lang (tu k-1 ket thuc, tu k bat dau)."""
    if k <= 0:
        return max(0.0, words[0].start - 0.04)
    return (words[k - 1].end + words[k].start) / 2


def gap_after(words, m: int, total_sec: float) -> float:
    if m >= len(words) - 1:
        return min(total_sec, words[-1].end + 0.04)
    return (words[m].end + words[m + 1].start) / 2


def _fade(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def join(parts: list[np.ndarray], sr: int, fade_ms: float = FADE_MS) -> tuple[np.ndarray, list[int]]:
    """Noi cac doan voi crossfade ngan (ke gia mao hay dung de tranh tieng 'tach').
    Tra ve song va vi tri (mau) cua tung moi noi tren song moi."""
    nf = int(sr * fade_ms / 1000)
    out = np.asarray(parts[0], float)
    joints = []
    for p in parts[1:]:
        p = np.asarray(p, float)
        n = min(nf, len(out), len(p))
        if n > 1:
            f = _fade(n)
            mix = out[-n:] * (1 - f) + p[:n] * f
            joints.append(len(out) - n // 2)
            out = np.concatenate([out[:-n], mix, p[n:]])
        else:
            joints.append(len(out))
            out = np.concatenate([out, p])
    return out, joints


def replace_span(host: np.ndarray, a_sec: float, b_sec: float, new_seg: np.ndarray, sr: int):
    """Thay host[a:b] bang new_seg. Tra ve (song, (bat_dau, ket_thuc) doan moi theo giay)."""
    a, b = int(a_sec * sr), int(b_sec * sr)
    out, joints = join([host[:a], new_seg, host[b:]], sr)
    return out, (joints[0] / sr, joints[1] / sr)


def delete_span(host: np.ndarray, a_sec: float, b_sec: float, sr: int):
    """Cat bo host[a:b], noi 2 dau lai. Tra ve (song, moc_noi_giay)."""
    a, b = int(a_sec * sr), int(b_sec * sr)
    out, joints = join([host[:a], host[b:]], sr)
    return out, joints[0] / sr


def blend_process(host: np.ndarray, a_sec: float, b_sec: float, processed: np.ndarray, sr: int,
                  fade_ms: float = 20) -> np.ndarray:
    """Chen ban da xu ly (khu on/doi giong) vao dung doan [a,b] voi chuyen tiep mem
    (nhu 'chon vung roi xu ly' trong phan mem bien tap). Do dai file khong doi."""
    a, b = int(a_sec * sr), int(b_sec * sr)
    seg = np.asarray(processed, float)[:b - a]
    if len(seg) < b - a:
        seg = np.pad(seg, (0, b - a - len(seg)))
    out = np.asarray(host, float).copy()
    n = min(int(sr * fade_ms / 1000), (b - a) // 4)
    w = np.ones(b - a)
    if n > 1:
        w[:n] = _fade(n)
        w[-n:] = _fade(n)[::-1]
    out[a:b] = host[a:b] * (1 - w) + seg * w
    return out


def shift_times(times: list[tuple[float, float]], at_sec: float, delta_sec: float):
    """Doi cac moc nam SAU at_sec them delta (dung khi co nhieu thao tac lien tiep)."""
    return [(s + delta_sec if s >= at_sec else s, e + delta_sec if e >= at_sec else e) for s, e in times]
