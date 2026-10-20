# -*- coding: utf-8 -*-
"""AMI (NXT words.xml) -> tu co moc thoi gian theo nguoi noi, luoi hoat dong 10 ms, ty le noi chong.

Nguon: data/ami/ami_public_manual_1.6.2.zip (words/<hop>.<A-E>.words.xml, corpusResources/meetings.xml).
Cach doc lay tu .tmp_research/r1_ami/ami_windows.py + edit_feasibility.py:
  - tu = <w> khong phai dau cau (punc) co starttime < endtime;
  - hoat dong noi (activity) = cac tu cung nguoi gop khi khoang ho <= 0,3 s, tren luoi 10 ms;
  - 'ban' (busy) = tu + vocalsound/nonvocalsound (cuoi, ho, go ban) KHONG gop -> dung de kiem 'mot minh'
    (solo) va tim khoang lang that (lay tieng on nen cho doan TTS).
Thoi gian luon la giay TUYET DOI trong cuoc hop (trung truc voi tep wav cua cuoc hop)."""
from __future__ import annotations

import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AMI_DIR = os.path.join(BASE, "data", "ami")
ZIP_PATH = os.path.join(AMI_DIR, "ami_public_manual_1.6.2.zip")
AUDIO_DIR = os.path.join(AMI_DIR, "audio")
MEETINGS = ("ES2004a", "ES2004b", "ES2004c", "ES2004d", "IS1009a", "IS1009b", "TS3003a", "TS3003b")
MICS = ("Array1-01", "Mix-Headset")          # Array1-01 = 'dien thoai dat tren ban' (chinh) · Mix-Headset = doi chung
MIC_TAG = {"Array1-01": "arr", "Mix-Headset": "hs"}
RES = 0.01                                   # luoi 10 ms
MERGE_GAP = 0.30                             # tu cung nguoi cach <= 0,3 s -> mot doan noi
# fold theo SERIES (cung nguoi + cung phong trong mot series). Nguoi noi AMI khong co trong VIVOS nen fold nao cung
# khong ro ri nguoi noi; fold chi chon mo hinh fold nao cham. TS3003 = LOCKBOX (chi bao cao cuoi).
SERIES_FOLD = {"ES2004": 0, "IS1009": 1, "TS3003": 2}
LOCKBOX_SERIES = frozenset({"TS3003"})
FILLERS = frozenset({"um", "uh", "hmm", "mm", "mmm", "hm", "er", "erm", "ah", "oh", "eh", "huh", "mhm", "mm-hmm",
                     "uh-huh", "uh-uh", "hmm-mm", "ooh", "aha", "ha"})
STOP_EN = frozenset({"the", "a", "an", "and", "or", "but", "so", "to", "of", "in", "on", "at", "it", "is", "i", "you",
                     "we", "he", "she", "they", "that", "this", "be", "are", "was", "for", "with", "as", "if", "then",
                     "yeah", "yes", "no", "okay", "ok", "right", "well", "just", "like", "not", "do", "have", "has",
                     "its", "it's", "i'm", "that's", "there", "what", "um", "uh"})


@dataclass(frozen=True)
class AWord:
    """Mot tu AMI (co .start/.end nhu vaudit.asr.Word -> dung duoc voi vaudit.edits.gap_before/gap_after)."""
    start: float
    end: float
    text: str
    agent: str
    trunc: bool = False

    @property
    def norm(self) -> str:
        return normalize(self.text)


def normalize(text: str) -> str:
    """Chu thuong, chi giu a-z, dau nhay va gach noi (so khop cung tu giua hai lan noi)."""
    return re.sub(r"[^a-z'\-]", "", text.lower()).strip("-'")


def is_clean_word(w: AWord) -> bool:
    """Tu doc tron (khong cut, khong phai tieng dem) va chi gom chu cai tren CHU GOC (loai 'T_V_', 'twen-', so)
    -> dung duoc cho ghep / TTS."""
    n = w.norm
    return (bool(n) and not w.trunc and n not in FILLERS
            and re.fullmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?", w.text.strip()) is not None)


def is_content_word(w: AWord) -> bool:
    return is_clean_word(w) and w.norm not in STOP_EN and len(w.norm) >= 3


def frames(s: float, e: float, n: int | None = None, res: float = RES) -> tuple[int, int]:
    """Khoang [s, e) giay -> [i0, i1) chi so khung (it nhat 1 khung)."""
    i0 = max(0, int(np.floor(s / res + 1e-9)))
    i1 = max(i0 + 1, int(np.ceil(e / res - 1e-9)))
    if n is not None:
        i0, i1 = min(i0, n), min(i1, n)
    return i0, i1


def speech_runs(intervals: list[tuple[float, float]], merge_gap: float = MERGE_GAP) -> list[list[float]]:
    """Gop cac khoang (s, e) cach nhau <= merge_gap thanh doan noi lien."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Mat na bool -> cac doan [i0, i1) lien tiep bang True."""
    m = np.r_[False, np.asarray(mask, bool), False]
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(d[::2], d[1::2])]


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    if r <= 0:
        return np.asarray(mask, bool).copy()
    return np.convolve(np.asarray(mask, float), np.ones(2 * r + 1), "same") > 0.5


@dataclass
class Meeting:
    name: str
    agents: dict[str, str]                        # 'A' -> 'AMI_FEE013'
    words: dict[str, list[AWord]]                 # theo nguoi, sap theo thoi gian
    sounds: dict[str, list[tuple[float, float]]]  # vocalsound / nonvocalsound theo nguoi
    audio_dur: float
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def series(self) -> str:
        return self.name[:6]

    @property
    def fold(self) -> int:
        return SERIES_FOLD[self.series]

    @property
    def lockbox(self) -> bool:
        return self.series in LOCKBOX_SERIES

    @property
    def agent_list(self) -> list[str]:
        return sorted(self.words)

    @property
    def extent(self) -> tuple[float, float]:
        """Phan da phien am: tu dau tien -> tu cuoi cung (moi nguoi)."""
        ws = [w for v in self.words.values() for w in v]
        return (min(w.start for w in ws), max(w.end for w in ws)) if ws else (0.0, 0.0)

    @property
    def n_frames(self) -> int:
        return int(np.ceil(max(self.audio_dur, self.extent[1]) / RES)) + 2

    def speaker(self, agent: str) -> str:
        return self.agents.get(agent) or f"AMI_{self.name}_{agent}"

    def audio_path(self, mic: str) -> str:
        return os.path.join(AUDIO_DIR, f"{self.name}.{mic}.wav")

    def all_words(self) -> list[AWord]:
        if "all" not in self._cache:
            self._cache["all"] = sorted((w for v in self.words.values() for w in v), key=lambda w: (w.start, w.end))
        return self._cache["all"]

    def activity(self) -> np.ndarray:
        """(so nguoi x so khung) bool: doan noi (tu gop <= 0,3 s) — dung tinh ty le noi, noi chong, luot noi."""
        if "act" not in self._cache:
            n = self.n_frames
            A = np.zeros((len(self.agent_list), n), bool)
            for k, a in enumerate(self.agent_list):
                for s, e in speech_runs([(w.start, w.end) for w in self.words[a]]):
                    i0, i1 = frames(s, e, n)
                    A[k, i0:i1] = True
            self._cache["act"] = A
        return self._cache["act"]

    def word_grid(self) -> np.ndarray:
        """(so nguoi x so khung) bool: chi tu, khong gop (noi chong muc tu)."""
        if "wg" not in self._cache:
            n = self.n_frames
            G = np.zeros((len(self.agent_list), n), bool)
            for k, a in enumerate(self.agent_list):
                for w in self.words[a]:
                    i0, i1 = frames(w.start, w.end, n)
                    G[k, i0:i1] = True
            self._cache["wg"] = G
        return self._cache["wg"]

    def busy(self) -> np.ndarray:
        """(so nguoi x so khung) bool: tu + am thanh phi loi (cuoi, ho, tieng dong) cua tung nguoi."""
        if "busy" not in self._cache:
            B = self.word_grid().copy()
            for k, a in enumerate(self.agent_list):
                for s, e in self.sounds.get(a, []):
                    i0, i1 = frames(s, e, B.shape[1])
                    B[k, i0:i1] = True
            self._cache["busy"] = B
        return self._cache["busy"]

    def solo(self, agent: str, s: float, e: float, pad: float = 0.3) -> bool:
        """Khong ai KHAC phat ra am (tu / cuoi / tieng dong) trong [s - pad, e + pad]."""
        B = self.busy()
        i0, i1 = frames(s - pad, e + pad, B.shape[1])
        others = [k for k, a in enumerate(self.agent_list) if a != agent]
        return not B[others, i0:i1].any()

    def pauses(self, t0: float, t1: float, guard: float = 0.15, min_len: float = 0.2) -> list[tuple[float, float]]:
        """Khoang lang that trong [t0, t1): khong ai phat am, cach moi am >= guard giay, dai >= min_len."""
        B = self.busy()
        i0, i1 = frames(t0, t1, B.shape[1])
        free = ~dilate(B[:, i0:i1].any(0), int(round(guard / RES)))
        return [((i0 + a) * RES, (i0 + b) * RES) for a, b in bool_runs(free) if (b - a) * RES >= min_len]


# ---------------- doc zip ----------------
def open_zip(path: str = ZIP_PATH) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def read_agents(zf: zipfile.ZipFile) -> dict[str, dict]:
    """meetings.xml -> {hop: {'agents': {'A': 'AMI_FEE013', ...}, 'duration': giay}}."""
    root = ET.fromstring(zf.read("corpusResources/meetings.xml"))
    out = {}
    for m in root:
        obs = m.get("observation")
        if not obs:
            continue
        ag = {s.get("nxt_agent"): "AMI_" + s.get("global_name") for s in m if s.get("nxt_agent") and s.get("global_name")}
        out[obs] = {"agents": ag, "duration": float(m.get("duration") or 0)}
    return out


def _parse_words(xml_bytes: bytes, agent: str) -> tuple[list[AWord], list[tuple[float, float]]]:
    words, sounds = [], []
    for el in ET.fromstring(xml_bytes):
        s, e = el.get("starttime"), el.get("endtime")
        if s is None or e is None:
            continue
        try:
            s, e = float(s), float(e)
        except ValueError:
            continue
        if e <= s:
            continue
        if el.tag == "w" and el.get("punc") != "true":
            words.append(AWord(s, e, (el.text or "").strip(), agent, el.get("trunc") == "true"))
        elif el.tag in ("vocalsound", "nonvocalsound"):
            sounds.append((s, e))
    words.sort(key=lambda w: (w.start, w.end))
    return words, sorted(sounds)


def load_meeting(name: str, zf: zipfile.ZipFile | None = None, meta: dict | None = None) -> Meeting:
    """Nap mot cuoc hop: tu + am phi loi theo nguoi, ten nguoi noi toan cuc, do dai audio (Array1-01 neu co)."""
    own = zf is None
    zf = zf or open_zip()
    try:
        meta = meta if meta is not None else read_agents(zf)
        words, sounds = {}, {}
        for fn in zf.namelist():
            mm = re.match(rf"words/{re.escape(name)}\.([A-Z])\.words\.xml$", fn)
            if mm:
                words[mm.group(1)], sounds[mm.group(1)] = _parse_words(zf.read(fn), mm.group(1))
    finally:
        if own:
            zf.close()
    if not words:
        raise FileNotFoundError(f"khong co words/{name}.*.words.xml trong {ZIP_PATH}")
    info = meta.get(name, {"agents": {}, "duration": 0.0})
    dur = info["duration"]
    wav = os.path.join(AUDIO_DIR, f"{name}.{MICS[0]}.wav")
    if os.path.exists(wav):
        import soundfile as sf
        dur = sf.info(wav).duration
    return Meeting(name, dict(info["agents"]), words, sounds, float(dur))


def load_all(names: tuple[str, ...] | list[str] = MEETINGS, zip_path: str = ZIP_PATH) -> dict[str, Meeting]:
    with open_zip(zip_path) as zf:
        meta = read_agents(zf)
        return {n: load_meeting(n, zf, meta) for n in names}


def overlap_ratio(A: np.ndarray) -> float:
    """Ty le noi chong = thoi gian >= 2 nguoi / thoi gian >= 1 nguoi (0 neu khong ai noi)."""
    c = np.asarray(A).sum(0)
    sp = int((c >= 1).sum())
    return float((c >= 2).sum() / sp) if sp else 0.0
