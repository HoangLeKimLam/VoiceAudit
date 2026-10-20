# -*- coding: utf-8 -*-
"""ASR tieng Viet (wav2vec2-base-vietnamese-250h, CTC) kem MOC THOI GIAN TUNG TU.

Dung de: (1) biet noi dung goc cua file host, (2) cat/chen DUNG ranh gioi tu,
(3) kiem tra lai noi dung file da chinh sua co con dung nhu mong doi khong.
"""
from dataclasses import dataclass

import numpy as np
import torch

from . import SR

ASR_NAME = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
_state = {}


@dataclass(frozen=True)
class Word:
    text: str
    start: float  # giay
    end: float


def _load():
    if "model" not in _state:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        _state["proc"] = Wav2Vec2Processor.from_pretrained(ASR_NAME)
        _state["model"] = Wav2Vec2ForCTC.from_pretrained(ASR_NAME).eval()
        torch.set_num_threads(max(1, (torch.get_num_threads())))
    return _state["proc"], _state["model"]


def transcribe_words(y: np.ndarray, sr: int = SR) -> tuple[str, list[Word]]:
    """Tra ve (transcript, danh sach Word co moc thoi gian) tu giai ma CTC tham lam."""
    proc, model = _load()
    inp = proc(y.astype(np.float32), sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inp).logits[0]
    ids = logits.argmax(-1).numpy()
    frame_sec = len(y) / sr / len(ids)

    tok = proc.tokenizer
    pad_id = tok.pad_token_id
    delim = tok.word_delimiter_token  # "|"
    id2tok = {v: k for k, v in tok.get_vocab().items()}

    words, cur_chars, cur_start, cur_end = [], [], None, None
    prev = None
    for f, i in enumerate(ids):
        if i == prev:
            if cur_chars and i != pad_id and id2tok.get(i) != delim:
                cur_end = f
            continue
        prev = i
        if i == pad_id:
            continue
        t = id2tok.get(int(i), "")
        if t == delim or t == " ":
            if cur_chars:
                words.append(Word("".join(cur_chars), cur_start * frame_sec, (cur_end + 1) * frame_sec))
                cur_chars, cur_start, cur_end = [], None, None
            continue
        if not cur_chars:
            cur_start = f
        cur_chars.append(t)
        cur_end = f
    if cur_chars:
        words.append(Word("".join(cur_chars), cur_start * frame_sec, (cur_end + 1) * frame_sec))
    text = " ".join(w.text for w in words)
    return text, words


def cer(ref: str, hyp: str) -> float:
    """Ty le loi ky tu (Levenshtein / do dai ref), bo qua khoang trang."""
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    if not r:
        return 0.0 if not h else 1.0
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev_diag, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev_diag + (r[i - 1] != h[j - 1]))
            prev_diag, d[j] = d[j], cur
    return float(d[len(h)]) / len(r)
