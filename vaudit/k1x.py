# -*- coding: utf-8 -*-
"""K1x - cac dong Constraint con thieu cua K1 (xem VoiceAudit.html, bang Constraint):
  ngon ngu        whisper-small, xac suat ngon ngu o buoc giai ma dau tien                  -> canh bao 'ngon ngu khac'
  su kien am      AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593), cua so 10,24 s   -> 04 'nen che tieng noi' · 04 'khong phai ghi am loi noi' · canh bao 'kieu noi'
  noi chong       pyannote segmentation 3.0 (ONNX, powerset 7 lop), ti le khung >= 2 nguoi / khung co tieng noi -> canh bao 'noi chong'
  RT60 mu         do doc duong cong tat dan nang luong sau cac diem dung cua loi noi        -> canh bao 'phong vang'
  SNR mu          WADA (Kim & Stern 2008; bang tinh bang mo phong) · ket hop voi uoc luong theo ASR cua k1.py
Moi ham nhan song mono float32 16 kHz; mo hinh nap mot lan (lazy). Chi dung CPU."""
from __future__ import annotations

import os

import numpy as np

SR = 16000
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_ONNX = os.path.join(BASE, ".tmp_research", "models", "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
AST_ID, WHISPER_ID = "MIT/ast-finetuned-audioset-10-10-0.4593", "openai/whisper-small"
_CACHE: dict = {}


# ------------------------------------------------------------------ ngon ngu
def _whisper():
    if "whisper" not in _CACHE:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        proc = WhisperProcessor.from_pretrained(WHISPER_ID)
        model = WhisperForConditionalGeneration.from_pretrained(WHISPER_ID).eval()
        from transformers.models.whisper.tokenization_whisper import LANGUAGES
        ids = {c: proc.tokenizer.convert_tokens_to_ids(f"<|{c}|>") for c in LANGUAGES}
        _CACHE["whisper"] = (proc, model, ids)
    return _CACHE["whisper"]


def language_probs(y: np.ndarray, windows: int = 3) -> dict:
    """Xac suat ngon ngu trung binh tren toi da `windows` cua so 30 s (dau, giua, cuoi) -> {'vi': p, 'en': p, 'top': ma, 'p_top': p}."""
    import torch
    proc, model, ids = _whisper()
    W = 30 * SR
    n = len(y)
    starts = [0] if n <= W else sorted({int(s) for s in np.linspace(0, n - W, windows)})
    acc = np.zeros(len(ids))
    codes = list(ids)
    tok = torch.tensor([ids[c] for c in codes])
    for s in starts:
        x = proc.feature_extractor(y[s:s + W].astype(np.float32), sampling_rate=SR, return_tensors="pt").input_features
        dec = torch.tensor([[model.config.decoder_start_token_id]])
        with torch.no_grad():
            lg = model(input_features=x, decoder_input_ids=dec).logits[0, -1]
        acc += torch.softmax(lg[tok], 0).numpy()
    p = acc / len(starts)
    top = int(np.argmax(p))
    return {"vi": float(p[codes.index("vi")]), "en": float(p[codes.index("en")]), "top": codes[top], "p_top": float(p[top])}


# ------------------------------------------------------------------ su kien am thanh (AST)
SPEECH_LABELS = {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", "Conversation",
                 "Narration, monologue", "Babbling", "Speech synthesizer"}
SHOUT_LABELS = {"Shout", "Yell", "Screaming", "Bellow", "Whoop", "Children shouting"}
SING_LABELS = {"Singing", "Choir", "Chant", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping"}
IGNORE_LABELS = {"Silence", "Noise", "White noise", "Pink noise", "Environmental noise", "Static", "Hum", "Buzz", "Inside, small room",
                 "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural", "Reverberation",
                 "Echo", "Mains hum", "Throbbing", "Cacophony", "Sound effect", "Breathing", "Cough", "Throat clearing", "Sigh", "Gasp", "Sniff",
                 "Burping, eructation", "Hiccup", "Chewing, mastication", "Snoring"}


def _ast():
    if "ast" not in _CACHE:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        fx = ASTFeatureExtractor.from_pretrained(AST_ID)
        m = ASTForAudioClassification.from_pretrained(AST_ID).eval()
        names = [m.config.id2label[i] for i in range(len(m.config.id2label))]
        grp = {"speech": [i for i, n in enumerate(names) if n in SPEECH_LABELS],
               "whisper": [i for i, n in enumerate(names) if n == "Whispering"],
               "shout": [i for i, n in enumerate(names) if n in SHOUT_LABELS],
               "sing": [i for i, n in enumerate(names) if n in SING_LABELS],
               "tvradio": [i for i, n in enumerate(names) if n in {"Television", "Radio"}]}
        used = set(sum(grp.values(), [])) | {i for i, n in enumerate(names) if n in IGNORE_LABELS}
        grp["music"] = [i for i, n in enumerate(names) if n in {"Music", "Musical instrument", "Pop music", "Background music", "Theme music",
                                                                 "Jingle (music)", "Soundtrack music"} and i not in grp["sing"]]
        used |= set(grp["music"])
        grp["event"] = [i for i in range(len(names)) if i not in used]
        _CACHE["ast"] = (fx, m, grp)
    return _CACHE["ast"]


def audio_events(y: np.ndarray, max_windows: int = 8) -> dict:
    """AST tren toi da `max_windows` cua so 10,24 s trai deu; moi nhom: max va trung binh (theo cua so) cua xac suat lon nhat trong nhom."""
    import torch
    fx, m, grp = _ast()
    W = int(10.24 * SR)
    n = len(y)
    starts = [0] if n <= W else sorted({int(s) for s in np.linspace(0, n - W, min(max_windows, max(1, int(n / W) + 1)))})
    per = {k: [] for k in grp}
    for s in starts:
        x = fx(y[s:s + W].astype(np.float32), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(m(**x).logits[0]).numpy()
        for k, idx in grp.items():
            per[k].append(float(p[idx].max()) if idx else 0.0)
    out = {}
    for k, v in per.items():
        out[k + "_max"], out[k + "_mean"] = float(np.max(v)), float(np.mean(v))
    return out


# ------------------------------------------------------------------ noi chong (pyannote segmentation 3.0)
def _seg():
    if "seg" not in _CACHE:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.intra_op_num_threads = 2
        _CACHE["seg"] = ort.InferenceSession(SEG_ONNX, so)
    return _CACHE["seg"]


def overlap_share(y: np.ndarray) -> dict:
    """Ti le khung >= 2 nguoi noi dong thoi / khung co tieng noi (cua so 10 s khong chong, lop powerset 4-6 = chong)."""
    sess = _seg()
    W = 10 * SR
    n = len(y)
    pad = (-n) % W
    x = np.pad(y.astype(np.float32), (0, pad)).reshape(-1, 1, W)
    cls = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: x[i:i + 4]})[0].argmax(-1).ravel() for i in range(0, len(x), 4)])
    cls = cls[:int(n / W * 589) + 1] if pad else cls
    speech = cls > 0
    ov = cls >= 4
    return {"noi_chong": float(ov.sum() / max(1, speech.sum())), "khung_noi": int(speech.sum())}


# ------------------------------------------------------------------ RT60 mu
def rt60_blind(y: np.ndarray, hop_s: float = 0.01) -> dict:
    """Uoc luong RT60 tu cac doan tat dan tu do: sau moi diem dung cua loi noi, neu dac trung nang luong (dB) giam >= 15 dB trong 0,12-0,6 s
    thi do doc (hoi quy tuyen tinh) -> RT60 = 60 / (-doc). Lay trung vi cua 30 % so doc thoai nhat (duoi vang ro nhat) de khong bi
    nhung doan dung tu nhien (tat nhanh) keo xuong. Tra rt60=None neu qua it doan tat dan."""
    h = int(hop_s * SR)
    k = len(y) // h
    if k < 100:
        return {"rt60": None, "n_decay": 0}
    e = 10 * np.log10((y[:k * h].astype(np.float64).reshape(k, h) ** 2).mean(1) + 1e-12)
    e = np.convolve(e, np.ones(3) / 3, mode="same")
    floor = np.percentile(e, 5)
    slopes = []
    i = 5
    while i < k - 60:
        if e[i] > floor + 25 and e[i] - e[i + 1] >= 0.0:
            j = i
            while j + 1 < k and e[j + 1] <= e[j] + 0.5 and j - i < 60:
                j += 1
            drop = e[i] - e[j]
            if 12 <= j - i <= 60 and drop >= 15 and e[j] > floor + 3:
                t = np.arange(i, j + 1) * hop_s
                sl = np.polyfit(t, e[i:j + 1], 1)[0]
                if sl < -5:
                    slopes.append(-60.0 / sl)
            i = j + 1
        else:
            i += 1
    if len(slopes) < 3:
        return {"rt60": None, "n_decay": len(slopes)}
    s = np.sort(slopes)
    top = s[int(0.7 * len(s)):]
    return {"rt60": float(np.median(top)), "n_decay": len(slopes), "rt60_median": float(np.median(s))}


# ------------------------------------------------------------------ SNR mu (WADA)
def _wada_table(n: int = 400000, seed: int = 0) -> tuple:
    """g(SNR) = log E|z| - E log|z| voi z = s + n, |s| ~ Gamma(0,4) (dau ngau nhien), n Gaussian; tinh bang mo phong Monte Carlo."""
    rng = np.random.default_rng(seed)
    s = rng.gamma(0.4, 1.0, n) * rng.choice([-1.0, 1.0], n)
    s = s / np.sqrt(np.mean(s ** 2))
    g0 = rng.normal(0, 1, n)
    db = np.arange(-20, 61)
    g = []
    for d in db:
        z = np.abs(s * 10 ** (d / 20) + g0)
        g.append(np.log(z.mean()) - np.log(z + 1e-12).mean())
    return db, np.array(g)


def wada_snr(y: np.ndarray) -> float:
    """SNR mu theo phan bo bien do (WADA): ti so E|z| / exp(E log|z|) tra nguoc ra SNR qua bang mo phong."""
    if "wada" not in _CACHE:
        _CACHE["wada"] = _wada_table()
    db, g = _CACHE["wada"]
    a = np.abs(y.astype(np.float64) / (np.abs(y).max() + 1e-12))
    a = np.maximum(a, 1e-10)
    v = np.log(a.mean()) - np.log(a).mean()
    order = np.argsort(g)
    return float(np.interp(v, g[order], db[order]))


# ------------------------------------------------------------------ nguong + danh gia mot kenh (k1.check(extended=True) goi)
THRESHOLDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k1x_thresholds.json")
DEFAULTS = {"snr_04_below": 10.0, "rt60_warn_above": 1.0, "overlap_warn_above": 0.10, "lang_warn_below": 0.5, "style_p": 0.5, "type_music": 0.6,
            "type_speech_below": 0.25}


def thresholds() -> dict:
    """Nguong da hieu chuan (vaudit/k1x_thresholds.json, do v6/k1x/calibrate.py ghi); thieu thi dung gia tri mac dinh theo bang Constraint."""
    import json
    t = dict(DEFAULTS)
    if os.path.exists(THRESHOLDS_PATH):
        t.update(json.load(open(THRESHOLDS_PATH, encoding="utf-8")))
    return t


def bg_score(ev: dict, nen: dict | None) -> float | None:
    """Diem 'nen che tieng noi' = hoi quy logistic da hieu chuan tren dac trung AST (None neu chua hieu chuan)."""
    if not nen:
        return None
    x = (np.array([ev[f] for f in nen["feats"]]) - np.array(nen["mean"])) / np.array(nen["scale"])
    return float(x @ np.array(nen["coef"]) + nen["bias"])


def assess(y: np.ndarray, lang: bool = True, overlap: bool = True, events: bool = False, rt60: bool = False) -> tuple:
    """Cac dong Constraint con thieu cho MOT kenh mono 16 kHz -> (measures, warnings, reasons). Chi goi sau khi K1 co ban khong ra 04.
    Mac dinh CHI ngon ngu + noi chong (hai bo uoc luong tot, re). SNR / nen / RT60 do vaudit/cond.py (hoc tren XLS-R) thay cho cac uoc luong mu o day;
    events=True: them AST (nhac, kieu noi hat / thi tham / la het, ca 04 'nen che tieng noi' theo AST - yeu, AUC 0,66); rt60=True: RT60 mu theo duong tat dan (yeu khi vang >= 1 s)."""
    T = thresholds()
    m, w, r = {}, [], []
    if lang:
        lp = language_probs(y)
        m["ngon_ngu"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lp.items()}
        if lp["vi"] + lp["en"] < T["lang_warn_below"]:
            w.append(f"ngôn ngữ khác (nhận ra “{lp['top']}”, không phải tiếng Việt / Anh)")
    if events:
        ev = audio_events(y)
        m["su_kien_am"] = {k: round(v, 3) for k, v in ev.items() if k.endswith(("_max", "_mean"))}
        sc = bg_score(ev, T.get("nen"))
        if sc is not None:
            m["diem_nen"] = round(sc, 2)
            if sc > T["nen"]["thr_fpr2"]:
                r.append("04 · “nền che tiếng nói” — nhạc, TV, loa hoặc tiếng động vượt 10 % năng lượng")
        if ev["music_mean"] >= T["type_music"] and ev["speech_mean"] < T["type_speech_below"]:
            r.append("04 · “không phải ghi âm lời nói” — nhạc / bài hát chiếm phần lớn tệp")
        if max(ev["sing_max"], ev["whisper_max"], ev["shout_max"]) >= T["style_p"]:
            w.append("kiểu nói ngoài phạm vi (hát, thì thầm hoặc la hét)")
    if overlap:
        ov = overlap_share(y)
        m["noi_chong_pct"] = round(100 * ov["noi_chong"], 1)
        if ov["noi_chong"] > T["overlap_warn_above"]:
            w.append(f"nói chồng ước lượng {100 * ov['noi_chong']:.0f} % thời lượng có tiếng nói")
    if rt60:
        rt = rt60_blind(y)
        m["rt60_s"] = None if rt["rt60"] is None else round(rt["rt60"], 2)
        if rt["rt60"] is not None and rt["rt60"] > T["rt60_warn_above"]:
            w.append(f"phòng vang — RT60 ước lượng {rt['rt60']:.1f} s")
    m["snr_wada_db"] = round(wada_snr(y), 1)
    return m, w, r
