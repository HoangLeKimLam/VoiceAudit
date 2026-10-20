# -*- coding: utf-8 -*-
"""K1 · Loai am thanh (Constraint C-17) va Kieu noi (C-18) bang mo hinh phan loai am thanh DA HUAN LUYEN SAN, dong bang:
AST AudioSet (MIT/ast-finetuned-audioset-10-10-0.4593, 527 lop, cua so 10,24 s). Khong hoc gi them; chi gop lop AudioSet thanh nhom:
  music    Music, Musical instrument, Song, Background music, ... (the loai nhac)     -> 04 'khong phai ghi am loi noi' khi nhac / bai hat
  sing     Singing, Choir, Chant, Rapping, Humming ...                                    chiem phan lon tep va gan nhu khong co loi noi
  whisper  Whispering                                                                  -> canh bao 'kieu noi ngoai pham vi'
  shout    Shout, Yell, Screaming, Bellow, Children shouting
  tvradio  Television, Radio                    (chi ghi so do: AudioSet khong tach duoc podcast / phat thanh / sach noi voi loi doc)
  synth    Speech synthesizer                   (chi ghi so do: KHONG chan - giong may ca file la thao tac 03, khong phai loi Constraint)
  child    Child speech, kid speaking           (chi ghi so do: 'nguoi lon' - chua co du lieu hieu chuan, de tranh bao nham giong nu)
  speech   Speech, Male / Female speech, Conversation, Narration, monologue
Nguong (TYPE_*, STYLE_*) la gia tri dat tay chua hieu chuan tren du lieu nhac / hat / thi tham THAT (chua co tren may) - xem calib.
  python -m vaudit.events <tep.wav ...>                      in diem nhom va (so do, canh bao, ly do)
  python -m vaudit.events calib --set natvi --limit 40       diem tren tep goc tu nhien hop le -> .tmp_v6/results/k1_events_calib_<set>.md
  python -m vaudit.events real                               doan 30 s that ngoai pham vi (archive.org: nhac co loi / khong loi CC BY / BY-SA,
                                                             sach noi LibriVox PD, phat thanh cu PD, podcast CC BY-SA; nguon + giay phep:
                                                             data_labels/v6/k1axes/events_real/sources.json) -> .tmp_v6/results/k1_events_real.md
                                                             (r5, review 12/10: 16 doan CC NC / ND cua ban truoc danh dau 'khong_dung' - DECISIONS 1
                                                             chi cho CC BY / BY-SA / CC0 / PD - va bi bo qua; thay bang doan CC BY / BY-SA)
  python -m vaudit.events test                               test (tep tong hop + MUSAN noise o data/musan_noise neu co)
Fixture TONG HOP cho Kieu noi (12/10, .tmp_v6/results/wp2_diag/style_exp.py: thi tham / hat / la het tao bang WORLD tu 3 tep VoxPopuli): AST van nghe la
'Speech' (whisper max 0,05-0,26, sing / shout ~0) -> KHONG dung duoc lam ca duong; dong Kieu noi can du lieu that (chua co).
Bai hat THAT co loi (10 doan CC, nay 'khong_dung' vi giay phep NC / ND, do 12/10): nhom 'sing' max trung vi 0,10 (lon nhat 0,43) < STYLE_P 0,5 tren
10/10 doan -> phan 'hat' cua dong Kieu noi KHONG kich hoat ca tren bai hat that (nhac van bi 04 qua nhom music). Bo thay the CC BY / BY-SA (r5):
sing max trung vi 0,07 tren 12 bai hat (12/12 van 04 qua music); nhac KHONG loi chi 3/6 bi 04 (ambient / piano music TB 0,29-0,60) - k1_events_real.md."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

SR = 16000
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
WIN_S = 10.24
MAX_WINDOWS = 8
TYPE_MUSIC_MEAN = 0.60          # nhac (hoac hat) trung binh cac cua so >= 0,6 ...
TYPE_SPEECH_BELOW = 0.25        # ... va tieng noi trung binh < 0,25 -> 04 'khong phai ghi am loi noi'
STYLE_P = 0.50                  # cua so co hat / thi tham / la het >= 0,5 ...
STYLE_SHARE = 0.25              # ... chiem >= 25 % so cua so (tep <= 10 s: 1 cua so) -> canh bao 'kieu noi ngoai pham vi'
GROUP_LABELS = {
    "speech": {"Speech", "Male speech, man speaking", "Female speech, woman speaking", "Conversation", "Narration, monologue", "Babbling"},
    "child": {"Child speech, kid speaking"},
    "synth": {"Speech synthesizer"},
    "whisper": {"Whispering"},
    "shout": {"Shout", "Yell", "Screaming", "Bellow", "Children shouting", "Battle cry"},
    "sing": {"Singing", "Choir", "Chant", "Mantra", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping",
             "Humming", "Yodeling", "Lullaby"},
    "tvradio": {"Television", "Radio"},
    "music": {"Music", "Musical instrument", "Song", "Pop music", "Rock music", "Hip hop music", "Electronic music", "Classical music",
              "Background music", "Theme music", "Jingle (music)", "Soundtrack music", "Video game music", "Christmas music", "Dance music",
              "Wedding music", "Happy music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music", "Music of Asia",
              "Vocal music", "A capella", "Music for children", "Folk music", "Jazz", "Guitar", "Piano", "Drum"},
}
_CACHE: dict = {}


def model_dir() -> str | None:
    """Thu muc snapshot AST trong cache Hugging Face cuc bo (HF_HOME, mac dinh D:/hf-cache); None neu chua tai (khong goi mang)."""
    roots = [os.environ.get("HF_HOME", ""), "D:/hf-cache"]
    for r in [x for x in roots if x]:
        for d in glob.glob(os.path.join(r, "hub", "models--" + AST_ID.replace("/", "--"), "snapshots", "*")):
            if os.path.exists(os.path.join(d, "config.json")) and glob.glob(os.path.join(d, "model.safetensors")) + glob.glob(os.path.join(d, "pytorch_model.bin")):
                return d
    return None


def available() -> bool:
    """Co mo hinh AST cuc bo VA khong bi tat (VA_K1_EVENTS=0) -> K1 bat dong Loai am thanh / Kieu noi."""
    return os.environ.get("VA_K1_EVENTS", "1") != "0" and model_dir() is not None


def _ast():
    if "ast" not in _CACHE:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        d = model_dir()
        if d is None:
            raise FileNotFoundError(f"chua co {AST_ID} trong cache cuc bo (xem queue job tai mo hinh)")
        fx = ASTFeatureExtractor.from_pretrained(d, local_files_only=True)
        m = ASTForAudioClassification.from_pretrained(d, local_files_only=True).eval()
        names = [m.config.id2label[i] for i in range(len(m.config.id2label))]
        grp = {g: [i for i, n in enumerate(names) if n in labs] for g, labs in GROUP_LABELS.items()}
        _CACHE["ast"] = (fx, m, grp)
    return _CACHE["ast"]


def window_starts(n: int, max_windows: int = MAX_WINDOWS) -> list:
    """Vi tri bat dau cac cua so 10,24 s trai deu (tep ngan hon: mot cua so tu 0)."""
    W = int(WIN_S * SR)
    if n <= W:
        return [0]
    k = min(max_windows, int(np.ceil(n / W)))
    return sorted({int(s) for s in np.linspace(0, n - W, k)})


def scores(y: np.ndarray, max_windows: int = MAX_WINDOWS) -> dict:
    """Xac suat (sigmoid) lon nhat trong moi nhom, theo tung cua so -> {nhom: [p cua so 1, ...]}."""
    import torch
    fx, m, grp = _ast()
    W = int(WIN_S * SR)
    y = np.nan_to_num(np.asarray(y, np.float32))
    per = {g: [] for g in grp}
    for s in window_starts(len(y), max_windows):
        x = fx(y[s:s + W], sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            p = torch.sigmoid(m(**x).logits[0]).numpy()
        for g, idx in grp.items():
            per[g].append(float(p[idx].max()) if idx else 0.0)
    return per


def decide(per: dict) -> tuple:
    """Luat tu diem theo cua so -> (measures, warnings, reasons) - tach rieng de test khong can mo hinh."""
    from .k1 import WARN, reason
    mean = {g: float(np.mean(v)) for g, v in per.items()}
    mx = {g: float(np.max(v)) for g, v in per.items()}
    nwin = len(next(iter(per.values())))
    style = np.max([per["sing"], per["whisper"], per["shout"]], axis=0)
    share = float(np.mean(style >= STYLE_P))
    M = {"su_kien_am": {f"{g}_mean": round(mean[g], 3) for g in per} | {f"{g}_max": round(mx[g], 3) for g in per},
         "su_kien_so_cua_so": nwin, "kieu_noi_ti_le_cua_so": round(share, 3)}
    W, R = [], []
    nonspeech = max(mean["music"], mean["sing"])
    if nonspeech >= TYPE_MUSIC_MEAN and mean["speech"] < TYPE_SPEECH_BELOW:
        R.append(reason("khong_phai_loi_noi", "nhạc / bài hát chiếm phần lớn tệp (mô hình AST AudioSet)"))
    elif share >= STYLE_SHARE:
        kinds = [k for k, lab in (("sing", "hát"), ("whisper", "thì thầm"), ("shout", "la hét")) if mx[k] >= STYLE_P]
        W.append(WARN["kieu_noi"])
        M["kieu_noi_chi_tiet"] = ", ".join(lab for k, lab in (("sing", "hát"), ("whisper", "thì thầm"), ("shout", "la hét")) if k in kinds)
    return M, W, R


def assess(y: np.ndarray) -> tuple:
    """(measures, warnings, reasons) cho MOT kenh mono 16 kHz (cung dang k1.analyse_channel)."""
    return decide(scores(y))


# ------------------------------------------------------------------ hieu chuan tren tep that hop le (nhe: <= vai chuc tep)
def calib(set_name: str, limit: int) -> None:
    import json

    import soundfile as sf
    man = json.load(open(os.path.join(BASE, "data_labels", "v6", set_name, "manifest.json"), encoding="utf-8"))
    ok = sorted([e for e in man if e["kind"] == "nguyen_ban" and e["cfg"] == "goc" and not e["meta"].get("lockbox")
                 and e["meta"].get("k1", {}).get("decision") in ("nhan", "canh_bao")], key=lambda e: e["id"])
    step = max(1, len(ok) // max(1, limit))
    rows = []
    for e in ok[::step][:limit]:
        y, sr = sf.read(os.path.join(BASE, "data_labels", "realistic_v4", e["wav"]), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        if sr != SR:
            continue
        per = scores(y)
        M, W, R = decide(per)
        rows.append((e["id"], M["su_kien_am"], W, R))
    n = len(rows)
    if not n:                                   # khong co tep goc hop le 16 kHz (sua 11/10: tung sap o np.quantile tren mang rong)
        print(f"{set_name}: 0 tep goc hop le de hieu chinh - bo qua", flush=True)
        return
    L = [f"# K1 su kien am (AST) tren {n} tep goc hop le cua {set_name}\n",
         f"- 04 'khong phai ghi am loi noi': {sum(bool(r[3]) for r in rows)}/{n}",
         f"- canh bao 'kieu noi ngoai pham vi': {sum(bool(r[2]) for r in rows)}/{n}\n", "| nhom | p50 mean | p95 mean | p95 max | max max |", "|---|---|---|---|---|"]
    for g in GROUP_LABELS:
        a = np.array([r[1][f"{g}_mean"] for r in rows])
        b = np.array([r[1][f"{g}_max"] for r in rows])
        L.append(f"| {g} | {np.median(a):.3f} | {np.quantile(a, .95):.3f} | {np.quantile(b, .95):.3f} | {b.max():.3f} |")
    L += ["", "Tep bi bao: " + ", ".join(r[0] for r in rows if r[2] or r[3])]
    out = os.path.join(BASE, ".tmp_v6", "results", f"k1_events_calib_{set_name}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


REAL = os.path.join(BASE, "data_labels", "v6", "k1axes", "events_real")
REAL_EXPECT = {"sing": "04", "music": "04", "book": "04*", "radio": "04*", "podcast": "04*"}     # * hop dong 04, AST khong tach duoc voi loi noi
ALLOWED_LICENSE = ("creativecommons.org/licenses/by/", "creativecommons.org/licenses/by-sa/", "creativecommons.org/publicdomain/",
                   "creativecommons.org/licenses/publicdomain")    # DECISIONS.md (1): archive.org chi CC BY / BY-SA / CC0 / PD


def license_ok(url: str) -> bool:
    """Giay phep duoc phep theo quyet dinh nguoi dung (1): CC BY, CC BY-SA, CC0 / dau PD; NC / ND -> khong."""
    u = str(url or "").lower()
    return any(a in u for a in ALLOWED_LICENSE) and not any(x in u for x in ("-nc", "-nd", "/nc", "/nd"))


def usable(doc: dict) -> list:
    """Muc sources.json DUOC dung: khong danh dau 'khong_dung' VA giay phep trong danh sach cho phep (kiem lai, khong tin co trong tep)."""
    return [s for s in doc["files"] if not s.get("khong_dung") and license_ok(s.get("license"))]


def real(root: str = REAL, max_windows: int = 3) -> dict:
    """Doan that ngoai pham vi (sources.json): ti le 04 'khong phai ghi am loi noi' / canh bao kieu noi theo nhom + diem nhom AST."""
    import json

    import soundfile as sf
    doc = json.load(open(os.path.join(root, "sources.json"), encoding="utf-8"))
    rows = []
    for s in usable(doc):
        y, sr = sf.read(os.path.join(root, s["file"]), dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        if sr != SR:
            continue
        M, W, R = decide(scores(y, max_windows))
        rows.append({"file": s["file"], "cat": s["cat"], "license": s["license"], "R": R, "W": W, "s": M["su_kien_am"]})
    n_skip = len(doc["files"]) - len(usable(doc))
    L = ["# K1 su kien am (AST) tren doan THAT ngoai pham vi (archive.org, 30 s / tep)\n",
         f"Doan dung: {len(rows)}; bo qua {n_skip} doan 'khong_dung' / giay phep ngoai CC BY / BY-SA / CC0 / PD (DECISIONS 1).\n",
         f"Nguong: nhac / hat TB >= {TYPE_MUSIC_MEAN} va tieng noi TB < {TYPE_SPEECH_BELOW} -> 04; kieu noi: cua so >= {STYLE_P} chiem >= {STYLE_SHARE}.\n",
         "| nhom | n | du kien | 04 'không phải ghi âm lời nói' | canh bao kieu noi | music TB (trung vi) | sing max (trung vi) | speech TB (trung vi) | tvradio max (trung vi) |",
         "|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for cat in REAL_EXPECT:
        rs = [r for r in rows if r["cat"] == cat]
        if not rs:
            continue
        med = lambda k: float(np.median([r["s"][k] for r in rs]))       # noqa: E731
        out[cat] = {"n": len(rs), "r04": sum(bool(r["R"]) for r in rs) / len(rs), "warn": sum(bool(r["W"]) for r in rs) / len(rs)}
        L.append(f"| {cat} | {len(rs)} | {REAL_EXPECT[cat]} | {sum(bool(r['R']) for r in rs)}/{len(rs)} | {sum(bool(r['W']) for r in rs)}/{len(rs)} | "
                 f"{med('music_mean'):.2f} | {med('sing_max'):.2f} | {med('speech_mean'):.2f} | {med('tvradio_max'):.2f} |")
    L += ["", "| tep | nhom | 04 | canh bao | music TB | sing max | speech TB |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['file'][:60]} | {r['cat']} | {'x' if r['R'] else ''} | {'x' if r['W'] else ''} | {r['s']['music_mean']:.2f} | {r['s']['sing_max']:.2f} | "
          f"{r['s']['speech_mean']:.2f} |" for r in rows]
    open(os.path.join(BASE, ".tmp_v6", "results", "k1_events_real.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    return out


# ------------------------------------------------------------------ test
def test_decide_rules() -> None:
    from .k1 import CONTRACT_WARNINGS

    def per(n: int = 1, **kw) -> dict:
        return {g: list(kw.get(g, [0.02] * n)) for g in GROUP_LABELS}
    M, W, R = decide(per(speech=[0.9]))
    assert not W and not R and M["su_kien_so_cua_so"] == 1
    M, W, R = decide(per(2, music=[0.9, 0.8], speech=[0.1, 0.05]))
    assert R and "“không phải ghi âm lời nói”" in R[0] and not W
    M, W, R = decide(per(music=[0.9], speech=[0.8]))            # nhac nen + co loi noi: khong phai loi dong nay (cond 'nen')
    assert not R
    M, W, R = decide(per(4, speech=[0.9] * 4, whisper=[0.7, 0.1, 0.1, 0.1]))
    assert W == ["kiểu nói ngoài phạm vi"] and W[0] in CONTRACT_WARNINGS and M["kieu_noi_chi_tiet"] == "thì thầm"
    M, W, R = decide(per(8, speech=[0.9] * 8, shout=[0.9] + [0.0] * 7))   # 1/8 cua so la het: khong canh bao
    assert not W
    M, W, R = decide(per(speech=[0.9], synth=[0.99]))            # giong may: khong chan o K1
    assert not W and not R


def test_license_filter() -> None:
    """DECISIONS 1: chi CC BY / BY-SA / CC0 / PD; NC / ND hoac 'khong_dung' -> bo (review 12/10: 16/26 doan cu la NC / ND)."""
    ok = ["http://creativecommons.org/licenses/by/3.0/", "http://creativecommons.org/licenses/by-sa/4.0/", "http://creativecommons.org/publicdomain/zero/1.0/",
          "http://creativecommons.org/publicdomain/mark/1.0/", "http://creativecommons.org/licenses/publicdomain/"]
    bad = ["http://creativecommons.org/licenses/by-nc/3.0/", "http://creativecommons.org/licenses/by-nc-nd/3.0/", "http://creativecommons.org/licenses/by-nd/2.0/uk/",
           "http://creativecommons.org/licenses/by-nc-sa/2.5/", "", None]
    assert all(license_ok(u) for u in ok) and not any(license_ok(u) for u in bad)
    doc = {"files": [{"license": ok[0]}, {"license": ok[1], "khong_dung": True}, {"license": bad[0]}]}
    assert usable(doc) == [doc["files"][0]]
    if os.path.exists(os.path.join(REAL, "sources.json")):                  # tep that: moi doan dung co giay phep cho phep
        import json
        d = json.load(open(os.path.join(REAL, "sources.json"), encoding="utf-8"))
        assert all(license_ok(s["license"]) for s in usable(d)) and all(s.get("khong_dung") for s in d["files"] if not license_ok(s.get("license")))


def test_window_starts() -> None:
    assert window_starts(5 * SR) == [0]
    s = window_starts(600 * SR)
    assert len(s) == MAX_WINDOWS and s[0] == 0 and s[-1] == 600 * SR - int(WIN_S * SR)
    assert len(window_starts(25 * SR)) == 3


def test_model_on_signals() -> None:
    """Can mo hinh cuc bo (bo qua neu chua co). Tieng noi that tu nhien: khong 04, it kieu noi; MUSAN noise: diem tieng noi thap hon."""
    if not available():
        print("    (bo qua: chua co mo hinh AST cuc bo)")
        return
    import json

    import soundfile as sf
    man = json.load(open(os.path.join(BASE, "data_labels", "v6", "naten", "manifest.json"), encoding="utf-8"))
    sp = [e for e in man if e["kind"] == "nguyen_ban" and e["cfg"] == "goc" and e["meta"].get("k1", {}).get("decision") in ("nhan", "canh_bao")][:3]
    sp_scores = []
    for e in sp:
        y, _ = sf.read(os.path.join(BASE, "data_labels", "realistic_v4", e["wav"]), dtype="float32")
        per = scores(y, max_windows=2)
        M, W, R = decide(per)
        assert not R, (e["id"], M)
        sp_scores.append(np.mean(per["speech"]))
    noise = sorted(glob.glob(os.path.join(BASE, "data", "musan_noise", "noise", "free-sound", "*.wav")))[:4]
    nz = []
    for f in noise:
        y, sr = sf.read(f, dtype="float32")
        y = y.mean(1) if y.ndim > 1 else y
        if sr != SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        per = scores(y[:int(20 * SR)], max_windows=2)
        nz.append(np.mean(per["speech"]))
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(5 * SR) / SR).astype(np.float32)
    tone_speech = np.mean(scores(tone)["speech"])
    print(f"    speech mean: loi noi that {np.round(sp_scores, 2)} | MUSAN noise {np.round(nz, 2)} | tone 440 Hz {tone_speech:.2f}")
    assert min(sp_scores) > 0.5 and (not nz or np.median(nz) < min(sp_scores)) and tone_speech < 0.5


def test_real_out_of_scope_clips() -> None:
    """Doan that (archive.org, chi doan giay phep CC BY / BY-SA / CC0 / PD - usable()): nhac co / khong loi -> 04 'khong phai ghi am loi noi'
    >= 80 % (r5 12/10, bo thay the: bai hat 12/12, nhac khong loi 3/6 -> 15/18 voi 3 cua so nhu K1; bo cu NC / ND: 15/16;
    1 cua so kem hon - quyet dinh theo trung binh cua so);
    sach noi -> KHONG 04 (AST khong tach sach noi voi loi noi - han che ghi trong not_impl). Can mo hinh + du lieu (bo qua neu thieu)."""
    if not available() or not os.path.exists(os.path.join(REAL, "sources.json")):
        print("    (bo qua: chua co mo hinh AST hoac doan that)")
        return
    import json

    import soundfile as sf
    doc = json.load(open(os.path.join(REAL, "sources.json"), encoding="utf-8"))
    hit, n_mus, book04 = 0, 0, 0
    for s in usable(doc):
        if s["cat"] not in ("sing", "music", "book"):
            continue
        y, _ = sf.read(os.path.join(REAL, s["file"]), dtype="float32")
        M, W, R = decide(scores(y))                    # so cua so nhu K1 that (30 s -> 3)
        if s["cat"] == "book":
            book04 += bool(R)
        else:
            n_mus += 1
            hit += bool(R) and "“không phải ghi âm lời nói”" in R[0]
    print(f"    nhac 04: {hit}/{n_mus} | sach noi 04: {book04}")
    assert n_mus and hit / n_mus >= 0.8 and book04 == 0, (hit, n_mus, book04)


def run_tests() -> None:
    import time
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    fails = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"  ok   {fn.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:      # noqa: BLE001  bao loi, chay tiep
            fails += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(tests) - fails}/{len(tests)} test dat")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if BASE not in sys.path:
        sys.path.insert(0, BASE)
    a = sys.argv[1:]
    if a and a[0] == "test":
        run_tests()
    elif a and a[0] == "calib":
        st = a[a.index("--set") + 1] if "--set" in a else "natvi"
        lim = int(a[a.index("--limit") + 1]) if "--limit" in a else 40
        calib(st, lim)
    elif a and a[0] == "real":
        real()
    else:
        import soundfile as sf
        for f in a:
            y, sr = sf.read(f, dtype="float32")
            y = y.mean(1) if y.ndim > 1 else y
            print(f, decide(scores(y)))
