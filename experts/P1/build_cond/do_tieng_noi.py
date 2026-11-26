"""Bộ đo cho bản sửa cách đếm thời lượng lời nói của K1 (hàng cứng «quá ít tiếng nói»: ≥ 2 s VÀ ≥ 30 % độ dài). CHƯA sửa vaudit/k1.py.

Vấn đề (NHAT_KY_P1 2.14, 22/11): K1 đếm lời nói = khung trong ±0,1 s quanh ký tự CTC (wav2vec2 tiếng Việt). Ký tự thưa (nói nhanh,
ồn, tiếng Anh) → khoảng giữa bị coi là im lặng → 33/927 đoạn vi thật bị chặn oan, en 2 ca.
Ứng viên: gộp ký tự cách nhau < `gap` giây thành một đoạn lời rồi mới đếm (gap = 0 ≡ cách hiện tại). Ngưỡng hàng cứng GIỮ NGUYÊN.

Chọn gap không tự chấm:
  - ghi âm thật p1_tunhien_v1 (đáp án accept), chia theo kênh: fold = crc32(person) % 5; CHỌN trên fold 0–2, BÁO trên fold 3–4;
  - nhãn nghe (Gemini, gemini/KQ_TY_LE_NOI*_2211.jsonl) để biết ca bị chặn nào thật sự đủ lời;
  - ca ÂM TÍNH tự dựng trong bộ nhớ (không ghi tệp): chỉ ồn MUSAN, âm thuần, gần im lặng, lời sạch 1,2 s / 20 % độ dài → PHẢI vẫn bị chặn;
    lời sạch 45 % → phải qua. Lời sạch lấy từ p1_dieukien_v1/goc (FLEURS, SNR sẵn có ≥ 25 dB).
Dùng:  .venv/Scripts/python.exe experts/P1/build_cond/do_tieng_noi.py cache | eval | selftest
"""
from __future__ import annotations

import json
import os
import random
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dung_bo import ROOT, SR, musan_pool  # noqa: E402

sys.path.insert(0, ROOT)
BO = os.path.join(ROOT, "data_labels", "v6", "p1_tunhien_v1")
COND = os.path.join(ROOT, "data_labels", "v6", "p1_dieukien_v1")
CACHE = os.path.join(BO, "_kiem", "ctc_token.jsonl")
WIN, PAD = 0.03, 0.10                       # như vaudit/k1.asr_speech_mask
GAPS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
MIN_S, MIN_SHARE = 2.0, 0.30                # hàng cứng, không đổi


def speech_s(t_tok: list[float], dur: float, gap: float) -> float:
    """Giây lời nói: khung 30 ms trong ±PAD quanh ký tự, cộng khoảng giữa hai ký tự liền nhau cách < gap (gap = 0: cách hiện tại)."""
    n = int(dur / WIN)
    if not t_tok or n == 0:
        return 0.0
    tc = (np.arange(n) + 0.5) * WIN
    t = np.asarray(sorted(t_tok))
    j = np.searchsorted(t, tc)
    prev, nxt = t[np.clip(j - 1, 0, len(t) - 1)], t[np.clip(j, 0, len(t) - 1)]
    near = np.minimum(np.abs(tc - prev), np.abs(tc - nxt)) <= PAD
    inside = (j > 0) & (j < len(t)) & (nxt - prev < gap)
    return float((near | inside).sum() * WIN)


def blocked(sp: float, dur: float) -> bool:
    sp = round(sp, 1)                       # như K1: m["tieng_noi_s"] = round(sp_s, 1) rồi mới so ngưỡng
    return sp < MIN_S or sp < MIN_SHARE * dur


def tokens(y: np.ndarray) -> list[float]:
    """Mốc thời gian (s) các khung CTC khác 'trống', đúng như vaudit.k1.asr_speech_mask."""
    import torch

    from vaudit.asr import _load
    from vaudit.features import LONG_S, chunk_frames
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
    return [round(float(t), 3) for t in (np.where(lg.argmax(1) != blank)[0] + 0.5) * dt]


# ----------------------------------------------------------------------------- ca âm tính / dương tính tự dựng (trong bộ nhớ)
def synth_cases(seed: int = 0) -> list[dict]:
    import soundfile as sf
    rng, nr = random.Random(seed), np.random.default_rng(seed)
    goc = sorted(f for f in os.listdir(os.path.join(COND, "goc")) if f.endswith(".wav"))
    noise = musan_pool("noise", 10.0)
    out = []
    for i in range(8):
        nz, _ = sf.read(os.path.join(ROOT, rng.choice(noise)["path"]), dtype="float32", frames=10 * SR)
        out.append({"ten": f"chi_on_{i}", "y": 0.1 * nz / (np.abs(nz).max() + 1e-9), "phai_chan": True})
    out.append({"ten": "am_thuan", "y": 0.3 * np.sin(2 * np.pi * 1000 * np.arange(10 * SR) / SR).astype(np.float32), "phai_chan": True})
    out.append({"ten": "gan_im_lang", "y": nr.normal(0, 3e-4, 10 * SR).astype(np.float32), "phai_chan": True})
    for i, f in enumerate(goc[:8]):
        s, _ = sf.read(os.path.join(COND, "goc", f), dtype="float32")
        for ten, keep_s, total_s, must in (("loi_1.2s", 1.2, 10.0, True), ("loi_20pct", 2.4, 12.0, True), ("loi_45pct", 4.5, 10.0, False)):
            if len(s) < keep_s * SR:
                continue
            y = nr.normal(0, 1e-3, int(total_s * SR)).astype(np.float32)
            a = rng.randrange(0, len(y) - int(keep_s * SR))
            y[a:a + int(keep_s * SR)] += s[:int(keep_s * SR)]
            out.append({"ten": f"{ten}_{i}", "y": y, "phai_chan": must})
    return out


# ----------------------------------------------------------------------------- lệnh
def cache() -> None:
    import soundfile as sf
    cases = json.load(open(os.path.join(BO, "cases.json"), encoding="utf-8"))
    done = {json.loads(x)["file"] for x in open(CACHE, encoding="utf-8")} if os.path.exists(CACHE) else set()
    todo = [c for c in cases if c["file"] not in done]
    for i, c in enumerate(todo):
        y, _ = sf.read(os.path.join(BO, c["file"]), dtype="float32")
        row = {"file": c["file"], "dur": round(len(y) / SR, 3), "t": tokens(y)}
        with open(CACHE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{i + 1}/{len(todo)} {c['file']}", flush=True)


def gemini_labels() -> dict:
    lab = {}
    for f in ("KQ_TY_LE_NOI_2211.jsonl", "KQ_TY_LE_NOI_VI_2211.jsonl"):
        p = os.path.join("F:/CODING/Voiceaudit_ai002/gemini", f)
        for x in open(p, encoding="utf-8") if os.path.exists(p) else []:
            d = json.loads(x)
            if "tep" in d:
                lab[d["tep"].split("p1_tunhien_v1/")[1]] = bool(d["du_30pct"] and d["du_2s"])
    return lab


def evaluate(show_test: bool = False) -> str:
    """show_test=False: cột fold 3–4 bị ẩn → chọn gap chỉ nhìn dev; `eval --test` sau khi đã ghi gap chọn vào NHAT_KY."""
    from chay_k1 import wilson
    cases = {c["file"]: c for c in json.load(open(os.path.join(BO, "cases.json"), encoding="utf-8"))}
    rows = [r for r in (json.loads(x) for x in open(CACHE, encoding="utf-8")) if r["file"] in cases and cases[r["file"]]["mong_doi"] == "accept"]
    fold = lambda r: zlib.crc32(cases[r["file"]]["base"].encode()) % 5
    lab = gemini_labels()
    k1r = {r["file"]: r for r in (json.loads(x) for x in open(os.path.join(BO, "_kiem", "k1_ket_qua.jsonl"), encoding="utf-8"))}
    one = [r for r in rows if r["file"] in k1r and len(k1r[r["file"]]["measures"].get("kenh", [])) == 1]
    # gap = 0 phải tái tạo QUYẾT ĐỊNH «quá ít tiếng nói» của K1 trên mọi tệp (số giây lệch < 0,3 % ở tệp dài do làm tròn số khung)
    lech = [r["file"] for r in one if blocked(speech_s(r["t"], r["dur"], 0.0), r["dur"])
            != any("quá ít tiếng nói" in x for x in k1r[r["file"]].get("reasons", []))]
    if not one or lech:
        raise SystemExit(f"gap=0 không tái tạo quyết định K1 trên {len(lech)}/{len(one)} tệp: {lech[:5]}")
    syn =[{**s, "t": tokens(s["y"]), "dur": len(s["y"]) / SR} for s in synth_cases()]
    L = ["| gap (s) | chặn «ít tiếng nói», dev fold 0–2 | test fold 3–4 | nhãn nghe: chặn mà đủ lời / chặn đúng mà bị thả | ca tự dựng sai |", "|---|---|---|---|---|"]
    for g in GAPS:
        b = {r["file"]: blocked(speech_s(r["t"], r["dur"], g), r["dur"]) for r in rows}
        dev = [f for f in b if fold(next(r for r in rows if r["file"] == f)) < 3]
        tst = [f for f in b if f not in set(dev)]
        cell = lambda fs: f"{sum(b[f] for f in fs)}/{len(fs)} = {100 * sum(b[f] for f in fs) / max(1, len(fs)):.1f} % " \
                          f"(Wilson {100 * wilson(sum(b[f] for f in fs), len(fs))[1]:.1f} % trên)"
        lf = set(dev) if not show_test else set(b)                     # nhãn nghe: chỉ fold dev khi đang chọn
        oan = sum(1 for f, ok in lab.items() if f in lf and ok and b[f])
        tha = sum(1 for f, ok in lab.items() if f in lf and not ok and not b[f])
        bad = [s["ten"] for s in syn if blocked(speech_s(s["t"], s["dur"], g), s["dur"]) != s["phai_chan"]]
        L.append(f"| {g} | {cell(dev)} | {cell(tst) if show_test else '(ẩn)'} | {oan} / {tha} (n nhãn {len(lab)}) | {len(bad)}/{len(syn)} {bad[:4]} |")
    out = ("# Thử gộp ký tự CTC cho «quá ít tiếng nói»\n\nCHỌN gap trên cột dev; cột test chỉ đọc sau khi chọn. "
           f"Kiểm tái tạo: gap = 0 cho đúng quyết định «quá ít tiếng nói» của K1 trên {len(one)}/{len(one)} tệp.\n\n" + "\n".join(L) + "\n")
    open(os.path.join(BO, "_kiem", "DO_TIENG_NOI.md"), "w", encoding="utf-8").write(out)
    return out


# ----------------------------------------------------------------------------- bộ dò bổ sung (23/11, Huấn chốt: chỉ dùng thứ có sẵn trên máy)
# Mọi ứng viên = khung CTC (như K1) HỢP với một mặt nạ khác → số giây lời chỉ TĂNG → chỉ đổi được tệp «chặn» sang «nhận».
# Nên chỉ cần tính trên tệp đang bị chặn + tệp có nhãn nghe + ca tự dựng (tệp đang nhận chắc chắn vẫn nhận).
AST_WIN = 2.0
AST_THR = (0.3, 0.5, 0.7)


def masks(y: np.ndarray, t_tok: list[float]) -> dict:
    """Mặt nạ khung 30 ms: ctc (như K1), nang_luong (vaudit.k1.speech_mask), webrtc (mode 2), ast_<thr> (lớp lời nói AST theo cửa sổ 2 s)."""
    import torch
    import webrtcvad

    from vaudit import k1, k1x
    db = k1.frame_db(y)
    n = len(db)
    tc = (np.arange(n) + 0.5) * WIN
    t = np.asarray(sorted(t_tok)) if t_tok else np.array([1e9])
    j = np.searchsorted(t, tc)
    ctc = np.minimum(np.abs(tc - t[np.clip(j - 1, 0, len(t) - 1)]), np.abs(tc - t[np.clip(j, 0, len(t) - 1)])) <= PAD
    ene = k1.speech_mask(db)[0]
    v, fn = webrtcvad.Vad(2), int(WIN * SR)
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    web = np.array([v.is_speech(pcm[i * fn:(i + 1) * fn].tobytes(), SR) for i in range(n)])
    fx, m, grp = k1x._ast()
    w = int(AST_WIN * SR)
    p = []
    for s in range(0, max(1, len(y) - w // 2), w):
        x = fx(y[s:s + w].astype(np.float32), sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            p.append(float(torch.sigmoid(m(**x).logits[0]).numpy()[grp["speech"]].max()))
    pw = np.asarray(p)[np.clip((tc // AST_WIN).astype(int), 0, len(p) - 1)]      # khung → cửa sổ AST chứa nó (không trôi ở tệp dài)
    out = {"ctc": ctc, "ctc|nang_luong": ctc | ene, "ctc|webrtc&nang_luong": ctc | (web & ene)}
    for th in AST_THR:
        out[f"ctc|nang_luong&ast>{th}"] = ctc | (ene & (pw > th))
        out[f"ctc|webrtc&ast>{th}"] = ctc | (web & (pw > th))
    return out


def alt() -> str:
    import soundfile as sf
    cases = {c["file"]: c for c in json.load(open(os.path.join(BO, "cases.json"), encoding="utf-8"))}
    rows = {r["file"]: r for r in (json.loads(x) for x in open(CACHE, encoding="utf-8"))
            if r["file"] in cases and cases[r["file"]]["mong_doi"] == "accept"}
    dev = {f for f in rows if zlib.crc32(cases[f]["base"].encode()) % 5 < 3}
    lab = {f: ok for f, ok in gemini_labels().items() if f in dev}
    pick = sorted(f for f in dev if blocked(speech_s(rows[f]["t"], rows[f]["dur"], 0.0), rows[f]["dur"]) or f in lab)
    score = {}
    for f in pick:
        y, _ = sf.read(os.path.join(BO, f), dtype="float32")
        score[f] = {k: blocked(mk.sum() * WIN, len(y) / SR) for k, mk in masks(y, rows[f]["t"]).items()}
    syn = {}
    for s in synth_cases():
        syn[s["ten"]] = ({k: blocked(mk.sum() * WIN, len(s["y"]) / SR) for k, mk in masks(s["y"], tokens(s["y"])).items()}, s["phai_chan"])
    keys = list(next(iter(score.values())).keys())
    L = ["| bộ dò | chặn «ít tiếng nói», dev | nhãn nghe dev: chặn oan / chặn đúng bị thả | ca tự dựng sai |", "|---|---|---|---|"]
    for k in keys:
        nb = sum(score[f][k] for f in pick if f in dev)
        oan = sum(1 for f, ok in lab.items() if ok and score[f][k])
        tha = sum(1 for f, ok in lab.items() if not ok and not score[f][k])
        bad = [t for t, (d, must) in syn.items() if d[k] != must]
        L.append(f"| `{k}` | {nb}/{len(dev)} = {100 * nb / len(dev):.1f} % | {oan} / {tha} (n {len(lab)}) | {len(bad)}/{len(syn)} {bad[:3]} |")
    out = ("# Bộ dò lời nói bổ sung cho «quá ít tiếng nói» (chỉ fold dev 0–2)\n\n"
           "Mọi ứng viên = khung CTC của K1 HỢP mặt nạ khác, nên chỉ thả bớt tệp bị chặn; tệp đang nhận không đổi.\n\n" + "\n".join(L) + "\n")
    open(os.path.join(BO, "_kiem", "DO_TIENG_NOI_ALT.md"), "w", encoding="utf-8").write(out)
    return out


CHON = "ctc|nang_luong&ast>0.3"                       # Huấn duyệt 23/11 trên dev (NHAT_KY 2.14), TRƯỚC khi mở test
HANG_CUNG = ("“quá nhỏ”", "“băng hẹp”", "“cắt đỉnh nặng”")   # vi phạm thật, đo độc lập xác nhận → báo riêng (DE_XUAT 2111)
IT_NOI = "“quá ít tiếng nói”"


def test() -> str:
    """Mở fold test 3–4 cho đúng 1 bộ dò đã chọn: tỉ lệ K1 ra 04 trên tệp accept, trước / sau khi thay cách đo tiếng nói."""
    import soundfile as sf

    from chay_k1 import wilson
    cases = {c["file"]: c for c in json.load(open(os.path.join(BO, "cases.json"), encoding="utf-8"))}
    k1r = {r["file"]: r for r in (json.loads(x) for x in open(os.path.join(BO, "_kiem", "k1_ket_qua.jsonl"), encoding="utf-8"))
           if r["file"] in cases and cases[r["file"]]["mong_doi"] == "accept"}
    lab = gemini_labels()
    tok = {r["file"]: r["t"] for r in (json.loads(x) for x in open(CACHE, encoding="utf-8"))}
    reasons = {f:[s for s in r.get("reasons", []) if s.startswith("04")] for f, r in k1r.items()}
    new = {}                                                    # tệp có «ít tiếng nói»: đo lại bằng bộ dò đã chọn
    for f in sorted(f for f, rs in reasons.items() if any(IT_NOI in s for s in rs) or f in lab):
        y, _ = sf.read(os.path.join(BO, f), dtype="float32")
        if y.ndim > 1:
            raise SystemExit(f"{f}: nhiều kênh, bộ dò chưa hỗ trợ")
        new[f] = blocked(masks(y, tok[f])[CHON].sum() * WIN, len(y) / SR)
    def chan(f, moi, bo_cung):
        rs = [s for s in reasons[f] if not (bo_cung and any(h in s for h in HANG_CUNG))]
        khac = [s for s in rs if IT_NOI not in s]
        return bool(khac) or (any(IT_NOI in s for s in rs) and (new[f] if moi else True))
    fold = lambda f: "dev" if zlib.crc32(cases[f]["base"].encode()) % 5 < 3 else "test"
    cell = lambda fs, **kw: (lambda k, n: f"{k}/{n} = {100 * k / max(1, n):.1f} % ({100 * wilson(k, n)[0]:.1f}–{100 * wilson(k, n)[1]:.1f})")(
        sum(chan(f, **kw) for f in fs), len(fs))
    L = ["| tập | lang | K1 hiện tại | bộ dò mới | bộ dò mới, trừ hàng cứng thật |", "|---|---|---|---|---|"]
    for sp in ("dev", "test", "tất cả"):
        for lg in ("en", "vi", "chung"):
            fs = [f for f in k1r if (sp == "tất cả" or fold(f) == sp) and (lg == "chung" or f.split("/")[0] == lg)]
            n_cung = sum(1 for f in fs if reasons[f] and all(any(h in s for h in HANG_CUNG) for s in reasons[f]))
            fs2 = [f for f in fs if not (reasons[f] and all(any(h in s for h in HANG_CUNG) for s in reasons[f]))]
            L.append(f"| {sp} | {lg} | {cell(fs, moi=False, bo_cung=False)} | {cell(fs, moi=True, bo_cung=False)} | "
                     f"{cell(fs2, moi=True, bo_cung=True)} (bỏ {n_cung}) |")
    tl = {f: ok for f, ok in lab.items() if f in new and fold(f) == "test"}
    oan = [f for f, ok in tl.items() if ok and new[f]]
    tha = [f for f, ok in tl.items() if not ok and not new[f]]
    out = (f"# Mở test cho bộ dò `{CHON}` (chọn trên dev trước)\n\nWilson 95 % trong ngoặc. Nhãn nghe Gemini (sơ bộ) fold test: "
           f"n = {len(tl)}, còn chặn oan {len(oan)} {oan}, chặn đúng bị thả {len(tha)} {tha}.\n\n" + "\n".join(L) + "\n")
    open(os.path.join(BO, "_kiem", "DO_TIENG_NOI_TEST.md"), "w", encoding="utf-8").write(out)
    return out


def selftest() -> None:
    t = [0.1, 0.2, 1.0, 1.1]
    assert abs(speech_s(t, 2.0, 0.0) - speech_s(t, 2.0, 0.05)) < 1e-9          # gap nhỏ hơn khoảng 2 ký tự sát nhau: như cũ
    assert speech_s(t, 2.0, 1.0) > speech_s(t, 2.0, 0.0)                         # gộp lấp khoảng 0,2 → 1,0
    assert speech_s([], 5.0, 1.0) == 0.0
    assert speech_s([0.5], 3.0, 1.0) == speech_s([0.5], 3.0, 0.0)                # 1 ký tự: không có khoảng để gộp
    assert blocked(1.9, 5.0) and blocked(2.5, 10.0) and not blocked(3.1, 10.0)
    print("do_tieng_noi selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:] or ["selftest"]
    {"cache": cache, "eval": lambda: print(evaluate("--test" in a)), "alt": lambda: print(alt()), "test": lambda: print(test()), "selftest": selftest}[a[0]]()
