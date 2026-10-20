# -*- coding: utf-8 -*-
"""NHAN DANG NGON NGU (coverage B-12 'LID >= 0,9') cho doan host cao - Whisper DONG BANG (pretrained, quyet dinh 6), CPU.

  Moi doan wav 16 kHz: bo trich dac trung Whisper (pad / cat 30 s dau) -> encoder -> MOT buoc decoder sau
  <|startoftranscript|> -> softmax CHI tren cac token ngon ngu -> p_target (ngon ngu dich cua host: vi | en), p_vi, p_en, top.
  Mo hinh mac dinh openai/whisper-small (da co trong D:/hf-cache, HF_HUB_OFFLINE=1). KHONG huan luyen, khong tinh chinh.
  Ra: meta/lid.jsonl {file, lang, tier, top, p_target, p_vi, p_en, dur_used, model}; ghi tiep, bo qua tep da co
  (--redo: lam lai). hosts.py doc tep nay: p_target < 0,9 -> bo ('lid_lech'); chua co -> bo ('chua_lid').
  --calib: do tren du lieu NGOAI tap loc: duong = mau host tu nhien cung ngon ngu (data/natural_v6/hosts*), am = host tu
  nhien ngon ngu kia + doan cua kenh 'vi' da biet noi tieng Anh (PANTADO, ke ca da chuyen vao _quarantine)
  -> meta/lid_calib.json (ty le duong < nguong, am >= nguong, o 0,9 va o LID_MIN theo ngon ngu: vi 0,9 / en 0,5).
NANG VUA (~1,2 GB RAM, ~1-3 s / doan voi 4 luong CPU): tron bo = hang doi; o day thu --limit / --budget-min.

CLI: python -m v6.scrape.lid [--lang vi|en] [--limit N] [--budget-min 8] [--threads 4] [--model openai/whisper-small]
     python -m v6.scrape.lid --calib [--n-nat 30]"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np

HF = "D:/hf-cache"
os.environ["HF_HOME"] = HF                         # truoc moi import huggingface_hub / transformers
os.environ["HF_HUB_CACHE"] = HF + "/hub"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from v6.scrape import util as U  # noqa: E402

MODEL = "openai/whisper-small"
# nguong p(ngon ngu dich) THEO NGON NGU (lid_calib 12/10, n = 24 / lop - nho): vi 0,9 (duong min 0,957, am max 0,003);
# en 0,5 vi tieng Anh giong la (VoxPopuli / EdAcc) cho p_en 0,11-0,99 -> nguong 0,9 bo 50 % host tieng Anh that
# (am: host tieng Viet p_en max 0,008). B-12 'LID >= 0,9' hieu la do chinh xac LID tren mau hieu chinh, xem lid_calib.json.
LID_MIN = {"vi": 0.9, "en": 0.5}
LID_MIN_DEFAULT = 0.9
MAX_S = 30.0
BATCH = 4
LID_JSONL = os.path.join(U.META, "lid.jsonl")
LID_CALIB = os.path.join(U.META, "lid_calib.json")
OFFLANG = {"vi": ("UC2JPCC4OKl5oQUeiLBFR--w",)}      # kenh 'vi' noi tieng Anh (PANTADO) -> am that cho dich vi
NAT_HOSTS = {"vi": os.path.join(U.BASE, "data", "natural_v6", "hosts", "hosts.jsonl"),
             "en": os.path.join(U.BASE, "data", "natural_v6", "hosts_en", "hosts.jsonl")}


def lid_min(lang: str | None) -> float:
    """Nguong p(ngon ngu dich) cua hosts cho ngon ngu lang."""
    return LID_MIN.get(str(lang or ""), LID_MIN_DEFAULT)


class WhisperLID:
    """Bo nhan dang ngon ngu: probs(list mang 16 kHz) -> list {ma_ngon_ngu: p} (chi token ngon ngu, tong = 1)."""

    def __init__(self, name: str = MODEL, threads: int | None = None) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        if threads:
            torch.set_num_threads(int(threads))
        self.torch = torch
        self.name = name
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name).eval()
        for p in self.model.parameters():            # dong bang (chi suy luan)
            p.requires_grad_(False)
        gc = self.model.generation_config
        l2i = dict(getattr(gc, "lang_to_id", None) or {})
        if not l2i:
            raise RuntimeError(f"{name}: generation_config khong co lang_to_id (khong phai Whisper da ngon ngu)")
        self.codes = [k.strip("<|>") for k in l2i]
        self.ids = torch.tensor(list(l2i.values()))
        self.sot = int(gc.decoder_start_token_id)

    def probs(self, ys: list[np.ndarray]) -> list[dict[str, float]]:
        torch = self.torch
        ys = [np.asarray(y, np.float32)[:int(MAX_S * U.SR)] for y in ys]
        feats = self.proc.feature_extractor(ys, sampling_rate=U.SR, return_tensors="pt").input_features
        with torch.inference_mode():
            dec = torch.full((len(ys), 1), self.sot, dtype=torch.long)
            logits = self.model(input_features=feats, decoder_input_ids=dec).logits[:, -1]
            p = torch.softmax(logits[:, self.ids].float(), dim=-1).numpy()
        return [{c: float(v) for c, v in zip(self.codes, row)} for row in p]


def summarize_probs(pr: Mapping[str, float], target: str) -> dict[str, Any]:
    top = max(pr, key=pr.get) if pr else None
    return {"top": top, "p_top": round(pr.get(top, 0.0), 4) if top else None,
            "p_target": round(pr.get(target, 0.0), 4), "p_vi": round(pr.get("vi", 0.0), 4),
            "p_en": round(pr.get("en", 0.0), 4)}


def read16k(path: str) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != U.SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=U.SR)
    return y


seg_wav = U.seg_wav     # duong dan wav cua doan, ke ca doan da chuyen vao _quarantine


def run_batches(lid: WhisperLID, items: list[tuple[dict, str]], target_of, budget_min: float,
                out_path: str | None = None) -> list[dict]:
    """items: [(dong, wav)]; target_of(dong) -> 'vi' | 'en'. Ghi tiep tung lo vao out_path (neu co)."""
    t0, out = time.time(), []
    for i in range(0, len(items), BATCH):
        if time.time() - t0 > budget_min * 60:
            print("het ngan sach thoi gian; chay lai de tiep tuc", flush=True)
            break
        chunk = items[i:i + BATCH]
        ys = [read16k(w) for _, w in chunk]
        rows = []
        for (r, _), y, pr in zip(chunk, ys, lid.probs(ys)):
            rows.append({**r, "lang": target_of(r), **summarize_probs(pr, target_of(r)),
                         "dur_used": round(min(len(y) / U.SR, MAX_S), 2), "model": lid.name, "lid_at": U.now_iso()})
        if out_path:
            U.append_jsonl(out_path, rows)
        out += rows
        print(f"  [{min(i + BATCH, len(items))}/{len(items)}] {time.time() - t0:.0f} s "
              f"{[(r['file'][-18:], r['top'], r['p_target']) for r in rows]}", flush=True)
    return out


def _q(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 4)


def nat_items(lang: str, n: int, rng: random.Random) -> list[tuple[dict, str]]:
    from v6.scrape.qc import wav_map
    path = NAT_HOSTS[lang]
    rows = [r for r in U.read_jsonl(path) if 4.0 <= float(r.get("dur") or 0) <= 30.0]
    rng.shuffle(rows)
    base, wmap, out = os.path.dirname(path), None, []
    for r in rows:
        if len(out) >= n:
            break
        w = os.path.join(base, str(r.get("source")), str(r.get("file")) + ".wav")
        if not os.path.exists(w):
            wmap = wav_map(base) if wmap is None else wmap
            w = wmap.get(str(r.get("file")))
        if w and os.path.exists(w):
            out.append(({"file": r["file"], "nat_lang": lang, "source": r.get("source")}, w))
    return out


def calib(lid: WhisperLID, n_nat: int, seed: int, budget_min: float, out_path: str = LID_CALIB) -> dict:
    rng = random.Random(seed)
    nat = {lang: run_batches(lid, nat_items(lang, n_nat, rng), lambda r: r["nat_lang"], budget_min / 3)
           for lang in ("vi", "en")}
    off = [(s, seg_wav(s)) for s in U.read_jsonl(U.SEGMENTS) if s.get("file")
           and s.get("channel_id") in OFFLANG["vi"]]
    off = [(dict(file=s["file"], channel_id=s["channel_id"]), w) for s, w in off if w][:n_nat]
    offr = run_batches(lid, off, lambda r: "vi", budget_min / 3)
    res: dict[str, Any] = {"model": lid.name, "lid_min": LID_MIN, "n_nat": n_nat, "seed": seed,
                           **calib_stats(nat, offr), "rows": {"nat": nat, "offlang_vi": offr}}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    return res


def _frac(xs: list[float], thr: float, above: bool) -> float:
    return round(sum((x >= thr) if above else (x < thr) for x in xs) / max(1, len(xs)), 4)


def calib_stats(nat: Mapping[str, list], offr: list) -> dict:
    """Thong ke hieu chinh (ham thuan): ty le duong duoi nguong / am tren nguong o 0,9 VA o nguong theo ngon ngu."""
    out = {}
    for lang, other in (("vi", "en"), ("en", "vi")):
        pos = [r["p_target"] for r in nat.get(lang, [])]
        neg = [r[f"p_{lang}"] for r in nat.get(other, [])]
        thr = lid_min(lang)
        d = {"thr": thr, "n_pos": len(pos), "pos_frac_below_0.9": _frac(pos, 0.9, False),
             "pos_frac_below_thr": _frac(pos, thr, False), "pos_min": _q(pos, 0), "pos_p05": _q(pos, 0.05),
             "top1_acc_pos": round(sum(r["top"] == lang for r in nat.get(lang, [])) / max(1, len(pos)), 4),
             "top_wrong_pos": sorted({r["top"] for r in nat.get(lang, []) if r["top"] != lang}),
             "n_neg_other_lang": len(neg), "neg_frac_above_thr": _frac(neg, thr, True), "neg_max": _q(neg, 1.0)}
        if lang == "vi":
            pv = [r["p_vi"] for r in offr]
            d.update({"n_neg_offlang_channel": len(pv), "neg_offlang_frac_above_thr": _frac(pv, thr, True),
                      "neg_offlang_median": _q(pv, 0.5), "neg_offlang_max": _q(pv, 1.0),
                      "neg_offlang_top": {t: sum(r["top"] == t for r in offr) for t in sorted({r["top"] for r in offr})},
                      "note_offlang": "kenh PANTADO co ca loi tieng Viet (phu huynh / giao vien) -> doan p_vi cao co the dung"})
        out[lang] = d
    return out


def todo_segments(segs: Iterable[Mapping[str, Any]], done: set, lang: str | None, redo: bool = False,
                  gated: Mapping[str, Mapping[str, Any]] | None = None, allow_flags=()) -> list[tuple[dict, str]]:
    """Doan can LID: co wav, chua lam; gated (cong hien tai) != None -> chi video gate.hold_reason None."""
    from v6.scrape.gate import hold_reason
    out = []
    for s in segs:
        if not s.get("file") or (lang and s.get("lang") != lang) or (s["file"] in done and not redo):
            continue
        if gated is not None and hold_reason(gated.get(s.get("id")), allow_flags) is not None:
            continue
        w = seg_wav(s)
        if w:
            out.append(({"file": s["file"], "tier": s.get("tier"), "id": s.get("id"), "seg_lang": s.get("lang")}, w))
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["vi", "en"])
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--n-nat", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-flag", action="append", default=[], help="nhu segment/hosts --allow-flag")
    a = ap.parse_args()
    t0 = time.time()
    lid = WhisperLID(a.model, a.threads)
    print(f"nap {a.model}: {time.time() - t0:.0f} s", flush=True)
    if a.calib:
        res = calib(lid, a.n_nat, a.seed, a.budget_min)
        print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))
        print("->", LID_CALIB)
        return
    from v6.scrape import gate as G
    done = set(U.latest_by(U.read_jsonl(LID_JSONL), "file"))
    items = todo_segments(U.read_jsonl(U.SEGMENTS), done, a.lang, a.redo, G.current(), a.allow_flag)[:a.limit]
    print(f"can LID: {len(items)} doan", flush=True)
    rows = run_batches(lid, items, lambda r: r["seg_lang"], a.budget_min, LID_JSONL)
    bad = [r for r in rows if r["p_target"] < lid_min(r["lang"])]
    print(f"lid: {len(rows)} doan, {len(bad)} duoi nguong {LID_MIN} -> {LID_JSONL}")


# ---------------- kiem thu (khong nap mo hinh) ----------------
def test_summarize_and_todo() -> None:
    s = summarize_probs({"vi": 0.95, "en": 0.04, "zh": 0.01}, "vi")
    assert s["top"] == "vi" and s["p_target"] == 0.95 and s["p_en"] == 0.04
    s = summarize_probs({"vi": 0.3, "en": 0.7}, "vi")
    assert s["top"] == "en" and s["p_target"] == 0.3
    segs = [{"file": "a", "lang": "vi", "wav": "khong/co.wav"}, {"file": None}, {"file": "b", "lang": "en", "wav": ""}]
    assert todo_segments(segs, set(), None) == []                 # wav khong ton tai -> bo qua
    assert lid_min("vi") == 0.9 and lid_min("en") == 0.5 and lid_min(None) == LID_MIN_DEFAULT
    nat = {"vi": [{"p_target": 0.99, "top": "vi", "p_vi": 0.99, "p_en": 0.0}],
           "en": [{"p_target": 0.6, "top": "en", "p_vi": 0.01, "p_en": 0.6},
                  {"p_target": 0.2, "top": "cy", "p_vi": 0.0, "p_en": 0.2}]}
    st = calib_stats(nat, [{"p_vi": 0.95, "top": "vi"}, {"p_vi": 0.01, "top": "en"}])
    assert st["en"]["pos_frac_below_0.9"] == 1.0 and st["en"]["pos_frac_below_thr"] == 0.5 and st["en"]["top1_acc_pos"] == 0.5
    assert st["vi"]["neg_offlang_frac_above_thr"] == 0.5 and st["vi"]["neg_frac_above_thr"] == 0.0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        test_summarize_and_todo()
        print("PASS test_summarize_and_todo")
    else:
        main()
