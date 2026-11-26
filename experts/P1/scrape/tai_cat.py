"""Bước 2–4 cào ghi âm tự nhiên: tải → cắt đoạn → dựng bộ `p1_tunhien_v1` (nhãn «01 có khả năng», PHU_DU_LIEU mục 2.2 ý 3).

  fetch    chọn video qua cổng (ung_vien_<lang>.jsonl), tối đa MAX_VID_PER_CH video / kênh; tải bestaudio, GIỮ NGUYÊN tệp YouTube trả ra
           (data/p1_scrape/goc/<lang>/<id>.<đuôi>) rồi chuyển sang wav 16 kHz mono 16 bit (chỉ đổi định dạng → vẫn 01, mục 2.4).
  segment  mỗi video: 1 đoạn dài (1–10 phút, luân phiên 60/120/240/400/560 s như v6/scrape/long_sets.py) + 1 đoạn 16–64 s + 2 đoạn 3–16 s,
           KHÔNG chồng nhau, bỏ 30 s đầu / cuối (nhạc mở / kết). Mỗi đoạn là MỘT khúc liền của một bản ghi (mục 1.3, không ghép).
           Cổng đoạn CHỈ xét dấu vết biên tập (mục 2.3 ý 3) + đúng ngôn ngữ; KHÔNG lọc theo điều kiện K1 (ồn, ít tiếng nói, cắt đỉnh…)
           để không làm đẹp giả số chặn oan. Số đo khác chỉ ghi lại.
  build    ghi wav + manifest H1 + thẻ bộ vào data_labels/v6/p1_tunhien_v1/.
Dùng:  .venv/Scripts/python.exe experts/P1/scrape/tai_cat.py fetch --lang vi | segment --lang vi | build | selftest
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cao_tunhien import NHAC_NEN_NGHE, OUT, ROOT, nonspeech  # noqa: E402

sys.path.insert(0, ROOT)
SR = 16000
MAX_VID_PER_CH = 2                     # 21/11 cào bù: 3 → 2 (Huấn chốt, giữ tổng < 6 giờ máy theo mục 7.6); video kênh cũ đã tải vẫn giữ
EDGE_S = 30.0
LONG_S = (60, 120, 240, 400, 560)
MAX_ZERO_S = 0.1                       # chuỗi số 0 kỹ thuật số > 0,1 s giữa đoạn = dấu vết biên tập / mất tín hiệu → loại đoạn
MIN_LANG_P = 0.5                       # xác suất ngôn ngữ đích (whisper-small, vaudit.k1x.language_probs) — ngưỡng cảnh báo của K1x
BO = "p1_tunhien_v1"
DST = os.path.join(ROOT, "data_labels", "v6", BO)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rows(lang: str) -> list[dict]:
    return [json.loads(x) for x in open(os.path.join(OUT, f"ung_vien_{lang}.jsonl"), encoding="utf-8")]


# ----------------------------------------------------------------------------- chọn + tải
def pick_videos(cands: list[dict], per_ch: int, seed: int = 0, done: frozenset = frozenset()) -> list[dict]:
    """Video qua cổng (cổng lúc cào + luật nonspeech hiện hành, vì dòng cũ mang ly_do_loai của cổng cũ), mỗi kênh tối đa per_ch.
    Video đã tải (done) giữ trước, chỉ bù cho đủ per_ch → cào bù không làm kênh vượt hạn mức, không đổi lựa chọn cũ."""
    rng = random.Random(seed)
    by: dict = {}
    for r in cands:
        if not r["ly_do_loai"] and not nonspeech(r):
            by.setdefault(r["channel_id"], []).append(r)
    out = []
    for ch in sorted(by):
        g = sorted(by[ch], key=lambda r: r["id"])
        rng.shuffle(g)
        g.sort(key=lambda r: r["id"] not in done)      # sort ổn định: đã tải lên đầu, phần còn lại giữ thứ tự ngẫu nhiên
        out += g[:per_ch]
    return out


def download(r: dict, lang: str) -> dict:
    import yt_dlp
    d = os.path.join(OUT, "goc", lang)
    os.makedirs(d, exist_ok=True)
    opts = {"format": "bestaudio", "outtmpl": os.path.join(d, "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
            "sleep_interval": 2, "max_sleep_interval": 5}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(r["webpage_url"], download=True)
    src = os.path.join(d, f"{r['id']}.{info['ext']}")
    wav = os.path.join(OUT, "wav16", lang, f"{r['id']}.wav")
    os.makedirs(os.path.dirname(wav), exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", wav], check=True)
    return {"id": r["id"], "tep_goc": os.path.relpath(src, ROOT).replace("\\", "/"), "sha256_goc": sha256(src),
            "acodec": info.get("acodec"), "abr": info.get("abr"), "asr": info.get("asr"),
            "wav16": os.path.relpath(wav, ROOT).replace("\\", "/")}


def fetch(lang: str) -> None:
    log = os.path.join(OUT, f"tai_{lang}.jsonl")
    done = {json.loads(x)["id"] for x in open(log, encoding="utf-8")} if os.path.exists(log) else set()
    todo = [r for r in pick_videos(rows(lang), MAX_VID_PER_CH, done=frozenset(done)) if r["id"] not in done]
    print(f"{lang}: cần tải {len(todo)} video", flush=True)
    for i, r in enumerate(todo):
        try:
            row = download(r, lang)
        except Exception as e:  # noqa: BLE001  (video lỗi: ghi lại, tải tiếp)
            row = {"id": r["id"], "loi": f"{type(e).__name__}: {e}"[:300]}
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{i + 1}/{len(todo)} {r['id']} {'LỖI' if 'loi' in row else row['acodec']}", flush=True)


# ----------------------------------------------------------------------------- cắt đoạn (hàm thuần + cổng)
def plan_segments(dur: float, k: int, rng: random.Random) -> list[tuple[str, float, float]]:
    """[(nhóm độ dài, bắt đầu, kết thúc)] không chồng nhau trong [EDGE_S, dur − EDGE_S]. k = số thứ tự video (luân phiên độ dài dài)."""
    lo, hi = EDGE_S, dur - EDGE_S
    want = [("1-10m", float(LONG_S[k % len(LONG_S)])), ("16-64s", rng.uniform(20, 60)),
            ("3-16s", rng.uniform(4, 15)), ("3-16s", rng.uniform(4, 15))]
    out, t = [], lo
    for grp, L in want:
        if grp == "1-10m" and hi - lo < L:                   # video ngắn: bỏ đoạn dài, không cắt ngắn lại (giữ đúng nhóm)
            continue
        if t + L > hi:
            break
        gap = rng.uniform(0, max(0.0, (hi - t - L) * 0.2))  # rải đoạn dọc video
        out.append((grp, round(t + gap, 2), round(t + gap + L, 2)))
        t = t + gap + L + 1.0
    return out


def zero_run_s(y: np.ndarray, thr: float = 3e-5) -> float:
    z = np.r_[0, (np.abs(y) < thr).astype(np.int8), 0]
    d = np.diff(z)
    runs = np.where(d == -1)[0] - np.where(d == 1)[0]
    return float(runs.max() / SR) if len(runs) else 0.0


def seg_gate(y: np.ndarray, lang: str, lang_p: dict) -> list[str]:
    """Lý do loại đoạn: CHỈ dấu vết biên tập + sai ngôn ngữ (không xét điều kiện ghi mà K1 sẽ chấm)."""
    r = []
    if zero_run_s(y) > MAX_ZERO_S:
        r.append(f"so_0:{zero_run_s(y):.2f}s")
    if lang_p.get(lang, 0.0) < MIN_LANG_P:
        r.append(f"ngon_ngu:{lang_p.get('top')}:{lang_p.get(lang, 0.0):.2f}")
    return r


def segment(lang: str) -> None:
    from vaudit import k1x
    tai = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(OUT, f"tai_{lang}.jsonl"), encoding="utf-8")) if "loi" not in r}
    meta = {r["id"]: r for r in rows(lang)}
    out = os.path.join(OUT, f"doan_{lang}.jsonl")
    done = {json.loads(x)["video"] for x in open(out, encoding="utf-8")} if os.path.exists(out) else set()
    rng = random.Random(1)
    for k, vid in enumerate(sorted(tai)):
        if vid in done:
            continue
        x, _ = sf.read(os.path.join(ROOT, tai[vid]["wav16"]), dtype="float32")
        for grp, a, b in plan_segments(len(x) / SR, k, rng):
            y = x[int(a * SR): int(b * SR)]
            lp = k1x.language_probs(y)
            row = {"video": vid, "channel_id": meta[vid]["channel_id"], "nhom_do_dai": grp, "start": a, "end": b,
                   "lang_p": {kk: round(float(v), 3) for kk, v in lp.items() if kk != "top"} | {"top": lp.get("top")},
                   "so_0_dai_s": round(zero_run_s(y), 3), "ly_do_loai": seg_gate(y, lang, lp)}
            with open(out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{k + 1}/{len(tai)} {vid}", flush=True)


# ----------------------------------------------------------------------------- dựng bộ
def record(s: dict, m: dict, t: dict, lang: str, idx: int) -> dict:
    """Một bản ghi manifest H1 cho một đoạn đã qua cổng."""
    rid = f"p1n_{lang}_{s['video']}_{idx}"
    abr = t.get("abr")
    return {"id": rid, "kind": "tunhien", "cfg": "goc", "src": s["video"], "wav": f"../v6/{BO}/{lang}/{rid}.wav",
            "meta": {"lang": lang, "source": "yt", "person": f"yt:{m['channel_id']}", "speaker": f"yt:{m['channel_id']}",
                     "channel_id": m["channel_id"], "recording": s["video"], "url": m["webpage_url"], "license": m["license"],
                     "level1": ["01_nguyen_ban"], "level2": [], "edits": [], "nhan_chac": "01_co_kha_nang",
                     "kenh": f"yt_{t.get('acodec')}_{round(abr) if abr else '?'}k", "nhom_do_dai": s["nhom_do_dai"],
                     "doan_trong_video_s": [s["start"], s["end"]], "cho_noi": [], "so_nguoi_noi": None, "thiet_bi": None,
                     "truy_van": m["truy_van"], "ngay_cao": m["doc_luc"][:10]}}


def build() -> None:
    man = []
    for lang in ("vi", "en"):
        p = os.path.join(OUT, f"doan_{lang}.jsonl")
        if not os.path.exists(p):
            continue
        meta = {r["id"]: r for r in rows(lang)}
        tai = {r["id"]: r for r in (json.loads(x) for x in open(os.path.join(OUT, f"tai_{lang}.jsonl"), encoding="utf-8")) if "loi" not in r}
        segs = [json.loads(x) for x in open(p, encoding="utf-8")]
        cnt: dict = {}
        for s in segs:
            if s["ly_do_loai"] or nonspeech(meta[s["video"]]):   # luật «không phải lời nói» thêm 21/11 (cao_tunhien.NONSPEECH)
                continue
            cnt[s["video"]] = cnt.get(s["video"], 0) + 1
            rec = record(s, meta[s["video"]], tai[s["video"]], lang, cnt[s["video"]])
            x, _ = sf.read(os.path.join(ROOT, tai[s["video"]]["wav16"]), dtype="int16", start=int(s["start"] * SR), stop=int(s["end"] * SR))
            path = os.path.join(DST, lang, rec["id"] + ".wav")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            sf.write(path, x, SR, subtype="PCM_16")
            man.append(rec)
    os.makedirs(DST, exist_ok=True)
    json.dump(man, open(os.path.join(DST, "manifest.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cases = [case_of(r) for r in man]
    json.dump(cases, open(os.path.join(DST, "cases.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # cho chay_k1 --bo
    card(man)
    quarantine({f"{r['meta']['lang']}/{r['id']}.wav" for r in man})
    print(f"{len(man)} đoạn → {os.path.relpath(DST, ROOT)}")


def case_of(r: dict) -> dict:
    """Đáp án một đoạn: «accept»; video Gemini nghe có nhạc nền rõ (cao_tunhien.NHAC_NEN_NGHE) → «khong_chac», ca riêng, không chấm."""
    lang, vid = r["meta"]["lang"], r["meta"]["url"].split("v=")[-1]
    nn = vid in NHAC_NEN_NGHE
    return {"file": f"{lang}/{r['id']}.wav", "ca": f"tunhien.{lang}." + ("nhac_nen" if nn else r["meta"]["nhom_do_dai"]),
            "mong_doi": "khong_chac" if nn else "accept", "ly_do": None, "ca_bien": False, "base": r["meta"]["person"]}


def quarantine(keep: set[str]) -> None:
    """Tệp wav đã dựng lần trước nhưng không còn trong manifest → chuyển vào _quarantine/ kèm lý do (PHU_DU_LIEU: không xoá)."""
    meta = {r["id"]: r for lang in ("vi", "en") if os.path.exists(os.path.join(OUT, f"ung_vien_{lang}.jsonl")) for r in rows(lang)}
    q = os.path.join(DST, "_quarantine")
    for lang in ("vi", "en"):
        d = os.path.join(DST, lang)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            rel = f"{lang}/{f}"
            if rel in keep:
                continue
            vid = f[len(f"p1n_{lang}_"):].rsplit("_", 1)[0]
            why = nonspeech(meta.get(vid, {}))
            os.makedirs(os.path.join(q, lang), exist_ok=True)
            os.replace(os.path.join(d, f), os.path.join(q, lang, f))
            with open(os.path.join(q, "LY_DO.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"file": rel, "video": vid, "ly_do": f"khong_loi_noi:{','.join(why)}" if why else "khong_con_trong_manifest",
                                     "luat": "cao_tunhien.NONSPEECH (tiêu đề + thẻ), Huấn chốt 21/11/2026"}, ensure_ascii=False) + "\n")
            print(f"_quarantine ← {rel} ({why or 'không còn trong manifest'})")


def card(man: list[dict]) -> None:
    """Thẻ bộ (mục 2.5 ý 9): nguồn, giấy phép, ngày, ai làm, cách làm, số tệp + số người nói theo ô."""
    import collections
    cell = collections.defaultdict(lambda: [0, set()])
    for r in man:
        c = cell[(r["meta"]["lang"], r["meta"]["nhom_do_dai"])]
        c[0] += 1
        c[1].add(r["meta"]["person"])
    L = [f"# Thẻ bộ `{BO}`", "",
         "- **Là gì:** ghi âm tự nhiên cào từ YouTube, nhãn **«01 có khả năng»** (PHU_DU_LIEU mục 2.2 ý 3): chỉ để đo chặn oan / vu oan, "
         "không làm mốc 01 chắc chắn. Nhãn 01 đến từ nguồn (ghi âm thật, không có thao tác tự làm), không phải nghe rồi đoán.",
         "- **Ai làm:** Huấn (P1); mã do Claude Code (Claude Opus 5.5) viết theo hướng Huấn chốt 19/11.",
         "- **Cách làm:** `experts/P1/scrape/cao_tunhien.py` (tìm video bộ lọc Creative Commons + cổng mục 1.8: giấy phép CC BY trong metadata, "
         "kênh không chuyên = không tích xanh và ≤ 100 000 người theo dõi, bỏ phụ lục A, bỏ kênh tập khoá v2 crc32 % 5 == 0, bỏ tôn giáo / trẻ em / "
         "trường phổ thông theo từ khoá) → `tai_cat.py` (yt-dlp bestaudio, giữ tệp gốc; ffmpeg → wav 16 kHz mono; mỗi video ≤ 1 đoạn dài + 1 vừa "
         "+ 2 ngắn, liền mạch, bỏ 30 s đầu / cuối; cổng đoạn: chuỗi số 0 ≤ 0,1 s, ngôn ngữ đích ≥ 0,5 theo whisper-small). "
         "**Không lọc theo điều kiện K1** (ồn, ít tiếng nói…) để số chặn oan không bị làm đẹp.",
         f"- **Người nói:** `person = yt:<channel_id>` (gom theo kênh → một kênh một fold). Mỗi kênh ≤ {MAX_VID_PER_CH} video.",
         "- **Truyền tải:** YouTube đã nén lại (`meta.kenh`, ví dụ `yt_opus_130k`) → nén ≥ 24 kbps, tính chỉ tiêu.",
         "- **Hạn chế:** chưa đếm số người nói trong đoạn (`so_nguoi_noi` = null); cổng vết cắt chỉ xét số 0 kỹ thuật số "
         "(chưa xét nhảy nền / nhảy âm lượng / click); từ khoá nội dung có thể loại nhầm (an toàn hơn để lọt).",
         "- **Đã kiểm:** *(điền sau)* `kiem_bo --moi` · `lockbox check` · nghe tay ≥ 20 tệp", "",
         "| ngôn ngữ | nhóm độ dài | số đoạn | số kênh (person) |", "|---|---|---|---|"]
    L += [f"| {lg} | {g} | {c[0]} | {len(c[1])} |" for (lg, g), c in sorted(cell.items())]
    open(os.path.join(DST, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def selftest() -> None:
    rng = random.Random(0)
    for k in range(10):
        segs = plan_segments(900.0, k, rng)
        assert all(EDGE_S <= a < b <= 900 - EDGE_S for _, a, b in segs)
        assert all(segs[i][2] < segs[i + 1][1] for i in range(len(segs) - 1)), segs
        assert segs[0][0] == "1-10m" and abs(segs[0][2] - segs[0][1] - LONG_S[k % 5]) < 0.01
    assert all(g != "1-10m" for g, _, _ in plan_segments(150.0, 4, rng))          # video ngắn: không có đoạn dài
    assert zero_run_s(np.r_[np.ones(100), np.zeros(SR // 5), np.ones(100)].astype(np.float32)) == 0.2
    y = np.ones(SR, np.float32)
    assert seg_gate(y, "vi", {"vi": 0.9}) == [] and seg_gate(y, "vi", {"vi": 0.2, "top": "en"})[0].startswith("ngon_ngu")
    c = [{"channel_id": "A", "id": str(i), "ly_do_loai": []} for i in range(5)] + [{"channel_id": "B", "id": "b", "ly_do_loai": ["x"]}]
    assert len(pick_videos(c, 3)) == 3
    assert {r["id"] for r in pick_videos(c, 3, done=frozenset({"0", "4"}))} >= {"0", "4"}            # đã tải: giữ, chỉ bù
    assert len(pick_videos(c + [{"channel_id": "A", "id": "9", "ly_do_loai": []}], 3, done=frozenset({"0", "1", "2"}))) == 3
    assert pick_videos([{"channel_id": "C", "id": "p", "ly_do_loai": [], "title": "Podcast #1"}], 3) == []   # luật hiện hành
    print("tai_cat selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:] or ["selftest"]
    lang = a[a.index("--lang") + 1] if "--lang" in a else None
    {"fetch": lambda: fetch(lang), "segment": lambda: segment(lang), "build": build, "selftest": selftest}[a[0]]()
