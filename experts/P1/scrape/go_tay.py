"""Gỡ tay một video khỏi p1_tunhien_v1 (Huấn duyệt), không dựng lại cả bộ:
thêm vào loai_tay.json (lần build sau vẫn gỡ) → chuyển mọi đoạn wav vào _quarantine/ + LY_DO.jsonl → bỏ khỏi manifest.json, cases.json.

  .venv/Scripts/python.exe experts/P1/scrape/go_tay.py run <video_id> "<lý do>" | selftest
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tai_cat import DST  # noqa: E402

LOAI_TAY = os.path.join(HERE, "loai_tay.json")


def go(dst: str, loai_tay: str, vid: str, ly_do: str) -> list[str]:
    man = json.load(open(os.path.join(dst, "manifest.json"), encoding="utf-8"))
    cases = json.load(open(os.path.join(dst, "cases.json"), encoding="utf-8"))
    bo = [r for r in man if r["meta"]["recording"] == vid]
    files = sorted(f"{r['meta']['lang']}/{r['id']}.wav" for r in bo)
    if not files:
        raise SystemExit(f"Không có đoạn nào của video {vid}")
    missing = [f for f in files if not os.path.exists(os.path.join(dst, f))]
    if missing:
        raise SystemExit(f"Thiếu tệp wav: {missing}")
    # kiểm xong mới ghi
    lt = json.load(open(loai_tay, encoding="utf-8"))
    json.dump({**lt, vid: ly_do}, open(loai_tay, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    q = os.path.join(dst, "_quarantine")
    for f in files:
        os.makedirs(os.path.join(q, os.path.dirname(f)), exist_ok=True)
        os.replace(os.path.join(dst, f), os.path.join(q, f))
        with open(os.path.join(q, "LY_DO.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"file": f, "video": vid, "ly_do": f"khong_loi_noi:nghe:{ly_do}",
                                 "luat": "go_tay.py (loai_tay.json), Huấn duyệt"}, ensure_ascii=False) + "\n")
    json.dump([r for r in man if r["meta"]["recording"] != vid], open(os.path.join(dst, "manifest.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump([c for c in cases if c["file"] not in set(files)], open(os.path.join(dst, "cases.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    return files


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "vi"))
        man = [{"id": f"p1n_vi_X_{i}", "meta": {"lang": "vi", "recording": "X" if i < 3 else "Y"}} for i in range(1, 5)]
        for r in man:
            open(os.path.join(d, "vi", r["id"] + ".wav"), "wb").close()
        json.dump(man, open(os.path.join(d, "manifest.json"), "w"))
        json.dump([{"file": f"vi/{r['id']}.wav"} for r in man], open(os.path.join(d, "cases.json"), "w"))
        lt = os.path.join(d, "lt.json")
        json.dump({"A": "a"}, open(lt, "w"))
        out = go(d, lt, "X", "bài hát")
        assert out == ["vi/p1n_vi_X_1.wav", "vi/p1n_vi_X_2.wav"]
        assert len(json.load(open(os.path.join(d, "manifest.json")))) == 2 and len(json.load(open(os.path.join(d, "cases.json")))) == 2
        assert json.load(open(lt, encoding="utf-8")) == {"A": "a", "X": "bài hát"}
        assert os.path.exists(os.path.join(d, "_quarantine", "vi", "p1n_vi_X_1.wav")) and not os.path.exists(os.path.join(d, "vi", "p1n_vi_X_1.wav"))
        assert sum(1 for _ in open(os.path.join(d, "_quarantine", "LY_DO.jsonl"), encoding="utf-8")) == 2
    print("go_tay selftest dat")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:]
    if a[:1] == ["selftest"]:
        selftest()
    elif a[:1] == ["run"] and len(a) == 3:
        print("\n".join(f"_quarantine ← {f}" for f in go(DST, LOAI_TAY, a[1], a[2])))
    else:
        raise SystemExit(__doc__)
