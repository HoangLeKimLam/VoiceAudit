"""Chạy K1 (vaudit.k1.check, extended=False, events=None: lõi K1 + loại âm thanh AST; extended=True cần mô hình nói chồng sherpa-onnx-pyannote trong gói nhóm trưởng) trên bộ p1_dieukien_v1 và chấm theo từng ô của cases.json.

Kết quả thô: <bộ>/_kiem/k1_ket_qua.jsonl (1 dòng / tệp, chạy lại thì bỏ qua tệp đã có → dừng giữa chừng không mất).
Báo cáo:     <bộ>/_kiem/KET_QUA_K1.md

Cách chấm một tệp (K1 trả nhan / canh_bao / 04, đáp án accept / canh_bao / 04):
  - trục K1 TỰ KIỂM được (goc, loai): đúng / chặn oan / bỏ lọt như bình thường;
  - trục K1 chưa kiểm khi thiếu vaudit/cond_models.joblib (snr, nen, vang → not_impl): không chấm đúng / sai,
    chỉ đếm «04 vì lý do khác» (chặn oan do dòng Constraint khác) và cảnh báo lạ.
Dùng:  .venv/Scripts/python.exe experts/P1/build_cond/chay_k1.py [run [--limit N] | report]   (cần ffmpeg trên PATH + mô hình HF)
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dung_bo import OUT, ROOT  # noqa: E402

if "--bo" in sys.argv:                 # bộ khác (vd p1_tunhien_v1: ghi âm tự nhiên, mọi đoạn đáp án «accept» → đo chặn oan)
    OUT = os.path.join(ROOT, "data_labels", "v6", sys.argv[sys.argv.index("--bo") + 1])
KIEM = os.path.join(OUT, "_kiem")
TAG = "_" + sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""      # --tag 2611: tệp kết quả riêng, không ghi đè lần đo cũ
DEV = "--dev" in sys.argv                                                               # chỉ fold 0–2 theo base (giữ kín test fold 3–4)
RAW = os.path.join(KIEM, f"k1_ket_qua{TAG}.jsonl")
DEC = {"nhan": "accept", "canh_bao": "canh_bao", "04": "04", "tu_choi": "04"}
IGNORE_WARN = {"0,3 s đầu và cuối không được soi"}


def done_files() -> set:
    if not os.path.exists(RAW):
        return set()
    return {json.loads(x)["file"] for x in open(RAW, encoding="utf-8")}


def cac_ca() -> list:
    cases = json.load(open(os.path.join(OUT, "cases.json"), encoding="utf-8"))
    if DEV:
        from cham_dieu_kien import fold
        cases = [c for c in cases if fold(c["base"]) <= 2]
    return cases


def run(limit: int | None = None) -> None:
    from vaudit import k1
    cases = cac_ca()
    todo = [c for c in cases if c["file"] not in done_files()][:limit]
    os.makedirs(KIEM, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(todo):
        t = time.time()
        try:
            r = k1.check(os.path.join(OUT, c["file"]), extended=False, events=None)
            row = {"file": c["file"], "decision": r["decision"], "reasons": r["reasons"], "warnings": r["warnings"],
                   "not_impl": r["not_impl"], "measures": r.get("measures", {}), "giay": round(time.time() - t, 1)}
        except Exception as e:  # noqa: BLE001  (ghi lỗi từng tệp, chạy tiếp tệp sau)
            row = {"file": c["file"], "loi": f"{type(e).__name__}: {e}"[:500], "giay": round(time.time() - t, 1)}
        with open(RAW, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        print(f"{i + 1}/{len(todo)} {c['file']} → {row.get('decision', row.get('loi'))} ({row['giay']} s, tổng {time.time() - t0:.0f} s)", flush=True)


def score(c: dict, r: dict) -> str:
    """Nhãn chấm cho một tệp."""
    if "loi" in r:
        return "lỗi chạy"
    got, want = DEC[r["decision"]], c["mong_doi"]
    if want == "khong_chac":                      # đáp án chưa chắc (vd nhạc nền rõ trong ghi âm thật): chỉ ghi K1 ra gì, không chấm
        return f"không chấm (K1: {got})"
    truc = c["ca"].split(".")[0]
    if truc in ("snr", "nen", "vang"):
        other = [x for x in r["reasons"] if c["ly_do"] is None or c["ly_do"] not in x]
        return "04 vì lý do khác" if got == "04" and other else "trục chưa kiểm"
    if got == want:
        return "đúng"
    if want == "04":
        return "bỏ lọt" if got != "04" else "04 sai lý do"
    return "chặn oan" if got == "04" else ("cảnh báo thừa" if got == "canh_bao" else "thiếu cảnh báo")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson 95 % cho tỉ lệ k/n (PHU_DU_LIEU mục 1.1: mỗi số kèm n và khoảng tin cậy)."""
    if n == 0:
        return 0.0, 1.0
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def report() -> str:
    cases = {c["file"]: c for c in cac_ca()}
    allrows = [json.loads(x) for x in open(RAW, encoding="utf-8")]
    rows = [r for r in allrows if r["file"] in cases]          # tệp đã chuyển vào _quarantine: giữ dòng thô, không chấm
    by: dict = collections.defaultdict(list)
    for r in rows:
        by[cases[r["file"]]["ca"]].append((cases[r["file"]], r))
    L = [f"# K1 trên {os.path.basename(OUT)}" + (f" (tag {TAG[1:]})" if TAG else "") + (" · chỉ dev fold 0–2" if DEV else ""), "",
         f"{len(rows)}/{len(cases)} tệp đã chạy · `vaudit.k1.check(extended=False, events=None)` (chưa kiểm ngôn ngữ / nói chồng) · "
         "thiếu `vaudit/cond_models.joblib` → SNR / nền / vang K1 **không tự kiểm** (not_impl)"
         + (f" · bỏ {len(allrows) - len(rows)} dòng của tệp không còn trong cases.json (đã chuyển `_quarantine/`)" if len(allrows) > len(rows) else ""), "",
         "| ca | đáp án | n | K1: nhận / cảnh báo / 04 | chấm | lý do 04 hay gặp | cảnh báo hay gặp |", "|---|---|---|---|---|---|---|"]
    for ca in sorted(by):
        g = by[ca]
        dec = collections.Counter(DEC.get(r.get("decision"), "lỗi") for _, r in g)
        sc = collections.Counter(score(c, r) for c, r in g)
        rs = collections.Counter(x.split(" — ")[0] for _, r in g for x in r.get("reasons", []))
        ws = collections.Counter(w for _, r in g for w in r.get("warnings", []) if w not in IGNORE_WARN)
        fmt = lambda cnt: ", ".join(f"{k} ×{v}" for k, v in cnt.most_common(3)) or "—"
        L.append(f"| {ca} | {g[0][0]['mong_doi']} | {len(g)} | {dec['accept']} / {dec['canh_bao']} / {dec['04']} | {fmt(sc)} | {fmt(rs)} | {fmt(ws)} |")
    valid = [(c, r) for g in by.values() for c, r in g if c["mong_doi"] in ("accept", "canh_bao") and "loi" not in r]
    for lang in sorted({c["file"].split("/")[0] for c, _ in valid if "/" in c["file"]} | {"*"}):
        v = [(c, r) for c, r in valid if lang == "*" or c["file"].startswith(lang + "/")]
        k = sum(DEC[r["decision"]] == "04" for _, r in v)
        lo, hi = wilson(k, len(v))
        L.append(f"\n**Chặn oan ({'tất cả' if lang == '*' else lang}):** {k}/{len(v)} = {100 * k / max(1, len(v)):.1f} % "
                 f"(Wilson 95 %: {100 * lo:.1f}–{100 * hi:.1f} %) · đích ≤ 2 %")
    t = [r["giay"] for r in rows]
    L += ["", f"Thời gian: trung bình {sum(t) / len(t):.1f} s / tệp (CPU), tổng {sum(t) / 60:.0f} phút.",
          "", "Chú giải: «trục chưa kiểm» = K1 không có mô hình cho trục này nên không chấm đúng / sai; "
          "«04 vì lý do khác» = K1 chặn tệp vì một dòng Constraint KHÁC trục đang thử (thường là chặn oan)."]
    out = "\n".join(L) + "\n"
    open(os.path.join(KIEM, f"KET_QUA_K1{TAG}.md"), "w", encoding="utf-8").write(out)
    return out


def selftest() -> None:
    lo, hi = wilson(0, 240)
    assert lo == 0.0 and abs(hi - 0.0158) < 1e-3                     # 0/240 → cận trên ≈ 1,6 %
    lo, hi = wilson(15, 1000)
    assert abs(lo - 0.0091) < 1e-3 and abs(hi - 0.0246) < 1e-3
    c = {"ca": "goc", "mong_doi": "accept", "ly_do": None}
    assert score(c, {"decision": "nhan", "reasons": []}) == "đúng"
    assert score(c, {"decision": "04", "reasons": ["04 · “băng hẹp”"]}) == "chặn oan"
    assert score({"ca": "loai.nhac", "mong_doi": "04", "ly_do": "không phải ghi âm lời nói"}, {"decision": "nhan", "reasons": []}) == "bỏ lọt"
    s = {"ca": "snr.5dB", "mong_doi": "04", "ly_do": "ồn quá"}
    assert score(s, {"decision": "nhan", "reasons": []}) == "trục chưa kiểm"
    assert score(s, {"decision": "04", "reasons": ["04 · “quá ít tiếng nói”"]}) == "04 vì lý do khác"
    assert score(s, {"decision": "04", "reasons": ["04 · “ồn quá”"]}) == "trục chưa kiểm"
    assert score(c, {"loi": "x"}) == "lỗi chạy"
    assert score({"ca": "tunhien.en.nhac_nen", "mong_doi": "khong_chac", "ly_do": None}, {"decision": "04", "reasons": []}) == "không chấm (K1: 04)"
    print("chay_k1 selftest ok")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    a = sys.argv[1:] or ["run"]
    if a[0] == "run":
        run(int(a[a.index("--limit") + 1]) if "--limit" in a else None)
        print(report())
    elif a[0] == "report":
        print(report())
    else:
        selftest()
