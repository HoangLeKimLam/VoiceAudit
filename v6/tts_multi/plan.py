# -*- coding: utf-8 -*-
"""WP7b / B1 - KE HOACH SINH GIONG MAY nhieu may (03_chen_giong_may) cho bo ttsmulti_vi / ttsmulti_en.

Chon ban goc (host) + nguoi bi nhai giong + mau giong + loi moi, chia quota deu theo
  may (engine, vai tro train | heldout tu engines.json) x muc che giau (vung | trung_binh | ky) x thao tac x fold nguoi noi
  (v6.boundary.detect.person_fold = crc32(meta.person) % 5; meta.fold giu dinh nghia cu C.assign_fold([speaker])).
  thao tac  may_thay    thay 1-12 tu bang LOI MOI (so, ngay, ten, phu dinh, so tien... hop cau) doc bang giong may
            may_chen    chen 1-8 tu moi vao khoang giua hai tu
            may_ca_file ca tep do giong may doc lai (tung cau mot, <= 18 tu vi / 25 tu en moi phan)
  Moi doan sua <= 5 s (uoc luong = so am tiet x thoi luong am tiet trung vi cua chinh nguoi do), cach mep tep >= 0,5 s.
Nguon duoc phep (DECISIONS.md muc 2): host vi = vox, celeb ; en = edacc, ami, voxp (bo nguoi 'voxp:None' = khong ro danh tinh)
  MAC DINH BO celeb (DEFAULT_EXCLUDE): muc 2 vua liet ke Vietnam-Celeb vua cam nhai 'nguoi noi tieng' -> cho nguoi dung
  quyet dinh; bat lai bang --include-sources celeb.
  + VIVOS (vi) / LibriSpeech dev|test-clean (en) lam dich them cho may_ca_file (loi chuan). CAM vss / vsasv / vivoice / tep cao.
Mau giong: tu BAN GHI KHAC cua cung nguoi (khong bao gio tu chinh tep host), cat 3-10 s o ranh gioi tu, ghi <work>/refs/.
  tang mau giong (ref_stratum):
    khac_phien   biet chac khac phien (khac cuoc hop AMI / khac chuong sach LibriSpeech)
                 + voxp: ma bai phat bieu tu audio_id cua parquet VoxPopuli (khac bai = khac phien)
    khac_tep_xa  vox / celeb KHONG co ma video: hai tep cach >= FAR_GAP hang parquet TOAN CUC (cong don qua cac manh,
                 so hang doc tu metadata parquet) -> UOC LUONG khac video; kem bang chung row_gap + ltas_dist_db
    khac_tep_ke  cung nguon tren nhung cach < FAR_GAP hang, HOAC khong tinh duoc khoang cach (thieu parquet) -> coi la
                 CUNG video. Do 11/10: khoi hang cua moi nguoi celeb dai trung vi 8,5 hang (p90 27; buoc hang lien tiep
                 p99 = 6) -> gan nhu moi mau giong celeb roi vao tang nay; vox: buoc lien tiep p95 18, p99 59.
    cung_phien   biet chac cung phien (EdAcc: moi nguoi chi o mot cuoc hoi thoai; AMI cung cuoc hop; voxp cung bai
                 phat bieu; VIVOS mot phien doc)
  Tang CHINH chi dung khac_phien | khac_tep_xa (quyet dinh 2: nhai tu ban ghi KHAC). Tang SO SANH (~P_SAME muc, co
  want_same_session=True) chi dung cung_phien | khac_tep_ke; thieu thi roi ve tang chinh. Host khong co mau giong hop tang
  chinh (vd EdAcc) chi vao tang so sanh.
Chia (12/10, round_robin): tinh TRUOC nguon cung (eligibility: host nao co mau giong tang chinh / so sanh), roi chia
  VONG TRON qua cac o may x muc x thao tac (moi vong moi o mot muc, thu tu o xoay vong), moi o lay fold nguoi it muc
  nhat truoc, trong fold lay nguoi IT dung nhat truoc; host chi bi danh dau 'da dung' SAU khi tao xong muc. Tang so
  sanh <= P_SAME moi o (tran cung); tang chinh het -> o dung (khong bu bang tang so sanh). Nguon thieu -> moi o co
  ~so muc nhu nhau (quota tu thu nho; tom tat ghi underfilled, cell_min/max, supply theo fold). O RONG -> PlanError,
  khong ghi gi (tru --allow-empty-cells). Lap lai: plan / summary / queue / jobs cu doi sang _work/_old/<thoi diem>/;
  da co cand / qa / manifest -> tu choi tru --force-replan (doi ca chung, khong xoa).
Dau ra (thu muc bo <root>/data_labels/v6/<set>/_work/):
  plan.jsonl          moi dong = 1 muc (1 tep se sinh): host, thao tac, loi moi, cau mang (carrier >= 10 tu), mau giong, job
  jobs/<engine>.jsonl dung giao dien I3: {job_id, text, lang, ref_wav, ref_text, seed, n_candidates}
  plan_summary.json   bang quota engine x tier x op x fold, so tang mau giong, so nguoi, so LOCKBOX, nguon cung
CLI: python -m v6.tts_multi.plan --set ttsmulti_vi [--per-cell 20] [--n-cand 3] [--refs 1] [--root R] [--registry F]
     [--exclude-sources S1,S2] [--include-sources celeb] [--person-cap-frac 0.03] [--host-dirs hosts,hosts_v2]
     [--allow-empty-cells] [--force-replan]      (loi ke hoach -> in tom tat, ma thoat 2)
     (thuong goi qua: python -m v6.tts_multi.build_sets plan ...; nhe: chi doc json + cat wav mau giong)"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import re
import shutil
import sys
import time
import zlib
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE not in sys.path:
    sys.path.insert(0, BASE)
from v6 import common as C  # noqa: E402

SR = 16000
NAT = os.path.join(BASE, "data", "natural_v6")
REGISTRY = os.path.join(BASE, "v6", "tts_multi", "engines.json")
SETS = {"ttsmulti_vi": ("vi", "v6ttsmvi"), "ttsmulti_en": ("en", "v6ttsmen"), "vcmulti": ("vi+en", "v6vcm")}
HOST_SOURCES = {"vi": ("vox", "celeb"), "en": ("edacc", "ami", "voxp")}
FORBIDDEN = ("vss", "vsasv", "vivoice", "viet", "phoaudio", "scrape", "real_pilot", "yt")
OPS = ("may_thay", "may_chen", "may_ca_file")
TIERS = ("vung", "trung_binh", "ky")
OP_CODE = {"may_thay": "mt", "may_chen": "mc", "may_ca_file": "mf"}
TIER_CODE = {"vung": "vu", "trung_binh": "tb", "ky": "ky"}
EDGE_S, MAX_SEG_S, MAX_FRAC = 0.5, 5.0, 0.5
REF_DUR = (3.0, 10.0)
REF_TARGET_S = 7.0
CARRIER_MIN, CARRIER_MAX = 10, 20
PART_MAX = {"vi": 18, "en": 25}
P_SAME = 0.2                       # ty le muc co chu dich lay mau giong cung phien (tang so sanh)
FAR_GAP = 40                       # vox / celeb: cach >= 40 hang parquet toan cuc = uoc luong khac video (xem docstring)
SESSION_GAP = FAR_GAP              # ten cu (giu cho ma ngoai)
DEFAULT_EXCLUDE = ("celeb",)       # cho quyet dinh nguoi dung ve 'nguoi noi tieng' (DECISIONS.md muc 2)
PARQUET_GLOB = {"vox": os.path.join("voxvietnam", "data", "test-*.parquet"),
                "celeb": os.path.join("vietnam_celeb", "data", "test-*.parquet"),
                "voxp": os.path.join("en_voxpopuli", "en", "test-*.parquet")}
MAIN_STRATA = ("khac_phien", "khac_tep_xa")
SAME_STRATA = ("cung_phien", "khac_tep_ke")
STRATA = MAIN_STRATA + SAME_STRATA
SINGLE_SESSION = ("vivos",)        # nguon ma moi nguoi chi co mot phien ghi (cum = nguoi nhung that ra cung phien)
PERSON_CAP_FRAC = 0.03             # moi nguoi <= 3 % so muc cua bo (toi thieu 2)
EXTRA_FRAC = 0.25                  # ty le may_ca_file lay tu VIVOS / LibriSpeech
LOW_TIER_SCALE = 1 / 3            # may giong co san (needs_vc: MMS, Kokoro) = bac chat luong thap -> ~<= 10 % so muc
PAUSE_MIN = 0.15
# so dang ky du phong khi WP7a chua ghi engines.json (ten theo audit1_tts; engines.json that se thay the)
FALLBACK_ENGINES = {
    "vixtts": {"kind": "tts", "langs": ["vi"], "role": "train", "license": "CPML", "notes": "fallback"},
    "f5_vi": {"kind": "tts", "langs": ["vi"], "role": "train", "license": "CC-BY-4.0 zalopay", "notes": "fallback"},
    "vizipvoice": {"kind": "tts", "langs": ["vi"], "role": "train", "license": "Apache-2.0", "notes": "fallback"},
    "xtts_v2": {"kind": "tts", "langs": ["en"], "role": "train", "license": "CPML", "notes": "fallback"},
    "f5_en": {"kind": "tts", "langs": ["en"], "role": "train", "license": "CC-BY-NC-4.0", "notes": "fallback"},
    "zipvoice_distill": {"kind": "tts", "langs": ["en"], "role": "train", "license": "Apache-2.0", "notes": "fallback"},
    "vieneu": {"kind": "tts", "langs": ["vi", "en"], "role": "heldout", "license": "Apache-2.0", "notes": "fallback"},
    "seedvc": {"kind": "vc", "langs": ["vi", "en"], "role": "train", "license": "GPL-3.0", "notes": "fallback"},
    "openvoice_v2": {"kind": "vc", "langs": ["vi", "en"], "role": "train", "license": "MIT", "notes": "fallback"},
    "knnvc": {"kind": "vc", "langs": ["vi", "en"], "role": "heldout", "license": "MIT", "notes": "fallback",
              "ref_seconds": 300},
}

# ---------------- bo loi moi (trung tinh: so, ngay, ten hu cau, phu dinh, so tien) ----------------
LEX = {
    "vi": {
        "num": ["hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "mười lăm", "hai mươi", "ba mươi",
                "bốn mươi lăm", "một trăm", "hai trăm", "năm trăm", "một nghìn"],
        "unit": ["người", "ngày", "tuần", "tháng", "năm", "lần", "phút", "tiếng", "cuốn sách", "cây số"],
        "money": ["triệu đồng", "nghìn đồng"],
        "month": ["một", "hai", "ba", "tư", "năm", "sáu", "bảy", "tám", "chín", "mười", "mười một", "mười hai"],
        # o so co mien gia tri rieng (ngay 1-31, gio sang 5-11, so nho 2-10): tranh 'ngay mot tram thang chin'
        "day": ["một", "hai", "ba", "năm", "bảy", "mười", "mười hai", "mười lăm", "hai mươi", "hai mươi lăm",
                "ba mươi"],
        "hour": ["năm", "sáu", "bảy", "tám", "chín", "mười", "mười một"],
        "small": ["hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"],
        "name": ["Minh", "Lan", "Hùng", "Mai", "Tuấn", "Hoa", "Nam", "Thảo"],
        "place": ["Hà Nội", "Đà Nẵng", "Cần Thơ", "Huế", "Hải Phòng", "Nha Trang"],
        "time": ["hôm qua", "hôm nay", "ngày mai", "tuần trước", "tuần sau", "năm ngoái", "sáng nay", "tối qua"],
        "one": ["không", "chưa", "đã", "sẽ", "rất", "cũng", "vẫn", "mới", "luôn", "thường", "hai", "ba", "năm", "mười"],
        "tpl": ["{num}", "{num} {unit}", "{time}", "không phải", "chưa bao giờ", "rất nhiều", "một chút", "anh {name}",
                "chị {name}", "ở {place}", "khoảng {num} {unit}", "vào {time}", "không bao giờ quên", "đã đến {place}",
                "gần {num} {unit}", "khoảng {num} {unit} nữa", "vào ngày {day} tháng {month}", "anh {name} đã nói rằng",
                "tôi không đồng ý với điều đó", "chúng tôi đã gặp nhau ở {place}", "số tiền khoảng {num} {money}",
                "mất khoảng {small} {unit} để làm", "tôi đã ở {place} {small} {unit}",
                "và sau đó chúng tôi ở lại thêm {small} ngày nữa", "tôi nhớ là vào khoảng {hour} giờ sáng hôm đó",
                "lúc đó tôi chưa biết gì về chuyện này cả", "chị {name} nói là sẽ quay lại vào {time}"],
    },
    "en": {
        "num": ["two", "three", "four", "five", "seven", "ten", "twelve", "fifteen", "twenty", "thirty", "forty five",
                "a hundred", "two hundred", "a thousand"],
        "unit": ["people", "days", "weeks", "months", "years", "times", "minutes", "hours", "books", "miles"],
        "money": ["dollars", "pounds", "euros"],
        "month": ["may", "june", "july", "march", "october"],
        "ord": ["first", "second", "third", "tenth", "twentieth"],
        "hour": ["five", "six", "seven", "eight", "nine", "ten", "eleven"],
        "small": ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        "name": ["John", "Sarah", "Peter", "Emma", "David", "Laura"],
        "place": ["London", "Glasgow", "Leeds", "Dublin", "Boston", "Toronto"],
        "time": ["yesterday", "today", "tomorrow", "last week", "next week", "last year", "this morning", "on monday"],
        "one": ["not", "never", "always", "really", "also", "still", "just", "often", "two", "three", "ten"],
        "tpl": ["{num}", "{num} {unit}", "{time}", "not really", "quite a lot", "a little bit", "my friend {name}",
                "in {place}", "about {num} {unit}", "back in {place}", "i did not agree", "about {num} {unit} later",
                "on the {ord} of {month}", "{name} told me that", "i do not agree with that",
                "we met each other in {place}", "it cost about {num} {money}", "it took about {small} {unit} to finish",
                "and then we decided to stay for {small} more days", "i remember it was around {hour} in the morning",
                "at that time i did not know anything about it", "{name} said she would come back {time}"],
    },
}
_SLOT_RE = re.compile(r"\{(\w+)\}")
NUM_SLOTS = ("num", "day", "hour", "small")    # o chi so: doan bi thay co tu chi so -> chon mau co mot trong cac o nay


def norm_text(t: str, lang: str) -> str:
    """Chu thuong, bo dau cau va <unk>; vi GIU dau thanh (CER phan biet dau), en chi a-z va dau nhay."""
    t = re.sub(r"<[^>]*>|\[[^\]]*\]", " ", str(t).lower())
    t = re.sub(r"[^a-z' ]+", " ", t) if lang == "en" else re.sub(r"[^\w' ]+|_|\d", " ", t)
    return " ".join(t.split())


def n_syll(words: list[str], lang: str) -> int:
    """vi: moi tu (cach trong) = 1 am tiet; en: so cum nguyen am moi tu (>= 1)."""
    if lang != "en":
        return sum(len(w.split()) for w in words)
    return sum(max(1, len(re.findall(r"[aeiouy]+", w.lower()))) for w in " ".join(words).split())


def _fill(tpl: str, lex: dict, rng: random.Random) -> str:
    return _SLOT_RE.sub(lambda m: rng.choice(lex[m.group(1)]), tpl)


def new_phrase(lang: str, n: int, removed: list[str], rng: random.Random) -> list[str]:
    """Loi moi dai ~n tu (+-1) hop cau: doan bi thay co tu chi so -> mau co {num}; n = 1 -> tu don (phu dinh / pho tu /
    so); khac voi doan bi thay. Tra danh sach tu (tach theo khoang trong)."""
    lex = LEX[lang]
    nums = {w for s in NUM_SLOTS for x in lex.get(s, []) for w in x.split()}
    has_num = any(w.lower() in nums for w in removed)
    rem = " ".join(removed).lower()
    for _ in range(60):
        if n <= 1 and not has_num:
            out = [rng.choice(lex["one"])]
        else:
            pool = [t for t in lex["tpl"] if (not has_num or any("{" + s + "}" in t for s in NUM_SLOTS))]
            cand = [_fill(t, lex, rng).split() for t in rng.sample(pool, len(pool))]
            near = [c for c in cand if abs(len(c) - n) <= (0 if n <= 2 else 1)]
            out = near[0] if near else min(cand, key=lambda c: abs(len(c) - n))
        if " ".join(out).lower() != rem:
            return out
    return out


# ---------------- nguon du lieu ----------------
def _allowed(source: str, person: str) -> bool:
    s = str(source).lower()
    return not any(f in s for f in FORBIDDEN) and not str(person).endswith(":None")


HOST_KEEP = ("group", "lockbox_v2", "fold_v2", "role")   # truong phu cua kho host mo rong (WP5 hosts_v2) -> meta


def load_hosts(lang: str, sources: tuple | None = None, dirs: tuple | None = None) -> list[dict]:
    """hosts.jsonl (natural_v6) cua ngon ngu, chi nguon duoc phep; them 'wav' tuyet doi.
    dirs: cac thu muc con cua natural_v6 (mac dinh 'hosts' | 'hosts_en'); vd ('hosts', 'hosts_v2') them kho ban goc mo
    rong cua WP5 (v6.data.expand_hosts: nguoi MOI, mac dinh lockbox_v2) — cung dinh dang, ten tep '<nguon>_<parquet>_<dong>'.
    Tep trung ten giua cac kho -> giu ban dau tien."""
    keep = sources or HOST_SOURCES[lang]
    out, seen = [], set()
    for sub in dirs or (("hosts",) if lang == "vi" else ("hosts_en",)):
        d = os.path.join(NAT, sub)
        p = os.path.join(d, "hosts.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(f"chua co kho host {p} (vd hosts_v2: viec xep hang cua WP5 v6.data.expand_hosts hosts)")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["source"] not in keep or not _allowed(r["source"], r["person"]) or r["file"] in seen:
                    continue
                seen.add(r["file"])
                out.append(annotate_rows({**r, "lang": lang, "wav": os.path.join(d, r["source"], r["file"] + ".wav"),
                                          "extra": False, "pool": sub}))
    return out


_PQ_CACHE: dict[str, dict | None] = {}


def parquet_meta(source: str) -> dict | None:
    """{'offsets': {manh: hang dau toan cuc}, 'sessions': [ma phien theo hang] | None} cua parquet goc; khong co -> None.
    Chi doc metadata (so hang) + cot audio_id (voxp), khong doc am thanh."""
    if source in _PQ_CACHE:
        return _PQ_CACHE[source]
    pat = PARQUET_GLOB.get(source)
    files = sorted(glob.glob(os.path.join(NAT, pat))) if pat else []
    meta = None
    if files:
        try:
            import pyarrow.parquet as pq
            offs, tot, sess = {}, 0, ([] if source == "voxp" else None)
            for f in files:
                m = re.search(r"-(\d+)(?:-of-\d+)?$", os.path.splitext(os.path.basename(f))[0])
                pf = pq.ParquetFile(f)
                offs[int(m.group(1)) if m else len(offs)] = tot
                tot += pf.metadata.num_rows
                if sess is not None:
                    sess += [str(a).rsplit("_", 1)[0] for a in pf.read(columns=["audio_id"]).column(0).to_pylist()]
            meta = {"offsets": offs, "sessions": sess}
        except Exception as ex:                    # thieu pyarrow / tep hong -> khong co bang chung (bao thu: KE)
            print(f"[plan] parquet_meta({source}) loi: {ex}", file=sys.stderr)
            meta = None
    _PQ_CACHE[source] = meta
    return meta


def annotate_rows(h: dict) -> dict:
    """Them 'grow' (hang parquet toan cuc) va 'session' (voxp: ma bai phat bieu) neu suy duoc tu ten tep."""
    src = str(h.get("source"))
    ri = row_index(h.get("file", "")) if src in PARQUET_GLOB else None
    meta = parquet_meta(src) if ri else None
    if meta and ri[0] in meta["offsets"]:
        g = meta["offsets"][ri[0]] + ri[1]
        h["grow"] = g
        if meta["sessions"] is not None and g < len(meta["sessions"]):
            h["session"] = f"{src}:{meta['sessions'][g]}"
    return h


def load_vivos(limit_per_spk: int = 40) -> list[dict]:
    """VIVOS co loi chuan (data/vivos_labeled): moi cau = mot 'host' khong co moc tu (chi may_ca_file / dich VC)."""
    out = []
    for split in ("train", "test"):
        p = os.path.join(BASE, "data", "vivos_labeled", f"metadata_{split}.jsonl")
        if not os.path.exists(p):
            continue
        cnt: Counter = Counter()
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                spk = r["speaker_id"]
                if cnt[spk] >= limit_per_spk:
                    continue
                cnt[spk] += 1
                f = os.path.splitext(r["file"])[0]
                out.append({"file": f, "source": "vivos", "cluster": f"vivos:{spk}", "person": f"vivos:{spk}",
                            "dur": float(r["duration"]), "text": r["sentence"], "words": None, "ref": r["sentence"],
                            "lang": "vi", "wav": os.path.join(BASE, "data", "vivos_labeled", r["file"]), "extra": True})
    return out


def load_librispeech(limit_per_spk: int = 30) -> list[dict]:
    """LibriSpeech dev-clean + test-clean (loi chuan, cum = chuong sach). Thoi luong doc tu tep khi can (sf.info)."""
    root = os.path.join(BASE, "data", "librispeech", "LibriSpeech")
    out = []
    for sub in ("dev-clean", "test-clean"):
        for tr in sorted(glob.glob(os.path.join(root, sub, "*", "*", "*.trans.txt"))):
            spk, chap = os.path.basename(tr).split(".")[0].split("-")
            with open(tr, encoding="utf-8") as fh:
                rows = [ln.strip().split(" ", 1) for ln in fh if ln.strip()]
            for uid, txt in rows[:limit_per_spk]:
                out.append({"file": uid, "source": "librispeech", "cluster": f"ls:{spk}-{chap}", "person": f"ls:{spk}",
                            "dur": None, "text": txt.lower(), "words": None, "ref": txt, "lang": "en",
                            "wav": os.path.join(os.path.dirname(tr), uid + ".flac"), "extra": True})
    return out


def librispeech_gender() -> dict[str, str]:
    p = os.path.join(BASE, "data", "librispeech", "LibriSpeech", "SPEAKERS.TXT")
    out = {}
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            if line.startswith(";"):
                continue
            f = [x.strip() for x in line.split("|")]
            if len(f) >= 2 and f[0].isdigit():
                out[f"ls:{f[0]}"] = {"F": "female", "M": "male"}.get(f[1], "")
    return out


def paired_real(lang: str) -> dict[str, str]:
    """host_file -> id ban nguyen_ban goc trong natvi / naten (cap that-gia cung tep cho danh gia + ABX)."""
    mp = os.path.join(C.V6DATA, "natvi" if lang == "vi" else "naten", "manifest.json")
    if not os.path.exists(mp):
        return {}
    with open(mp, encoding="utf-8") as fh:
        man = json.load(fh)
    return {m["meta"]["host_file"]: m["id"] for m in man if m["kind"] == "nguyen_ban" and m["cfg"] == "goc"}


def person_fold(person: str) -> int:
    """= v6.boundary.detect.person_fold (crc32(person) % 5 qua C.assign_fold) — khong nap detect (nang)."""
    return C.assign_fold([str(person)])


def person_stats(hosts: list[dict], lang: str) -> dict[str, dict]:
    """Nguoi -> toc do am tiet (am tiet / s tieng noi, bo khoang lang >= 0,15 s), cac khoang lang, so tep, phut."""
    acc: dict[str, dict] = defaultdict(lambda: {"syll": 0, "speech": 0.0, "pauses": [], "n_files": 0, "minutes": 0.0,
                                                "rates": []})
    for h in hosts:
        st = acc[h["person"]]
        st["n_files"] += 1
        st["minutes"] += (h.get("dur") or 0.0) / 60
        ws = h.get("words")
        if not ws or len(ws) < 3:
            continue
        gaps = [b[1] - a[2] for a, b in zip(ws, ws[1:])]
        span = ws[-1][2] - ws[0][1]
        pz = [g for g in gaps if g >= PAUSE_MIN]
        sp = max(1e-3, span - sum(pz))
        ns = n_syll([w[0] for w in ws], lang)
        st["syll"] += ns
        st["speech"] += sp
        st["pauses"] += pz
        st["rates"].append(ns / sp)
    out = {}
    for p, st in acc.items():
        rate = st["syll"] / st["speech"] if st["speech"] > 0 else (4.5 if lang == "vi" else 4.0)
        out[p] = {"rate": round(rate, 3), "syll_dur": round(1 / rate, 4), "pauses": [round(x, 3) for x in st["pauses"]],
                  "n_files": st["n_files"], "minutes": round(st["minutes"], 2), "rates": [round(r, 3) for r in st["rates"]]}
    return out


# ---------------- mau giong ----------------
_ROW_RE = re.compile(r"-(\d+)(?:-of-\d+)?_(\d+)$")


def row_index(file: str) -> tuple[int, int] | None:
    """(so parquet, hang) tu ten tep host '<src>_test-00000-of-00038_00510' / 'voxp_test-00000_00012'; khong co -> None."""
    m = _ROW_RE.search(str(file))
    return (int(m.group(1)), int(m.group(2))) if m else None


def row_gap(a: str, b: str) -> int | None:
    """Khoang cach hang parquet giua hai tep CUNG manh (khac manh -> None = khong biet; ref_stratum coi la KE)."""
    ra, rb = row_index(a), row_index(b)
    if ra is None or rb is None or ra[0] != rb[0]:
        return None
    return abs(ra[1] - rb[1])


def host_gap(a: dict, b: dict) -> int | None:
    """Khoang cach hang parquet: theo hang toan cuc 'grow' neu ca hai co, khong thi cung manh (row_gap), khong -> None."""
    if a.get("grow") is not None and b.get("grow") is not None:
        return abs(int(a["grow"]) - int(b["grow"]))
    return row_gap(a["file"], b["file"])


def ref_stratum(host: dict, ref_host: dict) -> str:
    """khac_phien | cung_phien (biet phien: cum AMI/LibriSpeech, ma bai voxp) · khac_tep_xa | khac_tep_ke (cum == nguoi:
    uoc luong theo hang parquet toan cuc; khong tinh duoc -> KE, bao thu theo quyet dinh 2)."""
    if str(host.get("source")) in SINGLE_SESSION:
        return "cung_phien"
    if host.get("session") and ref_host.get("session"):
        return "khac_phien" if host["session"] != ref_host["session"] else "cung_phien"
    if host["cluster"] == host["person"] or ref_host["cluster"] == ref_host["person"]:
        g = host_gap(host, ref_host)
        return "khac_tep_xa" if g is not None and g >= FAR_GAP else "khac_tep_ke"
    return "khac_phien" if ref_host["cluster"] != host["cluster"] else "cung_phien"


def ltas_dist_db(a: np.ndarray, b: np.ndarray) -> float | None:
    """Khoang cach pho trung binh (dB, dai 1/3 quang tam 100-7000 Hz, bo muc trung binh) — bang chung phu 'khac phien'."""
    from scipy.signal import welch
    if len(a) < 2048 or len(b) < 2048:
        return None
    f, pa = welch(np.asarray(a, float), SR, nperseg=1024)
    _, pb = welch(np.asarray(b, float), SR, nperseg=1024)
    cs = 100 * 2 ** (np.arange(0, 19) / 3)
    da, db = [], []
    for fc in cs[cs < 7000]:
        m = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        if m.any():
            da.append(10 * np.log10(pa[m].mean() + 1e-14))
            db.append(10 * np.log10(pb[m].mean() + 1e-14))
    d = (np.asarray(da) - np.mean(da)) - (np.asarray(db) - np.mean(db))
    return round(float(np.sqrt(np.mean(d ** 2))), 2)


def ref_window(h: dict) -> tuple[float, float, str, str] | None:
    """(t0, t1, ref_text, nguon_loi) cua doan 3-10 s gan 7 s nhat, cat o ranh gioi tu; tep ngan co loi chuan -> ca tep."""
    dur = h.get("dur")
    if dur is None:
        dur = sf.info(h["wav"]).duration
        h["dur"] = dur
    if h.get("ref") and REF_DUR[0] <= dur <= REF_DUR[1]:
        return 0.0, float(dur), str(h["ref"]).lower(), "chuan"
    ws = h.get("words")
    if not ws:
        return (0.0, float(dur), str(h.get("text", "")), "chuan" if h.get("ref") else "asr") \
            if REF_DUR[0] <= dur <= REF_DUR[1] else None
    best = None
    for i in range(len(ws)):
        for j in range(i, len(ws)):
            d = ws[j][2] - ws[i][1]
            if d > REF_DUR[1] - 0.2:
                break
            if d >= REF_DUR[0] - 0.2:
                sc = abs(d - REF_TARGET_S)
                if best is None or sc < best[0]:
                    best = (sc, i, j)
    if best is None:
        return None
    _, i, j = best
    t0 = max(0.0, ws[i][1] - 0.1)
    t1 = min(float(dur), ws[j][2] + 0.1)
    return round(t0, 3), round(t1, 3), " ".join(w[0] for w in ws[i:j + 1]), "asr"


def write_ref(h: dict, win: tuple, ref_dir: str) -> dict:
    """Cat mau giong ra <ref_dir>/<ref_id>.wav (16 kHz mono). Tra mo ta ref (id, wav, text, nguon)."""
    t0, t1, text, how = win
    rid = f"{h['file']}_{int(t0 * 1000)}_{int(t1 * 1000)}"
    rid = re.sub(r"[^A-Za-z0-9_.\-]", "_", rid)
    p = os.path.join(ref_dir, rid + ".wav")
    if not os.path.exists(p):
        y, sr = sf.read(h["wav"], dtype="float32", always_2d=True)
        y = y.mean(1)
        if sr != SR:
            from scipy.signal import resample_poly
            g = math.gcd(SR, sr)
            y = resample_poly(y, SR // g, sr // g).astype(np.float32)
        os.makedirs(ref_dir, exist_ok=True)
        sf.write(p, y[int(t0 * SR):int(t1 * SR)], SR)
    return {"ref_id": rid, "wav": p, "text": text, "text_src": how, "src_file": h["file"], "src_cluster": h["cluster"],
            "dur": round(t1 - t0, 3)}


def _load_mono16(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(1)
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def pick_refs(host: dict, by_person: dict, n: int, want_same: bool, rng: random.Random, ref_dir: str) -> list[dict]:
    """n mau giong tu tep KHAC cua cung nguoi, CHI trong tang cho phep: want_same -> SAME_STRATA (tang so sanh),
    nguoc lai MAIN_STRATA (khac phien that hoac uoc luong). Thu tu: khac_phien / cung_phien truoc, roi tep xa nhat.
    Moi mau giong kem bang chung: stratum, row_gap (hang parquet), ltas_dist_db (pho host - mau giong)."""
    allowed = SAME_STRATA if want_same else MAIN_STRATA
    others = [(h, ref_stratum(host, h)) for h in by_person[host["person"]] if h["file"] != host["file"]]
    others = [(h, s) for h, s in others if s in allowed]
    rng.shuffle(others)
    big = 10 ** 6
    others.sort(key=lambda hs: (allowed.index(hs[1]),
                                0 if want_same else -(host_gap(host, hs[0]) or big)))
    out, y_host = [], None
    for h, s in others:
        if len(out) >= n:
            break
        w = ref_window(h)
        if w is None:
            continue
        r = write_ref(h, w, ref_dir)
        if y_host is None:
            y_host = _load_mono16(host["wav"])
        r.update(stratum=s, row_gap=host_gap(host, h), session=h.get("session"),
                 ltas_dist_db=ltas_dist_db(y_host, sf.read(r["wav"], dtype="float32")[0]))
        out.append(r)
    return out


# ---------------- chon doan sua ----------------
def _gap_mid(ws, k: int, dur: float, before: bool) -> float:
    if before:
        return ws[0][1] - 0.04 if k <= 0 else (ws[k - 1][2] + ws[k][1]) / 2
    return min(dur, ws[-1][2] + 0.04) if k >= len(ws) - 1 else (ws[k][2] + ws[k + 1][1]) / 2


def _clean(w: str) -> str:
    """Bo the <unk> / [..] dinh trong tu ASR (vd "<unk>di")."""
    return re.sub(r"<[^>]*>|\[[^\]]*\]", "", str(w)).strip()


def _span_len(op: str, rng: random.Random) -> int:
    if op == "may_chen":
        return rng.choice([1, 1, 2, 3, rng.randint(4, 8)])
    return rng.choices([1, 2, 3, rng.randint(4, 6), rng.randint(7, 12)], weights=[30, 20, 15, 20, 15])[0]


def pick_edit(h: dict, op: str, lang: str, st: dict, rng: random.Random) -> dict | None:
    """Doan sua (thay k..m / chen truoc tu j) + loi moi + cau mang chua no. None neu tep khong hop."""
    ws, dur = h.get("words"), float(h["dur"])
    if not ws or len(ws) < 6:
        return None
    syll_dur = st["syll_dur"]
    # may_chen: uu tien chen o KHOANG LANG (>= PAUSE_MIN, ~ ranh gioi cum tu) nhu ke tan cong that; khong co -> bat ky
    pauses = [j for j in range(2, len(ws) - 1) if ws[j][1] - ws[j - 1][2] >= PAUSE_MIN]
    for t in range(30):
        n = _span_len(op, rng)
        new = new_phrase(lang, n, [], rng) if op == "may_chen" else None
        at_pause = None
        if op == "may_chen":
            at_pause = bool(pauses) and t < 20
            j = rng.choice(pauses) if at_pause else rng.randint(2, len(ws) - 2)
            a = b = (ws[j - 1][2] + ws[j][1]) / 2
            k, m, removed = j, j - 1, []
        else:
            n2 = max(1, min(n, len(ws) - 3))
            k = rng.randint(1, len(ws) - 1 - n2)
            m = k + n2 - 1
            a, b = _gap_mid(ws, k, dur, True), _gap_mid(ws, m, dur, False)
            removed = [w[0] for w in ws[k:m + 1]]
            new = new_phrase(lang, n2, removed, rng)
        est = n_syll(new, lang) * syll_dur + 0.1
        if not (a >= EDGE_S and b <= dur - EDGE_S and est <= MAX_SEG_S and (b - a) <= MAX_SEG_S
                and max(b - a, est) <= MAX_FRAC * dur):
            continue
        c0, c1 = k, m + 1                         # cau mang: mo rong hai phia toi >= 10 tu
        while (c1 - c0) + len(new) - len(removed) < CARRIER_MIN and (c0 > 0 or c1 < len(ws)):
            if c0 > 0:
                c0 -= 1
            if (c1 - c0) + len(new) - len(removed) < CARRIER_MIN and c1 < len(ws):
                c1 += 1
        while (c1 - c0) + len(new) - len(removed) > CARRIER_MAX and (c0 < k or c1 > m + 1):
            if c1 > m + 1:
                c1 -= 1
            elif c0 < k:
                c0 += 1
        left = [c for c in (_clean(w[0]) for w in ws[c0:k]) if c]
        right = [c for c in (_clean(w[0]) for w in ws[m + 1:c1]) if c]
        carrier = left + new + right
        return {"k": k, "m": m, "a": round(a, 4), "b": round(b, 4), "removed": " ".join(removed),
                "inserted": " ".join(new), "n_words": len(new), "n_syll": n_syll(new, lang), "est_s": round(est, 3),
                "carrier": " ".join(carrier), "carrier_lo": len(left), "carrier_hi": len(left) + len(new) - 1,
                "carrier_c0": c0, "carrier_c1": c1, **({"at_pause": at_pause} if op == "may_chen" else {})}
    return None


def whole_parts(h: dict, lang: str) -> list[str]:
    """Loi ca tep tach thanh phan <= PART_MAX tu (tung cau mot; khong dung noi chunk ben trong may)."""
    raw = str(h.get("ref") or "") if h.get("extra") else str(h["text"])
    words = [c for c in (_clean(w) for w in raw.split()) if c]
    k = PART_MAX[lang]
    n = max(1, math.ceil(len(words) / k))
    size = math.ceil(len(words) / n)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] if words else []


# ---------------- so dang ky may ----------------
def load_registry(path: str | None = None) -> tuple[dict, str]:
    """engines.json (I3, do WP7a ghi) -> (dict, nguon). Chua co -> FALLBACK_ENGINES (ghi ro 'fallback')."""
    p = path or REGISTRY
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            return {k: v for k, v in json.load(fh).items() if not k.startswith("_") and isinstance(v, dict)}, p
    return dict(FALLBACK_ENGINES), "fallback"


def engines_for(reg: dict, kind: str, lang: str, skip_failed_smoke: bool = True) -> list[str]:
    """May cung loai + ngon ngu. Bo may co ket qua thu nhanh (smoke, do WP7a ghi) ok=False; may chua thu van giu."""
    def langs(e):
        v = e.get("langs") or []
        return [v] if isinstance(v, str) else list(v)

    def failed(e):
        sm = e.get("smoke")
        return skip_failed_smoke and isinstance(sm, dict) and sm.get("ok") is False
    return sorted(n for n, e in reg.items() if e.get("kind") == kind and (lang in langs(e) or "any" in langs(e))
                  and not failed(e))


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9+]", "", name).lower()[:24] or "eng"


# ---------------- lap ke hoach ----------------
def set_dirs(set_name: str, root: str | None = None) -> dict[str, str]:
    base = os.path.join(root, "data_labels", "v6", set_name) if root else os.path.join(C.V6DATA, set_name)
    w = os.path.join(base, "_work")
    return {"set": base, "work": w, "jobs": os.path.join(w, "jobs"), "refs": os.path.join(w, "refs"),
            "cand": os.path.join(w, "cand"), "qa": os.path.join(w, "qa"), "logs": os.path.join(w, "logs"),
            "plan": os.path.join(w, "plan.jsonl"), "summary": os.path.join(w, "plan_summary.json"),
            "manifest": os.path.join(base, "manifest.json")}


def _fold_pools(hosts: list[dict], rng: random.Random | None = None) -> dict[int, list[dict]]:
    """fold nguoi (person_fold) -> danh sach host; rng tuy chon de tron (thu tu chon do _Alloc.free quyet dinh)."""
    pools: dict[int, list[dict]] = defaultdict(list)
    for h in hosts:
        pools[person_fold(h["person"])].append(h)
    if rng is not None:
        for f in pools:
            rng.shuffle(pools[f])
    return pools


class PlanError(ValueError):
    """Ke hoach khong dat (o may x muc x thao tac RONG, hoac da co dau ra cu): KHONG ghi plan/jobs.
    .summary = tom tat (nguon cung, o rong) de nguoi dung quyet dinh (vd bat celeb, nang tran nguoi, them kho host)."""

    def __init__(self, msg: str, summary: dict):
        super().__init__(msg)
        self.summary = summary


def _refable(h: dict, cache: dict) -> bool:
    k = h["file"]
    if k not in cache:
        try:
            cache[k] = ref_window(h) is not None
        except (OSError, RuntimeError, ValueError):     # tep hong / thieu -> khong dung lam mau giong
            cache[k] = False
    return cache[k]


def eligibility(hosts: list[dict], by_person: dict[str, list[dict]]) -> dict[str, dict]:
    """file host -> {'main': co ban ghi khac lam mau giong tang CHINH, 'same': co mau giong tang SO SANH}.
    Chi metadata (+ sf.info khi thieu thoi luong): tinh TRUOC khi chia de biet nguon cung that su."""
    cache: dict[str, bool] = {}
    main, same = set(MAIN_STRATA), set(SAME_STRATA)
    out = {}
    for h in hosts:
        sts = {ref_stratum(h, o) for o in by_person[h["person"]] if o["file"] != h["file"] and _refable(o, cache)}
        out[h["file"]] = {"main": bool(sts & main), "same": bool(sts & same)}
    return out


def round_robin(cells: list, quota: dict, fill, p_same: float = 0.0) -> dict:
    """Chia VONG TRON qua cac o: moi vong moi o (con quota, chua CAN) lay MOT muc; thu tu o xoay theo vong -> nguon
    khan hiem chia deu cho moi may / muc / thao tac (truoc 12/10: may dau bang chu cai lay het host).
    Tang so sanh la TRAN CUNG: muc thu n cua o chi xin cung phien khi floor(p_same * n) tang (=> <= p_same moi o);
    xin khong duoc -> lay tang chinh; tang chinh het nguon -> o CAN (khong bu bang tang so sanh).
    fill(cell, want_same, vong) -> muc | None. Tra {'made', 'same', 'dead', 'rounds'}."""
    made, same, dead = Counter(), Counter(), set()
    rnd = 0
    while cells:
        prog = False
        k = rnd % len(cells)
        for cell in cells[k:] + cells[:k]:
            if cell in dead or made[cell] >= quota[cell]:
                continue
            want = p_same > 0 and math.floor(p_same * (made[cell] + 1) + 1e-9) > same[cell]
            it = fill(cell, want, rnd)
            if it is None and want:
                want = False
                it = fill(cell, False, rnd)
            if it is None:
                dead.add(cell)
                continue
            made[cell] += 1
            same[cell] += int(want)
            prog = True
        rnd += 1
        if not prog:
            break
    return {"made": made, "same": same, "dead": dead, "rounds": rnd}


class _Alloc:
    """Trang thai chia chung (TTS + VC): tep / nguoi da dung, dem theo o / may / fold, (tep, loai) khong hop."""

    MAX_TRY = 40                                   # so host thu toi da moi fold moi o truoc khi sang fold khac

    def __init__(self, cap: int, seed: int):
        self.cap, self.seed = cap, seed
        self.used_file: set = set()
        self.used_person: Counter = Counter()
        self.cell_fold: dict = defaultdict(Counter)
        self.eng_fold: dict = defaultdict(Counter)
        self.all_fold: Counter = Counter()
        self.bad: set = set()

    def folds(self, cell, eng: str, rnd: int) -> list[int]:
        """Fold it muc nhat trong o truoc (roi trong may, roi ca bo; hoa -> xoay theo vong)."""
        return sorted(range(C.N_FOLDS), key=lambda f: (self.cell_fold[cell][f], self.eng_fold[eng][f], self.all_fold[f],
                                                         (f - rnd) % C.N_FOLDS))

    def free(self, pool: list[dict], ok, major=lambda h: 0, minor=lambda h: 0) -> list[dict]:
        """Host con dung duoc (chua dung, nguoi duoi tran, ok(h)), xep: major(h) (vd uu tien kho phu cho may_ca_file),
        nguoi IT dung truoc (trai deu nguoi), minor(h) (vd co ban that ghep cap), crc32 co dinh."""
        c = [h for h in pool if h["file"] not in self.used_file and self.used_person[h["person"]] < self.cap and ok(h)]
        c.sort(key=lambda h: (major(h), self.used_person[h["person"]], minor(h),
                              zlib.crc32(f"{self.seed}|{h['file']}".encode())))
        return c[:self.MAX_TRY]

    def take(self, h: dict, cell, eng: str, f: int) -> None:
        """Danh dau host da dung — CHI goi sau khi muc da tao xong (mau giong + doan sua hop le)."""
        self.used_file.add(h["file"])
        self.used_person[h["person"]] += 1
        self.cell_fold[cell][f] += 1
        self.eng_fold[eng][f] += 1
        self.all_fold[f] += 1


def archive_work(D: dict, force: bool = False, extra_dirs: list[str] | None = None) -> str | None:
    """Truoc khi ghi ke hoach moi: DOI (khong xoa) plan.jsonl, plan_summary.json, queue.json, jobs/*.jsonl (ca hybrid_*)
    + extra_dirs (vd vcsrc/ cua vc_sets) sang <work>/_old/<thoi diem>/ -> queue_cmds (glob jobs/*.jsonl) khong vo
    phai job cua ke hoach cu.
    Da co dau ra (cand/, qa/, manifest.json, thu muc tep da ghep) -> TU CHOI tru force (id muc dem lai tu 0: ung vien
    cu se bi nham la cua job moi); force -> doi ca chung sang _old. Tra thu muc _old hoac None."""
    heavy = [p for p in (D["cand"], D["qa"]) if os.path.isdir(p) and any(os.scandir(p))]
    if os.path.exists(D["manifest"]):
        heavy.append(D["manifest"])
        heavy += [os.path.join(D["set"], x) for x in sorted(os.listdir(D["set"]))
                  if x != "_work" and os.path.isdir(os.path.join(D["set"], x))]
    if heavy and not force:
        raise PlanError(f"da co dau ra cua ke hoach cu ({len(heavy)} muc, vd {heavy[0]}) -> --force-replan "
                        "(doi sang _work/_old, khong xoa)", {"heavy": heavy})
    moves = [p for p in (D["plan"], D["summary"], os.path.join(D["work"], "queue.json")) if os.path.exists(p)]
    moves += sorted(glob.glob(os.path.join(D["jobs"], "*.jsonl"))) + (heavy if force else [])
    moves += [p for p in extra_dirs or [] if os.path.isdir(p)]
    if not moves:
        return None
    dst = base = os.path.join(D["work"], "_old", time.strftime("%m%d_%H%M%S"))
    k = 0
    while os.path.exists(dst):                       # hai lan luu tru trong cung mot giay: khong ghi de ban truoc
        k += 1
        dst = f"{base}_{k}"
    for p in moves:
        q = os.path.join(dst, os.path.relpath(p, D["set"]))
        os.makedirs(os.path.dirname(q), exist_ok=True)
        shutil.move(p, q)
    return dst


def _quota(reg: dict, eng: str, per_cell: int) -> int:
    return per_cell if not reg.get(eng, {}).get("needs_vc") else max(1, math.ceil(per_cell * LOW_TIER_SCALE))


def _whole_ok(h: dict) -> bool:
    """Host dung duoc cho may_ca_file: co loi (ASR / chuan), <= 25 s (doc thoi luong neu thieu)."""
    if not (h.get("ref") if h.get("extra") else h.get("text")):
        return False
    try:
        h["dur"] = float(h.get("dur") or sf.info(h["wav"]).duration)
    except (OSError, RuntimeError):
        return False
    return h["dur"] <= 25


def _tts_item(h: dict, cell: tuple, want: bool, ctx: dict) -> tuple[dict, list] | None:
    """Mot muc (mau giong o tang yeu cau + doan sua / cac phan ca tep) hoac None; danh dau ly do khong hop."""
    eng, tier, op = cell
    lang, reg, rng = ctx["lang"], ctx["reg"], ctx["rng"]
    refs = pick_refs(h, ctx["by_person"], ctx["n_refs"], want, rng, ctx["D"]["refs"])
    if not refs:
        ctx["elig"][h["file"]]["same" if want else "main"] = False
        return None
    if op == "may_ca_file":
        texts = whole_parts(h, lang)
        edit = {"parts_text": texts} if texts else None
    else:
        edit = pick_edit(h, op, lang, ctx["stats"][h["person"]], rng)
        texts = [edit["carrier"]] if edit else []
    if not edit:
        ctx["alloc"].bad.add((h["file"], op == "may_ca_file"))
        return None
    iid = f"{OP_CODE[op]}{TIER_CODE[tier]}_{slug(eng)}_{ctx['counter']:05d}"
    ctx["counter"] += 1
    parts, jobs = [], []
    for pi, txt in enumerate(texts):
        jids = []
        for ri, r in enumerate(refs):
            jid = f"{iid}_p{pi}_r{ri}"
            jobs.append({"job_id": jid, "text": txt, "lang": lang, "ref_wav": r["wav"], "ref_text": r["text"],
                         "seed": zlib.crc32(f"{ctx['seed']}|{jid}".encode()) % 100000, "n_candidates": ctx["n_cand"]})
            jids.append(jid)
        parts.append({"part": pi, "text": txt, "jobs": jids})
    host = {k: h.get(k) for k in ("file", "source", "cluster", "person", "dur", "text", "words", "wav", "ref", "info",
                                  "extra")}
    host.update({k: h[k] for k in ("pool",) + HOST_KEEP if k in h})
    ent = reg.get(eng, {})
    item = {"item_id": iid, "set": ctx["set"], "prefix": ctx["prefix"], "lang": lang, "op": op, "tier": tier,
            "engine": eng, "role": ent.get("role", "train"), "host": host, "edit": edit, "parts": parts, "refs": refs,
            "ref_stratum": refs[0]["stratum"], "want_same_session": want, "person_fold": person_fold(h["person"]),
            "fold": C.assign_fold([h["cluster"]]), "paired_real": ctx["pairs"].get(h["file"]),
            "seed": zlib.crc32(f"{ctx['seed']}|{iid}".encode()) % 100000, "clone": not ent.get("needs_vc", False),
            "family": ent.get("family", eng)}
    return item, jobs


def supply_stats(elig: dict, hosts: list[dict]) -> dict:
    """Nguon cung tang chinh / so sanh theo fold nguoi (de bao cao khi ke hoach thieu)."""
    out: dict = {"main_hosts": Counter(), "same_hosts": Counter(), "main_persons": defaultdict(set)}
    for h in hosts:
        e, f = elig.get(h["file"]), person_fold(h["person"])
        if not e:
            continue
        if e["main"]:
            out["main_hosts"][f] += 1
            out["main_persons"][f].add(h["person"])
        if e["same"]:
            out["same_hosts"][f] += 1
    return {"main_hosts_by_fold": dict(sorted(out["main_hosts"].items())),
            "same_hosts_by_fold": dict(sorted(out["same_hosts"].items())),
            "main_persons_by_fold": {f: len(v) for f, v in sorted(out["main_persons"].items())},
            "main_persons": len(set().union(*out["main_persons"].values())) if out["main_persons"] else 0}


def make_plan(set_name: str, per_cell: int = 20, n_cand: int = 3, n_refs: int = 1, seed: int = 2026,
              root: str | None = None, registry: str | None = None, engines: list[str] | None = None,
              hosts: list[dict] | None = None, extra: list[dict] | None = None, ops: tuple = OPS,
              tiers: tuple = TIERS, pairs: dict | None = None, exclude_sources: tuple = DEFAULT_EXCLUDE,
              person_cap_frac: float = PERSON_CAP_FRAC, allow_empty_cells: bool = False, force: bool = False,
              host_dirs: tuple | None = None) -> dict:
    """Lap ke hoach cho mot bo ttsmulti_<lang>: ghi plan.jsonl + jobs/<engine>.jsonl + plan_summary.json. Tra tom tat.
    Chia VONG TRON (round_robin) qua cac o may x muc x thao tac, moi o lay fold it muc nhat truoc; nguon thieu ->
    moi o cung co ~so muc nhu nhau (quota TU THU NHO theo nguon cung tang chinh, bao 'underfilled' + supply).
    O nao RONG -> PlanError (khong ghi gi) tru allow_empty_cells. Ghi de ke hoach cu: archive_work (doi sang _old).
    hosts / extra truyen vao de kiem thu (mac dinh doc natural_v6 [+ host_dirs] + VIVOS / LibriSpeech).
    exclude_sources: bo han nguon (mac dinh celeb, xem DEFAULT_EXCLUDE)."""
    lang, prefix = SETS[set_name]
    if lang not in ("vi", "en"):
        raise ValueError(f"{set_name}: plan chi cho bo ttsmulti_vi / ttsmulti_en (VC: vc_sets)")
    rng = random.Random(seed)
    D = set_dirs(set_name, root)
    reg, reg_src = load_registry(registry)
    engs = engines or engines_for(reg, "tts", lang)
    if not engs:
        raise ValueError(f"khong co may TTS nao cho {lang} trong {reg_src}")
    hosts = hosts if hosts is not None else load_hosts(lang, dirs=host_dirs)
    extra = extra if extra is not None else (load_vivos() if lang == "vi" else load_librispeech())
    hosts = [h for h in hosts if h["source"] not in exclude_sources]
    extra = [h for h in extra if h["source"] not in exclude_sources]
    bad = [h["file"] for h in hosts + extra if not _allowed(h["source"], h["person"])]
    if bad:
        raise ValueError(f"nguon cam trong danh sach host: {bad[:3]}")
    by_person: dict[str, list[dict]] = defaultdict(list)
    for h in hosts + extra:
        by_person[h["person"]].append(h)
    multi = [h for h in hosts + extra if len(by_person[h["person"]]) >= 2]
    elig = eligibility(multi, by_person)
    pairs = paired_real(lang) if pairs is None else pairs
    seg = _fold_pools([h for h in multi if not h.get("extra") and h.get("words") and len(h["words"]) >= 6])
    whole = _fold_pools([h for h in multi if _whole_ok(h)])
    cells = [(e, t, o) for e in engs for t in tiers for o in ops]
    quota = {c: _quota(reg, c[0], per_cell) for c in cells}
    n_total = sum(quota.values())
    cap = max(2, math.ceil(person_cap_frac * n_total))
    A = _Alloc(cap, seed)
    ctx = {"lang": lang, "reg": reg, "rng": rng, "by_person": by_person, "n_refs": n_refs, "D": D, "elig": elig,
           "stats": person_stats(hosts + extra, lang), "alloc": A, "counter": 0, "seed": seed, "n_cand": n_cand,
           "set": set_name, "prefix": prefix, "pairs": pairs}
    items, jobs = [], defaultdict(list)
    os.makedirs(D["refs"], exist_ok=True)

    def fill(cell, want, rnd):
        eng, _, op = cell
        is_whole = op == "may_ca_file"
        prefer_x = is_whole and rng.random() < EXTRA_FRAC
        st = "same" if want else "main"
        for f in A.folds(cell, eng, rnd):
            for h in A.free((whole if is_whole else seg).get(f, []),
                            lambda x: elig[x["file"]][st] and (x["file"], is_whole) not in A.bad,
                            lambda x: (bool(x.get("extra")) != prefer_x) if is_whole else 0,
                            lambda x: x["file"] not in pairs):
                got = _tts_item(h, cell, want, ctx)
                if got:
                    A.take(h, cell, eng, f)
                    items.append(got[0])
                    jobs[eng] += got[1]
                    return got[0]
        return None
    rr = round_robin(cells, quota, fill, P_SAME)
    summ = summarize(items, reg_src, n_total, cap)
    key = [f"{e}|{t}|{o}" for e, t, o in cells]
    empty = [k for k in key if not summ["cells"].get(k)]
    summ.update(engines=engs, exclude_sources=list(exclude_sources), per_cell=per_cell, n_cand=n_cand, seed=seed,
                far_gap=FAR_GAP, p_same=P_SAME, host_dirs=list(host_dirs or ()), underfilled=len(items) < n_total,
                empty_cells=empty, cell_min=min((summ["cells"].get(k, 0) for k in key), default=0),
                cell_max=max((summ["cells"].get(k, 0) for k in key), default=0), rounds=rr["rounds"],
                ref_main_share=round(sum(it["ref_stratum"] in MAIN_STRATA for it in items) / max(1, len(items)), 3),
                supply=supply_stats(elig, multi))
    if empty and not allow_empty_cells:
        raise PlanError(f"{set_name}: {len(empty)} o rong (vd {empty[:3]}); nguon cung tang chinh {summ['supply']} "
                        "-> bat nguon (--include-sources celeb), nang --person-cap-frac, them --host-dirs, hoac "
                        "--allow-empty-cells", summ)
    summ["archived_to"] = archive_work(D, force)
    os.makedirs(D["jobs"], exist_ok=True)
    write_plan(D["plan"], items)
    for eng, js in jobs.items():
        with open(os.path.join(D["jobs"], f"{eng}.jsonl"), "w", encoding="utf-8") as fh:
            for j in js:
                fh.write(json.dumps(j, ensure_ascii=False) + "\n")
    with open(D["summary"], "w", encoding="utf-8") as fh:
        json.dump(summ, fh, ensure_ascii=False, indent=1)
    return summ


def resolve_excludes(exclude: str | None, include: str = "") -> tuple:
    """Chuoi CLI -> tuple nguon bi bo: exclude None = DEFAULT_EXCLUDE; tru di cac nguon trong include."""
    ex = DEFAULT_EXCLUDE if exclude is None else tuple(filter(None, exclude.split(",")))
    inc = set(filter(None, (include or "").split(",")))
    return tuple(x for x in ex if x not in inc)


def summarize(items: list[dict], reg_src: str, n_target: int, cap: int) -> dict:
    cell = Counter(f"{it['engine']}|{it['tier']}|{it['op']}" for it in items)
    lock: Counter = Counter()                  # ly do LOCKBOX gop theo loai (english_speaker, ami_TS3003...)
    for it in items:
        r = C.lockbox_reason({"id": it["item_id"], "meta": {"speaker": it["host"]["cluster"], "lang": it["lang"],
                                                           "host_file": it["host"]["file"]}})
        lock[(r or "mo").split(":")[0]] += 1
    per_p = Counter(it["host"]["person"] for it in items)
    return {"n_items": len(items), "n_target": n_target, "n_jobs": sum(len(p["jobs"]) for it in items for p in it["parts"]),
            "registry": reg_src, "person_cap": cap, "cells": dict(sorted(cell.items())),
            "folds": dict(sorted(Counter(it["person_fold"] for it in items).items())),
            "fold_x_engine": dict(sorted(Counter(f"{it['engine']}|{it['person_fold']}" for it in items).items())),
            "roles": dict(Counter(it["role"] for it in items)),
            "ref_stratum": dict(Counter(it["ref_stratum"] for it in items)),
            "comparison_share": round(sum(bool(it.get("want_same_session")) for it in items) / max(1, len(items)), 3),
            "sources": dict(Counter(it["host"]["source"] for it in items)),
            "persons": len(per_p), "max_per_person": max(per_p.values(), default=0),
            "max_person_share": round(max(per_p.values(), default=0) / max(1, len(items)), 3),
            "paired_real": sum(1 for it in items if it.get("paired_real")),
            "lockbox": dict(lock)}


PATH_KEYS = ("wav", "src_wav", "ref_wav", "target_ref_wav")


def abs_paths(x):
    """Truong duong dan TUONG DOI (ke hoach dong goi tu may goc, v6.data.goi_lane) -> tuyet doi theo goc kho nay; duong tuyet doi giu nguyen (ham thuan)."""
    if isinstance(x, dict):
        return {k: os.path.join(BASE, v) if k in PATH_KEYS and isinstance(v, str) and v and not os.path.isabs(v) else abs_paths(v)
                for k, v in x.items()}
    if isinstance(x, list):
        return [abs_paths(v) for v in x]
    return x


def read_plan(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [abs_paths(json.loads(line)) for line in fh if line.strip()]


def write_plan(path: str, items: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def add_plan_args(ap: argparse.ArgumentParser) -> None:
    """Tuy chon ke hoach dung chung cho plan / vc_sets / build_sets."""
    ap.add_argument("--exclude-sources", default=None, help="bo nguon, cach dau phay (mac dinh: celeb)")
    ap.add_argument("--include-sources", default="", help="bat lai nguon bi bo mac dinh (vd celeb)")
    ap.add_argument("--person-cap-frac", type=float, default=PERSON_CAP_FRAC)
    ap.add_argument("--host-dirs", default=None, help="kho host trong natural_v6, cach dau phay (vd hosts,hosts_v2)")
    ap.add_argument("--allow-empty-cells", action="store_true", help="van ghi ke hoach khi co o rong (bao trong tom tat)")
    ap.add_argument("--force-replan", action="store_true",
                    help="da co cand/qa/manifest cua ke hoach cu -> doi sang _work/_old roi lap lai (khong xoa)")


def plan_kwargs(a: argparse.Namespace) -> dict:
    return {"exclude_sources": resolve_excludes(a.exclude_sources, a.include_sources),
            "person_cap_frac": a.person_cap_frac, "allow_empty_cells": a.allow_empty_cells, "force": a.force_replan,
            "host_dirs": tuple(filter(None, a.host_dirs.split(","))) if a.host_dirs else None}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Ke hoach sinh giong may nhieu may (WP7b)")
    ap.add_argument("--set", required=True, choices=["ttsmulti_vi", "ttsmulti_en"])
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--n-cand", type=int, default=3)
    ap.add_argument("--refs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--root", default=None)
    ap.add_argument("--registry", default=None)
    ap.add_argument("--engines", default=None, help="danh sach may, cach dau phay (mac dinh: moi may tts cua ngon ngu)")
    add_plan_args(ap)
    a = ap.parse_args()
    try:
        s = make_plan(a.set, a.per_cell, a.n_cand, a.refs, a.seed, a.root, a.registry,
                      a.engines.split(",") if a.engines else None, **plan_kwargs(a))
    except PlanError as ex:
        print(json.dumps({"loi": str(ex), "summary": ex.summary}, ensure_ascii=False, indent=1, default=str))
        sys.exit(2)
    print(json.dumps(s, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
