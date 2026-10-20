# -*- coding: utf-8 -*-
"""Hang so dung chung cho v6.io — nhe (chi numpy), de schema/report dung ma khong nap torch."""
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE, "models", "v6")
TMP_DIR = os.path.join(BASE, ".tmp_v6")
V4_DIR = os.path.join(BASE, "data_labels", "realistic_v4")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
SCHEMA_ID = "voiceaudit.v6.ho_so/1"
ID_PREFIX = "v6io_"            # moi ban ghi do CLI tao ra: tien to rieng, khong trung cache dung chung

# cau hinh v5 da kiem (run_pipeline_v5/v5b): phai dat TRUOC khi import run_pipeline_*
PIPE_ENV = {"TAGGER": "fuse", "VERIF": "oof", "DECIDE": "calib", "EDGE10": "1", "DROP_CFLUX": "1",
            "KEEP_FPR": "0.10", "USE_GATE": "1"}

LOCAL = ["02_nen_cuc_bo", "02_giong_cuc_bo", "03_ghep_giong_that", "03_chen_giong_may"]
GLOBAL = ["02_nen_toan_cuc", "02_giong_toan_cuc"]
CLS = ["none"] + LOCAL
L04 = "04_khong_du_co_so"
DROP = "_bo"

# level2 -> (loai, pham vi, muc do) — Scope: muc do CO DINH theo thao tac
OP = {"02_nen_toan_cuc": ("khử ồn", "cả file", "vô hại"),
      "02_nen_cuc_bo": ("khử ồn", "một đoạn", "đáng ngờ"),
      "02_giong_toan_cuc": ("đổi giọng", "cả file", "sai lệch"),
      "02_giong_cuc_bo": ("đổi giọng", "một đoạn", "sai lệch"),
      "03_ghep_giong_that": ("ghép lời thật", "một đoạn", "sai lệch"),
      "03_chen_giong_may": ("chèn giọng máy", "một đoạn", "sai lệch nặng")}
SEV = {"02_nen_toan_cuc": 1, "02_nen_cuc_bo": 2, "02_giong_toan_cuc": 3, "02_giong_cuc_bo": 3,
       "03_ghep_giong_that": 4, "03_chen_giong_may": 5}
L1_NAME = {"01": "Không chỉnh sửa", "02": "AI chỉnh sửa hậu kỳ", "03": "Ghép hoặc chèn đoạn", "04": "Không đủ cơ sở"}
SEV_COLOR = {"vô hại": "#3b8f5a", "đáng ngờ": "#c98a0b", "sai lệch": "#d0601b", "sai lệch nặng": "#8e1b1b"}

# hang so co dinh cua dac ta (#bounds, #bind, #tree)
EDGE_S = 0.3                 # 0,3 s dau va cuoi khong soi
SCOPE_FRAC = 0.5             # mot doan <= 50 % do dai tep
CHUNK_S, CHUNK_STEP, LONG_S = 16.0, 14.0, 20.0
MAX_LOCAL_PER_CHUNK = 2
OVERLAP_MAX = 0.3
STRONG = 0.80
NEN_CB_WITH_TC = 0.9
G_HI, G_LO = 0.6, 0.4
WARN_EDGE = "0,3 s đầu và cuối không được soi"
WARN_MORE = "có thể còn chỗ sửa khác (tối đa 2 đoạn mỗi khúc 16 s, cộng các thao tác cả file)"
EXIT_OK, EXIT_REJECT, EXIT_04, EXIT_ERR = 0, 2, 3, 4

# tuyen bo gioi han (#io: bat buoc 3 cau dau)
LIMITS = ["Sàng lọc, không phải giám định tư pháp.",
          "Nhãn 01 nghĩa là không thấy dấu vết của 6 thao tác được soi, không phải chứng nhận bản thật.",
          "Các điều kiện ghi mà hệ thống không tự kiểm được: ghi một lần liên tục, một thiết bị, khoảng cách tới micro "
          "(≤ 2 m, không đổi xa–gần đột ngột) — nếu không đạt, chỗ tạm dừng hay đổi thiết bị có thể bị báo như mối ghép.",
          "Không chép lời, không nói kiểu con (thay hay chèn), công cụ đã dùng, nội dung đúng hay sai, hay người nói là ai.",
          "Mốc thời gian là một vùng, không cam kết đúng từng âm tiết; 0,3 s đầu và cuối tệp không được soi.",
          "Hệ thống soi 8 thao tác (khử ồn, đổi giọng, ghép lời thật, xoá lời, giọng máy; cả tệp hoặc một đoạn). Cắt đầu–cuối và đổi âm lượng cả tệp "
          "không tính là chỉnh sửa; thao tác khác không có nhãn riêng và không được cam kết phát hiện.",
          "Phiên bản thử nghiệm: chưa có cổng sàng lọc và chưa có bộ kiểm chứng độc lập; giọng máy một đoạn và cả tệp chưa được phát hiện thành loại riêng.",
          "VoiceAudit v9 là bản THỬ NGHIỆM: mô hình học trên dữ liệu mô phỏng tiếng Việt và tiếng Anh; "
          "số đo so với tệp thật lấy từ kho tệp tự nhiên mô phỏng, chưa phải ghi âm ngoài đời."]
LIMITS_REQUIRED = LIMITS[:3]


def setup_env() -> None:
    """Dat cau hinh pipeline v5 (neu chua dat) + duong dan repo. Goi truoc moi import run_pipeline_*."""
    for k, v in PIPE_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("HF_HOME", os.path.join(BASE, ".hf_cache"))
    if BASE not in sys.path:
        sys.path.insert(0, BASE)


def env_snapshot() -> dict:
    return {k: os.environ.get(k) for k in PIPE_ENV}


def mmss(t: float) -> str:
    """Giay -> 'mm:ss,cc' (dau phay thap phan kieu Viet)."""
    t = max(0.0, float(t))
    cs = int(np.floor(t * 100 + 0.5 + 1e-9))          # lam tron nua len (1,125 -> 1,13)
    m, rest = divmod(cs, 6000)
    return f"{m:02d}:{rest // 100:02d},{rest % 100:02d}"


def vn(x: float, d: int = 2, sign: bool = False) -> str:
    """So kieu Viet: 0,86 · −44,5 (dau tru that)."""
    if x is None or not np.isfinite(x):
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s.replace("-", "−").replace(".", ",")


def overlap(a: tuple, b: tuple) -> float:
    """Ti le chong lan tren doan ngan hon (nhu run_pipeline_eval.overlap)."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return inter / max(1e-6, min(a[1] - a[0], b[1] - b[0]))


def chunk_starts(dur: float) -> list:
    """Diem bat dau cac khuc 16 s chong 2 s (cung cong thuc v5b.decide_any)."""
    return [float(s) for s in np.arange(0.0, max(dur - 2.0, 1e-6), CHUNK_STEP)]


def l1_of(rows: list) -> str:
    """Nhan cap 1 = ham cua cac dong cap 2 con lai (K8)."""
    s = {x["level2"][:2] for x in rows if x["level2"] != DROP}
    return "03" if "03" in s else "02" if "02" in s else "04" if "04" in s else "01"
