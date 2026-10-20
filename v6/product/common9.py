# -*- coding: utf-8 -*-
"""Mo rong hang so v6.io cho DAC TA 27/09 (8 thao tac): them 03 XOA LOI (mot doan) va 03 GIONG MAY CA FILE; doi ten nhan 02 'Chinh sua hau ky' (da bo chu AI).
Cap nhat TAI CHO (list / dict dung chung) nen moi module v6.io da nap deu thay; PHAI goi patch() TRUOC KHI nap v6.io.schema / v6.io.render / v6.io.report
(schema dung GLOBAL + LOCAL luc nap). Khong dong vao ma v5 khac: chi them cuoi danh sach."""
from __future__ import annotations

from v6.io import common as C

NEW_LOCAL = ["03_xoa_loi"]
NEW_GLOBAL = ["03_giong_may_toan_cuc"]
NEW_OP = {"03_xoa_loi": ("xoá lời", "một đoạn", "sai lệch"), "03_giong_may_toan_cuc": ("giọng máy", "cả file", "sai lệch nặng")}
NEW_SEV = {"03_xoa_loi": 4, "03_giong_may_toan_cuc": 5}
LANES = ["xoá lời"]


def patch() -> None:
    for lv in NEW_LOCAL:
        if lv not in C.LOCAL:
            C.LOCAL.append(lv)
            C.CLS.append(lv)
    for lv in NEW_GLOBAL:
        if lv not in C.GLOBAL:
            C.GLOBAL.append(lv)
    C.OP.update(NEW_OP)
    C.SEV.update(NEW_SEV)
    C.L1_NAME["02"] = "Chỉnh sửa hậu kỳ"
    from v6.io import render as R                      # nap sau khi da sua C.CLS: cac ten lan phai khop CLS[1:]
    for n in LANES:
        if n not in R.LANE_NAMES:
            R.LANE_NAMES.append(n)
    C.LIMITS[:] = LIMITS_V9
    C.LIMITS_REQUIRED[:] = LIMITS_V9[:3]


LIMITS_V9 = [
    "Sàng lọc, không phải giám định tư pháp.",
    "Nhãn 01 nghĩa là không thấy dấu vết của các thao tác được soi, không phải chứng nhận bản thật.",
    "Các điều kiện ghi mà hệ thống không tự kiểm được: ghi một lần liên tục, một thiết bị, khoảng cách tới micro "
    "(≤ 2 m, không đổi xa–gần đột ngột), tốc độ phát gốc — nếu không đạt, chỗ tạm dừng hay đổi thiết bị có thể bị báo như mối ghép, "
    "và tốc độ lệch > 12 % có thể bị báo là đổi giọng cả file.",
    "Không chép lời, không nói kiểu con (thay hay chèn), công cụ đã dùng, nội dung đúng hay sai, hay người nói là ai.",
    "Mốc thời gian là một vùng, không cam kết đúng từng âm tiết; 0,3 s đầu và cuối tệp không được soi.",
    "Số đo độ chính xác lấy từ lời nói tự nhiên tiếng Việt và tiếng Anh trộn thao tác mô phỏng; thao tác che giấu kỹ (khớp âm sắc, nền âm, "
    "nén lại) và phòng vang (RT60 ≥ 0,3 s) làm tỉ lệ tìm thấy giảm mạnh — xem bảng theo vùng điều kiện trong báo cáo kiểm thử.",
    "VoiceAudit v9 là bản THỬ NGHIỆM: chỉ thử trên giọng máy viXTTS (một bộ tổng hợp), chưa thử trên ghi âm thiết bị thật.",
]
