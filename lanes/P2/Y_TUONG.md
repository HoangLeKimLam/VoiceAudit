# P2 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 0. Mã baseline hiện có

**Mã baseline hiện có — không nằm trong bộ gửi, xin nhóm trưởng khi cần** (sau khi đã tự đề xuất hướng): bộ lành tính `v6/coverage/build_benign.py`, đo `v6/coverage/eval_benign.py --tag`, luật `v6/boundary/rules.py --tag` (báo cáo `rules_all.md`, mô hình cũ 02/10).

## 1. Hướng đi gợi ý
1. **Chặng 1 — Hiểu vu oan.** (a) Bất thường tiếng Việt: vu oan theo nguồn × fold × người nói, báo cả hai cách khi bỏ VSS; (b) ghi âm thật cào (khi lane L của P0 có `scores.npz`): kênh nào, kiểu âm thanh nào (Zoom NS? chuẩn hoá độ lớn? AGC? nén YouTube?). *Xong khi* giải thích được ≥ 80 % tệp vu oan.
2. **Chặng 2 — Bộ âm cứng v1 + cổng 01.** Đoạn tệp thật làm kênh P3 / P4 / P5 nổ, chia đúng fold, manifest + thẻ, phiên bản; cổng 01 baseline → cải tiến.
3. **Chặng 3 — Phủ cơ chế lành tính + dữ liệu thật mới.** T2.1–T2.6 đo lại bằng hệ thống hiện tại (≥ 8 cấu hình nén, nén lặp, DTX, thiết bị, ±5 % tốc độ, cắt đầu–cuối, tạm dừng); T2.2 truyền tải thật qua app (cần người); thêm ghi âm thật tiếng Việt không chuyên (mục tiêu ≥ 20 h, ≥ 30 nhóm).

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
- Luật tạm dừng ghi (`rules.py`): yếu (25 → 23 %); luật đoạn số 0, che mất gói, cổng ồn thì có ích.
- VietSuperSpeech: 17 % vu oan, 5 người nói chiếm 57/70 vu oan hàng đầu — nghi video đã sửa → luôn báo cả hai cách.
- Bộ lọc vết cắt cho dữ liệu cào: precision 0,90 nhưng **recall 0,10** — bản gốc cào chỉ là "01 có khả năng".
- Luật LID cấp video bằng từ vựng ASR (17/10): không tách được — đã bỏ.

## 3. Ý mở đầu
Mô hình "dấu vết bình thường" một lớp theo nguồn; hiệu chỉnh ngưỡng theo nguồn (cuộc gọi / họp / tin nhắn) — đề xuất cho P0; nghe mẫu có hệ thống; tự thu chuỗi truyền tải thật; nguồn ghi âm thật mới trong chính sách.

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Số đo tay | `rules.py` (đoạn số 0, che mất gói, cổng ồn) có ích; luật tạm dừng yếu; số đo tay làm cổng 01: AUC 0,598 / 0,657 | Dấu vết bình thường đo tay: MDCT nén kép, dải trên f99, sàn nhiễu theo thời gian — để tách **khử ồn ứng dụng (01)** với khử ồn công cụ (02) |
| Ảnh phổ | Khối S (thống kê ảnh phổ) báo nhầm 48,3 % tệp lành tính ns3 — cẩn thận khi dùng cho 01 | CNN ảnh phổ / PANNs nhận dạng kênh truyền, nén, ứng dụng |
| Embedding | Cổng từ điểm khung: 0,786 / 0,829; kết hợp với số đo 0,800 / 0,853 (lạc quan) | Cổng 01 trên điểm max các kênh + vector gộp + số đo tay; hiệu chỉnh theo nguồn |
