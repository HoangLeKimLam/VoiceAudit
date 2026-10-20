# P0 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 1. Hướng đi gợi ý
1. **Chặng 1 — Chốt nền đo.** (a) Tệp dài đã chấm lại 19/10 (`e2e_v9F_long_len.md`: vu oan 51 → 15,3 %, Req 2 0,75 → 0,35, 03 đúng 48 → 5 %); ghi âm thật đã chấm lại 19/10 (`e2e_v9F_scrape_len.md`: vu oan 15,6 → 13,0 %; gần như mọi tệp < 20 s nên ngưỡng theo độ dài ít tác dụng — phần còn lại là lệch miền, 84/95 tệp gán 02); (b) điều tra cùng P2 vì sao ngưỡng vi không giữ ngân sách (lệch fold, VSS); (c) đo lại bộ điều kiện bằng hệ thống hiện tại (`e2e_test --sets benign cover cover2 chain --tag _dieukien`), cập nhật ma trận ô (`v6/coverage/matrix.py`).
2. **Chặng 2 — Ghép theo nhịp.** Hằng tuần ghép gói lane qua `ghep_thu` rồi e2e; công bố đường biên A–D có lencal (T0.3); T0.1 nhiều thao tác, T0.2 vùng xám; phân bổ / điều chỉnh ngân sách vu oan (CHUNG §3.1) khi có số mới.
3. **Chặng 3 — Chốt.** Tập khoá T0.9 một lần; báo cáo cuối theo điều kiện chấm; cập nhật `VoiceAudit.html` (mục hiện trạng).

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
Xem `CHUNG.md §6` — stacker v6d, ngưỡng theo chế độ điều kiện, trung bình mô hình fold vs ngưỡng OOF fold đơn, ngưỡng một chiều chỉ cho tệp dài.

## 3. Ý mở đầu
Luật gộp khúc → tệp khác (kiểm định đa giả thuyết theo kênh; hai tầng: một khúc rất mạnh HOẶC ≥ 2 khúc vừa), kiểm chứng viên độc lập trước 03, huấn luyện đa nhiệm từ dữ liệu các lane, điểm vận hành theo nguồn tệp.

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Kết hợp / giải mã | Hợp nhất muộn FT-2 ⊕ TCN tốt hơn từng bên; stacker không hơn; lencal hai chiều giữ vu oan phẳng theo độ dài | Kiểm chứng viên độc lập trước 03 (v4 có, v9 mất); luật gộp khúc → tệp hai tầng; giải mã nhiều đỉnh; định tuyến / tiền xử lý theo điều kiện; ngưỡng theo nguồn |
| Xếp hạng lại dòng | Embedding nhiều kênh: 54,2 % ở báo nhầm 4,3 % vs 50,2 % ở 5,2 % | Chưa tích hợp vào sản phẩm |
| Mọi họ ở cấp lane | — | Ghép gói của lane bất kể họ phương pháp, miễn đúng chỗ cắm và qua `ghep_thu` |
