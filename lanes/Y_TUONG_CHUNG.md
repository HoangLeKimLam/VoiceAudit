# Ý tưởng tham khảo chung — kinh nghiệm cũ của cả nhóm

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng.** Đây là các thử nghiệm đã làm trên hệ thống hiện tại, đúng **trong cấu hình đã ghi** — không phải lệnh cấm. Được thử lại nếu có cách hoặc lý do mới (ghi rõ khác ở đâu). Ý tưởng riêng từng mảng: `lanes/<Px>/Y_TUONG.md`; bản đồ mọi họ phương pháp: `lanes/HUONG_PHUONG_PHAP.md`.

## Đã thử, chưa hiệu quả

- Bộ đặc trưng K2 kiểu v5 / v6b (`FileFeat`, DeepFense…): RTF 5,5 → vượt Req 15. Bài học là **ngân sách thời gian chạy**, không phải «chỉ được dùng XLS-R»: mọi hướng (kể cả mô hình nền khác) dùng được nếu nằm trong ngân sách RTF của lane.
- Vector gộp cả tệp để bắt ghép / xoá: AUC 0,45–0,59 — ranh giới cần bằng chứng **cục bộ**.
- Stacker v6d (trộn chuyên gia bằng mô hình thứ hai): không hơn — giới hạn là **độ tách của đặc trưng**, không phải cách trộn.
- Dừng sớm theo CE (FT-3): kém hơn FT-2 rõ (xoá en 55 → 16 %).
- Ngưỡng riêng theo chế độ điều kiện (`regime_thr.py`): ±3 điểm, không lợi.
- Khử vang WPE trước bộ phát hiện: không lợi (vang 0,3 s 12 → 12 %).
- Trung bình 5 mô hình fold trong sản phẩm trong khi ngưỡng từ OOF fold đơn: lệch điểm (đã sửa bằng mô hình cuối / định tuyến fold).
- int8 vs fp32 XLS-R: như nhau — giữ int8.
- Ngưỡng một chiều chỉ cho tệp dài (lencal mốc 10 s / 20 s): kém hơn hai chiều mốc trung vị 8,8 s.
- Luật vu oan theo từ vựng ASR cấp video cho LID tiếng Anh (17/10): tiếng Việt qua ASR tiếng Anh chồng lấn tiếng Anh thật.
- Dừng sớm theo loss CE (FT-3) kém hơn số lượt cố định ở cấu hình đã thử (xoá en 55 → 16 %) — vì vậy baseline dùng số lượt cố định.

## Giả thuyết đang mở (chưa kiểm — được thách thức bằng số liệu)

- Vì sao tiếng Việt vẫn vu oan 5,3 % trên tệp 01 gốc ở ngân sách 3 % (trước đổi nhãn 20/10: 5,2 %): Nghi: lệch giữa các fold (fold 0 chứa ~56 % natvi) và nguồn VietSuperSpeech nghi đã bị sửa — P0 + P2 điều tra.
