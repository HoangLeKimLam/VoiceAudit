# P1 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 0. Mã baseline hiện có

**Mã baseline hiện có — không nằm trong bộ gửi, xin nhóm trưởng khi cần** (sau khi đã tự đề xuất hướng): học bộ ước lượng điều kiện trên vector gộp `v6/k1x/cond_learn.py` (`extract | eval`, báo cáo `cond_learn.md`); sinh đầu vào khó `v6/coverage/robust_inputs.py`.

## 1. Hướng đi gợi ý
1. **Chặng 1 — Test chủ động 24 hàng (T1.1) + sửa chặn oan rõ nhất.** Sinh ≥ 12 tệp mỗi hàng × 3 mức × 2 ngôn ngữ; kiểm "quá nhỏ" tiếng Anh 4,7 % (ngưỡng mức âm có quá chặt với EdAcc / AMI?); đo lại Req 9.
2. **Chặng 2 — Hàng học + khung tệp.** SNR → 04 bằng bộ ước lượng học (mục tiêu AUC ≥ 0,92, nhiều loại nhiễu thật), nền, loại âm thanh, kiểu nói, số người nói; T1.3 kênh, T1.4 độ dài (2,5 s … > 10 phút, 250 MB); PDF 2 kênh.
3. **Chặng 3 — Người ngoài hiểu bằng chứng.** T1.8 khảo sát Req 13 đúng quy trình ở `VoiceAudit_DanhGia.docx` mục 4.3 (3–5 người ngoài nhóm; 40 hồ sơ, 8 cho mỗi loại sửa; làm hết bước 1 rồi mới sang bước 2; chấm tự động; báo kèm theo từng loại, từng người, khoảng tin cậy, mức đồng thuận) — quy trình đo là cố định, còn cách làm ảnh / giải thích dễ hiểu hơn là tự do; chỉnh ảnh / giải thích theo kết quả; T1.7 Assumption (phát lại qua loa rồi thu; sửa > 50 %; đoạn < 0,1 s).

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
- Ước lượng mù: WADA-SNR AUC 0,71, SNR qua ASR 0,66, AST nhạc / TV nền 0,66 (TV gần như vô hình), RT60 theo suy giảm không ước được ≥ 1 s → dùng **bộ học trên vector gộp XLS-R**.
- Chạy K1 khi máy cạn bộ nhớ: tiến trình con không khởi được → ra giả "không đọc được tệp" (1 891 tệp cover phải chạy lại). Chạy K1 hàng loạt qua `run_serial.sh`.
- Đáp án mong đợi sai (mp3 16 / 24 kbps): kiểm đáp án trước khi kết luận K1 sai.

## 3. Ý mở đầu
Bộ ước lượng SNR / nền với MUSAN, DEMAND, DNS; bộ phân loại loại âm thanh (podcast / phát thanh / sách nói / nhạc) và kiểu nói (hát / thì thầm / la hét); đếm người nói, nói chồng; cách trình bày ảnh và giải thích dễ hiểu hơn cho người không chuyên.

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Số đo tay (DSP) | WADA-SNR 0,711; RT60 theo suy giảm hỏng ở ≥ 1 s; luật băng thông 20/20 tệp hẹp → 04, 0/200 tự nhiên bị lật | Brouhaha SNR / C50 (giấy phép NC — hỏi); DSP cho loại âm thanh / kiểu nói; số đo tay làm **dấu hiệu dễ hiểu** cho Req 13 |
| Ảnh phổ / mô hình sẵn | AST nền nhạc / TV 0,661 (TV 0,462) | PANNs / BEATs cho loại âm thanh, nhạc, hát, la hét; thiết kế ảnh phổ K9 dễ hiểu cho người ngoài |
| Embedding | Gộp XLS-R học: SNR 0,838 · nền 0,907 · RT60 0,997; SQUIM 0,824 nhưng 18,5 % 04 oan; LID whisper-small vi 100 % / en 83 % (n nhỏ) | WavLM / Whisper encoder gộp cho kiểu nói, loại âm thanh, đếm người nói, nói chồng |
