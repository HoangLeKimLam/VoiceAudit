# P3 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 0. Hệ thống hiện tại của mảng — cách baseline đang làm

```
XLS-R khung L12 (PCA 64) + tầng 6 ──► [kênh den, voi] ──► cột 2, 3 của điểm khung (T, 6) ─┐
XLS-R vector gộp thô (L8/12/16)  ──► [đầu nen, giong] ──► xác suất cả tệp (cột 0, 1 h2) ──┤
                                                                                            ▼
                         K7 (P0): vùng vượt ngưỡng 0,4–8 s → dòng 02_nen_cuc_bo / 02_giong_cuc_bo
                                  đầu vượt ngưỡng → dòng 02_nen_toan_cuc / 02_giong_toan_cuc
                                                                                            ▼
                         K9 (P1): ảnh + dấu hiệu từ `signs.py` của bạn
```

Baseline hiện tại được tạo bằng: TCN vùng `tcn_reg` (`python -m v6.boundary.tcn train --tag tcn_reg --regions`) ⊕ FT-2 LoRA (cột chung, P0 giữ) trung bình logit = **fuse1** cột 2, 3; đầu cả tệp **h2** (`python -m v6.boundary.heads2 run --tag h2`).

**Mã baseline hiện có — không nằm trong bộ gửi, xin nhóm trưởng khi cần** (sau khi đã tự đề xuất hướng): TCN vùng `v6/boundary/tcn.py --regions`; đầu cả tệp `v6/boundary/heads2.py`; sinh dữ liệu 02 `v6/natural/build_natural.py` (`DEN_FAMS` / `VOC_FAMS`, `_process`) và `vaudit/edits.py blend_process`.

## 1. Hướng đi gợi ý
1. **Chặng 1 — Chuyên gia `den` / `voi` riêng v1, sạch trên âm thanh thật.** Giữ khung baseline (TCN vùng / LoRA trên XLS-R), thêm **bộ âm cứng v1 của P2** (đoạn tệp thật làm kênh nổ: ghi âm thật cào, tệp dài, lành tính, kênh ứng dụng) — đúng fold. Mục tiêu: ở ngân sách 0,5 %, phát hiện ≥ baseline, vu oan trên dev ghi âm thật giảm rõ. Nộp qua `ghep_thu`.
2. **Chặng 2 — Đầu cả tệp chịu được miền thật.** `giong` (25 tệp vu oan ghi âm thật) và `nen` (14–16 % sau đổi nhãn 20/10; trước 16–31 %): thêm vector gộp ghi âm thật làm âm, đặc trưng mới cho khử ồn cả tệp (sàn nhiễu ở đoạn yên vs có tiếng, phương sai phổ); giữ Req 5 (khử ồn ứng dụng = 01).
3. **Chặng 3 — Đa dạng công cụ + định vị.** Thêm công cụ khử ồn / đổi giọng (một số cần GPU → hỏi), tập chưa thấy, Req 10 / 11 cho đoạn 02, kiểm chứng viên tiêu chí tay.

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
- Lớp 3 của bộ ranh giới để bắt 02 cục bộ: 1–10 % → phải dùng **nhãn vùng** (`tcn_reg`).
- Đầu cả tệp trên khung đã trừ trung vị: mất thông tin (AUC khử ồn 0,75) → phải dùng **vector gộp thô** (0,86).
- Bộ gắn 02 cục bộ kiểu v6b: cần ~9 % vu oan để đạt 78 % — quá đắt.
- Khử ồn cả tệp kẹt ~30 % ở mọi cách đã thử — cần ý tưởng mới, đừng chỉ chỉnh siêu tham số.

## 3. Ý mở đầu
Mô hình sẵn khác (WavLM, BEATs), đặc trưng sàn nhiễu theo thời gian, LFCC, cao độ, hàm mất mát xếp hạng, lấy mẫu âm cứng trực tuyến, kiểm chứng viên ΔMFCC / sàn nhiễu trong–ngoài đoạn, đầu cả tệp hai tầng (yên / có tiếng).

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Số đo tay | **Mạnh nhất cho khử ồn cả tệp**: H 0,877 / 0,872 vs embedding 0,834; LODO 0,928 vs 0,772; khử ồn WebRTC báo nhầm 3,3 % vs 15 %. Nhưng gắn thẳng vào sản phẩm (Q9b) làm Req 4 vi 0,165 → 0,243 — **cách gắn / hiệu chỉnh là bài toán mở**. Đoạn: `i_mel` 0,85 | Musical noise (G2) tách riêng, HNR 4–8 kHz, sàn nhiễu theo dải / theo thời gian, ΔMFCC trong–ngoài đoạn làm kiểm chứng viên |
| Ảnh phổ | Ảnh phổ tự dựng: khử ồn cả tệp 0,88–0,92, **đổi giọng AI 0,93–1,00**; AST + E 0,901 (200 tệp) | PANNs / BEATs / AudioMAE / CNN ảnh tinh chỉnh cho khử ồn & đổi giọng (cả tệp + cửa sổ trượt) |
| Embedding | h2 khử ồn 0,863, đổi giọng 0,941 (vi); thêm L8 / L16 không lợi; fuse1 đổi giọng đoạn 0,943 vi | WavLM, ECAPA (đổi giọng = đổi đặc tính người nói), tinh chỉnh riêng cho `den` / `voi` |
| Kết hợp | EHST khử ồn cả tệp 0,897 / 0,890; số đo tay làm kiểm chứng viên: khử ồn +0,026 | Kết hợp có hiệu chỉnh riêng cho miền thật (ghi âm thật làm âm) |
