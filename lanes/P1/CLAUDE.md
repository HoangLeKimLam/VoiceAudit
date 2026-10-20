# P1 — Khung tất định: Input · Constraint · 04 · Output

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ chính **V1** (kiểm tệp · đo tín hiệu · ước lượng điều kiện · người nói và phòng · quyết định điều kiện) và **V6** (phiếu bằng chứng · ảnh · giải thích, bảng, PDF · khảo sát người ngoài); trong **V5** giữ mức độ theo loại và luật nhãn; trong **V4** giữ tệp hai kênh và cùng P0 giữ giới hạn vùng soi. Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

```
tệp âm thanh ──► [K1: container · mã hoá · kênh · độ dài · tín hiệu · người nói / phòng · LID] ──► vi phạm cứng → 04 + lý do
                     │ hợp lệ: kênh đã chuẩn hoá, ngôn ngữ (chọn bộ ngưỡng), cảnh báo
                     ▼
              (XLS-R + chuyên gia P2–P5 + K7 của P0)
                     │ dòng thao tác H4
                     ▼
              [K8: luật nhãn tệp (1)–(5), mức độ] ──► [K9: hồ sơ · ảnh · dấu hiệu · bảng · PDF · 12 cảnh báo]
```
Bạn **không** phát hiện chỉnh sửa. Bạn quyết định tệp nào được giám định và kết quả trình bày ra sao — nên bạn giữ Req 9, 12, 13, 14.

*Sơ đồ trên là hệ thống hiện tại (baseline). Cố định chỉ là chỗ cắm — nhận gì, nộp gì đúng hợp đồng (CHUNG §4) và trong ngân sách; bên trong khối của bạn làm bằng cách nào cũng được (mô hình nền khác, âm thanh gốc, số đo tay, kết hợp…).*

## 2. Đích đến
Mỗi hàng trong 24 hàng Constraint có test chủ động đúng đáp án (dưới · đúng · trên ngưỡng × vi / en); K1 chặn oan ≤ 2 % tệp hợp lệ (tệp bị chặn báo riêng theo lý do — hợp đồng không tính chúng vào Req 9); **Req 9 ≤ 8 %** cho tệp ra 04 do phân tích (luật nhãn K8 của bạn + vùng xám của P0); **Req 12 = Req 14 = 100 %** trên mọi tập; **Req 13 ≥ 70 %** (khảo sát người ngoài hai bước: bước 1 đoán loại sửa khi chưa thấy nhãn — metric chốt; bước 2 xem cả hồ sơ, trả lời 3 câu đọc hiểu — metric phụ; quy trình đo cố định ở `VoiceAudit_DanhGia.docx` mục 4.3); PDF mở được tệp 2 kênh.

## 3. Hiểu bài toán của mảng

Bạn trả lời: tệp có **đủ cơ sở để xét** không (vấn đề 1), nhãn và mức độ có **đúng luật** không (vấn đề 5), hồ sơ có giúp người nhận **hiểu và tự kiểm lại** không (vấn đề 6). Bạn không phát hiện chỉnh sửa.

**Bài toán khó ở đâu (bằng chứng hiện có):**
- **Chặn đúng mà không chặn oan**: K1 đang chặn tệp hợp lệ vi 3,8 %, en 5,9 % (chủ yếu «quá nhỏ» tiếng Anh 4,7 %) — đích ≤ 2 %.
- Vài điều kiện **khó đo thẳng trên tín hiệu** (tiếng ồn, nhạc / TV nền, vang): đo mù kém (WADA-SNR AUC 0,71; TV gần như vô hình); bộ học hiện đạt SNR 0,84, nền 0,91, nhưng SNR chưa dùng để ra 04.
- **24 hàng Constraint** chưa có bài kiểm tra chủ động đủ 3 mức × 2 ngôn ngữ; đáp án mong đợi đôi khi sai.
- Req 9 (04 do phân tích ≤ 8 %) phụ thuộc cả luật nhãn của bạn lẫn vùng xám của P0.
- Req 13 chưa đo; tệp hai kênh có 4 trường hợp, PDF hai kênh chưa kiểm.

**Câu hỏi còn mở:** đặt ngưỡng các bộ ước lượng học thế nào để chặn oan ≤ 2 % mà vẫn chặn đúng; ảnh và giải thích thế nào thì người không chuyên hiểu được.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
| Hạng mục | Số | Nguồn / lệnh |
|---|---|---|
| Req 12 · Req 14 | **100 %** mọi lần chấm (kể cả tệp 1–10 phút) | `e2e_v9F_*.md` |
| Req 9 (04 do phân tích) | 1,8 % tệp tự tạo (60 tệp/kiểu) · 6,2 % ghi âm thật · 6,2 % tệp dài có lencal | như trên |
| K1 chặn tệp hợp lệ (natvi / naten, 60 tệp/kiểu) | vi 3,8 % (không phải lời nói 25, quá ít tiếng nói 19, ồn 7…) · **en 5,9 %, "quá nhỏ" 30/642 = 4,7 %** | `e2e_v9F_n60_TAM.md` |
| K1 trên 135 ca container / mã hoá | 129/135 (6 ca lệch do đáp án sai) | `python -m v6.coverage.k1_axes run` |
| LID (K1x, whisper-small) | 1 568/1 568 ghi âm thật · 317/321 tệp dài | e2e |
| Bộ ước lượng học trên vector gộp | SNR AUC 0,84 · nền (nhạc / TV) 0,91 · RT60 0,997; **SNR chỉ cảnh báo, chưa → 04** | báo cáo `cond_learn.md` (P0 giữ) |
| Req 13 — bước 1 đoán loại (chốt) · bước 2 đọc hiểu hồ sơ (phụ) | **chưa khảo sát** | quy trình: `VoiceAudit_DanhGia.docx` mục 4.3 |

## 5. Bạn sở hữu (chỉ P1 sửa)
Có trong bộ gửi: `vaudit/k1.py` · `vaudit/k1x.py` · `vaudit/cond.py` · `v6/k1x/eval_signal.py` · `v6/coverage/k1_axes*.py` · `v6/product/ticket.py` · `v6/product/signals.py` · phần dựng hồ sơ / PDF / thông điệp 04 trong `voiceaudit_v9.py` · `data_labels/v6/k1axes/`. Mã cũ học bộ ước lượng điều kiện (`v6/k1x/cond_learn.py`) và sinh đầu vào khó (`robust_inputs.py`) không nằm trong bộ gửi — xin P0 **sau khi** tự đề xuất hướng. (Kho phân vị `refbank.py` là của **P2**; dấu hiệu từng thao tác lấy từ `signs.py` của lane chuyên gia.)

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P1/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng
- Hàng **cứng** → 04 với **đúng cụm lý do** trong `VoiceAudit.html`; hàng **học** → cảnh báo hoặc 04 với ngưỡng chặn oan ≤ 2 %.
- Không đổi lược đồ hồ sơ H5 / #bind mà không qua P0; mọi trường mới có test trong `test_ticket.py`.
- "Không tự kiểm được" (một lần ghi, tốc độ phát) → tuyên bố giới hạn, không đoán.

## 7. Nộp
Thay đổi K1 / K8 / K9 kèm: bảng T1.x trước / sau, Req 9 trên natvi / naten (60 tệp / kiểu) không tăng, `python -m v6.product.test_ticket` + `selftest` + `test_pipeline_modes` đều đạt. Bài đo T1.1–T1.8. Lệnh: `python -m v6.coverage.k1_axes build | run | merge | lang`, `python -m v6.k1x.eval_signal`.

## 8. Phối hợp
Nhận `signs.py` của **P3 / P4 / P5** và kho phân vị của **P2**; thông điệp 04 và cảnh báo thống nhất cho mọi lane.

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
