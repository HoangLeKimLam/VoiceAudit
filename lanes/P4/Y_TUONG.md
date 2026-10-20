# P4 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 0. Hệ thống hiện tại của mảng — cách baseline đang làm

```
XLS-R khung L12 (PCA 64) + tầng 6 ──► [kênh ghep, xoa, doan_ghep] ──► cột 0, 1, 4 của điểm khung (T, 6)
                                                                                 ▼
          K7 (P0): đỉnh vượt ngưỡng (cách ≥ 0,3 s), hai đỉnh ghép ≤ 5 s → một đoạn, ≤ 2 dòng / 14 s
                   → dòng 03_ghep_giong_that (thay / chèn) · 03_xoa_loi (quanh mối nối)
                                                                                 ▼
          K9 (P1): ảnh khoanh mối nối + dấu hiệu từ `signs.py` của bạn
```

Baseline hiện tại được tạo bằng: TCN ranh giới + FT-2 LoRA (hạng 16 × 6 tầng cuối XLS-R, cửa sổ 4 s, **8 lượt cố định**) trung bình logit = **fuse1** cột 0, 1.

**Mã baseline hiện có — không nằm trong bộ gửi, xin nhóm trưởng khi cần** (sau khi đã tự đề xuất hướng): TCN ranh giới `v6/boundary/tcn.py train --tag`; FT-2 LoRA `v6/boundary/ft.py`, `ft_cv.py train|eval|compare --fold k --tag`; so OOF `v6/boundary/cmp_oof.py all`; sinh dữ liệu 03 `build_natural.make_one` (`xoa_loi`, `ghep_thay`, `ghep_chen`, `_donor_seg`, `_pick_span`), `v6/natural/build_tiers.py`, `v6/coverage/build_edge.py`.

## 1. Hướng đi gợi ý
1. **Chặng 1 — Xoá lời (7b) + làm rõ vu oan vi.** Chuyên gia xoá lời riêng (cửa sổ 8 s, trọng số lớp, nhiều hạt giống, nhãn mốc ±60 ms, đặc trưng mép: B_lfcc, flux phổ, bậc sàn nhiễu, pha). Song song: tìm vì sao ngưỡng vi không giữ được ngân sách. Nộp qua `ghep_thu` ở ngân sách.
2. **Chặng 2 — Ghép kỹ** (vi 24–29 %): đặc trưng trong / ngoài đoạn (độ giống người nói trái / phải mốc, sàn nhiễu, phòng), nguồn lời khác bản ghi / khác người.
3. **Chặng 3 — Bền điều kiện + sửa tay thật.** Đo lại vang / TV / nén / ứng dụng / chồng điều kiện; T4.7 ≥ 150 ca người ngoài sửa bằng Audacity / Audition / Reaper; định vị T4.8 trong e2e; chuẩn ngoài PartialSpoof.

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
- HGB trên khung < TCN (ghép vi 26 → 49 % ở 1 %); WavLM L6 < XLS-R L12.
- Học thêm AMI dài: hầu như không giúp ghép chép-dán trên âm thanh họp xa micro.
- Khử vang WPE trước bộ phát hiện: không lợi; **vang là chế độ khó nhất** — cần dữ liệu / mô hình, không phải ngưỡng.
- FT-3 (dừng sớm CE + chain / benign): xoá en 55 → 16 % — đừng dừng sớm theo CE.

## 3. Ý mở đầu
Đầu "trong / ngoài đoạn" (ECAPA trái / phải mốc), mô hình mép riêng cho xoá lời, tăng cường vang / nén / ứng dụng **áp sau khi sửa**, đa hạt giống, kiểm chứng viên tiêu chí tay, giải mã khác trong kênh của bạn (đề xuất cho P0).

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Số đo tay | `flux_moc` 0,81–0,94 khi biết vị trí (kỹ 0,69–0,73) — **chưa dùng trong sản phẩm**; group delay 0,55–0,79; ENF thô 0,34–0,61; H tại mối xoá ≤ 0,68 vs E 0,94–0,98 | `flux_moc` làm kiểm chứng viên / đặc trưng mốc; ENF thật; đồng cấu âm, gãy pha, ngắt hơi, nền theo dải tại mối nối; B1 ΔPSD + PELT; B6 khoảng lặng lượt lời; số đo tay làm đầu vào khung |
| Ảnh phổ | mel / CQT ghép 0,79, xoá 0,70–0,75 (biết vị trí); vị trí không biết 0,56–0,59 | CNN ảnh quanh mốc làm kiểm chứng viên; PANNs / BEATs khung |
| Embedding | Probe cục bộ xlsr 0,96 / 0,97 / 0,91; fuse1 ghép 0,938, xoá 0,789 (vi); FT-2 làm **xoá kỹ vi tụt 16 → 7 %** | Signal Pointer Nets (có mã mở); đầu BISL / FPN-Transformer; ECAPA trái–phải mốc; gộp L6 + L12; WavLM-Large / HuBERT |
| Kết hợp / giải mã | Xếp hạng lại dòng: 54,2 % ở 4,3 % vs 50,2 % ở 5,2 % (chưa tích hợp) | Giải mã nhiều đỉnh (≥ 2 chỗ sửa / khúc); kiểm chứng viên mốc |
