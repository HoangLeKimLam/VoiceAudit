# Phân việc VoiceAudit v9 — mỗi người một lane

Cập nhật 19/10/2026. Thay cho phần lane của `CAY_PHAN_RA_V2_VA_PHAN_VIEC_6_NGUOI.md` (04/10); danh mục test bắt buộc vẫn ở §4B của tệp đó.

## Ai làm gì

| Lane | Vai trò | Một câu | Vấn đề lớn nhất hiện nay | Tệp giao việc |
|---|---|---|---|---|
| **P0** | Nhóm trưởng | Ghép gói, luật quyết định + ngưỡng (K7), chấm đầu-cuối, tập khoá | Tệp dài đã chấm lại (vu oan 51 → 15 %, nhưng bắt tệp sửa 0,75 → 0,35); ghi âm thật 15,6 → 13,0 %; đường biên A–D | [P0/CLAUDE.md](P0/CLAUDE.md) |
| **P1** | Khung tất định | K1 (Constraint → 04), K8 luật nhãn, K9 hồ sơ · ảnh · PDF | 24 hàng Constraint chưa có test chủ động; SNR chưa → 04; khảo sát Req 13 chưa làm | [P1/CLAUDE.md](P1/CLAUDE.md) |
| **P2** | Nhãn 01 | Ghi âm thật, truyền tải, cổng 01, kho phân vị, **bộ âm cứng** | Vu oan 13,0 % trên ghi âm thật (đã có ngưỡng theo độ dài); cổng 01 chưa có | [P2/CLAUDE.md](P2/CLAUDE.md) |
| **P3** | Nhãn 02 | Khử ồn · đổi giọng (cả tệp · một đoạn) | Kênh đoạn 02 là **nguồn vu oan số 1**; khử ồn cả tệp chỉ 28–46 % | [P3/CLAUDE.md](P3/CLAUDE.md) |
| **P4** | Nhãn 03 lời thật | Ghép / chèn · xoá lời | Ở ngân sách: xoá lời 23–30 % (kỹ 6–17 %), 7b 0,20 / 0,35; vu oan vi bất thường 4,3 %; **tệp dài có lencal: 03 đúng 5 %**; chưa có sửa tay thật | [P4/CLAUDE.md](P4/CLAUDE.md) |
| **P5** | Nhãn 03 giọng máy | Một đoạn · cả tệp · máy hay thật | Bật kênh thì bắt gần hết (mẫu chỉ 38 tệp) nhưng **giọng thật vi bị gọi là máy** (vu oan 4,1–5,5 %); chưa có tiếng Anh; cần GPU | [P5/CLAUDE.md](P5/CLAUDE.md) |

**Người mới đọc đầu tiên: [BAT_DAU.md](BAT_DAU.md)** — tổng quan hệ thống bằng lời thường, thuật ngữ, bốn rào (cái gì cố định, cái gì tự do), danh sách tài liệu / mã / dữ liệu nhận, ngày đầu tiên. Sau đó luật chung: [CHUNG.md](CHUNG.md). Dữ liệu — có gì, độ phủ, gói từng người, luật khi thêm: [DU_LIEU.md](DU_LIEU.md) và `lanes/<Px>/DU_LIEU.md` (đóng gói: `python -m v6.data.goi_lane`). Yêu cầu, metric và công thức đo: `VoiceAudit_DanhGia.docx` (phần 2–5). Bản đồ mọi họ phương pháp (số đo tay · ảnh phổ · embedding · kết hợp — bằng chứng, chỗ cắm, khoảng trống): [HUONG_PHUONG_PHAP.md](HUONG_PHUONG_PHAP.md).

## Ba thứ giữ cả nhóm không lệch mục tiêu và ghép được (chi tiết ở `CHUNG.md`)

1. **Hệ thống chung** (`CHUNG.md §1`): sơ đồ luồng K1 → XLS-R → kênh của từng lane → K7 → K8 / K9, 8 thao tác ↔ lane, 15 yêu cầu ↔ lane chịu, số đo hiện tại.
2. **Ngân sách vu oan theo kênh** (`CHUNG.md §3.1`): Req 4 ≤ 3 % chia cho từng kênh (ghép / xoá 0,5 %, khử ồn / đổi giọng vùng 0,5 %, đầu cả tệp 0,25 %, giọng máy 0,25 %). Mỗi lane tối đa phát hiện **trong ngân sách của mình** — không ai nới kênh mình cho đẹp vì vu oan cộng dồn khi ghép.
3. **Ghép thử** (`python -m v6.product.ghep_thu`, `CHUNG.md §3.2`): mỗi lane tự cắm cột điểm của mình vào hệ thống trên OOF và xem ngay Req 1–4, 7a, 7b, 6, 8a, vu oan theo kênh, ô nào tụt — **trước** khi nộp. `--name nen` tái tạo đúng baseline.

## Mỗi lane có gì (cùng một khung)

**`lanes/<Px>/CLAUDE.md` — mục tiêu và nhiệm vụ, không nói cách giải:**
1. **Vị trí trong hệ thống** — nhận gì từ đâu, nộp gì cho ai (chỗ cắm cố định; bên trong làm cách nào cũng được).
2. **Đích đến** — chỉ tiêu cụ thể, ở ngân sách của kênh mình.
3. **Hiểu bài toán của mảng** — khó ở đâu, bằng chứng đang hỏng chỗ nào, câu hỏi còn mở.
4. **Baseline phải vượt** — số để vượt + lệnh tái tạo (không phải kiến trúc phải theo).
5. **Sở hữu** · 6. **Luật riêng** · 7. **Nộp và ghép thử** · 8. **Phối hợp**.

**`lanes/<Px>/DU_LIEU.md`** — trường hợp dữ liệu phải phủ (cố định) + dữ liệu mẫu nhận được + gợi ý nguồn (tự do).

**`lanes/<Px>/Y_TUONG.md`** — ý tưởng của nhóm trưởng: hướng đi gợi ý, đã thử, ý mở đầu, bản đồ phương pháp. **Chỉ đọc sau khi đã tự đề xuất hướng** — để Claude của thành viên không bị neo theo cách nhóm đang làm. Kinh nghiệm chung: `Y_TUONG_CHUNG.md`.

## Bắt đầu (cho từng người)

1. Đọc [BAT_DAU.md](BAT_DAU.md), cài đặt theo mục 10 của tệp đó, mở Claude Code ở thư mục gốc của bộ gửi (đặt ở đâu cũng được, nên là đường ngắn).
2. Câu đầu: dùng đúng câu ở `BAT_DAU.md` mục 11 — Claude đọc BAT_DAU, CHUNG và tệp lane, trả lời 5 câu tự kiểm, tái tạo baseline, rồi đề xuất 2–3 hướng của riêng mình (thay P3 bằng lane của mình).
3. Sau mỗi đợt việc: ghi `lanes/<lane>/BAO_CAO_<ddmm>.md` (giả thuyết · làm gì · số trước / sau · ô tụt · bước tiếp).
4. Muốn đổi hợp đồng / điểm vận hành / chỉ tiêu: viết `lanes/DE_XUAT/<ngay>_<lane>_<ten>.md` để P0 duyệt.

## Nhịp ghép

- Mỗi lane nộp **gói chuyên gia** (`CHUNG.md §4`) khi có cải tiến qua được bài đo trên OOF / dev.
- **Hằng tuần** P0 ghép các gói mới, chạy bảng ô trước / sau, gửi lại từng lane "ô nào tụt, vì sao".
- Tập khoá: P0 chấm **một lần**, cuối cùng.

## Việc nhóm trưởng (người dùng) cần quyết trước khi chia cho người khác

1. **Bộ gửi theo lane** (đã làm 20/10): mỗi người nhận một thư mục `python -m v6.data.goi_lane copy --lane <Px> --dest <ổ ngoài>` gồm tài liệu + mã cần thiết + dữ liệu mẫu đúng lane (BAT_DAU mục 10.2–10.3). Danh sách từng tệp: `python -m v6.data.goi_lane liet-ke --lane <Px> --out <tep>`. Mã không còn ghi cứng `D:\TưDuyTTNT` — đặt bộ gửi ở đâu cũng được (đường ngắn). Các bước gửi: `P0/DU_LIEU.md` mục 4.
2. **Quản lý mã**: thư mục này **chưa là kho git** — nhóm trưởng tự tạo, mỗi lane một nhánh, P0 gộp. Kho chỉ nên chứa phần mã / tài liệu của bộ gửi (không đưa mã và báo cáo thử nghiệm cũ — thiên lệch). Tập khoá đã đóng băng lại 20/10 — chạy lại `lockbox freeze` mỗi khi dữ liệu đổi (`P0/DU_LIEU.md`).
2b. **Môi trường**: `requirements.txt` đã có (xuất 20/10 từ `.venv`, thêm torch CPU và các thư viện cài riêng). Mô hình Hugging Face của sản phẩm: `python -m v6.product.tai_mo_hinh` trên máy thành viên.
3. **GPU cho P5** (và công cụ AI của P3): đang hoãn theo quyết định 12/10.
4. **Người làm thật**: sửa tay ≥ 150 ca (P4 T4.7, P0 T0.5), gửi qua ứng dụng thật (P2 T2.2), ghi âm thiết bị thật 30–50 tệp (P0 T0.6), khảo sát ≥ 3 người (P1 T1.8).
