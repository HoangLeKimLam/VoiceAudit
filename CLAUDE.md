# Nhánh `lane/p1` — bạn là **P1**

> Claude Code tự đọc tệp này khi mở kho. Tệp chỉ cho biết bạn là ai, đọc gì, đối chiếu với ai — **không** phải hướng giải.

## Bạn là
**P1** — Khung cố định: kiểm điều kiện K1 (Constraint → 04), luật nhãn, hồ sơ · ảnh · PDF, khảo sát Req 13. Việc của lane: `lanes/P1/CLAUDE.md`. Dữ liệu của lane: `lanes/P1/DU_LIEU.md`.

## Đọc theo thứ tự (lần đầu)
1. `VoiceAudit_6Muc.html` — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. `lanes/BAT_DAU.md` → `lanes/CHUNG.md` → `lanes/P1/CLAUDE.md` → `lanes/DU_LIEU.md` → `lanes/P1/DU_LIEU.md`.
3. **Chưa** đọc `lanes/P1/Y_TUONG.md`, `lanes/Y_TUONG_CHUNG.md`, `lanes/HUONG_PHUONG_PHAP.md` cho tới khi đã tự đề xuất 2–3 hướng của riêng mình (`lanes/BAT_DAU.md` mục 11).

## Cố định — để ghép không xung đột (còn lại tự do)
- Phạm vi phải phủ, dạng nhãn và định dạng: `lanes/DU_LIEU.md` mục 2–3, `lanes/P1/DU_LIEU.md` mục 1–3.
- Hợp đồng nộp và ngân sách vu oan: `lanes/CHUNG.md` §3–4. Đo trên cùng thước: `python -m v6.product.ghep_thu`.
- Trước khi nộp dữ liệu: `python -m v6.data.kiem_bo <manifest> --moi` phải ĐẠT, và `python -m v6.data.lockbox check <manifest>`.
- Luật cứng: `lanes/CHUNG.md` §2.

## Kiểm tính đồng nhất với lane khác
Đọc thêm (không sửa) phần ranh giới của các lane liên quan:
- **P3 / P4 / P5** — nộp `signs.py` (dấu hiệu từng thao tác) cho hồ sơ của bạn: gói chuyên gia ở `lanes/CHUNG.md` §4.
- **P2** — kho phân vị tệp thật (`refbank`): `lanes/P2/CLAUDE.md`.
- **Mọi lane** — thông điệp 04 và cảnh báo phải thống nhất; tệp vi phạm điều kiện không mang nhãn sửa: `lanes/DU_LIEU.md` mục 2.2.

Bảng nhãn chung cho mọi lane: `lanes/DU_LIEU.md` mục 2.1–2.3. Thấy mâu thuẫn giữa các tệp thì **không tự sửa tệp lane khác** — ghi `lanes/DE_XUAT/<ngày>_P1_<tên>.md` cho nhóm trưởng (P0).

## Dữ liệu
Kho git không có dữ liệu. Giải nén `du_lieu_P1.tar` (nhóm trưởng gửi) vào chính thư mục này. Dữ liệu đó chỉ là ví dụ — việc chính là tự thu thập dữ liệu đúng dạng để phủ phạm vi (`lanes/P1/DU_LIEU.md`, đầu tệp).
