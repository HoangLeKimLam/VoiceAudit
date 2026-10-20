# Nhánh `lane/p2` — bạn là **P2**

> Claude Code tự đọc tệp này khi mở kho. Tệp chỉ cho biết bạn là ai, đọc gì, đối chiếu với ai — **không** phải hướng giải.

## Bạn là
**P2** — Nhãn 01: ghi âm thật, truyền tải qua ứng dụng, cổng 01, kho tệp thật, bộ âm cứng. Việc của lane: `lanes/P2/CLAUDE.md`. Dữ liệu của lane: `lanes/P2/DU_LIEU.md`.

## Đọc theo thứ tự (lần đầu)
1. `VoiceAudit_6Muc.html` — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. `lanes/BAT_DAU.md` → `lanes/CHUNG.md` → `lanes/P2/CLAUDE.md` → `lanes/DU_LIEU.md` → `lanes/P2/DU_LIEU.md`.
3. **Chưa** đọc `lanes/P2/Y_TUONG.md`, `lanes/Y_TUONG_CHUNG.md`, `lanes/HUONG_PHUONG_PHAP.md` cho tới khi đã tự đề xuất 2–3 hướng của riêng mình (`lanes/BAT_DAU.md` mục 11).

## Cố định — để ghép không xung đột (còn lại tự do)
- Phạm vi phải phủ, dạng nhãn và định dạng: `lanes/DU_LIEU.md` mục 2–3, `lanes/P2/DU_LIEU.md` mục 1–3.
- Hợp đồng nộp và ngân sách vu oan: `lanes/CHUNG.md` §3–4. Đo trên cùng thước: `python -m v6.product.ghep_thu`.
- Trước khi nộp dữ liệu: `python -m v6.data.kiem_bo <manifest> --moi` phải ĐẠT, và `python -m v6.data.lockbox check <manifest>`.
- Luật cứng: `lanes/CHUNG.md` §2.

## Kiểm tính đồng nhất với lane khác
Đọc thêm (không sửa) phần ranh giới của các lane liên quan:
- **P3** — ranh giới khử ồn: đơn sơ của ứng dụng = 01 (bạn), chuyên nghiệp / mạnh = 02 (P3): `lanes/P3/DU_LIEU.md` mục 2.
- **P3 / P4 / P5** — bạn cấp bộ âm cứng (tệp thật, đúng fold theo nhóm kênh) cho họ học tránh vu oan.
- **P1** — kho phân vị tệp thật cho hồ sơ: `lanes/P1/CLAUDE.md`.
- **P0** — bất thường vu oan tiếng Việt: điều tra cùng nhóm trưởng.

Bảng nhãn chung cho mọi lane: `lanes/DU_LIEU.md` mục 2.1–2.3. Thấy mâu thuẫn giữa các tệp thì **không tự sửa tệp lane khác** — ghi `lanes/DE_XUAT/<ngày>_P2_<tên>.md` cho nhóm trưởng (P0).

## Dữ liệu
Kho git không có dữ liệu. Giải nén `du_lieu_P2.tar` (nhóm trưởng gửi) vào chính thư mục này. Dữ liệu đó chỉ là ví dụ — việc chính là tự thu thập dữ liệu đúng dạng để phủ phạm vi (`lanes/P2/DU_LIEU.md`, đầu tệp).
