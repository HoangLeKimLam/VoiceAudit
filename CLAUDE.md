# Nhánh `lane/p3` — bạn là **P3**

> Claude Code tự đọc tệp này khi mở kho. Tệp chỉ cho biết bạn là ai, đọc gì, đối chiếu với ai — **không** phải hướng giải.

## Bạn là
**P3** — Nhãn 02: khử ồn chuyên nghiệp / mạnh, đổi giọng (cả tệp · một đoạn). Việc của lane: `lanes/P3/CLAUDE.md`. Dữ liệu của lane: `lanes/P3/DU_LIEU.md`.

## Đọc theo thứ tự (lần đầu)
1. `VoiceAudit_6Muc.html` — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. `lanes/BAT_DAU.md` → `lanes/CHUNG.md` → `lanes/P3/CLAUDE.md` → `lanes/DU_LIEU.md` → `lanes/P3/DU_LIEU.md`.
3. **Chưa** đọc `lanes/P3/Y_TUONG.md`, `lanes/Y_TUONG_CHUNG.md`, `lanes/HUONG_PHUONG_PHAP.md` cho tới khi đã tự đề xuất 2–3 hướng của riêng mình (`lanes/BAT_DAU.md` mục 11).

## Cố định — để ghép không xung đột (còn lại tự do)
- Phạm vi phải phủ, dạng nhãn và định dạng: `lanes/DU_LIEU.md` mục 2–3, `lanes/P3/DU_LIEU.md` mục 1–3.
- Hợp đồng nộp và ngân sách vu oan: `lanes/CHUNG.md` §3–4. Đo trên cùng thước: `python -m v6.product.ghep_thu`.
- Trước khi nộp dữ liệu: `python -m v6.data.kiem_bo <manifest> --moi` phải ĐẠT, và `python -m v6.data.lockbox check <manifest>`.
- Luật cứng: `lanes/CHUNG.md` §2.

## Kiểm tính đồng nhất với lane khác
Đọc thêm (không sửa) phần ranh giới của các lane liên quan:
- **P2** — ranh giới khử ồn: đơn sơ của ứng dụng = 01 (P2), chuyên nghiệp / mạnh = 02 (bạn): `lanes/P2/DU_LIEU.md` mục 2.
- **P5** — chuyển giọng (VC) = 02 (bạn), giọng máy tạo từ văn bản = 03 (P5); P5 sinh `vcmulti` theo yêu cầu của bạn: `lanes/P5/DU_LIEU.md` mục 2.
- **P4** — tệp vừa ghép vừa khử ồn / đổi giọng: mỗi thao tác một dòng riêng: `lanes/P4/DU_LIEU.md` mục 2.
- **P1** — nhận `signs.py` của bạn cho hồ sơ: `lanes/P1/CLAUDE.md`.

Bảng nhãn chung cho mọi lane: `lanes/DU_LIEU.md` mục 2.1–2.3. Thấy mâu thuẫn giữa các tệp thì **không tự sửa tệp lane khác** — ghi `lanes/DE_XUAT/<ngày>_P3_<tên>.md` cho nhóm trưởng (P0).

## Dữ liệu
Kho git không có dữ liệu. Giải nén `du_lieu_P3.tar` (nhóm trưởng gửi) vào chính thư mục này. Dữ liệu đó chỉ là ví dụ — việc chính là tự thu thập dữ liệu đúng dạng để phủ phạm vi (`lanes/P3/DU_LIEU.md`, đầu tệp).
