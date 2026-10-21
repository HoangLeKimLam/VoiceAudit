# VoiceAudit — kho làm việc của nhóm

Kho chỉ có **tài liệu và mã**; dữ liệu gửi riêng (mỗi người một tệp `.tar` qua Drive).
**Mỗi người chỉ làm theo mục mang tên mình** ở dưới: copy từng khối, dán vào Git Bash hoặc Claude Code — không phải sửa chữ nào.

## Phân công

| Thành viên | Lane | Việc | Nhánh git | Dữ liệu | Hướng dẫn |
|---|---|---|---|---|---|
| Nhóm trưởng | P0 | Ghép các phần, giải mã + ngưỡng, chấm đầu–cuối, tập khoá | `main` | — | — |
| Huấn | P1 | Khung cố định: kiểm điều kiện K1 (Constraint → 04), luật nhãn, hồ sơ · ảnh · PDF, khảo sát Req 13 | `lane/p1` | `du_lieu_P1.tar` · 5,1 GB | [mục Huấn](#huấn-p1) |
| Nhật | P2 | Nhãn 01: ghi âm thật, truyền tải qua ứng dụng, cổng 01, kho tệp thật, bộ âm cứng | `lane/p2` | `du_lieu_P2.tar` · 13,2 GB | [mục Nhật](#nhật-p2) |
| Khôi | P3 | Nhãn 02: khử ồn chuyên nghiệp / mạnh, đổi giọng (cả tệp · một đoạn) | `lane/p3` | `du_lieu_P3.tar` · 6,3 GB | [mục Khôi](#khôi-p3) |
| Long | P4 | Nhãn 03 lời thật: ghép / chèn, xoá lời | `lane/p4` | `du_lieu_P4.tar` · 6,6 GB | [mục Long](#long-p4) |
| Minh | P5 | Nhãn 03 giọng máy: một đoạn · cả tệp · máy hay thật | `lane/p5` | `du_lieu_P5.tar` · 6,5 GB | [mục Minh](#minh-p5) |

Chi tiết việc từng lane: `lanes/<Px>/CLAUDE.md`. Ranh giới nhãn giữa các lane: `lanes/DU_LIEU.md` mục 2.

## Trước khi bắt đầu (mọi người, một lần)

1. Gửi tên tài khoản GitHub cho nhóm trưởng. Mở email mời từ GitHub, bấm **Accept invitation**.
2. Cài Git, Python 3.13 và ffmpeg — mở **PowerShell**, dán:
   ```powershell
   winget install --id Git.Git -e
   winget install --id Python.Python.3.13 -e
   winget install --id Gyan.FFmpeg -e
   ```
   Cài xong thì đóng và mở lại cửa sổ.
3. Cài **Claude Code**: extension «Claude Code» trong VS Code, hoặc bản dòng lệnh.
4. Ổ `D:` trống khoảng 10 GB, cộng dung lượng gói dữ liệu của bạn.

Từ đây mọi khối `bash` dán vào **Git Bash**, mỗi khối một lần; không gõ dấu `$`.

---

## Huấn (P1)

**P1** — Khung cố định: kiểm điều kiện K1 (Constraint → 04), luật nhãn, hồ sơ · ảnh · PDF, khảo sát Req 13. Nhánh git `lane/p1` · dữ liệu `du_lieu_P1.tar` (khoảng 5,1 GB, nhóm trưởng gửi link Drive).

**Bước 1 — Lấy kho đúng lane.** Mở Git Bash, dán:
```bash
git clone -b lane/p1 --single-branch https://github.com/HoangLeKimLam/VoiceAudit.git D:/VoiceAudit
cd /d/VoiceAudit
git branch
```
Phải thấy `* lane/p1`.

**Bước 2 — Giải nén dữ liệu.** Tải `du_lieu_P1.tar` về ổ `D:\` (thành `D:\du_lieu_P1.tar`), rồi dán:
```bash
cd /d/VoiceAudit
tar -xf /d/du_lieu_P1.tar -C /d/VoiceAudit
git status
```
Phải thấy `nothing to commit, working tree clean` (dữ liệu không lọt vào git).

**Bước 3 — Cài môi trường** (vài phút):
```bash
cd /d/VoiceAudit
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m v6.product.tai_mo_hinh
```

Lệnh cuối tải mô hình (khoảng 3,2 GB, một lần) và in 4 dòng `ok`.

**Bước 4 — Kiểm cài đặt:**
```bash
cd /d/VoiceAudit
.venv/Scripts/python.exe -m v6.product.ghep_thu selftest
.venv/Scripts/python.exe -m v6.data.kiem_bo selftest
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name nen
.venv/Scripts/python.exe -m v6.product.golden
.venv/Scripts/python.exe -m v6.product.test_ticket
```
Phải thấy:
- `ghep_thu selftest dat` và `kiem_bo selftest dat`;
- tệp `.tmp_v6/results/ghep_thu_nen.md` ghi Req 1 / 2 / 3 / 4 đúng: **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 %;
- các lệnh còn lại: dòng cuối `N/N dat`, `N/N PASS` hoặc `OK: 0 test hong`.

Lệch hay báo lỗi thì chụp màn hình gửi nhóm trưởng.

**Bước 5 — Câu đầu tiên cho Claude.** Mở Claude Code tại thư mục `D:\VoiceAudit` (VS Code: File → Open Folder → `D:\VoiceAudit`), dán nguyên khối:
```text
Tôi phụ trách P1 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

Đọc kỹ, hết từng tệp, theo đúng thứ tự:
1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
   tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P1/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P1/DU_LIEU.md.
4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
CHƯA đọc lanes/P1/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
là hệ thống hiện tại, không phải đáp án.

Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P1, mỗi ý ghi tệp
và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P1/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
@lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
Bước 5 — Ghi báo cáo đầu tiên lanes/P1/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
(DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
```
Đọc câu trả lời bước 1 của Claude, so với tài liệu: khớp thì cho làm tiếp, lệch thì hỏi nhóm trưởng.

**Các phiên sau** — mở phiên mới, dán:
```text
Tiếp tục việc lane P1. Đọc CLAUDE.md gốc, @lanes/P1/CLAUDE.md và báo cáo mới nhất
lanes/P1/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
```

---

## Nhật (P2)

**P2** — Nhãn 01: ghi âm thật, truyền tải qua ứng dụng, cổng 01, kho tệp thật, bộ âm cứng. Nhánh git `lane/p2` · dữ liệu `du_lieu_P2.tar` (khoảng 13,2 GB, nhóm trưởng gửi link Drive).

**Bước 1 — Lấy kho đúng lane.** Mở Git Bash, dán:
```bash
git clone -b lane/p2 --single-branch https://github.com/HoangLeKimLam/VoiceAudit.git D:/VoiceAudit
cd /d/VoiceAudit
git branch
```
Phải thấy `* lane/p2`.

**Bước 2 — Giải nén dữ liệu.** Tải `du_lieu_P2.tar` về ổ `D:\` (thành `D:\du_lieu_P2.tar`), rồi dán:
```bash
cd /d/VoiceAudit
tar -xf /d/du_lieu_P2.tar -C /d/VoiceAudit
git status
```
Phải thấy `nothing to commit, working tree clean` (dữ liệu không lọt vào git).

**Bước 3 — Cài môi trường** (vài phút):
```bash
cd /d/VoiceAudit
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m v6.product.tai_mo_hinh
```

Lệnh cuối tải mô hình (khoảng 3,2 GB, một lần) và in 4 dòng `ok`.

**Bước 4 — Kiểm cài đặt:**
```bash
cd /d/VoiceAudit
.venv/Scripts/python.exe -m v6.product.ghep_thu selftest
.venv/Scripts/python.exe -m v6.data.kiem_bo selftest
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name nen
.venv/Scripts/python.exe -m v6.scrape.test_scrape
.venv/Scripts/python.exe -m v6.product.golden
```
Phải thấy:
- `ghep_thu selftest dat` và `kiem_bo selftest dat`;
- tệp `.tmp_v6/results/ghep_thu_nen.md` ghi Req 1 / 2 / 3 / 4 đúng: **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 %;
- các lệnh còn lại: dòng cuối `N/N dat`, `N/N PASS` hoặc `OK: 0 test hong`.

Lệch hay báo lỗi thì chụp màn hình gửi nhóm trưởng.

**Bước 5 — Câu đầu tiên cho Claude.** Mở Claude Code tại thư mục `D:\VoiceAudit` (VS Code: File → Open Folder → `D:\VoiceAudit`), dán nguyên khối:
```text
Tôi phụ trách P2 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

Đọc kỹ, hết từng tệp, theo đúng thứ tự:
1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
   tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P2/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P2/DU_LIEU.md.
4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
CHƯA đọc lanes/P2/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
là hệ thống hiện tại, không phải đáp án.

Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P2, mỗi ý ghi tệp
và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P2/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
@lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
Bước 5 — Ghi báo cáo đầu tiên lanes/P2/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
(DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
```
Đọc câu trả lời bước 1 của Claude, so với tài liệu: khớp thì cho làm tiếp, lệch thì hỏi nhóm trưởng.

**Các phiên sau** — mở phiên mới, dán:
```text
Tiếp tục việc lane P2. Đọc CLAUDE.md gốc, @lanes/P2/CLAUDE.md và báo cáo mới nhất
lanes/P2/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
```

---

## Khôi (P3)

**P3** — Nhãn 02: khử ồn chuyên nghiệp / mạnh, đổi giọng (cả tệp · một đoạn). Nhánh git `lane/p3` · dữ liệu `du_lieu_P3.tar` (khoảng 6,3 GB, nhóm trưởng gửi link Drive).

**Bước 1 — Lấy kho đúng lane.** Mở Git Bash, dán:
```bash
git clone -b lane/p3 --single-branch https://github.com/HoangLeKimLam/VoiceAudit.git D:/VoiceAudit
cd /d/VoiceAudit
git branch
```
Phải thấy `* lane/p3`.

**Bước 2 — Giải nén dữ liệu.** Tải `du_lieu_P3.tar` về ổ `D:\` (thành `D:\du_lieu_P3.tar`), rồi dán:
```bash
cd /d/VoiceAudit
tar -xf /d/du_lieu_P3.tar -C /d/VoiceAudit
git status
```
Phải thấy `nothing to commit, working tree clean` (dữ liệu không lọt vào git).

**Bước 3 — Cài môi trường** (vài phút):
```bash
cd /d/VoiceAudit
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Bước 4 — Kiểm cài đặt:**
```bash
cd /d/VoiceAudit
.venv/Scripts/python.exe -m v6.product.ghep_thu selftest
.venv/Scripts/python.exe -m v6.data.kiem_bo selftest
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name nen
```
Phải thấy:
- `ghep_thu selftest dat` và `kiem_bo selftest dat`;
- tệp `.tmp_v6/results/ghep_thu_nen.md` ghi Req 1 / 2 / 3 / 4 đúng: **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 %.

Lệch hay báo lỗi thì chụp màn hình gửi nhóm trưởng.

**Bước 5 — Câu đầu tiên cho Claude.** Mở Claude Code tại thư mục `D:\VoiceAudit` (VS Code: File → Open Folder → `D:\VoiceAudit`), dán nguyên khối:
```text
Tôi phụ trách P3 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

Đọc kỹ, hết từng tệp, theo đúng thứ tự:
1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
   tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P3/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P3/DU_LIEU.md.
4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
CHƯA đọc lanes/P3/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
là hệ thống hiện tại, không phải đáp án.

Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P3, mỗi ý ghi tệp
và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P3/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
@lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
Bước 5 — Ghi báo cáo đầu tiên lanes/P3/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
(DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
```
Đọc câu trả lời bước 1 của Claude, so với tài liệu: khớp thì cho làm tiếp, lệch thì hỏi nhóm trưởng.

**Các phiên sau** — mở phiên mới, dán:
```text
Tiếp tục việc lane P3. Đọc CLAUDE.md gốc, @lanes/P3/CLAUDE.md và báo cáo mới nhất
lanes/P3/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
```

---

## Long (P4)

**P4** — Nhãn 03 lời thật: ghép / chèn, xoá lời. Nhánh git `lane/p4` · dữ liệu `du_lieu_P4.tar` (khoảng 6,6 GB, nhóm trưởng gửi link Drive).

**Bước 1 — Lấy kho đúng lane.** Mở Git Bash, dán:
```bash
git clone -b lane/p4 --single-branch https://github.com/HoangLeKimLam/VoiceAudit.git D:/VoiceAudit
cd /d/VoiceAudit
git branch
```
Phải thấy `* lane/p4`.

**Bước 2 — Giải nén dữ liệu.** Tải `du_lieu_P4.tar` về ổ `D:\` (thành `D:\du_lieu_P4.tar`), rồi dán:
```bash
cd /d/VoiceAudit
tar -xf /d/du_lieu_P4.tar -C /d/VoiceAudit
git status
```
Phải thấy `nothing to commit, working tree clean` (dữ liệu không lọt vào git).

**Bước 3 — Cài môi trường** (vài phút):
```bash
cd /d/VoiceAudit
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Bước 4 — Kiểm cài đặt:**
```bash
cd /d/VoiceAudit
.venv/Scripts/python.exe -m v6.product.ghep_thu selftest
.venv/Scripts/python.exe -m v6.data.kiem_bo selftest
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name nen
```
Phải thấy:
- `ghep_thu selftest dat` và `kiem_bo selftest dat`;
- tệp `.tmp_v6/results/ghep_thu_nen.md` ghi Req 1 / 2 / 3 / 4 đúng: **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 %.

Lệch hay báo lỗi thì chụp màn hình gửi nhóm trưởng.

**Bước 5 — Câu đầu tiên cho Claude.** Mở Claude Code tại thư mục `D:\VoiceAudit` (VS Code: File → Open Folder → `D:\VoiceAudit`), dán nguyên khối:
```text
Tôi phụ trách P4 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

Đọc kỹ, hết từng tệp, theo đúng thứ tự:
1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
   tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P4/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P4/DU_LIEU.md.
4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
CHƯA đọc lanes/P4/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
là hệ thống hiện tại, không phải đáp án.

Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P4, mỗi ý ghi tệp
và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P4/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
@lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
Bước 5 — Ghi báo cáo đầu tiên lanes/P4/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
(DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
```
Đọc câu trả lời bước 1 của Claude, so với tài liệu: khớp thì cho làm tiếp, lệch thì hỏi nhóm trưởng.

**Các phiên sau** — mở phiên mới, dán:
```text
Tiếp tục việc lane P4. Đọc CLAUDE.md gốc, @lanes/P4/CLAUDE.md và báo cáo mới nhất
lanes/P4/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
```

---

## Minh (P5)

**P5** — Nhãn 03 giọng máy: một đoạn · cả tệp · máy hay thật. Nhánh git `lane/p5` · dữ liệu `du_lieu_P5.tar` (khoảng 6,5 GB, nhóm trưởng gửi link Drive).

**Bước 1 — Lấy kho đúng lane.** Mở Git Bash, dán:
```bash
git clone -b lane/p5 --single-branch https://github.com/HoangLeKimLam/VoiceAudit.git D:/VoiceAudit
cd /d/VoiceAudit
git branch
```
Phải thấy `* lane/p5`.

**Bước 2 — Giải nén dữ liệu.** Tải `du_lieu_P5.tar` về ổ `D:\` (thành `D:\du_lieu_P5.tar`), rồi dán:
```bash
cd /d/VoiceAudit
tar -xf /d/du_lieu_P5.tar -C /d/VoiceAudit
git status
```
Phải thấy `nothing to commit, working tree clean` (dữ liệu không lọt vào git).

**Bước 3 — Cài môi trường** (vài phút):
```bash
cd /d/VoiceAudit
py -3.13 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m v6.product.tai_mo_hinh
```

Lệnh cuối tải mô hình (khoảng 3,2 GB, một lần) và in 4 dòng `ok`.

**Bước 4 — Kiểm cài đặt:**
```bash
cd /d/VoiceAudit
.venv/Scripts/python.exe -m v6.product.ghep_thu selftest
.venv/Scripts/python.exe -m v6.data.kiem_bo selftest
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name nen
.venv/Scripts/python.exe -m v6.tts_multi.test_engines
.venv/Scripts/python.exe -m v6.tts_multi.test_pipeline
```
Phải thấy:
- `ghep_thu selftest dat` và `kiem_bo selftest dat`;
- tệp `.tmp_v6/results/ghep_thu_nen.md` ghi Req 1 / 2 / 3 / 4 đúng: **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 %;
- các lệnh còn lại: dòng cuối `N/N dat`, `N/N PASS` hoặc `OK: 0 test hong`.

Lệch hay báo lỗi thì chụp màn hình gửi nhóm trưởng.

**Bước 5 — Câu đầu tiên cho Claude.** Mở Claude Code tại thư mục `D:\VoiceAudit` (VS Code: File → Open Folder → `D:\VoiceAudit`), dán nguyên khối:
```text
Tôi phụ trách P5 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

Đọc kỹ, hết từng tệp, theo đúng thứ tự:
1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
   tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P5/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P5/DU_LIEU.md.
4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
CHƯA đọc lanes/P5/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
là hệ thống hiện tại, không phải đáp án.

Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P5, mỗi ý ghi tệp
và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P5/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
@lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
Bước 5 — Ghi báo cáo đầu tiên lanes/P5/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
(DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
```
Đọc câu trả lời bước 1 của Claude, so với tài liệu: khớp thì cho làm tiếp, lệch thì hỏi nhóm trưởng.

**Các phiên sau** — mở phiên mới, dán:
```text
Tiếp tục việc lane P5. Đọc CLAUDE.md gốc, @lanes/P5/CLAUDE.md và báo cáo mới nhất
lanes/P5/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
```

---

## Làm việc hằng ngày (mọi người)

- Lấy bản cập nhật của nhóm trưởng: `git pull`.
- Lưu việc của bạn lên nhánh mình:
  ```bash
  git add <tệp>
  git commit -m "<mô tả ngắn>"
  git push
  ```
- Không đưa dữ liệu lên git (đã chặn sẵn). Trọng số và OOF `.pkl` của gói chuyên gia gửi nhóm trưởng qua Drive.
- Sau mỗi đợt việc ghi `lanes/<Px>/BAO_CAO_<ngày><tháng>.md`; nộp và đề xuất theo `lanes/BAT_DAU.md` mục 12.

## Gặp lỗi

| Thấy | Làm |
|---|---|
| `$: command not found` | đừng gõ dấu `$` ở đầu lệnh |
| `Repository not found` hoặc lỗi 403 khi clone | chưa bấm Accept invitation, hoặc Git Bash đăng nhập sai tài khoản GitHub |
| `py: command not found` | thay `py -3.13 -m venv .venv` bằng `python -m venv .venv` |
| `tar: ... Cannot open` | tệp `.tar` chưa nằm đúng ở `D:\` |
| báo «không có tệp» dù tệp có | đường dẫn quá dài: kho phải ở `D:/VoiceAudit` |
| máy không có ổ `D:` | trong mọi lệnh thay `D:/` thành `C:/` và `/d/` thành `/c/` |
| số ở bước 4 lệch | chụp màn hình gửi nhóm trưởng |

## Kho gồm gì

- Nhánh `main`: tài liệu hệ thống + mã chung (đo, kiểm, ghép thử).
- Nhánh `lane/p1` … `lane/p5`: `main` + mã riêng của lane + `CLAUDE.md` ở gốc (Claude Code tự đọc để biết bạn là lane nào, đối chiếu với lane nào).
- Tài liệu bắt đầu: `lanes/BAT_DAU.md`.
