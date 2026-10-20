# P0 — Nhóm trưởng: quyết định · tích hợp · ngưỡng · chấm hệ thống

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ chính **V4** (đặt ngưỡng từng thao tác · giải mã dòng · gộp dòng nhiều chuyên gia · giới hạn vùng soi), **V5** (kiểm chứng độc lập · nhiều thao tác và vùng xám · thứ tự áp thao tác), **V7** (khung 20 ms · vector cả tệp · tốc độ xử lý · tệp dài và ghi âm thật · kiểm thử và tập khoá); cùng P2 hợp nhất **chuyên gia sàng lọc** của **V2**. Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

Bạn giữ **phần chung và các mối nối**: XLS-R dùng chung, mô hình đa nhiệm FT-2, **K7** (giải mã, ngưỡng theo ngôn ngữ × độ dài, gộp dòng, luật vùng xám), ghép gói các lane, chấm đầu-cuối, tập khoá, hàng đợi máy. Mọi lane nộp cho bạn; bạn trả lại "ô nào tụt, vì sao".

*Sơ đồ trên là hệ thống hiện tại (baseline). Cố định chỉ là chỗ cắm — nhận gì, nộp gì đúng hợp đồng (CHUNG §4) và trong ngân sách; bên trong khối của bạn làm bằng cách nào cũng được (mô hình nền khác, âm thanh gốc, số đo tay, kết hợp…).*

## 2. Đích đến
Một **điểm vận hành công bố** có đủ 15 Req đo được kèm khoảng tin cậy, đúng điều kiện chấm (`VoiceAudit.html`); mọi ô ma trận phủ có số hoặc ghi "trống có lý do"; tập khoá chấm **một lần**; RTF ≤ 0,6.

## 3. Hiểu bài toán của mảng

Bạn trả lời: từ điểm của các chuyên gia, hệ thống có ra **đúng dòng thao tác và đúng nhãn** không, giữ **vu oan tổng ≤ 3 %**, và con số công bố có **trung thực** không.

**Bài toán khó ở đâu (bằng chứng hiện có):**
- Vu oan **cộng dồn**: mỗi kênh nổ nhầm một chút, ghép lại thành nhiều — hiện tiếng Việt 5,3 % trên tệp 01 gốc ngay cả khi mỗi kênh giữ ngân sách.
- **Độ dài tệp**: tệp dài có nhiều cơ hội nổ nhầm. Ngưỡng theo độ dài làm vu oan tệp 1–10 phút giảm 51 → 15 % nhưng bắt sửa tụt 0,75 → 0,35, nhãn 03 đúng 48 → 5 %.
- **Lệch miền**: ghi âm thật vu oan 13 % — gần hết là tệp < 20 s, 84 / 95 tệp bị gán 02; điều kiện thật khác dữ liệu tự tạo.
- **Đo trung thực**: ngưỡng chỉ chọn trên OOF / dev; tập khoá chấm một lần; tập khoá tiếng Anh người mới chưa có.

**Câu hỏi còn mở:** vì sao ngưỡng tiếng Việt không giữ được ngân sách (lệch fold? nguồn VSS?); gộp điểm từ khúc lên tệp thế nào để tệp dài giữ được cả vu oan lẫn phát hiện; một điểm vận hành hay theo nguồn tệp.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
- Bảng hệ thống: `CHUNG.md §1.3–1.4`. Tái tạo OOF: `python -m v6.product.ghep_thu --name nen` (điểm F) và `--name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025` (ngân sách 3 %: Req 4 en 3,8 %, vi 7,0 %; trên tệp 01 gốc en 2,6 % đạt, vi 5,3 % chưa).
- **Đổi nhãn 20/10, mô hình chưa học lại** (`CHUNG.md §1.4`): 212 tệp WebRTC NS mức ≤ 2 nay là 01 vừa bị tính vu oan, vừa đẩy ngưỡng kênh khử ồn đoạn lên. Baseline sạch cần học lại trên nhãn mới — việc của bạn (`P0/DU_LIEU.md`).
- **Ngưỡng theo độ dài** (số trước đổi nhãn; 18/10, `lencal_v9F.json`, `python -m v6.product.lencal check | regress`): dev vu oan tệp 1–10 phút en 27→98 % thành 0–7,5 %, vi 40→91 % thành 4–9 %; AMI 14/24 → 3/24; tệp ngắn en giữ Req 2, vi Req 2 0,563 → 0,550 / Req 4 9,4 → 7,9 %. **Chưa chấm lại trên tập kiểm thử** (lane L).
- Danh mục test §4B: ~13/46 đo đủ, 18 một phần, 15 chưa có. T0.8 RTF đạt (≈ 90 s / 300 s); T0.7 chênh vi–en 0,07–0,09 (chỉ tiêu ≤ 0,08).

## 5. Bạn sở hữu (chỉ P0 sửa)
`v6/product/pipeline.py`, `e2e_test.py`, `lencal.py`, `ghep_thu.py`, `calib.py` · `v6/boundary/fuse9.py`, `fusion.py`, `rowmerge.py`, `ens6.py` · `thresholds_v9*.json`, `lencal_v9*.json`, `calib_*.json` · FT-2 đa nhiệm (`ft_ft2_*.pt`, `v6/boundary/ft.py`, `ft_cv.py`) · `v6/scrape/long_sets.py` · hàng đợi `.tmp_v6/wf/*.sh`, `queue_watch.py`, `run_serial.sh` · `VoiceAudit.html`.

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P0/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng
- Ngưỡng / điểm vận hành / ngân sách chỉ đặt trên OOF hoặc dev; mỗi biến thể chốt chấm tập kiểm thử **một lần**, có `--tag`.
- Tập khoá: một lần, cuối cùng, sau khi mọi lane chốt.
- Mỗi lần ghép: `lanes/P0/GHEP_<ddmm>.md` — bảng ô trước / sau, ô tụt và nguyên nhân, gửi từng lane.
- Một việc nặng một lúc trên máy; e2e cần ≥ 6 GB bộ nhớ ảo trống.

## 7. Việc mở cần quyết với người dùng
Gửi bộ gửi theo lane (mã đã bỏ đường dẫn cứng 20/10; quy trình: `P0/DU_LIEU.md` mục 4); `git init` + nhánh theo lane; GPU cho P5 / P3; người làm thật (sửa tay, app thật, thiết bị thật, khảo sát).

## 8. Phối hợp
Nhận gói P1–P5 (CHUNG §4) + báo cáo `ghep_thu`; P2 cấp bộ âm cứng — bạn kiểm không lane nào làm tăng vu oan trên bộ của P2.

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
