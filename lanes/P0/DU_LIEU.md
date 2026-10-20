# P0 — Dữ liệu: giữ gì, chia gì, còn phải làm gì

> Đọc sau `lanes/DU_LIEU.md`. P0 giữ toàn bộ dữ liệu trên máy gốc, chia gói cho P1–P5, giữ tập khoá và tập chấm cuối, và là người **nhận bộ mới** của các lane — chốt cuối để ghép không xung đột.

## 1. Bạn giữ

Mọi thứ ở `DU_LIEU.md` mục 4, gồm cả phần không gửi đi: tập khoá, bản tải gốc `data/scrape/raw`, thư mục cách ly, kho gốc voxvietnam / vietnam_celeb (còn người nói chưa dùng), bản lưu manifest trước khi đổi nhãn (`.tmp_v6/docbak/manifest_truoc_n7_2010/`, `manifest_truoc_qd_2010/`).

## 2. Việc dữ liệu của bạn

| Việc | Vì sao | Cách làm |
|---|---|---|
| ~~Đóng băng lại tập khoá~~ — **đã làm 20/10** (+1 850 mục cào) | | chạy lại `freeze` mỗi khi manifest / hosts / `engines.json` đổi |
| ~~Đổi nhãn khử ồn ứng dụng, gắn cờ 554 tệp nhân bản~~ — **đã làm 20/10** (quyết định 18) | | 971 tệp WebRTC NS mức ≤ 2 → 01 (`meta.edits_khong_tinh`), 106 tệp bớt một thao tác; 618 đoạn không rõ mức + 6 id trùng ở `chain` → `tam_loai`; `load_items` và `goi_lane` đã bỏ các cờ này |
| **Học lại baseline theo nhãn mới** | mô hình hiện có học nhãn cũ: 212 tệp nay là 01 vừa bị tính vu oan, vừa đẩy ngưỡng kênh khử ồn đoạn lên (`CHUNG.md §1.4`) | chạy lại đúng công thức baseline (TCN + FT-2 LoRA, 5 fold) một lần để mọi lane có mốc sạch — FT-2 cần GPU hoặc CPU rất lâu → **hỏi người dùng trước**; lane học kênh của mình theo nhãn mới |
| Sinh lại **618 đoạn WebRTC NS** có ghi `ns_level` | đang tạm loại (`tam_loai`) | cùng P3; mức ≤ 2 → không ghi thao tác, mức 3 → 02 |
| **Chốt số thang khử ồn ứng dụng** (01 / 02) | thang hai bậc hiện là tạm: mặc định / đơn sơ = 01, bật mạnh / AI = 02 | P2 đo khử ồn trên ứng dụng thật, P3 đo công cụ — đề xuất số, bạn duyệt rồi cập nhật `DU_LIEU.md` mục 2.2 |
| Dựng **tập khoá tiếng Anh** từ 17 người VoxPopuli chưa dùng | đã chốt nguồn 20/10 (`curate --add voxp`) | dựng bộ từ kho gốc `data/natural_v6/en_voxpopuli`, rồi `freeze` |
| Chia gói cho P1–P5 | thành viên làm trên máy riêng | `python -m v6.data.goi_lane plan`, rồi `copy --lane <Px> --dest <ổ ngoài>` qua `run_serial.sh` |
| Sinh **tập công cụ chưa thấy** | «Điều kiện chấm»: mỗi loại ≥ 1 bộ không có khi học | chỉ P0 dùng 4 công cụ để dành (`vieneu`, `styletts2`, `knnvc`, `resenh`) — khi GPU được phép |
| Đủ tập chấm theo «Điều kiện chấm» | chỉ tiêu chốt | 8 cấu hình nén; ≥ 3 bộ khử ồn, ≥ 2 đổi giọng, ≥ 2 sinh giọng; xoá lời, giọng máy cả tệp; sửa ngắn / dài; nhiều người nói; một phần ghi âm thật (MIST) |
| Ghi âm **thiết bị thật** 30–50 tệp | phủ thiết bị; tệp chắc chắn 01 | nhiều điện thoại, máy ghi âm, laptop, phòng khác nhau — cùng P2 |
| Sửa tay ≥ 150 ca | dữ liệu sửa thật | cùng P4; giữ một phần làm tập khoá |
| Tệp dài tiếng Việt nhiều nhóm hơn | tệp dài vi chỉ 11 nhóm | lấy từ kênh mới P2 cào được |
| Duyệt kênh YouTube mới | chính sách cào | chỉ kênh không chuyên, CC BY / BY-SA / CC0 |
| VSS | 37 % natvi, 17 % vu oan lịch sử | báo số tiếng Việt cả khi có và khi bỏ VSS |

## 3. Nhận bộ mới từ các lane — chốt chống xung đột

Mỗi bộ mới phải qua đủ các bước dưới đây **trước** khi vào dữ liệu chung:

1. `python -m v6.data.kiem_bo data_labels/v6/<bo>/manifest.json --moi` — không còn lỗi (mã nhãn, quy ước xoá lời, giọng máy có công cụ sinh, chuyển giọng không phải 03, WebRTC có `ns_level`, chính sách nhái giọng, id trùng giữa các bộ).
2. `python -m v6.data.lockbox check data_labels/v6/<bo>/manifest.json` — không chạm tập khoá.
3. Có thẻ README: nguồn, giấy phép, công cụ (công cụ nào giữ kín), số tệp / người nói, lane làm.
4. Cập nhật `DU_LIEU.md` mục 3 (phủ) và mục 4 (kho); chạy lại `lockbox freeze` nếu bộ có nguồn mới.
5. Hai lane gán khác nhau cho cùng một loại dữ liệu → **giữ cả hai, không gộp**, ghi vào `DU_LIEU.md` mục 7 «Còn mở» rồi quyết theo ranh giới mục 2.2.

## 4. Gửi bộ gửi cho thành viên

Mỗi người nhận **một thư mục bộ gửi** = tài liệu + mã cần thiết + dữ liệu mẫu đúng lane (`BAT_DAU.md` mục 10.2–10.3). Mã chọn theo tệp thực sự chạy khi làm các việc tài liệu bắt buộc; mã / báo cáo thử nghiệm cũ không gửi.

1. Xem trước: `python -m v6.data.goi_lane plan` (dung lượng, số mục bị loại vì tập khoá / chính sách).
2. Danh sách đầy đủ để kiểm: `python -m v6.data.goi_lane liet-ke --lane <Px> --them "*" --out .tmp_v6/goi/DANH_SACH_<Px>.txt` (mỗi dòng một tệp, theo gói; «(ban da loc)» = siêu dữ liệu đã bỏ tập khoá).
3. Chép ra ổ ngoài (lần lượt từng lane): `bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.goi_lane copy --lane <Px> --dest E:/goi_<Px> [--them ...]`. Sửa tài liệu / mã sau này → gửi lại phần đổi: `... copy --lane <Px> --dest E:/goi_<Px> --chi tai_lieu,ma_chung`.
4. Gửi thư mục (ổ cứng / Drive) + nhắn: «đọc `lanes/BAT_DAU.md`, cài theo mục 10.2, rồi làm theo mục 11».
5. Thành viên P1, P2, P5 tự tải mô hình Hugging Face (`python -m v6.product.tai_mo_hinh`, khoảng 3,2 GB) — không đóng gói.
6. Gửi qua git (mã + tài liệu) và Drive (dữ liệu): tách bằng `--phan ma` / `--phan du_lieu`. Hướng dẫn từng lệnh: `D:\VoiceAudit_gui\HUONG_DAN_GUI.md` — nhánh `main` = phần chung, `lane/p1…p5` = `main` + mã riêng của lane; dữ liệu là một tệp `.tar` riêng cho từng người.
