# Đề xuất P3 — 06/11/2026: tệp thật trùng byte đặt ngưỡng `nen` tiếng Việt; một nhóm người chiếm 38 % tệp thật vi

**Người đề xuất:** P3 · **Gửi:** P0 (chủ dữ liệu natvi / quy tắc ngưỡng) · **Loại:** dữ liệu + ghi chú đo — P3 **không** tự sửa manifest hay `ghep_thu`.

## 1. Bằng chứng

1. **Hai tệp thật trùng byte** trong `data_labels/v6/natvi` (cfg `goc`):
   `v6nat_nguyen_ban_0318` và `v6nat_nguyen_ban_0393` — cùng md5 `b9c45de6c0…`, cùng người `celeb:id00882`, cùng fold 4, hai `host_file` khác tên
   (`celeb_test-00000-of-00010_00044`, `_00051`). Quét toàn bộ natvi / naten `goc` có `.wav`: đây là **cặp trùng duy nhất**.
2. Cặp này là **hai điểm cao nhất** của đầu `nen` trên tệp thật vi ở mọi mô hình đã thử (p3_v1: 0,979; p3_v2 các nhánh: 0,96 / 0,96).
   Ngưỡng đầu cả tệp ở ngân sách 0,25 % = phân vị 99,75 % trên ~750 tệp thật vi của 4 fold còn lại ≈ điểm thứ 2 → với fold 0–3, ngưỡng `nen` vi
   do chính cặp trùng này quyết định (đếm hai lần một bản ghi).
3. Người `celeb:id00882` (và `celeb:id00945`) có các bản ghi «trông như đã khử ồn» theo cả đặc trưng XLS-R lẫn số đo vật lý (sàn nhiễu sâu, ít nhiễu nền) —
   có thể nguồn gốc đã qua xử lý trước khi vào kho.
4. **Một nhóm người chiếm 38 % tệp thật vi:** `vss:nguyenkhangofficial` = 355 / 939 tệp thật vi, toàn bộ ở fold 0
   (kèm 160 / 361 tệp `nen_toan_cuc` vi). Mô hình fold 0 không thấy miền `vss` khi học; ngưỡng fold 0 tính trên tệp thật không có `vss`.

Lệnh tái tạo: `python experts/P3/p3_v2/diag_heads.py experts/P3/p3_v1/heads_p3.pkl 0` (in tệp thật điểm cao nhất theo ngôn ngữ).

## 2. Đề xuất
- (a) Bỏ một trong hai tệp trùng (chuyển `_quarantine` kèm nhật ký, luật cứng 7) hoặc đánh dấu `trung_lap` để `Ctx9` không đếm hai lần.
- (b) Xem lại nguồn `celeb:id00882`, `celeb:id00945`: nếu bản gốc đã qua khử ồn mạnh thì không còn là «không chỉnh sửa» chuẩn để đặt ngưỡng 02.
- (c) Ghi chú cho mọi lane: ở ngân sách 0,25 % / 0,5 %, ngưỡng vi do 2–4 tệp quyết định; nên báo kèm số **theo fold** và khoảng tin cậy.
- (d) Cân nhắc chia `vss:nguyenkhangofficial` theo video (nhóm kênh → nhóm video) nếu chính sách fold cho phép — hiện một fold mang gần nửa miền vi.

Cho tới khi P0 quyết, P3 vẫn đo đúng `ghep_thu` hiện hành và báo số như cũ.

## Theo mẫu
- **Lane / người đề xuất:** P3
- **Muốn đổi gì:** bỏ / đánh dấu tệp thật trùng byte `v6nat_nguyen_ban_0393` (bản sao của `_0318`) khỏi tập đặt ngưỡng.
- **Vì sao:** mục 1 (OOF, không dùng tập chấm).
- **Ghép thử:** chưa chạy với dữ liệu đã sửa — P3 không sửa manifest.
- **Ảnh hưởng tới lane khác:** mọi kênh / đầu đặt ngưỡng trên tệp thật vi (đổi rất ít trừ khi tệp này nằm ở đuôi của kênh đó).
- **Cách quay lại nếu không ổn:** khôi phục từ `_quarantine`.
- **P0 duyệt:**
