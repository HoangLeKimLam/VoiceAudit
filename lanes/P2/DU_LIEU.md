# P2 — Dữ liệu: phạm vi, nhãn, ví dụ

> Đọc sau `lanes/DU_LIEU.md`. **Cố định:** phạm vi phải phủ (mục 1), nhãn và ranh giới (mục 2), kiểm trước khi nộp (`DU_LIEU.md` mục 6.5). **Tự do:** nguồn, cách cào / ghi, số lượng. Dữ liệu bạn nhận (mục 4) chỉ là **ví dụ**.
>
> **Việc của bạn với dữ liệu:** tự thu thập — cào nguồn mở, tự ghi, tự sinh, tìm kho mở — **dữ liệu đúng dạng ở mục 2–3** cho **đủ các trường hợp ở mục 1**. Dữ liệu mẫu ở mục 4 **không phủ hết** phạm vi; nguồn ở mục 6 chỉ là điểm xuất phát. Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết.
>
> **Dạng dữ liệu cần thu:** **ghi âm thật không chỉnh sửa (01)** — tự ghi trên thiết bị / ứng dụng thật, hoặc cào kênh không chuyên có giấy phép mở qua chuỗi `v6/scrape`; phủ sáu biến đổi lành tính và khử ồn đơn sơ của ứng dụng; tiếng Việt và tiếng Anh, nhiều người, nhiều vùng miền, có cả tệp dài; ghi rõ thiết bị / ứng dụng.

## 1. Phạm vi bạn phải phủ

| Trường hợp | Vì sao |
|---|---|
| Sáu hàng «Nhãn 01 phủ những gì»: nén khi ghi (MP3, AAC, Opus) · nén lại khi gửi qua nền tảng, nhiều lần · tinh chỉnh mặc định của thiết bị · đổi định dạng / tần số lấy mẫu · cắt đầu–cuối, đổi âm lượng tổng, tốc độ ± 5 % · khử ồn đơn sơ có sẵn của ứng dụng | định nghĩa 01; đích lane: mỗi hàng ≥ 10 tệp, ≥ 5 người nói |
| Ghi âm thật tiếng Việt nhiều người, nhiều vùng miền, không hậu kỳ | Req 4 trên ghi âm thật; kho tệp thật làm mốc so sánh |
| Tệp **chắc chắn 01** trên thiết bị thật | bản cào chỉ là «01 có khả năng» |
| Truyền tải qua **ứng dụng thật** (tin nhắn thoại, cuộc gọi, họp trực tuyến) | Input «nhận lại qua ứng dụng» |
| 8 cấu hình nén, nén lặp, DTX, tự chỉnh mức, tạm dừng ghi | «Điều kiện chấm»: ≥ 24 kbps tính chỉ tiêu, còn lại báo riêng |
| **Khử ồn đơn sơ của từng ứng dụng thật**, đo mức tác động | chốt số cho thang khử ồn (01 / 02) — cùng P3 |
| Tiếng Anh tự nhiên đa dạng; hội thoại 2–4 người tiếng Việt | Req 4 en; số người nói |
| Bộ âm cứng: đoạn tệp thật làm các kênh báo nhầm | cho P3 / P4 / P5 học tránh vu oan |

## 2. Nhãn của bạn và ranh giới

- Nhãn của bạn là **01**: không có `edits`, `level2 = []`, `level1 = ["01_nguyen_ban"]`.
- Biến đổi lành tính (sáu hàng ở trên) **không ghi là thao tác**; ghi mô tả vào `meta.kenh_meta` / `meta.ung_dung` để báo riêng.
- **Khử ồn:** đơn sơ của ứng dụng (WebRTC NS mức 0–2, chế độ mặc định) → 01, của bạn. Chuyên nghiệp hoặc mạnh (mức 3, chế độ «Cao», AI, công cụ khử ồn) → 02, của P3. Khi tự ghi, ghi rõ ứng dụng và mức đã đặt.
- **Tự ghi** dùng `source = "tuthu"` — cách đánh dấu tệp 01 chắc chắn. Bản cào giữ `source` gốc (`yt`, `fosdem`, `ccc`) và chỉ là «01 có khả năng».
- Bản tự sửa dựng từ ghi âm thật (như trong `scrapevi`) theo đúng quy ước nhãn của thao tác đó (`DU_LIEU.md` mục 2.3).

## 3. Bản ghi mẫu

```json
{"id": "p2x_tuthu_0007", "kind": "nguyen_ban", "cfg": "goc", "src": "p2x_tuthu_0007",
 "wav": "../v6/p2x_tuthu/nguyen_ban/p2x_tuthu_0007.wav",
 "meta": {"lang": "vi", "source": "tuthu", "person": "tuthu:nguoi05", "speaker": "tuthu:nguoi05",
          "level1": ["01_nguyen_ban"], "level2": [], "edits": [],
          "thiet_bi": "iPhone", "ung_dung": "Zalo — gọi thoại, ghi lại", "khu_on_ung_dung": "mặc định"}}
```

## 4. Dữ liệu mẫu bạn nhận (ví dụ — không phủ hết phạm vi)

Nhóm trưởng chọn từ dữ liệu sẵn có để bạn hiểu định dạng, tái tạo baseline và bắt đầu nhanh — **gợi ý, không bắt buộc dùng**. Dòng «*thêm*» là dữ liệu khác có sẵn trên máy gốc, xin nếu thấy có ích.

| Gói | Nội dung | Dung lượng |
|---|---|---|
| chung · hệ thống | điểm OOF + manifest natvi / naten · trọng số chạy đầu–cuối | 0,19 GB |
| vector cả tệp | `xlsr_pooled` — chỉ có cho tệp tự tạo; ghi âm cào / tệp dài cần tự trích nếu muốn dùng | 0,4 GB |
| ghi âm thật ngoài đời | `scrapevi`, `scrapeen`, `longvi`, `longen`, `benign`; bản cắt `data/scrape/hosts_*`; `real_pilot`; kho phân vị 160 tệp VIVOS (NC — nội bộ) | 10,5 GB |
| khung XLS-R | ghép thử cần (chỉ kiểm có tệp) | 2,0 GB |
| *thêm* | 60 tệp mỗi kiểu 0,8 GB | |

Đã loại tập khoá (kênh cào `crc32 % 5 == 0`) — cả âm thanh lẫn dòng manifest / `hosts.jsonl`. Bản tải gốc `data/scrape/raw` không gửi. Trạng thái chuỗi cào (`data/scrape/meta`, kèm theo mã chuỗi cào) cũng đã gỡ mọi dòng của 102 kênh tập khoá; các tệp chuỗi tự tạo lại được (vector giọng, mẫu nghe, báo cáo QC) không gửi. Nếu chuỗi tìm lại một kênh tập khoá, hoặc kênh mới rơi vào quy tắc trên, dữ liệu đó **là tập khoá**: `lockbox check` sẽ báo — không học, không chọn ngưỡng, giao lại P0.

## 5. Khi tự thêm dữ liệu

- **Cào:** chỉ qua chuỗi `v6/scrape` (`DU_LIEU.md` mục 6.1) — bạn giữ mã này. `meta.person` của bản cào = nhóm kênh, nên fold và tập khoá tính theo kênh.
- **Tự ghi:** `DU_LIEU.md` mục 6.2.
- Dữ liệu cào là **tập kiểm thử của cả nhóm**: khi cấp làm âm cứng / dữ liệu học, giữ đúng fold theo nhóm kênh; không đưa kênh tập khoá vào.

## 6. Gợi ý nguồn (điểm xuất phát — tự tìm thêm)

| Cho | Gợi ý | Lưu ý |
|---|---|---|
| Ghi âm thật tiếng Việt | YouTube CC BY kênh không chuyên qua `v6/scrape` | kênh mới nhóm trưởng duyệt |
| Đa thiết bị, giọng đọc | Common Voice vi / en (CC0), FLEURS (CC BY 4.0) | là giọng đọc, không thay hội thoại tự nhiên |
| Tiếng Anh tự nhiên | People's Speech (CC BY / BY-SA), VoxPopuli (CC0), FOSDEM / CCC (CC BY) | — |
| Tệp chắc chắn 01, ứng dụng thật | tự ghi trên nhiều điện thoại, máy ghi âm, laptop, tai nghe Bluetooth; gửi qua Zalo, Messenger, Viber, Telegram, WhatsApp; gọi VoLTE / Zalo; họp Zoom / Meet / Teams | chỉ giọng người tham gia, có đồng ý |
| Biến đổi lành tính | ffmpeg trên tệp thật; chuỗi chuyển tiếp qua ứng dụng thật | — |
| Hội thoại nhiều người tiếng Việt | tự ghi nhóm; kho nghiên cứu hội thoại | kho NC → hỏi nhóm trưởng |

## 7. Việc cần người thật

Gửi tệp qua ứng dụng thật, gọi điện, họp trực tuyến · tự ghi nhiều thiết bị, hội thoại nhóm · nhóm trưởng duyệt kênh YouTube mới.
