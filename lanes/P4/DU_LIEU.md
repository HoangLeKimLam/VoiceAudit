# P4 — Dữ liệu: phạm vi, nhãn, ví dụ

> Đọc sau `lanes/DU_LIEU.md`. **Cố định:** phạm vi phải phủ (mục 1), nhãn và ranh giới (mục 2), kiểm trước khi nộp (`DU_LIEU.md` mục 6.5). **Tự do:** nguồn, cách sinh, công cụ, số lượng. Dữ liệu bạn nhận (mục 4) chỉ là **ví dụ**.
>
> **Việc của bạn với dữ liệu:** tự thu thập — cào nguồn mở, tự ghi, tự sinh, tìm kho mở — **dữ liệu đúng dạng ở mục 2–3** cho **đủ các trường hợp ở mục 1**. Dữ liệu mẫu ở mục 4 **không phủ hết** phạm vi; nguồn ở mục 6 chỉ là điểm xuất phát. Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết.
>
> **Dạng dữ liệu cần thu:** **lời thật bị ghép (thay / chèn) hoặc bị xoá** — mỗi chỗ sửa ≤ 5 s, có mốc và nguồn lời ghép; ưu tiên sửa tay thật; ghép cùng và khác bản ghi; sửa rồi mới qua điều kiện ngoài đời; có cả tệp dài.

## 1. Phạm vi bạn phải phủ

| Trường hợp | Vì sao |
|---|---|
| **Sửa tay thật** ≥ 150 ca: thay, chèn, xoá; 1 tiếng → 1 câu | danh mục test; tập sửa tay báo riêng |
| Ghép **cùng bản ghi** (cùng thiết bị, cùng phòng) | Scope «cùng bản ghi» |
| Ghép từ **bản ghi khác**, khác thiết bị, khác phòng | Scope «bản ghi khác» |
| Xoá từ **một âm tiết** đến một câu | Scope; Assumption: phần bị xoá ≥ 1 âm tiết |
| Mỗi đoạn chèn / thay ≤ 5 s, ≥ 0,1 s; lời sau sửa vẫn mạch lạc | Scope + Assumption |
| Sửa rồi mới qua vang, ồn, TV, nén, ứng dụng, chồng điều kiện | điều kiện ngoài đời áp sau khi sửa |
| Chỗ sửa ngắn trong **tệp dài** 1–10 phút | Input 3 giây – 10 phút |
| Cả vi và en; số lượng đủ để báo riêng có khoảng tin cậy 95 % | «Điều kiện chấm» |

## 2. Nhãn của bạn và ranh giới

| Của bạn | Không phải của bạn |
|---|---|
| `03_ghep_giong_that`, `op = "ghep_thay"` (`sub = "thay_that"`) / `"ghep_chen"` (`sub = "chen_that"`), có `donor_file` | đoạn chèn / thay **do máy tạo** → `03_chen_giong_may` (P5) — kể cả khi nhái rất giống |
| xoá lời: `03_ghep_giong_that` + `op = "xoa_loi"` + **`sub = "cat_bo"`** + `joint` (không dùng mã `03_xoa_loi`) | khử ồn / đổi giọng kèm theo → thêm dòng `02_*` riêng (đa thao tác), không gộp vào dòng của bạn |

## 3. Bản ghi mẫu

```json
{"id": "p4x_tay_0012", "kind": "ghep_thay", "cfg": "goc", "src": "p4x_tay_0012",
 "wav": "../v6/p4x_suatay/ghep_thay/p4x_tay_0012.wav",
 "meta": {"lang": "vi", "source": "vox", "person": "vox:id00092", "speaker": "vox:id00092",
          "level1": ["03_ghep_chen"], "level2": ["03_ghep_giong_that"], "source_edit": "tay",
          "edits": [{"level2": "03_ghep_giong_that", "kind": "segment", "op": "ghep_thay", "sub": "thay_that",
                     "start": 3.20, "end": 4.05, "donor_file": "vox_test-00001-of-00038_00258",
                     "removed": "ngày mai", "inserted": "hôm qua", "tool": "Audacity", "nguoi_sua": "ngoai_nhom_03"}]}}
```

Bản ghi xoá lời mẫu: `lanes/DU_LIEU.md` mục 2.4.

## 4. Dữ liệu mẫu bạn nhận (ví dụ — không phủ hết phạm vi)

Nhóm trưởng chọn từ dữ liệu sẵn có để bạn hiểu định dạng, tái tạo baseline và bắt đầu nhanh — **gợi ý, không bắt buộc dùng**. Dòng «*thêm*» là dữ liệu khác có sẵn trên máy gốc, xin nếu thấy có ích.

| Gói | Nội dung | Dung lượng |
|---|---|---|
| chung | điểm OOF hệ thống + manifest natvi / naten | 0,12 GB |
| khung + vector XLS-R | đặc trưng đã trích (đã bỏ id có cờ) | 2,4 GB |
| âm thanh 03 lời thật | natvi / naten: ghép thay / chèn, xoá, các mức kỹ / vụng, đa thao tác, tệp thật; `amilong` | 4,0 GB |
| MIST | bộ ngoài tiếng Việt có bản gốc thật | 0,14 GB |
| *thêm* | kho gốc sinh 1,75 GB · bộ phủ điều kiện 1,6 GB · đặc trưng họ khác | |

## 5. Khi tự thêm dữ liệu

- Theo `DU_LIEU.md` mục 6. Ca sửa tay ghi `meta.source_edit = "tay"`, công cụ, người sửa, mốc từng chỗ sửa.
- **Không học trên tập sửa tay** — chỉ chấm, báo riêng (luật lane).
- `meta.person` = người nói của bản gốc; nguồn lời ghép (`donor_file`) cùng người nói khi là ghép lời thật của chính người đó.

## 6. Gợi ý nguồn (điểm xuất phát — tự tìm thêm)

| Cho | Gợi ý | Lưu ý |
|---|---|---|
| Sửa tay | Audacity, Adobe Audition, Reaper, CapCut, ứng dụng sửa âm trên điện thoại | tệp gốc là giọng người tham gia hoặc kho nghiên cứu |
| Vang, ồn sau khi sửa | OpenSLR 28 (Apache 2.0), BUT ReverbDB (CC BY 4.0), MUSAN (CC BY 4.0) | — |
| Tệp dài | AMI đầy đủ (`data/ami`), ICSI (CC BY 4.0), tệp dài của P2 | — |
| Kiểm tra ngoài (không phải trường hợp phủ) | PartialSpoof, LlamaPartialSpoof, ADD 2023 track 2 | kiểm giấy phép; NC → chỉ đo, hỏi trước |

## 7. Việc cần người thật

≥ 150 ca sửa tay của người ngoài nhóm (cùng P0), với một trang hướng dẫn chung (kiểu sửa, độ dài, cách xuất mốc) · tự ghi một người trên hai thiết bị để ghép khác thiết bị.
