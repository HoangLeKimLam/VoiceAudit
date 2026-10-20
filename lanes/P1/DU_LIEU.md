# P1 — Dữ liệu: phạm vi, nhãn, ví dụ

> Đọc sau `lanes/DU_LIEU.md`. **Cố định:** phạm vi phải phủ (mục 1), quy ước ghi đáp án và nhãn (mục 2–3), kiểm trước khi nộp (`DU_LIEU.md` mục 6.5). **Tự do:** nguồn, cách tạo, công cụ, số lượng. Dữ liệu bạn nhận (mục 4) chỉ là **ví dụ**.
>
> **Việc của bạn với dữ liệu:** tự thu thập — cào nguồn mở, tự ghi, tự sinh, tìm kho mở — **dữ liệu đúng dạng ở mục 2–3** cho **đủ các trường hợp ở mục 1**. Dữ liệu mẫu ở mục 4 **không phủ hết** phạm vi; nguồn ở mục 6 chỉ là điểm xuất phát. Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết.
>
> **Dạng dữ liệu cần thu:** tệp **kiểm điều kiện** — ghi âm (thật, hoặc biến đổi từ tệp sạch) nằm dưới / đúng / trên ngưỡng của từng hàng Constraint, mỗi tệp kèm **đáp án mong đợi** (accept / cảnh báo / 04 + lý do), không mang nhãn sửa; tệp **ngoài phạm vi** (phát lại qua loa, phá dấu vết, băng hẹp); hồ sơ cho khảo sát Req 13.

## 1. Phạm vi bạn phải phủ

| Trường hợp | Vì sao (bản nộp / đích lane) |
|---|---|
| Mỗi hàng trong 24 hàng Constraint ở 3 mức — dưới · đúng · trên ngưỡng — × vi / en | test chủ động 24 hàng; K1 chặn oan ≤ 2 % |
| Tiếng ồn thật nhiều loại, SNR quanh 10 và 15 dB | hàng «tiếng nói so với tiếng ồn» (< 10 dB → 04, 10–15 dB → cảnh báo) |
| Phòng vang quanh RT60 1,0 s | hàng «phòng» (> 1,0 s → cảnh báo) |
| Nhạc / TV / loa nền quanh 10 % năng lượng | hàng «âm không phải tiếng nói» (> 10 % → 04) |
| Không phải ghi âm lời nói: podcast, phát thanh, sách nói, nhạc | hàng «loại âm thanh» (→ 04) |
| Hát, thì thầm, la hét (người lớn) | hàng «kiểu nói» (→ cảnh báo) |
| Ngôn ngữ khác Việt / Anh | hàng «ngôn ngữ» (→ cảnh báo) |
| 2–4 người, > 4 người, nói chồng quanh 10 % | hàng «số người nói», «nói chồng» |
| Hai kênh: một kênh < 2 s tiếng nói · giống hệt · hai micro một cảnh · hai nguồn riêng | hàng «hai kênh»; PDF hai kênh |
| 8 định dạng, độ sâu bit, tần số lấy mẫu, tốc độ bit quanh 24 kbps, độ dài quanh 3 giây và 10 phút, 250 MB | các hàng «tệp» |
| Ngoài phạm vi: phát lại qua loa, cố ý phá dấu vết, băng hẹp | Assumption — phải ra cảnh báo / 04 đúng |
| 40 hồ sơ cho khảo sát Req 13 (8 mỗi loại sửa) | Req 13 (`VoiceAudit_DanhGia.docx` mục 4.3) |

## 2. Nhãn của bạn và ranh giới

- Dữ liệu của bạn là **tệp kiểm điều kiện**, không phải tệp có sửa: mỗi tệp ghi **đáp án mong đợi** (`accept`, cảnh báo nào, hay `04` với lý do nào), không ghi `edits`.
- Tệp kiểm sinh từ một tệp sạch giữ `meta.person` / `base` của tệp gốc (để fold đúng) và **không mang nhãn sửa**. Muốn ca «điều kiện + có sửa» thì lấy tệp đã sửa của lane khác làm gốc và giữ nguyên nhãn của nó.
- Phát lại qua loa, phá dấu vết → đánh dấu `meta.ngoai_pham_vi`, không gán nhãn sửa.
- Quy ước đáp án hiện có: `data_labels/v6/k1axes/cases.json`, mỗi ca `{file, ca: "<trục>.<mức>", mong_doi: "accept" | "04", base: <id tệp gốc>}`. Bạn sở hữu bộ này — được mở rộng (thêm lý do, cảnh báo), miễn ghi rõ trong thẻ.

## 3. Bản ghi mẫu

```json
{"file": "tinhieu.snr_8dB__v6nat_nguyen_ban_0001.wav", "ca": "tinhieu.snr_8dB",
 "mong_doi": "04", "ly_do": "ồn quá", "base": "v6nat_nguyen_ban_0001"}
```

## 4. Dữ liệu mẫu bạn nhận (ví dụ — không phủ hết phạm vi)

Nhóm trưởng chọn từ dữ liệu sẵn có để bạn hiểu định dạng, tái tạo baseline và bắt đầu nhanh — **gợi ý, không bắt buộc dùng**. Dòng «*thêm*» là dữ liệu khác có sẵn trên máy gốc, xin nếu thấy có ích.

| Gói | Nội dung | Dung lượng |
|---|---|---|
| chung | điểm OOF hệ thống + manifest natvi / naten | 0,12 GB |
| hệ thống | trọng số chạy đầu–cuối (mô hình Hugging Face tải riêng: `python -m v6.product.tai_mo_hinh`) | 0,07 GB |
| khung XLS-R | ghép thử cần (chỉ kiểm có tệp) | 2,0 GB |
| 60 tệp mỗi kiểu | natvi + naten, chọn cố định (đã bỏ bản ghi có cờ) | 0,8 GB |
| ngoài phạm vi | 224 tệp `ngoai_pham_vi_*` (144 tệp đã có trong gói 60 tệp mỗi kiểu) | 0,01 GB |
| kiểm điều kiện | `k1axes` (135 ca + trục v2, sự kiện thật, ngôn ngữ thật) + `suites` | 2,1 GB |
| *thêm:* MIST | bộ ngoài tiếng Việt có bản gốc thật | 0,14 GB |

## 5. Khi tự thêm dữ liệu

Theo `DU_LIEU.md` mục 6. Riêng phần của bạn: mỗi tệp kiểm có đáp án mong đợi ghi sẵn để chấm tự động; tên tệp giữ dạng `<trục>.<mức>__<id gốc>`; kiểm lại đáp án trước khi kết luận K1 sai.

## 6. Gợi ý nguồn (điểm xuất phát — tự tìm thêm)

| Cho | Gợi ý | Lưu ý |
|---|---|---|
| Tiếng ồn | MUSAN đầy đủ (CC BY 4.0), DEMAND, FSD50K, tự ghi quán / đường / xe | kiểm giấy phép từng đoạn, bỏ đoạn NC |
| Phòng vang | OpenSLR 28 RIRS_NOISES (Apache 2.0), BUT ReverbDB (CC BY 4.0), tự ghi phòng thật | — |
| Nhạc / TV nền | phần nhạc của MUSAN, FMA (CC theo từng bài), tự ghi có TV / nhạc | — |
| Sách nói, podcast | LibriVox (phạm vi công cộng), podcast giấy phép CC | không cào ngoài chuỗi `v6/scrape` |
| Ngôn ngữ khác | Common Voice (CC0), FLEURS (CC BY 4.0) | — |
| Nhiều người, nói chồng | AMI, ICSI (CC BY 4.0), trộn có kiểm soát, tự ghi hội thoại | — |
| Hai kênh, định dạng | hai điện thoại ghi cùng cảnh; ffmpeg cho định dạng / tốc độ bit / độ dài | — |

## 7. Việc cần người thật

Khảo sát Req 13 (3–5 người ngoài nhóm, khoảng 1 giờ mỗi người) · tự ghi hai kênh, thì thầm, la hét, hát, TV / nhạc nền, hội thoại nhóm.
