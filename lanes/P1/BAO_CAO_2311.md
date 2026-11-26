# P1 · Báo cáo 23/11 và việc xin nhóm trưởng

**Người gửi:** Huấn (P1), Claude Code soạn. **Người đọc:** Claude Code của nhóm trưởng (P0).
**Nhánh:** `lane/p1`. Chưa commit. `git diff` hiện chỉ có `vaudit/k1.py` (+30 dòng).
**Chi tiết:** `lanes/P1/NHAT_KY/NHAT_KY_P1.md` mục 2.14.

## Tóm tắt

Có 2 việc.
1. P1 sửa **cách đo thời lượng lời nói** của K1 (dòng «quá ít tiếng nói»). Không đổi ngưỡng nào.
2. P1 xin **5 thứ** để chạy đủ bài kiểm tra bắt buộc (`lanes/P1/CLAUDE.md` dòng 61) và bật các dòng Constraint còn thiếu. Xem mục 3.

## 1. Bản sửa K1

**Giả thuyết.** K1 đo lời nói bằng khung CTC của ASR tiếng Việt (`nguyenvulebinh/wav2vec2-base-vietnamese-250h`), dùng cho cả hai ngôn ngữ. Trên ghi âm thật có vang, có ồn, hoặc nói tiếng Anh, ASR không ra ký tự, nên K1 đếm thiếu giây lời nói và chặn oan.

**Đã làm.** Trong `vaudit/k1.py`:
- thêm hàm `ast_speech_frames`;
- thêm một nhánh trong `analyse_channel`.

Nhánh mới chỉ chạy khi số giây đo theo CTC sắp làm tệp bị chặn (`< 2 s` hoặc `< 30 %`). Khi đó K1 lấy hợp:

`ctc ∪ (năng lượng ∩ AST Speech > 0,3)`

- **AST Speech:** xác suất lớn nhất của nhóm lớp Speech (`k1x.SPEECH_LABELS`), tính trên mỗi cửa sổ 2 s.
- **năng lượng:** mặt nạ `k1.speech_mask`.

Hệ quả:
- Phép hợp chỉ làm **tăng** số giây lời nói. Tệp đang được nhận vẫn được nhận, và không chạy chậm thêm.
- Nếu AST lỗi, K1 giữ số đo CTC (tức vẫn chặn) và ghi lỗi vào `measures.kenh[i].tieng_noi_ast_loi`.
- Ở chế độ nhẹ `_light()` của bài kiểm tra, AST không được nạp.

Trường mới trong `measures.kenh[i]`:
- `tieng_noi_do`: cách đo đã dùng;
- `tieng_noi_ast_loi`: chỉ có khi AST lỗi.

Không đổi lược đồ hồ sơ H5 / #bind.

**Cách chọn, tránh «học thuộc đề».**
- Bộ đánh giá: `data_labels/v6/p1_tunhien_v1`, gồm ghi âm YouTube CC BY từ kênh không chuyên. Đáp án là accept, nhãn «01 có khả năng».
- Chia theo video: `crc32(base) % 5`. Fold 0–2 dùng để chọn, fold 3–4 dùng để chấm.
- Có 9 ứng viên: năng lượng, webrtcvad, AST ở các ngưỡng 0,3 / 0,5 / 0,7, và các tổ hợp. Cả 9 được so trên dev cùng với nhãn nghe và 34 ca tự dựng.
- Huấn duyệt cách đo **trước khi** mở fold test.
- Mã chọn: `experts/P1/build_cond/do_tieng_noi.py` (`alt`, `test`, `selftest`).

**Số trước / sau.** Chặn oan trên fold test, Wilson 95 %. Số này tính bằng mô phỏng trên kết quả K1 đã lưu. Chạy lại `k1.check` đã sửa cho kết quả khớp đúng (xem mục 2).

| | K1 cũ | K1 sửa | K1 sửa, trừ ca vi phạm hàng cứng thật* |
|---|---|---|---|
| en | 7/220 = 3,2 % | 5/220 = 2,3 % | 1/216 = 0,5 % (0,1–2,6) |
| vi | 30/491 = 6,1 % | 24/491 = 4,9 % | 16/483 = 3,3 % (2,0–5,3) |
| chung | 37/711 = 5,2 % | 29/711 = 4,1 % | 17/699 = 2,4 % (1,5–3,9) |

\* Ca vi phạm hàng cứng thật là ca «quá nhỏ», «băng hẹp» hoặc «cắt đỉnh nặng» mà đo độc lập xác nhận. Cách đếm này đang chờ P0 duyệt ở `lanes/DE_XUAT/2111_P1_chan-oan-hang-cung.md`.

**Chỗ còn yếu.**
- vi chưa đạt mục tiêu ≤ 2 %.
- Trên fold test có 56 tệp Gemini nghe sơ bộ. Trong đó:
  - 11 ca vẫn bị chặn oan, cả 11 đều vì «quá ít tiếng nói», phần lớn là đoạn ngắn 5–13 s;
  - không có ca nào đáng chặn mà bị thả.
- Trên dev có 1 ca bị thả sai sát ngưỡng: `K-bsHVojEYg_3`. Gemini nghe được 29 % lời nói, đoạn có vỗ tay; cách mới đếm 42 %.
- Mức «chung» vẫn nằm trên mục tiêu.

## 2. Bài kiểm tra hồi quy đã chạy

| Lệnh | Kết quả |
|---|---|
| `python -m v6.product.test_pipeline_modes` | 15/15 đạt |
| `python -m v6.product.ghep_thu selftest` | đạt |
| `python -m v6.product.test_ticket` | 38/39. Bài hỏng là `test_schema_cli_and_scan_on_v9_record`, **có sẵn từ trước**: đã `git stash` bản sửa và chạy lại, vẫn hỏng y hệt. Xem mục 3, ý 4. |
| `python -m vaudit.k1 test` | không chạy được: `ImportError: cannot import name 'test_k1'`. Tệp `vaudit/test_k1.py` không có trong bộ gửi. |
| `python -m v6.coverage.k1_axes run` | chưa chạy: máy không có `data_labels/v6/k1axes/v2`. |
| Req 9 trên natvi / naten | chưa chạy: máy không có natvi / naten. |
| Chạy lại `k1.check` đã sửa trên mọi tệp đang bị chặn vì «quá ít tiếng nói» (`p1_tunhien_v1`, `p1_dieukien_v1`) | **Khớp mô phỏng.** Trên `p1_tunhien_v1` có 38 tệp accept bị chặn vì «quá ít tiếng nói», trong đó 13 tệp được thả (en 3, vi 10), đúng như mô phỏng (62 → 49 trên cả bộ). Trên `p1_dieukien_v1`, cả 12 ca «loại âm thanh» có đáp án 04 **vẫn bị chặn**. AST lỗi: 0 lần. |

## 3. Xin nhóm trưởng

1. **`vaudit/cond_models.joblib` và checkpoint TCN.** Máy P1 chưa có, nên K1 / pipeline ghi `not_impl` cho 3 dòng Constraint:
   - SNR ≥ 10 dB;
   - âm không phải lời nói ≤ 10 %;
   - RT60 ≤ 1,0 s.

   Con số chặn oan ở mục 1 **chưa gồm** 3 dòng này. Khi bật, số có thể tăng, và P1 sẽ đo lại.
2. **`model.onnx` của `sherpa-onnx-pyannote-segmentation-3-0`.** Thư mục `.tmp_research/models/sherpa-onnx-pyannote-segmentation-3-0/` trên máy chỉ có script, không có `model.onnx`. Thiếu tệp này thì không chạy được `extended=True`, nên 2 dòng «Ngôn ngữ» và «Nói chồng ≤ 10 %» chưa kiểm.
3. **Link Drive `du_lieu_P1.tar`** (`TONG_HOP_P1.md` mục 6.1). Cần cho `k1axes` (135 ca + trục v2) và natvi / naten, để chạy bảng T1.x trước / sau và Req 9 theo `lanes/P1/CLAUDE.md` dòng 61.
4. **`vaudit/test_k1.py`**: `k1.py` gọi đến tệp này (`python -m vaudit.k1 test`) nhưng nó không có trong bộ gửi. Ngoài ra, `v6/product/test_ticket.py` dòng 511 ghi cứng `tempfile.TemporaryDirectory(dir="D:\\tmp_va")`, nên bài đó hỏng trên máy không có thư mục này. Nhóm muốn tạo thư mục đó, hay sửa để dùng thư mục tạm mặc định?
5. **P0 duyệt** `lanes/DE_XUAT/2111_P1_chan-oan-hang-cung.md`. Nội dung: ghi âm thật mà đo độc lập xác nhận vi phạm «quá nhỏ», «băng hẹp» hoặc «cắt đỉnh nặng» thì tính là chặn đúng, báo riêng, không đổi ngưỡng. Báo cáo của P1 luôn ghi cả hai cách đếm.

## 4. Bước tiếp của P1

- Huấn duyệt rồi mới commit `vaudit/k1.py`.
- Phần vi còn lại (đoạn ngắn bị đếm thiếu lời nói): tìm nguyên nhân **chỉ trên dev**. Fold test đã mở một lần, nên muốn có số chấm khách quan cho bản sửa tiếp theo thì cần bộ test mới (cào thêm hoặc dùng tập của P0).
- Nhận được mục 3, ý 1–3 thì chạy đủ bảng T1.x, Req 9, và đo lại chặn oan với SNR / nền / vang.
