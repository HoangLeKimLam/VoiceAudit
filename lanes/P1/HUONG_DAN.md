# Hướng dẫn đọc và chạy lại việc của P1 (nhánh `lane/p1-bao-cao-2611`)

**Ai đọc:** nhóm trưởng và Claude của nhóm trưởng.
**Mục đích:** biết đọc gì trước, mỗi thư mục dùng làm gì, chạy lại các số trong báo cáo bằng lệnh nào, cần xin dữ liệu gì từ máy P1.

## 1. Đọc theo thứ tự này

1. `lanes/P1/BAO_CAO_2611.md`: tóm tắt, pipeline, vấn đề từ 23/11, việc cần quyết. **Đọc trước nhất.**
2. `lanes/P1/PHAN_TICH_K1_2611.md`: nguyên nhân đã đo của từng hàng K1 còn sai, ứng viên sửa, luật chấm chốt trước (mục 5).
3. `lanes/DE_XUAT/2611_P1_cach-do-on.md`: cách đo đồng thuận cho hàng ồn (Huấn duyệt 26/11, chưa cài).
4. Khi cần thì đọc thêm:
   - `lanes/DE_XUAT/2511_P1_chunk-frames.md`: lỗi `features.py`, chạm lane khác;
   - `lanes/DE_XUAT/2111_P1_chan-oan-hang-cung.md`: cách đếm chặn oan hàng cứng;
   - `lanes/P1/KICH_BAN_TU_GHI_KIEU_NOI.md`;
   - `lanes/P1/DAC_TA_DU_LIEU_TU_TAO_2511.md`.
5. Luật nền vẫn là tệp gốc của nhóm: `lanes/CHUNG.md`, `lanes/BAT_DAU.md`, `lanes/PHU_DU_LIEU_5_LANE.md`, `lanes/P1/CLAUDE.md`.

**Không có trong git** (theo `.gitignore` và luật «log chỉ để trên máy»):
- `data/`, `data_labels/`, `*.jsonl`, `*.wav`, `*.npz`, `*.joblib`, `*.log`;
- nhật ký chi tiết `lanes/P1/NHAT_KY/`.

Số trong báo cáo đều ghi kèm tên tệp nguồn; muốn đối chiếu thì xin gói dữ liệu ở mục 4.

## 2. Bản đồ mã

| thư mục / tệp | làm gì |
|---|---|
| `vaudit/k1.py` | K1 sản phẩm. Nhánh này có hai bản sửa của P1: hợp mặt nạ lời với AST khi sắp chặn «quá ít tiếng nói» (23/11), và hàng logits trống tính là blank (C4, 25/11). Xem `git diff lane/p1 -- vaudit/k1.py` |
| `experts/P1/scrape/` | cào YouTube CC BY (`tai_cat.py`, `cao_tunhien.py`, `tai_v2.py`), kiểm ứng viên Gemini (`kiem_ung_vien_gemini.py`), gỡ tay có lý do (`go_tay.py`, `loai_tay.json`), chọn danh sách nghe mù cho Gemini (`chon_soat_gemini_v2.py`) |
| `experts/P1/build_cond/dung_bo*.py` | dựng bộ điều kiện (giọng sạch + ồn + vang, nhãn từ tham số dựng) |
| `experts/P1/build_cond/do_*.py` | đo: Brouhaha / DeepFilterNet (`do_dieu_kien.py`), vang (`do_vang.py`), GPU (`do_gpu.py`), **đo độc lập để đếm theo cách P1** (`do_doc_lap.py`) |
| `experts/P1/build_cond/dac_trung*.py`, `huan_luyen*.py` | đặc trưng và học bộ ước SNR / nền / RT60 (mlp_F1, S1–S3) |
| `experts/P1/build_cond/cham_*.py`, `so_ba_cach.py`, `de_xuat_bien.py` | chấm bộ ước theo cả hai cách đếm; so ba cách đo ồn |
| `experts/P1/build_cond/chay_k1.py` | chạy K1 trên bộ ghi âm thật, ghi nối tiếp (chạy lại thì bỏ qua tệp đã có) |
| `experts/P1/build_cond/nghe_*.py`, `chon_nghe_*.py` | trang nghe tay (HTML) và chấm kết quả nghe |
| `experts/P1/coverage/k1_axes_p1.py` | bọc `v6/coverage/k1_axes.py` (không sửa tệp gốc) để dựng / chạy T1.x trên 24 tệp gốc thật của P1 |
| `experts/P1/coverage/theo_hang.py`, `so_c4.py` | tóm T1.x theo hàng Constraint; so trước / sau một bản sửa |
| `experts/P1/coverage/bo_sung_hang.py` | bộ kiểm bổ sung cho Ngôn ngữ / Nói chồng / Loại âm thanh |
| `experts/P1/coverage/chan_doan_*.py` | chẩn đoán hàng Số người nói và Dải tần, chỉ đọc |
| `experts/P1/coverage/test_so_tep.py` | bài kiểm hàng «Số tệp: đúng 1», không cần dữ liệu |

Mọi script có `selftest` khi có hàm thuần cần kiểm. Script đã có tệp kết quả thì **không ghi đè**: phải chuyển tệp cũ đi trước, không xoá.

## 3. Môi trường

| môi trường | dùng cho | ghi chú |
|---|---|---|
| `.venv` (của nhóm) | K1, chấm, dựng bộ, T1.x | numpy 2; **không cài pyannote vào đây** |
| `D:\venv_brouhaha` | Brouhaha, DeepFilterNet, đo độc lập | Python 3.11, torch 2.5.1 cpu, numpy < 2, pyannote.audio 3.3.0, `huggingface_hub < 0.26`. Chạy với `-I` |
| `D:\venv_gpu` | đặc trưng w2v / Brouhaha trên GPU | torch 2.7.1+cu128. **Bắt buộc tắt TF32**; bật thì lệch CPU tới 0,57 dB |

**Biến môi trường:** `HF_HOME=F:/hf_cache`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONIOENCODING=utf-8`. Thêm `ffmpeg` vào `PATH`.

**Mô hình ngoài:**
- whisper-small (ngôn ngữ) và AST AudioSet (loại âm thanh / kiểu nói / bù mặt nạ lời) nằm trong `HF_HOME`.
- pyannote segmentation 3.0, bản onnx của sherpa-onnx (nói chồng), nằm ở `.tmp_research/models/sherpa-onnx-pyannote-segmentation-3-0/`.
- `pyannote/brouhaha`: kho có cổng; người dùng tự đồng ý điều khoản trên Hugging Face.

## 4. Dữ liệu cần xin từ máy P1 để chạy lại

| bộ (`data_labels/v6/…`) | cỡ | dùng để chạy lại |
|---|---|---|
| `p1_tunhien_v1`, `p1_tunhien_v2` (kèm `_kiem/`) | khoảng 1 900 đoạn | chặn oan K1, đo độc lập, Gemini soát |
| `k1axes/p1_v1` | 3 060 tệp | T1.x theo hàng |
| `p1_k1bosung_v1` | 168 tệp, 93 MB | ba hàng bổ sung |
| `p1_dieukien_v3`, `v4`, `v4b` (kèm `_kiem/`) | | bộ ước SNR, so ba cách đo |
| `data/p1_raw` (FLEURS, MUSAN, DEMAND, RIR), `data/p1_scrape/goc` | 33,5 GB / khoảng 62 GB | chỉ cần khi dựng lại bộ |

Ngược lại, P1 cần nhóm trưởng gửi: `du_lieu_P1.tar` (natvi / naten, checkpoint bộ phát hiện), `vaudit/cond_models.joblib`, `vaudit/test_k1.py`.

## 5. Lệnh chạy lại các số trong báo cáo

Chạy từ gốc repo, với biến môi trường ở mục 3.

```bash
# K1 trên tệp thật (chỉ dev fold 0–2; ghi _kiem/k1_ket_qua_<tag>.jsonl và KET_QUA_K1_<tag>.md)
.venv/Scripts/python.exe experts/P1/build_cond/chay_k1.py run --bo p1_tunhien_v2 --tag 2611 --dev
.venv/Scripts/python.exe experts/P1/build_cond/chay_k1.py run --bo p1_tunhien_v1 --tag 2611

# Đo độc lập cho tệp K1 chặn (cách đếm P1; luật trong BAO_CAO_2611.md mục 4)
D:/venv_brouhaha/Scripts/python.exe -I experts/P1/build_cond/do_doc_lap.py selftest
D:/venv_brouhaha/Scripts/python.exe -I experts/P1/build_cond/do_doc_lap.py run p1_tunhien_v2   # kèm 120 tệp T1.x kiểm phương pháp
D:/venv_brouhaha/Scripts/python.exe -I experts/P1/build_cond/do_doc_lap.py run p1_tunhien_v1

# T1.x: bảng theo hàng Constraint, và so trước / sau bản sửa C4
.venv/Scripts/python.exe experts/P1/coverage/theo_hang.py p1_v1_c4
.venv/Scripts/python.exe experts/P1/coverage/so_c4.py p1_v1 p1_v1_c4

# Bộ kiểm bổ sung (chế độ mở rộng của K1)
.venv/Scripts/python.exe experts/P1/coverage/bo_sung_hang.py run
.venv/Scripts/python.exe experts/P1/coverage/bo_sung_hang.py tom

# Chẩn đoán Số người nói / Dải tần
.venv/Scripts/python.exe experts/P1/coverage/chan_doan_nguoi.py run
.venv/Scripts/python.exe experts/P1/coverage/chan_doan_nguoi.py tom
.venv/Scripts/python.exe experts/P1/coverage/chan_doan_bangtan.py

# Hàng ồn: so ba cách đo trên cùng thước
.venv/Scripts/python.exe experts/P1/build_cond/so_ba_cach.py run

# Bài kiểm không cần dữ liệu
.venv/Scripts/python.exe experts/P1/coverage/test_so_tep.py
```

Thời gian trên CPU, máy chỉ chạy một việc:
- K1 khoảng 6,6 s mỗi tệp; v1 đủ bộ khoảng 2,5 giờ;
- đo độc lập 141 tệp khoảng 10 phút.

## 6. Luật cần giữ khi chạy hoặc sửa

- **Test chỉ mở một lần:** fold 3–4 của mỗi bộ, theo `crc32(base) % 5`. Mọi lần chọn cấu hình dùng dev (fold 0–2). Đã mở rồi thì lần sửa sau dùng bộ test mới.
- **Chốt luật chấm trước khi xem số.** Báo `k/n` kèm Wilson 95 %, và báo **cả hai cách đếm** chặn oan (nhóm và P1).
- **Hàng cứng:** không đổi ngưỡng, chỉ sửa cách đo hoặc cách đếm.
- **Không xoá dữ liệu:** muốn gỡ thì chuyển vào `_quarantine/` và ghi lý do vào `_quarantine/LY_DO.jsonl`. Không gán nhãn 04 bằng tay cho ghi âm thật.
- **Đổi K1** thì kèm: bảng T1.x trước / sau, `python -m v6.product.test_ticket`, `selftest`, `test_pipeline_modes` (xem `lanes/P1/CLAUDE.md` mục 7).
- **Gemini** chỉ nghe sơ bộ: phiên mới, nghe mù, có ca đối chứng. Kết quả thành nhãn nghe tay khi Huấn xác nhận.
