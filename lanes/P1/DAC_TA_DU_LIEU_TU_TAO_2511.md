# Đặc tả dữ liệu P1 tự tạo thay gói P0 (25/11/2026)

**Người đọc:** Huấn (P1). Claude Code soạn, rút từ mã gốc: `v6/coverage/k1_axes*.py`, `v6/natural/build_natural.py`, `v6/natural/req_check.py`, `v6/product/ticket.py`, `lanes/P1/CLAUDE.md`.

**Bối cảnh:** P0 cho P1 tự tạo dữ liệu, miễn theo đúng logic. Gói `du_lieu_P1.tar` phục vụ ba phép đo dưới đây. Mỗi mục ghi: phép đo cần gì, tự tạo được tới đâu, và phần còn thiếu.

## 1. Bộ k1axes (bảng T1.x, «24 hàng Constraint có test chủ động»)

- **Cần:** mỗi ca có tệp gốc sạch, cộng một phép biến đổi lên tệp (mã hoá, đổi tần số lấy mẫu, độ sâu bit, kênh, độ dài, lọc băng, mức, cắt đỉnh, chèn mất tín hiệu, pha loãng tiếng nói, ghép người nói).
- **Đáp án:** tính lại lúc chạy, từ định nghĩa ca và phép đo độc lập (Welch 6–7 kHz, thời lượng tiếng nói). Không đọc đáp án từ tệp.
- **Tự tạo được: toàn bộ.** Mã `k1_axes.py` (P1 sở hữu) chỉ cần đổi **nguồn tệp gốc**. Mã đó đọc `natvi` / `naten` qua `pick()` và `_bases_by_id()`. Tệp gốc cần thoả:
  - một người nói, ≥ 8 s;
  - băng thông ≥ 7 kHz, âm lượng ≥ −35 dBFS, cắt đỉnh < 0,5 %;
  - K1 nhận hoặc cảnh báo;
  - mỗi người một tệp; một nửa vi, một nửa en.
- Nguồn ồn cho ca «chỉ tiếng ồn»: MUSAN noise. Chỉ lấy giấy phép CC BY / BY-SA / CC0 / PD, lọc bằng `dung_bo.musan_pool`.
- Thiết kế cụ thể của lượt dựng: xem tin nhắn duyệt ngày 25/11 (nhóm A).

## 2. Bộ «tệp hợp lệ có sửa» kiểu natvi / naten

Mục đích với P1: đếm K1 chặn oan **theo từng kiểu thao tác** (baseline: 60 tệp / kiểu, vi 3,8 %, en 5,9 %). Bộ này cũng là mẫu số của Req 9.

**Kiểu và nhãn** (`PLAN` trong `build_natural.py`):

| Kiểu | Nhãn | Cách tạo |
|---|---|---|
| `nguyen_ban` | 01 | Bản gốc, không sửa |
| `nen_toan_cuc` / `nen_cuc_bo` | 02 | Khử ồn cả tệp / một đoạn (họ N1–N5, cộng N7 WebRTC NS) |
| `giong_toan_cuc` / `giong_cuc_bo` | 02 | Đổi giọng cả tệp / một đoạn (họ V1–V6) |
| `ghep_thay` | 03 | Thay 1–12 từ bằng đoạn giọng thật của **cùng người** (60 % cùng bản ghi, 40 % bản ghi khác) |
| `ghep_chen` | 03 | Chèn 1–8 từ thật của cùng người vào khoảng giữa hai từ |
| `xoa_loi` | 03 | Cắt 1–12 từ rồi nối lại. Nhãn = vùng 0,5 s quanh mỗi mối nối |

**Ràng buộc khi tạo:**
- đoạn sửa ≤ 5 s và ≤ 50 % tệp, cách hai đầu tệp ≥ 0,5 s;
- mối nối crossfade 0 / 5 / 20 ms;
- mốc từ lấy bằng ASR (`vaudit/asr.py`: wav2vec2 tiếng Việt; tiếng Anh dùng wav2vec2-base-960h). Tiếng Anh chỉ giữ bản gốc có CER ≤ 0,35 so với transcript.

**Lược đồ một bản ghi `manifest.json`** (giữ nguyên để `req_check` / `report_natural` đọc được):

```json
{"id": "p1nat_...", "kind": "ghep_thay", "cfg": "goc", "src": "...", "wav": "<đường dẫn tương đối>",
 "meta": {"host_file": "...", "speaker": "...", "person": "...", "source": "...", "cluster": "...",
          "orig_text": "...", "expected_text": "...", "level1": ["03_ghep_chen"], "level2": ["03_ghep_..."],
          "edits": [{"level2": "...", "start": 0.0, "end": 0.0}], "lang": "vi", "fold": 0, "lockbox": false}}
```

- `fold` = crc32(person) % 5. `cluster` = video (với nguồn YouTube) hoặc người nói.
- Mỗi bộ dùng một tiền tố id riêng. Dựng xong chạy `kiem_bo --moi`, `kiem_bo --tat-ca` và `lockbox check`.

**Những chỗ không làm y như P0 được, cần Huấn quyết:**
1. **Kiểu `giong_*` (đổi giọng):** luật P1 cấm «đổi giọng AI trên giọng cào». Ba hướng:
   - bỏ hai kiểu này;
   - chỉ làm trên corpus có giấy phép cho phép biến đổi (FLEURS CC BY 4.0), và ghi rõ trong thẻ bộ;
   - hoặc dùng biến đổi tín hiệu không phải AI (đổi cao độ / formant bằng DSP). Hướng này cũng phải hỏi, vì luật cấm cả «đổi giọng».
2. **Kiểu `nen_*` (khử ồn):** không phải đổi giọng, nhưng là xử lý AI trên giọng cào. Cần Huấn xác nhận được phép.
3. **Kênh điện thoại** (`v6/phone/channels.py`, 4 kênh chính + 27 cấu hình): máy không có mã này. Bảng chính của Req chỉ tính tệp gốc và kênh ≥ 24 kbps. Đề xuất: lượt đầu chỉ làm `cfg = goc`.
4. **Họ khử ồn N1–N5 và đổi giọng V1–V6** (`v6/tools/backends`): máy không có mã này. Nếu làm, chỉ dùng những gì đang có (DeepFilterNet trong `D:\venv_gpu`, WebRTC NS) và ghi lệch so với P0.

**Cỡ tối thiểu:** 60 tệp × 8 kiểu × 2 ngôn ngữ, nếu bỏ `giong_*` thì 6 kiểu. Mỗi ô ≥ 10 tệp và ≥ 5 người nói. Gốc lấy từ ghi âm thật P1, chỉ fold dev nếu dùng để chọn.

## 3. Req 9 (04 do phân tích ≤ 8 %): dữ liệu tự tạo không đủ

**Định nghĩa** (`req_check.py`, dòng 179):
- Mỗi tệp **không** bị K1 chặn cộng 1 vào mẫu số.
- Tử số đếm tệp có nhãn cuối là 04.
- Tệp ra 04 ở bước phân tích theo luật (5) (`ticket.channel_label`): kênh **không có dòng thao tác nào** **và** có xác suất cả tệp (đã hiệu chỉnh) nằm trong vùng xám **0,4–0,6**.

**Cần:** xác suất của các bộ phát hiện (checkpoint `ft_*`, `tcn_*`, `heads_*` và bộ hiệu chỉnh `calib`). Đó là mô hình của P0 / các lane chuyên gia. Máy hiện **không có** `v6/engine`, checkpoint, hay tệp kết quả `engine_*.json`.

**Hệ quả:** tự tạo đủ dữ liệu ở mục 2 vẫn **không** tính được Req 9.

**P1 làm được, không cần P0:**
- (a) K1 chặn oan trên bộ mục 2. Tệp bị chặn không vào Req 9, nhưng làm đổi mẫu số.
- (b) Test đơn vị cho luật (5) với xác suất giả (đúng 0,4 / 0,6, có dòng / không dòng), đặt trong `test_ticket.py`.
- (c) Ghi Req 9 là «không đo được: thiếu điểm bộ phát hiện» và giữ baseline 1,8 % / 6,2 % làm mốc.

Chỉ có thể đo Req 9 thật khi có **một trong hai**: checkpoint cộng `v6/engine`, hoặc tệp điểm `engine_<tag>.json` của P0 trên bộ dữ liệu cùng id.
