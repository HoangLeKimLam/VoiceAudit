# Dữ liệu VoiceAudit — phạm vi, nhãn, cách thêm để ghép không xung đột

> Cho mọi thành viên và nhóm trưởng. Cập nhật 20/10/2026 (sau quyết định 18). Đọc sau `lanes/BAT_DAU.md`; phần riêng của từng người: `lanes/<Px>/DU_LIEU.md`.
>
> **Chỉ ba thứ cố định**, để dữ liệu của mọi người ghép được và không lẫn nhãn:
> 1. **Phạm vi phải phủ** — các trường hợp lấy từ bản đề xuất đã nộp (mục 3).
> 2. **Quy ước nhãn và định dạng** — một bảng nhãn chung, ranh giới giữa các lane, mã trong manifest (mục 2, 6).
> 3. **Kiểm trước khi nộp** — `python -m v6.data.kiem_bo <manifest> --moi` phải «ĐẠT», và kiểm tập khoá (mục 6.5).
>
> **Mọi thứ khác là tự do:** tìm nguồn nào, cào / ghi / sinh bằng công cụ gì, bao nhiêu, theo thứ tự nào. Dữ liệu trên máy gốc và các gói gửi đi **chỉ là ví dụ** để hiểu định dạng và tái tạo baseline — không bắt buộc dùng, và **không phủ hết** phạm vi.
>
> **Việc của mỗi người là tự thu thập** dữ liệu đúng dạng (mục 2) để phủ đủ phạm vi của mảng mình (mục 3 và `lanes/<Px>/DU_LIEU.md`).

---

## 1. Bốn loại dữ liệu

| Loại | Là gì | Dùng để |
|---|---|---|
| **Tệp tự tạo** | Lấy ghi âm thật rồi tự sửa, nên biết chắc loại và vị trí sửa (`natvi`, `naten`, `chain`, `cover`…) | học, chọn ngưỡng trên OOF, đo phát hiện |
| **Ghi âm thật ngoài đời** | Cào từ nguồn mở, tự ghi, tệp dài, tệp lành tính | đo **vu oan** ngoài đời. Bản cào chỉ là «01 có khả năng»; bản **tự ghi** có kiểm soát mới là 01 chắc chắn |
| **Đặc trưng đã trích** | Vector XLS-R mỗi 20 ms, vector cả tệp và vài họ đặc trưng khác, tính sẵn cho các tệp tự tạo | thử nhanh mà không chạy lại mô hình nền — không bắt buộc dùng |
| **Kho gốc để sinh thêm** | Bản gốc của các kho nghiên cứu, giọng tham chiếu, tiếng ồn | sinh dữ liệu sửa mới |

---

## 2. Nhãn — một bảng chung cho cả nhóm

### 2.1 Bốn nhãn và ai giữ

| Nhãn | Gồm | Không gồm (→ nhãn nào) | Lane |
|---|---|---|---|
| **01** Không chỉnh sửa | bản ghi bình thường, kể cả: nén khi ghi, gửi lại qua ứng dụng (nhiều lần cũng được), tinh chỉnh mặc định của thiết bị, đổi định dạng / tần số lấy mẫu, cắt đầu–cuối, đổi âm lượng tổng, đổi tốc độ ± 5 %, **khử ồn đơn sơ có sẵn của ứng dụng ghi / gọi** | khử ồn chuyên nghiệp (→ 02); mọi thao tác đổi chuỗi lời (→ 03) | P2 |
| **02** Chỉnh sửa hậu kỳ | lời có sẵn, không thêm bớt, chỉ đổi cách nghe: **khử ồn chuyên nghiệp**; **đổi giọng** (dịch cao độ ≥ 2 nửa cung, hoặc chuyển giọng người khác — kể cả chuyển giọng bằng AI) | khử ồn đơn sơ của ứng dụng (→ 01); dịch < 2 nửa cung (không phải thao tác); giọng do máy tạo từ văn bản (→ 03) | P3 |
| **03** Ghép hoặc chèn đoạn | chuỗi lời bị đổi: **ghép lời thật** (chèn / thay bằng lời người thật lấy từ chỗ khác), **xoá lời**, **giọng máy** (đoạn chèn / thay do máy tạo, hoặc cả tệp là giọng máy) | — | P4 (lời thật, xoá) · P5 (giọng máy) |
| **04** Không đủ cơ sở | hệ thống tự ra khi tệp vi phạm điều kiện hoặc kết quả lưng chừng — **không gán tay** khi làm dữ liệu | — | P1 · P0 |

Nhiều thao tác trong một tệp → mỗi thao tác một dòng `edits`; nhãn tệp = nặng nhất (03 > 02).

### 2.2 Ranh giới hay nhầm — phải gán đúng như sau

| Tình huống | Nhãn | Lane |
|---|---|---|
| Khử ồn **đơn sơ** có sẵn khi ghi hoặc gọi, người dùng không chỉnh: WebRTC NS **mức 0–2**, Zoom / Teams «Tự động / Thấp / Vừa», ghi âm điện thoại thường, tin nhắn thoại | **01** — không ghi là thao tác | P2 |
| Khử ồn **chuyên nghiệp hoặc mạnh**, lúc ghi hay sau khi ghi: WebRTC NS **mức 3**, Zoom / Teams «Cao», iPhone Voice Isolation / Enhance Recording, khử ồn AI, công cụ khử ồn (noisereduce, afftdn, RNNoise dạng bộ lọc, Audacity…) | **02** khử ồn (cả tệp hay một đoạn theo vùng áp) | P3 |
| Chuyển giọng (VC): người thật nói, đổi sang giọng khác | **02** đổi giọng — không phải 03 | P3 |
| Giọng do máy tạo (TTS, nhân bản giọng từ văn bản) chèn vào hoặc cả tệp | **03** giọng máy | P5 |
| Đoạn chèn là **người thật** nói (cùng hoặc khác bản ghi) | **03** ghép lời thật — không phải giọng máy (Req 6) | P4 |
| Cắt bỏ lời rồi nối lại | **03** xoá lời | P4 |
| Phát lại qua loa rồi ghi; cố ý phá dấu vết | ngoài phạm vi — đánh dấu `meta.ngoai_pham_vi`, không gán nhãn sửa | P1 |
| Tệp vi phạm điều kiện (quá ngắn, quá ồn…) | không gán nhãn sửa — hệ thống tự ra 04 | P1 |

Thang khử ồn ở trên là **thang tạm** (quyết định 18): các hãng không công bố mức dB; số chốt sẽ đo trên ghi âm thật của từng ứng dụng (P2, P3).

### 2.3 Mã trong manifest (hợp đồng H2)

| Thao tác | `level2` của dòng `edits` | Trường bắt buộc của dòng | `level1` của tệp |
|---|---|---|---|
| (không có thao tác) | `level2 = []`, `edits = []` | — | `["01_nguyen_ban"]` |
| Khử ồn cả tệp / một đoạn | `02_nen_toan_cuc` / `02_nen_cuc_bo` | `kind` (`global` / `segment`), `start`, `end` (đoạn), công cụ (`tool`); WebRTC NS ghi thêm `ns_level` | có `"02_ai_hau_ky"` |
| Đổi giọng cả tệp / một đoạn | `02_giong_toan_cuc` / `02_giong_cuc_bo` | như trên; dịch cao độ ghi `f0_shift_st`; chuyển giọng AI ghi `op = "doi_giong_ai"` + công cụ | có `"02_ai_hau_ky"` |
| Ghép lời thật (thay / chèn) | `03_ghep_giong_that` | `kind = segment`, `start`, `end`, `op` (`ghep_thay` / `ghep_chen`), `sub` (`thay_that` / `chen_that`), `donor_file` (nguồn lời) | có `"03_ghep_chen"` |
| Xoá lời | `03_ghep_giong_that` (**không** dùng `03_xoa_loi`) | `kind = segment`, `start`, `end`, `op = "xoa_loi"`, **`sub = "cat_bo"`**, `joint` (mốc nối) | có `"03_ghep_chen"` |
| Giọng máy một đoạn / cả tệp | `03_chen_giong_may` | đoạn: `kind = segment`, `start`, `end`, `op` (`may_thay` / `may_chen`), `sub` (`thay_the` / `chen_them`); cả tệp: `kind = global`, `op = "may_ca_file"`; luôn ghi công cụ sinh (`tts` / `engine`) và `role` | có `"03_ghep_chen"` |

Mỗi tệp: `level2` của tệp = tập `level2` của các dòng `edits`. Bộ kiểm `kiem_bo` kiểm đúng các điều này.

### 2.4 Bản ghi mẫu

```json
{"id": "p2x_tuthu_0001", "kind": "nguyen_ban", "cfg": "goc", "src": "p2x_tuthu_0001",
 "wav": "../v6/p2x_tuthu/nguyen_ban/p2x_tuthu_0001.wav",
 "meta": {"lang": "vi", "source": "tuthu", "person": "tuthu:nguoi03", "speaker": "tuthu:nguoi03",
          "level1": ["01_nguyen_ban"], "level2": [], "edits": [],
          "thiet_bi": "điện thoại Android", "ung_dung": "tin nhắn thoại Zalo"}}
```

```json
{"id": "p4x_xoa_0001", "kind": "xoa_loi", "cfg": "goc", "src": "p4x_xoa_0001",
 "wav": "../v6/p4x_suatay/xoa_loi/p4x_xoa_0001.wav",
 "meta": {"lang": "vi", "source": "vox", "person": "vox:id00092", "speaker": "vox:id00092",
          "level1": ["03_ghep_chen"], "level2": ["03_ghep_giong_that"],
          "edits": [{"level2": "03_ghep_giong_that", "kind": "segment", "op": "xoa_loi", "sub": "cat_bo",
                     "start": 6.59, "end": 7.09, "joint": 6.84, "deleted": "là", "tool": "Audacity", "nguoi_sua": "ngoai_nhom_02"}]}}
```

Mẫu riêng cho nhãn của từng người: `lanes/<Px>/DU_LIEU.md` mục 3.

---

## 3. Phạm vi phải phủ — và hiện đang có gì

Bảng là **bản đồ trường hợp phải phủ** (lấy từ Input, Constraint, Scope, Assumption và «Điều kiện chấm» của bản đề xuất đã nộp). Cột «Còn thiếu» chỉ là hiện trạng; cách lấp là tự do.

| # | Trục | Bản nộp cần | Hiện có | Còn thiếu | Lane |
|---|---|---|---|---|---|
| 1 | Ngôn ngữ | Việt, Anh; báo riêng | vi 35 h, en 23 h tự tạo | — | — |
| 2 | Người nói | người chấm ≠ người học; đa dạng | vi 128 nhóm (VSS chiếm 37 %, có 17 % vu oan lịch sử); ghi âm thật vi 11 nhóm | nhiều nhóm ghi âm thật tiếng Việt hơn; dựng tập khoá tiếng Anh từ người VoxPopuli chưa dùng | P2 · P0 |
| 3 | Thiết bị ghi | điện thoại, máy ghi âm, phần mềm họp | kho nghiên cứu, YouTube, FOSDEM | ghi âm có kiểm soát trên thiết bị thật | P0 · P2 |
| 4 | Truyền tải | gửi lại qua ứng dụng; cuộc gọi | 4 kênh **giả lập** (Zalo tin nhắn, WhatsApp / Telegram, gọi ghi lại, VoLTE) | truyền qua **ứng dụng thật** | P2 |
| 5 | Nén | 8 cấu hình: MP3, AAC, Opus, AMR-WB, chuyển tiếp nhiều lần | một phần trong k1axes và kênh giả lập | đủ 8 cấu hình × chuyển tiếp nhiều lần | P2 · P1 |
| 6 | Độ dài tệp | 3 giây – 10 phút; báo riêng ≤ 20 s / dài hơn | tự tạo gần như toàn tệp ngắn; tệp dài 405 | tệp có sửa dài hơn 20 s | P0 · P3 · P4 |
| 7 | Số người nói | 1 / 2–4; nói chồng | AMI, EdAcc, Earnings-22 (en) | hội thoại nhiều người tiếng Việt | P2 · P1 |
| 8 | Hai kênh | 4 trường hợp (kênh im, giống hệt, hai micro một cảnh, hai nguồn riêng) | chưa có bộ riêng | bộ tệp hai kênh | P1 |
| 9 | Điều kiện ghi | ồn, nhạc / TV nền, vang, cắt đỉnh, mất tín hiệu, băng hẹp, quá nhỏ, không phải lời nói, kiểu nói lạ | k1axes (tự sinh), tiếng ồn mẫu 57 MB | ồn và phòng thật đa dạng; hát, thì thầm, la hét; nhạc / TV nền thật | P1 |
| 10 | Khử ồn (02) | ≥ 3 bộ, ≥ 1 bộ chưa thấy khi học | 5 bộ (afftdn, noisereduce ×2, RNNoise ×2) + WebRTC NS mức 3 | khử ồn **AI**; công cụ phổ thông; số đo khử ồn của ứng dụng thật để chốt thang | P3 |
| 11 | Đổi giọng (02) | ≥ 2 bộ, ≥ 1 chưa thấy | 6 bộ dịch cao độ + chuyển giọng VSASV (chỉ vi) | chuyển giọng **AI** (nhất là tiếng Anh); ứng dụng đổi giọng phổ thông | P3 |
| 12 | Ghép / xoá lời thật (03) | 1 tiếng – 1 câu, mỗi đoạn ≤ 5 s | máy sinh nhiều mức, đa thao tác | **sửa tay thật ≥ 150 ca**; sửa trong vang / ồn; ghép khác thiết bị | P4 |
| 13 | Giọng máy (03) | ≥ 2 bộ, ≥ 1 chưa thấy; có thể nhái rất giống | 240 tệp vi hợp lệ (1 công cụ; chấm OOF hiện chỉ 38 tệp có đặc trưng); **tiếng Anh: 0** | nhiều công cụ, nhiều người, vi + en | P5 |
| 14 | Ghi âm thật | một phần tập chấm là ghi âm thật (vd. MIST) | cào vi ~6 h ngắn + ~5 h dài / 11 nhóm, en ~30 h ngắn + ~28 h dài / 41 nhóm; MIST | nhiều nhóm tiếng Việt hơn | P2 |
| 15 | Ngoài phạm vi | phát lại qua loa, phá dấu vết, băng hẹp | 224 tệp `ngoai_pham_vi_*` | đủ để kiểm cảnh báo | P1 |
| 16 | Tập khoá | người nói chưa dùng, công cụ chưa thấy | luật v2 (mục 6.5); 4 công cụ để dành | bộ tiếng Anh người mới (VoxPopuli) | P0 |

---

## 4. Kho hiện có trên máy gốc (ví dụ)

| Bộ (`data_labels/v6/<bộ>`) | Là gì | Số tệp | Ngôn ngữ · nguồn | Dung lượng |
|---|---|---|---|---|
| `natvi` · `naten` | tự tạo, đủ các thao tác + mức thường / kỹ / vụng / đa thao tác (naten chưa có giọng máy) | 13 864 · 9 064 | vi: VSS, celeb, vox, VSASV · en: VoxPopuli, EdAcc, AMI | 4,0 · 2,5 GB |
| `scrapevi` · `scrapeen` | ghi âm thật cào + bản tự sửa | 1 477 · 1 750 | YouTube · FOSDEM, YouTube, CCC | 0,7 · 0,6 GB |
| `longvi` · `longen` | tệp 1–10 phút | 200 · 205 | YouTube · FOSDEM, YouTube | 1,1 · 1,4 GB |
| `benign` · `amilong` · `chain` · `cover` · `cover2` | lành tính · họp dài · chuỗi truyền tải · phủ điều kiện | 540 · 110 · 2 009 · 2 644 · 499 | vi / en | 0,2 · 0,2 · 0,6 · 0,9 · 0,1 GB |
| `aug` | tăng cường (chỉ có đặc trưng) | 8 318 | vi / en | 8 MB |
| `k1axes`, `suites`, `mist_vi`, `mist_pair` | kiểm điều kiện · bộ kiểm nhỏ · bộ ngoài MIST | — | — | 1,9 · 0,2 · 0,14 GB |
| `ttsmulti_vi` · `ttsmulti_en` · `vcmulti` | **kế hoạch** sinh giọng máy / chuyển giọng nhiều công cụ + giọng tham chiếu — chưa sinh | — | — | 0,1 · 0,3 · 1,2 GB |

Đặc trưng đã trích (`data_labels/realistic_v4/`): `xlsr_frames` 2,0 GB, `xlsr_pooled` 0,4 GB, `spec_frames` 2,5 GB, `hand_frames` 0,6 GB, `ssl6_cache` 3,8 GB, `ssl_cache` 2,7 GB, `deepfense_cache` 3,1 GB — có cho tệp tự tạo, **chưa có** cho ghi âm cào và tệp dài.
Kho gốc: `data/natural_v6` 18 GB, `data/scrape` 22 GB, `data/real_pilot` 0,5 GB, `data/ami` 0,9 GB, `data/librispeech` 0,7 GB (sách nói — ngoài phạm vi), `data/musan_noise` 57 MB, `models/viXTTS` 1,8 GB.
Định dạng âm thanh của các bộ: **wav 16 kHz, một kênh, 16 bit**.

**Cờ trong manifest (quyết định 18)** — bản ghi có cờ không dùng để học hay chấm, bộ nạp chung (`v6.boundary.detect.load_items`) tự bỏ qua:
- `meta.chinh_sach` — 554 tệp giọng máy nhân bản người không được phép nhái (430 natvi + 124 bản tăng cường).
- `meta.tam_loai` — 618 đoạn WebRTC NS không ghi mức + 6 bản ghi trùng id ở `chain`.
- `meta.edits_khong_tinh` — 1 077 dòng WebRTC NS mức ≤ 2 đã chuyển khỏi `edits` (971 tệp thành 01).

---

## 5. Chia cho từng người (dữ liệu mẫu — nhóm trưởng gợi ý)

Dung lượng dưới đây chỉ để nhóm trưởng chép gửi. Dữ liệu bạn tự thu nhiều hay ít là việc của bạn. Cột «Gói thêm» là dữ liệu sẵn có khác — gợi ý, xin nếu thấy có ích.

P0 giữ toàn bộ trên máy gốc. P1–P5 nhận **bộ gửi cơ bản** (tài liệu + mã cần thiết + dữ liệu mẫu), gói thêm khi cần — đã loại tập khoá và mọi bản ghi có cờ. Mọi bộ có chung: tài liệu, mã chung (đo / kiểm / ghép thử), gói chung (OOF hệ thống, manifest natvi / naten, 2 tệp v4 ghép thử đọc) và khung XLS-R (ghép thử cần). Dung lượng đo thật (20/10).

| Lane | Thêm vào phần chung | Bộ cơ bản | Gói thêm |
|---|---|---|---|
| **P1** | mã sản phẩm · công cụ đo K1 · trọng số hệ thống · 60 tệp mỗi kiểu · ngoài phạm vi · k1axes + suites | **5,1 GB** | MIST 0,14 |
| **P2** | mã sản phẩm · chuỗi cào + trạng thái (đã lọc tập khoá) · trọng số hệ thống · vector cả tệp · ghi âm thật ngoài đời (scrape, long, benign, bản cắt, real_pilot, kho phân vị) | **13,2 GB** | 60 tệp mỗi kiểu 0,8 |
| **P3** | vector XLS-R · âm thanh khử ồn, đổi giọng, tệp thật, benign | **6,3 GB** | giọng tham chiếu 0,17 · kho gốc sinh 1,75 · bộ phủ 1,5 · đặc trưng họ khác 12,4 |
| **P4** | vector XLS-R · âm thanh ghép, xoá, đa thao tác, tệp thật, amilong · MIST | **6,6 GB** | kho gốc sinh 1,75 · bộ phủ 1,5 · đặc trưng họ khác 12,4 |
| **P5** | hạ tầng sinh · vector XLS-R · âm thanh giọng máy hợp lệ + ghép thật vi / en (Req 6) + tệp thật · giọng tham chiếu · kho gốc sinh (đã ghi sẵn chỉ số dòng `grow` / `session`) | **6,5 GB** | viXTTS 1,75 · đặc trưng họ khác 12,4 |

Đóng gói (nhóm trưởng): `python -m v6.data.goi_lane plan` để xem trước; `bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.data.goi_lane copy --lane P3 --dest E:\goi_P3 [--them tham_chieu_sinh]`. Thư mục đích nằm ngoài kho gốc. Bộ gửi gồm cả tài liệu và mã — thành viên đặt thư mục đó ở đâu cũng được (nên là đường ngắn).

---

## 6. Thêm dữ liệu mới — cách làm để ghép không xung đột

### 6.1 Tự cào
Đi **đúng chuỗi cào của hệ thống** (`v6/scrape`, P2 giữ mã) để mọi bản cào cùng giấy phép, cùng lọc, cùng cách gom người nói và cùng luật tập khoá: `bash .tmp_v6/cache/scrape_queue.sh <bước>` theo thứ tự `discover → fetch → probe_filter → segment → align → lid → spk → hosts_qc → build_sets` (bước `search` chỉ gợi ý kênh, không tự thêm). Kênh mới phải được nhóm trưởng duyệt. Lane khác cần dữ liệu cào: chạy cùng chuỗi với thẻ riêng, hoặc gửi danh sách nguồn cho P2. **Không tự viết bộ cào riêng ra định dạng khác.**

### 6.2 Tự ghi (thiết bị, ứng dụng thật)
`source = "tuthu"`, `person = "tuthu:<mã người>"` (cùng một người → cùng một mã ở mọi bộ), ghi thêm `thiet_bi`, `ung_dung`, mức khử ồn nếu chỉnh; người tham gia đồng ý; chỉ giọng của người tham gia.

### 6.3 Tự sinh dữ liệu sửa
Công cụ, cách sinh là tự do. Bắt buộc: dòng `edits` đủ trường ở mục 2.3; ghi công cụ, phiên bản, tham số; WebRTC NS ghi `ns_level`; nhãn đúng ranh giới mục 2.2.

### 6.4 Đặt tên và định dạng
- Bộ mới: `data_labels/v6/<px>_<tên>/manifest.json`; âm thanh dưới `data_labels/v6/<px>_<tên>/<kind>/<id>.wav`; trường `wav` = đường dẫn tương đối từ `data_labels/realistic_v4` (ví dụ `../v6/<px>_<tên>/<kind>/<id>.wav`).
- **id duy nhất trên toàn kho**, có tiền tố riêng (ví dụ `p3x_…`). Không ghi vào bộ của người khác; không tự đổi nhãn bộ có sẵn — muốn đổi thì đề xuất P0 (đổi nhãn làm đổi baseline của cả nhóm).
- `meta.person` ghi nhất quán: hệ thống **tự tính fold** từ chuỗi này (`person_fold = crc32 % 5`), không đọc `meta.fold`.
- wav 16 kHz, một kênh, 16 bit.

### 6.5 Kiểm trước khi nộp
1. `python -m v6.data.kiem_bo data_labels/v6/<bộ>/manifest.json --moi` → phải **ĐẠT** (cấu trúc, id, nhãn, trường bắt buộc, ranh giới nhãn, chính sách nhái giọng).
2. `python -m v6.data.lockbox check data_labels/v6/<bộ>/manifest.json` — bộ có mục tập khoá chỉ dùng để chấm cuối, không học / chọn ngưỡng. Tập khoá v2: ghi âm cào thuộc kênh `crc32 % 5 == 0`; mọi thứ sinh bằng công cụ **để dành** (`vieneu`, `styletts2`, `knnvc`, `resenh` trong `v6/tts_multi/engines.json`); người mới của vox, celeb, VoxPopuli. **Không sinh dữ liệu học bằng công cụ để dành.**
3. Trong mã học của bạn: `from v6.data.kiem_bo import dung_de_hoc` để bỏ bản ghi có cờ.
4. Thẻ `data_labels/v6/<bộ>/README.md`: nguồn · giấy phép · ngày · ai làm · cách làm (công cụ, phiên bản, tham số) · số tệp theo loại · số người nói · đã kiểm chưa.

### 6.6 Chính sách dữ liệu
- Cào: chỉ CC BY / BY-SA / CC0; YouTube chỉ kênh không chuyên (từ 17/10; dữ liệu cũ của vài kênh lớn được giữ theo quyết định của nhóm trưởng); kênh mới nhóm trưởng duyệt; không tôn giáo, trẻ em, trường phổ thông.
- Kho nghiên cứu giấy phép NC / ND mới: hỏi nhóm trưởng. Kho NC đang dùng (VIVOS, VietMed, MMS-TTS…): chỉ nội bộ, không phát tán.
- Nhái giọng (TTS / VC): chỉ người trong kho nghiên cứu (VIVOS, vox, EdAcc, AMI, VoxPopuli, LibriSpeech); **không** người cào được, celeb, VSS, VSASV, viVoice; giọng tham chiếu là bản ghi **khác** của cùng người; mỗi người ≤ 6 % số tệp một bộ.
- Không xoá — gỡ = chuyển `_quarantine` kèm nhật ký lý do.

---

## 7. Đã chốt và còn mở

**Đã chốt (quyết định 18, 20/10):**
1. 554 tệp giọng máy nhân bản celeb / VSS → cờ `chinh_sach`, không dùng.
2. Khử ồn theo bậc: đơn sơ của ứng dụng (WebRTC NS mức 0–2…) = 01; chuyên nghiệp / mạnh (WebRTC NS mức 3, chế độ «Cao», AI, công cụ khử ồn) = 02. Đã áp trên mọi bộ; 618 đoạn WebRTC không ghi mức tạm loại.
3. Tập khoá tiếng Anh người mới = VoxPopuli chưa dùng + kênh cào tiếng Anh.
4. `vcmulti` (chuyển giọng AI, 02): P3 dùng và đo; P5 giữ hạ tầng sinh.
5. Baseline đã chấm lại theo nhãn mới (`CHUNG.md §1.4`). Mô hình hiện tại vẫn học theo nhãn cũ: P0 học lại baseline một lần cho mốc sạch chung (`P0/DU_LIEU.md`), mỗi lane học kênh của mình theo nhãn mới.

**Còn mở:**
- Sinh lại 618 đoạn WebRTC NS có ghi mức (P3); đo khử ồn trên ứng dụng thật để chốt thang (P2, P3).
- VSS chiếm 37 % natvi và có 17 % vu oan lịch sử — số tiếng Việt báo cả khi có và khi bỏ VSS.
