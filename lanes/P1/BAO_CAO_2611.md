# Báo cáo P1 ngày 26/11/2026: pipeline hiện tại, vấn đề đã gặp từ 23/11, việc cần quyết

**Người viết:** P1 (Huấn), Claude soạn.
**Ai đọc:** nhóm trưởng, P0, và Claude của nhóm trưởng.
**Nhánh:** `lane/p1-bao-cao-2611`, tách từ `lane/p1`. Nhánh cũ giữ nguyên.

**Lưu ý khi đọc:**
- Kho git chỉ có **mã và tài liệu**. Dữ liệu (`data/`, `data_labels/`), số đo (`*.jsonl`), log (`*.log`) và nhật ký chi tiết `lanes/P1/NHAT_KY/` nằm trên máy P1, không có trong git.
- Mọi số bên dưới chép từ các tệp đó, kèm tên tệp để đối chiếu khi có gói dữ liệu.
- Mốc bắt đầu: 23/11, ngày Huấn chốt làm lại dữ liệu và mô hình từ đầu. Trước mốc này chỉ giữ luật P0 và bài học sự cố.

---

## 0. Tóm tắt

| chỉ số | trước | nay | đích |
|---|---|---|---|
| Hàng Constraint có bài kiểm chủ động và đạt | 9/24 (25/11) | **11/24** | 24/24 |
| K1 chặn oan v1, cách đếm của nhóm | 63/1 377 = 4,6 % (22/11) | **48/1 368 = 3,5 %** (2,7–4,6) | ≤ 2 % |
| K1 chặn oan v1, cách đếm P1 (chặn đúng nếu đo độc lập xác nhận vi phạm) | chưa tính | **20/1 368 = 1,5 %** (0,9–2,2); vi 2,1 %, en 0,2 % | ≤ 2 % |
| K1 chặn oan v2 dev, cách đếm của nhóm | — | **21/265 = 7,9 %** (5,2–11,8) | ≤ 2 % |
| K1 chặn oan v2 dev, cách đếm P1 | — | **10/265 = 3,8 %** (2,1–6,8) | ≤ 2 % |
| Hàng ồn SNR, cách đo đồng thuận (dev, chưa cài) | — | chặn oan có trọng số 1,8 % (v4 dev), 1,9 % (v3 dev) | ≤ 2 % |

- Số v1 và v2 là **hai bộ khác nhau**, không so trực tiếp được.
- v1 đo lại sau hai bản sửa 23/11 và 25/11. Fold 3–4 của v1 đã mở ngày 23/11, nên số v1 chỉ để báo. v2 dev là bộ sạch, nhưng mới có 265 tệp.
- Trên cả hai bộ, nguồn chặn oan chính là **«quá ít tiếng nói» ở tệp tiếng Việt**.
- Req 9, 12, 13, 14 chưa đo: cần pipeline đầy đủ và gói `du_lieu_P1.tar` (natvi / naten, checkpoint bộ phát hiện), máy P1 chưa có. P1 ưu tiên làm đúng các hàng K1 trước.

---

## 1. Pipeline hiện tại

### 1.1 Sản phẩm (phần P1 giữ)

```
tệp âm thanh ─► K1 (vaudit/k1.py; chế độ mở rộng thêm vaudit/k1x.py, vaudit/events.py)
                 ├─ hàng cứng vi phạm ─► 04 + đúng cụm lý do
                 ├─ hàng mềm ─► cảnh báo
                 └─ hợp lệ ─► (XLS-R + chuyên gia P2–P5 + K7 của P0) ─► K8 luật nhãn ─► K9 hồ sơ / PDF
               vaudit/cond.py: SNR / nền / RT60, chạy trong pipeline, không ở K1
               (thiếu cond_models.joblib trên máy P1, nên chưa chạy được)
```

**Bên trong K1, mỗi kênh:**
- **Mặt nạ lời nói:** khung đủ to (năng lượng) VÀ ASR tiếng Việt (CTC) nghe ra chữ.
  - Sửa 23/11 (Huấn duyệt, chọn trên dev): khi sắp chặn «quá ít tiếng nói» thì hợp thêm khung «đủ to VÀ AST AudioSet nghe ra lời > 0,3». Phép hợp chỉ tăng số giây, nên tệp đang nhận không đổi.
  - Sửa 25/11 (C4): `chunk_frames` để trống một số hàng logits; hàng toàn 0 bị đọc thành chữ ảo, ví dụ tệp 80 s có thêm 9,2 s lời ảo. Nay hàng trống tính là blank. Đã đo trên T1.x: 21 ca sửa đúng, 0 ca tụt (`experts/P1/coverage/so_c4.py`). Lỗi gốc nằm ở `features.py` của lane khác; đề xuất ở `lanes/DE_XUAT/2511_P1_chunk-frames.md`.
- **Các số đo khác:**
  - dải tần thực (`bandwidth_hz`), âm lượng lời, cắt đỉnh (bình nguyên; tệp nén dùng đuôi biên độ), mất tín hiệu (chuỗi số 0 giữa lời);
  - số người nói (resemblyzer 1,6 s, gom cụm cosine 0,35);
  - chế độ mở rộng: ngôn ngữ (whisper-small), nói chồng (pyannote segmentation 3.0 bản onnx của sherpa), loại âm thanh / kiểu nói (AST).
- Thay đổi K1 chưa commit trước đây: chỉ hai bản sửa trên, nằm trong `vaudit/k1.py` của nhánh này.

### 1.2 Quy trình đo của P1 (không thiên lệch)

1. **Chốt luật trước, xem số sau.** Luật chấm của mỗi lần đo ghi trong nhật ký trước khi chạy.
2. **Chia theo video / người nói** bằng `crc32(base) % 5`: fold 0–2 là dev để chọn, fold 3–4 là test chỉ để báo. Test chỉ mở một lần, sau khi Huấn duyệt. Bộ cào không có fold 0 (đó là tập khoá lockbox v2).
3. Mọi tỉ lệ báo `k/n` kèm Wilson 95 %.
4. **Hai cách đếm chặn oan** (Huấn chốt 24/11):
   - nhóm: mọi tệp thật hợp lệ bị 04 là chặn oan;
   - P1: tệp bị 04 mà một phép đo độc lập, khác phương pháp với bộ chặn, xác nhận vi phạm hàng cứng thì tính là chặn đúng.

   Không sửa `mong_doi`; xác nhận ghi ra tệp riêng.
5. Nhãn suy từ tham số dựng phải có ít nhất một phép đo độc lập trên tệp đầu ra. `kiem_bo` và `lockbox` chỉ kiểm định dạng.
6. **Gemini Neon nghe hộ:** nghe mù, phiên mới, trộn ca đối chứng, xáo thứ tự. Kết quả chỉ là «nghe sơ bộ», thành nhãn nghe tay khi Huấn xác nhận. Nhãn mức ồn của Gemini không dùng để đếm.

### 1.3 Dữ liệu (đều trên máy P1, không trong git)

| bộ | là gì | dùng cho |
|---|---|---|
| `p1_tunhien_v1` | 1 415 đoạn ghi âm thật vi / en, cào YouTube CC BY (kênh không chuyên, tối đa 3 video / kênh, đã soát tiêu đề và Gemini nghe mù, gỡ khoảng 63 video sai loại) | đo chặn oan K1, nhãn «01 có khả năng» |
| `p1_tunhien_v2` | 486 đoạn từ 158 video Gemini đề xuất, Claude kiểm giấy phép / kênh | như trên; v2 là bộ sạch cho bản sửa C4 (fold 3–4 v1 đã mở ngày 23/11) |
| `k1axes/p1_v1` (T1.x) | 3 060 tệp dựng từ 24 tệp gốc thật (12 vi, 12 en, fold 0–2) × các ca của `v6/coverage/k1_axes.py` | bài kiểm chủ động cho các hàng K1 |
| `p1_k1bosung_v1` | 168 tệp: ngôn ngữ (sách nói MUSAN 6 tiếng), nói chồng (trộn hai câu FLEURS 0 / 5 / 20 / 40 %), loại âm thanh (nhạc MUSAN) | ba hàng mà T1.x chưa có |
| `p1_dieukien_v3` / `v4` / `v4b` | giọng sạch FLEURS + ồn DEMAND / MUSAN + RIR, SNR đặt trước | học và chấm bộ ước SNR / nền / vang |

---

## 2. Việc cần quyết: dùng lại khoảng 63 video đã gỡ làm bộ kiểm hàng «Loại âm thanh»

### 2.1 Bối cảnh
- Constraint «Loại âm thanh»: phải là ghi âm lời nói, **không phải podcast, phát thanh, sách nói, nhạc**; P1 xếp thêm lồng tiếng / game vào nhóm này. Vi phạm thì ra 04.
- Hiện **chưa có ca kiểm nào** cho podcast / phát thanh / sách nói / lồng tiếng. K1 tự ghi trong `not_impl` rằng AudioSet không tách được các loại này, chỉ chặn nhạc.

### 2.2 Nguồn có sẵn
- Từ 21 đến 25/11, khi soát `p1_tunhien_v1`, Huấn đã duyệt gỡ khoảng 63 video. Danh sách kèm lý do ở `experts/P1/scrape/cao_tunhien.py` (`LOAI_THEO_NGHE`) và `experts/P1/scrape/loai_tay.json`.
- Phân loại (đếm từ hai danh sách): sách nói / đọc truyện / giọng TTS khoảng 22, truyền hình hoặc chương trình đăng lại khoảng 13, nhạc / hát khoảng 10, podcast 7, lồng tiếng 5, còn lại 1–2 mỗi loại (game, review phim dựng, stock footage, tôn giáo, trẻ em).
- Âm thanh gốc vẫn còn trong `data/p1_scrape/goc`. Đã kiểm 16 video trong `LOAI_THEO_NGHE`: còn đủ cả 16. Các video đều đã qua lọc giấy phép CC BY lúc cào.

### 2.3 Chỗ vướng luật
- Luật dữ liệu P1 §4: «ghi âm thật cào về chỉ được nhãn 01 có khả năng; không gán nhãn 04 bằng tay; nhãn đến từ cách tạo tệp, không đến từ việc nghe».
- Nhãn «podcast / sách nói / …» của các video này đến từ tiêu đề / thẻ / tên kênh cộng việc Gemini nghe mù và Huấn duyệt, tức là **có phần từ việc nghe**.

### 2.4 Phương án

| | A. Dùng làm bộ kiểm riêng (P1 đề xuất) | B. Không dùng |
|---|---|---|
| Cách làm | Bộ mới `p1_loaiam_v1`: cắt đoạn như v1, gắn `ngoai_pham_vi`. Thẻ README ghi nguồn nhãn «tiêu đề / kênh + Gemini nghe + Huấn duyệt». Bỏ video tôn giáo và trẻ em. Chỉ dùng để **đo bao nhiêu tệp K1 lọt**, không dùng để học hay chọn ngưỡng | Chỉ dùng sách nói MUSAN (nhãn theo kho, có en và 6 tiếng khác, **không có vi**) |
| Được | Có số đo thật cho podcast / phát thanh / lồng tiếng, cả vi lẫn en | Giữ nguyên luật §4 |
| Mất | Một ngoại lệ có ghi rõ với luật §4 | Hàng «Loại âm thanh» tiếp tục chỉ là tuyên bố giới hạn cho podcast / phát thanh / lồng tiếng |
| Giới hạn | Podcast 7, lồng tiếng 5 video: chưa đủ ≥ 5 người nói mỗi ô ở cả vi và en | |

Sau khi đo, nếu K1 lọt nhiều thì Huấn chọn: giữ tuyên bố giới hạn, hoặc mở việc dựng bộ phát hiện riêng (dấu hiệu dàn dựng như nhạc hiệu, nén động mạnh, micro phòng thu). Việc sau là nghiên cứu, cần dev / test riêng.

---

## 3. Vấn đề đã gặp từ 23/11

Mỗi vấn đề ghi: là gì, bằng chứng, nguồn gốc, trạng thái.

### 3.1 Cách đo hàng ồn (SNR ≥ 10 dB)

**Tiến trình:**
- Bộ ước học (run21) bị loại. Thử thêm ba bộ ước mạnh hơn (S1: thống kê khung Brouhaha; S2: đầu torch trên 13 lớp w2v-VI; S3).
- Phép thử khả thi cần sai số σ ≤ 2 dB; tốt nhất đạt 2,87 dB. Vì vậy **dừng hướng «một bộ ước đủ chính xác»**.
- Chuyển sang **cách đo đồng thuận**: chỉ 04 khi mlp_F1 VÀ S2 cùng đoán < 10 dB; chỉ một bộ đoán < 10 dB thì cảnh báo «tiếng ồn cao». Huấn duyệt 26/11; cách cũ giữ để đối chiếu.
- Đề xuất đầy đủ: `lanes/DE_XUAT/2611_P1_cach-do-on.md`.

**So ba cách đo trên cùng thước** (chỉ dev):

| bộ | chỉ số | cách cũ C3v2 | mlp_F1 | đồng thuận |
|---|---|---|---|---|
| v4 dev | chặn oan có trọng số | 6,8 % | 4,7 % | 1,8 % |
| v3 dev (chưa từng dùng để chọn) | chặn oan có trọng số | 12,8 % | 6,5 % | **1,9 %** |
| | bắt được < 10 dB | 82,8 % | 81,8 % | 77,7 % |
| tệp thật v1 dev (656) | cách đếm nhóm | 20,9 % | 13,6 % | 7,5 % |
| | cách đếm P1 (WADA xác nhận) | 6,2 % | 3,5 % | 1,1 % |

**Vấn đề còn lại** (bảng 14 mục, tóm lại):

| # | vấn đề | trạng thái |
|---|---|---|
| 1 | Nhãn SNR dữ liệu dựng chỉ tính ồn thêm vào, bỏ qua ồn sẵn của bản gốc | Chờ Huấn nghe 12 bản gốc |
| 2 | Trọng số chặn oan lấy từ một phép đo có sai số | Sẽ tính lại với hai phép đo khác, báo khoảng dao động |
| 3 | Dev đã dùng để chọn bốn vòng; 1,8 % sát mốc 2 %, bootstrap 0,4–4,3 % | Đóng băng cấu hình, mở test một lần khi Huấn duyệt |
| 4–6 | Phép xác nhận WADA yếu (chạm trần 60 dB ở 158/657 tệp); mẫu nghe tay cũ không đại diện; thang nghe chưa gắn mốc 10 dB | Thay bằng nghe mù 49 tệp bị chặn + 15 đối chứng, có tệp mốc 10 dB |
| 7 | Hai script lọc ra 656 và 657 tệp | Dùng một hàm lọc chung |
| 11 | Nền ngắt quãng bị đoán cao | Ứng viên đặc trưng mới, thử trên dev mới |
| — | Cách đo của nhóm (`cond.py`, XLS-R) không chạy được để so | Thiếu `cond_models.joblib` |

### 3.2 Các hàng K1 còn sai

Nguyên nhân dưới đây đều **đã đo**, chưa sửa K1. Chi tiết và ứng viên: `lanes/P1/PHAN_TICH_K1_2611.md`.

| hàng | sai | nguyên nhân đo được | ứng viên sửa cách đo |
|---|---|---|---|
| Số người nói | 47/72 | Ca 2 người en hỏng do bộ kiểm (mục 3.3). 6 người ra 4–5: người ít cửa sổ bị luật 10 % bỏ; nhúng resemblyzer tách người kém (khoảng cách giữa người 0,26–0,29 < ngưỡng 0,35). 27 tệp một người bị báo «nhiều người nói» vì tách thành cụm phụ 15–17 % | N1 sửa bộ kiểm; N2 cụm tối thiểu ≥ 3 cửa sổ; N3 nhúng ECAPA / sherpa |
| Dải tần ≥ 6 kHz | 13 tệp G.711 / AMR-NB được nhận | Tệp gốc chạm 0 dBFS; nâng mẫu thì bị xén; méo tạo nền phẳng tới 8 kHz, lên xuống theo giọng, nên K1 tính là dải tần | B1: băng hẹp khi có «vách» ≥ 25 dB rồi nền phẳng ≤ 6 dB |
| Cắt đỉnh | 9 sai quyết định | Tệp không nén: luật bình nguyên bỏ sót mẫu chạm trần đứng lẻ (5,6 % đo ra 4,87 %). Opus: vọt đỉnh làm đuôi biên độ trải | C1 đếm mọi mẫu chạm trần khi tệp có bình nguyên; C2 hiệu chuẩn riêng cho Opus |
| Mất tín hiệu | 24/72 tệp nén | Opus 24k: khoảng mất tắt dần −58 đến −87 dBFS, không về 0. MP3 / AAC: mẫu dư làm mốc đầu trễ 0,05–0,07 s | D1 (tệp nén): khoảng mất là đoạn thấp hơn sàn nghỉ của tệp ≥ 15 dB |
| Nói chồng | 20 %: 20/24 · 40 %: 23/24 | Cả 5 tệp sót là en; vi đúng hết | chưa chẩn đoán |
| Loại âm thanh | nhạc có hát 8/12 | 4 bài có hát được nhận (`music-fma-0016`, `-0075`, `music-jamendo-0141`, `-0144`) | chưa chẩn đoán |
| Lượng tiếng nói | 4 tệp ở mức 20 % | Bài kiểm và K1 đếm giây lời theo hai cách khác nhau, ca sát ngưỡng | thống nhất cách đếm |

**Hai giả thuyết đã bị số đo bác** (ghi để khỏi lặp lại):
- dải tần sai do ảnh gương khi nâng mẫu: phổ phẳng, không có đỉnh ở 7,75–8 kHz;
- số người nói en sai do ASR tiếng Việt làm mất cửa sổ tiếng Anh: số cửa sổ trên tệp gốc không lệch theo ngôn ngữ.

### 3.3 Lỗi trong bộ kiểm và nguồn gốc

| lỗi | hậu quả | nguồn gốc | xử lý |
|---|---|---|---|
| Ca «2 người»: người thứ hai luôn là tệp gốc đầu tiên cùng ngôn ngữ, nên 11/12 tệp en dùng chung en_00, mà 8 s đầu của en_00 chỉ có lời rời rạc (0 cửa sổ dùng được) | 11 ca en «2 người» thực chất K1 chỉ thấy 1 người | **Mã dựng của nhóm** (`v6/coverage/k1_axes.py`: `_concat_people` cùng danh sách `others`) cộng **đặc điểm dữ liệu** chưa được ghi lại | N1: xoay vòng người ghép, đo độc lập ≥ 3 s lời mỗi người |
| 5/24 tệp gốc có đỉnh đúng 0 dBFS | Bản băng hẹp bị xén đỉnh và lọt hàng Dải tần | **Đặc điểm dữ liệu** (video được chuẩn hoá tới 0 dBFS) cộng bước chọn tệp gốc của P1 (`k1_axes_p1.pick()`, mã AI viết) không kiểm đỉnh | Ghi đỉnh vào provenance; thêm ca «băng hẹp + xén» có chủ đích |
| Ca nói chồng 0 % đặt người thứ hai sau 0,3 s trên nền số 0 | Cả 24 tệp bị cảnh báo «mất tín hiệu» (K1 báo đúng) | **Mã P1 do AI viết** (`bo_sung_hang.py`) | Bản sau lấp khoảng trống bằng nền của nguồn |
| Lần dựng `p1_k1bosung_v1` đầu dừng giữa chừng (FLEURS vi chỉ có 59 câu sạch, cần 108); phép kiểm số lượng đặt sau bước ghi | 48 tệp dở | **Mã P1 do AI viết** | Chuyển vào `_quarantine/` kèm lý do; kiểm nguyên liệu trước khi ghi |

### 3.4 Dữ liệu và nhãn
- **v2, Gemini soát nội dung** (150 đoạn thật + 12 đối chứng):
  - Đối chứng đúng 12/12 ở các trường có lời nói / ngôn ngữ / nhạc, 11/12 ở trường loại.
  - 9/150 đoạn bị gắn cờ (5 đọc văn bản, 2 nhạc / hát, 1 không lời, 1 chương trình); chờ Huấn nghe xác nhận.
  - Nhạc nền nghe thấy ở 31/150 đoạn, chỉ để biết.
- **Nhãn ngôn ngữ** của sách nói MUSAN lấy từ chú thích của kho, chưa có phép đo độc lập.
- **Truy vấn cào trái Constraint** (21/11): các truy vấn «podcast style chat», «talkshow nhỏ», «tâm sự chia sẻ» kéo về khoảng 40 % đoạn vi không tự nhiên. Đã bỏ truy vấn, gỡ video, thêm bước soát tiêu đề trước khi dựng. Đây cũng là nguồn của khoảng 63 video ở mục 2.

### 3.5 Vận hành
- Trước hôm nay, việc của P1 chưa được commit. Lần này commit mã và tài liệu vào nhánh mới.
- `vaudit/test_k1.py` không có trong bộ gửi, nên chưa chạy được đủ bộ test bắt buộc khi đổi K1. `test_ticket` ghi cứng đường dẫn `D:\tmp_va`. Cả hai là lỗi có sẵn, đã kiểm bằng `git stash`.
- Ổ D đầy: `data/p1_raw` (33,5 GB) đã chuyển sang ổ F qua junction; đối chiếu đủ 66 169 tệp và đúng số byte.
- Chạy song song ba việc CPU làm mỗi việc chậm khoảng gấp đôi.

---

## 4. Số chặn oan K1 trên tệp thật (26/11), chi tiết

### 4.1 v1: 1 368 tệp accept (`chay_k1.py run --bo p1_tunhien_v1 --tag 2611`)

| lý do 04 | 22/11 | 26/11 | đo độc lập xác nhận vi phạm (26/11) |
|---|---|---|---|
| quá ít tiếng nói | 38 | 26 | 12; 14 không xác nhận (13 vi, 1 en; Brouhaha đo 31–100 % lời) |
| quá nhỏ | 10 | 10 | 10/10 |
| băng hẹp | 6 | 6 | 6/6 |
| cắt đỉnh nặng | 5 | 5 | chưa có phép đo độc lập, nên tính chặn oan |
| quá 4 người | 2 | 1 | như trên |
| **tổng** | **61** | **48** | 28 xác nhận, 20 chặn oan |

- Cách đếm nhóm: **48/1 368 = 3,5 %** (en 12/450 = 2,7 %, vi 36/918 = 3,9 %). Cách đếm P1: **20/1 368 = 1,5 %** (0,9–2,2) (en 1/450 = 0,2 %, vi 19/918 = 2,1 %).
- So 22/11 trên cùng tệp: gỡ được 14 (13 «quá ít tiếng nói» nhờ AST 23/11 và C4 25/11, 1 «quá 4 người»).
- **Có 1 ca tụt:** `vi/p1n_vi_Op30S3w9RAI_1` mới bị chặn sau C4 (K1 đếm 29,7 s / 120 s, sát mốc 30 %). Brouhaha đo 99 % lời. Hai đoạn khác cùng video đã bị chặn từ 22/11 với cùng mâu thuẫn, nên video này là ca rõ để chẩn đoán mặt nạ lời của K1.
- 5 ca cắt đỉnh bị tính chặn oan chỉ vì chưa có phép đo cắt đỉnh khác phương pháp. Nếu thêm phép đo đó (chốt luật trước) thì số theo cách đếm P1 có thể giảm.

### 4.2 v2 dev: 265 tệp (`chay_k1.py run --bo p1_tunhien_v2 --tag 2611 --dev`)

- 265 tệp, fold 1–2. Đã kiểm không có tệp test.
- Phép đo độc lập: `experts/P1/build_cond/do_doc_lap.py`, mặt nạ lời nói là VAD Brouhaha. Luật chốt trước khi đo. Phương pháp đã kiểm trên tệp dựng biết đáp án: tách sạch băng rộng / băng hẹp quanh −50 dB, nhận đúng mức −41,5 dBFS. Riêng tỉ lệ lời, Brouhaha đếm dư vài điểm, nên phép xác nhận này nghiêng về bảo thủ.

| lý do 04 | tệp (video) | đo độc lập xác nhận vi phạm |
|---|---|---|
| quá ít tiếng nói | 10 (7), đều vi | 3; 7 không xác nhận. Riêng `Q_J_ZYAFw6Y`: Brouhaha thấy 42–98 % lời, K1 chỉ đếm 0,4–45,7 s |
| quá nhỏ | 6 (3), đều en | 6/6 (−42,4 đến −44,6 dBFS) |
| băng hẹp | 2 (1) | 2/2 |
| không phải ghi âm lời nói | 2 | không đo độc lập được, tính chặn oan |
| quá 4 người nói | 1 | như trên |

- Cách đếm nhóm: **21/265 = 7,9 %**. Cách đếm P1: **10/265 = 3,8 %**; en 1/114 = 0,9 %, vi 9/151 = 6,0 %.
- **Nguồn chặn oan chính: «quá ít tiếng nói» ở tệp vi.** Đây là việc ưu tiên tiếp theo.

---

## 5. Việc tiếp và việc chờ quyết

| việc | ai quyết / làm |
|---|---|
| Dùng hay không dùng khoảng 63 video đã gỡ (mục 2) | Huấn, có thể hỏi nhóm trưởng / P0 |
| Duyệt ứng viên sửa cách đo các hàng K1 (N1–N4, B1–B2, C1–C3, D1) rồi chấm trên dev theo luật chốt ở `PHAN_TICH_K1_2611.md` mục 5 | Huấn duyệt, P1 làm |
| Chẩn đoán «quá ít tiếng nói» trên tệp vi thật (nguồn chặn oan chính; bắt đầu từ `Op30S3w9RAI` và `Q_J_ZYAFw6Y`, nơi Brouhaha đo > 90 % lời mà K1 đếm < 30 %) | P1 |
| Thêm phép đo độc lập cho cắt đỉnh (chốt luật trước) để đếm theo cách P1 | Huấn duyệt, P1 làm |
| Huấn nghe: 9 đoạn v2 Gemini gắn cờ, 12 bản gốc (nhãn SNR), trang nghe 64 tệp hàng ồn | Huấn |
| Tự ghi bộ kiểm hàng Kiểu nói (thì thầm / la hét / hát), ≥ 5 người mỗi ngôn ngữ: kịch bản `lanes/P1/KICH_BAN_TU_GHI_KIEU_NOI.md` | Huấn tổ chức |
| Cài cách đo đồng thuận vào `vaudit/cond.py`, mở test một lần | Huấn duyệt |
| Gói `du_lieu_P1.tar` (natvi / naten, checkpoint, `cond_models.joblib`) để đo Req 9 / 12 / 14 và so với `cond.py` | nhóm trưởng |
| `vaudit/test_k1.py` còn thiếu trong bộ gửi | nhóm trưởng |

## 6. Tệp trong nhánh này

| tệp | nội dung |
|---|---|
| `lanes/P1/BAO_CAO_2611.md` | báo cáo này |
| `lanes/P1/HUONG_DAN.md` | hướng dẫn đọc nhánh, môi trường, dữ liệu cần xin, lệnh chạy lại |
| `lanes/P1/PHAN_TICH_K1_2611.md` | phân tích các hàng K1 còn sai, ứng viên, luật chấm |
| `lanes/P1/KICH_BAN_TU_GHI_KIEU_NOI.md` | kịch bản tự ghi hàng Kiểu nói |
| `lanes/P1/BAO_CAO_2311.md`, `DAC_TA_DU_LIEU_TU_TAO_2511.md` | báo cáo 23/11, đặc tả dữ liệu tự tạo |
| `lanes/DE_XUAT/2611_P1_cach-do-on.md` | đề xuất cách đo đồng thuận hàng ồn |
| `lanes/DE_XUAT/2511_P1_chunk-frames.md`, `2111_P1_chan-oan-hang-cung.md` | đề xuất sửa `features.py`; đề xuất cách đếm chặn oan hàng cứng |
| `vaudit/k1.py` | hai bản sửa: AST hợp mặt nạ lời (23/11), hàng logits trống (C4, 25/11) |
| `experts/P1/scrape/` | cào, kiểm giấy phép, chọn danh sách cho Gemini |
| `experts/P1/build_cond/` | dựng bộ điều kiện, đo Brouhaha / DFN, học và chấm bộ ước SNR, chạy K1 trên bộ thật, đo độc lập |
| `experts/P1/coverage/` | bọc T1.x cho tệp gốc P1, bộ kiểm bổ sung, chẩn đoán người nói / dải tần, test hàng «Số tệp» |
