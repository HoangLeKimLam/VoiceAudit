# Phân tích các hàng K1 còn sai và kế hoạch dữ liệu cho hai hàng chưa kiểm (26/11)

Huấn giao 26/11: (1) hàng Số người nói, (2) ba hàng tín hiệu (Dải tần, Cắt đỉnh, Mất tín hiệu), (4) kế hoạch dữ liệu cho Kiểu nói và podcast / phát thanh.
- Chỉ phân tích và đề xuất. **Chưa sửa K1, chưa sinh hay tải dữ liệu.**
- Số liệu lấy từ T1.x tag `p1_v1_c4`: 24 tệp gốc lấy từ fold 0–2, tức dev.
- Script chẩn đoán, chỉ đọc: `experts/P1/coverage/chan_doan_nguoi.py` và `chan_doan_bangtan.py`. Các phép đo khác chạy tay, kết quả ghi trong từng mục.
- Mỗi ứng viên sửa cách đo phải chấm trên hai thứ. Luật chấm chốt trước, ở mục 5.
  - T1.x hàng đó: ca dựng.
  - Tệp thật hợp lệ dev: số chặn oan hoặc cảnh báo oan mới phát sinh.

## 1. Số người nói (47/72 đúng)

- **Cách K1 đếm** (`count_speakers`):
  - nhúng resemblyzer trên cửa sổ 1,6 s (bước 0,8 s); chỉ lấy cửa sổ có ≥ 60 % khung thuộc mặt nạ tiếng nói (năng lượng ∩ ASR tiếng Việt);
  - gom cụm average-linkage, cắt ở khoảng cách cosine 0,35;
  - đếm cụm có ≥ max(2 cửa sổ, 10 %).
- **Số liệu:** `_chan_doan/nguoi_2611.jsonl`, gồm số cửa sổ của từng người, cỡ cụm, khoảng cách giữa người và trong một người.

| ca | đúng | nguyên nhân đo được |
|---|---|---|
| 2 người, en | 1/12 | **Lỗi dựng bộ kiểm.** `_concat_people` lấy «người thứ hai» là tệp gốc đầu tiên cùng ngôn ngữ, nên 11/12 tệp en dùng chung en_00. 8 s đầu của en_00 cho **0 cửa sổ đạt** (lời rời rạc, không có đoạn 1,6 s nào ≥ 60 % là lời). K1 vì vậy chỉ thấy một người: số cửa sổ theo người là [7, 0], [3, 0], [9, 0]… |
| 2 người, vi | 10/12 | Ngược lại, người ghép vi_00 cho 4–7 cửa sổ. Hai ca sai (vi_09, vi_11) là do **chính tệp gốc** chỉ có 1–2 cửa sổ, vỡ thành các cụm 1 cửa sổ và bị luật «≥ 2 cửa sổ» bỏ |
| 6 người | 12/24 (K1 ra 4 hoặc 5) | (a) Một hai người có rất ít cửa sổ: ở en, người en_00 luôn chỉ có 1 cửa sổ, nên tệp en thực chất chỉ có 5 người K1 thấy được. Cụm 2–3 cửa sổ bị luật 10 % bỏ (38 cửa sổ → cần ≥ 4). (b) Nhúng tách người kém: khoảng cách nhỏ nhất giữa hai người 0,255–0,29, khoảng cách trong một người 0,17–0,24, đều dưới ngưỡng cắt 0,35, nên có cặp người bị gộp |
| 1 người bị báo «nhiều người nói» (27 tệp ở các hàng khác) | – | Cùng một người tách thành một cụm chính và một cụm phụ chiếm 15–17 %: [10, 2], [30, 6], [147, 27]. Dồn ở en_00 bản dài (`dodai.*`, chính tệp gốc này cũng có cụm lẻ) và vi_01 sau khi cắt đỉnh (tệp gốc vi_01 không tách) |

- **Giả thuyết «ASR tiếng Việt làm mất cửa sổ tiếng Anh» bị bác.** Số cửa sổ trên tệp gốc một người là en 4–19, vi 2–21; không lệch theo ngôn ngữ. Lỗi en là do người ghép chung en_00.
- **Ảnh hưởng thật:**
  - Trên tệp thật v1 (22/11), «nhiều người nói» bật ở 271/1 817 tệp (15 %); cảnh báo này không chặn.
  - «quá 4 người nói» ra 04 ở 2 tệp; chưa biết là chặn đúng hay oan.
- **Ứng viên:**
  - **N1. Sửa bộ kiểm** (làm trước, không đụng K1):
    - mỗi tệp gốc ghép với người khác nhau (xoay vòng);
    - mỗi phần người phải có ≥ 3 s lời nói, đo độc lập bằng `oracle.speech_s` theo phần; ghi vào provenance;
    - tách riêng ca «người thứ hai chỉ nói ngắn» để đo giới hạn.
  - **N2. Luật cụm:**
    - cụm tối thiểu ≥ 3 cửa sổ VÀ ≥ 10 %;
    - quyết định «> 4 người» chỉ khi cụm thứ 5 đủ lớn.

    Gỡ được phần lớn 27 cảnh báo thừa (cụm phụ 2 cửa sổ), nhưng có thể làm sót ca 2 người nói ngắn (vi_04: [6, 2]). Phải chấm trên N1.
  - **N3. Nhúng tốt hơn:** ECAPA (speechbrain, có trong `venv_brouhaha`) hoặc mô hình nhúng onnx của sherpa (cùng nguồn với mô hình phân đoạn pyannote đã có). Cần kiểm trọng số có sẵn trên máy chưa; nếu phải tải thì xin Huấn.
  - **N4. Dùng mô hình phân đoạn pyannote 3.0 đã có** (chế độ mở rộng): lấy số người nói cục bộ của mỗi khúc 10 s làm sàn cho phép đếm. Nặng hơn, chỉ thử sau N2 / N3.
- **Thứ tự đề xuất:** N1, rồi N2 (rẻ, chấm ngay trên nhúng đã lưu trong `nguoi_2611.npz`, không cần chạy lại K1), rồi N3 nếu 6 người vẫn < 90 %.

## 2. Ba hàng tín hiệu

### 2.1 Dải tần thực ≥ 6 kHz: 13 tệp sai quyết định, tất cả là G.711 / AMR-NB phải ra 04 nhưng được nhận

- **Nguyên nhân (đã đo):**
  - Cả 5 tệp gốc gây lỗi (vi_01, vi_03, vi_06, en_03, en_11) có đỉnh đúng 1,000, tức 0 dBFS. Các tệp gốc đạt có đỉnh 0,37–0,91.
  - Khi mã hoá rồi nâng lên 16 kHz, sóng vượt ±1 và bị xén (0,003–0,1 % số mẫu).
  - Méo do xén tạo một nền **phẳng** ở 4,5–8 kHz, thấp hơn mức 1–3 kHz 42–50 dB. Nền này lên xuống theo giọng, nên thoả cả hai điều kiện của `bandwidth_hz`: cao hơn mốc «−55 dB» và có độ động ≥ 6 dB. K1 vì vậy trả 8 000 Hz.
  - Tệp băng hẹp không bị xén thì nền 4,5–8 kHz chỉ ở −57 đến −81 dB và K1 trả đúng 4 500 Hz.
  - Giả thuyết ban đầu (ảnh gương của dải 0–250 Hz) **bị bác**: phổ phẳng, không có đỉnh ở 7,75–8 kHz.
- **Không coi đây là lỗi riêng của bộ kiểm.** Ghi âm điện thoại được chuẩn hoá tới 0 dBFS rồi nâng mẫu cũng sẽ như vậy.
- **Ứng viên:**
  - **B1 «vách + nền phẳng»** (đề xuất): coi là băng hẹp khi cả hai điều kiện đúng:
    - mức trung vị 4,5–8 kHz thấp hơn mức 3–3,5 kHz ít nhất 25 dB;
    - các dải 250 Hz trong 4,5–8 kHz chênh nhau ≤ 6 dB.

    Ngưỡng 25 / 6 dB cần chọn trên T1.x dev. Hai điều kiện này tách được «méo hoặc ồn phẳng» khỏi phổ giọng thật, vốn giảm dần và có cấu trúc âm xát.
  - B2: nâng mốc −55 dB lên khoảng −40 dB. Đơn giản hơn nhưng dễ chặn oan micro tối tiếng: tệp thật có dải 6–7 kHz thấp hơn mức 1–3 kHz 19–33 dB, theo docstring của K1.
- **Bộ kiểm:** thêm ca «băng hẹp + xén» có chủ đích. Ghi đỉnh tệp gốc vào provenance để không phụ thuộc vào việc tệp gốc nào tình cờ chạm trần.

### 2.2 Cắt đỉnh: 9 tệp sai quyết định, 16 tệp chỉ dính cảnh báo lạ

- **Tệp không nén, ca 5,6 % (en_00, en_07 ra «cảnh báo» thay vì 04):**
  - Luật `_plateau` chỉ đếm chuỗi ≥ 2 mẫu liền chạm trần, bỏ sót mẫu chạm trần đứng lẻ.
  - Đếm mọi mẫu chạm trần cho đúng 5,63 %; luật hiện hành chỉ ra 4,87–5,52 %. Mức hụt lớn nhất ở en_00 và en_07, có lẽ do giọng nhiều thành phần cao. Ở ca 8 % hụt tới 1,2 điểm, nhưng vẫn trên 5 %.
  - **C1 (đề xuất):**
    - nếu phần lớn mẫu chạm trần nằm trong bình nguyên (bình nguyên ≥ 50 % số mẫu chạm trần) thì đếm mọi mẫu chạm trần;
    - ngược lại (âm thuần, xung nhọn) giữ luật bình nguyên.

    Như vậy âm thuần 1 kHz vẫn ra 0, đúng lý do luật bình nguyên được thêm ngày 11/10.
- **Opus 48k, ca 8 % (7 tệp; ước lượng ra 2,7–5,0 %):**
  - Opus vọt đỉnh mạnh nên «đuôi biên độ» bị trải. Docstring K1 đã ghi đây là hạn chế đã biết.
  - **C2:** hiệu chuẩn `LOSSY_CLIP_FRAC` riêng cho Opus. K1 biết mã hoá từ container. Chọn trên T1.x Opus 0 / 1,4 / 3 / 8 %.
  - **C3:** ước mức cắt bằng chỗ dồn của các cực đại cục bộ (tệp bị cắt có nhiều cực đại dồn sát một mức). Phức tạp hơn; chỉ thử nếu C2 không đủ.
- **16 tệp còn lại** chỉ sai vì cảnh báo lạ «nhiều người nói», dồn ở tệp gốc vi_01 và vi_07. Sửa hàng Số người nói sẽ gỡ được.

### 2.3 Mất tín hiệu: tệp không nén 120/120 đúng; tệp nén sai 24/72

| loại | sai | đo trong khoảng mất |
|---|---|---|
| MP3 64k, AAC 32k | 6 ca lệch mốc (đầu trễ 0,05–0,07 s; sai số cho phép 0,04 s) | Đường bao 10 ms đã xuống −100 dB chỉ 25 ms sau điểm mất. Vài mẫu dư do nén còn > 3e-5 làm chuỗi «toàn số 0» bắt đầu muộn |
| Opus 24k | 12 ca không phát hiện, 6 ca lệch mốc | Khoảng mất **không về 0**: tắt dần theo hàm mũ, khoảng −4 dB mỗi 25 ms, mức trung bình −58 đến −87 dBFS, nên không qua ngưỡng ≈ −90 dBFS |

- **D1 (đề xuất, chỉ cho tệp nén):**
  - Dùng đường bao năng lượng 10 ms thay cho điều kiện «mọi mẫu < 3e-5».
  - Khoảng mất là chuỗi khung ≥ 0,1 s thấp hơn **sàn nghỉ của chính tệp** ít nhất 15 dB, nằm giữa lời nói. Sàn nghỉ lấy từ `floor` mà `speech_mask` đã tính.
  - Điểm đầu: lùi về trước dọc đoạn tắt dần đơn điệu, tới khung lời cuối.
  - Lý do: khoảng nghỉ tự nhiên nằm ở sàn ồn của phòng; khoảng mất thật rơi xuống dưới sàn đó 20–40 dB.
- **Rủi ro:** tệp «cổng ồn» (ứng dụng tắt tiếng giữa các từ) cũng rơi dưới sàn. K1 đã có cờ `so_nguoi_noi_tin_cay` cho loại tệp này; phải đưa ca `cong_on` vào bộ chấm.
- **Sai số mốc ±0,04 s** là lựa chọn của bộ kiểm, không phải chữ của Constraint. Nếu D1 vẫn lệch vì khung mã hoá (MP3 ở 16 kHz là 36 ms) thì báo riêng. Không nới sai số khi chưa hỏi Huấn.

## 4. Kế hoạch dữ liệu: Kiểu nói và Loại âm thanh (podcast / phát thanh / sách nói / lồng tiếng)

### 4.1 Hiện trạng K1
- `vaudit/events.py` dùng AST AudioSet và có nhóm `whisper`, `shout`, `sing` cho hàng Kiểu nói. Ngưỡng `STYLE_P` = 0,5 và `STYLE_SHARE` = 25 % là **đặt tay, chưa hiệu chuẩn trên dữ liệu thật**.
  - Ca tổng hợp WORLD (giọng thì thầm / hát / la hét dựng bằng bộ mã hoá giọng WORLD) không dùng được: AST vẫn nghe là «Speech».
  - Trên bài hát thật, nhóm `sing` không kích hoạt (trung vị 0,07); các tệp này bị 04 qua nhóm nhạc.
- Với podcast, phát thanh và sách nói, K1 **tự tuyên bố không tách được** (`not_impl`), chỉ chặn nhạc.
- Trên tệp thật v1 ngày 22/11, «kiểu nói ngoài phạm vi» bật ở 1/1 817 tệp.

### 4.2 Nguồn có sẵn trên máy (không cần tải)

| ca | nguồn | số lượng | nguồn nhãn |
|---|---|---|---|
| sách nói | MUSAN librivox (PD) | en và 6 tiếng khác; **không có vi** | theo kho (provenance) |
| podcast | video đã gỡ khỏi `p1_tunhien_v1` (`LOAI_THEO_NGHE`, `loai_tay.json`), âm thanh gốc còn ở `data/p1_scrape/goc` | 7 video | tiêu đề / thẻ, cộng Gemini nghe và Huấn duyệt gỡ |
| sách nói / đọc truyện (giọng người hoặc TTS) | như trên | khoảng 15 video | như trên |
| truyền hình đăng lại | như trên | khoảng 10 video | như trên |
| lồng tiếng | như trên | 5 video | như trên |
| hát (người nói hát) | như trên (`2zCSmOlcd1g` hát mộc, `m5Vle27My1o` bolero, `mmFgpMmymYc`), cộng MUSAN có hát | khoảng 3 video và 12 tệp MUSAN | như trên / theo kho |
| thì thầm, la hét | **không có** | 0 | |

- Không dùng: video tôn giáo (`m2o91ylZ6nQ`) và bài hát thiếu nhi (`kO99Wl90w98`), theo luật cào.
- **Vướng luật:** §4 ghi «không gán nhãn 04 bằng tay; nhãn đến từ cách tạo tệp, không đến từ việc nghe». Nhãn của các video đã gỡ đến từ tiêu đề / thẻ cộng việc nghe. Nếu dùng thì phải là **bộ kiểm riêng, gắn `ngoai_pham_vi`, chỉ để đo hạn chế của K1**, ghi rõ nguồn nhãn trong thẻ bộ. Cần Huấn cho phép ngoại lệ này.
- Rào cản về độ phủ:
  - Mỗi ô cần ≥ 10 tệp và ≥ 5 người nói, có cả vi lẫn en.
  - Podcast 7 video và lồng tiếng 5 video là chưa đủ. Phải tách nhiều đoạn mỗi video, nhưng số người nói vẫn bị giới hạn bởi số kênh.

### 4.3 Nguồn phải tải hoặc tự ghi (cần Huấn duyệt)

| ca | phương án | ưu | nhược |
|---|---|---|---|
| thì thầm, la hét, hát | **A. Tự ghi:** ≥ 5 người mỗi ngôn ngữ, mỗi người đọc cùng một đoạn ở giọng thường / thì thầm / la hét / hát, ghi bằng điện thoại | Nhãn đến từ cách tạo tệp, đúng §4. Có cặp đối chứng cùng người, cùng nội dung | Cần người. Có thể ghép với khảo sát Req 13 (3–5 người ngoài nhóm, khoảng 1 giờ mỗi người) |
| thì thầm | B. YouTube CC BY: ASMR thì thầm vi / en, qua `tai_v2.py` (kiểm giấy phép từ metadata, tối đa 3 video / kênh) | Có giọng vi thật | Nhãn từ truy vấn / tiêu đề. ASMR hay kèm tiếng động kích thích (gõ, vò giấy). Ít người nói vi |
| thì thầm, la hét | C. Freesound, chỉ CC0 / CC BY | Nhiều mẫu la hét / hét | Cần khoá API do Huấn tự tạo. Mẫu ngắn, nhiều mẫu không có lời; khó có vi |
| podcast, phát thanh vi | D. YouTube CC BY: kênh podcast / radio tự nhận, truy vấn riêng cho bộ kiểm (không trộn vào bộ «tự nhiên») | Nhãn theo metadata kênh | Kênh podcast phần lớn là kênh chuyên, trái luật «kênh không chuyên». Luật đó áp cho bộ tự nhiên; với bộ kiểm cần Huấn quyết |

- **Đề xuất:** A cho Kiểu nói (là nguồn duy nhất cho nhãn đúng §4 và đủ vi), và 4.2 cộng D cho Loại âm thanh.
- **Người lớn / trẻ em:** không lấy được dữ liệu trẻ em theo luật cào. Đề xuất giữ tuyên bố giới hạn như hiện nay: K1 chỉ ghi số đo `child`.

### 4.4 Câu hỏi đo trước khi có dữ liệu
- K1 đã tự ghi là không tách được podcast / phát thanh / sách nói, nên bộ kiểm 4.2 chủ yếu để **đo cỡ hạn chế** (bao nhiêu tệp lọt). Sau đó Huấn chọn:
  - (a) giữ tuyên bố giới hạn;
  - (b) mở việc dựng bộ phát hiện riêng (dấu hiệu dàn dựng: nhạc hiệu, nén động mạnh / LRA thấp, micro phòng thu). Đây là việc nghiên cứu, cần dev / test riêng.

## 5. Luật chấm ứng viên (chốt trước khi chạy)

- Chọn trên T1.x dev và tệp thật dev (v1 fold 0–2, v2 fold 1–2). Test (fold 3–4) chỉ mở một lần, sau khi Huấn duyệt.
- Mỗi ứng viên báo đủ ba thứ, kèm Wilson 95 %:
  - số ca dựng đúng của hàng;
  - số tệp thật hợp lệ bị **thêm** 04 hoặc thêm cảnh báo của hàng đó;
  - số ca đang đúng nay sai.
- Ứng viên chỉ được chọn khi không thêm chặn oan nào trên tệp thật dev. Nếu có thêm, phải có phép đo độc lập xác nhận là chặn đúng, theo cách đếm hàng cứng 24/11.
- Sau khi cài: chạy bộ bắt buộc của `lanes/P1/CLAUDE.md` dòng 61, gồm bảng T1.x trước / sau, `test_ticket`, `selftest`, `test_pipeline_modes`. `vaudit/test_k1.py` thiếu từ bộ gửi; ghi rõ là lỗi có sẵn.
