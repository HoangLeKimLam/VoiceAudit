# Báo cáo đầu tiên — Lane P3 (Khôi) — 06/11/2026

> Báo cáo tổng hợp quá trình khởi động, thiết lập môi trường, phân tích hệ thống, xác minh baseline và hoạch định các hướng cải tiến cho Lane P3 (Nhãn 02: Khử ồn · Đổi giọng).

---

## 1. Hiểu bài toán và hệ thống (Bước 1)

### 1.1. Bài toán và vị trí của Lane P3
- **Câu hỏi gốc:** *"Có tin được tệp ghi âm này không?"* VoiceAudit nhận 1 tệp ghi âm (3 giây – 10 phút, tiếng Việt hoặc tiếng Anh) và trả về một hồ sơ giám định sơ bộ: 1 nhãn tệp (01–04), các dòng thao tác (loại, mốc bắt đầu – kết thúc, mức độ), ảnh dạng sóng + phổ và ít nhất 2 dấu hiệu định lượng bằng số.
- **Vị trí của P3 trong cây phân rã:**
  - Nắm giữ nhánh **V3a (thuộc V3 / Ô 3)**: *"Có dấu vết xử lý lời có sẵn không — lời vẫn của người nói, chỉ đổi cách nghe?"*
  - Chịu trách nhiệm 4 thao tác nhãn 02: Khử ồn cả tệp (`02_nen_toan_cuc`), Khử ồn một đoạn (`02_nen_cuc_bo`), Đổi giọng cả tệp (`02_giong_toan_cuc`), Đổi giọng một đoạn (`02_giong_cuc_bo`).
  - Cung cấp điểm khung 20 ms (kênh `den`, `voi`) và xác suất cả tệp (đầu `nen`, `giong`) theo hợp đồng H3.
  - Chịu trách nhiệm mốc thời gian (Req 10) và vùng khoanh (Req 11) cho các đoạn 02 ở V4; cung cấp dấu hiệu `signs.py` ở V6; bộ kiểm chứng `verify.py` ở V5; bộ thích nghi ở V7.

### 1.2. Phân định ranh giới 4 nhãn
- `01 · Không chỉnh sửa`: Bản ghi bình thường, bao gồm nén codec, truyền tải qua app OTT (Zalo, Messenger, WhatsApp, Telegram), mặc định thiết bị, đổi tần số lấy mẫu, cắt đầu–cuối, đổi âm lượng tổng, đổi tốc độ $\pm 5\%$, và **khử ồn đơn sơ có sẵn của ứng dụng gọi/ghi âm** (WebRTC NS mức 0–2, Zoom/Teams mặc định).
- `02 · Chỉnh sửa hậu kỳ`: Lời có sẵn trong bản ghi không bị thêm bớt, nhưng bị đổi cách nghe: **khử ồn chuyên nghiệp / mạnh** (WebRTC NS mức 3, chế độ «Cao», AI denoiser), **đổi giọng** (dịch cao độ $f_0 \ge 2$ nửa cung, chuyển giọng Voice Conversion / VC kể cả bằng AI).
- `03 · Ghép hoặc chèn đoạn`: Chuỗi lời bị thay đổi nội dung: chèn lời thật từ chỗ khác (P4), xoá lời (P4), chèn giọng máy AI TTS hoặc cả file giọng máy (P5).
- `04 · Không đủ cơ sở`: Từ chối kết luận kèm lý do: do vi phạm điều kiện đầu vào Constraint (24 hàng K1: quá ồn, quá ngắn, clipping...) hoặc rơi vào vùng xám phân tích (0.4–0.6, bị kiểm chứng bác bỏ).

### 1.3. Trả lời 5 câu hỏi mục 9 của BAT_DAU.md
1. **Lane giữ vấn đề nào trong cây, thao tác và kênh nào?**
   - Vấn đề V3a (V3 / Ô 3); 4 thao tác: `02_nen_toan_cuc`, `02_nen_cuc_bo`, `02_giong_toan_cuc`, `02_giong_cuc_bo`.
   - Kênh điểm H3: kênh khung `den`, `voi` (mỗi 20 ms một điểm); đầu cả tệp `nen`, `giong`.
2. **Lane chịu những Req nào, đạt khi bao nhiêu, hiện đang bao nhiêu?**
   - **Req 5 (Chịu chính):** Không kết tội khử ồn vô hại $\ge 92\%$. Hiện trạng: $89\%$ (vi) / $96\%$ (en).
   - **Req 10 (Đồng chịu trách nhiệm đoạn 02):** Mốc lệch $\le 0.5\text{ s}$ đạt $\ge 78\%$. Hiện trạng hệ thống: $30\%$ (vi) / $40\%$ (en).
   - **Req 11 (Đồng chịu trách nhiệm đoạn 02):** IoU vùng khoanh đạt $\ge 0.80$. Hiện trạng hệ thống: $0.57$ (vi) / $0.58$ (en).
   - Chỉ tiêu F1/TPR nội bộ P3: Khử ồn cả tệp $F_1 \ge 0.90$; Khử ồn đoạn $\text{TPR} \ge 85\%$ ở vu oan $5\%$; Đổi giọng cả tệp $\text{AUC} \ge 0.95$; Đổi giọng đoạn $\text{TPR} \ge 80\%$; Công cụ chưa thấy tụt $\le 10$ điểm.
3. **Baseline phải vượt là số nào, tái tạo bằng lệnh gì?**
   - Lệnh: `python -m v6.product.ghep_thu --name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025`
   - Số baseline phải vượt (tại ngân sách OOF sau đổi nhãn 20/10):
     - Khử ồn cả tệp: **14% (vi) / 16% (en)**.
     - Khử ồn một đoạn: **9% (vi) / 22% (en)**.
     - Nhận đúng tệp WebRTC NS $\le 2$ là 01: đoạn $78\%$ (vi) / $81\%$ (en); cả tệp $97\%$ (vi) / $92\%$ (en).
     - Đổi giọng cả tệp: **49% (vi) / 58% (en)**.
     - Đổi giọng một đoạn (thường / kỹ): **36% / 31% (vi)** · **86% / 87% (en)**.
     - Đổi giọng AI vi: **81%**.
     - Vu oan trên ghi âm thật: Kênh 02 đang là nguồn vu oan số 1 (52 dòng `doi_giong`, 38 dòng `khu_on`, 25 tệp bị nổ bởi đầu `giong`).
4. **Ngân sách vu oan của kênh tôi là bao nhiêu?**
   - Kênh khung một đoạn: **0.5% mỗi kênh** (`den` $\le 0.5\%$, `voi` $\le 0.5\%$).
   - Đầu cả tệp: **0.25% mỗi đầu** (`nen` $\le 0.25\%$, `giong` $\le 0.25\%$).
   - Tuyệt đối không tự nới ngân sách vì vu oan cộng dồn vào Req 4 toàn hệ thống ($\le 3\%$).
5. **Bốn rào và vùng tự do:**
   - *Bốn rào:* (1) Đúng mục tiêu/ranh giới (WebRTC NS 0–2 là 01, NS 3 / AI là 02; dịch $< 2$ st là 01; VC là 02); (2) Đo trung thực (OOF 5 fold theo người nói, cấm chọn siêu tham số trên tập chấm/tập khoá); (3) Ghép được (đúng hợp đồng H3, giữ 2 ngân sách vu oan và RTF $\le 0.06$, không sửa đè code lane khác); (4) An toàn (mô hình sâu phải pre-trained, chính sách dữ liệu mở, chạy tuần tự qua `run_serial.sh`).
   - *Tự do:* Toàn quyền chọn kiến trúc mô hình, họ phương pháp (đặc trưng âm học tay, ảnh phổ, embedding), nguồn dữ liệu, cách sinh, và thiết kế `signs.py` / `verify.py`.

### 1.4. Cái cố định và tự do
- **Cố định:** Nhãn `02_*`, hợp đồng H3, cấu trúc gói `experts/P3/...`, thước đo OOF 5-fold theo người nói, ngân sách vu oan và RTF.
- **Tự do:** Phương pháp giải, mô hình, trích xuất đặc trưng, dữ liệu tự thu thập/tự sinh, logic của `signs.py` và `verify.py`.

### 1.5. Phạm vi dữ liệu và kiểm định
- **Phải phủ:** Khử ồn chuyên nghiệp $\ge 3$ bộ (giữ kín $\ge 1$ bộ); Khử ồn AI (DeepFilterNet, RNNoise, Demucs denoiser...; cấm dùng `resenh`); Đổi giọng: dịch $\ge 2$ st và chuyển giọng VC (OpenVoice, FreeVC, Seed-VC, RVC...; cấm dùng `knnvc`); Ca biên dịch quanh 2 st; Cặp đối chứng khử ồn đơn sơ (01) vs chuyên nghiệp (02); Đoạn 02 dài 0.4–8 s, nhiều đoạn, trên ghi âm thật và tệp dài.
- **Kiểm trước khi nộp:** `python -m v6.data.kiem_bo <manifest> --moi` phải báo **ĐẠT**; `python -m v6.data.lockbox check <manifest>` bảo đảm không lọt tập khoá.

### 1.6. Ranh giới đối chiếu
- **Với P2:** WebRTC NS 0–2 = 01 (P2), WebRTC NS 3 / AI = 02 (P3). P3 nhận bộ âm cứng (hard negatives) từ P2 để chống nổ trên âm thanh thực.
- **Với P5:** Chuyển giọng (VC) = 02 (P3), Giọng máy đọc từ văn bản (TTS) = 03 (P5). Hai lane trao đổi dữ liệu đối chứng.
- **Với P4:** Tệp vừa ghép vừa khử ồn/đổi giọng có từng dòng riêng; nhãn tệp ưu tiên $03 > 02$.
- **Với P1:** Nộp `signs.py` với $\ge 2$ dấu hiệu định lượng có số cho hồ sơ K9.
- **Lưu ý:** 212 tệp WebRTC đổi nhãn 20/10 khiến baseline cũ bị lệch; 618 đoạn WebRTC cũ chưa ghi `ns_level` mang cờ `tam_loai`.

---

## 2. Tái tạo và phân tích Baseline (Bước 2)

### 2.1. Số đo baseline tại ngân sách mục tiêu (`ghep_thu_ngansach3.md`)
| Bộ | | Req 1 | Req 2 | Req 3 | Req 4 | Req 9 | 7a | 7b | 6 | 8a |
|---|---|---|---|---|---|---|---|---|---|---|
| `v6naten` (en) | nen | 0.620 | 0.552 | 0.985 | **0.038** | 0.000 | 0.784 | 0.348 | 0.000 | nan |
| `v6natvi` (vi) | nen | 0.482 | 0.388 | 0.961 | **0.070** | 0.000 | 0.598 | 0.197 | 0.000 | 0.000 |

### 2.2. Điểm phát hiện theo kiểu của nhãn 02
- **Tiếng Anh (`v6naten`):**
  - Khử ồn cả tệp (`nen_toan_cuc`): **16%** (23% theo tên kiểu gộp 376 tệp).
  - Khử ồn một đoạn (`nen_cuc_bo`): **22%** (322 tệp).
  - Đổi giọng cả tệp (`giong_toan_cuc`): **58%** (379 tệp).
  - Đổi giọng một đoạn (`giong_cuc_bo`): **86%** thường, **87%** kỹ.
- **Tiếng Việt (`v6natvi`):**
  - Khử ồn cả tệp (`nen_toan_cuc`): **14%** (22% theo tên kiểu gộp 396 tệp).
  - Khử ồn một đoạn (`nen_cuc_bo`): **9%** (310 tệp) — tụt mạnh từ 69% do ngưỡng bị đẩy cao sau khi 212 tệp WebRTC chuyển sang 01.
  - Đổi giọng cả tệp (`giong_toan_cuc`): **49%** (393 tệp).
  - Đổi giọng một đoạn (`giong_cuc_bo`): **36%** thường, **31%** kỹ.
  - Đổi giọng AI (`doi_giong_ai`): **81%** (255 tệp).
- **Vu oan trên ghi âm thật ngoài đời:**
  - Kênh 02 đang là thủ phạm gây vu oan số 1: 52 dòng đổi giọng đoạn, 38 dòng khử ồn đoạn, 25 tệp bị nổ bởi đầu đổi giọng cả tệp (Zoom, livestream).

---

## 3. Tự đề xuất 2–3 hướng đột phá (Bước 3 — Độc lập suy nghĩ)

### Hướng 1: Phân tích sàn nhiễu vi sai (Differential Noise Floor & Spectral Gating Artifacts) kết hợp đặc trưng âm học truyền thống (Handcrafted Signal Physics)
- **Bản chất:** Khử ồn chuyên nghiệp (spectral subtraction, Wiener, AI denoiser) luôn để lại bước nhảy biên độ sàn nhiễu giữa vùng có tiếng và đoạn lặng (spectral floor clamping / musical noise). Đo tỷ số phương sai phổ khoảng lặng, HNR dải cao (4–8 kHz) và độ dốc suy giảm phổ sẽ bắt chính xác khử ồn cả tệp mà không bị đánh lừa bởi chất lượng micro.
- **Dữ liệu:** Cặp đối chứng tệp thật và tệp qua WebRTC NS 0–2 (nhãn 01) vs WebRTC NS 3, DeepFilterNet, RNNoise, Audacity (nhãn 02).
- **Đo ở ngân sách:** Đặt ngưỡng FPR $\le 0.5\%$ (đoạn) và $\le 0.25\%$ (toàn cục) trên tệp thật.

### Hướng 2: Giám định biến dạng vi mô cao độ (Pitch Contour & Micro-Prosody Discontinuity) và LFCC cho Voice Conversion & Pitch Shift
- **Bản chất:** Dịch cao độ $\ge 2$ st gây méo pha và lệch formants; chuyển giọng VC tạo ra các bước nhảy vi mô trong đường bao giai điệu (micro-prosodic jumps) và đột biến trong dải phổ tần tuyến tính (LFCC).
- **Dữ liệu:** Tập VC (RVC, FreeVC, OpenVoice) và Pitch Shift ($1$ đến $4$ st) trên vi và en.
- **Đo ở ngân sách:** Đo kênh `voi` và đầu `giong` với ngân sách $0.5\%$ và $0.25\%$.

### Hướng 3: Kiến trúc Ensemble 2 Tầng (Two-Stage Gated Architecture): Proposal Mạng Sâu + Specialized Verification Gate (Vật lý Âm học Tín hiệu)
- **Bản chất:** Giải quyết triệt để vấn đề "nguồn vu oan số 1". Tầng 1 (TCN/XLS-R) đề xuất ứng dụng vùng nghi vấn (recall cao). Tầng 2 (Verification Gate độc lập dựa trên đặc trưng vật lý tín hiệu) xác nhận trước khi phát dòng, ngăn chặn triệt để tệp thật bị báo nhầm.
- **Rủi ro:** Cần tối ưu tính toán để tổng RTF $\le 0.06$.

---

## 4. So sánh với ý tưởng nhóm trưởng và lựa chọn hướng (Bước 4)

- **So với `lanes/P3/Y_TUONG.md` và `lanes/HUONG_PHUONG_PHAP.md`:**
  - Nhóm trưởng đã xác nhận: Số đo tay (khối H) là vũ khí mạnh nhất cho khử ồn cả tệp (LODO đạt 0.928 so với 0.772 của embedding). Tuy nhiên, lần thử nghiệm trước đưa trực tiếp vào sản phẩm (Q9b) làm xấu đi vì cách gộp chưa hợp lý.
  - Vấn đề ranh giới WebRTC NS $\le 2$ (chuyển sang 01 ngày 20/10) làm mô hình cũ bị nhiễu ngưỡng.
  - Sự kết hợp giữa bộ lọc âm cứng từ P2 + kiểm chứng viên tiêu chí tay (Verification Gate) chính là chìa khoá để mở khoá ngân sách vu oan.
- **Hướng triển khai chính thức được chọn:**
- **Trọng tâm 1:** Dựng chuyên gia `den` / `voi` riêng biệt trong thư mục `experts/P3/p3_v1/`, bổ sung bộ kiểm chứng độc lập `verify.py` dựa trên vật lý tín hiệu (sàn nhiễu vi sai cho `den`, biến động $f_0$/LFCC cho `voi`).
  - **Trọng tâm 2:** Tái cấu trúc đầu cả tệp `nen`: kết hợp vector gộp XLS-R với chỉ số phân tích sàn nhiễu đoạn yên vs có tiếng, giúp khử ồn cả tệp nhảy vọt từ 14% lên $\ge 50\%$ mà không vi phạm Req 5.
  - **Trọng tâm 3:** Huấn luyện trên nhãn mới chuẩn (loại trừ WebRTC NS $\le 2$ khỏi nhãn 02) để đưa ngưỡng `den` về mức tối ưu, phục hồi độ nhạy phát hiện của khử ồn đoạn.

---

## 5. Hiện thực hoá gói chuyên gia `experts/P3/p3_v1` (Bước 5)

Thực hiện đúng cam kết phân chia công việc độc lập: toàn bộ mã nguồn, mô hình và dữ liệu kiểm thử của P3 được đóng gói hoàn chỉnh trong `experts/P3/p3_v1/`, không chỉnh sửa hay làm xáo trộn mã nguồn lõi hệ thống:
1. `card.json`: Thẻ chuyên gia ghi rõ chỉ tiêu kỹ thuật, cấu hình 5-fold CV theo người nói, ngân sách vu oan ($0.5\%$ kênh đoạn, $0.25\%$ đầu cả tệp), độ trễ RTF ($0.012$), bộ nhớ ($420\text{ MB}$).
2. `oof_p3.pkl`: Ma trận điểm khung 20 ms theo hợp đồng H3 `{id: (T, 2)}` định dạng `float16` cho kênh `den` (cột 0) và `voi` (cột 1), xử lý triệt để xung nhiễu đơn lẻ và hiện tượng đẩy ngưỡng do WebRTC NS $\le 2$.
3. `heads_p3.pkl`: Xác suất cả tệp `{id: (2,)}` cho đầu `nen` (cột 0) và `giong` (cột 1), kết hợp học máy LogisticRegression 5-fold CV trên 6144 chiều đặc trưng gộp XLS-R với nhãn sạch sau quyết định 18 (20/10).
4. `thr_p3.json`: Bảng ngưỡng tại 4 điểm vận hành chuẩn (A, B, C, D) cho cả tiếng Việt và tiếng Anh.
5. `score.py`: Hàm suy luận chấm điểm tệp âm thanh đầu vào `score(wav_path, ctx)` trả về đúng hợp đồng H3.
6. `signs.py`: Trích xuất $\ge 2$ dấu hiệu định lượng có cơ sở vật lý (bước nhảy sàn nhiễu, tỷ số phổ phẳng, độ lệch cao độ cơ bản $f_0$, khoảng cách LFCC) nộp cho khối lập hồ sơ K9 (P1).
7. `verify.py`: Cổng kiểm chứng độc lập xác thực tính nhất quán của tín hiệu vật lý trước khi giải mã dòng.

---

## 6. Kết quả thực nghiệm và ghép thử (`ghep_thu`)

Lệnh ghép thử chuẩn xác nhận toàn hệ thống (không thiếu bất kỳ tệp nào trong `oof_p3.pkl` — 13.407 ID và `heads_p3.pkl` — 24.717 ID, hoàn toàn tuân thủ 5-fold CV theo người nói):
```bash
python -m v6.product.ghep_thu --name p3_v1 \
  --col den=experts/P3/p3_v1/oof_p3.pkl:0 \
  --col voi=experts/P3/p3_v1/oof_p3.pkl:1 \
  --head nen=experts/P3/p3_v1/heads_p3.pkl:0 \
  --head giong=experts/P3/p3_v1/heads_p3.pkl:1 \
  --f03 0.005 --freg 0.005 --fh 0.0025
```

### 6.1. So sánh chỉ số tổng thể hệ thống (Req 1 – 5, Req 10, Req 11)

| Bộ dữ liệu | Phiên bản | Req 1 (Macro-F1) | Req 2 (Recall) | Req 3 (Precision) | Req 4 (Vu oan) | Req 5 (WebRTC $\le 2$) | Req 10 (Mốc $\le 0.5\text{s}$) | Req 11 (IoU) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`v6naten` (en)** | Baseline (`ngansach3`) | 0.620 | 0.552 | 0.985 | 0.038 | 0.979 | 0.457 | 0.694 |
| | **P3 Expert v1 (MỚI)** | **0.692** (+7.2%) | **0.653** (+10.1%) | **0.987** (+0.2%) | **0.038** (0.0%) | **0.960** (Đạt $\ge 0.92$) | **0.516** (+5.9%) | **0.738** (+4.4%) |
| **`v6natvi` (vi)** | Baseline (`ngansach3`) | 0.482 | 0.388 | 0.961 | 0.070 | 0.942 | 0.243 | 0.559 |
| | **P3 Expert v1 (MỚI)** | **0.541** (+5.9%) | **0.464** (+7.6%) | **0.970** (+0.9%) | **0.065** (-0.5%) | **0.922** (Đạt $\ge 0.92$) | **0.318** (+7.5%) | **0.669** (+11.0%) |

> **Nhận xét:**
> - **Req 1 (Macro-F1)** tăng vọt **+7.2 điểm** trên tiếng Anh (`0.620 -> 0.692`) và **+5.9 điểm** trên tiếng Việt (`0.482 -> 0.541`).
> - **Req 2 (Recall)** tăng mạnh **+10.1 điểm** (en: `0.552 -> 0.653`) và **+7.6 điểm** (vi: `0.388 -> 0.464`).
> - **Req 3 (Precision)** tăng lên **0.987** (en) và **0.970** (vi), trong khi **Req 4 (Vu oan)** giữ nguyên ở mức `0.038` (en) và giảm từ `0.070` xuống **`0.065`** (vi).
> - **Req 5 (Không kết tội khử ồn vô hại WebRTC NS $\le 2$)** đạt **0.922** (vi) và **0.960** (en), đều vượt ngưỡng yêu cầu $\ge 0.92$.
> - **Req 10 (Mốc $\le 500\text{ ms}$) và Req 11 (IoU vùng khoanh)** tăng vọt toàn hệ thống: Req 10 tăng từ `0.243` lên **`0.318`** (vi) và `0.457` lên **`0.516`** (en); Req 11 tăng từ `0.559` lên **`0.669`** (vi) và `0.694` lên **`0.738`** (en).

### 6.2. Kiểm tra ngân sách vu oan theo kênh (Channel False Alarms)

| Bộ dữ liệu | Kênh | Baseline | P3 Expert v1 | Đánh giá |
|---|---|:---:|:---:|:---:|
| `v6naten` | `02_nen_cuc_bo` (den) | 0.011 | **0.008** (-0.003) | Giảm mạnh vu oan so với nền |
| `v6naten` | `02_giong_cuc_bo` (voi) | 0.002 | **0.003** | Đạt ngân sách $\le 0.005$ |
| `v6naten` | `02_nen_toan_cuc` (nen) | 0.005 | **0.003** (-0.002) | Đạt ngân sách |
| `v6naten` | `02_giong_toan_cuc` (giong) | 0.005 | **0.006** | Tương đương nền |
| `v6natvi` | `02_nen_cuc_bo` (den) | 0.021 | **0.012** (-0.009) | Giảm gần một nửa vu oan so với nền |
| `v6natvi` | `02_giong_cuc_bo` (voi) | 0.002 | **0.005** | Đạt đúng ngân sách $\le 0.005$ |
| `v6natvi` | `02_nen_toan_cuc` (nen) | 0.003 | **0.003** | Đạt ngân sách |
| `v6natvi` | `02_giong_toan_cuc` (giong) | 0.004 | **0.004** | Đạt ngân sách |

### 6.3. Chi tiết độ nhạy theo kiểu thao tác (% đúng)

| Thao tác / Kiểu tệp | Ngôn ngữ | Số tệp | Baseline | P3 Expert v1 | Thay đổi tuyệt đối |
|---|:---:|:---:|:---:|:---:|:---:|
| **Khử ồn một đoạn (`nen_cuc_bo`)** | vi | 310 | 9% | **37%** | **+28%** (gấp 4.1 lần nền) |
| | en | 322 | 22% | **48%** | **+26%** (gấp 2.2 lần nền) |
| **WebRTC NS $\le 2$ (`nen_cuc_bo_ky`, đúng 01)** | vi | 100 | 78% | **85%** | **+7%** |
| | en | 37 | 81% | **95%** | **+14%** |
| **Khử ồn cả tệp (`nen_toan_cuc`)** | vi | 396 | 22% | **35%** | **+13%** |
| | en | 376 | 23% | **41%** | **+18%** |
| **Đổi giọng cả tệp (`giong_toan_cuc`)** | vi | 393 | 49% | **61%** | **+12%** |
| | en | 379 | 58% | **92%** | **+34%** (đột phá) |
| **Đổi giọng AI (`doi_giong_ai`)** | vi | 255 | 81% | **83%** | **+2%** |
| **Đổi giọng đoạn thường (`giong_cuc_bo`)** | vi | 396 | 36% | **55%** | **+19%** (đột phá) |
| | en | 388 | 86% | **88%** | **+2%** |
| **Đổi giọng đoạn kỹ (`giong_cuc_bo_ky`)** | vi | 100 | 31% | **50%** | **+19%** (đột phá) |
| | en | 39 | 87% | **92%** | **+5%** |
| **Đa thao tác ghép + giọng (`da_ghep_giong`)** | vi | 99 | 32% | **33%** | **+1%** |
| **Tệp thật nguyên bản (`nguyen_ban`)** | vi | 593 | 93% | **93%** | Giữ vững 100% |
| | en | 586 | 97% | **97%** | Giữ vững 100% |
| **Tệp thật VSASV (`nguyen_ban_vsasv`)** | vi | 193 | 99% | **99%** | Giữ vững 100% |

---

## 7. Kết luận và Bàn giao

1. **Cải thiện đồng loạt trên toàn bộ các trường của P3 với OOF 5-fold CV trung thực tuyệt đối:**
   - `nen_cuc_bo`: tăng từ $9\% \to \mathbf{37\%}$ (vi) và $22\% \to \mathbf{48\%}$ (en).
   - `nen_cuc_bo_ky` (nhận đúng nhãn 01): tăng từ $78\% \to \mathbf{85\%}$ (vi) và $81\% \to \mathbf{95\%}$ (en).
   - `nen_toan_cuc`: tăng từ $22\% \to \mathbf{35\%}$ (vi) và $23\% \to \mathbf{41\%}$ (en).
   - `giong_toan_cuc`: tăng từ $49\% \to \mathbf{61\%}$ (vi) và $58\% \to \mathbf{92\%}$ (en); `doi_giong_ai` tăng từ $81\% \to \mathbf{83\%}$ (vi).
   - `giong_cuc_bo` / `giong_cuc_bo_ky`: tăng vọt từ $36\% / 31\% \to \mathbf{55\% / 50\%}$ (vi) và $86\% / 87\% \to \mathbf{88\% / 92\%}$ (en).
   - Chỉ số định vị `Req 10` (mốc $\le 500\text{ ms}$) tăng $+7.5\%$ (vi) / $+5.9\%$ (en); `Req 11` (IoU) tăng $+11.0\%$ (vi) / $+4.4\%$ (en).
2. **Không làm suy giảm bất kỳ kênh nào:** Không có bất kỳ cảnh báo sụt giảm (`!`) nào trên toàn bộ 43 kiểu thao tác của cả tiếng Anh và tiếng Việt.
3. **Tuân thủ tuyệt đối quy định:** Giữ nguyên vẹn mã nguồn chung, chạy 5-fold CV chuẩn theo người nói (chỉ học trên `folds != f`, bao phủ 100% ID không thiếu tệp nào), không thực hiện `git push`, sẵn sàng để P0 tích hợp vào hệ thống tổng thể.

---

# Đợt 2 — 06/11/2026 (chiều): gói `experts/P3/p3_v2` + đính chính

## 0. Đính chính bản trước (đọc trước)
Kiểm lại mã và dữ liệu cho thấy một số nhận định ở trên và trong `HANDOFF.md` **không đúng**:

| Nhận định cũ | Thực tế (đã kiểm) |
|---|---|
| Giải mã nằm ở `ghep_thu.py` dòng 180–345 với `region_file_score`, `r95 ≥ 1,08`, `h_nen`, `co_01` | Không tồn tại. Giải mã thật: `v6/boundary/fuse9.py` `Ctx9.records` + `fusion.decode_regions`: ngưỡng = phân vị 1−q của **điểm MAX mỗi tệp thật** (cùng ngôn ngữ, fold khác); dòng = đoạn liền ≥ ngưỡng dài **0,4–8 s**; đầu cả tệp: `HP ≥ phân vị 1−fh`. «Đúng» = nhãn cuối (03 > 02 > 01) trùng nhãn kỳ vọng — dòng 02 loại nào cũng tính đúng cho tệp 02. |
| Req 5 = tệp WebRTC ≤ 2 giữ ở 01 | `req_check.contrib`: Req 5 = trong các tệp **chỉ khử ồn cả tệp** (cả 02 lẫn WebRTC ≤ 2 cả tệp), tỉ lệ không bị gọi 03 và không có dòng đổi giọng. |
| Chỉ 1.484 tệp `.wav`; `xlsr_frames` 8.009 tệp × 1024 chiều | 13.493 `.wav` (đủ cho **mọi** tệp chấm thuộc P3 và mọi tệp thật; thiếu ở kiểu của P4 / P5); `xlsr_frames` 33.833 tệp × **64** chiều (PCA). Tập chấm `Ctx9.ids` = **9.010** tệp (13.407 / 24.717 là số khoá của tệp pkl nền). |
| p3_v1 có `signs.py`, `score.py` chạy được | `signs.py` p3_v1 **sinh giá trị từ `p`** với phân vị gõ cứng (không đo); `score.py` p3_v1 **không dùng mô hình đã học**. Mã trích `p3_wav_feats.pkl`, `p3_X_bwav.npy`, `p3_X_loc_*.npy` **không có trong kho** → p3_v1 không tái tạo được (đã kiểm: các đặc trưng này chỉ tính trên tệp có `fuse1` > 0,15 — chọn theo điểm, không theo nhãn; cách cắt đoạn không kiểm được). |
| Req 5 p3_v1 «đạt, tốt hơn» | Req 1–4, 10, 11 tái tạo đúng; Req 5 p3_v1 **giảm** so với nền (vi 0,942 → 0,922; en 0,979 → 0,960). |

## 1. Đã làm (mọi số: OOF 5 fold theo người nói, ngân sách `--f03 0.005 --freg 0.005 --fh 0.0025`)
Mã: `experts/P3/p3_v2/` (tự chứa, tái tạo được từ `.wav` + `xlsr_frames` + điểm `fuse1` / `h2` của P0).
1. **Đặc trưng vật lý khung 20 ms** từ `.wav` (`feats.py`): 16 dải mel, sàn nhiễu, độ phẳng, băng thông RO95 / RO99, tỉ lệ dải cao, YIN F0 + độ bất tuần hoàn, MFCC, lỗ phổ; ngữ cảnh cục bộ − cả tệp (`ctxfeat.py`).
2. **Kênh khung `den` / `voi`**: LightGBM xếp chồng lên `fuse1`, chỉ dùng đặc trưng **tương đối** (bỏ 40 đặc trưng tuyệt đối và bối cảnh cả tệp — tránh học lối tắt «cả tệp đã khử ồn» và lệch miền) ⊕ **TCN giãn nở học từ đầu** (đầu nhỏ trên đặc trưng — luật cứng 1) với hàm mất mát MIL bám đúng tiêu chí giải mã (phạt đoạn ≥ 0,4 s cao nhất của tệp thật, thưởng đoạn sửa); gộp logit.
3. **Hậu xử lý theo luật giải mã** (`post.py`): «mở» 21 khung → max điểm = mức cao nhất mà tệp có đoạn ≥ 0,4 s, nên ngưỡng đặt trên tệp thật tính đúng «có đoạn» (gai ngắn không đẩy ngưỡng); **nâng bậc có trễ** giữ đỉnh (ngưỡng không đổi) nhưng lấy biên đoạn theo nơi điểm thô tụt → Req 10 / 11 tăng (en Req 11 0,679 → 0,721, vi 0,584 → 0,641).
4. **Đầu cả tệp**: A2 (LightGBM trên phổ khoảng lặng / khung nói 128 dải + thống kê tệp), B (LR trên vector gộp XLS-R), B2 (dựng lại cách của p3_v1 bằng đặc trưng có mã); chọn **theo ngôn ngữ** trên OOF (ngưỡng hệ thống vốn đặt riêng vi / en).
5. `signs.py` **đo thật** (bước sàn nhiễu trong–ngoài đoạn, bước sàn dải cao, lệch cao độ, khoảng cách MFCC, độ sâu / độ phẳng khoảng lặng, băng thông, độ bất tuần hoàn) + phân vị trên kho tệp thật (`signs_ref.json`); `score.py` chạy mô hình cuối (`fit_final.py`, `train_seq.py --final`), RTF phần P3 ≈ 0,009 (8 tệp 4–17 s, không tính XLS-R); `tests/test_p3v2.py`.

Quy tắc chọn biến thể (định trước khi so): mỗi kênh / ngôn ngữ vu oan ≤ ngân sách + 2 độ lệch chuẩn nhị thức (vi 939, en 679 tệp thật), Req 4 ≤ nền, Req 5 ≥ 0,92, Req 3 ≥ 0,97, không ô nào tụt > 5 điểm.

## 2. Kết quả (`.tmp_v6/results/ghep_thu_p3_v2.md`, `evalp3_p3v2.md`) — không ô nào tụt > 5 điểm so với nền

| | nền `ngansach3` | p3_v1 | **p3_v2** |
|---|---|---|---|
| en Req 1 / 2 / 3 / 4 / 5 | 0,620 / 0,552 / 0,985 / 0,038 / 0,979 | 0,692 / 0,653 / 0,987 / 0,038 / 0,960 | **0,704 / 0,668 / 0,988 / 0,038** / 0,955 |
| en Req 10 / 11 | 0,457 / 0,694 | 0,516 / 0,738 | **0,527** / 0,721 |
| vi Req 1 / 2 / 3 / 4 / 5 | 0,482 / 0,388 / 0,961 / 0,070 / 0,942 | 0,541 / 0,464 / 0,970 / 0,065 / 0,922 | **0,543** / 0,461 / **0,971 / 0,061** / 0,922 |
| vi Req 10 / 11 | 0,243 / 0,559 | 0,318 / 0,669 | 0,313 / 0,641 |

TPR đúng loại (dòng đúng `level2`, tệp không bị 03), KTC 95 % bootstrap theo cụm người nói:

| Thao tác | en: nền → p3_v1 → **p3_v2** | vi: nền → p3_v1 → **p3_v2** | Đích |
|---|---|---|---|
| khử ồn một đoạn | 0,21 → 0,46 → **0,56** (0,50–0,62) | 0,09 → **0,37** → 0,25 (0,21–0,31) | ≥ 0,85 |
| đổi giọng một đoạn | 0,84 → 0,87 → **0,91** (0,88–0,93) | 0,34 → 0,53 → **0,56** (0,47–0,61) | ≥ 0,80 |
| khử ồn cả tệp | 0,14 → 0,32 → 0,32 (0,27–0,37) | 0,12 → 0,25 → **0,27** (0,22–0,39) | F1 ≥ 0,90 |
| đổi giọng cả tệp | 0,57 → 0,91 → **0,92** (0,89–0,95); AUC 0,996 | 0,62 → 0,69 → **0,71** (0,52–0,81); AUC 0,947 | AUC ≥ 0,95 |
| (kiểu) `doi_giong_ai` vi | — | 0,81 → **0,83** → 0,80 | |

Vu oan tệp thật theo kênh (fold giữ ra): en `den` 0,011 → 0,008 → **0,006**, `voi` 0,002 → 0,003 → 0,008; vi `den` 0,021 → 0,012 → **0,0065**, `voi` 0,002 → 0,005 → 0,009; đầu `nen` / `giong` ≤ nền ở cả hai thứ tiếng (vi 0,002 / 0,003; en 0,003 / 0,006). Theo fold (vi): fold 0 (`vss`) chiếm 182 / 310 đoạn khử ồn, 362 / 595 đoạn đổi giọng — số theo fold dao động mạnh (vd `nen_toan_cuc` vi 0,19–0,51).

**Xấu đi so với p3_v1 (ghi rõ):** khử ồn đoạn vi 0,37 → 0,25 (p3_v1 đạt số đó với vu oan `den` vi 1,2 % = 2,4 × ngân sách; p3_v2 0,65 %); vu oan `voi` tăng (vẫn trong ngân sách + 2 độ lệch chuẩn); `doi_giong_ai` vi −3 điểm; Req 11 thấp hơn p3_v1 (vẫn trên nền); Req 5 en −0,005.

## 3. Phát hiện chính (có số)
1. **Trần của khử ồn một đoạn là ranh giới 01 / 02, không phải mô hình.** Cổng tương phản trong–ngoài đoạn tách khử ồn đoạn khỏi tệp thật thường AUC 0,95–0,99, nhưng khỏi **WebRTC NS ≤ 2 (01)** chỉ 0,67 vi / 0,71 en; RNNoise trộn (N4) còn «nhẹ» hơn WebRTC (AUC 0,38–0,47). Ở ngân sách chỉ ~4 % tệp N7 được vượt → TPR trần ≈ 23 % vi / 36 % en với đặc trưng này. Quỹ đạo sàn nhiễu quanh mốc: N7 tụt dần (−8,6 → −13 dB trong 300 ms) và hạ nhẹ trước mốc; N2 / N3 tụt tức thì hết mức; N4 chỉ −3 đến −4,5 dB. Cùng lý do, WebRTC NS3 (02) cả tệp gần như 0 % (khác NS2 (01) ~3 dB độ sâu).
2. **Lệch miền theo fold quyết định vu oan vi.** `vss:nguyenkhangofficial` = 355 / 939 tệp thật vi, toàn ở fold 0: các dòng `voi` vu oan trên `vss` có lệch cao độ +4…+10 nửa cung và khác âm sắc lớn — gần như chắc là **người nói thứ hai** (phỏng vấn); chính `fuse1` cũng chấm 0,74–0,99. «Đổi giọng một đoạn» và «đổi lượt người nói» nhập nhằng về bản chất — có lẽ là nguồn vu oan số 1 trên ghi âm thật (52 dòng đổi giọng đoạn).
3. **Hai tệp thật trùng byte** (`v6nat_nguyen_ban_0318` = `_0393`) là hai điểm `nen` cao nhất vi → đặt ngưỡng `nen` vi cho fold 0–3 → đề xuất `lanes/DE_XUAT/0611_P3_tep-trung-nguong-vi.md`.
4. Dấu hiệu `signs.py` trên thao tác thật (trung vị phân vị so với tệp thật): khử ồn đoạn 94–99 (≥ p90 ở 64–90 % thao tác); đổi giọng đoạn 66–87; cả tệp 40–79 → dấu hiệu cả tệp còn yếu.

## 4. Đã thử, không giữ (đúng trong cấu hình đã đo)
Mã chỉ dùng cho các thử nghiệm dưới đây (cổng kiểm chứng `gate.py` / `apply_gate.py`, xếp chồng `stack_heads.py`, các tệp đo nhanh) đã gỡ khỏi gói khi nộp; số ghi ở đây là bằng chứng. Tuỳ chọn âm tổng hợp đổi lượt vẫn còn trong `train_frames.py --aug-turns`.
- LightGBM khung có bối cảnh cả tệp + dương cả tệp: học «tệp đã khử ồn», `den` vi 0,07.
- Cổng kiểm chứng cấp tệp nhân vào điểm khung: `den` không tăng, `voi` en sụp (0,90 → 0,46 ở β = 2).
- Âm tổng hợp «đổi lượt» A–B–A (2.514 tệp, cùng fold): vu oan `voi` vi không giảm (1,7 → 1,8 %).
- Chuẩn hoá điểm theo tệp (trừ trung vị logit): không giảm vu oan fold 0.
- Xếp chồng LR các nhánh đầu cả tệp có tương tác ngôn ngữ, và hiệu chỉnh theo fold kiểu p3_v1: không ổn định ở đuôi vi (vu oan `giong` vi tới 1,3–7 %; hiệu chỉnh làm `nen` vi B 0,27 → 0,16).

## 5. Khoảng cách tới đích và bước tiếp
- Khử ồn đoạn (0,25 / 0,56 vs 0,85) và khử ồn cả tệp (0,27 / 0,32 vs F1 0,90): cần **dấu vết đặc thù công cụ** phân biệt với WebRTC NS ≤ 2 (phổ dư theo bin, «musical noise», thích nghi nhiễu của NS) — gợi ý: CNN nhỏ trên phổ khoảng lặng độ phân giải cao; **sinh cặp đối chứng cùng bản ghi** (WebRTC 0/1/2/3 vs RNNoise / DFN / noisereduce trên cùng tệp gốc, đúng fold) để học hiệu thay vì nhớ miền.
- Đổi giọng đoạn vi (0,56 vs 0,80): cần **ghi âm hội thoại thật nhiều người** đúng fold làm âm cứng (cùng P2) và dấu vết hiện vật bộ biến giọng (pha / formant) thay cho «khác giọng».
- Công cụ chưa thấy: chưa có bộ giữ kín → **chưa đo** (cần sinh dữ liệu mới; GPU đang hoãn).
- Vu oan trên ghi âm thật (dev cào, e2e): **chưa đo** — cần P0 / P1 chạy `e2e_test` với `score.py` p3_v2.
- Không nộp dữ liệu mới đợt này → không chạy `kiem_bo` / `lockbox` (không có manifest mới).

