# Luật chung cho mọi lane (P0–P5) — VoiceAudit v9

> Đọc sau `lanes/BAT_DAU.md` (tổng quan cho người mới) và trước tệp lane của mình. Cập nhật 19/10/2026. Nguồn gốc phân việc: `CAY_PHAN_RA_V2_VA_PHAN_VIEC_6_NGUOI.md` (04/10); chỉ tiêu: `VoiceAudit.html` (mục Requirement).
> Tinh thần: **chặt ở "làm cái gì", "so với cái gì", "đo thế nào", "nộp ra sao"; tự do ở "làm bằng cách nào".**

---

## 0. CÁI GÌ CỐ ĐỊNH, CÁI GÌ CHỈ LÀ GỢI Ý — đọc kỹ trước mọi thứ khác

| CỐ ĐỊNH (không được lệch — để các mảng không xung đột khi ghép) | CHỈ LÀ GỢI Ý (tham khảo, được bỏ qua hoặc làm khác) |
|---|---|
| Mục tiêu của mảng (chỉ tiêu Req mảng chịu) và phạm vi mảng (nhãn / thao tác / kênh mảng sở hữu) | "Hướng đi", thứ tự các chặng, ý tưởng, danh sách công cụ / mô hình / đặc trưng được nêu |
| Thước đo: OOF 5 fold theo người nói, ngân sách vu oan (§3.1), `ghep_thu`, tiêu chí ghép (§3.3) | Giả thuyết về nguyên nhân (vd. "nghi lệch fold", "nghi Zoom NS") — là giả thuyết của nhóm trưởng, **chưa phải kết luận** |
| Hợp đồng đầu vào / đầu ra (§4), luật cứng (§2), chính sách dữ liệu, an toàn máy | Bằng chứng cũ và mục "đã thử, chưa hiệu quả" — **đúng trong cấu hình đã đo**, không phải lệnh cấm |

**Với Claude của từng thành viên:** đừng coi tài liệu này là lộ trình bắt buộc. Trước khi làm theo gợi ý, hãy **tự đề xuất ≥ 2–3 hướng của riêng mình** (tìm tài liệu, bài báo, mã mở mới hơn; nghĩ lại bài toán từ đầu), so với gợi ý, rồi chọn theo **số đo của mảng ở ngân sách**. Được thử lại hướng "đã thử" nếu có cách làm hoặc lý do mới (ghi rõ khác ở đâu). Được thách thức giả thuyết của nhóm trưởng bằng số liệu. Giới hạn duy nhất: **không lệch mục tiêu / phạm vi mảng và không làm hỏng mảng khác** (kiểm bằng `ghep_thu`) — mọi thứ còn lại là vùng sáng tạo.

**Mọi họ phương pháp đều mở** — số đo tay, ảnh phổ (kể cả mô hình ảnh phổ đã huấn luyện sẵn), embedding, kết hợp / kiểm chứng, hoặc hướng chưa ai nghĩ tới.

**Để không bị neo theo hệ thống trên máy gốc:** các tệp `CHUNG.md`, `BAT_DAU.md`, `<Px>/CLAUDE.md` chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép. Ý tưởng, hướng gợi ý, thử nghiệm cũ, bản đồ phương pháp nằm riêng (`Y_TUONG_CHUNG.md`, `<Px>/Y_TUONG.md`, `HUONG_PHUONG_PHAP.md`) — **chỉ đọc sau khi đã tự đề xuất hướng**. Kiến trúc hiện tại (XLS-R → kênh → giải mã) là **baseline để vượt**, không phải khuôn phải theo. Dữ liệu trên máy gốc và các gói gửi đi **chỉ là ví dụ** để hiểu định dạng và tái tạo baseline (`DU_LIEU.md`).

---

## 1. HỆ THỐNG CHUNG — ai cũng phải nắm

### 1.1 Bài toán và nhãn
Nhận **một tệp ghi âm lời nói**, trả **một nhãn** + **phiếu thao tác** (mỗi thao tác: loại, phạm vi, mốc thời gian, mức độ, ảnh, ≥ 2 dấu hiệu có số, giải thích).

| Nhãn | Nghĩa | Thao tác (8, theo phạm vi) | Lane chịu |
|---|---|---|---|
| **01** | Không chỉnh sửa | — (gồm cả nén, truyền tải ứng dụng, mặc định thiết bị, **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi** — mặc định, không chỉnh) | **P2** |
| **02** | Chỉnh sửa hậu kỳ | khử ồn cả tệp · khử ồn một đoạn · đổi giọng cả tệp · đổi giọng một đoạn | **P3** |
| **03** | Ghép hoặc chèn đoạn | ghép lời thật (một đoạn) · xoá lời (quanh mối nối) | **P4** |
| | | giọng máy một đoạn · giọng máy cả tệp | **P5** |
| **04** | Không đủ cơ sở | Constraint vi phạm (24 hàng) · vùng xám · bị kiểm chứng bác | **P1** (Constraint) · **P0** (xám) |

Thứ tự nhãn tệp: Constraint 04 → **03 > 02 > 04 (phân tích) > 01**. Nhãn theo **dấu vết kỹ thuật**, không theo nội dung.

### 1.2 Cây vấn đề (gốc là vấn đề ban đầu) và luồng xử lý

**Cây vấn đề** — bản gốc là cây trong `VoiceAudit_TomTat.html` §02 (cùng cây: `CAY_PHAN_RA_V2.html`, `CAY_PHAN_RA_V2_VA_PHAN_VIEC_6_NGUOI.md §1`; V1–V7 = «ô 1–7» trong TomTat = «Vấn đề con 1–7» trong cây tương tác): *Có tin được tệp ghi âm này không?* → V1 có đủ cơ sở để xét không (P1) · V2 có bị sửa không — chuyên gia sàng lọc (= cổng 01), kho tệp thật (P2, P0) · V3 sửa kiểu gì — **chỉ đo**, ba nhóm: 02 xử lý lời có sẵn (P3) · 03 lời thật (P4) · 03 giọng máy (P5) · V4 ở đâu, phạm vi nào — đặt ngưỡng từng thao tác, giải mã, gộp dòng (P0, P1) · V5 vô hại hay sai lệch, đủ chắc chưa (P1, P0) · V6 người nhận hiểu và kiểm lại được không (P1) · V7 dùng được ngoài đời không (P0). Mảng của bạn trả lời vấn đề con nào (mục 1 của tệp lane) thì đó là **mục tiêu không được lệch**.

**Luồng xử lý hiện tại** (một cách hiện thực các câu trả lời trên — là baseline, không phải kiến trúc bắt buộc) và ai giữ khối nào:

```
tệp âm thanh
  │
  ▼
[K1 kiểm điều kiện · LID chọn bộ ngưỡng vi/en · cảnh báo] ──── P1 ──► vi phạm cứng → 04 + lý do
  │
  ▼
[XLS-R đóng băng, khúc 32 s, int8] ─────────────────────────── P0 giữ (dùng chung)
  │  khung 20 ms (L12, PCA 64) · tầng 6 · vector gộp thô (L8/12/16 mean+std)
  ├──► kênh khung ghep · xoa · doan_ghep ───────────────────── P4 ┐
  ├──► kênh khung den (khử ồn vùng) · voi (đổi giọng vùng) ──── P3 ├─ điểm khung (T, 6) = H3
  ├──► kênh khung giong_may ─────────────────────────────────── P5 ┘   (nay: fuse1 = FT-2 LoRA ⊕ TCN, P0 ghép)
  ├──► đầu cả tệp nen · giong (P3) · tts_file (P5) ──────────────── vector gộp → xác suất cả tệp (h2)
  └──► cổng 01 (chưa có) ─────────────────────────────────────── P2
  │
  ▼
[K7 giải mã: đỉnh / vùng vượt ngưỡng · ngưỡng theo ngôn ngữ × ĐỘ DÀI (lencal) · gộp dòng · ≤ 2 dòng / 14 s] ── P0
  │  dòng thao tác H4 {level2, kind, start, end, p, nm, via}
  ▼
[K8 luật nhãn tệp] ── P1        [K9 hồ sơ · ảnh · dấu hiệu (signs.py của lane, phân vị kho thật của P2) · PDF] ── P1
```

- "XLS-R" của dự án = `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` (**đã tinh chỉnh chống giả mạo**, không phải XLS-R gốc).
- Mã: `voiceaudit_v9.py` → `v6/product/pipeline.py` (`Product`) → `ticket.py`. Ngưỡng `.tmp_v6/results/thresholds_v9F.json` (điểm vận hành F), ngưỡng theo độ dài `lencal_v9F.json`.
- **Kênh giọng máy đang TẮT** (`ftts = 0`): bật ở fold 0 làm vu oan tệp thật vi 14 → 42 %.

### 1.3 Mười lăm yêu cầu — chỉ tiêu, hiện trạng, ai chịu

Định nghĩa 14 metric (12 định lượng, 2 định tính), công thức tính và bảng ánh xạ yêu cầu ↔ metric (mỗi yêu cầu dùng metric nào, đánh giá ở vấn đề nào của cây): `VoiceAudit_DanhGia.docx`, phần 2–5. Báo cáo của lane dùng đúng tên và công thức đó. Tệp đánh giá đánh số yêu cầu STT 1–17 (tách 7a / 7b / 8a / 8b); bảng nối hai cách đánh số: `lanes/BAT_DAU.md` mục 5.

| Req | Nội dung | Chỉ tiêu | Hiện trạng (vi / en) | Chịu chính |
|---|---|---|---|---|
| 1 | macro-F1 01/02/03 | ≥ 0,86 | e2e 0,65 / 0,72 · OOF 0,55 / 0,68 | P0 (cả nhóm) |
| 2 | không bỏ sót tệp sửa | ≥ 0,90 | e2e 0,57 / 0,67 · tệp dài có lencal 0,35 | P0 (P3 P4 P5) |
| 3 | báo "có sửa" đáng tin | ≥ 0,97 | e2e 0,95 · ghi âm thật 0,85 | P0 · P2 |
| 4 | vu oan tệp thật | ≤ 3 % | OOF ngân sách 3 %: 7,0 % / 3,8 % (tệp 01 gốc 5,3 % / 2,6 %) · e2e 7,7 % · ghi âm thật 13,0 % (có lencal) · tệp 1–10 ph 51 % → **15,3 % có lencal** | **P2** (mọi kênh trong ngân sách §3) |
| 5 | khử ồn vô hại không bị kết tội | ≥ 92 % | 89 % / 96 % | P3 (P2) |
| 6 | ghép thật không bị gọi giọng máy | ≤ 5 % | chưa đo trong hệ thống (kênh tắt); OOF khi bật ở ngân sách: 7,6 % / 1,4 % | P5 (P4) |
| 7a | F1 ghép lời thật | ≥ 0,80 | OOF 0,63 / 0,79 | P4 |
| 7b | F1 xoá lời | ≥ 0,65 | OOF 0,26 / 0,44 | P4 |
| 8a | F1 giọng máy một đoạn | ≥ 0,86 | chưa đo trong hệ thống (kênh tắt); OOF khi bật ở ngân sách: vi 0,17 (31 tệp) | P5 |
| 8b | giọng máy cả tệp | ≥ 0,90 | 0 % khi kênh tắt; OOF khi bật ở ngân sách: 29 % (vi, 7 tệp) | P5 |
| 9 | từ chối đúng mức (04 **do phân tích** / tệp được chấm; tệp bị loại vì Constraint không tính, báo riêng theo lý do) | ≤ 8 % | 1,8–6,2 % | P1 (luật nhãn) · P0 (vùng xám) |
| 10 | mốc lệch ≤ 500 ms | ≥ 78 % | 30 % / 40 % | P3 P4 P5 (đoạn mình) |
| 11 | IoU vùng khoanh | ≥ 0,80 | 0,57 / 0,58 | P3 P4 P5 |
| 12 | bằng chứng đủ (ảnh + ≥ 2 dấu hiệu) | 100 % | 100 % | P1 |
| 13 | bằng chứng dễ hiểu — khảo sát hai bước: người ngoài đoán đúng loại khi chưa thấy nhãn (chốt) · đọc hiểu cả hồ sơ (phụ) | ≥ 70 % | chưa khảo sát | P1 |
| 14 | hồ sơ không mâu thuẫn | 100 % | 100 % | P1 |
| 15 | RTF CPU 4 nhân | ≤ 0,6 | tệp 1–10 phút 0,32 · tệp ngắn 0,5–0,6 (chi phí cố định) | P0 (ngân sách RTF từng lane) |

Điều kiện chấm (`VoiceAudit.html`): chỉ chấm tệp không bị Constraint chuyển 04; người nói ở tập chấm không có trong tập học; báo riêng theo ngôn ngữ, độ dài tệp (≤ 20 s / dài hơn), độ dài chỗ sửa, số người nói, tệp hai kênh; mỗi chỉ tiêu kèm khoảng tin cậy 95 %; chỉ tiêu chốt chấm **một lần trên tập khoá**.

### 1.4 Số đo hệ thống mới nhất

| Tập | Req 1 | Req 2 | Req 3 | Req 4 | Nguồn |
|---|---|---|---|---|---|
| OOF natvi + naten, điểm F | vi 0,546 · en 0,682 | 0,491 · 0,655 | 0,956 · 0,976 | 10,1 % · 7,3 % | `ghep_thu_nen.md` |
| OOF, **ngân sách 3 %** (§3) | vi 0,482 · en 0,620 | 0,388 · 0,552 | 0,961 · 0,985 | **7,0 %** · **3,8 %** | `ghep_thu_ngansach3.md` |
| e2e natvi + naten 60 tệp/kiểu (tạm) | 0,67 | 0,60 | 0,95 | 7,7 % | `e2e_v9F_n60_TAM.md` |
| e2e ghi âm thật cào (trước lencal) | 0,72 | 0,67 | 0,83 | 15,6 % | `e2e_v9F_scrape.md` |
| e2e ghi âm thật cào, **có lencal** | 0,71 | 0,65 | 0,85 | **13,0 %** (vi 11,1 · en 14,7) | `e2e_v9F_scrape_len.md` |
| e2e tệp 1–10 phút (trước lencal) | 0,54 | 0,75 | 0,77 | 51 % | `e2e_v9F_long.md` |
| e2e tệp 1–10 phút, **có lencal** (cùng 321 tệp, chấm một lần 19/10) | 0,38 | 0,35 | 0,84 | **15,3 %** (en 24 · vi 9) | `e2e_v9F_long_len.md` |

**Tệp dài sau lencal:** vu oan giảm mạnh nhưng nhãn 03 đúng chỉ còn 5 % (trước 48 %), xoá lời 0 % — phần lớn "bắt được" trước đó là vu oan trùng may (68 % tệp thật en bị báo). Muốn bắt chỗ sửa ngắn trong tệp dài, điểm ở chỗ sửa phải vượt điểm lớn nhất của cả một tệp thật — phép thử thật cho mọi hướng của P3, P4. Không chỉnh ngưỡng theo bộ này (luật cứng 2).

**Đổi nhãn 20/10 (quyết định 18), mô hình chưa học lại.** Hai dòng OOF trên chấm lại sau khi đổi nhãn; mô hình vẫn là bản học trên nhãn cũ. Số e2e ở các dòng dưới chấm **trước** khi đổi nhãn. Đổi nhãn tác động theo hai đường:
- 212 tệp WebRTC NS mức ≤ 2 trong OOF nay là 01 nhưng mô hình vẫn coi là 02 → tính thành vu oan: vi 16,7 % (23 / 138), en 13,5 % (10 / 74). Trên tệp 01 gốc, Req 4 gần như không đổi: vi 5,3 % (42 / 786), en 2,6 % (15 / 586).
- Các tệp này cũng vào tập «tệp thật» dùng đặt ngưỡng → ngưỡng kênh khử ồn một đoạn bị đẩy lên → phát hiện khử ồn một đoạn rơi mạnh (vi 69 → 9 %, en 76 → 22 %).

Baseline sạch cần học lại trên nhãn mới (việc của P0, `P0/DU_LIEU.md`). Đến lúc đó, so sánh cải tiến với các số ở trên, theo **cùng** bộ nhãn.

Chưa rõ vì sao vi vẫn vu oan 5,3 % trên tệp 01 gốc ở ngân sách 3 % (kênh ghép / xoá vu oan 4,3 % tệp 01 gốc vi dù ngân sách 0,5 % mỗi kênh; **không** phải bão hoà điểm). Câu hỏi mở — giả thuyết của nhóm trưởng ở `Y_TUONG_CHUNG.md`.

---

## 2. LUẬT CỨNG (không được phá)

1. **Mô hình nền lớn (mạng sâu học biểu diễn âm thanh) phải là mô hình đã huấn luyện sẵn** — đóng băng, LoRA, tinh chỉnh một phần; không học từ đầu. Đầu nhỏ / bộ phân loại nhỏ đặt trên đặc trưng (TCN, MLP, cây, hồi quy…) được học từ đầu.
2. **Không chọn ngưỡng / siêu tham số / biến thể trên tập chấm.** Chọn trên OOF (5 fold theo người nói) hoặc dev. Tập chấm (e2e fold ngoài, `scrape*`, `long*`) chỉ để **báo cáo**, mỗi biến thể đã chốt chạy một lần. **Tập khoá** chỉ P0, **một lần**, cuối cùng.
3. **Fold theo NGƯỜI NÓI**: `v6.boundary.detect.person_fold(meta)` (crc32 % 5); dữ liệu cào: `meta.person` = nhóm kênh. Mô hình / ngưỡng fold k chỉ từ dữ liệu fold ≠ k.
4. **Chọn điểm dừng, số lượt học, siêu tham số trên dữ liệu học / dev theo fold người nói** — không nhìn tập chấm (cùng tinh thần luật 2). Cách học là tự do.
5. **Hợp đồng đóng băng** (§4) và **ngân sách vu oan** (§3): muốn đổi → đề xuất `lanes/DE_XUAT/<ngay>_<lane>_<ten>.md`, P0 duyệt.
6. **Không sửa tệp lane khác sở hữu.** Cần sửa → đề xuất cho chủ lane / P0.
7. **Không xoá dữ liệu** — chuyển `_quarantine` kèm nhật ký. Không ghi đè kết quả người khác: mọi kết quả có `--tag` / hậu tố riêng.
8. **Chính sách dữ liệu**: cào chỉ CC BY / BY-SA / CC0, YouTube chỉ **kênh không chuyên**; tự thêm nguồn / kênh / kho không cần duyệt, miễn đúng phạm vi dữ liệu của lane và qua `kiem_bo`; không tôn giáo, không trẻ em / trường phổ thông; **không nhái giọng người cào được / người nổi tiếng**; dữ liệu nội bộ. **Khử ồn:** **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi phổ thông** (WebRTC NS mức 0–2, chế độ mặc định) là **01**, dù áp cả tệp hay một đoạn — bản chất vẫn là một bản ghi bình thường. Khử ồn **chuyên nghiệp / mạnh** (ứng dụng ghi âm chuyên nghiệp có lọc ồn sâu, WebRTC NS mức 3, chế độ «Cao», công cụ khử ồn AI), bật lúc ghi hay áp sau khi ghi, là **02**. Tiêu chí là **loại ứng dụng / công cụ**, không phải cả tệp hay một đoạn (`DU_LIEU.md` mục 2.2). Bản ghi có `meta.chinh_sach` (vd. giọng máy cũ nhân bản người không được phép nhái) **không dùng để học** — quyết định 18.
9. **Máy yếu bộ nhớ**: việc nặng tuần tự qua `bash .tmp_v6/wf/run_serial.sh <lệnh>`; việc mạng nhẹ `run_nolock.sh`. **GPU đang hoãn** — dùng GPU phải hỏi người dùng.
10. **Không sửa `VoiceAudit_6Muc.html`** (bản nộp). Đổi `VoiceAudit.html` → qua P0.
11. **Báo cáo trung thực**: mỗi kết luận kèm số + tệp nguồn + khoảng tin cậy khi có; số xấu đi cũng phải ghi.
12. **Mã mới** có docstring + CLI + **test hàm thuần** chạy không cần mô hình; không phá test của phần mã bạn nhận (`BAT_DAU.md` mục 10.2, bước 4 «Kiểm cài đặt»; P0 chạy đủ bộ test khi ghép).

## 3. CHỐNG LỆCH MỤC TIÊU VÀ ĐỂ GHÉP DỄ

### 3.1 Ngân sách vu oan theo kênh (cách giữ Req 4 ≤ 3 % khi ghép)
Mỗi kênh được chia **phần vu oan trên tệp thật** (phân vị điểm lớn nhất / tệp thật cùng ngôn ngữ, fold khác). Mỗi lane **tối đa hoá phát hiện TRONG ngân sách của mình** — không ai được "nới" kênh mình cho đẹp, vì vu oan cộng dồn khi ghép.

| Kênh | Lane | Điểm F hiện tại | **Ngân sách mục tiêu** | Cờ `ghep_thu` |
|---|---|---|---|---|
| ghep · xoa | P4 | 1 % mỗi kênh | **0,5 % mỗi kênh** | `--f03 0.005` |
| den · voi | P3 | 3 % mỗi kênh | **0,5 % mỗi kênh** | `--freg 0.005` |
| đầu nen · giong | P3 | 0,5 % mỗi đầu | **0,25 % mỗi đầu** | `--fh 0.0025` |
| giong_may · tts_file | P5 | tắt | **0,25 % cho kênh, 0,25 % cho đầu** (giữ chỗ) | `--ftts 0.0025` (một cờ đặt cả hai) |
| cổng 01 (giảm vu oan chung) | P2 | — | — | — |

Tổng ≤ 3 % (cận hợp; thực tế thấp hơn). Cột «Điểm F hiện tại» là mức đang đặt trong sản phẩm; **KPI và baseline chính thức của lane tính ở «Ngân sách mục tiêu»**. Vu oan trong bảng baseline của lane là của riêng kênh đó; Req 4 hệ thống là hợp của mọi kênh và đầu, nên vẫn có thể vượt 3 % khi ngưỡng không giữ được trên một ngôn ngữ. KPI chính của P3 / P4 / P5 = **% phát hiện đúng ở ngân sách mục tiêu** (bảng baseline trong tệp lane). Hiện (20/10, mô hình còn học nhãn cũ) ngân sách này cho Req 4 en 3,8 %, vi 7,0 %; tính riêng trên tệp 01 gốc: en 2,6 % (đạt), vi 5,3 % (chưa) — xem §1.4.

### 3.2 Công cụ ghép thử — tự kiểm trước khi nộp
```
bash .tmp_v6/wf/run_serial.sh .venv/Scripts/python.exe -m v6.product.ghep_thu --name <lane>_<ten> \
     --col den=experts/P3/<ten>/oof_<ten>.pkl:0 --col voi=experts/P3/<ten>/oof_<ten>.pkl:1 [--mode replace|mean] \
     [--head giong=experts/P3/<ten>/heads_<ten>.pkl:0] --f03 0.005 --freg 0.005 --fh 0.0025
```
Cắm cột của lane vào hệ thống đang chạy (fuse1 / h2), đặt lại ngưỡng đúng quy tắc, ra `.tmp_v6/results/ghep_thu_<ten>.md`: Req 1–4, 9, 7a, 7b, 6, 8a theo ngôn ngữ (**nền vs mới**), vu oan tệp thật **theo kênh** (kiểm ngân sách), % đúng theo kiểu (ô tụt > 5 điểm đánh dấu `!`). Tái tạo baseline (chấm lại 20/10): `--name nen` không cờ ngân sách → điểm F, en 0,682 / 0,655 / 0,976 / 7,3 %, vi 0,546 / 0,491 / 0,956 / 10,1 %; `--name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025` → ngân sách 3 %, en 0,620 / 0,552 / 0,985 / 3,8 %, vi 0,482 / 0,388 / 0,961 / 7,0 % (Req 1 / 2 / 3 / 4). Mỗi lần chạy ghi đè `ghep_thu_<name>.md` — luôn đặt `--name` riêng cho từng thử.

### 3.3 Thế nào là một cải tiến được ghép
1. Trên **OOF 5 fold theo người nói**, ở **ngân sách của kênh mình**: phát hiện tăng (hoặc vu oan giảm ở cùng phát hiện) so với **baseline lane** — có khoảng tin cậy hoặc ≥ 2 hạt giống.
2. `ghep_thu` (cùng ngân sách §3.1): **không ô nào của lane khác tụt > 5 điểm**, vu oan kênh mình không vượt ngân sách, Req 4 hệ thống không tăng.
3. Bảng đối chứng của lane (T*.6 / T*.9 / T5.4–5.5) không xấu đi; RTF trong ngân sách lane (P3 / P4 / P5 ≤ 0,06 mỗi lane; P2 ≤ 0,02) **khi nộp để ghép vào sản phẩm**. Lúc khám phá được chạy chậm hơn (quyết định 5: tìm hướng trước, tối ưu tốc độ sau) — khi nộp mà vượt ngân sách thì kèm kế hoạch tăng tốc.
4. Có thẻ `card.json`, báo cáo `lanes/<lane>/BAO_CAO_<ddmm>.md`, kèm tệp `ghep_thu_<ten>.md`.

## 4. Hợp đồng (đóng băng)

| Mã | Nội dung | Tham chiếu |
|---|---|---|
| H1 | manifest `data_labels/v6/<bo>/manifest.json`: `[{id, kind, cfg, src, wav, meta{level2[], edits[], person, speaker, fold, lang, source, ...}}]`; `wav` tương đối `data_labels/realistic_v4` | kiểm: `v6/data/kiem_bo.py` (mọi lane có) · bộ dựng hiện có: `v6/scrape/build_sets.py` (P2), `v6/natural/build_natural.py` (P0) |
| H2 | mã nhãn cấp 2 **đang dùng trong dữ liệu**: `02_nen_toan_cuc 02_nen_cuc_bo 02_giong_toan_cuc 02_giong_cuc_bo 03_ghep_giong_that 03_chen_giong_may`. **Xoá lời** dùng `03_ghep_giong_that` + `edits[].sub = "cat_bo"` (`op = xoa_loi`) — mã `03_xoa_loi` có trong bản đề xuất nhưng **không dùng** trong dữ liệu; bộ mới giữ đúng quy ước này, không trộn hai cách. Chuyển giọng AI là 02 (`02_giong_*`), không phải 03. Không đổi nhãn của bộ có sẵn — muốn đổi thì đề xuất cho P0 | `VoiceAudit.html`, `v6/natural/req_check.py edit_type` |
| H3 | điểm khung 20 ms `(T, C) float16`, OOF `{id: mảng}` 5 fold theo người; đầu cả tệp `{"ids": [...], "P": (N, k)}` | `.tmp_v6/cache/boundary_oof_fuse1.pkl` (cột: ghep 0, xoa 1, den 2, voi 3, spl 4, ttsr 5 = kênh giọng máy `giong_may`), `.tmp_v6/cache/heads2_h2.pkl` (nen 0, giong 1, ttsf 2 = đầu giọng máy cả tệp `tts_file`). Tên dùng trong `ghep_thu --col / --head` là tên cột: `ghep xoa den voi spl ttsr` · `nen giong ttsf` |
| H4 | dòng thao tác `{level2, kind (segment / global), start, end, p, nm, via}` | `v6/boundary/fusion.py`, `rowmerge.py` |
| H5 | hồ sơ `voiceaudit.v6.ho_so/1` + #bind | `v6/product/ticket.py` (P1) |
| H6 | chấm: `v6/natural/req_check.py`, `v6/product/ghep_thu.py` (OOF hệ thống — mọi lane có), `v6/product/e2e_test.py` (đầu-cuối — P1, P2 có; P0 chạy khi ghép) | |

Học trên dữ liệu nào, đầu vào nào cũng được; khi nộp vẫn kèm điểm OOF trên natvi / naten (đủ id, 5 fold theo người nói) — để ghép thử tính được Req 1–4 trên cùng một bộ, so cùng một thước với baseline; số trên dữ liệu riêng của bạn báo thêm bên cạnh. Hướng không cho điểm mỗi 20 ms: tự quy về điểm khung, hoặc đề xuất đổi hợp đồng.

**Gói chuyên gia**: `experts/<lane>/<ten>_<phienban>/` gồm `card.json` (kênh, ngôn ngữ, dữ liệu, fold, seed, số đo ở ngân sách, RTF 300 s, bộ nhớ), `oof_<ten>.pkl` (H3, **đủ id của natvi / naten goc**, 5 fold), `weights/`, `score.py` (wav, ctx → {kênh: (T,)} hoặc {đầu: p}), `signs.py` (≥ 2 dấu hiệu có số + phân vị kho thật), `verify.py` (tuỳ chọn), `tests/`.

## 5. Cách đo chung
- **OOF hệ thống** (nhanh, không XLS-R): `ghep_thu` (§3.2); đường biên đầy đủ: `python -m v6.boundary.fuse9 sweep --bound <tag> --reg <tag> --heads <tag>`.
- **Theo độ dài** (P1, P2 có mã; P0 chạy khi ghép): `python -m v6.product.lencal regress` (tệp ngắn), `check` (chuỗi giả dài).
- **Đầu-cuối** (P1, P2 có mã; P0 chạy khi ghép; chậm ~8–16 s / tệp, cần ≥ 6 GB bộ nhớ ảo trống): `python -m v6.product.e2e_test --sets <bo...> --n <tep/kieu> --n-real <tep that> --tag _<lane>_<ten> --resume` → `e2e_v9F_<tag>.md` + `scores.npz` từng tệp.
- **Báo cáo**: `lanes/<lane>/BAO_CAO_<ddmm>.md` — giả thuyết · làm gì · số trước / sau (cùng thước, cùng ngân sách) · ô tụt · bước tiếp.

## 6. Kinh nghiệm cũ

Các thử nghiệm đã làm và kết quả nằm ở [`Y_TUONG_CHUNG.md`](Y_TUONG_CHUNG.md) và `lanes/<Px>/Y_TUONG.md` — **đọc sau khi đã tự đề xuất hướng**, để không bị neo theo cách nhóm đang làm.

## 7. Khi nào PHẢI hỏi (P0 hoặc người dùng)
Dùng GPU · xoá / chuyển dữ liệu lớn · đổi hợp đồng / ngân sách / điểm vận hành / chỉ tiêu · chạm tập khoá · việc > 6 giờ máy (để P0 xếp hàng) · đổi chính sách dữ liệu.

## 8. Làm việc với Claude
- Mở Claude Code ở **thư mục gốc của bộ gửi**, câu đầu theo `lanes/BAT_DAU.md` mục 11: đọc @lanes/BAT_DAU.md, @lanes/CHUNG.md và @lanes/<Px>/CLAUDE.md (**chưa** đọc các tệp ý tưởng); trả lời 5 câu tự kiểm; tái tạo baseline bằng `ghep_thu`; tự đề xuất 2–3 hướng; **sau đó** mới đọc `<Px>/Y_TUONG.md`, `Y_TUONG_CHUNG.md`, `HUONG_PHUONG_PHAP.md` để so.
- Claude của lane **nên chủ động** tìm và thử hướng mới (không chỉ làm theo gợi ý của tệp lane) — xem §0; **luôn** so với baseline lane ở ngân sách §3.1 bằng `ghep_thu`; **phải** dừng hỏi ở §7; **phải** ghi báo cáo sau mỗi đợt, kể cả hướng tự nghĩ đã thử và kết quả.
- Quyết định đã chốt của người dùng nằm ở `.tmp_v6/wf/DECISIONS.md` và §2 tệp này — không làm trái; gặp mâu thuẫn thì hỏi (§7).
