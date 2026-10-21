# Bắt đầu từ đây — VoiceAudit cho người mới vào nhóm

> Tệp này dành cho thành viên **chưa biết gì về hệ thống** và cho Claude của họ. Đọc hết mất khoảng 30 phút.
> Đọc xong, bạn phải trả lời được 5 câu ở mục 9. Chưa trả lời được thì hỏi lại trước khi làm.
> Hệ thống mô tả ở đây là **baseline để vượt**, không phải đáp án để làm theo. Tài liệu chỉ rào bốn thứ (mục 6) — còn lại bạn tự do tìm hướng mới.
> **Dữ liệu trên máy gốc và các gói gửi đi chỉ là ví dụ** để hiểu định dạng và tái tạo baseline — bạn được tự xây dữ liệu của mình, miễn phủ đúng các trường hợp của mảng (`DU_LIEU.md`).
> **Ý tưởng của nhóm trưởng nằm riêng** (`Y_TUONG_CHUNG.md`, `<Px>/Y_TUONG.md`, `HUONG_PHUONG_PHAP.md`) — chỉ đọc sau khi đã tự đề xuất hướng (mục 11).
> Cập nhật 19/10/2026.

---

## 1. Bài toán trong một phút

**Câu hỏi gốc:** *Có tin được tệp ghi âm này không?*

Ghi âm cuộc gọi, tin nhắn thoại, cuộc họp ngày càng được đưa ra làm bằng chứng, trong khi ai cũng có công cụ khử ồn, đổi giọng, cắt ghép, nhái giọng bằng AI. Các công cụ phát hiện hiện có chỉ trả lời «thật hay giả» cho cả tệp, coi mọi chỉnh sửa là giả mạo, và không chỉ ra đoạn nào bị sửa.

**VoiceAudit** nhận **1 tệp ghi âm** (tiếng Việt hoặc tiếng Anh, 3 giây – 10 phút) và trả một **hồ sơ giám định sơ bộ**:

- **Một nhãn cho cả tệp:**

  | Nhãn | Nghĩa |
  |---|---|
  | 01 · Không chỉnh sửa | Không thấy dấu vết sửa (không phải chứng nhận bản thật). Nén, gửi qua ứng dụng, thiết bị, **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi** (mặc định, không chỉnh) vẫn là 01 |
  | 02 · Chỉnh sửa hậu kỳ | Lời có sẵn trong bản ghi, không thêm bớt; chỉ đổi cách nghe: khử ồn, đổi giọng |
  | 03 · Ghép hoặc chèn đoạn | Chuỗi lời bị đổi: chèn lời thật từ chỗ khác, xoá lời, chèn giọng máy, hoặc cả tệp là giọng máy |
  | 04 · Không đủ cơ sở | Từ chối kết luận, luôn kèm lý do |

- **Mỗi chỗ sửa một dòng:** loại sửa, phạm vi (cả tệp hay một đoạn), mốc bắt đầu – kết thúc, mức độ (vô hại → sai lệch nặng).
- **Mỗi chỗ sửa một ảnh** (dạng sóng + phổ, tô đúng vùng sửa) và **một đoạn giải thích** có ít nhất 2 dấu hiệu bằng số.

**5 loại sửa × phạm vi = 8 thao tác** hệ thống phải bắt:

| Loại sửa | Cả tệp | Một đoạn | Nhãn |
|---|---|---|---|
| Khử ồn | ✓ | ✓ | 02 |
| Đổi giọng | ✓ | ✓ | 02 |
| Ghép lời thật (chèn / thay bằng lời người thật lấy từ chỗ khác) | — | ✓ | 03 |
| Xoá lời (cắt bỏ rồi nối lại) | — | ✓ | 03 |
| Giọng máy (giọng do AI tạo) | ✓ | ✓ | 03 |

**Nguyên tắc quan trọng nhất:** nhãn dựa trên **dấu vết kỹ thuật** của thao tác, **không** dựa trên nội dung lời. Hệ thống không nghe hiểu lời nói, không biết ai nói thật hay nói dối.

---

## 2. Cây bảy vấn đề — bản đồ của cả dự án

Câu hỏi gốc được chia thành 7 vấn đề. Mỗi lane giữ một hoặc vài vấn đề; **vấn đề lane bạn giữ là mục tiêu không được lệch.**

| Vấn đề | Câu hỏi | Trả ra | Lane giữ |
|---|---|---|---|
| 1 | Có đủ cơ sở để xét không? | báo cáo điều kiện, hoặc 04 kèm lý do (dừng) | P1 |
| 2 | Tệp có bị chỉnh sửa không? | có dấu vết, hoặc 01 (dừng) — sàng lọc, không vu oan tệp thật | P2 (P0 ghép) |
| 3 | Nếu có, sửa kiểu gì? | 8 đường điểm theo thời gian — **chỉ đo, chưa quyết** | P3 (02) · P4 (03 lời thật) · P5 (03 giọng máy) |
| 4 | Sửa ở đâu, phạm vi nào? | dòng thao tác có mốc — đặt ngưỡng từng thao tác, gộp dòng | P0 · P1 (tệp hai kênh) |
| 5 | Vô hại hay làm sai lệch? Đủ chắc chưa? | nhãn và mức độ, hoặc 04 | P1 (luật nhãn) · P0 (vùng xám) |
| 6 | Người nhận có hiểu và kiểm lại được không? | hồ sơ: ảnh, giải thích, PDF | P1 |
| 7 | Có dùng được ngoài đời không? | số đo dùng chung, tốc độ, tệp dài, ghi âm thật | P0 · P2 |

Cây đầy đủ (tầng 2 có 9 câu hỏi kỹ thuật, tầng 3 có 36 vấn đề con): `VoiceAudit_TomTat.html` phần 2 (bản gốc), `CAY_PHAN_RA_V2.html` (bản bấm được).
Tài liệu lane gọi các vấn đề này là **V1–V7**; TomTat gọi là **ô 1–7**; tài liệu đánh giá gọi là **vấn đề 1–7**. Cùng một thứ.

---

## 3. Hệ thống chạy thế nào (theo thứ tự một tệp đi qua)

```
tệp ghi âm
 │
 ├─1─ Kiểm điều kiện ............................ P1   tệp hỏng, quá ồn, quá ngắn… → 04 + lý do, dừng
 │
 ├─2─ Trích đặc trưng dùng chung (XLS-R) ........ P0   mô hình đã huấn luyện sẵn biến âm thanh thành
 │                                                     1 vector mỗi 20 ms + 1 vector cho cả tệp
 ├─3─ Các «chuyên gia» chấm điểm ................ P3 P4 P5   mỗi thao tác một «kênh»: 1 điểm mỗi 20 ms;
 │                                                          thao tác cả tệp có «đầu cả tệp»: 1 xác suất
 ├─4─ Sàng lọc «cổng 01» (chưa có) .............. P2   tệp không có gì vượt mức tệp thật → 01
 │
 ├─5─ Giải mã ................................... P0   so điểm với ngưỡng (theo ngôn ngữ × độ dài tệp),
 │                                                     tìm đoạn vượt ngưỡng, gộp thành dòng thao tác
 ├─6─ Ra nhãn ................................... P1   thứ tự: 04 điều kiện → 03 → 02 → 04 phân tích → 01
 │
 └─7─ Dựng hồ sơ ................................ P1   ảnh, dấu hiệu có số (mỗi lane viết signs.py,
                                                       so với kho tệp thật của P2), PDF
```

Trong tài liệu lane, các khối có tên mã: bước 1 = **K1**, bước 5 = **K7**, bước 6 = **K8**, bước 7 = **K9**.
Mã chính: `voiceaudit_v9.py` → `v6/product/pipeline.py` → `v6/product/ticket.py`.

**Mô hình nền** hiện dùng là `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` — một XLS-R **đã tinh chỉnh chống giả mạo**, giữ đóng băng.

Luồng này là **cách nhóm đang làm**, không phải cách duy nhất. Lane được dùng mô hình nền khác hoặc họ phương pháp khác cho phần của mình, miễn mô hình đã huấn luyện sẵn, xuất đúng định dạng trao đổi và nằm trong ngân sách (mục 6–7).

---

## 4. Sáu lane — ai làm gì, nối với ai

| Lane | Phần việc | Nhận từ | Giao cho | Khó khăn lớn nhất hiện nay |
|---|---|---|---|---|
| **P0** · nhóm trưởng | Ghép các phần, giải mã + ngưỡng, chấm đầu–cuối, tập khoá | điểm của P3 P4 P5, cổng 01 của P2 | dòng thao tác cho P1 | tệp dài: chỉnh ngưỡng theo độ dài làm vu oan giảm 51 → 15 % nhưng khả năng bắt sửa tụt 0,75 → 0,35 |
| **P1** · khung cố định | Kiểm điều kiện, luật nhãn, hồ sơ · ảnh · PDF, khảo sát người ngoài | dòng thao tác (P0), `signs.py` (P3 P4 P5), kho tệp thật (P2) | hồ sơ cuối cùng | 24 điều kiện chưa có bài kiểm tra chủ động; khảo sát chưa làm |
| **P2** · nhãn 01 | Ghi âm thật, truyền tải qua ứng dụng, cổng 01, kho tệp thật, bộ âm cứng | dữ liệu cào, điểm các kênh | bộ âm cứng cho P3 P4 P5; kho tệp thật cho P1; cổng 01 cho P0 | vu oan 13 % trên ghi âm thật; cổng 01 chưa có |
| **P3** · nhãn 02 | Khử ồn, đổi giọng (cả tệp · một đoạn) | dữ liệu mẫu đúng nhãn (có sẵn đặc trưng XLS-R — không bắt buộc dùng) | điểm kênh khử ồn / đổi giọng + `signs.py` | kênh một đoạn là nguồn vu oan số 1; khử ồn cả tệp chỉ bắt 28–46 % |
| **P4** · nhãn 03 lời thật | Ghép / chèn, xoá lời | dữ liệu mẫu đúng nhãn (có sẵn đặc trưng XLS-R — không bắt buộc dùng) | điểm kênh ghép / xoá + `signs.py` | xoá lời F1 0,26 / 0,44 (cần 0,65); chưa có ca sửa tay thật |
| **P5** · nhãn 03 giọng máy | Giọng máy một đoạn, cả tệp | dữ liệu mẫu đúng nhãn (có sẵn đặc trưng XLS-R — không bắt buộc dùng) | điểm kênh giọng máy + `signs.py` | bật kênh thì giọng thật tiếng Việt bị gọi là máy; kênh đang tắt; cần GPU (đang hoãn) |

Mỗi lane có tệp giao việc riêng `lanes/<Px>/CLAUDE.md`, cùng một khung: vị trí trong hệ thống · đích đến · baseline phải vượt · hướng đi gợi ý · đã thử · phần mã sở hữu · luật riêng · cách nộp · phối hợp.

---

## 5. Thế nào là «đạt» — 17 yêu cầu

Yêu cầu lấy từ mục Requirement của bản đề xuất đã nộp. Định nghĩa đầy đủ, metric và công thức: `VoiceAudit_DanhGia.docx` (phần 2–5).
**Hai cách đánh số:** tài liệu đánh giá đánh STT 1–17; tài liệu lane dùng số Req của bản nộp (1–15, có 7a/7b/8a/8b). Bảng dưới nối hai cách.

| STT | Req | Yêu cầu | Đạt khi | Hiện trạng (vi / en) | Lane chịu |
|---|---|---|---|---|---|
| 1 | 1 | Phân loại đúng | macro-F1 ≥ 0,86 | 0,65 / 0,72 | P0 (cả nhóm) |
| 2 | 2 | Không bỏ sót tệp bị sửa | ≥ 0,90 | 0,57 / 0,67 · tệp dài 0,35 | P0 (P3 P4 P5) |
| 3 | 3 | Báo «có sửa» đáng tin | ≥ 0,97 | 0,95 · ghi âm thật 0,85 | P0 · P2 |
| 4 | 4 | Không vu oan tệp thật | ≤ 3 % | 7,7 % · ghi âm thật 13,0 % · tệp dài 15,3 % · OOF ngân sách 3 %: 7,0 % / 3,8 % | **P2** (mọi kênh) |
| 5 | 5 | Không kết tội khử ồn vô hại | ≥ 92 % | 89 % / 96 % | P3 |
| 6 | 6 | Không nhầm lời thật thành giọng máy | ≤ 5 % | chưa đo (kênh tắt); khi bật: 7,6 % / 1,4 % | P5 (P4) |
| 7 | 7a | Bắt được ghép lời thật | F1 ≥ 0,80 | 0,63 / 0,79 | P4 |
| 8 | 7b | Bắt được xoá lời | F1 ≥ 0,65 | 0,26 / 0,44 | P4 |
| 9 | 8a | Bắt được giọng máy một đoạn | F1 ≥ 0,86 | chưa đo (kênh tắt); khi bật: 0,17 (vi, 31 tệp) | P5 |
| 10 | 8b | Bắt được giọng máy cả tệp | ≥ 0,90 | 0 % (kênh tắt); khi bật: 29 % (vi, 7 tệp) | P5 |
| 11 | 9 | Từ chối đúng mức | ≤ 8 % | 1,8–6,2 % ✓ | P1 · P0 |
| 12 | 10 | Mốc thời gian chính xác (lệch ≤ 0,5 giây) | ≥ 78 % | 30 % / 40 % | P3 P4 P5 |
| 13 | 11 | Vùng khoanh trùng chỗ sửa (IoU) | ≥ 0,80 | 0,57 / 0,58 | P3 P4 P5 |
| 14 | 12 | Bằng chứng đầy đủ | 100 % | 100 % ✓ | P1 |
| 15 | 13 | Bằng chứng dễ hiểu (khảo sát người ngoài, định tính) | ≥ 70 % | chưa khảo sát | P1 |
| 16 | 14 | Hồ sơ không tự mâu thuẫn | 100 % | 100 % ✓ | P1 |
| 17 | 15 | Xử lý kịp thời (giây xử lý ÷ giây âm thanh, máy 4 nhân, không GPU) | ≤ 0,6 | 0,32 tệp dài · 0,5–0,6 tệp ngắn | P0 |

Cột «Hiện trạng» là ở điểm vận hành F của hệ thống; **baseline chính thức của từng lane tính ở ngân sách mục tiêu** (mục 7; số ở tệp lane, mục Baseline).

**Vấn đề lớn nhất của cả nhóm hiện nay là Req 4 — vu oan tệp thật ngoài đời.** Mỗi kênh nổ nhầm một chút, cộng lại thành nhiều. Vì vậy mỗi kênh chỉ được nổ nhầm trong **ngân sách** của mình (mục 7).

---

## 6. Rào ở đâu, tự do ở đâu

Tài liệu chỉ rào **bốn thứ** — để mỗi người làm đúng việc và các phần ghép lại thành một hệ thống thống nhất. Ngoài bốn thứ này, bạn tự do.

| Rào | Nghĩa là | Vì sao |
|---|---|---|
| **1. Đúng mục tiêu, đúng nhiệm vụ** | Lane giữ đúng vấn đề, thao tác, kênh của mình (mục 2, 4) và các Req mình chịu (mục 5); **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi phổ thông** (WebRTC NS mức 0–2, chế độ mặc định) là **01**, dù áp cả tệp hay một đoạn — bản chất vẫn là một bản ghi bình thường. Khử ồn **chuyên nghiệp / mạnh** (ứng dụng ghi âm chuyên nghiệp có lọc ồn sâu, WebRTC NS mức 3, chế độ «Cao», công cụ khử ồn AI), bật lúc ghi hay áp sau khi ghi, là **02**. Tiêu chí là **loại ứng dụng / công cụ**, không phải cả tệp hay một đoạn (`DU_LIEU.md` mục 2.2) | để 6 phần cộng lại trả lời đúng câu hỏi gốc, không ai làm chồng hay bỏ trống |
| **2. Đo trung thực, cùng một thước** | Chọn ngưỡng / tham số trên OOF hoặc dev, **không** trên tập chấm; chia dữ liệu theo người nói; tập khoá chỉ P0 chấm một lần cuối; báo số thật kèm tệp nguồn, kể cả số xấu đi | để số của các lane so được với nhau và không «học thuộc đề» |
| **3. Ghép được, thống nhất khi ghép** | Xuất đúng định dạng trao đổi (hợp đồng H1–H6); giữ hai ngân sách ở mục 7; chạy ghép thử trước khi nộp; không sửa mã lane khác sở hữu (muốn đổi thì đề xuất, mục 12); mã mới có test, không phá test cũ | để phần của bạn cắm vào là chạy, không làm hỏng phần người khác |
| **4. An toàn** | Mô hình nền lớn phải là mô hình đã huấn luyện sẵn (đầu nhỏ / bộ phân loại nhỏ trên đặc trưng thì học từ đầu được); chính sách dữ liệu (dưới đây); không xoá dữ liệu — chuyển `_quarantine`; việc nặng chạy tuần tự, GPU phải hỏi; không sửa bản đã nộp `VoiceAudit_6Muc.html` | quyết định đã chốt của nhóm |

*Chính sách dữ liệu:* cào chỉ giấy phép CC BY / BY-SA / CC0; YouTube chỉ kênh không chuyên (kênh chuyên nghiệp thường đã qua hậu kỳ, không còn là bản ghi thật đơn sơ); không tôn giáo, không trẻ em, không trường phổ thông; không nhái giọng người thật cào được hay người nổi tiếng; dữ liệu nội bộ.

*Lấy dữ liệu:* **tự do** — tự tìm nguồn, kênh, kho nghiên cứu, tự ghi, tự sinh, **không cần xin duyệt**, miễn **đúng phạm vi dữ liệu của lane** (`DU_LIEU.md`, `<Px>/DU_LIEU.md`) và đúng chính sách trên. Ghi nguồn + giấy phép vào thẻ bộ và chạy `kiem_bo` trước khi dùng: lệnh này tự bắt bản ghi rơi vào tập khoá, sai ranh giới nhãn, sai chính sách nhái giọng.

**Tự do — mọi thứ còn lại:**
- **Cách làm:** mô hình, kiến trúc, đặc trưng, cách sinh dữ liệu, thứ tự các bước. Được dùng mô hình nền khác XLS-R, miễn đã huấn luyện sẵn và nằm trong ngân sách thời gian chạy.
- **Mọi họ phương pháp:** số đo tay, ảnh phổ, embedding, kết hợp, hoặc hướng chưa ai nghĩ tới. `lanes/HUONG_PHUONG_PHAP.md` là bản đồ tham khảo, không phải danh sách giới hạn.
- **Dữ liệu:** dữ liệu có sẵn chỉ là ví dụ; tự tìm, tự ghi, tự sinh đều được, miễn đúng chính sách và phủ đúng trường hợp (`DU_LIEU.md`).
- **Ý tưởng của nhóm trưởng** (`<Px>/Y_TUONG.md`: hướng đi, chặng, ý mở đầu) là gợi ý — được bỏ, đổi thứ tự, hoặc thay bằng hướng của bạn.
- **Giả thuyết của nhóm trưởng** (ví dụ nguyên nhân vu oan) chưa phải kết luận — được thách thức bằng số liệu.
- **«Đã thử, chưa hiệu quả»** chỉ đúng trong cấu hình đã ghi — được thử lại nếu có cách hoặc lý do mới (ghi rõ khác ở đâu).

**Với Claude của từng thành viên:** đừng neo vào hệ thống hiện tại, dữ liệu có sẵn hay ý tưởng của nhóm trưởng. Làm theo thứ tự: (1) hiểu mục tiêu và bài toán của mảng (`<Px>/CLAUDE.md` mục 2–3); (2) **tự** tìm 2–3 hướng (tài liệu, bài báo, mã mở mới hơn; nghĩ lại bài toán từ đầu) — **chưa** đọc các tệp ý tưởng; (3) rồi mới đọc `<Px>/Y_TUONG.md`, `Y_TUONG_CHUNG.md`, `HUONG_PHUONG_PHAP.md` để so; (4) chọn theo số đo ở ngân sách. Chỉ dừng lại hỏi khi đụng tới bốn rào trên.

Chi tiết từng rào: `lanes/CHUNG.md` mục 0, 2, 3, 4.

---

## 7. Hai con số giữ cho các phần ghép được

- **Ngân sách vu oan:** Req 4 ≤ 3 % được chia cho từng kênh — ghép, xoá: 0,5 % mỗi kênh; khử ồn, đổi giọng một đoạn: 0,5 % mỗi kênh; đầu cả tệp (khử ồn, đổi giọng): 0,25 % mỗi đầu; giọng máy: 0,25 % cho kênh `giong_may` và 0,25 % cho đầu `tts_file` (cùng cờ `--ftts`). Mỗi kênh nổ nhầm một chút thì cộng lại thành nhiều, nên không ai nới kênh mình cho đẹp số.
- **Ngân sách thời gian chạy:** cả hệ thống ≤ 0,6 giây xử lý cho mỗi giây âm thanh; phần thêm của mỗi lane P3 / P4 / P5 ≤ 0,06, của P2 ≤ 0,02 — áp **khi nộp để ghép vào sản phẩm**. Lúc khám phá được chạy chậm hơn (quyết định của nhóm trưởng: tìm hướng trước, tối ưu tốc độ sau).

Vì vậy thước đo chính của P3 / P4 / P5 là **bắt được bao nhiêu khi giữ đúng ngân sách**, không phải bắt được bao nhiêu nói chung. Muốn đổi ngân sách thì đề xuất (mục 12).

---

## 8. Bảng thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **«Ghép» — ba nghĩa** | (1) **ghép lời thật**: loại sửa của P4 (chèn / thay bằng lời người thật); (2) **ghép hệ thống**: P0 nối phần của các lane lại; (3) **ghép thử** (`ghep_thu`): công cụ thử nối trước khi nộp |
| **Mức sửa: thường · kỹ · vụng** | Trong dữ liệu tự tạo: thường = cắt giữa khoảng lặng, khớp âm lượng; kỹ = khớp âm sắc, nền, phủ tiếng nền, nén lại (khó nghe ra); vụng = cắt lẹm vào từ, không làm mượt (dễ lộ). Tên kiểu có hậu tố `_ky`, `_vung` (= vụng) |
| **Kênh** | Bộ phát hiện của một thao tác một đoạn, cho 1 điểm 0–1 mỗi 20 ms. Tên kênh: `ghep`, `xoa`, `doan_ghep` (P4); `den` = khử ồn một đoạn, `voi` = đổi giọng một đoạn (P3); `giong_may` (P5) |
| **Đầu cả tệp** | Bộ phát hiện cho thao tác phủ cả tệp, cho 1 xác suất: `nen` = khử ồn cả tệp, `giong` = đổi giọng cả tệp (P3); `tts_file` = giọng máy cả tệp (P5) |
| **Vu oan** | Tệp thật (không sửa) bị báo là đã sửa — Req 4 |
| **Ngân sách vu oan** | Phần vu oan mỗi kênh được phép, cộng lại ≤ 3 % (mục 7) |
| **Ngưỡng · điểm vận hành F** | Bộ ngưỡng đang dùng: `.tmp_v6/results/thresholds_v9F.json` |
| **lencal** | Ngưỡng chỉnh theo độ dài tệp (tệp dài có nhiều cơ hội nổ nhầm hơn): `lencal_v9F.json` |
| **OOF** (out-of-fold) | Điểm của một tệp do mô hình **chưa từng học tệp đó và người nói đó** tính ra. Dùng để chọn ngưỡng và so sánh mà không «học thuộc đề» |
| **Fold theo người nói** | Chia dữ liệu 5 phần sao cho mỗi người nói chỉ nằm trong 1 phần; mô hình phần k chỉ học từ các phần khác |
| **Baseline** | Số của hệ thống hiện tại mà lane phải vượt; có lệnh tái tạo trong tệp lane |
| **Ghép thử** (`ghep_thu`) | Công cụ cắm điểm mới của lane vào hệ thống trên OOF, xem ngay các Req thay đổi ra sao, không cần chạy lại XLS-R. Chạy trước khi nộp |
| **e2e** (đầu–cuối) | Chạy cả hệ thống trên âm thanh, từng tệp; chậm (8–16 giây / tệp) |
| **Tập chấm · tập khoá** | Tập chấm: chỉ để báo cáo. Tập khoá: phần dữ liệu giữ kín, P0 chấm một lần cuối dự án |
| **natvi · naten** | Bộ tự tạo: lấy ghi âm thật tiếng Việt / Anh rồi tự sửa bằng 8 thao tác, nên biết chắc loại và vị trí sửa |
| **scrapevi · scrapeen** | Ghi âm thật cào được (giấy phép mở, kênh không chuyên) — để đo vu oan ngoài đời |
| **longvi · longen** | Tệp dài 1–10 phút |
| **benign** | Tệp lành tính (nén, gửi qua ứng dụng, thiết bị…) — phải ra 01 |
| **Bộ âm cứng** | Các đoạn tệp thật làm kênh nổ nhầm; P2 gom lại để các lane học tránh |
| **Kho tệp thật** (kho phân vị, `refbank`) | Số đo của tệp thật, để hồ sơ viết được «cao hơn X % tệp thật» |
| **Gói chuyên gia** | Thư mục `experts/<lane>/<tên>/` một lane nộp: `card.json`, điểm OOF, trọng số, `score.py`, `signs.py` |
| **Hợp đồng H1–H6** | Định dạng dữ liệu trao đổi giữa các lane (manifest, mã nhãn, điểm khung, dòng thao tác, hồ sơ, cách chấm) — đóng băng |
| **K1 · K7 · K8 · K9** | Tên khối: kiểm điều kiện · giải mã · luật nhãn · dựng hồ sơ (mục 3) |
| **V1–V7** | Vấn đề 1–7 của cây (mục 2) |
| **XLS-R** | Mô hình nền wav2vec2 đã tinh chỉnh chống giả mạo, đóng băng, dùng chung |
| **FT-2 · LoRA** | Cách tinh chỉnh nhẹ mô hình nền; FT-2 là bản tinh chỉnh chung của P0 |
| **TCN** | Mạng nhỏ đọc chuỗi vector 20 ms để chấm điểm ranh giới / vùng |
| **fuse1 · h2** | Tên bộ điểm OOF hệ thống đang dùng: `boundary_oof_fuse1.pkl` (kênh), `heads2_h2.pkl` (đầu cả tệp) |
| **Constraint** | 24 điều kiện của tệp đầu vào (mục 4 bản đề xuất); vi phạm → 04 |
| **Vùng xám** | Kết quả lưng chừng, chưa đủ chắc → 04 |
| **LID** | Nhận dạng ngôn ngữ (chọn bộ ngưỡng tiếng Việt hay tiếng Anh) |
| **RTF** | Thời gian xử lý ÷ độ dài âm thanh — Req 15 |

---

## 9. Năm câu bạn phải trả lời được trước khi làm

1. Lane của tôi giữ **vấn đề nào** trong cây (mục 2), thao tác và kênh nào?
2. Lane tôi chịu **những Req nào**, đạt khi bao nhiêu, hiện đang bao nhiêu (mục 5 + tệp lane mục 2)?
3. **Baseline** phải vượt là số nào, tái tạo bằng lệnh gì (tệp lane mục 4 «Baseline»)?
4. **Ngân sách vu oan** của kênh tôi là bao nhiêu (mục 7)?
5. Bốn rào áp vào lane tôi cụ thể là gì, và tôi **được tự do** ở những đâu (mục 6)?

Nhờ Claude của bạn trả lời 5 câu này trước (mục 11), rồi so với tài liệu. Lệch chỗ nào thì hỏi nhóm trưởng.

---

## 10. Bạn nhận những gì

### 10.1 Tài liệu — đọc theo thứ tự

| Thứ tự | Tệp | Ai đọc | Để làm gì |
|---|---|---|---|
| 1 | `lanes/BAT_DAU.md` (tệp này) | người | hiểu toàn hệ thống |
| 2 | `VoiceAudit_TomTat.html` | người | cây vấn đề, đề tài tóm tắt |
| 3 | `VoiceAudit_6Muc.html` | người | định nghĩa bài toán đã nộp: Input, Output, Requirement, Constraint, Scope, Assumption — **không sửa** |
| 4 | `VoiceAudit_DanhGia.docx` (phần 2–5) | người + Claude | yêu cầu, metric, công thức đo |
| 5 | `lanes/CHUNG.md` | người + Claude | luật chung, luồng, hợp đồng, ngân sách, ghép thử |
| 6 | `lanes/<Px>/CLAUDE.md` | người + Claude | việc của lane bạn |
| 7 | `lanes/DU_LIEU.md` rồi `lanes/<Px>/DU_LIEU.md` | người + Claude | trường hợp dữ liệu phải phủ, dữ liệu mẫu có gì, định hướng tìm thêm, luật khi thêm dữ liệu |
| sau khi tự đề xuất hướng | `lanes/<Px>/Y_TUONG.md`, `lanes/Y_TUONG_CHUNG.md`, `lanes/HUONG_PHUONG_PHAP.md` | người + Claude | ý tưởng, thử nghiệm cũ, bản đồ phương pháp — để so, không phải để làm theo |
| tra cứu | `VoiceAudit.html` | Claude | tài liệu hệ thống đầy đủ: chỉ tiêu chi tiết, «01 phủ những gì», hiện trạng |
| tra cứu | `CAY_PHAN_RA_V2.html`, `CAY_PHAN_RA_V2_VA_PHAN_VIEC_6_NGUOI.md` (§4B: danh mục test bắt buộc) | Claude | cây 36 vấn đề con, test |
| tra cứu | `.tmp_v6/wf/DECISIONS.md` | Claude | quyết định đã chốt của nhóm trưởng — không làm trái |
| tra cứu | `lanes/DE_XUAT/README.md` | người + Claude | cách đề xuất thay đổi |

**Cây phân rã là cố định**: nó định hình bài toán và cách cả hệ thống vận hành. Các phần «hiện trạng», «cách làm», «hướng thử» trong `VoiceAudit.html` và hai tệp cây mô tả **hệ thống hiện tại (baseline)**. Đọc để hiểu, không phải đáp án. Cách giải từng vấn đề con là việc của bạn. Phần chia lane trong `CAY_PHAN_RA_V2_VA_PHAN_VIEC_6_NGUOI.md` (04/10) đã được thay bằng `lanes/`.

### 10.2 Bộ gửi và cài đặt

Nhóm trưởng gửi mỗi người **một thư mục bộ gửi** của lane mình, gồm tài liệu, mã và dữ liệu mẫu. Danh sách từng tệp nằm trong `GOI_<Px>.json` ở gốc thư mục. Mã được chọn theo đúng các tệp **thực sự chạy** khi bạn làm những việc tài liệu yêu cầu: đo, kiểm, cắm vào hệ thống.

| Phần | Ai nhận | Gồm | Để làm gì |
|---|---|---|---|
| tài liệu | mọi người | mục 10.1, toàn bộ `lanes/`, `DECISIONS.md` | hiểu hệ thống và nhiệm vụ |
| mã chung | mọi người | ghép thử, `kiem_bo`, `lockbox` (cùng 17 tệp mã chung chúng nạp) · `run_serial.sh`, `run_nolock.sh` · `requirements.txt` · `tai_mo_hinh` · ngưỡng `thresholds_v9F.json` · `lockbox_v2.json` · bảng fold `folds_v2_groups.json` · danh mục công cụ sinh `engines.json` (vai trò học / để dành — kiểm tập khoá cần) · báo cáo baseline `ghep_thu_*.md`, `e2e_v9F_*.md` | đo trên cùng một thước, kiểm nhãn và tập khoá trước khi nộp, cắm vào hệ thống |
| mã sản phẩm | P1, P2 | hệ thống chạy đầu–cuối `voiceaudit_v9.py` (K1, hồ sơ, PDF) · cấu hình · mô hình phân đoạn ONNX (trong gói dữ liệu) | P1: K1 và hồ sơ nằm trong sản phẩm · P2: chạy trên ghi âm thật |
| công cụ đo K1 | P1 | `k1_axes`, `eval_signal` | chấm K1 trên bộ ca có đáp án |
| chuỗi cào | P2 | `v6/scrape` · trạng thái cào (đã gỡ kênh tập khoá) · danh sách kênh đã duyệt · lời chép kho nghiên cứu (từ vựng cho cổng ngôn ngữ) | cào đúng như hệ thống (fold theo nhóm kênh, tập khoá) |
| hạ tầng sinh | P5 | `v6/tts_multi` (danh mục `engines.json`: vai trò học / để dành) · mô hình DNSMOS | sinh giọng máy và chuyển giọng (cả `vcmulti` cho P3) |

**Không gửi:** mã huấn luyện và mã thử nghiệm của các hướng cũ, cùng các báo cáo thử nghiệm — để không thiên lệch cách làm. Muốn xem mã baseline của mảng mình thì xin nhóm trưởng **sau bước 3** ở mục 11 (giống `Y_TUONG.md`). Mã mới của bạn đặt ở chỗ riêng (`experts/<Px>/` hoặc thư mục mã của lane) và giao qua hợp đồng (CHUNG §4). Không cần sửa mã hiện có.

**Cài đặt:**
1. Windows. Đặt thư mục bộ gửi **ở đâu cũng được** (mã tự tính gốc kho), nhưng chọn **đường ngắn**, ví dụ `D:\VoiceAudit` (dưới khoảng 90 ký tự). Bộ đệm mô hình trong kho dài thêm khoảng 160 ký tự; vượt giới hạn 260 ký tự của Windows sẽ báo «không có tệp», trừ khi bật `LongPathsEnabled`.
2. Python 3.13: `python -m venv .venv`, rồi `.venv\Scripts\python.exe -m pip install -r requirements.txt` (torch bản CPU, cùng vài thư viện ghi rõ trong tệp).
3. P1, P2, P5: tải mô hình **một lần** bằng `python -m v6.product.tai_mo_hinh` (khoảng 3,2 GB, ghim đúng phiên bản; sản phẩm chạy offline). Kiểm lại: `python -m v6.product.tai_mo_hinh --kiem`.
4. Kiểm cài đặt — mọi người: `python -m v6.product.ghep_thu selftest` · `python -m v6.data.kiem_bo selftest`, rồi `python -m v6.product.ghep_thu --name nen` (mục 10.3, phải ra đúng số). Thêm theo lane:
   - P1: `python -m v6.product.golden` · `python -m v6.product.test_ticket`;
   - P2: `python -m v6.scrape.test_scrape` · `python -m v6.product.golden`;
   - P5: `python -m v6.tts_multi.test_engines` · `python -m v6.tts_multi.test_pipeline`.
5. Việc nặng chạy qua `bash .tmp_v6/wf/run_serial.sh <lệnh>` (Git Bash). Thư mục tạm và bộ đệm đổi bằng `VA_TMP`, `VA_CACHE`. ffmpeg phải có trong PATH, hoặc đặt `VA_FFMPEG_DIR`.

### 10.3 Dữ liệu — theo lane

Âm thanh và đặc trưng nằm trong bộ gửi (nhiều GB). Gói đã **loại tập khoá** (cả âm thanh lẫn dòng siêu dữ liệu) và chỉ chứa dữ liệu các loại sửa của lane.

**Việc chính của bạn với dữ liệu là tự thu thập**: cào nguồn mở, tự ghi, tự sinh, tìm kho mở — đúng **dạng** (nhãn, định dạng) để phủ **đủ các trường hợp** của mảng (`lanes/<Px>/DU_LIEU.md` mục 1–3). Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết. Cái cố định chỉ là phạm vi phải phủ và luật khi thêm dữ liệu (`lanes/DU_LIEU.md` mục 2–3, 6).

**Gói gửi kèm chỉ là dữ liệu mẫu**, do nhóm trưởng chọn từ dữ liệu sẵn có: để hiểu định dạng, tái tạo baseline và bắt đầu nhanh. Gói **không phủ hết** phạm vi. Bạn **không bắt buộc** dùng nó, cũng không bắt buộc dùng đặc trưng XLS-R đã trích. Cột cuối là dữ liệu sẵn có khác — **gợi ý**, xin nếu thấy có ích.

| Lane | Bộ gửi cơ bản (dữ liệu mẫu + mã) | Có sẵn khác — gợi ý, xin nếu muốn |
|---|---|---|
| Mọi lane | **gói chung**: điểm OOF hệ thống, manifest natvi / naten, 2 tệp v4 mà ghép thử đọc (0,12 GB) + khung XLS-R (1,97 GB, ghép thử cần) | |
| P1 | + mã sản phẩm, công cụ đo K1, trọng số hệ thống, 60 tệp mỗi kiểu, ngoài phạm vi, `k1axes` + `suites` — **5,1 GB** | MIST 0,14 GB |
| P2 | + mã sản phẩm, chuỗi cào, trọng số hệ thống, vector cả tệp, ghi âm thật ngoài đời — **13,2 GB** | 60 tệp mỗi kiểu 0,8 GB |
| P3 | + vector XLS-R, âm thanh khử ồn / đổi giọng — **6,3 GB** | giọng tham chiếu, kho gốc sinh, bộ phủ, đặc trưng họ khác |
| P4 | + vector XLS-R, âm thanh ghép / xoá, MIST — **6,6 GB** | kho gốc sinh, bộ phủ, đặc trưng họ khác |
| P5 | + hạ tầng sinh, vector XLS-R, âm thanh giọng máy hợp lệ + ghép thật vi / en, giọng tham chiếu, kho gốc sinh — **6,5 GB** | mô hình viXTTS, đặc trưng họ khác |

Có gói chung là chạy được: `python -m v6.product.ghep_thu --name nen` phải ra đúng baseline điểm F — **en** 0,682 / 0,655 / 0,976 / 7,3 % · **vi** 0,546 / 0,491 / 0,956 / 10,1 % (Req 1 / 2 / 3 / 4, chấm lại 20/10 sau đổi nhãn). Ra khác là cài đặt hoặc dữ liệu thiếu — báo nhóm trưởng. Mô hình trong gói còn học trên nhãn cũ — đọc `CHUNG.md §1.4` trước khi so số.

---

## 11. Ngày đầu tiên

1. Đọc tệp này và `VoiceAudit_TomTat.html` (30 phút).
2. Đọc `VoiceAudit_6Muc.html` và `VoiceAudit_DanhGia.docx` phần 2–5 (30 phút).
3. Cài đặt, chạy các lệnh kiểm ở 10.2.
   Ở gốc kho có **`CLAUDE.md` của nhánh bạn**; Claude Code tự đọc tệp này khi mở kho. Tệp ghi bạn là lane nào, đọc gì trước, và phải đối chiếu với lane nào để giữ đồng nhất (ranh giới nhãn, phần nhận / nộp).
4. Mở Claude Code tại thư mục bộ gửi, gõ câu đầu (thay **mọi** chữ `P3` bằng lane của bạn):

   ```
   Tôi phụ trách P3 trong dự án VoiceAudit. Đây là phiên đầu tiên: bạn chưa có ngữ cảnh gì.
   CLAUDE.md ở gốc kho cho biết nhánh và lane của tôi.

   Đọc kỹ, hết từng tệp, theo đúng thứ tự:
   1. @VoiceAudit_6Muc.html — bản đề xuất đã nộp, gốc của mọi mục tiêu (không sửa).
   2. @VoiceAudit_TomTat.html (cây vấn đề) và VoiceAudit_DanhGia.docx phần 2–5 (metric, cách đo;
      tệp Word — trích chữ bằng python-docx có sẵn trong .venv).
   3. @lanes/BAT_DAU.md, @lanes/CHUNG.md, @lanes/P3/CLAUDE.md, @lanes/DU_LIEU.md, @lanes/P3/DU_LIEU.md.
   4. Phần ranh giới của các lane mà CLAUDE.md gốc bảo đối chiếu.
   CHƯA đọc lanes/P3/Y_TUONG.md, lanes/Y_TUONG_CHUNG.md, lanes/HUONG_PHUONG_PHAP.md.
   VoiceAudit.html và CAY_PHAN_RA_* chỉ tra cứu khi cần; phần «hiện trạng / cách làm» trong đó
   là hệ thống hiện tại, không phải đáp án.

   Rồi làm lần lượt; xong mỗi bước thì dừng, chờ tôi xác nhận:
   Bước 1 — Hiểu. (a) Tóm tắt bằng lời của bạn: bài toán, 4 nhãn, cây vấn đề, luồng xử lý,
   6 lane ghép với nhau ra sao. (b) Trả lời 5 câu ở mục 9 của BAT_DAU.md cho P3, mỗi ý ghi tệp
   và mục nguồn. (c) Cái cố định và cái tự do của lane tôi. (d) Phạm vi dữ liệu phải phủ,
   dạng dữ liệu cần thu, cách kiểm trước khi nộp. (e) Ranh giới với các lane cần đối chiếu.
   Chỗ nào tài liệu mâu thuẫn hay không rõ thì ghi ra, đừng tự đoán.
   Bước 2 — Tái tạo baseline bằng lệnh ở mục «Baseline» của tệp lane (chạy qua
   bash .tmp_v6/wf/run_serial.sh); báo số có khớp tài liệu không.
   Bước 3 — Tự đề xuất 2–3 hướng của riêng bạn (tìm tài liệu, bài báo, mã mở; nghĩ lại bài toán
   từ đầu). Mỗi hướng: lý do, dữ liệu cần thu, cách đo ở ngân sách của kênh, rủi ro.
   Bước 4 — Chỉ sau bước 3 mới đọc @lanes/P3/Y_TUONG.md, @lanes/Y_TUONG_CHUNG.md,
   @lanes/HUONG_PHUONG_PHAP.md; so với hướng của bạn (giống / khác), rồi đề xuất chọn hướng
   theo số đo. Không coi ý tưởng của nhóm trưởng là đáp án.
   Bước 5 — Ghi báo cáo đầu tiên lanes/P3/BAO_CAO_<ngày><tháng>.md: phần hiểu ở bước 1, số baseline
   ở bước 2, hướng đã chọn và lý do — để các phiên sau đọc lại.

   Dữ liệu: tự tìm, tự thu thoải mái, không cần xin duyệt — miễn đúng phạm vi dữ liệu của lane
   (DU_LIEU.md) và chính sách dữ liệu (BAT_DAU.md mục 6); ghi nguồn vào thẻ bộ, chạy kiem_bo trước khi dùng.
   Luật: không sửa tệp của lane khác, không sửa VoiceAudit_6Muc.html; việc nặng chạy qua
   run_serial.sh; dùng GPU, chạm tập khoá thì hỏi nhóm trưởng trước.
   ```

   **Các phiên sau** (không cần đọc lại từ đầu):
   ```
   Tiếp tục việc lane P3. Đọc CLAUDE.md gốc, @lanes/P3/CLAUDE.md và báo cáo mới nhất
   lanes/P3/BAO_CAO_*.md; tóm tắt đang ở đâu và việc tiếp theo, chờ tôi xác nhận rồi làm tiếp.
   ```

5. So câu trả lời bước 1 của Claude với tài liệu. Khớp thì bắt đầu; lệch thì hỏi nhóm trưởng.
6. Chọn hướng theo số đo — hướng của bạn hay ý tưởng của nhóm trưởng đều được. Ghi lý do chọn vào báo cáo đầu tiên.

**Việc không cần mã, làm được ngay:** sửa tay ≥ 150 ca (P4, P0) · ghi âm bằng thiết bị thật 30–50 tệp (P0) · gửi tệp qua ứng dụng thật (P2) · khảo sát ≥ 3 người ngoài nhóm (P1).

---

## 12. Nộp, hỏi, báo cáo

- **Sau mỗi đợt việc:** ghi `lanes/<Px>/BAO_CAO_<ngày><tháng>.md` — giả thuyết · làm gì · số trước / sau (cùng thước, cùng ngân sách) · ô nào tụt · bước tiếp.
- **Có cải tiến:** chạy ghép thử → không ô nào của lane khác tụt quá 5 điểm, không vượt ngân sách → nộp gói chuyên gia `experts/<Px>/<tên>_<phiên bản>/` (`CHUNG.md` mục 3.3, 4).
- **Muốn đổi định dạng, ngân sách, ngưỡng, chỉ tiêu:** viết `lanes/DE_XUAT/<ngày>_<Px>_<tên>.md`, nhóm trưởng duyệt.
- **Phải hỏi trước khi:** dùng GPU · xoá hoặc chuyển dữ liệu lớn · chạm tập khoá · chạy việc quá 6 giờ máy · đổi chính sách dữ liệu. Thêm nguồn / kênh / kho dữ liệu **không** cần hỏi (mục 6, «Lấy dữ liệu»).
- **Nhịp ghép:** hằng tuần nhóm trưởng ghép các gói mới, gửi lại từng lane «ô nào tụt, vì sao».
