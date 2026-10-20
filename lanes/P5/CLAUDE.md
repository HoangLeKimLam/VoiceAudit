# P5 — Nhãn 03 giọng máy: một đoạn · cả tệp · máy hay thật

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ nhóm **«Có thay nội dung bằng giọng máy không? (03)»** của **V3** (giọng máy cả tệp · giọng máy một đoạn); góp cho kênh của mình ở V5 (kiểm chứng độc lập), V6 (phiếu bằng chứng — `signs.py`), V7 (bộ thích nghi riêng); chịu mốc và vùng khoanh (Req 10, 11) cho đoạn giọng máy ở V4. Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

```
tệp âm thanh ──► [ phần của P5 — làm bằng cách nào cũng được ] ──► nộp theo hợp đồng H3 (CHUNG §4):
                                                                     · điểm mỗi 20 ms: cột giong_may (đoạn giọng máy)
                                                                     · xác suất cả tệp: tts_file (cả tệp là giọng máy)
                                                                     · signs.py: ≥ 2 dấu hiệu có số cho hồ sơ
             ──► P0 (K7): vùng vượt ngưỡng → 03_chen_giong_may; dòng ghép trùng vùng máy đổi thành giọng máy (Req 6, 8a)
             ──► P1 (K8, K9): nhãn, hồ sơ                                      (kênh và đầu giọng máy hiện đang TẮT: ftts = 0)
```

**Cố định chỉ là chỗ cắm trên** — nộp gì, đúng hợp đồng, trong ngân sách. Bên trong phần của bạn dùng mô hình nền nào, đầu vào nào (âm thanh gốc, đặc trưng có sẵn, số đo tay…), học trên dữ liệu nào là tự do. Hướng không cho điểm mỗi 20 ms (ví dụ tìm ứng viên mối nối rồi phân loại) vẫn được: tự quy kết quả về điểm khung, hoặc đề xuất đổi hợp đồng (`lanes/DE_XUAT/`). Khi nộp, kèm điểm OOF trên natvi / naten để ghép thử so cùng một thước với baseline. Hệ thống hiện tại của mảng (cách baseline đang làm) ở `Y_TUONG.md` mục 0.

## 2. Đích đến
**8a F1 giọng máy một đoạn ≥ 0,86**, **8b cả tệp ≥ 0,90**, **Req 6 ≤ 5 %** ghép thật bị gọi giọng máy — ở **ngân sách 0,25 % mỗi kênh** (CHUNG §3.1), vi và en, có ≥ 2 công cụ chưa thấy (tụt ≤ 10 điểm). Khi đạt → đề xuất P0 **bật kênh**.

*Ghi chú:* Công cụ chưa thấy = công cụ bạn giữ kín khi học (ghi trong thẻ) cùng công cụ để dành của P0 (`vieneu`, `styletts2`). Req 10, 11 hiện chỉ có số chung cho mọi loại sửa — đo riêng phần của mảng là một phần việc.

## 3. Hiểu bài toán của mảng

Bạn trả lời: có **đoạn hay cả tệp giọng do máy tạo** không, và **không gọi nhầm lời thật là máy** (Req 6: ghép lời thật không bị gọi là giọng máy).

**Bài toán khó ở đâu (bằng chứng hiện có):**
- Phát hiện đoạn cao (31 / 31 tệp ở ngân sách, tiếng Việt — mẫu quá nhỏ để tin), nhưng **giọng thật tiếng Việt bị gọi là máy**: vu oan 4,1–5,5 % so với ngân sách 0,25 %; bật kênh làm Req 4 hệ thống vi 7,0 → 14,7 %. Vì vậy kênh đang tắt.
- **Dữ liệu ít**: 240 tệp giọng máy dùng được (gồm biến thể kênh; chỉ 38 bản gốc nằm trong OOF), một công cụ, ít người nói; tiếng Anh 0.
- Giọng máy mới **nhái rất giống** người thật; phải chịu được công cụ chưa thấy.
- Req 6 tiếng Việt 7,6 % (chỉ tiêu ≤ 5 %).
- Kết quả cao trên giọng máy **chưa che giấu** chưa nói gì về giọng máy đã qua nén, ứng dụng, khử ồn.

**Câu hỏi còn mở:** vì sao giọng thật tiếng Việt bị gọi là máy (nguồn? người nói? fold?); dấu hiệu nào bền qua công cụ mới và hậu xử lý.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
Tái tạo số (bật kênh ở ngân sách 0,25 %): `python -m v6.product.ghep_thu --name ngansach3_tts --f03 0.005 --freg 0.005 --fh 0.0025 --ftts 0.0025` → `.tmp_v6/results/ghep_thu_ngansach3_tts.md` (chấm lại 20/10: đã bỏ 554 tệp trái chính sách và đổi nhãn khử ồn ứng dụng).

| Ở ngân sách 0,25 % (OOF) | vi | en | trước 20/10 (vi) |
|---|---|---|---|
| giọng máy thay / chèn một đoạn | **100 % / 100 %** (18 / 13 tệp) | chưa có dữ liệu | 96 % / 97 % |
| giọng máy cả tệp | 29 % (7 tệp) | chưa có | 60 % (20 tệp) |
| **vu oan tệp 01 gốc do kênh đoạn / đầu cả tệp** (ngưỡng đặt trên fold khác để đạt 0,25 %; đo trên fold giữ lại) | **4,1 % / 5,5 %** (tính cả tệp vừa chuyển 01 — số in trong `ghep_thu`: 4,0 % / 5,2 %) | 0,2 % / 1,0 % (`ghep_thu`: 0,2 % / 0,9 %) | 3,7 % / 4,7 % |
| Req 6 · 8a | **7,6 %** · 0,17 | 1,4 % · — | 6,9 % · 0,36 |
| Req 4 hệ thống khi bật (OOF, ngân sách 3 %) | vi 7,0 → **14,7 %** | en 3,8 → 4,8 % | 5,2 → 12,5 % |

Các số trên là **khi bật kênh** trên OOF. Trong hệ thống đang chạy kênh tắt, nên bảng chung (`CHUNG.md §1.3`) ghi 8a «chưa đo», 8b 0 %. `--ftts` đặt ngân sách cho **cả** kênh `giong_may` (ngưỡng `ttsr`) và đầu `tts_file` (ngưỡng `ttsf`). F1 8a chỉ 0,17 dù bắt đủ 31 đoạn — phần mất do báo nhầm trên tệp thật hay do mốc / vùng là một câu hỏi mở. **Số dương chỉ có 38 tệp** sau khi bỏ 554 tệp trái chính sách (`DU_LIEU.md` mục 4) — khoảng tin cậy rất rộng; mô hình baseline vẫn đã học trên các tệp đó. Không dùng chúng để học mới.

**Đọc baseline**: phát hiện giọng máy trên mẫu nhỏ thì mạnh; vấn đề là **giọng thật tiếng Việt bị gọi là máy** (ngưỡng từ fold khác không giữ được ngân sách; bật kênh ở fold 0 làm vu oan tệp thật vi 14 → 42 %, số trước 20/10). Dữ liệu giọng máy dùng được chỉ 240 tệp vi (một công cụ, ít người nói — `DU_LIEU.md`); tiếng Anh chưa có. Bộ `ttsmulti_vi`, `ttsmulti_en`, `vcmulti` mới có thư mục làm việc (hàng WP7b 78 việc, GPU ~11,4 h).

## 5. Bạn sở hữu (chỉ P5 sửa)
Phần hệ thống bạn chịu khi ghép (qua hợp đồng H3, CHUNG §4): kênh khung `giong_may` (cột 5, ngưỡng `ttsr`) và đầu cả tệp `tts_file` (2, ngưỡng `ttsf`); dữ liệu nhãn 03 giọng máy (`data_labels/v6/ttsmulti_vi`, `ttsmulti_en`). Có trong bộ gửi: hạ tầng sinh chung `v6/tts_multi` (kể cả sinh `vcmulti` theo yêu cầu của P3 — dữ liệu chuyển giọng AI là nhãn 02, P3 dùng và đo; quyết định 18). Bạn nộp **chuyên gia riêng** theo hợp đồng, P0 ghép. Mã sinh cũ (`vaudit/tts.py`, `build_tiers.make_tts`, `.tmp_v6/wf/tts_*.py`) không nằm trong bộ gửi — xin P0 **sau khi** tự đề xuất hướng.

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P5/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng (chặt — rủi ro pháp lý / đạo đức)
- **Chỉ nhái giọng người nói của kho nghiên cứu công khai**: VIVOS, VoxVietnam, EdAcc, AMI, VoxPopuli, LibriSpeech. **Không** Vietnam-Celeb, **không** người trong dữ liệu cào, **không** VSS / VSASV / viVoice. Mẫu giọng từ **bản ghi khác** của cùng người; trần mỗi người 6 % số tệp của một bộ. Không dùng mô hình cấm nhái khi chưa đồng ý (vd. `hynt/F5-TTS-Vietnamese-ViVoice`). Không phát tán âm thanh sinh ra.
- **GPU đang hoãn** — phải được người dùng cho phép; cần 6–7 GB bộ nhớ trống, chạy một mình.
- So sánh luôn ở ngân sách §3.1 bằng `ghep_thu --ftts 0.0025`.

## 7. Nộp và ghép thử
Gói `experts/P5/<ten>_<phienban>/` + `python -m v6.product.ghep_thu --name p5_<ten> --col ttsr=experts/P5/<ten>/oof_<ten>.pkl:0 [--head ttsf=experts/P5/<ten>/heads_<ten>.pkl:0] --f03 0.005 --freg 0.005 --fh 0.0025 --ftts 0.0025`. Đạt theo CHUNG §3.3 **và** Req 4 hệ thống khi bật không vượt mức khi tắt. Bài đo T5.1–T5.7.

## 8. Phối hợp
Đối chứng P4↔P5 (ghép thật), P3↔P5 (đổi giọng AI); xin **P0** xếp hàng GPU; `signs.py` cho **P1**.

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
