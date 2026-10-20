# P3 — Nhãn 02: khử ồn · đổi giọng (cả tệp · một đoạn)

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ nhóm **«Có xử lý lời có sẵn không? (02)»** của **V3** (khử ồn cả tệp · đổi giọng cả tệp · khử ồn một đoạn · đổi giọng một đoạn); góp cho kênh của mình ở V5 (kiểm chứng độc lập), V6 (phiếu bằng chứng — `signs.py`), V7 (bộ thích nghi riêng); chịu mốc và vùng khoanh (Req 10, 11) cho đoạn 02 ở V4. Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

```
tệp âm thanh ──► [ phần của P3 — làm bằng cách nào cũng được ] ──► nộp theo hợp đồng H3 (CHUNG §4):
                                                                     · điểm mỗi 20 ms: cột den (khử ồn một đoạn), voi (đổi giọng một đoạn)
                                                                     · và / hoặc xác suất cả tệp: nen (khử ồn cả tệp), giong (đổi giọng cả tệp)
                                                                     · signs.py: ≥ 2 dấu hiệu có số cho hồ sơ
             ──► P0 (K7): đặt ngưỡng, khoanh đoạn 0,4–8 s, gộp dòng ──► P1 (K8, K9): nhãn, mức độ, hồ sơ
```

**Cố định chỉ là chỗ cắm trên** — nộp gì, đúng hợp đồng, trong ngân sách. Bên trong phần của bạn dùng mô hình nền nào, đầu vào nào (âm thanh gốc, đặc trưng có sẵn, số đo tay…), học trên dữ liệu nào là tự do. Hướng không cho điểm mỗi 20 ms (ví dụ tìm ứng viên mối nối rồi phân loại) vẫn được: tự quy kết quả về điểm khung, hoặc đề xuất đổi hợp đồng (`lanes/DE_XUAT/`). Khi nộp, kèm điểm OOF trên natvi / naten để ghép thử so cùng một thước với baseline. Hệ thống hiện tại của mảng (cách baseline đang làm) ở `Y_TUONG.md` mục 0.

## 2. Đích đến (khi nào lane xong)
- Mỗi thao tác 02 đạt chỉ tiêu ở **ngân sách 0,5 % mỗi kênh vùng, 0,25 % mỗi đầu** (CHUNG §3.1), cả vi và en: khử ồn cả tệp F1 ≥ 0,90; khử ồn đoạn TPR ≥ 85 %; đổi giọng đoạn TPR ≥ 80 %; đổi giọng cả tệp AUC ≥ 0,95; **Req 5 ≥ 92 %**; tập công cụ chưa thấy tụt ≤ 10 điểm.
- Vu oan kênh của bạn trên **ghi âm thật ngoài đời** (phần dev của dữ liệu cào, đúng fold) cũng nằm trong ngân sách — không chỉ trên natvi / naten.

*Ghi chú:* Tập công cụ chưa thấy = công cụ bạn giữ kín khi học (≥ 1 khử ồn, ≥ 1 đổi giọng; ghi trong thẻ) cùng công cụ để dành của P0 (`resenh` khử ồn AI, `knnvc` chuyển giọng). Req 10, 11 hiện chỉ có số chung cho mọi loại sửa — đo riêng phần của mảng là một phần việc.

## 3. Hiểu bài toán của mảng

Bạn trả lời: tệp có bị **xử lý lời có sẵn** không — khử ồn hay đổi giọng, cả tệp hay một đoạn, ở đâu — mà **không kết tội tệp thật** và **không kết tội khử ồn của ứng dụng** (01).

**Bài toán khó ở đâu (bằng chứng hiện có):**
- Kênh 02 một đoạn là **nguồn vu oan số 1** trên ghi âm thật: trong 114 tệp vu oan có 52 dòng đổi giọng đoạn, 38 dòng khử ồn đoạn; 25 tệp chỉ do đầu đổi giọng cả tệp (Zoom, livestream).
- **Khử ồn cả tệp** khó: ở ngân sách chỉ bắt 14 % vi / 16 % en. Đổi giọng một đoạn tiếng Việt 36 % (kỹ 31 %).
- **Ranh giới khử ồn ứng dụng (01) / chuyên nghiệp (02)** vừa chốt lại (20/10): WebRTC NS mức ≤ 2 nay là 01. Mô hình hiện có học theo nhãn cũ, nên vẫn chấm các tệp này là 02 (xem mục 4).
- Req 5: khử ồn vô hại không bị kết tội ≥ 92 % (hiện 89 / 96 %), trong khi dấu vết khử ồn ứng dụng và khử ồn công cụ giống nhau.
- Công cụ ngoài đời (khử ồn AI, ứng dụng đổi giọng) khác bộ dùng sinh dữ liệu hiện có; phải chịu được công cụ chưa thấy (tụt ≤ 10 điểm).
- Mốc và vùng khoanh cho đoạn 02 (Req 10, 11) còn thấp.

**Câu hỏi còn mở:** dấu vết nào của 02 còn lại sau nén và ứng dụng; bắt khử ồn cả tệp thế nào mà không kết tội tệp ghi qua ứng dụng; ranh giới 2 nửa cung tách bằng gì.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
Tái tạo số: `python -m v6.product.ghep_thu --name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025` → `.tmp_v6/results/ghep_thu_ngansach3.md`. Bảng «% đúng theo kiểu» trong tệp đó gộp theo **tên kiểu**; số dưới đây tách theo **nhãn** (chấm lại 20/10, mô hình còn học nhãn cũ).

| % phát hiện đúng ở ngân sách (OOF) | vi | en | trước đổi nhãn (vi / en) |
|---|---|---|---|
| khử ồn cả tệp, nhãn 02 | **14 %** (358 tệp) | **16 %** (339) | 16 % / 31 % |
| khử ồn một đoạn, nhãn 02 | **9 %** (310) | **22 %** (322) | 69 % / 76 % |
| tệp WebRTC NS mức ≤ 2 nay là 01 — gán đúng 01: một đoạn / cả tệp | 78 % / 97 % | 81 % / 92 % | trước là 02 (kiểu `nen_cuc_bo_ky` — toàn bộ là loại này) |
| đổi giọng cả tệp `giong_toan_cuc` | 49 % | 58 % | 47 % / 56 % |
| đổi giọng một đoạn / kỹ | **36 % / 31 %** | 86 % / 87 % | 45 % / 41 % · 86 % / 90 % |
| đổi giọng AI (vi) | 81 % | — | 81 % |
| vu oan tệp 01 gốc: den · voi · nen · giong | 0 · 0,3 · 0,3 · 0,5 % | 0 · 0,2 · 0 · 0,5 % | |
| vu oan trên 212 tệp vừa chuyển 01: den · nen | 13,8 · 0,7 % | 9,5 · 4,1 % | |

**Đọc bảng:** khử ồn một đoạn rơi mạnh không phải vì kênh kém đi. 212 tệp WebRTC vừa chuyển sang 01 vào tập «tệp thật» dùng đặt ngưỡng; mô hình cũ chấm chúng cao, nên ngưỡng `den` bị đẩy lên — vu oan `den` trên tệp 01 gốc về 0, phát hiện cũng rơi theo. Học lại trên nhãn mới (P0, hoặc chính mô hình của bạn) sẽ đổi baseline này — luôn so cải tiến với baseline **cùng bộ nhãn**.

Trên ghi âm thật ngoài đời (e2e, điểm F 3 %): kênh của bạn là **nguồn vu oan số 1** — 114 tệp vu oan có 52 dòng đổi giọng đoạn, 38 dòng khử ồn đoạn, và **25 tệp chỉ do đầu đổi giọng cả tệp** (Zoom / livestream lệch miền); tệp dài: 91 + 47 dòng.

## 5. Bạn sở hữu (chỉ P3 sửa)
Phần hệ thống bạn chịu khi ghép (qua hợp đồng H3, CHUNG §4): kênh khung `den` (cột 2) và `voi` (cột 3) trong OOF điểm khung; đầu cả tệp `nen` (0) và `giong` (1) trong OOF đầu cả tệp; dữ liệu nhãn 02 (`P3/DU_LIEU.md`). Bạn nộp **chuyên gia riêng** theo hợp đồng, P0 ghép — không cần sửa mã hệ thống. Mã baseline hiện có của các phần này không nằm trong bộ gửi: danh sách ở `P3/Y_TUONG.md` §0, xin P0 **sau khi** tự đề xuất hướng.
Dữ liệu chuyển giọng AI `data_labels/v6/vcmulti` (nhãn 02): **bạn dùng và đo**; P5 giữ hạ tầng sinh chung (`v6/tts_multi`) và sinh theo yêu cầu của bạn (quyết định 18).

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P3/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng
- **Khử ồn:** **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi phổ thông** (WebRTC NS mức 0–2, chế độ mặc định) là **01**, dù áp cả tệp hay một đoạn — bản chất vẫn là một bản ghi bình thường. Khử ồn **chuyên nghiệp / mạnh** (ứng dụng ghi âm chuyên nghiệp có lọc ồn sâu, WebRTC NS mức 3, chế độ «Cao», công cụ khử ồn AI), bật lúc ghi hay áp sau khi ghi, là **02**. Tiêu chí là **loại ứng dụng / công cụ**, không phải cả tệp hay một đoạn (`DU_LIEU.md` mục 2.2) — phần của bạn không được gọi khử ồn đơn sơ của ứng dụng là 02. Đã chốt 20/10: WebRTC NS mức 0–2 = 01, mức 3 = 02; thang cho các ứng dụng khác còn tạm, chờ đo trên ứng dụng thật (`DU_LIEU.md` mục 2.2, 7).
- Âm cứng từ ghi âm thật dùng **đúng fold**. Giữ kín ≥ 1 bộ khử ồn và ≥ 1 bộ đổi giọng làm tập chưa thấy.
- So sánh luôn ở **ngân sách §3.1**, bằng `ghep_thu`; không tự đổi ngân sách.

## 7. Nộp và ghép thử
Gói `experts/P3/<ten>_<phienban>/` (CHUNG §4) + chạy:
`python -m v6.product.ghep_thu --name p3_<ten> --col den=experts/P3/<ten>/oof_<ten>.pkl:0 --col voi=experts/P3/<ten>/oof_<ten>.pkl:1 [--head giong=...:0] --f03 0.005 --freg 0.005 --fh 0.0025`
Đạt khi đúng tiêu chí CHUNG §3.3. Bài đo T3.1–T3.6 (`CAY_PHAN_RA_V2… §4B`).

## 8. Phối hợp
Nhận bộ âm cứng từ **P2**; đối chứng P3↔P5 (đổi giọng AI ↔ giọng máy), P2↔P3 (khử ồn ứng dụng ↔ nén / thiết bị); nộp `signs.py` cho **P1**.

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
