# P4 — Nhãn 03 lời thật: ghép / chèn · xoá lời

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ nhóm **«Có thay nội dung bằng lời thật không? (03)»** của **V3** (mối ghép lời thật · mối xoá lời · vùng đoạn ghép); góp cho kênh của mình ở V5 (kiểm chứng độc lập), V6 (phiếu bằng chứng — `signs.py`), V7 (bộ thích nghi riêng); chịu mốc và vùng khoanh (Req 10, 11) cho đoạn 03 lời thật ở V4. Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

```
tệp âm thanh ──► [ phần của P4 — làm bằng cách nào cũng được ] ──► nộp theo hợp đồng H3 (CHUNG §4):
                                                                     · điểm mỗi 20 ms: cột ghep (mối ghép / chèn), xoa (mối xoá lời),
                                                                       doan_ghep (vùng đoạn ghép — tùy chọn)
                                                                     · signs.py: ≥ 2 dấu hiệu có số cho hồ sơ
             ──► P0 (K7): đỉnh vượt ngưỡng → dòng 03 (thay / chèn, xoá quanh mối nối) ──► P1 (K8, K9): nhãn, hồ sơ
```

**Cố định chỉ là chỗ cắm trên** — nộp gì, đúng hợp đồng, trong ngân sách. Bên trong phần của bạn dùng mô hình nền nào, đầu vào nào (âm thanh gốc, đặc trưng có sẵn, số đo tay…), học trên dữ liệu nào là tự do. Hướng không cho điểm mỗi 20 ms (ví dụ tìm ứng viên mối nối rồi phân loại) vẫn được: tự quy kết quả về điểm khung, hoặc đề xuất đổi hợp đồng (`lanes/DE_XUAT/`). Khi nộp, kèm điểm OOF trên natvi / naten để ghép thử so cùng một thước với baseline. Hệ thống hiện tại của mảng (cách baseline đang làm) ở `Y_TUONG.md` mục 0.

## 2. Đích đến
**Req 7a F1 ghép ≥ 0,80**, **7b F1 xoá ≥ 0,65**, mốc ≤ 500 ms ≥ 78 % (Req 10), IoU ≥ 0,80 (Req 11) — ở **ngân sách 0,5 % mỗi kênh** (CHUNG §3.1), cả vi và en; không nhóm điều kiện nào < 60 % mức gốc (mức gốc = tỉ lệ phát hiện trên tệp không áp điều kiện); tập sửa tay ≥ 60 % (báo riêng).

*Ghi chú:* Req 10, 11 hiện chỉ có số chung cho mọi loại sửa — đo riêng phần của mảng là một phần việc.

## 3. Hiểu bài toán của mảng

Bạn trả lời: lời có bị **chèn, thay hoặc xoá bằng lời người thật** không, và **mối nối ở đâu**.

**Bài toán khó ở đâu (bằng chứng hiện có):**
- **Xoá lời** rất yếu: F1 0,26 / 0,44 ở điểm F, 0,20 / 0,35 ở ngân sách — xoá chỉ để lại một mối nối ngắn, không có đoạn lạ.
- **Ghép kỹ** tiếng Việt chỉ 25–31 %.
- **Tệp dài**: khi ngưỡng tính theo độ dài, 03 đúng còn 4–7 %, xoá 0 % — điểm ở chỗ sửa phải vượt điểm cao nhất của cả một tệp thật dài.
- **Vang** là chế độ khó nhất (đo bằng mô hình cũ: RT60 0,3 / 0,6 / 1,0 s → 10 / 2 / 0 %).
- Vu oan tiếng Việt do kênh ghép + xoá 4,3 % trên tệp 01 gốc, vượt ngân sách — chưa rõ vì sao.
- Chưa có ca **sửa tay thật**; dữ liệu máy sinh có thể khác cách người sửa thật.

**Câu hỏi còn mở:** dấu hiệu nào của mối nối còn lại sau nén và vang; tách mối nối sửa khỏi ngắt tự nhiên (tạm dừng, đổi người nói) bằng gì.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
Tái tạo: `python -m v6.product.ghep_thu --name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025` → `.tmp_v6/results/ghep_thu_ngansach3.md` (chấm lại 20/10 sau đổi nhãn; số của bạn gần như không đổi so với trước).

Mức sửa trong dữ liệu tự tạo: **thường** — cắt giữa khoảng lặng, khớp âm lượng, làm mượt; **kỹ** — khớp âm sắc, nền và âm lượng, phủ tiếng nền, nén lại cả tệp (khó nghe ra); **vụng** — cắt lẹm vào từ, không làm mượt, lệch âm lượng (dễ lộ).

| % phát hiện đúng ở ngân sách (OOF) | vi | en |
|---|---|---|
| ghép thay: thường / kỹ / vụng | 51 % / **25 %** / 56 % | 74 % / 58 % / 88 % |
| ghép chèn: thường / kỹ / vụng | 51 % / **31 %** / 82 % | 77 % / 63 % / 90 % |
| **xoá lời: thường / kỹ / vụng** | **23 % / 6 %** / 20 % | **30 % / 17 %** / 42 % |
| đa thao tác: ghép + đổi giọng / ghép + khử ồn / hai chỗ | 32 / 24 / 53 % | 49 / 62 / 92 % |
| **7a · 7b** | **0,598 · 0,197** | **0,784 · 0,348** |
| vu oan tệp 01 gốc do kênh ghép + xoá | **4,3 %** (ngân sách hai kênh cộng lại 1 %) | 1,9 % |

- Ghi âm thật (e2e, điểm F): ghép thay / chèn / xoá vi 60 / 58 / **15 %**, en 82 / 90 / 45 %. Tệp 1–10 phút vi 32 / 28 / 21 %, en 100 / 93 / 56 % → **có lencal (chấm một lần 19/10): vi 4 / 4 / 0 %, en 7 / 20 / 0 %** — phần lớn số cũ là vu oan trùng may (68 % tệp thật en bị báo). Tệp dài là phép thử thật: điểm ở chỗ sửa phải vượt điểm lớn nhất của cả tệp thật (`e2e_v9F_long_len.md`).
- Điều kiện (đo bằng mô hình CŨ 02/10 — cần đo lại): vang RT60 0,3 / 0,6 / 1,0 s → 10 / 2 / 0 %; TV 8 % → 22 %; chồng điều kiện → 6 %.
- **Vu oan vi 4,3 % ở ngân sách 0,5 %/kênh là bất thường** (không phải bão hoà điểm): chưa rõ vì sao (giả thuyết của nhóm trưởng ở `Y_TUONG_CHUNG.md`) — làm rõ cùng P0 + P2 trước khi tin số vi.

## 5. Bạn sở hữu (chỉ P4 sửa)
Phần hệ thống bạn chịu khi ghép (qua hợp đồng H3, CHUNG §4): kênh khung `ghep` (cột 0), `xoa` (cột 1), `doan_ghep` (cột 4, `fspl = 0` → đang tắt) trong OOF điểm khung; dữ liệu nhãn 03 lời thật (`P4/DU_LIEU.md`). Giải mã đỉnh thành dòng thao tác (`v6/boundary/detect.py decode_file`, có trong bộ gửi) ảnh hưởng toàn hệ thống — muốn đổi thì đề xuất P0. Bạn nộp **chuyên gia riêng** theo hợp đồng, P0 ghép. Mã baseline hiện có (mô hình ranh giới, sinh dữ liệu 03) không nằm trong bộ gửi: danh sách ở `P4/Y_TUONG.md` §0, xin P0 **sau khi** tự đề xuất hướng.

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P4/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng
- Chỉnh sửa 1 âm tiết → 1 câu, chèn ≤ 5 s, cắt tại ranh giới từ; nội dung sau sửa mạch lạc (Assumption).
- Âm cứng từ ghi âm thật dùng đúng fold. **Không học trên tập sửa tay** — chỉ chấm, báo riêng.
- So sánh luôn ở ngân sách §3.1 bằng `ghep_thu`.

## 7. Nộp và ghép thử
Gói `experts/P4/<ten>_<phienban>/` + `python -m v6.product.ghep_thu --name p4_<ten> --col ghep=experts/P4/<ten>/oof_<ten>.pkl:0 --col xoa=experts/P4/<ten>/oof_<ten>.pkl:1 --f03 0.005 --freg 0.005 --fh 0.0025`. Đạt theo CHUNG §3.3. Bài đo T4.1–T4.9.

## 8. Phối hợp
Âm cứng từ **P2**; đối chứng P4↔P5 (Req 6: ghép thật không bị gọi giọng máy); `signs.py` cho **P1**.

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
