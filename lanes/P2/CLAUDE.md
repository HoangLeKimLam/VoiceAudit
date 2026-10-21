# P2 — Nhãn 01: ghi âm thật · truyền tải · dấu vết bình thường · cổng 01

Đọc trước: `lanes/BAT_DAU.md` (tổng quan hệ thống, thuật ngữ, bốn rào — dành cho người mới) rồi `lanes/CHUNG.md` (hệ thống chung, luật cứng, ngân sách vu oan, ghép thử).

> **Tệp này chỉ nói mục tiêu, bài toán, nhiệm vụ và cách ghép — cách giải là của bạn.** Cố định: mục tiêu và phạm vi mảng, thước đo, hợp đồng, ngân sách, luật (`CHUNG.md §0`, `BAT_DAU.md` mục 6). Ý tưởng của nhóm trưởng, hướng đi gợi ý và các thử nghiệm cũ nằm riêng ở [`Y_TUONG.md`](Y_TUONG.md) — **chỉ đọc sau khi bạn đã tự đề xuất 2–3 hướng**, để khỏi bị neo theo cách nhóm đang làm.

## 1. Vị trí của bạn trong hệ thống
**Trong cây phân rã** (bản gốc: `VoiceAudit_TomTat.html` §02; cùng cây: `CAY_PHAN_RA_V2.html`; V1–V7 = «ô 1–7» trong TomTat): giữ chính **V2** (chuyên gia sàng lọc — chính là «cổng 01» trong tệp này — và kho tệp thật; P0 hợp nhất), và trong **V7** cùng P0 giữ «tệp dài và ghi âm thật». Các vấn đề con bạn giữ là **mục tiêu không được lệch** (CHUNG §1.2).

```
nguồn ghi âm thật (cào, kho mở, tự thu) ──► [chuỗi cào + QC: v6/scrape] ──► bộ tệp thật (scrapevi / scrapeen / benign)
                                                                             │
                     ┌───────────────────────────────────────────────────────┴───────────────┐
                     ▼                                                                       ▼
   bộ âm cứng (đoạn tệp thật làm kênh nổ) ──► P3 / P4 / P5 học           kho phân vị tệp thật ──► K9 (P1): "phân vị so với tệp thật"
                                                                             │
   điểm max các kênh + vector gộp ──► [cổng 01 — chưa có] ──► K7 (P0): tệp qua cổng 01 thì không báo sửa
```
Bạn là **người giữ Req 4**: mọi kênh khác chỉ được nổ trong ngân sách; bạn đo, giải thích và cung cấp dữ liệu để họ giữ được ngân sách trên **âm thanh thật ngoài đời**.

*Sơ đồ trên là hệ thống hiện tại (baseline). Cố định chỉ là chỗ cắm — nhận gì, nộp gì đúng hợp đồng (CHUNG §4) và trong ngân sách; bên trong khối của bạn làm bằng cách nào cũng được (mô hình nền khác, âm thanh gốc, số đo tay, kết hợp…).*

## 2. Đích đến
- **Req 4 ≤ 3 %** trên natvi / naten ở điểm vận hành công bố, và **vu oan trên ghi âm thật ngoài đời báo riêng** (cào, tệp dài, kênh ứng dụng, thiết bị), mỗi cơ chế lành tính ≤ 8 %.
- Cổng 01: AUC ≥ 0,98; ngưỡng sàng lọc chọn để **≤ 10 % tệp thật bị cổng báo «có dấu vết»** (đúng bản đề xuất: tệp không qua sàng lọc → 01), mà Req 2 không tụt quá 5 điểm.
- **Chỗ cắm của cổng 01:** nộp xác suất cả tệp «có dấu vết sửa» theo dạng đầu cả tệp H3 (`{"ids": [...], "P": (N, 1)}`, OOF đủ id natvi / naten để ghép thử so được); P0 cắm vào K7: tệp dưới ngưỡng cổng → 01, không báo dòng nào. Đầu vào của cổng làm gì cũng được.
- **«Giữ Req 4» nghĩa là:** bạn đo và giải thích vu oan trên ghi âm thật, cung cấp dữ liệu (âm cứng, tệp chắc chắn 01) để các lane khác giữ ngân sách, và cổng 01 là đòn bẩy trực tiếp của bạn. Thành công của P2 đo bằng: chỉ tiêu cổng 01, phần vu oan được giải thích theo cơ chế, và Req 4 hệ thống trước / sau khi cắm cổng (ghép thử).
- Kho phân vị tệp thật thay kho 160 tệp VIVOS; mọi hàng "01 phủ những gì" của `VoiceAudit.html` có ≥ 10 tệp, ≥ 5 người nói.

## 3. Hiểu bài toán của mảng

Bạn trả lời: **tệp thật có ra 01 không** — giữ Req 4 ≤ 3 % trên dữ liệu tự tạo **và trên ghi âm thật ngoài đời**; sàng lọc «có dấu vết sửa hay không» (cổng 01); cung cấp kho tệp thật làm mốc so sánh.

**Bài toán khó ở đâu (bằng chứng hiện có):**
- Vu oan trên ghi âm thật **13 %** (vi 11,1 · en 14,7), phần lớn bị gán 02 (84 / 95 tệp). Cơ chế gây ra **chưa được tách** — ứng viên: nén, khử ồn của ứng dụng, tự chỉnh mức, livestream.
- Ngưỡng theo độ dài (lencal) làm vu oan tệp 1–10 phút giảm 51 → 15 % nhưng Req 2 tụt 0,75 → 0,35 — vu oan tệp dài chưa giải xong.
- Khử ồn: **khử ồn đơn sơ có sẵn của ứng dụng ghi âm / gọi phổ thông** (WebRTC NS mức 0–2, chế độ mặc định) là **01**, dù áp cả tệp hay một đoạn — bản chất vẫn là một bản ghi bình thường. Khử ồn **chuyên nghiệp / mạnh** (ứng dụng ghi âm chuyên nghiệp có lọc ồn sâu, WebRTC NS mức 3, chế độ «Cao», công cụ khử ồn AI), bật lúc ghi hay áp sau khi ghi, là **02**. Tiêu chí là **loại ứng dụng / công cụ**, không phải cả tệp hay một đoạn (`DU_LIEU.md` mục 2.2). Dấu vết của khử ồn đơn sơ (01) và khử ồn công cụ (02) có thể giống nhau — phải tách được bằng thứ khác ngoài «có khử ồn hay không».
- Dữ liệu cào chỉ là **«01 có khả năng»**: không có đáp án chắc chắn; bộ lọc vết cắt bắt đúng 0,90 nhưng chỉ bắt được 0,10.
- Tiếng Việt vẫn vu oan 5,3 % trên tệp 01 gốc ở ngân sách 3 %; nguồn VSS có 17 % vu oan lịch sử (5 người nói chiếm 57 / 70 vu oan hàng đầu); ghi âm thật tiếng Việt chỉ 11 nhóm.
- Khử ồn mặc định của ứng dụng (01) và khử ồn bằng công cụ (02) để lại dấu vết giống nhau.
- Cổng 01 chưa có.

**Câu hỏi còn mở:** cơ chế lành tính nào gây vu oan nhiều nhất; dấu vết nào tách «bình thường» khỏi «bị sửa»; lấy đâu ra tệp **chắc chắn 01** đủ đa dạng.

## 4. Baseline phải vượt (tái tạo được — số để vượt, không phải kiến trúc phải theo)
| Hạng mục | Số | Tái tạo |
|---|---|---|
| Tệp thật → 01, OOF ngân sách 3 % | tệp 01 gốc: vi 93 % · VSASV 99 % · en 97 %; **Req 4 vi 7,0 % · en 3,8 %** — tệp 01 gốc 5,3 % · 2,6 %, 212 tệp WebRTC vừa chuyển 01 16,7 % · 13,5 % (mô hình còn học nhãn cũ, `CHUNG.md §1.4`) | `ghep_thu --name ngansach3 --f03 0.005 --freg 0.005 --fh 0.0025` |
| Tệp thật → 01, OOF điểm F | Req 4 vi 10,1 % · en 7,3 % | `ghep_thu --name nen` |
| **Ghi âm thật cào** (e2e, điểm F) | **13,0 %** có lencal (vi 11,1 · en 14,7; 84/95 tệp bị gán 02) · trước lencal 15,6 % (vi 13,7 · en 17,3); 99/114 tệp bị gán 02; 26 tệp chỉ do đầu đổi giọng cả tệp | `e2e_v9F_scrape.md`, `e2e_v9F_scrape_len.md` |
| Tệp thật dài 1–10 phút | 51 % trước lencal → **15,3 % có lencal** (en 24 % · vi 9 %; chấm một lần 19/10) | `e2e_v9F_long.md`, `e2e_v9F_long_len.md` |
| 4 kênh ứng dụng giả lập | 3,3 % (150 tệp) | `e2e_v9F.md` (13/10) |
| Bộ lành tính — **mô hình CŨ 02/10** | rớt gói 2 % → 30 % (sau luật), 5 % → 12 %, tạm dừng 23 %, cổng ồn 3 % | `rules_all.md` |
| Dữ liệu thật cào sau QC | vi **11,2 h** = 6,0 h bản cắt ngắn + 5,2 h tệp dài (56 video, 11 nhóm) · en 58,5 h = 30,4 h ngắn + 28,0 h dài (41 nhóm) — **chưa có vector XLS-R** cho phần này (muốn dùng phải tự trích) | `data/scrape/meta/hosts_stats.json` |
| Cổng 01 | **chưa có** — baseline đề xuất: hồi quy logistic trên [điểm max 6 kênh + 3 đầu] | — |

**Bất thường cần làm rõ:** ở ngân sách 0,5 % / kênh, kênh ghép + xoá vẫn vu oan 4,3 % tệp 01 gốc vi (en 1,9 %), và kênh giọng máy (nếu bật) 4,1–5,5 % vi — ngưỡng từ fold khác không giữ được ngân sách trên tiếng Việt. Chưa rõ vì sao — câu hỏi mở của P2 cùng P0 (giả thuyết của nhóm trưởng ở `Y_TUONG_CHUNG.md`).

## 5. Bạn sở hữu (chỉ P2 sửa)
Có trong bộ gửi: chuỗi cào `v6/scrape/*` (trừ `long_sets.py` của P0) và trạng thái `data/scrape/` · bộ `data_labels/v6/scrapevi`, `scrapeen`, `benign` · kho phân vị `v6/product/refbank.py` · chuyên gia mới `cong01` (cổng 01, nộp theo hợp đồng). Mã baseline cũ của bộ lành tính và luật (`build_benign`, `eval_benign`, `rules.py`) không nằm trong bộ gửi — danh sách ở `P2/Y_TUONG.md` §0, xin P0 **sau khi** tự đề xuất hướng.

*Sở hữu chỉ để không sửa chồng lên nhau khi ghép. Mã mới của bạn đặt ở chỗ riêng (ví dụ `experts/P2/`, thư mục mã mới của lane) — không bắt buộc sửa tiếp trên các tệp này.*

## 6. Luật riêng
- YouTube chỉ **kênh không chuyên**; kênh mới **tự thêm** vào whitelist, không cần duyệt, miễn không chuyên + CC BY / BY-SA / CC0; ghi lý do chọn kênh vào thẻ bộ (`fetch.py NO_NEW_FETCH` chặn các kênh chuyên nghiệp đã biết).
- **Dữ liệu cào là tập kiểm thử của cả nhóm**: khi cấp làm âm cứng / dữ liệu học, chia 5 fold người nói (`meta.person` = nhóm kênh); mô hình fold k chỉ học phần fold ≠ k. Ghi rõ trong thẻ bộ âm cứng.
- Không xoá âm gốc khi chưa được phép; gỡ = `_quarantine`.

## 7. Nộp và ghép thử
- Bộ âm cứng: `experts/P2/am_cung_v<k>/` (manifest H1 + thẻ: nguồn, fold, kênh nổ, ngày).
- Cổng 01: gói `experts/P2/cong01_<phienban>/` + báo cáo AUC / vu oan / phát hiện mất; P0 ghép vào K7.
- Bài đo T2.1–T2.8. Lệnh: e2e `python -m v6.product.e2e_test --sets benign scrapevi scrapeen --tag _p2_<ten>`, chuỗi cào `bash .tmp_v6/cache/scrape_queue.sh <buoc>`.

## 8. Phối hợp
Cấp bộ âm cứng cho **P3 / P4 / P5**, kho phân vị cho **P1**; điều tra bất thường vi cùng **P0**; đối chứng P2↔P3 (khử ồn ứng dụng = 01 ↔ khử ồn công cụ = 02).

## 9. Ý tưởng tham khảo

Hướng đi gợi ý, thử nghiệm cũ, ý mở đầu, bản đồ phương pháp của mảng: [`Y_TUONG.md`](Y_TUONG.md). Đọc **sau** khi đã tự đề xuất 2–3 hướng; dùng để so, không phải để làm theo.
