# Cây phân rã V2 và phân việc 6 người song song (04/10/2026)

Nguồn gốc: cây **K1–K9 · 3 tầng · 12 artifact** trong `VoiceAudit.html` (mục "Decomposition") và sáu mục đã nộp (Input · Output · Requirement · Constraint · Scope · Assumption). V2 **giữ khung đó**, cập nhật theo hệ thống đang chạy (v9) và chia việc theo **nhãn × trường hợp**, mỗi người làm trọn một lát từ dữ liệu tới mô hình. Tệp `CAY_PHAN_RA_PHU_TOAN_MIEN.md` giữ làm nhật ký số đo.
Đọc theo thứ tự: §1 cây V2 (gốc là vấn đề ban đầu; §1.1 luồng xử lý hiện thực) · §2 ai giữ ô nào · §3 để không ai chờ ai · §4 sáu lane · §5 ghép · §6 sổ phủ sáu mục.

## 0. Ba giai đoạn và nguyên tắc "tự đủ"

| Giai đoạn | Làm gì | Ai |
|---|---|---|
| **GĐ0 · khảo sát khả thi** (đang chạy) | chạy thử cả hệ thống để nhóm trưởng nắm: cái gì làm được, yếu ở đâu, số đo nền; **không** phải sản phẩm cuối | P0 |
| **GĐ1 · mỗi người một vấn đề, tự đủ** | mỗi lane **tự tìm dữ liệu, tự sinh, tự huấn luyện, tự hiệu chỉnh, tự chấm** cho đủ nhãn và đủ trường hợp của phần mình (§4, §4B); số đo nền của GĐ0 chỉ làm mốc so sánh | cả 6 người, song song |
| **GĐ2 · ghép** | nạp các "gói chuyên gia" vào khung chung, chạy test ăn khớp (§5), chấm hệ thống | P0 dẫn, mọi lane sửa phần của mình |

"Tự đủ" nghĩa là: lane có **bộ dữ liệu riêng** (manifest + tag riêng), **huấn luyện riêng** (đã huấn luyện sẵn, không học từ đầu), **ngưỡng riêng**, **test riêng** và **chạy được khi hàng xóm chưa xong** (§3). Hệ thống v9 + tinh chỉnh hiện có là *mẫu* để mỗi lane đối chiếu và là *khung ghép*; lane được phép thay hoàn toàn cách làm miễn đúng hợp đồng.

## 1. Cây V2 — gốc là vấn đề ban đầu (sửa 19/10/2026)

Tầng cao nhất là **vấn đề của người nhận tệp**, không phải luồng xử lý. Mỗi vấn đề con có câu hỏi kỹ thuật, chỉ tiêu đo nó, người giữ, và khâu hiện thực (K1–K9) ghi trong ngoặc. Bản tương tác: `CAY_PHAN_RA_V2.html`.

```
VẤN ĐỀ BAN ĐẦU: Có tin được tệp ghi âm này không?
  (→ tệp mang dấu vết của thao tác nào, ở đâu, mức độ nào; hoặc vì sao chưa đủ cơ sở kết luận)
├─ V1 Có đủ cơ sở để xét không? ..................................... P1        [Constraint · tệp bị loại báo riêng theo lý do]
│    điều kiện tệp · đo tín hiệu · ước lượng điều kiện · người nói và phòng · quyết định 04        (K1)
├─ V2 Tệp có bị chỉnh sửa không? .................................... P2 · P0   [Req 2, 3, 4]
│    cổng 01 (sàng lọc: có dấu vết hay 01) · kho tệp thật                                           (K5)
├─ V3 Nếu có, sửa kiểu gì? ............................................ P3 P4 P5  [Req 1]
│    ├─ V3a xử lý lời có sẵn (02): khử ồn / đổi giọng × cả tệp / một đoạn ...... P3  [Req 5]       (K3, K4)
│    ├─ V3b thay nội dung bằng lời thật (03): ghép / chèn · xoá lời ............ P4  [7a, 7b]      (K4)
│    └─ V3c thay nội dung bằng giọng máy (03): một đoạn · cả tệp ............... P5  [6, 8a, 8b]  (K3, K4)
├─ V4 Sửa ở đâu, phạm vi nào? ........................................ P0 · P1   [Req 10, 11: P3/P4/P5 chịu cho đoạn của mình]
│    đặt ngưỡng từng thao tác / ngân sách vu oan · giải mã dòng · gộp dòng · vùng soi · hai kênh   (K7, K8)
├─ V5 Vô hại hay làm sai lệch? Kết luận đã đủ chắc chưa? .............. P1 · P0   [Req 1, 9, 14]
│    luật nhãn · mức độ theo loại · kiểm chứng độc lập · nhiều thao tác, vùng xám · trình tự (mở)  (K6, K7, K8)
├─ V6 Người nhận có hiểu và kiểm lại được không? ...................... P1        [Req 12, 13]
│    phiếu bằng chứng · ảnh · giải thích, bảng, PDF · khảo sát người ngoài                          (K9)
└─ V7 Có dùng được ngoài đời không? ................................... P0        [Req 15 · tệp dài, ghi âm thật]
     biểu diễn chung đo một lần · tốc độ · tệp dài và ghi âm thật · kiểm thử và tập khoá            (K2)
```

Vì sao 7 vấn đề con: V1–V5 là bốn câu người nhận tệp cần biết (có sửa không · sửa kiểu gì · ở đâu · vô hại hay sai lệch) cộng điều kiện tiên quyết "có đủ cơ sở không"; V6, V7 là điều làm câu trả lời **dùng được** (hiểu và kiểm lại được; kịp thời, đúng ngoài đời). Nguồn: `TRUU_TUONG_HOA_VA_TONG_QUAT_HOA_VoiceAudit.md` §0.1, A1.

### 1.1 Góc nhìn hiện thực — luồng xử lý K1–K9 (cách hệ thống trả lời các vấn đề con trên)

```
(1) FILE ──► TẦNG 1 · ĐO MỘT LẦN ──► TẦNG 2 · PHÁT HIỆN ──► TẦNG 3 · QUYẾT ĐỊNH VÀ HỒ SƠ ──► (9)(10)(11)(12)
             K1 Kiểm điều kiện          K3 Toàn cục (cả file)       K7 Cấp 2 + kiểm chứng độc lập
             K2 Biểu diễn               K4 Khung 20 ms (đoạn)       K8 Nhãn tệp
                                        K5 Cổng 01                  K9 Phiếu · ảnh · giải thích · PDF
                                        K6 Trình tự (hướng mở)
```
12 artifact giữ nguyên số: (1) tệp · (2) báo cáo điều kiện · (3) kho đặc trưng · (4) xác suất toàn cục · (5) điểm khung · (6) điểm cổng 01 · (7) dòng cấp 2 ứng viên · (8) kết quả kiểm chứng · (9) phiếu thao tác + nhãn · (10) phiếu bằng chứng · (11) ảnh · (12) hồ sơ.

| Khâu | Bản v4 trong HTML | Hệ thống hiện chạy (v9) | V2 phải đạt | Chủ |
|---|---|---|---|---|
| **K1** kiểm điều kiện | cổng cứng + cờ cảnh báo | container + đo tín hiệu + bộ học SNR / nền / RT60 | **24 hàng Constraint** mỗi hàng một test chủ động; thêm loại âm thanh, kiểu nói, số người nói, hai kênh | P1 |
| **K2** biểu diễn | tiêu chí tay + 5 backbone | XLS-R đóng băng (khung L12 PCA 64, vector gộp) + 6 tầng LoRA | một khung chung; mỗi lane có bộ thích nghi riêng trong ngân sách RTF; tiêu chí tay chỉ còn ở dấu hiệu và kiểm chứng | P0 (khung) · lane (bộ thích nghi) |
| **K3** toàn cục | rừng ngẫu nhiên | đầu trên vector gộp thô (khử ồn, đổi giọng, giọng máy cả file) | AUC ≥ 0,95 mỗi đầu, hai ngôn ngữ | P3 · P5 |
| **K4** khung | HGB + 1D-CNN 5 lớp | TCN đóng băng ⊕ TCN tinh chỉnh, 6 kênh | mỗi kênh một chủ (§2); định vị ≤ 500 ms, IoU ≥ 0,80 | P3 · P4 · P5 |
| **K5** cổng 01 | hồi quy logistic, ≤ 10 % file sạch bị báo | **chưa có** (chỉ ngưỡng từng chuyên gia) | dựng lại: AUC ≥ 0,98, ≤ 10 % tệp thật qua cổng, 8/8 cấu hình nén ra 01 | P2 (chuyên gia) · P0 (hợp nhất) |
| **K6** trình tự | hướng mở | — | giữ hướng mở (ngoài output) | P0 (backlog) |
| **K7** cấp 2 + kiểm chứng | ngưỡng FPR ≤ 10 %, NMS, bộ kiểm chứng độc lập | ngưỡng theo ngôn ngữ, `rowmerge`, **chưa có bộ kiểm chứng độc lập** | mọi dòng đoạn phải được một bộ nhìn **khác** (tiêu chí tay trên đoạn) xác nhận; hạ về 04 / bỏ theo luật | P0 (khung) · lane (kiểm chứng viên) |
| **K8** nhãn tệp | luật suy từ dòng cấp 2 | đủ luật (1)–(5), 8 thao tác | giữ; test luật | P1 |
| **K9** phiếu · ảnh · PDF | 6 hàm dựng hình | hồ sơ 8 thao tác, ảnh, ≥ 2 dấu hiệu, #bind | dấu hiệu theo kho thật tự nhiên; khảo sát Req 13; PDF, hai kênh | P1 |

## 2. Ai giữ ô nào — mỗi nhãn, mỗi trường hợp có đúng một chủ

| Lane | Người | Nhãn / thao tác giữ | Kênh điểm (hợp đồng H3) |
|---|---|---|---|
| **P0** | Nhóm trưởng | tệp **nhiều thao tác**; vùng xám → 04; nhãn tệp khi xung đột | hợp nhất, không có kênh riêng |
| **P1** | Khung tất định | **04** do Constraint (24 hàng); mọi **Output** (nhãn, lý do, ảnh, giải thích, bảng, PDF, 12 cảnh báo) | không |
| **P2** | Nhãn 01 | **01**: mọi biến đổi đồng đều (nén, truyền tải ứng dụng, mặc định thiết bị, đổi định dạng, tốc độ ±5 %, cắt đầu–cuối, đổi âm lượng); ghi âm thật; kho tham chiếu | `cong01` |
| **P3** | Nhãn 02 | khử ồn cả file · khử ồn một đoạn · đổi giọng cả file · đổi giọng một đoạn (+ mức vô hại / đáng ngờ / sai lệch) | `khu_on` `doi_giong` · đầu `nen_toan` `giong_toan` |
| **P4** | Nhãn 03 lời thật | ghép / chèn lời thật · xoá lời | `ghep` `xoa` `doan_ghep` |
| **P5** | Nhãn 03 giọng máy | giọng máy một đoạn · giọng máy cả file · phân loại máy / thật | `giong_may` · đầu `tts_file` |

**Trục trường hợp phủ** — mỗi lane áp lên **nhãn của mình**, cả hai ngôn ngữ, và chịu số đo của từng ô (≥ 10 tệp, ≥ 5 người nói):

| Trục | Giá trị |
|---|---|
| mức che giấu | vụng · vừa · **kỹ** (khớp kênh, nền, cross-fade, khử ồn phủ sau) |
| độ dài đoạn | 1 âm tiết · 1 từ · vài từ · 1 câu; chèn ≤ 5 s; tổng cục bộ ≤ 50 % |
| vị trí | giữa · sát mép 0,3–0,55 s · hai chỗ sát nhau (< 1 s) · hai chỗ cách xa |
| điều kiện Constraint | gốc · SNR 10–25 dB · nhạc / TV 5–10 % · RT60 ≤ 1 s · mức −40…−26 dBFS · cắt đỉnh ≤ 1 % · dải 6–8 kHz · tốc độ ±5 % |
| truyền tải | WAV · nén ≥ 24 kbps (MP3, AAC, Opus, AMR-WB, Vorbis) · qua Zalo / Messenger / WhatsApp / Telegram · rớt gói · DTX |
| nguồn tệp | điện thoại · máy ghi âm · họp trực tuyến · nhận lại qua ứng dụng nhắn tin |
| kênh · độ dài tệp | 1 · 2 giống · 2 cùng cảnh · 3–16 s · 16–64 s · 1–10 phút |
| công cụ | **thấy khi học** và **chưa từng thấy** (báo riêng; bắt buộc ≥ 1 công cụ giữ kín mỗi lane) |
| đối chứng | tệp thật nhiều nguồn · tệp ngoài phạm vi (kỳ vọng 04) |

**Quy ước nhãn chéo** (để huấn luyện độc lập mà không nhầm lẫn nhau). Mỗi mẫu là *dương* (thao tác của mình), *âm* hoặc *bỏ qua*:

| Chuyên gia | Âm (không được báo) | Bỏ qua (nhãn do lane khác chịu) |
|---|---|---|
| `ghep` `xoa` `doan_ghep` (P4) | tệp thật, lành tính, tệp chỉ có 02 | đoạn giọng máy (P5 gán loại) |
| `khu_on` `doi_giong` `nen_toan` `giong_toan` (P3) | tệp thật, lành tính (nén, thiết bị), tệp chỉ có 03 | vùng bị thao tác 03 chồng lên |
| `giong_may` `tts_file` (P5) | tệp thật, **ghép lời thật** (Req 6), **đổi giọng AI** (02) | — |
| `cong01` (P2) | tệp thật mọi nguồn / mọi truyền tải | — (dương = mọi thao tác trong 8 loại) |
Ba cặp dễ nhầm trao đổi **dữ liệu đối chứng** (không trao đổi mô hình): P3↔P5 (đổi giọng AI ↔ giọng máy), P4↔P5 (ghép thật ↔ giọng máy), P2↔P3 (khử ồn ứng dụng ↔ nén / mặc định thiết bị). Mỗi lane đăng ≥ 300 tệp đối chứng ở ngày đầu từ bộ đã có và làm mới ở mỗi mốc.

## 3. Để sáu người chạy song song, không ai chờ ai

1. **Bộ khởi động ngày 0** đã nằm trong kho — ai cũng chạy được ngay: manifest các bộ (`data_labels/v6/*`: natvi, naten, cover, cover2, chain, benign, amilong, aug, tts…), khung XLS-R + vector gộp đã trích, fold theo người (`crc32 % 5`), `detect.load_items`, `tcn.py` · `ft.py` · `ft_cv.py`, `rowmerge.py`, `req_check` (Req 1–11 gồm 6, 8a, 8b), `cells.py`, `regime_table.py`, `v6/product/` (pipeline, ticket, selftest, e2e) và `vaudit/k1.py`.
2. **Hợp đồng đóng băng** (đổi = đề xuất bằng văn bản, P0 duyệt): H1 manifest · H2 tám mã nhãn `02_nen_toan_cuc 02_nen_cuc_bo 02_giong_toan_cuc 02_giong_cuc_bo 03_ghep_giong_that 03_xoa_loi 03_chen_giong_may(đoạn) 03_chen_giong_may(cả file)` · H3 điểm khung 20 ms `(T, C) float16` + đầu cả file + `cong01` · H4 dòng thao tác `{level2, kind, start, end, p, nm, via}` · H5 hồ sơ `voiceaudit.v6.ho_so/1` + #bind · H6 chấm (`req_check`, ô, vùng).
3. **Mỗi lane nộp một "gói chuyên gia"** (cùng khuôn cho cả sáu lane):
   ```
   experts/<lane>/<ten>_<phienban>/
     card.json   lane, kênh, ngôn ngữ, bộ dữ liệu, fold, seed, số đo, RTF 300 s, bộ nhớ
     oof_<ten>.pkl   {id: (T, C) float16}, 5 fold theo người nói
     thr_<ten>.json  ngưỡng theo ngôn ngữ ở 4 điểm vận hành A–D (từ tệp thật fold khác)
     weights/    đã huấn luyện sẵn (đóng băng hoặc LoRA); KHÔNG học từ đầu
     score.py    score(wav, ctx) -> {kênh: mảng (T,)} hoặc {đích: số}
     signs.py    signs(wav, hàng, ctx) -> ≥ 2 dấu hiệu có số + phân vị so với kho thật
     verify.py   verify(wav, đoạn, ctx) -> (nhãn, điểm)  # nhìn khác bộ phát hiện (tiêu chí tay)
     tests/      đơn vị + bảng ô của lane
   ```
   `ctx` do P0 cấp (khung L12, tầng 6, vector gộp, điều kiện K1) — tính **một lần**; ngày 0 đã có bản lưu đĩa cho mọi tệp hiện có nên không lane nào phải chờ.
4. **Ghép = đổi cấu hình**, không ghép mã: P0 nạp gói theo tên + phiên bản; thiếu gói mới thì dùng gói cũ. Mỗi tuần P0 chạy "chuyến ghép" và gửi lại từng lane bảng ô trước / sau (ô nào tụt, vì sao).
5. **Giả lập hàng xóm**: P1 dựng ảnh / hồ sơ từ dòng giả của `selftest`; các chuyên gia dùng K1 đã lưu trong `meta.k1`; P0 chạy bộ ghép với gói giả (điểm ngẫu nhiên đúng lược đồ). Không ai cần bản thật của người khác để làm việc.
6. **Ngân sách RTF** (CPU 4 nhân, 300 s, không tính nạp mô hình; tổng ≤ 0,6, dự trữ 0,11): XLS-R int8 dùng chung 0,12 · K1 0,12 · P2 ≤ 0,02 · P3 ≤ 0,06 · P4 ≤ 0,06 · P5 ≤ 0,06 · hợp nhất + giải mã ≤ 0,02 · dựng hồ sơ ≤ 0,03. Mỗi bộ thích nghi 6 tầng ≈ 0,05. Vượt ngân sách → đề xuất gộp vào bộ thích nghi chung (P0 huấn luyện đa nhiệm từ dữ liệu của các lane).
7. **Nguyên tắc học chung (H9)**: mô hình sâu đã huấn luyện sẵn; không dừng sớm theo loss CE (đã đo: làm kém đi, xếp hạng đỉnh vẫn tốt lên); ngưỡng và siêu tham số **không** chọn trên tập chấm; ngưỡng đặt trên tệp thật tự nhiên của fold khác, riêng từng ngôn ngữ; tập khoá chỉ P0 chạm, một lần.

## 4. Sáu lane

Mỗi lane gồm: **TÌM** dữ liệu thật · **SINH** dữ liệu · **mô hình và phương pháp** · **đo** · **nộp** · **xong khi**. Mọi lane phủ **cả vi và en**.

### P0 — Nhóm trưởng: quyết định · tích hợp · chấm hệ thống
Sở hữu K2 (khung chung), K5 hợp nhất, K7 (khung và luật), tệp nhiều thao tác, G, H. Req 1, 2, 3, 15 (hệ thống), duyệt 4, 9, 14.
- **TÌM:** ghi âm thật cho **tập chấm hệ thống** (30–50 tệp: phòng vang, ngoài trời, qua loa; điện thoại, máy ghi âm, họp trực tuyến); nhờ **người ngoài tự sửa** bằng Audacity / Audition / công cụ AI → tập chưa từng thấy.
- **SINH:** tệp 2–3 thao tác khác nhãn (ghép từ bộ của P3–P5); nhóm vùng xám 0,4–0,6; tệp dài 1–10 phút; tập "04": Constraint bị vi phạm có chủ đích + xác suất lưng chừng.
- **Làm:** hợp nhất điểm (trung bình logit TCN đóng băng ⊕ tinh chỉnh, đã đo tốt hơn từng bên); ngưỡng theo ngôn ngữ ở 4 điểm vận hành, đường biên báo nhầm ↔ bắt được; cổng 01 (điểm của P2 ⊕ bằng chứng chuyên gia); bộ kiểm chứng độc lập (luật: khớp → giữ; bác điểm ≥ 0,80 → 04; bác yếu → bỏ); `rowmerge` + `typed_merge`; mỗi đoạn một thao tác, chồng > 30 % giữ đoạn mạnh; ≤ 2 dòng / 14 s + cảnh báo "còn chỗ khác"; nhãn tệp nhiều thao tác; đo RTF 300 s đầu-cuối; hàng đợi GPU, bộ nhớ, đóng gói; chạy ma trận ô, bảng vùng, e2e, robust, **chấm khoá một lần**.
- **Nộp:** bảng chỉ tiêu Req 1–15 (đường biên và số đo), nhật ký "ô tụt", phiên bản hệ thống.
- **Xong khi:** hồ sơ đầu-cuối đạt Req 12, 14 = 100 %, RTF 300 s ≤ 0,6, mọi ô có số đo, gói nào trượt ngân sách đều có quyết định.

### P1 — Khung tất định: Input · Constraint · 04 · Output
Sở hữu K1, K8, K9. Req 9, 12, 13, 14 (khung), 12 cảnh báo.
- **TÌM:** loại âm thanh (podcast, phát thanh, sách nói: LibriVox + sách nói tiếng Việt; nhạc: MUSAN-music, FMA); kiểu nói (hát: VocalSet, NUS-48E; thì thầm: wTIMIT; la hét: CREMA-D); ngôn ngữ khác (FLEURS, Common Voice); đếm người nói và nói chồng (AMI, AliMeeting, VoxConverse, LibriMix); RIR thật (OpenSLR 26 / 28, BUT ReverbDB); nhiễu (MUSAN, DEMAND, FSD50K, DNS Challenge); ghi âm thật từ nhóm: mỗi thiết bị ≥ 30 tệp.
- **SINH:** từng hàng Constraint **dưới / đúng / trên ngưỡng** (≥ 12 tệp mỗi hàng, hai ngôn ngữ): định dạng (8) × bit × tốc độ bit × tần số lấy mẫu × số kênh (1, 2 giống, 2 cùng cảnh, 2 nguồn riêng, 3) × độ dài (2,5 s … 10 phút+) × dải 4–8 kHz × mức × cắt đỉnh 0,5–6 % × mất tín hiệu × SNR 5–25 dB × nhạc / TV 5–30 % × RT60 0,3–2,0 s × tỉ lệ tiếng nói; phát lại qua loa rồi ghi lại (Assumption); sửa > 50 % và đoạn < 0,1 s (Assumption).
- **Mô hình:** bộ học trên vector gộp XLS-R cho SNR, nền, RT60 (có sẵn: AUC 0,84 / 0,91 / 0,997 → mục tiêu SNR ≥ 0,92 bằng nhiều loại nhiễu thật); bộ phân loại loại âm thanh và kiểu nói; nhận dạng ngôn ngữ; đếm người nói; nói chồng. Ngưỡng chọn để ≤ 2 % tệp hợp lệ bị chặn (tệp bị chặn báo riêng theo lý do; hợp đồng không tính chúng vào Req 9 — Req 9 chỉ đếm tệp ra 04 do phân tích).
- **Tất định:** K8 luật nhãn (1)–(5) và mức độ cố định; K9 ảnh dạng sóng + phổ (cả file / vùng ≤ 1 s cho xoá lời), giải thích tiếng Việt, bảng mốc / điểm–ngưỡng / số đo, PDF (hash, phiên bản, giờ, tuyên bố giới hạn), kênh trái / phải, #bind; dấu hiệu lấy từ `signs.py` của các lane và phân vị từ kho tham chiếu của P2.
- **Khảo sát Req 13:** ≥ 3 người ngoài nhóm xem ảnh, đoán loại thao tác (1 trong 5), ≥ 70 % đúng; thiết kế phiếu, tuyển người, thống kê.
- **Xong khi:** mỗi hàng Constraint có test chủ động đúng đáp án; Req 12 = 14 = 100 % trên tập kiểm; khảo sát ≥ 70 %; PDF mở được cả tệp 2 kênh.

### P2 — Nhãn 01: dấu vết bình thường · truyền tải · ghi âm thật · cổng 01
Sở hữu mọi thứ **không phải chỉnh sửa**, K5 (chuyên gia), kho tham chiếu. Req 4 (chính), 3, 5 (phối hợp P3).
- **TÌM:** ghi âm thật **không sửa** nhiều nguồn, hai ngôn ngữ — cuộc gọi băng rộng, tin nhắn thoại, máy ghi âm, họp trực tuyến (AMI, EdAcc, VoxPopuli, VietSuperSpeech, VoxVietnam, Vietnam-Celeb); **tự thu** chuỗi truyền tải thật: cùng một bản ghi gửi qua Zalo, Messenger, WhatsApp, Telegram (iOS / Android), thu lại bản nhận; ghi lại có tạm dừng, đổi thiết bị.
- **SINH:** 8+ cấu hình nén (MP3 32–128 kbps, AAC, Opus VoIP / audio, AMR-WB, Vorbis), nén lặp 2–4 lần, DTX, rớt gói 2–5 % (che giấu mất gói), mặc định thiết bị (cao thông 100 / 300 Hz, AGC, EQ cố định, chuẩn hoá độ lớn), đổi định dạng và tần số, tốc độ ±5 %, đổi âm lượng, cắt đầu–cuối, nén kênh điện thoại; **ca nhìn giống chỉnh sửa**: tạm dừng ghi, cắt dán đầu–cuối, im lặng dài.
- **Mô hình:** chuyên gia `cong01` — "có dấu vết của 8 thao tác không?" trên vector gộp + điểm tối đa của các kênh (dùng phiên bản có sẵn của chúng), hồi quy logistic / GBM, ngưỡng ≤ 10 % tệp thật qua cổng; mô hình hoá **nguồn "thật" đáng ngờ** (VietSuperSpeech 17 % báo nhầm: kiểm bằng nghe và báo cả hai cách); bảng phân vị tệp thật (kho tham chiếu) cho **mọi** dấu hiệu, theo ngôn ngữ và nguồn.
- **Đo:** báo nhầm theo từng cơ chế ≤ 8 %; 8/8 cấu hình nén ra 01; AUC cổng ≥ 0,98; không lane nào làm báo nhầm tệp thật tăng trên bộ của P2.
- **Nộp:** bộ tệp thật + truyền tải (manifest), gói `cong01`, kho tham chiếu, **bộ âm cứng** cho P3 / P4 / P5 (phiên bản mới mỗi mốc).
- **Xong khi:** Req 4 báo bằng đường biên (≤ 3 % hoặc ghi rõ mức bắt được tương ứng), mọi hàng "01 phủ những gì" của HTML có ≥ 10 tệp ≥ 5 người, kho tham chiếu thay kho 160 tệp VIVOS.

### P3 — Nhãn 02: khử ồn · đổi giọng (cả file · một đoạn)
Req 5, F1 bốn thao tác 02, định vị (10, 11) cho đoạn 02.
- **TÌM:** công cụ khử ồn khác nhau — DeepFilterNet, RNNoise, Facebook Denoiser, noisereduce (spectral gating), Resemble Enhance, WebRTC NS, khử ồn của Zoom / Teams / Meet (thu qua cuộc gọi), khử vang (WPE); công cụ đổi giọng — RVC, Seed-VC, so-vits-svc, OpenVoice, FreeVC, kNN-VC, dịch cao độ (rubberband, ffmpeg, librosa); giọng đích từ VCTK / LibriTTS / VSASV; **giữ kín ≥ 1 bộ khử ồn và ≥ 1 bộ đổi giọng** làm tập chưa từng thấy.
- **SINH:** cả file và một đoạn (≤ 50 %), cường độ từ rất nhẹ (khử ồn < 0,5, dịch < 1 nửa cung) tới mạnh (±2–4 nửa cung, nhiều nửa cung với AI), chuyển tiếp mềm / cứng, đoạn ngắn 0,5 s tới 8 s, hai đoạn trong một tệp, trên nền SNR 10–25 dB, vang ≤ 1 s, kênh ứng dụng; **ca lành tính của 02** (khử ồn ứng dụng áp đều, tốc độ ±5 %) để Req 5.
- **Mô hình:** kênh vùng `khu_on`, `doi_giong` (TCN trên khung đóng băng + bộ thích nghi LoRA riêng), đầu cả file `nen_toan`, `giong_toan` (vector gộp thô; phép thử phương sai phổ yên / có tiếng); kiểm chứng viên bằng tiêu chí tay trên đoạn (ΔMFCC, sàn nhiễu, LFCC, cao độ).
- **Đo:** F1 từng thao tác, TPR ở báo nhầm 1 / 3 / 5 %, Req 5 ≥ 92 %, theo ô điều kiện, tập chưa từng thấy; khử ồn vô hại ≥ 92 % không bị gọi đổi giọng / 03.
- **Xong khi:** mỗi thao tác 02 đạt F1 tương ứng chỉ tiêu ở cả hai ngôn ngữ trên tập công cụ đã thấy, tụt ≤ 10 điểm trên tập chưa thấy; có thẻ + `signs` + `verify`.

### P4 — Nhãn 03 lời thật: ghép / chèn · xoá lời
Req 7a, 7b, 10, 11 (cho 03 lời thật). Lát **khó nhất** (xoá lời và sửa kỹ).
- **TÌM:** tiếng nói tự nhiên có chép lời + mã người nói để cắt tại ranh giới từ — vi: VIVOS, VoxVietnam, Vietnam-Celeb, VietSuperSpeech, FPT Open Speech, Common Voice; en: LibriSpeech, VCTK, AMI, EdAcc, VoxPopuli; công cụ gióng từ (wav2vec2 CTC vi / en, MFA); **sửa thật bằng tay**: ≥ 150 ca làm trên Audacity / Audition / Reaper bởi người không trong nhóm → tập chưa từng thấy; tập chuẩn định vị PartialSpoof.
- **SINH:** thay / chèn / xoá tại ranh giới từ, từ 1 âm tiết tới 1 câu, chèn ≤ 5 s, nguồn lời cùng bản ghi / bản ghi khác / cùng người / người khác; **vụng · vừa · kỹ** (khớp mức, khớp EQ và nền, cross-fade 5–50 ms, thêm phòng); sát mép 0,3–0,55 s; hai chỗ xoá < 1 s; xoá nhiều từ; 1–10 phút với ≤ 2 đoạn / 16 s; nội dung sau sửa kiểm bằng ASR đọc lại (lời mạch lạc, đúng Assumption); chồng điều kiện (vang, TV, nén, ứng dụng).
- **Mô hình:** kênh `ghep`, `xoa`, `doan_ghep`: TCN đóng băng ⊕ **TCN tinh chỉnh** (LoRA hạng 16 × 6 tầng, cửa sổ 4 s, 8 lượt cố định — công thức đã đo: ghép kỹ vi 28 → 49 %, xoá en 36 → 55 %); xoá lời cần thử riêng: cửa sổ 8 s, trọng số lớp, đa hạt giống; nhãn mốc ±60 ms; kiểm chứng viên bằng tiêu chí mép (B_lfcc, flux, bậc sàn nhiễu) và trong / ngoài đoạn.
- **Đo:** TPR ở báo nhầm 1 / 3 / 5 %, F1 7a, 7b, mốc ≤ 500 ms, IoU, theo kiểu tệp (vụng / vừa / kỹ / vùng), theo vang / TV / nén, tệp dài, tập sửa tay.
- **Xong khi:** 7a ≥ 0,80 và 7b ≥ 0,65 ở ít nhất một điểm vận hành công bố; mốc ≥ 78 %, IoU ≥ 0,80 hoặc báo rõ khoảng cách và nguyên nhân từng ô.

### P5 — Nhãn 03 giọng máy: một đoạn · cả file · máy hay thật
Req 6, 8a, 8b, định vị cho đoạn máy.
- **TÌM:** bộ công cụ nhái giọng — viXTTS, XTTS-v2, F5-TTS, CosyVoice, StyleTTS2, YourTTS, OpenVoice, Piper, MMS-TTS, VITS / VietTTS (vi); dữ liệu giả mạo chuẩn — ASVspoof 2019 / 2021 / 5, PartialSpoof, HAD, WaveFake, MLAAD, In-the-Wild; **giữ kín ≥ 2 công cụ** làm tập chưa thấy; giọng người nói tham chiếu từ chính các bộ tự nhiên.
- **SINH:** **nhái đúng người nói** (mẫu 6 s) đọc cả câu rồi cắt đúng từ (CER 0,02–0,05); một đoạn: thay / chèn ≤ 5 s; cả file; vụng · vừa · kỹ (khớp kênh, nền, mức, cross-fade, phát lại qua loa rồi thu); vi và en (≥ 300 tệp mỗi loại mỗi ngôn ngữ, ≥ 30 người nói); đối chứng: ghép lời thật (Req 6), đổi giọng AI (P3), đọc thật nói nhanh / chậm.
- **Mô hình:** kênh vùng `giong_may` + đầu cả file `tts_file` (vector gộp) + **bộ phân loại dòng máy / thật** trên đoạn (đặc trưng đoạn, AUC hiện 0,988 trên vi chưa che giấu) + kiểm chứng viên (độ giống giọng tổng hợp, mép).
- **Đo:** Req 6 ≤ 5 %, F1 8a ≥ 0,86, 8b ≥ 0,90, theo công cụ đã / chưa thấy, theo mức che giấu, vi / en.
- **Xong khi:** có bộ giọng máy hai ngôn ngữ đủ ô, ba số 6, 8a, 8b được đo (nay chưa đo được) và thẻ chuyên gia hoàn chỉnh.

## 4B. Danh mục test bắt buộc — đủ nhãn, đủ trường hợp

Quy tắc chung: mỗi bộ test có **tên cố định**, ≥ 10 tệp và ≥ 5 người nói mỗi ô, **cả vi và en**, người nói ở tập chấm không có trong tập học, ngưỡng không chọn trên tập chấm; công cụ **thấy** và **chưa từng thấy** báo riêng; mỗi số ghi kèm n. Mục "đạt" là mức chỉ tiêu của nhãn đó; chưa đạt thì báo mức bắt được và nguyên nhân từng ô.

**P1 — Constraint · Output · Assumption**
| Bộ | Nội dung | Đạt |
|---|---|---|
| T1.1 hàng Constraint | 24 hàng × (dưới · đúng ngưỡng · trên) × 2 ngôn ngữ, ≥ 12 tệp mỗi hàng; hàng cứng ra 04 + đúng lý do; hàng cảnh báo ra đúng cảnh báo | hàng cứng 100 % · hàng học: AUC mục §4 |
| T1.2 container | 8 định dạng × (16-bit · 24-bit · float) × (16 · 22,05 · 44,1 · 48 kHz) × tốc độ bit nén (< 24 · ≥ 24 kbps) | không sập; đúng 04 / cảnh báo |
| T1.3 kênh | 1 · 2 giống · 2 cùng cảnh (lệch ≤ 50 ms) · 1 kênh < 2 s tiếng nói · 2 nguồn riêng · 3 kênh | đúng quy tắc, mốc ghi theo kênh |
| T1.4 độ dài | 2,5 · 3 · 16 · 64 s · 5 · 10 phút · > 10 phút · 250 MB | đúng cổng; RTF theo độ dài |
| T1.5 điều kiện | SNR 5–25 dB · nhạc / TV 5–30 % · RT60 0,3–2,0 s · mức · cắt đỉnh 0,5–6 % · mất tín hiệu · tiếng nói < 30 % · loại âm thanh · kiểu nói · ngôn ngữ khác · 1–6 người nói · nói chồng 0–30 % | ≤ 2 % tệp hợp lệ bị chặn oan; bắt ≥ 80 % tệp vi phạm rõ |
| T1.6 hồ sơ | 8 thao tác × 2 phạm vi, hai kênh, nhiều dòng, #bind, ảnh, ≥ 2 dấu hiệu, PDF | 0 hồ sơ tự mâu thuẫn (Req 14), 100 % dòng đủ ảnh + dấu hiệu (Req 12) |
| T1.7 Assumption | phát lại qua loa rồi thu · sửa > 50 % · đoạn < 0,1 s · xoá < 1 âm tiết | cảnh báo / 04 đúng hoặc ghi vào tuyên bố giới hạn |
| T1.8 khảo sát | ≥ 3 người ngoài nhóm, mỗi người ≥ 40 ảnh, loại thao tác 1 trong 5 | ≥ 70 % (Req 13) |

**P2 — nhãn 01 và cổng 01**
| Bộ | Nội dung | Đạt |
|---|---|---|
| T2.1 nén | ≥ 8 cấu hình (MP3 · AAC · Opus VoIP / audio · AMR-WB · Vorbis × bitrate), nén lặp 2–4 lần | 8/8 ra 01 (≤ 8 % báo nhầm mỗi cấu hình) |
| T2.2 truyền tải thật | cùng bản ghi qua Zalo · Messenger · WhatsApp · Telegram × iOS / Android | báo nhầm ≤ 8 % mỗi ứng dụng |
| T2.3 thiết bị | cao thông 100 / 300 Hz · AGC · EQ cố định · chuẩn hoá độ lớn · đổi định dạng / tần số | ≤ 8 % |
| T2.4 biến đổi cả tệp | tốc độ ±5 % · đổi âm lượng · cắt đầu–cuối | ra 01 |
| T2.5 mất gói | rớt gói 2 / 5 % · DTX | ≤ 8 % |
| T2.6 ca dễ nhầm | tạm dừng ghi · đổi thiết bị · im lặng dài · tốc độ lệch 5–12 % | đúng như Constraint "không tự kiểm được": ghi vào tuyên bố, không quá 15 % báo nhầm |
| T2.7 tệp thật theo nguồn | điện thoại · máy ghi âm · họp trực tuyến · tin nhắn thoại × vi / en, báo theo nguồn và người nói | Req 4: báo nhầm và đường biên; nguồn lệch (VSS) báo cả hai cách |
| T2.8 cổng 01 | AUC; độ nhạy ở ≤ 10 % tệp thật qua cổng | AUC ≥ 0,98 |

**P3 — nhãn 02** (mỗi hàng: ×vụng/vừa/kỹ, × độ dài, × vị trí, × điều kiện, × công cụ thấy / chưa thấy)
| Bộ | Nội dung | Đạt |
|---|---|---|
| T3.1 khử ồn cả file | ≥ 6 công cụ × cường độ (< 0,5 … 1,0) × SNR 10–25 dB × vang ≤ 1 s | F1 ≥ 0,90; tập chưa thấy tụt ≤ 10 điểm |
| T3.2 khử ồn một đoạn | 0,5–8 s, ≤ 50 % tệp, 1 và 2 đoạn, chuyển tiếp mềm / cứng | TPR ≥ 85 % ở báo nhầm 5 % |
| T3.3 đổi giọng cả file | dịch 2 / 3 / 4 nửa cung · ≥ 4 công cụ AI | AUC ≥ 0,95 |
| T3.4 đổi giọng một đoạn | như T3.2 | TPR ≥ 80 % |
| T3.5 vô hại | khử ồn áp đều bởi ứng dụng (Zoom · Teams · WebRTC) và công cụ cả file | **≥ 92 % không bị gọi đổi giọng / 03 (Req 5)** |
| T3.6 đối chứng | tệp thật · nén / thiết bị (P2) · giọng máy (P5) · tệp chỉ 03 | không báo 02 ngoài ô cho phép |

**P4 — nhãn 03 lời thật**
| Bộ | Nội dung | Đạt |
|---|---|---|
| T4.1 ghép thay | 1 âm tiết … 1 câu × vụng / vừa / kỹ × nguồn (cùng bản ghi · khác bản ghi · cùng người · khác người) | F1 7a ≥ 0,80 |
| T4.2 ghép chèn | như T4.1, chèn ≤ 5 s | như trên |
| T4.3 xoá lời | 1 âm tiết … 1 câu × vụng / vừa / kỹ; xoá nhiều từ | F1 7b ≥ 0,65 |
| T4.4 vị trí | sát mép 0,3–0,55 s · 1 từ · 40–50 % tệp · hai chỗ xoá < 1 s · hai đoạn cùng tệp | mỗi nhóm ≥ 70 % |
| T4.5 điều kiện | vang 0,3–1,0 s · TV 8 % · SNR 10–25 · nén · kênh ứng dụng · chồng điều kiện | không nhóm nào < 60 % mức gốc |
| T4.6 tệp dài | 16–64 s · 1–10 phút, ≤ 2 dòng / 16 s | ≥ 60 % |
| T4.7 sửa tay | ≥ 150 ca do người ngoài làm bằng Audacity / Audition / Reaper | báo riêng, mục tiêu ≥ 60 % |
| T4.8 định vị | mốc ≤ 500 ms · IoU · mốc xoá ≤ 500 ms cách vùng khoanh | ≥ 78 % · ≥ 0,80 |
| T4.9 đối chứng | tệp thật · lành tính · tệp chỉ 02 | không báo 03 ngoài ô cho phép |

**P5 — nhãn 03 giọng máy**
| Bộ | Nội dung | Đạt |
|---|---|---|
| T5.1 một đoạn | thay / chèn ≤ 5 s × ≥ 4 công cụ × vụng / vừa / kỹ × vi / en | F1 8a ≥ 0,86 |
| T5.2 cả file | ≥ 4 công cụ × vi / en × mức | ≥ 0,90 (8b) |
| T5.3 công cụ chưa thấy | ≥ 2 công cụ giữ kín | báo riêng; tụt ≤ 10 điểm |
| T5.4 ghép lời thật | tệp có ghép lời thật (P4) | **≤ 5 % bị gắn giọng máy (Req 6)** |
| T5.5 đổi giọng AI | tệp 02 của P3 | không gắn 03 máy; báo mức nhầm |
| T5.6 chuẩn ngoài | ASVspoof 2019 / 2021 / 5 · PartialSpoof · In-the-Wild | báo AUC / EER để so |
| T5.7 điều kiện · truyền tải | vang · nén · ứng dụng · phát lại qua loa | không ô nào < 60 % mức gốc |

**P0 — hệ thống**
| Bộ | Nội dung | Đạt |
|---|---|---|
| T0.1 nhiều thao tác | 5 tổ hợp 2 thao tác + 3 thao tác + 2 đoạn cục bộ cùng tệp, mọi nhãn tệp | nhãn tệp đúng thứ tự (04 > 03 > 02 > 01); ≥ 60 % thao tác đúng loại |
| T0.2 vùng xám | xác suất 0,4–0,6; đoạn bị kiểm chứng bác | đúng 04 + lý do, tỉ lệ 04 ≤ 8 % (Req 9) |
| T0.3 chuỗi nhãn | Req 1–5, 9 theo đường biên và 4 điểm vận hành A–D, vi / en | công bố đường biên; chỉ tiêu đạt nào ghi rõ |
| T0.4 ma trận ô | 72 nhóm × 11 nhãn + bảng vùng điều kiện | không ô trống có nghĩa |
| T0.5 tập người ngoài sửa | do người ngoài nhóm làm, công cụ chưa thấy | báo riêng |
| T0.6 ghi âm thật hệ thống | 30–50 tệp thiết bị thật | báo riêng |
| T0.7 vi–en | chênh macro-F1 | ≤ 0,08 |
| T0.8 RTF | 300 s đầu-cuối, 4 nhân, 8 GB, không GPU | ≤ 180 s (Req 15) |
| T0.9 khoá | 950 biến thể codec lạ + người nói crc lẻ (en) | chấm một lần |

## 5. Ghép lại và cổng

**Test ăn khớp khi ghép** (P0 chạy mỗi chuyến, gửi kết quả cho cả sáu lane):
1. **Hợp đồng**: lược đồ gói, cột điểm, tên nhãn, ngưỡng đủ 4 điểm vận hành, `signs` ≥ 2, `verify` có mặt.
2. **Ma trận nhầm lẫn chéo**: mỗi chuyên gia chạy trên bộ của **mọi** lane; hàng "âm" của bảng quy ước nhãn chéo (§2) phải ≤ mức báo nhầm đã đặt.
3. **Không tụt ô**: bảng ô của từng lane trước / sau khi nạp vào hệ thống (độ chênh > 3 điểm phải có lý do).
4. **Cùng tập chấm**: mọi lane và hệ thống đo trên cùng fold, cùng danh sách tệp; số của lane và số trong hệ thống đối chiếu được.
5. **Đủ ba cặp nhầm**: đổi giọng AI ↔ giọng máy, ghép thật ↔ giọng máy, khử ồn ứng dụng ↔ nén / thiết bị có bảng riêng.
6. **Phép thử thay thế**: gỡ từng gói khỏi hệ thống, số của các lane còn lại không đổi.

| Cổng | Điều kiện | Người |
|---|---|---|
| G0 giao diện | gói chuyên gia qua kiểm lược đồ (cột, ngưỡng, dấu hiệu, kiểm chứng viên); test hợp đồng xanh | mỗi lane → P0 |
| G1 cục bộ | đạt "xong khi" của lane; bảng ô của lane đủ; **không làm tụt ô của lane khác** khi nạp vào hệ thống | mỗi lane |
| G2 hợp nhất | hợp nhất điểm, ngưỡng, `typed_merge`, cổng 01, kiểm chứng; chọn điểm vận hành A–D | P0 |
| G3 đầu-cuối | e2e + robust + RTF 300 s; Req 12, 14 = 100 %; mọi hàng Constraint có test | P0 + P1 |
| G4 khoá | chấm đúng một lần, công bố đường biên lẫn số đo | P0 |
Va chạm: lỗi tính cho lane **sinh dữ liệu sai**; đổi hợp đồng cần văn bản; không ai sửa tệp của lane khác (mỗi lane có thư mục `lanes/<tên>/` và tag bộ dữ liệu riêng).

## 6. Sổ phủ sáu mục — không sót

| Mục | Mục con | Chủ |
|---|---|---|
| **Input** | 1 tệp / lần, có tiếng người nói | P1 |
| | nguồn: điện thoại · máy ghi âm · họp trực tuyến · nhận lại qua ứng dụng nhắn tin | P2 (dữ liệu thật + truyền tải), P1 (cfg) |
| **Output chính** | nhãn 01–04 | P1 (luật K8), P2 (01), P0 (xung đột) |
| | lý do khi 04 (Constraint · xám · bị bác) | P1 · P0 |
| | thao tác: loại · phạm vi · mốc · mức độ · kênh | P3 · P4 · P5 (điểm), P1 (dựng) |
| | ảnh mỗi thao tác · giải thích ≥ 2 dấu hiệu có số theo kho tham chiếu | P1 (dựng) · lane (dấu hiệu) · P2 (kho) |
| **Output phụ** | cảnh báo: nén mạnh · ồn cao · cắt đỉnh · mất tín hiệu · phòng vang · kiểu nói · ngôn ngữ khác · nhiều người nói · nói chồng · 0,3 s đầu cuối | P1 |
| | cảnh báo: nghi 03 chưa đủ chắc · còn chỗ sửa khác | P0 |
| | bảng mốc · điểm–ngưỡng · số đo · PDF (hash, phiên bản, giờ, tuyên bố) | P1 |
| **Requirement** | 1, 2, 3, 15 | P0 |
| | 4 (chính) · 3 | P2 (P0 duyệt) |
| | 5 | P3 |
| | 6 · 8a · 8b | P5 |
| | 7a · 7b | P4 |
| | 9 · 12 · 13 · 14 | P1 |
| | 10 · 11 | mỗi lane cho thao tác của mình, P0 tổng hợp |
| **Constraint** | TỆP: số tệp, 250 MB, 8 định dạng, bit, tốc độ bit, tần số, độ dài, số kênh + quy tắc hai kênh | P1 |
| | TÍN HIỆU: dải ≥ 6 kHz, âm lượng, cắt đỉnh, mất tín hiệu, SNR, nền > 10 %, tiếng nói ≥ 2 s và ≥ 30 % | P1 |
| | NGƯỜI NÓI VÀ PHÒNG: loại âm thanh, kiểu nói, ngôn ngữ, số người nói, nói chồng, RT60 | P1 |
| | VÙNG SOI: 0,3 s đầu cuối; ≤ 2 đoạn / khúc 16 s | P1 (cảnh báo) · P0 (luật) |
| | KHÔNG TỰ KIỂM ĐƯỢC: một lần ghi, tốc độ phát (+ ca minh hoạ) | P1 (tuyên bố) · P2 (ca dữ liệu) |
| **Scope** | khử ồn · đổi giọng (cả file / một đoạn) | P3 |
| | ghép lời thật · xoá lời | P4 |
| | giọng máy (một đoạn / cả file) | P5 |
| | mức độ cố định theo loại | P1 |
| | "báo khi": cả file ≥ 0,6; đoạn vượt ngưỡng + kiểm chứng độc lập | P0 (luật) · lane (kiểm chứng viên) |
| | nhãn tệp (1)–(5); mỗi đoạn một thao tác; chồng > 30 % | P1 (luật) · P0 (hợp nhất) |
| | 01 nghĩa là gì; cắt đầu–cuối và đổi âm lượng không phải chỉnh sửa | P2 |
| | thao tác khác không cam kết; "không làm" | P1 (tuyên bố) |
| **Assumption** | không phát lại qua loa; micro không che | P1 (cảnh báo) · P2 (ca đối chứng) |
| | nén đồng đều, DTX; ứng dụng khử ồn đều | P2 · P3 |
| | người sửa: cục bộ ≤ 50 %, đoạn ≥ 0,1 s, xoá ≥ 1 âm tiết, lời mạch lạc, công cụ phổ thông, không phá dấu vết | P3 · P4 · P5 (dữ liệu tuân thủ), P0 (kiểm dữ liệu) |
| | người dùng nhận 04; chờ ≤ 3 phút / 5 phút | P1 (thông điệp 04) · P0 (RTF) |
| **Ngoài 6 mục nhưng có trong HTML** | cờ che giấu K6; chỉ tiêu chênh vi–en ≤ 0,08; tập chống nén 8 cấu hình; ghi âm thật; tập người ngoài tự sửa | P0 (K6, vi–en) · P2 (nén, ghi âm thật) · P0 + P4 (tự sửa) |

## 7. Cập nhật 05/10/2026 (khớp mục «Hiện trạng và hướng v2» trong VoiceAudit.html)

- `VoiceAudit.html` đã viết lại thân theo v2: mục Hiện trạng và hướng v2, Cây phân rã v2, Tiêu chí đánh giá v2 (17 yêu cầu một định nghĩa, 8 thao tác), Số đo tay kết hợp embedding, Sổ lệch. Các mục v4 giữ làm lịch sử, có chú thích. Sáu khối đã khoá không đổi (`VoiceAudit_6Muc.html` giống hệt bản nộp).
- Đã sửa trong hệ thống: hồ sơ không còn ghi kiểm chứng giả; luật nền cục bộ ≥ 0,9 khi có nền cả tệp; tuyên bố giới hạn theo 8 thao tác; nhãn khúc 32 s; ngôn ngữ nhận dạng chọn bộ ngưỡng; AMR-WB tính là nén; thiếu đầu cả tệp thì báo lỗi; kiểm điều kiện mở rộng (ngôn ngữ, nói chồng) bật mặc định.
- Thí nghiệm số đo tay (`v6/product/hand.py`, `v6/boundary/hand_exp.py`): hợp với khử ồn đoạn và giọng máy (nhịp, nền, mel); không giúp xoá lời và ghép kỹ; làm kiểm chứng chung chỉ thêm AUC +0,003. Xếp hạng lại dòng bằng điểm embedding nhiều kênh tăng bắt được 6–9 điểm ở cùng mức báo nhầm. Chưa nối vào sản phẩm.
- Còn mở: cổng không dấu vết, bộ kiểm chứng trong sản phẩm, giọng máy, loại âm thanh và kiểu nói, chốt ngân sách báo nhầm cả tệp, sửa mã chấm theo định nghĩa mới, khảo sát người ngoài.
