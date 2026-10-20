# Bản đồ hướng phương pháp — mọi họ đều mở

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng** — bản đồ này tổng hợp những gì nhóm đã đo trên hệ thống hiện tại, dùng để so và tìm khoảng trống, không phải để chọn hướng từ đây.
> Cập nhật 19/10/2026, tổng hợp từ toàn bộ kết quả trong `.tmp_v6/results/`, tài liệu và lịch sử các phiên làm việc.
> **Đây KHÔNG phải xếp hạng hay lộ trình.** Mỗi bằng chứng chỉ đúng trong cấu hình đã đo (dữ liệu, cách chia, cách đưa vào). Họ nào "yếu" ở chỗ này có thể mạnh ở chỗ khác hoặc khi làm cách khác. Mọi họ — **số đo tay, ảnh phổ, embedding, kết hợp** — đều được dùng, nâng cấp, thay thế, miễn **không lệch mục tiêu / phạm vi mảng** và qua được `ghep_thu` ở ngân sách (`CHUNG.md §0, §3`).

## 1. Chỗ cắm vào hệ thống (họ nào cũng cắm được, không xung đột)

| Chỗ cắm | Đầu ra (hợp đồng) | Họ phù hợp (gợi ý, không giới hạn) | Ai |
|---|---|---|---|
| **Kênh khung** | điểm 20 ms `(T, k)` (H3) | embedding khung (XLS-R, WavLM…), ảnh phổ theo cửa sổ trượt (CNN / AST / PANNs), chuỗi số đo tay theo khung (`v6/fusion/hand_frames.py`, `v6/fusion/spec_frames.py`), kết hợp | P3, P4, P5 |
| **Đầu cả tệp** | xác suất / tệp | vector gộp embedding, số đo tay cả tệp, ảnh phổ cả tệp, kết hợp | P3, P5 (P2: cổng 01) |
| **Kiểm chứng viên đoạn** | xác nhận / bác một dòng (H4) | số đo tay trong–ngoài đoạn, ảnh phổ quanh mốc, embedding người nói trái–phải mốc, bộ phân loại máy / thật | chuyên gia của nhãn đó; luật dùng do P0 |
| **Xếp hạng lại dòng** | điểm mới cho dòng ứng viên | embedding nhiều kênh, đặc trưng đoạn | lane + P0 |
| **Dấu hiệu bằng chứng** | ≥ 2 dấu hiệu có số + phân vị kho thật (K9) | **số đo tay là chủ lực** (dễ hiểu cho Req 13), ảnh phổ cắt vùng | lane → P1 |
| **Cổng 01** | "có dấu vết của 8 thao tác không?" | điểm max các kênh + vector gộp + số đo tay | P2 |
| **Bộ ước lượng điều kiện K1** | SNR / nền / RT60 / loại âm thanh / kiểu nói | embedding gộp (đang dùng), ảnh phổ, DSP | P1 |
| **Tiền xử lý / định tuyến theo điều kiện** | (đề xuất cho P0) | khử ồn nhẹ / khử vang trước dò, ngưỡng theo nguồn | P0 + lane |

Ràng buộc chung cho mọi họ: mô hình sâu phải **đã huấn luyện sẵn** (đóng băng / LoRA / tinh chỉnh — kể cả CNN ảnh ImageNet, AST, PANNs, BEATs tinh chỉnh trên ảnh phổ đều hợp lệ); **RTF** trong ngân sách lane (P3 / P4 / P5 ≤ 0,06 mỗi lane; tạm thời được phép thử chậm hơn để chứng minh giá trị trước, tối ưu tốc độ sau — chỉ đạo 07/10); GPU phải được người dùng cho phép.

## 2. Họ A — Số đo tay (đặc trưng tín hiệu thủ công)

**Đã đo (trong cấu hình ghi bên cạnh):**
| Số đo | Thao tác | Kết quả | Nguồn |
|---|---|---|---|
| Khối H cả tệp (19 đặc trưng + 9 tiêu chí + `dn_*` nhiễu sau khử ồn) | **khử ồn cả tệp** | H 0,877 / 0,872 vs embedding 0,834 / 0,836; EHST 0,897 / 0,890; + G_* v5 0,918 / 0,904; LODO N2: E 0,772 vs **H 0,928**; khử ồn WebRTC báo nhầm ở q 1 %: H 3,3 % vs h3 15 % | `fx_file.md`, `fx_file_EG-EHSTG.md` |
| Khối H cả tệp | đổi giọng cả tệp | H 0,796 / 0,697 vs E 0,940 / 0,984 | `fx_file.md` |
| Khối H + S | giọng máy cả tệp (vi, 20 dương) | H 0,736, S 0,915, EH 0,969, E 0,873 | `fx_file.md` |
| Đưa đầu fxEHST vào sản phẩm (Q9b) | hệ thống vi | Req 4 0,165 → **0,243**, Req 5 0,836 → 0,760 — xấu đi (cách gắn này) | `req_check_v9fxH1_*.md` |
| `flux_moc` (M9, `v6/natural/criteria.py`) | ghép / xoá, **biết vị trí** | 0,81–0,94 (kỹ 0,69–0,73; kênh ứng dụng 0,64–0,86) — **chưa vào sản phẩm** | `criteria_auc*.md` |
| `pha_gay_moc` (group delay) · `enf_nhay_moc` (ENF thô) · `f0_jitter` (proxy) | mối nối / đổi giọng | 0,55–0,79 · 0,34–0,61 · 0,39–0,48 | như trên |
| `hand.py` theo đoạn | đoạn sửa vs đoạn đối chứng | khử ồn đoạn `i_mel` 0,85; giọng máy đoạn `e_mel_mean` 0,86; ghép ≤ 0,64 | `hand_trend.md` |
| `hand.py` làm kiểm chứng viên dòng 03 | đúng / sai dòng | E 0,861, H 0,691, EH 0,864 (cách gộp này) | `hand_verify.md` |
| Khối H/S/T theo dòng, mối xoá | ghép / xoá | H ≤ 0,68 vs E 0,87–0,98 | `fx_rows_novss.md`, `fx_joint.md` |
| Luật lành tính (`rules.py`) | 01 | cổng ồn 12 → 3 %, rớt gói 2 % 45 → 30 %, 5 % 30 → 12 % | `rules_all.md` |
| Bộ v4 FileFeat (biết vị trí, dữ liệu mô phỏng) | mọi thao tác | `I_dmfcc_var_ratio` 0,983 (khử ồn đoạn), `G_lfcc0` 0,988, `G_noisefloor_db` 0,914 (khử ồn cả tệp) — chậm (RTF 5,5) | `data_labels/realistic_v4/SANG_LOC_DAC_TRUNG.md` |

**Chưa từng đo thật (khoảng trống mở):** B1 ΔPSD đoạn yên + PELT · B3 / T1 MDCT nén kép, Benford, PFR · B4 ENF thật (chỉ có bản thô) · B5 ΔRT60 / ΔDRR theo đoạn · B6 z-score khoảng lặng lượt lời · I1 jitter / shimmer thật (Praat / WORLD chưa chạy) · I3 phương sai MGD · I4 HNR 4–8 kHz · I5 bộ dò hơi thở · G1 độ dốc f99 / kiểm upsample · G2 musical noise tách riêng · T2 / T3 lệch hai tầng · DNSMOS đầy đủ · đồng cấu âm, gãy pha, ngắt hơi, nền theo dải tại mối nối · `hand.py` làm **đầu vào khung** cho TCN hoặc đầu cả tệp · `hand.py` / `flux_moc` chưa được `pipeline` gọi.

## 3. Họ B — Ảnh phổ và mô hình âm thanh đã huấn luyện sẵn

**Đã đo:**
| Biểu diễn / mô hình | Thao tác | Kết quả | Nguồn |
|---|---|---|---|
| Ảnh phổ tự dựng (STFT, mel, MFCC, LFCC, CQT, group delay, modulation), probe | khử ồn cả tệp · đổi giọng AI · ghép · xoá | 0,88–0,92 · **0,93–1,00** · mel / CQT 0,79 · 0,70–0,75; vị trí không biết: mel 0,59 (xlsr 0,65) | `probe_auc.md`, `gate_probe.md` |
| AST (AudioSet) gộp, probe | khử ồn / đổi giọng cả tệp | 0,85 / 0,77; cục bộ 0,48–0,63 | `probe_auc.md` |
| AST gộp + 527 logit (khối A), mẫu 200 tệp | khử ồn cả tệp | EA 0,901 vs E 0,788 (CI +0,010…+0,189) — **chưa chạy đủ quy mô** | `fx_file_n200_*.md` |
| AST 10 đặc trưng | K1 nhạc / TV nền | 0,661 (TV 0,462) — bộ học trên XLS-R gộp 0,91 | `k1x_calib.md` |
| Khối S (hồ sơ log-mel, GLCM, modulation — thống kê tay, không phải CNN) | cả tệp | thêm vào E: +0,020; nhưng báo nhầm 48,3 % tệp lành tính ns3 ở 5 % | `fx_file.md` |
| DeepFense (WavLM-Large) + đầu CNN 1D | định vị (mô phỏng v4) | khung AUC 0,93–0,99, IoU 0,39–0,76 — quá chậm (19 s CPU / 60 s) | `data_labels/realistic_v4/TONG_HOP_CAC_HUONG.md` |
| AASIST3 · W2V2-AASIST | probe / cửa sổ | khử ồn cả tệp 0,60; cửa sổ dừng ở 56/630 tệp | `probe_auc.md` |

**Chưa từng đo (mở):** **mọi mô hình ảnh phổ đã huấn luyện sẵn ở mức khung** — PANNs CNN14, BEATs, AudioMAE, EfficientAT, CNN ảnh (ImageNet) tinh chỉnh trên log-mel / CQT / LFCC; AST đóng băng + đầu nhỏ cho khử ồn (M5); khối A đủ quy mô; `fx_stack` / `fx_tcn` (luồng khung H + S + T, đã trích cho 25,6 k tệp, chờ GPU); AASIST vs WavLM cho giọng máy trong đoạn; ảnh phổ quanh mốc làm kiểm chứng viên.

## 4. Họ C — Embedding tự giám sát và tinh chỉnh

> "XLS-R" của dự án là `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification` — **đã tinh chỉnh chống giả mạo**, không phải XLS-R gốc.

**Đã đo:**
| Embedding / cách học | Thao tác | Kết quả | Nguồn |
|---|---|---|---|
| Probe cục bộ (biết vị trí): xlsr_df L12 · WavLM L6 · PhoWhisper · resemblyzer | ghép thay / chèn / xoá | 0,96 / 0,97 / 0,91 · 0,92 / 0,93 / 0,85 · 0,81 / 0,86 / 0,70 · ≤ 0,63 | `probe_auc.md` |
| Vector gộp cả tệp | ghép / xoá | 0,45–0,66 — **gộp không thấy mối nối** | `gate_probe.md` |
| Khung L12 + TCN vs HGB | ghép (TPR 1/3/5 %) | vi 26/46/55 → 49/61/67; en 36/66/73 → 60/78/82 | `CAY_PHAN_RA_PHU_TOAN_MIEN.md` |
| FT-2 LoRA hạng 16 tầng 7–12, cửa sổ 4 s, 8 lượt | mọi kênh | ghép vi 0,861 → 0,921, xoá en 0,805 → 0,890; **xoá kỹ vi 16 → 7 %** | `ft_cmp_ft2_f0.md` |
| fuse1 = FT-2 ⊕ TCN (5 fold) | vi: ghép / xoá / đổi giọng đoạn | 0,938 / 0,789 / 0,943 (en ghép 0,981, xoá 0,896) | `cmp_oof.md` |
| h2: gộp thô L12 PCA96 | khử ồn / đổi giọng / giọng máy cả tệp | 0,863 / 0,941 / 0,886 (vi); thêm L8 / L16 không lợi | `heads2_h2.md`, `fx_file_L8-12-16_*.md` |
| tts_expert (khung L12) | giọng máy đoạn vs ghép thật | 0,988 (chỉ viXTTS, chưa che giấu); cả tệp 0,621 | `tts_expert.md` |
| cond_learn (gộp XLS-R) | SNR / nền / RT60 | 0,838 / 0,907 / 0,997 | `cond_learn.md` |

**Chưa từng đo (mở):** WavLM-Large, HuBERT / DistilHuBERT, Whisper encoder làm khung; gộp nhiều tầng (L6 + L12) cho khung; ECAPA-TDNN tinh chỉnh triplet (người nói trái–phải mốc); tinh chỉnh WavLM; đầu kiểu BISL / FPN-Transformer; Signal Pointer Nets (xoá / chèn, có mã mở); cắt tầng / ONNX / OpenVINO cho tốc độ; độ dài khúc (16 s sản phẩm vs cả tệp khi trích nghiên cứu — chưa tách bạch).

## 5. Họ D — Kết hợp, kiểm chứng, giải mã

- Đã đo: **hợp nhất muộn (trung bình logit) FT-2 + TCN tốt hơn từng bên** (fuse2 / fuse1); stacker v6d không hơn; số đo tay gộp làm kiểm chứng viên +0,003; **xếp hạng lại dòng bằng embedding nhiều kênh: 54,2 % ở báo nhầm 4,3 % vs 50,2 % ở 5,2 % — chưa tích hợp**; trình tự / cờ che giấu bằng công thức tay 0,30–0,36, bằng RF 0,742; ngưỡng theo độ dài hai chiều (lencal) giữ vu oan phẳng.
- Mở: kiểm chứng viên độc lập trước nhãn 03 (v4 có, v9 mất); giải mã nhiều đỉnh (≥ 2 chỗ sửa cùng loại / khúc); luật gộp khúc → tệp hai tầng; định tuyến / tiền xử lý theo điều kiện; ngưỡng theo nguồn tệp.

## 6. Họ E — Dữ liệu (sinh và thu)

- Đã có: natvi / naten (mức thường / kỹ / vụng, đa thao tác), cover / cover2 / chain / benign, ghi âm thật cào (không chuyên), tệp dài 1–10 phút.
- Mở: gom tệp theo giọng (speaker embedding) để ghép cùng giọng; giọng máy tiếng Anh và đa công cụ (GPU); hai kênh kiểu C thật, app ghi cuộc gọi 8 kHz; sửa tay bởi người ngoài; truyền tải thật qua app; thiết bị thật.

## 7. Nguyên tắc từ chỉ đạo của người dùng qua các phiên (để không lệch hướng dự án)

- Kết hợp **số đo tay + xu hướng + ảnh phổ + embedding** (23/09, 28/09, 30/09, 05/10, 07/10) — tạm bỏ qua thời gian chạy khi chứng minh giá trị, tối ưu sau.
- **Bám giọng, không bám kênh**: đặc trưng chỉ đo kênh / thiết bị dễ cho số đẹp ảo (97,9 % ảo do G2 / B5, 24/09).
- Dữ liệu mô phỏng phải **đúng nội dung gốc**, không chèn nội dung vô nghĩa (25/09); đủ mức che giấu vụng → kỹ, giọng tự nhiên, học chủ yếu trên dữ liệu tự nhiên (29/09, 01/10).
- Một tệp có thể chịu **nhiều thao tác** (25/09); phủ hết không gian Constraint kể cả ca biên (01–02/10); **không phình hệ thống** quá mức (22/09).
- Chỉ mô hình đã huấn luyện sẵn (25/09). Trung thực về khả thi, không lạc quan (23/09).

## 8. Số liệu KHÔNG được trích như kết quả
Các con số minh hoạ trong `BAN_TONG_HOP_CUOI.md`, `HE_THONG_FULL_CHOT.md` (jitter 0,61 → 0,58 → 0,09 %, PFR 2,1 vs 6,8, PSD −51,9 → −40,5 dB, `spoof_score = 0,83`) và bảng "0,__" trong `CHI_SO_THONG_KE_VA_BANG_CHUNG.md` là **mô phỏng / chỗ trống**, không có tệp kết quả.
