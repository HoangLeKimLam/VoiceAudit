# P5 — Ý tưởng tham khảo

> **Đọc sau khi bạn (và Claude) đã tự đề xuất ≥ 2–3 hướng cho mảng.** Mọi thứ dưới đây là ý tưởng, giả thuyết và kinh nghiệm cũ của nhóm trưởng — **không phải lộ trình, không phải lệnh cấm**. Bằng chứng cũ chỉ đúng trong cấu hình đã ghi; được thử lại với cách mới, được thách thức bằng số liệu. Mục tiêu, thước đo, hợp đồng vẫn ở `CLAUDE.md`.

## 0. Hệ thống hiện tại của mảng — cách baseline đang làm

```
XLS-R khung + tầng 6 ──► [kênh giong_may] ──► cột 5 của điểm khung (T, 6)  ─┐
XLS-R vector gộp     ──► [đầu tts_file]   ──► cột 2 của h2 (xác suất cả tệp) ┤   (cả hai đang TẮT: ftts = 0)
                                                                             ▼
        K7 (P0): vùng vượt ngưỡng ttsr → 03_chen_giong_may (đoạn); đầu ≥ ttsf → 03_chen_giong_may (cả tệp)
                 typed_merge: dòng ghép trùng vùng giọng máy đổi loại thành giọng máy (Req 6, 8a)
                                                                             ▼
        K9 (P1): ảnh + dấu hiệu từ `signs.py` của bạn
```

Baseline hiện tại được tạo bằng: fuse1 cột 5 (FT-2 cột giọng máy) + đầu `tts_file` của h2.

**Mã baseline hiện có — không nằm trong bộ gửi, xin nhóm trưởng khi cần** (sau khi đã tự đề xuất hướng): sinh giọng máy cũ `vaudit/tts.py`, `v6/natural/build_tiers.py make_tts`, `.tmp_v6/wf/tts_*.py`.

## 1. Hướng đi gợi ý
1. **Chặng 1 (CPU, làm ngay) — Vì sao giọng thật vi bị gọi là máy.** Tệp thật nào nổ (nguồn VSS / VSASV / celeb? người nói? fold?), điểm theo fold; thử đầu / kênh hiệu chỉnh theo nguồn, âm cứng thật cho kênh máy. Mục tiêu: vu oan kênh máy vi ≤ 0,25 % mà giữ ≥ 90 % phát hiện đoạn.
2. **Chặng 2 (CPU) — Chuẩn bị sinh dữ liệu đủ đa dạng.** Hoàn tất hàng WP7b (văn bản, người nói tham chiếu đúng luật, cấu hình ≥ 4 công cụ / ngôn ngữ, giữ kín ≥ 2), viết bộ đo 6 / 8a / 8b chuẩn; đề xuất bật kênh khi đạt.
3. **Chặng 3 (GPU, khi người dùng cho phép)** — sinh vi + en (≥ 300 tệp mỗi loại mỗi ngôn ngữ, ≥ 30 người nói), trích khung + vector gộp, chuyên gia `giong_may` riêng, tập chưa thấy, chuẩn ngoài (ASVspoof, In-the-Wild, PartialSpoof).

## 2. Đã thử, chưa hiệu quả (đúng trong cấu hình đã ghi — được thử lại nếu có cách / lý do mới)
- Bật kênh giọng máy thẳng ở điểm vận hành thường: vu oan tệp thật vi tăng mạnh (14 → 42 % ở fold 0; 5,2 → 12,5 % ở ngân sách 0,25 % trước 20/10; sau đổi nhãn 7,0 → 14,7 %).
- Nạp XTTS trên GPU khi máy thiếu bộ nhớ → sập; chạy một mình, đóng ứng dụng khác.
- Bộ phân loại đoạn máy / thật AUC 0,988 chỉ trên giọng máy **chưa che giấu** — đừng coi là đã giải xong.

## 3. Ý mở đầu
Công cụ TTS / nhái giọng hợp luật (viXTTS, XTTS-v2, F5-TTS, CosyVoice, StyleTTS2, YourTTS, OpenVoice, Piper, MMS-TTS, VITS); bộ phát hiện sẵn có (AASIST3, wav2vec2-deepfake, XLS-R SLS) làm đặc trưng; che giấu (khớp kênh, nền, mức, cross-fade, phát lại qua loa rồi thu); hiệu chỉnh theo nguồn.

## 4. Bản đồ hướng phương pháp cho mảng (chi tiết `lanes/HUONG_PHUONG_PHAP.md`)


| Họ | Đã thấy | Chưa thử / mở |
|---|---|---|
| Số đo tay | Khối H giọng máy cả tệp 0,736 (vi, 20 dương); tỉ lệ ngắt / nhịp không tách được nguồn | Jitter / shimmer thật (Praat / WORLD), HNR 4–8 kHz, bộ dò hơi thở — tín hiệu "quá mượt" |
| Ảnh phổ / mô hình chống giả mạo sẵn | Khối S giọng máy cả tệp 0,915 (n = 20); ảnh phổ tự dựng đổi giọng AI 0,93–1,00; AASIST3 mới đo 1 ô (0,60) | AASIST3 / XLS-R SLS / mô hình chống giả mạo sẵn làm đặc trưng; CNN ảnh tinh chỉnh |
| Embedding | tts_expert khung L12 0,988 (chỉ viXTTS, chưa che giấu); cả tệp 0,621; EH 0,969 | WavLM / gộp nhiều tầng; hiệu chỉnh theo nguồn để **giọng thật vi không bị gọi là máy** |
| Kết hợp | `typed_merge` đổi loại dòng ghép trùng vùng máy | Bộ phân loại máy / thật làm kiểm chứng viên dòng (Req 6) |
