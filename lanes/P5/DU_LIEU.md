# P5 — Dữ liệu: phạm vi, nhãn, ví dụ

> Đọc sau `lanes/DU_LIEU.md`. **Cố định:** phạm vi phải phủ (mục 1), nhãn và ranh giới (mục 2), kiểm trước khi nộp (`DU_LIEU.md` mục 6.5), chính sách nhái giọng. **Tự do:** công cụ sinh, nguồn, cách làm, số lượng. Dữ liệu bạn nhận (mục 4) chỉ là **ví dụ**.
>
> **Việc của bạn với dữ liệu:** tự thu thập — cào nguồn mở, tự ghi, tự sinh, tìm kho mở — **dữ liệu đúng dạng ở mục 2–3** cho **đủ các trường hợp ở mục 1**. Dữ liệu mẫu ở mục 4 **không phủ hết** phạm vi; nguồn ở mục 6 chỉ là điểm xuất phát. Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết.
>
> **Dạng dữ liệu cần thu:** **giọng do máy tạo** — TTS / nhân bản giọng chèn, thay một đoạn hoặc cả tệp, có mốc; ghi công cụ sinh và vai trò; ≥ 2 công cụ (≥ 1 giữ kín); chỉ nhái người được phép; kèm giọng thật đa dạng làm âm.

## 1. Phạm vi bạn phải phủ

| Trường hợp | Vì sao |
|---|---|
| Giọng máy **một đoạn** (chèn / thay, 1 tiếng – 1 câu, mỗi đoạn ≤ 5 s) và **cả tệp**, vi + en | Scope; 8a, 8b |
| ≥ 2 công cụ sinh, ≥ 1 công cụ giữ kín không dùng khi học | «Điều kiện chấm» |
| Giọng máy **nhái rất giống** người trong bản ghi | Assumption |
| Giọng máy **đã qua** nén, ứng dụng, khử ồn, ghép khớp phòng | điều kiện ngoài đời |
| Giọng thật đa dạng làm âm (đúng fold) | vu oan kênh giọng máy |
| Ghép lời **thật** để đo Req 6 (không gọi nhầm là máy) | Req 6 — dùng chung dữ liệu của P4 |

## 2. Nhãn của bạn và ranh giới

| Của bạn | Không phải của bạn |
|---|---|
| `03_chen_giong_may`: một đoạn `op = "may_thay"` (`sub = "thay_the"`) / `"may_chen"` (`sub = "chen_them"`); cả tệp `kind = "global"`, `op = "may_ca_file"`. Luôn ghi công cụ sinh (`tts` / `engine`), `role`, người tham chiếu | **chuyển giọng (VC)** — người thật nói rồi đổi giọng → `02_giong_*` (P3) · đoạn chèn do **người thật** nói → `03_ghep_giong_that` (P4) |

`role` = `"train"` hoặc `"heldout"` theo `v6/tts_multi/engines.json`. Công cụ để dành (`vieneu`, `styletts2`) **không dùng để sinh dữ liệu học**.

## 3. Bản ghi mẫu

```json
{"id": "p5x_may_0003", "kind": "may_chen", "cfg": "goc", "src": "p5x_may_0003",
 "wav": "../v6/p5x_kokoro/may_chen/p5x_may_0003.wav",
 "meta": {"lang": "en", "source": "voxp", "person": "voxp:1234", "speaker": "voxp:1234",
          "level1": ["03_ghep_chen"], "level2": ["03_chen_giong_may"],
          "edits": [{"level2": "03_chen_giong_may", "kind": "segment", "op": "may_chen", "sub": "chen_them",
                     "start": 2.40, "end": 3.65, "inserted": "and then we agreed", "tts": "kokoro",
                     "role": "train", "ref_person": "voxp:1234"}]}}
```

## 4. Dữ liệu mẫu bạn nhận (ví dụ — không phủ hết phạm vi)

Nhóm trưởng chọn từ dữ liệu sẵn có để bạn hiểu định dạng, tái tạo baseline và bắt đầu nhanh — **gợi ý, không bắt buộc dùng**. Dòng «*thêm*» là dữ liệu khác có sẵn trên máy gốc, xin nếu thấy có ích.

| Gói | Nội dung | Dung lượng |
|---|---|---|
| chung | điểm OOF hệ thống + manifest natvi / naten | 0,12 GB |
| khung + vector XLS-R | đặc trưng đã trích (đã bỏ id có cờ) | 2,4 GB |
| âm thanh 03 giọng máy | natvi `may_*` hợp lệ (240 tệp, người vox); ghép lời thật vi / en (Req 6); tệp thật vi / en | 2,1 GB |
| giọng tham chiếu + kế hoạch sinh | khoảng 810 giọng tham chiếu, kế hoạch `ttsmulti_vi/en`, `vcmulti` (đã bỏ dòng để dành / tập khoá; đường dẫn tương đối) | 0,17 GB |
| kho gốc sinh | `data/natural_v6/hosts`, `hosts_en` (đã ghi sẵn `grow` / `session` để lập kế hoạch không cần parquet kho gốc) + tiếng ồn MUSAN | 1,75 GB |
| hạ tầng sinh | `v6/tts_multi` + DNSMOS | 0,002 GB |
| *thêm* | mô hình viXTTS 1,75 GB · đặc trưng họ khác | |

554 tệp giọng máy cũ nhân bản celeb / VSS (cờ `chinh_sach`) không có trong gói và không dùng.

## 5. Khi tự thêm dữ liệu

- Theo `DU_LIEU.md` mục 6. Mỗi tệp sinh ghi `tts` / `engine`, `role`, `ref_person`, `seed`; đóng manifest cho `ttsmulti_vi`, `ttsmulti_en` theo H1. Công cụ mới → đề xuất P0 thêm vào `engines.json` (vai trò học hay để dành).
- **Nhái giọng:** chỉ người trong kho nghiên cứu (VIVOS, vox, EdAcc, AMI, VoxPopuli, LibriSpeech); không celeb, không người cào, không VSS / VSASV / viVoice; tham chiếu là bản ghi **khác** của cùng người; mỗi người ≤ 6 % số tệp một bộ. Không phát tán âm thanh sinh ra.
- `vcmulti` (chuyển giọng AI, nhãn 02): bạn giữ hạ tầng sinh, sinh theo yêu cầu của P3.

## 6. Gợi ý nguồn (điểm xuất phát — tự tìm thêm)

| Cho | Gợi ý | Lưu ý |
|---|---|---|
| Sinh trên CPU (GPU đang hoãn) | Kokoro (Apache 2.0, nhẹ), MMS-TTS (CC BY-NC 4.0 — nội bộ), Piper (kiểm giấy phép từng giọng) | công cụ mới → đề xuất P0 |
| Sinh theo kế hoạch | `ttsmulti_vi` (viXTTS, F5-vi, ViZipVoice, MMS-vie), `ttsmulti_en` (XTTS-v2, F5-en, ZipVoice-distill, Kokoro) | GPU phải hỏi |
| Kiểm tra ngoài (không phải trường hợp phủ) | ASVspoof 2019 LA / 2021 / ASVspoof 5, In-the-Wild, MLAAD, PartialSpoof | kiểm giấy phép; NC → chỉ đo, hỏi trước |

## 7. Việc cần người thật

Nhóm trưởng cho phép dùng GPU khi máy đủ bộ nhớ (hiện đang hoãn).
