# P3 — Dữ liệu: phạm vi, nhãn, ví dụ

> Đọc sau `lanes/DU_LIEU.md`. **Cố định:** phạm vi phải phủ (mục 1), nhãn và ranh giới (mục 2), kiểm trước khi nộp (`DU_LIEU.md` mục 6.5). **Tự do:** công cụ, nguồn, cách sinh, số lượng. Dữ liệu bạn nhận (mục 4) chỉ là **ví dụ** — không bắt buộc dùng, không bắt buộc dùng đặc trưng XLS-R.
>
> **Việc của bạn với dữ liệu:** tự thu thập — cào nguồn mở, tự ghi, tự sinh, tìm kho mở — **dữ liệu đúng dạng ở mục 2–3** cho **đủ các trường hợp ở mục 1**. Dữ liệu mẫu ở mục 4 **không phủ hết** phạm vi; nguồn ở mục 6 chỉ là điểm xuất phát. Lấy bao nhiêu, từ đâu, bằng công cụ gì là bạn quyết.
>
> **Dạng dữ liệu cần thu:** **lời nói thật đã qua xử lý** — khử ồn chuyên nghiệp / mạnh, đổi giọng (dịch cao độ ≥ 2 nửa cung, chuyển giọng VC), cả tệp hoặc một đoạn có mốc; từ nhiều công cụ (≥ 1 giữ kín); kèm cặp đối chứng với khử ồn đơn sơ của ứng dụng (01).

## 1. Phạm vi bạn phải phủ

| Trường hợp | Vì sao |
|---|---|
| Khử ồn **chuyên nghiệp / mạnh**, cả tệp và một đoạn, từ ≥ 3 bộ — trong đó ≥ 1 bộ giữ kín không dùng khi học | «Điều kiện chấm» |
| Khử ồn **AI** và công cụ người dùng phổ thông | loại người dùng gặp nhiều nhất ngoài đời |
| Đổi giọng: dịch cao độ ≥ 2 nửa cung, và chuyển giọng (VC, cả bằng AI), vi + en, cả tệp và một đoạn, ≥ 2 bộ (≥ 1 giữ kín) | Scope + «Điều kiện chấm» |
| Dịch cao độ **quanh ngưỡng 2 nửa cung** | biên định nghĩa đổi giọng |
| Cặp đối chứng **khử ồn đơn sơ của ứng dụng (01)** với khử ồn chuyên nghiệp (02) trên cùng bản ghi | Req 5; tách 01 / 02 — cùng P2 |
| 02 trên **ghi âm thật** và **tệp dài** (đúng fold) | vu oan trên ghi âm thật |
| Đoạn 02 dài 0,4–8 s; nhiều đoạn | dải khoanh của hệ thống; Req 10, 11 |

## 2. Nhãn của bạn và ranh giới

| Của bạn | Không phải của bạn |
|---|---|
| `02_nen_toan_cuc`, `02_nen_cuc_bo` — khử ồn **chuyên nghiệp / mạnh** | khử ồn đơn sơ có sẵn của ứng dụng (WebRTC NS mức 0–2, chế độ mặc định) → **01**, không ghi là thao tác |
| `02_giong_toan_cuc`, `02_giong_cuc_bo` — dịch ≥ 2 nửa cung; **chuyển giọng (VC)** kể cả AI, ghi `op = "doi_giong_ai"` | dịch < 2 nửa cung → không phải thao tác · giọng do máy tạo từ văn bản → **03 giọng máy** (P5) |
| dữ liệu chuyển giọng `vcmulti` — bạn dùng và đo (P5 sinh theo yêu cầu của bạn) | — |

Luôn ghi công cụ (`tool`) và tham số (`f0_shift_st` khi dịch cao độ); **WebRTC NS ghi `ns_level`** (mức ≤ 2 thì không ghi là thao tác). Nhiều thao tác trong một tệp: mỗi thao tác một dòng.

## 3. Bản ghi mẫu

```json
{"id": "p3x_nen_0001", "kind": "nen_toan_cuc", "cfg": "goc", "src": "p3x_nen_0001",
 "wav": "../v6/p3x_ai/nen_toan_cuc/p3x_nen_0001.wav",
 "meta": {"lang": "en", "source": "voxp", "person": "voxp:1234", "speaker": "voxp:1234",
          "level1": ["02_ai_hau_ky"], "level2": ["02_nen_toan_cuc"],
          "edits": [{"level2": "02_nen_toan_cuc", "kind": "global", "tool": "DeepFilterNet3", "tool_family": "dfn"}]}}
```

```json
{"id": "p3x_giong_0001", "kind": "giong_cuc_bo", "cfg": "goc", "src": "p3x_giong_0001",
 "wav": "../v6/p3x_ai/giong_cuc_bo/p3x_giong_0001.wav",
 "meta": {"lang": "vi", "source": "vox", "person": "vox:id00092", "speaker": "vox:id00092",
          "level1": ["02_ai_hau_ky"], "level2": ["02_giong_cuc_bo"],
          "edits": [{"level2": "02_giong_cuc_bo", "kind": "segment", "start": 2.10, "end": 4.35,
                     "tool": "rubberband", "f0_shift_st": 3.0}]}}
```

## 4. Dữ liệu mẫu bạn nhận (ví dụ — không phủ hết phạm vi)

Nhóm trưởng chọn từ dữ liệu sẵn có để bạn hiểu định dạng, tái tạo baseline và bắt đầu nhanh — **gợi ý, không bắt buộc dùng**. Dòng «*thêm*» là dữ liệu khác có sẵn trên máy gốc, xin nếu thấy có ích.

| Gói | Nội dung | Dung lượng |
|---|---|---|
| chung | điểm OOF hệ thống + manifest natvi / naten | 0,12 GB |
| khung + vector XLS-R | đặc trưng đã trích (đã bỏ id có cờ) | 2,4 GB |
| âm thanh 02 | natvi / naten kiểu `nen_*`, `giong_*`, `doi_giong_ai`, `da_ghep_giong`, `da_ghep_nen`, tệp thật; `benign` | 3,8 GB |
| *thêm* | giọng tham chiếu cho chuyển giọng (đã lọc tập khoá) 0,17 GB · kho gốc sinh 1,75 GB · bộ phủ điều kiện 1,6 GB · đặc trưng họ khác | |

Lưu ý: tên kiểu `nen_*` trong natvi / naten giờ gồm cả tệp WebRTC NS mức ≤ 2 đã chuyển sang **01** (xem `meta.edits_khong_tinh`). Chấm phần của bạn theo **nhãn**, không theo tên kiểu.

## 5. Khi tự thêm dữ liệu

- Theo `DU_LIEU.md` mục 6. Ghi công cụ, phiên bản, tham số trong `edits` để chia tập **công cụ chưa thấy**; công cụ giữ kín ghi trong thẻ bộ.
- Không dùng `resenh` (khử ồn AI), `knnvc` (chuyển giọng) — để dành cho tập khoá.
- Việc còn mở của dữ liệu: **618 đoạn WebRTC NS không ghi mức** (cờ `tam_loai`) — sinh lại có ghi `ns_level` nếu bạn cần chúng.

## 6. Gợi ý nguồn (điểm xuất phát — tự tìm thêm)

| Cho | Gợi ý | Lưu ý |
|---|---|---|
| Khử ồn AI | DeepFilterNet3 (`dfn`, MIT / Apache, chạy CPU), mô hình tăng cường của SpeechBrain, Demucs denoiser (CC BY-NC → hỏi) | không dùng `resenh` |
| Khử ồn phổ thông | Audacity, CapCut, Adobe Podcast Enhance, Krisp; chế độ «Cao» của Zoom / Teams; iPhone Voice Isolation | dịch vụ trực tuyến: chỉ tải lên giọng của chính thành viên |
| Chuyển giọng AI | OpenVoice v2, FreeVC24, Seed-VC (danh mục `engines.json`), RVC (MIT) | người đích chỉ trong kho nghiên cứu; GPU đang hoãn |
| Đổi giọng phổ thông | ứng dụng đổi giọng điện thoại / máy tính; SoX, Praat | áp lên giọng của chính thành viên |

## 7. Việc cần người thật

Cặp đối chứng khử ồn ứng dụng / công cụ (cùng P2) · áp công cụ và ứng dụng phổ thông lên giọng của chính thành viên.
