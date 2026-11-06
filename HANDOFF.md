# HANDOFF — Lane P3 (Nhãn 02: khử ồn chuyên nghiệp / mạnh · đổi giọng) — cập nhật 06/11/2026 chiều

> Bản trước của tệp này mô tả sai cơ chế chấm (xem «Đính chính» trong `lanes/P3/BAO_CAO_0611.md`, Đợt 2 §0). Bản này thay thế toàn bộ.
> Luật: `lanes/CHUNG.md` §2 (không sửa `v6/`, OOF 5 fold theo người nói, không chọn trên tập chấm, không sửa tệp lane khác, không `git push` khi chưa được yêu cầu).

## 1. Cơ chế chấm THẬT (đã đọc mã)
- `python -m v6.product.ghep_thu ...` → `v6/boundary/fuse9.py` `Ctx9.records`:
  - Ngưỡng kênh khung (den, voi) = phân vị `1 − freg` của **điểm MAX mỗi tệp thật** (`items[i]["real"]` = không có `level2`; gồm WebRTC NS ≤ 2 nay là 01), cùng ngôn ngữ, **fold khác** fold đang chấm.
  - Dòng 02 cục bộ = đoạn **liền** điểm ≥ ngưỡng, dài **0,4–8 s** (`fusion.decode_regions`). Đoạn > 8 s bị bỏ.
  - Đầu cả tệp (nen, giong): dòng toàn cục nếu điểm ≥ phân vị `1 − fh` trên tệp thật cùng ngôn ngữ fold khác.
  - Nhãn tệp: 04 (K1) > 03 > 02 > 01; «đúng» = nhãn cuối trùng nhãn kỳ vọng (dòng 02 loại nào cũng đúng cho tệp 02; dòng 03 của P4 làm tệp 02 thành sai).
  - Req 5 (`v6/natural/req_check.py contrib`): trong tệp **chỉ khử ồn cả tệp** (02 và WebRTC ≤ 2 cả tệp), tỉ lệ không bị 03 và không có dòng đổi giọng.
- Hệ quả thiết kế: biến điểm khung bằng phép «mở» 21 khung → max điểm = mức có đoạn ≥ 0,4 s → ngưỡng tệp thật không bị gai ngắn đẩy lên (`experts/P3/p3_v2/post.py`). Ở ngân sách 0,5 % / 0,25 %, ngưỡng vi do **2–4 tệp** quyết định → số rất nhạy, luôn báo KTC và theo fold.

## 2. Dữ liệu trên máy này
- Tập chấm `Ctx9.ids` = 9.010 tệp natvi / naten `goc`; 1.618 tệp thật (vi 939, en 679).
- `.wav`: 13.493 tệp — đủ cho mọi kiểu của P3 + mọi tệp thật + benign (540) + biến thể `app_*`; **không có** `.wav` cho kiểu của P4 / P5 (ghép, xoá, giọng máy).
- `data_labels/realistic_v4/xlsr_frames/<id>.npy` = (T, **64**) PCA-64; `xlsr_pooled/<id>.npy` = 6144.
- Cache P3 v2 (tạo lại được): `.tmp_v6/cache/p3v2_ff/` (đặc trưng khung, `feats.py run`, ~80 s), `p3v2_pause.pkl` (`feats_pause.py`), `p3v2_raw_*.pkl`, `p3v2_seq_s1.pkl`, `p3v2_heads_raw.pkl`.
- Cache p3_v1 (`p3_wav_feats.pkl`, `p3_X_bwav.npy`, `p3_X_loc_*.npy`) **không có mã tạo** → p3_v1 không tái tạo được; `signs.py` / `score.py` của p3_v1 là giả lập.

## 3. Hiện trạng: gói `experts/P3/p3_v2/` (nộp được, tự chứa)
Số OOF ở ngân sách (`.tmp_v6/results/ghep_thu_p3_v2.md`; chi tiết + KTC: `lanes/P3/BAO_CAO_0611.md` Đợt 2):

| | en | vi | đích |
|---|---|---|---|
| Req 1 / 4 / 5 | 0,704 / 0,038 / 0,955 | 0,543 / 0,061 / 0,922 | — / ≤ nền / ≥ 0,92 |
| khử ồn đoạn TPR | 0,56 | 0,25 | ≥ 0,85 |
| đổi giọng đoạn TPR | 0,91 | 0,56 | ≥ 0,80 |
| khử ồn cả tệp TPR (F1 chưa tính riêng) | 0,32 | 0,27 | F1 ≥ 0,90 |
| đổi giọng cả tệp AUC | 0,996 | 0,947 | ≥ 0,95 |
| Req 10 / 11 | 0,527 / 0,721 | 0,313 / 0,641 | ≥ 0,78 / ≥ 0,80 |

Tái tạo từ đầu (tuần tự; tổng ~2 giờ CPU):
```
python experts/P3/p3_v2/feats.py run --jobs 20                 # dac trung khung tu .wav
python experts/P3/p3_v2/feats_pause.py --jobs 8                # pho khoang lang / khung noi
python experts/P3/p3_v2/train_frames.py --tag st2 --stack --drop-file-ctx --rel-only --jobs 8
python experts/P3/p3_v2/train_seq.py --tag s1 --epochs 10      # TCN, ~60-80 phut
python experts/P3/p3_v2/train_heads.py ; python experts/P3/p3_v2/train_heads2.py ; python experts/P3/p3_v2/train_heads_b2.py
python experts/P3/p3_v2/make_mix.py                            # ens55 / ens73 (gop logit st2 & seq) + heads_vi_mix.pkl
python experts/P3/p3_v2/compose.py --den "rawp:ens55:9,9,21" --voi "vi=rawmx:ens55:9,9,1;en=rawp:ens73:9,9,21" \
   --nen "vi=file:experts/P3/p3_v2/heads_vi_mix.pkl;en=file:experts/P3/p3_v2/heads_b2.pkl" \
   --giong "vi=file:experts/P3/p3_v2/heads_vi_mix.pkl;en=file:experts/P3/p3_v2/heads_b2.pkl" --name p3v2 --out experts/P3/p3_v2 --eval
python -m v6.product.ghep_thu --name p3_v2 --col den=experts/P3/p3_v2/oof_p3v2.pkl:0 --col voi=experts/P3/p3_v2/oof_p3v2.pkl:1 \
   --head nen=experts/P3/p3_v2/heads_p3v2.pkl:0 --head giong=experts/P3/p3_v2/heads_p3v2.pkl:1 --f03 0.005 --freg 0.005 --fh 0.0025
python experts/P3/p3_v2/fit_final.py ; python experts/P3/p3_v2/train_seq.py --final --epochs 10   # trong so san pham -> weights/
python experts/P3/p3_v2/tests/test_p3v2.py
```
Đo nhanh mọi biến thể trong bộ nhớ (cùng thước `ghep_thu`, thêm Req 5/10/11, TPR đúng loại, AUC, KTC): `experts/P3/p3_v2/evalp3.py` (`P3Eval().run(den=..., voi=..., nen=..., giong=...)`).

## 4. Nút thắt đã định lượng (đừng thử lại hướng cũ mà không có ý mới)
1. **Khử ồn đoạn ↔ WebRTC NS ≤ 2 (01)**: AUC 0,67 vi / 0,71 en với đặc trưng tương phản trong–ngoài đoạn; RNNoise trộn (N4) không tách được. Cần dấu vết đặc thù công cụ hoặc cặp đối chứng cùng bản ghi.
2. **WebRTC NS3 (02) ↔ NS2 (01) cả tệp**: gần như không tách (TPR ~0–6 %).
3. **Fold 0 vi = `vss:nguyenkhangofficial`** (38 % tệp thật vi, phỏng vấn nhiều người) → vu oan `voi` vi do đổi lượt người nói; âm tổng hợp A–B–A không giúp.
4. **Tệp trùng byte** `v6nat_nguyen_ban_0318` = `_0393` đặt ngưỡng `nen` vi → `lanes/DE_XUAT/0611_P3_tep-trung-nguong-vi.md` (chờ P0).

## 5. Việc tiếp theo (ưu tiên)
1. Sinh **cặp đối chứng cùng bản ghi** cho khử ồn (WebRTC 0–3, RNNoise, DFN, noisereduce, afftdn) trên tệp gốc đúng fold + giữ kín ≥ 1 công cụ (tập chưa thấy) → `kiem_bo --moi` + `lockbox check` trước khi nộp.
2. CNN nhỏ trên phổ khoảng lặng độ phân giải cao (phân biệt công cụ), thử làm nhánh thêm cho `den` / `nen`.
3. Âm cứng hội thoại thật nhiều người (cùng P2) cho `voi`; đo vu oan trên dev ghi âm thật (cần P0 / P1 chạy e2e với `score.py` p3_v2).
4. Đo RTF tệp 300 s; tách F1 khử ồn cả tệp riêng; đo Req 10 / 11 riêng phần 02.
