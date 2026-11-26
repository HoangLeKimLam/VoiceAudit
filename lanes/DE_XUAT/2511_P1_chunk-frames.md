# Sửa lỗi ghép khúc `vaudit/features.chunk_frames` (hàng không được gán)

- **Lane / người đề xuất:** P1 (Huấn), 25/11/2026. Claude Code soạn.
- **Muốn đổi gì:** `chunk_frames` hiện để trống một số khung của đầu ra. P1 muốn mọi khung đều được gán khung của khúc có tâm gần nhất.
- **Tệp đụng tới:** `vaudit/features.py`. Tệp này không thuộc P1, nên cần P0 duyệt.

## Lỗi

`chunk_frames` chạy mô hình trên các khúc 16 s chồng nhau 2 s, áp cho tệp dài hơn 20 s. Sau đó hàm ghép kết quả các khúc vào một lưới chung có bước `dt`, theo `idx = (tc / dt).astype(int)`.

Lỗi xảy ra khi tâm khung của một khúc rơi sát số nguyên của lưới:
- Điều này đúng ở các khúc có `s0 / dt` lẻ đúng ,5 (bắt đầu ở 56 s, 168 s, …, vì dt = 16/799 s), và ở khúc cuối ngắn hơn, có `d` khác `dt`.
- Phép cắt phần lẻ đẩy hai khung vào cùng một ô, nên ô bên cạnh không được gán.
- Ô không được gán giữ giá trị **0**.

## Tái hiện

Script `experts/P1/coverage/tai_hien_chunk_frames.py` tái hiện lỗi mà không cần mô hình hay dữ liệu:
- Tệp 80 s: 83/3 995 hàng không được gán, nằm ở giây 58–69.
- Độ dài 21–600 s, bước 1 s: **545/580** độ dài có hàng không được gán; ngắn nhất 21 s.
- Theo độ dài: 80 s → 83 hàng; 300 s → 276 hàng; 600 s → 424 hàng (khoảng 1,3 % số khung).

## Hậu quả đã đo

- **K1, «lượng tiếng nói»** (`k1.asr_speech_mask`):
  - Hàng logits toàn 0 cho argmax = 0, khác `pad_token_id` = 109, nên bị đọc thành «có chữ». Nới ±0,1 s ra **9,2 s tiếng nói ảo** trên tệp 80 s chỉ có 14 s lời.
  - Hệ quả: K1 nhận nhầm `tinhieu.tiengnoi_15pct` 13/24 và `tiengnoi_20pct` 12/24 trên bộ `k1axes/p1_v1`.
  - Nguồn số: `.tmp_v6/results/k1_axes_p1_v1.json`, NHAT_KY_P1 mục 2.17 và 2.19.
- **Đặc trưng bộ phát hiện** (`_w2v_hidden`, dòng 72–83): các khung không được gán thành vector **0** thay vì trạng thái ẩn wav2vec2 thật. Lỗi này có ở mọi tệp dài hơn 20 s, kể cả khi huấn luyện lẫn khi chạy.
  - **P1 chưa đo được** ảnh hưởng lên điểm bộ phát hiện hay Req 9, vì máy P1 không có checkpoint và `v6/engine`.

## Bản sửa đề xuất

Thay ánh xạ xuôi (khung khúc → ô) bằng ánh xạ ngược (ô → khung của khúc có tâm gần nhất):

```python
    dt = parts[0][1]
    T = int(round(n / SR / dt))
    tc = (np.arange(T) + 0.5) * dt                                   # tâm từng ô đầu ra
    cen = np.array([s0 + h.shape[0] * d / 2 for s0, d, h in parts])
    k = np.abs(tc[:, None] - cen[None, :]).argmin(1)                 # khúc có tâm gần nhất
    out = np.zeros((T, parts[0][2].shape[1]), np.float32)
    for i, (s0, d, h) in enumerate(parts):
        r = np.where(k == i)[0]
        out[r] = h[np.clip(((tc[r] - s0) / d).astype(int), 0, h.shape[0] - 1)]
    return out, dt
```

Kiểm bản sửa bằng `fn` giả có gắn nhãn khúc và khung, trên độ dài 21–600 s:
- hàng trống: **0**, bản cũ có 119 585;
- trong các hàng bản cũ đã gán, 6,3 % đổi sang khung khác, lệch tối đa 1 khung (20 ms) trong cùng khúc;
- tệp ≤ 20 s không đổi, vì không đi qua `chunk_frames`.

## Ảnh hưởng tới lane khác

- Mọi bộ phát hiện dùng `_w2v_hidden` có thể đã được huấn luyện và hiệu chỉnh trên đặc trưng có hàng 0. Sửa xong thì đầu vào đổi trên tệp dài hơn 20 s.
- **Cần ghép thử** trước khi nhận bản sửa: điểm theo kênh, vu oan, Req 9 trên natvi / naten, trước và sau.
- P0 / các lane chuyên gia có thể quyết huấn luyện lại hoặc giữ nguyên.

## Phần P1 tự làm, không cần sửa `features.py`

- Ứng viên **C4**: trong `vaudit/k1.py` (tệp của P1), coi hàng không được gán (`~lg.any(1)`) là blank. Chỉ ảnh hưởng dòng «lượng tiếng nói» của K1.
- C4 đang được so với 4 ứng viên khác trên dev, theo luật chốt trước ở NHAT_KY_P1 mục 2.19.
- Nếu P0 nhận bản sửa `chunk_frames`, C4 vẫn đúng, vì khi đó không còn hàng 0 để bỏ.

## Cách quay lại nếu không ổn

- Bản sửa chỉ nằm trong một hàm. Hoàn lại bằng `git revert` commit sửa.
- C4 chỉ thêm một phép `&` trong `asr_speech_mask`.

## P0 duyệt

(P0 ghi ngày · đồng ý / không · lý do)
