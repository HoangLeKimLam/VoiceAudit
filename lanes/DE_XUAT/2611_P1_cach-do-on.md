# Sửa cách đo hàng «SNR ≥ 10 dB»: chỉ chặn khi hai bộ ước độc lập cùng đo dưới 10 dB

- **Lane / người đề xuất:** P1 (Huấn), 26/11/2026. Claude Code soạn.
- **Muốn đổi gì:** hàng ồn chỉ trả 04 «ồn quá» khi **cả hai** bộ ước SNR dùng đặc trưng khác nhau cùng đoán dưới 10 dB. Khi chỉ một bộ đoán dưới 10 dB thì trả cảnh báo «tiếng ồn cao» thay vì chặn.
- **Không đổi:** ngưỡng hàng cứng vẫn là 10 dB trên cả hai bộ ước, không có biên trừ. Hợp đồng nhận / nộp và cụm từ cảnh báo giữ nguyên.
- **Trạng thái:** Huấn duyệt 26/11 làm cách đo P1 chọn; cách đo cũ giữ lại để đối chiếu và báo cáo. Chưa cài vào mã, chưa mở test. Số liệu dưới đây đều trên dev (fold 0–2).

## Vì sao

Mỗi bộ ước đơn lẻ sai số quá lớn so với yêu cầu chặn oan ≤ 2 %.

- Sai số của bộ tốt nhất ở vùng 8–16 dB là σ ≈ 2,9–3,1 dB. Mô phỏng cho thấy cần σ ≤ khoảng 1,6 dB thì một bộ ước không lệch mới đạt chặn oan ≤ 2 %.
- Ba hướng cải thiện mô hình đều không qua: thêm dữ liệu sát ngưỡng (NHAT_KY 2.21), thêm Brouhaha theo khung, thêm w2v 13 lớp (2.22). Chặn oan theo phân bố thật (a*) vẫn 4,7–5,4 %.
- Sai số của hai bộ ước chỉ tương quan 0,35 ở vùng 8–16 dB. Vì vậy yêu cầu hai bộ đồng thuận loại được phần lớn ca sai của từng bộ.
- Bộ đo đang nằm trong `vaudit/cond.py` (XLS-R, của nhóm) ghi trong chú thích của chính nó: tệp 5 dB chỉ bị chặn 26 %, tệp 8 dB 14 %. Số đó đo trên dữ liệu khác nên không so thẳng với bảng dưới, nhưng cho thấy hàng này hiện gần như không được thực thi.

Hai bộ ước:

| bộ ước | đầu vào | mô hình |
|---|---|---|
| mlp_F1 | 13 số đo có sẵn (Brouhaha, DeepFilterNet, tỉ lệ nền theo dải, khoảng nghỉ) | MLP nhỏ |
| S2 | lớp ẩn của wav2vec2 VI, cả 13 lớp, gộp trên khung lời và khung nghỉ | đầu MLP trên trung bình có trọng số theo lớp |

## Số đo trước / sau

Nguồn: `experts/P1/build_cond/de_xuat_bien.py` → `data_labels/v6/p1_dieukien_v4/_kiem/DE_XUAT_BIEN_2611.md`. Dự đoán OOF 3 fold chia theo người nói, train trên v4 + v4b, chấm trên v4 dev.

### Dữ liệu dựng (v4 dev, 3 758 tệp, SNR thật biết từ cách trộn)

| | mlp_F1 một mình (tốt nhất của P1 trước đề xuất) | đề xuất (đồng thuận) |
|---|---|---|
| Chặn oan theo phân bố thật (a*), cần ≤ 2 % | 4,7 % | **1,8 %** |
| Chặn oan, tệp thật 10–12 dB | 158/528 = 29,9 % | 56/528 = 10,6 % |
| Chặn oan, 12–15 dB | 56/372 = 15,1 % | 10/372 = 2,7 % |
| Chặn oan, 15–20 dB | 10/247 = 4,0 % | 4/247 = 1,6 % |
| Chặn oan, ≥ 20 dB | 17/983 = 1,7 % | 11/983 = 1,1 % |
| Bắt tệp ≤ 5 dB | 425/460 = 92 % | 419/460 = 91 % |
| Bắt tệp 5–8 dB | 407/477 = 85 % | 366/477 = 77 % |
| Bắt tệp 8–10 dB | 391/590 = 66 % | 271/590 = 46 % |
| Bắt mọi tệp < 10 dB | 1223/1527 = 80,1 % (78,0–82,0) | 1056/1527 = 69,2 % (66,8–71,4) |

Tệp vi phạm không bị chặn phần lớn vẫn nhận cảnh báo, không lọt im lặng:

| SNR thật | chặn | cảnh báo (một bộ đo dưới 10 dB) | không báo gì |
|---|---|---|---|
| < 10 dB | 1056/1527 = 69,2 % | 344/1527 = 22,5 % | 127/1527 = 8,3 % (7,0–9,8) |
| ≤ 5 dB | 419/460 = 91,1 % | 36/460 = 7,8 % | 5/460 = 1,1 % (0,5–2,5) |
| ≥ 10 dB (hợp lệ) | 81/2130 = 3,8 % | 281/2130 = 13,2 % | còn lại |

### Tệp thật (`p1_tunhien_v1` dev, 657 tệp «nên nhận»)

Báo cả hai cách đếm theo luật P1.

| cách đếm | mlp_F1 một mình | đề xuất (đồng thuận) |
|---|---|---|
| Cách nhóm: mọi tệp bị chặn là chặn oan | 89/657 = 13,5 % (11,1–16,4) | 49/657 = 7,5 % (5,7–9,7) |
| Cách P1: tệp bị chặn mà WADA-SNR độc lập cũng đo < 10 dB thì tính chặn đúng | 23/657 = 3,5 % (2,3–5,2) | **7/657 = 1,1 % (0,5–2,2)** |
| Số tệp nhận cảnh báo | chưa tính | 65/657 = 9,9 % (7,8–12,4) |

Trên 24 tệp thật Huấn đã nghe mù (NHAT_KY 2.21), số tệp còn bị chặn theo nhãn tai:

| nhãn tai (ý nghĩa sau hiệu chỉnh tai) | số tệp | mlp_F1 một mình chặn | đề xuất chặn |
|---|---|---|---|
| `khong` (sạch, chặn là oan) | 6 | 2 | **0** |
| `nhe` (11/14 trên 10 dB, có lẽ oan) | 8 | 8 | 4 |
| `ro` (dưới 10 dB, chặn đúng) | 9 | 8 | 5 |
| `che_loi` | 1 | 1 | 1 |

## Cách chọn (luật chốt trước khi có số, NHAT_KY 2.24)

So 18 cấu hình: ba luật (mlp_F1, S2, đồng thuận), mỗi luật thử biên trừ m = 0 … 5 dB.

- Điều kiện: (a*) ≤ 2 % và bắt tệp ≤ 5 dB ≥ 80 %. Trong số đạt, lấy cấu hình bắt < 10 dB cao nhất.
- Bốn cấu hình đạt: đồng thuận m = 0, 1, 2 và S2 m = 4. Đồng thuận m = 0 bắt cao nhất (69,2 %; kế tiếp 51,5 %).
- Biên trừ trên một bộ ước đơn lẻ kém hơn hẳn. Ví dụ mlp_F1 phải trừ 3 dB mới xuống 1,9 %, khi đó chỉ còn bắt 38,8 % tệp < 10 dB và 78 % tệp ≤ 5 dB.

## Những điều chưa chắc

- **(a*) 1,8 % sát mức 2 % và khoảng tin cậy rộng.** Lấy mẫu lại theo người nói 1 000 lần: trung vị 1,8 %, khoảng 0,4–4,3 %, 63 % số lần ≤ 2 %. Cấu hình lại được chọn trên chính dev nên con số này lạc quan.
- **Một bản gốc nghi ồn chi phối kết quả.** Bản gốc `fleurs_en_dev_5484…` chiếm phần lớn ca chặn oan ≥ 20 dB (Brouhaha đo chính bản gốc chỉ 6 dB). Nếu bỏ bản này thì (a*) của đề xuất còn 0,7 % và của mlp_F1 một mình còn 3,6 %. Đây chỉ là phép thử độ nhạy; có bỏ hay không do kết quả Huấn nghe 12 tệp gốc quyết định.
- **WADA-SNR là phép xác nhận yếu.** WADA chạm trần 60 dB ở 158/657 tệp thật dev, nên cách đếm P1 có thể đếm thừa chặn oan.
- **Giá phải trả là thả sót ở 8–10 dB:** bắt 66 % → 46 %. Phần lớn chuyển thành cảnh báo, không lọt im lặng.
- **Ô yếu nhất là nền ngắt quãng:** đề xuất chỉ bắt 61/198 = 30,8 % (24,8–37,6) tệp < 10 dB loại này, trong khi các loại nền khác bắt 62–86 %. Cả hai bộ ước đều đoán SNR của nền ngắt quãng cao hơn thật.
- **Theo ngôn ngữ:** chặn oan trên mọi tệp dựng ≥ 10 dB là vi 33/1043 = 3,2 %, en 48/1087 = 4,4 %. Hai con số này chưa cân theo phân bố thật nên cao hơn (a*).
- **Req 9 trên natvi / naten chưa đo được** trên máy này.

## Ghép thử

Chưa làm. Cần Huấn duyệt các bước sau, theo thứ tự:

1. Chốt cách đo đồng thuận là cách P1 chọn. Cách đo của nhóm (`vaudit/cond.py` hiện có) vẫn báo song song.
2. Mở test **một lần**: fold 3–4 của v4 (dữ liệu dựng) và `p1_tunhien_v2` (486 đoạn thật mới, chưa dùng để chọn gì).
3. Nếu test đạt: cài vào `vaudit/cond.py`, chạy lại mã thật và so khớp với mô phỏng, rồi chạy `test_ticket`, `selftest`, `test_pipeline_modes` và bảng T1.x trước / sau.

## Ảnh hưởng tới lane khác

- `vaudit/cond.py` nằm trong bộ gửi của P1. Đề xuất không đổi hợp đồng đầu ra: vẫn 04 «ồn quá» và cảnh báo «tiếng ồn cao».
- Thêm chi phí chạy: S2 dùng lại wav2vec2 VI mà K1 đã chạy để đếm tiếng nói. mlp_F1 cần Brouhaha và DeepFilterNet; Brouhaha có trọng số OpenRAIL sau cổng, phải kiểm điều khoản trước khi đưa vào sản phẩm.
- Số tệp nhận cảnh báo «tiếng ồn cao» sẽ tăng (khoảng 10 % tệp thật hợp lệ).

## Cách quay lại nếu không ổn

Thay đổi nằm trong một hàm quyết định của `cond.py`. Bỏ điều kiện đồng thuận là về cách đo cũ; mô hình và ngưỡng cũ không bị xoá.

- **P0 duyệt:** (theo Huấn chốt 24/11, cách đo do P1 tự quyết; gửi P0 để biết)
