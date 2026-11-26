# Cách đếm «chặn oan» khi ghi âm thật vi phạm hàng cứng (quá nhỏ, băng hẹp, cắt đỉnh nặng)

> **Cập nhật 22/11** (sau khi đo bộ vi cào bù, 927 đoạn accept; luật lọc chốt trước khi mở kết quả K1):
> - vi số thô: 48/927 = 5,2 % (Wilson 95 %: 3,9–6,8 %).
> - Trong đó vi phạm hàng cứng thật, đo độc lập xác nhận:
>   - «cắt đỉnh nặng» 5 đoạn / 2 video: −4 dBFS, 7–11 % mẫu ≥ 99 % đỉnh;
>   - «băng hẹp» 4 / 1 video: năng lượng 6–8 kHz thấp hơn dải lời nói 47–82 dB;
>   - «quá nhỏ» 2: −40,5 / −40,8 dBFS, **sát ngưỡng**.
> - Đếm theo đề xuất: vi 37/916 = 4,0 % (2,9–5,5 %).
> - Nguồn chặn oan lớn nhất còn lại là **«quá ít tiếng nói» 33 đoạn** (K1 đếm thiếu thời lượng lời nói). P1 đang sửa **cách đo**, không đổi ngưỡng.
> - Huấn chốt 22/11: «cắt đỉnh nặng» tính cùng nguyên tắc với «quá nhỏ» và «băng hẹp».
> - **Huấn chốt 24/11:** «quá nhỏ» tính **đúng theo công thức K1**: mức tiếng nói (khung đủ to ∩ ASR ra chữ) < −40 dBFS là chặn đúng. Đo độc lập và nghe (Gemini) chỉ để kiểm tra công thức đo đúng, **không** thay đổi cách đếm.
>   - Trên `p1_tunhien_v1`: 10 tệp accept (5 kênh; en 8, vi 2), K1 −40,1 … −44,7 dBFS.
>   - Đo độc lập không dùng ASR: RMS cả tệp −42,4 … −46,8 dBFS; mức 30 % khung to nhất −37,8 … −42,4 dBFS.
> - **Huấn chốt 24/11 (sau):** P1 toàn quyền về cách đo và cách đếm, không chờ P0 duyệt nữa. Đề xuất này thành **quyết định của P1**: vi phạm hàng cứng đo xác nhận được thì tính là chặn đúng. Báo cáo của P1 vẫn ghi đủ cả hai cách đếm (theo P1 và kiểu baseline).
> - **Huấn chốt 24/11 (mở rộng):** áp cùng nguyên tắc cho hàng **ồn (SNR < 10 dB)**.
>   - Tệp thật bị chặn vì ồn mà một phép đo độc lập, khác phương pháp với bộ đo chặn, cũng ra < 10 dB → chặn đúng.
>   - Không đo độc lập được → chặn oan, báo riêng.
>   - Chi tiết ở `NHAT_KY_P1.md` mục 2.16.
>   - Hai câu hỏi ở cuối (chuẩn hoá hoặc hạ ngưỡng −40 dBFS; cuộc gọi băng hẹp) đụng tới ngưỡng và hợp đồng. Chỉ đổi khi có minh chứng thực nghiệm rõ ràng bác bỏ ngưỡng cũ; hiện **chưa có**, nên giữ nguyên.

- **Lane / người đề xuất:** P1, Huấn. Claude Code soạn; Huấn chốt hướng ngày 21/11/2026.

- **Muốn đổi gì:** Không đổi ngưỡng. Xin P0 xác nhận cách đếm chỉ tiêu «K1 chặn oan ≤ 2 % tệp hợp lệ». Ghi âm thật mà đo độc lập xác nhận vi phạm hàng **cứng** «quá nhỏ» (< −40 dBFS) hoặc «băng hẹp» (dải tần thực < 6 kHz) thì tính là **chặn đúng**, báo **riêng**, không tính vào chặn oan.

- **Vì sao:**
  - Hai hàng này khớp với điều kiện ghi trong `PHU_DU_LIEU_5_LANE.md` mục 1.5: «mức tiếng nói −40 … −26 dBFS», «dải tần thực 6–8 kHz». Tệp ngoài khoảng đó không phải «tệp hợp lệ» theo hợp đồng.
  - Nhưng baseline `e2e_v9F_n60_TAM.md` lại tính «quá nhỏ» en 30/642 = 4,7 % là chặn oan. Hai cách đếm cho kết quả rất khác nhau.

  **Số đo** (bộ `data_labels/v6/p1_tunhien_v1`):
  - Nguồn: ghi âm YouTube CC BY, kênh không chuyên, đã lọc theo mục 1.8 và Constraint «Loại âm thanh».
  - Nhãn: «01 có khả năng», đáp án «accept».
  - Lệnh: `vaudit.k1.check(extended=False)`; chưa có `cond_models.joblib`, nên chưa kiểm SNR / nền / vang.

  | | en | vi |
  |---|---|---|
  | n (đáp án accept) | 454 | 131 |
  | K1 ra 04 | 15 | 2 |
  | trong đó «quá nhỏ», đo độc lập xác nhận | 8 (3 video; −40,5 … −45,8 dBFS) | 0 |
  | trong đó «băng hẹp», đo độc lập xác nhận | 2 (2 video; năng lượng 6–8 kHz thấp hơn dải 0,3–3,4 kHz 56–61 dB) | 0 |
  | **chặn oan, đếm kiểu baseline** | 15/454 = 3,3 % (Wilson 95 %: 2,0–5,4 %) | 2/131 = 1,5 % (0,4–5,4 %) |
  | **chặn oan, đếm theo đề xuất** | 5/444 = 1,1 % (0,5–2,6 %) | 2/131 = 1,5 % (0,4–5,4 %) |

  Cách đo độc lập:
  - Mức âm: trung bình công suất trên khung webrtcvad (mode 2). Lệch so với K1 0,4–1,1 dB, cùng chiều.
  - Dải tần: tỉ lệ năng lượng FFT 6–7,8 kHz so với 0,3–3,4 kHz.
  - Hai tệp băng hẹp: một video «Oral History Interview» thu bằng máy ghi âm cũ; một đoạn «Virtual Town Hall» qua họp trực tuyến.
  - Chi tiết: `lanes/P1/NHAT_KY/NHAT_KY_P1.md` mục 2.14.

- **Ghép thử:** chưa chạy. Đề xuất này chỉ đổi cách **báo cáo** chỉ tiêu K1, không đổi quyết định của hệ thống.

- **Ảnh hưởng tới lane khác:**
  - P2 (ghi âm thật, cổng 01): tệp thật < −40 dBFS hoặc băng hẹp vẫn ra 04 như hiện nay. Chỉ khác cách đếm trong báo cáo của P1.
  - P0: nếu đồng ý, nên ghi rõ định nghĩa «tệp hợp lệ» = thoả mọi hàng cứng đo được trực tiếp trên tín hiệu.

- **Câu hỏi kèm cho P0** (không bắt buộc trả lời ngay):
  1. Ghi âm thật nhỏ hơn −40 dBFS khá phổ biến: 3/144 video en trong bộ, cả bản ghi họp trực tuyến. Nhóm có muốn chuẩn hoá âm lượng trước khi đo, hoặc hạ ngưỡng, không? Nếu có thì phải sửa hợp đồng ở mục 1.5 và Constraint.
  2. Mục 1.5 đưa «ghi cuộc gọi» và «họp trực tuyến» vào phạm vi. Cuộc gọi băng hẹp (AMR-NB, 4 kHz) bị hàng «băng hẹp» chặn hết. Đó có đúng là ý định của nhóm không?

- **Cách quay lại nếu không ổn:** báo cáo của P1 luôn ghi **cả hai** con số (kiểu baseline và theo đề xuất), nên bỏ đề xuất thì không cần đo lại.

- **P0 duyệt:** (P0 ghi ngày · đồng ý / không · lý do)
