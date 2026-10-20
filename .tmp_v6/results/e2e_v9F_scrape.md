# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 1575 tep tu nhien (scrapevi, scrapeen), 2026-10-18 00:49

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 60 tep / kieu co thao tac, 600 tep / kieu tep that (goc), 0 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'scrapevi': {'goc': {}}, 'scrapeen': {'goc': {'lockbox': 683}}}
- **Req 14 (ho so khong tu mau thuan)**: 1575/1575
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 746/746 thao tac
- PDF: 1575/1575 tep co bao_cao.pdf
- Ngon ngu dung: 1568/1568 tep duoc cham (nguon: {'lid': 1568})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 98, 'K5_khong_con_dong': 98}

**CHI TIEU CHINH - tang goc (tep goc)**: chay 1575 tep, bi loai (Constraint 04 / tu choi) 7, DUOC CHAM 1568

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.715 [bootstrap theo nguoi noi 0.684-0.742] (01 0.738 · 02 0.696 · 03 0.712) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.672 [0.640-0.703] (562/836) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.831 [0.801-0.858] (562/676) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.156 [0.131-0.184] (114/732) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.062 [0.052-0.076] (98/1568) · <= 0,08
- tep that ra 01 = 0.769 [0.737-0.798] (563/732) · tep co thao tac 03 gan dung 03 = 0.583 [0.531-0.633] (208/357)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 805 | 5 | 800 | 0.758 | 0.749 [0.705-0.788] (313/418) | 0.173 [0.138-0.214] (66/382) | 0.039 [0.027-0.054] (31/800) |
| vi | 770 | 2 | 768 | 0.665 | 0.596 [0.548-0.642] (249/418) | 0.137 [0.105-0.177] (48/350) | 0.087 [0.069-0.109] (67/768) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 228 | 4 | 224 | 0.756 | 0.737 [0.649-0.809] (84/114) | 0.164 [0.106-0.244] (18/110) | 0.018 [0.007-0.045] (4/224) |
| fold 1 | 248 | 1 | 247 | 0.765 | 0.712 [0.630-0.782] (94/132) | 0.122 [0.074-0.194] (14/115) | 0.057 [0.034-0.093] (14/247) |
| fold 2 | 494 | 2 | 492 | 0.709 | 0.655 [0.597-0.708] (184/281) | 0.128 [0.089-0.180] (27/211) | 0.079 [0.059-0.107] (39/492) |
| fold 3 | 468 | 0 | 468 | 0.694 | 0.652 [0.590-0.709] (159/244) | 0.179 [0.134-0.234] (40/224) | 0.060 [0.042-0.085] (28/468) |
| fold 4 | 137 | 0 | 137 | 0.640 | 0.631 [0.509-0.738] (41/65) | 0.208 [0.131-0.316] (15/72) | 0.095 [0.056-0.156] (13/137) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| <= 20 s | 1570 | 7 | 1563 | 0.716 | 0.671 [0.639-0.703] (558/831) | 0.156 [0.131-0.184] (114/732) | 0.063 [0.052-0.076] (98/1563) |
| > 20 s | 5 | 0 | 5 | 0.750 | 0.800 [0.376-0.964] (4/5) | - (n=0) | 0.000 [0.000-0.434] (0/5) |

**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)

| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |
|---|---|---|---|---|---|---|
| goc | scrapeen | 805 | 5 | quá ít tiếng nói | 3 | 0.004 |
| goc | scrapeen | 805 | 5 | quá nhỏ | 2 | 0.002 |
| goc | scrapevi | 770 | 2 | quá ít tiếng nói | 1 | 0.001 |
| goc | scrapevi | 770 | 2 | quá nhỏ | 1 | 0.001 |

**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)

| | 01 | 02 | 03 | 04 | K1 04 |
|---|---|---|---|---|---|
| **01** | 563 | 99 | 15 | 55 | 3 |
| **02** | 116 | 323 | 4 | 36 | 1 |
| **03** | 115 | 27 | 208 | 7 | 3 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| goc | scrapeen | ghep_chen | 60 | 0 | 60 | 90% | 80%-95% | 4 | 0 | {'fosdem': 30, 'yt': 30} |
| goc | scrapeen | ghep_thay | 60 | 0 | 60 | 82% | 70%-89% | 5 | 0 | {'yt': 30, 'fosdem': 30} |
| goc | scrapeen | giong_cuc_bo | 60 | 0 | 60 | 93% | 84%-97% | 4 | 0 | {'fosdem': 31, 'yt': 29} |
| goc | scrapeen | giong_toan_cuc | 60 | 0 | 60 | 75% | 63%-84% | 10 | 5 | {'fosdem': 30, 'yt': 30} |
| goc | scrapeen | nen_cuc_bo | 60 | 0 | 60 | 85% | 74%-92% | 8 | 0 | {'fosdem': 30, 'yt': 29, 'ccc': 1} |
| goc | scrapeen | nen_toan_cuc | 76 | 0 | 76 | 34% | 25%-45% | 45 | 10 | {'fosdem': 45, 'yt': 30, 'ccc': 1} |
| goc | scrapeen | nguyen_ban | 369 | 3 | 366 | 79% | 74%-83% | 288 | 16 | {'fosdem': 246, 'yt': 122, 'ccc': 1} |
| goc | scrapeen | xoa_loi | 60 | 2 | 58 | 45% | 33%-58% | 26 | 0 | {'yt': 29, 'fosdem': 30, 'ccc': 1} |
| goc | scrapevi | ghep_chen | 60 | 1 | 59 | 58% | 45%-69% | 21 | 1 | {'yt': 60} |
| goc | scrapevi | ghep_thay | 60 | 0 | 60 | 60% | 47%-71% | 20 | 1 | {'yt': 60} |
| goc | scrapevi | giong_cuc_bo | 60 | 0 | 60 | 78% | 66%-87% | 11 | 2 | {'yt': 60} |
| goc | scrapevi | giong_toan_cuc | 60 | 0 | 60 | 57% | 44%-68% | 20 | 6 | {'yt': 60} |
| goc | scrapevi | nen_cuc_bo | 60 | 0 | 60 | 83% | 72%-91% | 7 | 3 | {'yt': 60} |
| goc | scrapevi | nen_toan_cuc | 83 | 1 | 82 | 54% | 43%-64% | 41 | 12 | {'yt': 83} |
| goc | scrapevi | nguyen_ban | 327 | 0 | 327 | 75% | 70%-79% | 245 | 37 | {'yt': 327} |
| goc | scrapevi | xoa_loi | 60 | 0 | 60 | 15% | 8%-26% | 39 | 5 | {'yt': 60} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| scrapeen | ccc | 2 | 0.50 | 2 | 1.00 |
| scrapeen | fosdem | 260 | 0.12 | 210 | 0.74 |
| scrapeen | yt | 120 | 0.29 | 206 | 0.67 |
| scrapevi | yt | 350 | 0.14 | 418 | 0.56 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 4.74 | 0.37 |
| xlsr | 1.31 | 0.11 |
| tcn | 0.02 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 6.08 | 0.48 |
| tong_s | 6.44 | 0.51 |
| K1 + XLS-R + TCN + dau ca file | 6.08 | 0.48 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.24 | 0.03 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.38 | 0.12 |

Fold nguoi noi cua mau: {0: 228, 1: 248, 2: 494, 3: 468, 4: 137} · loi chay (khong co ho so, khong tinh): 0
