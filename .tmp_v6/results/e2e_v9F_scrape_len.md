# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 1575 tep tu nhien (scrapevi, scrapeen), 2026-10-19 16:10

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 60 tep / kieu co thao tac, 600 tep / kieu tep that (goc), 0 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'scrapevi': {'goc': {}}, 'scrapeen': {'goc': {'lockbox': 683}}}
- **Req 14 (ho so khong tu mau thuan)**: 1575/1575
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 685/685 thao tac
- PDF: 1575/1575 tep co bao_cao.pdf
- Ngon ngu dung: 1568/1568 tep duoc cham (nguon: {'lid': 1568})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 99, 'K5_khong_con_dong': 99}

**CHI TIEU CHINH - tang goc (tep goc)**: chay 1575 tep, bi loai (Constraint 04 / tu choi) 7, DUOC CHAM 1568

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.712 [bootstrap theo nguoi noi 0.681-0.740] (01 0.739 · 02 0.713 · 03 0.686) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.651 [0.618-0.682] (544/836) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.851 [0.822-0.877] (544/639) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.130 [0.107-0.156] (95/732) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.063 [0.052-0.076] (99/1568) · <= 0,08
- tep that ra 01 = 0.788 [0.757-0.816] (577/732) · tep co thao tac 03 gan dung 03 = 0.541 [0.489-0.592] (193/357)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 805 | 5 | 800 | 0.765 | 0.734 [0.690-0.775] (307/418) | 0.147 [0.115-0.186] (56/382) | 0.035 [0.024-0.050] (28/800) |
| vi | 770 | 2 | 768 | 0.651 | 0.567 [0.519-0.614] (237/418) | 0.111 [0.083-0.149] (39/350) | 0.092 [0.074-0.115] (71/768) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 228 | 4 | 224 | 0.756 | 0.728 [0.640-0.801] (83/114) | 0.136 [0.084-0.213] (15/110) | 0.022 [0.010-0.051] (5/224) |
| fold 1 | 248 | 1 | 247 | 0.760 | 0.689 [0.606-0.762] (91/132) | 0.122 [0.074-0.194] (14/115) | 0.061 [0.037-0.098] (15/247) |
| fold 2 | 494 | 2 | 492 | 0.701 | 0.626 [0.568-0.681] (176/281) | 0.090 [0.058-0.136] (19/211) | 0.081 [0.060-0.109] (40/492) |
| fold 3 | 468 | 0 | 468 | 0.701 | 0.635 [0.573-0.693] (155/244) | 0.143 [0.103-0.195] (32/224) | 0.056 [0.038-0.080] (26/468) |
| fold 4 | 137 | 0 | 137 | 0.621 | 0.600 [0.479-0.710] (39/65) | 0.208 [0.131-0.316] (15/72) | 0.095 [0.056-0.156] (13/137) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| <= 20 s | 1570 | 7 | 1563 | 0.712 | 0.650 [0.617-0.681] (540/831) | 0.130 [0.107-0.156] (95/732) | 0.063 [0.052-0.077] (99/1563) |
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
| **01** | 577 | 84 | 11 | 60 | 3 |
| **02** | 120 | 325 | 2 | 32 | 1 |
| **03** | 133 | 24 | 193 | 7 | 3 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| goc | scrapeen | ghep_chen | 60 | 0 | 60 | 85% | 74%-92% | 6 | 0 | {'fosdem': 30, 'yt': 30} |
| goc | scrapeen | ghep_thay | 60 | 0 | 60 | 82% | 70%-89% | 6 | 0 | {'yt': 30, 'fosdem': 30} |
| goc | scrapeen | giong_cuc_bo | 60 | 0 | 60 | 93% | 84%-97% | 4 | 0 | {'fosdem': 31, 'yt': 29} |
| goc | scrapeen | giong_toan_cuc | 60 | 0 | 60 | 75% | 63%-84% | 11 | 4 | {'fosdem': 30, 'yt': 30} |
| goc | scrapeen | nen_cuc_bo | 60 | 0 | 60 | 85% | 74%-92% | 7 | 1 | {'fosdem': 30, 'yt': 29, 'ccc': 1} |
| goc | scrapeen | nen_toan_cuc | 76 | 0 | 76 | 38% | 28%-49% | 47 | 9 | {'fosdem': 45, 'yt': 30, 'ccc': 1} |
| goc | scrapeen | nguyen_ban | 369 | 3 | 366 | 82% | 77%-85% | 299 | 14 | {'fosdem': 246, 'yt': 122, 'ccc': 1} |
| goc | scrapeen | xoa_loi | 60 | 2 | 58 | 40% | 28%-53% | 29 | 0 | {'yt': 29, 'fosdem': 30, 'ccc': 1} |
| goc | scrapevi | ghep_chen | 60 | 1 | 59 | 46% | 34%-58% | 28 | 2 | {'yt': 60} |
| goc | scrapevi | ghep_thay | 60 | 0 | 60 | 55% | 42%-67% | 24 | 1 | {'yt': 60} |
| goc | scrapevi | giong_cuc_bo | 60 | 0 | 60 | 77% | 65%-86% | 13 | 1 | {'yt': 60} |
| goc | scrapevi | giong_toan_cuc | 60 | 0 | 60 | 57% | 44%-68% | 20 | 6 | {'yt': 60} |
| goc | scrapevi | nen_cuc_bo | 60 | 0 | 60 | 82% | 70%-89% | 9 | 2 | {'yt': 60} |
| goc | scrapevi | nen_toan_cuc | 83 | 1 | 82 | 54% | 43%-64% | 38 | 14 | {'yt': 83} |
| goc | scrapevi | nguyen_ban | 327 | 0 | 327 | 76% | 71%-80% | 249 | 41 | {'yt': 327} |
| goc | scrapevi | xoa_loi | 60 | 0 | 60 | 17% | 9%-28% | 40 | 4 | {'yt': 60} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| scrapeen | ccc | 2 | 0.50 | 2 | 1.00 |
| scrapeen | fosdem | 260 | 0.10 | 210 | 0.71 |
| scrapeen | yt | 120 | 0.25 | 206 | 0.68 |
| scrapevi | yt | 350 | 0.11 | 418 | 0.54 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 4.83 | 0.39 |
| xlsr | 1.47 | 0.11 |
| tcn | 0.02 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 6.41 | 0.51 |
| tong_s | 6.77 | 0.54 |
| K1 + XLS-R + TCN + dau ca file | 6.41 | 0.51 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.40 | 0.03 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.48 | 0.13 |

Fold nguoi noi cua mau: {0: 228, 1: 248, 2: 494, 3: 468, 4: 137} · loi chay (khong co ho so, khong tinh): 0
