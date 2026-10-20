# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 323 tep tu nhien (longvi, longen), 2026-10-18 05:05

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 30 tep / kieu co thao tac, 60 tep / kieu tep that (goc), 0 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'longvi': {'goc': {}}, 'longen': {'goc': {'lockbox': 82}}}
- **Req 14 (ho so khong tu mau thuan)**: 323/323
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 703/703 thao tac
- PDF: 323/323 tep co bao_cao.pdf
- Ngon ngu dung: 317/321 tep duoc cham (nguon: {'lid': 321})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 16, 'K5_khong_con_dong': 16}

**CHI TIEU CHINH - tang goc (tep goc)**: chay 323 tep, bi loai (Constraint 04 / tu choi) 2, DUOC CHAM 321

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.540 [bootstrap theo nguoi noi 0.482-0.605] (01 0.443 · 02 0.629 · 03 0.547) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.753 [0.693-0.805] (168/223) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.771 [0.710-0.821] (168/218) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.510 [0.413-0.607] (50/98) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.050 [0.031-0.079] (16/321) · <= 0,08
- tep that ra 01 = 0.418 [0.326-0.517] (41/98) · tep co thao tac 03 gan dung 03 = 0.483 [0.396-0.572] (58/120)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 123 | 0 | 123 | 0.561 | 0.927 [0.849-0.966] (76/82) | 0.683 [0.530-0.804] (28/41) | 0.016 [0.004-0.057] (2/123) |
| vi | 200 | 2 | 198 | 0.501 | 0.652 [0.571-0.726] (92/141) | 0.386 [0.271-0.516] (22/57) | 0.071 [0.043-0.115] (14/198) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 35 | 0 | 35 | 0.461 | 0.625 [0.427-0.788] (15/24) | 0.545 [0.280-0.787] (6/11) | 0.000 [0.000-0.099] (0/35) |
| fold 1 | 40 | 0 | 40 | 0.621 | 0.929 [0.774-0.980] (26/28) | 0.583 [0.320-0.807] (7/12) | 0.000 [0.000-0.088] (0/40) |
| fold 2 | 104 | 2 | 102 | 0.486 | 0.764 [0.654-0.847] (55/72) | 0.600 [0.423-0.754] (18/30) | 0.088 [0.047-0.159] (9/102) |
| fold 3 | 97 | 0 | 97 | 0.600 | 0.742 [0.626-0.833] (49/66) | 0.323 [0.186-0.499] (10/31) | 0.062 [0.029-0.128] (6/97) |
| fold 4 | 47 | 0 | 47 | 0.452 | 0.697 [0.527-0.826] (23/33) | 0.643 [0.388-0.837] (9/14) | 0.021 [0.004-0.111] (1/47) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| > 20 s | 323 | 2 | 321 | 0.540 | 0.753 [0.693-0.805] (168/223) | 0.510 [0.413-0.607] (50/98) | 0.050 [0.031-0.079] (16/321) |

**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)

| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |
|---|---|---|---|---|---|---|
| goc | longen | 123 | 0 | - | 0 | 0.000 |
| goc | longvi | 200 | 2 | quá ít tiếng nói | 2 | 0.010 |

**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)

| | 01 | 02 | 03 | 04 | K1 04 |
|---|---|---|---|---|---|
| **01** | 41 | 30 | 20 | 7 | 1 |
| **02** | 14 | 72 | 14 | 3 | 0 |
| **03** | 32 | 24 | 58 | 6 | 1 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| goc | longen | ghep_chen | 15 | 0 | 15 | 93% | 70%-99% | 0 | 1 | {'fosdem': 12, 'yt': 3} |
| goc | longen | ghep_thay | 15 | 0 | 15 | 100% | 80%-100% | 0 | 0 | {'fosdem': 10, 'yt': 4, 'ccc': 1} |
| goc | longen | giong_cuc_bo | 12 | 0 | 12 | 67% | 39%-86% | 1 | 0 | {'yt': 2, 'fosdem': 10} |
| goc | longen | giong_toan_cuc | 5 | 0 | 5 | 80% | 38%-96% | 1 | 0 | {'fosdem': 5} |
| goc | longen | nen_cuc_bo | 13 | 0 | 13 | 62% | 36%-82% | 1 | 0 | {'fosdem': 11, 'yt': 2} |
| goc | longen | nen_toan_cuc | 9 | 0 | 9 | 22% | 6%-55% | 0 | 1 | {'yt': 4, 'fosdem': 4, 'ccc': 1} |
| goc | longen | nguyen_ban | 38 | 0 | 38 | 32% | 19%-47% | 12 | 0 | {'fosdem': 27, 'yt': 10, 'ccc': 1} |
| goc | longen | xoa_loi | 16 | 0 | 16 | 56% | 33%-77% | 2 | 0 | {'yt': 6, 'fosdem': 10} |
| goc | longvi | ghep_chen | 25 | 0 | 25 | 28% | 14%-48% | 9 | 1 | {'yt': 25} |
| goc | longvi | ghep_thay | 25 | 0 | 25 | 32% | 17%-52% | 11 | 1 | {'yt': 25} |
| goc | longvi | giong_cuc_bo | 25 | 0 | 25 | 68% | 48%-83% | 4 | 2 | {'yt': 25} |
| goc | longvi | giong_toan_cuc | 10 | 0 | 10 | 60% | 31%-83% | 4 | 0 | {'yt': 10} |
| goc | longvi | nen_cuc_bo | 25 | 0 | 25 | 88% | 70%-96% | 2 | 0 | {'yt': 25} |
| goc | longvi | nen_toan_cuc | 10 | 0 | 10 | 60% | 31%-83% | 2 | 2 | {'yt': 10} |
| goc | longvi | nguyen_ban | 55 | 1 | 54 | 52% | 39%-65% | 28 | 5 | {'yt': 55} |
| goc | longvi | xoa_loi | 25 | 1 | 24 | 21% | 9%-40% | 10 | 3 | {'yt': 25} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| longen | ccc | 2 | 1.00 | 1 | 1.00 |
| longen | fosdem | 28 | 0.61 | 61 | 0.72 |
| longen | yt | 11 | 0.82 | 20 | 0.75 |
| longvi | yt | 57 | 0.39 | 141 | 0.50 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 27.64 | 0.17 |
| xlsr | 22.27 | 0.14 |
| tcn | 0.34 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 50.39 | 0.33 |
| tong_s | 51.05 | 0.33 |
| K1 + XLS-R + TCN + dau ca file | 49.77 | 0.32 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.59 | 0.00 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.49 | 0.01 |

Fold nguoi noi cua mau: {0: 35, 1: 40, 2: 104, 3: 97, 4: 47} · loi chay (khong co ho so, khong tinh): 0
