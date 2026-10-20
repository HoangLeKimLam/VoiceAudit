# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 323 tep tu nhien (longvi, longen), 2026-10-19 01:46

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 30 tep / kieu co thao tac, 60 tep / kieu tep that (goc), 0 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'longvi': {'goc': {}}, 'longen': {'goc': {'lockbox': 82}}}
- **Req 14 (ho so khong tu mau thuan)**: 323/323
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 126/126 thao tac
- PDF: 323/323 tep co bao_cao.pdf
- Ngon ngu dung: 317/321 tep duoc cham (nguon: {'lid': 321})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 20, 'K5_khong_con_dong': 20}

**CHI TIEU CHINH - tang goc (tep goc)**: chay 323 tep, bi loai (Constraint 04 / tu choi) 2, DUOC CHAM 321

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.382 [bootstrap theo nguoi noi 0.327-0.441] (01 0.489 · 02 0.561 · 03 0.095) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.345 [0.286-0.410] (77/223) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.837 [0.748-0.899] (77/92) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.153 [0.095-0.237] (15/98) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.062 [0.041-0.094] (20/321) · <= 0,08
- tep that ra 01 = 0.765 [0.672-0.838] (75/98) · tep co thao tac 03 gan dung 03 = 0.050 [0.023-0.105] (6/120)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 123 | 0 | 123 | 0.447 | 0.549 [0.441-0.652] (45/82) | 0.244 [0.138-0.393] (10/41) | 0.016 [0.004-0.057] (2/123) |
| vi | 200 | 2 | 198 | 0.337 | 0.227 [0.166-0.303] (32/141) | 0.088 [0.038-0.189] (5/57) | 0.091 [0.058-0.139] (18/198) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 35 | 0 | 35 | 0.408 | 0.333 [0.180-0.533] (8/24) | 0.182 [0.051-0.477] (2/11) | 0.000 [0.000-0.099] (0/35) |
| fold 1 | 40 | 0 | 40 | 0.361 | 0.607 [0.424-0.764] (17/28) | 0.250 [0.089-0.532] (3/12) | 0.000 [0.000-0.088] (0/40) |
| fold 2 | 104 | 2 | 102 | 0.296 | 0.181 [0.109-0.285] (13/72) | 0.100 [0.035-0.256] (3/30) | 0.098 [0.054-0.171] (10/102) |
| fold 3 | 97 | 0 | 97 | 0.489 | 0.424 [0.312-0.544] (28/66) | 0.097 [0.033-0.249] (3/31) | 0.062 [0.029-0.128] (6/97) |
| fold 4 | 47 | 0 | 47 | 0.335 | 0.333 [0.198-0.504] (11/33) | 0.286 [0.117-0.546] (4/14) | 0.085 [0.034-0.199] (4/47) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| > 20 s | 323 | 2 | 321 | 0.382 | 0.345 [0.286-0.410] (77/223) | 0.153 [0.095-0.237] (15/98) | 0.062 [0.041-0.094] (20/321) |

**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)

| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |
|---|---|---|---|---|---|---|
| goc | longen | 123 | 0 | - | 0 | 0.000 |
| goc | longvi | 200 | 2 | quá ít tiếng nói | 2 | 0.010 |

**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)

| | 01 | 02 | 03 | 04 | K1 04 |
|---|---|---|---|---|---|
| **01** | 75 | 15 | 0 | 8 | 1 |
| **02** | 45 | 53 | 0 | 5 | 0 |
| **03** | 89 | 18 | 6 | 7 | 1 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| goc | longen | ghep_chen | 15 | 0 | 15 | 20% | 7%-45% | 8 | 1 | {'fosdem': 12, 'yt': 3} |
| goc | longen | ghep_thay | 15 | 0 | 15 | 7% | 1%-30% | 9 | 0 | {'fosdem': 10, 'yt': 4, 'ccc': 1} |
| goc | longen | giong_cuc_bo | 12 | 0 | 12 | 83% | 55%-95% | 2 | 0 | {'yt': 2, 'fosdem': 10} |
| goc | longen | giong_toan_cuc | 5 | 0 | 5 | 80% | 38%-96% | 1 | 0 | {'fosdem': 5} |
| goc | longen | nen_cuc_bo | 13 | 0 | 13 | 77% | 50%-92% | 3 | 0 | {'fosdem': 11, 'yt': 2} |
| goc | longen | nen_toan_cuc | 9 | 0 | 9 | 44% | 19%-73% | 4 | 1 | {'yt': 4, 'fosdem': 4, 'ccc': 1} |
| goc | longen | nguyen_ban | 38 | 0 | 38 | 76% | 61%-87% | 29 | 0 | {'fosdem': 27, 'yt': 10, 'ccc': 1} |
| goc | longen | xoa_loi | 16 | 0 | 16 | 0% | 0%-19% | 10 | 0 | {'yt': 6, 'fosdem': 10} |
| goc | longvi | ghep_chen | 25 | 0 | 25 | 4% | 1%-20% | 22 | 2 | {'yt': 25} |
| goc | longvi | ghep_thay | 25 | 0 | 25 | 4% | 1%-20% | 21 | 1 | {'yt': 25} |
| goc | longvi | giong_cuc_bo | 25 | 0 | 25 | 12% | 4%-30% | 18 | 4 | {'yt': 25} |
| goc | longvi | giong_toan_cuc | 10 | 0 | 10 | 60% | 31%-83% | 4 | 0 | {'yt': 10} |
| goc | longvi | nen_cuc_bo | 25 | 0 | 25 | 52% | 33%-70% | 12 | 0 | {'yt': 25} |
| goc | longvi | nen_toan_cuc | 10 | 0 | 10 | 60% | 31%-83% | 4 | 2 | {'yt': 10} |
| goc | longvi | nguyen_ban | 55 | 1 | 54 | 80% | 67%-88% | 43 | 6 | {'yt': 55} |
| goc | longvi | xoa_loi | 25 | 1 | 24 | 0% | 0%-14% | 19 | 3 | {'yt': 25} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| longen | ccc | 2 | 1.00 | 1 | 1.00 |
| longen | fosdem | 28 | 0.14 | 61 | 0.44 |
| longen | yt | 11 | 0.36 | 20 | 0.15 |
| longvi | yt | 57 | 0.09 | 141 | 0.20 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 26.99 | 0.18 |
| xlsr | 23.33 | 0.14 |
| tcn | 0.36 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 51.09 | 0.32 |
| tong_s | 51.42 | 0.32 |
| K1 + XLS-R + TCN + dau ca file | 50.90 | 0.32 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.38 | 0.00 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.41 | 0.01 |

Fold nguoi noi cua mau: {0: 35, 1: 40, 2: 104, 3: 97, 4: 47} · loi chay (khong co ho so, khong tinh): 0
