# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 2225 tep tu nhien (natvi, naten), 2026-10-18 13:25

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 60 tep / kieu co thao tac, 150 tep / kieu tep that (goc), 20 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'natvi': {'goc': {'ngoai_pham_vi': 200}, 'app_that': {}}, 'naten': {'goc': {'lockbox': 1637, 'ngoai_pham_vi': 24}, 'app_that': {'lockbox': 294}}}
- **Req 14 (ho so khong tu mau thuan)**: 2225/2225
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 954/954 thao tac
- PDF: 2225/2225 tep co bao_cao.pdf
- Ngon ngu dung: 2009/2124 tep duoc cham (nguon: {'lid': 2124})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 44, 'K5_khong_con_dong': 44}
- K1 da luu vs K1x that: luu 04 79 tep -> that 04 76; luu khong 04 2058 tep -> that 04 25

**CHI TIEU CHINH - tang goc (tep goc)**: chay 2027 tep, bi loai (Constraint 04 / tu choi) 91, DUOC CHAM 1936

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.670 [bootstrap theo nguoi noi 0.640-0.713] (01 0.634 · 02 0.727 · 03 0.649) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.597 [0.571-0.623] (823/1378) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.950 [0.934-0.963] (823/866) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.077 [0.058-0.102] (43/558) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.018 [0.013-0.025] (35/1936) · <= 0,08
- tep that ra 01 = 0.905 [0.878-0.927] (505/558) · tep co thao tac 03 gan dung 03 = 0.498 [0.466-0.530] (456/916)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 642 | 38 | 604 | 0.719 | 0.668 [0.623-0.711] (290/434) | 0.076 [0.045-0.126] (13/170) | 0.028 [0.018-0.045] (17/604) |
| vi | 1385 | 53 | 1332 | 0.647 | 0.565 [0.533-0.596] (533/944) | 0.077 [0.055-0.108] (30/388) | 0.014 [0.009-0.021] (18/1332) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 1145 | 33 | 1112 | 0.649 | 0.602 [0.569-0.635] (509/845) | 0.116 [0.083-0.160] (31/267) | 0.023 [0.016-0.034] (26/1112) |
| fold 1 | 248 | 16 | 232 | 0.600 | 0.564 [0.492-0.633] (106/188) | 0.045 [0.013-0.151] (2/44) | 0.013 [0.004-0.037] (3/232) |
| fold 2 | 251 | 26 | 225 | 0.771 | 0.657 [0.573-0.732] (88/134) | 0.011 [0.002-0.060] (1/91) | 0.013 [0.005-0.038] (3/225) |
| fold 3 | 173 | 8 | 165 | 0.668 | 0.574 [0.477-0.666] (58/101) | 0.047 [0.016-0.129] (3/64) | 0.012 [0.003-0.043] (2/165) |
| fold 4 | 210 | 8 | 202 | 0.699 | 0.564 [0.470-0.653] (62/110) | 0.065 [0.030-0.135] (6/92) | 0.005 [0.001-0.028] (1/202) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| <= 20 s | 2018 | 90 | 1928 | 0.670 | 0.596 [0.570-0.622] (817/1370) | 0.077 [0.058-0.102] (43/558) | 0.018 [0.013-0.025] (35/1928) |
| > 20 s | 9 | 1 | 8 | 0.462 | 0.750 [0.409-0.929] (6/8) | - (n=0) | 0.000 [0.000-0.324] (0/8) |

**BAO RIENG - tep THAT qua kenh ung dung (khu on mac dinh = 01, quyet dinh 4)**: chay 198 tep, bi loai (Constraint 04 / tu choi) 10, DUOC CHAM 188

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.950 [bootstrap theo nguoi noi 0.924-0.974] (01 0.950) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = - (n=0) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.000 [0.000-0.299] (0/9) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.048 [0.025-0.088] (9/188) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.048 [0.025-0.088] (9/188) · <= 0,08
- tep that ra 01 = 0.904 [0.854-0.939] (170/188) · tep co thao tac 03 gan dung 03 = - (n=0)

**Tep that qua kenh ung dung, theo (bo, kenh)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| naten · app_goi_ghi_lai | 22 | 1 | 21 | 0.950 | - (n=0) | 0.048 [0.008-0.227] (1/21) | 0.048 [0.008-0.227] (1/21) |
| naten · app_volte_ghi_cuoc_goi | 21 | 4 | 17 | 0.938 | - (n=0) | 0.000 [0.000-0.184] (0/17) | 0.118 [0.033-0.343] (2/17) |
| naten · app_whatsapp_telegram | 23 | 3 | 20 | 0.974 | - (n=0) | 0.000 [0.000-0.161] (0/20) | 0.050 [0.009-0.236] (1/20) |
| naten · app_zalo_tin_nhan | 26 | 1 | 25 | 0.913 | - (n=0) | 0.080 [0.022-0.250] (2/25) | 0.080 [0.022-0.250] (2/25) |
| natvi · app_goi_ghi_lai | 25 | 0 | 25 | 1.000 | - (n=0) | 0.000 [0.000-0.133] (0/25) | 0.000 [0.000-0.133] (0/25) |
| natvi · app_volte_ghi_cuoc_goi | 27 | 0 | 27 | 0.981 | - (n=0) | 0.000 [0.000-0.125] (0/27) | 0.037 [0.007-0.183] (1/27) |
| natvi · app_whatsapp_telegram | 27 | 1 | 26 | 0.939 | - (n=0) | 0.115 [0.040-0.290] (3/26) | 0.000 [0.000-0.129] (0/26) |
| natvi · app_zalo_tin_nhan | 27 | 0 | 27 | 0.898 | - (n=0) | 0.111 [0.039-0.281] (3/27) | 0.074 [0.021-0.234] (2/27) |

**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)

| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |
|---|---|---|---|---|---|---|
| app_that | naten | 92 | 9 | quá nhỏ | 9 | 0.098 |
| app_that | natvi | 106 | 1 | quá ít tiếng nói | 1 | 0.009 |
| goc | naten | 642 | 38 | quá nhỏ | 30 | 0.047 |
| goc | naten | 642 | 38 | độ dài ngoài 3 giây – 10 phút | 5 | 0.008 |
| goc | naten | 642 | 38 | quá ít tiếng nói | 5 | 0.008 |
| goc | natvi | 1385 | 53 | không phải ghi âm lời nói | 25 | 0.018 |
| goc | natvi | 1385 | 53 | quá ít tiếng nói | 19 | 0.014 |
| goc | natvi | 1385 | 53 | ồn quá | 7 | 0.005 |
| goc | natvi | 1385 | 53 | băng hẹp | 2 | 0.001 |
| goc | natvi | 1385 | 53 | độ dài ngoài 3 giây – 10 phút | 2 | 0.001 |
| goc | natvi | 1385 | 53 | quá 4 người nói | 1 | 0.001 |
| goc | natvi | 1385 | 53 | nền che tiếng nói | 1 | 0.001 |

**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)

| | 01 | 02 | 03 | 04 | K1 04 |
|---|---|---|---|---|---|
| **01** | 505 | 21 | 22 | 10 | 31 |
| **02** | 128 | 305 | 11 | 18 | 25 |
| **03** | 402 | 51 | 456 | 7 | 35 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| app_that | naten | nen_toan_cuc | 5 | 0 | 5 | 80% | 38%-96% | 4 | 1 | {'voxp': 2, 'edacc': 3} |
| app_that | naten | nguyen_ban | 87 | 9 | 78 | 90% | 81%-95% | 70 | 5 | {'edacc': 32, 'ami': 28, 'voxp': 27} |
| app_that | natvi | nen_toan_cuc | 1 | 0 | 1 | 100% | 21%-100% | 1 | 0 | {'vss': 1} |
| app_that | natvi | nguyen_ban | 82 | 0 | 82 | 91% | 83%-96% | 75 | 1 | {'vss': 37, 'vox': 21, 'celeb': 24} |
| app_that | natvi | nguyen_ban_vsasv | 23 | 1 | 22 | 91% | 72%-97% | 20 | 2 | {'vsasv': 23} |
| goc | naten | da_ghep_giong | 20 | 1 | 19 | 37% | 19%-59% | 4 | 0 | {'voxp': 6, 'edacc': 9, 'ami': 5} |
| goc | naten | da_ghep_nen | 21 | 0 | 21 | 67% | 45%-83% | 5 | 1 | {'ami': 7, 'edacc': 4, 'voxp': 10} |
| goc | naten | da_hai_cho | 21 | 2 | 19 | 89% | 69%-97% | 2 | 0 | {'voxp': 7, 'edacc': 7, 'ami': 7} |
| goc | naten | ghep_chen | 31 | 0 | 31 | 77% | 60%-89% | 5 | 0 | {'voxp': 12, 'edacc': 12, 'ami': 7} |
| goc | naten | ghep_chen_ky | 24 | 1 | 23 | 70% | 49%-84% | 7 | 0 | {'voxp': 8, 'edacc': 8, 'ami': 8} |
| goc | naten | ghep_chen_vung | 21 | 1 | 20 | 90% | 70%-97% | 1 | 0 | {'edacc': 8, 'voxp': 8, 'ami': 5} |
| goc | naten | ghep_thay | 32 | 1 | 31 | 81% | 64%-91% | 4 | 1 | {'edacc': 16, 'voxp': 9, 'ami': 7} |
| goc | naten | ghep_thay_ky | 24 | 1 | 23 | 48% | 29%-67% | 12 | 0 | {'voxp': 6, 'edacc': 11, 'ami': 7} |
| goc | naten | ghep_thay_vung | 23 | 2 | 21 | 81% | 60%-92% | 3 | 0 | {'edacc': 8, 'voxp': 8, 'ami': 7} |
| goc | naten | giong_cuc_bo | 32 | 3 | 29 | 83% | 65%-92% | 4 | 0 | {'ami': 8, 'edacc': 12, 'voxp': 12} |
| goc | naten | giong_cuc_bo_ky | 19 | 0 | 19 | 84% | 62%-94% | 2 | 0 | {'voxp': 9, 'edacc': 8, 'ami': 2} |
| goc | naten | giong_toan_cuc | 31 | 4 | 27 | 56% | 37%-72% | 6 | 6 | {'voxp': 12, 'edacc': 12, 'ami': 7} |
| goc | naten | nen_cuc_bo | 30 | 2 | 28 | 82% | 64%-92% | 5 | 0 | {'edacc': 13, 'voxp': 9, 'ami': 8} |
| goc | naten | nen_cuc_bo_ky | 20 | 0 | 20 | 95% | 76%-99% | 1 | 0 | {'voxp': 12, 'edacc': 6, 'ami': 2} |
| goc | naten | nen_toan_cuc | 46 | 3 | 43 | 28% | 17%-43% | 31 | 8 | {'voxp': 19, 'edacc': 19, 'ami': 8} |
| goc | naten | nguyen_ban | 166 | 9 | 157 | 92% | 86%-95% | 144 | 1 | {'voxp': 62, 'edacc': 59, 'ami': 45} |
| goc | naten | xoa_loi | 31 | 3 | 28 | 36% | 21%-54% | 18 | 0 | {'voxp': 10, 'edacc': 14, 'ami': 7} |
| goc | naten | xoa_loi_ky | 24 | 1 | 23 | 17% | 7%-37% | 19 | 0 | {'voxp': 8, 'ami': 7, 'edacc': 9} |
| goc | naten | xoa_loi_vung | 26 | 4 | 22 | 50% | 31%-69% | 11 | 0 | {'edacc': 12, 'voxp': 7, 'ami': 7} |
| goc | natvi | da_ghep_giong | 48 | 0 | 48 | 31% | 20%-45% | 16 | 0 | {'vss': 20, 'celeb': 15, 'vox': 13} |
| goc | natvi | da_ghep_nen | 44 | 4 | 40 | 40% | 26%-55% | 19 | 1 | {'vss': 20, 'celeb': 11, 'vox': 13} |
| goc | natvi | da_hai_cho | 47 | 0 | 47 | 53% | 39%-67% | 21 | 0 | {'vss': 20, 'celeb': 15, 'vox': 12} |
| goc | natvi | doi_giong_ai | 43 | 8 | 35 | 83% | 67%-92% | 6 | 0 | {'vsasv': 43} |
| goc | natvi | ghep_chen | 48 | 1 | 47 | 43% | 30%-57% | 26 | 0 | {'vox': 14, 'celeb': 14, 'vss': 20} |
| goc | natvi | ghep_chen_ky | 47 | 3 | 44 | 27% | 16%-42% | 31 | 0 | {'vss': 20, 'vox': 16, 'celeb': 11} |
| goc | natvi | ghep_chen_vung | 51 | 0 | 51 | 82% | 70%-90% | 8 | 0 | {'vss': 20, 'celeb': 16, 'vox': 15} |
| goc | natvi | ghep_thay | 42 | 1 | 41 | 66% | 51%-78% | 10 | 1 | {'vss': 20, 'celeb': 12, 'vox': 10} |
| goc | natvi | ghep_thay_ky | 44 | 2 | 42 | 29% | 17%-44% | 30 | 0 | {'celeb': 11, 'vss': 20, 'vox': 13} |
| goc | natvi | ghep_thay_vung | 51 | 1 | 50 | 62% | 48%-74% | 17 | 0 | {'celeb': 16, 'vox': 15, 'vss': 20} |
| goc | natvi | giong_cuc_bo | 50 | 2 | 48 | 67% | 53%-78% | 13 | 0 | {'celeb': 13, 'vss': 20, 'vox': 17} |
| goc | natvi | giong_cuc_bo_ky | 50 | 0 | 50 | 74% | 60%-84% | 13 | 0 | {'vss': 20, 'celeb': 16, 'vox': 14} |
| goc | natvi | giong_toan_cuc | 44 | 1 | 43 | 56% | 41%-70% | 18 | 1 | {'vox': 11, 'celeb': 13, 'vss': 20} |
| goc | natvi | may_ca_file | 20 | 0 | 20 | 0% | 0%-16% | 13 | 3 | {'vss': 4, 'celeb': 9, 'vox': 7} |
| goc | natvi | may_chen | 28 | 0 | 28 | 75% | 57%-87% | 6 | 0 | {'vss': 11, 'vox': 11, 'celeb': 6} |
| goc | natvi | may_thay | 40 | 0 | 40 | 68% | 52%-80% | 13 | 0 | {'vox': 11, 'vss': 18, 'celeb': 11} |
| goc | natvi | nen_cuc_bo | 41 | 2 | 39 | 82% | 67%-91% | 6 | 0 | {'celeb': 11, 'vox': 10, 'vss': 20} |
| goc | natvi | nen_cuc_bo_ky | 42 | 0 | 42 | 90% | 78%-96% | 3 | 0 | {'vss': 20, 'celeb': 12, 'vox': 10} |
| goc | natvi | nen_toan_cuc | 96 | 2 | 94 | 46% | 36%-56% | 59 | 11 | {'vox': 14, 'vss': 61, 'celeb': 21} |
| goc | natvi | nguyen_ban | 210 | 8 | 202 | 91% | 86%-94% | 183 | 1 | {'vss': 91, 'vox': 53, 'celeb': 66} |
| goc | natvi | nguyen_ban_vsasv | 156 | 12 | 144 | 97% | 92%-99% | 139 | 0 | {'vsasv': 156} |
| goc | natvi | xoa_loi | 49 | 0 | 49 | 22% | 13%-36% | 37 | 0 | {'vss': 26, 'celeb': 13, 'vox': 10} |
| goc | natvi | xoa_loi_ky | 46 | 3 | 43 | 7% | 2%-19% | 40 | 0 | {'vox': 9, 'vss': 24, 'celeb': 13} |
| goc | natvi | xoa_loi_vung | 48 | 3 | 45 | 44% | 31%-59% | 24 | 0 | {'vss': 25, 'vox': 12, 'celeb': 11} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| naten | ami | 36 | 0.00 | 89 | 0.60 |
| naten | edacc | 65 | 0.17 | 181 | 0.56 |
| naten | voxp | 69 | 0.03 | 164 | 0.73 |
| natvi | celeb | 71 | 0.04 | 253 | 0.47 |
| natvi | vox | 54 | 0.04 | 253 | 0.43 |
| natvi | vsasv | 144 | 0.03 | 35 | 0.83 |
| natvi | vss | 119 | 0.17 | 403 | 0.57 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 4.43 | 0.46 |
| xlsr | 1.10 | 0.11 |
| tcn | 0.02 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 5.57 | 0.57 |
| tong_s | 5.99 | 0.61 |
| K1 + XLS-R + TCN + dau ca file | 4.88 | 0.57 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.26 | 0.04 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.50 | 0.17 |

Fold nguoi noi cua mau: {0: 1233, 1: 281, 2: 279, 3: 197, 4: 235} · loi chay (khong co ho so, khong tinh): 1
