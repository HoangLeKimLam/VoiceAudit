# Kiem thu dau-cuoi v9 - diem van hanh F (dinh tuyen fold (ngoai mau)), 1409 tep tu nhien (natvi, naten), 2026-10-13 11:25

- Cau hinh: k1x BAT · ngon ngu LID (khong bao truoc) · PDF bat · bound/reg/heads = fuse1/fuse1/h2 · tep nguong thresholds_v9F.json (sha1 3cf1466784) · kiem sha1 checkpoint: verified · strict BAT · hieu chinh: verified
- Mau: toi da 20 tep / kieu co thao tac, 150 tep / kieu tep that (goc), 20 tep that / (bo, kenh ung dung); KHONG loc theo K1 da luu: K1x chay that quyet dinh
- Loai khoi MAU khi lap mau (khong phai Constraint): {'natvi': {'goc': {'ngoai_pham_vi': 200}, 'app_that': {}}, 'naten': {'goc': {'lockbox': 1637, 'ngoai_pham_vi': 24}, 'app_that': {'lockbox': 260}}}
- **Req 14 (ho so khong tu mau thuan)**: 1409/1409
- **Req 12 (moi thao tac co anh + >= 2 dau hieu + giai thich)**: 530/530 thao tac
- PDF: 1409/1409 tep co bao_cao.pdf
- Ngon ngu dung: 1245/1332 tep duoc cham (nguon: {'lid': 1332})
- Xac suat ca tep: co hieu chinh (OOF 5 fold; dinh tuyen fold: mo hinh ngoai mau, buoc hieu chinh TRONG MAU cho fold cua tung tep (bo hieu chinh hoc ca fold do))
- Dau ca tep theo fold: heads_final_h2_f{k} hoc lai 10/10, khong trung het OOF (xem heads2_h2_perfold.md)
- 04 do phan tich (luat 5) theo duong: {'K3_vung_xam': 21, 'K5_khong_con_dong': 21}
- K1 da luu vs K1x that: luu 04 65 tep -> that 04 62; luu khong 04 1284 tep -> that 04 15

**CHI TIEU CHINH - tang goc (tep goc)**: chay 1249 tep, bi loai (Constraint 04 / tu choi) 67, DUOC CHAM 1182

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.704 [bootstrap theo nguoi noi 0.668-0.735] (01 0.717 · 02 0.742 · 03 0.654) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = 0.607 [0.572-0.641] (459/756) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.944 [0.920-0.962] (459/486) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.063 [0.044-0.091] (27/426) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.012 [0.007-0.020] (14/1182) · <= 0,08
- tep that ra 01 = 0.932 [0.904-0.952] (397/426) · tep co thao tac 03 gan dung 03 = 0.503 [0.460-0.546] (259/515)

**Theo ngon ngu (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| en | 509 | 37 | 472 | 0.729 | 0.677 [0.625-0.725] (224/331) | 0.078 [0.044-0.134] (11/141) | 0.017 [0.009-0.033] (8/472) |
| vi | 740 | 30 | 710 | 0.680 | 0.553 [0.505-0.600] (235/425) | 0.056 [0.035-0.089] (16/285) | 0.008 [0.004-0.018] (6/710) |

**Theo fold nguoi noi (tang goc; fold 0 chua ~56 % natvi -> mo hinh / nguong fold 0 hoc tren it du lieu Viet)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| fold 0 | 443 | 13 | 430 | 0.696 | 0.631 [0.574-0.684] (186/295) | 0.111 [0.068-0.175] (15/135) | 0.014 [0.006-0.030] (6/430) |
| fold 1 | 172 | 12 | 160 | 0.654 | 0.560 [0.470-0.647] (65/116) | 0.045 [0.013-0.151] (2/44) | 0.013 [0.003-0.044] (2/160) |
| fold 2 | 251 | 26 | 225 | 0.771 | 0.657 [0.573-0.732] (88/134) | 0.011 [0.002-0.060] (1/91) | 0.013 [0.005-0.038] (3/225) |
| fold 3 | 173 | 8 | 165 | 0.668 | 0.574 [0.477-0.666] (58/101) | 0.047 [0.016-0.129] (3/64) | 0.012 [0.003-0.043] (2/165) |
| fold 4 | 210 | 8 | 202 | 0.699 | 0.564 [0.470-0.653] (62/110) | 0.065 [0.030-0.135] (6/92) | 0.005 [0.001-0.028] (1/202) |

**Theo do dai tep (tang goc)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| <= 20 s | 1241 | 66 | 1175 | 0.704 | 0.606 [0.571-0.641] (454/749) | 0.063 [0.044-0.091] (27/426) | 0.012 [0.007-0.020] (14/1175) |
| > 20 s | 8 | 1 | 7 | 0.455 | 0.714 [0.359-0.918] (5/7) | - (n=0) | 0.000 [0.000-0.354] (0/7) |

**BAO RIENG - tep THAT qua kenh ung dung (khu on mac dinh = 01, quyet dinh 4)**: chay 160 tep, bi loai (Constraint 04 / tu choi) 10, DUOC CHAM 150

- **Req 1 macro-F1 01/02/03** (04 = bo sot nhan dung) = 0.958 [bootstrap theo nguoi noi 0.931-0.983] (01 0.958) · chi tieu >= 0,86
- **Req 2 khong bo sot** (tep sua -> 02/03; ra 04 = khong bat) = - (n=0) · >= 0,90
- **Req 3 dang tin** (02/03 ma that bi sua) = 0.000 [0.000-0.434] (0/5) · >= 0,97
- **Req 4 vu oan** (tep that -> 02/03; tep that ra 04 van o mau so) = 0.033 [0.014-0.076] (5/150) · <= 0,03
- **Req 9 tu choi dung muc** (04 phan tich / tep duoc cham) = 0.047 [0.023-0.093] (7/150) · <= 0,08
- tep that ra 01 = 0.920 [0.865-0.954] (138/150) · tep co thao tac 03 gan dung 03 = - (n=0)

**Tep that qua kenh ung dung, theo (bo, kenh)** (tep duoc cham; Wilson 95 %)

| Nhom | chay | K1 04 | cham | Req 1 macro-F1 | Req 2 | Req 4 | Req 9 |
|---|---|---|---|---|---|---|---|
| naten · app_goi_ghi_lai | 20 | 1 | 19 | 0.944 | - (n=0) | 0.053 [0.009-0.246] (1/19) | 0.053 [0.009-0.246] (1/19) |
| naten · app_volte_ghi_cuoc_goi | 20 | 4 | 16 | 0.968 | - (n=0) | 0.000 [0.000-0.194] (0/16) | 0.062 [0.011-0.283] (1/16) |
| naten · app_whatsapp_telegram | 20 | 3 | 17 | 0.970 | - (n=0) | 0.000 [0.000-0.184] (0/17) | 0.059 [0.010-0.270] (1/17) |
| naten · app_zalo_tin_nhan | 20 | 1 | 19 | 0.944 | - (n=0) | 0.053 [0.009-0.246] (1/19) | 0.053 [0.009-0.246] (1/19) |
| natvi · app_goi_ghi_lai | 20 | 0 | 20 | 1.000 | - (n=0) | 0.000 [0.000-0.161] (0/20) | 0.000 [0.000-0.161] (0/20) |
| natvi · app_volte_ghi_cuoc_goi | 20 | 0 | 20 | 0.974 | - (n=0) | 0.000 [0.000-0.161] (0/20) | 0.050 [0.009-0.236] (1/20) |
| natvi · app_whatsapp_telegram | 20 | 1 | 19 | 0.973 | - (n=0) | 0.053 [0.009-0.246] (1/19) | 0.000 [0.000-0.168] (0/19) |
| natvi · app_zalo_tin_nhan | 20 | 0 | 20 | 0.889 | - (n=0) | 0.100 [0.028-0.301] (2/20) | 0.100 [0.028-0.301] (2/20) |

**Tep bi loai theo ly do** (Constraint 04 / tu choi cua K1x chay that; dieu kien hoc 04 cua Product tinh nhu K1)

| Tang | Bo | chay | bi loai (K1 04 / tu choi) | ly do | so tep | ti le / tep chay |
|---|---|---|---|---|---|---|
| app_that | naten | 80 | 9 | quá nhỏ | 9 | 0.113 |
| app_that | natvi | 80 | 1 | quá ít tiếng nói | 1 | 0.013 |
| goc | naten | 509 | 37 | quá nhỏ | 30 | 0.059 |
| goc | naten | 509 | 37 | quá ít tiếng nói | 5 | 0.010 |
| goc | naten | 509 | 37 | độ dài ngoài 3 giây – 10 phút | 4 | 0.008 |
| goc | natvi | 740 | 30 | không phải ghi âm lời nói | 14 | 0.019 |
| goc | natvi | 740 | 30 | quá ít tiếng nói | 11 | 0.015 |
| goc | natvi | 740 | 30 | ồn quá | 4 | 0.005 |
| goc | natvi | 740 | 30 | độ dài ngoài 3 giây – 10 phút | 2 | 0.003 |
| goc | natvi | 740 | 30 | băng hẹp | 1 | 0.001 |

**Ma tran nham lan - tang goc** (hang: nhan dung, cot: nhan he thong; 'K1 04' = bi loai, khong cham)

| | 01 | 02 | 03 | 04 | K1 04 |
|---|---|---|---|---|---|
| **01** | 397 | 11 | 16 | 2 | 24 |
| **02** | 66 | 167 | 2 | 6 | 18 |
| **03** | 219 | 31 | 259 | 6 | 25 |

**Dung nhan theo (tang, bo, kieu)** - tep duoc cham, Wilson 95 %

| Tang | Bo | Kieu | chay | K1 04 | cham | dung | khoang 95 % | ra 01 | ra 04 | nguon |
|---|---|---|---|---|---|---|---|---|---|---|
| app_that | naten | nguyen_ban | 80 | 9 | 71 | 92% | 83%-96% | 65 | 4 | {'edacc': 28, 'ami': 28, 'voxp': 24} |
| app_that | natvi | nguyen_ban | 60 | 0 | 60 | 93% | 84%-97% | 56 | 1 | {'vss': 20, 'vox': 20, 'celeb': 20} |
| app_that | natvi | nguyen_ban_vsasv | 20 | 1 | 19 | 89% | 69%-97% | 17 | 2 | {'vsasv': 20} |
| goc | naten | da_ghep_giong | 20 | 1 | 19 | 37% | 19%-59% | 4 | 0 | {'voxp': 6, 'edacc': 9, 'ami': 5} |
| goc | naten | da_ghep_nen | 20 | 0 | 20 | 65% | 43%-82% | 5 | 1 | {'ami': 7, 'edacc': 4, 'voxp': 9} |
| goc | naten | da_hai_cho | 20 | 2 | 18 | 89% | 67%-97% | 2 | 0 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | ghep_chen | 20 | 0 | 20 | 70% | 48%-85% | 5 | 0 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | ghep_chen_ky | 20 | 1 | 19 | 68% | 46%-85% | 6 | 0 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | ghep_chen_vung | 20 | 1 | 19 | 89% | 69%-97% | 1 | 0 | {'edacc': 8, 'voxp': 7, 'ami': 5} |
| goc | naten | ghep_thay | 20 | 1 | 19 | 84% | 62%-94% | 2 | 1 | {'edacc': 7, 'voxp': 6, 'ami': 7} |
| goc | naten | ghep_thay_ky | 20 | 1 | 19 | 42% | 23%-64% | 11 | 0 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | ghep_thay_vung | 20 | 2 | 18 | 83% | 61%-94% | 2 | 0 | {'edacc': 7, 'voxp': 6, 'ami': 7} |
| goc | naten | giong_cuc_bo | 20 | 3 | 17 | 76% | 53%-90% | 4 | 0 | {'ami': 7, 'edacc': 7, 'voxp': 6} |
| goc | naten | giong_cuc_bo_ky | 19 | 0 | 19 | 84% | 62%-94% | 2 | 0 | {'voxp': 9, 'edacc': 8, 'ami': 2} |
| goc | naten | giong_toan_cuc | 20 | 4 | 16 | 56% | 33%-77% | 4 | 3 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | nen_cuc_bo | 20 | 2 | 18 | 83% | 61%-94% | 3 | 0 | {'edacc': 7, 'voxp': 6, 'ami': 7} |
| goc | naten | nen_cuc_bo_ky | 20 | 0 | 20 | 95% | 76%-99% | 1 | 0 | {'voxp': 12, 'edacc': 6, 'ami': 2} |
| goc | naten | nen_toan_cuc | 20 | 3 | 17 | 6% | 1%-27% | 14 | 2 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | nguyen_ban | 150 | 9 | 141 | 91% | 86%-95% | 129 | 1 | {'voxp': 52, 'edacc': 53, 'ami': 45} |
| goc | naten | xoa_loi | 20 | 2 | 18 | 44% | 25%-66% | 10 | 0 | {'voxp': 6, 'edacc': 7, 'ami': 7} |
| goc | naten | xoa_loi_ky | 20 | 1 | 19 | 21% | 9%-43% | 15 | 0 | {'voxp': 6, 'ami': 7, 'edacc': 7} |
| goc | naten | xoa_loi_vung | 20 | 4 | 16 | 44% | 23%-67% | 9 | 0 | {'edacc': 7, 'voxp': 6, 'ami': 7} |
| goc | natvi | da_ghep_giong | 20 | 0 | 20 | 30% | 15%-52% | 6 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | da_ghep_nen | 20 | 3 | 17 | 59% | 36%-78% | 5 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | da_hai_cho | 20 | 0 | 20 | 50% | 30%-70% | 10 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | doi_giong_ai | 20 | 3 | 17 | 82% | 59%-94% | 3 | 0 | {'vsasv': 20} |
| goc | natvi | ghep_chen | 20 | 0 | 20 | 20% | 8%-42% | 15 | 0 | {'vox': 7, 'celeb': 7, 'vss': 6} |
| goc | natvi | ghep_chen_ky | 20 | 2 | 18 | 22% | 9%-45% | 14 | 0 | {'vss': 6, 'vox': 7, 'celeb': 7} |
| goc | natvi | ghep_chen_vung | 20 | 0 | 20 | 80% | 58%-92% | 4 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | ghep_thay | 20 | 0 | 20 | 70% | 48%-85% | 4 | 1 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | ghep_thay_ky | 20 | 2 | 18 | 28% | 12%-51% | 13 | 0 | {'celeb': 7, 'vss': 6, 'vox': 7} |
| goc | natvi | ghep_thay_vung | 20 | 0 | 20 | 50% | 30%-70% | 9 | 0 | {'celeb': 7, 'vox': 7, 'vss': 6} |
| goc | natvi | giong_cuc_bo | 20 | 0 | 20 | 65% | 43%-82% | 6 | 0 | {'celeb': 7, 'vss': 6, 'vox': 7} |
| goc | natvi | giong_cuc_bo_ky | 20 | 0 | 20 | 75% | 53%-89% | 5 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | giong_toan_cuc | 20 | 1 | 19 | 53% | 32%-73% | 9 | 0 | {'vox': 7, 'celeb': 7, 'vss': 6} |
| goc | natvi | may_ca_file | 20 | 0 | 20 | 0% | 0%-16% | 13 | 3 | {'vss': 4, 'celeb': 9, 'vox': 7} |
| goc | natvi | may_chen | 20 | 0 | 20 | 80% | 58%-92% | 3 | 0 | {'vss': 7, 'vox': 7, 'celeb': 6} |
| goc | natvi | may_thay | 20 | 0 | 20 | 70% | 48%-85% | 6 | 0 | {'vox': 7, 'vss': 6, 'celeb': 7} |
| goc | natvi | nen_cuc_bo | 20 | 2 | 18 | 89% | 67%-97% | 2 | 0 | {'celeb': 7, 'vox': 7, 'vss': 6} |
| goc | natvi | nen_cuc_bo_ky | 20 | 0 | 20 | 95% | 76%-99% | 1 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | nen_toan_cuc | 20 | 0 | 20 | 35% | 18%-57% | 12 | 1 | {'vox': 7, 'vss': 6, 'celeb': 7} |
| goc | natvi | nguyen_ban | 150 | 4 | 146 | 92% | 86%-95% | 134 | 1 | {'vss': 50, 'vox': 50, 'celeb': 50} |
| goc | natvi | nguyen_ban_vsasv | 150 | 11 | 139 | 96% | 92%-98% | 134 | 0 | {'vsasv': 150} |
| goc | natvi | xoa_loi | 20 | 0 | 20 | 25% | 11%-47% | 15 | 0 | {'vss': 6, 'celeb': 7, 'vox': 7} |
| goc | natvi | xoa_loi_ky | 20 | 1 | 19 | 5% | 1%-25% | 18 | 0 | {'vox': 7, 'vss': 6, 'celeb': 7} |
| goc | natvi | xoa_loi_vung | 20 | 1 | 19 | 32% | 15%-54% | 12 | 0 | {'vss': 6, 'vox': 7, 'celeb': 7} |

**Theo nguon - tang goc** (tep duoc cham; tep that: vu oan; tep sua: dung nhan)

| Bo | Nguon | that n | vu oan | sua n | dung |
|---|---|---|---|---|---|
| naten | ami | 36 | 0.00 | 85 | 0.59 |
| naten | edacc | 53 | 0.19 | 125 | 0.55 |
| naten | voxp | 52 | 0.02 | 121 | 0.76 |
| natvi | celeb | 49 | 0.02 | 140 | 0.44 |
| natvi | vox | 50 | 0.04 | 146 | 0.45 |
| natvi | vsasv | 139 | 0.04 | 17 | 0.82 |
| natvi | vss | 47 | 0.17 | 122 | 0.61 |

**Thoi gian** (CPU, int8; RTF = giay xu ly / giay am thanh; tep ngan -> RTF cao vi chi phi co dinh)

| Tang | giay trung vi | RTF trung vi |
|---|---|---|
| k1 | 3.59 | 0.46 |
| xlsr | 0.95 | 0.11 |
| tcn | 0.02 | 0.00 |
| decode | 0.01 | 0.00 |
| total | 4.57 | 0.57 |
| tong_s | 4.92 | 0.61 |
| K1 + XLS-R + TCN + dau ca file | 4.50 | 0.57 |
| K9 phiếu + giải thích | 0.00 | 0.00 |
| K9 ảnh | 0.24 | 0.04 |
| K9 báo cáo HTML | 0.00 | 0.00 |
| K9 PDF (Chrome) | 1.49 | 0.18 |

Fold nguoi noi cua mau: {0: 493, 1: 205, 2: 279, 3: 197, 4: 235} · loi chay (khong co ho so, khong tinh): 0
