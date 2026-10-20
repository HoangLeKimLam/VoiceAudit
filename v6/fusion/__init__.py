# -*- coding: utf-8 -*-
"""v6.fusion - THI NGHIEM KET HOP dac trung (WP8a muc tep / muc dong; WP8b muc khung):
  E  embedding (vector gop XLS-R, diem OOF fuse1 / ft2)      H  so do tay ca tep (nen, pho, cao do, nhip, chat giong)
  T  xu huong theo thoi gian (buoc nhay nhieu thang, suy giam sau cau, troi nen)
  S  mo ta anh pho tu lam (profile log-mel, GLCM, pho dieu bien)  A  AST pooled (mo hinh pho DA HUAN LUYEN)  Q  DNSMOS
Module: common_fx (nap tep nhe, fold nguoi noi, AUC co xu ly hoa, TPR theo fold, bootstrap cap theo cum, khop mo hinh khoi)
        trend (thong ke buoc / gay doc / GLR)  file_stats (trich so do ca tep, CLI extract)  spec_models (AST / DNSMOS)
        rows_fx (bo kiem chung dong tren ung vien fuse1)  cond_fx (nguong theo dieu kien)  tts_fx (giong may, bo-mot-dong-co)
        report_fx (bang markdown)  ; chay thi nghiem ca tep: python -m v6.boundary.hand_file_exp
Fold: LUON LA v6.boundary.detect.person_fold (crc32(person) % 5), KHONG dung meta.fold."""
