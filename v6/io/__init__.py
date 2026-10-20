# -*- coding: utf-8 -*-
"""v6.io — bo chay THU NGHIEM input -> output day du theo dac ta VoiceAudit.html (#io, #ev, #bind, #bounds).

Chi de nghien cuu/kiem thu. Cac mo-dun:
  common      hang so dung chung (ten thao tac, muc do, bac, dinh dang so)
  train_final hoc MO HINH CUOI tren 630 tep v4 (+ bo tang cuong) -> models/v6 (NANG, lead chay)
  extract     K2 mot tep trong bo nho (giong run_codec_variants.features)
  decide      K3..K8 bang mo hinh cuoi; tep > 20 s theo v6.chunkfile (du phong v5b.decide_any)
  explain     giai thich tieng Viet + ly do 04
  render      anh bang chung (dang song + pho mel, to dung doan) + anh tong quan
  report      bao cao HTML + PDF (Chrome headless)
  schema      luoc do JSON 'voiceaudit.v6.ho_so/1' + luat #bind
  stub        mo hinh GIA LAP + tin hieu tong hop (chi de tu kiem ma)
  conformance chay CLI tren danh sach tep va kiem C1..Cn
Chay tu thu muc goc repo:  python -m v6.io.<ten> ...  hoac  python voiceaudit_v6.py ..."""
