# evalp3 `p3v2`

spec: {'den': {'vi': 'rawp:ens55:9,9,21', 'en': 'rawp:ens55:9,9,21'}, 'voi': {'vi': 'rawmx:ens55:9,9,1', 'en': 'rawp:ens73:9,9,21'}, 'nen': {'vi': 'file:experts/P3/p3_v2/heads_vi_mix.pkl', 'en': 'file:experts/P3/p3_v2/heads_b2.pkl'}, 'giong': {'vi': 'file:experts/P3/p3_v2/heads_vi_mix.pkl', 'en': 'file:experts/P3/p3_v2/heads_b2.pkl'}}

| bo | Req1 | Req2 | Req3 | Req4 | Req5 | Req10 | Req11 |
|---|---|---|---|---|---|---|---|
| v6naten nen | 0.620 | 0.552 | 0.985 | 0.038 | 0.979 | 0.457 | 0.694 |
| v6naten MOI | 0.704 | 0.668 | 0.988 | 0.038 | 0.955 | 0.527 | 0.721 |
| v6natvi nen | 0.482 | 0.388 | 0.961 | 0.070 | 0.942 | 0.243 | 0.559 |
| v6natvi MOI | 0.543 | 0.461 | 0.971 | 0.061 | 0.922 | 0.313 | 0.641 |

| bo | kieu | n | dung nen |
|---|---|---|---|
| v6naten | da_ghep_giong | 39 | 0.49 -> 0.49 |
| v6naten | da_ghep_nen | 39 | 0.62 -> 0.62 |
| v6naten | giong_cuc_bo | 388 | 0.86 -> 0.91 |
| v6naten | giong_cuc_bo_ky | 39 | 0.87 -> 0.95 |
| v6naten | giong_toan_cuc | 379 | 0.58 -> 0.93 |
| v6naten | nen_cuc_bo | 322 | 0.22 -> 0.57 |
| v6naten | nen_cuc_bo_ky@01 | 37 | 0.81 -> 0.92 |
| v6naten | nen_toan_cuc | 339 | 0.16 -> 0.37 |
| v6naten | nen_toan_cuc@01 | 37 | 0.92 -> 0.92 |
| v6naten | nguyen_ban | 586 | 0.97 -> 0.97 |
| v6natvi | da_ghep_giong | 99 | 0.32 -> 0.33 |
| v6natvi | da_ghep_nen | 96 | 0.25 -> 0.25 |
| v6natvi | doi_giong_ai | 255 | 0.81 -> 0.80 |
| v6natvi | giong_cuc_bo | 396 | 0.36 -> 0.57 |
| v6natvi | giong_cuc_bo_ky | 100 | 0.31 -> 0.61 |
| v6natvi | giong_toan_cuc | 393 | 0.49 -> 0.66 |
| v6natvi | nen_cuc_bo | 310 | 0.09 -> 0.27 |
| v6natvi | nen_cuc_bo_ky@01 | 100 | 0.78 -> 0.91 |
| v6natvi | nen_toan_cuc | 358 | 0.14 -> 0.30 |
| v6natvi | nen_toan_cuc@01 | 38 | 0.97 -> 0.95 |
| v6natvi | nguyen_ban | 593 | 0.93 -> 0.93 |
| v6natvi | nguyen_ban_vsasv | 193 | 0.99 -> 0.99 |

| bo | thao tac (dung loai) | n | TPR |
|---|---|---|---|
| v6naten | 02_giong_cuc_bo | 388 | 0.85 -> 0.90 |
| v6naten | 02_giong_cuc_bo:da_ghep_giong | 39 | 0.72 -> 0.90 |
| v6naten | 02_giong_cuc_bo:giong_cuc_bo_ky | 39 | 0.87 -> 0.95 |
| v6naten | 02_giong_toan_cuc | 379 | 0.57 -> 0.92 |
| v6naten | 02_nen_cuc_bo | 322 | 0.21 -> 0.56 |
| v6naten | 02_nen_toan_cuc | 339 | 0.14 -> 0.33 |
| v6naten | 02_nen_toan_cuc:da_ghep_nen | 37 | 0.08 -> 0.27 |
| v6natvi | 02_giong_cuc_bo | 396 | 0.36 -> 0.56 |
| v6natvi | 02_giong_cuc_bo:da_ghep_giong | 99 | 0.30 -> 0.54 |
| v6natvi | 02_giong_cuc_bo:giong_cuc_bo_ky | 100 | 0.30 -> 0.59 |
| v6natvi | 02_giong_toan_cuc | 393 | 0.49 -> 0.66 |
| v6natvi | 02_giong_toan_cuc:doi_giong_ai | 255 | 0.81 -> 0.78 |
| v6natvi | 02_nen_cuc_bo | 310 | 0.09 -> 0.25 |
| v6natvi | 02_nen_toan_cuc | 358 | 0.12 -> 0.27 |
| v6natvi | 02_nen_toan_cuc:da_ghep_nen | 83 | 0.11 -> 0.28 |

| bo | kenh | vu oan tep that |
|---|---|---|
| v6naten | 02_giong_cuc_bo | 0.0076 |
| v6naten | 02_giong_toan_cuc/toan | 0.0061 |
| v6naten | 02_nen_cuc_bo | 0.0061 |
| v6naten | 02_nen_toan_cuc/toan | 0.0030 |
| v6natvi | 02_giong_cuc_bo | 0.0087 |
| v6natvi | 02_giong_toan_cuc/toan | 0.0032 |
| v6natvi | 02_nen_cuc_bo | 0.0065 |
| v6natvi | 02_nen_toan_cuc/toan | 0.0022 |

AUC dau: v6naten:giong 0.996 · v6naten:nen 0.920 · v6natvi:giong 0.947 · v6natvi:nen 0.885
