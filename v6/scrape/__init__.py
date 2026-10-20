# -*- coding: utf-8 -*-
"""v6.scrape - CAO DU LIEU THAT (YouTube CC tung video + archive.org / media.ccc.de / FOSDEM), chi noi bo nghien cuu.

Chuoi buoc (moi buoc mot module, ghi tiep jsonl trong data/scrape/meta):
  discover -> gate -> fetch -> probe -> filter -> segment -> align -> lid -> [spk_cluster] -> hosts (giao dien I4) -> build_sets
  gate.current(): moi buoc sau (fetch / segment / hosts) TINH LAI cong trong bo nho, khong tin gated.jsonl cu.
  lid: nhan dang ngon ngu Whisper dong bang (B-12 'LID >= 0,9'); spk_cluster: hop kenh co chung nguoi noi (nhom chong ro ri).
  qc: trung lap (van tay duong bao nang luong, 8 pha env8) + selfcheck + mau nghe tay 5 % (coverage B-12); report: bao cao.
  validate_cuts: cay vet cat da biet vao am thanh sach de do recall/precision cua bo loc cat (filter).
  quarantine: CHUYEN (khong xoa) tep cua kenh bi loai sau khi tai vao data/scrape/_quarantine.
Khong nhai giong nguoi trong am thanh cao (quyet dinh 2). Moi lenh python chay qua .tmp_v6/wf/run_serial.sh."""
