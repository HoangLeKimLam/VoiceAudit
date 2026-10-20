#!/usr/bin/env bash
# HANG DOI WP6 (cao du lieu that) - moi buoc la vong lap CU THE, moi lan chay python qua run_serial (khoa may) va < 10 phut.
# Dung: bash .tmp_v6/cache/scrape_queue.sh <buoc>     (buoc: xem 'case' cuoi tep; chay theo thu tu phu thuoc)
# Moi buoc dung khi: ke hoach rong / khong con thong bao 'het ngan sach' / ma thoat 3 (YouTube doi xac minh: KHONG cookie,
# cho vai gio roi chay lai voi RATE=1M) / qua MAXIT vong. Nhat ky: data/scrape/logs/q_<buoc>_<n>.log
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
RS="bash $ROOT/.tmp_v6/wf/run_serial.sh"
PY=.venv/Scripts/python.exe
LOG=data/scrape/logs
MAXIT=${MAXIT:-40}
mkdir -p "$LOG"

# lap mot lenh python co --budget-min cho den khi no khong con bao het ngan sach (hoac het MAXIT vong)
repeat_budget() {
  local tag=$1; shift
  local i=1
  while [ $i -le "$MAXIT" ]; do
    $RS $PY "$@" > "$LOG/q_${tag}_$i.log" 2>&1
    local rc=$?
    tail -3 "$LOG/q_${tag}_$i.log"
    [ $rc -eq 3 ] && { echo "[$tag] ma thoat 3 (bot check / 429): DUNG, cho vai gio, RATE=1M"; return 3; }
    [ $rc -ne 0 ] && { echo "[$tag] loi rc=$rc (xem $LOG/q_${tag}_$i.log)"; return $rc; }
    grep -q "het ngan sach" "$LOG/q_${tag}_$i.log" || { echo "[$tag] xong sau $i vong"; return 0; }
    i=$((i + 1))
  done
  echo "[$tag] dung sau MAXIT=$MAXIT vong"
}

FOSDEM_ROOMS_2024=${FOSDEM_ROOMS_2024:-all}     # 'all' = moi phong (discover fosdem-rooms --year 2024 de xem danh sach)

case "${1:-}" in
  discover)       # ~150 phut mang: tab (lam sau tab da cham tran 60), metadata mau ngau nhien, nguon mo BY/BY-SA/CC0
    for lang in vi en; do
      repeat_budget "tabs_$lang" -m v6.scrape.discover tabs --lang $lang --max-channels 60 --per-tab 200 --prio 3 --budget-min 9
      repeat_budget "meta_$lang" -m v6.scrape.discover meta --lang $lang --per-channel 30 --budget-min 9
    done
    repeat_budget ia -m v6.scrape.discover ia --rows 100 --budget-min 9
    repeat_budget fosdem -m v6.scrape.discover fosdem --year 2024 --room $FOSDEM_ROOMS_2024 --max-per-room 20 --budget-min 9
    repeat_budget ccc -m v6.scrape.discover ccc --conf 37c3 --lang en --max-events 60 --budget-min 9
    $RS $PY -m v6.scrape.gate > "$LOG/q_gate.log" 2>&1; tail -20 "$LOG/q_gate.log" ;;
  search)         # tuy chon: goi y kenh moi (KHONG tu them vao whitelist; nguoi dung duyet)
    $RS $PY -m v6.scrape.discover search --q "buổi học trực tuyến" --q "hội thảo khoa học" --q "tọa đàm" \
        --q "town hall meeting" --q "public lecture" --n 200 --budget-min 9 > "$LOG/q_search.log" 2>&1
    $RS $PY -m v6.scrape.discover agg --top 60 > "$LOG/q_search_agg.log" 2>&1; tail -60 "$LOG/q_search_agg.log" ;;
  fetch)          # ~420 phut mang, ~17 GB tren D: (ton giao xep cuoi; can nguoi dung xac nhan truoc khi --religious mix)
    i=1
    while [ $i -le "$MAXIT" ]; do
      $RS $PY -m v6.scrape.fetch --per-channel 15 --max-hours-channel 12 --max-gb 18 --order prefer --rate ${RATE:-2M} \
          --budget-min 9 > "$LOG/q_fetch_$i.log" 2>&1
      rc=$?
      tail -2 "$LOG/q_fetch_$i.log"
      [ $rc -eq 3 ] && { echo "fetch: bot check -> DUNG; chay lai sau vai gio voi RATE=1M"; exit 3; }
      grep -q "ke hoach tai: 0 tep" "$LOG/q_fetch_$i.log" && { echo "fetch: ke hoach rong -> xong"; break; }
      i=$((i + 1))
    done ;;
  probe_filter)   # ~70 phut CPU, RAM ~1 GB (filter ghi ca frames/<id>.env8.npy)
    $RS $PY -m v6.scrape.probe > "$LOG/q_probe.log" 2>&1; tail -3 "$LOG/q_probe.log"
    repeat_budget filter -m v6.scrape.filter --budget-min 9
    repeat_budget env8 -m v6.scrape.qc env8 --budget-min 9 ;;
  validate)       # ~15 phut CPU: recall / precision bo loc cat tren nhieu du lieu hon (2 cau hinh bo do)
    $RS $PY -m v6.scrape.validate_cuts --n-files 25 --per-file 2 --excerpt-s 300 --edits 8 > "$LOG/q_validate.log" 2>&1
    tail -30 "$LOG/q_validate.log" ;;
  segment)        # ~25 phut CPU
    $RS $PY -m v6.scrape.segment --max-std 40 --max-long 6 > "$LOG/q_segment.log" 2>&1; tail -3 "$LOG/q_segment.log" ;;
  align)          # ~1 000 phut CPU (~90 h doan, ~6 x thoi gian thuc), RAM ~1,5 GB
    for lang in vi en; do repeat_budget "align_$lang" -m v6.scrape.align --lang $lang --budget-min 9; done ;;
  lid)            # ~600 phut CPU (Whisper small dong bang, ~1,5 s / doan, 4 luong), RAM ~1,5 GB
    repeat_budget lid_calib -m v6.scrape.lid --calib --n-nat 200 --threads 4 --budget-min 9
    repeat_budget lid -m v6.scrape.lid --threads 4 --budget-min 9 ;;
  spk)            # ~30 phut CPU (resemblyzer, <= 60 doan / kenh)
    repeat_budget spk -m v6.scrape.spk_cluster --max-per-channel 60 --budget-min 9 ;;
  hosts_qc)       # ~20 phut CPU: hosts I4 + hieu chinh proxy + trung lap (ca voi host tu nhien) + selfcheck + mau nghe
    $RS $PY -m v6.scrape.hosts --max-hours-group 3 > "$LOG/q_hosts.log" 2>&1; cat "$LOG/q_hosts.log"
    $RS $PY -m v6.scrape.hosts --calib-lex > "$LOG/q_lex.log" 2>&1
    $RS $PY -m v6.scrape.qc dedup --stride 4 --against data/natural_v6/hosts/hosts.jsonl \
        --against data/natural_v6/hosts_en/hosts.jsonl --limit-against 6000 > "$LOG/q_dedup.log" 2>&1; tail -5 "$LOG/q_dedup.log"
    $RS $PY -m v6.scrape.qc selfcheck --stride 4 --phases 8 > "$LOG/q_selfcheck.log" 2>&1; grep selfcheck "$LOG/q_selfcheck.log"
    $RS $PY -m v6.scrape.qc listen --frac 0.05 > "$LOG/q_listen.log" 2>&1
    $RS $PY -m v6.scrape.report > "$LOG/q_report.log" 2>&1 ;;
  build_sets)     # ~45 phut CPU, RAM ~1,5 GB: data_labels/v6/scrapevi|scrapeen (CHI giong that + DSP, khong TTS)
    for lang in vi en; do
      $RS $PY -m v6.scrape.build_sets --lang $lang > "$LOG/q_build_sets_$lang.log" 2>&1; tail -10 "$LOG/q_build_sets_$lang.log"
    done ;;
  *) echo "buoc: discover | search | fetch | probe_filter | validate | segment | align | lid | spk | hosts_qc | build_sets"; exit 2 ;;
esac
