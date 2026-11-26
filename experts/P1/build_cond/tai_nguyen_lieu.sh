#!/usr/bin/env bash
# Tải nguyên liệu cho bộ kiểm điều kiện K1 của P1 (ồn, vang, nhạc nền) và ghi nguồn gốc.
#
# Mỗi tệp tải: lưu ở data/p1_raw/<bo>/, kiểm md5 nếu nguồn công bố, tính sha256, giải nén,
# rồi append một dòng JSON vào data/p1_raw/NGUON.jsonl:
#   {bo, tep, url, giay_phep, trang_nguon, bytes, md5_nguon, md5_kiem, sha256, ngay_tai, giai_nen_vao}
# Chạy lại an toàn: tệp đã có và đúng kích thước thì bỏ qua (curl -C - để tải tiếp).
#
# Dùng:  bash experts/P1/build_cond/tai_nguyen_lieu.sh [musan] [rirs] [demand]   (không đối số = cả ba)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RAW="$ROOT/data/p1_raw"
LOG="$RAW/NGUON.jsonl"
mkdir -p "$RAW"

# $1 bo · $2 url · $3 ten tep · $4 giay phep · $5 trang nguon · $6 md5 nguon ("" neu khong co) · $7 sha256 nguon (tuy chon)
tai() {
  local bo="$1" url="$2" tep="$3" lic="$4" trang="$5" md5n="$6" shan="${7:-}"
  local dir="$RAW/$bo" out="$RAW/$bo/$tep"
  mkdir -p "$dir"
  if grep -qF "\"tep\": \"$bo/$tep\"" "$LOG" 2>/dev/null; then echo "da co: $bo/$tep"; return; fi
  echo "tai: $url"
  curl -L --fail --retry 5 --retry-delay 10 -C - -o "$out" "$url"
  local md5k="" sha bytes
  if [ -n "$md5n" ]; then
    md5k="$(md5sum "$out" | cut -d' ' -f1)"
    [ "$md5k" = "$md5n" ] || { echo "LOI md5 $tep: $md5k != $md5n" >&2; return 1; }
  fi
  sha="$(sha256sum "$out" | cut -d' ' -f1)"
  [ -z "$shan" ] || [ "$sha" = "$shan" ] || { echo "LOI sha256 $tep: $sha != $shan" >&2; return 1; }
  bytes="$(stat -c %s "$out")"
  case "$tep" in
    *.zip)    unzip -q -o "$out" -d "$dir" ;;
    *.tar.gz) tar -xzf "$out" -C "$dir" ;;
  esac
  printf '{"bo": "%s", "tep": "%s/%s", "url": "%s", "giay_phep": "%s", "trang_nguon": "%s", "bytes": %s, "md5_nguon": "%s", "md5_kiem": "%s", "sha256": "%s", "ngay_tai": "%s", "giai_nen_vao": "data/p1_raw/%s/"}\n' \
    "$bo" "$bo" "$tep" "$url" "$lic" "$trang" "$bytes" "$md5n" "$md5k" "$sha" "$(date -Iseconds)" "$bo" >> "$LOG"
  echo "xong: $bo/$tep ($bytes bytes)"
}

musan() {
  tai musan https://www.openslr.org/resources/17/musan.tar.gz musan.tar.gz "CC BY 4.0" https://www.openslr.org/17/ ""
}

rirs() {
  tai rirs_noises https://www.openslr.org/resources/28/rirs_noises.zip rirs_noises.zip "Apache 2.0" https://www.openslr.org/28/ ""
}

demand() {   # ban 16 kHz; md5 lay tu Zenodo API (records/1227121)
  local base=https://zenodo.org/api/records/1227121/files
  while read -r f m; do
    tai demand "$base/$f/content" "$f" "CC BY 4.0" https://zenodo.org/records/1227121 "$m"
  done <<'LIST'
PRESTO_16k.zip b98d2e6854eeebb397f29a8ad7457092
DWASHING_16k.zip 7e5ee9437ce9409c5f9a779b6212a240
PCAFETER_16k.zip 99927d148128254141a9417d051510bb
OOFFICE_16k.zip 7b61cc2d182d5a654cb9c3101ddd4041
OMEETING_16k.zip 62f7cfe7fe6d30b7d8a215fe37c2dfd2
DKITCHEN_16k.zip 7ffbf52d7f4699f96927846103dc8788
TMETRO_16k.zip 95daf4df678e13b120e14211e6d89571
PSTATION_16k.zip d7448009f6c2aeb6ba570375df1750a3
OHALLWAY_16k.zip fe918bbb0e63e73d09ba7f4843ef33f1
DLIVING_16k.zip 46741384d9e434a0bd8b3ec1830b6052
NRIVER_16k.zip 54264db61d3fe073fb81f2e40e0d19b5
NPARK_16k.zip 80f1385a34d7f1705758926b57f138ce
NFIELD_16k.zip a740046c6f4e174e16f5d568aaec5024
SPSQUARE_16k.zip 205d0e7b8fe74504a2f8d252fc414b9e
TBUS_16k.zip 706b11b0d8504f9f3b3f3211e91b3863
STRAFFIC_16k.zip 2efa87262f272bbf9ba578088e81939c
TCAR_16k.zip 4d930012796bd298932245a26189f973
LIST
}

fleurs() {   # loi noi sach TAM (vi_vn, en_us; dev + test); sha256 = LFS oid tu HF tree API
  local hf=https://huggingface.co/datasets/google/fleurs/resolve/main/data
  local trang=https://huggingface.co/datasets/google/fleurs
  while read -r l s sh; do
    mkdir -p "$RAW/fleurs/$l"
    tai "fleurs/$l" "$hf/$l/audio/$s.tar.gz" "$s.tar.gz" "CC BY 4.0" "$trang" "" "$sh"
    tai "fleurs/$l" "$hf/$l/$s.tsv" "$s.tsv" "CC BY 4.0" "$trang" ""
  done <<'LIST'
vi_vn dev 8821a394c99069409b3ce7bb5cd14b10b709f2ee2fea16f42b4701e1cc9ef673
vi_vn test b359261216ef14b8eea159abce794e270c2a0be903218cf1e1e8f7c7c1565b5c
en_us dev 2658fda72f199e12676ecac9415094667a4e14e149b146e568ea00b2a2f0954c
en_us test d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356
LIST
}

[ $# -eq 0 ] && set -- rirs demand fleurs musan
for b in "$@"; do "$b"; done
echo "NGUON.jsonl: $(wc -l < "$LOG") dong"
