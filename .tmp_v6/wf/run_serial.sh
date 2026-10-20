#!/usr/bin/env bash
# Chay MOT lenh nang tai mot thoi diem tren may (khoa thu muc). Dung: bash .tmp_v6/wf/run_serial.sh <lenh ...>
# Xep hang theo thu tu den (ve trong lock_q/): ai cho truoc duoc khoa truoc, de hai lan hang doi khong dan nhau.
# Cho khoa toi da WAIT_MIN phut (mac dinh 2880 = 48 h). Ghi ai dang giu khoa vao lock.d/owner.
# 13/10: mac dinh 6 luong / viec (may 10 nhan 16 luong; truoc la 2).
# Thu muc goc cua kho tinh tu vi tri tep nay (.tmp_v6/wf/) — dat kho o dau cung duoc.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
L="$ROOT/.tmp_v6/wf/lock.d"
Q="$ROOT/.tmp_v6/wf/lock_q"
WAIT_MIN=${WAIT_MIN:-2880}
mkdir -p "$Q"
T="$Q/$(date +%s%N)_$$"
: > "$T"
trap 'rm -f "$T"' EXIT
alive() { kill -0 "$1" 2>/dev/null || tasklist //FI "PID eq $1" 2>/dev/null | grep -q "$1"; }
n=0
while true; do
  # bo ve cua tien trinh da chet
  for f in "$Q"/*; do
    [ -e "$f" ] || continue
    p=${f##*_}
    alive "$p" || rm -f "$f"
  done
  first=$(ls "$Q" 2>/dev/null | sort | head -1)
  if [ "$Q/$first" = "$T" ] && mkdir "$L" 2>/dev/null; then break; fi
  if [ -f "$L/owner" ]; then
    pid=$(cut -d' ' -f1 "$L/owner" 2>/dev/null)
    if [ -n "$pid" ] && ! alive "$pid"; then rm -rf "$L"; continue; fi
  fi
  n=$((n + 1)); [ $n -gt $((WAIT_MIN * 6)) ] && { echo "run_serial: het thoi gian cho khoa" >&2; exit 98; }
  sleep 10
done
rm -f "$T"
echo "$$ $(date +%H:%M:%S) $*" > "$L/owner"
trap 'rm -rf "$L"' EXIT
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-6}
# Thu muc tam / bo dem: dat VA_TMP, VA_CACHE de doi; mac dinh dung thu muc cua may goc neu co, khong thi trong kho.
if [ -z "${VA_TMP:-}" ]; then VA_TMP="D:/tmp_va"; [ -d "$VA_TMP" ] || VA_TMP="$ROOT/.tmp_v6/tmp"; fi
mkdir -p "$VA_TMP"
if [ -z "${VA_CACHE:-}" ]; then VA_CACHE="D:/hf-cache"; [ -d "$VA_CACHE" ] || VA_CACHE="$ROOT/.cache"; fi
export PYTHONIOENCODING=utf-8 TMP="$VA_TMP" TEMP="$VA_TMP" TORCH_HOME="$VA_CACHE/torch" TTS_HOME="$VA_CACHE/tts" PIP_CACHE_DIR="$VA_CACHE/pip" UV_CACHE_DIR="$VA_CACHE/uv" XDG_CACHE_HOME="$VA_CACHE/xdg"
# ffmpeg cua may goc (neu co); may khac: cai ffmpeg vao PATH hoac dat VA_FFMPEG_DIR
FF_DIR=${VA_FFMPEG_DIR:-/c/Users/acer/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin}
[ -d "$FF_DIR" ] && export PATH="$FF_DIR:$PATH"
"$@"
