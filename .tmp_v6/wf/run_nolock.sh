#!/usr/bin/env bash
# Nhu run_serial.sh nhung KHONG lay khoa: chi cho viec nhe chi dung mang (tim nguon, tai), RAM < 1 GB, khong nap torch.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2} NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2}
# Goc kho tinh tu vi tri tep (.tmp_v6/wf/); thu muc tam / bo dem / ffmpeg giong run_serial.sh (VA_TMP, VA_CACHE, VA_FFMPEG_DIR).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -z "${VA_TMP:-}" ]; then VA_TMP="D:/tmp_va"; [ -d "$VA_TMP" ] || VA_TMP="$ROOT/.tmp_v6/tmp"; fi
mkdir -p "$VA_TMP"
if [ -z "${VA_CACHE:-}" ]; then VA_CACHE="D:/hf-cache"; [ -d "$VA_CACHE" ] || VA_CACHE="$ROOT/.cache"; fi
export PYTHONIOENCODING=utf-8 TMP="$VA_TMP" TEMP="$VA_TMP" TORCH_HOME="$VA_CACHE/torch" TTS_HOME="$VA_CACHE/tts" PIP_CACHE_DIR="$VA_CACHE/pip" UV_CACHE_DIR="$VA_CACHE/uv" XDG_CACHE_HOME="$VA_CACHE/xdg"
FF_DIR=${VA_FFMPEG_DIR:-/c/Users/acer/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin}
[ -d "$FF_DIR" ] && export PATH="$FF_DIR:$PATH"
"$@"
