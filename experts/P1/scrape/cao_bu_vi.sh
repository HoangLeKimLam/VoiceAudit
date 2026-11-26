#!/usr/bin/env bash
# Cào bù vi 21/11 (Huấn chốt): tìm (tiếp) → tải; cắt đoạn chạy SONG SONG với tải (mỗi 5 phút cắt các video đã tải xong),
# rồi cắt lần cuối. segment bỏ qua video đã cắt; fetch chỉ ghi tai_vi.jsonl sau khi tải + ffmpeg xong → không cắt tệp dở.
set -u
cd /d/VoiceAudit
export PATH="/c/Users/phand/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.2-full_build/bin:$PATH"
export HF_HOME="F:/hf_cache" PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
L=data/p1_scrape
$PY experts/P1/scrape/cao_tunhien.py discover --lang vi --per-query 100 >> ${L}_discover_vi_2.log 2>&1 || echo "discover loi $?"
$PY experts/P1/scrape/tai_cat.py fetch --lang vi > ${L}_fetch_vi_2.log 2>&1 &
FETCH=$!
while kill -0 "$FETCH" 2>/dev/null; do
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY experts/P1/scrape/tai_cat.py segment --lang vi >> ${L}_segment_vi_2.log 2>&1
  for _ in $(seq 60); do kill -0 "$FETCH" 2>/dev/null || break; sleep 5; done
done
wait "$FETCH"; echo "fetch exit $?"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY experts/P1/scrape/tai_cat.py segment --lang vi >> ${L}_segment_vi_2.log 2>&1
echo "segment exit $?"
