#!/usr/bin/env bash
# Cai cac moi truong python RIENG (uv) cho tung may sinh giong khong song chung duoc voi .venv chinh.
# Dung:  bash v6/tts_multi/setup_envs.sh <env ...>|all|list|status
#        bash v6/tts_multi/setup_envs.sh replay <env ...>|all   # tai lap CHINH XAC tu ban freeze da ghi
#   env: f5 zipvoice vieneu kokoro seedvc dfn styletts2 resenh | models (tai trong so lon bang curl)
# Moi env: D:/va_gen/envs/<env> (duong ASCII), Python 3.11.14 (uv tai ve D:/va_gen/uv_python),
#   torch 2.8.0+cu128 (chay voi driver 576.93 / CUDA 12.9) -> cung mot ban wheel, uv hardlink tu cache D:.
# Ghi lai: D:/va_gen/envs/<env>/FREEZE.txt (uv pip freeze) + ban sao D:/va_gen/freeze/<env>.txt (con lai ke ca khi
#   xoa env) + dong tom tat vao D:/va_gen/logs/installed.tsv (ngay, env, trang thai, torch, python).
# TAI LAP: CHI 'replay' (uv pip sync tu freeze) cho dung TUNG ban. Lan cai dau chi ghim ban chinh (torch, f5-tts,
#   kokoro, vieneu, deepfilternet, styletts2, resenh, commit git); nhieu goi phu (zipvoice deps, transformers cua
#   vieneu, numpy/soundfile cua f5 ...) do uv giai luc cai -> co the khac neu cai lai vao ngay khac.
#   (Moi 'replay' chi moi thu --dry-run cho dfn/zipvoice: "Would make no changes".)
# Banh xe local D:/va_gen/dl/wheels: 'wheels' tai tu files.pythonhosted.org neu thieu + kiem sha256 (chay tu dong
#   truoc moi lan cai/replay). Ma nguon git: clone() LOI neu khong o dung commit ghim.
# Dung them:  bash v6/tts_multi/setup_envs.sh wheels
#             bash v6/tts_multi/setup_envs.sh selftest   # kiem clone() tren repo git tam (dung / sai commit, sua cuc bo)
# Chi cai / tai ve (khong chay python nang) -> duoc chay NGOAI khoa run_serial.sh; moi bo nho dem tren D:.
# Ngoai le: rut EMA cua zalopay F5 (python, can khoa) — env_models chi nhac lenh, khong tu chay.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # goc kho, tinh tu vi tri tep nay (v6/tts_multi/)
# Thu muc rieng cua may (duong ASCII): doi bang VA_GEN / VA_CACHE / VA_TMP; mac dinh = may goc.
VA=${VA_GEN:-D:/va_gen}
VC=${VA_CACHE:-D:/hf-cache}
VT=${VA_TMP:-D:/tmp_va}
SRC_DIR=${SRC_DIR:-$VA/src}     # noi clone ma nguon git (selftest doi sang thu muc tam)
export UV_CACHE_DIR=$VC/uv PIP_CACHE_DIR=$VC/pip UV_PYTHON_INSTALL_DIR=$VA/uv_python
export HF_HOME=$VC TORCH_HOME=$VC/torch TMP=$VT TEMP=$VT UV_LINK_MODE=hardlink
export UV_HTTP_TIMEOUT=600
mkdir -p "$VA/envs" "$VA/logs" "$VA/src" "$VT"
PYVER=3.11.14
FREEZE_DIR=$VA/freeze
PIPER_LINKS=https://k2-fsa.github.io/icefall/piper_phonemize.html
TORCH_IDX=https://download.pytorch.org/whl/cu128
TORCH_PINS="torch==2.8.0+cu128 torchaudio==2.8.0+cu128"
WHEELS=$VA/dl/wheels
# Phien ban ma nguon git (khoa commit de tai lap)
VIZIP_COMMIT=7c09abc215c775ea0c6377bd018e90aa45bddcc8
SEEDVC_COMMIT=51383efd921027683c89e5348211d93ff12ac2a8

log() { echo "[$(date +%H:%M:%S)] $*"; }

py_of() { echo "$VA/envs/$1/Scripts/python.exe"; }

mkenv() {  # mkenv <env>  : tao venv + torch cu128
  local e=$1 py
  py=$(py_of "$e")
  [ -x "$py" ] || uv venv -q --python $PYVER "$VA/envs/$e" || return 1
  printf "torch==2.8.0+cu128\ntorchaudio==2.8.0+cu128\ntorchvision==0.23.0+cu128\n" > "$VA/envs/$e/constraints.txt"
  uv pip install -q --python "$py" --index-url $TORCH_IDX $TORCH_PINS || return 1
}

pipi() {  # pipi <env> <args...> : cai tu PyPI, giu torch cu128 (rang buoc)
  local e=$1; shift
  uv pip install -q --python "$(py_of "$e")" -c "$VA/envs/$e/constraints.txt" \
    --extra-index-url $TORCH_IDX --index-strategy unsafe-best-match "$@"
}

clone() {  # clone <url> <dir> <commit> : LOI neu khong dung duoc dung commit ghim (khong im lang de ban khac)
  local d=$SRC_DIR/$2 head
  [ -d "$d/.git" ] || git clone -q "$1" "$d" || return 1
  if [ "$(git -C "$d" rev-parse HEAD)" != "$3" ]; then
    git -C "$d" cat-file -e "$3^{commit}" 2>/dev/null || git -C "$d" fetch -q origin "$3" || { log "LOI: khong tai duoc commit $3 cua $2"; return 1; }
    git -C "$d" checkout -q "$3" || { log "LOI: checkout $3 that bai trong $d (co sua doi cuc bo?)"; return 1; }
  fi
  head=$(git -C "$d" rev-parse HEAD)
  [ "$head" = "$3" ] || { log "LOI: $2 o $head, khac commit ghim $3"; return 1; }
  echo "$2 $head"
}

# Banh xe dung boi cac env (giong het ban tren PyPI — sha256 da doi chieu voi pypi.org/pypi/<goi>/<ban>/json 11/10)
WHEEL_SPECS="
deepfilternet-0.5.6-py3-none-any.whl d/deepfilternet 99f5688d954fcfa8f853bf8bb8c3b2a59e4f9dc5d95643c9e6a32053234ba7c6
f5_tts-1.1.22-py3-none-any.whl f/f5-tts f0505dfb5463645caa526bace346ed1c89bcc9acb9ef42fdffd56c2c4c0a09d1
kokoro-0.9.4-py3-none-any.whl k/kokoro a129dc6364a286bd6a92c396e9862459d3d3e45f2c15596ed5a94dcee5789efd
misaki-0.9.4-py3-none-any.whl m/misaki 90e2eeb169786c014c429e5058d2ea6bcd02d651f2a24450ba6c9ffc0f8da15a
resemble_enhance-0.0.1-py3-none-any.whl r/resemble-enhance 9aa565cba75b93e2f82603d72c19aadaf4fc5237ca0babca7dd0853b5bf3432a
vieneu-3.8.3-py3-none-any.whl v/vieneu 7388d166e65746f5bb075bf8094d324f8131a82393680b712260d6f2f983ef06
"

wheels() {  # tai (neu thieu) + kiem sha256 moi banh xe local tu files.pythonhosted.org
  mkdir -p "$WHEELS"
  local f p h got
  while read -r f p h; do
    [ -n "$f" ] || continue
    if [ ! -s "$WHEELS/$f" ]; then
      curl -fsSL --retry 3 -o "$WHEELS/$f.part" "https://files.pythonhosted.org/packages/py3/$p/$f" && mv "$WHEELS/$f.part" "$WHEELS/$f" || { log "LOI tai $f"; return 1; }
    fi
    got=$(sha256sum "$WHEELS/$f" | cut -d' ' -f1)
    [ "$got" = "$h" ] || { log "LOI sha256 $f: $got != $h"; return 1; }
  done <<< "$WHEEL_SPECS"
  log "banh xe ok (sha256 khop): $WHEELS"
}

selftest() {  # kiem clone() khong can mang: repo tam 2 commit -> dung commit / commit khong co / sua cuc bo chan
  local t up c1 c2 fails=0 g="git -c user.name=va -c user.email=va@local -c init.defaultBranch=main"
  t=$(mktemp -d "$VT/setup_selftest_XXXXXX") || return 1
  up=$t/up
  $g init -q "$up" && echo 1 > "$up/a.txt" && $g -C "$up" add a.txt && $g -C "$up" commit -q -m c1 || return 1
  c1=$(git -C "$up" rev-parse HEAD)
  echo 2 > "$up/a.txt" && $g -C "$up" commit -q -am c2 || return 1
  c2=$(git -C "$up" rev-parse HEAD)
  SRC_DIR=$t/src
  chk() {  # chk <ten> <ma mong doi: 0|loi> <lenh...>
    local name=$1 want=$2 rc; shift 2
    "$@" > "$t/out.txt" 2>&1; rc=$?
    if { [ "$want" = 0 ] && [ $rc -eq 0 ]; } || { [ "$want" = loi ] && [ $rc -ne 0 ]; }; then echo "PASS $name"
    else echo "FAIL $name (rc=$rc): $(tr '\n' ' ' < "$t/out.txt")"; fails=$((fails + 1)); fi
  }
  chk "clone moi ve commit cu c1" 0 clone "$up" r "$c1"
  [ "$(git -C "$t/src/r" rev-parse HEAD)" = "$c1" ] || { echo "FAIL HEAD khac c1"; fails=$((fails + 1)); }
  chk "doi sang c2 trong repo co san" 0 clone "$up" r "$c2"
  chk "commit khong ton tai -> loi" loi clone "$up" r 0123456789abcdef0123456789abcdef01234567
  [ "$(git -C "$t/src/r" rev-parse HEAD)" = "$c2" ] || { echo "FAIL HEAD bi doi sau lan loi"; fails=$((fails + 1)); }
  echo x > "$t/src/r/a.txt"
  chk "sua cuc bo chan checkout -> loi" loi clone "$up" r "$c1"
  SRC_DIR=$VA/src
  rm -rf "$t"
  [ $fails -eq 0 ] && echo "selftest OK" || echo "selftest LOI: $fails"
  return $fails
}

# ---------------- tung env ----------------
env_f5() {        # F5-TTS 1.1.22: zalopay/vietnamese-tts (vi, F5TTS_Base) + F5TTS_v1_Base (en)
  mkenv f5 || return 1
  pipi f5 "$WHEELS/f5_tts-1.1.22-py3-none-any.whl" "torchcodec==0.7.0" soundfile numpy || return 1
}

env_zipvoice() {  # ViZipVoice (vi) + ZipVoice-Distill (en): cung ma nguon iamdinhthuan/ViZipvoice (co ca infer_zipvoice)
  clone https://github.com/iamdinhthuan/ViZipvoice ViZipvoice $VIZIP_COMMIT || return 1
  mkenv zipvoice || return 1
  pipi zipvoice --find-links https://k2-fsa.github.io/icefall/piper_phonemize.html \
    piper_phonemize lhotse huggingface_hub safetensors vocos pydub soundfile numpy cn2an inflect \
    "soe-vinorm>=0.3.2" jieba pypinyin "setuptools<81" "typing_extensions>=4.15.0" tensorboard || return 1
}

env_vieneu() {    # VieNeu-TTS v3 Turbo (vi+en, Apache-2.0) - KHONG cai extra [watermark]
  mkenv vieneu || return 1
  pipi vieneu "$WHEELS/vieneu-3.8.3-py3-none-any.whl" transformers safetensors soundfile || return 1
}

env_kokoro() {    # Kokoro-82M (en, giong co san) - espeak-ng qua espeakng-loader (khong can MSI)
  mkenv kokoro || return 1
  pipi kokoro "$WHEELS/kokoro-0.9.4-py3-none-any.whl" "$WHEELS/misaki-0.9.4-py3-none-any.whl" \
    "misaki[en]==0.9.4" soundfile \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" || return 1
}

env_seedvc() {    # Seed-VC (GPL-3.0, repo da luu tru) - bo 3 dong torch nightly + doi torch 2.4 -> 2.8 cu128
  clone https://github.com/Plachtaa/seed-vc seed-vc $SEEDVC_COMMIT || return 1
  mkenv seedvc || return 1
  grep -v -E "^(torch|torchvision|torchaudio)( |=|$)|FreeSimpleGUI|sounddevice|^gradio" "$VA/src/seed-vc/requirements.txt" \
    > "$VA/envs/seedvc/req_local.txt"
  pipi seedvc -r "$VA/envs/seedvc/req_local.txt" "torchvision==0.23.0+cu128" || return 1
}

env_dfn() {       # DeepFilterNet 0.5.6 (khu on AI, kind denoise_ai) - numpy<2
  mkenv dfn || return 1
  pipi dfn "$WHEELS/deepfilternet-0.5.6-py3-none-any.whl" "numpy<2" "soundfile<0.13" "soxr==0.5.0.post1" || return 1
}

env_resenh() {    # Resemble-Enhance 0.0.1 (MIT; khu on + lam dep AI, kind denoise_ai, held-out) - cai --no-deps vi
                  # METADATA ghim deepspeed 0.12.4 / torch 2.1.1 (deepspeed chi dung khi HOC; adapter gia lap module)
  mkenv resenh || return 1
  uv pip install -q --python "$(py_of resenh)" --no-deps "$WHEELS/resemble_enhance-0.0.1-py3-none-any.whl" || return 1
  pipi resenh "torchvision==0.23.0+cu128" "librosa==0.10.2.post1" "matplotlib==3.9.2" "numpy==1.26.4" \
    "omegaconf==2.3.0" "pandas==2.2.3" "ptflops==0.7.4" "rich==13.9.4" "scipy==1.14.1" "soundfile==0.12.1" \
    "tqdm==4.67.1" "resampy==0.4.3" "tabulate==0.9.0" "celluloid==0.2.0" || return 1
  local f=enhancer_stage2
  dl https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/$f/hparams.yaml $VA/models/resemble_enhance/$f/hparams.yaml || return 1
  dl https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/$f/ds/G/default/mp_rank_00_model_states.pt \
    $VA/models/resemble_enhance/$f/ds/G/default/mp_rank_00_model_states.pt || return 1
}

env_styletts2() { # StyleTTS2 LibriTTS (en, held-out) qua goi pip styletts2 (am vi bang gruut, khong can espeak)
  mkenv styletts2 || return 1
  pipi styletts2 "styletts2==0.1.6" "numpy<2" || return 1
}

# ---------------- trong so tai truoc bang curl (khong can python; tiep tuc duoc khi dut: -C -) ----------------
MD=$VA/models
COQ=$VC/tts/tts         # = get_user_data_dir("tts") khi TTS_HOME=D:/hf-cache/tts (run_serial.sh)
HFU=https://huggingface.co
dl() {  # dl <url> <file>
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] && [ ! -f "$2.part" ] && return 0
  touch "$2.part"
  curl -sSL --retry 5 --retry-delay 5 -C - -o "$2" "$1" && rm -f "$2.part"
}
env_models() {
  local f
  for f in model.pth config.json vocab.json hash.md5 speakers_xtts.pth; do
    dl $HFU/coqui/XTTS-v2/resolve/main/$f $COQ/tts_models--multilingual--multi-dataset--xtts_v2/$f || return 1; done
  dl https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/WavLM-Large.pt $COQ/wavlm/WavLM-Large.pt || return 1
  dl $HFU/charactr/vocos-mel-24khz/resolve/main/config.yaml $MD/vocos-mel-24khz/config.yaml || return 1
  dl $HFU/charactr/vocos-mel-24khz/resolve/main/pytorch_model.bin $MD/vocos-mel-24khz/pytorch_model.bin || return 1
  for f in checkpoint-1860000.pt tokens.txt model.json; do
    dl $HFU/contextboxai/ViZipvoice/resolve/main/$f $MD/vizipvoice/$f || return 1; done
  for f in model.pt model.json tokens.txt; do
    dl $HFU/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/$f $MD/zipvoice_distill/$f || return 1; done
  for f in model_1250000.safetensors vocab.txt; do
    dl $HFU/SWivid/F5-TTS/resolve/main/F5TTS_v1_Base/$f $MD/f5_v1_base/$f || return 1; done
  # zalopay/vietnamese-tts (CC-BY-4.0): 5,4 GB (co trang thai toi uu) -> engines/f5.py --convert rut EMA ra safetensors
  for f in vocab.txt model_1290000.pt; do
    dl $HFU/zalopay/vietnamese-tts/resolve/main/$f $MD/f5_vi_zalopay/$f || return 1; done
  [ -s $MD/f5_vi_zalopay/model_1290000_ema.safetensors ] || log "CON THIEU: rut EMA zalopay (python, qua khoa): bash" \
    "$REPO/.tmp_v6/wf/run_serial.sh $(py_of f5) -m v6.tts_multi.engines.f5 --convert" \
    "$MD/f5_vi_zalopay/model_1290000.pt $MD/f5_vi_zalopay/model_1290000_ema.safetensors"
}

record() {  # record <env> <status>
  local e=$1 st=$2 py tv pv
  py=$(py_of "$e")
  if [ -x "$py" ]; then
    uv pip freeze --python "$py" > "$VA/envs/$e/FREEZE.txt" 2>/dev/null
    mkdir -p "$FREEZE_DIR" && cp -f "$VA/envs/$e/FREEZE.txt" "$FREEZE_DIR/$e.txt"
  fi
  tv=$(grep -E "^torch==" "$VA/envs/$e/FREEZE.txt" 2>/dev/null | head -1)
  pv=$(grep -E "^version_info" "$VA/envs/$e/pyvenv.cfg" 2>/dev/null | sed 's/.*= *//')
  printf "%s\t%s\t%s\t%s\t%s\n" "$(date +%F_%T)" "$e" "$st" "${tv:-none}" "${pv:-none}" >> "$VA/logs/installed.tsv"
}

replay_env() {  # replay_env <env> : venv Python $PYVER + uv pip sync dung ban freeze (ma nguon git o commit ghim)
  local e=$1 fz=$FREEZE_DIR/$1.txt py
  [ -s "$fz" ] || fz=$VA/envs/$e/FREEZE.txt
  [ -s "$fz" ] || { echo "khong co freeze cho $e (chay cai thuong truoc)"; return 1; }
  case $e in
    zipvoice) clone https://github.com/iamdinhthuan/ViZipvoice ViZipvoice $VIZIP_COMMIT || return 1 ;;
    seedvc) clone https://github.com/Plachtaa/seed-vc seed-vc $SEEDVC_COMMIT || return 1 ;;
  esac
  py=$(py_of "$e")
  [ -x "$py" ] || uv venv -q --python $PYVER "$VA/envs/$e" || return 1
  uv pip sync -q --python "$py" --extra-index-url $TORCH_IDX --index-strategy unsafe-best-match \
    --find-links $PIPER_LINKS "$fz" || return 1
}

status() {  # tung env: co python khong, ban torch, dong installed.tsv cuoi
  local e py
  for e in $ALL; do
    py=$(py_of "$e")
    printf "%-10s python=%s %s | %s\n" "$e" "$([ -x "$py" ] && echo yes || echo NO)" \
      "$(grep -E '^torch==' "$VA/envs/$e/FREEZE.txt" 2>/dev/null | head -1)" \
      "$(awk -F'\t' -v e="$e" '$2 == e' "$VA/logs/installed.tsv" 2>/dev/null | tail -1 | tr '\t' ' ')"
  done
}

ALL="f5 zipvoice vieneu kokoro seedvc dfn styletts2 resenh"
[ $# -eq 0 ] && { echo "dung: $0 <env...>|all|list|status|wheels|selftest | replay <env...>|all"; exit 1; }
[ "$1" = list ] && { echo "$ALL models"; exit 0; }
[ "$1" = status ] && { status; exit 0; }
[ "$1" = selftest ] && { selftest; exit $?; }
mode=install
if [ "$1" = replay ]; then mode=replay; shift; [ $# -eq 0 ] && { echo "replay can <env...>|all"; exit 1; }; fi
[ "$1" = wheels ] && { wheels; exit $?; }
[ "$1" = all ] && set -- $ALL
rc=0
case " $* " in *" models "*) [ $# -eq 1 ] || wheels || exit 1 ;; *) wheels || exit 1 ;; esac
for e in "$@"; do
  if [ $mode = replay ]; then
    [ "$e" = models ] && { log "models khong co freeze: dung '$0 models'"; continue; }
    log "== tai lap env $e tu freeze"
    if replay_env "$e" > "$VA/logs/replay_$e.log" 2>&1; then record "$e" replay-ok; log "   ok ($e)"
    else record "$e" replay-FAIL; log "   LOI ($e) - xem $VA/logs/replay_$e.log"; rc=1; fi
    continue
  fi
  log "== cai env $e"
  if "env_$e" > "$VA/logs/setup_$e.log" 2>&1; then record "$e" ok; log "   ok ($e)"
  else record "$e" FAIL; log "   LOI ($e) - xem $VA/logs/setup_$e.log"; rc=1; fi
done
exit $rc
