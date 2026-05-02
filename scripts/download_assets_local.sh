#!/usr/bin/env bash
set -euo pipefail

# Download the local assets expected by Lever-LM:
# - MSCOCO 2014 images and captions annotations
# - VQAv2 train/val annotations and questions, plus repo-specific JSON preprocessing
# - OpenFlamingo 3B and 9B checkpoint.pt files
#
# The script is intentionally resumable. Re-running it will continue partial
# downloads and skip extracted/preprocessed files that already exist.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

COCO_PATH="${COCO_PATH:-$ROOT_DIR/downloads/mscoco}"
VQAV2_PATH="${VQAV2_PATH:-$ROOT_DIR/downloads/vqav2}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/openflamingo}"
PYTHON_BIN="${PYTHON_BIN:-/home/fupental/project/miniconda3/envs/leverlm/bin/python}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

download_file() {
  local url="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"
  log "Downloading $(basename "$dst")"
  wget -q -c -O "$dst" "$url"
}

unzip_if_missing() {
  local archive="$1"
  local dest="$2"
  local expected="$3"
  if [[ -e "$expected" ]]; then
    log "Skip unzip $(basename "$archive"); found $expected"
    return 0
  fi
  log "Unzipping $(basename "$archive")"
  unzip -q "$archive" -d "$dest"
}

log "Using COCO_PATH=$COCO_PATH"
log "Using VQAV2_PATH=$VQAV2_PATH"
log "Using CHECKPOINT_PATH=$CHECKPOINT_PATH"
log "Using PYTHON_BIN=$PYTHON_BIN"

mkdir -p "$COCO_PATH/mscoco2014" "$VQAV2_PATH" "$CHECKPOINT_PATH"

download_file \
  "http://images.cocodataset.org/zips/train2014.zip" \
  "$COCO_PATH/mscoco2014/train2014.zip"
download_file \
  "http://images.cocodataset.org/zips/val2014.zip" \
  "$COCO_PATH/mscoco2014/val2014.zip"
download_file \
  "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
  "$COCO_PATH/mscoco2014/annotations_trainval2014.zip"

unzip_if_missing "$COCO_PATH/mscoco2014/train2014.zip" "$COCO_PATH/mscoco2014" "$COCO_PATH/mscoco2014/train2014"
unzip_if_missing "$COCO_PATH/mscoco2014/val2014.zip" "$COCO_PATH/mscoco2014" "$COCO_PATH/mscoco2014/val2014"
unzip_if_missing "$COCO_PATH/mscoco2014/annotations_trainval2014.zip" "$COCO_PATH/mscoco2014" "$COCO_PATH/mscoco2014/annotations"

download_file \
  "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip" \
  "$VQAV2_PATH/v2_Annotations_Train_mscoco.zip"
download_file \
  "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip" \
  "$VQAV2_PATH/v2_Annotations_Val_mscoco.zip"
download_file \
  "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip" \
  "$VQAV2_PATH/v2_Questions_Train_mscoco.zip"
download_file \
  "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip" \
  "$VQAV2_PATH/v2_Questions_Val_mscoco.zip"

unzip_if_missing "$VQAV2_PATH/v2_Annotations_Train_mscoco.zip" "$VQAV2_PATH" "$VQAV2_PATH/v2_mscoco_train2014_annotations.json"
unzip_if_missing "$VQAV2_PATH/v2_Annotations_Val_mscoco.zip" "$VQAV2_PATH" "$VQAV2_PATH/v2_mscoco_val2014_annotations.json"
unzip_if_missing "$VQAV2_PATH/v2_Questions_Train_mscoco.zip" "$VQAV2_PATH" "$VQAV2_PATH/v2_OpenEnded_mscoco_train2014_questions.json"
unzip_if_missing "$VQAV2_PATH/v2_Questions_Val_mscoco.zip" "$VQAV2_PATH" "$VQAV2_PATH/v2_OpenEnded_mscoco_val2014_questions.json"

if [[ ! -f "$VQAV2_PATH/vqav2_hf/vqav2_mscoco_train2014.json" || ! -f "$VQAV2_PATH/vqav2_hf/vqav2_mscoco_val2014.json" ]]; then
  log "Preprocessing VQAv2 into vqav2_hf JSON files"
  "$PYTHON_BIN" lever_lm/dataset_module/preprocess/vqav2_hf.py --root_path "$VQAV2_PATH"
else
  log "Skip VQAv2 preprocessing; vqav2_hf JSON files already exist"
fi

log "Downloading OpenFlamingo checkpoint.pt files"
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import hf_hub_download
import os

base = os.environ["CHECKPOINT_PATH"]
for hf_root in ["OpenFlamingo-3B-vitl-mpt1b", "OpenFlamingo-9B-vitl-mpt7b"]:
    out_dir = os.path.join(base, hf_root)
    os.makedirs(out_dir, exist_ok=True)
    path = hf_hub_download(
        repo_id=f"openflamingo/{hf_root}",
        filename="checkpoint.pt",
        local_dir=out_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"{hf_root}: {path}")
PY

log "Verifying expected files"
test -d "$COCO_PATH/mscoco2014/train2014"
test -d "$COCO_PATH/mscoco2014/val2014"
test -f "$COCO_PATH/mscoco2014/annotations/captions_train2014.json"
test -f "$COCO_PATH/mscoco2014/annotations/captions_val2014.json"
test -f "$VQAV2_PATH/vqav2_hf/vqav2_mscoco_train2014.json"
test -f "$VQAV2_PATH/vqav2_hf/vqav2_mscoco_val2014.json"
test -f "$CHECKPOINT_PATH/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"
test -f "$CHECKPOINT_PATH/OpenFlamingo-9B-vitl-mpt7b/checkpoint.pt"

log "All requested local assets are ready."
