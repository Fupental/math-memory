#!/usr/bin/env bash
set -euo pipefail

# Full single-GPU pipeline for the local machine:
# 1. Generate Lever-LM training data on VQAv2 using OpenFlamingo 3B.
# 2. Train the GPT2-based Lever-LM transformer.
# 3. Record data generation duration, train.py process duration, and the
#    epoch-0-to-fit-end transformer training duration emitted by train.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results/pipeline_logs results/timing
RUN_ID="vqa_local_3b_$(date '+%Y%m%d_%H%M%S')"
LOG_PATH="results/pipeline_logs/${RUN_ID}.log"
SUMMARY_PATH="results/timing/${RUN_ID}_summary.env"

exec >> "$LOG_PATH" 2>&1

timestamp() {
  date '+%F %T'
}

write_summary() {
  local key="$1"
  local value="$2"
  printf '%s=%q\n' "$key" "$value" >> "$SUMMARY_PATH"
}

echo "[PIPELINE] run_id=${RUN_ID}"
echo "[PIPELINE] log_path=${LOG_PATH}"
echo "[PIPELINE] summary_path=${SUMMARY_PATH}"
echo "[PIPELINE] started_at=$(timestamp)"
write_summary "RUN_ID" "$RUN_ID"
write_summary "LOG_PATH" "$ROOT_DIR/$LOG_PATH"
write_summary "SUMMARY_PATH" "$ROOT_DIR/$SUMMARY_PATH"
write_summary "PIPELINE_START_AT" "$(timestamp)"

DATA_FILE="vqa-vqav2-flamingo_3B-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:5000.json"
write_summary "DATA_FILE" "$DATA_FILE"

echo "[PIPELINE] Step 1/2 generate_data started_at=$(timestamp)"
gen_start=$(date +%s)
WANDB_MODE=disabled \
HYDRA_FULL_ERROR=1 \
INFER_MODEL=flamingo_3B \
LOAD_FROM_LOCAL=true \
bash scripts/generate_data.sh vqa vqav2_local "[0]"
gen_end=$(date +%s)
gen_seconds=$((gen_end - gen_start))
echo "[PIPELINE] Step 1/2 generate_data finished_at=$(timestamp) seconds=${gen_seconds}"
write_summary "GENERATE_DATA_SECONDS" "$gen_seconds"
write_summary "GENERATE_DATA_FINISHED_AT" "$(timestamp)"

echo "[PIPELINE] Step 2/2 train.py started_at=$(timestamp)"
train_start=$(date +%s)
WANDB_MODE=disabled \
HYDRA_FULL_ERROR=1 \
INFER_MODEL=flamingo_3B \
INFER_MODEL_NAME=OpenFlamingo-3B-vitl-mpt1b \
DATA_FILE="$DATA_FILE" \
TRAIN_STRATEGY=auto \
TRAIN_PRECISION=16 \
bash scripts/train_lever_lm.sh vqa vqav2_local 1 query_img_text_icd_img_text
train_end=$(date +%s)
train_seconds=$((train_end - train_start))
echo "[PIPELINE] Step 2/2 train.py finished_at=$(timestamp) seconds=${train_seconds}"
write_summary "TRAIN_PY_SECONDS_WALL" "$train_seconds"
write_summary "TRAIN_PY_FINISHED_AT" "$(timestamp)"

pipeline_end=$(date +%s)
pipeline_seconds=$((pipeline_end - gen_start))
echo "[PIPELINE] finished_at=$(timestamp) total_seconds=${pipeline_seconds}"
write_summary "PIPELINE_TOTAL_SECONDS" "$pipeline_seconds"
write_summary "PIPELINE_FINISHED_AT" "$(timestamp)"
