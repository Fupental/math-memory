#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-/home/fu_zhihang/projects/LeverLM/LeverLM}
cd "${ROOT_DIR}"

source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

export HF_HOME=${HF_HOME:-/home/fu_zhihang/projects/LeverLM/data/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

DATA=${DATA:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/generated_data/math_memory_shot2_cand64_repeat1_beam5_seed42_scored.json}
EXPERIENCE_FILE=${EXPERIENCE_FILE:-data/experiences.json}
EMBEDDING_CACHE_DIR=${EMBEDDING_CACHE_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
SCORER_MODEL=${SCORER_MODEL:-Qwen/Qwen3-8B}

RESULT_ROOT=${RESULT_ROOT:-/home/fu_zhihang/projects/LeverLM/data/rce_temperature_search_$(date +%Y%m%d_%H%M%S)}
TEMPERATURES=${TEMPERATURES:-"0.5 1 2 5 100000000"}
REWARD_FIELD=${REWARD_FIELD:-total_delta}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SEED=${SEED:-42}
SHOT_NUM=${SHOT_NUM:-2}

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_EPOCHS=${MAX_EPOCHS:-100}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-128}
SCORER_BATCH_SIZE=${SCORER_BATCH_SIZE:-16}
SCORER_MAX_LENGTH=${SCORER_MAX_LENGTH:-4096}
BEST_METRIC=${BEST_METRIC:-val_accuracy}
ONLINE_EVAL_EVERY=${ONLINE_EVAL_EVERY:-1}
ONLINE_EVAL_LIMIT=${ONLINE_EVAL_LIMIT:-}

mkdir -p "${RESULT_ROOT}"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
echo "temperature,best_epoch,best_val_loss,test_best_accuracy,test_best_correct,test_best_total,test_best_delta,test_best_unique_pair,run_dir,ckpt_dir" > "${SUMMARY_CSV}"

echo "Result root: ${RESULT_ROOT}"
echo "Data: ${DATA}"
echo "Temperatures: ${TEMPERATURES}"
echo "Reward field: ${REWARD_FIELD}"
echo "Best metric: ${BEST_METRIC}"

for temperature in ${TEMPERATURES}; do
  temp_name=$(echo "${temperature}" | tr '.-' 'p_')
  run_name="rce_temp${temp_name}"
  run_dir="${RESULT_ROOT}/${run_name}"
  ckpt_dir="${run_dir}/model_cpk/rce_shot${SHOT_NUM}_cand64_repeat1_beam5_seed${SEED}"
  mkdir -p "${ckpt_dir}" "${run_dir}/metrics/best"

  echo "===== RUN ${run_name} ====="

  cat > "${run_dir}/train_command.txt" <<EOF
python math_memory_train.py \\
  --generated-file "${DATA}" \\
  --experience-file "${EXPERIENCE_FILE}" \\
  --output-dir "${ckpt_dir}" \\
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \\
  --embedding-model "${EMBEDDING_MODEL}" \\
  --embedding-device cuda \\
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \\
  --loss-type rce \\
  --rce-reward-field "${REWARD_FIELD}" \\
  --rce-temperature "${temperature}" \\
  --batch-size "${BATCH_SIZE}" \\
  --max-epochs "${MAX_EPOCHS}" \\
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \\
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \\
  --lr "${LR}" \\
  --weight-decay "${WEIGHT_DECAY}" \\
  --best-metric "${BEST_METRIC}" \\
  --online-eval-every "${ONLINE_EVAL_EVERY}" \\
  ${ONLINE_EVAL_LIMIT:+--online-eval-limit "${ONLINE_EVAL_LIMIT}" \\}
  --scorer-model "${SCORER_MODEL}" \\
  --scorer-device cuda \\
  --scorer-dtype bf16 \\
  --scorer-batch-size "${SCORER_BATCH_SIZE}" \\
  --scorer-max-length "${SCORER_MAX_LENGTH}" \\
  --device cuda
EOF

  online_eval_limit_args=()
  if [[ -n "${ONLINE_EVAL_LIMIT}" ]]; then
    online_eval_limit_args=(--online-eval-limit "${ONLINE_EVAL_LIMIT}")
  fi

  python math_memory_train.py \
    --generated-file "${DATA}" \
    --experience-file "${EXPERIENCE_FILE}" \
    --output-dir "${ckpt_dir}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-device cuda \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    --loss-type rce \
    --rce-reward-field "${REWARD_FIELD}" \
    --rce-temperature "${temperature}" \
    --batch-size "${BATCH_SIZE}" \
    --max-epochs "${MAX_EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --best-metric "${BEST_METRIC}" \
    --online-eval-every "${ONLINE_EVAL_EVERY}" \
    "${online_eval_limit_args[@]}" \
    --scorer-model "${SCORER_MODEL}" \
    --scorer-device cuda \
    --scorer-dtype bf16 \
    --scorer-batch-size "${SCORER_BATCH_SIZE}" \
    --scorer-max-length "${SCORER_MAX_LENGTH}" \
    --device cuda \
    2>&1 | tee "${run_dir}/rce_train.log"

  if [[ -f "${ckpt_dir}/best.pt" ]]; then
    cat > "${run_dir}/eval_best_command.txt" <<EOF
python math_memory_eval.py \\
  --method lever_lm \\
  --checkpoint "${ckpt_dir}/best.pt" \\
  --experience-file "${EXPERIENCE_FILE}" \\
  --output-dir "${run_dir}/metrics/best" \\
  --shot-num "${SHOT_NUM}" \\
  --seed "${SEED}" \\
  --train-ratio "${TRAIN_RATIO}" \\
  --compute-final-delta \\
  --scorer-model "${SCORER_MODEL}" \\
  --scorer-device cuda \\
  --scorer-dtype bf16 \\
  --scorer-batch-size "${SCORER_BATCH_SIZE}" \\
  --scorer-max-length "${SCORER_MAX_LENGTH}" \\
  --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \\
  --embedding-model "${EMBEDDING_MODEL}" \\
  --embedding-device cuda \\
  --embedding-batch-size "${EMBEDDING_BATCH_SIZE}"
EOF

    python math_memory_eval.py \
      --method lever_lm \
      --checkpoint "${ckpt_dir}/best.pt" \
      --experience-file "${EXPERIENCE_FILE}" \
      --output-dir "${run_dir}/metrics/best" \
      --shot-num "${SHOT_NUM}" \
      --seed "${SEED}" \
      --train-ratio "${TRAIN_RATIO}" \
      --compute-final-delta \
      --scorer-model "${SCORER_MODEL}" \
      --scorer-device cuda \
      --scorer-dtype bf16 \
      --scorer-batch-size "${SCORER_BATCH_SIZE}" \
      --scorer-max-length "${SCORER_MAX_LENGTH}" \
      --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
      --embedding-model "${EMBEDDING_MODEL}" \
      --embedding-device cuda \
      --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
      2>&1 | tee "${run_dir}/eval_best.log"
  else
    echo "Skip best.pt: ${ckpt_dir}/best.pt not found"
  fi

  TEMPERATURE_VALUE="${temperature}" RUN_DIR="${run_dir}" CKPT_DIR="${ckpt_dir}" SUMMARY_CSV="${SUMMARY_CSV}" \
  python - <<'PY'
import csv
import json
import os
from pathlib import Path
import torch

run_dir = Path(os.environ["RUN_DIR"])
ckpt_dir = Path(os.environ["CKPT_DIR"])
summary_csv = Path(os.environ["SUMMARY_CSV"])

def read_json(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)

best_metric = read_json(run_dir / "metrics" / "best" / "lever_lm_metrics.json")

best_epoch = ""
best_val_loss = ""
best_ckpt = ckpt_dir / "best.pt"
if best_ckpt.exists():
    payload = torch.load(best_ckpt, map_location="cpu")
    metadata = payload.get("metadata", {})
    best_epoch = metadata.get("best_epoch", "")
    best_val_loss = metadata.get("best_val_loss", "")

row = {
    "temperature": os.environ["TEMPERATURE_VALUE"],
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "test_best_accuracy": best_metric.get("accuracy", ""),
    "test_best_correct": best_metric.get("correct", ""),
    "test_best_total": best_metric.get("total", ""),
    "test_best_delta": best_metric.get("mean_final_delta", ""),
    "test_best_unique_pair": best_metric.get("unique_pair_count", ""),
    "run_dir": str(run_dir),
    "ckpt_dir": str(ckpt_dir),
}
with summary_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(json.dumps(row, indent=2, ensure_ascii=False))
PY
done

echo "===== SUMMARY ====="
cat "${SUMMARY_CSV}"
