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

SFT_CKPT=${SFT_CKPT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_sft1080_grpo_refkl_20260505_202911/model_cpk/sft_shot2_cand64_repeat1_beam5_seed42/best.pt}
EXPERIENCE_FILE=${EXPERIENCE_FILE:-data/experiences.json}
EMBEDDING_CACHE_DIR=${EMBEDDING_CACHE_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
SCORER_MODEL=${SCORER_MODEL:-Qwen/Qwen3-8B}

RESULT_ROOT=${RESULT_ROOT:-/home/fu_zhihang/projects/LeverLM/data/ppo_selector_compare_split80_$(date +%Y%m%d_%H%M%S)}
SEEDS=${SEEDS:-"42"}

TRAIN_RATIO=${TRAIN_RATIO:-0.8}
CRITIC_MODE=${CRITIC_MODE:-shared}
BATCH_SIZE=${BATCH_SIZE:-16}
GROUP_SIZE=${GROUP_SIZE:-40}
PPO_EPOCHS=${PPO_EPOCHS:-4}
MAX_STEPS=${MAX_STEPS:-100}
PPO_MINIBATCH_SIZE=${PPO_MINIBATCH_SIZE:-64}
LR=${LR:-5e-6}
CLIP_EPS=${CLIP_EPS:-0.1}
VALUE_CLIP_EPS=${VALUE_CLIP_EPS:-0.2}
VALUE_COEF=${VALUE_COEF:-0.5}
ENTROPY_COEF=${ENTROPY_COEF:-0.001}
REF_KL_COEF=${REF_KL_COEF:-0.05}
TARGET_KL=${TARGET_KL:-0}
GRPO_VAL_RATIO=${GRPO_VAL_RATIO:-0.1}
VAL_EVAL_EVERY=${VAL_EVAL_EVERY:-5}
SAVE_EVERY=${SAVE_EVERY:-5}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-0,5,10,15,20,25,30}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_K=${TOP_K:-32}
SCORER_BATCH_SIZE=${SCORER_BATCH_SIZE:-16}
SCORER_MAX_LENGTH=${SCORER_MAX_LENGTH:-4096}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-128}

mkdir -p "${RESULT_ROOT}"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
echo "seed,selector,checkpoint,checkpoint_path,test_accuracy,test_correct,test_total,mean_final_delta,unique_first_memory_count,unique_second_memory_count,unique_pair_count,best_step,best_metric,best_metric_value,run_dir" > "${SUMMARY_CSV}"

echo "Result root: ${RESULT_ROOT}"
echo "SFT checkpoint: ${SFT_CKPT}"

evaluate_checkpoint() {
  local seed="$1"
  local selector="$2"
  local run_dir="$3"
  local ckpt_dir="$4"
  local ckpt_label="$5"
  local ckpt_path="$6"

  if [[ ! -f "${ckpt_path}" ]]; then
    return 0
  fi

  local out_dir="${run_dir}/metrics/${ckpt_label}"
  python math_memory_eval.py \
    --method lever_lm \
    --checkpoint "${ckpt_path}" \
    --experience-file "${EXPERIENCE_FILE}" \
    --output-dir "${out_dir}" \
    --shot-num 2 \
    --seed "${seed}" \
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
    2>&1 | tee "${run_dir}/eval_${ckpt_label}.log"

  SEED_VALUE="${seed}" SELECTOR_VALUE="${selector}" RUN_DIR="${run_dir}" CKPT_DIR="${ckpt_dir}" \
  CKPT_LABEL="${ckpt_label}" CKPT_PATH="${ckpt_path}" SUMMARY_CSV="${SUMMARY_CSV}" \
  python - <<'PY'
import csv
import json
import os
import torch
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
ckpt_path = Path(os.environ["CKPT_PATH"])
metric_path = run_dir / "metrics" / os.environ["CKPT_LABEL"] / "lever_lm_metrics.json"
if not metric_path.exists():
    raise SystemExit(f"missing metrics: {metric_path}")

with metric_path.open(encoding="utf-8") as f:
    metric = json.load(f)

payload = torch.load(ckpt_path, map_location="cpu")
metadata = payload.get("metadata", {})

row = {
    "seed": os.environ["SEED_VALUE"],
    "selector": os.environ["SELECTOR_VALUE"],
    "checkpoint": os.environ["CKPT_LABEL"],
    "checkpoint_path": str(ckpt_path),
    "test_accuracy": metric.get("accuracy", ""),
    "test_correct": metric.get("correct", ""),
    "test_total": metric.get("total", ""),
    "mean_final_delta": metric.get("mean_final_delta", ""),
    "unique_first_memory_count": metric.get("unique_first_memory_count", ""),
    "unique_second_memory_count": metric.get("unique_second_memory_count", ""),
    "unique_pair_count": metric.get("unique_pair_count", ""),
    "best_step": metadata.get("best_step", ""),
    "best_metric": metadata.get("best_metric", ""),
    "best_metric_value": metadata.get("best_metric_value", ""),
    "run_dir": str(run_dir),
}
with Path(os.environ["SUMMARY_CSV"]).open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(json.dumps(row, indent=2, ensure_ascii=False))
PY
}

run_training() {
  local seed="$1"
  local selector="$2"
  local best_metric="$3"
  local eval_every="$4"

  local run_dir="${RESULT_ROOT}/seed${seed}_${selector}"
  local ckpt_dir="${run_dir}/model_cpk/ppo_from_sft_best"
  local log_file="${run_dir}/ppo_train.log"
  mkdir -p "${ckpt_dir}"

  echo "===== TRAIN seed=${seed} selector=${selector} ====="
  python math_memory_ppo_train.py \
    --critic-mode "${CRITIC_MODE}" \
    --init-mode checkpoint \
    --checkpoint "${SFT_CKPT}" \
    --reference-checkpoint "${SFT_CKPT}" \
    --experience-file "${EXPERIENCE_FILE}" \
    --output-dir "${ckpt_dir}" \
    --embedding-cache-dir "${EMBEDDING_CACHE_DIR}" \
    --embedding-model "${EMBEDDING_MODEL}" \
    --embedding-device cuda \
    --embedding-batch-size "${EMBEDDING_BATCH_SIZE}" \
    --train-ratio "${TRAIN_RATIO}" \
    --seed "${seed}" \
    --shot-num 2 \
    --group-size "${GROUP_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top-k "${TOP_K}" \
    --reward-mode delta_logprob \
    --credit-mode reward_to_go \
    --lr "${LR}" \
    --batch-size "${BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --ppo-epochs "${PPO_EPOCHS}" \
    --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}" \
    --clip-eps "${CLIP_EPS}" \
    --value-clip-eps "${VALUE_CLIP_EPS}" \
    --value-coef "${VALUE_COEF}" \
    --entropy-coef "${ENTROPY_COEF}" \
    --ref-kl-coef "${REF_KL_COEF}" \
    --target-kl "${TARGET_KL}" \
    --early-stop-patience 0 \
    --grpo-val-ratio "${GRPO_VAL_RATIO}" \
    --eval-every "${eval_every}" \
    --best-metric "${best_metric}" \
    --save-every "${SAVE_EVERY}" \
    --checkpoint-steps "${CHECKPOINT_STEPS}" \
    --scorer-model "${SCORER_MODEL}" \
    --scorer-device cuda \
    --scorer-dtype bf16 \
    --scorer-batch-size "${SCORER_BATCH_SIZE}" \
    --scorer-max-length "${SCORER_MAX_LENGTH}" \
    2>&1 | tee "${log_file}"

  evaluate_checkpoint "${seed}" "${selector}" "${run_dir}" "${ckpt_dir}" "best" "${ckpt_dir}/best.pt"
  evaluate_checkpoint "${seed}" "${selector}" "${run_dir}" "${ckpt_dir}" "last" "${ckpt_dir}/last.pt"
  evaluate_checkpoint "${seed}" "${selector}" "${run_dir}" "${ckpt_dir}" "init" "${ckpt_dir}/init.pt"

  IFS=',' read -ra steps <<< "${CHECKPOINT_STEPS}"
  for step in "${steps[@]}"; do
    if [[ "${step}" == "0" ]]; then
      continue
    fi
    printf -v padded "%06d" "${step}"
    evaluate_checkpoint "${seed}" "${selector}" "${run_dir}" "${ckpt_dir}" "step_${padded}" "${ckpt_dir}/step_${padded}.pt"
  done
}

for seed in ${SEEDS}; do
  run_training "${seed}" "train_window_final_delta" "train_window_final_delta" "0"
  run_training "${seed}" "val_accuracy" "val_accuracy" "${VAL_EVAL_EVERY}"
done

echo "===== SUMMARY ====="
cat "${SUMMARY_CSV}"
