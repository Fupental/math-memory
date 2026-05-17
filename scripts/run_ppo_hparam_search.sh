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

RESULT_ROOT=${RESULT_ROOT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_ppo_hparam_search_$(date +%Y%m%d_%H%M%S)}

# Default grid: reproduce the strongest SFT+PPO setting found so far.
# Override any of these from the shell, e.g. BATCH_SIZES="4 8 16".
BATCH_SIZES=${BATCH_SIZES:-"16"}
GROUP_SIZES=${GROUP_SIZES:-"40"}
PPO_EPOCHS_LIST=${PPO_EPOCHS_LIST:-"4"}
MAX_STEPS_LIST=${MAX_STEPS_LIST:-"100"}

# Extra knobs worth sweeping after the core 4-parameter search.
LR_LIST=${LR_LIST:-"5e-6"}
CLIP_EPS_LIST=${CLIP_EPS_LIST:-"0.1"}
REF_KL_COEF_LIST=${REF_KL_COEF_LIST:-"0.05"}
ENTROPY_COEF_LIST=${ENTROPY_COEF_LIST:-"0.001"}
SEEDS=${SEEDS:-"42"}

TRAIN_RATIO=${TRAIN_RATIO:-0.8}
CRITIC_MODE=${CRITIC_MODE:-shared}
TEMPERATURE=${TEMPERATURE:-0.7}
TOP_K=${TOP_K:-32}
PPO_MINIBATCH_SIZE=${PPO_MINIBATCH_SIZE:-64}
VALUE_CLIP_EPS=${VALUE_CLIP_EPS:-0.2}
VALUE_COEF=${VALUE_COEF:-0.5}
TARGET_KL=${TARGET_KL:-0.03}
GRPO_VAL_RATIO=${GRPO_VAL_RATIO:-0.1}
BEST_METRIC=${BEST_METRIC:-train_window_final_delta}
EVAL_EVERY=${EVAL_EVERY:-0}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-3}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA:-0}
SAVE_EVERY=${SAVE_EVERY:-25}
SCORER_BATCH_SIZE=${SCORER_BATCH_SIZE:-16}
SCORER_MAX_LENGTH=${SCORER_MAX_LENGTH:-4096}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-128}

mkdir -p "${RESULT_ROOT}"
SUMMARY_CSV="${RESULT_ROOT}/summary.csv"
echo "run_name,seed,batch_size,group_size,ppo_epochs,max_steps,lr,clip_eps,ref_kl_coef,entropy_coef,best_metric,best_step,best_metric_value,best_val_accuracy,best_val_correct,best_val_total,test_best_accuracy,test_best_correct,test_best_total,test_best_delta,test_best_unique_pair,test_last_accuracy,test_last_correct,test_last_total,test_last_delta,test_last_unique_pair,run_dir" > "${SUMMARY_CSV}"

echo "Result root: ${RESULT_ROOT}"
echo "SFT checkpoint: ${SFT_CKPT}"
echo "Train ratio: ${TRAIN_RATIO}"
echo "Best metric: ${BEST_METRIC}"
echo "Eval every: ${EVAL_EVERY}"

for seed in ${SEEDS}; do
  for batch_size in ${BATCH_SIZES}; do
    for group_size in ${GROUP_SIZES}; do
      for ppo_epochs in ${PPO_EPOCHS_LIST}; do
        for max_steps in ${MAX_STEPS_LIST}; do
          for lr in ${LR_LIST}; do
            for clip_eps in ${CLIP_EPS_LIST}; do
              for ref_kl_coef in ${REF_KL_COEF_LIST}; do
                for entropy_coef in ${ENTROPY_COEF_LIST}; do
                  run_name="seed${seed}_b${batch_size}_g${group_size}_ep${ppo_epochs}_steps${max_steps}_lr${lr}_clip${clip_eps}_kl${ref_kl_coef}_ent${entropy_coef}"
                  run_dir="${RESULT_ROOT}/${run_name}"
                  ckpt_dir="${run_dir}/model_cpk/ppo_from_sft_best"
                  log_file="${run_dir}/ppo_train.log"
                  mkdir -p "${ckpt_dir}"

                  echo "===== RUN ${run_name} ====="
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
                    --group-size "${group_size}" \
                    --temperature "${TEMPERATURE}" \
                    --top-k "${TOP_K}" \
                    --reward-mode delta_logprob \
                    --credit-mode reward_to_go \
                    --lr "${lr}" \
                    --batch-size "${batch_size}" \
                    --max-steps "${max_steps}" \
                    --ppo-epochs "${ppo_epochs}" \
                    --ppo-minibatch-size "${PPO_MINIBATCH_SIZE}" \
                    --clip-eps "${clip_eps}" \
                    --value-clip-eps "${VALUE_CLIP_EPS}" \
                    --value-coef "${VALUE_COEF}" \
                    --entropy-coef "${entropy_coef}" \
                    --ref-kl-coef "${ref_kl_coef}" \
                    --target-kl "${TARGET_KL}" \
                    --grpo-val-ratio "${GRPO_VAL_RATIO}" \
                    --eval-every "${EVAL_EVERY}" \
                    --best-metric "${BEST_METRIC}" \
                    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
                    --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
                    --save-every "${SAVE_EVERY}" \
                    --scorer-model "${SCORER_MODEL}" \
                    --scorer-device cuda \
                    --scorer-dtype bf16 \
                    --scorer-batch-size "${SCORER_BATCH_SIZE}" \
                    --scorer-max-length "${SCORER_MAX_LENGTH}" \
                    2>&1 | tee "${log_file}"

                  for ckpt_name in best last; do
                    if [[ -f "${ckpt_dir}/${ckpt_name}.pt" ]]; then
                      python math_memory_eval.py \
                        --method lever_lm \
                        --checkpoint "${ckpt_dir}/${ckpt_name}.pt" \
                        --experience-file "${EXPERIENCE_FILE}" \
                        --output-dir "${run_dir}/metrics/${ckpt_name}" \
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
                        2>&1 | tee "${run_dir}/eval_${ckpt_name}.log"
                    fi
                  done

                  RUN_NAME="${run_name}" RUN_DIR="${run_dir}" CKPT_DIR="${ckpt_dir}" SUMMARY_CSV="${SUMMARY_CSV}" \
                  SEED_VALUE="${seed}" BATCH_SIZE_VALUE="${batch_size}" GROUP_SIZE_VALUE="${group_size}" \
                  PPO_EPOCHS_VALUE="${ppo_epochs}" MAX_STEPS_VALUE="${max_steps}" LR_VALUE="${lr}" \
                  CLIP_EPS_VALUE="${clip_eps}" REF_KL_COEF_VALUE="${ref_kl_coef}" ENTROPY_COEF_VALUE="${entropy_coef}" \
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

best_eval = {}
eval_path = ckpt_dir / "ppo_eval_history.csv"
if eval_path.exists():
    with eval_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if rows:
        best_eval = max(rows, key=lambda r: float(r.get("val_accuracy") or -1))

best_metric = read_json(run_dir / "metrics" / "best" / "lever_lm_metrics.json")
last_metric = read_json(run_dir / "metrics" / "last" / "lever_lm_metrics.json")
best_metadata = {}
best_ckpt = ckpt_dir / "best.pt"
if best_ckpt.exists():
    payload = torch.load(best_ckpt, map_location="cpu")
    best_metadata = payload.get("metadata", {})

row = {
    "run_name": os.environ["RUN_NAME"],
    "seed": os.environ["SEED_VALUE"],
    "batch_size": os.environ["BATCH_SIZE_VALUE"],
    "group_size": os.environ["GROUP_SIZE_VALUE"],
    "ppo_epochs": os.environ["PPO_EPOCHS_VALUE"],
    "max_steps": os.environ["MAX_STEPS_VALUE"],
    "lr": os.environ["LR_VALUE"],
    "clip_eps": os.environ["CLIP_EPS_VALUE"],
    "ref_kl_coef": os.environ["REF_KL_COEF_VALUE"],
    "entropy_coef": os.environ["ENTROPY_COEF_VALUE"],
    "best_metric": best_metadata.get("best_metric", ""),
    "best_step": best_metadata.get("best_step", best_eval.get("step", "")),
    "best_metric_value": best_metadata.get("best_metric_value", ""),
    "best_val_accuracy": best_eval.get("val_accuracy", ""),
    "best_val_correct": best_eval.get("val_correct", ""),
    "best_val_total": best_eval.get("val_total", ""),
    "test_best_accuracy": best_metric.get("accuracy", ""),
    "test_best_correct": best_metric.get("correct", ""),
    "test_best_total": best_metric.get("total", ""),
    "test_best_delta": best_metric.get("mean_final_delta", ""),
    "test_best_unique_pair": best_metric.get("unique_pair_count", ""),
    "test_last_accuracy": last_metric.get("accuracy", ""),
    "test_last_correct": last_metric.get("correct", ""),
    "test_last_total": last_metric.get("total", ""),
    "test_last_delta": last_metric.get("mean_final_delta", ""),
    "test_last_unique_pair": last_metric.get("unique_pair_count", ""),
    "run_dir": str(run_dir),
}
with summary_csv.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    writer.writerow(row)
print(json.dumps(row, indent=2, ensure_ascii=False))
PY
                done
              done
            done
          done
        done
      done
    done
  done
done

echo "===== SUMMARY ====="
cat "${SUMMARY_CSV}"
