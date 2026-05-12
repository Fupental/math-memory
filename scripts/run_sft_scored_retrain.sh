#!/usr/bin/env bash
set -euo pipefail

cd /root/Lever-LM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

DATA=/root/autodl-tmp/leverlm_math_memory_anchor400_r1/generated_data/math_memory_shot2_cand64_repeat1_beam5_seed42_scored.json
CACHE=/root/autodl-tmp/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings
RUN_DIR=/root/autodl-tmp/leverlm_math_memory_sft_scored_retrain_$(date +%Y%m%d_%H%M%S)
CKPT_DIR=${RUN_DIR}/model_cpk/sft_scored_shot2_cand64_repeat1_beam5_seed42

if [[ ! -f "${DATA}" ]]; then
  echo "Generated file not found: ${DATA}" >&2
  exit 1
fi

mkdir -p "${CKPT_DIR}" "${RUN_DIR}/metrics/best" "${RUN_DIR}/metrics/last"

echo "RUN_DIR=${RUN_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
echo "DATA=${DATA}"

python math_memory_train.py \
  --generated-file "${DATA}" \
  --experience-file data/experiences.json \
  --output-dir "${CKPT_DIR}" \
  --embedding-cache-dir "${CACHE}" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --batch-size 64 \
  --max-epochs 100 \
  --early-stop-patience 5 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/sft_train.log"

python math_memory_eval.py \
  --method lever_lm \
  --checkpoint "${CKPT_DIR}/best.pt" \
  --experience-file data/experiences.json \
  --output-dir "${RUN_DIR}/metrics/best" \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.8 \
  --compute-final-delta \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --embedding-cache-dir "${CACHE}" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  2>&1 | tee "${RUN_DIR}/eval_best.log"

python math_memory_eval.py \
  --method lever_lm \
  --checkpoint "${CKPT_DIR}/last.pt" \
  --experience-file data/experiences.json \
  --output-dir "${RUN_DIR}/metrics/last" \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.8 \
  --compute-final-delta \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --embedding-cache-dir "${CACHE}" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  2>&1 | tee "${RUN_DIR}/eval_last.log"

echo "DONE"
echo "RUN_DIR=${RUN_DIR}"
echo "CKPT_DIR=${CKPT_DIR}"
