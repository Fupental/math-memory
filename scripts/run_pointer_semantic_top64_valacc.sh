#!/usr/bin/env bash
set -euo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

export HF_HOME=${HF_HOME:-/home/fu_zhihang/projects/LeverLM/data/hf_cache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

DATA=${DATA:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_semantic_top64_beam5_seed42/generated_data/math_memory_semantic_top64_shot2_beam5_seed42_scored.json}
CACHE=${CACHE:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
RUN_DIR=${RUN_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_pointer_rce_semantic_top64_valacc_$(date +%Y%m%d_%H%M%S)}
CKPT_DIR=${CKPT_DIR:-${RUN_DIR}/model_cpk/pointer_rce_semantic_top64_valacc}
OUT_DIR=${OUT_DIR:-${RUN_DIR}/metrics/best}

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$OUT_DIR"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "CKPT_DIR=$CKPT_DIR"
  echo "OUT_DIR=$OUT_DIR"
  echo "DATA=$DATA"
  echo "HOSTNAME=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi || true
} | tee "$RUN_DIR/run_info.log"

python math_memory_pointer_train.py \
  --generated-file "$DATA" \
  --experience-file data/experiences.json \
  --output-dir "$CKPT_DIR" \
  --embedding-cache-dir "$CACHE" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --candidate-mode generated \
  --eval-candidate-mode semantic \
  --candidate-num 64 \
  --rce-reward-field total_delta \
  --rce-temperature 0.1 \
  --batch-size 16 \
  --max-epochs 100 \
  --early-stop-patience 5 \
  --lr 1e-4 \
  --weight-decay 1e-3 \
  --best-metric val_accuracy \
  --online-eval-every 1 \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --device cuda \
  --infer-batch-size 128 \
  2>&1 | tee "$RUN_DIR/pointer_train.log"

python math_memory_pointer_eval.py \
  --checkpoint "$CKPT_DIR/best.pt" \
  --experience-file data/experiences.json \
  --output-dir "$OUT_DIR" \
  --candidate-mode semantic \
  --candidate-seed 42 \
  --candidate-num 64 \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.8 \
  --compute-final-delta \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --embedding-cache-dir "$CACHE" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --device cuda \
  --infer-batch-size 128 \
  2>&1 | tee "$RUN_DIR/eval_semantic_top64.log"

echo "DONE"
echo "RUN_DIR=$RUN_DIR"
echo "METRICS_DIR=$OUT_DIR"
