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

DATA=${DATA:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/generated_data/math_memory_shot2_cand64_repeat1_beam5_seed42_scored_with_candidates.json}
CACHE=${CACHE:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
RUN_DIR=${RUN_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_pointer_rce_random64_generated_valacc_20seeds_$(date +%Y%m%d_%H%M%S)}
CKPT_DIR=${CKPT_DIR:-${RUN_DIR}/model_cpk/pointer_rce_random64_generated_valacc}
METRICS_DIR=${METRICS_DIR:-${RUN_DIR}/metrics/best_repeated}
CANDIDATE_SEEDS=${CANDIDATE_SEEDS:-"1 2 3 4 5 6 7 8 9 10 42 100 123 456 789 1000 2024 2025 2026 3407"}
VAL_CANDIDATE_SEED=${VAL_CANDIDATE_SEED:-42}

mkdir -p "$RUN_DIR" "$CKPT_DIR" "$METRICS_DIR"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "CKPT_DIR=$CKPT_DIR"
  echo "METRICS_DIR=$METRICS_DIR"
  echo "DATA=$DATA"
  echo "CANDIDATE_SEEDS=$CANDIDATE_SEEDS"
  echo "VAL_CANDIDATE_SEED=$VAL_CANDIDATE_SEED"
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
  --eval-candidate-mode random \
  --candidate-seed "$VAL_CANDIDATE_SEED" \
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

for candidate_seed in $CANDIDATE_SEEDS; do
  echo "===== Evaluate candidate_seed=${candidate_seed} ====="
  python math_memory_pointer_eval.py \
    --checkpoint "$CKPT_DIR/best.pt" \
    --experience-file data/experiences.json \
    --output-dir "$METRICS_DIR" \
    --candidate-mode random \
    --candidate-seed "$candidate_seed" \
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
    2>&1 | tee "$RUN_DIR/eval_seed${candidate_seed}.log"
done

python math_memory_summarize_pointer.py \
  --metrics-dir "$METRICS_DIR" \
  2>&1 | tee "$RUN_DIR/pointer_repeated_summary.log"

echo "DONE"
echo "RUN_DIR=$RUN_DIR"
echo "SUMMARY_JSON=$METRICS_DIR/pointer_repeated_summary.json"
echo "SUMMARY_CSV=$METRICS_DIR/pointer_repeated_summary.csv"
