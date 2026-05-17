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

VARIANT=${VARIANT:-A_step_delta}
RUN_ROOT=${RUN_ROOT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_pointer_grpo_old_random64_seed42_$(date +%Y%m%d_%H%M%S)}
BASE_CKPT=${BASE_CKPT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_pointer_rce_random64_20260516_114841/model_cpk/pointer_rce_random64/best.pt}
CACHE=${CACHE:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
CKPT_DIR=${CKPT_DIR:-${RUN_ROOT}/${VARIANT}/model_cpk/pointer_grpo}
METRICS_DIR=${METRICS_DIR:-${RUN_ROOT}/${VARIANT}/metrics/best_repeated}
CANDIDATE_SEEDS=${CANDIDATE_SEEDS:-"1 2 3 4 5 6 7 8 9 10 42 100 123 456 789 1000 2024 2025 2026 3407"}

SCORER_DEVICE_MAP=${SCORER_DEVICE_MAP:-}
SCORER_MAX_MEMORY=${SCORER_MAX_MEMORY:-}
EMBEDDING_DEVICE_MAP=${EMBEDDING_DEVICE_MAP:-}
EMBEDDING_MAX_MEMORY=${EMBEDDING_MAX_MEMORY:-}

case "$VARIANT" in
  A_step_delta)
    REWARD_MODE=delta_logprob
    CREDIT_MODE=step
    CREDIT_GAMMA=0.3
    CORRECTNESS_BONUS=1.0
    ;;
  B_reward_to_go_delta)
    REWARD_MODE=delta_logprob
    CREDIT_MODE=reward_to_go
    CREDIT_GAMMA=0.3
    CORRECTNESS_BONUS=1.0
    ;;
  C_discounted_delta)
    REWARD_MODE=delta_logprob
    CREDIT_MODE=discounted
    CREDIT_GAMMA=0.3
    CORRECTNESS_BONUS=1.0
    ;;
  D_step_delta_correctness)
    REWARD_MODE=delta_plus_correctness
    CREDIT_MODE=step
    CREDIT_GAMMA=0.3
    CORRECTNESS_BONUS=1.0
    ;;
  *)
    echo "Unknown VARIANT=$VARIANT" >&2
    exit 2
    ;;
esac

mkdir -p "$RUN_ROOT" "$CKPT_DIR" "$METRICS_DIR"

scorer_extra=()
if [[ -n "$SCORER_DEVICE_MAP" ]]; then
  scorer_extra+=(--scorer-device-map "$SCORER_DEVICE_MAP")
fi
if [[ -n "$SCORER_MAX_MEMORY" ]]; then
  scorer_extra+=(--scorer-max-memory "$SCORER_MAX_MEMORY")
fi
embedding_extra=()
if [[ -n "$EMBEDDING_DEVICE_MAP" ]]; then
  embedding_extra+=(--embedding-device-map "$EMBEDDING_DEVICE_MAP")
fi
if [[ -n "$EMBEDDING_MAX_MEMORY" ]]; then
  embedding_extra+=(--embedding-max-memory "$EMBEDDING_MAX_MEMORY")
fi

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "VARIANT=$VARIANT"
  echo "BASE_CKPT=$BASE_CKPT"
  echo "CKPT_DIR=$CKPT_DIR"
  echo "METRICS_DIR=$METRICS_DIR"
  echo "REWARD_MODE=$REWARD_MODE"
  echo "CREDIT_MODE=$CREDIT_MODE"
  echo "CREDIT_GAMMA=$CREDIT_GAMMA"
  echo "CORRECTNESS_BONUS=$CORRECTNESS_BONUS"
  echo "CANDIDATE_SEEDS=$CANDIDATE_SEEDS"
  echo "HOSTNAME=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  nvidia-smi || true
} | tee "$RUN_ROOT/${VARIANT}_run_info.log"

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("ERROR: CUDA is not available; aborting to avoid accidental CPU training.")
    sys.exit(1)
print("CUDA preflight ok:", torch.cuda.get_device_name(0))
PY

python math_memory_pointer_grpo_train.py \
  --checkpoint "$BASE_CKPT" \
  --reference-checkpoint "$BASE_CKPT" \
  --experience-file data/experiences.json \
  --output-dir "$CKPT_DIR" \
  --embedding-cache-dir "$CACHE" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  "${embedding_extra[@]}" \
  --train-ratio 0.8 \
  --seed 42 \
  --candidate-mode random \
  --candidate-seed 42 \
  --candidate-num 64 \
  --group-size 20 \
  --temperature 1.0 \
  --reward-mode "$REWARD_MODE" \
  --credit-mode "$CREDIT_MODE" \
  --credit-gamma "$CREDIT_GAMMA" \
  --correctness-bonus "$CORRECTNESS_BONUS" \
  --lr 5e-6 \
  --batch-size 8 \
  --max-steps 200 \
  --clip-eps 0.1 \
  --entropy-coef 0.001 \
  --ref-kl-coef 0.05 \
  --best-window 20 \
  --best-metric train_window_final_delta \
  --early-stop-patience 0 \
  --save-every 25 \
  --checkpoint-steps 0,25,50,100,200 \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  "${scorer_extra[@]}" \
  --device cuda \
  2>&1 | tee "$RUN_ROOT/${VARIANT}_train.log"

for candidate_seed in $CANDIDATE_SEEDS; do
  echo "===== $VARIANT evaluate candidate_seed=${candidate_seed} ====="
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
    "${scorer_extra[@]}" \
    --embedding-cache-dir "$CACHE" \
    --embedding-model Qwen/Qwen3-Embedding-0.6B \
    --embedding-device cuda \
    --embedding-batch-size 128 \
    "${embedding_extra[@]}" \
    --device cuda \
    --infer-batch-size 128 \
    2>&1 | tee "$RUN_ROOT/${VARIANT}_eval_seed${candidate_seed}.log"
done

python math_memory_summarize_pointer.py \
  --metrics-dir "$METRICS_DIR" \
  2>&1 | tee "$RUN_ROOT/${VARIANT}_summary.log"

echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
echo "VARIANT=$VARIANT"
echo "CKPT_DIR=$CKPT_DIR"
echo "SUMMARY_JSON=$METRICS_DIR/pointer_repeated_summary.json"
