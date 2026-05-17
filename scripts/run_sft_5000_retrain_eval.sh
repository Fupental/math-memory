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

DATA=${DATA:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/generated_data/math_memory_shot2_cand64_repeat1_beam5_seed42.json}
EXPERIENCE_FILE=${EXPERIENCE_FILE:-data/experiences.json}
CACHE=${CACHE:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}
RUN_DIR=${RUN_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_sft_5000_retrain_$(date +%Y%m%d_%H%M%S)}
CKPT_DIR=${CKPT_DIR:-${RUN_DIR}/model_cpk/sft_shot2_cand64_repeat1_beam5_seed42}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}
SCORER_MODEL=${SCORER_MODEL:-Qwen/Qwen3-8B}

BATCH_SIZE=${BATCH_SIZE:-64}
MAX_EPOCHS=${MAX_EPOCHS:-100}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-5}
LR=${LR:-1e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-3}
TRAIN_RATIO=${TRAIN_RATIO:-0.8}
SEED=${SEED:-42}
SHOT_NUM=${SHOT_NUM:-2}

mkdir -p "$CKPT_DIR" "$RUN_DIR/metrics/best" "$RUN_DIR/metrics/last"

cat > "$RUN_DIR/train_command.txt" <<EOF
python math_memory_train.py \\
  --generated-file "$DATA" \\
  --experience-file "$EXPERIENCE_FILE" \\
  --output-dir "$CKPT_DIR" \\
  --embedding-cache-dir "$CACHE" \\
  --embedding-model "$EMBEDDING_MODEL" \\
  --embedding-device cuda \\
  --embedding-batch-size 128 \\
  --batch-size "$BATCH_SIZE" \\
  --max-epochs "$MAX_EPOCHS" \\
  --early-stop-patience "$EARLY_STOP_PATIENCE" \\
  --lr "$LR" \\
  --weight-decay "$WEIGHT_DECAY" \\
  --device cuda
EOF

python math_memory_train.py \
  --generated-file "$DATA" \
  --experience-file "$EXPERIENCE_FILE" \
  --output-dir "$CKPT_DIR" \
  --embedding-cache-dir "$CACHE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --batch-size "$BATCH_SIZE" \
  --max-epochs "$MAX_EPOCHS" \
  --early-stop-patience "$EARLY_STOP_PATIENCE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --device cuda \
  2>&1 | tee "$RUN_DIR/sft_train.log"

cat > "$RUN_DIR/eval_best_command.txt" <<EOF
python math_memory_eval.py \\
  --method lever_lm \\
  --checkpoint "$CKPT_DIR/best.pt" \\
  --experience-file "$EXPERIENCE_FILE" \\
  --output-dir "$RUN_DIR/metrics/best" \\
  --shot-num "$SHOT_NUM" \\
  --seed "$SEED" \\
  --train-ratio "$TRAIN_RATIO" \\
  --compute-final-delta \\
  --scorer-model "$SCORER_MODEL" \\
  --scorer-device cuda \\
  --scorer-dtype bf16 \\
  --scorer-batch-size 16 \\
  --scorer-max-length 4096 \\
  --embedding-cache-dir "$CACHE" \\
  --embedding-model "$EMBEDDING_MODEL" \\
  --embedding-device cuda \\
  --embedding-batch-size 128
EOF

python math_memory_eval.py \
  --method lever_lm \
  --checkpoint "$CKPT_DIR/best.pt" \
  --experience-file "$EXPERIENCE_FILE" \
  --output-dir "$RUN_DIR/metrics/best" \
  --shot-num "$SHOT_NUM" \
  --seed "$SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --compute-final-delta \
  --scorer-model "$SCORER_MODEL" \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --embedding-cache-dir "$CACHE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  2>&1 | tee "$RUN_DIR/eval_best.log"

cat > "$RUN_DIR/eval_last_command.txt" <<EOF
python math_memory_eval.py \\
  --method lever_lm \\
  --checkpoint "$CKPT_DIR/last.pt" \\
  --experience-file "$EXPERIENCE_FILE" \\
  --output-dir "$RUN_DIR/metrics/last" \\
  --shot-num "$SHOT_NUM" \\
  --seed "$SEED" \\
  --train-ratio "$TRAIN_RATIO" \\
  --compute-final-delta \\
  --scorer-model "$SCORER_MODEL" \\
  --scorer-device cuda \\
  --scorer-dtype bf16 \\
  --scorer-batch-size 16 \\
  --scorer-max-length 4096 \\
  --embedding-cache-dir "$CACHE" \\
  --embedding-model "$EMBEDDING_MODEL" \\
  --embedding-device cuda \\
  --embedding-batch-size 128
EOF

python math_memory_eval.py \
  --method lever_lm \
  --checkpoint "$CKPT_DIR/last.pt" \
  --experience-file "$EXPERIENCE_FILE" \
  --output-dir "$RUN_DIR/metrics/last" \
  --shot-num "$SHOT_NUM" \
  --seed "$SEED" \
  --train-ratio "$TRAIN_RATIO" \
  --compute-final-delta \
  --scorer-model "$SCORER_MODEL" \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 16 \
  --scorer-max-length 4096 \
  --embedding-cache-dir "$CACHE" \
  --embedding-model "$EMBEDDING_MODEL" \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  2>&1 | tee "$RUN_DIR/eval_last.log"

echo "RUN_DIR=$RUN_DIR"
echo "CKPT_DIR=$CKPT_DIR"
echo "best metrics: $RUN_DIR/metrics/best"
echo "last metrics: $RUN_DIR/metrics/last"
