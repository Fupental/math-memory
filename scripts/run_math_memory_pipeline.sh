#!/usr/bin/env bash
set -euo pipefail

original_args=("$@")
experience_file=""
result_dir="${RESULT_DIR:-/tmp/leverlm_math_memory_results}"
stage="all"
shot_num=2
candidate_num=64
repeat=4
beam_size=5
seed=42
train_ratio=0.8
scorer_model="Qwen/Qwen3-8B"
score_mode="delta_logprob"
embedding_model="Qwen/Qwen3-Embedding-0.6B"
scorer_batch_size=4
embedding_batch_size=16
max_epochs=100
early_stop_patience=5
early_stop_min_delta=0.0
batch_size=64
test_limit=""
anchor_limit=""
smoke=0
n_embd=512
n_head=8
n_layer=2
device="cuda"
scorer_device="cuda"
embedding_device="cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experience-file) experience_file="$2"; shift 2 ;;
    --result-dir) result_dir="$2"; shift 2 ;;
    --stage) stage="$2"; shift 2 ;;
    --shot-num) shot_num="$2"; shift 2 ;;
    --candidate-num) candidate_num="$2"; shift 2 ;;
    --repeat) repeat="$2"; shift 2 ;;
    --beam-size) beam_size="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --train-ratio) train_ratio="$2"; shift 2 ;;
    --scorer-model) scorer_model="$2"; shift 2 ;;
    --score-mode) score_mode="$2"; shift 2 ;;
    --embedding-model) embedding_model="$2"; shift 2 ;;
    --scorer-batch-size) scorer_batch_size="$2"; shift 2 ;;
    --embedding-batch-size) embedding_batch_size="$2"; shift 2 ;;
    --max-epochs) max_epochs="$2"; shift 2 ;;
    --early-stop-patience) early_stop_patience="$2"; shift 2 ;;
    --early-stop-min-delta) early_stop_min_delta="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --test-limit) test_limit="$2"; shift 2 ;;
    --anchor-limit) anchor_limit="$2"; shift 2 ;;
    --n-embd) n_embd="$2"; shift 2 ;;
    --n-head) n_head="$2"; shift 2 ;;
    --n-layer) n_layer="$2"; shift 2 ;;
    --device) device="$2"; shift 2 ;;
    --scorer-device) scorer_device="$2"; shift 2 ;;
    --embedding-device) embedding_device="$2"; shift 2 ;;
    --smoke) smoke=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$experience_file" ]]; then
  echo "--experience-file is required" >&2
  exit 1
fi

mkdir -p "$result_dir"
log_file="${result_dir}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_file") 2>&1
echo "Pipeline log: $log_file"
printf "Command:"
printf " %q" "$0" "${original_args[@]}"
printf "\n"

mock_data_args=()
fast_dev_args=()
if [[ "$smoke" == "1" ]]; then
  scorer_model="mock"
  embedding_model="mock"
  candidate_num=8
  repeat=1
  beam_size=2
  anchor_limit=3
  test_limit=10
  max_epochs=1
  early_stop_patience=5
  early_stop_min_delta=0.0
  batch_size=2
  n_embd=64
  n_head=4
  n_layer=2
  device="cpu"
  scorer_device="cpu"
  embedding_device="cpu"
  mock_data_args=(--mock-data)
  fast_dev_args=(--fast-dev-run)
fi

run_name="shot${shot_num}_cand${candidate_num}_repeat${repeat}_beam${beam_size}_seed${seed}"
generated_file="${result_dir}/generated_data/math_memory_${run_name}.json"
ckpt_dir="${result_dir}/model_cpk/math_memory_${run_name}"
metrics_dir="${result_dir}/metrics/math_memory_${run_name}"
embedding_cache_dir="${result_dir}/cache/math_memory_embeddings"

extra_generate_args=()
if [[ -n "$anchor_limit" ]]; then
  extra_generate_args+=(--anchor-limit "$anchor_limit")
fi

extra_eval_args=()
if [[ -n "$test_limit" ]]; then
  extra_eval_args+=(--test-limit "$test_limit")
fi

run_generate() {
  python math_memory_generate.py \
    --experience-file "$experience_file" \
    --output-file "$generated_file" \
    --shot-num "$shot_num" \
    --candidate-num "$candidate_num" \
    --repeat "$repeat" \
    --beam-size "$beam_size" \
    --seed "$seed" \
    --train-ratio "$train_ratio" \
    --scorer-model "$scorer_model" \
    --score-mode "$score_mode" \
    --scorer-device "$scorer_device" \
    --scorer-batch-size "$scorer_batch_size" \
    "${mock_data_args[@]}" \
    "${extra_generate_args[@]}"
}

run_train() {
  python math_memory_train.py \
    --generated-file "$generated_file" \
    --experience-file "$experience_file" \
    --output-dir "$ckpt_dir" \
    --embedding-cache-dir "$embedding_cache_dir" \
    --embedding-model "$embedding_model" \
    --embedding-device "$embedding_device" \
    --embedding-batch-size "$embedding_batch_size" \
    --device "$device" \
    --seed "$seed" \
    --batch-size "$batch_size" \
    --max-epochs "$max_epochs" \
    --early-stop-patience "$early_stop_patience" \
    --early-stop-min-delta "$early_stop_min_delta" \
    --n-embd "$n_embd" \
    --n-head "$n_head" \
    --n-layer "$n_layer" \
    "${fast_dev_args[@]}"
}

run_eval_lever_lm() {
  python math_memory_eval.py \
    --method lever_lm \
    --checkpoint "${ckpt_dir}/last.pt" \
    --experience-file "$experience_file" \
    --output-dir "$metrics_dir" \
    --shot-num "$shot_num" \
    --seed "$seed" \
    --train-ratio "$train_ratio" \
    --scorer-model "$scorer_model" \
    --scorer-device "$scorer_device" \
    --scorer-batch-size "$scorer_batch_size" \
    --embedding-cache-dir "$embedding_cache_dir" \
    --embedding-model "$embedding_model" \
    --embedding-device "$embedding_device" \
    --embedding-batch-size "$embedding_batch_size" \
    --device "$device" \
    "${mock_data_args[@]}" \
    "${extra_eval_args[@]}"
}

run_eval_rs() {
  python math_memory_eval.py \
    --method rs \
    --experience-file "$experience_file" \
    --output-dir "$metrics_dir" \
    --shot-num "$shot_num" \
    --seed "$seed" \
    --train-ratio "$train_ratio" \
    --scorer-model "$scorer_model" \
    --scorer-device "$scorer_device" \
    --scorer-batch-size "$scorer_batch_size" \
    --embedding-cache-dir "$embedding_cache_dir" \
    "${mock_data_args[@]}" \
    "${extra_eval_args[@]}"
}

case "$stage" in
  all)
    run_generate
    run_train
    run_eval_lever_lm
    run_eval_rs
    ;;
  generate) run_generate ;;
  train) run_train ;;
  eval_lever_lm) run_eval_lever_lm ;;
  eval_rs) run_eval_rs ;;
  *) echo "Unknown stage: $stage" >&2; exit 1 ;;
esac

echo "Generated data: $generated_file"
echo "Checkpoint dir: $ckpt_dir"
echo "Metrics dir: $metrics_dir"
