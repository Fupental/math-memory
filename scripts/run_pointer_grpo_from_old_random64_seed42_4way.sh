#!/usr/bin/env bash
set -euo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM

RUN_ROOT=${RUN_ROOT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_pointer_grpo_old_random64_seed42_4way_$(date +%Y%m%d_%H%M%S)}
VARIANTS=${VARIANTS:-"A_step_delta B_reward_to_go_delta C_discounted_delta D_step_delta_correctness"}

mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT" | tee "$RUN_ROOT/4way_run_info.log"
echo "VARIANTS=$VARIANTS" | tee -a "$RUN_ROOT/4way_run_info.log"

for variant in $VARIANTS; do
  echo "===== RUN $variant =====" | tee -a "$RUN_ROOT/4way_run_info.log"
  RUN_ROOT="$RUN_ROOT" VARIANT="$variant" \
    bash scripts/run_pointer_grpo_from_old_random64_seed42_one.sh
done

echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
