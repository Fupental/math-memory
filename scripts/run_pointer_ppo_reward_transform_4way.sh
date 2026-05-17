#!/usr/bin/env bash
set -euo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM

RUN_ROOT=${RUN_ROOT:-/home/fu_zhihang/projects/LeverLM/data/leverlm_pointer_ppo_reward_transform_seed42_4way_$(date +%Y%m%d_%H%M%S)}
VARIANTS=${VARIANTS:-"A_discounted_none B_discounted_group_zscore C_discounted_sigmoid D_discounted_tanh"}

mkdir -p "$RUN_ROOT"
echo "RUN_ROOT=$RUN_ROOT" | tee "$RUN_ROOT/run_info.log"
echo "VARIANTS=$VARIANTS" | tee -a "$RUN_ROOT/run_info.log"

for variant in $VARIANTS; do
  echo "===== RUN $variant =====" | tee -a "$RUN_ROOT/run_info.log"
  VARIANT="$variant" \
  RUN_ROOT="$RUN_ROOT" \
  bash scripts/run_pointer_ppo_reward_transform_one.sh
done

echo "DONE"
echo "RUN_ROOT=$RUN_ROOT"
