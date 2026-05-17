#!/bin/bash
#SBATCH --job-name=ptr_grpo
#SBATCH --partition=RTX4090
#SBATCH --nodelist=main-5,main-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
mkdir -p logs

: "${VARIANT:?Set VARIANT=A_step_delta|B_reward_to_go_delta|C_discounted_delta|D_step_delta_correctness}"
: "${RUN_ROOT:?Set RUN_ROOT to the shared output directory}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

srun --cpu-bind=none --export=ALL,RUN_ROOT="$RUN_ROOT",VARIANT="$VARIANT" \
  bash scripts/run_pointer_grpo_from_old_random64_seed42_one.sh
