#!/bin/bash
#SBATCH --job-name=gen_sem20k_resume
#SBATCH --partition=RTX4090
#SBATCH --nodelist=main-5,main-6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
mkdir -p logs

source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

export HF_HOME=/home/fu_zhihang/projects/LeverLM/data/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR=/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_semantic_topk_chunks_20k_seed42_main4_gpu1_20260517_070632
OUT=${RUN_DIR}/generated_data/math_memory_semantic_topk_chunks_shot2_cand64_repeat4_beam5_seed42_scored.json
CACHE=/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings

{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "HOSTNAME=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "OUT=${OUT}"
  echo "CACHE=${CACHE}"
  nvidia-smi
} | tee -a "${RUN_DIR}/slurm_resume_info.log"

RUN_DIR="${RUN_DIR}" \
OUT="${OUT}" \
CACHE="${CACHE}" \
SCORER_BATCH_SIZE=20 \
bash scripts/run_generate_semantic_chunks_20k.sh
