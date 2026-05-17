#!/bin/bash
#SBATCH --job-name=cuda_main1_each
#SBATCH --partition=RTX4090
#SBATCH --nodelist=main-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -uo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
mkdir -p logs

echo "===== MAIN1 SINGLE-GPU ARRAY DIAG ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"

echo "===== NVIDIA SMI ====="
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
nvidia-smi || true

source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

echo "===== PYTORCH CUDA ====="
python - <<'PY' || true
import os
import sys

print("python:", sys.executable)
for name in [
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "NVIDIA_VISIBLE_DEVICES",
]:
    print(f"{name}={os.environ.get(name)}")

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device0:", torch.cuda.get_device_name(0))
        x = torch.randn(2048, 2048, device="cuda:0")
        y = x @ x.T
        torch.cuda.synchronize()
        print("matmul_ok:", float(y[0, 0].detach().cpu()))
except Exception as exc:
    print("PYTORCH_CUDA_EXCEPTION:", repr(exc))
    raise
PY

echo "===== DONE ====="
