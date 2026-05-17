#!/bin/bash
#SBATCH --job-name=cuda_main1_test
#SBATCH --partition=RTX4090
#SBATCH --nodelist=main-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
mkdir -p logs

echo "===== BASIC SLURM ENV ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
echo "PATH=$PATH"

echo "===== NVIDIA SMI IN BATCH CONTEXT ====="
nvidia-smi -L || true
nvidia-smi || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true

echo "===== NVIDIA DEVICE FILES ====="
ls -l /dev/nvidia* || true
ls -l /proc/driver/nvidia/gpus || true

source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

run_python_cuda_check() {
  local label="$1"
  echo "===== PYTORCH CUDA CHECK: ${label} ====="
  python - <<'PY'
import os
import subprocess
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
        for idx in range(torch.cuda.device_count()):
            print("device", idx, torch.cuda.get_device_name(idx))
            x = torch.randn(1024, 1024, device=f"cuda:{idx}")
            y = x @ x.T
            torch.cuda.synchronize(idx)
            print("matmul_ok", idx, float(y[0, 0].detach().cpu()))
except Exception as exc:
    print("PYTORCH_CUDA_EXCEPTION:", repr(exc))
    raise
PY
}

run_python_cuda_check "batch default env" || true

echo "===== SRUN ENV AND CUDA CHECK ====="
srun --cpu-bind=none bash -lc '
set -uo pipefail
echo "--- inside srun ---"
hostname
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math
python - <<'"'"'PY'"'"'
import os, sys
print("python:", sys.executable)
for name in ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "NVIDIA_VISIBLE_DEVICES"]:
    print(f"{name}={os.environ.get(name)}")
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("name0:", torch.cuda.get_device_name(0))
        x = torch.randn(1024, 1024, device="cuda:0")
        y = x @ x.T
        torch.cuda.synchronize()
        print("matmul_ok:", float(y[0, 0].detach().cpu()))
except Exception as exc:
    print("PYTORCH_CUDA_EXCEPTION:", repr(exc))
    raise
PY
'

echo "===== TRY CUDA_DEVICE_ORDER=PCI_BUS_ID IN SRUN ====="
srun --cpu-bind=none bash -lc '
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID
source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math
python - <<'"'"'PY'"'"'
import os
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA_DEVICE_ORDER=", os.environ.get("CUDA_DEVICE_ORDER"))
try:
    import torch
    print("available=", torch.cuda.is_available(), "count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        x = torch.ones(1, device="cuda")
        print("tensor_ok=", x.item())
except Exception as exc:
    print("PYTORCH_CUDA_EXCEPTION:", repr(exc))
    raise
PY
'

echo "===== DONE ====="
