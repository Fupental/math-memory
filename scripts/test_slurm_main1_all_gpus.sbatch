#!/bin/bash
#SBATCH --job-name=cuda_main1_all
#SBATCH --partition=RTX4090
#SBATCH --nodelist=main-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

cd /home/fu_zhihang/projects/LeverLM/LeverLM
mkdir -p logs

echo "===== MAIN1 ALL-GPU DIAG ====="
date
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

echo "===== FULL NVIDIA SMI ====="
nvidia-smi -L || true
nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true
nvidia-smi || true

source /home/fu_zhihang/miniconda3/etc/profile.d/conda.sh
conda activate leverlm_math

echo "===== PYTORCH DEFAULT VISIBLE SET ====="
python - <<'PY' || true
import os
import torch
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("available=", torch.cuda.is_available(), "count=", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "===== TEST EACH LOGICAL CUDA_VISIBLE_DEVICES ENTRY ====="
IFS=',' read -ra GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
for gpu in "${GPUS[@]}"; do
  echo "--- TEST CUDA_VISIBLE_DEVICES=${gpu} ---"
  CUDA_VISIBLE_DEVICES="$gpu" python - <<'PY' || true
import os
print("env CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("available=", torch.cuda.is_available(), "count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("name0=", torch.cuda.get_device_name(0))
        x = torch.randn(1024, 1024, device="cuda:0")
        y = x @ x.T
        torch.cuda.synchronize()
        print("matmul_ok=", float(y[0, 0].detach().cpu()))
except Exception as exc:
    print("EXCEPTION=", repr(exc))
PY
done

echo "===== DONE ====="
