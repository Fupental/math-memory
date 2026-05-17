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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
SCORER_BATCH_SIZE=${SCORER_BATCH_SIZE:-20}

RUN_DIR=${RUN_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_random64_20k_seed42}
OUT=${OUT:-${RUN_DIR}/generated_data/math_memory_random64_shot2_cand64_repeat4_beam5_seed42_scored.json}

mkdir -p "${RUN_DIR}/generated_data"

{
  echo "RUN_DIR=$RUN_DIR"
  echo "OUT=$OUT"
  echo "HOSTNAME=$(hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
  echo "SCORER_BATCH_SIZE=$SCORER_BATCH_SIZE"
  echo "COMMAND=$0"
  nvidia-smi || true
} | tee "$RUN_DIR/run_info.log"

python math_memory_generate.py \
  --experience-file data/experiences.json \
  --output-file "$OUT" \
  --candidate-num 64 \
  --repeat 4 \
  --beam-size 5 \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.8 \
  --anchor-limit 1000 \
  --score-mode delta_logprob \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size "$SCORER_BATCH_SIZE" \
  --scorer-max-length 4096 \
  2>&1 | tee -a "$RUN_DIR/generate.log"

python - "$OUT" <<'PY' | tee "$RUN_DIR/summary.log"
import json
import sys
from collections import Counter
from pathlib import Path

p = Path(sys.argv[1])
d = json.load(open(p, encoding="utf-8"))
rows = d["data"]
counts = Counter(row["query_id"] for row in rows)
required = ["total_score", "total_delta", "empty_score", "full_score", "candidate_ids"]
missing = {key: sum(1 for row in rows if key not in row) for key in required}
summary = {
    "output_file": str(p),
    "rows": len(rows),
    "unique_queries": len(counts),
    "min_rows_per_query": min(counts.values()) if counts else 0,
    "max_rows_per_query": max(counts.values()) if counts else 0,
    "metadata": d.get("metadata", {}),
    "missing_required_fields": missing,
}
summary_path = p.with_name(f"{p.stem}_summary.json")
json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

echo "DONE"
echo "OUT=$OUT"
