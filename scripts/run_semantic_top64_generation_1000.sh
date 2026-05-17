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

RUN_DIR=${RUN_DIR:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_semantic_top64_beam5_seed42}
OUT=${OUT:-${RUN_DIR}/generated_data/math_memory_semantic_top64_shot2_beam5_seed42_scored.json}
CACHE=${CACHE:-/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_anchor400_r1/cache/math_memory_embeddings}

mkdir -p "${RUN_DIR}/generated_data"

python math_memory_generate_semantic.py \
  --experience-file data/experiences.json \
  --output-file "$OUT" \
  --embedding-cache-dir "$CACHE" \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128 \
  --recall-device cuda \
  --recall-batch-size 128 \
  --candidate-num 64 \
  --beam-size 5 \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.8 \
  --anchor-limit 1000 \
  --score-mode delta_logprob \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 32 \
  --scorer-max-length 4096 \
  2>&1 | tee -a "$RUN_DIR/semantic_generate.log"

python - <<'PY'
import json
from collections import Counter
from pathlib import Path

p = Path("/home/fu_zhihang/projects/LeverLM/data/leverlm_math_memory_semantic_top64_beam5_seed42/generated_data/math_memory_semantic_top64_shot2_beam5_seed42_scored.json")
d = json.load(open(p, encoding="utf-8"))
rows = d["data"]
q = Counter(row["query_id"] for row in rows)
summary = {
    "output_file": str(p),
    "rows": len(rows),
    "unique_queries": len(q),
    "min_rows_per_query": min(q.values()) if q else 0,
    "max_rows_per_query": max(q.values()) if q else 0,
    "is_complete": d["metadata"].get("is_complete"),
    "candidate_mode": d["metadata"].get("candidate_mode"),
    "candidate_num": d["metadata"].get("candidate_num"),
    "beam_size": d["metadata"].get("beam_size"),
    "shot_num": d["metadata"].get("shot_num"),
}
summary_path = p.with_name(f"{p.stem}_summary.json")
json.dump(summary, open(summary_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
