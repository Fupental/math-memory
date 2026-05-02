task=${1:-caption}
dataset=${2:-coco2017}
gpu_ids=${3:-"[0]"}
infer_model=${INFER_MODEL:-flamingo_9B}
load_from_local=${LOAD_FROM_LOCAL:-true}
python_bin=${PYTHON_BIN:-/home/fupental/project/miniconda3/envs/leverlm/bin/python}


"${python_bin}" generate_data.py beam_size=5 \
                               cand_num=64 \
                               sample_num=5000 \
                               gpu_ids="${gpu_ids}" \
                               task=${task} \
                               dataset=${dataset} \
                               infer_model=${infer_model} \
                               infer_model.load_from_local=${load_from_local} \
                               few_shot_num=2
