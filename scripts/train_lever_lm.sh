task=${1:-caption}
dataset=${2:-coco2017}
device_num=${3:-1}
lever_lm=${4:-query_img_icd_img_text}
infer_model=${INFER_MODEL:-flamingo_9B}
infer_model_name=${INFER_MODEL_NAME:-OpenFlamingo-9B-vitl-mpt7b}
strategy=${TRAIN_STRATEGY:-auto}
precision=${TRAIN_PRECISION:-16}
python_bin=${PYTHON_BIN:-/home/fupental/project/miniconda3/envs/leverlm/bin/python}


run_train() {
    local data_file=${DATA_FILE:-$1}
 
    local ex_name_prefix="main_${task}"

    if [ "${task}" == "vqa" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        "${python_bin}" train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        epochs=20 \
                        val_step=80 \
                        ex_name="${ex_name_prefix}_${lever_lm}" \
                        trainer_args.devices=${device_num} \
                        trainer_args.strategy=${strategy} \
                        trainer_args.precision=${precision} \
                        +infer_model=${infer_model} \
                        dataset=${dataset} \
                        task=${task}

    elif [ "${task}" == "caption" ]; then
        echo "==========Begin: ${ex_name_prefix}-LeverLM: ${lever_lm}==========" 
        "${python_bin}" train.py train="${lever_lm}" \
                        data_files="${data_file}" \
                        epochs=20 \
                        val_step=80 \
                        ex_name="${ex_name_prefix}_freeze_adapter_non_norm_${lever_lm}" \
                        trainer_args.devices=${device_num} \
                        trainer_args.strategy=${strategy} \
                        trainer_args.precision=${precision} \
                        +infer_model=${infer_model} \
                        dataset=${dataset} \
                        task=${task} \
                        train.lever_lm.norm=false \
                        train.lever_lm.freeze_prefix_list="[img_model,sen_model]" \
                        train.lever_lm.adapter=true
    fi
}

run_train "${task}-${dataset}-${infer_model_name}-RandSampler-scorer:infoscore-construct_order:left-beam_size:5-few_shot:2-candidate_num:64-sample_num:5000.json"
