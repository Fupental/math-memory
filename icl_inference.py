import datetime
import json
import os
import random
import uuid

import hydra
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoProcessor

from lever_lm.utils import init_interface
from open_mmicl.icl_inferencer import ICLInferecer
from open_mmicl.metrics.cider_calculator import compute_cider
from open_mmicl.metrics.vqa_metrics import compute_vqa_accuracy
from open_mmicl.retriever import *
from utils import (
    caption_postprocess,
    get_lever_lm_path,
    init_lever_lm,
    load_ds,
    vqa_postprocess,
)

"""
ICL 推理与评估入口。

论文训练好 Lever-LM 后，并不直接让它回答 caption/VQA/SST2，而是让它为每个
测试 query 生成一串 ICD 样本 id。随后 ICLInferecer 会把这些 ICD 拼进 LVLM 的
prompt，让冻结的 LVLM 完成真正任务。这个文件同时评估随机检索、CLIP 相似度检索、
zero-shot 和 Lever-LM 检索，便于复现实验表格。
"""


def record(result_json_path: str, new_data: dict):
    """把某个 retriever 的指标增量写入 metrics.json，支持多方法连续评估。"""
    recorded_data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, "r") as f:
            recorded_data = json.load(f)

    with open(result_json_path, "w") as f:
        recorded_data.update(new_data)
        json.dump(recorded_data, f, indent=4)


def evaluate_retriever(
    retriever_name,
    inferencer,
    retriever,
    ds,
    base_info,
    shot_num_list,
    result_json_path,
    cfg,
):
    """对一个 retriever 在多个 shot_num 下跑完整 ICL 评估。"""
    retriever_res = {}
    info = base_info + retriever_name
    for shot_num in shot_num_list:
        logger.info(
            f"Now begin test {cfg.task.task_name}: {retriever_name} with {shot_num=}"
        )
        output_files = info + f"-bs:{cfg.inference_bs}-{shot_num=}"
        # retrieve 返回形如 List[List[int]] 的 ICD id；外层按测试样本排列。
        icd_idx_list = retriever.retrieve(shot_num)

        if cfg.task.task_name == "caption":
            metric = inference_caption(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ann_path=cfg.dataset.val_coco_annotation_file,
                output_json_filename=output_files,
                model_name=cfg.infer_model.name,
            )
        elif cfg.task.task_name == "vqa":
            metric = inference_vqa(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                val_ques_path=cfg.dataset.val_ques_path,
                val_ann_path=cfg.dataset.val_ann_path,
                output_json_filename=output_files,
                model_name=cfg.infer_model.name,
            )
        elif cfg.task.task_name == "sst2":
            metric = inference_cls(
                inferencer=inferencer,
                ds=ds,
                icd_idx_list=icd_idx_list,
                output_json_filename=output_files,
            )
        retriever_res[f"{shot_num=}"] = metric
        logger.info(f"{output_files}: {metric=}")
        record(result_json_path, {info: retriever_res})


def inference_cls(
    inferencer,
    ds,
    icd_idx_list,
    output_json_filename,
):
    """SST2 这类分类任务使用 ppl_inference，根据候选标签困惑度决定预测。"""
    output_dict = inferencer.ppl_inference(
        ds["train"],
        ds["validation"],
        icd_idx_list,
        output_json_filename=output_json_filename,
    )
    predictions = [v["prediction"] for k, v in output_dict.items()]
    targets = ds["validation"]["label"]
    metrics = {}

    # 计算并存储准确率
    metrics["accuracy"] = accuracy_score(targets, predictions)

    # 计算并存储宏平均和加权平均精确率
    metrics["precision_macro"] = precision_score(targets, predictions, average="macro")

    # 计算并存储宏平均和加权平均召回率
    metrics["recall_macro"] = recall_score(targets, predictions, average="macro")

    # 计算并存储宏平均和加权平均F1分数
    metrics["f1_macro"] = f1_score(targets, predictions, average="macro")
    return metrics["accuracy"]


def init_retriever(retriever_name, ds, cfg):
    """根据配置创建不同 ICD 选择方法。

    ZeroShot 不提供 ICD；RandomRetriever 随机采样；MMTopKRetriever 用 CLIP
    图文相似度检索；LeverLMRetriever 加载训练好的 Lever-LM 逐步生成 ICD 序列。
    """
    if retriever_name == "ZeroShot":
        return ZeroRetriever(ds["train"], ds["validation"])
    elif retriever_name == "RandomRetriever":
        return RandRetriever(
            ds["train"],
            ds["validation"],
            seed=cfg.seed,
            fixed=cfg.random_retrieval_fixed,
        )
    elif retriever_name.startswith("MMTopKRetriever"):
        mode = retriever_name.split("-")[-1]
        index_field = (
            cfg.task.icd_text_feature_field
            if mode.endswith("t")
            else cfg.task.image_field
        )
        test_field = (
            cfg.task.image_field
            if mode.startswith("i")
            else cfg.task.icd_text_feature_field
        )

        cache_file = os.path.join(
            cfg.result_dir,
            "cache",
            f'{cfg.task.task_name}-{cfg.dataset.name}-{cfg.mmtopk_clip_name.split("/")[-1]}-{mode}-'
            f"index_field:{index_field}-test_data_num:{cfg.test_data_num}-"
            f"test_field:{test_field}-emb_cache.pth",
        )
        return MMTopkRetriever(
            ds["train"],
            ds["validation"],
            mode=mode,
            index_field=index_field,
            test_field=test_field,
            clip_model_name=cfg.mmtopk_clip_name,
            cache_file=cache_file,
            reversed_order=cfg.mmtopk_reversed_order,
            batch_size=32,
            num_workers=8,
        )
    elif retriever_name == "LeverLMRetriever":
        lever_lm_path = get_lever_lm_path(cfg)
        lever_lm, processor = init_lever_lm(cfg, lever_lm_path=lever_lm_path)
        return LeverLMRetriever(
            ds["train"],
            ds["validation"],
            lever_lm=lever_lm,
            processor=processor,
            query_image_field=cfg.train.lever_lm_ds.query_image_field,
            query_text_field=cfg.train.lever_lm_ds.query_text_field,
            icd_image_field=cfg.train.lever_lm_ds.icd_image_field,
            icd_text_field=cfg.train.lever_lm_ds.icd_text_field,
            device=cfg.device,
            infer_batch_size=cfg.lever_lm_bs,
            infer_num_workers=cfg.lever_lm_num_workers,
            reverse_seq=cfg.reverse_seq,
        )

    return None


def inference_caption(
    inferencer,
    ds,
    icd_idx_list,
    val_ann_path,
    output_json_filename,
    model_name,
):
    """执行 image captioning 推理，并用 COCO CIDEr 评分。"""
    output_dict = inferencer.inference(
        train_ds=ds["train"],
        test_ds=ds["validation"],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    pred_coco = []
    for idx in output_dict:
        pred_coco.append(
            {
                "image_id": output_dict[idx]["image_id"],
                "caption": caption_postprocess(
                    output_dict[idx]["prediction"], model_name
                ),
            }
        )
    cider_score = compute_cider(pred_coco, val_ann_path)
    return cider_score * 100


def inference_vqa(
    inferencer,
    ds,
    icd_idx_list,
    val_ques_path,
    val_ann_path,
    output_json_filename,
    model_name,
):
    """执行 VQA 推理，并调用官方风格的 VQA accuracy 计算。"""
    output_dict = inferencer.inference(
        train_ds=ds["train"],
        test_ds=ds["validation"],
        ice_idx_list=icd_idx_list,
        output_json_filename=output_json_filename,
    )
    preds = []
    for idx in output_dict:
        preds.append(
            {
                "answer": vqa_postprocess(
                    output_dict[idx]["prediction"], model_name=model_name
                ),
                "question_id": output_dict[idx]["question_id"],
            }
        )
    random_uuid = str(uuid.uuid4())

    with open(f"{random_uuid}.json", "w") as f:
        f.write(json.dumps(preds, indent=4))
    acc = compute_vqa_accuracy(f"{random_uuid}.json", val_ques_path, val_ann_path)
    # delete the temporary file
    os.remove(f"{random_uuid}.json")
    return acc


@hydra.main(version_base=None, config_path="./configs", config_name="inference.yaml")
def main(cfg: DictConfig):
    """Hydra 入口：加载数据、LVLM interface、retriever，然后逐项评估。"""
    logger.info(f"{cfg=}")
    result_dir = os.path.join(
        cfg.result_dir,
        "icl_inference",
        cfg.infer_model.name,
        cfg.task.task_name,
        cfg.ex_name,
    )
    result_json_path = os.path.join(result_dir, "metrics.json")

    test_data_num = cfg.test_data_num
    index_data_num = cfg.index_data_num

    # ds["train"] 是 ICD 池，ds["validation"] 是要评估的 query 集。
    ds = load_ds(cfg)

    if index_data_num != -1:
        ds["train"] = ds["train"].select(
            random.sample(range(len(ds["train"])), index_data_num)
        )
    if test_data_num != -1:
        ds["validation"] = ds["validation"].select(range(test_data_num))

    # interface 统一封装 Flamingo/IDEFICS/Qwen 的 prompt 拼接、生成和概率接口。
    interface = init_interface(cfg, device=cfg.device)

    inferencer = ICLInferecer(
        interface=interface,
        train_ds=ds["train"],
        test_ds=ds["validation"],
        generation_kwargs=cfg.task.gen_args,
        other_save_field=cfg.task.other_save_field,
        num_workers=cfg.num_workers,
        num_proc=cfg.num_proc,
        batch_size=cfg.inference_bs,
        output_json_filepath=os.path.join(result_dir, "generation_metainfo"),
    )

    base_info = f"{str(datetime.datetime.now())}-{test_data_num=}-"

    # 通过开关控制本次要跑哪些 baseline/Lever-LM 方法。
    retriever_list = [
        ("ZeroShot", [0] if cfg.test_zero_shot else []),
        ("RandomRetriever", cfg.shot_num_list if cfg.test_random else []),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2t',
            cfg.shot_num_list if cfg.test_i2t else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-i2i',
            cfg.shot_num_list if cfg.test_i2i else [],
        ),
        (
            f'MMTopKRetriever-{cfg.mmtopk_clip_name.split("/")[-1]}-t2t',
            cfg.shot_num_list if cfg.test_t2t else [],
        ),
        (
            "LeverLMRetriever",
            cfg.shot_num_list if cfg.test_lever_lm else [],
        ),
    ]

    for retriever_name, shot_nums in retriever_list:
        if shot_nums:  # Only initialize and evaluate if shot_nums is not empty
            retriever_instance = init_retriever(retriever_name, ds, cfg)
            evaluate_retriever(
                retriever_name,
                inferencer,
                retriever_instance,
                ds,
                base_info,
                shot_nums,
                result_json_path,
                cfg,
            )


def shuffle_2d_list(matrix):
    new_matrix = [row.copy() for row in matrix]
    if len(new_matrix[0]) == 1:
        return new_matrix
    for i, row in enumerate(tqdm(new_matrix)):
        while row == matrix[i]:
            random.shuffle(row)
    return new_matrix


if __name__ == "__main__":
    load_dotenv()
    main()
