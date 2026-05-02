import json
import os
from typing import List

import more_itertools
import torch
from datasets import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)


def init_interface(cfg, **kwargs):
    """根据 Hydra 配置初始化真正负责回答问题的 LVLM/LLM 接口。

    Lever-LM 自己只负责选择 ICD；InfoScore 打分、数据生成和最终 ICL 推理都
    还需要调用 Flamingo、IDEFICS、Qwen 等基础模型。这里把不同模型封装成
    open_mmicl 统一接口。
    """

    if "flamingo" in cfg.infer_model.name:
        from open_mmicl.interface import FlamingoInterface

        return FlamingoInterface(
            lang_encoder_path=cfg.infer_model.lang_encoder_path,
            tokenizer_path=cfg.infer_model.tokenizer_path,
            flamingo_checkpoint_dir=cfg.infer_model.flamingo_checkpoint_dir,
            cross_attn_every_n_layers=cfg.infer_model.cross_attn_every_n_layers,
            hf_root=cfg.infer_model.hf_root,
            precision=cfg.precision,
            device=kwargs["device"],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            icd_join_char=cfg.infer_model.icd_join_char,
            load_from_local=cfg.infer_model.load_from_local,
            instruction=cfg.task.instruction,
            init_device=cfg.infer_model.init_device,
            image_field=cfg.task.image_field,
            label_field=cfg.task.output_column,
        )
    elif "idefics" in cfg.infer_model.name:
        from open_mmicl.interface import IDEFICSInterface

        return IDEFICSInterface(
            hf_root=cfg.infer_model.hf_root,
            load_from_local=cfg.infer_model.load_from_local,
            precision=cfg.precision,
            device=kwargs["device"],
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            instruction=cfg.task.instruction,
            icd_join_char=cfg.infer_model.icd_join_char,
            image_field=cfg.task.image_field,
            label_field=cfg.task.output_column,
        )
    elif "Qwen" in cfg.infer_model.name:
        from open_mmicl.interface import LLMInterface

        return LLMInterface(
            cfg.infer_model.model_name,
            cfg.infer_model.model_name,
            precision=cfg.precision,
            input_ids_field_name="input_ids",
            prompt_template=cfg.task.template,
            column_token_map=cfg.task.column_token_map,
            instruction=cfg.task.instruction,
            icd_join_char=cfg.infer_model.icd_join_char,
            label_field=cfg.task.output_column,
            device=kwargs["device"],
        )

    else:
        raise ValueError(
            "infer_model name error, now only support ['flamingo, idefics']"
        )


def recall_sim_feature(test_vec, train_vec, top_k=200):
    """用 FAISS 内积检索相似样本。

    图像/文本相似度采样器都先把特征归一化，因此内积等价于 cosine similarity。
    返回值中的 index 就是候选 ICD 的 dataset id。
    """

    import faiss

    logger.info(f"embedding shape: {train_vec.shape}")
    dim = train_vec.shape[-1]
    index_feat = faiss.IndexFlatIP(dim)
    index_feat.add(train_vec)
    dist, index = index_feat.search(test_vec, top_k)
    return dist, index


@torch.inference_mode()
def encode_text(
    train_ds,
    data_key,
    device,
    model_type="openai/clip-vit-large-patch14",
    batch_size=128,
):
    """批量编码文本字段，供 TextSimSampler 构造 candidate set。"""

    model = CLIPTextModelWithProjection.from_pretrained(model_type).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    final_text_feature = []

    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        text_list = [i[data_key] for i in batch]
        inputs = tokenizer(text_list, padding=True, return_tensors="pt").to(device)
        text_feature = model(**inputs).text_embeds
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        final_text_feature.append(text_feature)

    final_text_feature = torch.cat(final_text_feature, dim=0)
    return final_text_feature.detach().cpu().numpy()


@torch.inference_mode()
def encode_image(
    train_ds,
    data_key,
    device,
    model_type="openai/clip-vit-large-patch14",
    batch_size=128,
):
    """批量编码图像字段，供 ImgSimSampler 构造 candidate set。"""

    model = CLIPVisionModelWithProjection.from_pretrained(model_type).to(device)
    processor = AutoProcessor.from_pretrained(model_type)
    model.eval()

    final_image_feature = []
    for batch in more_itertools.chunked(tqdm(train_ds), batch_size):
        images = [i[data_key] for i in batch]
        inputs = processor(images=images, return_tensors="pt").to(device)
        image_feature = model(**inputs).image_embeds
        image_feature /= image_feature.norm(dim=-1, keepdim=True)
        final_image_feature.append(image_feature)

    final_image_feature = torch.cat(final_image_feature, dim=0)
    return final_image_feature.detach().cpu().numpy()


def data_split(generated_data, train_ratio):
    """把 generate_data 的结果按 query id 划分为训练/验证监督。

    generated_data 是 beam search 产生的多组 ICD 序列和分数。划分时按最后的
    query/anchor id 做切分，避免同一个 query 的不同 ICD 序列同时出现在 train
    和 validation 中。
    """

    # 获得有多少条 query/anchor 数据。
    test_dataset_id_set = {
        v[-1] for d in generated_data for v in generated_data[d]["id_list"]
    }
    test_dataset_len = len(test_dataset_id_set)

    # 计算多少 query 用于训练，剩下部分用于监督 val loss。
    train_data_len = int(train_ratio * test_dataset_len)
    train_idx_set = set(sorted(list(test_dataset_id_set))[:train_data_len])
    val_idx_set = test_dataset_id_set - train_idx_set

    train_data_list = list()
    val_data_list = list()
    train_data_score = list()
    val_data_score = list()
    for d in generated_data:
        for i in range(len(generated_data[d]["id_list"])):
            query_idx = generated_data[d]["id_list"][i][-1]
            if int(query_idx) in train_idx_set:
                train_data_list.append(generated_data[d]["id_list"][i])
                train_data_score.append(generated_data[d]["score_list"][i])
            elif int(query_idx) in val_idx_set:
                val_data_list.append(generated_data[d]["id_list"][i])
                val_data_score.append(generated_data[d]["score_list"][i])
            else:
                raise ValueError()

    print(f"the train size {len(train_data_list)}, the test size {len(val_data_list)}")

    train_data = {
        "icd_seq": train_data_list,
        "icd_score": train_data_score,
    }
    val_data = {
        "icd_seq": val_data_list,
        "icd_score": val_data_score,
    }
    return train_data, val_data


def collate_fn(batch, processor: CLIPProcessor):
    """Lever-LM 训练的 batch 拼接函数。

    Dataset 返回的是原始图像/文本对象和 ICD token id。这里统一调用 CLIPProcessor
    得到模型输入张量，并把 ICD 特征 reshape 成 [batch, icd_num, ...]，使模型能
    对每条 query 的每个 ICD 前缀位置加对应特征。
    """

    bs = len(batch)
    collate_dict = {
        "icd_seq_idx": torch.stack([item["icd_seq_idx"] for item in batch]),
    }
    query_input = [d["query_input"] for d in batch]

    query_text_input = (
        [q["text"] for q in query_input] if "text" in query_input[0] else None
    )
    query_image_input = (
        [q["images"] for q in query_input] if "images" in query_input[0] else None
    )
    if query_text_input or query_image_input:
        query_input = processor(
            images=query_image_input,
            text=query_text_input,
            padding=True,
            return_tensors="pt",
        )
        collate_dict["query_input"] = query_input

    icd_input_list = [d["icd_input"] for d in batch]
    icd_image_input = icd_text_input = None
    if "text" in icd_input_list[0]:
        icd_num = len(icd_input_list[0]["text"])
        icd_text_input = [i["text"] for i in icd_input_list]
        icd_text_input = [i for icd_text in icd_text_input for i in icd_text]
    if "images" in icd_input_list[0]:
        icd_num = len(icd_input_list[0]["images"])
        icd_image_input = [i["images"] for i in icd_input_list]
        icd_image_input = [i for icd_image in icd_image_input for i in icd_image]

    if icd_image_input or icd_text_input:
        icd_input = processor(
            images=icd_image_input,
            text=icd_text_input,
            padding=True,
            return_tensors="pt",
        )
        if "input_ids" in icd_input:
            icd_input["input_ids"] = icd_input["input_ids"].view(bs, icd_num, -1)
            icd_input["attention_mask"] = icd_input["attention_mask"].view(
                bs, icd_num, -1
            )
        if "pixel_values" in icd_input:
            icd_input["pixel_values"] = icd_input["pixel_values"].view(
                bs, icd_num, *icd_input["pixel_values"].shape[1:]
            )
        collate_dict["icd_input"] = icd_input
    return collate_dict


def load_feature_cache(cfg, cache_path, encoding_method, train_ds, data_key):
    """读取或构造 CLIP 特征缓存。"""

    if os.path.exists(cache_path):
        features = torch.load(cache_path)
    else:
        features = encoding_method(
            train_ds,
            data_key,
            cfg.device,
            cfg.sim_model_type,
            cfg.candidate_set_encode_bs,
        )
        torch.save(features, cache_path)
    return features


def beam_filter(score_list, data_id_list, beam_size):
    """beam search 的 top-k 过滤工具。

    `score_list` 是每条候选 ICD 序列的累计分数，`data_id_list` 是对应的 id 序列。
    返回分数最高的 beam_size 条，供下一轮继续扩展。
    """

    score_list = torch.tensor(score_list)
    score_value, indices = torch.topk(score_list, beam_size)
    return score_value.tolist(), [data_id_list[idx] for idx in indices]
