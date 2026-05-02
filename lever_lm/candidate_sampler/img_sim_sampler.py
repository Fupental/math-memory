import os

import torch
from loguru import logger

from lever_lm.utils import encode_image, recall_sim_feature

from .base_sampler import BaseSampler


class ImgSimSampler(BaseSampler):
    """基于图像相似度的候选 ICD 采样器。

    先用 CLIP image encoder 给训练集所有图像编码，再用 FAISS 内积检索每个
    anchor 最相似的图像样本。由于最相似的第一个通常是 anchor 自己，代码会
    取 top_k=candidate_num+1 后丢掉第一项。
    """

    def __init__(
        self,
        candidate_num,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        cache_dir,
        dataset_name,
        overwrite,
        clip_model_name,
        feature_cache_filename,
        img_field_name,
        device,
        candidate_set_encode_bs,
        anchor_idx_list=None,
    ):
        super().__init__(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            other_info=feature_cache_filename.replace("openai/", ""),
            anchor_idx_list=anchor_idx_list,
        )
        self.clip_model_name = clip_model_name
        self.feature_cache_filename = feature_cache_filename.replace("openai/", "")
        self.feature_cache = os.path.join(self.cache_dir, self.feature_cache_filename)
        self.img_field_name = img_field_name
        self.device = device
        self.bs = candidate_set_encode_bs

    def sample(self, anchor_set, train_ds):
        if self.img_field_name not in train_ds.keys():
            raise ValueError(
                f'dataset\'s keys {train_ds.keys()} do not include "{self.img_field_name}".'
            )
        # CLIP 特征缓存可以避免每次 generate_data 都重新编码整个训练集。
        if os.path.exists(self.feature_cache):
            logger.info(f"feature cache {self.feature_cache} exists, loding...")
            features = torch.load(self.feature_cache)
        else:
            features = encode_image(
                train_ds,
                self.img_field_name,
                self.device,
                self.clip_model_name,
                self.bs,
            )
            logger.info(f"saving the features cache in {self.feature_cache} ...")
            torch.save(features, self.feature_cache)
        test_feature = features[anchor_set]
        # recall_sim_feature 返回相似度和样本 id；这里仅保留候选 ICD id。
        _, sim_sample_idx = recall_sim_feature(
            test_feature, features, top_k=self.candidate_num + 1
        )
        sim_sample_idx = sim_sample_idx[:, 1:].tolist()
        candidate_set_idx = {idx: cand for idx, cand in zip(anchor_set, sim_sample_idx)}
        return candidate_set_idx
