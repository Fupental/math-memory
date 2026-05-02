import os

import torch
from loguru import logger

from lever_lm.utils import encode_text, recall_sim_feature

from .base_sampler import BaseSampler


class TextSimSampler(BaseSampler):
    """基于文本相似度的候选 ICD 采样器。

    与 ImgSimSampler 对称，只是编码字段换成文本。caption/VQA 等任务中，
    它可以用问题、caption 或其他文本字段召回语义相近的 ICD 候选。
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
        text_field_name,
        device,
        candidate_set_encode_bs,
        anchor_idx_list=None,
    ):
        super().__init__(
            candidate_num=candidate_num,
            sampler_name=sampler_name,
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            other_info=feature_cache_filename.replace("openai/", ""),
            anchor_idx_list=anchor_idx_list,
        )
        self.clip_model_name = clip_model_name
        self.feature_cache_filename = feature_cache_filename.replace("openai/", "")
        self.feature_cache = os.path.join(self.cache_dir, self.feature_cache_filename)
        self.text_field_name = text_field_name
        self.device = device
        self.bs = candidate_set_encode_bs

    @torch.inference_mode()
    def sample(self, anchor_set, train_ds):
        # 文本 CLIP 特征同样会缓存，减少重复构造 candidate set 的成本。
        if os.path.exists(self.feature_cache):
            logger.info(f"feature cache {self.feature_cache} exists, loding...")
            features = torch.load(self.feature_cache)
        else:
            features = encode_text(
                train_ds,
                self.text_field_name,
                self.device,
                self.clip_model_name,
                self.bs,
            )
            logger.info(f"saving the features cache in {self.feature_cache} ...")
            torch.save(features, self.feature_cache)
        test_feature = features[anchor_set]
        # top-1 通常是 anchor 自身，因此取 candidate_num+1 后跳过第一列。
        _, sim_sample_idx = recall_sim_feature(
            test_feature, features, top_k=self.candidate_num + 1
        )
        sim_sample_idx = sim_sample_idx[:, 1:].tolist()
        candidate_set_idx = {idx: cand for idx, cand in zip(anchor_set, sim_sample_idx)}
        return candidate_set_idx
