import json
import os
import random
from typing import Any

from loguru import logger


class BaseSampler:
    """采样anchor set并为每个anchor的Candidate Set采样。

    论文的数据构造阶段不会在整个训练集上穷举所有 ICD 组合，而是先为每个
    anchor/query 构造一个 candidate set，再在这个较小集合里用 scorer 和
    beam search 选择高分 ICD 序列。本类统一处理：

    - anchor set 的抽样和缓存；
    - 每个 anchor 对应 candidate set 的缓存；
    - 不同采样策略的公共参数。
    """

    def __init__(
        self,
        candidate_num,
        sampler_name,
        cache_dir,
        anchor_sample_num,
        index_ds_len,
        overwrite,
        dataset_name,
        other_info='',
        anchor_idx_list=None
    ) -> None:
        self.candidate_num = candidate_num
        self.sampler_name = sampler_name
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.overwrite = overwrite
        self.anchor_sample_num = anchor_sample_num
        self.anchor_set_cache_fn = os.path.join(
            cache_dir, f'{dataset_name}-anchor_sample_num:{self.anchor_sample_num}.json'
        )
        cache_fn = (
            f"{dataset_name}-{self.sampler_name}-"
            f"anchor_sample_num: {self.anchor_sample_num}:{self.candidate_num}"
            f"{'-' if other_info else '' + other_info}.json"
        )
        self.cache_file = os.path.join(self.cache_dir, cache_fn)
        self.index_ds_len = index_ds_len
        if anchor_idx_list is None:
            self.anchor_idx_list = self.sample_anchor_set()
        else:
            # 混合采样器会复用同一组 anchor，保证各子采样器对齐。
            self.anchor_idx_list = anchor_idx_list

    def __call__(self, train_ds) -> Any:
        total_data = {}
        total_data['anchor_set'] = self.anchor_idx_list
        data = self.load_cache_file()
        if data is not None:
            total_data['candidate_set'] = data
            return total_data
        data = self.sample(self.anchor_idx_list, train_ds)
        self.save_cache_file(data)
        total_data['candidate_set'] = data
        return total_data

    def sample(self, *args, **kwargs):
        # 子类实现具体策略：随机、图像相似度、文本相似度或混合。
        raise NotImplemented

    def load_cache_file(self):
        # candidate set 可能很贵，默认优先复用缓存，除非 overwrite=True。
        if not os.path.exists(self.cache_file) or self.overwrite:
            logger.info(
                f'the candidate set cache {self.cache_file} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            return
        else:
            logger.info(
                f'the candidate set cache {self.cache_file} exists, reloding...'
            )
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            data = {int(k): v for k, v in data.items()}
            return data

    def save_cache_file(self, data):
        # data 的结构通常是 {anchor_idx: [candidate_idx, ...]}。
        with open(self.cache_file, 'w') as f:
            logger.info(f'save the candidate data to {self.cache_file}')
            json.dump(data, f)

    def sample_anchor_set(self):
        """采样要用来构造训练监督的 anchor/query 集合。"""

        logger.info(self.anchor_set_cache_fn)
        if os.path.exists(self.anchor_set_cache_fn) and not self.overwrite:
            logger.info('the anchor_set_cache_filename exists, loding...')
            anchor_idx_list = json.load(open(self.anchor_set_cache_fn, 'r'))
        else:
            logger.info(
                f'the anchor set cache {self.anchor_set_cache_fn} not exists or set overwrite mode. (overwrite: {self.overwrite})'
            )
            anchor_idx_list = random.sample(#这一步就是抽样
                range(0, self.index_ds_len), self.anchor_sample_num
            )
            with open(self.anchor_set_cache_fn, 'w') as f:
                logger.info(f'save to {self.anchor_set_cache_fn}...')
                json.dump(anchor_idx_list, f)
        return anchor_idx_list
