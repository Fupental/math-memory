import random

from .base_sampler import BaseSampler


class RandSampler(BaseSampler):
    """随机Candidate set采样器。

    对每个 anchor，从训练集中随机挑选 candidate_num 个样本作为候选 ICD，
    并显式排除 anchor 自己。它是最简单的 candidate set 构造方式，也常作为
    混合采样器里的探索项。
    """

    def __init__(
        self,
        candidate_num,
        sampler_name,
        anchor_sample_num,
        index_ds_len,
        dataset_name,
        cache_dir,
        overwrite,
        anchor_idx_list=None,
    ):
        super().__init__(
            candidate_num=candidate_num,
            dataset_name=dataset_name,
            sampler_name=sampler_name,
            anchor_sample_num=anchor_sample_num,
            index_ds_len=index_ds_len,
            cache_dir=cache_dir,
            overwrite=overwrite,
            anchor_idx_list=anchor_idx_list,
        )

    def sample(self, anchor_set, train_ds):
        candidate_set_idx = {}
        for s_idx in anchor_set:
            # 重新采样直到候选集中不包含 query 自己，避免把待答样本当示例。
            random_candidate_set = random.sample(
                range(0, len(train_ds)), self.candidate_num
            )
            while s_idx in random_candidate_set:
                random_candidate_set = random.sample(
                    list(range(0, len(train_ds))), self.candidate_num
                )
            candidate_set_idx[s_idx] = random_candidate_set
        return candidate_set_idx
