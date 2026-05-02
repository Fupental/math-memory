from typing import Dict, List

import datasets
import torch
from torch.utils.data import Dataset


class BaseLeverLMDataset(Dataset):
    """Lever-LM 训练样本的基础 Dataset。

    `generate_data.py` 输出的监督数据里，每条高分序列都是：

    [icd_1, icd_2, ..., query_id]

    前面的 id 是被选作 in-context demonstration 的样本，最后一个 id 是
    当前 anchor/query。训练时这里会把它转换成：

    [BOS, QUERY, icd_1, icd_2, ..., EOS]

    其中 query 的图像/文本内容会通过 `query_input` 交给模型，在 QUERY 位置
    注入特征；ICD 本身的图文内容由子类 LeverLMDataset 继续补充。
    """

    def __init__(
        self,
        data: Dict,
        index_ds: datasets.Dataset,
        eos_token_id: int,
        bos_token_id: int,
        query_token_id: int,
        query_image_field: str = None,
        query_text_field: str = None,
        threshold: float = 0.0,
        reverse_seq: bool = False,
    ):
        super().__init__()

        self.threshold = threshold
        self.reverse_seq = reverse_seq

        self.icd_idx_seq_list = []
        self.x_id_list = []

        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.query_token_id = query_token_id

        self.query_image_field = query_image_field
        self.query_text_field = query_text_field

        # 只保留 scorer 打分高于 threshold 的 ICD 序列，降低噪声监督。
        icd_seq_list = data["icd_seq"]
        icd_score_list = data["icd_score"]

        self.index_ds = index_ds
        for icd_seq, icd_score in zip(icd_seq_list, icd_score_list):
            if icd_score < self.threshold:
                continue
            idx_list = icd_seq[:-1]
            if self.reverse_seq:
                idx_list = reversed(idx_list)
            self.icd_idx_seq_list.append(list(idx_list))
            # x_id_list 保存 query/anchor 的样本 id，用来读取 query 图文特征。
            self.x_id_list.append(icd_seq[-1])

    def __getitem__(self, index):
        icd_seq_idx = self.icd_idx_seq_list[index]
        add_sp_token_seq_idx = (
            [self.bos_token_id, self.query_token_id] + icd_seq_idx + [self.eos_token_id]
        )

        test_sample_id = self.x_id_list[index]
        query_input = {}
        # 这里仍保持原始字段，真正 tokenization/图像预处理在 collate_fn 中完成。
        if self.query_image_field:
            img = self.index_ds[test_sample_id][self.query_image_field]
            query_input["images"] = img
        if self.query_text_field:
            text = self.index_ds[test_sample_id][self.query_text_field]
            query_input["text"] = text
        return {
            "query_input": query_input,
            "icd_seq_idx": torch.tensor(add_sp_token_seq_idx, dtype=torch.long),
        }

    def __len__(self):
        return len(self.x_id_list)
