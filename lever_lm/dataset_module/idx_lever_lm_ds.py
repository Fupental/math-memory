from typing import List

import datasets

from .base_lever_lm_ds import BaseLeverLMDataset


class IdxLeverLMDataset(BaseLeverLMDataset):
    """只使用 ICD index 序列的轻量 Dataset 变体。

    该类不额外读取 ICD 图像或文本，而是把 `icd_seq_idx` 同时作为 `icd_input`。
    在当前主链路中更常用的是 LeverLMDataset；这里保留给只建模编号序列的配置。
    """

    def __init__(
        self,
        data_list: List,
        index_ds: datasets.Dataset,
        clip_processor_name: str,
        index_set_length: int,
        image_field: str,
    ):
        eos_token_id = index_set_length
        bos_token_id = index_set_length + 1
        query_token_id = index_set_length + 2
        super().__init__(
            data_list,
            index_ds,
            clip_processor_name,
            eos_token_id,
            bos_token_id,
            query_token_id,
            image_field,
        )

    def __getitem__(self, index):
        data_dict = super().__getitem__(index)
        # 让后续训练接口仍能拿到 icd_input 字段，但内容只是 token id 序列。
        data_dict["icd_input"] = data_dict["icd_seq_idx"]
        return data_dict

    def __len__(self):
        return len(self.x_id_list)
