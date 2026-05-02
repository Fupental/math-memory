from typing import List, Optional

import datasets
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ProcessorMixin

from open_mmicl.retriever.base_retriever import BaseRetriever


class LeverLMRetriever(BaseRetriever):
    """用训练好的 Lever-LM 做 ICL 示例检索。

    open_mmicl 的推理流程需要一个 retriever 返回每条测试样本对应的 ICD index
    列表。传统 retriever 可能是随机或相似度 TopK；本类则把测试样本 query
    输入 Lever-LM，让模型自回归生成 ICD id 序列，再交给 LVLM 作为上下文示例。
    """

    def __init__(
        self,
        index_ds: datasets.Dataset,
        test_ds: datasets.Dataset,
        lever_lm: torch.nn.Module,
        processor: ProcessorMixin,
        query_image_field: Optional[str] = None,
        query_text_field: Optional[str] = None,
        icd_image_field: Optional[str] = None,
        icd_text_field: Optional[str] = None,
        device: str = "cpu",
        infer_batch_size: int = 1,
        infer_num_workers: int = 0,
        reverse_seq: bool = False,
    ):
        """Initialize the LeverLMRetriever.

        Args:
            index_ds (datasets.Dataset): The dataset used for creating the index.
            test_ds (datasets.Dataset): The dataset used for testing.
            lever_lm (torch.nn.Module): ICD Language Model used for retrieval.
            processor (ProcessorMixin): The processor for preparing input data.
            query_image_field (Optional[str]): The field name for query images in the dataset.
            query_text_field (Optional[str]): The field name for query text in the dataset.
            icd_image_field (Optional[str]): The field name for images in the ICD dataset.
            icd_text_field (Optional[str]): The field name for text in the ICD dataset.
            device (str): The computing device ('cpu' or 'cuda').
            infer_batch_size (int): The batch size for inference.
            infer_num_workers (int): The number of workers for data loading during inference.
        """
        super().__init__(index_ds, test_ds)
        self.lever_lm = lever_lm
        self.processor = processor
        self.device = device
        self.query_image_field = query_image_field
        self.query_text_field = query_text_field
        self.infer_batch_size = infer_batch_size
        self.infer_num_workers = infer_num_workers
        self.icd_text_field = icd_text_field
        self.icd_image_field = icd_image_field
        self.reverse_seq = reverse_seq

    def retrieve(self, ice_num) -> List[List[int]]:
        """Retrieve indices from the index dataset using the LeverLM model.

        Args:
            ice_num (int): The number of indices to retrieve for each test case.

        Returns:
            List[List[int]]: A list of lists containing the retrieved indices. Each sublist corresponds to a test case.
        """
        # open_mmicl 统一调用 retrieve；内部实际走 Lever-LM generation。
        return self.lever_lm_generation(ice_num)

    @torch.inference_mode()
    def lever_lm_generation(self, ice_num: int) -> List[List[int]]:
        """Generate indices using the LeverLM model.

        Args:
            ice_num (int): The number of indices to generate for each test case.

        Returns:
            List[List[int]]: A list of lists containing the generated indices. Each sublist corresponds to a test case.
        """
        self.lever_lm = self.lever_lm.to(self.device)
        self.lever_lm.eval()
        icd_idx_list = []
        # 特殊 token id 必须与训练阶段 LeverLMDataset 的约定一致。
        bos_token_id = len(self.index_ds) + 1
        query_token_id = len(self.index_ds) + 2

        test_ds_ = self.test_ds.map()

        def prepare(examples):
            # 只预处理 query 本身；候选 ICD 不由相似度检索给出，而由 Lever-LM 生成。
            images = texts = None
            if self.query_image_field:
                images = [i for i in examples[self.query_image_field]]
            if self.query_text_field:
                texts = [i for i in examples[self.query_text_field]]

            data_dict = self.processor(
                images=images,
                text=texts,
                padding=True,
                return_tensors="pt",
            )
            return data_dict

        test_ds_.set_transform(prepare)
        dataloader = DataLoader(
            test_ds_,
            batch_size=self.infer_batch_size,
            shuffle=False,
            num_workers=self.infer_num_workers,
        )

        for query_input in tqdm(dataloader, ncols=100):
            query_input = {k: v.to(self.device) for k, v in query_input.items()}
            bs = len(query_input[list(query_input.keys())[0]])
            # 每条 query 都从 [BOS, QUERY] 前缀开始生成 ice_num 个 ICD id。
            init_icd_idx = torch.tensor(
                [[bos_token_id, query_token_id] for _ in range(bs)]
            ).to(self.device)
            res = self.lever_lm.generation(
                query_input=query_input,
                init_icd_idx=init_icd_idx,
                shot_num=ice_num,
                index_ds=self.index_ds,
                processor=self.processor,
                icd_image_field=self.icd_image_field,
                icd_text_field=self.icd_text_field,
                device=self.device,
            )
            # generation 返回包含特殊 token 的完整序列，推理只需要 ICD id。
            res = [r[2 : 2 + ice_num] for r in res]
            icd_idx_list.extend(res)
        if self.reverse_seq:
            # 某些配置会把生成顺序和最终 prompt 中 ICD 拼接顺序反过来。
            icd_idx_list = [list(reversed(s)) for s in icd_idx_list]

        return icd_idx_list
