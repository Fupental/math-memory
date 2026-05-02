import torch
from torch import nn


class BaseLeverLM(nn.Module):
    """Lever-LM 模型基类。

    论文中的 Lever-LM 本质上不是直接生成自然语言，而是生成一串 ICD
    （in-context demonstration）样本编号。因此子类都需要维护两类信息：

    1. 哪些 query 特征要编码进模型输入，例如图像特征、文本特征；
    2. 已经选出的 ICD 是否也需要用图像/文本特征重新编码，作为下一步生成的条件。
    """

    def __init__(
        self,
        adapter: bool = False,
        norm: bool = False,
        query_encoding_flag: list = None,
        icd_encoding_flag: list = None,
    ) -> None:
        super().__init__()
        if query_encoding_flag is None:
            query_encoding_flag = []

        self._adapter = adapter
        self._norm = norm
        self.query_encoding_flag = query_encoding_flag
        self.icd_encoding_flag = icd_encoding_flag

    def forward(*args, **kwargs):
        # 具体模型可以是 GPT2 或 LSTM，前向逻辑由子类实现。
        raise NotImplemented()

    def freeze_prefix(self, freeze_prefix_list):
        """按参数名前缀冻结模块。

        训练 Lever-LM 时常见做法是冻结 CLIP 编码器，只训练轻量 LM 或 adapter。
        配置文件里的 freeze_prefix_list 会传到这里，匹配到的参数不再更新。
        """

        if freeze_prefix_list is None:
            return
        for n, p in self.named_parameters():
            for prefix in freeze_prefix_list:
                if n.startswith(prefix):
                    p.requires_grad = False

    @torch.inference_mode()
    def generation(self, *args, **kwargs):
        # 推理阶段逐步生成 ICD index 序列，具体策略由子类实现。
        raise NotImplemented()
