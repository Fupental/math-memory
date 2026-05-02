import torch
from torch import nn
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    GPT2Config,
    GPT2LMHeadModel,
)

from .base_lever_lm import BaseLeverLM


class GPT2LeverLM(BaseLeverLM):
    """基于 GPT-2 架构的 Lever-LM。

    这里的“词表”不是普通文本 token，而是：

    - 0 到 index_ds_size-1：训练/索引数据集中每个样本的 ICD 编号；
    - index_ds_size 起的 3 个 id：EOS、BOS、QUERY 等特殊位置 token。

    模型要学习的是在给定 query 特征和已有 ICD 前缀时，下一个最合适的 ICD id。
    """

    def __init__(
        self,
        lm_config,
        index_ds_size: int,
        clip_name: str = "openai/clip-vit-base-patch32",
        adapter: bool = False,
        norm: bool = False,
        freeze_prefix_list: list = None,
        query_encoding_flag: list = None,
        icd_encoding_flag: list = None,
    ):
        super().__init__(
            adapter,
            norm,
            query_encoding_flag,
            icd_encoding_flag,
        )
        # 每个可被选作 ICD 的样本占一个 token，再额外预留特殊 token。
        vocab_size = index_ds_size + 3
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=lm_config["n_embd"],
            n_head=lm_config["n_head"],
            n_layer=lm_config["n_layer"],
            eos_token_id=vocab_size,
            bos_token_id=vocab_size + 1,
        )
        self.lm_model = GPT2LMHeadModel(config)

        # 只有配置中需要图像/文本特征时才加载对应的 CLIP 编码器。
        need_encoder = set(self.query_encoding_flag + self.icd_encoding_flag)
        if "image" in need_encoder:
            self.img_model = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        if "text" in need_encoder:
            self.sen_model = CLIPTextModelWithProjection.from_pretrained(clip_name)

        # adapter 用来把 CLIP projection_dim 映射到 LM embedding 维度。
        # 这样可以冻结大编码器，只训练较小的投影层和 LM。
        if self._adapter:
            if "image" in need_encoder:
                self.img_adapter = nn.Sequential(
                    nn.Linear(
                        self.img_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
            if "text" in need_encoder:
                self.sen_adapter = nn.Sequential(
                    nn.Linear(
                        self.sen_model.config.projection_dim, lm_config.n_embd * 4
                    ),
                    nn.ReLU(),
                    nn.Linear(lm_config.n_embd * 4, lm_config.n_embd),
                )
        self.freeze_prefix(freeze_prefix_list)

    def forward(self, query_input, icd_input, icd_seq_idx):
        """训练/推理共用的前向过程。

        `icd_seq_idx` 是形如 [BOS, QUERY, icd_1, ..., EOS] 的 token 序列。
        代码先取 GPT2 token embedding，再把 query 的图文特征加到 QUERY 位置；
        如果已经有 ICD 前缀，则把每个 ICD 的图文特征加到对应 ICD token 位置。
        """

        text_embeds = image_embeds = None
        inputs_embeds = self.lm_model.transformer.wte(icd_seq_idx)

        # QUERY token 是一个占位符，真正的 query 条件通过 CLIP 特征注入。
        if "image" in self.query_encoding_flag:
            image_embeds = self.img_model(query_input["pixel_values"])["image_embeds"]
            if self._adapter:
                image_embeds = self.img_adapter(image_embeds)
            if self._norm:
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += image_embeds
        if "text" in self.query_encoding_flag:
            text_embeds = self.sen_model(
                input_ids=query_input["input_ids"],
                attention_mask=query_input["attention_mask"],
            )["text_embeds"]
            if self._adapter:
                text_embeds = self.sen_adapter(text_embeds)
            if self._norm:
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            inputs_embeds[:, 1] += text_embeds

        # 推理第一步还没有选择任何 ICD，此时只依赖 BOS/QUERY 前缀。
        if icd_input is None:
            lm_output = self.lm_model(inputs_embeds=inputs_embeds)
            return lm_output

        # 训练和多步生成时，已选 ICD 的图文特征也作为条件注入。
        if "text" in self.icd_encoding_flag:
            bs, icd_num, icd_seq_len = icd_input["input_ids"].shape
            icd_input["input_ids"] = icd_input["input_ids"].view(-1, icd_seq_len)
            icd_input["attention_mask"] = icd_input["attention_mask"].view(
                -1, icd_seq_len
            )

            icd_text_features = self.sen_model(
                input_ids=icd_input["input_ids"],
                attention_mask=icd_input["attention_mask"],
            )["text_embeds"]
            if self._adapter:
                icd_text_features = self.sen_adapter(icd_text_features)
            if self._norm:
                icd_text_features = icd_text_features / icd_text_features.norm(
                    dim=-1, keepdim=True
                )
            icd_text_features = icd_text_features.view(bs, icd_num, -1)
            inputs_embeds[:, 2 : 2 + icd_num] += icd_text_features
        if "image" in self.icd_encoding_flag:
            bs, icd_num = icd_input["pixel_values"].shape[:2]
            img_shape = icd_input["pixel_values"].shape[-3:]
            icd_input["pixel_values"] = icd_input["pixel_values"].view(-1, *img_shape)
            icd_img_features = self.img_model(icd_input["pixel_values"])["image_embeds"]

            if self._adapter:
                icd_img_features = self.img_adapter(icd_img_features)
            if self._norm:
                icd_img_features = icd_img_features / icd_img_features.norm(
                    dim=-1, keepdim=True
                )
            icd_img_features = icd_img_features.view(bs, icd_num, -1)
            inputs_embeds[:, 2 : 2 + icd_num] += icd_img_features

        # labels 使用完整 ICD token 序列，GPT2LMHeadModel 内部会做 next-token loss。
        output = self.lm_model(inputs_embeds=inputs_embeds, labels=icd_seq_idx)
        return output

    @torch.inference_mode()
    def generation(
        self,
        query_input,
        init_icd_idx,
        shot_num,
        index_ds,
        processor,
        device,
        icd_image_field,
        icd_text_field,
    ):
        """
        Generate for one batch.

        从 [BOS, QUERY] 开始，每一步贪心选择一个 ICD id，并把新 ICD 的图文特征
        加入下一步输入。生成出的前两个 token 是特殊 token，调用方通常会截掉。
        """
        icd_input = None
        icd_seq_idx = init_icd_idx
        sp_token_begin = len(index_ds)
        bs = len(icd_seq_idx)

        for s_n in range(shot_num):
            out = self.forward(query_input, icd_input, icd_seq_idx)["logits"][:, -1, :]
            # 不能生成 EOS/BOS/QUERY 等特殊 token，也不能重复选择已有 ICD。
            out[:, sp_token_begin:] = -torch.inf
            for icd_idx in icd_seq_idx:
                out[:, icd_idx] = -torch.inf

            next_token_idx = torch.softmax(out, dim=-1).argmax(dim=-1)  # bs, 1

            icd_seq_idx = torch.cat(
                [icd_seq_idx, next_token_idx.unsqueeze(dim=1)], dim=1
            )
            icd_text_list = icd_img_list = None
            if "text" in self.icd_encoding_flag:
                icd_text_list = [
                    index_ds[idx][icd_text_field]
                    for i in range(bs)
                    for idx in icd_seq_idx.tolist()[i][2:]
                ]
            if "image" in self.icd_encoding_flag:
                icd_img_list = [
                    index_ds[idx][icd_image_field]
                    for i in range(bs)
                    for idx in icd_seq_idx.tolist()[i][2:]
                ]
            if icd_text_list or icd_img_list:
                # processor 先处理成扁平 batch，再 reshape 回 [bs, 已选 ICD 数, ...]。
                flatten_icd_input = processor(
                    text=icd_text_list,
                    images=icd_img_list,
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                icd_input = {}
                for k in flatten_icd_input:
                    other_dim = flatten_icd_input[k].shape[1:]
                    icd_input[k] = flatten_icd_input[k].view(bs, s_n + 1, *other_dim)
        return icd_seq_idx.detach().cpu().tolist()
