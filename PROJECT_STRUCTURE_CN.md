# Lever-LM 项目中文导读

本文档面向阅读论文和源码的人，解释 Lever-LM 的论文概念如何落到代码里，以及每个主要目录和文件的作用。

## 1. 核心思想

Lever-LM 要解决的问题是：给一个强大的 LVLM/LLM 做 in-context learning 时，哪些示例应该放进 prompt，顺序又应该怎么排。

关键术语对应如下：

- LVLM：真正执行 caption、VQA、分类等任务的大模型，例如 Flamingo、IDEFICS、Qwen。代码里由 `open_mmicl/interface/` 下的接口封装。
- ICL：in-context learning，把若干示例放进 prompt，让基础模型参考示例回答当前 query。
- ICD：in-context demonstration，即一个被放进 prompt 的示例样本。代码里通常用训练集样本的 index 表示一个 ICD。
- Query / Anchor：当前要构造监督或推理的目标样本。`generate_data.py` 里称为 anchor；推理时就是 test sample。
- Candidate set：每个 anchor 周围可选的 ICD 候选集合，由 `lever_lm/candidate_sampler/` 生成。
- Scorer：评估“加入某个 ICD 后基础模型表现提升多少”的打分器。当前主链路支持 InfoScore 和 caption 任务中的 CIDEr 增益。
- Lever-LM：一个轻量语言模型，但它生成的不是文本 token，而是 ICD 样本 index 序列。
- Retriever：推理时为每个测试样本返回 ICD index 列表的组件。Lever-LM 训练好后会作为一种 retriever 使用。

一句话概括：先用大模型和 scorer 离线搜索出高质量 ICD 序列，再训练小模型 Lever-LM 学会“给定 query 生成 ICD index 序列”，最后推理时用 Lever-LM 替代昂贵搜索。

## 2. 三阶段数据流

### 阶段一：构造训练数据

入口：`generate_data.py`

数据流：

```text
原始训练集
  -> 抽样 anchor/query
  -> 为每个 anchor 构造 candidate set
  -> 用 LVLM + scorer 评估候选 ICD
  -> beam search 得到高分 ICD 序列
  -> 保存 generated_data JSON
```

生成数据中，一条 ICD 序列通常是：

```text
[icd_1, icd_2, ..., query_id]
```

最后一个 id 是 anchor/query，前面是给它选出来的 ICD。训练阶段会再补上 BOS、QUERY、EOS 等特殊 token。

相关文件：

- `generate_data.py`：主入口，负责加载数据、初始化 LVLM 接口、调用 sampler、并行构造监督数据。
- `utils.py`：提供 `get_info_score`、`get_cider_score`、`load_ds` 等构造数据时用到的公共逻辑。
- `configs/generate_data.yaml`：默认数据构造配置。
- `configs/generate_data_random.yaml`：随机构造数据的配置变体。
- `scripts/generate_data.sh`：调用数据构造入口的脚本包装。

### 阶段二：训练 Lever-LM

入口：`train.py`

数据流：

```text
generated_data JSON
  -> data_split 按 query id 分 train/val
  -> LeverLMDataset 转成训练样本
  -> collate_fn 编码 query/ICD 图文特征
  -> GPT2LeverLM 或 LSTMLeverLM 学习预测 ICD token
  -> 保存 checkpoint
```

Lever-LM 的词表不是普通文本词表。它把索引数据集中每个样本都当作一个可生成 token：

```text
0 ... len(index_ds)-1     -> 可选 ICD 样本 id
len(index_ds)             -> EOS
len(index_ds)+1           -> BOS
len(index_ds)+2           -> QUERY
```

训练序列形如：

```text
[BOS, QUERY, icd_1, icd_2, ..., EOS]
```

模型会把 query 的图像/文本 CLIP 特征加到 QUERY token embedding 上；如果配置启用 ICD 图像/文本特征，也会把每个 ICD 的特征加到对应 ICD token embedding 上。

相关文件：

- `train.py`：Lightning 训练入口，定义 `LeverLM` 和 `ICDSeqDataModule`。
- `lever_lm/dataset_module/base_lever_lm_ds.py`：把 generated data 中的 ICD 序列转成 token id 序列。
- `lever_lm/dataset_module/lever_lm_ds.py`：进一步读取 ICD 的图像/文本内容。
- `lever_lm/utils.py`：提供 `collate_fn`、CLIP 编码、相似度检索等工具。
- `lever_lm/models/gpt2_lever_lm.py`：GPT-2 版本 Lever-LM。
- `lever_lm/models/lstm_lever_lm.py`：LSTM 版本 Lever-LM。
- `configs/train.yaml`：训练主配置。
- `configs/train/*.yaml`：不同 query/ICD 特征组合的配置入口。
- `configs/train/lever_lm/*.yaml`：具体模型结构配置。
- `configs/train/lever_lm_ds/*.yaml`：具体 Dataset 字段配置。
- `scripts/train_lever_lm.sh`：训练脚本包装。

### 阶段三：ICL 推理评估

入口：`icl_inference.py`

数据流：

```text
test query
  -> retriever 返回 ICD index 列表
  -> open_mmicl interface 拼 prompt
  -> LVLM/LLM 生成答案
  -> caption/VQA/SST2 指标评估
```

可比较的 retriever 包括：

- ZeroShot：不使用 ICD。
- Random：随机选 ICD。
- MMTopK：按图像/文本相似度检索 ICD。
- LeverLM：加载训练好的 Lever-LM，自回归生成 ICD index 序列。

相关文件：

- `icl_inference.py`：推理评估入口，控制不同 retriever 的实验。
- `open_mmicl/retriever/lever_lm_retriever.py`：把 Lever-LM 包装成 open_mmicl retriever。
- `open_mmicl/retriever/rand_retriever.py`：随机检索。
- `open_mmicl/retriever/mm_topk_retriever.py`：多模态相似度 TopK 检索。
- `open_mmicl/retriever/zero_retriever.py`：零样本基线。
- `open_mmicl/interface/`：不同基础模型的 prompt 构造、输入准备和生成封装。
- `open_mmicl/metrics/`：caption CIDEr、VQA accuracy 等指标。
- `configs/inference.yaml`：推理主配置。
- `scripts/inference.sh`：推理脚本包装。

## 3. 目录说明

### `configs/`

Hydra 配置目录，控制数据集、模型、采样器、训练和推理。

- `configs/dataset/`：数据集路径和加载方式，例如 COCO、VQAv2、SST2。
- `configs/infer_model/`：基础 LVLM/LLM 配置，例如 Flamingo、IDEFICS、Qwen。
- `configs/sampler/`：candidate set 采样策略配置。
- `configs/task/`：任务模板和字段映射，例如 caption、vqa、sst2。
- `configs/train/`：Lever-LM 训练组合配置。

常见命名含义：

- `query_img`：query 使用图像特征。
- `query_text`：query 使用文本特征。
- `icd_img`：ICD 使用图像特征。
- `icd_text`：ICD 使用文本特征。
- `icd_idx`：ICD 只使用样本编号 token，不额外编码图文内容。

### `lever_lm/`

Lever-LM 自身实现。

- `models/`：模型定义。核心是把 ICD 样本 index 作为 LM token 来生成。
- `dataset_module/`：把离线构造的 ICD 序列 JSON 转成训练样本。
- `candidate_sampler/`：为每个 anchor 生成候选 ICD 集合。
- `load_ds_utils.py`：封装 COCO、VQAv2、HuggingFace 数据集加载。
- `utils.py`：CLIP 特征编码、FAISS 检索、batch collate、beam top-k 过滤等工具。

### `open_mmicl/`

项目内置的 ICL 推理框架。

- `interface/`：把不同基础模型统一成 `prepare_input`、`generate`、`get_cond_prob` 等接口。
- `retriever/`：实现不同 ICD 检索策略。
- `metrics/`：任务指标计算。
- `icl_inferencer.py`：批量调用 interface 完成 ICL 推理。

### `scripts/`

常用命令封装。

- `generate_data.sh`：构造 Lever-LM 训练数据。
- `train_lever_lm.sh`：训练 Lever-LM。
- `inference.sh`：加载 retriever 做 ICL 推理评估。
- `get_sub_ds.py`：构造数据子集的辅助脚本。
- `model_size.sh`：查看模型规模的辅助脚本。

### 根目录入口文件

- `generate_data.py`：构造训练监督。
- `train.py`：训练 Lever-LM。
- `icl_inference.py`：推理评估。
- `generate_data_random.py`：随机策略的数据构造变体。
- `utils.py`：跨阶段公共函数，尤其是数据加载和 scorer。
- `README.md` / `README_CN.md`：官方使用说明。
- `.env.example`：环境变量示例，说明数据路径、checkpoint 路径、结果目录等需要如何配置。
- `.env`：本地真实配置文件，可能包含私有路径或密钥，不应提交或写入文档。
- `requirements.txt`：Python 依赖。
- `requirements_repo/`：外部依赖源码目录，例如 OpenICL；通常不作为 Lever-LM 主链路阅读重点。

## 4. 常用配置组合

caption 任务常见组合：

```text
task=caption
dataset=coco2017 或 karpathy_local
infer_model=flamingo_9B / idefics_9B
train=query_img_icd_img_text
```

含义：query 使用图像特征，ICD 使用图像和文本特征，适合图像描述任务。

VQA 任务常见组合：

```text
task=vqa
dataset=vqav2_local 或 vqav2_online
infer_model=flamingo_9B / idefics_9B
train=query_img_text_icd_img_text
```

含义：query 同时包含问题文本和图像，ICD 也包含问题/答案相关文本和图像。

纯文本或分类任务可以使用：

```text
task=sst2
train=query_text_icd_text
```

含义：query 和 ICD 都主要由文本特征驱动。

## 5. 建议阅读路线

1. 先读 `generate_data.py`，理解 anchor、candidate set、scorer、beam search 如何产生训练监督。
2. 再读 `lever_lm/dataset_module/`，确认 generated data 怎样变成 `[BOS, QUERY, ICD..., EOS]`。
3. 接着读 `lever_lm/models/gpt2_lever_lm.py`，看 ICD index 如何作为 token，被 query/ICD 图文特征增强。
4. 然后读 `train.py`，理解 Lightning 训练流程和 checkpoint 保存。
5. 最后读 `icl_inference.py` 和 `open_mmicl/retriever/lever_lm_retriever.py`，理解训练好的 Lever-LM 如何在测试时生成 ICD 序列。
