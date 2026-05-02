# Repository Guidelines

## Communication

始终使用中文和用户沟通。解释代码时优先给出具体文件、函数和数据流，不只描述概念。涉及实验结果时写清楚使用的数据集、配置、样本数、checkpoint 和输出目录。

## Project Overview

Lever-LM 是一个研究型 Python 项目，用轻量模型为 ICL 自动选择 demonstration / ICD 序列。主流程分三步：

1. `generate_data.py`：用冻结的大模型和 scorer 离线搜索高质量 ICD 序列。
2. `train.py`：训练 Lever-LM 学习生成这些 ICD index 序列。
3. `icl_inference.py`：把训练好的 Lever-LM 作为 retriever，为测试 query 生成 ICD，再交给基础 LVLM/LLM 推理。

仓库还包含 math-memory 扩展流程：

1. `math_memory_generate.py`：基于 MMLU-Pro math 和经验库生成 memory 序列监督。
2. `math_memory_train.py`：训练 `MathMemoryLeverLM` 生成 memory id。
3. `math_memory_eval.py`：比较 Lever-LM retrieval 与随机 memory retrieval。
4. `scripts/run_math_memory_pipeline.sh`：串联 generate/train/eval。

## Repository Structure

- `configs/`：Hydra 配置，按 dataset、task、sampler、infer_model、train 等分组。
- `lever_lm/`：核心模型、数据集封装、候选采样器和工具函数。
- `lever_lm/models/`：GPT-2 和 LSTM 版本的 Lever-LM。
- `lever_lm/dataset_module/`：把 generated data 转成训练样本。
- `lever_lm/candidate_sampler/`：为 anchor/query 构造 candidate set。
- `lever_lm/math_memory/`：MMLU-Pro math + experience memory 的数据、embedding、模型、scorer。
- `open_mmicl/`：内置 ICL 框架，包含 interface、retriever、metrics。
- `scripts/`：常用实验脚本。
- `requirements_repo/`：外部依赖源码，例如 OpenICL；不要轻易改动。

## Core Data Flow

原始 Lever-LM 流程中，generated data 的一条序列通常是：

```text
[icd_1, icd_2, ..., query_id]
```

训练时会转成：

```text
[BOS, QUERY, icd_1, icd_2, ..., EOS]
```

ICD 样本 index 本身就是模型词表 token。`len(index_ds)`、`len(index_ds)+1`、`len(index_ds)+2` 分别用于 EOS、BOS、QUERY 等特殊 token。

默认 scorer 是 InfoScore。它衡量加入候选 ICD 后，基础模型对正确答案的条件概率提升：

```text
score(c) = P(y | x, existing ICDs + c) - P(y | x, existing ICDs)
```

训练阶段通常只用 score 做 threshold 过滤，不把 score 当连续权重加入 loss。

math-memory 当前生成阶段默认使用：

```text
score = log P(correct | query + prefix + candidate)
      - log P(correct | query + prefix)
```

也就是 `--score-mode delta_logprob`。如需复现旧行为，使用 `--score-mode absolute_logprob`。

## Development Commands

推荐环境：

```bash
conda create -n leverlm python=3.10
conda activate leverlm
pip install -r requirements.txt
pip install -e requirements_repo/OpenICL
```

快速语法检查：

```bash
python -m py_compile generate_data.py train.py icl_inference.py
python -m py_compile math_memory_generate.py math_memory_train.py math_memory_eval.py
bash -n scripts/run_math_memory_pipeline.sh
```

原始主流程常用命令：

```bash
bash scripts/generate_data.sh caption coco2017 "[0,1,2,3]"
bash scripts/train_lever_lm.sh caption coco2017 1 query_img_icd_img_text
bash scripts/inference.sh caption coco2017 0 query_img_icd_img_text
```

math-memory smoke 示例：

```bash
RESULT_DIR=/tmp/leverlm_math_memory_smoke \
bash scripts/run_math_memory_pipeline.sh \
  --experience-file data/experiences.json \
  --smoke
```

只生成 math-memory 数据：

```bash
python math_memory_generate.py \
  --experience-file data/experiences.json \
  --output-file /tmp/math_memory_generated.json \
  --mock-data \
  --scorer-model mock \
  --scorer-device cpu
```

## Coding Conventions

- Python 代码使用 4 空格缩进。
- 函数和变量使用 `snake_case`，类名使用 `PascalCase`。
- 新增配置遵循 Hydra 目录结构，复用配置放到 `configs/<group>/`。
- 保持入口脚本清晰，复杂逻辑抽成局部辅助函数。
- 不要引入与现有栈不一致的新框架，除非任务明确需要。
- 不要把实验输出、checkpoint、缓存数据集、日志或下载模型加入仓库。

## Testing and Verification

本项目没有完整测试套件。修改后至少做以下验证：

- 改 Python 文件后运行相关 `py_compile`。
- 改 shell 脚本后运行 `bash -n <script>`。
- 改数据构造逻辑后，用 mock 或小样本跑到生成 JSON，并检查 metadata 与样本字段。
- 改训练或推理逻辑后，先减少样本数、epoch、batch size 做 smoke run，再跑完整 GPU 任务。

如果环境缺少依赖，例如 `transformers`、`datasets`、`pyarrow`，在最终说明里明确指出验证停在哪一步。

## Configuration and Paths

- 使用 `.env.example` 作为本地配置模板。
- 常见环境变量包括 `RESULT_DIR`、`CHECKPOINT_PATH`、`COCO_PATH`、`VQAV2_PATH`。
- 本地真实 `.env`、私有路径、token、凭据不要提交或写入公开说明。
- Hugging Face 缓存、模型权重和数据集通常很大，应放在外部缓存目录，例如 `/root/autodl-tmp/hf_cache`。

## Agent Workflow

- 修改前先阅读相关入口、配置和调用链。
- 发现工作树中已有用户改动时，不要回滚，必须在其基础上继续。
- 只改与任务直接相关的文件；避免顺手重构无关代码。
- 给用户汇报时说明改了什么、验证了什么、还有什么没法验证。
- 解释主流程时区分原始 Lever-LM 和 math-memory 扩展，避免混用概念。
