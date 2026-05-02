# Math Memory

Math Memory 是基于 Lever-LM 改造的数学经验检索实验仓库。项目目标是在 MMLU-Pro math 子集上，把从 DAPO/经验库中提取的文本经验作为 memory bank，训练一个两层 Transformer 检索器，让它根据测试题生成若干条 experience id，再把这些 experience 拼入 Qwen3-8B prompt 中进行选择题评测。

## 核心工作流

1. 准备经验库：`data/experiences.json` 保存所有 experience 文本，加载后会按 `G0, G1, ...` 自然排序并映射成 dense memory id。
2. 构造训练序列：`math_memory_generate.py` 对 MMLU-Pro math 的训练 split 采样候选 experience，使用 Qwen3-8B 对正确答案的 log probability 增益打分，通过 beam search 生成监督序列。
3. 训练检索器：`math_memory_train.py` 读取生成序列，训练 `MathMemoryLeverLM`。模型输入是 query embedding，目标是自回归预测 memory id 序列。
4. 测试检索策略：`math_memory_eval.py` 比较 LeverLM 生成的 memory 序列和随机采样 RS 策略。两者后续都用同一个 Qwen3 scorer 评测答案正确率。
5. 分析固定序列质量：`math_memory_eval_generated.py` 可以把已经构造出的 memory 序列直接作为固定 retrievals 送入 Qwen3，评估 beam/random 序列本身质量。

## 主要文件说明

- `math_memory_generate.py`：主训练数据生成脚本。支持 `delta_logprob`/`absolute_logprob`，支持断点续跑；每个 query 生成完成后会实时写入主 JSON 和 `.partial.jsonl`。
- `math_memory_generate_random.py`：生成随机 memory 序列，用于和 beam search 训练序列做固定序列质量对比。
- `math_memory_train.py`：训练两层 Transformer 检索器。包含 early stopping，保存 `best.pt`、`last.pt` 和 `loss_history.csv`。
- `math_memory_eval.py`：评估 LeverLM 或 RS 在 test split 上的表现。`--seed` 固定 train/test split，`--rs-seed` 只控制 RS memory 采样。
- `math_memory_eval_generated.py`：评估固定 generated/random memory 序列文件。
- `math_memory_summarize_rs.py`：汇总多次 RS seed 评测，输出 mean/std/min/max。
- `lever_lm/math_memory/data.py`：加载 experience、MMLU-Pro math split，构造 Qwen3 answer prompt。
- `lever_lm/math_memory/scoring.py`：Qwen3 选择题 scorer。单 token 选项字母使用 prompt-only logits，一次 forward 读取所有选项概率。
- `lever_lm/math_memory/model.py`：`MathMemoryLeverLM` 模型定义。包含 GPT2 风格两层 causal Transformer 和 adapter。
- `lever_lm/math_memory/embeddings.py`：Qwen3-Embedding/Mock embedder 以及 embedding cache 工具。
- `scripts/run_math_memory_pipeline.sh`：端到端脚本，串联 generate、train、eval，并把终端输出保存到实验目录日志文件。
- `data/experiences.json`：经验库文本。
- `AGENTS.md`：本仓库的协作/代理使用说明。

原始 Lever-LM/OpenMMICL 相关文件仍保留：`generate_data.py`、`generate_data_random.py`、`train.py`、`icl_inference.py`、`open_mmicl/`、`configs/` 等。

## 环境准备

推荐使用 Conda：

```bash
conda create -n leverlm_math python=3.10
conda activate leverlm_math
pip install -r requirements.txt
```

如果需要 OpenICL 相关原始流程，请按原 Lever-LM 依赖安装 `requirements_repo/OpenICL`。Math Memory 主流程主要依赖 PyTorch、Transformers、datasets、tqdm 等。

Hugging Face 模型和数据集默认会使用本地缓存；离线环境需要提前缓存：

- `TIGER-Lab/MMLU-Pro`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-Embedding-0.6B`

## 运行实验

### 1. 端到端主实验

```bash
bash scripts/run_math_memory_pipeline.sh \
  --experience-file data/experiences.json \
  --result-dir /root/autodl-tmp/leverlm_math_memory_split50_repeat3_beam10 \
  --score-mode delta_logprob \
  --train-ratio 0.5 \
  --candidate-num 64 \
  --repeat 3 \
  --beam-size 10 \
  --shot-num 2 \
  --scorer-batch-size 32 \
  --embedding-batch-size 256 \
  --batch-size 128 \
  --max-epochs 100 \
  --early-stop-patience 5
```

输出结构：

```text
<result-dir>/generated_data/      # generated JSON 和 .partial.jsonl 断点文件
<result-dir>/model_cpk/           # best.pt、last.pt、loss_history.csv
<result-dir>/metrics/             # predictions、metrics.csv、json 指标
<result-dir>/pipeline_*.log        # 完整终端日志
```

### 2. 断点续跑生成阶段

`math_memory_generate.py` 每完成一个 query 会保存一次。如果在 `Generating D_M` 中断，重新运行同一个 pipeline 命令即可继续生成。若要强制重来，删除对应 `generated_data` 文件，或直接调用底层脚本加 `--overwrite`。

### 3. 只重新训练

如果 generated data 已经存在，只重新训练：

```bash
bash scripts/run_math_memory_pipeline.sh \
  --stage train \
  --experience-file data/experiences.json \
  --result-dir /root/autodl-tmp/leverlm_math_memory_split50_repeat3_beam10 \
  --score-mode delta_logprob \
  --train-ratio 0.5 \
  --candidate-num 64 \
  --repeat 3 \
  --beam-size 10 \
  --shot-num 2 \
  --embedding-batch-size 256 \
  --batch-size 128 \
  --max-epochs 100 \
  --early-stop-patience 5
```

### 4. 评估 best.pt / last.pt

```bash
python math_memory_eval.py \
  --method lever_lm \
  --checkpoint /path/to/best.pt \
  --experience-file data/experiences.json \
  --output-dir /path/to/metrics_best \
  --shot-num 2 \
  --seed 42 \
  --train-ratio 0.5 \
  --scorer-model Qwen/Qwen3-8B \
  --scorer-device cuda \
  --scorer-dtype bf16 \
  --scorer-batch-size 8 \
  --scorer-max-length 4096 \
  --embedding-cache-dir /path/to/cache/math_memory_embeddings \
  --embedding-model Qwen/Qwen3-Embedding-0.6B \
  --embedding-device cuda \
  --embedding-batch-size 128
```

### 5. 多 seed RS baseline

固定同一个 test split，只改变 `--rs-seed`：

```bash
for rs_seed in 1 2 3 4 5 42 100 123 2024 2026; do
  python math_memory_eval.py \
    --method rs \
    --experience-file data/experiences.json \
    --output-dir /path/to/rs_repeats \
    --shot-num 2 \
    --seed 42 \
    --rs-seed "$rs_seed" \
    --train-ratio 0.5 \
    --scorer-model Qwen/Qwen3-8B \
    --scorer-device cuda \
    --scorer-dtype bf16 \
    --scorer-batch-size 8 \
    --scorer-max-length 4096 \
    --embedding-cache-dir /path/to/cache/math_memory_embeddings
done

python math_memory_summarize_rs.py --metrics-dir /path/to/rs_repeats
```

## 关键实验参数

- `--train-ratio`：划分 MMLU-Pro math train/test 的比例。
- `--anchor-limit`：限制用于构造训练序列的 train query 数量。
- `--candidate-num`：每个 anchor query 随机采样多少个候选 experience。
- `--repeat`：每个 query 重新采样 candidate set 的次数。
- `--beam-size`：beam search 保留多少条序列。
- `--shot-num`：最终检索多少条 experience。
- `--score-mode delta_logprob`：按加入 memory 后正确答案 log probability 的边际增益排序。
- `--early-stop-patience`：验证集 loss 连续多少个 epoch 不提升后停止训练。

训练序列总数约为：

```text
train_query_count × repeat × beam_size
```

例如 MMLU-Pro math 共 1351 道，`train_ratio=0.5` 时约 675 个 train query；`repeat=3`、`beam_size=10` 会生成约 20250 条训练序列。

## 注意事项

- `.env`、缓存、实验结果、论文 PDF 和本地抽取文本不会提交到 Git。
- Qwen3 scorer 当前对选项字母使用 next-token log probability，不生成长文本答案。
- `best.pt` 按验证集 LM loss 保存，`last.pt` 是最后一个 epoch；最终应分别评估 downstream accuracy 后再决定报告哪个。
