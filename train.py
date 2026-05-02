import json
import os
import time
from datetime import datetime, timezone
from functools import partial

import hydra
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    TQDMProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import optim
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, get_cosine_schedule_with_warmup

from lever_lm.utils import data_split, collate_fn
from utils import load_ds


SCRIPT_START_PERF = time.perf_counter()
SCRIPT_START_AT = datetime.now(timezone.utc)


"""
训练 Lever-LM 的入口。

generate_data.py 已经把“高分 ICD 序列”保存成 JSON。这里会把这些 id 序列转成
语言模型训练样本：训练 Lever-LM 在看到 BOS、Query 以及前面已生成的 ICD 后，
预测下一个 ICD token。论文里的“tiny LM levers LVLM”就是在这个文件中完成训练。
"""


class LeverLM(pl.LightningModule):
    """Pytorch Lightning 包装器，负责训练循环、日志、优化器和学习率计划。

    self.lever_lm 才是真正的模型主体，可能是 GPT2LeverLM 或 LSTMLeverLM。
    LightningModule 只把 batch 喂给模型并取出 loss。
    """

    def __init__(self, lever_lm, lr, weight_decay=1e-2, warm_steps=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["lever_lm"])
        self.lever_lm = lever_lm

    def training_step(self, batch, batch_idx):
        # batch 已经由 collate_fn 整理成 query_input、icd_input、icd_seq_idx。
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log(
            "train_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.lever_lm(**batch)
        loss = output["loss"]
        self.log("val_loss", loss, batch_size=len(batch["icd_seq_idx"]), sync_dist=True)
        return loss

    def configure_optimizers(self):
        # Lever-LM 规模较小，训练时对所有可训练参数使用 AdamW。
        optimizer = optim.AdamW(
            self.lever_lm.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # warm_steps 支持两种写法：整数表示固定 step 数，浮点数表示总 step 比例。
        step_batches = self.trainer.estimated_stepping_batches
        if isinstance(self.hparams.warm_steps, float):
            warm_steps = self.hparams.warm_steps * step_batches
        elif isinstance(self.hparams.warm_steps, int):
            warm_steps = self.hparams.warm_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(self.hparams.warm_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class TrainingTimingCallback(pl.Callback):
    """记录用户关心的两个时间：

    1. train.py 进程完整运行时间：从 Python 进入脚本开始算。
    2. Transformer 训练时间：从 epoch 0 真正开始训练算，到 fit 结束。

    记录会写到 results/timing/*.json，同时在终端打印一行 `[TIMING]`，方便从
    长日志里 grep。
    """

    def __init__(self, cfg):
        super().__init__()
        timing_dir = hydra.utils.to_absolute_path(
            os.path.join(cfg.result_dir, "timing")
        )
        os.makedirs(timing_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_ex_name = str(cfg.ex_name).replace("/", "_")
        self.path = os.path.join(
            timing_dir,
            f"train_timing_{cfg.task.task_name}_{safe_ex_name}_{ts}.json",
        )
        self.records = {
            "script_start_at_utc": SCRIPT_START_AT.isoformat(),
            "task": cfg.task.task_name,
            "dataset": cfg.dataset.name,
            "train_config": cfg.train.lever_lm._target_,
            "data_files": cfg.data_files,
            "ex_name": cfg.ex_name,
        }
        self.epoch0_start_perf = None
        self.fit_end_perf = None

    def _write(self):
        with open(self.path, "w") as f:
            json.dump(self.records, f, indent=2)

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == 0 and self.epoch0_start_perf is None:
            self.epoch0_start_perf = time.perf_counter()
            self.records["epoch0_train_start_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.records["script_to_epoch0_start_seconds"] = (
                self.epoch0_start_perf - SCRIPT_START_PERF
            )
            self._write()
            print(
                f"[TIMING] epoch0_train_start_at_utc={self.records['epoch0_train_start_at_utc']}"
            )

    def on_fit_end(self, trainer, pl_module):
        self.fit_end_perf = time.perf_counter()
        self.records["fit_end_at_utc"] = datetime.now(timezone.utc).isoformat()
        if self.epoch0_start_perf is not None:
            self.records["transformer_training_seconds_epoch0_to_fit_end"] = (
                self.fit_end_perf - self.epoch0_start_perf
            )
        self.records["train_py_seconds_until_fit_end"] = (
            self.fit_end_perf - SCRIPT_START_PERF
        )
        self.records["max_epochs"] = trainer.max_epochs
        self.records["global_step"] = trainer.global_step
        self.records["timing_file"] = self.path
        self._write()
        print(
            "[TIMING] "
            f"transformer_training_seconds_epoch0_to_fit_end="
            f"{self.records.get('transformer_training_seconds_epoch0_to_fit_end'):.3f} "
            f"train_py_seconds_until_fit_end={self.records['train_py_seconds_until_fit_end']:.3f} "
            f"timing_file={self.path}"
        )


class ICDSeqDataModule(pl.LightningDataModule):
    """把生成好的 ICD 序列 JSON 封装成 Lightning DataLoader。

    关键数据流：
    1. 读取 result_dir/generated_data/<data_files>.json。
    2. data_split 按 query id 划分训练/验证。
    3. cfg.train.lever_lm_ds 决定样本中使用 query 图像、query 文本、ICD 图像或 ICD 文本。
    4. collate_fn 用 CLIPProcessor 把图文输入批量化。
    """

    def __init__(
        self,
        cfg,
    ):
        """
        dataset_para: The dataset parameters
        dataset: The *.py file name of the dataset class
        dataset_name: The dataset Class name
        """
        super().__init__()
        data_files_path = os.path.join(cfg.result_dir, "generated_data", cfg.data_files)
        with open(data_files_path, "r") as f:
            data = json.load(f)
        # data_split 输出的不是 HuggingFace Dataset，而是 Lever-LM 专用的序列列表。
        self.train_data_list, self.val_data_list = data_split(data, cfg.train_ratio)
        # Hydra 根据 train/query_* 配置选择具体 Dataset 类与字段组合。
        self.ds_factory = hydra.utils.instantiate(cfg.train.lever_lm_ds, _partial_=True)
        self.index_ds = load_ds(cfg, "train")
        self.processor = CLIPProcessor.from_pretrained(cfg.train.lever_lm.clip_name)

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            # index_ds 是原始样本表，Dataset 会用 ICD id 回查图像/文本特征。
            self.trainset = self.ds_factory(
                data_list=self.train_data_list, index_ds=self.index_ds
            )
            self.valset = self.ds_factory(
                data_list=self.val_data_list, index_ds=self.index_ds
            )

    def train_dataloader(self):
        global collate_fn
        return DataLoader(
            self.trainset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            shuffle=True,
            collate_fn=partial(collate_fn, processor=self.processor),
            pin_memory=True,
        )

    def val_dataloader(self):
        global collate_fn
        return DataLoader(
            self.valset,
            batch_size=self.hparams.cfg.batch_size,
            num_workers=self.hparams.cfg.num_workers,
            collate_fn=partial(collate_fn, processor=self.processor),
            shuffle=False,
        )


@hydra.main(version_base=None, config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """训练主函数：创建日志、checkpoint、模型和 DataModule。"""
    pl.seed_everything(cfg.seed)

    logger = WandbLogger(**cfg.wandb_args)
    # 同时保存 train loss 最优和 val loss 最优，便于后续 inference 选择 checkpoint。
    tl_model_cpk_callback = ModelCheckpoint(
        filename="min_tl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="train_loss",
        save_last=False,
        save_top_k=1,
        mode="min",
        dirpath=cfg.dirpath,
    )
    vl_model_cpk_callback = ModelCheckpoint(
        filename="min_vl-{epoch}-{train_loss:.5f}-{val_loss:.5f}",
        monitor="val_loss",
        save_last=True,
        save_top_k=1,
        mode="min",
        dirpath=cfg.dirpath,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            TQDMProgressBar(),
            TrainingTimingCallback(cfg),
            tl_model_cpk_callback,
            vl_model_cpk_callback,
        ],
        **cfg.trainer_args,
    )
    # cfg.train.lever_lm 指向具体模型配置，如 query_img_icd_img_text_lever_lm。
    lever_lm = hydra.utils.instantiate(cfg.train.lever_lm)
    model = LeverLM(lever_lm, cfg.lr, cfg.weight_decay, cfg.warm_steps)
    data_module = ICDSeqDataModule(cfg)
    trainer.fit(model, data_module)


if __name__ == "__main__":
    load_dotenv()
    main()
