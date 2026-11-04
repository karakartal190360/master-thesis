import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

import os
from collections import defaultdict
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from data import RawDataset, SpeechFakeDataset
from utils import compute_eer

torch.set_float32_matmul_precision('high')

BACKBONE = "hubert_xlarge"  # MMS, wav2vec_small, wav2vec_large, xlsr_1b, hubert_xlarge
if BACKBONE == "MMS":
    from antideepfake.mms import DeepfakeDetector
    model = DeepfakeDetector()
elif BACKBONE == "wav2vec_large":
    from antideepfake.wav2vec_large import DeepfakeDetector
    model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-large-anti-deepfake")
elif BACKBONE == "wav2vec_small":
    from antideepfake.wav2vec_small import DeepfakeDetector
    model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-small-anti-deepfake")
elif BACKBONE == "xlsr_1b":
    from antideepfake.xlsr_1b import DeepfakeDetector
    model = DeepfakeDetector.from_pretrained("nii-yamagishilab/xls-r-1b-anti-deepfake")
elif BACKBONE == "hubert_xlarge":
    from antideepfake.hubert_xlarge import DeepfakeDetector
    model = DeepfakeDetector.from_pretrained("nii-yamagishilab/hubert-xlarge-anti-deepfake")
else:
    raise NotImplementedError

MAX_LEN = 32000
CUDA_INDEX = 6


class DataModule(L.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.test_in_the_wild_dataset = RawDataset(
            data_dir="./In-the-Wild/release_in_the_wild",
            protocol_path="In-the-Wild/wild_labels.txt",
            max_len=MAX_LEN,
        )
        self.test_zh_famous_dataset = RawDataset(
            data_dir="./zh_famous_samples",
            protocol_path="./zh_famous_samples/labels.txt",
            max_len=MAX_LEN,
        )
        self.test_famous_figures = SpeechFakeDataset(data_dir="./famous_figures_samples", max_len=MAX_LEN)

    def test_dataloader(self):
        # 返回已知和未知说话人的 DataLoader 列表，用于批量测试
        return [
            DataLoader(self.test_in_the_wild_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
            DataLoader(self.test_zh_famous_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
            DataLoader(self.test_famous_figures, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
        ]


class SpeakerSpoofDetectorModule(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()  # 保存超参数

        # 实例化模型
        print(f"[*] Loading {BACKBONE} model")
        self.model = model

    def forward(self, wav):
        pred, emb = self.model(wav)
        return pred

    def on_test_epoch_start(self):
        self.test_outputs = defaultdict(lambda: {'scores': [], 'labels': []})

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        wav, label, *_ = batch
        # 预测相似度分数 (通常取正类的概率)
        pred = self(wav)[:, 1]

        self.test_outputs[dataloader_idx]['scores'].extend(pred.cpu().numpy().tolist())
        self.test_outputs[dataloader_idx]['labels'].extend(label.cpu().numpy().tolist())

    def on_test_epoch_end(self):
        dataloader_indices = sorted(self.test_outputs.keys())
        tags = ["test/in_the_wild", "test/zh_famous", "test/famous_figures"]

        for dataloader_idx in dataloader_indices:
            data = self.test_outputs[dataloader_idx]
            tag = tags[dataloader_idx]

            y_true = np.array(data['labels'])
            y_score = np.array(data['scores'])

            eer, eer_threshold = compute_eer(y_score, y_true)
            auc = roc_auc_score(y_true, y_score)

            # 4. 记录和打印结果
            self.log(f'{tag}_EER', eer, prog_bar=True)
            self.log(f'{tag}_AUC', auc, prog_bar=True)
            print(f"[{tag}] EER: {eer * 100:.4f}%, AUC: {auc:.4f} (Threshold: {eer_threshold:.4f})")

        self.test_outputs.clear()


if __name__ == '__main__':
    L.seed_everything(42)
    BATCH_SIZE = 16
    data_module = DataModule(batch_size=BATCH_SIZE)
    data_module.setup()

    model_module = SpeakerSpoofDetectorModule()

    logger = TensorBoardLogger("exp/", name=f"baseline1-{BACKBONE}")
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[CUDA_INDEX],
        logger=logger,
    )

    print("\n✅ 开始测试...")
    trainer.test(model_module, data_module.test_dataloader())
