import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

import os
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import glob


class BaseDataset(Dataset):
    def __init__(self, data_dir, max_len=64320, sr=16000):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.segment_length = max_len
        self.sr = sr
        self.suffix = ""

    def cut_or_pad(self, wav):
        if wav.shape[0] < self.segment_length:
            wav = torch.cat([wav, torch.zeros(self.segment_length - wav.shape[0])])
        else:
            # 随机截断 (训练时使用) 或固定截断 (测试时使用，简化处理)
            start = random.randint(0, wav.shape[0] - self.segment_length)
            wav = wav[start: start + self.segment_length]
        return wav

    def load_audio(self, key):
        full_path = str(self.data_dir / f"{key}{self.suffix}")
        wav, sr = torchaudio.load(full_path)
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            wav = resampler(wav)
        wav = wav.squeeze()
        with torch.no_grad():
            wav = torch.nn.functional.layer_norm(wav, wav.shape)
        wav = self.cut_or_pad(wav)
        return wav


class RawDataset(BaseDataset):
    def __init__(self, data_dir, protocol_path=None, protocol_list=None, is_enroll=False, max_len=64320):
        super().__init__(data_dir, max_len=max_len)
        self.is_enroll = is_enroll

        # 优先从 list 加载，否则读取文件
        if protocol_list:
            lines = protocol_list
        else:
            with open(protocol_path, "r") as f:
                lines = f.readlines()

        self.samples = []
        for line in lines:
            spk, filename, _, _, label = line.strip().split()

            if self.is_enroll:
                self.samples.append((spk, filename))
            else:
                l_code = 1 if label == "bonafide" else 0
                self.samples.append((spk, filename, l_code))

        if filename.endswith(".wav"):
            self.suffix = ""
        elif os.path.exists(self.data_dir / f"{filename}.wav"):
            self.suffix = ".wav"
        else:
            self.suffix = ".flac"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_enroll:
            spk, filename = self.samples[idx]
            wav = self.load_audio(filename)
            return wav, spk
        else:
            spk, filename, label = self.samples[idx]
            wav = self.load_audio(filename)
            return wav, torch.tensor(label, dtype=torch.long), spk


class SpeechFakeDataset(BaseDataset):
    def __init__(self, data_dir, max_len=64320):
        super().__init__(data_dir, max_len=max_len)

        self.samples = []
        files = glob.glob(str(self.data_dir / '**' / '*.wav'), recursive=True)
        for path in files:
            spk, label, key = path.split('/')[-3:]
            key = f"{spk}/{label}/{key}"
            if label == "bonafide":
                self.samples.append((spk, key, 1))
            elif label == "spoof":
                self.samples.append((spk, key, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        spk, key, label = self.samples[idx]
        wav = self.load_audio(key)
        label = torch.tensor(label, dtype=torch.long)
        return wav, label, spk


class PairDataset(BaseDataset):
    """
    修改后的 Dataset，用于加载 Reference/Test 样本对。
    假设 Reference/Test 样本路径结构与原始 load_audio 兼容。
    """

    def __init__(self, data_dir, protocol_path):
        super().__init__(data_dir)
        self.data_dir = Path(data_dir)
        self.protocol_df = pd.read_csv(protocol_path, sep='\t')
        key = self.protocol_df.iloc[0]["Reference_Key"]
        if key.endswith(".wav"):
            self.suffix = ""
        elif os.path.exists(self.data_dir / f"{key}.wav"):
            self.suffix = ".wav"
        else:
            self.suffix = ".flac"

    def __len__(self):
        return len(self.protocol_df)

    def __getitem__(self, idx):
        row = self.protocol_df.iloc[idx]

        ref_key = row['Reference_Key']
        test_key = row['Test_Key']
        sim_label = row['Similarity_Label']
        spk_id = row['Speaker_ID']

        audio_R = self.load_audio(ref_key)
        audio_T = self.load_audio(test_key)

        # 相似度标签，需要 float 类型用于 BCEWithLogitsLoss
        sim_label = torch.tensor(sim_label, dtype=torch.float)

        return audio_R, audio_T, sim_label  # R, T, Label


class PairDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, protocol_path, batch_size=16, num_workers=8):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.protocol_path = Path(protocol_path)

    def setup(self, stage=None):
        # 训练集
        self.train_dataset = PairDataset(
            data_dir=self.data_dir,
            protocol_path=self.protocol_path
        )
        self.test_in_the_wild_dataset = PairDataset(
            data_dir="./In-the-Wild/release_in_the_wild",
            protocol_path="./in_the_wild_pairs.tsv"
        )
        self.test_zh_famous_dataset = PairDataset(
            data_dir="./zh_famous_samples",
            protocol_path="./zh_famous_samples/protocol.tsv"
        )
        self.test_famous_figures_dataset = PairDataset(
            data_dir="./famous_figures_samples",
            protocol_path="./famous_figures_pairs.tsv"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        # 返回已知和未知说话人的 DataLoader 列表，用于批量测试
        return [
            # DataLoader(self.test_known_dataset, batch_size=self.batch_size, num_workers=self.num_workers),
            DataLoader(self.test_in_the_wild_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
            DataLoader(self.test_zh_famous_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
            DataLoader(self.test_famous_figures_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                       pin_memory=True),
        ]

