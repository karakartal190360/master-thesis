import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from utils import compute_eer
from antideepfake.mms import DeepfakeDetector

BACKBONE = "MMS"
cuda_index = 6
DEVICE = f"cuda:{cuda_index}"
# os.environ["HF_HUB_OFFLINE"] = "1"

asv_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)
feat_model = DeepfakeDetector().m_ssl.to(DEVICE).eval()

class RobustProfileDetector:
    def __init__(self, n_components: int=5):
        self.n_components = n_components

        self.gmm = None

    def _train_gmm(self, vectors, n_components=2):
        """
        训练一个轻量级 GMM。
        原理：GMM 试图找到几个高斯分布的组合，使其生成这些数据的概率最大。
        对于小样本（One-shot/Few-shot），我们限制分量数并使用对角协方差。
        """
        if len(vectors) < self.n_components * 2:
            # 样本极少时，退化为单高斯
            n_components = 1

        # 协方差类型 'diag'：假设各维度独立，参数量少，适合高维少样本。
        # 'spherical' 参数更少，也可以尝试。
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar=1e-3, random_state=42)
        gmm.fit(vectors)
        return gmm

    def _compute_gmm_score(self, gmm, vector):
        """
        计算 Log-Likelihood (对数似然)。
        值越大（通常是负数，越接近0），表示该样本属于该分布的概率越大。
        """
        # score_samples 返回的是每个样本的 log-likelihood
        return gmm.score_samples(vector.reshape(1, -1))[0]

    def _normalize_scores(self, scores):
        # 1. 调整参数以匹配你的数据分布
        # 根据你提供的 raw_ll，中心点设在 0 附近可能更合适，缩放因子需要加大
        center = -2000
        scale = 200  # 增大 scale 可以让 Sigmoid 曲线更平缓，覆盖更广的 LL 范围

        # 2. 计算 z 值
        z = (scores - center) / scale

        # 3. 数值稳定处理：限制 z 的范围
        # 防止 np.exp(-z) 在 z 为极小的负数时溢出，或在 z 为极大正数时分母变 0
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def fit(self, audio_paths: list, verbose=False):
        """
        audio_paths: list of audio_path (str)
        """
        # --- 收集数据 ---
        feats = []

        for audio_p in audio_paths:
            try:
                wav, sr = torchaudio.load(audio_p)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(sr, 16000)
                    wav = resampler(wav)
                wav = wav[0]
                with torch.no_grad():
                    wav = torch.nn.functional.layer_norm(wav, wav.shape)
                wav = wav.to(DEVICE).unsqueeze(0)
                feat = feat_model(wav).squeeze(0)
                feat = feat.mean(dim=0).cpu().numpy()
                feats.append(feat)
            except Exception as e:
                print(f"Error processing {audio_p}: {e}")
                continue

        feats = np.array(feats)

        self.gmm = self._train_gmm(feats, n_components=self.n_components)

    def predict(self, test_audio_path):
        try:
            wav, sr = torchaudio.load(test_audio_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                wav = resampler(wav)
            wav = wav[0]
            with torch.no_grad():
                wav = torch.nn.functional.layer_norm(wav, wav.shape)
            wav = wav.to(DEVICE).unsqueeze(0)
            feat = feat_model(wav).squeeze(0)
            feat = feat.mean(dim=0).cpu().numpy()
        except Exception as e:
            print(f"Error processing {test_audio_path}: {e}")
            return 0.5, "Extract Fail"

        raw_ll = self._compute_gmm_score(self.gmm, feat)
        score = self._normalize_scores(raw_ll)
        return score, "Success"


class HybridPOIDetector(RobustProfileDetector):
    def __init__(self):
        # 初始化原有的音素检测器
        super().__init__()
        self.target_gmm = None  # 存储目标人的声纹GMM

    def _extract_embedding(self, wav_path):
        """提取全局 Speaker Embedding"""
        x, _ = librosa.load(wav_path, sr=16000)
        x = torch.from_numpy(x).float()
        # SpeechBrain 自动处理采样率重采样
        embedding = asv_model.encode_batch(x).squeeze()
        return embedding.cpu().numpy()

    def _normalize_asv_scores(self, scores):
        # 1. 调整参数以匹配你的数据分布
        # 根据你提供的 raw_ll，中心点设在 0 附近可能更合适，缩放因子需要加大
        center = -1800
        scale = 100  # 增大 scale 可以让 Sigmoid 曲线更平缓，覆盖更广的 LL 范围

        # 2. 计算 z 值
        z = (scores - center) / scale

        # 3. 数值稳定处理：限制 z 的范围
        # 防止 np.exp(-z) 在 z 为极小的负数时溢出，或在 z 为极大正数时分母变 0
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def fit(self, reference_paths: list, top_k=12, verbose=False):
        """
        注册阶段：同时构建音素画像和声纹中心
        """
        # 1. 运行原有的音素级注册
        super().fit(reference_paths, verbose=verbose)

        # 2. 运行 ASV 注册 (计算 Embedding 质心)
        embeddings = []
        for path in reference_paths:
            emb = self._extract_embedding(path)
            embeddings.append(emb)

        n_comp = 5 if len(embeddings) > 10 else 1
        self.target_gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type='diag',
            reg_covar=1e-4,
            max_iter=200,
            random_state=42
        )
        self.target_gmm.fit(embeddings)

        if verbose:
            print("[*] ASV Profile Built (Centroid computed)")

    def predict(self, test_audio_path, alpha=0.8):
        """
        alpha: 权重系数 (0~1)。
               alpha 越大，越看重音素细节(防伪)；
               alpha 越小，越看重整体音色(认人)。
               建议设为 0.5 - 0.7。
        """
        # 1. 获取音素级分数 (Phoneme LLR Score)
        # 注意：这里调用父类的 predict，我们需要它返回分数
        # 假设父类 predict 返回 (score, level_msg)
        phn_score, phn_msg = super().predict(test_audio_path)

        test_emb = self._extract_embedding(test_audio_path)
        asv_ll = self.target_gmm.score_samples(test_emb.reshape(1, -1))[0]
        asv_score = self._normalize_asv_scores(asv_ll)

        # 4. 融合
        final_score = alpha * phn_score + (1 - alpha) * asv_score

        return final_score, f"{phn_msg} + ASV"


def run_poi_specificity_experiment(enroll_path, test_path, audio_root):
    """
    针对 POI 特异性设计的实验主程序
    protocol 格式: [ref_key, test_key, label, speaker, attack_type]
    label: 1(Bonafide), 0(Spoof)
    """
    # 1. 加载数据
    enroll_df = pd.read_csv(enroll_path, sep=',', names=['speaker', 'path', 'source', 'label'], header=0)
    test_df = pd.read_csv(test_path, sep=',', names=['speaker', 'path', 'source', 'label'], header=0)

    # 建立数据库，方便快速检索
    spk_profiles = {}
    for spk, group in tqdm(enroll_df.groupby('speaker'), desc="Enrolling"):
        # print(f"{spk}: {len(group)} / {len(enroll_df)}")
        enroll_paths = group["path"].map(str).tolist()
        if not enroll_paths[0].endswith(".wav"):
            enroll_paths = [os.path.join(audio_root, path + ".wav") for path in enroll_paths]
        else:
            enroll_paths = [os.path.join(audio_root, path) for path in enroll_paths]

        if len(enroll_paths) < 5:
            continue

        detector = HybridPOIDetector()
        detector.fit(enroll_paths)
        spk_profiles[spk] = detector

    # 3. 构造测试 Trials
    print(f"[*] Constructing Specificity Trials...")

    # 4. 运行预测
    results = []
    print(f"[*] Scoring {len(test_df)} trials...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        detector = spk_profiles[row['speaker']]
        if not str(row["path"]).endswith(".wav"):
            row["path"] = str(row["path"]) + ".wav"
        test_path = os.path.join(audio_root, row['path'])

        try:
            fused_score, msg = detector.predict(test_path)
        except Exception as e:
            print(e)
            continue

        row["label"] = 1 if row["label"] == "bonafide" else 0
        row['score'] = fused_score  # 融合分数
        results.append(row)

    res_df = pd.DataFrame(results, columns=['speaker', 'path', 'label', 'score'])

    # 5. 指标计算与分析
    print("\n" + "=" * 30)
    print(" POI Specificity Report ")
    print("=" * 30)

    metrics = []
    # --- 任务 1: 传统防伪 (TG vs TS) ---
    # A. 计算融合分数 (Fused) 指标
    eer, _ = compute_eer(res_df['score'], res_df['label'])
    auc = roc_auc_score(res_df['label'], res_df['score'])
    metrics.append({
        'speaker': "ALL", 'task': 'Spoof_Detection_Fused', 'eer': eer, 'auc': auc
    })
    print(f"EER: {eer * 100:.4f}%, AUC: {auc:.4f}")

    # 分说话人计算 (仅记录融合分数的结果，若需记录消融项可参照此逻辑扩展)
    for spk, group in res_df.groupby('speaker'):
        eer, _ = compute_eer(group['score'], group['label'])
        auc = roc_auc_score(group['label'], group['score'])
        if eer is not None:
            metrics.append({
                'speaker': spk, 'task': 'Spoof_Detection', 'eer': eer, 'auc': auc
            })

    print("=" * 30)
    return metrics


def save_results(results, name):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    results = run_poi_specificity_experiment(
        enroll_path="./protocols/zh_famous_enroll.csv",
        test_path="./protocols/zh_famous_test.csv",
        audio_root="./zh_famous/wav",
    )
    save_results(results, f"results/zh_famous_{BACKBONE}.txt")

    results = run_poi_specificity_experiment(
        enroll_path="./protocols/famous_figures_enroll.csv",
        test_path="./protocols/famous_figures_test.csv",
        audio_root="./FamousFigure/wav",
    )
    save_results(results, f"results/famous_figures_{BACKBONE}.txt")
