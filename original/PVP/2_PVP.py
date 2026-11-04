import json
import os
import torch
import librosa
import random
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np

from speechbrain.inference.speaker import EncoderClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score

from analyzer import UniversalPhonemeAnalyzer, RobustProfileDetector
from utils import compute_eer

BACKBONE = "MMS"  # MMS, wav2vec_small, wav2vec_large, xlsr_1b, hubert_xlarge
cuda_index = 0
DEVICE = f"cuda:{cuda_index}"
# os.environ["HF_HUB_OFFLINE"] = "1"


asv_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)


# 继承你之前的检测器
class HybridPOIDetector(RobustProfileDetector):
    def __init__(self, feature_extractor):
        # 初始化原有的音素检测器
        super().__init__(feature_extractor)
        self.target_gmm = None  # 存储目标人的声纹GMM

    def _extract_embedding(self, wav_path):
        """提取全局 Speaker Embedding"""
        x, _ = librosa.load(wav_path, sr=16000)
        x = torch.from_numpy(x).float()
        embedding = asv_model.encode_batch(x).squeeze()
        return embedding.cpu().numpy()

    def _normalize_asv_scores(self, scores):
        center = -1800
        scale = 100 
        z = (scores - center) / scale
        z = np.clip(z, -500, 500)

        return 1 / (1 + np.exp(-z))

    def fit(self, reference_paths: list, top_k=12, verbose=False):
        """
        注册阶段：同时构建音素画像和声纹中心
        """
        # 1. 运行原有的音素级注册
        super().fit(reference_paths, top_k, verbose)

        # 2. 运行 ASV 注册 (计算 Embedding 质心)
        embeddings = []
        for path in tqdm(reference_paths, desc="enrolling", leave=False):
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

        # print(f"phn_score: {phn_score:.4f}, asv_score: {asv_score:.4f}, final_score: {final_score:.4f}")

        return final_score, phn_score, f"{phn_msg} + ASV"


def run_poi_specificity_experiment(enroll_path, test_path, audio_root):
    """
    针对 POI 特异性设计的实验主程序
    protocol 格式: [ref_key, test_key, label, speaker, attack_type]
    label: 1(Bonafide), 0(Spoof)
    """
    # 1. 加载数据
    enroll_df = pd.read_csv(enroll_path, sep=',', names=['speaker', 'path', 'source', 'label'], header=0)
    test_df = pd.read_csv(test_path, sep=',', names=['speaker', 'path', 'source', 'label'], header=0)
    analyzer = UniversalPhonemeAnalyzer(device=DEVICE, feat_model_name=BACKBONE)

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

        detector = HybridPOIDetector(analyzer)
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
            fused_score, phn_score, msg = detector.predict(test_path)
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
