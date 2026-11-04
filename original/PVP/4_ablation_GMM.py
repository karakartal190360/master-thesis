import os
import torch
import librosa
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics import roc_auc_score

from utils import compute_eer
from analyzer import UniversalPhonemeAnalyzer

BACKBONE = "MMS"
cuda_index = 2
DEVICE = f"cuda:{cuda_index}"
# os.environ["HF_HUB_OFFLINE"] = "1"

asv_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb",
    run_opts={"device": DEVICE},
)


class RobustProfileDetector:
    def __init__(self, feature_extractor, n_components: int = 5):
        self.extractor = feature_extractor
        # 消融实验中，n_components 不再用于训练 GMM，但保留参数接口以兼容代码
        self.n_components = n_components

        # 存储音素质心: {phoneme: mean_vector}
        self.phoneme_centroids = {}
        self.top_k_set = set()

        # Level 3: 大类质心
        self.broad_class_centroids = {}

        # 映射逻辑保持不变
        try:
            with open("vocab.json", "r") as f:
                broad_map = json.load(f)
        except:
            broad_map = {}
        self.broad_map = broad_map
        self.char_to_class = {}
        for cls, chars in self.broad_map.items():
            for c in chars:
                self.char_to_class[c] = cls

    def _compute_cosine_score(self, centroid, vector):
        """
        计算余弦相似度。范围 [-1, 1]，值越大越相似。
        """
        v1 = centroid.reshape(1, -1)
        v2 = vector.reshape(1, -1)
        # 手动计算以减少依赖
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0: return 0.0
        return np.dot(v1, v2.T)[0][0] / (norm1 * norm2 + 1e-6)

    def fit(self, audio_paths: list, top_k=12, verbose=False):
        phn_data = defaultdict(list)
        class_data = defaultdict(list)

        if verbose: audio_paths = tqdm(audio_paths, desc="Aligning...")
        for audio_p in audio_paths:
            feats = self.extractor.get_phoneme_features_from_file(audio_p)
            for item in feats:
                vec, char = item['feature'], item['char'].lower()
                phn_data[char].append(vec)

                cls_name = self.char_to_class.get(char) or self.char_to_class.get(
                    ''.join([i for i in char if not i.isdigit()]))
                if cls_name: class_data[cls_name].append(vec)

        # --- 消融建模：计算质心替代 GMM ---
        candidates = []
        for phn, vecs in phn_data.items():
            if len(vecs) < 3: continue
            vecs = np.array(vecs)

            # 计算质心 (Centroid)
            centroid = np.mean(vecs, axis=0)
            self.phoneme_centroids[phn] = centroid

            # 计算紧密度权重：使用训练样本到质心的平均余弦相似度
            sims = [self._compute_cosine_score(centroid, v) for v in vecs]
            weight = np.mean(sims)
            candidates.append((phn, weight))

        candidates.sort(key=lambda x: x[1], reverse=True)
        self.top_k_set = set([x[0] for x in candidates[:top_k]])

        # --- Level 3 大类建模 ---
        for cls_name, vecs in class_data.items():
            if len(vecs) < 5: continue
            vecs = np.array(vecs)
            centroid = np.mean(vecs, axis=0)
            self.broad_class_centroids[cls_name] = centroid

    def predict(self, test_audio_path):
        feats = self.extractor.get_phoneme_features_from_file(test_audio_path)
        if not feats: return 0.5, "Extract Fail"

        l1_scores, l2_scores, l3_scores = [], [], []

        for item in feats:
            char, vec = item['char'].lower(), item['feature']

            if char in self.phoneme_centroids:
                sim = self._compute_cosine_score(self.phoneme_centroids[char], vec)

                l2_scores.append(sim)
                if char in self.top_k_set:
                    l1_scores.append(sim)

            cls_name = self.char_to_class.get(char) or self.char_to_class.get(
                ''.join([i for i in char if not i.isdigit()]))
            if cls_name and cls_name in self.broad_class_centroids:
                sim = self._compute_cosine_score(self.broad_class_centroids[cls_name], vec)
                l3_scores.append(sim)

        if len(l1_scores) >= 1:
            return np.average(l1_scores), "Level 1 (Top-K Cosine)"
        elif len(l2_scores) >= 3:
            return np.average(l2_scores), "Level 2 (All Cosine)"
        elif len(l3_scores) >= 1:
            return np.average(l3_scores), "Level 3 (Broad Cosine)"
        return 0.5, "Level 4 (Failure)"


class HybridPOIDetector(RobustProfileDetector):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.target_asv_centroid = None  # 存储质心替代 GMM

    def _extract_embedding(self, wav_path):
        x, _ = librosa.load(wav_path, sr=16000)
        x = torch.from_numpy(x).float()
        embedding = asv_model.encode_batch(x).squeeze()
        return embedding.cpu().numpy()

    def fit(self, reference_paths: list, top_k=8, verbose=False):
        # 1. 音素级注册 (已在父类中改为 Cosine)
        super().fit(reference_paths, top_k, verbose)

        # 2. ASV 注册：计算均值向量
        embeddings = []
        for path in reference_paths:
            emb = self._extract_embedding(path)
            embeddings.append(emb)

        # 消融 GMM，只计算均值
        self.target_asv_centroid = np.mean(embeddings, axis=0)

    def predict(self, test_audio_path, alpha=0.8):
        # 1. 获取音素级 Cosine 分数
        phn_score, phn_msg = super().predict(test_audio_path)

        # 2. 获取 ASV Cosine 分数
        test_emb = self._extract_embedding(test_audio_path)
        asv_score = self._compute_cosine_score(self.target_asv_centroid, test_emb)

        final_score = alpha * phn_score + (1 - alpha) * asv_score
        return final_score, f"{phn_msg} + ASV_Cosine"


def run_poi_specificity_experiment(enroll_path, test_path, audio_root, analyzer):
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
            fused_score, msg = detector.predict(test_path)
            if msg == "Extract Fail": continue
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
    analyzer = UniversalPhonemeAnalyzer(device=DEVICE, feat_model_name=BACKBONE)

    results = run_poi_specificity_experiment(
        enroll_path="./protocols/zh_famous_enroll.csv",
        test_path="./protocols/zh_famous_test.csv",
        audio_root="./zh_famous/wav",
        analyzer=analyzer
    )
    save_results(results, f"results/zh_famous_{BACKBONE}.txt")

    results = run_poi_specificity_experiment(
        enroll_path="./protocols/famous_figures_enroll.csv",
        test_path="./protocols/famous_figures_test.csv",
        audio_root="./FamousFigure/wav",
        analyzer=analyzer
    )
    save_results(results, f"results/famous_figures_{BACKBONE}.txt")
