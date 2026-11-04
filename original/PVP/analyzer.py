import os
import torch
import librosa
import numpy as np
import pandas as pd
import json
import joblib
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from sklearn.mixture import GaussianMixture


class UniversalPhonemeAnalyzer:
    def __init__(
            self,
            asr_model_name: str = "facebook/wav2vec2-xlsr-53-espeak-cv-ft",
            feat_model_name: str = "MMS",
            device: str = None,
            granularity: str = "phoneme",
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.granularity = granularity

        # 加载 ASR 模型用于音素识别 (CTC)
        print(f"[*] Loading Aligner (CTC): {asr_model_name}")
        self.asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name).to(self.device).eval()

        # 加载特征提取模型
        print(f"[*] Loading Feature Extractor: {feat_model_name}")
        if feat_model_name == "MMS":
            from backbones.mms import DeepfakeDetector
            model = DeepfakeDetector().m_ssl
        elif feat_model_name == "wav2vec_large":
            from backbones.wav2vec_large import DeepfakeDetector
            model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-large-anti-deepfake").m_ssl
        elif feat_model_name == "wav2vec_small":
            from backbones.wav2vec_small import DeepfakeDetector
            model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-small-anti-deepfake").m_ssl
        elif feat_model_name == "xlsr_1b":
            from backbones.xlsr_1b import DeepfakeDetector
            model = DeepfakeDetector.from_pretrained("nii-yamagishilab/xls-r-1b-anti-deepfake").m_ssl
        elif feat_model_name == "hubert_xlarge":
            from backbones.hubert_xlarge import DeepfakeDetector
            model = DeepfakeDetector.from_pretrained("nii-yamagishilab/hubert-xlarge-anti-deepfake").m_ssl
        else:
            raise NotImplementedError


        self.feat_model = model.to(self.device).eval()

    def get_phoneme_features_from_file(self, audio_path: str) -> List[Dict]:
        try:
            audio, _ = librosa.load(audio_path, sr=16000)
            inputs = self.asr_processor(audio, return_tensors="pt", sampling_rate=16000)
            input_values = inputs.input_values.to(self.device)

            with torch.no_grad():
                # 1. 识别音素及其偏移量
                asr_logits = self.asr_model(input_values).logits
                predicted_ids = torch.argmax(asr_logits, dim=-1)
                # output_char_offsets=True 是关键，返回每个字符在序列中的 start/end frame
                transcription = self.asr_processor.decode(predicted_ids[0], output_char_offsets=True)

                features = self.feat_model(input_values).squeeze(0)

            raw_offsets = transcription.char_offsets
            if not raw_offsets:
                return []

            num_frames = features.shape[0]
            final_data = []

            for u in raw_offsets:
                s_idx, e_idx = u["start_offset"], u["end_offset"]
                # 确保索引不越界
                if s_idx >= num_frames: continue
                e_idx = min(e_idx + 1, num_frames)  # e_idx 通常是最后一帧，+1 包含该帧

                segment = features[s_idx:e_idx]
                if segment.size(0) > 0:
                    feat_mean = segment.mean(dim=0).cpu().numpy()
                    final_data.append({
                        "char": u["char"],
                        "feature": feat_mean
                    })
            return final_data
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []


class RobustProfileDetector:
    def __init__(self, feature_extractor, n_components: int=5):
        self.extractor = feature_extractor
        self.n_components = n_components

        # 存储 GMM 模型: {phoneme: GMM_object}
        self.phoneme_gmms = {}
        # 存储 GMM 的权重/置信度: {phoneme: weight}
        self.phoneme_weights = {}

        self.top_k_set = set()

        # Level 3: 粗糙画像 (大类) - 同样使用 GMM
        self.broad_class_gmms = {}
        self.broad_class_weights = {}

        # 定义音素大类映射
        try:
            with open("vocab.json", "r") as f:
                broad_map = json.load(f)
        except:
            # Fallback map if json not found
            broad_map = {}
        self.broad_map = broad_map
        self.char_to_class = {}
        for cls, chars in self.broad_map.items():
            for c in chars:
                self.char_to_class[c] = cls

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

    def _finalize_weights(self, phoneme_weights):
        weights = np.array(list(phoneme_weights.values()))
        w_min = weights.min()
        w_max = weights.max()

        # 将权重线性映射到 [0.1, 1.0]
        for phn in phoneme_weights:
            raw_w = phoneme_weights[phn]
            if w_max > w_min:
                normalized_w = 0.1 + 0.9 * (raw_w - w_min) / (w_max - w_min)
            else:
                normalized_w = 1.0
            phoneme_weights[phn] = normalized_w

    def fit(self, audio_paths: list, top_k=12, verbose=False):
        """
        audio_paths: list of audio_path (str)
        """
        # --- 收集数据 ---
        phn_data = defaultdict(list)
        class_data = defaultdict(list)

        if verbose:
            audio_paths = tqdm(audio_paths, desc="Aligning...")
        for audio_p in audio_paths:
            feats = self.extractor.get_phoneme_features_from_file(audio_p)
            for item in feats:
                vec = item['feature']
                char = item['char'].lower()

                phn_data[char].append(vec)

                # 收集大类
                found_class = None
                if char in self.char_to_class:
                    found_class = self.char_to_class[char]
                else:
                    clean_char = ''.join([i for i in char if not i.isdigit()])
                    if clean_char in self.char_to_class:
                        found_class = self.char_to_class[clean_char]

                if found_class:
                    class_data[found_class].append(vec)

        # --- 构建 Level 1 & 2 (具体音素 GMM) ---
        candidates = []
        if verbose:
            phn_data = tqdm(phn_data.items(), desc="GMM modeling for level1 & 2...")
        else:
            phn_data = phn_data.items()
        for phn, vecs in phn_data:
            if len(vecs) < 5: continue  # GMM 需要比均值更多一点的数据

            vecs = np.array(vecs)

            # 训练 GMM
            gmm = self._train_gmm(vecs, n_components=self.n_components)

            self.phoneme_gmms[phn] = gmm

            # 计算该 GMM 的“紧密度”或“置信度”
            # 用训练数据自身的平均 Log-Likelihood (LL) 来衡量
            # LL 越高，说明数据越集中，模型拟合越好
            avg_ll = gmm.score(vecs)

            # 权重设计：LL 越高权重越大
            # 注意 LL 是负数，需要处理一下
            weight = np.exp(avg_ll / 1000)  # 经验缩放
            self.phoneme_weights[phn] = weight

            candidates.append((phn, weight))

        # 选出 Top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        self.top_k_set = set([x[0] for x in candidates[:top_k]])

        if verbose:
            print(f"[*] Top K: {self.top_k_set}")
            class_data = tqdm(class_data.items(), desc="GMM modeling for level3...")
        else:
            class_data = class_data.items()
        # --- 构建 Level 3 (大类 GMM) ---
        for cls_name, vecs in class_data:
            if len(vecs) < 10: continue
            vecs = np.array(vecs)

            # 大类数据多，可以使用更多的分量
            gmm = self._train_gmm(vecs, n_components=self.n_components)
            self.broad_class_gmms[cls_name] = gmm

            avg_ll = gmm.score(vecs)
            self.broad_class_weights[cls_name] = np.exp(avg_ll / 1000)

        self._finalize_weights(self.phoneme_weights)
        self._finalize_weights(self.broad_class_weights)
        # print(f"weights normalized.")

    def predict(self, test_audio_path):
        feats = self.extractor.get_phoneme_features_from_file(test_audio_path)
        if not feats: return 0.5, "Extract Fail"

        # 容器
        l1_scores, l1_weights = [], []
        l2_scores, l2_weights = [], []
        l3_scores, l3_weights = [], []

        for item in feats:
            char = item['char'].lower()
            vec = item['feature']

            # Level 1 & 2 Check
            if char in self.phoneme_gmms:
                gmm = self.phoneme_gmms[char]
                raw_ll = self._compute_gmm_score(gmm, vec)

                # 归一化分数 [0, 1]
                # 这里比较 tricky，GMM 是生成模型，不直接给 0-1 的相似度
                # 我们可以用相对似然：和“通用背景模型 UBM”比，或者简单的 Sigmoid
                score = self._normalize_scores(raw_ll)
                w = self.phoneme_weights[char]

                l2_scores.append(score)
                l2_weights.append(w)

                if char in self.top_k_set:
                    l1_scores.append(score)
                    l1_weights.append(w)

            # Level 3 Check
            cls_name = None
            if char in self.char_to_class:
                cls_name = self.char_to_class[char]
            else:
                clean_char = ''.join([i for i in char if not i.isdigit()])
                if clean_char in self.char_to_class:
                    cls_name = self.char_to_class[clean_char]

            if cls_name and cls_name in self.broad_class_gmms:
                gmm = self.broad_class_gmms[cls_name]
                raw_ll = self._compute_gmm_score(gmm, vec)
                score = self._normalize_scores(raw_ll)
                w = self.broad_class_weights[cls_name]

                l3_scores.append(score)
                l3_weights.append(w)

        # --- 分级决策 ---
        if len(l1_scores) >= 1:
            return np.average(l1_scores, weights=l1_weights), "Level 1 (Top-K GMM)"
        elif len(l2_scores) >= 3:
            return np.average(l2_scores, weights=l2_weights), "Level 2 (All GMM)"
        elif len(l3_scores) >= 1:
            return np.average(l3_scores, weights=l3_weights), "Level 3 (Broad GMM)"
        else:
            return 0.5, "Level 4 (Failure)"

    def save(self, filepath):
        """只保存统计模型数据，排除 PyTorch 模块"""
        # 创建一个不包含 extractor 的状态字典
        state = {
            'n_components': self.n_components,
            'phoneme_gmms': self.phoneme_gmms,
            'phoneme_weights': self.phoneme_weights,
            'top_k_set': self.top_k_set,
            'broad_class_gmms': self.broad_class_gmms,
            'broad_class_weights': self.broad_class_weights,
            'broad_map': self.broad_map,
            'char_to_class': self.char_to_class
        }
        try:
            joblib.dump(state, filepath)
            print(f"成功保存 GMM 数据至: {filepath}")
        except Exception as e:
            print(f"保存失败: {e}")

    @classmethod
    def load(cls, filepath, feature_extractor):
        """加载数据并重新绑定 extractor"""
        try:
            state = joblib.load(filepath)
            # 先用旧 extractor 初始化一个空实例
            instance = cls(feature_extractor, n_components=state['n_components'])
            # 还原属性
            instance.phoneme_gmms = state['phoneme_gmms']
            instance.phoneme_weights = state['phoneme_weights']
            instance.top_k_set = state['top_k_set']
            instance.broad_class_gmms = state['broad_class_gmms']
            instance.broad_class_weights = state['broad_class_weights']
            instance.broad_map = state['broad_map']
            instance.char_to_class = state['char_to_class']
            return instance
        except Exception as e:
            print(f"加载失败: {e}")
            return None