# note: fairseq is not compatible with matplotlib(numpy>2.0), please try to use a separate env to run this code...

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import glob
import pickle

from analyzer import UniversalPhonemeAnalyzer, RobustProfileDetector

cuda_index = 6
DEVICE = f"cuda:{cuda_index}"
os.environ["HF_HUB_OFFLINE"] = "1"


class PhonemeSimilarityVisualizer(RobustProfileDetector):
    def __init__(self, feature_extractor):
        super().__init__(feature_extractor)
        self.spk_profiles = {}
        # 存储每个音素的所有特征向量，用于计算全局锚点
        self.raw_data = defaultdict(lambda: defaultdict(list))
        # 存储全局锚点: {phoneme: mean_vector}
        self.global_anchors = {}

    def enroll_speakers(self, spk_refs_dict):
        print(f"[*] Enrolling {len(spk_refs_dict)} speakers and building global anchors...")
        for spk_id, enroll_files in tqdm(spk_refs_dict.items()):
            enroll_files = enroll_files[:500]

            # 收集特征用于计算全局锚点
            for p in tqdm(enroll_files, desc=f"[*] Enrolling {spk_id}"):
                feats = self.extractor.get_phoneme_features_from_file(p)
                for item in feats:
                    self.raw_data[spk_id][item['char'].lower()].append(item['feature'])

    def compute_anchors_and_scores(self, min_samples_per_spk=10):
        """计算全局锚点及各说话人的特异性分数"""
        # 1. 计算每个音素的全局平均特征 (Anchor)
        phoneme_pool = defaultdict(list)
        for spk in self.raw_data:
            for phn, vecs in self.raw_data[spk].items():
                phoneme_pool[phn].extend(vecs)

        for phn, all_vecs in phoneme_pool.items():
            if len(all_vecs) >= 10:  # 至少有10个样本才计算锚点
                self.global_anchors[phn] = np.mean(all_vecs, axis=0)

        # 2. 计算特异性分数 (1 - CosineSimilarity)
        # 结果结构: scores[spk][phn] = score
        spk_scores = defaultdict(dict)
        for spk in self.raw_data:
            for phn, vecs in self.raw_data[spk].items():
                if phn in self.global_anchors and len(vecs) >= min_samples_per_spk:
                    # 计算该人该音素的平均特征
                    spk_phn_mean = np.mean(vecs, axis=0).reshape(1, -1)
                    anchor = self.global_anchors[phn].reshape(1, -1)

                    # 计算余弦距离作为特异性指标
                    sim = cosine_similarity(spk_phn_mean, anchor)[0][0]
                    spk_scores[spk][phn] = 1 - sim  # 距离越大，越独特

        return spk_scores

def plot_heatmap(spk_scores, top_k=8, save_path="cosine_specificity_heatmap.pdf"):
    """提取 Top-K 并集并绘图，包含列归一化逻辑"""
    # 1. 提取每人 Top-K 的并集
    union_phonemes = set()
    for spk in spk_scores:
        # 排序获取得分最高的音素
        sorted_phns = sorted(spk_scores[spk].items(), key=lambda x: x[1], reverse=True)
        union_phonemes.update([p[0] for p in sorted_phns[:top_k]])

    sorted_phns = sorted(list(union_phonemes))
    sorted_spks = sorted(list(spk_scores.keys()))

    raw_spks = sorted(list(spk_scores.keys()))
    formatted_spks = [s.replace("_", " ") for s in raw_spks]

    # 2. 构建 DataFrame
    plot_matrix = []
    for spk in sorted_spks:
        # 使用 np.nan 替代 0，可以避免在计算最小值时受缺失数据干扰
        row = [spk_scores[spk].get(phn, np.nan) for phn in sorted_phns]
        plot_matrix.append(row)

    df = pd.DataFrame(plot_matrix, index=formatted_spks, columns=[f"/{p}/" for p in sorted_phns])

    # 步骤 A: 列标准化 (Column-wise Z-Score)
    # 意义：抹平音素本身的难度差异，数值变为“偏离全人群平均值的程度（标准差单位）”
    df_norm = df.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)

    # 步骤 B: 行归一化 (Row-wise Min-Max)
    # 意义：在每个说话人自己的维度里，寻找他最突出的特征。
    # 0 代表该音素是该说话人最“平庸”的音，1 代表该音素是该说话人最“招牌”的音。
    df_norm = df_norm.apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0,
        axis=1
    )
    # 将缺失值填充为 0（代表没有独特性或数据不足）
    df_norm = df_norm.fillna(0)

    # 3. 绘图
    plt.figure(figsize=(12, 7))
    # 使用修正后的 df_norm
    sns.heatmap(
        df_norm,
        annot=False,
        cmap="YlOrRd",
        # cbar_kws={'label': 'Relative Uniqueness (Normalized 0-1 per Phoneme)'}
    )

    # plt.title(f"Speaker-Specific Phoneme Uniqueness Map (Column-Normalized, Top-{top_k} Union)", fontsize=16)
    # plt.xlabel("Phonemes", fontsize=15)
    # plt.ylabel("Speakers", fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=13)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"[*] Heatmap saved to {save_path}")
    plt.show()


# --- 执行脚本 ---
if __name__ == "__main__":

    # 准备 10 位说话人的数据
    files = glob.glob("./famous_figures_samples/**/*.wav", recursive=True)
    spk_refs = defaultdict(list)
    for path in files:
        spk, label, _ = path.split("/")[-3:]
        # if label == "spoof": continue
        spk_refs[spk].append(path)

    # 注册并绘图
    if os.path.exists("./scores.pkl"):
        with open("./scores.pkl", "rb") as f:
            scores = pickle.load(f)
    else:
        analyzer = UniversalPhonemeAnalyzer(device=DEVICE)
        visualizer = PhonemeSimilarityVisualizer(analyzer)
        visualizer.enroll_speakers(spk_refs)
        scores = visualizer.compute_anchors_and_scores()
        with open("scores.pkl", "wb") as f:
            pickle.dump(scores, f)

    plot_heatmap(scores)