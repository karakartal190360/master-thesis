import os
import torch
import torchaudio
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from huggingface_hub import PyTorchModelHubMixin


# === Wrapper for the SSL model ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Model config used to build SSL architecture
        cfg = Wav2Vec2Config(
            quantize_targets=True,
            extractor_mode="layer_norm",
            layer_norm_first=True,
            final_dim=768,
            latent_temp=(2.0, 0.1, 0.999995),
            encoder_layerdrop=0.0,
            dropout_input=0.0,
            dropout_features=0.0,
            dropout=0.0,
            attention_dropout=0.0,
            conv_bias=True,
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=4096,
            encoder_attention_heads=16,
            feature_grad_mult=1.0
        )
        # Initialize SSL model with random weights
        self.model = Wav2Vec2Model(cfg)

    def extract_feat(self, input_data):
        # If input has shape (B, T, 1), squeeze the last dim
        if input_data.ndim == 3:
            input_data = input_data[:, :, 0]
        # Extract features
        with torch.no_grad():
            features = self.model(input_data, mask=False, features_only=True)['x']
        return features

    def forward(self, input_data):
        return self.extract_feat(input_data)


# === The actual deepfake detection model using SSL frontend + FC backend ===
class DeepfakeDetector(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self):
        super().__init__()
        self.ssl_orig_output_dim = 1024
        self.num_classes = 2

        # Frontend: SSL model
        self.m_ssl = SSLModel()

        # Backend: Pooling + Classification
        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(output_size=1)
        self.proj_fc = torch.nn.Linear(
            in_features=self.ssl_orig_output_dim,
            out_features=self.num_classes,
        )

    def forward(self, wav):
        emb = self.m_ssl.extract_feat(wav)  # [B, T, D]
        emb = emb.transpose(1, 2)           # [B, D, T]
        pooled_emb = self.adap_pool1d(emb)  # [B, D, 1]
        pooled_emb = pooled_emb.squeeze(-1) # [B, D]
        logits = self.proj_fc(pooled_emb)   # [B, 2]
        return logits, pooled_emb

# # === Load AntiDeepfake model from Hugging Face===
# model = DeepfakeDetector.from_pretrained("nii-yamagishilab/wav2vec-large-anti-deepfake").m_ssl
# model.eval()
# x = torch.randn(10, 64320)
# logits = model(x)
# print(logits.shape)