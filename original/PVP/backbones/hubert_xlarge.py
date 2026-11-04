import os
import torch
import torchaudio
import fairseq
from huggingface_hub import PyTorchModelHubMixin


ssl_path = "pretrained_models/hubert_xtralarge_ll60k.pt"

# === Wrapper for the SSL model ===
class SSLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ssl_path])
        self.model = model[0]

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
        self.ssl_orig_output_dim = 1280
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
# model = DeepfakeDetector().m_ssl
# model.eval()
# x = torch.randn(10, 64320)
# logits = model(x)
# print(logits.shape)