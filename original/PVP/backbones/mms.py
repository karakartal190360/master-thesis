import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
from collections import OrderedDict

# to download checkpoint, see: https://huggingface.co/nii-yamagishilab/mms-300m-anti-deepfake

global_input_dim = 1024
mms_configs = {
    "_name": "wav2vec2",
    "extractor_mode": "layer_norm",
    "encoder_layers": 24,
    "encoder_embed_dim": 1024,
    "encoder_ffn_embed_dim": 4096,
    "encoder_attention_heads": 16,
    "activation_fn": "gelu",
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "activation_dropout": 0.0,
    "encoder_layerdrop": 0.0,
    "dropout_input": 0.0,
    "dropout_features": 0.0,
    "final_dim": 768,
    "layer_norm_first": True,
    "conv_feature_layers": "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
    "conv_bias": True,
    "logit_temp": 0.1,
    "quantize_targets": True,
    "quantize_input": False,
    "same_quantizer": False,
    "target_glu": False,
    "feature_grad_mult": 1.0,
    "quantizer_depth": 1,
    "quantizer_factor": 3,
    "latent_vars": 320,
    "latent_groups": 2,
    "latent_dim": 0,
    "mask_length": 10,
    "mask_prob": 0.65,
    "mask_selection": "static",
    "mask_other": 0.0,
    "no_mask_overlap": False,
    "mask_min_space": 1,
    "num_negatives": 100,
    "negatives_from_everywhere": False,
    "cross_sample_negatives": 0,
    "codebook_negatives": 0,
    "conv_pos": 128,
    "conv_pos_groups": 16,
    "latent_temp": [2.0, 0.1, 0.999995],
}


class MMS300M(torch.nn.Module):
    def __init__(self, freeze=False):
        """SSLModel(model_name)
        Args:
          model_name: string, pass to global configurations to get model specific config
        """
        super().__init__()
        # model-specific config
        cfg = Wav2Vec2Config(**mms_configs)
        # randomly initialized model
        self.model = Wav2Vec2Model(cfg)
        # load pretrained model weights
        checkpoint_path = "pretrained_models/mms_300m.ckpt"
        state_dict = torch.load(checkpoint_path)
        state = OrderedDict()
        for name, param in state_dict.items():
            if name.startswith("m_ssl.model."):
                state[name[12:]] = param
        self.model.load_state_dict(state)
        self.out_dim = global_input_dim

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def extract_feat(self, input_data):
        """feature = extract_feat(input_data)
        Args:
          input_data: tensor, waveform, (batch, T)

        Return:
          feature: tensor, feature, (batch, frame, hidden_dim)
        """
        if (
                next(self.model.parameters()).device != input_data.device
                or next(self.model.parameters()).dtype != input_data.dtype
        ):
            self.model.to(input_data.device, dtype=input_data.dtype)
            # self.model.eval()
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)["x"]
        return emb

    def forward(self, input_data):
        return self.extract_feat(input_data)


class DeepfakeDetector(torch.nn.Module):
    """ Model definition
    Args:
      model_name: string, pass to global configurations to get model specific config
    """

    def __init__(self):
        super().__init__()
        # number of output class
        self.v_out_class = 2

        ####
        # create network
        ####
        self.m_ssl = MMS300M(freeze=True)

        self.adap_pool1d = torch.nn.AdaptiveAvgPool1d(
            output_size=1
        )
        self.proj_fc = torch.nn.Linear(
            in_features=self.m_ssl.out_dim,
            out_features=self.v_out_class,
        )

        # load pretrained
        checkpoint_path = "pretrained_models/mms_300m.ckpt"
        state_dict = torch.load(checkpoint_path)
        state = OrderedDict()
        for name, param in state_dict.items():
            if name.startswith("proj_fc."):
                state[name[8:]] = param
        self.proj_fc.load_state_dict(state)

    def forward(self, wav):
        # [batch, frame, hidden_dim]
        emb = self.m_ssl.extract_feat(wav)
        # [batch, hidden_dim, frame]
        emb = emb.transpose(1, 2)
        # [batch, hidden_dim, 1]
        pooled_emb = self.adap_pool1d(emb)
        # [batch, hidden_dim]
        pooled_emb = pooled_emb.view(emb.size(0), -1)
        # [batch, 2]
        pred = self.proj_fc(pooled_emb)
        return pred, pooled_emb


if __name__ == "__main__":
    model = MMS300M()
    x = torch.randn(10, 64320)
    emb = model.extract_feat(x)
    print(emb.shape)