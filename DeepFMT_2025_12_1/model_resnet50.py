# src/model_resnet50.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import Config
from resnet50_1d import ResNet50Encoder1D

class model_resnet50(nn.Module):
    """
    ResNet50-based OptimalFMTNet.
    - Replace ResNetEncoder (3 blocks) with full ResNet50 (16 blocks).
    - Everything else (Geo, Transformer, Heads) stays identical.
    """
    def __init__(self):
        super().__init__()

        # 1. ResNet50 Encoder (replaces old 3-block ResNet)
        self.encoder = ResNet50Encoder1D(dropout=0.2)
        cnn_out_dim = 2048  # ResNet50 output

        # 2. Geo projection (unchanged)
        self.geo_project = nn.Sequential(
            nn.Linear(Config.GEO_DIM, 64),
            nn.ReLU()
        )

        fusion_dim = cnn_out_dim + 64  # 2112

        # 3. Transformer (unchanged)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 4. Attention Pooling (unchanged)
        self.attn_pool = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Softmax(dim=1)
        )

        # 5. Output heads (unchanged)
        self.head_mt = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

        self.head_mw = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape

        # 1. Extract waveform features
        feat_wav = self.encoder(wav.view(b * s, c, t)).view(b, s, -1)  # (B, S, 2048)

        # 2. Fuse geo features
        feat_geo = self.geo_project(geo)  # (B, S, 64)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)  # (B, S, 2112)

        # 3. Transformer
        if padding_mask is not None:
            feat = self.transformer(feat, src_key_padding_mask=padding_mask)
        else:
            feat = self.transformer(feat)

        # 4. Pooling
        weights = self.attn_pool(feat)  # (B, S, 1)
        global_vec = torch.sum(feat * weights, dim=1)  # (B, 2112)

        # 5. Predict MT
        mt_raw = self.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)

        # 6. Predict Mw
        log_amp_scaled = (log_amp + 6.0) / 6.0
        mw_in = torch.cat([global_vec, log_amp_scaled], dim=1)
        mw_pred = self.head_mw(mw_in)

        return mt_pred, mw_pred