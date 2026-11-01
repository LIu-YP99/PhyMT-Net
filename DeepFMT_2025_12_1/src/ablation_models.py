import torch
import torch.nn as nn
from .model import ResNetEncoder
from .config import Config

class PlainCNNEncoder(nn.Module):
    """
    普通CNN骨干 (Standard CNN backbone) - Without residual connections
    """
    def __init__(self, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ModelNoGeo(nn.Module):
    """消融实验 1: 无几何特征注入"""
    def __init__(self):
        super().__init__()
        self.encoder = PlainCNNEncoder(dropout=0.2)
        fusion_dim = 256  # Only CNN features, no geo projection
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=1024,
            dropout=0.2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) 
        
        self.attn_pool = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Softmax(dim=1))
        
        self.head_mt = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Linear(256, 6))
        self.head_mw = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim + 1, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        feat = self.encoder(wav.view(b*s, c, t)).view(b, s, -1)  # Only waveform feature
        
        feat = self.transformer(feat)
        weights = self.attn_pool(feat)
        global_vec = torch.sum(feat * weights, dim=1)
        
        mt_raw = self.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        log_amp_scaled = (log_amp + 6.0) / 6.0
        mw_pred = self.head_mw(torch.cat([global_vec, log_amp_scaled], dim=1))
        return mt_pred, mw_pred

class ModelNoTrans(nn.Module):
    """消融实验 2: 无 Transformer"""
    def __init__(self):
        super().__init__()
        self.encoder = PlainCNNEncoder(dropout=0.2)
        self.geo_project = nn.Sequential(nn.Linear(Config.GEO_DIM, 64), nn.ReLU())
        fusion_dim = 256 + 64
        
        # No Transformer! Directly to pooling
        # self.attn_pool = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Softmax(dim=1))
        # self.global_pool = feat.mean(dim=1)  # Global Average Pooling


        self.head_mt = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Linear(256, 6))
        self.head_mw = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim + 1, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        feat_wav = self.encoder(wav.view(b*s, c, t)).view(b, s, -1)
        feat_geo = self.geo_project(geo)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)
        
        # Skip transformer
        
        global_vec = feat.mean(dim=1)
        
        mt_raw = self.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        log_amp_scaled = (log_amp + 6.0) / 6.0
        mw_pred = self.head_mw(torch.cat([global_vec, log_amp_scaled], dim=1))
        return mt_pred, mw_pred

class ModelPlainCNN(nn.Module):
    """消融实验 3: 普通CNN骨干"""
    def __init__(self):
        super().__init__()
        self.encoder = PlainCNNEncoder(dropout=0.2)
        self.geo_project = nn.Sequential(nn.Linear(Config.GEO_DIM, 64), nn.ReLU())
        fusion_dim = 256 + 64
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=1024,
            dropout=0.2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) 
        
        self.attn_pool = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Softmax(dim=1))
        
        self.head_mt = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Linear(256, 6))
        self.head_mw = nn.Sequential(nn.Dropout(0.3), nn.Linear(fusion_dim + 1, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        feat_wav = self.encoder(wav.view(b*s, c, t)).view(b, s, -1)
        feat_geo = self.geo_project(geo)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)
        
        feat = self.transformer(feat)
        weights = self.attn_pool(feat)
        global_vec = torch.sum(feat * weights, dim=1)
        
        mt_raw = self.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        log_amp_scaled = (log_amp + 6.0) / 6.0
        mw_pred = self.head_mw(torch.cat([global_vec, log_amp_scaled], dim=1))
        return mt_pred, mw_pred

class ModelCoupledHead(nn.Module):
    """消融实验 4: 无解耦预测头 (Unified output head)"""
    def __init__(self):
        super().__init__()
        self.encoder = PlainCNNEncoder(dropout=0.2)
        self.geo_project = nn.Sequential(nn.Linear(Config.GEO_DIM, 64), nn.ReLU())
        fusion_dim = 256 + 64
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=8, dim_feedforward=1024,
            dropout=0.2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) 
        self.attn_pool = nn.Sequential(nn.Linear(fusion_dim, 1), nn.Softmax(dim=1))
        
        # Unified head for both MT (6) and Mw (1) -> 7 outputs
        self.head_unified = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        feat_wav = self.encoder(wav.view(b*s, c, t)).view(b, s, -1)
        feat_geo = self.geo_project(geo)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)
        
        feat = self.transformer(feat)
        weights = self.attn_pool(feat)
        global_vec = torch.sum(feat * weights, dim=1)
        
        log_amp_scaled = (log_amp + 6.0) / 6.0
        unified_in = torch.cat([global_vec, log_amp_scaled], dim=1)
        
        preds = self.head_unified(unified_in)
        mt_raw = preds[:, :6]
        mw_pred = preds[:, 6:7]
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        return mt_pred, mw_pred

# The full model is the original OptimalFMTNet from model.py
