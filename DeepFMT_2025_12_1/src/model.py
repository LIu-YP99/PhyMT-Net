import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config

class ResidualBlock1D(nn.Module):
    """
    1D 残差块：保留波形细节，允许构建更深的网络
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x) 
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class ResNetEncoder(nn.Module):
    """
    基于 ResNet 的波形特征提取器
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual Stages
        self.layer1 = ResidualBlock1D(32, 64, stride=2, dropout=dropout)
        self.layer2 = ResidualBlock1D(64, 128, stride=2, dropout=dropout)
        self.layer3 = ResidualBlock1D(128, 256, stride=2, dropout=dropout)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        # x: (B, 3, T)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_pool(x)
        return x.squeeze(-1) # Output dim: 256

class OptimalFMTNet(nn.Module): # 类名保持为 OptimalFMTNet 以兼容 train.py
    def __init__(self):
        super().__init__()
        
        # 1. 强力波形编码器 (ResNet)
        # 恢复你觉得好的 dropout 设置
        self.encoder = ResNetEncoder(dropout=0.2)
        cnn_out_dim = 256 
        
        # 2. 特征融合层
        self.geo_project = nn.Sequential(
            nn.Linear(Config.GEO_DIM, 64),
            nn.ReLU()
        )
        
        fusion_dim = cnn_out_dim + 64 # 320
        
        # 3. Transformer (不加 Mask，让它自己学)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, 
            nhead=8,                
            dim_feedforward=1024,
            dropout=0.2,            
            batch_first=True,
            norm_first=True         
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3) 
        
        # 4. Attention Pooling
        self.attn_pool = nn.Sequential(
            nn.Linear(fusion_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 5. 输出头
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

    # 【关键修改】兼容 train.py 的传参，但忽略 mask
    def forward(self, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        
        # 1. 提取波形特征
        feat_wav = self.encoder(wav.view(b*s, c, t)).view(b, s, -1)
        
        # 2. 融合几何特征
        feat_geo = self.geo_project(geo)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)
        
        # 3. Transformer
        feat = self.transformer(feat) # 不用 mask，让它自己学
        
        # 4. Pooling
        weights = self.attn_pool(feat) # (B, S, 1)
        global_vec = torch.sum(feat * weights, dim=1)
        
        # 5. 预测
        mt_raw = self.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        # 【关键修改】手动归一化 log_amp
        # 假设 log_amp 范围是 -9 到 -2 左右
        # (log_amp + 6) / 6 把它大致映射到 -0.5 到 0.5 附近
        log_amp_scaled = (log_amp + 6.0) / 6.0
        
        mw_in = torch.cat([global_vec, log_amp_scaled], dim=1)
        mw_pred = self.head_mw(mw_in)
        
        return mt_pred, mw_pred