# src/ablation_models.py

import torch
import torch.nn as nn

class FMNet(nn.Module):
    """
    FMNet baseline (PyTorch version).
    Architecture faithfully reproduced from the provided Keras code.
    - 16 trainable layers (Conv2D + MaxPooling + LeakyReLU + BatchNorm)
    - Encoder: 7 downsampling blocks
    - Decoder: 7 upsampling blocks
    - Output: 6-D moment tensor (adapted from 3 angles for fair comparison)
    """
    def __init__(self):
        super().__init__()
        
        # ===== Encoder (Downsampling) =====
        self.encoder = nn.Sequential(
            # Block 1: (1, 48, 128) -> (64, 24, 64)
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 2: (64, 24, 64) -> (64, 12, 32)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 3: (64, 12, 32) -> (64, 6, 16)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 4: (64, 6, 16) -> (64, 3, 8)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 5: (64, 3, 8) -> (64, 1, 4)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(3, 2), stride=(3, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 6: (64, 1, 4) -> (128, 1, 2)
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128, momentum=0.99, eps=0.001),
            
            # Block 7 (Bottleneck): (128, 1, 2) -> (128, 1, 1)
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=1),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128, momentum=0.99, eps=0.001),
        )
        
        # ===== Decoder (Upsampling) =====
        self.decoder = nn.Sequential(
            # Block 1: (128, 1, 1) -> (128, 1, 2)
            nn.Conv2d(128, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128, momentum=0.99, eps=0.001),
            
            # Block 2: (128, 1, 2) -> (128, 1, 4)
            nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128, momentum=0.99, eps=0.001),
            
            # Block 3: (128, 1, 4) -> (64, 1, 8)
            nn.Conv2d(128, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 4: (64, 1, 8) -> (64, 3, 16)
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 5: (64, 3, 16) -> (64, 6, 32)
            nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64, momentum=0.99, eps=0.001),
            
            # Block 6: (64, 6, 32) -> (32, 12, 64)
            nn.Conv2d(64, 32, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32, momentum=0.99, eps=0.001),
            
            # Block 7: (32, 12, 64) -> (16, 24, 128)
            nn.Conv2d(32, 16, kernel_size=(1, 3), stride=1, padding=1),
            nn.Upsample(scale_factor=(1, 2)),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16, momentum=0.99, eps=0.001),
        )
        
        # ===== Output Head (6-D MT) =====
        self.head_mt = nn.Conv2d(16, 6, kernel_size=(1, 3), stride=1, padding=1)
        
        # ===== Bottleneck head (for comparison) =====
        self.bottleneck = nn.Sequential(
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 6)
        )

    def forward(self, wav, geo=None, amp=None, mask=None):
        """
        Args:
            wav: (B, S, C, T) - 输入波形
        Returns:
            mt_norm: (B, 6) - 归一化矩张量
            None: 占位（与你的模型接口一致）
        """
        B, S, C, T = wav.shape
        # 重塑为 FMNet 期望的输入: (B, 1, S*C, T)
        x = wav.view(B, 1, S * C, T)  # (B, 1, 48, 128)
        
        # Encode
        encoded = self.encoder(x)  # (B, 128, 1, 1)
        
        # Bottleneck output (for feature analysis)
        bottleneck = encoded
        
        # Decode
        decoded = self.decoder(encoded)  # (B, 16, 24, 128)
        
        # MT output
        mt_raw = self.head_mt(decoded).mean(dim=[2, 3])  # (B, 6)
        
        # Normalize to unit sphere
        mt_norm = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        
        return mt_norm, None  # None 占位，表示不输出 Mw