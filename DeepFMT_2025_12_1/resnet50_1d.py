# src/resnet50_1d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck1D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)

        self.conv2 = nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(mid_channels)

        self.conv3 = nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))

        out += identity
        return self.relu(out)


class ResNet50Encoder1D(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Layer1: 64 → 256
        self.layer1 = self._make_layer(64, 256, num_blocks=3, stride=1, dropout=dropout)

        # Layer2: 256 → 512
        self.layer2 = self._make_layer(256, 512, num_blocks=4, stride=2, dropout=dropout)

        # ✅ Layer3: 512 → 1024（关键修正）
        self.layer3 = self._make_layer(512, 1024, num_blocks=6, stride=2, dropout=dropout)

        # Layer4: 1024 → 2048
        self.layer4 = self._make_layer(1024, 2048, num_blocks=3, stride=2, dropout=dropout)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, dropout):
        layers = []
        # First block with stride (downsampling)
        layers.append(Bottleneck1D(in_ch, out_ch, stride=stride, dropout=dropout))
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(Bottleneck1D(out_ch, out_ch, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return x.squeeze(-1)