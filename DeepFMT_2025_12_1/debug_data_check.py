import torch
import numpy as np
import sys
import os
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from src.config import Config
from src.dataset import DynamicSeismicDataset

def check():
    print("=== 开始数据体检 ===")
    print(f"Data Root: {Config.DATA_ROOT}")
    
    # 1. 加载数据集
    ds = DynamicSeismicDataset(split='train', augment=False)
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    print("正在读取一个 Batch...")
    try:
        # 解包 6 个返回值
        wav, geo, mt, mw, amp, mask = next(iter(loader))
    except Exception as e:
        print(f"❌ DataLoader 读取失败: {e}")
        return

    # 2. 检查 Waveform (波形)
    print(f"\n[1] Waveform 检查:")
    print(f"    Shape: {wav.shape}")
    print(f"    Min: {wav.min().item():.4f}")
    print(f"    Max: {wav.max().item():.4f}")
    print(f"    Mean: {wav.mean().item():.4f}")
    
    if torch.all(wav == 0):
        print("❌ 严重错误: 波形数据全是 0！")
        print("   原因推测: generate_data.py 生成时出错，或者 dataset.py 读取路径不对。")
    elif torch.isnan(wav).any():
        print("❌ 严重错误: 波形数据包含 NaN！")
    else:
        print("✅ 波形数据正常 (非全0，无NaN)。")

    # 3. 检查 Geometry (几何)
    print(f"\n[2] Geometry 检查:")
    print(f"    Sample: {geo[0, 0, :]}") # 打印第一个台站的几何
    if torch.all(geo == 0):
        print("❌ 严重错误: 几何特征全是 0！")
    else:
        print("✅ 几何特征正常。")

    # 4. 检查 Labels (MT & Mw)
    print(f"\n[3] Labels 检查:")
    print(f"    MT (First): {mt[0]}")
    print(f"    Mw (First): {mw[0]}")
    if torch.isnan(mt).any() or torch.isnan(mw).any():
        print("❌ 严重错误: 标签包含 NaN！")
    else:
        print("✅ 标签正常。")

    # 5. 检查 Log Amp
    print(f"\n[4] Log Amp 检查:")
    print(f"    Values: {amp.flatten()}")
    if (amp < -10).all():
        print("⚠️ 警告: Log Amp 全是极小值 (-9.0)，说明归一化时 event_max=0。")
    elif torch.isnan(amp).any():
        print("❌ 严重错误: Log Amp 包含 NaN！")
    else:
        print("✅ Log Amp 正常。")

if __name__ == "__main__":
    check()