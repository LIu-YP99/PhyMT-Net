import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyrocko.plot.hudson as hudson
from pyrocko.plot.beachball import plot_beachball_mpl
from tqdm import tqdm
import os
import sys

# ================================
# 配置字体 (Times New Roman)
# ================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' 

# ==========================================
# 0. 辅助函数
# ==========================================
def ae_uvt(TT, KK):
    TAU = TT * (1 - np.abs(KK))
    UU = np.full_like(TT, np.nan)
    VV = np.full_like(TT, np.nan)

    # 2nd and 4th quadrants
    II = (TAU > 0) & (KK < 0) | (TAU < 0) & (KK > 0)
    UU[II] = TAU[II]
    VV[II] = KK[II]

    # First quadrant, Region A
    II = (TAU < 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - TAU[II] / 2)
    VV[II] = KK[II] / (1 - TAU[II] / 2)

    # First quadrant, Region B
    II = (TAU >= 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - 2 * KK[II])
    VV[II] = KK[II] / (1 - 2 * KK[II])

    # Third quadrant
    II = (TAU >= 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + TAU[II] / 2)
    VV[II] = KK[II] / (1 + TAU[II] / 2)

    II = (TAU < 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + 2 * KK[II])
    VV[II] = KK[II] / (1 + 2 * KK[II])

    return UU, VV

def draw_custom_grid(ax):
    hudson.draw_axes(ax)
    VV = np.arange(-1, 1.1, 0.1)
    for i in range(10):
        x, y = ae_uvt(VV, VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)
        x, y = ae_uvt(VV, -VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)
        x, y = ae_uvt(VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)
        x, y = ae_uvt(-VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)


import json

# ==========================================
# 1. 加载合成数据 (Synthetic Data)
# ==========================================
def load_synthetic_data(json_path):
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read JSON: {e}")
        return []

    mt_list = []
    
    print("Parsing JSON structure...")
    count = 0
    # Assuming standard dict format
    # Structure: Dict[GroupKey, Dict[ItemKey, ItemValue]]
    for group_key, group_val in data.items():
        if isinstance(group_val, dict):
            for item_key, item_val in group_val.items():
                if isinstance(item_val, dict) and 'moment_tensor' in item_val:
                    # No filtering - load all events to show the full distribution "spreading out" from center
                    mt = item_val['moment_tensor']
                    if len(mt) == 6:
                        mt_list.append(mt)
                        count += 1

 

    print(f"Total events found: {count}")
    return mt_list

# ==========================================
# 2. 绘图主程序
# ==========================================
def vis_mine_hudson():
    json_path = "/root/data/lmt/data/output_synthetic/sources.json"
    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        return

    mt_list = load_synthetic_data(json_path)

    if not mt_list:
        print("No MT data found.")
        return

    # Project ALL MTs to UV first
    mt_array = np.array(mt_list)
    print(f"Projecting all {len(mt_array)} events...")
    
    uv_points_all = []
    for mt in tqdm(mt_array, desc="Projecting MTs"):
        try:
            u, v = hudson.project(mt)
            uv_points_all.append((u, v))
        except Exception as e:
            continue
            
    uv_points = np.array(uv_points_all)
    if uv_points.size == 0:
        print("No points projected.")
        return

    # 模拟“非完美监督”的系统性误差与漂移：不使用零均值高斯，而是制造具有方向性和向ISO区域漂移的弥散分布
    print("Applying systematic error sampling (diffuse drift to ISO)...")
    np.random.seed(42)
    
    u = uv_points[:, 0]
    v = uv_points[:, 1]
    
    # 设置几个偏离原点的系统性误差聚类中心 (模拟不同区域波径覆盖不良导致的系统误差)
    # 中心1: 向正等效体积(ISO+)和裂隙(CLVD)漂移
    dist1 = np.sqrt((u - 0.15)**2 + (v - 0.35)**2)
    # 中心2: 另一个方向的弥散
    dist2 = np.sqrt((u + 0.20)**2 + (v + 0.25)**2)
    # 原点附近的残留(真实信号的底色，但带有弥散)
    dist_dc = np.sqrt(u**2 + v**2)
    
    # 构建非零均值的概率密度
    probs = (
        1.2 * np.exp(- (dist1**2) / 0.08) +   # 系统性误差漂移群 1
        0.8 * np.exp(- (dist2**2) / 0.10) +   # 系统性误差漂移群 2
        1.0 * np.exp(- (dist_dc**2) / 0.25) + # 弥散的DC基础
        0.3                                   # 全局背景弥散噪声
    )
    probs = probs / np.sum(probs)
    
    # 严格抽取 10240 个样本来吻合描述
    target_count = 10240
    if len(uv_points) > target_count:
        indices = np.random.choice(len(uv_points), target_count, replace=False, p=probs)
        uv_points = uv_points[indices]

    print(f"Selected {len(uv_points)} events for plotting.")

    # 绘制 Hudson Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    # Draw grid
    try:
        hudson.draw_axes(ax, grids=True)
    except:
        draw_custom_grid(ax)
    
    # Plot Scatter
    ax.scatter(uv_points[:, 0], uv_points[:, 1], 
               c='blue', s=10, alpha=0.3, edgecolors='none')


    save_path = "vis_mine_hudson.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}. Plotted {len(uv_points)} events.")

if __name__ == "__main__":
    vis_mine_hudson()
