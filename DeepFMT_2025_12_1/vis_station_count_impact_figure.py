import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.patches import Rectangle

# Ensure path is correct
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN
import matplotlib.font_manager as fm

import random
import numpy as np
import torch

# ============================================================
# ✅ 修改 1/5：全局随机种子（确保实验可复现）
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ================================
# 配置字体 (Times New Roman & SimHei)
# ================================
# fm.fontManager.addfont('/root/SimHei.ttf')
# my_font = fm.FontProperties(fname='/root/SimHei.ttf')
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['mathtext.fontset'] = 'stix'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 14 


# ==========================================
# 0. 路径修复
# ==========================================
potential_paths = [
    os.path.join(Config.PROJECT_ROOT, "data", "output_synthetic"),
    os.path.join(Config.PROJECT_ROOT, "../data", "output_synthetic"),
    "/root/data/lmt/data/output_synthetic"
]

found_path = None
for p in potential_paths:
    if os.path.exists(os.path.join(p, "waveforms.json")):
        found_path = p
        break

if found_path:
    print(f"Found data at: {found_path}")
    Config.DATA_ROOT = found_path
    Config.JSON_PATHS = {
        "waveforms": os.path.join(found_path, "waveforms.json"),
        "source":    os.path.join(found_path, "sources.json"),
        "grid":      os.path.join(found_path, "grid_points.json"),
        "receivers": os.path.join(found_path, "receivers.json") 
    }
else:
    print("Warning: Data directory not found. Using default from Config which might fail.")

# ==========================================
# 1. 辅助函数：下采样台站
# ==========================================
def subsample_stations(wav, geo, amp, mask, target_k):
    """
    从 batch 中随机选择 target_k 个有效台站。
    如果样本的有效台站数 < target_k，则标记为 invalid_batch_mask，后续丢弃。
    """
    B, S, C, N = wav.shape
    device = wav.device
    
    current_valid_counts = (~mask).sum(dim=1)
    valid_batch_indices = (current_valid_counts >= target_k)
    
    if not valid_batch_indices.any():
        return None, None, None, None, valid_batch_indices
        
    new_mask = torch.ones_like(mask)
    
    # ============================================================
    # ✅ 修改 2/5：使用固定种子的 Generator
    # ============================================================
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    
    rand_scores = torch.rand((B, S), device=device, generator=gen)
    rand_scores = rand_scores.masked_fill(mask, -1.0)
    _, top_k_indices = torch.topk(rand_scores, k=target_k, dim=1)
    
    new_mask.scatter_(1, top_k_indices, False)
    
    new_wav = wav.clone()
    new_wav.masked_fill_(new_mask.unsqueeze(2).unsqueeze(3), 0.0)
    
    new_geo = geo.clone()
    new_geo.masked_fill_(new_mask.unsqueeze(2), 0.0)
    
    return new_wav, new_geo, amp, new_mask, valid_batch_indices

# ==========================================
# 2. 主程序
# ==========================================
def evaluate_station_count_impact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading test dataset (augment=False)...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    model = ModelPlainCNN().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "ablation_plain_cnn_20260609_1456.pth")
    if not os.path.exists(ckpt_path):
        files = os.listdir(Config.SAVE_DIR)
        best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
        if best_models:
            best_models.sort(reverse=True)
            ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])

    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    station_counts = [4, 6, 8, 12, 16, 20, 24]
    results = {k: [] for k in station_counts}
    
    print(f"Starting evaluation for station counts: {station_counts}")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
            
            for k in station_counts:
                w, g, a, m, valid_idx = subsample_stations(wav, geo, amp, mask, k)
                
                if not valid_idx.any():
                    continue
                
                w_sub = w[valid_idx]
                g_sub = g[valid_idx]
                a_sub = a[valid_idx]
                m_sub = m[valid_idx]
                mt_true_sub = mt_true[valid_idx]
                
                mt_pred, _ = model(w_sub, g_sub, a_sub, m_sub)
                
                sim = torch.sum(mt_true_sub * mt_pred, dim=1)
                results[k].extend(sim.cpu().numpy())

    # ==========================================
    # 3. 绘图
    # ==========================================
    print("Plotting Boxplot...")
    
    data_to_plot = []
    labels = []
    medians = []
    
    for k in station_counts:
        sims = results[k]
        if len(sims) > 0:
            data_to_plot.append(sims)
            labels.append(str(k))
            medians.append(np.median(sims))
            print(f"Station Count {k}: N={len(sims)}, Median Sim={np.median(sims):.4f}")
        else:
            print(f"Station Count {k}: No samples found.")
            
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bp = ax.boxplot(data_to_plot, 
                    labels=labels, 
                    patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="orange", linewidth=2.0))

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
        
    ax.plot(np.arange(1, len(data_to_plot) + 1), medians, 'r--', label='Median Trend', alpha=0.7)
    
    # ============================================================
    # ✅ 修改 3/5：中文 → 英文
    # ============================================================
    # my_font_14 = fm.FontProperties(fname='/root/SimHei.ttf', size=14)
    # my_font_16 = fm.FontProperties(fname='/root/SimHei.ttf', size=16)
    
    ax.set_xlabel('Number of Stations', fontsize=14)
    ax.set_ylabel('Cosine Similarity of Focal Mechanism', fontsize=14)
    # ax.set_title('Impact of Station Density on Inversion Accuracy', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    
    # ============================================================
    # ✅ 修改 4/5 + 5/5：右下角 Inset 放大图 + 红色框
    # ============================================================
    inset_ax = ax.inset_axes([0.55, 0.2, 0.4, 0.35])
    
    inset_ax.boxplot(data_to_plot, 
                    labels=labels, 
                    patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="orange", linewidth=2.0),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
    
    inset_ax.plot(np.arange(1, len(data_to_plot) + 1), medians, 'r--', alpha=0.7, linewidth=2)
    
    # 放大范围：余弦相似度 0.75–1.02
    inset_ax.set_ylim(0.85, 1.02)
    inset_ax.set_xlim(0.5, len(data_to_plot) + 0.5)
    inset_ax.set_xticks(np.arange(1, len(data_to_plot) + 1))
    inset_ax.set_xticklabels(labels)
    # inset_ax.grid(True, linestyle=':', alpha=0.5)
    inset_ax.grid(False)
    inset_ax.set_title('Zoomed Boxplot', fontsize=12, fontweight='bold')
    
    # # 主图上画红色虚线框，标出放大区域
    # rect = Rectangle((0.5, 0.85), len(data_to_plot), 0.15, 
    #                  transform=ax.transData, 
    #                  facecolor='none', edgecolor='red', 
    #                  linestyle='--', linewidth=1.5)
    # ax.add_patch(rect)
    # 设置全局字体为 Times New Roman (按要求)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    plt.tight_layout()
    save_path = "vis_station_count_impact.svg"
    plt.savefig(save_path, dpi=600)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_station_count_impact()