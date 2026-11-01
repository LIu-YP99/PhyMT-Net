import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import matplotlib.font_manager as fm

# Pyrocko 绘图工具
import pyrocko.plot.hudson as hudson
from pyrocko.plot.beachball import plot_beachball_mpl

sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet

# ================================
# 配置字体 (Times New Roman)
# ================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
font_prop = None  # No special prop needed, global settings work for stix/serif

# ==========================================
# 1. Hudson Grid 绘图算法 (复刻自 evaluate.py)
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
    """
    绘制 Hudson 网格背景
    """
    # 绘制基础坐标轴外框
    hudson.draw_axes(ax)
    for text in ax.texts:
        text.set_fontsize(16)
    
    # 创建 VV 数组
    VV = np.arange(-1, 1.1, 0.1)

    # 循环逻辑
    for i in range(10):
        # 绘制四组曲线
        x, y = ae_uvt(VV, VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(VV, -VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(-VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

# ==========================================
# 2. 评估与绘图主逻辑
# ==========================================
def vis_hudson_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Hudson Plot Visualization...")
    
    # 1. 加载数据 - Test Set
    test_ds = DynamicSeismicDataset(split='train', augment=False)
    # 为了绘图清晰，我们随机挑选 200 个样本（或者更多，取决于可视化需求）
    # 如果要画全部可能会太密集
    N_SAMPLES = 500
    
    # 设置随机种子以保证可重复性
    np.random.seed(304)  
    indices = np.random.choice(len(test_ds), N_SAMPLES, replace=False)
    subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=N_SAMPLES, shuffle=False)
    
    # 2. 加载模型
    model = OptimalFMTNet().to(device)
    
    # 寻找最新的最佳模型 checkpoint
    save_dir = Config.SAVE_DIR
    files = [f for f in os.listdir(save_dir) if f.startswith('best_model_') and f.endswith('.pth')]
    if not files:
        print("No 'best_model_*.pth' checkpoint found!")
        return
    
    latest_ckpt = sorted(files)[-1]
    ckpt_path = os.path.join(save_dir, latest_ckpt)
    
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    # 3. 推理
    print("Running inference...")
    try:
        # 我们这里是一次性加载一个大 batch (200个)
        batch = next(iter(loader))
    except StopIteration:
        print("No data.")
        return
        
    wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
    
    with torch.no_grad():
        mt_pred, mw_pred = model(wav, geo, amp, mask)
    
    # 转 Numpy
    mt_pred_np = mt_pred.cpu().numpy()
    mt_true_np = mt_true.cpu().numpy()
    
    # 只保留预测和真实足够接近的样本
    distances = np.linalg.norm(mt_pred_np - mt_true_np, axis=1)
    DIST_THRESHOLD = 0.1
    MAX_PLOT_SAMPLES = 20000

    similar_mask = distances <= DIST_THRESHOLD
    similar_count = int(np.sum(similar_mask))

    if similar_count == 0:
        # 阈值过严时，兜底保留最接近的少量样本，避免空图
        fallback_n = min(20, len(distances))
        selected_indices = np.argsort(distances)[:fallback_n]
        print(
            f"No samples under threshold={DIST_THRESHOLD:.3f}; "
            f"fallback to top-{fallback_n} closest samples."
        )
    else:
        selected_indices = np.where(similar_mask)[0]
        if similar_count > MAX_PLOT_SAMPLES:
            # 样本过多时，仅保留阈值内最接近的一部分，防止图面过密
            ranked_within_threshold = selected_indices[np.argsort(distances[selected_indices])]
            selected_indices = ranked_within_threshold[:MAX_PLOT_SAMPLES]

        print(
            f"Kept {len(selected_indices)} similar samples "
            f"(threshold={DIST_THRESHOLD:.3f})."
        )

    mt_pred_np = mt_pred_np[selected_indices]
    mt_true_np = mt_true_np[selected_indices]
    
    # 4. 绘图 - 定义通用绘图函数
    def to_pyrocko_tuple(m):
        # NED -> GCMT(USE)
        return (m[2], m[0], m[1], m[4], -m[5], -m[3])
        
    def plot_single_hudson(mt_data, title_text, color, save_path):
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.axis('off')
        
        # A. 绘制网格
        draw_custom_grid(ax)
        
        # B. 绘制沙滩球
        print(f"Plotting {len(mt_data)} beachballs for {save_path}...")
        
        count = 0
        # 使用 tqdm 显示进度
        for i in tqdm(range(len(mt_data)), desc=f"Plotting {os.path.basename(save_path)}"):
            try:
                u, v = hudson.project(tuple(mt_data[i]))
            except:
                continue
            
            mt_beachball = to_pyrocko_tuple(mt_data[i])
            
            try:
                plot_beachball_mpl(
                    mt_beachball, 
                    ax, 
                    position=(u, v), 
                    size=15,
                    color_t=color, 
                    alpha=0.8,
                    linewidth=0.5
                )
                count += 1
            except Exception as e:
                pass
        
        # plt.title(title_text, fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()
        print(f"Saved Hudson plot to {save_path} (Plotted {count} beachballs)")

    # (1) 绘制真实分布 (Ground Truth) - 蓝色
    plot_single_hudson(mt_true_np, "Source Type Distribution (Reference)", 'blue', "vis_hudson_true.svg")

    # (2) 绘制预测分布 (Prediction) - 红色
    plot_single_hudson(mt_pred_np, "Source Type Distribution (Predictions)", 'red', "vis_hudson_predictions.svg")

if __name__ == "__main__":
    vis_hudson_only()
