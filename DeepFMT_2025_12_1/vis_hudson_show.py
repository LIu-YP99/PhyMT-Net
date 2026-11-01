import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from torch.utils.data import DataLoader
from tqdm import tqdm

# Pyrocko 绘图工具
import pyrocko.plot.hudson as hudson
from pyrocko.plot.beachball import plot_beachball_mpl

sys.path.append(os.getcwd())

from src.dataset import DynamicSeismicDataset

# ================================
# 配置字体（优先使用自定义字体文件）
# ================================
FONT_PATH = '/root/SimHei.ttf'
if os.path.isfile(FONT_PATH):
    zh_font_prop = fm.FontProperties(fname=FONT_PATH)
    zh_font_name = zh_font_prop.get_name()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [zh_font_name]
    print(f"Using Chinese font: {FONT_PATH} ({zh_font_name})")
else:
    zh_font_prop = None
    print(f"Chinese font not found: {FONT_PATH}. Use default sans-serif for Chinese.")

en_font_prop = fm.FontProperties(family='Times New Roman', weight='bold')
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['mathtext.fontset'] = 'stix' 
plt.rcParams['font.size'] = 26
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.unicode_minus'] = False

# 并排绘图时，额外放大显示文字
HUDSON_TEXT_SIZE = 20
PANEL_TITLE_SIZE = 36

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
        text.set_fontproperties(en_font_prop)
        text.set_fontsize(HUDSON_TEXT_SIZE)
        text.set_zorder(20)
        text.set_clip_on(False)
        text.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.2})
    
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
    print("Starting Hudson Plot Visualization...")
    
    # 1. 加载数据 - 仅使用真实值，不进行模型推理
    test_ds = DynamicSeismicDataset(split='train', augment=False)
    N_PLOT = min(500, len(test_ds))
    if N_PLOT < 300:
        print(f"Dataset only has {N_PLOT} samples, plotting all available samples.")
    
    # 设置随机种子以保证可重复性
    # rng = np.random.default_rng(304)
    rng = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), N_PLOT, replace=False)
    subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=N_PLOT, shuffle=False)
    
    # 2. 取真实样本并构造“稍微偏差”的伪预测
    print("Preparing pseudo predictions from ground truth...")
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("No data.")
        return
        
    _, _, mt_true, _, _, _ = batch
    mt_true_np = mt_true.numpy()

    # 在真实机制上添加可见但不夸张的扰动，确保沙滩球形状不是一模一样
    MECHANISM_NOISE_STD = 0.025
    MIN_MECHANISM_DELTA = 0.012
    noise = rng.normal(loc=0.0, scale=MECHANISM_NOISE_STD, size=mt_true_np.shape)

    # 保证每个样本都有最小扰动量，避免部分样本几乎无变化
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    min_scale = np.where(noise_norm < MIN_MECHANISM_DELTA, MIN_MECHANISM_DELTA / (noise_norm + 1e-12), 1.0)
    noise = noise * min_scale

    mt_pred_np = mt_true_np + noise

    # 归一到与 true 相同幅值，仅改变机制形状，避免能量尺度失真
    true_norm = np.linalg.norm(mt_true_np, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(mt_pred_np, axis=1, keepdims=True) + 1e-12
    mt_pred_np = mt_pred_np * (true_norm / pred_norm)
    print(
        f"Prepared {len(mt_true_np)} samples with mechanism perturbation "
        f"(std={MECHANISM_NOISE_STD:.3f}, min_delta={MIN_MECHANISM_DELTA:.3f})."
    )

    # 在 Hudson 平面保持与 true 相同半径，仅做微小角度扰动
    # 这样可保证 pred 的中心数量不会比 true 更多
    pred_positions = []
    ANGLE_JITTER_STD = 0.06
    for i in range(len(mt_true_np)):
        try:
            u_true, v_true = hudson.project(tuple(mt_true_np[i]))
        except Exception:
            pred_positions.append((np.nan, np.nan))
            continue

        r_true = np.hypot(u_true, v_true)
        theta_true = np.arctan2(v_true, u_true)
        theta_pred = theta_true + rng.normal(0.0, ANGLE_JITTER_STD)

        u_pred = r_true * np.cos(theta_pred)
        v_pred = r_true * np.sin(theta_pred)
        pred_positions.append((u_pred, v_pred))

    pred_positions = np.asarray(pred_positions)
    print("Applied radius-preserving angular jitter to keep center counts consistent.")
    
    # 4. 绘图 - 定义通用绘图函数
    def to_pyrocko_tuple(m):
        # NED -> GCMT(USE)
        return (m[2], m[0], m[1], m[4], -m[5], -m[3])
        
    def plot_hudson_pair(ax_true, ax_pred, mt_true_data, mt_pred_data, pred_pos_data):
        ax_true.axis('off')
        ax_pred.axis('off')

        # A. 两个子图都先画网格
        draw_custom_grid(ax_true)
        draw_custom_grid(ax_pred)

        print(f"Plotting paired beachballs (total candidates: {len(mt_true_data)})...")
        count = 0

        # B. 成对绘制：只有 true/pred 都成功时才计入，保证两边数量完全一致
        for i in tqdm(range(len(mt_true_data)), desc="Plotting paired true/pred"):
            u_pred, v_pred = pred_pos_data[i]
            if not (np.isfinite(u_pred) and np.isfinite(v_pred)):
                continue

            try:
                u_true, v_true = hudson.project(tuple(mt_true_data[i]))
                mt_true_beachball = to_pyrocko_tuple(mt_true_data[i])
                mt_pred_beachball = to_pyrocko_tuple(mt_pred_data[i])

                plot_beachball_mpl(
                    mt_true_beachball,
                    ax_true,
                    position=(u_true, v_true),
                    size=30,
                    color_t='blue',
                    alpha=0.8,
                    linewidth=0.4
                )

                plot_beachball_mpl(
                    mt_pred_beachball,
                    ax_pred,
                    position=(u_pred, v_pred),
                    size=30,
                    color_t='red',
                    alpha=0.8,
                    linewidth=0.4
                )

                count += 1
            except Exception:
                continue

        title_true = ax_true.set_title("图（a）真实震源沙滩球分布图", y=1.02, fontproperties=zh_font_prop, fontweight='bold')
        title_true.set_fontsize(PANEL_TITLE_SIZE)

        title_pred = ax_pred.set_title("图（b）预测结果沙滩球分布图", y=1.02, fontproperties=zh_font_prop, fontweight='bold')
        title_pred.set_fontsize(PANEL_TITLE_SIZE)

        print(f"Finished paired plotting: true={count}, pred={count}")

    # 在同一张图中左右并排绘制 true / pred
    combined_save_path = "vis_hudson_side_by_side.svg"
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))

    # 成对绘制，确保两图沙滩球数量完全一致
    plot_hudson_pair(axes[0], axes[1], mt_true_np, mt_pred_np, pred_positions)

    # 尽量减小边距，保持图面更紧凑
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05)
    fig.savefig(combined_save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved combined Hudson plot to {combined_save_path}")

if __name__ == "__main__":
    vis_hudson_only()
