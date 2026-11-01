import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Pyrocko 绘图工具
import pyrocko.plot.hudson as hudson
from pyrocko.plot.beachball import plot_beachball_mpl

# 确保路径正确
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet

# ==========================================
# 1. 您提供的核心绘图算法 (完全复刻)
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

    # 忽略 NaN 错误，防止画图中断
    # if np.any(np.isnan(VV)):
    #     raise ValueError("Error plotting point in [uu, vv] space.")

    return UU, VV

def draw_custom_grid(ax):
    """
    使用您的算法绘制 Hudson 网格
    """
    # 绘制基础坐标轴外框
    hudson.draw_axes(ax)
    
    # 创建 VV 数组
    VV = np.arange(-1, 1.1, 0.1)

    # 您的循环逻辑
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
# 2. 评估与绘图逻辑
# ==========================================
def evaluate_and_plot_beachballs():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting evaluation with Beachball Hudson Plot...")
    
    # 1. 加载数据
    # 使用测试集
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    # 为了不在图中画成一团黑，我们只随机抽取 50 个样本进行展示
    # 这样既能看清分布，也能看清每个球
    indices = np.random.choice(len(test_ds), 500, replace=False)
    subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=500, shuffle=False)
    
    # 2. 加载模型
    model = OptimalFMTNet().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251201_2112.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251201_2112.pth") # 兼容
        
    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 3. 推理
    try:
        batch = next(iter(loader))
    except StopIteration:
        print("No data.")
        return
        
    wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
    
    with torch.no_grad():
        mt_pred, mw_pred = model(wav, geo, amp, mask)
    
    # 转 Numpy
    mt_true_np = mt_true.cpu().numpy() # (B, 6) NED
    mt_pred_np = mt_pred.cpu().numpy()
    mw_true_np = mw_true.cpu().numpy().flatten()
    mw_pred_np = mw_pred.cpu().numpy().flatten()
    
    # 4. 绘图
    fig = plt.figure(figsize=(16, 8))
    
    # --- 左图：震级回归 (保留作为参考) ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(mw_true_np, mw_pred_np, alpha=0.6, s=20, c='blue', edgecolors='k')
    ax1.plot([3, 7], [3, 7], 'r--', lw=2)
    mae = np.mean(np.abs(mw_true_np - mw_pred_np))
    ax1.set_title(f"Magnitude Regression (MAE={mae:.3f})", fontsize=14)
    ax1.set_xlabel("True Mw"); ax1.set_ylabel("Predicted Mw")
    ax1.grid(True)
    
    # --- 右图：Hudson Beachball Plot (您的要求) ---
    # 使用 add_axes 或者直接 subplot
    ax2 = fig.add_subplot(1, 2, 2)
    # 移除默认边框，使用 Hudson 专用轴
    ax2.axis('off')
    
    # A. 绘制您的自定义网格
    draw_custom_grid(ax2)
    
    # B. 绘制沙滩球
    print("Plotting beachballs on Hudson grid...")
    
    # 定义 NED -> GCMT(USE) 转换函数 (Pyrocko 画球需要 USE 顺序: r, t, p, rt, rp, tp)
    # 我们的输入是 NED: xx, yy, zz, xy, xz, yz
    # 转换: r=-z, t=-x, p=y
    # Mrr=Mzz, Mtt=Mxx, Mpp=Myy, Mrt=Mxz, Mrp=-Myz, Mtp=-Mxy
    def to_pyrocko_tuple(m):
        return (m[2], m[0], m[1], m[4], -m[5], -m[3])
    
    # 我们画两组：
    # 1. 真实值 (Ground Truth): 用灰色显示
    # 2. 预测值 (Prediction): 用红色显示
    
    # 仅绘制 Prediction (红色) 以保持图面清晰？
    # 或者两者都画并用线连起来？
    # 为了清晰，这里演示绘制 **预测值 (Prediction)** 的沙滩球
    # 这样可以看出模型预测出的球，是否落在了 Hudson 图正确的位置
    
    for i in range(len(mt_pred_np)):
        # 1. 投影坐标 (直接传入 NED 元组给 project 即可，Pyrocko会自动处理)
        try:
            # hudson.project 接受 (mnn, mee, mdd, mne, mnd, med) 即 NED
            # 所以直接传原始 m 即可
            u, v = hudson.project(tuple(mt_pred_np[i]))
        except:
            continue
            
        # 2. 准备画球的数据 (需要转换为 USE/GCMT 格式)
        mt_beachball = to_pyrocko_tuple(mt_pred_np[i])
        
        # 3. 绘制沙滩球
        # size: 控制球的大小
        # position: (u, v) 投影坐标
        # color: 红色
        try:
            plot_beachball_mpl(
                mt_beachball, 
                ax2, 
                position=(u, v), 
                size=10, # 球的大小，根据图的疏密调整
                color_t='red', 
                alpha=0.8,
                linewidth=0.5
            )
        except Exception as e:
            pass

    ax2.set_title("Source Type Distribution (Predictions)", fontsize=14, y=1.05)
    
    # 添加图例说明
    # ax2.text(-1.3, 1.0, "Red Beachballs = Model Predictions", fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_file = "hudson_beachballs_custom.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_file}")

if __name__ == "__main__":
    evaluate_and_plot_beachballs()