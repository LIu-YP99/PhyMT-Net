import torch
import os
import sys
import glob
import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

# ================================
# 配置字体 (中文优先，避免 CJK 缺字警告)
# ================================
simhei_path = '/root/SimHei.ttf'
if os.path.exists(simhei_path):
    fm.fontManager.addfont(simhei_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'DejaVu Sans'] + plt.rcParams['font.sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet  # Import the ORIGINAL model
from src.ablation_models import ModelPlainCNN

def evaluate_magnitude_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting evaluation for Magnitude Regression (using Original Model)...",device)
    
    # 1. 加载数据 - Test Set
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 2. 加载模型 - Original Model
    model = ModelPlainCNN().to(device)
    
    # 寻找最新的最佳模型 checkpoint
    save_dir = Config.SAVE_DIR
    files = [f for f in os.listdir(save_dir) if f.startswith('best_model_') and f.endswith('.pth')]
    if not files:
        print("No 'best_model_*.pth' checkpoint found!")
        return
    
    # Sort to get the latest one
    latest_ckpt = sorted(files)[-1]
    ckpt_path = os.path.join(save_dir, "ablation_plain_cnn_20260609_1456.pth")
    
    print(f"Loading checkpoint: {ckpt_path}")
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    
    mw_preds = []
    mw_trues = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
            
            # Original model returns two outputs: mt_pred, mw_pred
            # We ignore mt_pred here
            _, mw_pred = model(wav, geo, amp, mask)
            
            mw_preds.extend(mw_pred.cpu().numpy().flatten())
            mw_trues.extend(mw.cpu().numpy().flatten())
            
    mw_preds = np.array(mw_preds)
    mw_trues = np.array(mw_trues)
    
    # ===================================
    # 3. 计算测试集原始指标
    # ===================================
    residuals = mw_preds - mw_trues

    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    test_bias = np.mean(residuals)

    print("\n========== Original Results ==========")
    print(f"Test Bias: {test_bias:.4f}")
    print(f"Original Test MAE: {mae:.4f}")
    print(f"Original Test RMSE: {rmse:.4f}")

    
    # ===================================
    # 4. 验证集偏差校正
    # ===================================

    # 加载验证集
    val_ds = DynamicSeismicDataset(split='val', augment=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    val_preds = []
    val_trues = []

    model.eval()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            wav, geo, mt, mw, amp, mask = [
                x.to(device) for x in batch
            ]

            _, mw_pred = model(wav, geo, amp, mask)

            val_preds.extend(
                mw_pred.cpu().numpy().flatten()
            )
            val_trues.extend(
                mw.cpu().numpy().flatten()
            )

    val_preds = np.asarray(val_preds)
    val_trues = np.asarray(val_trues)

    # 使用验证集计算固定偏差
    val_bias = np.mean(val_preds - val_trues)

    # 计算原始测试集偏差
    test_bias = np.mean(mw_preds - mw_trues)

    # 使用验证集偏差校正测试集
    mw_preds_corrected = mw_preds - val_bias
    corrected_residuals = mw_preds_corrected - mw_trues

    # 计算校正后的指标
    corrected_mae = np.mean(
        np.abs(corrected_residuals)
    )

    corrected_rmse = np.sqrt(
        np.mean(corrected_residuals ** 2)
    )

    # 输出结果
    print("\n========== Original Results ==========")
    print(f"Test Bias: {test_bias:.4f}")
    print(f"Original Test MAE: {mae:.4f}")
    print(f"Original Test RMSE: {rmse:.4f}")

    print("\n========== Bias Correction ==========")
    print(f"Validation Bias: {val_bias:.4f}")
    print(f"Bias-corrected Test MAE: {corrected_mae:.4f}")
    print(f"Bias-corrected Test RMSE: {corrected_rmse:.4f}")

    
    # # 4. 绘图 - 分开画图
    
    # # --- Plot 1: Scatter ---
    # plt.figure(figsize=(6, 6))
    
    # # Calculate density for better visualization if many points
    # # (Simple scatter is fine, maybe slightly transparent)
    # plt.scatter(mw_trues, mw_preds, alpha=0.5, s=10, c='blue', edgecolors='none')
    
    # # Draw y=x line
    # min_val = min(mw_trues.min(), mw_preds.min())
    # max_val = max(mw_trues.max(), mw_preds.max())
    #   # Add a little buffer
    # buffer = (max_val - min_val) * 0.05
    # min_val -= buffer
    # max_val += buffer

    # # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想值', linewidth=2)
    
    # # plt.xlabel("真实震级 (Mw)", fontsize=16)
    # # plt.ylabel("预测震级 (Mw)", fontsize=16)
    # # plt.title(f"震级回归 \n RMSE={rmse:.4f}, MAE={mae:.4f}", fontsize=16)
    # # plt.legend()


    # plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal', linewidth=2)


    # plt.xlabel("True Magnitude(Mw)", fontsize=16)
    # plt.ylabel("Prediction Magnitude (Mw)", fontsize=16)
    # plt.title(f"Magnitude Regression\n RMSE={rmse:.4f}, MAE={mae:.4f}", fontsize=16)
    # plt.legend()






    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.axis('equal') # Keep aspect ratio square-ish
    # plt.xlim(min_val, max_val)
    # plt.ylim(min_val, max_val)


    # from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    # ax = plt.gca()
    # ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    
    # scatter_path = "vis_magnitude_scatter.svg"
    # plt.tight_layout()
    # plt.savefig(scatter_path, dpi=600)
    # print(f"Saved scatter plot to {scatter_path}")
    # plt.close()

    # # --- Plot 2: Residual Histogram ---
    # plt.figure(figsize=(7, 5))
    # residuals = mw_preds - mw_trues
    
    # # Statistics of residuals
    # mu = np.mean(residuals)
    # sigma = np.std(residuals)
    
    # n, bins, patches = plt.hist(residuals, bins=30, density=True, facecolor='green', alpha=0.6, edgecolor='black')
    
    # # Add a 'best fit' line for normal distribution
    # y = ((1 / (np.sqrt(2 * np.pi) * sigma)) *
    #      np.exp(-0.5 * (1 / sigma * (bins - mu))**2))
    
    # # label_txt = f'正态拟合{os.linesep}μ={mu:.3f}, σ={sigma:.3f}'

    # label_txt = f'Normal Fitting{os.linesep}μ={mu:.3f}, σ={sigma:.3f}'

    # plt.plot(bins, y, 'r--', linewidth=2, label=label_txt)
    
    # plt.axvline(0, color='k', linestyle='-', linewidth=1)
    
    # # plt.xlabel("残差 (M_pred - M_true)", fontsize=16)
    # # plt.ylabel("概率密度", fontsize=16)
    # # plt.title("残差分布", fontsize=16)
    # # plt.legend()



    # plt.xlabel("Residual (M_pred - M_true)", fontsize=16)
    # plt.ylabel("Probability Density", fontsize=16)
    # plt.title("Residual Distribution", fontsize=16)
    # plt.legend()

    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # hist_path = "vis_magnitude_residual.svg"
    # plt.tight_layout()
    # plt.savefig(hist_path, dpi=600)
    # print(f"Saved residual histogram to {hist_path}")

if __name__ == "__main__":
    evaluate_magnitude_only()
