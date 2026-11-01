import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from obspy.imaging.beachball import beach
from torch.utils.data import DataLoader

# 确保路径正确
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet

# ==========================================
# 1. 坐标系转换 (NED -> GCMT)
# ==========================================
def convert_ned_to_gcmt(m_ned):
    mxx, myy, mzz, mxy, mxz, myz = m_ned
    mrr = mzz
    mtt = mxx
    mpp = myy
    mrt = mxz
    mrp = -myz
    mtp = -mxy
    return np.array([mrr, mtt, mpp, mrt, mrp, mtp])

# ==========================================
# 2. 指标与校验
# ==========================================
def is_valid_mt(mt):
    if np.all(mt == 0): return False
    if np.any(np.isnan(mt)): return False
    if np.linalg.norm(mt) < 1e-9: return False
    return True

def calculate_metrics(mt_pred, mt_true):
    if not is_valid_mt(mt_true): return 0.0, 0.0, 0.0
    
    norm_p = np.linalg.norm(mt_pred)
    norm_t = np.linalg.norm(mt_true)
    cosine = np.dot(mt_pred, mt_true) / (norm_p * norm_t + 1e-9)
    
    def get_iso(mt_arr):
        if not is_valid_mt(mt_arr): return 0.0
        m = np.array([[mt_arr[0], mt_arr[3], mt_arr[4]],
                      [mt_arr[3], mt_arr[1], mt_arr[5]],
                      [mt_arr[4], mt_arr[5], mt_arr[2]]])
        try:
            vals = np.linalg.eigvalsh(m)
            return (np.sum(vals) / np.sum(np.abs(vals))) * 100
        except: return 0.0

    return cosine, get_iso(mt_pred), get_iso(mt_true)

# ==========================================
# 3. 绘图主程序 (UI 修复版)
# ==========================================
def plot_comparison(num_samples=10, save_path="best_model_20251202_2257.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading data from {Config.DATA_ROOT}")
    
    # 1. 加载数据
    # test_ds = DynamicSeismicDataset(augment=False)  #真实数据
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    n_data = len(test_ds)
    if n_data == 0:
        print("No data in test set.")
        return
        
    batch_size = min(n_data, num_samples)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True) 
    
    # 2. 模型
    model = OptimalFMTNet().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251202_2257.pth")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    print(f"Loading weights...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 3. 获取数据
    try:
        wav, geo, mt_true, mw_true, amp, mask = next(iter(loader))
    except StopIteration:
        return

    wav = wav.to(device)
    geo = geo.to(device)
    amp = amp.to(device)
    mask = mask.to(device)
    
    # 4. 预测
    with torch.no_grad():
        mt_pred_tensor, mw_pred_tensor = model(wav, geo, amp, mask)
        
    mt_p_np = mt_pred_tensor.cpu().numpy()
    mw_p_np = mw_pred_tensor.cpu().numpy().flatten()
    mt_t_np = mt_true.numpy()
    mw_t_np = mw_true.numpy().flatten()
    
    # 5. 绘图配置
    rows = len(mt_p_np)
    cols = 2
    # 增加宽度 (8 -> 10)，确保文字有地方放
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.5))
    
    if rows == 1: axes = np.array([axes])
    
    if rows > 1:
        fig.suptitle("Focal Mechanism Comparison", fontsize=16, y=1.005)
        axes[0,0].set_title("Ground Truth (GCMT)", fontweight='bold', fontsize=14)
        axes[0,1].set_title("Prediction", fontweight='bold', fontsize=14)

    print("Plotting beachballs...")
    for i in range(rows):
        ax_t, ax_p = axes[i, 0], axes[i, 1]
        
        m_t_ned = mt_t_np[i]
        m_p_ned = mt_p_np[i]
        
        sim, iso_p, iso_t = calculate_metrics(m_p_ned, m_t_ned)
        mw_err = mw_p_np[i] - mw_t_np[i]
        
        m_t_plot = convert_ned_to_gcmt(m_t_ned)
        m_p_plot = convert_ned_to_gcmt(m_p_ned)
        
        # 【关键修改】设置坐标轴范围，给左右留白
        # 球的直径是 0.9，中心在 (0.5, 0.5)
        # 我们把 X 轴范围设为 -0.5 到 1.5，这样左右就各空出了 0.5 的空间写字
        LIMITS = (-0.5, 1.5)
        
        # --- 画真值 ---
        if is_valid_mt(m_t_plot):
            try:
                b = beach(m_t_plot, xy=(0.5, 0.5), width=0.9, linewidth=1, facecolor='gray', alpha=0.8)
                ax_t.add_collection(b)
                # 文字放在左侧 (x=0.0)，稍微往左偏一点
                ax_t.text(-0.1, 0.5, f"Mw: {mw_t_np[i]:.2f}\nISO: {iso_t:.0f}%", 
                          ha='right', va='center', fontsize=11, fontweight='bold')
            except: pass
        else:
            ax_t.text(0.5, 0.5, "No Data", ha='center')
            
        ax_t.set_aspect("equal")
        ax_t.set_xlim(LIMITS) # 扩大视野
        ax_t.set_ylim(0, 1)
        ax_t.axis('off')
        
        # --- 画预测 ---
        if is_valid_mt(m_p_plot):
            try:
                col = 'red' if sim > 0.7 else 'blue'
                if not is_valid_mt(m_t_plot): col = 'purple'
                
                b = beach(m_p_plot, xy=(0.5, 0.5), width=0.9, linewidth=1, facecolor=col, alpha=0.8)
                ax_p.add_collection(b)
                
                info = f"Mw: {mw_p_np[i]:.2f} (Err:{mw_err:+.2f})\nISO: {iso_p:.0f}%\nSim: {sim:.3f}"
                # 文字放在右侧 (x=1.1)
                ax_p.text(1.1, 0.5, info, ha='left', va='center', fontsize=11, fontweight='bold')
            except: pass
        
        ax_p.set_aspect("equal")
        ax_p.set_xlim(LIMITS) # 扩大视野
        ax_p.set_ylim(0, 1)
        ax_p.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"Done! Saved to {save_path}")

if __name__ == "__main__":
    plot_comparison(num_samples=11)