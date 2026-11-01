import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

sys.path.append(os.getcwd())
from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet

# --- Hudson 绘图辅助函数 (纯Matplotlib版，不依赖Pyrocko) ---
def ae_uvt(TT, KK):
    TAU = TT * (1 - np.abs(KK))
    UU = np.full_like(TT, np.nan)
    VV = np.full_like(TT, np.nan)
    II = (TAU > 0) & (KK < 0) | (TAU < 0) & (KK > 0)
    UU[II] = TAU[II]; VV[II] = KK[II]
    II = (TAU < 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - TAU[II] / 2); VV[II] = KK[II] / (1 - TAU[II] / 2)
    II = (TAU >= 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - 2 * KK[II]); VV[II] = KK[II] / (1 - 2 * KK[II])
    II = (TAU >= 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + TAU[II] / 2); VV[II] = KK[II] / (1 + TAU[II] / 2)
    II = (TAU < 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + 2 * KK[II]); VV[II] = KK[II] / (1 + 2 * KK[II])
    return UU, VV

def draw_hudson_grid(ax):
    # 绘制边框
    x_b = [0, 1, 0, -1, 0]; y_b = [1, 0, -1, 0, 1]
    ax.plot(x_b, y_b, 'k-', lw=1.5)
    # 绘制网格
    VV = np.linspace(-1, 1, 100)
    for val in [-0.5, 0, 0.5]:
        x, y = ae_uvt(VV, val * np.ones_like(VV)); ax.plot(x, y, 'k--', lw=0.5, alpha=0.3)
        x, y = ae_uvt(VV, -val * np.ones_like(VV)); ax.plot(x, y, 'k--', lw=0.5, alpha=0.3)
        x, y = ae_uvt(val * np.ones_like(VV), VV); ax.plot(x, y, 'k--', lw=0.5, alpha=0.3)
        x, y = ae_uvt(-val * np.ones_like(VV), VV); ax.plot(x, y, 'k--', lw=0.5, alpha=0.3)
    ax.text(0, 1.05, '+ISO', ha='center'); ax.text(0, -1.15, '-ISO', ha='center')
    ax.text(-1.1, 0, '+CLVD', ha='right'); ax.text(1.1, 0, '-CLVD', ha='left')
    ax.axis('off')

def get_Tk(mt_batch):
    # 计算 T, k
    m_np = mt_batch.cpu().numpy()
    # 构建 3x3
    m_33 = np.zeros((len(m_np), 3, 3))
    m_33[:,0,0]=m_np[:,0]; m_33[:,1,1]=m_np[:,1]; m_33[:,2,2]=m_np[:,2]
    m_33[:,0,1]=m_np[:,3]; m_33[:,1,0]=m_np[:,3]
    m_33[:,0,2]=m_np[:,4]; m_33[:,2,0]=m_np[:,4]
    m_33[:,1,2]=m_np[:,5]; m_33[:,2,1]=m_np[:,5]
    
    vals = np.linalg.eigvalsh(m_33)
    vals = np.sort(vals, axis=1)[:, ::-1] # 降序 l1>=l2>=l3
    
    iso = np.mean(vals, axis=1)
    dev = vals - iso[:, None]
    max_dev = np.max(np.abs(dev), axis=1) + 1e-9
    T = 2 * dev[:, 1] / max_dev
    k = iso / (np.max(np.abs(vals), axis=1) + 1e-9)
    return T, k

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Test Data...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    
    model = OptimalFMTNet().to(device)
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, "best_model_20251202_2257.pth"), map_location=device))
    model.eval()
    
    mw_true, mw_pred = [], []
    mt_true_list, mt_pred_list = [], []
    
    print("Inferencing...")
    with torch.no_grad():
        for batch in tqdm(loader):
            wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
            mt_out, mw_out = model(wav, geo, amp, mask)
            
            mw_true.extend(mw.cpu().numpy().flatten())
            mw_pred.extend(mw_out.cpu().numpy().flatten())
            mt_true_list.append(mt)
            mt_pred_list.append(mt_out)
            
    mt_true = torch.cat(mt_true_list, dim=0)
    mt_pred = torch.cat(mt_pred_list, dim=0)
    
    # 计算 Hudson 参数
    T_t, k_t = get_Tk(mt_true)
    T_p, k_p = get_Tk(mt_pred)
    
    # --- 绘图 ---
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. 震级回归
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(mw_true, mw_pred, alpha=0.1, s=5, c='b')
    ax1.plot([3, 7], [3, 7], 'r--')
    ax1.set_title("Magnitude Regression")
    ax1.set_xlabel("True Mw"); ax1.set_ylabel("Predicted Mw")
    ax1.grid(True, alpha=0.5)
    
    # 2. Hudson Plot
    ax2 = fig.add_subplot(gs[0, 1:])
    draw_hudson_grid(ax2)
    # 投影
    u_t, v_t = ae_uvt(T_t, k_t)
    u_p, v_p = ae_uvt(T_p, k_p)
    # 降采样绘制
    idx = np.random.choice(len(u_t), min(len(u_t), 3000), replace=False)
    ax2.scatter(u_t[idx], v_t[idx], c='gray', alpha=0.2, label='True', s=10)
    ax2.scatter(u_p[idx], v_p[idx], c='red', alpha=0.2, label='Pred', s=10)
    ax2.legend()
    ax2.set_title("Source Type Distribution")
    
    # 3. ISO Error Histogram
    ax3 = fig.add_subplot(gs[1, 0])
    iso_err = (k_p - k_t) * 100 # 百分比
    ax3.hist(iso_err, bins=50, color='green', alpha=0.7)
    ax3.set_title("ISO Component Error Distribution")
    ax3.set_xlabel("ISO Error (%)"); ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.5)
    
    # 4. Similarity Histogram
    ax4 = fig.add_subplot(gs[1, 1])
    sim = np.sum(mt_true.cpu().numpy() * mt_pred.cpu().numpy(), axis=1)
    ax4.hist(sim, bins=50, color='orange', alpha=0.7)
    ax4.set_title("Cosine Similarity Distribution")
    ax4.set_xlabel("Similarity (1.0 is best)"); ax4.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig("vis_statistics.png", dpi=300)
    print("Saved vis_statistics.png")

if __name__ == "__main__":
    main()