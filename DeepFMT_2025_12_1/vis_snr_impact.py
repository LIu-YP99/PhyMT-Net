import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure path is correct
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

# ==========================================
# 0. 修复路径配置
# ==========================================
# 检测实际数据路径
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
# 1. 辅助函数
# ==========================================
def get_Tk(mt_batch):
    """
    计算 Hudson 参数 T 和 k
    mt_batch: (B, 6) tensor, NED 坐标系
    """
    m_np = mt_batch.cpu().numpy()
    # 构建 3x3 矩阵
    # NED: xx, yy, zz, xy, xz, yz
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
    
    # Hudson T: [-1, 1]
    T = 2 * dev[:, 1] / max_dev
    # Hudson k: [-1, 1] (k = ISO / Max(|eigenvals|))
    # 注意：这里的分母取所有特征值的最大绝对值，确保 k 在 [-1, 1]
    k = iso / (np.max(np.abs(vals), axis=1) + 1e-9)
    return T, k

def add_noise(wav, snr_db):
    """
    向波形添加指定 SNR 的高斯噪声
    wav: (B, C, N) 归一化后的波形
    snr_db: 信噪比 (dB)
    """
    # 假设信号峰值为 1 (因为已经归一化)
    # SNR = 20 * log10(A_signal / A_noise)
    # => A_noise = A_signal * 10^(-SNR/20)
    # 取 A_signal = 1
    
    # 也可以用 RMS 功率计算：
    # SNR = 10 * log10(P_signal / P_noise)
    # P_signal = mean(wav^2)
    # P_noise = P_signal * 10^(-SNR/10)
    # noise_std = sqrt(P_noise)
    
    # 这里我们使用基于 RMS 的更准确的方法
    # wav: (B, S, C, N)
    # S is stations
    
    # 假设信号峰值为 1 (因为已经归一化)
    # SNR = 20 * log10(A_signal / A_noise)
    # => A_noise = A_signal * 10^(-SNR/20)
    # 取 A_signal = 1
    
    sigma = 10 ** (-snr_db / 20.0)
    
    noise = torch.randn_like(wav) * sigma
    return wav + noise

# ==========================================
# 2. 主程序
# ==========================================
def evaluate_snr_impact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载数据 (无增强)
    print("Loading test dataset (augment=False)...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    # 为了加快速度，如果测试集很大，可以取子集，或者只跑部分 batch
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # # 2. 加载模型
    # model = OptimalFMTNet().to(device)
    
    # # 尝试找到最佳模型
    # ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251201_2112.pth")
    # if not os.path.exists(ckpt_path):
    #      files = os.listdir(Config.SAVE_DIR)
    #      best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
    #      if best_models:
    #          best_models.sort(reverse=True)
    #          ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])
    
    # print(f"Loading weights from {ckpt_path}...")
    # model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # model.eval()
    # 2. 加载模型
    model = ModelPlainCNN().to(device)
    
    # 尝试找到最佳模型
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
    
    # 3. 定义 SNR 列表
    # 范围：-5 dB 到 30 dB
    snr_levels = [-5, 0, 5, 10, 15, 20, 25, 30]
    
    avg_similarities = []
    avg_iso_errors = []
    
    print(f"Starting evaluation on SNR levels: {snr_levels}")
    
    for snr in snr_levels:
        print(f"Processing SNR = {snr} dB ...")
        
        sim_list = []
        iso_err_list = []
        
        # 只跑一部分数据以节省时间？或者跑全量。这里跑全量。
        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
                
                # 手动注入噪声
                wav_noisy = add_noise(wav, snr)
                
                # 模型推理
                mt_pred, mw_pred = model(wav_noisy, geo, amp, mask)
                
                # 计算指标
                # 1. Cosine Similarity
                # mt_true 和 mt_pred 应该都是归一化的
                sim = torch.sum(mt_true * mt_pred, dim=1) # (B,)
                sim_list.extend(sim.cpu().numpy())
                
                # 2. ISO Error
                # 需要计算 T, k
                _, k_true = get_Tk(mt_true)
                _, k_pred = get_Tk(mt_pred)
                
                # 误差定义为绝对误差 |k_pred - k_true|
                # 或者按照 vis_statistics 里的 (k_p - k_t) * 100
                # 用户要求“各向同性分量误差”，通常是 MAE
                iso_err = np.abs(k_pred - k_true)
                iso_err_list.extend(iso_err)
        
        # 汇总当前 SNR 的结果
        avg_sim = np.mean(sim_list)
        avg_iso = np.mean(iso_err_list)
        
        avg_similarities.append(avg_sim)
        avg_iso_errors.append(avg_iso)
        
        print(f"  -> Avg Sim: {avg_sim:.4f}, Avg ISO Err: {avg_iso:.4f}")

    # ==========================================
    # 3. 绘图
    # ==========================================
    print("Plotting results...")
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 设置网格
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # 绘制左轴：震源机制相似度 (蓝色实线)
    color1 = 'tab:blue'
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Mechanism Similarity', color=color1, fontsize=12)
    l1, = ax1.plot(snr_levels, avg_similarities, color=color1, marker='o', linestyle='-', linewidth=2, label='Mechanism Similarity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.0, 1.05) # 相似度通常在 0~1 之间 (如果是绝对相似度) 或 -1~1
    
    # 绘制右轴：各向同性分量误差 (橙色虚线)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Isotropic Component Error', color=color2, fontsize=12)
    l2, = ax2.plot(snr_levels, avg_iso_errors, color=color2, marker='s', linestyle='--', linewidth=2, label='Isotropic Component Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    # ISO 误差范围假设在 0~1 之间 (k 是 -1~1)
    # 自动缩放即可，或者稍微留点余地
    
    # 标题
    plt.title('Model Performance vs. SNR', fontsize=14)
    
    # 图例 (合并双轴图例)
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=10)
    
    plt.tight_layout()
    save_path = "vis_snr_impact.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_snr_impact()
