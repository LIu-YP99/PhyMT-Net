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
    m_33 = np.zeros((len(m_np), 3, 3))
    m_33[:,0,0]=m_np[:,0]; m_33[:,1,1]=m_np[:,1]; m_33[:,2,2]=m_np[:,2]
    m_33[:,0,1]=m_np[:,3]; m_33[:,1,0]=m_np[:,3]
    m_33[:,0,2]=m_np[:,4]; m_33[:,2,0]=m_np[:,4]
    m_33[:,1,2]=m_np[:,5]; m_33[:,2,1]=m_np[:,5]

    vals = np.linalg.eigvalsh(m_33)
    vals = np.sort(vals, axis=1)[:, ::-1]

    iso = np.mean(vals, axis=1)
    dev = vals - iso[:, None]
    max_dev = np.max(np.abs(dev), axis=1) + 1e-9

    T = 2 * dev[:, 1] / max_dev
    k = iso / (np.max(np.abs(vals), axis=1) + 1e-9)
    return T, k

def add_noise(wav, snr_db):
    """向波形添加指定 SNR 的高斯噪声"""
    sigma = 10 ** (-snr_db / 20.0)
    noise = torch.randn_like(wav) * sigma
    return wav + noise

@torch.no_grad()
def estimate_uncertainty(model, wav, geo, amp, mask, mc_samples=10):
    """
    蒙特卡洛 Dropout 估计预测不确定性
    """
    model.eval()
    mt_preds = []
    for _ in range(mc_samples):
        mt_pred, _ = model(wav, geo, amp, mask)
        mt_preds.append(mt_pred)
    mt_preds = torch.stack(mt_preds)
    mt_std = mt_preds.std(dim=0).mean(dim=1)  # (B,)
    return mt_std.cpu().numpy()

# ==========================================
# 2. 主程序
# ==========================================
def evaluate_snr_impact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据 (无增强)
    print("Loading test dataset (augment=False)...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    # 2. 加载模型
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

    # 3. 定义 SNR 列表
    snr_levels = [-5, 0, 5, 10, 15, 20, 25, 30]

    avg_similarities = []
    avg_iso_errors = []
    avg_uncertainties = []

    print(f"Starting evaluation on SNR levels: {snr_levels}")

    for snr in snr_levels:
        print(f"Processing SNR = {snr} dB ...")

        sim_list = []
        iso_err_list = []
        uncert_list = []

        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]

                wav_noisy = add_noise(wav, snr)

                mt_pred, mw_pred = model(wav_noisy, geo, amp, mask)

                # Cosine Similarity
                sim = torch.sum(mt_true * mt_pred, dim=1)
                sim_list.extend(sim.cpu().numpy())

                # ISO Error
                _, k_true = get_Tk(mt_true)
                _, k_pred = get_Tk(mt_pred)
                iso_err = np.abs(k_pred - k_true)
                iso_err_list.extend(iso_err)

                # Uncertainty (MC Dropout)
                mt_std = estimate_uncertainty(model, wav_noisy, geo, amp, mask)
                uncert_list.extend(mt_std)

        avg_sim = np.mean(sim_list)
        avg_iso = np.mean(iso_err_list)
        avg_unc = np.mean(uncert_list)

        avg_similarities.append(avg_sim)
        avg_iso_errors.append(avg_iso)
        avg_uncertainties.append(avg_unc)

        print(f"  -> Avg Sim: {avg_sim:.4f}, Avg ISO Err: {avg_iso:.4f}, Avg Unc: {avg_unc:.4f}")

    # ==========================================
    # 4. 绘图（无误差带 + 右上角图例 + 不确定性范围 0-0.3）
    # ==========================================
    print("Plotting results...")

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # 左轴：Cosine Similarity
    color1 = 'tab:blue'
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', color=color1, fontsize=12)
    l1, = ax1.plot(snr_levels, avg_similarities, color=color1, marker='o', linestyle='-', linewidth=2, label='Cosine Similarity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)

    # 右轴1：Isotropic Component Error
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Isotropic Component Error', color=color2, fontsize=12)
    l2, = ax2.plot(snr_levels, avg_iso_errors, color=color2, marker='s', linestyle='--', linewidth=2, label='Isotropic Component Error')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.15, 0.4)

    # 右轴2：Prediction Uncertainty（偏移，避免与 ISO 重叠）
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    color3 = 'tab:green'
    ax3.set_ylabel('Prediction Uncertainty', color=color3, fontsize=12)
    l3, = ax3.plot(snr_levels, avg_uncertainties, color=color3, marker='^', linestyle='-.', linewidth=2, label='Prediction Uncertainty')
    ax3.tick_params(axis='y', labelcolor=color3)
    ax3.set_ylim(0.0, 0.3)

    plt.title('Model Performance vs. SNR', fontsize=14)

    # 图例放在右上角
    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    save_path = "vis_snr_impact_uncertainty.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_snr_impact()
