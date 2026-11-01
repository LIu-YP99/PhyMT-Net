import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.ablation_models import ModelPlainCNN

# ============================================================
# 全局配置
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ============================================================
# 路径修复
# ============================================================
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
    Config.DATA_ROOT = found_path
    Config.JSON_PATHS = {
        "waveforms": os.path.join(found_path, "waveforms.json"),
        "source":    os.path.join(found_path, "sources.json"),
        "grid":      os.path.join(found_path, "grid_points.json"),
        "receivers": os.path.join(found_path, "receivers.json")
    }

# ============================================================
# 辅助函数
# ============================================================
def calc_cosine_similarity(mt1, mt2):
    """计算两个矩张量之间的余弦相似度"""
    mt1_norm = mt1 / (np.linalg.norm(mt1) + 1e-10)
    mt2_norm = mt2 / (np.linalg.norm(mt2) + 1e-10)
    return np.dot(mt1_norm, mt2_norm)

def get_azimuth(geo_row):
    """从 geo 张量中提取方位角（度）"""
    sin_az = geo_row[0].item()
    cos_az = geo_row[1].item()
    return (np.degrees(np.arctan2(sin_az, cos_az)) + 360) % 360

# ============================================================
# 主程序
# ============================================================
def visualize_mt_consistency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    print("Loading Test Dataset...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    total_events = len(test_ds)

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

    # 3. 扫描，寻找 Median Similarity 附近的典型事件
    print("Scanning for representative event...")
    all_sims = []
    with torch.no_grad():
        for idx in range(min(500, total_events)):
            wav, geo, mt_true, mw_true, amp, mask = test_ds[idx]
            wav_b = wav.unsqueeze(0).to(device)
            geo_b = geo.unsqueeze(0).to(device)
            amp_b = amp.unsqueeze(0).to(device)
            mask_b = mask.unsqueeze(0).to(device)
            
            mt_pred, _ = model(wav_b, geo_b, amp_b, mask_b)
            sim = calc_cosine_similarity(mt_true.cpu().numpy().flatten(), mt_pred.cpu().numpy().flatten())
            all_sims.append((idx, sim))

    all_sims.sort(key=lambda x: x[1])
    median_idx = len(all_sims) // 2
    selected_idx = all_sims[median_idx][0]
    selected_sim = all_sims[median_idx][1]
    print(f"Selected representative event: Index={selected_idx}, MT_Sim={selected_sim:.4f}")

    # 4. 对该事件进行完整推理
    wav, geo, mt_true, mw_true, amp, mask = test_ds[selected_idx]

    wav_b = wav.unsqueeze(0).to(device)
    geo_b = geo.unsqueeze(0).to(device)
    amp_b = amp.unsqueeze(0).to(device)
    mask_b = mask.unsqueeze(0).to(device)

    with torch.no_grad():
        mt_pred, mw_pred = model(wav_b, geo_b, amp_b, mask_b)

    mt_true_np = mt_true.cpu().numpy().flatten()
    mt_pred_np = mt_pred.cpu().numpy().flatten()
    final_sim = calc_cosine_similarity(mt_true_np, mt_pred_np)
    mw_pred_val = mw_pred.item()

    # 5. 获取事件信息
    key = test_ds.event_keys[selected_idx]
    evt_data = test_ds.events[key]

    # 6. 选择 4 个有效台站
    valid_indices = torch.where(~mask)[0].tolist()
    plot_indices = random.sample(valid_indices, min(4, len(valid_indices)))
    print(f"Selected Station Indices: {plot_indices}")

    # 7. 绘图（只画观测波形，标注 MT 相似度）
    fig, axes = plt.subplots(len(plot_indices), 3, figsize=(18, 3.5 * len(plot_indices)), sharex=True)

    if len(plot_indices) == 1:
        axes = np.array([axes])

    fig.suptitle(
        f"MT Consistency Validation (Event {selected_idx})\n"
        f"True Mw: {mw_true.item():.2f} | Pred Mw: {mw_pred_val:.2f} | MT Sim: {final_sim:.4f}",
        fontsize=14, fontweight='bold'
    )

    components = ['Z', 'R', 'T']

    for row_i, sta_idx in enumerate(plot_indices):
        w_meta = evt_data['waveforms'][sta_idx]
        sta_code = w_meta.get('receiver_code', f'STA{sta_idx}')

        # 方位角
        azimuth_deg = get_azimuth(geo[sta_idx])

        # 观测波形
        obs_wav = wav[sta_idx].cpu().numpy()  # (3, T)

        for col_i, comp in enumerate(components):
            ax = axes[row_i, col_i]

            obs_data = obs_wav[col_i]

            # 归一化
            obs_norm = obs_data / (np.max(np.abs(obs_data)) + 1e-9)

            # 绘制观测波形
            ax.plot(obs_norm, 'k-', label='Observed', lw=1.5, alpha=0.9)

            # 标题：台站 + 方位角 + MT 相似度
            title_text = f"{sta_code} | Az: {azimuth_deg:.0f}° | MT Sim: {final_sim:.3f}"
            ax.set_title(title_text, fontsize=9, fontweight='bold')

            # 网格
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-1.3, 1.3)

            # 图例
            if row_i == 0 and col_i == 0:
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

            # Y 轴标签
            if col_i == 0:
                ax.set_ylabel(f"{sta_code}\nAz: {azimuth_deg:.0f}°", fontsize=9, labelpad=8)

            # X 轴标签
            if row_i == len(plot_indices) - 1:
                ax.set_xlabel("Time (samples)", fontsize=10)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    save_path = "mt_consistency_validation.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    visualize_mt_consistency()