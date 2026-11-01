import sys, os, json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import torch

sys.path.append(os.getcwd())
from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet


import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 0. 路径修复（保持你现有逻辑）
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
    print(f"Found data at: {found_path}")
    Config.DATA_ROOT = found_path
    Config.JSON_PATHS = {
        "waveforms": os.path.join(found_path, "waveforms.json"),
        "source":    os.path.join(found_path, "sources.json"),
        "grid":      os.path.join(found_path, "grid_points.json"),
        "receivers": os.path.join(found_path, "receivers.json")
    }
else:
    print("Warning: Data directory not found.")

# ============================================================
# 1. 加噪函数
# ============================================================
def add_noise(wav, snr_db):
    sigma = 10 ** (-snr_db / 20.0)
    noise = torch.randn_like(wav) * sigma
    return wav + noise

# ============================================================
# 2. 主程序
# ============================================================
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    model = OptimalFMTNet().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20260609_2332.pth")
    if not os.path.exists(ckpt_path):
        files = os.listdir(Config.SAVE_DIR)
        best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
        if best_models:
            best_models.sort(reverse=True)
            ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])

    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    snr_levels = [-5, 0, 5, 10, 15, 20, 25, 30, 35, 40]

    all_mae, all_rmse, all_r2 = [], [], []

    print(f"Starting evaluation on SNR levels: {snr_levels}")

    for snr in snr_levels:
        print(f"Processing SNR = {snr} dB ...")

        mw_true_list, mw_pred_list = [], []

        with torch.no_grad():
            for batch in tqdm(loader, leave=False):
                wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
                wav_noisy = add_noise(wav, snr)
                mt_pred, mw_pred = model(wav_noisy, geo, amp, mask)

                mw_true_list.append(mw_true.cpu().numpy())
                mw_pred_list.append(mw_pred.cpu().numpy())

        y_true = np.concatenate(mw_true_list).flatten()
        y_pred = np.concatenate(mw_pred_list).flatten()

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)

        all_mae.append(mae)
        all_rmse.append(rmse)
        all_r2.append(r2)

        print(f"  -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # ============================================================
    # 3. 绘图：MAE / RMSE / R²（左轴 + 右轴）
    # ============================================================
    print("Plotting magnitude metrics...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('MAE / RMSE', color='tab:blue', fontsize=12)

    l1, = ax1.plot(snr_levels, all_mae,  'o-',  color='tab:blue',  linewidth=2, label='MAE')
    l2, = ax1.plot(snr_levels, all_rmse, 's--', color='tab:cyan',  linewidth=2, label='RMSE')

    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_xticks(snr_levels)

    ax2 = ax1.twinx()
    ax2.set_ylabel('$R^2$', color='tab:orange', fontsize=12)
    l3, = ax2.plot(snr_levels, all_r2, '^-.', color='tab:orange', linewidth=2, label='$R^2$')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='best', fontsize=11, framealpha=0.9)

    plt.title('Magnitude Metrics vs. SNR', fontsize=14)
    plt.tight_layout()

    out = "vis_snr_magnitude_metrics.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {out}")

    # 保存数据
    data = {
        "snr": snr_levels,
        "mae": all_mae,
        "rmse": all_rmse,
        "r2": all_r2
    }
    with open("snr_magnitude_metrics.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved snr_magnitude_metrics.json")

if __name__ == "__main__":
    evaluate()