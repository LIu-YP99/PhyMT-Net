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

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

# ================================
# 字体配置
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
    print(f"Chinese font not found: {FONT_PATH}. Use default sans-serif.")

en_font_prop = fm.FontProperties(family='Times New Roman', weight='bold')
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 26
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 26
plt.rcParams['axes.unicode_minus'] = False

HUDSON_TEXT_SIZE = 20
PANEL_TITLE_SIZE = 36

# ==========================================
# Hudson 网格算法
# ==========================================
def ae_uvt(TT, KK):
    TAU = TT * (1 - np.abs(KK))
    UU = np.full_like(TT, np.nan)
    VV = np.full_like(TT, np.nan)

    II = (TAU > 0) & (KK < 0) | (TAU < 0) & (KK > 0)
    UU[II] = TAU[II]
    VV[II] = KK[II]

    II = (TAU < 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - TAU[II] / 2)
    VV[II] = KK[II] / (1 - TAU[II] / 2)

    II = (TAU >= 4 * KK) & (TAU >= 0) & (KK >= 0)
    UU[II] = TAU[II] / (1 - 2 * KK[II])
    VV[II] = KK[II] / (1 - 2 * KK[II])

    II = (TAU >= 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + TAU[II] / 2)
    VV[II] = KK[II] / (1 + TAU[II] / 2)

    II = (TAU < 4 * KK) & (TAU <= 0) & (KK <= 0)
    UU[II] = TAU[II] / (1 + 2 * KK[II])
    VV[II] = KK[II] / (1 + 2 * KK[II])

    return UU, VV


def draw_custom_grid(ax):
    hudson.draw_axes(ax)
    for text in ax.texts:
        text.set_fontproperties(en_font_prop)
        text.set_fontsize(HUDSON_TEXT_SIZE)
        text.set_zorder(20)
        text.set_clip_on(False)
        text.set_bbox({'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 0.2})

    VV = np.arange(-1, 1.1, 0.1)
    for i in range(10):
        x, y = ae_uvt(VV, VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(VV, -VV[i] * np.ones_like(VV))
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

        x, y = ae_uvt(-VV[i] * np.ones_like(VV), VV)
        ax.plot(x, y, 'k--', linewidth=0.5, alpha=0.3)

# ==========================================
# 主逻辑：模型预测版
# ==========================================
def vis_hudson_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Hudson Plot Visualization (Model Prediction)...")
    print("Device:", device)

    # 1. 加载 TEST 数据集
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    N_PLOT = min(500, len(test_ds))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(test_ds), N_PLOT, replace=False)
    subset = torch.utils.data.Subset(test_ds, indices)
    loader = DataLoader(subset, batch_size=N_PLOT, shuffle=False)

    # 2. 加载模型
    # model = OptimalFMTNet().to(device)

    # ckpt_files = sorted([
    #     f for f in os.listdir(Config.SAVE_DIR)
    #     if f.startswith("best_model_") and f.endswith(".pth")
    # ])

    # if not ckpt_files:
    #     raise FileNotFoundError("No best_model_*.pth found.")

    # ckpt_path = os.path.join(Config.SAVE_DIR, ckpt_files[-1])
    # print(f"Loading checkpoint: {ckpt_path}")

    # model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # model.eval()


    # 2. 加载模型
    model = ModelPlainCNN().to(device)

    ckpt_path = os.path.join(Config.SAVE_DIR, Config.RESUME_CHECKPOINT)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. 推理
    batch = next(iter(loader))
    wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]

    with torch.no_grad():
        mt_pred, _ = model(wav, geo, amp, mask)

    mt_true_np = mt_true.cpu().numpy()
    mt_pred_np = mt_pred.cpu().numpy()

    print(f"Predicted {len(mt_true_np)} samples using PhyMT-Net.")

    # 4. Hudson 投影
    pred_positions = []
    for i in range(len(mt_true_np)):
        try:
            u_pred, v_pred = hudson.project(tuple(mt_pred_np[i]))
            pred_positions.append((u_pred, v_pred))
        except Exception:
            pred_positions.append((np.nan, np.nan))

    pred_positions = np.asarray(pred_positions)

    # 5. 绘图
    def to_pyrocko_tuple(m):
        return (m[2], m[0], m[1], m[4], -m[5], -m[3])

    def plot_hudson_pair(ax_true, ax_pred, mt_true_data, mt_pred_data, pred_pos_data):
        ax_true.axis('off')
        ax_pred.axis('off')

        draw_custom_grid(ax_true)
        draw_custom_grid(ax_pred)

        count = 0
        for i in tqdm(range(len(mt_true_data)), desc="Plotting true vs pred"):
            u_pred, v_pred = pred_pos_data[i]
            if not (np.isfinite(u_pred) and np.isfinite(v_pred)):
                continue

            try:
                mt_true_bb = to_pyrocko_tuple(mt_true_data[i])
                mt_pred_bb = to_pyrocko_tuple(mt_pred_data[i])

                plot_beachball_mpl(
                    mt_true_bb, ax_true,
                    position=hudson.project(tuple(mt_true_data[i])),
                    size=30, color_t='blue', alpha=0.8, linewidth=0.4
                )

                plot_beachball_mpl(
                    mt_pred_bb, ax_pred,
                    position=(u_pred, v_pred),
                    size=30, color_t='red', alpha=0.8, linewidth=0.4
                )

                count += 1
            except Exception:
                continue

        ax_true.set_title("图（a）真实震源沙滩球分布图",
                         y=1.02, fontproperties=zh_font_prop, fontweight='bold')
        ax_pred.set_title("图（b）模型预测沙滩球分布图",
                         y=1.02, fontproperties=zh_font_prop, fontweight='bold')

        print(f"Finished plotting: true={count}, pred={count}")

    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    plot_hudson_pair(axes[0], axes[1], mt_true_np, mt_pred_np, pred_positions)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.05)

    save_path = "vis_hudson_model_prediction.svg"
    fig.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)

    print(f"Saved Hudson plot with model predictions to {save_path}")


if __name__ == "__main__":
    vis_hudson_only()