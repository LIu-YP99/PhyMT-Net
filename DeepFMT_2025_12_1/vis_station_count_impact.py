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
import matplotlib.font_manager as fm

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

# ================================
# 配置字体 (Times New Roman & SimHei)
# ================================
fm.fontManager.addfont('/root/SimHei.ttf')
my_font = fm.FontProperties(fname='/root/SimHei.ttf')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'stix'

# ==========================================
# 0. 路径修复 (同 vis_snr_impact.py)
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
# 1. 辅助函数：下采样台站
# ==========================================
def subsample_stations(wav, geo, amp, mask, target_k):
    """
    从 batch 中随机选择 target_k 个有效台站。
    如果样本的有效台站数 < target_k，则标记为 invalid_batch_mask，后续丢弃。
    
    Returns:
        new_wav, new_geo, new_amp, new_mask (for model input)
        valid_indices (boolean tensor of shape (B,), indicating which samples are kept)
    """
    B, S, C, N = wav.shape
    device = wav.device
    
    # 1. 统计当前有效台站数
    # mask: True 表示 Padding/Invalid
    current_valid_counts = (~mask).sum(dim=1) # (B,)
    
    # 2. 标记满足条件的样本
    valid_batch_indices = (current_valid_counts >= target_k)
    
    if not valid_batch_indices.any():
        return None, None, None, None, valid_batch_indices
        
    # 3. 构建新的 mask
    # 我们只修改 mask，wav 和 geo 可以保持原样，因为模型中有 mask 就会忽略对应位置
    # 但为了保险，也可以把 mask 掉的位置 wav 置 0
    
    new_mask = torch.ones_like(mask) # Init all invalid (True)
    
    # 向量化随机选择
    # 生成随机分数矩阵 (B, S)
    rand_scores = torch.rand((B, S), device=device)
    
    # 将原始 mask 为 True (无效) 的位置分数设为 -1，确保不会被选中
    rand_scores = rand_scores.masked_fill(mask, -1.0)
    
    # 对分数排序，取前 target_k 个的索引
    # descending=True, 最大的在前
    _, top_k_indices = torch.topk(rand_scores, k=target_k, dim=1) # (B, K)
    
    # 更新 new_mask
    # scatter_(dim, index, src) -> src must be tensor or value
    # 我们要将 new_mask 在 top_k_indices 的位置设为 False (Valid)
    # new_mask 是 Bool，不能直接 scatter 0/1 (在旧版 torch 可能有问题)，用 scatter 赋值 False
    new_mask.scatter_(1, top_k_indices, False)
    
    # 4. 对于那些原始有效数 < target_k 的样本，上面的逻辑选出来的可能是无效台站
    # 但反正这些样本最后会被 valid_batch_indices 过滤掉，所以无所谓
    
    # 5. 为了模型输入干净，把新 mask 为 True 的 wav 设为 0
    # 模型虽然可能忽略了 mask 参数，但训练时 Padding 都是 0，所以这里必须置 0 来模拟
    new_wav = wav.clone()
    new_wav.masked_fill_(new_mask.unsqueeze(2).unsqueeze(3), 0.0)
    
    new_geo = geo.clone()
    new_geo.masked_fill_(new_mask.unsqueeze(2), 0.0)
    
    return new_wav, new_geo, amp, new_mask, valid_batch_indices

# ==========================================
# 2. 主程序
# ==========================================
def evaluate_station_count_impact():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 加载数据
    print("Loading test dataset (augment=False)...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # # 2. 加载模型
    # model = OptimalFMTNet().to(device)
    # ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251201_2112.pth")
    # # Fallback logic
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


    # 3. 定义台站数量列表
    # 我们的数据最大台站数是 24
    station_counts = [4, 6, 8, 12, 16, 20, 24]
    
    # 存储结果：字典 {k: [sim1, sim2, ...]}
    results = {k: [] for k in station_counts}
    
    print(f"Starting evaluation for station counts: {station_counts}")
    
    with torch.no_grad():
        # 为了效率，我们只遍历一次 DataLoader
        # 但对于每个 batch，我们需要遍历所有 station_counts 进行多次推理
        for batch in tqdm(loader):
            wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
            
            # 对每个目标 K 进行测试
            for k in station_counts:
                # 下采样
                w, g, a, m, valid_idx = subsample_stations(wav, geo, amp, mask, k)
                
                if not valid_idx.any():
                    continue
                
                # 只对满足条件的样本进行推理
                # 过滤 Batch
                w_sub = w[valid_idx]
                g_sub = g[valid_idx]
                a_sub = a[valid_idx]
                m_sub = m[valid_idx]
                mt_true_sub = mt_true[valid_idx]
                
                # 推理
                mt_pred, _ = model(w_sub, g_sub, a_sub, m_sub)
                
                # 计算相似度
                sim = torch.sum(mt_true_sub * mt_pred, dim=1)
                results[k].extend(sim.cpu().numpy())

    # ==========================================
    # 3. 绘图
    # ==========================================
    print("Plotting Boxplot...")
    
    # 准备数据
    data_to_plot = []
    labels = []
    means = []
    medians = []
    
    for k in station_counts:
        sims = results[k]
        if len(sims) > 0:
            data_to_plot.append(sims)
            labels.append(str(k))
            means.append(np.mean(sims))
            medians.append(np.median(sims))
            print(f"Station Count {k}: N={len(sims)}, Mean Sim={np.mean(sims):.4f}, Median={np.median(sims):.4f}")
        else:
            print(f"Station Count {k}: No samples found.")
            
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制箱线图
    # showfliers=False 不显示异常值点，保持图面整洁，或者 True 显示
    # patch_artist=True 填充颜色
    bp = ax.boxplot(data_to_plot, 
                    labels=labels, 
                    patch_artist=True,
                    showfliers=True,
                    medianprops=dict(color="orange", linewidth=2.0))
    
    # 美化箱体
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
        
    # 添加均值连线（可选，或者只看箱线图）
    # x 坐标从 1 开始
    x_coords = np.arange(1, len(data_to_plot) + 1)
    ax.plot(x_coords, medians, 'r--', label='中位数趋势', alpha=0.7)
    
    my_font_14 = fm.FontProperties(fname='/root/SimHei.ttf', size=14)
    my_font_16 = fm.FontProperties(fname='/root/SimHei.ttf', size=16)
    
    ax.set_xlabel('台站数量', fontproperties=my_font_14)
    ax.set_ylabel('震源机制解相似度', fontproperties=my_font_14)
    ax.set_title('反演性能随台站数量的变化', fontproperties=my_font_16)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(prop=my_font)
    
    # 限制 Y 轴
    ax.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    save_path = "vis_station_count_impact.svg"
    plt.savefig(save_path, dpi=600)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    evaluate_station_count_impact()
