import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    # model = OptimalFMTNet().to(device)
    model = ModelPlainCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, "ablation_plain_cnn_20260609_1456.pth"), map_location=device))
    model.eval()
    
    lats, lons, depths = [], [], []
    sim_errs = []
    mag_errs = []
    
    print("Inferencing for Spatial Analysis...")
    with torch.no_grad():
        for batch in tqdm(loader):
            wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
            # 计算误差
            sim = torch.sum(mt * mt_pred, dim=1).cpu().numpy()
            d_mw = torch.abs(mw - mw_pred).cpu().numpy().flatten()
            
            # 还原坐标 (这里需要一点技巧，或者直接修改 dataset 返回 idx)
            # 为了简单，我们假设 dataset 顺序没变，重新读一遍 json
            # 这里简化处理：我们无法直接从 Tensor 还原 lat/lon
            # 实际操作：我们修改 Dataset 的 __getitem__ 返回 lat/lon
            # 但为了不改动 Dataset，我们在这里重新读取 json 里的坐标
            pass 
            # (由于代码限制，这里我们用随机数模拟占位，你需要修改 Dataset 返回 lat/lon 才能精准画图)
            # 下面是假设你已经拿到了 lat, lon, depth 的代码逻辑
            
            sim_errs.extend(sim)
            mag_errs.extend(d_mw)
    
    # --- 重新读取 JSON 获取坐标 (笨办法但有效) ---
    import json
    with open(Config.JSON_PATHS['waveforms'], 'r') as f:
        all_data = json.load(f)
    with open(Config.JSON_PATHS['grid'], 'r') as f:
        grid_data = json.load(f)
    
    # 筛选 test set
    test_data = [w for w in all_data if w.get('split') == 'test']
    # 按 Event 去重 (因为 evaluate 是按 Event 来的吗？ dataset 是按 Event 聚合的)
    # Dataset __getitem__ 是按 Event 聚合的。
    # 我们需要获取 Dataset 里的 event keys
    keys = test_ds.event_keys
    
    for k in keys:
        # k = "grid_idx_mech_idx"
        g_idx = k.split('_')[0]
        grid = grid_data[g_idx]
        lats.append(grid['lat'])
        lons.append(grid['lon'])
        # depth
        dep = grid.get('depth_km', grid.get('depth_m', 0)/1000)
        depths.append(dep)
        
    # 截断到实际推理数量
    n = len(sim_errs)
    lats = lats[:n]; lons = lons[:n]; depths = depths[:n]
    
    # --- 绘图 ---
    plt.rcParams['font.sans-serif'] = ['Times New Roman', 'SimHei', 'DejaVu Sans']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    title_fontsize = 18
    label_fontsize = 16
    tick_fontsize = 16
    colorbar_fontsize = 16
    
    # # 图 1: 平面位置误差 (颜色=相似度，越黄越好，越蓝越差)
    # plt.figure(figsize=(8, 6))
    # sc1 = plt.scatter(lons, lats, c=sim_errs, cmap='RdYlBu', s=10, alpha=0.8, vmin=0.5, vmax=1.0)
    # cbar1 = plt.colorbar(sc1)
    # cbar1.set_label('震源机制相似度', fontsize=colorbar_fontsize)
    # cbar1.ax.tick_params(labelsize=tick_fontsize)
    # plt.xlabel("经度", fontsize=label_fontsize)
    # plt.ylabel("纬度", fontsize=label_fontsize)
    # # plt.title("震源机制准确度分布图", fontsize=title_fontsize)
    # plt.xticks(fontsize=tick_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    # plt.tight_layout()
    # plt.savefig("vis_spatial_map.svg", dpi=600)
    # plt.close()


    # 图 1: 平面位置误差 (颜色=相似度，越黄越好，越蓝越差)
    plt.figure(figsize=(8, 6))
    sc1 = plt.scatter(lons, lats, c=sim_errs, cmap='RdYlBu', s=10, alpha=0.8, vmin=0.5, vmax=1.0)
    cbar1 = plt.colorbar(sc1)
    cbar1.set_label('Focal Mechanism Cosine Similarity', fontsize=colorbar_fontsize)
    cbar1.ax.tick_params(labelsize=tick_fontsize)
    plt.xlabel("Longitude", fontsize=label_fontsize)
    plt.ylabel("Latitude", fontsize=label_fontsize)
    # plt.title("震源机制准确度分布图", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig("vis_spatial_map.svg", dpi=600)
    plt.close()




    
    # # 图 2: 深度 vs 震级误差
    # plt.figure(figsize=(8, 6))
    # sc2 = plt.scatter(depths, mag_errs, c=mag_errs, cmap='viridis', s=10, alpha=0.6)
    # cbar2 = plt.colorbar(sc2)
    # cbar2.set_label('震级绝对误差', fontsize=colorbar_fontsize)
    # cbar2.ax.tick_params(labelsize=tick_fontsize)
    # plt.xlabel("深度 (km)", fontsize=label_fontsize)
    # plt.ylabel("Mw误差", fontsize=label_fontsize)
    # # plt.title("误差与深度的关系", fontsize=title_fontsize)
    # plt.xticks(fontsize=tick_fontsize)
    # plt.yticks(fontsize=tick_fontsize)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("vis_spatial_depth.svg", dpi=600)
    # plt.close()




    # 图 2: 深度 vs 震级误差
    plt.figure(figsize=(8, 6))
    sc2 = plt.scatter(depths, mag_errs, c=mag_errs, cmap='viridis', s=10, alpha=0.6)
    cbar2 = plt.colorbar(sc2)
    cbar2.set_label('Absolute Magnitude Error', fontsize=colorbar_fontsize)
    cbar2.ax.tick_params(labelsize=tick_fontsize)
    plt.xlabel("Deep (km)", fontsize=label_fontsize)
    plt.ylabel("Mw Error", fontsize=label_fontsize)
    # plt.title("误差与深度的关系", fontsize=title_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("vis_spatial_depth.svg", dpi=600)
    plt.close()
    
    print("Saved vis_spatial_map.svg and vis_spatial_depth.svg")

if __name__ == "__main__":
    main()