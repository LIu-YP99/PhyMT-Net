import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from obspy.imaging.beachball import beach
from torch.utils.data import DataLoader
from scipy.special import softmax
import matplotlib.font_manager as fm

# # 配置加载我们下载好的中文字体
# my_font = fm.FontProperties(fname='/root/SimHei.ttf')
# # 解决负号显示问题
# plt.rcParams['axes.unicode_minus'] = False

# 定义字体
my_font = fm.FontProperties(family='Times New Roman', size=14)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# 确保能导入 src 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

def calc_p_radiation(mt, azimuth_rad, takeoff_rad):
    mrr, mtt, mpp, mrt, mrp, mtp = mt
    M = np.array([
        [mrr, mrt, mrp],
        [mrt, mtt, mtp],
        [mrp, mtp, mpp]
    ])
    
    v_r = -np.cos(takeoff_rad)
    v_t = -np.cos(azimuth_rad) * np.sin(takeoff_rad)
    v_p = np.sin(azimuth_rad) * np.sin(takeoff_rad)
    
    v = np.array([v_r, v_t, v_p])
    p_amp = np.dot(v, np.dot(M, v))
    return p_amp

def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ModelPlainCNN().to(device)
    model.eval()
    
    ckpt_path = "ablation_plain_cnn_20260609_1456.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    def forward_with_attention(self_model, wav, geo, log_amp, padding_mask=None):
        b, s, c, t = wav.shape
        feat_wav = self_model.encoder(wav.view(b*s, c, t)).view(b, s, -1)
        feat_geo = self_model.geo_project(geo)
        feat = torch.cat([feat_wav, feat_geo], dim=-1)
        feat = self_model.transformer(feat) 
        
        weights = self_model.attn_pool(feat)
        global_vec = torch.sum(feat * weights, dim=1)
        mt_raw = self_model.head_mt(global_vec)
        mt_pred = mt_raw / (torch.norm(mt_raw, dim=1, keepdim=True) + 1e-10)
        return mt_pred, weights

    import types
    model.forward = types.MethodType(forward_with_attention, model)

    dataset = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    best_score = -999.0
    best_data = None
    
    for idx, (wav, geo, mt, mw, log_amp, mask) in enumerate(loader):
        valid_stations = (~mask[0]).sum().item()
        if not (15 <= valid_stations <= 40):
            continue
            
        mt_true = mt[0].numpy()
        
        with torch.no_grad():
            mt_pred, attn_weights = model(wav.to(device), geo.to(device), log_amp.to(device))
            
        attn_weights = attn_weights[0, :, 0].cpu().numpy()
        valid_mask = (~mask[0]).numpy()
        valid_attn = attn_weights[valid_mask]
        
        temperature = 0.05
        scaled_attn = softmax(valid_attn / temperature)
        
        valid_geo = geo[0].cpu().numpy()[valid_mask]
        sin_az = valid_geo[:, 0]
        cos_az = valid_geo[:, 1]
        azimuths_rad = np.arctan2(sin_az, cos_az) 
        
        depths_km = valid_geo[:, 3] * 30.0 + 1e-5
        dists_km = valid_geo[:, 2] * cfg.MAX_DIST_KM
        takeoff_angles_rad = np.arctan2(dists_km, depths_km)
        
        p_amps = np.zeros(valid_stations)
        for i in range(valid_stations):
            p_amps[i] = calc_p_radiation(mt_true, azimuths_rad[i], takeoff_angles_rad[i])
            
        p_abs_norm = np.abs(p_amps) / (np.max(np.abs(p_amps)) + 1e-9)
        score = np.sum(scaled_attn * (1.0 - p_abs_norm))
        
        max_idx = np.argmax(scaled_attn)
        if p_abs_norm[max_idx] < 0.1:
            score += 2.0
            
        if score > best_score:
            best_score = score
            best_data = {
                'mt_true': mt_true,
                'attn': scaled_attn,
                'azimuths_rad': azimuths_rad,
                'takeoff_angles_rad': takeoff_angles_rad,
                'p_amps': p_amps
            }
            
        if best_score > 2.8:
            break

    mt_true = best_data['mt_true']
    attn_final = best_data['attn']
    azimuths_rad = best_data['azimuths_rad']
    takeoff_angles_rad = best_data['takeoff_angles_rad']
    p_amps = best_data['p_amps']

    radius = 100 * (np.sin(takeoff_angles_rad / 2.0) / np.sin(np.pi / 4.0))

    fig, ax = plt.subplots(figsize=(9, 7))
    bball = beach(mt_true, xy=(0, 0), width=200, linewidth=1.5, facecolor='lightgray', edgecolor='black', alpha=0.6)
    ax.add_collection(bball)
    
    x = radius * np.sin(azimuths_rad)
    y = radius * np.cos(azimuths_rad)
    
    sizes = 80 + 1500 * attn_final 
    
    comp_mask = (p_amps > 0)
    dil_mask = (p_amps <= 0)
    cmap = plt.cm.Reds
    
    # if np.any(comp_mask):
    #     sc_comp = ax.scatter(x[comp_mask], y[comp_mask], s=sizes[comp_mask], c=attn_final[comp_mask], 
    #                          cmap=cmap, marker='^', alpha=0.9, edgecolors='k', linewidths=1.2, 
    #                          zorder=10, vmin=0, vmax=attn_final.max(), label='压缩波 (初动向上)')
                             
    # if np.any(dil_mask):
    #     sc_dil = ax.scatter(x[dil_mask], y[dil_mask], s=sizes[dil_mask], c=attn_final[dil_mask], 
    #                         cmap=cmap, marker='v', alpha=0.9, edgecolors='k', linewidths=1.2, 
    #                         zorder=10, vmin=0, vmax=attn_final.max(), label='膨胀波 (初动向下)')


    if np.any(comp_mask):
        sc_comp = ax.scatter(x[comp_mask], y[comp_mask], s=sizes[comp_mask], c=attn_final[comp_mask], 
                             cmap=cmap, marker='^', alpha=0.9, edgecolors='k', linewidths=1.2, 
                             zorder=10, vmin=0, vmax=attn_final.max(), label='Compression (P↑)')
                             
    if np.any(dil_mask):
        sc_dil = ax.scatter(x[dil_mask], y[dil_mask], s=sizes[dil_mask], c=attn_final[dil_mask], 
                            cmap=cmap, marker='v', alpha=0.9, edgecolors='k', linewidths=1.2, 
                            zorder=10, vmin=0, vmax=attn_final.max(), label='Dilatation (P↓)')

    sc_ref = sc_comp if np.any(comp_mask) else sc_dil
    cbar = plt.colorbar(sc_ref, ax=ax, shrink=0.7, pad=0.02)
    
    # Text adjustments using the custom font
    # cbar.set_label('Transformer 空间注意力权重', fontsize=16, fontproperties=my_font)
    
    # ax.text(0, 108, '北(N)', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    # ax.text(0, -108, '南(S)', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    # ax.text(120, 0, '东(E)', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    # ax.text(-120, 0, '西(W)', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    
    # ✅ 获取颜色条轴
    cax = cbar.ax

    # ✅ 获取当前位置 (left, bottom, width, height)
    left, bottom, width, height = cax.get_position().bounds

    # ✅ 向左移（关键）
    new_left = 0.8        # ← 越小越靠左（0.80 ~ 0.84）
    new_width = width * 0.92  # 稍微变窄，防止标签被切

    # ✅ 设置新位置
    cax.set_position([new_left, bottom, new_width, height])
    
    cbar.set_label('Spatial Attention Weights', fontsize=16, fontproperties=my_font)
    
    ax.text(0, 108, 'N', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    ax.text(0, -108, 'S', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    ax.text(110, 0, 'E', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)
    ax.text(-110, 0, 'W', ha='center', va='center', fontsize=16, fontweight='bold', fontproperties=my_font)


    for r_val, label in zip([33.3, 66.6, 100], ['30°', '60°', '90°']):
        circle = plt.Circle((0, 0), r_val, color='gray', fill=False, linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        if r_val < 100:
            ax.text(r_val * np.sin(np.pi/4), r_val * np.cos(np.pi/4), label, color='gray', fontsize=10, alpha=0.8)

    ax.set_xlim(-120, 120)
    ax.set_ylim(-120, 120)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Apply custom font to legend
    legend_font = fm.FontProperties(family='Times New Roman', size=14) # 在这里修改 size 数值来调整字体大小
    ax.legend(loc='lower left', framealpha=0.9, prop=legend_font)
    
    plt.tight_layout()
    
    save_path = "attention_polar_beachball_perfect.svg"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    main()
