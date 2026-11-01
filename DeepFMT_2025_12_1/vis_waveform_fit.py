import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import instaseis
import obspy
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

# 确保路径正确
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet

# ================= 配置 =================
NUM_EVENTS_TO_PLOT = 5   # 随机画多少个地震
STATIONS_PER_EVENT = 3   # 每个地震画多少个台站
OUTPUT_DIR = "waveform_fitting_results" # 图片保存文件夹
# =======================================

def ned_to_use(m_ned):
    """
    NED (North, East, Down) -> Instaseis USE (Up, South, East)
    Transform: Up=-z, South=-x, East=y
    """
    return m_ned[2], m_ned[0], m_ned[1], m_ned[4], -m_ned[5], -m_ned[3]

def normalize_trace(data):
    """归一化波形到 [-1, 1] 以对比形状"""
    max_val = np.max(np.abs(data))
    if max_val > 0:
        return data / max_val
    return data

def visualize_multiple_fits():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. 加载 Instaseis 数据库
    print(f"Loading Instaseis DB from {Config.INSTASEIS_DB_PATH}...")
    try:
        db = instaseis.open_db(Config.INSTASEIS_DB_PATH)
    except Exception as e:
        print(f"Error: 无法加载数据库: {e}")
        return

    # 2. 加载数据与模型
    print("Loading Test Dataset...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    if len(test_ds) == 0:
        print("Test dataset is empty.")
        return

    model = OptimalFMTNet().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251202_2257.pth")
    
    if not os.path.exists(ckpt_path):
        # 兼容旧文件名
        ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20251202_2257.pth")
    
    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 3. 随机选择 N 个地震
    total_events = len(test_ds)
    indices = random.sample(range(total_events), min(NUM_EVENTS_TO_PLOT, total_events))
    
    print(f"Selected indices: {indices}")
    
    # 4. 循环处理每个地震
    for idx in tqdm(indices, desc="Generating Plots"):
        # --- 获取输入数据 ---
        wav, geo, mt_true, mw_true, amp, mask = test_ds[idx]
        
        # 增加 batch 维度
        wav_b = wav.unsqueeze(0).to(device)
        geo_b = geo.unsqueeze(0).to(device)
        amp_b = amp.unsqueeze(0).to(device)
        mask_b = mask.unsqueeze(0).to(device)
        
        # --- 模型预测 ---
        with torch.no_grad():
            mt_pred, mw_pred = model(wav_b, geo_b, amp_b, mask_b)
        
        # --- 还原物理参数 ---
        mt_pred_np = mt_pred.cpu().numpy().flatten()
        mw_pred_val = mw_pred.item()
        m0_pred = 10**(1.5 * mw_pred_val + 9.1)
        mt_pred_scaled = mt_pred_np * m0_pred # 绝对地震矩张量
        
        # --- 准备正演 Source ---
        key = test_ds.event_keys[idx]
        evt_data = test_ds.events[key]
        first_wave_meta = evt_data['waveforms'][0]
        g_idx = str(first_wave_meta['grid_point_idx'])
        
        # 获取震源位置
        if g_idx in test_ds.grids:
            grid = test_ds.grids[g_idx]
            depth_m = grid.get('depth_m', grid.get('depth_km', 10)*1000)
            lat_src = grid['lat']
            lon_src = grid['lon']
        else:
            # 兼容真实数据下载脚本可能没存 grid.json 的情况
            # 尝试从 waveform meta 反推或者直接跳过
            print(f"Warning: Grid info not found for event {idx}, skipping.")
            continue
        
        # Instaseis Source
        m_rr, m_tt, m_pp, m_rt, m_rp, m_tp = ned_to_use(mt_pred_scaled)
        origin_time = obspy.UTCDateTime(2020, 1, 1, 0, 0, 0) # 固定时间
        
        source = instaseis.Source(
            latitude=lat_src, longitude=lon_src, depth_in_m=depth_m,
            m_rr=m_rr, m_tt=m_tt, m_pp=m_pp, m_rt=m_rt, m_rp=m_rp, m_tp=m_tp,
            origin_time=origin_time
        )
        
        # --- 选择台站 ---
        # 找出 mask 为 False 的有效台站索引
        valid_indices = torch.where(~mask)[0].tolist()
        
        if len(valid_indices) == 0:
            continue
            
        # 随机选几个台站
        plot_indices = random.sample(valid_indices, min(STATIONS_PER_EVENT, len(valid_indices)))
        
        # --- 绘图 ---
        # 布局: 行=台站, 列=分量(Z, N, E)
        fig, axes = plt.subplots(len(plot_indices), 3, figsize=(15, 3*len(plot_indices)), sharex=True)
        if len(plot_indices) == 1: axes = np.array([axes]) # 保持维度一致
        
        fig.suptitle(f"Event {idx} | True Mw: {mw_true.item():.2f} | Pred Mw: {mw_pred_val:.2f}", fontsize=14)
        
        for row_i, sta_idx in enumerate(plot_indices):
            w_meta = evt_data['waveforms'][sta_idx]
            
            # 构建 Receiver
            rec = instaseis.Receiver(
                latitude=w_meta['receiver_lat'], longitude=w_meta['receiver_lon']
            )
            
            # A. 正演预测波形
            try:
                st_pred = db.get_seismograms(source=source, receiver=rec)
                # 预处理 (必须与训练一致)
                st_pred.filter('bandpass', freqmin=Config.FREQ_MIN, freqmax=Config.FREQ_MAX, corners=4, zerophase=True)
                st_pred.resample(Config.SAMPLE_RATE)
                st_pred.trim(starttime=origin_time, endtime=origin_time+Config.DURATION, pad=True, fill_value=0)
            except Exception as e:
                print(f"Instaseis error: {e}")
                continue
            
            # B. 获取观测波形 (Input)
            # Dataset 返回的是已经归一化过的 Tensor
            obs_tensor = wav[sta_idx].cpu().numpy() # (3, T)
            
            # C. 绘制三个分量
            components = ['Z', 'N', 'E']
            for col_i, comp in enumerate(components):
                ax = axes[row_i, col_i]
                
                # 观测数据 (黑实线)
                obs_data = obs_tensor[col_i]
                
                # 预测数据 (红虚线)
                # Instaseis 返回可能是 Z, R, T 或 Z, N, E，取决于设置，默认是 ZNE
                try:
                    pred_trace = st_pred.select(component=comp)[0]
                    pred_data = pred_trace.data[:Config.N_POINTS]
                    if len(pred_data) < Config.N_POINTS:
                        pred_data = np.pad(pred_data, (0, Config.N_POINTS - len(pred_data)), 'constant')
                except:
                    pred_data = np.zeros_like(obs_data)
                
                # 归一化对比形状
                ax.plot(normalize_trace(obs_data), 'k-', label='Observed', lw=1.2)
                ax.plot(normalize_trace(pred_data), 'r--', label='Synthetic', lw=1.2)
                
                if row_i == 0: ax.set_title(f"Component {comp}")
                if col_i == 0: ax.set_ylabel(f"Station\n{w_meta['receiver_code']}")
                
                # 只在第一个子图显示图例
                if row_i == 0 and col_i == 0:
                    ax.legend(loc='best', fontsize='small')
                    
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 留出标题空间
        save_name = os.path.join(OUTPUT_DIR, f"fit_event_{idx}.png")
        plt.savefig(save_name, dpi=150)
        plt.close() # 关闭图形释放内存
        
    print(f"Done! Saved {len(indices)} images to {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_multiple_fits()