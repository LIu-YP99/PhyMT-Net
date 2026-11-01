import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from obspy.imaging.beachball import beach
from torch.utils.data import DataLoader, ConcatDataset

# Ensure path is correct
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

# ==========================================
# 0. 配置覆盖 (指向真实数据)
# ==========================================
REAL_DATA_PATH = "/root/data/lmt/data/output_real_exhaustive"
if os.path.exists(REAL_DATA_PATH):
    print(f"Using Real Data from: {REAL_DATA_PATH}")
    Config.DATA_ROOT = REAL_DATA_PATH
    Config.JSON_PATHS = {
        "waveforms": os.path.join(REAL_DATA_PATH, "waveforms.json"),
        "source":    os.path.join(REAL_DATA_PATH, "sources.json"),
        "grid":      os.path.join(REAL_DATA_PATH, "grid_points.json"),
        "receivers": os.path.join(REAL_DATA_PATH, "receivers.json") 
    }
else:
    print(f"Error: Real data path {REAL_DATA_PATH} not found.")
    sys.exit(1)

# ==========================================
# 1. 辅助函数
# ==========================================
def convert_ned_to_gcmt(m_ned):
    # NED: x(N), y(E), z(D)
    # GCMT (USE): r(Up), t(South), p(East)
    # r = -z, t = -x, p = y
    # Mrr = Mzz
    # Mtt = Mxx
    # Mpp = Myy
    # Mrt = Mzx = Mxz
    # Mrp = Mzy = -Myz (注意：yz 在 NED 里是 y*z，在 USE 里是 r*p = (-z)*y = -yz)
    # Mtp = Mxy = -Mxy (t*p = (-x)*y = -xy)
    
    # 修正 Pyrocko/Obspy 的定义:
    # Obspy beachball uses (M11, M22, M33, M12, M13, M23) corresponds to (Mrr, Mtt, Mpp, Mrt, Mrp, Mtp)
    # NED: xx, yy, zz, xy, xz, yz
    mxx, myy, mzz, mxy, mxz, myz = m_ned
    mrr = mzz
    mtt = mxx
    mpp = myy
    mrt = mxz
    mrp = -myz 
    mtp = -mxy
    return np.array([mrr, mtt, mpp, mrt, mrp, mtp])

def get_event_id_and_time(dataset, event_key):
    # 从 Dataset 的结构中解析 ID
    # key = "grid_idx_mech_idx"
    evt_data = dataset.events[event_key]
    first_wave = evt_data['waveforms'][0]
    
    # 从文件名解析 ID: "waveforms/202010010031A_PB.B088.mseed" -> "202010010031A"
    fname = os.path.basename(first_wave['waveform_file'])
    event_id = fname.split('_')[0]
    
    # 获取时间 (从 sources.json)
    g_idx = str(first_wave['grid_point_idx'])
    m_idx = str(first_wave['mechanism_idx'])
    src = dataset.sources[g_idx][m_idx]
    origin_time = src.get('origin_time', 'N/A')
    
    # 格式化时间 2020-10-01T00:31:29.500000Z -> 2020-10-01 00:31
    if 'T' in origin_time:
        origin_time = origin_time.split('.')[0].replace('T', ' ')
        
    return event_id, origin_time

def calculate_metrics(mt_pred, mt_true):
    norm_p = np.linalg.norm(mt_pred)
    norm_t = np.linalg.norm(mt_true)
    cosine = np.dot(mt_pred, mt_true) / (norm_p * norm_t + 1e-9)
    
    def get_iso(mt_arr):
        # NED 3x3
        m = np.array([[mt_arr[0], mt_arr[3], mt_arr[4]],
                      [mt_arr[3], mt_arr[1], mt_arr[5]],
                      [mt_arr[4], mt_arr[5], mt_arr[2]]])
        try:
            vals = np.linalg.eigvalsh(m)
            # ISO % = tr(M) / sum(|eigenvals|)
            return (np.sum(vals) / (np.sum(np.abs(vals)) + 1e-9)) * 100
        except: return 0.0

    return cosine, get_iso(mt_pred), get_iso(mt_true)

# ==========================================
# 2. 绘图主程序
# ==========================================
def visualize_real_events():
    # 字体设置 (模拟论文)
    plt.rcParams['font.family'] = 'sans-serif' # or 'serif' for Times
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载所有 Split 的数据以确保有足够样本
    ds_list = []
    for split in ['test', 'val', 'train']:
        try:
            ds = DynamicSeismicDataset(split=split, augment=False)
            if len(ds) > 0: ds_list.append(ds)
        except: pass
        
    if not ds_list:
        print("No data found.")
        return
        
    # 我们需要保留 dataset 对象的引用来查元数据，所以不能简单 Concat
    # 我们手动遍历列表
    all_samples = []
    for ds in ds_list:
        for i in range(len(ds)):
            all_samples.append((ds, i))
            
    print(f"Total events found: {len(all_samples)}")
    
    # 随机选择 N 个
    N_SAMPLES = min(8, len(all_samples))
    indices = np.random.choice(len(all_samples), N_SAMPLES, replace=False)
    # 如果想固定看某些 Event，可以手动指定
    # indices = range(N_SAMPLES) 
    
    # 2. 加载模型
    model = ModelPlainCNN().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "/root/data/lmt/DeepFMT_2025_12_1/checkpoints/ablation_plain_cnn_20260609_1456.pth")
    # Fallback
    if not os.path.exists(ckpt_path):
         files = os.listdir(Config.SAVE_DIR)
         best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
         if best_models:
             best_models.sort(reverse=True)
             ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])
             
    print(f"Loading weights from {ckpt_path}...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 3. 准备绘图
    # 布局: N 行 1 列，每行是一个 subfigure 或者 GridSpec
    # 或者直接 N 行 3 列 (Info, True, Pred)
    fig, axes = plt.subplots(N_SAMPLES, 3, figsize=(10, N_SAMPLES * 2.0), 
                             gridspec_kw={'width_ratios': [1.2, 1, 1]})
    
    if N_SAMPLES == 1: axes = np.array([axes])
    
    # 标题头
    cols = ["Event Info", "Reference (GCMT)", "DeepFMT Prediction"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold', pad=10)

    print("Processing events...")
    
    for row_idx, sample_idx in enumerate(indices):
        ds, idx = all_samples[sample_idx]
        
        # 获取数据
        # Dataset __getitem__ returns: wav, geo, mt_true, mw_true, amp, mask
        wav, geo, mt_true, mw_true, amp, mask = ds[idx]
        
        # 增加 Batch 维
        wav = wav.unsqueeze(0).to(device)
        geo = geo.unsqueeze(0).to(device)
        amp = amp.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
        # 转 Numpy
        m_p = mt_pred.cpu().numpy()[0] # NED
        mw_p = mw_pred.cpu().numpy()[0].item()
        m_t = mt_true.numpy() # NED
        mw_t = mw_true.numpy().item()
        
        # 元数据
        key = ds.event_keys[idx]
        evt_id, evt_time = get_event_id_and_time(ds, key)
        
        # 指标
        sim, iso_p, iso_t = calculate_metrics(m_p, m_t)
        mw_err = mw_p - mw_t
        
        # 绘图 - Col 0: Info
        ax_info = axes[row_idx, 0]
        ax_info.axis('off')
        info_text = (
            f"ID: {evt_id}\n"
            f"Time: {evt_time}\n"
            f"Split: {ds.split.upper()}"
        )
        ax_info.text(0.1, 0.5, info_text, va='center', ha='left', fontsize=11, linespacing=1.8)
        # 添加一条竖线分割
        # ax_info.plot([1, 1], [0.2, 0.8], color='gray', linewidth=0.5)
        
        # 绘图 - Col 1: True
        ax_true = axes[row_idx, 1]
        ax_true.axis('off')
        m_t_gcmt = convert_ned_to_gcmt(m_t)
        try:
            b = beach(m_t_gcmt, xy=(0.5, 0.5), width=0.8, linewidth=0.8, facecolor='gray', alpha=0.9)
            ax_true.add_collection(b)
        except: pass
        ax_true.set_aspect('equal')
        ax_true.set_xlim(0, 1); ax_true.set_ylim(0, 1)
        # Label
        ax_true.text(0.5, 0.05, f"Mw {mw_t:.2f} | ISO {iso_t:.0f}%", 
                     ha='center', va='top', fontsize=10)

        # 绘图 - Col 2: Pred
        ax_pred = axes[row_idx, 2]
        ax_pred.axis('off')
        m_p_gcmt = convert_ned_to_gcmt(m_p)
        
        # 颜色根据相似度
        color = '#d62728' if sim > 0.8 else '#1f77b4' # Red if good, Blue if bad? Or Red for Pred.
        color = 'red' # 统一红色表示预测
        
        try:
            b = beach(m_p_gcmt, xy=(0.5, 0.5), width=0.8, linewidth=0.8, facecolor=color, alpha=0.8)
            ax_pred.add_collection(b)
        except: pass
        ax_pred.set_aspect('equal')
        ax_pred.set_xlim(0, 1); ax_pred.set_ylim(0, 1)
        
        # Label with Error
        # 使用数学符号
        label_text = f"Mw {mw_p:.2f} ({mw_err:+.1f})\nISO {iso_p:.0f}%"
        ax_pred.text(0.5, 0.05, label_text, ha='center', va='top', fontsize=10)
        
        # Similarity Score (右上角或右下角)
        ax_pred.text(0.95, 0.95, f"Sim: {sim:.2f}", ha='right', va='top', fontsize=9, 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    plt.tight_layout()
    # 调整顶部空间
    plt.subplots_adjust(top=0.92)
    
    save_file = "real_data_beachballs.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {save_file}")

if __name__ == "__main__":
    visualize_real_events()
