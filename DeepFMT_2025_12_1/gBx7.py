import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
from obspy.imaging.beachball import beach
from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.getcwd())
from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelPlainCNN

REAL_DATA_PATH = "/root/data/lmt/data/output_real_exhaustive"
if os.path.exists(REAL_DATA_PATH):
    Config.DATA_ROOT = REAL_DATA_PATH
    Config.JSON_PATHS = {
        "waveforms": os.path.join(REAL_DATA_PATH, "waveforms.json"),
        "source":    os.path.join(REAL_DATA_PATH, "sources.json"),
        "grid":      os.path.join(REAL_DATA_PATH, "grid_points.json"),
        "receivers": os.path.join(REAL_DATA_PATH, "receivers.json") 
    }
else:
    sys.exit(1)

def convert_ned_to_gcmt(m_ned):
    mxx, myy, mzz, mxy, mxz, myz = m_ned
    mrr = mzz
    mtt = mxx
    mpp = myy
    mrt = mxz
    mrp = -myz 
    mtp = -mxy
    return np.array([mrr, mtt, mpp, mrt, mrp, mtp])

def get_event_info(dataset, event_key):
    evt_data = dataset.events[event_key]
    first_wave = evt_data['waveforms'][0]
    fname = os.path.basename(first_wave['waveform_file'])
    event_id = fname.split('_')[0]
    g_idx = str(first_wave['grid_point_idx'])
    m_idx = str(first_wave['mechanism_idx'])
    src = dataset.sources[g_idx][m_idx]
    
    # Load grid coordinates
    with open(Config.JSON_PATHS["grid"], "r") as f:
        grids = json.load(f)
    g_info = grids[g_idx]
    
    return {
        "event_id": event_id,
        "origin_time": src.get('origin_time', 'N/A'),
        "lon": g_info.get('lon'),
        "lat": g_info.get('lat'),
        "depth_km": g_info.get('depth_km')
    }

def calculate_metrics(mt_pred, mt_true):
    norm_p = np.linalg.norm(mt_pred)
    norm_t = np.linalg.norm(mt_true)
    cosine = np.dot(mt_pred, mt_true) / (norm_p * norm_t + 1e-9)
    return cosine

def save_single_beachball(mt_gcmt, filename, color):
    # Save purely the beachball with a transparent background
    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    try:
        b = beach(mt_gcmt, xy=(0.5, 0.5), width=1.0, linewidth=1.0, facecolor=color, alpha=0.9)
        ax.add_collection(b)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.0)
    except Exception as e:
        print(f"Error drawing beachball: {e}")
    plt.close(fig)

def visualize_real_events():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    output_dir = "individual_results"
    os.makedirs(output_dir, exist_ok=True)
    
    ds_list = []
    for split in ['test', 'val', 'train']:
        try:
            ds = DynamicSeismicDataset(split=split, augment=False)
            if len(ds) > 0: ds_list.append(ds)
        except: pass
        
    if not ds_list: return
        
    all_samples = []
    for ds in ds_list:
        for i in range(len(ds)):
            all_samples.append((ds, i))
            
    print(f"Total events found: {len(all_samples)}")
    
    # To save meta-information
    results_list = []
    
    model = ModelPlainCNN().to(device)
    ckpt_path = os.path.join(Config.SAVE_DIR, "ablation_plain_cnn_20260609_1456.pth")
    if not os.path.exists(ckpt_path):
         files = os.listdir(Config.SAVE_DIR)
         best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
         if best_models:
             best_models.sort(reverse=True)
             ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])
             
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    for row_idx, sample_idx in enumerate(range(len(all_samples))):
        ds, idx = all_samples[sample_idx]
        wav, geo, mt_true, mw_true, amp, mask = ds[idx]
        wav, geo, amp, mask = wav.unsqueeze(0).to(device), geo.unsqueeze(0).to(device), amp.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        
        with torch.no_grad():
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
        m_p = mt_pred.cpu().numpy()[0]
        mw_p = mw_pred.cpu().numpy()[0].item()
        m_t = mt_true.numpy()
        mw_t = mw_true.numpy().item()
        
        key = ds.event_keys[idx]
        info = get_event_info(ds, key)
        sim = calculate_metrics(m_p, m_t)
        
        evt_id = info['event_id']
        info['sim'] = float(sim)
        info['mw_pred'] = float(mw_p)
        info['mw_true'] = float(mw_t)
        
        # Save independent beachballs
        pred_path = os.path.join(output_dir, f"{evt_id}_pred.png")
        true_path = os.path.join(output_dir, f"{evt_id}_true.png")
        
        save_single_beachball(convert_ned_to_gcmt(m_p), pred_path, color='red')
        save_single_beachball(convert_ned_to_gcmt(m_t), true_path, color='gray')
        
        results_list.append(info)
        print(f"Processed Event: {evt_id}")

    # Save metainfo to a json file
    with open(os.path.join(output_dir, "predictions_info.json"), "w") as f:
        json.dump(results_list, f, indent=4)
        
    print(f"All saved inside directory '{output_dir}'.")

if __name__ == "__main__":
    visualize_real_events()
