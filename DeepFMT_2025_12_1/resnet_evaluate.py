
import torch
import os
import sys
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

# Ensure path is correct
sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from model_resnet50 import model_resnet50
from src.metrics import (
    calculate_cosine_similarity,
    calculate_iso_error,
    calculate_mean_kagan_angle_dc_only,
    calculate_hudson_distance
)


def calculate_metrics():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 1. Load Data
    print("Loading test dataset...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    
    # 2. Load Model
    model = model_resnet50().to(device)
    
    # Logic to find best model
    ckpt_path = os.path.join(Config.SAVE_DIR, "best_model_20260703_1745.pth")




    if not os.path.exists(ckpt_path):
        files = os.listdir(Config.SAVE_DIR)
        best_models = [f for f in files if f.startswith("best_model") and f.endswith(".pth")]
        if best_models:
            best_models.sort(reverse=True)
            ckpt_path = os.path.join(Config.SAVE_DIR, best_models[0])
        else:
            print("Error: No checkpoint found.")
            return

    print(f"Loading weights from {ckpt_path}...")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return
        
    model.eval()
    
    # 3. Inference
    all_mw_true = []
    all_mw_pred = []
    all_mt_true = []
    all_mt_pred = []
    
    print("Starting inference on test set...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
            
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
            all_mw_true.append(mw_true.cpu().numpy())
            all_mw_pred.append(mw_pred.cpu().numpy())
            
            all_mt_true.append(mt_true.cpu().numpy())
            all_mt_pred.append(mt_pred.cpu().numpy())
            
    # Concatenate
    y_true_mw = np.concatenate(all_mw_true).flatten()
    y_pred_mw = np.concatenate(all_mw_pred).flatten()
    
    # MT shape: (N, 6)
    y_true_mt = np.concatenate(all_mt_true, axis=0)
    y_pred_mt = np.concatenate(all_mt_pred, axis=0)
    
    print(f"Inference done in {time.time() - start_time:.2f}s")
    
    # =========================================================
    # (1) 震级指标：MAE, RMSE, R2
    # =========================================================
    mae_mw = mean_absolute_error(y_true_mw, y_pred_mw)
    rmse_mw = np.sqrt(mean_squared_error(y_true_mw, y_pred_mw))
    r2_mw = r2_score(y_true_mw, y_pred_mw)
    
    print("\n" + "="*50)
    print(" >>> (1) Magnitude Metrics (Mw) <<<")
    print("="*50)
    print(f"  MAE  : {mae_mw:.4f}")
    print(f"  RMSE : {rmse_mw:.4f}")
    print(f"  R^2  : {r2_mw:.4f}")
    
    # =========================================================
    # (2) 机制指标
    # =========================================================
    print("\n" + "="*50)
    print(" >>> (2) Mechanism Metrics (MT) <<<")
    print("="*50)
    
    # a. Cosine Similarity
    cos_sim = calculate_cosine_similarity(y_true_mt, y_pred_mt)
    print(f"  a. Cosine Similarity (Avg) : {cos_sim:.4f}")
    
    # b. ISO Error
    iso_err = calculate_iso_error(y_true_mt, y_pred_mt)
    print(f"  b. ISO Error (% diff)      : {iso_err:.4f}")
    
    # c. Hudson Distance
    print("  Calculating Hudson Distance... (may take a moment if using pyrocko)")
    hudson_dist = calculate_hudson_distance(y_true_mt, y_pred_mt)
    print(f"  c. Hudson Distance (Avg)   : {hudson_dist:.4f}")
    
    # d. Kagan Angle
    print("  Calculating Kagan Angle... (may take a moment)")
    kagan_ang, n_dc, n_valid, n_undefined = (
        calculate_mean_kagan_angle_dc_only(
            y_true_mt, y_pred_mt
        )
    )

    print(f"Mean DC Kagan Angle: {kagan_ang:.4f}°")
    print(f"Pure DC reference samples: {n_dc}")
    print(f"Valid DC pairs: {n_valid}")
    print(f"Undefined predicted orientations: {n_undefined}")
    
    print("="*50)
    print("Done.")

if __name__ == "__main__":
    calculate_metrics()
