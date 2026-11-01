# evaluate_fmnet.py

import torch
import numpy as np
import sys
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from FMNet import FMNet
from src.metrics import (
    calculate_cosine_similarity,
    calculate_iso_error,
    calculate_mean_kagan_angle_dc_only,
    calculate_hudson_distance
)

def evaluate_fmnet():
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load Test Data
    print("Loading test dataset...")
    test_ds = DynamicSeismicDataset(split='test', augment=False)
    loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)

    # 2. Load FMNet Model
    model = FMNet().to(device)

    ckpt_path = "/root/data/lmt/DeepFMT_2025_12_1/checkpoints/fmnet_mt_20260703_1537.pth"

    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return

    print(f"Loading weights from {ckpt_path}...")
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(">>> Success! FMNet weights loaded.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 3. Inference
    all_mt_true = []
    all_mt_pred = []

    print("Starting inference on test set...")
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="FMNet Testing"):
            wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]

            mt_pred, _ = model(wav, geo, amp, mask)

            all_mt_true.append(mt_true.cpu().numpy())
            all_mt_pred.append(mt_pred.cpu().numpy())

    y_true_mt = np.concatenate(all_mt_true, axis=0)
    y_pred_mt = np.concatenate(all_mt_pred, axis=0)

    print(f"Inference done in {time.time() - start_time:.2f}s")
    print(f"Total samples: {len(y_true_mt)}")

    # ============================================================
    # 4. Calculate Metrics
    # ============================================================
    print("\n" + "="*60)
    print("  FMNet (MT-Only) Evaluation Results")
    print("="*60)

    # a. Cosine Similarity
    cos_sim = calculate_cosine_similarity(y_true_mt, y_pred_mt)
    print(f"\n  (a) Cosine Similarity (Avg) : {cos_sim:.4f}")

    # b. Kagan Angle
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

    # c. Hudson Distance
    print("  (c) Calculating Hudson Distance...")
    hudson_dist = calculate_hudson_distance(y_true_mt, y_pred_mt)
    print(f"      Hudson Distance (Avg)   : {hudson_dist:.4f}")

    # d. ISO Error
    iso_err = calculate_iso_error(y_true_mt, y_pred_mt)
    print(f"  (d) ISO Error(% diff)              : {iso_err:.4f}")

    print("\n" + "="*60)
    print("  Evaluation Complete.")
    print("="*60)


if __name__ == "__main__":
    evaluate_fmnet()