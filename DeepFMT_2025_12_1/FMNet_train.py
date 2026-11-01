# train_fmnet_mt.py

import torch
import torch.optim as optim
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from FMNet import FMNet
from src.loss_FMNet import GeodesicMSELossMTOnly
from src.utils import EarlyStopping

def train_fmnet():
    # 1. 准备时间戳
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"fmnet_mt_{start_time}.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    print("="*60)
    print(f"🚀 FMNet MT‑Only Training at {start_time}")
    print(f"Model File   : {model_name}")
    print(f"Config       : Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, "
          f"LR={Config.LEARNING_RATE}")
    print("="*60)

    # 2. 数据（与主模型完全一致）
    train_ds = DynamicSeismicDataset(split='train', augment=True)
    val_ds   = DynamicSeismicDataset(split='val',   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE,
        shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE,
        shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    # 3. 模型
    model = FMNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"FMNet Parameters: {total_params:,}")

    # 4. Optimizer（与主模型一致）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # 5. Scheduler: OneCycleLR（与主模型一致）
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )

    # 6. Loss: MT‑only（alpha=1.0）
    criterion = GeodesicMSELossMTOnly(alpha=1.0)

    # 7. Early Stopping（patience = 25，与主模型一致）
    save_path = os.path.join(Config.SAVE_DIR, model_name)
    early_stopping = EarlyStopping(
        patience=25,     # ✅ 与主模型一致
        verbose=True,
        path=save_path
    )

    log_file = os.path.join(Config.LOG_DIR, f"log_fmnet_{start_time}.txt")

    # 8. 训练循环
    for epoch in range(Config.EPOCHS):
        # ===== Train =====
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"FMNet Epoch {epoch+1}/{Config.EPOCHS}")

        for batch in pbar:
            wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]

            optimizer.zero_grad()

            mt_pred, _ = model(wav, geo, amp, mask)

            # ✅ 拆包（关键）
            loss, loss_shape, loss_mag = criterion(mt_pred, mt_true, None, None)

            if torch.isnan(loss):
                print("!! NaN detected. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # ✅ 与主模型一致
            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            pbar.set_postfix({
                'L': f"{loss.item():.4f}",
                'S': f"{loss_shape.item():.4f}"
            })

        # ===== Validation =====
        model.eval()
        val_losses = []
        val_shape_losses = []

        with torch.no_grad():
            for batch in val_loader:
                wav, geo, mt_true, mw_true, amp, mask = [x.to(device) for x in batch]
                mt_pred, _ = model(wav, geo, amp, mask)

                # ✅ 拆包
                loss, loss_shape, loss_mag = criterion(mt_pred, mt_true, None, None)

                val_losses.append(loss.item())
                val_shape_losses.append(loss_shape.item())

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        avg_val_s = np.mean(val_shape_losses)

        msg = (f"Epoch {epoch+1} | Train: {avg_train:.4f} "
               f"| Val: {avg_val:.4f} (S: {avg_val_s:.3f})")
        print(msg)

        with open(log_file, "a") as f:
            f.write(msg + "\n")

        # ===== Early Stopping =====
        early_stopping(avg_val, model)
        if early_stopping.early_stop:
            print("Early Stopping triggered!")
            break

    print(f"\n🎉 FMNet training complete!")
    print(f"Best val loss : {early_stopping.best_score:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train_fmnet()