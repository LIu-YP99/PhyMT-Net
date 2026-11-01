import torch
import torch.optim as optim
import os
import sys
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.getcwd())

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.model import OptimalFMTNet
from src.ablation_models import ModelNoGeo, ModelNoTrans, ModelPlainCNN, ModelCoupledHead
from src.loss import GeodesicMSELoss
from src.utils import EarlyStopping

def get_model(ablation_type):
    if ablation_type == "no_geo":
        return ModelNoGeo()
    elif ablation_type == "no_trans":
        return ModelNoTrans()
    elif ablation_type == "plain_cnn":
        return ModelPlainCNN()
    elif ablation_type == "coupled_head":
        return ModelCoupledHead()
    elif ablation_type == "full":
        return OptimalFMTNet()
    else:
        raise ValueError(f"Unknown ablation type: {ablation_type}")

def train_ablation(args):
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"ablation_{args.ablation_type}_{start_time}.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    print("="*60)
    print(f"🚀 ABLATION TRAINER STARTING at {start_time}")
    print(f"Ablation Type: {args.ablation_type}")
    print(f"Model File   : {model_name}")
    print("="*60)
    
    train_ds = DynamicSeismicDataset(split='train', augment=True)
    val_ds = DynamicSeismicDataset(split='val', augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    model = get_model(args.ablation_type).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, 
        epochs=Config.EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    criterion = GeodesicMSELoss(alpha=Config.LOSS_ALPHA, beta=Config.LOSS_BETA)
    
    save_path = os.path.join(Config.SAVE_DIR, model_name)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, verbose=True, path=save_path)
    
    log_file = os.path.join(Config.LOG_DIR, f"ablation_{args.ablation_type}_{start_time}.txt")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        losses, shp_losses, mag_losses = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        
        for batch in pbar:
            wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
            if torch.isnan(mt_pred).any():
                continue

            loss, l_shp, l_mag = criterion(mt_pred, mt, mw_pred, mw)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            shp_losses.append(l_shp.item())
            mag_losses.append(l_mag.item())
            
            curr_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'L': f"{loss.item():.3f}", 'S': f"{l_shp.item():.3f}", 'lr': f"{curr_lr:.1e}"})
            
        model.eval()
        v_losses, v_shp, v_mag = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
                mt_pred, mw_pred = model(wav, geo, amp, mask)
                l, ls, lm = criterion(mt_pred, mt, mw_pred, mw)
                v_losses.append(l.item()); v_shp.append(ls.item()); v_mag.append(lm.item())
        
        avg_t = np.mean(losses)
        avg_v = np.mean(v_losses)
        msg = f"Epoch {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f} (S:{np.mean(v_shp):.3f})"
        print(msg)
        
        with open(log_file, "a") as f: f.write(msg + "\n")
        
        early_stopping(avg_v, model)
        if early_stopping.early_stop:
            print("Early Stopping!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Experiments Runner")
    parser.add_argument("--ablation_type", type=str, required=True, 
                        choices=["no_geo", "no_trans", "plain_cnn", "coupled_head", "full"],
                        help="Type of ablation experiment to run")
    args = parser.parse_args()
    train_ablation(args)