import torch
import torch.optim as optim
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Ensure root path is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.dataset import DynamicSeismicDataset
from src.loss import GeodesicMSELoss
from src.utils import EarlyStopping
from compare.model import ComparisonModel

def train():
    # 1. Setup
    start_time = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"compare_model_{start_time}.pth"
    
    # Distributed Setup
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_distributed = world_size > 1
    
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        local_rank = 0

    is_main_process = (not is_distributed) or (rank == 0)

    # Output directories
    save_dir = os.path.join(Config.PROJECT_ROOT, "compare", "checkpoints")
    log_dir = os.path.join(Config.PROJECT_ROOT, "compare", "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    if is_main_process:
        print("="*60)
        print(f"🚀 COMPARISON TRAINER STARTING at {start_time}")
        print(f"Model File   : {model_name}")
        print(f"Config       : Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}")
        print("="*60)
    
    # 2. Data
    train_ds = DynamicSeismicDataset(split='train', augment=True)
    val_ds = DynamicSeismicDataset(split='val', augment=False)
    
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    
    per_device_batch = max(1, Config.BATCH_SIZE // world_size) if is_distributed else Config.BATCH_SIZE
    
    train_loader = DataLoader(
        train_ds, batch_size=per_device_batch, 
        shuffle=(not is_distributed), sampler=train_sampler,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=per_device_batch, 
        shuffle=False, sampler=val_sampler,
        num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=False
    )
    
    # 3. Model
    model = ComparisonModel().to(device)
    
    if is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LEARNING_RATE, 
        epochs=Config.EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3
    )
    
    criterion = GeodesicMSELoss(alpha=Config.LOSS_ALPHA, beta=Config.LOSS_BETA)
    
    save_path = os.path.join(save_dir, model_name)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, verbose=True, path=save_path)
    
    log_file = os.path.join(log_dir, f"log_{start_time}.txt")
    
    # 4. Loop
    for epoch in range(Config.EPOCHS):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        model.train()
        losses, shp_losses, mag_losses = [], [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", disable=not is_main_process)
        
        for batch in pbar:
            wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            # Comparison model only takes wav (and implicitly handles geo/mask inside if designed, but here only wav)
            # The prompt model doesn't mention geo/mask inputs explicitly, just (1, 48, 128) input.
            # My implementation takes (wav, geo, amp, mask) to match signature but ignores extra args if needed.
            mt_pred, mw_pred = model(wav, geo, amp, mask)
            
            if torch.isnan(mt_pred).any():
                if is_main_process: print("!! NaN detected. Skipping batch.")
                continue

            loss, l_shp, l_mag = criterion(mt_pred, mt, mw_pred, mw)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            shp_losses.append(l_shp.item())
            mag_losses.append(l_mag.item())
            
            if is_main_process:
                curr_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'L': f"{loss.item():.3f}", 'S': f"{l_shp.item():.3f}", 'lr': f"{curr_lr:.1e}"})
            
        # Validation
        model.eval()
        v_losses, v_shp, v_mag = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                wav, geo, mt, mw, amp, mask = [x.to(device) for x in batch]
                mt_pred, mw_pred = model(wav, geo, amp, mask)
                l, ls, lm = criterion(mt_pred, mt, mw_pred, mw)
                v_losses.append(l.item()); v_shp.append(ls.item()); v_mag.append(lm.item())
        
        # Aggregate
        avg_t = float(np.mean(losses)) if len(losses) > 0 else 0.0
        avg_v_local = float(np.mean(v_losses)) if len(v_losses) > 0 else 0.0
        avg_s_local = float(np.mean(v_shp)) if len(v_shp) > 0 else 0.0
        
        if is_distributed:
            t = torch.tensor([avg_v_local, avg_s_local], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_v = (t[0].item() / world_size)
            avg_s = (t[1].item() / world_size)
        else:
            avg_v = avg_v_local
            avg_s = avg_s_local

        if is_main_process:
            msg = f"Epoch {epoch+1} | Train: {avg_t:.4f} | Val: {avg_v:.4f} (S:{avg_s:.3f})"
            print(msg)
            with open(log_file, "a") as f: f.write(msg + "\n")
            
            model_to_save = model.module if is_distributed else model
            early_stopping(avg_v, model_to_save)
            if early_stopping.early_stop:
                print("Early Stopping!")
                break

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
