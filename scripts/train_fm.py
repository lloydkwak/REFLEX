import os
import torch
import torch.nn as nn
import numpy as np
import wandb
import argparse
import copy
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from src.planner.model import ReflexFMNetwork
from src.planner.flow_matching import ReflexFlowMatcher
from src.datasets.robomimic_dataset import RobomimicSE3Dataset

def set_seed(seed):
    """ Sets seeds for reproducibility across torch, numpy, and random. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class EMA:
    """ 
    Exponential Moving Average for weight smoothing. 
    Standard practice in Diffusion and Flow Matching models for inference stability.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].data = (1.0 - self.decay) * param.data + self.decay * self.shadow[name].data

    def apply_to_model(self, model_to_update):
        """ Overwrites the model weights with EMA shadow weights. """
        for name, param in model_to_update.named_parameters():
            param.data.copy_(self.shadow[name].data)

def train():
    parser = argparse.ArgumentParser(description="REFLEX Research-Grade Training Script")
    parser.add_argument("--project", type=str, default="REFLEX-Project")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., PickPlaceCan)")
    parser.add_argument("--exp_name", type=str, default="OT-CFM-Chunk-Tp16")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=16, help="Prediction horizon (chunk size)")
    args = parser.parse_args()
    
    set_seed(args.seed)

    # 1. Monitoring & Directory Setup
    wandb.init(project=args.project, name=f"{args.task}-{args.exp_name}", config=args)
    save_dir = f"checkpoints/{args.task}/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Dataset & Pipeline Setup
    # Uses Tp=16 action chunking and 6-channel multi-view fusion
    dataset = RobomimicSE3Dataset(args.data_path, prediction_horizon=args.tp)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # Network initialized with the specific prediction horizon
    model = ReflexFMNetwork(pred_horizon=args.tp).to(args.device)
    fm_engine = ReflexFlowMatcher(model).to(args.device) 
    
    ema = EMA(model, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')

    # 3. Training Loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch in pbar:
                img = batch["image"].to(args.device)  # (B, 6, 224, 224)
                pose = batch["pose"].to(args.device)  # (B, Tp * 6)
                
                optimizer.zero_grad()
                
                # Compute OT-CFM straight-path regression loss
                loss = fm_engine.compute_loss(x_1=pose, image=img) 
                
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()
                
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                wandb.log({"iter_loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        scheduler.step()
        
        wandb.log({"epoch_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "epoch": epoch})
        print(f"Epoch {epoch} Finished. Avg Loss: {avg_loss:.6f}")

        # 4. Checkpointing & EMA Deployment
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'ema_shadow': ema.shadow,
            'args': vars(args)
        }
        
        torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))
        
        if is_best:
            # Prepare best model for deployment with smoothed weights
            best_model_cpu = copy.deepcopy(model).cpu()
            ema.apply_to_model(best_model_cpu)
            torch.save(best_model_cpu.state_dict(), os.path.join(save_dir, "best_ema.pth"))
            print(f"New Best Model Saved with Loss: {best_loss:.6f}")

    wandb.finish()

if __name__ == "__main__":
    train()
