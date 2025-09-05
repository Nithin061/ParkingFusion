import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, fcn_resnet50
from tqdm import tqdm
from losses import get_loss_fn, focal_plus_dice_loss
from bev_dataset import BEVDataset
from models.unet import UNetSeg
from misc.utils import resolve_output_dir
from torch.utils.tensorboard import SummaryWriter

# ─────────────────────────────── MODELS ───────────────────────────────
def get_model(name: str, in_channels: int, num_classes: int):
    """
    name: 'fcn' or 'deeplabv3'
    """
    if name == "fcn":
        model = fcn_resnet50(pretrained=True, aux_loss=False)
        # adapt first conv & classifier
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        model.classifier[4]  = nn.Conv2d(512, num_classes, 1)
    elif name == "deeplabv3":
        model = deeplabv3_resnet50(pretrained=True)
        model.backbone.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        model.classifier[4]  = nn.Conv2d(256, num_classes, 1)
    elif name == "unet":
        model = UNetSeg(in_channels=in_channels, num_classes=num_classes, base_ch=32)
    else:
        raise ValueError(f"Unknown model '{name}'")
    return model

def dice_from_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum((1,2,3))
    denom = probs.sum((1,2,3)) + target.sum((1,2,3))
    dice  = (2*inter + eps) / (denom + eps)
    return dice.mean()

def dice_from_logits_threshold(logits, target, thresh = 0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs  = (probs > thresh).float()
    inter = (probs * target).sum((1,2,3))
    denom = probs.sum((1,2,3)) + target.sum((1,2,3))
    dice  = (2*inter + eps) / (denom + eps)
    return dice.mean()

@torch.no_grad()
def iou_from_logits(logits, target, thresh=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    pred  = (probs > thresh).float()
    inter = (pred * target).sum((1,2,3))
    union = (pred + target).clamp(0,1).sum((1,2,3))
    iou   = (inter + eps) / (union + eps)
    return iou.mean()

def bce_plus_dice_loss(logits, target, alpha: float = 0.5):
    """
    Returns:
      loss: scalar (alpha * BCE + (1-alpha) * (1 - Dice))
      bce:  scalar (for logging)
      dice: scalar (for logging, higher is better)
    """
    criterion = nn.BCEWithLogitsLoss() 
    bce  = criterion(logits, target)     # BCE on logits
    dice = dice_from_logits(logits, target)  # soft Dice (no threshold)
    loss = alpha * bce + (1.0 - alpha) * (1.0 - dice)
    return loss, bce, dice
 
# ─────────────────────────────── TRAINING ───────────────────────────────
def train(cfg_path: Path):
    cfg = yaml.safe_load(cfg_path.read_text())
    data_path  = Path(cfg["data_path"])
    gt_path  = Path(cfg["gt_path"])
    model_name  = cfg.get("model", "deeplabv3")   # 'fcn' or 'deeplabv3'
    batch_size  = cfg.get("batch_size", 8)
    lr          = cfg.get("lr", 1e-3)
    epochs      = cfg.get("epochs", 20)
    val_split   = cfg.get("val_split", 0.2)
    save_path   = Path(cfg.get("save_path", "freespace_model.pth"))
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir  = Path(resolve_output_dir(
        cfg.get("output_dir", "freespace_detector_output"), cfg_path.name)
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Dataset + split
    full_ds = BEVDataset(data_path=data_path, gt_path=gt_path)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    writer = SummaryWriter(log_dir=output_dir / "metrics")  # one per experiment
    # Model, loss, optimizer
    # determine in_channels from one sample
    sample_x, _, _ = full_ds[0]
    in_ch = sample_x.shape[0]
    model = get_model(model_name, in_channels=in_ch, num_classes=1).to(device)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = get_loss_fn(cfg.get('loss', 'focal'),
                            alpha=float(cfg.get('focal_alpha', 0.25)),
                            gamma=float(cfg.get('focal_gamma', 2.0)),
                            lam_dice=float(cfg.get('lam_dice', 0.3)))
    optimizer = optim.Adam(model.parameters(), lr=float(lr))
    # Best‐model tracking
    # 1) LR scheduler: watch val loss
    scheduler = ReduceLROnPlateau(
                                  optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=3,
                                  threshold=1e-4,
                                  cooldown=0,
                                  min_lr=0.0,)
    best_val = float('inf')
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for x,y,_ in tqdm(train_ld, desc=f"Epoch {epoch}/{epochs} [Train]"):
            x,y = x.to(device), y.to(device)
            out = model(x)["out"]
            # loss, _, _ = criterion(out, y)
            loss= criterion(out, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ld.dataset)

        model.eval()
        val_loss = 0.0
        total_iou, total_dice = 0.0, 0.0
        with torch.no_grad():
            for x,y,_ in tqdm(val_ld, desc=f"Epoch {epoch}/{epochs} [Val]"):
                x,y = x.to(device), y.to(device)
                out = model(x)["out"]
                # loss, _, _ = criterion(out, y)
                loss= criterion(out, y)
                val_loss += loss.item() * x.size(0)

                # compute predictions & metrics
                iou_batch = iou_from_logits(out, y)
                dice_batch = dice_from_logits_threshold(out, y)
                total_iou += iou_batch.item()
                total_dice += dice_batch.item()

        val_loss /= n_val
        avg_iou   = total_iou  / len(val_ld)
        avg_dice  = total_dice / len(val_ld)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / f"{save_path}_{epoch}.pth")
            print(f"  → New best model (val_loss={best_val:.4f}) saved to {output_dir}")
        print(f"Epoch {epoch}/{epochs} — train: {train_loss:.4f}, val: {val_loss:.4f} IoU: {avg_iou:.4f}  Dice: {avg_dice:.4f}")
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val",   val_loss,   epoch)
        writer.add_scalar("metrics/iou/val",    avg_iou,    epoch)
        writer.add_scalar("metrics/dice/val",   avg_dice,   epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        scheduler.step(val_loss)
    writer.close()

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=Path, required=True)
    args = parser.parse_args()
    train(args.config)