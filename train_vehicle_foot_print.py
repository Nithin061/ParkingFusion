# train_with_your_dataset.py
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import ResNet101_Weights
from foot_print_dataset import FootPrintDataset
from tqdm import tqdm
# --------------- model: DeepLabv3+ R101 -------------------
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def make_deeplab_r101(pretrained_backbone: bool = True) -> nn.Module:
    model = deeplabv3_resnet101(
            weights=None,
            weights_backbone=ResNet101_Weights.IMAGENET1K_V1,
            num_classes=1,
            aux_loss=None,
        )
    return model

# --------------- batch sanitizers (consume your dataset as-is) -----
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

@torch.no_grad()
def iou_f1(logits: torch.Tensor, target: torch.Tensor, thr=0.5):
    p = torch.sigmoid(logits)
    pred = (p >= thr).float()
    tp = (pred * target).sum(dim=(1,2,3))
    fp = (pred * (1 - target)).sum(dim=(1,2,3))
    fn = ((1 - pred) * target).sum(dim=(1,2,3))
    iou = tp / (tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    return iou.mean().item(), f1.mean().item()

def train_image_plane_with_your_dataset(
    root: str | Path,
    epochs=200,                 # paper
    batch_size=8,               # paper ~8–10
    lr=1e-3,                    # paper
    momentum=0.9,               # paper
    weight_decay=1e-4,
    pos_weight=2.0,             # tune per dataset
    pretrained_backbone=True,   # paper used R101 encoder; ImageNet helps
):
    # --- dataset & splits
    ds = FootPrintDataset(Path(root), Path(root), normalize=False)  # use exactly your class
    assert len(ds) > 0, "Dataset is empty (check your *_img.npy / *_mask.npy files)."

    n = len(ds)
    n_val = max(50, int(0.1 * n))
    n_tr  = n - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False,)

    # --- model, loss, optim (paper-style)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_deeplab_r101(pretrained_backbone=pretrained_backbone).to(device)
    model = model.to(device=device, dtype=torch.float32)
    # Paper uses BCE; we keep it strictly BCE here
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_iou = 0.0
    for ep in range(1, epochs + 1):
        # -------- train --------
        model.train()
        run_loss = 0.0
        seen = 0
        for x, y in tqdm(dl_tr):
            opt.zero_grad(set_to_none=True)
            logits = model(x)["out"]           # (B,1,H,W)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            bs = x.size(0)
            run_loss += loss.item() * bs
            seen += bs
        tr_loss = run_loss / max(1, seen)

        # -------- validate --------
        model.eval()
        v_loss = 0.0
        v_iou = 0.0
        v_f1  = 0.0
        v_seen = 0
        with torch.no_grad():
            for x, y in tqdm(dl_va):
                logits = model(x)["out"]
                v_loss += criterion(logits, y).item() * x.size(0)
                i, f = iou_f1(logits, y, thr=0.5)
                v_iou += i * x.size(0)
                v_f1  += f * x.size(0)
                v_seen += x.size(0)

        val_loss = v_loss / max(1, v_seen)
        val_iou  = v_iou / max(1, v_seen)
        val_f1   = v_f1  / max(1, v_seen)

        print(f"[{ep:03d}] train {tr_loss:.4f} | val {val_loss:.4f} | IoU {val_iou:.3f} | F1 {val_f1:.3f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save({"model": model.state_dict(),
                        "pretrained_backbone": pretrained_backbone},
                       Path(root) / f"deeplab_r101_vehicle_fp_best_{epochs}.pt")

    return model

def validate_image_plane_with_your_dataset(root: str | Path, batch_size=8):
    ds = FootPrintDataset(Path(root), Path(root), normalize=False)  # use exactly your class
    assert len(ds) > 0, "Dataset is empty (check your *_img.npy / *_mask.npy files)."

    n = len(ds)
    n_val = max(50, int(0.1 * n))
    n_tr  = n - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))

    dl_tr = DataLoader(tr, batch_size=batch_size, shuffle=True,)
    dl_va = DataLoader(va, batch_size=batch_size, shuffle=False,)

    # --- model, loss, optim (paper-style)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(root) / f"deeplab_r101_vehicle_fp_best.pt", map_location=device)
    model = make_deeplab_r101(pretrained_backbone=True).to(device)
    model.load_state_dict(ckpt.get("model", ckpt))

    # -------- validate --------
    model.eval()
    v_loss = 0.0
    v_iou = 0.0
    v_f1  = 0.0
    v_seen = 0
    with torch.no_grad():
        for x, y in tqdm(dl_va):
            logits = model(x)["out"]
            i, f = iou_f1(logits, y, thr=0.5)
            v_iou += i * x.size(0)
            v_f1  += f * x.size(0)
            v_seen += x.size(0)

    val_iou  = v_iou / max(1, v_seen)
    val_f1   = v_f1  / max(1, v_seen)
    print(f"IoU {val_iou:.3f} | F1 {val_f1:.3f}")
# python -i train_with_your_dataset.py
model = train_image_plane_with_your_dataset(r"./local/cfg_bev_foot_print_seg_gt",
                                            epochs=200, batch_size=8, pos_weight=2.0,
                                            pretrained_backbone=True)

validate_image_plane_with_your_dataset(r"./local/cfg_bev_foot_print_seg_gt", 6)
