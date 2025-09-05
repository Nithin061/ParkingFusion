
import torch
import torch.nn.functional as F
from typing import Callable, Tuple

def _as_float(targets: torch.Tensor, logits_shape: torch.Size) -> torch.Tensor:
    # Ensure targets are float and shaped like logits
    if targets.dtype != torch.float32 and targets.dtype != torch.float16 and targets.dtype != torch.bfloat16:
        targets = targets.float()
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
    if targets.shape != logits_shape:
        # Broadcast if needed
        targets = targets.expand(logits_shape)
    return targets

def binary_focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Numerically stable binary focal loss on logits.
    logits:  [B,1,H,W] (unnormalized)
    targets: [B,1,H,W] or [B,H,W] with 0/1 values
    """
    targets = _as_float(targets, logits.shape)
    # BCE with logits (no reduction) gives the stable log-sum-exp form under the hood
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # p = sigmoid(logits) without materializing full probs for gradients twice
    # We still need p for pt; sigmoid is stable in PyTorch.
    p = torch.sigmoid(logits)
    pt = p * targets + (1.0 - p) * (1.0 - targets)  # p_t
    # Class-balancing term (alpha) — weight positives vs negatives
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = w * (1.0 - pt).clamp(min=eps).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def dice_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Soft Dice loss on probabilities (sigmoid(logits)).
    Returns 1 - Dice.
    """
    targets = _as_float(targets, logits.shape)
    probs = torch.sigmoid(logits)
    # Flatten per-sample
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    denom = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    loss = 1.0 - dice  # per sample
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss

def focal_plus_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lam_dice: float = 0.3,
    reduction: str = "mean"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss = focal + lam_dice * dice.
    Returns (total, focal, dice) for logging.
    """
    fl = binary_focal_loss_with_logits(logits, targets, alpha=alpha, gamma=gamma, reduction=reduction)
    dl = dice_loss_from_logits(logits, targets, reduction=reduction)
    total = fl + lam_dice * dl
    return total, fl, dl

def get_loss_fn(kind: str, **kwargs) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Returns a callable loss function based on `kind`.
    Supported:
      - 'focal'        -> binary_focal_loss_with_logits
      - 'focal_dice'   -> lambda returns (total, fl, dl) like above
      - 'bce_dice'     -> classic BCE + Dice using BCEWithLogits
    kwargs may include alpha, gamma, lam_dice, pos_weight, etc.
    """
    kind = kind.lower()
    if kind == "focal":
        alpha = float(kwargs.get("alpha", 0.25))
        gamma = float(kwargs.get("gamma", 2.0))
        def _fn(logits, targets):
            return binary_focal_loss_with_logits(logits, targets, alpha=alpha, gamma=gamma, reduction="mean")
        return _fn

    if kind == "focal_dice":
        alpha = float(kwargs.get("alpha", 0.25))
        gamma = float(kwargs.get("gamma", 2.0))
        lam_dice = float(kwargs.get("lam_dice", 0.3))
        def _fn(logits, targets):
            total, _, _ = focal_plus_dice_loss(logits, targets, alpha=alpha, gamma=gamma, lam_dice=lam_dice, reduction="mean")
            return total
        return _fn

    if kind == "bce_dice":
        lam_dice = float(kwargs.get("lam_dice", 0.3))
        pos_weight = kwargs.get("pos_weight", None)
        def _fn(logits, targets):
            targets_ = _as_float(targets, logits.shape)
            bce = F.binary_cross_entropy_with_logits(
                logits, targets_, reduction="mean", pos_weight=pos_weight
            )
            dl = dice_loss_from_logits(logits, targets_, reduction="mean")
            return bce + lam_dice * dl
        return _fn

    raise ValueError(f"Unknown loss kind: {kind}")
