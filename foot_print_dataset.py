from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FootPrintDataset(Dataset):
    def __init__(self, data_path: Path, gt_path: Path, normalize: bool = False):
        self.data_path = Path(data_path)
        self.normalize = normalize

        # make sure you only try to load tokens for which all three files exist
        gt = {p.stem.strip('_mask') for p in (self.data_path).glob("*_mask.npy")}
        cam = {p.stem.strip('_img') for p in (self.data_path).glob("*_img.npy")}
        self.tokens = sorted(gt & cam)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = self.tokens[idx]

        # load
        cam = np.load(self.data_path/f"{tok}_img.npy")   # H×W or H×W×C
        mask = np.load(self.data_path/f"{tok}_mask.npy")    # H×W (0/1)
        # cam = cam / 255
        # prepare camera: ensure shape → C×H×W
        if cam.ndim == 2:
            cam = cam[None, ...]          # [1,H,W]
        # else:
        #     cam = cam.transpose(1, 2, 0)    # [C,H,W]

        # radar → [1,H,W]
        # rad = rad[None, ...]

        # stack → [1+C, H, W]
        x = cam.transpose(2, 0, 1)
        x = x.astype(np.float32)
        x = x / 255.0
        # optional joint min‐max normalize to [0,1]
        if self.normalize:
            mn, mx = x.min(), x.max()
            x = (x - mn) / (mx - mn + 1e-6)

        # mask → binary [1,H,W]
        y = mask.astype(np.float32)[None, ...]
        y=y/255.0

        return torch.from_numpy(x), torch.from_numpy(y)