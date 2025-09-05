from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BEVDataset(Dataset):
    def __init__(self, data_path: Path, gt_path: Path, normalize: bool = False):
        self.data_path = Path(data_path)
        self.gt_path = Path(gt_path)
        self.normalize = normalize

        # make sure you only try to load tokens for which all three files exist
        cam = {p.stem for p in (self.data_path/"camera_bev").glob("*.npy")}
        rad = {p.stem for p in (self.data_path/"radar_bev").glob("*.npy")}
        msk = {p.stem for p in (self.gt_path).glob("*.npy")}
        self.tokens = sorted(cam & rad & msk)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tok = self.tokens[idx]

        # load
        cam = np.load(self.data_path/"camera_bev"/f"{tok}.npy")   # H×W or H×W×C
        rad = np.load(self.data_path/"radar_bev"/f"{tok}.npy")    # H×W (0/1)
        mask = np.load(self.gt_path/f"{tok}.npy")         # H×W (0/255 or 0/1)
        # cam = cam / 255
        # prepare camera: ensure shape → C×H×W
        if cam.ndim == 2:
            cam = cam[None, ...]          # [1,H,W]
        # else:
        #     cam = cam.transpose(1, 2, 0)    # [C,H,W]

        # radar → [1,H,W]
        # rad = rad[None, ...]

        # stack → [1+C, H, W]
        x = np.concatenate([rad, cam], axis=0).astype(np.float32)

        # optional joint min‐max normalize to [0,1]
        if self.normalize:
            mn, mx = x.min(), x.max()
            x = (x - mn) / (mx - mn + 1e-6)

        # mask → binary [1,H,W]
        y = mask.astype(np.float64)[None, ...]

        return torch.from_numpy(x), torch.from_numpy(y), tok