import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from git import Repo
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import (BoxVisibility, points_in_box,
                                           transform_matrix, view_points)
from pyquaternion import Quaternion

from misc.utils import resolve_output_dir


class BEVFootPrintSegmentationGT:
    """
    Generates BEV occupancy masks from NuScenes using your original logic,
    with cleaner structure and robust logging.
    """
    def __init__(self, config: dict, nusc: NuScenes, output_dir: Path, log: logging.Logger):
        self.nusc = nusc
        # Pre‐compute grid shape from config
        self.long_range = config['bev']['long_range']
        self.lat_range  = config['bev']['lat_range']
        self.resolution = config['bev']['resolution']
        self.output_dir = output_dir
        self.log = log
        self.height = int(
            (self.long_range[1] -  self.long_range[0]) / self.resolution
        )
        self.width = int(
            ( self.lat_range[1] -  self.lat_range[0]) / self.resolution
        )
        self.bev_shape = (self.height, self.width)
        self.visualize = config['bev'].get('visualize', False)
        self.target = (640, 360)

    def visualize_bev(self, bev_mask: Optional[np.ndarray]) -> None:
        plt.imshow(bev_mask)
        plt.show()

    def adjust_K_for_resize(K, src=(1600,900), dst=(1365,768), crop=(0,0)):
        K = K.copy().astype(float)
        sx, sy = dst[0]/src[0], dst[1]/src[1]
        ox, oy = crop
        K[0,0] *= sx;  K[1,1] *= sy
        K[0,2] = sx*K[0,2] - ox
        K[1,2] = sy*K[1,2] - oy
        return K

    def make_foot_print_seg_mask(self, token: str) -> np.ndarray:
        sample = self.nusc.get('sample', token)
        cam_token = sample['data']['CAM_FRONT']
        image_path, boxes, K = self.nusc.get_sample_data(cam_token,
                                    box_vis_level=BoxVisibility.ANY,
                                    use_flat_vehicle_coordinates=False)
        sd = self.nusc.get('sample_data', cam_token)
        H, W = sd['height'], sd['width']
        out = np.zeros((H, W, 1), np.uint8)   # black background
        input_img = cv2.imread(image_path)

        for box in boxes:
            name = getattr(box, 'name', '')
            if not (isinstance(name, str) and name.startswith('vehicle.')):
                continue
            if name in ['vehicle.bicycle', 'vehicle.motorcycle', 'vehicle.construction']:
                continue
            # 3D corners in cam frame; bottom face = indices 0..3
            C3d = box.bottom_corners()
            if np.any(C3d[2, :4] <= 1e-6):
                continue  # skip if any bottom corner behind camera

            # Project bottom 4 corners to pixels
            uv = view_points(C3d[:, :4], K, normalize=True)[:2].T 
            # uv[:, 0] = np.clip(uv[:, 0], 0, W - 1)
            # uv[:, 1] = np.clip(uv[:, 1], 0, H - 1)
            pts = uv.astype(np.int32)
            cv2.fillPoly(out, [pts], 255)
        
        out = cv2.resize(out, self.target)
        input_img = cv2.resize(input_img, self.target)
        return out, input_img

    def run(self, token: str) -> None:
        bev_mask, img_out = self.make_foot_print_seg_mask(token)
        try:
            save_mask_path = self.output_dir / f"{token}_mask.npy"
            save_image_path = self.output_dir / f"{token}_img.npy"
            if self.visualize:
                self.nusc.render_sample_data(self.nusc.get('sample', token)['data']['CAM_FRONT'])
                self.visualize_bev(img_out)
                #self.visualize_bev(bev_mask)
            np.save(save_mask_path, bev_mask)
            np.save(save_image_path, img_out)

        except Exception as e:
            self.log.error(f"Error rendering/saving sample {token}: {e}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BEV masks from NuScenes dataset.")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to the configuration YAML file.")
    parser.add_argument("--log", type=str, default="INFO",
                        help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return parser.parse_args()

def setup_logging(log_path, level="INFO"):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),                         # console
            logging.FileHandler(log_path, mode='w', encoding="utf-8")  # file
        ],
    )


def main() -> None:
    args = parse_args()
    
    config_path = Path(args.config)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    nusc = NuScenes(
        version=cfg['data_source']['version'],
        dataroot=cfg['data_source']['path'],
        verbose=True
    )
    output_dir = Path(resolve_output_dir(cfg['output_dir'], config_path))

    # Setup logging
    log_path = output_dir / "bev_footprint_seg_gt.log"
    setup_logging(log_path)
    log = logging.getLogger(__name__)
    bev_gt = BEVFootPrintSegmentationGT(cfg['data_representation'], nusc, output_dir, log)

    # Preserve your original sampling of all samples:
    for s in nusc.sample[0:]:
        bev_gt.run(s['token'])

if __name__ == "__main__":
    main()