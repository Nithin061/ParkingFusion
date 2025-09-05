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


class BEVFreeSpaceGT:
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
    
    def load(self, token: str):
        """
        Load sample and compute global->ego transform via LIDAR_TOP ego_pose.
        (Same logic preserved.)
        """
        try:
            sample = self.nusc.get("sample", token)
            lidar_data = self.nusc.get("sample_data", sample["data"]["CAM_FRONT"])
            ego_pose = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
            global_to_ego = transform_matrix(
                ego_pose["translation"],
                Quaternion(ego_pose["rotation"]),
                inverse=True
            )
            return sample, global_to_ego
        except Exception as e:
            self.log.error(f"Error loading data for sample {token}: {e}")
            return None, None
    
    def visualize_bev(self, bev_mask: Optional[np.ndarray]) -> None:
        plt.imshow(bev_mask)
        plt.show()

    @staticmethod
    def box_bottom_corners_global_to_ego_xy(box, global_to_ego: np.ndarray) -> np.ndarray:
        """
        Take bottom corners (global), transform to ego, return (4,2) as XY.
        """
        corners = box.bottom_corners()                    # (3,4)
        corners = np.vstack((corners, np.ones((1, 4))))   # (4,4)
        corners_ego = global_to_ego @ corners             # (4,4)
        return corners_ego[:2, :].T  

    def corners_within_ranges(self, corners_xy: np.ndarray) -> bool:
        """Check if any corner lies within configured ranges."""
        x_ok = (corners_xy[:, 0] >= self.long_range[0]) & (corners_xy[:, 0] <= self.long_range[1])
        y_ok = (corners_xy[:, 1] >= self.lat_range[0])  & (corners_xy[:, 1] <= self.lat_range[1])
        return np.any(x_ok & y_ok)

    def get_radar_points_global(self, sample) -> Optional[np.ndarray]:
        """
        Load RADAR_FRONT and return points in GLOBAL frame, shape (3, N).
        Returns None if sensor missing.
        """
        radar_token = sample["data"].get("RADAR_FRONT")
        if radar_token is None:
            return None

        radar_data = self.nusc.get("sample_data", radar_token)
        radar_path = self.nusc.get_sample_data_path(radar_token)
        radar_pc   = RadarPointCloud.from_file(radar_path)

        radar_calib   = self.nusc.get("calibrated_sensor", radar_data["calibrated_sensor_token"])
        radar_to_ego  = transform_matrix(radar_calib["translation"], Quaternion(radar_calib["rotation"]), inverse=False)
        radar_pose    = self.nusc.get("ego_pose", radar_data["ego_pose_token"])
        ego_to_global = transform_matrix(radar_pose["translation"], Quaternion(radar_pose["rotation"]), inverse=False)

        radar_to_global = ego_to_global @ radar_to_ego
        pts_h = np.vstack((radar_pc.points[:3, :], np.ones((1, radar_pc.points.shape[1]))))  # (4,N)
        return (radar_to_global @ pts_h)[:3]

    def anns_in_cam_fov(self, nusc: NuScenes, sample_token: str, cam_channel: str = 'CAM_FRONT'):
        sample = nusc.get('sample', sample_token)
        sd_token = sample['data'][cam_channel]

        # Returns only boxes that pass the chosen visibility test in this camera view.
        data_path, boxes, K = nusc.get_sample_data(
            sd_token,
            box_vis_level=BoxVisibility.ALL  # use BoxVisibility.ALL for stricter “entire box in view”
        )
        ann_tokens = [b.token for b in boxes]  # sample_annotation tokens
        return ann_tokens

    # def ann_in_cam_fov(self, nusc, ann_token, cam_channel="CAM_FRONT", margin_px=0):
    #     """
    #     Return True if the annotation's *bottom* corners project inside the camera image.
    #     Uses the camera's timestamp extrinsics (ego_pose + calibrated_sensor).
    #     """
    #     ann   = nusc.get('sample_annotation', ann_token)
    #     samp  = nusc.get('sample', ann['sample_token'])
    #     sd    = nusc.get('sample_data', samp['data'][cam_channel])

    #     W, H  = sd['width'], sd['height']
    #     cs    = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    #     pose  = nusc.get('ego_pose', sd['ego_pose_token'])
    #     K     = np.array(cs['camera_intrinsic'])

    #     # Get box in WORLD
    #     box_w = nusc.get_box(ann_token).copy()

    #     # ---- WORLD -> EGO @ cam_time
    #     box_e = box_w.copy()
    #     box_e.translate(-np.array(pose['translation']))
    #     box_e.rotate(Quaternion(pose['rotation']).inverse)

    #     # Identify bottom-face corners in EGO frame (z-up world/ego)
    #     corners_e = box_e.corners()              # (3, 8)
    #     z = corners_e[2]
    #     bottom_idx = np.where(np.isclose(z, z.min(), atol=1e-3))[0]   # 4 indices

    #     # ---- EGO -> CAMERA
    #     box_c = box_e.copy()
    #     box_c.translate(-np.array(cs['translation']))
    #     box_c.rotate(Quaternion(cs['rotation']).inverse)

    #     # Corners in camera frame (z-forward)
    #     corners_c = box_c.corners()              # (3, 8)
    #     bc = corners_c[:, bottom_idx]            # bottom corners only

    #     # Must be in front of camera
    #     in_front = bc[2, :] > 0
    #     if not in_front.any():
    #         return False

    #     # Project to pixels and test bounds with margin
    #     pts = view_points(bc, K, normalize=True)   # (3, 4)
    #     u, v = pts[0, :], pts[1, :]

    #     inside = ((u >= -margin_px) & (u < W + margin_px) &
    #             (v >= -margin_px) & (v < H + margin_px) &
    #             in_front)

    #     return bool(inside.all())


    def make_bev_mask(self, sample_token: str) -> Optional[np.ndarray]:
        """
        Generate BEV occupancy mask for one sample..
        """
        sample, global_to_ego = self.load(sample_token)
        if sample is None or global_to_ego is None:
            return None

        radar_points_global = self.get_radar_points_global(sample)
        occupied_mask = np.zeros(self.bev_shape, dtype=np.uint8)
        front_cam_fov_anns = self.anns_in_cam_fov(self.nusc, sample_token, cam_channel='CAM_FRONT')
        for ann_token in sample["anns"]:
            try:
                ann = self.nusc.get("sample_annotation", ann_token)
                box = self.nusc.get_box(ann_token)
                corners_xy = self.box_bottom_corners_global_to_ego_xy(
                    box, global_to_ego
                )

                visibility = self.nusc.get("visibility", ann["visibility_token"])["token"]
                # Keep your exact logic:
                # but the code actually skips only visibility '1'.
                
                visible_to_radar = points_in_box(box, radar_points_global)

                # check not visible to camera and not visible to radar
                # if not (ann_token in front_cam_fov_anns and visibility in [ '4']) \
                #     and (not visible_to_radar.any()):
                #     continue
                # is_camera_fov = self.ann_in_cam_fov(self.nusc, ann_token, "CAM_FRONT")
                if  (not (ann_token in front_cam_fov_anns and visibility in ['4'])) \
                    and (not visible_to_radar.any()):
                    continue
                # Check if any corner is within configured ranges
                if not self.corners_within_ranges(corners_xy):
                    continue

                # Convert to pixel coordinates
                px = (corners_xy[:, 1] - self.lat_range[0]) / self.resolution
                py = (corners_xy[:, 0] - self.long_range[0]) / self.resolution
                polygon = np.stack([px, py], axis=1).astype(np.int32)

                cv2.fillPoly(occupied_mask, [polygon], color=1)

            except Exception as e:
                self.log.warning(f"Failed to process annotation {ann_token}: {e}")

        return occupied_mask
    
    def run(self, token: str) -> None:

        bev_mask = self.make_bev_mask(token)
        try:
            save_path = self.output_dir / f"{token}.npy"
            rotated = np.rot90(bev_mask, k=2)
            if self.visualize:
                self.nusc.render_sample_data(self.nusc.get('sample', token)['data']['CAM_FRONT'])
                self.visualize_bev(rotated)
            np.save(save_path, rotated)
            
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
    log_path = output_dir / "bev_freespace_gt.log"
    setup_logging(log_path)
    log = logging.getLogger(__name__)
    bev_gt = BEVFreeSpaceGT(cfg['data_representation'], nusc, output_dir, log)

    # Preserve your original sampling of all samples:
    for s in nusc.sample[0:1]:
        bev_gt.run(s['token'])

if __name__ == "__main__":
    main()