# radar_occupancy.py

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import RadarPointCloud
from bev_base import BaseBEV  # wherever your BaseBEV lives
from typing import Optional
from matplotlib import pyplot as plt
import cv2

class RadarPointOccupancy(BaseBEV):
    def __init__(self, config: dict, nusc: NuScenes, edt_clip_m: float = 10.0,):
        super().__init__(config)
        self.nusc = nusc
        # Pre‐compute grid shape from config
        self.long_range = config['bev']['long_range']
        self.lat_range  = config['bev']['lat_range']
        self.resolution = config['bev']['resolution']
        self.height = int(
            (self.long_range[1] -  self.long_range[0]) / self.resolution
        )
        self.width = int(
            ( self.lat_range[1] -  self.lat_range[0]) / self.resolution
        )
        self.visualize = config['bev'].get('visualize', False)
        self.edt_clip_m = edt_clip_m

    def visualize_bev(self, bev_image: Optional[np.ndarray]) -> None:
        """
        Save a rotated copy (k=2) of the mask to .npy if save_path is provided.
        (All plotting lines remain omitted/commented to preserve your behavior.)
        """
        if bev_image is None:
            self.log.warning("Cannot visualize: Invalid BEV mask.")
            return
        plt.imshow(bev_image[0], extent=[0, self.width, 0, self.height])
        plt.xlabel("Lateral (m)")
        plt.ylabel("Longitudinal (m)")
        plt.show()

    def load(self, token: str):
        # 1) load raw pointcloud + sensor‐to‐ego calib
        radar_sample_data_token = self.nusc.get('sample', token)['data']['RADAR_FRONT']
        sd = self.nusc.get('sample_data', radar_sample_data_token)
        pc = RadarPointCloud.from_file(self.nusc.get_sample_data_path(sd['token']))
        calib = self.nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
        # radar_pc.points shape: (18, N) → You can extract x, y, z using [:3, :]
        radar_points = pc.points[:3, :]  # shape (3, N)
        features = pc.points[5, :].astype(np.float32)
        return (radar_points, features, calib)

    def transform(self, raw):
        # 2) transform into ego‐frame XY
        points, _, calib = raw
        T = transform_matrix(calib['translation'],
                             Quaternion(calib['rotation']),
                             inverse=False)
        pts_h = np.vstack((points, np.ones((1, points.shape[1]))))  # (4,N)
        points_in_egoframe = (T @ pts_h)[:3, :] # (3,N)
        return points_in_egoframe

    def make_bev(self, points_in_egoframe, features):
        # 3) rasterize XY into an occupancy‐grid BEV
        x, y, _ = points_in_egoframe
        lg_min, _ = self.long_range
        lt_min, _ = self.lat_range

        x_idx = ((x - lg_min) / self.resolution).astype(int)
        y_idx = ((y - lt_min) / self.resolution).astype(int)

        bev = np.zeros((self.height, self.width), dtype=np.uint8)
        valid = (x_idx >= 0) & (x_idx < self.height) & (y_idx >= 0) & (y_idx < self.width)
        bev[x_idx[valid], y_idx[valid]] = 1

        # Euclidean Distance Transform on FREE cells
        free = (bev == 0).astype(np.uint8)
        edt_px = cv2.distanceTransform(free, distanceType=cv2.DIST_L2, maskSize=3).astype(np.float32)
        edt_m  = edt_px * self.resolution
        edt_m  = np.clip(edt_m, 0.0, self.edt_clip_m)
        edt    = edt_m / max(self.edt_clip_m, 1e-6)

        # per-cell Rcs (max), robusr nomalize to [0,1]
        rcs_grid = np.full((self.height, self.width), -np.inf, dtype=np.float32)
        np.maximum.at(rcs_grid, (x_idx[valid], y_idx[valid]), features[valid])
        mask = np.isfinite(rcs_grid)
        if mask.any():
            p1, p99 = np.percentile(rcs_grid[mask], [1, 99])
            rcs_norm = (np.clip(rcs_grid, p1, p99) - p1) / max(p99 - p1, 1e-6)
            rcs_norm[~mask] = 0.0
        else:
            rcs_norm = np.zeros_like(rcs_grid, dtype=np.float32)
        

        bev = np.rot90(bev, k=2)
        edt = np.rot90(edt, k=2)
        rcs_norm = np.rot90(rcs_norm, k=2)

        # Stack channels
        bev = np.stack([bev, edt, rcs_norm], axis=0)
        return bev

    def run(self, token):
        """ Run the full BEV pipeline for a given sample token.
        """
        raw = self.load(token)  # Load raw radar points and calibration data
        processed = self.transform(raw)  # Transform points to ego frame
        bev_image = self.make_bev(processed, raw[1])    # Rasterize into BEV occupancy grid
        if self.visualize:
            self.nusc.render_sample_data(self.nusc.get('sample', token)['data']['CAM_FRONT'])
            self.visualize_bev(bev_image)
        return bev_image  # Return the final BEV occupancy grid
