from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from bev_base import BaseBEV  # wherever your BaseBEV lives
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion


def quaternion_to_euler(quat):
    q = Quaternion(quat)
    yaw, pitch, roll = q.yaw_pitch_roll
    return roll, pitch, yaw

def motion_cancellation_matrix(roll, pitch, K):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(pitch),  np.sin(pitch)],
        [0, -np.sin(pitch), np.cos(pitch)]
    ])
    Rz = np.array([
        [ np.cos(roll), np.sin(roll), 0],
        [-np.sin(roll), np.cos(roll), 0],
        [ 0,            0,           1]
    ])
    R_corr = Rz @ Rx
    return K @ R_corr @ np.linalg.inv(K)

def motion_cancellation_matrix_from_calib(K, R_EC, roll_rad, pitch_rad, yaw_rad=0.0):
    """
    Cancel small ego rotations (roll,pitch[,yaw]) on the image.
    K: (3x3) intrinsics
    R_EC: (3x3) camera->ego rotation (from calibrated_sensor['rotation'])
    Angles are about EGO axes (x=roll, y=pitch, z=yaw), in radians.
    """
    R_CE = R_EC.T
    omega_E = np.array([roll_rad, pitch_rad, yaw_rad], dtype=float)  # ego axes
    omega_C = R_CE @ omega_E                                         # camera axes
    R_corr, _ = cv2.Rodrigues(-omega_C)                              # inverse to cancel
    H = K @ R_corr @ np.linalg.inv(K)
    return H

def ipm_nuscenes(img, calib, X_max, Y_min, Y_max, res):

    K = np.array(calib['camera_intrinsic'])
    cam_rot_quat = calib['rotation']
    cam_trans = calib['translation']

    bev_H = int((Y_max - Y_min) / res)
    bev_W = int(X_max / res)

    xs = np.linspace(0, X_max, bev_W)
    ys = np.linspace(Y_max, Y_min, bev_H)
    Xg, Yg = np.meshgrid(xs, ys)
    ego_pts = np.stack([Xg.ravel(), Yg.ravel(), np.zeros(bev_H*bev_W)], axis=1)

    R_cam2ego = Quaternion(cam_rot_quat).rotation_matrix
    R_ego2cam = R_cam2ego.T
    t_cam2ego = np.array(cam_trans)
    t_ego2cam = -R_ego2cam @ t_cam2ego

    pts_cam = (ego_pts @ R_ego2cam.T) + t_ego2cam
    in_front = pts_cam[:,2] > 0
    pts_cam_f = pts_cam[in_front]

    uv = pts_cam_f[:,:2] / pts_cam_f[:,2:3]
    uv = uv @ np.array([[K[0,0],0],[0,K[1,1]]]) \
         + np.array([K[0,2],K[1,2]])
    uv = uv.astype(np.float32)

    remap_x = np.full(bev_H*bev_W, -1, dtype=np.float32)
    remap_y = np.full(bev_H*bev_W, -1, dtype=np.float32)
    remap_x[in_front] = uv[:,0]
    remap_y[in_front] = uv[:,1]
    remap_x = remap_x.reshape(bev_H, bev_W)
    remap_y = remap_y.reshape(bev_H, bev_W)

    return cv2.remap(img, remap_x, remap_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


# 3) The concrete BEV class
class CameraIPMBEV(BaseBEV):
    def __init__(self, config: dict, nusc: NuScenes):
        super().__init__(config)
        # Pre‐compute grid shape from config
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

    def visualize_bev(self, bev_image: Optional[np.ndarray]) -> None:
        """
        Save a rotated copy (k=2) of the mask to .npy if save_path is provided.
        (All plotting lines remain omitted/commented to preserve your behavior.)
        """
        if bev_image is None:
            self.log.warning("Cannot visualize: Invalid BEV mask.")
            return
        bev_image = cv2.cvtColor(bev_image, cv2.COLOR_RGB2BGR)  # Convert BGR to RGB for matplotlib
        plt.imshow(bev_image)
        plt.xlabel("Lateral (m)")
        plt.ylabel("Longitudinal (m)")
        plt.show()

    def load(self, token: str):
        sample   = self.nusc.get('sample', token)
        cam_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        ego_pos  = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        calib    = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        # img      = cv2.imread(self.nusc.get_sample_data_path(cam_data['token']))
        img = cv2.imread('./bottom_corners_black.png')  # use a fixed image for testing
        return img, calib, ego_pos

    def transform(self, raw):
        img, calib, ego_pos = raw
        K = np.array(calib['camera_intrinsic'])
        roll, pitch, _ = quaternion_to_euler(ego_pos['rotation'])
        # H = motion_cancellation_matrix(roll, pitch, K)
        # img = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
        return img

    def make_bev(self, raw):
        img = self.transform(raw)
        bev = ipm_nuscenes(img, raw[1],
                           self.long_range[1],
                           self.lat_range[0],
                           self.lat_range[1],
                           self.resolution)
        bev = cv2.rotate(bev, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        # Sobel gradients (ksize must be 1, 3, 5, or 7; 3 is standard)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Gradient magnitude
        edges = np.sqrt(gx*gx + gy*gy)
        m = edges.max()
        if m > 1e-6:
            edges /= m
        # ensure numeric range
        gray  = np.clip(gray,  0.0, 1.0).astype(np.float32)
        edges = np.clip(edges, 0.0, 1.0).astype(np.float32)
        return np.stack([gray, edges], axis=0)

    def run(self, token):
        """ Run the full BEV pipeline for a given sample token.
        """
        raw = self.load(token)
        bev = self.make_bev(raw)
        if self.visualize:
            self.nusc.render_sample_data(self.nusc.get('sample', token)['data']['CAM_FRONT'])
            self.visualize_bev(bev[0])
        return bev

