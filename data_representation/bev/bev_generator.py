import argparse
from pathlib import Path

import numpy as np
import yaml
from camera_ipm_bev import CameraIPMBEV
from git import Repo
from nuscenes.nuscenes import NuScenes
from radar_points_occupancy_bev import RadarPointOccupancy


def resolve_output_dir(output_dir_template, config_path):
    # Get repo root (up to 'freespace-detector')
    repo_root = Repo(Path(__file__).parent, search_parent_directories=True).working_dir

    # Get config file stem (filename without extension)
    config_file_stem = Path(config_path).stem

    # Replace placeholders
    output_dir = output_dir_template.replace("{REPO}", str(repo_root))
    output_dir = output_dir.replace("{CONFIG_FILE_STEM}", config_file_stem)
    return output_dir

def main():
    p = argparse.ArgumentParser(
        description="Generate camera & radar BEV maps in .npy format"
    )
    p.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="Path to config.yaml"
    )
    args = p.parse_args()

    # Load config
    cfg = yaml.safe_load(args.config.read_text())
    ds  = cfg["data_source"]
    out_base = Path(resolve_output_dir(cfg['output_dir'], args.config))

    # Create output dirs
    cam_dir = out_base / "camera_bev"
    rad_dir = out_base / "radar_bev"
    cam_dir.mkdir(parents=True, exist_ok=True)
    rad_dir.mkdir(parents=True, exist_ok=True)

    # Load nuScenes once
    nusc = NuScenes(version=ds["version"], dataroot=ds["path"], verbose=False)
    tokens = [s["token"] for s in nusc.sample]

    # Instantiate BEV generators
    cam_gen = CameraIPMBEV(config=cfg['data_representation'], nusc=nusc)
    rad_gen = RadarPointOccupancy(config=cfg['data_representation'], nusc=nusc)

    # Generate and save
    for token in tokens:
        cam_bev = cam_gen.run(token)
        np.save(cam_dir / f"{token}.npy", cam_bev)

        rad_bev = rad_gen.run(token)
        np.save(rad_dir / f"{token}.npy", rad_bev)

        print(f"[{token}] camera & radar BEV saved")

if __name__ == "__main__":
    main()
