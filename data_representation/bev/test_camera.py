#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import yaml
from camera_ipm_bev import CameraIPMBEV  # adjust import if needed
from git import Repo
from nuscenes.nuscenes import NuScenes


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
        description="Standalone script to generate radar BEV occupancy grid"
    )
    p.add_argument(
        "--config", "-c",
        required=True,
        help="Path to config.yaml"
    )
    args = p.parse_args()

    # 1) Load YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)


    # 2) init NuScenes once
    ds = cfg["data_source"]
    nusc = NuScenes(version=ds["version"], dataroot=ds["path"], verbose=False)

    # 3) pick first sample token
    sample_token = nusc.sample[0]['token']
    # sample_token = '82597244941b4e79aa3e1e9cc6386f8b'
    # sample = nusc.sample[0]
    # sample_token = sample['token']
    print(f"Using sample token: {sample_token}")

    # 4) build & run BEV generator
    bev_gen = CameraIPMBEV(config=cfg['data_representation'], nusc=nusc)
    bev = bev_gen.run(sample_token)  # Use the first sample token

if __name__ == "__main__":
    main()
