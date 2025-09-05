
import argparse
import os
from pathlib import Path

import cv2
import yaml
from git import Repo
from nuscenes.nuscenes import NuScenes
from radar_points_occupancy_bev import \
    RadarPointOccupancy  # adjust import if needed


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

    # 2) Prepare NuScenes once
    ds = cfg["data_source"]
    nusc = NuScenes(version=ds["version"], dataroot=ds["path"])

    # 3) Instantiate your RadarOccupancy generator
    gen = RadarPointOccupancy(config=cfg['data_representation'], nusc=nusc)

    # 4) Run and get BEV
    sample_token = '3ffc4360d1084e6eae5067e87d79503f'
    sample = nusc.sample[0]
    bev = gen.run(sample_token)  # Use the provided token

    # 5) Save output
    output_dir = resolve_output_dir(cfg['output_dir'], args.config)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{sample['token']}_radar_bev.png")
    cv2.imwrite(out_path, bev[0]*255)
    print(f"Saved BEV to {out_path}")

if __name__ == "__main__":
    main()
