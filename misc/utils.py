from pathlib import Path
from git import Repo

def resolve_output_dir(output_dir_template, config_path):
    # Get repo root (up to 'freespace-detector')
    repo_root = Repo(Path(__file__).parent, search_parent_directories=True).working_dir

    # Get config file stem (filename without extension)
    config_file_stem = Path(config_path).stem

    # Replace placeholders
    output_dir = output_dir_template.replace("{REPO}", str(repo_root))
    output_dir = output_dir.replace("{CONFIG_FILE_STEM}", config_file_stem)
    return output_dir