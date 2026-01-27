import sys

# sys.path.append('/home/aigouy/miniconda3/pkgs/cuda-nvcc-tools-12.4.131-h99ab3db_0/bin')
# sys.path.append('/home/aigouy/miniconda3/bin')
# sys.path.append('/home/aigouy/miniconda3/condabin')
import json
import os
import tensorflow as tf

html = 'https://github.com/baigouy/FISH-Dist'

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVcICES"] = ""

print('entered the script')
print('__file__',__file__)

import os.path
import traceback
from batoolset.files.tools import smart_name_parser
from batoolset.utils.loadlist import loadlist
from fishdist.fish_analysis_pipeline import \
    run_analysis
import shutil
import argparse



print('really executing the script')

DEFAULT_CONFIG_NAME = "config.json"


def find_config_in_root(root_path):
    if root_path is None:
        return None

    candidate = os.path.join(root_path, DEFAULT_CONFIG_NAME)
    return candidate if os.path.isfile(candidate) else None


def get_valid_folder(prompt):
    """Prompt the user for a folder path and validate it exists."""
    while True:
        folder = input(prompt).strip()
        if os.path.exists(folder) and os.path.isdir(folder):
            return folder
        print(f"Error: '{folder}' is not a valid directory. Please try again.")

def move_to_DONE_when_DONE(lst, add_extras=False):
    """
    Move a list of files (and optionally associated extra files) into a 'DONE' subdirectory.

    Args:
        lst (list of str): List of file paths to move.
        add_extras (bool, default=False): If True, also move additional files in the same parent
            directory with extensions .png, .npy, or .pdf.

    Returns:
        None: Files are moved to a 'DONE' folder; the function does not return a value.

    Example:
        ```python
        move_to_DONE_when_DONE(['/path/to/file1.txt', '/path/to/file2.txt'], add_extras=True)

        ```
    """

    extras = []
    if add_extras:
        # add the extra reg files
        extras = loadlist(smart_name_parser(lst[0], 'parent')+'/*.png')
        extras += loadlist(smart_name_parser(lst[0], 'parent')+'/*.npy')
        extras += loadlist(smart_name_parser(lst[0], 'parent')+'/*.pdf')

    extras.extend(lst)

    for file in extras:
        filename = smart_name_parser(file,'short')
        parent = smart_name_parser(file, 'parent')
        associated_folder = smart_name_parser(file,'short_no_ext')
        dest = parent+'/DONE'
        os.makedirs(dest, exist_ok=True)

        try:
            shutil.move(file, os.path.join(dest,filename))
            try:
                if os.path.exists(smart_name_parser(file,'full_no_ext')):
                    shutil.move(smart_name_parser(file,'full_no_ext'), dest)
            except:
                print('no output folder --> skipping')
        except:
            traceback.print_exc()
            # print error


def print_folder_logic_with_values():
    """Prints the folder structure and workflow logic with actual values."""
    print("\n" + "=" * 60)
    print("FOLDER STRUCTURE, WORKFLOW LOGIC, AND USAGE")
    print("=" * 60)

    logic = f"""

For a step-by-step tutorial, refer to the online documentation:
{html}
    
1. FOLDER STRUCTURE:

FISH_Dist_analysis/
├── colocs/          # For colocalization analysis and chromatic aberration correction computation
├── controls/        # Stores chromatic aberration corrections from colocs/
└── distances/       # For distance measurements between FISH spots

2. WORKFLOW:
- COLOCALIZATION ANALYSIS:
  * Place images in 'colocs/' to generate chromatic aberration corrections
  * Correction matrices must be saved in 'controls/' to be correct chromatic aberrations of distances

- DISTANCE ANALYSIS:
  * Place images in 'distances/' to measure FISH spot distances
  * If chromatic aberration correction matrices (.npy) exist in 'controls/', they'll be applied to distances images 

3. COMMAND LINE USAGE:

You can run the analysis in several ways:

A) Using a root path (recommended for most users):
   python your_script.py --root-path "/media/teamPrudhomme/EqpPrudhomme2/FISH_Dist_analysis"

B) Using a configuration file (JSON format, recommended for advanced users):
   python your_script.py --config path/to/config.json

Example config.json:
{{
    "root_path": "/media/teamPrudhomme/EqpPrudhomme2/FISH_Dist_analysis",
    "reference_channel": 1,
    "second_spot_channel": [2, 3],
    "pairing_threshold": 30,
    "area_threshold": null,
    "run_seg": true,
    "run_distance_measurement": true,
    "run_ctrls": true,
    "run_gaussian_fit": true,
    "nuclear_model_to_use": "/path/to/nuclear_model",
    "spot_model_to_use": "/path/to/spot_model"
}}

For a step-by-step tutorial, refer to the online documentation:
{html}

"""

    print(logic)
    print("="*60 + "\n")

def print_current_analysis(
    nucleus_channel,
    reference_channel,
    folders,
    controls_path,
    second_spot_channel=None,
    run_gaussian_fit=False,
    nuclear_model_to_use=None,
    spot_model_to_use=None
):
    logic = f"""
CURRENT ANALYSIS:
- Nucleus channel: {nucleus_channel}
- Reference channel: {reference_channel}
- Second spot channel(s): {second_spot_channel}
- Scanning folders: {', '.join(folders)}
- Using registration files from: {controls_path}
- Processing images: *.czi files in the specified folders
- Results will be moved to 'DONE/' subfolders
- Run Gaussian fit: {run_gaussian_fit}
- Nuclear model: {nuclear_model_to_use}
- Spot model: {spot_model_to_use}
"""
    print(logic)
    print("="*60 + "\n")

def get_channel_indices():
    """Prompt the user for channel indices with default values."""
    try:
        first_spot_channel = int(input("Enter the index of the first FISH channel (default: 1): ") or 1)
        second_spot_channel = int(input("Enter the index of the second FISH channel (default: 2): ") or 2)
        return first_spot_channel, second_spot_channel
    except ValueError:
        print("Invalid input. Using default values.")
        return 1, 2

def parse_arguments():
    """Parse command line arguments with all required options."""
    parser = argparse.ArgumentParser(
        description="FISH-Dist: Automated 3D Distance Quantification for Confocal FISH Images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # -------------------------
    # Path arguments
    # -------------------------
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--root-path",
        type=str,
        help="Root path containing 'colocs/', 'controls/', and 'distances/' subfolders"
    )
    path_group.add_argument(
        "--colocs-path",
        type=str,
        help="Path to the colocs folder"
    )

    parser.add_argument(
        "--controls-path",
        type=str,
        help="Path to the controls folder (required if using --colocs-path)"
    )

    parser.add_argument(
        "--distances-path",
        type=str,
        help="Path to the distances folder (required if using --colocs-path)"
    )

    # TODO add an override in case there is a config.json file in the root folder


    # -------------------------
    # Channel arguments
    # -------------------------
    parser.add_argument(
        "--nucleus-channel",
        type=int,
        default=0,
        help="Index of the nucleus channel"
    )

    parser.add_argument(
        "--reference-channel",
        type=int,
        default=1,
        help="Index of the reference FISH channel for registration"
    )

    parser.add_argument(
        "--second-spot-channel",
        type=int,
        # nargs="*",
        default=2,
        help="Indices of second spot channels comma separated if more than 1 (default 2)"
    )

    # -------------------------
    # Configuration
    # -------------------------
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file with all settings"
    )

    # -------------------------
    # Analysis parameters
    # -------------------------
    parser.add_argument(
        "--pairing-threshold",
        type=float,
        default=2.5, # in µm
        help="Threshold (µm) for pairing spots between channels" # TODO --> FIX THAT --> it is not in nm, it is in micrometer and this nb is scaled later by the pixel size which makes no sense
    )

    parser.add_argument(
        "--area-threshold",
        type=float,
        default=None,
        help="Area threshold for segmentation"
    )

    # -------------------------
    # Pipeline steps (ALL ENABLED BY DEFAULT)
    # -------------------------
    parser.add_argument(
        "--no-seg",
        dest="run_seg",
        action="store_false",
        help="Disable segmentation"
    )
    parser.set_defaults(run_seg=True)

    parser.add_argument(
        "--no-distance-measurements",
        dest="run_distance_measurements",
        action="store_false",
        help="Disable distance measurements"
    )
    parser.set_defaults(run_distance_measurements=True)

    parser.add_argument(
        "--no-ctrls",
        dest="run_ctrls",
        action="store_true",
        help="Disable controls"
    )
    parser.set_defaults(run_ctrls=False)

    parser.add_argument(
        "--no-gaussian-fit",
        dest="run_gaussian_fit",
        action="store_false",
        help="Disable Gaussian fitting"
    )
    parser.set_defaults(run_gaussian_fit=True)

    # -------------------------
    # Models
    # -------------------------
    parser.add_argument(
        "--nuclear-model-to-use",
        type=str,
        default="nuclear_model_0",
        help="Nuclear model to use"
    )

    parser.add_argument(
        "--spot-model-to-use",
        type=str,
        default="spot_model_0",
        help="Spot detection model to use"
    )

    args = parser.parse_args()

    # -------------------------
    # Conditional requirement: if no config, require a path
    # -------------------------
    if not args.config and not (args.root_path or args.colocs_path):
        parser.error("one of the arguments --root-path or --colocs-path is required if --config is not provided")

    # If --colocs-path is used, require controls and distances paths
    if args.colocs_path and (not args.controls_path or not args.distances_path):
        parser.error("--controls-path and --distances-path are required when using --colocs-path")

    # Auto-detect config.json in root-path if --config not provided
    if not args.config and args.root_path:
        auto_config = find_config_in_root(args.root_path)
        if auto_config:
            args.config = auto_config

    return args

def load_config(config_path):
    """Load configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config

def validate_paths(args):
    """Validate and set up the paths based on user input."""
    if args.config:
        # If using config file, paths should be set there
        return

    if args.root_path:
        # If root path is provided, construct subfolder paths
        args.colocs_path = os.path.join(args.root_path, 'colocs')
        args.controls_path = os.path.join(args.root_path, 'controls')
        args.distances_path = os.path.join(args.root_path, 'distances')
    else:
        # If individual paths are provided, verify they exist
        if not os.path.exists(args.colocs_path):
            raise ValueError(f"Colocs path does not exist: {args.colocs_path}")
        if not os.path.exists(args.controls_path):
            raise ValueError(f"Controls path does not exist: {args.controls_path}")
        if not os.path.exists(args.distances_path):
            raise ValueError(f"Distances path does not exist: {args.distances_path}")

def merge_args_with_config(args):
    """Merge command line arguments with configuration file settings."""
    if args.config:
        try:
            config = load_config(args.config)

            # Override arguments with config file values if they exist
            if 'nucleus_channel' in config:
                args.nucleus_channel = config['nucleus_channel']
            if 'reference_channel' in config:
                args.reference_channel = config['reference_channel']
            if 'second_spot_channel' in config:
                args.second_spot_channel = config['second_spot_channel']
            if 'pairing_threshold' in config:
                args.pairing_threshold = config['pairing_threshold']
            if 'area_threshold' in config:
                args.area_threshold = config['area_threshold']
            if 'run_seg' in config:
                args.run_seg = config['run_seg']
            if 'run_distance_measurements' in config:
                args.run_distance_measurements = config['run_distance_measurements']
            if 'run_ctrls' in config:
                args.run_ctrls = config['run_ctrls']
            if 'run_gaussian_fit' in config:
                args.run_gaussian_fit = config['run_gaussian_fit']
            if 'nuclear_model_to_use' in config:
                args.nuclear_model_to_use = config['nuclear_model_to_use']
            if 'spot_model_to_use' in config:
                args.spot_model_to_use = config['spot_model_to_use']

            # Handle paths from config
            if 'root_path' in config:
                args.root_path = config['root_path']
                validate_paths(args)
            else:
                if 'colocs_path' in config:
                    args.colocs_path = config['colocs_path']
                if 'controls_path' in config:
                    args.controls_path = config['controls_path']
                if 'distances_path' in config:
                    args.distances_path = config['distances_path']
                validate_paths(args)

        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
            sys.exit(1)

    return args

if __name__ == '__main__':
    pass