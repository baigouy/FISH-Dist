python your_script.py


python your_script.py --first-channel -2 --second-channel -1


python your_script.py --paths "/path/to/folder1" "/path/to/folder2"

python your_script.py --first-channel 3 --second-channel 4 --paths "/path1" "/path2" --pairing-threshold 40


Basic Command (Using Root Path)

python your_script.py \
    --nucleus-channel 0 \
    --reference-channel 1 \
    --second-spot-channel 2 3 4 \
    --root-path "/media/teamPrudhomme/EqpPrudhomme2/FISH_Dist_analysis" \
    --pairing-threshold 30 \
    --area-threshold 35 \
    --run-seg \
    --run-distance-measurements \
    --run-ctrls \
    --run-gaussian-fit \
    --nuclear-model-to-use "nuclear_model_0" \
    --spot-model-to-use "spot_model_0"

Using Individual Paths

python your_script.py \
    --nucleus-channel 0 \
    --reference-channel 1 \
    --second-spot-channel 2 \
    --colocs-path "/path/to/colocs" \
    --controls-path "/path/to/controls" \
    --distances-path "/path/to/distances" \
    --pairing-threshold 30 \
    --area-threshold 35 \
    --run-seg \
    --run-distance-measurements \
    --run-ctrls \
    --run-gaussian-fit \
    --nuclear-model-to-use "nuclear_model_0" \
    --spot-model-to-use "spot_model_0"


```
FISH_Dist_analysis/
    ├── colocs/
    ├── controls/
    └── distances/
```

Example **config.json**: (fill with your own values and folders)
```
{
    "nucleus_channel": 0,
    "reference_channel": 1,
    "second_spot_channel": [2, 3, 4],
    "pairing_threshold": 30,
    "area_threshold": null,
    "run_seg": true,
    "run_distance_measurements": true,
    "run_ctrls": true,
    "run_gaussian_fit": true,
    "nuclear_model_to_use": "/path/to/nuclear_model",
    "spot_model_to_use": "/path/to/spot_model",
    "root_path": "/path/to/root_folder_containg_the_3_folder_distances_controls_and_colocs"
}


```

"pairing_threshold": 30, --> shall I offer that ??? --> MAYBE or not in the path --> I need to think


Using root path:
```
python your_script.py --nucleus-channel 0 --reference-channel 1 --root-path "/media/teamPrudhomme/EqpPrudhomme2/To be analyzed"
```
Using individual paths:
```
python your_script.py --nucleus-channel 0 --reference-channel 1 \
    --colocs-path "/path/to/colocs" \
    --controls-path "/path/to/controls" \
    --distances-path "/path/to/distances"
```

Now, you can run your script with the new arguments:
python your_script.py --nucleus-channel 0 --reference-channel 1 \
    --root-path "/media/teamPrudhomme/EqpPrudhomme2/To be analyzed" \
    --pairing-threshold 30 --area-threshold 35 --run-seg --run-distance-measurements --run-ctrls \
    --models-path "/path/to/models"



Using a configuration file:
```
python your_script.py --config path/to/config.json
```

TODO can I read the root path from a file or ask the user for it if not found -−> samrt idea


---

### **Folder Logic**

#### **1. `colocs/`**
- **Purpose**: Contains files used to measure colocalization between FISH spots.
- **Usage**: Place your `.czi` images here to generate a **registration matrix** for chromatic aberration correction.
- **Output**: The registration matrix is generated and should be moved to the `controls/` folder.

---

#### **2. `controls/`**
- **Purpose**: Stores the registration matrix files generated from the `colocs/` folder.
- **Usage**: After generating the registration matrix in `colocs/`, move it to this folder.
- **Note**: The script automatically checks this folder for registration matrices when processing images in `distances/`.

---

#### **3. `distances/`**
- **Purpose**: Contains files used to compute distances between pairs of FISH spots.
- **Usage**: Place your `.czi` images here to measure distances between spots.
- **Note**: If a registration matrix is available in the `controls/` folder, the script will apply chromatic aberration correction to the images in this folder.

---

### **Workflow Summary**
1. **Generate Registration Matrix**: Place images in `colocs/` to generate the registration matrix.
2. **Move Registration Matrix**: Copy the generated registration matrix to the `controls/` folder.
3. **Measure Distances**: Place images in `distances/` to measure distances between FISH spots, with optional chromatic aberration correction applied if a registration matrix is available in `controls/`.



# Quick Start Guide

This script automatically analyzes FISH images in specific folders.
It detects spots, measures distances, and moves finished files to a **"DONE"** folder.

---

## Before You Start
- Make sure you have **Python 3.10+** and the `FISH_Dist` Conda environment set up.
- Your images should be in `.czi` format and placed in the folders listed in the script.

---

## How to Run the Script

1. **Open a terminal** and activate the `FISH_Dist` environment:
   ```bash
   conda activate FISH_Dist
   ```

2. **Run the script**:
   ```bash
   python /path/to/your_script.py
   ```

---

## What You Can Change

You can edit the script to customize the analysis:

| Option                     | Description                                                                                     | Default Value                     |
|----------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------|
| `folders_to_scan`          | List of folders containing `.czi` images to analyze.                                           | `['/media/teamPrudhomme/EqpPrudhomme2/To be analyzed/2048', ...]` |
| `PAIRING_THRESHOLD`        | Threshold (in nanometers) for pairing spots between channels.                                   | `30`                              |
| `first_spot_channel`       | Index of the first FISH channel (adjust if channel order changes).                              | `-2` (for 2048px) / `1` (for 3072px) |
| `second_spot_channel`      | Index of the second FISH channel.                                                              | `-1`                              |
| `RUN_SEG`                  | Run segmentation.                                                                              | `True`                            |
| `RUN_REG`                  | Run registration (only for coloc folders).                                                     | `False`                           |
| `RUN_DISTANCE_MEASUREMENTS`| Run distance measurements.                                                                    | `True`                            |
| `RUN_CTRLS`                | Run controls.                                                                                  | `True`                            |

---

## What to Expect

- The script will scan the folders you specified for `.czi` images.
- It will process each image, detect spots, and measure distances.
- Once done, it will move the processed files to a **"DONE"** folder.
- If the script can't connect to the fileserver, it will try to remount it automatically.

---

## Tips

- If you have control images, place them in:
  ```
  /media/teamPrudhomme/EqpPrudhomme2/To be analyzed/controls/
  ```
- If no control images are found, the script will use default paths for registration.
```

---