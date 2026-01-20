# FISH-Dist Configuration Guide

FISH-Dist can be configured via a **JSON configuration file** instead of providing all command-line arguments. This is particularly useful if you run multiple experiments with the same setup, or want to save channel assignments, thresholds, and model paths.

### JSON Structure

| Key/Parameters              | Type | Description                                                                                    | Default / Required |
|-----------------------------|------|------------------------------------------------------------------------------------------------|------------------|
| `root_path`                 | string | Root folder containing `colocs/`, `controls/`, and `distances/`                                | Required if not specifying paths individually |
| `nucleus_channel`           | int | Index of the nuclear channel in your images                                                    | 0 |
| `reference_channel`         | int | Index of the FISH channel used as reference for chromatic aberration correction                | 1 |
| `second_spot_channel`       | int or list of int | Index/indices of the other FISH channels to pair with the reference                            | 2 |
| `pairing_threshold`         | int | Maximum distance (nm) to pair spots across channels                                            | 3000 |
| `area_threshold`            | float | Minimum area in px for detected objects in segmentation (optional)                             | null |
| `run_seg`                   | bool | Run segmentation step                                                                          | true |
| `run_gaussian_fit`          | bool | Run 3D Gaussian fitting for sub-pixel localization                                             | true |
| `run_distance_measurements` | bool | Run distance measurements                                                                      | true |
| `run_ctrls`                 | bool | Creates a set of images that allows for visual inspection of segmentation, pairs and distances | true |
| `nuclear_model_to_use`      | string | Path to a custom nuclear segmentation model (tensorflow)                                       | `"nuclear_model_0"` |
| `spot_model_to_use`         | string | Path to a custom spot detection model (tensorflow)                                             | `"spot_model_0"` |

### Example `config.json`  Files

Below are several example `config.json` files illustrating common usage scenarios.  
You only need to include the parameters relevant to your experiment—all others will fall back to sensible defaults.

### Example 1 — Minimal Configuration (Default Channel Convention)

Use this configuration if your images follow the default channel order:

- Channel 0 → Nucleus  
- Channel 1 → Reference FISH  
- Channel 2 → Second FISH  

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis"
}
```

**Note:** the absence of a comma at the end of the last parameter is mandatory

This is equivalent to running FISH-Dist without any configuration file, as long as your data follow the default convention.

### Example 2 — Custom Channel Assignment

Use this configuration if your channels are ordered differently in the image file.        

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "nucleus_channel": 1,
    "reference_channel": 2,
    "second_spot_channel": 3
}
```

### Example 3 — Multiple FISH Channels (One-to-Many Pairing)

Use this configuration when measuring distances between one reference FISH channel and multiple other FISH channels.

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "reference_channel": 1,
    "second_spot_channel": [2, 3, 4]
}
```

Distances will be computed between the reference channel and each listed channel independently.

### Example 4 — Distance Measurement Only (Reuse Existing Controls)

Use this configuration when chromatic aberration correction has already been computed and you want to skip control generation.

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "run_ctrls": false,
    "run_distance_measurements": true
}
```

### Example 5 — Disable Gaussian Fitting (Faster Runtime)

Use this configuration to speed up processing by skipping sub-pixel Gaussian fitting.

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "run_gaussian_fit": false
}
```        

### Example 6 — Custom Pairing and Segmentation Parameters

Use this configuration when adjusting pairing distance or segmentation thresholds.

```json        
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "pairing_threshold": 2000,
    "area_threshold": 50
}
```

### Example 7 — Full Explicit Configuration

This example explicitly defines most parameters and is useful for reproducibility.

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "nucleus_channel": 0,
    "reference_channel": 1,
    "second_spot_channel": [2, 3],
    "pairing_threshold": 3000,
    "area_threshold": null,
    "run_seg": true,
    "run_gaussian_fit": true,
    "run_distance_measurements": true,
    "run_ctrls": false
}
```

### Example — Using Custom Segmentation Models

Use this configuration when you want to override the default deep learning models for nuclear segmentation while keeping the default FISH segmentation model.

```json
{
    "root_path": "/media/teamXXX/FISH_Dist_analysis",
    "nuclear_model_to_use": "/path/to/my_great_nuclear_model"
}
```

## Using the Configuration File

To run FISH-Dist with a configuration file, simply use:
        
```bash
# Run FISH-Dist using an explicit configuration file
conda activate FISH_Dist
python fish_dist.py --config /path/to/config.json
```

Alternatively, if `config.json` is located inside the analysis root folder:

```bash
# Run FISH-Dist as ususal and it will read the config file automatically
conda activate FISH_Dist
python fish_dist.py --root-path /path/to/FISH_Dist_analysis 
```

The script will automatically override any command-line arguments with the values in the JSON file.