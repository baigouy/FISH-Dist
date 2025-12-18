# FISH-Dist: FISH Spot Detection and Analysis Pipeline

A comprehensive Python pipeline for detecting, quantifying, and analyzing FISH (Fluorescence In Situ Hybridization) spots in 3D microscopy images using deep learning and chromatic aberration correction.

## Overview

FISH-Dist performs:
- 3D nuclei and spot segmentation using deep learning

- Sub-pixel spot localization with Gaussian fitting

- Chromatic aberration correction via affine transformation

- Spot pairing and distance quantification

- Visualization

## Installation

```bash
pip install numpy scipy scikit-image pandas seaborn matplotlib
pip install bigfish  # for Gaussian sub-pixel fitting
```

## Quick Start

### Basic Pipeline Execution

```python
from fishdist.fish_analysis_pipeline import run_analysis

# Define your image paths
paths = [
    '/path/to/image1.tif',
    '/path/to/image2.tif',
]

# Path to save correction matrices
correction_matrix_path = '/path/to/output/corrections/'

# Run full analysis
run_analysis(
    paths=paths,
    correction_matrix_save_path=correction_matrix_path,
    PAIRING_THRESHOLD=60,  # nm
    ch_nuclei=0,  # Nuclear channel index
    first_spot_channel=1,  # First FISH channel
    second_spot_channel=2,  # Second FISH channel
    RUN_SEG=True,
    RUN_REG=True,
    RUN_DISTANCE_MEASUREMENTS=True,
    RUN_CTRLS=True
)
```

## Pipeline Stages

### 1. Segmentation

Segments nuclei and spots using pre-trained deep learning models:

```python
segment_spots_and_nuclei(
    paths=paths,
    ch_nuclei=0,
    first_spot_channel=1,
    second_spot_channel=2,
    channels_to_blur=[1, 2],  # Optional blur before segmentation
    blur_mode='recursive2D',
    deep_learning='rapid'
)
```

**Output**: Binary masks saved as `ch0.tif`, `ch1.tif`, `ch2.tif`

### 2. Spot Detection

Detects spot coordinates with optional Gaussian sub-pixel refinement:

```python
detect_spots_and_nuclei(
    paths=paths,
    ch_nuclei=0,
    first_spot_channel=1,
    second_spot_channel=2,
    area_threshold=5,           # Minimum spot area
    threshold_spot_ch1=0.5,     # Binarization threshold
    threshold_spot_ch2=0.5,
    threshold_nuclei=0.5
)
```

**Output**: Spot coordinates saved to `FISH.db` tables:
- `spots_ch1`: First channel spots
- `spots_ch2`: Second channel spots
- `nuclei`: Nuclear centroids

### 3. Spot Pairing

Pairs spots between channels based on proximity:

```python
pair_spots(
    paths=paths,
    PAIRING_THRESHOLD=250  # nm
)
```

**Output**: Tables in `FISH.db`:
- `points_n_distances3D`: All spot pairs
- `points_n_distances3D_only_in_nuclei`: Nuclear-restricted pairs

### 4. Chromatic Aberration Correction

Computes global affine transformation from paired spots:

```python
compute_affine_transform_for_images(
    paths=paths,
    correction_matrix_save_path='/path/to/output/',
    db_to_read='points_n_distances3D_only_in_nuclei',
    USE_AFFINE_TRAFO_WITH_SHEAR=True
)
```

**Output**:
- `affine_chromatic_aberration_correction.npy`: Transformation matrix
- `voxel_size.npy`: Voxel scaling factors
- Histogram plots before/after correction

Apply correction to all datasets:

```python
apply_affine_transform_to_all_images(
    paths=paths,
    root_folder_path_to_affine_trafo_matrix='/path/to/corrections/',
    table_name_to_apply_correction_to='points_n_distances3D_only_in_nuclei'
)
```

### 5. Visualization

Generate violin plots with statistical annotations:

```python
plot_analysis(
    paths=paths,
    distance_cut_off=2.5,  # µm
    group_files_by_name_similarity=True,
    output_file_name='violin_plot.pdf',
    table_names=[
        'points_n_distances3D_only_in_nuclei_chromatic_aberrations_corrected'
    ]
)
```

## Citation

If you use this FISH-Dist, please cite the associated manuscript. TODO.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 3rd Party Licenses

<font color='red'>IMPORTANT: If you disagree with any of the licenses below, <u>please uninstall FISH-Dist</u>. Additionally, ensure you review the licenses of all 3<sup>rd</sup> party dependencies used by the project, as they may have their own terms and conditions.</font>

| Library name            | Use                                                                         | Link                                          | License            |
|-------------------------|-----------------------------------------------------------------------------|-----------------------------------------------|--------------------|
| **tensorflow**          | Deep learning library                                                       | https://pypi.org/project/tensorflow/          | Apache 2.0         |
| **segmentation-models** | Models                                                                      | https://pypi.org/project/segmentation-models/ | MIT                |
| **matplotlib**          | Plots images and graphs                                                     | https://pypi.org/project/matplotlib/          | PSF                |
| **numpy**               | Array/Image computing                                                       | https://pypi.org/project/numpy/               | BSD                |
| **scikit-image**        | Image processing                                                            | https://pypi.org/project/scikit-image/        | BSD (Modified BSD) |
| **scipy**               | Great library to work with numpy arrays                                     | https://pypi.org/project/scipy/               | BSD                | 
| **pandas**              | Data analysis toolkit                                                       | https://pypi.org/project/pandas/              | BSD (BSD-3-Clause) |
| **BAtoolset**           | Compilation of functions I find useful                                      | https://github.com/baigouy/BAtoolset              | BSD-2-Clause license |
| **EPySeg**           | A deep learning and image processing package for segmentation and analysis                                 | https://github.com/baigouy/EPySeg              | BSD |
| **affine_matrix_from_points**           | C. Gohlke code to calculate the affine transform between two sets of points | https://github.com/cgohlke/transformations/blob/deb1a195dab70f0f36365a104f9b70505e37b473/transformations/transformations.py#L920              | BSD 3-Clause |
