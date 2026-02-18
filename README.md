[![Documentation Status](https://readthedocs.org/projects/fish-dist/badge/?version=latest)](https://fish-dist.readthedocs.io/en/latest/)

# FISH-Dist: Automated 3D Distance Quantification for Confocal FISH Images

**FISH-Dist** is a Python pipeline for detecting and **quantifying FISH** (Fluorescence In Situ Hybridization) probes in **3D confocal microscopy images**.
It is designed for distance measurements between pairs of fluorescent signals across imaging channels.

## Overview

FISH-Dist provides:

- 3D nuclei segmentation (deep learning)
- FISH spot segmentation (deep learning)
- Sub-pixel spot localization (3D Gaussian fitting)
- Spot pairing across channels
- Chromatic aberration correction (linear/affine)
- Pairwise inter-spot distance measurements
- Result visualization and reporting

## Install Conda

FISH-Dist requires Conda. If you don’t have it installed:

Download and **install Miniconda** (lightweight) or Anaconda (full package) from:  

  - [Miniconda (recommended)](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions:~:text=Miniconda-,Installing%20Miniconda,-Copy%20page)  
  - [Anaconda](https://www.anaconda.com/products/distribution)

💡 Tip: Miniconda is usually faster and smaller; you can always install extra packages later.

## Opening the Command Line

FISH-Dist runs from a **command-line interface (CLI)**. The steps to open the terminal depend on your operating system:

| Operating System | How to open the command line |
|------------------|------------------------------|
| **Windows**      | Press `Win + R`, type `cmd`, and hit Enter. <br> Or search for **Command Prompt** in the Start menu. <br> If you installed Anaconda/Miniconda, you can also use **Anaconda Prompt**. |
| **MacOS**        | Press `Cmd + Space`, type `Terminal`, and hit Enter. <br> Or navigate to **Applications → Utilities → Terminal**. |
| **Linux**        | Press `Ctrl + Alt + T` (works in most distributions). <br> Or search for **Terminal** in your applications menu. |

## Verify Conda Installation

After installing Conda, open a terminal (see [Opening the Command Line](#opening-the-command-line) instructions above) and type:

```bash
conda --version
```

You should see something like:

```text
conda 24.10.0
```

If the command returns `command not found`, you need to [install Conda](#install-conda) first.

## Installation / Setup (one-time)

In your [command line](#opening-the-command-line), type the following commands:

```bash
# Create and activate the FISH-Dist environment
conda create -y -n FISH_Dist python==3.10.12
conda activate FISH_Dist

# Upgrade pip and install FISH-Dist
pip install --upgrade pip
pip install fishdist
```

## Required Folder Structure

```text
    FISH_Dist_analysis/
    ├── colocs/        
    ├── controls/      
    └── distances/     
```

| Folder       | Description                                                           |
| ------------ |-----------------------------------------------------------------------|
| `colocs/`    | Colocalization images for computing chromatic aberration.             |
| `controls/`  | Stores registration matrices (`.npy`) from `colocs/`.                 |
| `distances/` | Images for measuring distances between genomic loci.                  |

**Note:** Correction matrices can be reused across multiple datasets acquired under identical microscope settings, provided they are collected within a short time frame to ensure consistent chromatic aberrations.

**Note2:** For a complete description of input formats, database tables, distance tables, and generated plots, see the **[FISH-Dist Inputs and Outputs](inputs_outputs.md)** guide.

## Example workflow

FISH-Dist operates in **two sequential phases**. Understanding this order is essential for correct results.

### Step 1 — Chromatic aberration correction

- Acquire colocalization images:
    - 1 nuclear channel
    - 2 (or more) FISH channels targeting the same genomic locus
    
- Place images in `colocs/`.

- Run the analysis:

```bash
conda activate FISH_Dist
python -m fishdist --root-path /path/to/FISH_Dist_analysis
```

- Store generated `.npy` matrices in `controls/`.

**Note:** Step 1 only needs to be performed once per microscope setup and imaging configuration, as long as chromatic aberrations remain stable.

### Step 2 — Distance Measurement

- Acquire distance-measurement images:

    - Use the same microscope and imaging settings as for colocalization

    - Acquire images shortly after the colocalization experiment to ensure stable chromatic aberrations

    - Images should contain FISH probes targeting distinct genomic loci

- Ensure appropriate `.npy` matrices are in `controls/`.

- Place distance images in `distances/`.

- Run FISH-Dist:

```bash
conda activate FISH_Dist
python -m fishdist --root-path FISH_Dist_analysis
```

- Outputs:

    - Processed images → `distances/DONE/`
    - Distance tables, plots, and reports

    - See the **[FISH-Dist Inputs and Outputs](inputs_outputs.md)** for a full list and description of all generated tables, plots, and additional files.

- Quality Control:
   - Verify consistency between affine and linear corrections
   - If **inconsistent, repeat Step 1** with newly acquired colocalization images.

## Summary

```text
Colocalization images → colocs/ → Correction (.npy)
                                     ↓
                                 controls/
                                     ↓
Distance images → distances/ → Distance measurements → Check for consistency
```

For detailed information about the input formats and description of all outputs, see **[FISH-Dist Inputs and Outputs](inputs_outputs.md)**.

### Default Channel Convention

By default, FISH-Dist assumes the following channel layout in **3D confocal microscopy images**:

| Channel | Purpose                                                                                               |
|--------:|-------------------------------------------------------------------------------------------------------|
| **0** | Nucleus (used for nuclear segmentation)                                                               |
| **1** | Reference FISH channel (channel used for aligning all other channels to)                              |
| **2** | Second FISH channel (used for colocalization with the reference channel or for distance measurements) |

**Note:**  
If your datasets follow this channel order **and** you use the same channel configuration for both colocalization and distance measurements, **no configuration file is required**. You can run the pipeline using only the command-line interface.

## Configuration File (Optional)

By default, FISH-Dist follows the [Default Channel Convention](#default-channel-convention); if your data follow this channel order and you are not modifying any parameters, no configuration file is required.

### When is a configuration file required?

A JSON configuration file becomes necessary if you want to:

- Use a different channel order
- Analyze more than two FISH channels
- Change thresholds or analysis parameters
- Use custom segmentation models
- Enable or disable specific pipeline steps
- Re-run only part of the pipeline (e.g. skip registration or Gaussian fitting)

For a full description of all available parameters and advanced examples, see
👉 [Advanced settings (configuration file)](CONFIGURATION.md)

## Uninstall

To **completely remove** the `FISH_Dist` environment and all its dependencies, run:

```bash
conda remove --name FISH_Dist --all
```

## Citation

If you use this FISH-Dist, please cite the associated manuscript. **TODO**.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 3rd Party Licenses

⚠️ IMPORTANT: If you disagree with any of the licenses below, uninstall FISH-Dist. Review all licenses of third-party dependencies.

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

