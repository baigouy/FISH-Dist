# FISH-Dist inputs and outputs

## Input Formats

The software is designed to support all major scientific image formats, but currently implements support for:

* **`.tif`** (Tagged Image File Format)
* **`.czi`** (Zeiss CZI format)

[//]: # (Future versions will support:)

[//]: # ()
[//]: # (* `.lif` &#40;Leica LIF format&#41;)

[//]: # (* `.lsm` &#40;Zeiss LSM format&#41;)

---

# Output Files

## 1. Raw Deep Learning Outputs (`chX.tif`)

For each channel, the software generates a `.tif` file containing the raw output of the deep learning model:

* `chX.tif` (where `X` is the channel index)

These files can represent:

* Nuclear segmentation masks
* Detected FISH / oligo paint spots

---

## 2. SQLite Database (`FISH.db`)

The software outputs a comprehensive SQLite database (`FISH.db`) containing all processed data.

The database can be explored using **[DB Browser for SQLite](https://sqlitebrowser.org/dl/)** (download available on the official website).  
⚠️ **Warning:** Be sure to close the database file before running or re-running the analysis to avoid SQLite file lock issues.

---

# Database Structure & Contents

---

## Image Metadata

### Table: `nb_channels`

| Column        | Type    | Description                     |
| ------------- | ------- | ------------------------------- |
| `nb_channels` | INTEGER | Number of channels in the image |

---

### Table: `voxel_size`

| Column | Type  | Description                    |
| ------ | ----- | ------------------------------ |
| `vz`   | FLOAT | Voxel size in µm (Z dimension) |
| `vy`   | FLOAT | Voxel size in µm (Y dimension) |
| `vx`   | FLOAT | Voxel size in µm (X dimension) |

---

## Nuclear Information

Nuclear centroids are stored in the table `nuclei`

### Table: `nuclei`

| Column | Type  | Description                          |
| ------ | ----- | ------------------------------------ |
| `z`    | FLOAT | Z-coordinate of the nucleus centroid |
| `y`    | FLOAT | Y-coordinate of the nucleus centroid |
| `x`    | FLOAT | X-coordinate of the nucleus centroid |

---

## Spot Detection

Detected FISH / oligo-paint spot coordinates are stored in the table `spots_chX` (where X is the channel index).

### Table: `spots_chX`

| Column                    | Type  | Description                                                                                       |
| ------------------------- | ----- |---------------------------------------------------------------------------------------------------|
| `z`                       | FLOAT | Z-coordinate of the FISH / oligo paint spot                                                       |
| `y`                       | FLOAT | Y-coordinate of the FISH / oligo paint spot                                                       |
| `x`                       | FLOAT | X-coordinate of the FISH / oligo paint spot                                                       |
| `score`                   | FLOAT | Fraction of FISH spot pixels overlapping with the nuclear mask (1 = nuclear, 0 = outside nucleus) |
| `successful_gaussian_fit` | FLOAT | Gaussian fit success (1 = yes, 0 = no)                                                            |

---

## Penetrance Estimation

### Table: `penetrance_estimate`

| Column         | Type    | Description                             |
| -------------- | ------- | --------------------------------------- |
| `nb_nuclei`    | INTEGER | Number of analyzed nuclei               |
| `nb_spots_chX` | INTEGER | Number of detected spots in channel `X` |

---

# Distance Analysis

Distance analysis is performed between spots detected in channel `X` and channel `Y`.
All distances are computed in 3D using voxel-size corrected coordinates.

For each analysis level, both a **database table** and a corresponding **histogram plot** are generated.

Each histogram:

* Displays the distribution of 3D distances
* Includes a vertical line indicating the **median distance**

---

## 1. All Spot Pairs (Raw Coordinates)

### Table: `points_n_distances3D_chX_chY`

| Column     | Type  | Description                                                |
| ---------- | ----- |------------------------------------------------------------|
| `z1`       | FLOAT | Z-coordinate of the FISH / oligo paint spot in channel `X` |
| `y1`       | FLOAT | Y-coordinate of the FISH / oligo paint spot in channel `X` |
| `x1`       | FLOAT | X-coordinate of the FISH / oligo paint spot in channel `X` |
| `z2`       | FLOAT | Z-coordinate of the FISH / oligo paint spot in channel `Y` |
| `y2`       | FLOAT | Y-coordinate of the FISH / oligo paint spot in channel `Y` |
| `x2`       | FLOAT | X-coordinate of the FISH / oligo paint spot in channel `Y` |
| `distance` | FLOAT | 3D Euclidean distance between paired spots                 |

**Corresponding histogram file:**

* `pairs_distances_chX_chY.png`

This plot shows the full distribution of all pairwise 3D distances (no nuclear filtering, no chromatic aberration correction).

---

## 2. Spot Pairs Restricted to Nuclei

### Table: `points_n_distances3D_only_in_nuclei_chX_chY`

Same structure as above, but only for spot pairs assigned to nuclei.

**Corresponding histogram file:**

* `pairs_distances_only_in_nuclei_chX_chY.png`

This plot shows the distance distribution restricted to intra-nuclear spot pairs (without chromatic aberration correction).

---

## 3. Affine Chromatic Aberration Correction

### Table: `points_n_distances3D_chX_chY_acc`

Same structure as `points_n_distances3D_chX_chY`, but channel `Y` coordinates are corrected using an affine chromatic aberration (acc) transformation.

**Corresponding histogram file:**

* `points_n_distances3D_chX_chY_acc.png`

This plot allows direct comparison between raw and affine-corrected distance distributions.

---

## 4. Affine Chromatic Aberration Correction (Nuclei Only)

### Table: `points_n_distances3D_only_in_nuclei_chX_chY_acc`

Same structure as above, restricted to nuclear spot pairs.

**Corresponding histogram file:**

* `points_n_distances3D_only_in_nuclei_chX_chY_acc.png`

This plot provides a view of affine chromatic aberration correction (acc) within nuclei and can be compared to the raw distances to assess the effect of the correction.

---

## 5. Linear Chromatic Aberration Correction (Nuclei Only)

### Table: `points_n_distances3D_only_in_nuclei_chX_chY_lcc`

Same structure as the nuclei-only table, but using linear chromatic aberration correction (lcc) of channel `Y`.

**Corresponding histogram file:**

* `points_n_distances3D_only_in_nuclei_chX_chY_lcc.png`

* This plot provides a view of linear chromatic aberration correction (lcc) within nuclei and can be compared to the raw distances to assess the effect of the correction.

---

## Master Violin Plot: `YYYY-MM-DD_HH-MM-SS_violin_plot_chX_chY.pdf`

* Combines all relevant distance distributions from input files with similar names into a single violin plot
* Shows the **median distance** as a dashed line
* Displays **quantiles** as a box plot inside the violin
* Replace `X` and `Y` with the corresponding channel indices to match the dataset
* `YYYY-MM-DD_HH-MM-SS` is the **date and time when the plot was generated**:

  * `YYYY` → year (4 digits)
  * `MM` → month
  * `DD` → day
  * `HH` → hour (24h format)
  * `MM` → minute
  * `SS` → second

---

## Additional Files from Colocalization Analysis

In addition to the main outputs, the colocalization analysis generates several supplementary files:

### 1. `chX_chY_acc.npy`

* Contains the **affine chromatic aberration (ACC) transformation** between the two sets of points (`chX` ↔ `chY`)
* Used to correct channel `Y` coordinates relative to channel `X`
* Replace `X` and `Y` with the corresponding channel indices

### 2. `voxel_size.npy`

* Stores the **voxel dimensions** of the image (Z, Y, X)
* Acts as a consistency check to ensure that distance measurements are computed using the same acquisition settings; performing registration or distance measurements on datasets with different acquisition conditions would be meaningless
