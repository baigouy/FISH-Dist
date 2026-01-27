# FISH-Dist Tutorial (Demo Dataset)

This tutorial walks through a complete FISH-Dist analysis using provided demo data.
You will learn how to:

1. Perform chromatic aberration correction  
2. Measure distances between genomic loci  

This demo focuses on running the pipeline on example datasets; installation, setup, and general workflow details are described in the main [README.md](README.md).

---

## 1. Setup

### Requirements

* Conda installed see [Install Conda](https://fish-dist.readthedocs.io/en/latest/#install-conda)
* FISH-Dist environment available created see [Installation / Setup (one-time)](https://fish-dist.readthedocs.io/en/latest/#installation-setup-one-time)

---

## 2. Download and prepare demo data

### Step 2.1 — Folder structure

Download the template folder:

```
FISH_Dist_analysis.zip
```

From:

[https://gitlab.com/baigouy/models/-/raw/master/FISH_Dist_analysis.zip](https://gitlab.com/baigouy/models/-/raw/master/FISH_Dist_analysis.zip)

Unzip the archive.

After unzipping, you should have **exactly one** parent folder named:

```text
FISH_Dist_analysis/
```

containing:

```text
FISH_Dist_analysis/
├── colocs/
├── controls/
└── distances/
```

> ⚠️ **Do not rename these folders.**
> FISH-Dist relies on this exact directory structure.

---

## 3. Step 1 — Chromatic aberration correction

This step computes correction matrices using colocalization images and provides an estimate of the spatial resolution achievable with the imaging setup.

### 3.1 Add demo colocalization images

Download:

```
colocs.zip
```

From:

[https://gitlab.com/baigouy/models/-/raw/master/coloc.zip](https://gitlab.com/baigouy/models/-/raw/master/coloc.zip)

[//]: # ([https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/colocs.zip]&#40;https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/colocs.zip&#41;)

Unzip the archive.

Copy the **contents** into:

```text
FISH_Dist_analysis/colocs/
```

After copying, the directory should look like:

```text
FISH_Dist_analysis/
└── colocs/
    ├── coloc_1.tif
    └── coloc_2.tif
```

These images contain:

* 1 nuclear channel
* 2 Oligopaint/FISH channels targeting the **same genomic locus**

---

### 3.2 Run chromatic aberration correction

From anywhere on your system, run:

```bash
conda activate FISH_Dist
python -m fishdist --root-path /path/to/FISH_Dist_analysis
```

Replace `/path/to/FISH_Dist_analysis` with the actual path to your `FISH_Dist_analysis` folder.

---

### 3.3 Outputs

After completion:

* Correction matrices (`.npy`) are generated in:

```text
colocs/DONE/
```

These files should then be moved to:

```text
controls/
```

The matrices encode affine transformations correcting chromatic aberration.

> ℹ️ **Important:**
> This step only needs to be performed once per microscope setup and imaging configuration, as long as chromatic aberrations remain stable.

---

## 4. Step 2 — Distance measurement

This step measures distances between **distinct genomic loci**.

---

### Option A — Run with provided correction matrices (recommended for demo)

To skip Step 1, download:

```
controls.zip
distances.zip
```

From:

[https://gitlab.com/baigouy/models/-/raw/master/controls.zip](https://gitlab.com/baigouy/models/-/raw/master/controls.zip)
[https://gitlab.com/baigouy/models/-/raw/master/distances.zip](https://gitlab.com/baigouy/models/-/raw/master/distances.zip)

[//]: # ([https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/distances.zip]&#40;https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/distances.zip&#41;)



Unzip **each archive separately**.

Copy or merge their contents so that your folder structure is:

```text
FISH_Dist_analysis/
├── controls/     ← contains .npy files
└── distances/    ← contains demo distance images
```

---

### Option B — Use your own correction matrices

If you completed Step 1 yourself, ensure that:

* `.npy` files are present in `controls/`

You still need to download the distance images from:

[https://gitlab.com/baigouy/models/-/raw/master/distances.zip](https://gitlab.com/baigouy/models/-/raw/master/distances.zip)

[//]: # ([https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/distances.zip]&#40;https://filedn.eu/lP1mui2ztv7VO2qrPEt28Vf/distances.zip&#41;)

* Distance images are placed in `distances/`

---

### 4.1 Run distance measurement

```bash
conda activate FISH_Dist
python -m fishdist --root-path /path/to/FISH_Dist_analysis
```

As above, replace `/path/to/FISH_Dist_analysis` with the full path to your analysis folder.

---

### 4.2 Outputs

After processing:

* Processed images are copied to:

```text
distances/DONE/
```

* Distance tables, plots, and summary reports are generated automatically.