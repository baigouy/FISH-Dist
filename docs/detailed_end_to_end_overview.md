## Detailed End-to-End Workflow of FISH-Dist with Purpose and Rationale for Each Module

**FISH-Dist** is designed as a sequential pipeline to accurately detect and quantify 3D distances between FISH probes. Below is an end-to-end description of each key module, including its purpose.

### 1. 3D Nuclei Segmentation (Deep Learning)
- **Purpose & Rationale:** Segments nuclei in 3D confocal images using a deep learning model, achieving higher accuracy and robustness than traditional thresholding methods, particularly in noisy or densely packed images. For **DNA FISH**, which labels genomic loci, **signals outside the nucleus (false positive) represent irrelevant background**, so precise nuclear segmentation ensures that only biologically meaningful nuclear signals are analyzed.  
  *(Note: If applying FISH-Dist to RNA FISH or other cytoplasmic targets, the nuclear mask is not recommended; however, it is essential for DNA FISH analysis.)*

### 2. FISH Spot Segmentation (Deep Learning)
- **Purpose & Rationale:** Segments individual FISH spots in 3D images using a deep learning model, allowing precise detection even for weak signals. Accurate spot segmentation is essential for downstream distance measurements, as errors here may affect the calculated inter-spot distances and their biological interpretation.

### 3. Sub-Pixel Spot Localization (3D Gaussian Fitting)
- **Purpose & Rationale:** Refines FISH spot coordinates beyond pixel resolution using 3D Gaussian fitting, providing sub-pixel accuracy. This precision is critical because even small localization errors can substantially affect calculated inter-spot distances, which directly impacts quantitative analysis of genomic loci positioning.

### 4. Spot Pairing Across Channels
- **Purpose & Rationale:** Matches corresponding FISH spots between different fluorescence channels based on spatial proximity. Accurate pairing is essential not only for calculating inter-spot distances but also for deriving chromatic aberration corrections: when most pairs are correctly matched, the resulting median distances and corrections are meaningful; if pairing is inaccurate, both the distance measurements and corrections become unreliable.

### 5. Chromatic Aberration Correction (Linear/Affine)
- **Purpose & Rationale:** Corrects spatial distortions between fluorescence channels caused by optical system imperfections using linear or affine models. Accurate correction is especially critical for genomic loci in close proximity, where misalignments can dominate measurement errors; for loci that are farther apart, the relative impact of chromatic aberrations is smaller. Without proper correction, inter-spot distances can be biased, potentially compromising biological interpretation.

### 6. Pairwise Inter-Spot Distance Measurements
- **Purpose & Rationale:** Computes distances between paired FISH spots across channels using the corrected, sub-pixel accurate positions from previous modules. Accurate distance measurements are essential for quantitative analysis of genomic loci positioning, as even small errors in localization or pairing can significantly affect downstream statistical interpretations.  

  *(Note: Discrepancies between distances calculated using linear versus affine chromatic corrections may indicate changes in the microscope's chromatic aberrations, signaling the need to acquire a new colocalization dataset for accurate correction.)*

### 7. Result Visualization and Reporting
- **Purpose & Rationale:** Generates images, plots, and tables that allow users to interpret biological results. For a detailed description of all input formats, generated tables, plots, and files, see the **[FISH-Dist Inputs and Outputs](inputs_outputs.md)** guide.