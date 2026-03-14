# SomaScan Analysis Guide

This guide covers SomaScan-specific functionality in pyprideap, including data reading, QC flag handling, control analyte management, outlier detection, normalization (including cross-version lifting), and the preprocessing pipeline.

## Reading SomaScan Data

pyprideap supports two SomaScan file formats.

```python
import pyprideap as pp

# ADAT format (standard SomaLogic output)
dataset = pp.read("raw_data.adat")

# CSV with SeqId.* columns
dataset = pp.read("somascan_data.csv")
```

### ADAT format

The ADAT reader handles both legacy and modern formats:

- **Legacy format**: sections delimited by `COL_DATA`, `ROW_DATA`, `TABLE_BEGIN`
- **Modern format**: starts with `TABLE_BEGIN`

Extracted metadata:
- **Samples**: `SampleId`, `SampleType`, `PlateId`, `ScannerID`, plus normalization scale columns (`HybControlNormScale`, `Med.Scale.*`, etc.)
- **Features**: `SeqId`, `UniProt`, `Target` (protein name), `Dilution`, `SomaId`
- **Expression**: RFU (Relative Fluorescence Units) matrix, samples × SeqId

### Expression values

SomaScan reports **RFU** (Relative Fluorescence Units) on a **linear scale**. Higher RFU means higher protein concentration. Unlike Olink's log2 NPX, RFU values are always positive:

```
RFU ∝ protein concentration (linear relationship)
```

Typical RFU values range from hundreds to tens of thousands.

## QC Flags

SomaScan uses two QC flags adapted from the SomaDataIO R package.

### RowCheck (Sample-level QC)

Evaluates normalization scale factors for each sample. A sample passes only when **all** normalization scales fall within the acceptance range.

```
RowCheck = PASS  if all(0.4 ≤ scale ≤ 2.5) for each normalization scale column
RowCheck = FLAG  otherwise
```

Normalization scale columns detected: `*NormScale*`, `Med.Scale.*`

```python
from pyprideap.processing.somascan.qc_flags import (
    add_row_check,
    filter_by_row_check,
    get_row_check_summary,
)

# Add RowCheck column to sample metadata
ds = add_row_check(dataset)

# Check summary
summary = get_row_check_summary(ds)
print(f"PASS: {summary['PASS']}, FLAG: {summary['FLAG']}")

# Remove flagged samples
ds_filtered = filter_by_row_check(dataset)

# Custom thresholds
ds_strict = filter_by_row_check(dataset, low=0.5, high=2.0)
```

### ColCheck (Feature-level QC)

Evaluates calibrator QC ratios for each analyte. Analytes with ratios outside the acceptance range are flagged:

```
ColCheck = PASS  if 0.8 ≤ QC_ratio ≤ 1.2
ColCheck = FLAG  otherwise
```

ColCheck values come from the ADAT file's feature metadata.

```python
from pyprideap.processing.somascan.qc_flags import (
    filter_by_col_check,
    get_col_check_summary,
)

# Check summary
summary = get_col_check_summary(dataset)
print(f"PASS: {summary['PASS']}, FLAG: {summary['FLAG']}")

# Remove flagged analytes
ds_filtered = filter_by_col_check(dataset)
```

## Control Analytes

SomaScan includes several categories of control analytes that are not human protein targets. These should be removed before biological analysis.

### Control categories

| Category | Count | Purpose |
|----------|-------|---------|
| `HybControlElution` | 12 | Hybridization efficiency controls |
| `Spuriomer` | 20 | Non-specific binding controls |
| `NonBiotin` | 10 | Biotinylation controls |
| `NonHuman` | 22 | Non-human protein controls |
| `NonCleavable` | 4 | Cleavage controls |

```python
from pyprideap.processing.somascan.controls import (
    classify_control_analytes,
    remove_control_analytes,
    get_control_seqids,
    is_control_analyte,
)

# See which analytes are controls
classification = classify_control_analytes(dataset)
# Returns dict: {SeqId: category_name} for control analytes

# Remove all control analytes
ds_clean = remove_control_analytes(dataset)

# Remove only specific categories
ds_clean = remove_control_analytes(dataset, categories=["Spuriomer", "NonBiotin"])

# Check a specific SeqId
is_control_analyte("2171-12")  # True (HybControlElution)

# Get SeqIds for specific categories
hyb_seqids = get_control_seqids("HybControlElution")
```

## Outlier Detection

### MAD-based outlier detection

SomaScan uses a Median Absolute Deviation (MAD) approach combined with fold-change criteria to detect analyte-level outliers per sample:

```
For each sample × analyte value:
  stat_outlier = |value − median(analyte)| > 6 × MAD(analyte)
  fold_outlier = (value / median > fc_crit) OR (median / value > fc_crit)
  outlier = stat_outlier AND fold_outlier
```

The dual criterion requires both statistical extremity (6 × MAD) and biological significance (fold-change threshold).

```python
from pyprideap.processing.somascan.outliers import calc_outlier_map, get_outlier_ids

# Compute outlier map (boolean matrix: samples × analytes)
outlier_map = calc_outlier_map(dataset, fc_crit=5.0)

# Inspect per-sample outlier counts
print(outlier_map.n_outliers_per_sample)
print(outlier_map.outlier_fraction_per_sample)

# Get sample indices where ≥5% of analytes are outliers
flagged = get_outlier_ids(outlier_map, flags=0.05)
print(f"Flagged samples: {flagged}")
```

### OutlierMap properties

The `OutlierMap` object provides convenient access to outlier statistics:

```python
outlier_map.matrix                     # Boolean DataFrame (samples × analytes)
outlier_map.n_outliers_per_sample      # Series: count per sample
outlier_map.n_outliers_per_analyte     # Series: count per analyte
outlier_map.outlier_fraction_per_sample  # Series: fraction per sample
```

## eLOD (Estimated Limit of Detection)

SomaScan's eLOD is computed from buffer samples using a MAD-based formula robust to outliers:

```
eLOD = median(buffer_RFU) + 3.3 × 1.4826 × MAD(buffer_RFU)
```

- **1.4826** converts MAD to the equivalent of standard deviation for normal distributions
- **3.3** multiplier targets approximately 95% detection probability
- Best suited for non-core matrices (cell lysate, CSF); use carefully for plasma/serum

```python
from pyprideap.processing.lod import compute_soma_elod

# Compute eLOD from buffer samples
elod = compute_soma_elod(dataset)  # Returns Series: SeqId → eLOD value

# Use in LOD statistics
stats = pp.compute_lod_stats(dataset, lod=elod)
print(f"Above-eLOD rate: {stats.above_lod_rate:.1%}")
```

## Normalization

### Scale analytes (multiplicative)

SomaScan normalization is **multiplicative** on the RFU (linear) scale, unlike Olink's additive NPX normalization:

```
RFU_scaled = RFU × scalar
```

```python
from pyprideap.processing.normalization import scale_analytes

# scalars: dict or Series mapping expression column → scalar
scalars = {"2171-12": 1.05, "2178-55": 0.98, ...}
ds_scaled = scale_analytes(dataset, scalars)
```

### Cross-version lifting (lift_somascan)

Calibrate data between SomaScan assay versions (5k ↔ 7k ↔ 11k). Scalars are derived from matched reference populations:

```python
from pyprideap.processing.normalization import lift_somascan, validate_lift_requirements

# Validate that the dataset meets lifting requirements
validation = validate_lift_requirements(dataset, bridge="v4.0_to_v4.1")
# Checks: ANML normalized, correct signal space, plasma/serum matrix

# Apply lift scalars
lifted = lift_somascan(
    dataset,
    scalars=lift_scalars,           # dict: SeqId → scalar
    target_version="7k",           # Target assay version
    bridge="v4.0_to_v4.1",        # Bridge direction
    validate=True,                  # Run requirement checks first
)
```

#### Version mapping

| Version String | Menu Size |
|---------------|-----------|
| V3, v3.0 | 1.1k |
| V3.2, v3.2 | 1.3k |
| V4, v4.0 | 5k |
| V4.1, v4.1 | 7k |
| V5, v5.0 | 11k |

#### Lift requirements

`validate_lift_requirements` checks:
1. Bridge direction is valid (e.g., `v4.0_to_v4.1`)
2. Data is ANML normalized (`ProcessSteps` contains "ANML")
3. Current signal space matches the "from" side of the bridge
4. Sample matrix is plasma or serum

### Lift quality assessment

After lifting, assess the quality of calibration using Lin's Concordance Correlation Coefficient (CCC):

```
CCC = (2 × ρ × σx × σy) / [(μx − μy)² + σx² + σy²]
```

Where ρ is Pearson correlation, σ is standard deviation, and μ is mean.

```python
from pyprideap.processing.normalization import assess_lift_quality

quality = assess_lift_quality(original=dataset, lifted=lifted_dataset)

# Result columns: analyte, ccc, pearson_r, median_original, median_lifted, scalar
good_lift = quality[quality["ccc"] > 0.9]
print(f"{len(good_lift)} of {len(quality)} analytes have CCC > 0.9")
```

## Preprocessing Pipeline

The SomaScan pipeline chains common preprocessing steps in the recommended order:

```python
from pyprideap.processing.somascan.pipeline import preprocess_somascan

processed, report = preprocess_somascan(
    dataset,
    filter_features=True,     # Remove control analytes (Spuriomer, NonBiotin, etc.)
    filter_controls=True,     # Keep only SampleType == "Sample"
    filter_rowcheck=True,     # Remove samples failing RowCheck
    filter_outliers=False,    # Remove MAD-based outlier samples
    log10=False,              # Apply log10 transformation
    center_scale=False,       # Z-score standardization
    fc_crit=5.0,              # Fold-change criterion for outlier detection
    outlier_flags=0.05,       # Fraction threshold for flagging samples
)
```

### Pipeline steps in order

1. **Filter features** — Remove control analytes (HybControlElution, Spuriomer, NonBiotin, NonHuman, NonCleavable)
2. **Filter control samples** — Keep only `SampleType == "Sample"` (remove Buffer, Calibrator, QC)
3. **Filter by RowCheck** — Remove samples where normalization scales are outside [0.4, 2.5]
4. **Filter outliers** — Compute MAD-based outlier map, remove samples with ≥5% analyte outliers
5. **Log10 transform** — `RFU_log10 = log10(RFU)` (optional, for downstream analysis)
6. **Center and scale** — Z-score: `(x − mean) / SD` per analyte (optional, for clustering/PCA)

### PreprocessingReport

The report object tracks what was removed at each step:

```python
print(f"Samples: {report.n_samples_before} → {report.n_samples_after}")
print(f"Features: {report.n_features_before} → {report.n_features_after}")
print(f"Controls removed: {report.n_controls_removed}")
print(f"RowCheck failures: {report.n_rowcheck_removed}")
print(f"Outliers removed: {report.n_outliers_removed}")
```

## QC Report

Generate an interactive HTML report with SomaScan-specific sections:

```python
pp.qc_report(dataset, "somascan_qc_report.html")
```

SomaScan-specific sections:
- **Expression distributions** — per-sample RFU histogram overlays
- **Normalization scale factors** — box/strip plots of HybControlNormScale, Med.Scale.* columns
- **RowCheck / ColCheck summary** — bar charts of PASS vs FLAG counts
- **Outlier map** — heatmap of MAD-based outlier detection results
- **Control analyte analysis** — RFU distribution of control vs biological analytes
- **PCA / t-SNE** — dimensionality reduction colored by SampleType
- **Correlation heatmap** — sample-to-sample correlation
- **CV distribution** — coefficient of variation per analyte
- **Data completeness** — above/below eLOD per sample

## Complete Workflow Example

```python
import pyprideap as pp
from pyprideap.processing.somascan.pipeline import preprocess_somascan
from pyprideap.processing.somascan.qc_flags import get_row_check_summary, get_col_check_summary

# 1. Read ADAT file
dataset = pp.read("experiment.adat")
print(f"{len(dataset.samples)} samples, {len(dataset.features)} features")

# 2. Generate raw QC report
pp.qc_report(dataset, "raw_qc_report.html")

# 3. Check QC flags
row_summary = get_row_check_summary(dataset)
col_summary = get_col_check_summary(dataset)
print(f"RowCheck: {row_summary['PASS']} PASS, {row_summary['FLAG']} FLAG")
print(f"ColCheck: {col_summary['PASS']} PASS, {col_summary['FLAG']} FLAG")

# 4. Run preprocessing pipeline
processed, report = preprocess_somascan(
    dataset,
    filter_features=True,
    filter_controls=True,
    filter_rowcheck=True,
    filter_outliers=True,
    log10=True,              # Log-transform for downstream analysis
)
print(f"After preprocessing: {len(processed.samples)} samples, {len(processed.features)} features")

# 5. Compute eLOD and check detection
from pyprideap.processing.lod import compute_soma_elod
elod = compute_soma_elod(dataset)  # Use original (unfiltered) dataset for buffer samples
stats = pp.compute_lod_stats(processed, lod=elod)
print(f"Above-eLOD rate: {stats.above_lod_rate:.1%}")

# 6. Get detectable proteins
proteins = pp.get_proteins_above_lod(processed, lod=elod, threshold=50.0)
print(f"{len(proteins)} proteins above eLOD")

# 7. Generate post-processing report
pp.qc_report(processed, "processed_qc_report.html")
```

## Key Differences from Olink

| Aspect | Olink | SomaScan |
|--------|-------|----------|
| **Scale** | NPX (log2) | RFU (linear) |
| **Normalization** | Additive (on log2 scale) | Multiplicative (on linear scale) |
| **LOD method** | NCLOD / FixedLOD / Reported | eLOD from buffer samples |
| **QC flags** | SampleQC (PASS/WARN/FAIL) | RowCheck + ColCheck (PASS/FLAG) |
| **Control types** | NEGATIVE_CONTROL, PLATE_CONTROL | HybControlElution, Spuriomer, NonBiotin, NonHuman, NonCleavable |
| **Outlier detection** | IQR vs Median (per panel) | MAD-based (per analyte, fold-change criterion) |
| **Feature ID** | OlinkID | SeqId |
| **Panels** | Cardiometabolic, Inflammation, etc. | Dilution groups (20%, 0.5%, 0.005%) |
