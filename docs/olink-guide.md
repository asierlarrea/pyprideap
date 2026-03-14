# Olink Analysis Guide

This guide covers Olink-specific functionality in pyprideap, including data reading, LOD computation, QC, normalization, and statistical analysis for Olink Explore, Explore HT, Target, and Reveal platforms.

## Reading Olink Data

pyprideap supports three Olink file formats. All produce the same `AffinityDataset` structure.

```python
import pyprideap as pp

# NPX CSV (Explore, Target)
dataset = pp.read("olink_npx.csv")

# Parquet (Explore HT)
dataset = pp.read("explore_ht.parquet")

# Excel
dataset = pp.read("olink_data.xlsx")
```

### What the reader does

1. Detects the platform from OlinkID prefixes (OID0–OID5 → Explore/HT/Target/Reveal)
2. Extracts sample metadata: `SampleID`, `SampleName`, `PlateID`, `WellID`, `SampleType`, `SampleQC`, `PlateQC`
3. Extracts feature metadata: `OlinkID`, `UniProt`, `Assay`, `Panel`, `LOD`, `MissingFreq`
4. Pivots long-format NPX data into a wide expression matrix (samples × OlinkID)
5. Builds a per-sample LOD matrix in `metadata["lod_matrix"]` when a LOD column is present

### Expression values

Olink reports **NPX** (Normalized Protein eXpression) values on a **log2 scale**. Higher NPX means higher protein concentration:

```
NPX = log2(protein concentration in arbitrary units)
```

Typical NPX values range from approximately -10 to 40.

## Filtering

### Remove control samples

Olink data files include negative controls (`NEGATIVE_CONTROL`), plate controls (`PLATE_CONTROL`), and other non-biological samples. Remove them before analysis:

```python
# Remove all control sample types
ds = pp.filter_controls(dataset)
print(f"Kept {len(ds.samples)} of {len(dataset.samples)} samples")
```

### Filter by QC status

Each sample has a `SampleQC` flag: `PASS`, `WARN`, or `FAIL`.

```python
# Keep PASS and WARN (default)
ds = pp.filter_qc(dataset)

# Keep only PASS
ds = pp.filter_qc(dataset, keep=("PASS",))
```

## LOD (Limit of Detection)

### Resolution priority

pyprideap resolves LOD automatically in this order:

1. **Reported LOD** — from the NPX file's LOD column (per-sample, per-assay)
2. **NCLOD** — computed from negative control samples
3. **FixedLOD** — pre-computed reference values bundled with the library

### Reported LOD

If the data file contains a LOD column, it is stored as a matrix (samples × assays):

```python
lod = pp.get_reported_lod(dataset)
# Returns DataFrame (per-sample LOD) or Series (single LOD per assay)
```

### NCLOD (Negative Control LOD)

Computed from negative control samples using the OlinkAnalyze formula:

```
LOD = median(NC_NPX) + max(0.2, 3 × SD(NC_NPX))
```

- Requires **≥10 negative control samples** for reliable estimation
- The `max(0.2, ...)` floor prevents unrealistically low LOD when controls have very low variance

With plate adjustment enabled (default), a plate-specific offset is applied:

```
plate_LOD = base_LOD + (plate_median_NPX − global_median_NPX)
```

This accounts for plate-to-plate intensity differences.

```python
# Simple NCLOD (one value per assay)
lod = pp.compute_nclod(dataset, plate_adjusted=False)

# Plate-adjusted NCLOD (matrix: samples × assays)
lod = pp.compute_nclod(dataset, plate_adjusted=True)
```

### NCLOD with Count Data

For datasets that include extension count data, a more detailed LOD computation is available:

```python
from pyprideap.processing.lod import compute_nc_lod_detailed, compute_pc_normalized_lod

# Detailed per-assay LOD with method classification
detail = compute_nc_lod_detailed(dataset)
# detail.lod_method: "lod_npx" or "lod_count" per assay
# detail.lod_count: max(150, 2 × max(Count))

# PC-normalized LOD matrix (plate control adjusted)
lod_matrix = compute_pc_normalized_lod(dataset)
```

The LOD method for each assay is determined by extension count:

```
If MaxCount > 150 → lod_npx method (standard NPX-based)
If MaxCount ≤ 150 → lod_count method: LOD = log2(LODCount / ExtCount) − PCMedian
```

### FixedLOD

Pre-computed LOD values from Olink, specific to each reagent lot. pyprideap bundles FixedLOD files for Explore 3072, Explore HT, and Reveal:

```python
# Load bundled FixedLOD matched to dataset platform
lod = pp.load_fixed_lod(dataset)

# Load a custom FixedLOD file
lod = pp.load_fixed_lod(dataset, lod_file_path="custom_lod.csv")

# Get the path to the bundled file
path = pp.get_bundled_fixed_lod_path(dataset.platform)
```

### LOD Statistics

```python
stats = pp.compute_lod_stats(dataset)
print(f"LOD source: {stats.lod_source}")
print(f"Overall above-LOD rate: {stats.above_lod_rate:.1%}")
print(f"Assays with LOD: {stats.n_assays_with_lod}/{stats.n_assays_total}")

# Per-panel breakdown (if panels are available)
for panel, rate in stats.above_lod_per_panel.items():
    print(f"  {panel}: {rate:.1%}")
```

### Proteins above LOD

Get UniProt accessions for proteins detected in a sufficient fraction of samples:

```python
# Default: ≥50% of samples above LOD
proteins = pp.get_proteins_above_lod(dataset)

# Stricter: ≥80%
proteins = pp.get_proteins_above_lod(dataset, threshold=80.0)

# With a specific LOD source
proteins = pp.get_proteins_above_lod(dataset, lod=my_lod)
```

## QC Outlier Detection

### IQR vs Median outliers

This method detects samples with anomalous within-panel expression patterns, computed per panel:

```
For each panel:
  sample_IQR = Q75(NPX) − Q25(NPX)
  sample_Median = median(NPX)

  panel_mean_IQR, panel_SD_IQR = mean/SD across all samples
  panel_mean_Median, panel_SD_Median = mean/SD across all samples

  Outlier if:
    |sample_IQR − panel_mean_IQR| > n × panel_SD_IQR
    OR
    |sample_Median − panel_mean_Median| > n × panel_SD_Median
```

Default multiplier `n = 3.0` (3 standard deviations).

```python
from pyprideap.processing.olink.outliers import compute_iqr_median_outliers

result = compute_iqr_median_outliers(dataset, iqr_outlier_def=3.0, median_outlier_def=3.0)
print(f"Outliers: {result.n_outliers}/{result.n_samples}")
print(f"Outlier sample IDs: {result.outlier_sample_ids}")
```

## Preprocessing Pipeline

The Olink pipeline chains common preprocessing steps:

```python
from pyprideap.processing.olink.pipeline import preprocess_olink

processed, report = preprocess_olink(
    dataset,
    filter_controls=True,           # Remove NEGATIVE_CONTROL, PLATE_CONTROL, etc.
    filter_qc_outliers=True,        # Remove IQR/Median outliers
    filter_qc_warning=False,        # Also remove WARN samples (default: keep them)
    filter_lod=False,               # Remove assays below LOD detection rate
    lod_detection_rate=0.5,         # Min fraction of samples above LOD
    remove_uniprot_duplicates=False, # Remove duplicate UniProt mappings
    prep_for_dimred=False,          # Impute NaN, remove zero-variance for PCA/t-SNE
    iqr_outlier_def=3.0,
    median_outlier_def=3.0,
)

# Preprocessing report
print(f"Samples: {report.n_samples_before} → {report.n_samples_after}")
print(f"Features: {report.n_features_before} → {report.n_features_after}")
```

## Normalization

### Bridge normalization

Adjusts a target dataset to match a reference dataset using shared bridge samples. The adjustment is additive on the NPX (log2) scale, equivalent to multiplicative scaling in linear space:

```
adjustment[protein] = median(bridge in reference) − median(bridge in target)
adjusted_NPX = original_NPX + adjustment
```

```python
# Step 1: Select bridge samples automatically
bridges = pp.select_bridge_samples(dataset, n=8)

# Step 2: Normalize
normalized = pp.bridge_normalize(reference_ds, target_ds, bridge_samples=bridges)
```

### Bridge sample selection

Selects optimal bridge samples by:
1. Excluding control samples
2. Excluding IQR/Median QC outliers
3. Keeping only `SampleQC == "PASS"` samples
4. Filtering by below-LOD rate (< 50% missing by default)
5. Selecting `n` evenly-spaced samples across the MeanNPX range

```python
bridges = pp.select_bridge_samples(
    dataset,
    n=8,                       # Number of bridge samples
    sample_missing_freq=0.5,   # Max below-LOD rate
    exclude_qc_outliers=True,  # Apply IQR/Median outlier filter
    iqr_outlier_def=3.0,
    median_outlier_def=3.0,
)
```

### Subset normalization

Adjusts using a reference set of stable housekeeping proteins:

```
adjustment = median(ref_proteins in reference) − median(ref_proteins in target)
```

Applied uniformly to all proteins.

```python
normalized = pp.subset_normalize(reference_ds, target_ds, reference_proteins=["OID0001", "OID0002"])
```

### Reference median normalization

Shifts each protein to match pre-recorded reference medians:

```python
# reference_medians: dict or Series mapping OlinkID → target median NPX
normalized = pp.reference_median_normalize(dataset, reference_medians=ref_medians)
```

### Cross-product bridgeability assessment

Evaluate whether assays can be reliably bridged between two Olink products (e.g., Explore 3072 vs Explore HT):

```python
report = pp.assess_bridgeability(dataset1, dataset2)

# Key columns: protein_id, correlation, median_diff, detection rates, bridgeable flag
bridgeable = report[report["bridgeable"]]
print(f"{len(bridgeable)} of {len(report)} assays are bridgeable")
```

Criteria: correlation > 0.7 AND detection rate > 0.5 in both datasets.

### Quantile smoothing normalization

For cross-product normalization (Explore 3072 ↔ HT ↔ Reveal), a quantile smoothing approach maps the target's NPX distribution to the reference using ECDF and spline regression:

```python
from pyprideap.processing.normalization import quantile_smooth_normalize

normalized = quantile_smooth_normalize(
    reference=ref_ds,
    target=target_ds,
    bridge_samples=bridges,
)
```

Minimum bridge sample requirements by product pair:
- Explore 3072 ↔ HT: 40 samples
- Explore 3072 ↔ Reveal: 32 samples
- HT ↔ Reveal: 24 samples

### Multi-project normalization

Chain normalization across N datasets sequentially:

```python
from pyprideap.processing.normalization import normalize_n, NormalizationStep

steps = [
    NormalizationStep(order=1, name="Batch1", dataset=ds1, bridge_samples=b1,
                      normalization_type="bridge", normalize_to="Batch2"),
    NormalizationStep(order=2, name="Batch2", dataset=ds2, bridge_samples=b2,
                      normalization_type="bridge", normalize_to="Batch3"),
    NormalizationStep(order=3, name="Batch3", dataset=ds3, bridge_samples=None,
                      normalization_type=None, normalize_to=None),  # reference
]
normalized_datasets = normalize_n(steps)
```

## Statistical Testing

All tests run per-protein with Benjamini-Hochberg FDR correction. Requires `pip install "pyprideap[stats]"`.

### t-test

Two-sample Welch t-test (or paired t-test with `pair_id`). Requires exactly 2 groups.

```python
results = pp.ttest(dataset, group_var="Treatment")
results = pp.ttest(dataset, group_var="Treatment", pair_id="SubjectID")  # paired

# Result columns: protein_id, assay, estimate, statistic, p_value, adj_p_value, significant
significant = results[results["significant"]]
```

### Wilcoxon / Mann-Whitney U

Non-parametric alternative to the t-test:

```python
results = pp.wilcoxon(dataset, group_var="Treatment")
results = pp.wilcoxon(dataset, group_var="Treatment", pair_id="SubjectID")  # paired
```

### ANOVA

One-way ANOVA or ANCOVA (with covariates). Supports ≥2 groups.

```python
results = pp.anova(dataset, variables=["Treatment"])
results = pp.anova(dataset, variables=["Treatment"], covariates=["Age", "Sex"])

# Result columns: protein_id, assay, statistic, df_between, df_within, p_value, adj_p_value
```

### Post-hoc comparisons (Tukey HSD)

Pairwise comparisons after ANOVA:

```python
posthoc = pp.anova_posthoc(dataset, variable="Treatment")

# Result columns: protein_id, assay, contrast (g1-g2), estimate, p_value, adj_p_value, ci_lower, ci_upper
```

### Volcano plot data

```python
volcano = pp.compute_volcano(dataset, group_var="Treatment")
# Returns VolcanoData with log2FC and -log10(p) for plotting
```

## QC Report

Generate a comprehensive interactive HTML report:

```python
pp.qc_report(dataset, "olink_qc_report.html")
```

Olink-specific sections in the report:
- **Expression distributions** — per-sample NPX histogram overlays
- **QC × LOD summary** — stacked bar showing PASS/WARN/FAIL vs above/below LOD
- **LOD analysis** — % samples above LOD per assay
- **LOD comparison** — FixedLOD vs NCLOD vs Reported LOD
- **PCA / t-SNE** — dimensionality reduction colored by SampleType or Panel
- **Correlation heatmap** — sample-to-sample Pearson correlation
- **Expression heatmap** — clustered heatmap across all proteins
- **Data completeness** — above/below LOD per sample and per protein
- **IQR vs Median QC** — outlier detection scatter plot
- **UniProt duplicates** — assays mapping to the same UniProt accession

## Complete Workflow Example

```python
import pyprideap as pp

# 1. Read data
dataset = pp.read("olink_npx.csv")
print(f"{len(dataset.samples)} samples, {len(dataset.features)} features")

# 2. Generate QC report (before filtering)
pp.qc_report(dataset, "raw_qc_report.html")

# 3. Filter
ds = pp.filter_controls(dataset)
ds = pp.filter_qc(ds)

# 4. Check LOD
stats = pp.compute_lod_stats(ds)
print(f"Above-LOD rate: {stats.above_lod_rate:.1%}")

# 5. Get analyzable proteins
proteins = pp.get_proteins_above_lod(ds, threshold=50.0)
print(f"{len(proteins)} proteins above LOD")

# 6. Statistical testing (if group variable is available)
if "Treatment" in ds.samples.columns:
    results = pp.ttest(ds, group_var="Treatment")
    sig = results[results["significant"]]
    print(f"{len(sig)} significant proteins (FDR < 0.05)")

# 7. Generate post-filtering report
pp.qc_report(ds, "filtered_qc_report.html")
```
