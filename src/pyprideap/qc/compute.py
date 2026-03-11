from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform


@dataclass
class DistributionData:
    values: list[float]
    xlabel: str
    ylabel: str = "Count"
    title: str = "Expression Distribution"
    platform: str = ""


@dataclass
class QcSummaryData:
    categories: list[str]
    counts: list[int]
    title: str = "QC Summary"


@dataclass
class LodAnalysisData:
    assay_ids: list[str]
    above_lod_pct: list[float]
    panel: list[str]
    title: str = "LOD Analysis: % Samples Above LOD"


@dataclass
class PcaData:
    pc1: list[float]
    pc2: list[float]
    variance_explained: list[float]
    labels: list[str]
    groups: list[str]
    title: str = "PCA"


@dataclass
class CorrelationData:
    matrix: list[list[float]]
    labels: list[str]
    title: str = "Sample Correlation"


@dataclass
class MissingValuesData:
    missing_rate_per_sample: list[float]
    missing_rate_per_feature: list[float]
    sample_ids: list[str]
    feature_ids: list[str]
    title: str = "Missing Values"


@dataclass
class CvDistributionData:
    feature_ids: list[str]
    cv_values: list[float]
    dilution: list[str] = field(default_factory=list)
    title: str = "CV Distribution"


@dataclass
class DetectionRateData:
    sample_ids: list[str]
    rates: list[float]
    title: str = "Detection Rate per Sample"


def compute_distribution(dataset: AffinityDataset) -> DistributionData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    flat = numeric.values.flatten()
    flat = flat[~np.isnan(flat)]

    is_somascan = dataset.platform == Platform.SOMASCAN
    if is_somascan:
        flat = np.log10(flat[flat > 0])
        xlabel = "log10(RFU)"
        title = "RFU Distribution (log10)"
    else:
        xlabel = "NPX (log2)"
        title = "NPX Distribution"

    return DistributionData(
        values=flat.tolist(),
        xlabel=xlabel,
        title=title,
        platform=dataset.platform.value,
    )


def compute_qc_summary(dataset: AffinityDataset) -> QcSummaryData | None:
    if "SampleQC" not in dataset.samples.columns:
        return None
    counts = dataset.samples["SampleQC"].value_counts()
    return QcSummaryData(
        categories=counts.index.tolist(),
        counts=counts.values.tolist(),
    )


def compute_lod_analysis(dataset: AffinityDataset) -> LodAnalysisData | None:
    from pyprideap.lod import compute_lod_from_controls, get_lod_values

    lod = get_lod_values(dataset)
    if lod is None:
        try:
            lod = compute_lod_from_controls(dataset)
        except (ValueError, KeyError):
            return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    assay_ids = []
    above_lod_pct = []
    panels = []

    panel_col = dataset.features.get("Panel")
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]

    for i, col in enumerate(numeric.columns):
        if col not in lod.index:
            continue
        vals = numeric[col].dropna()
        if len(vals) == 0:
            pct = 0.0
        else:
            pct = float((vals > lod[col]).sum() / len(vals) * 100)

        assay_ids.append(str(dataset.features[id_col].iloc[i]) if i < len(dataset.features) else col)
        above_lod_pct.append(pct)
        panels.append(str(panel_col.iloc[i]) if panel_col is not None and i < len(panel_col) else "")

    if not assay_ids:
        return None

    return LodAnalysisData(assay_ids=assay_ids, above_lod_pct=above_lod_pct, panel=panels)


def compute_pca(dataset: AffinityDataset, n_components: int = 2) -> PcaData | None:
    try:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
    except ImportError:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return None

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)

    n_comp = min(n_components, *imputed.shape)
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(imputed)

    id_col = "SampleID" if "SampleID" in dataset.samples.columns else "SampleId"
    labels = (
        dataset.samples[id_col].astype(str).tolist()
        if id_col in dataset.samples.columns
        else [f"S{i}" for i in range(len(dataset.samples))]
    )
    groups = (
        dataset.samples["SampleType"].astype(str).tolist()
        if "SampleType" in dataset.samples.columns
        else [""] * len(labels)
    )

    return PcaData(
        pc1=transformed[:, 0].tolist(),
        pc2=transformed[:, 1].tolist() if n_comp >= 2 else [0.0] * len(labels),
        variance_explained=[float(v) for v in pca.explained_variance_ratio_],
        labels=labels,
        groups=groups,
    )


def compute_correlation(dataset: AffinityDataset, max_samples: int = 50) -> CorrelationData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if numeric.shape[0] > max_samples:
        numeric = numeric.sample(n=max_samples, random_state=42)

    corr = numeric.T.corr().fillna(0)

    id_col = "SampleID" if "SampleID" in dataset.samples.columns else "SampleId"
    if id_col in dataset.samples.columns:
        labels = dataset.samples.loc[numeric.index, id_col].astype(str).tolist()
    else:
        labels = [f"S{i}" for i in range(len(numeric))]

    return CorrelationData(
        matrix=[row.tolist() for row in corr.values],
        labels=labels,
    )


def compute_missing_values(dataset: AffinityDataset) -> MissingValuesData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    is_missing = numeric.isna()

    per_sample = is_missing.mean(axis=1).tolist()
    per_feature = is_missing.mean(axis=0).tolist()

    id_col = "SampleID" if "SampleID" in dataset.samples.columns else "SampleId"
    sample_ids = (
        dataset.samples[id_col].astype(str).tolist()
        if id_col in dataset.samples.columns
        else [f"S{i}" for i in range(len(dataset.samples))]
    )
    feature_ids = numeric.columns.tolist()

    return MissingValuesData(
        missing_rate_per_sample=per_sample,
        missing_rate_per_feature=per_feature,
        sample_ids=sample_ids,
        feature_ids=feature_ids,
    )


def compute_cv_distribution(dataset: AffinityDataset) -> CvDistributionData | None:
    if dataset.platform != Platform.SOMASCAN:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    stds = numeric.std()
    cv = (stds / means).fillna(0)

    feature_ids = numeric.columns.tolist()
    dilution = (
        dataset.features["Dilution"].astype(str).tolist()
        if "Dilution" in dataset.features.columns and len(dataset.features) == len(feature_ids)
        else []
    )

    return CvDistributionData(
        feature_ids=feature_ids,
        cv_values=cv.tolist(),
        dilution=dilution,
    )


def compute_detection_rate(dataset: AffinityDataset) -> DetectionRateData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    rates = numeric.notna().mean(axis=1).tolist()

    id_col = "SampleID" if "SampleID" in dataset.samples.columns else "SampleId"
    sample_ids = (
        dataset.samples[id_col].astype(str).tolist()
        if id_col in dataset.samples.columns
        else [f"S{i}" for i in range(len(dataset.samples))]
    )

    return DetectionRateData(sample_ids=sample_ids, rates=rates)


def compute_all(dataset: AffinityDataset) -> dict[str, object]:
    """Compute all applicable QC plot data for the dataset."""
    results: dict[str, object] = {}
    results["distribution"] = compute_distribution(dataset)
    results["qc_summary"] = compute_qc_summary(dataset)
    results["lod_analysis"] = compute_lod_analysis(dataset)
    results["pca"] = compute_pca(dataset)
    results["correlation"] = compute_correlation(dataset)
    results["missing_values"] = compute_missing_values(dataset)
    results["cv_distribution"] = compute_cv_distribution(dataset)
    results["detection_rate"] = compute_detection_rate(dataset)
    return {k: v for k, v in results.items() if v is not None}
