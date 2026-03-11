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
