from __future__ import annotations

from dataclasses import dataclass, field


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
