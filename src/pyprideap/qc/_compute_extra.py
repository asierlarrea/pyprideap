from __future__ import annotations

import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    MissingValuesData,
    PcaData,
)


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
