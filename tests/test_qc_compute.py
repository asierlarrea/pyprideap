import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    DistributionData,
    LodAnalysisData,
    MissingValuesData,
    PcaData,
    QcSummaryData,
    compute_all,
    compute_correlation,
    compute_cv_distribution,
    compute_detection_rate,
    compute_distribution,
    compute_lod_analysis,
    compute_missing_values,
    compute_pca,
    compute_qc_summary,
)


def _make_olink_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": ["S1", "S2"],
                "SampleType": ["SAMPLE", "SAMPLE"],
                "SampleQC": ["PASS", "PASS"],
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "Panel": ["Inf", "Inf"]}),
        expression=pd.DataFrame({"O1": [3.5, 4.1], "O2": [2.0, -0.5]}),
        metadata={},
    )


def _make_somascan_dataset():
    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=pd.DataFrame({"SampleId": ["S1", "S2"], "SampleType": ["Sample", "Sample"]}),
        features=pd.DataFrame(
            {
                "SeqId": ["10000-1", "10001-2"],
                "UniProt": ["P1", "P2"],
                "Target": ["T1", "T2"],
                "Dilution": ["20", "0.5"],
            }
        ),
        expression=pd.DataFrame({"SL1": [1234.5, 1100.2], "SL2": [5678.9, 4567.8]}),
        metadata={},
    )


class TestDataclassSerialization:
    def test_distribution_data_serializable(self):
        d = DistributionData(values=[1.0, 2.0], xlabel="NPX")
        result = json.dumps(asdict(d))
        assert '"values"' in result

    def test_all_dataclasses_serializable(self):
        instances = [
            DistributionData(values=[1.0], xlabel="x"),
            QcSummaryData(categories=["PASS"], counts=[10]),
            LodAnalysisData(assay_ids=["A1"], above_lod_pct=[90.0], panel=["P1"]),
            PcaData(pc1=[1.0], pc2=[2.0], variance_explained=[0.5, 0.3], labels=["S1"], groups=["G1"]),
            CorrelationData(matrix=[[1.0]], labels=["S1"]),
            MissingValuesData(
                missing_rate_per_sample=[0.1],
                missing_rate_per_feature=[0.2],
                sample_ids=["S1"],
                feature_ids=["F1"],
            ),
            CvDistributionData(feature_ids=["F1"], cv_values=[0.15]),
            DetectionRateData(sample_ids=["S1"], rates=[0.95]),
        ]
        for inst in instances:
            serialized = json.dumps(asdict(inst))
            assert isinstance(serialized, str)


class TestComputeDistribution:
    def test_olink_returns_npx_values(self):
        ds = _make_olink_dataset()
        result = compute_distribution(ds)
        assert result.xlabel == "NPX (log2)"
        assert len(result.values) == 4

    def test_somascan_returns_log10_rfu(self):
        ds = _make_somascan_dataset()
        result = compute_distribution(ds)
        assert "log10" in result.xlabel.lower() or "RFU" in result.xlabel
        assert len(result.values) == 4


class TestComputeQcSummary:
    def test_olink_returns_qc_counts(self):
        ds = _make_olink_dataset()
        ds.samples["SampleQC"] = ["PASS", "FAIL"]
        result = compute_qc_summary(ds)
        assert result is not None
        assert "PASS" in result.categories
        assert "FAIL" in result.categories

    def test_somascan_returns_none(self):
        ds = _make_somascan_dataset()
        result = compute_qc_summary(ds)
        assert result is None


class TestComputeLodAnalysis:
    def test_olink_with_lod_returns_data(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [1.0, 0.5]
        result = compute_lod_analysis(ds)
        assert result is not None
        assert len(result.assay_ids) == 2
        assert all(0 <= p <= 100 for p in result.above_lod_pct)

    def test_no_lod_returns_none(self):
        ds = _make_somascan_dataset()
        result = compute_lod_analysis(ds)
        assert result is None


class TestComputePca:
    def test_returns_pca_data(self):
        ds = _make_olink_dataset()
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert len(result.pc1) == 2
        assert len(result.variance_explained) == 2

    def test_labels_from_sample_id(self):
        ds = _make_olink_dataset()
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert result.labels == ["S1", "S2"]


class TestComputeCorrelation:
    def test_returns_square_matrix(self):
        ds = _make_olink_dataset()
        result = compute_correlation(ds)
        assert len(result.matrix) == 2
        assert len(result.matrix[0]) == 2

    def test_diagonal_is_one(self):
        ds = _make_olink_dataset()
        result = compute_correlation(ds)
        for i in range(len(result.matrix)):
            assert abs(result.matrix[i][i] - 1.0) < 1e-6


class TestComputeMissingValues:
    def test_no_missing_returns_zero_rates(self):
        ds = _make_olink_dataset()
        result = compute_missing_values(ds)
        assert all(r == 0.0 for r in result.missing_rate_per_sample)

    def test_with_nans(self):
        ds = _make_olink_dataset()
        ds.expression.iloc[0, 0] = np.nan
        result = compute_missing_values(ds)
        assert result.missing_rate_per_sample[0] > 0


class TestComputeCvDistribution:
    def test_somascan_returns_cv(self):
        ds = _make_somascan_dataset()
        result = compute_cv_distribution(ds)
        assert result is not None
        assert len(result.cv_values) == 2

    def test_olink_returns_none(self):
        ds = _make_olink_dataset()
        result = compute_cv_distribution(ds)
        assert result is None


class TestComputeDetectionRate:
    def test_full_detection(self):
        ds = _make_olink_dataset()
        result = compute_detection_rate(ds)
        assert all(r == 1.0 for r in result.rates)

    def test_partial_detection(self):
        ds = _make_olink_dataset()
        ds.expression.iloc[0, 0] = np.nan
        result = compute_detection_rate(ds)
        assert result.rates[0] == 0.5


class TestComputeAll:
    def test_olink_returns_expected_keys(self):
        ds = _make_olink_dataset()
        result = compute_all(ds)
        assert "distribution" in result
        assert "detection_rate" in result
        assert "missing_values" in result

    def test_somascan_includes_cv(self):
        ds = _make_somascan_dataset()
        result = compute_all(ds)
        assert "cv_distribution" in result
        assert result["cv_distribution"] is not None

    def test_none_values_excluded(self):
        ds = _make_somascan_dataset()
        result = compute_all(ds)
        assert all(v is not None for v in result.values())
