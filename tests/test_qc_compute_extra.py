import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.qc._compute_extra import (
    compute_correlation,
    compute_cv_distribution,
    compute_detection_rate,
    compute_missing_values,
    compute_pca,
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
