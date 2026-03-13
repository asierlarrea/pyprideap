"""Tests for sample filtering utilities."""

import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.filtering import filter_controls, filter_qc


def _make_dataset(sample_types=None, qc_values=None, n_features=3):
    n = len(sample_types) if sample_types else 4
    samples_data = {"SampleID": [f"S{i}" for i in range(n)]}
    if sample_types is not None:
        samples_data["SampleType"] = sample_types
    if qc_values is not None:
        samples_data["SampleQC"] = qc_values

    samples = pd.DataFrame(samples_data)
    features = pd.DataFrame(
        {
            "OlinkID": [f"OID{i}" for i in range(n_features)],
            "UniProt": [f"P{i}" for i in range(n_features)],
        }
    )
    expression = pd.DataFrame(
        [[1.0 + i + j for j in range(n_features)] for i in range(n)],
        columns=[f"OID{j}" for j in range(n_features)],
    )
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=samples,
        features=features,
        expression=expression,
    )


class TestFilterControls:
    def test_removes_control_samples(self):
        ds = _make_dataset(sample_types=["Sample", "CONTROL", "Sample", "Negative"])
        result = filter_controls(ds)
        assert len(result.samples) == 2
        assert list(result.samples["SampleType"]) == ["Sample", "Sample"]
        assert len(result.expression) == 2

    def test_no_sample_type_returns_unchanged(self):
        ds = _make_dataset(sample_types=None)
        result = filter_controls(ds)
        assert len(result.samples) == len(ds.samples)

    def test_no_controls_returns_unchanged(self):
        ds = _make_dataset(sample_types=["Sample", "Sample", "Sample", "Sample"])
        result = filter_controls(ds)
        assert len(result.samples) == 4

    def test_case_insensitive(self):
        ds = _make_dataset(sample_types=["Sample", "calibrator", "NEGATIVE CONTROL", "QC"])
        result = filter_controls(ds)
        assert len(result.samples) == 1
        assert result.samples["SampleType"].iloc[0] == "Sample"


class TestFilterQc:
    def test_keeps_pass_and_warn(self):
        ds = _make_dataset(qc_values=["PASS", "FAIL", "WARN", "NA"])
        result = filter_qc(ds)
        assert len(result.samples) == 2
        assert set(result.samples["SampleQC"]) == {"PASS", "WARN"}

    def test_no_qc_column_returns_unchanged(self):
        ds = _make_dataset(sample_types=["Sample"] * 4)
        result = filter_qc(ds)
        assert len(result.samples) == 4

    def test_custom_keep_values(self):
        ds = _make_dataset(qc_values=["PASS", "FAIL", "WARN", "NA"])
        result = filter_qc(ds, keep=("PASS",))
        assert len(result.samples) == 1
        assert result.samples["SampleQC"].iloc[0] == "PASS"

    def test_expression_aligned_with_samples(self):
        ds = _make_dataset(qc_values=["PASS", "FAIL", "PASS", "FAIL"])
        result = filter_qc(ds)
        assert len(result.expression) == len(result.samples) == 2
