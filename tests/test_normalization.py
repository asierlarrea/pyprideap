import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.normalization import (
    assess_bridgeability,
    bridge_normalize,
    lift_somascan,
    reference_median_normalize,
    scale_analytes,
    select_bridge_samples,
    subset_normalize,
)


def _make_dataset(sample_ids, expression_data, qc=None):
    """Build a minimal Olink dataset for normalization tests."""
    samples_dict = {
        "SampleID": sample_ids,
        "SampleType": ["SAMPLE"] * len(sample_ids),
    }
    if qc is not None:
        samples_dict["SampleQC"] = qc
    else:
        samples_dict["SampleQC"] = ["PASS"] * len(sample_ids)

    samples = pd.DataFrame(samples_dict)
    expr = pd.DataFrame(expression_data, index=range(len(sample_ids)))

    features_cols = {
        "OlinkID": list(expr.columns),
        "UniProt": [f"P{i}" for i in range(len(expr.columns))],
        "Panel": ["Inf"] * len(expr.columns),
    }

    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=samples,
        features=pd.DataFrame(features_cols),
        expression=expr,
        metadata={},
    )


class TestBridgeNormalize:
    def test_bridge_normalize_adjusts_values(self):
        """Bridge normalization should shift dataset2 values towards dataset1."""
        ds1 = _make_dataset(
            ["S1", "S2", "S3"],
            {"O1": [10.0, 11.0, 12.0], "O2": [5.0, 6.0, 7.0]},
        )
        ds2 = _make_dataset(
            ["S1", "S2", "S4"],
            {"O1": [8.0, 9.0, 10.0], "O2": [3.0, 4.0, 5.0]},
        )

        result = bridge_normalize(ds1, ds2, bridge_samples=[0, 1])

        # median(ds1 bridge O1) = 10.5, median(ds2 bridge O1) = 8.5 => adj = 2.0
        # So ds2 O1 values should increase by 2.0
        assert result.expression["O1"].iloc[0] == pytest.approx(10.0)
        assert result.expression["O1"].iloc[1] == pytest.approx(11.0)
        assert result.expression["O1"].iloc[2] == pytest.approx(12.0)

        # median(ds1 bridge O2) = 5.5, median(ds2 bridge O2) = 3.5 => adj = 2.0
        assert result.expression["O2"].iloc[0] == pytest.approx(5.0)

    def test_bridge_normalize_no_overlap_raises(self):
        """ValueError when no bridge samples are found in both datasets."""
        ds1 = _make_dataset(
            ["S1", "S2"],
            {"O1": [10.0, 11.0]},
        )
        ds2 = _make_dataset(
            ["S3", "S4"],
            {"O1": [8.0, 9.0]},
        )

        with pytest.raises(ValueError, match="No bridge samples"):
            bridge_normalize(ds1, ds2, bridge_samples=[10, 11])


class TestSubsetNormalize:
    def test_subset_normalize_adjusts_values(self):
        """Subset normalization using reference proteins shifts values."""
        ds1 = _make_dataset(
            ["S1", "S2", "S3"],
            {"O1": [10.0, 12.0, 14.0], "O2": [4.0, 6.0, 8.0]},
        )
        ds2 = _make_dataset(
            ["S4", "S5", "S6"],
            {"O1": [7.0, 9.0, 11.0], "O2": [1.0, 3.0, 5.0]},
        )

        result = subset_normalize(ds1, ds2, reference_proteins=["O1"])

        # median(ds1 O1) = 12.0, median(ds2 O1) = 9.0 => adj = 3.0
        assert result.expression["O1"].iloc[0] == pytest.approx(10.0)
        assert result.expression["O1"].iloc[1] == pytest.approx(12.0)

        # O2 is NOT a reference protein, so it is not adjusted
        assert result.expression["O2"].iloc[0] == pytest.approx(1.0)


class TestReferenceMedianNormalize:
    def test_reference_median_normalize(self):
        """Shift each protein so its median matches the reference median."""
        ds = _make_dataset(
            ["S1", "S2", "S3"],
            {"O1": [2.0, 4.0, 6.0], "O2": [1.0, 3.0, 5.0]},
        )

        ref_medians = {"O1": 10.0, "O2": 5.0}
        result = reference_median_normalize(ds, ref_medians)

        # current median(O1) = 4.0, ref = 10.0 => adj = 6.0
        assert result.expression["O1"].median() == pytest.approx(10.0)
        # current median(O2) = 3.0, ref = 5.0 => adj = 2.0
        assert result.expression["O2"].median() == pytest.approx(5.0)


class TestSelectBridgeSamples:
    def test_select_bridge_samples_returns_expected_count(self):
        """Should return up to n bridge samples."""
        ds = _make_dataset(
            ["S1", "S2", "S3", "S4", "S5"],
            {
                "O1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "O2": [5.0, 4.0, 3.0, 2.0, 1.0],
                "O3": [2.0, 3.0, 4.0, 5.0, 6.0],
            },
        )

        result = select_bridge_samples(ds, n=3)
        assert len(result) == 3
        assert all(isinstance(s, (int, np.integer)) for s in result)

    def test_select_bridge_samples_prefers_pass_qc(self):
        """Samples with SampleQC == 'PASS' should be preferred."""
        ds = _make_dataset(
            ["S1", "S2", "S3", "S4"],
            {
                "O1": [1.0, 2.0, 3.0, 4.0],
                "O2": [5.0, 4.0, 3.0, 2.0],
            },
            qc=["PASS", "PASS", "FAIL", "FAIL"],
        )

        result = select_bridge_samples(ds, n=2)
        assert len(result) == 2
        # All returned samples should be PASS (indices 0 and 1)
        for idx in result:
            assert ds.samples.loc[idx, "SampleQC"] == "PASS"


class TestAssessBridgeability:
    def test_assess_bridgeability_returns_dataframe(self):
        """Output should have the expected columns and one row per protein."""
        ds1 = _make_dataset(
            ["S1", "S2", "S3"],
            {"O1": [3.0, 4.0, 5.0], "O2": [1.0, 2.0, 3.0]},
        )
        ds2 = _make_dataset(
            ["S1", "S2", "S3"],
            {"O1": [3.5, 4.5, 5.5], "O2": [1.5, 2.5, 3.5]},
        )

        result = assess_bridgeability(ds1, ds2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {
            "protein_id",
            "correlation",
            "median_diff",
            "detection_rate_1",
            "detection_rate_2",
            "bridgeable",
        }
        assert expected_cols.issubset(set(result.columns))
        # Perfect linear shift => correlation should be 1.0
        assert result["correlation"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# SomaScan normalization (multiplicative)
# ---------------------------------------------------------------------------


def _make_somascan_dataset(sample_ids, expression_data):
    """Build a minimal SomaScan dataset for normalization tests."""
    samples = pd.DataFrame(
        {
            "SampleId": sample_ids,
            "SampleType": ["Sample"] * len(sample_ids),
        }
    )
    expr = pd.DataFrame(expression_data, index=range(len(sample_ids)))
    features = pd.DataFrame(
        {
            "SeqId": list(expr.columns),
            "UniProt": [f"P{i}" for i in range(len(expr.columns))],
            "Target": [f"T{i}" for i in range(len(expr.columns))],
            "Dilution": ["20"] * len(expr.columns),
        }
    )
    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expr,
        metadata={"AssayVersion": "v4.1"},
    )


class TestScaleAnalytes:
    def test_multiplies_rfu_values(self):
        ds = _make_somascan_dataset(
            ["S1", "S2"],
            {"SL0": [1000.0, 2000.0], "SL1": [500.0, 1500.0]},
        )
        result = scale_analytes(ds, {"SL0": 2.0, "SL1": 0.5})
        assert result.expression["SL0"].iloc[0] == pytest.approx(2000.0)
        assert result.expression["SL0"].iloc[1] == pytest.approx(4000.0)
        assert result.expression["SL1"].iloc[0] == pytest.approx(250.0)
        assert result.expression["SL1"].iloc[1] == pytest.approx(750.0)

    def test_partial_scalars_only_affects_matched(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0], "SL1": [500.0]},
        )
        result = scale_analytes(ds, {"SL0": 3.0})
        assert result.expression["SL0"].iloc[0] == pytest.approx(3000.0)
        # SL1 is unchanged
        assert result.expression["SL1"].iloc[0] == pytest.approx(500.0)

    def test_does_not_mutate_original(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        original_val = ds.expression["SL0"].iloc[0]
        scale_analytes(ds, {"SL0": 2.0})
        assert ds.expression["SL0"].iloc[0] == original_val

    def test_raises_when_no_keys_match(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        with pytest.raises(ValueError, match="None of the scalar keys"):
            scale_analytes(ds, {"NONEXISTENT": 2.0})

    def test_accepts_series(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0], "SL1": [500.0]},
        )
        scalars = pd.Series({"SL0": 2.0, "SL1": 0.5})
        result = scale_analytes(ds, scalars)
        assert result.expression["SL0"].iloc[0] == pytest.approx(2000.0)


class TestLiftSomascan:
    def test_lift_applies_scalars_with_default_1(self):
        ds = _make_somascan_dataset(
            ["S1", "S2"],
            {"SL0": [1000.0, 2000.0], "SL1": [500.0, 1500.0]},
        )
        # Only provide scalar for SL0; SL1 should default to 1.0
        result = lift_somascan(ds, {"SL0": 1.5})
        assert result.expression["SL0"].iloc[0] == pytest.approx(1500.0)
        assert result.expression["SL1"].iloc[0] == pytest.approx(500.0)

    def test_lift_rounds_to_1_decimal(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        result = lift_somascan(ds, {"SL0": 1.123456})
        assert result.expression["SL0"].iloc[0] == pytest.approx(1123.5)

    def test_lift_sets_signal_space_metadata(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        result = lift_somascan(ds, {"SL0": 1.0}, target_version="7k")
        assert result.metadata["SignalSpace"] == "7k"

    def test_lift_preserves_existing_metadata(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        result = lift_somascan(ds, {"SL0": 1.0}, target_version="11k")
        assert result.metadata["AssayVersion"] == "v4.1"
        assert result.metadata["SignalSpace"] == "11k"

    def test_lift_does_not_mutate_original(self):
        ds = _make_somascan_dataset(
            ["S1"],
            {"SL0": [1000.0]},
        )
        lift_somascan(ds, {"SL0": 2.0})
        assert ds.expression["SL0"].iloc[0] == pytest.approx(1000.0)
        assert "SignalSpace" not in ds.metadata
