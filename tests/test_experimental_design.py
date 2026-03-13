import pandas as pd
import pytest

from pyprideap.stats.design import randomize_plates


def _make_samples(n, paired_col=None, paired_values=None):
    """Build a minimal samples DataFrame with n rows."""
    df = pd.DataFrame(
        {
            "SampleID": [f"S{i}" for i in range(n)],
            "SampleType": ["SAMPLE"] * n,
        }
    )
    if paired_col is not None and paired_values is not None:
        df[paired_col] = paired_values
    return df


class TestRandomizePlates:
    def test_randomize_plates_assigns_all_samples(self):
        """Every sample should get a PlateNumber and WellPosition."""
        samples = _make_samples(20)
        result = randomize_plates(samples, n_plates=1, plate_size=88, seed=42)

        assert "PlateNumber" in result.columns
        assert "WellPosition" in result.columns
        assert result["PlateNumber"].notna().all()
        assert result["WellPosition"].notna().all()
        assert len(result) == 20

    def test_randomize_plates_respects_capacity(self):
        """No plate should exceed plate_size."""
        samples = _make_samples(50)
        result = randomize_plates(samples, n_plates=3, plate_size=20, seed=42)

        plate_counts = result["PlateNumber"].value_counts()
        assert all(count <= 20 for count in plate_counts.values)
        assert len(result) == 50

    def test_randomize_plates_reproducible_with_seed(self):
        """Same seed should produce the same plate assignments."""
        samples = _make_samples(30)

        result1 = randomize_plates(samples, n_plates=2, plate_size=20, seed=123)
        result2 = randomize_plates(samples, n_plates=2, plate_size=20, seed=123)

        pd.testing.assert_frame_equal(result1, result2)

    def test_randomize_plates_keep_paired(self):
        """Samples with the same paired value should end up on the same plate."""
        # 12 samples, 4 subjects x 3 time-points
        paired_values = ["subj1"] * 3 + ["subj2"] * 3 + ["subj3"] * 3 + ["subj4"] * 3
        samples = _make_samples(12, paired_col="SubjectID", paired_values=paired_values)

        result = randomize_plates(samples, n_plates=2, plate_size=10, keep_paired="SubjectID", seed=42)

        # Each subject's samples must be on the same plate
        for subj in ["subj1", "subj2", "subj3", "subj4"]:
            plates = result.loc[result["SubjectID"] == subj, "PlateNumber"].unique()
            assert len(plates) == 1, f"Subject {subj} is split across plates: {plates}"

    def test_randomize_plates_too_many_samples_raises(self):
        """ValueError when samples exceed total plate capacity."""
        samples = _make_samples(100)

        with pytest.raises(ValueError, match="Too many samples"):
            randomize_plates(samples, n_plates=1, plate_size=10, seed=42)
