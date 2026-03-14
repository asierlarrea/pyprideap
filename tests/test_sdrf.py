"""Tests for SDRF reader, merge, and volcano integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.io.readers.sdrf import get_grouping_columns, merge_sdrf, read_sdrf

DATA_DIR = Path(__file__).parent / "data"
SDRF_PATH = DATA_DIR / "test_sample.sdrf.tsv"


# ---------------------------------------------------------------------------
# read_sdrf
# ---------------------------------------------------------------------------


class TestReadSdrf:
    def test_reads_and_shortens_columns(self):
        df = read_sdrf(SDRF_PATH)
        assert "source name" in df.columns
        assert "organism" in df.columns
        assert "disease" in df.columns
        assert "sex" in df.columns
        assert "treatment" in df.columns

    def test_duplicate_columns_disambiguated(self):
        df = read_sdrf(SDRF_PATH)
        # Two characteristics[disease] columns -> "disease" and "disease 2"
        assert "disease" in df.columns
        assert "disease 2" in df.columns
        assert df["disease"].iloc[0] == "healthy"
        assert df["disease 2"].iloc[0] == "not applicable"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_sdrf("/nonexistent/file.sdrf.tsv")


# ---------------------------------------------------------------------------
# get_grouping_columns
# ---------------------------------------------------------------------------


class TestGetGroupingColumns:
    def test_identifies_suitable_columns(self):
        df = read_sdrf(SDRF_PATH)
        cols = get_grouping_columns(df)
        # disease (2 groups: healthy, COVID-19) and treatment (2 groups) and sex (2 groups)
        # should be found; organism (1 group) should not
        assert "organism" not in cols
        assert "disease" in cols
        assert "treatment" in cols
        assert "sex" in cols

    def test_excludes_high_cardinality(self):
        df = pd.DataFrame(
            {
                "source name": [f"S{i}" for i in range(50)],
                "group_ok": ["A", "B"] * 25,
                "group_too_many": [f"G{i}" for i in range(50)],
            }
        )
        cols = get_grouping_columns(df)
        assert "group_ok" in cols
        assert "group_too_many" not in cols

    def test_excludes_small_groups(self):
        df = pd.DataFrame(
            {
                "source name": ["S1", "S2", "S3", "S4"],
                "tiny": ["A", "A", "B", "C"],  # C has only 1 sample
            }
        )
        cols = get_grouping_columns(df)
        assert "tiny" not in cols


# ---------------------------------------------------------------------------
# merge_sdrf
# ---------------------------------------------------------------------------


def _make_dataset(sample_ids: list[str]) -> AffinityDataset:
    n = len(sample_ids)
    samples = pd.DataFrame({"SampleId": sample_ids})
    rng = np.random.default_rng(42)
    expression = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=["P1", "P2", "P3"],
    )
    features = pd.DataFrame(
        {
            "OlinkID": ["P1", "P2", "P3"],
            "UniProt": ["UP1", "UP2", "UP3"],
            "Assay": ["Assay1", "Assay2", "Assay3"],
        }
    )
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=samples,
        features=features,
        expression=expression,
        metadata={},
    )


class TestMergeSdrf:
    def test_merge_adds_columns(self):
        ds = _make_dataset(["S001", "S002", "S003"])
        sdrf = read_sdrf(SDRF_PATH)
        merged = merge_sdrf(ds, sdrf)
        assert "disease" in merged.samples.columns
        assert "sex" in merged.samples.columns
        assert len(merged.samples) == 3

    def test_merge_preserves_expression(self):
        ds = _make_dataset(["S001", "S002", "S003"])
        sdrf = read_sdrf(SDRF_PATH)
        merged = merge_sdrf(ds, sdrf)
        pd.testing.assert_frame_equal(merged.expression, ds.expression)

    def test_merge_unmatched_samples_get_nan(self):
        ds = _make_dataset(["S001", "S999"])
        sdrf = read_sdrf(SDRF_PATH)
        merged = merge_sdrf(ds, sdrf)
        assert merged.samples.loc[0, "disease"] == "healthy"
        assert pd.isna(merged.samples.loc[1, "disease"])

    def test_merge_auto_detects_sample_col(self):
        ds = _make_dataset(["S001", "S002"])
        sdrf = read_sdrf(SDRF_PATH)
        merged = merge_sdrf(ds, sdrf)
        assert "disease" in merged.samples.columns

    def test_merge_explicit_sample_col(self):
        ds = _make_dataset(["S001", "S002"])
        sdrf = read_sdrf(SDRF_PATH)
        merged = merge_sdrf(ds, sdrf, sample_col="SampleId")
        assert "disease" in merged.samples.columns


# ---------------------------------------------------------------------------
# Volcano integration (requires scipy + statsmodels)
# ---------------------------------------------------------------------------

scipy_stats = pytest.importorskip("scipy.stats")
statsmodels = pytest.importorskip("statsmodels")


class TestVolcanoIntegration:
    def _make_de_dataset(self) -> tuple[AffinityDataset, pd.DataFrame]:
        """Dataset with clear differential expression between two groups."""
        rng = np.random.default_rng(42)
        n_per_group = 20
        n_proteins = 50

        sample_ids = [f"S{i}" for i in range(n_per_group * 2)]
        groups = ["control"] * n_per_group + ["treated"] * n_per_group

        # Most proteins: no difference. First 5: significant difference.
        expr = rng.standard_normal((n_per_group * 2, n_proteins))
        for j in range(5):
            expr[n_per_group:, j] += 3.0  # large effect

        protein_ids = [f"OID{i:05d}" for i in range(n_proteins)]
        samples = pd.DataFrame({"SampleId": sample_ids, "treatment": groups})
        expression = pd.DataFrame(expr, columns=protein_ids)
        features = pd.DataFrame(
            {
                "OlinkID": protein_ids,
                "UniProt": [f"UP{i}" for i in range(n_proteins)],
                "Assay": [f"Assay{i}" for i in range(n_proteins)],
            }
        )
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=samples,
            features=features,
            expression=expression,
            metadata={},
        )

        # Build a minimal SDRF
        sdrf = pd.DataFrame(
            {
                "source name": sample_ids,
                "treatment": groups,
            }
        )
        return ds, sdrf

    def test_compute_volcano_from_ttest(self):
        from pyprideap.stats.differential import ttest
        from pyprideap.viz.qc.compute import compute_volcano

        ds, _ = self._make_de_dataset()
        result = ttest(ds, group_var="treatment")
        assert len(result) == 50

        vdata = compute_volcano(result)
        assert vdata is not None
        assert len(vdata.protein_ids) == 50
        # At least some should be significant
        assert sum(vdata.significant) >= 3

    def test_compute_sdrf_volcanoes(self):
        from pyprideap.viz.qc.report import _compute_sdrf_volcanoes

        ds, sdrf_df = self._make_de_dataset()
        # Write temporary SDRF
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sdrf.tsv", delete=False) as f:
            sdrf_df.to_csv(f, sep="\t", index=False)
            sdrf_path = f.name

        results = _compute_sdrf_volcanoes(ds, sdrf_path)
        Path(sdrf_path).unlink()

        assert "treatment" in results
        assert len(results["treatment"]) == 1  # exactly 2 groups -> 1 comparison
        label, vdata = results["treatment"][0]
        assert "control" in label and "treated" in label
        assert sum(vdata.significant) >= 3

    def test_qc_report_with_sdrf(self, tmp_path):
        pytest.importorskip("plotly")

        from pyprideap.viz.qc.report import qc_report

        ds, sdrf_df = self._make_de_dataset()
        # Write temporary SDRF
        sdrf_path = tmp_path / "test.sdrf.tsv"
        sdrf_df.to_csv(sdrf_path, sep="\t", index=False)

        output = tmp_path / "report.html"
        result = qc_report(ds, output, sdrf_path=sdrf_path)
        assert result.exists()

        html = result.read_text()
        assert "Differential Expression" in html
        assert "volcano-var-select" in html
        assert "volcano-plot" in html

    def test_qc_report_without_sdrf_has_no_volcano(self, tmp_path):
        pytest.importorskip("plotly")

        from pyprideap.viz.qc.report import qc_report

        ds, _ = self._make_de_dataset()
        output = tmp_path / "report.html"
        result = qc_report(ds, output)
        assert result.exists()

        html = result.read_text()
        assert 'class="volcano-plot"' not in html
