import json
from dataclasses import asdict

import pandas as pd

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
    compute_distribution,
    compute_lod_analysis,
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
