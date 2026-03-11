import json
from dataclasses import asdict

from pyprideap.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    DistributionData,
    LodAnalysisData,
    MissingValuesData,
    PcaData,
    QcSummaryData,
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
