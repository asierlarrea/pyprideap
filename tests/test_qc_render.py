import pytest

plotly = pytest.importorskip("plotly")

from pyprideap.viz.qc.compute import (  # noqa: E402
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    HeatmapData,
    LodAnalysisData,
    PcaData,
    QcLodSummaryData,
    UmapData,
    VolcanoData,
)
from pyprideap.viz.qc.render import (  # noqa: E402
    render_correlation,
    render_cv_distribution,
    render_data_completeness,
    render_distribution,
    render_heatmap,
    render_lod_analysis,
    render_pca,
    render_qc_summary,
    render_umap,
    render_volcano,
)


class TestRenderFunctions:
    def test_render_distribution(self):
        data = DistributionData(
            sample_ids=["S1", "S2"],
            sample_values=[[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            xlabel="NPX Value",
        )
        fig = render_distribution(data)
        assert fig is not None

    def test_render_qc_summary_with_lod(self):
        data = QcLodSummaryData(
            categories=["PASS & NPX > LOD", "PASS & NPX ≤ LOD", "WARN & NPX > LOD"],
            counts=[80, 15, 5],
        )
        fig = render_qc_summary(data)
        assert fig is not None

    def test_render_qc_summary_simple(self):
        data = QcLodSummaryData(categories=["PASS", "FAIL"], counts=[10, 2])
        fig = render_qc_summary(data)
        assert fig is not None

    def test_render_lod_analysis(self):
        data = LodAnalysisData(assay_ids=["A1", "A2"], above_lod_pct=[95.0, 80.0], panel=["Inf", "Inf"])
        fig = render_lod_analysis(data)
        assert fig is not None

    def test_render_pca(self):
        data = PcaData(
            pc1=[1.0, 2.0], pc2=[3.0, 4.0], variance_explained=[0.6, 0.3], labels=["S1", "S2"], groups=["A", "B"]
        )
        fig = render_pca(data)
        assert fig is not None

    def test_render_correlation(self):
        data = CorrelationData(matrix=[[1.0, 0.5], [0.5, 1.0]], labels=["S1", "S2"])
        fig = render_correlation(data)
        assert fig is not None

    def test_render_data_completeness(self):
        data = DataCompletenessData(
            sample_ids=["S1", "S2"],
            above_lod_rate=[0.7, 0.8],
            below_lod_rate=[0.3, 0.2],
            protein_ids=["P1", "P2"],
            missing_freq=[0.1, 0.4],
        )
        fig = render_data_completeness(data)
        assert fig is not None

    def test_render_cv_distribution(self):
        data = CvDistributionData(feature_ids=["F1", "F2"], cv_values=[0.1, 0.2])
        fig = render_cv_distribution(data)
        assert fig is not None

    def test_render_umap(self):
        data = UmapData(x=[1.0, 2.0], y=[3.0, 4.0], labels=["S1", "S2"], groups=["A", "B"])
        fig = render_umap(data)
        assert fig is not None

    def test_render_heatmap(self):
        data = HeatmapData(
            values=[[0.5, -0.3], [-0.5, 0.3]],
            sample_labels=["S1", "S2"],
            protein_labels=["P1", "P2"],
            sample_order=[0, 1],
            protein_order=[1, 0],
        )
        fig = render_heatmap(data)
        assert fig is not None

    def test_render_volcano(self):
        data = VolcanoData(
            protein_ids=["P1", "P2", "P3"],
            assay_names=["A1", "A2", "A3"],
            fold_change=[2.0, -1.5, 0.1],
            neg_log10_pval=[3.0, 2.5, 0.5],
            significant=[True, True, False],
            direction=["up", "down", "ns"],
        )
        fig = render_volcano(data)
        assert fig is not None
