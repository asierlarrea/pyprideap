"""Run once to generate binary test fixtures."""

import pandas as pd


def create_olink_parquet():
    df = pd.DataFrame(
        {
            "SampleID": ["S001", "S001", "S002", "S002"],
            "SampleType": ["SAMPLE", "SAMPLE", "SAMPLE", "SAMPLE"],
            "WellID": ["A1", "A1", "B1", "B1"],
            "PlateID": ["P1", "P1", "P1", "P1"],
            "DataAnalysisRefID": ["REF1", "REF1", "REF1", "REF1"],
            "OlinkID": ["OID00001", "OID00002", "OID00001", "OID00002"],
            "UniProt": ["P12345", "Q67890", "P12345", "Q67890"],
            "Assay": ["IL-6", "TNF", "IL-6", "TNF"],
            "Panel": ["Inflammation", "Inflammation", "Inflammation", "Inflammation"],
            "Block": ["B1", "B1", "B1", "B1"],
            "Normalization": ["Plate control", "Plate control", "Plate control", "Plate control"],
            "SampleQC": ["PASS", "PASS", "WARN", "PASS"],
            "ExploreVersion": ["v1.0", "v1.0", "v1.0", "v1.0"],
            "Count": [150, 200, 120, 180],
            "ExtNPX": [3.45, 2.10, 4.12, -0.50],
            "NPX": [3.45, 2.10, 4.12, -0.50],
        }
    )
    df.to_parquet("tests/data/olink_sample.parquet", index=False)


def create_olink_xlsx():
    df = pd.DataFrame(
        {
            "SampleID": ["S001", "S001", "S002", "S002"],
            "PlateID": ["P1", "P1", "P1", "P1"],
            "WellID": ["A1", "A1", "B1", "B1"],
            "SampleType": ["SAMPLE", "SAMPLE", "SAMPLE", "SAMPLE"],
            "OlinkID": ["OID00001", "OID00002", "OID00001", "OID00002"],
            "UniProt": ["P12345", "Q67890", "P12345", "Q67890"],
            "Assay": ["IL-6", "TNF", "IL-6", "TNF"],
            "Panel": ["Inflammation", "Inflammation", "Inflammation", "Inflammation"],
            "NPX": [3.45, 2.10, 4.12, -0.50],
            "LOD": [1.2, 0.8, 1.2, 0.8],
            "SampleQC": ["PASS", "PASS", "PASS", "PASS"],
        }
    )
    df.to_excel("tests/data/olink_sample.xlsx", index=False)


if __name__ == "__main__":
    create_olink_parquet()
    create_olink_xlsx()
