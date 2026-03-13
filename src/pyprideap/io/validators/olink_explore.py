from __future__ import annotations

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Level, ValidationResult

_REQUIRED_SAMPLE_COLS = {"SampleID", "SampleType", "SampleQC"}
_REQUIRED_FEATURE_COLS = {"OlinkID", "UniProt", "Panel"}
_VALID_QC_VALUES = {"NA", "PASS", "WARN", "FAIL"}
_NPX_MIN = -10.0
_NPX_MAX = 40.0


class OlinkExploreValidator:
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        results.extend(self._check_schema(dataset))
        results.extend(self._check_qc_values(dataset))
        results.extend(self._check_qc_consistency(dataset))
        results.extend(self._check_npx_range(dataset))
        results.extend(self._check_not_empty(dataset))
        results.extend(self._check_lod_present(dataset))
        results.extend(self._check_data_quality(dataset))
        return results

    def _check_schema(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for col in _REQUIRED_SAMPLE_COLS:
            if col not in ds.samples.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="olink.schema.missing_sample_column",
                        message=f"Missing required sample column: {col}",
                        details={"column": col},
                    )
                )
        for col in _REQUIRED_FEATURE_COLS:
            if col not in ds.features.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="olink.schema.missing_feature_column",
                        message=f"Missing required feature column: {col}",
                        details={"column": col},
                    )
                )
        return results

    def _check_qc_values(self, ds: AffinityDataset) -> list[ValidationResult]:
        if "SampleQC" not in ds.samples.columns:
            return []
        invalid = set(ds.samples["SampleQC"].dropna().unique()) - _VALID_QC_VALUES
        if invalid:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="olink.SampleQC.invalid_values",
                    message=f"Invalid SampleQC values: {invalid}. Must be one of {_VALID_QC_VALUES}",
                    details={"invalid_values": list(invalid)},
                )
            ]
        return []

    def _check_qc_consistency(self, ds: AffinityDataset) -> list[ValidationResult]:
        if "SampleQC" not in ds.samples.columns:
            return []
        results = []
        for idx in range(len(ds.samples)):
            row = ds.samples.iloc[idx]
            qc = row.get("SampleQC")
            if idx >= len(ds.expression):
                continue
            expr_row = ds.expression.iloc[idx]
            if qc in ("FAIL", "NA"):
                if not expr_row.isna().all():
                    results.append(
                        ValidationResult(
                            level=Level.ERROR,
                            rule="olink.npx.qc_consistency",
                            message=f"Sample {row.get('SampleID', idx)}: SampleQC={qc} but NPX contains non-NaN values",
                            details={"sample_index": idx, "qc": qc},
                        )
                    )
            elif qc in ("PASS", "WARN"):
                if expr_row.isna().all():
                    results.append(
                        ValidationResult(
                            level=Level.ERROR,
                            rule="olink.npx.qc_consistency",
                            message=f"Sample {row.get('SampleID', idx)}: SampleQC={qc} but all NPX values are NaN",
                            details={"sample_index": idx, "qc": qc},
                        )
                    )
        return results

    def _check_npx_range(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return []
        values = np.asarray(pd.to_numeric(pd.Series(ds.expression.values.flatten()), errors="coerce").dropna())
        if len(values) == 0:
            return []
        out_of_range = (values < _NPX_MIN) | (values > _NPX_MAX)
        if out_of_range.any():
            return [
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.npx.range",
                    message=f"NPX values outside expected log2 range [{_NPX_MIN}, {_NPX_MAX}]: "
                    f"{int(out_of_range.sum())} values out of range",
                    details={"count": int(out_of_range.sum()), "min": float(values.min()), "max": float(values.max())},
                )
            ]
        return []

    def _check_lod_present(self, ds: AffinityDataset) -> list[ValidationResult]:
        if "LOD" not in ds.features.columns:
            return [
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.lod.missing",
                    message="LOD column not found in features — LOD-based quality filtering will not be available",
                )
            ]
        lod_values = pd.to_numeric(ds.features["LOD"], errors="coerce")
        if lod_values.isna().all():
            return [
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.lod.empty",
                    message="LOD column exists but all values are missing or non-numeric",
                )
            ]
        return []

    def _check_not_empty(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="olink.expression.empty",
                    message="Expression matrix is empty — at least one data record required",
                )
            ]
        return []

    def _check_data_quality(self, ds: AffinityDataset) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        n_samples = len(ds.samples)
        n_features = len(ds.features)

        if ds.expression.empty:
            return results

        nan_frac = float(ds.expression.isna().mean().mean())
        if nan_frac > 0.5:
            results.append(
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.expression.high_nan",
                    message=f"Expression matrix is {nan_frac:.0%} NaN "
                    f"({n_samples} samples × {n_features} features). "
                    f"Data may have been pivoted incorrectly or is very sparse.",
                    details={"nan_fraction": nan_frac, "n_samples": n_samples, "n_features": n_features},
                )
            )

        if n_samples == n_features and n_samples > 10:
            results.append(
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.expression.square_matrix",
                    message=f"Expression matrix is square ({n_samples} × {n_features}), "
                    f"which is unusual. Verify sample and feature identifiers.",
                    details={"n_samples": n_samples, "n_features": n_features},
                )
            )

        non_nan_per_sample = ds.expression.notna().sum(axis=1)
        median_non_nan = float(non_nan_per_sample.median())
        if n_features > 0 and median_non_nan / n_features < 0.1:
            results.append(
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.expression.sparse_samples",
                    message=f"Samples have very few measured values "
                    f"(median {median_non_nan:.0f} of {n_features} features). "
                    f"This may indicate a parsing issue.",
                    details={"median_non_nan": median_non_nan, "n_features": n_features},
                )
            )

        return results
