from __future__ import annotations

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Level, ValidationResult

_REQUIRED_FEATURE_COLS = {"SeqId", "UniProt", "Target", "Dilution"}
_REQUIRED_SAMPLE_COLS = {"SampleId", "SampleType"}
_EXPECTED_HEADER_KEYS = {"AssayVersion", "AssayType"}


class SomaScanValidator:
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        results.extend(self._check_schema(dataset))
        results.extend(self._check_rfu_positive(dataset))
        results.extend(self._check_not_empty(dataset))
        results.extend(self._check_header_metadata(dataset))
        results.extend(self._check_data_quality(dataset))
        return results

    def _check_schema(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for col in _REQUIRED_SAMPLE_COLS:
            if col not in ds.samples.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="somascan.schema.missing_sample_column",
                        message=f"Missing required sample column: {col}",
                        details={"column": col},
                    )
                )
        for col in _REQUIRED_FEATURE_COLS:
            if col not in ds.features.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="somascan.schema.missing_feature_column",
                        message=f"Missing required feature column: {col}",
                        details={"column": col},
                    )
                )
        return results

    def _check_rfu_positive(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return []
        values = np.asarray(pd.to_numeric(pd.Series(ds.expression.values.flatten()), errors="coerce").dropna())
        if len(values) == 0:
            return []
        negative = values < 0
        if negative.any():
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="somascan.rfu.positive",
                    message=f"RFU values must be positive: {int(negative.sum())} negative values found",
                    details={"count": int(negative.sum()), "min": float(values.min())},
                )
            ]
        return []

    def _check_not_empty(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="somascan.expression.empty",
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
                    rule="somascan.expression.high_nan",
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
                    rule="somascan.expression.square_matrix",
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
                    rule="somascan.expression.sparse_samples",
                    message=f"Samples have very few measured values "
                    f"(median {median_non_nan:.0f} of {n_features} features). "
                    f"This may indicate a parsing issue.",
                    details={"median_non_nan": median_non_nan, "n_features": n_features},
                )
            )

        return results

    def _check_header_metadata(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for key in _EXPECTED_HEADER_KEYS:
            if key not in ds.metadata:
                results.append(
                    ValidationResult(
                        level=Level.WARNING,
                        rule="somascan.metadata.missing_header",
                        message=f"Missing expected ADAT header key: {key}",
                        details={"key": key},
                    )
                )
        return results
