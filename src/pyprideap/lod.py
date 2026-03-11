"""LOD (Limit of Detection) computation and analysis for Olink datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset
from pyprideap.filtering import _CONTROL_SAMPLE_TYPES

_MIN_CONTROLS_FOR_LOD = 10
_MIN_STD_FLOOR = 0.2


@dataclass
class LodStats:
    """LOD analysis results."""

    lod_source: str
    n_assays_with_lod: int
    n_assays_total: int
    above_lod_rate: float
    above_lod_per_sample: dict[str, float]
    above_lod_per_panel: dict[str, float]

    def summary(self) -> str:
        lines = [
            f"LOD source: {self.lod_source}",
            f"Assays with LOD: {self.n_assays_with_lod}/{self.n_assays_total}",
            f"Overall above-LOD rate: {self.above_lod_rate:.1%}",
        ]
        if self.above_lod_per_panel:
            lines.append("Per-panel above-LOD rates:")
            for panel, rate in sorted(self.above_lod_per_panel.items()):
                lines.append(f"  {panel}: {rate:.1%}")
        return "\n".join(lines)


def compute_lod_from_controls(dataset: AffinityDataset) -> pd.Series:
    """Calculate LOD per assay from negative control samples.

    LOD = median(NPX) + max(0.2, 3 * std(NPX)) for each assay, computed
    over samples whose SampleType indicates a negative control.
    This follows the Olink OlinkAnalyze R package methodology.

    Returns a Series indexed by feature (column) name with LOD values.
    Raises ValueError if no negative controls are found or too few exist.
    """
    if "SampleType" not in dataset.samples.columns:
        raise ValueError("SampleType column required to identify negative controls")

    is_control = dataset.samples["SampleType"].astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)

    # Further narrow to negative controls only
    is_negative = (
        dataset.samples["SampleType"]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin({"negative", "negative control", "negative_control", "neg"})
    )

    control_mask = is_negative if is_negative.any() else is_control
    if control_mask.sum() < _MIN_CONTROLS_FOR_LOD:
        raise ValueError(
            f"Need at least {_MIN_CONTROLS_FOR_LOD} control samples for LOD calculation, found {control_mask.sum()}"
        )

    control_expr = dataset.expression[control_mask]
    numeric_expr = control_expr.apply(pd.to_numeric, errors="coerce")

    medians = numeric_expr.median()
    stds = numeric_expr.std()
    # Olink formula: median + max(0.2, 3*std) — floor of 0.2 NPX
    lod = medians + np.maximum(_MIN_STD_FLOOR, 3 * stds)

    return pd.Series(lod)


def get_lod_values(dataset: AffinityDataset) -> pd.Series | None:
    """Extract LOD values from the dataset's features table.

    Returns a Series indexed by expression column name, or None if LOD
    is not available. Looks for a 'LOD' column in features and maps it
    to the expression columns via OlinkID.
    """
    if "LOD" not in dataset.features.columns:
        return None

    lod_series = pd.to_numeric(dataset.features["LOD"], errors="coerce")
    if lod_series.isna().all():
        return None

    # Map LOD to expression columns via OlinkID
    if "OlinkID" in dataset.features.columns:
        lod_map = dict(zip(dataset.features["OlinkID"], lod_series))
        return pd.Series({col: lod_map.get(col, np.nan) for col in dataset.expression.columns})

    # Fallback: assume features align with expression columns
    return pd.Series(lod_series.values, index=dataset.expression.columns[: len(lod_series)])


def compute_lod_stats(
    dataset: AffinityDataset,
    lod: pd.Series | None = None,
) -> LodStats:
    """Compute LOD-related statistics for the dataset.

    Args:
        dataset: The AffinityDataset to analyze.
        lod: Pre-computed LOD values per assay. If None, attempts to
             extract from the features table, then falls back to
             computing from controls.

    Returns:
        LodStats with detection rates overall, per-sample, and per-panel.
    """
    source = "provided"

    if lod is None:
        lod = get_lod_values(dataset)
        if lod is not None:
            source = "features_table"

    if lod is None:
        try:
            lod = compute_lod_from_controls(dataset)
            source = "computed_from_controls"
        except ValueError:
            return LodStats(
                lod_source="not_available",
                n_assays_with_lod=0,
                n_assays_total=len(dataset.expression.columns),
                above_lod_rate=0.0,
                above_lod_per_sample={},
                above_lod_per_panel={},
            )

    # Align LOD with expression columns
    lod_aligned = lod.reindex(dataset.expression.columns)
    n_assays_with_lod = int(lod_aligned.notna().sum())
    n_assays_total = len(dataset.expression.columns)

    # Compute above-LOD matrix
    expr_numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    above_lod = expr_numeric.gt(lod_aligned, axis=1)

    # Only consider assays that have LOD values
    has_lod = lod_aligned.notna()
    above_lod_masked = above_lod.loc[:, has_lod]
    expr_valid = expr_numeric.loc[:, has_lod].notna()

    total_valid = int(expr_valid.sum().sum())
    total_above = int(above_lod_masked.sum().sum())
    above_lod_rate = total_above / total_valid if total_valid > 0 else 0.0

    # Per-sample rates
    per_sample_above = above_lod_masked.sum(axis=1)
    per_sample_total = expr_valid.sum(axis=1)
    per_sample_rate = (per_sample_above / per_sample_total).fillna(0.0)

    sample_ids = dataset.samples.get("SampleID", pd.Series(range(len(dataset.samples))))
    above_lod_per_sample = dict(zip(sample_ids.astype(str), per_sample_rate.round(4)))

    # Per-panel rates
    above_lod_per_panel: dict[str, float] = {}
    if "Panel" in dataset.features.columns and "OlinkID" in dataset.features.columns:
        panel_map = dict(zip(dataset.features["OlinkID"], dataset.features["Panel"]))
        for panel_name in dataset.features["Panel"].dropna().unique():
            panel_cols = [
                c for c in dataset.expression.columns if panel_map.get(c) == panel_name and has_lod.get(c, False)
            ]
            if panel_cols:
                panel_above = int(above_lod_masked[panel_cols].sum().sum())
                panel_total = int(expr_valid[panel_cols].sum().sum())
                above_lod_per_panel[str(panel_name)] = panel_above / panel_total if panel_total > 0 else 0.0

    return LodStats(
        lod_source=source,
        n_assays_with_lod=n_assays_with_lod,
        n_assays_total=n_assays_total,
        above_lod_rate=above_lod_rate,
        above_lod_per_sample=above_lod_per_sample,
        above_lod_per_panel=above_lod_per_panel,
    )


def get_valid_proteins(
    dataset: AffinityDataset,
    lod: pd.Series | None = None,
) -> list[str]:
    """Return protein/assay IDs that pass QC and are above LOD.

    A protein is valid if at least one non-control sample has:
    - SampleQC in (PASS, WARN) — or no QC column present
    - NPX > LOD for that assay

    Args:
        dataset: The AffinityDataset to analyze.
        lod: Pre-computed LOD values. If None, resolved automatically.

    Returns:
        Sorted list of valid protein/assay identifiers (expression column names).
    """
    from pyprideap.filtering import filter_controls, filter_qc

    # Filter to biological samples with acceptable QC
    ds = filter_controls(dataset)
    ds = filter_qc(ds)

    if ds.expression.empty:
        return []

    # Resolve LOD
    if lod is None:
        lod = get_lod_values(dataset)
    if lod is None:
        try:
            lod = compute_lod_from_controls(dataset)
        except ValueError:
            # No LOD available — return all proteins that have any non-NaN values
            expr_numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
            return sorted(expr_numeric.columns[expr_numeric.notna().any()].tolist())

    lod_aligned = lod.reindex(ds.expression.columns)
    expr_numeric = ds.expression.apply(pd.to_numeric, errors="coerce")

    # For each assay, check if any sample has NPX > LOD
    above_lod = expr_numeric.gt(lod_aligned, axis=1)
    valid_cols = above_lod.any(axis=0)

    return sorted(valid_cols[valid_cols].index.tolist())
