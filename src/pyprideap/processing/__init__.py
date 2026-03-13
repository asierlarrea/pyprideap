"""Processing sub-package — filtering, normalization, and LOD computation."""

from pyprideap.processing.filtering import filter_controls, filter_qc
from pyprideap.processing.lod import (
    LodMethod,
    LodStats,
    compute_lod_from_controls,
    compute_lod_stats,
    compute_nclod,
    get_bundled_fixed_lod_path,
    get_reported_lod,
    get_valid_proteins,
    load_fixed_lod,
)
from pyprideap.processing.normalization import (
    assess_bridgeability,
    bridge_normalize,
    reference_median_normalize,
    select_bridge_samples,
    subset_normalize,
)

__all__ = [
    "filter_controls",
    "filter_qc",
    "LodMethod",
    "LodStats",
    "compute_lod_from_controls",
    "compute_lod_stats",
    "compute_nclod",
    "get_bundled_fixed_lod_path",
    "get_reported_lod",
    "get_valid_proteins",
    "load_fixed_lod",
    "assess_bridgeability",
    "bridge_normalize",
    "reference_median_normalize",
    "select_bridge_samples",
    "subset_normalize",
]
