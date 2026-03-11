"""pyprideap — Python library for PRIDE Affinity Proteomics (PAD) archive data."""

from importlib.metadata import version

from pyprideap.core import AffinityDataset, Level, Platform, ValidationResult
from pyprideap.filtering import filter_controls, filter_qc
from pyprideap.lod import LodStats, compute_lod_from_controls, compute_lod_stats, get_valid_proteins
from pyprideap.pride import PrideClient
from pyprideap.readers.registry import read
from pyprideap.stats import DatasetStats, compute_stats
from pyprideap.validators import validate

__version__ = version("pyprideap")

__all__ = [
    "__version__",
    "AffinityDataset",
    "DatasetStats",
    "Level",
    "LodStats",
    "Platform",
    "PrideClient",
    "ValidationResult",
    "compute_lod_from_controls",
    "compute_lod_stats",
    "compute_stats",
    "filter_controls",
    "filter_qc",
    "get_valid_proteins",
    "read",
    "validate",
]
