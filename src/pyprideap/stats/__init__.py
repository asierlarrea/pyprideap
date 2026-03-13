"""Stats sub-package — descriptive statistics, differential testing, experimental design."""

from pyprideap.stats.descriptive import DatasetStats, compute_stats
from pyprideap.stats.design import randomize_plates

__all__ = [
    "DatasetStats",
    "compute_stats",
    "randomize_plates",
]
