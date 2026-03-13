"""I/O sub-package — data reading and validation."""

from pyprideap.io.readers.registry import read
from pyprideap.io.validators import validate

__all__ = ["read", "validate"]
