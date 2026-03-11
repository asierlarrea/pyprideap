from __future__ import annotations

from typing import Protocol

from pyap.core import AffinityDataset, ValidationResult


class BaseValidator(Protocol):
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]: ...
