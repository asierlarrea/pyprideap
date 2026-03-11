from __future__ import annotations

from pyap.core import AffinityDataset, ValidationResult
from pyap.validators.olink_explore import OlinkExploreValidator


class OlinkTargetValidator(OlinkExploreValidator):
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results = super().validate(dataset)
        for r in results:
            r.rule = r.rule.replace("olink.", "olink_target.", 1)
        return results
