"""Capability #3: fairness regression detection, attributed to a cause.

Detecting that a fairness metric got worse between two model versions is the easy
half and the useless half. "Demographic parity went from 0.11 to 0.19" is not
actionable, because the two versions differ in *two* ways at once — someone
changed the configuration, and the training data moved underneath them — and the
remedy for each is opposite. Rolling back the model change is wasted work if the
regression came from the data.

So this package computes the metric (:mod:`.metrics`, :mod:`.evaluate`) and then
decomposes the change into a model effect and a data effect
(:mod:`.decompose`), by training the two counterfactuals that hold one factor
fixed. The decomposition is exact: the two effects sum to the observed change.
"""

from __future__ import annotations

from .metrics import (
    DIFFERENCE_METRICS,
    GROUP_METRICS,
    MIN_GROUP_N,
    GroupResult,
    differences,
    group_metrics,
)

__all__ = [
    "DIFFERENCE_METRICS",
    "GROUP_METRICS",
    "MIN_GROUP_N",
    "GroupResult",
    "differences",
    "group_metrics",
]
