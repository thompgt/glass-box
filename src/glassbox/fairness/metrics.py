"""Group fairness metrics, defined in this repository rather than imported.

fairlearn would supply all of these. It is deliberately not used, for one reason:
``audit.fairness_evaluations`` records a metric *value* as a durable claim, and a
claim is only checkable if the definition behind it is versioned alongside it.
"equalized_odds_difference = 0.19" means nothing in five years if it was produced
by whichever fairlearn happened to be resolved that afternoon and the definition
has since been revised. The metrics are thirty lines; the dependency is not worth
the ambiguity.

Three definitional decisions that a reader is entitled to disagree with, and can,
because they are here:

**Differences are max-min across groups, and therefore unsigned.** With two
groups this is the familiar signed gap in absolute value. With more it is the
worst pair. A regression is then unambiguously an *increase*, which is what makes
"total_delta > 0 means worse" a safe reading in :mod:`.decompose`.

**An undefined rate is NaN, never zero.** A group with no positive-label rows has
no true-positive rate. Substituting zero would manufacture the largest possible
disparity out of an absent denominator, and the resulting "regression" would be
attributable to nothing. Undefined groups are excluded from the max-min, and if
fewer than two groups survive, the difference itself is undefined.

**Small groups are measured but excluded from the difference.** A group of four
has a selection rate quantized to 0, 0.25, 0.5, 0.75, 1.0, and it will win the
max-min essentially at random. Since the point of this package is to attribute a
regression to a cause, and sampling noise has no cause, groups below
``min_group_n`` are recorded individually — with their ``n``, so the exclusion is
visible and reversible by anyone reading the table — but kept out of the
aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this, a group's rates are recorded but excluded from across-group
# differences. 30 is the conventional floor for a proportion estimate to be worth
# quoting; it is a default, not a law, and every caller can override it.
MIN_GROUP_N = 30

# Sensitive-attribute value used when the column is null. A null group is made
# explicit rather than dropped: the subjects whose demographic data is missing are
# usually not missing at random, and silently excluding them would remove exactly
# the population a fairness audit most needs to see.
MISSING_GROUP = "__missing__"

# Per-group metrics, written with ``group_value`` set.
GROUP_METRICS = (
    "selection_rate",
    "positive_label_rate",
    "true_positive_rate",
    "false_positive_rate",
    "accuracy",
)

# Across-group metrics, written with ``group_value`` null.
DIFFERENCE_METRICS = (
    "demographic_parity_difference",
    "equal_opportunity_difference",
    "false_positive_rate_difference",
    "equalized_odds_difference",
    "accuracy_difference",
)


@dataclass(frozen=True)
class GroupResult:
    """Every rate for one value of the sensitive attribute.

    ``n`` and ``n_positive_label`` are carried alongside the rates because a rate
    without its denominator cannot be audited: 1.0 from one row and 1.0 from four
    hundred are the same number and completely different evidence.
    """

    group_value: str
    n: int
    n_positive_label: int
    selection_rate: float
    positive_label_rate: float
    true_positive_rate: float
    false_positive_rate: float
    accuracy: float

    def as_dict(self) -> dict[str, float]:
        """The metrics only, keyed by name, with undefined ones omitted."""
        values = {name: getattr(self, name) for name in GROUP_METRICS}
        return {k: v for k, v in values.items() if _defined(v)}


def group_metrics(y_true, y_pred, sensitive) -> list[GroupResult]:
    """Rates per group, in sorted group order.

    The sort is not cosmetic. These become rows in an Iceberg table, and a row
    order that depended on set iteration would make two evaluations of the same
    model produce tables that differ without any metric differing.
    """
    y_true = _as_binary(y_true, "y_true")
    y_pred = _as_binary(y_pred, "y_pred")
    groups = np.asarray([_group_of(v) for v in sensitive], dtype=object)

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"sensitive={len(groups)}"
        )

    results = []
    for value in sorted(set(groups.tolist())):
        mask = groups == value
        results.append(_result_for(str(value), y_true[mask], y_pred[mask]))
    return results


def differences(
    groups: list[GroupResult], *, min_group_n: int = MIN_GROUP_N
) -> dict[str, float]:
    """Across-group disparities, computed only over groups large enough to trust.

    Returns only the metrics that are defined. A difference needs at least two
    groups with a defined rate, so a run where every group but one lacked
    positive labels yields no equal-opportunity entry rather than a zero.
    """
    eligible = [g for g in groups if g.n >= min_group_n]

    spread = {
        "demographic_parity_difference": _spread(eligible, "selection_rate"),
        "equal_opportunity_difference": _spread(eligible, "true_positive_rate"),
        "false_positive_rate_difference": _spread(eligible, "false_positive_rate"),
        "accuracy_difference": _spread(eligible, "accuracy"),
    }

    # Equalized odds is violated if *either* error rate differs across groups, so
    # it is the worse of the two rather than an average — an average would let a
    # large true-positive gap hide behind a small false-positive one.
    tpr, fpr = spread["equal_opportunity_difference"], spread["false_positive_rate_difference"]
    defined = [d for d in (tpr, fpr) if _defined(d)]
    if defined:
        spread["equalized_odds_difference"] = max(defined)

    return {k: v for k, v in spread.items() if _defined(v)}


def eligible_n(groups: list[GroupResult], *, min_group_n: int = MIN_GROUP_N) -> int:
    """Subjects actually behind a difference metric — the denominator to quote."""
    return sum(g.n for g in groups if g.n >= min_group_n)


# ---------------------------------------------------------------- internals ----

def _result_for(value: str, y_true: np.ndarray, y_pred: np.ndarray) -> GroupResult:
    n = int(y_true.size)
    positives = y_true == 1
    negatives = ~positives

    return GroupResult(
        group_value=value,
        n=n,
        n_positive_label=int(positives.sum()),
        selection_rate=_rate(int(y_pred.sum()), n),
        positive_label_rate=_rate(int(positives.sum()), n),
        true_positive_rate=_rate(int(y_pred[positives].sum()), int(positives.sum())),
        false_positive_rate=_rate(int(y_pred[negatives].sum()), int(negatives.sum())),
        accuracy=_rate(int((y_true == y_pred).sum()), n),
    )


def _spread(groups: list[GroupResult], metric: str) -> float:
    values = [v for v in (getattr(g, metric) for g in groups) if _defined(v)]
    if len(values) < 2:
        return math.nan
    return max(values) - min(values)


def _rate(numerator: int, denominator: int) -> float:
    """A rate with no denominator is undefined, not zero. See the module docstring."""
    return numerator / denominator if denominator else math.nan


def _defined(value: float) -> bool:
    return not math.isnan(value)


def _group_of(value) -> str:
    return MISSING_GROUP if value is None else str(value)


def _as_binary(values, name: str) -> np.ndarray:
    """Coerce to a 0/1 integer array, refusing anything that is not already binary.

    A silent cast is how a probability array ends up being treated as a decision
    array, which would make every selection rate wrong in a way no assertion
    downstream would catch.
    """
    array = np.asarray(values)
    if array.dtype == bool:
        return array.astype(np.int64)
    # Checked *before* the cast, not after. ``astype(int64)`` truncates 0.37 to 0,
    # so a post-cast check sees a clean binary array and passes — which is
    # precisely the mistake this is here to catch.
    extra = set(np.unique(array).tolist()) - {0, 1}
    if extra:
        raise ValueError(f"{name} must be binary 0/1; found {sorted(extra)[:5]}")
    return array.astype(np.int64, copy=False)
