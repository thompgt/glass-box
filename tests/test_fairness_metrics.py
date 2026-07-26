"""The metric definitions, on inputs whose right answer is arithmetic.

Everything here is hand-computed. These metrics are the foundation the whole
decomposition sits on — if ``demographic_parity_difference`` is subtly wrong,
the model/data attribution built on it is confidently wrong, which is worse than
absent. So the tests state the expected number rather than recomputing it with
the same code under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from glassbox.fairness.metrics import (
    MISSING_GROUP,
    differences,
    eligible_n,
    group_metrics,
)


def by_value(groups) -> dict:
    return {g.group_value: g for g in groups}


# --------------------------------------------------------------- per group ----

def test_rates_are_computed_within_each_group():
    #            group:  A  A  A  A   B  B  B  B
    y_true = np.array([1, 1, 0, 0,  1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0,  1, 1, 0, 0])
    sensitive = ["A"] * 4 + ["B"] * 4

    g = by_value(group_metrics(y_true, y_pred, sensitive))

    assert g["A"].n == 4
    assert g["A"].selection_rate == 0.5  # 2 of 4 predicted positive
    assert g["A"].true_positive_rate == 0.5  # 1 of 2 actual positives caught
    assert g["A"].false_positive_rate == 0.5  # 1 of 2 actual negatives flagged
    assert g["A"].accuracy == 0.5

    assert g["B"].selection_rate == 0.5
    assert g["B"].true_positive_rate == 1.0
    assert g["B"].false_positive_rate == 0.0
    assert g["B"].accuracy == 1.0


def test_the_positive_label_rate_describes_the_data_not_the_model():
    """Base rates are what make demographic parity and equalized odds conflict."""
    y_true = np.array([1, 1, 1, 0])
    sensitive = ["A"] * 4

    g = by_value(group_metrics(y_true, np.zeros(4), sensitive))

    assert g["A"].positive_label_rate == 0.75
    assert g["A"].n_positive_label == 3
    assert g["A"].selection_rate == 0.0


def test_groups_come_back_in_sorted_order():
    """These become audit rows; set iteration order must not reach the table."""
    groups = group_metrics([1, 0, 1], [1, 0, 1], ["Zeta", "Alpha", "Mu"])

    assert [g.group_value for g in groups] == ["Alpha", "Mu", "Zeta"]


def test_a_null_group_is_named_rather_than_dropped():
    """Missing demographic data is not missing at random; excluding it hides people."""
    groups = by_value(group_metrics([1, 0, 1], [1, 0, 0], ["A", None, None]))

    assert set(groups) == {"A", MISSING_GROUP}
    assert groups[MISSING_GROUP].n == 2


# ------------------------------------------------------- undefined vs zero ----

def test_a_group_with_no_positive_labels_has_no_true_positive_rate():
    """NaN, not 0.0 — a rate with an empty denominator is absent, not perfect."""
    groups = by_value(group_metrics([0, 0, 0], [1, 0, 0], ["A", "A", "A"]))

    assert math.isnan(groups["A"].true_positive_rate)
    assert groups["A"].false_positive_rate == pytest.approx(1 / 3)
    assert "true_positive_rate" not in groups["A"].as_dict()


def test_an_undefined_rate_is_left_out_of_the_difference_not_treated_as_zero():
    """Otherwise an absent denominator manufactures a maximal, causeless disparity."""
    # Group B has no positive labels at all, so it has no TPR to compare.
    y_true = [1] * 30 + [0] * 30 + [0] * 40
    y_pred = [1] * 30 + [0] * 30 + [0] * 40
    sensitive = ["A"] * 60 + ["B"] * 40

    groups = group_metrics(y_true, y_pred, sensitive)
    result = differences(groups)

    assert "equal_opportunity_difference" not in result
    assert result["demographic_parity_difference"] == pytest.approx(0.5)


def test_a_single_eligible_group_yields_no_differences_at_all():
    """A disparity between one group and nothing is not zero, it is unanswerable."""
    groups = group_metrics([1] * 40, [1] * 40, ["A"] * 40)

    assert differences(groups) == {}


# ------------------------------------------------------------- differences ----

def test_demographic_parity_is_the_spread_in_selection_rate():
    y_pred = [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30
    y_true = [1] * 80
    sensitive = ["A"] * 40 + ["B"] * 40

    result = differences(group_metrics(y_true, y_pred, sensitive))

    # A selects 30/40 = 0.75, B selects 10/40 = 0.25.
    assert result["demographic_parity_difference"] == pytest.approx(0.5)


def test_the_difference_is_unsigned_so_a_regression_is_always_an_increase():
    """decompose reads 'total_delta > 0' as 'worse'; that requires max-min."""
    y_true = [1] * 80
    favoured_a = [1] * 30 + [0] * 10 + [1] * 10 + [0] * 30
    favoured_b = [1] * 10 + [0] * 30 + [1] * 30 + [0] * 10
    sensitive = ["A"] * 40 + ["B"] * 40

    a = differences(group_metrics(y_true, favoured_a, sensitive))
    b = differences(group_metrics(y_true, favoured_b, sensitive))

    assert a["demographic_parity_difference"] == b["demographic_parity_difference"] > 0


def test_equalized_odds_takes_the_worse_error_rate_not_the_average():
    """An average would let a large true-positive gap hide behind a small false one."""
    #                     A: 20 pos, 20 neg          B: 20 pos, 20 neg
    y_true = [1] * 20 + [0] * 20 + [1] * 20 + [0] * 20
    # A catches every positive; B catches none. Both flag negatives identically.
    y_pred = [1] * 20 + [0] * 20 + [0] * 20 + [0] * 20
    sensitive = ["A"] * 40 + ["B"] * 40

    result = differences(group_metrics(y_true, y_pred, sensitive))

    assert result["equal_opportunity_difference"] == pytest.approx(1.0)
    assert result["false_positive_rate_difference"] == pytest.approx(0.0)
    assert result["equalized_odds_difference"] == pytest.approx(1.0)


def test_a_group_too_small_to_measure_is_recorded_but_not_compared():
    """A group of four has a selection rate quantized to quarters — that is noise.

    A and C are large and 0.1 apart. B is four people at 100%, which would
    otherwise take the spread to 0.9 on the strength of four coin flips.
    """
    y_true = [1] * 204
    y_pred = [1] * 50 + [0] * 50 + [1] * 4 + [1] * 60 + [0] * 40
    sensitive = ["A"] * 100 + ["B"] * 4 + ["C"] * 100

    groups = group_metrics(y_true, y_pred, sensitive)
    result = differences(groups, min_group_n=30)

    assert by_value(groups)["B"].selection_rate == 1.0, "the small group is still measured"
    assert result["demographic_parity_difference"] == pytest.approx(0.1)
    assert eligible_n(groups, min_group_n=30) == 200


def test_lowering_the_threshold_lets_the_small_group_back_in():
    """The exclusion is a stated parameter, not a hidden policy."""
    y_true = [1] * 204
    y_pred = [1] * 50 + [0] * 50 + [1] * 4 + [1] * 60 + [0] * 40
    sensitive = ["A"] * 100 + ["B"] * 4 + ["C"] * 100

    result = differences(group_metrics(y_true, y_pred, sensitive), min_group_n=4)

    assert result["demographic_parity_difference"] == pytest.approx(0.5)


# ------------------------------------------------------------- guardrails ----

def test_probabilities_are_refused_rather_than_truncated_to_zero():
    """int(0.87) == 0 would silently make every selection rate wrong."""
    with pytest.raises(ValueError, match="binary"):
        group_metrics([1, 0, 1], [0.87, 0.12, 0.55], ["A", "A", "A"])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="length mismatch"):
        group_metrics([1, 0], [1, 0, 1], ["A", "A", "A"])


def test_boolean_predictions_are_accepted():
    """numpy comparison operators produce these; rejecting them would be pedantry."""
    groups = by_value(group_metrics([1, 0], np.array([True, False]), ["A", "A"]))

    assert groups["A"].accuracy == 1.0
