"""Capability #2, part one — are the attributions actually attributions?

The claim a right-to-explanation endpoint makes is stronger than "here are some
numbers per feature". It is: *these values decompose this decision*. That is a
checkable property (additivity), and these tests check it rather than asserting
that the code ran.

The dispatch tests matter for a different reason. The refusal to explain an
unrecognized model is a design commitment — no KernelExplainer fallback — and a
commitment that nothing tests is a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.explain import (
    AdditivityError,
    PredictionExplainer,
    UnexplainableModelError,
    build_background,
    explainer_for,
    link_of,
    logit,
    reconciles,
)
from glassbox.ingest import ingest_adult
from glassbox.schemas import CREDIT_APPLICATIONS
from glassbox.snapshots import get_snapshot, materialize
from glassbox.train import features as F
from glassbox.train import train_baseline
from glassbox.train.registry import load_model_checked


@pytest.fixture(scope="session")
def fitted(adult_table):
    """A fitted baseline pipeline and the raw frame it was fit on.

    Fit directly rather than through ``train_baseline``, because everything below
    this line tests explanation arithmetic, which does not become more true for
    having been routed through Iceberg and MLflow first. The one test that is
    genuinely about the served path builds its own model the slow way.
    """
    from sklearn.linear_model import LogisticRegression

    from glassbox.train.baseline import BASELINE_HYPERPARAMS

    estimator = LogisticRegression(random_state=42, **BASELINE_HYPERPARAMS)
    frame = F.to_frame(adult_table)
    model = F.build_pipeline(adult_table, estimator)
    model.fit(frame, F.target_of(adult_table))
    return model, frame


@pytest.fixture(scope="session")
def served(fitted):
    """A model plus its explainer, held together as the serving path holds them."""
    model, frame = fitted
    return model, PredictionExplainer(model, build_background(frame, n=100)), frame


# --------------------------------------------------------------- additivity ----

def test_attributions_sum_to_the_score_they_explain(served):
    """The property that makes this an explanation and not a ranked list."""
    model, explainer, frame = served

    for i in (0, 1, 7, 42):
        row = frame.iloc[[i]]
        score = float(model.predict_proba(row)[0, 1])
        explanation = explainer.explain(row, score=score)

        total = explanation.base_value + sum(a.shap_value for a in explanation.attributions)
        assert total == pytest.approx(float(logit(score)), abs=1e-9)


def test_a_truncated_view_still_reconciles_via_the_residual(served):
    """A top-k explanation that does not add up is not a top-k explanation."""
    model, explainer, frame = served
    row = frame.iloc[[0]]
    score = float(model.predict_proba(row)[0, 1])
    explanation = explainer.explain(row, score=score)

    for k in (1, 5, 10, len(explanation.attributions)):
        assert reconciles(explanation, score, k), f"top-{k} view failed to reconcile"


def test_reconciling_against_probability_instead_of_log_odds_fails(served):
    """Guards the guard: proves the additivity check can actually fail.

    If the check passed against both spaces it would be measuring nothing, and
    every other test in this file would be vacuous.
    """
    model, explainer, frame = served
    row = frame.iloc[[0]]
    score = float(model.predict_proba(row)[0, 1])
    explanation = explainer.explain(row, score=score)

    total = explanation.base_value + sum(a.shap_value for a in explanation.attributions)
    assert abs(total - score) > 1e-3, (
        "log-odds and probability totals are indistinguishable for this row, so "
        "this row cannot demonstrate that the link function matters"
    )


def test_explanation_refuses_to_report_a_decomposition_that_does_not_add_up(served, monkeypatch):
    """Fail closed when SHAP's output space is not what link_of declared."""
    model, explainer, frame = served
    row = frame.iloc[[0]]

    original = explainer._shap_values
    monkeypatch.setattr(
        explainer, "_shap_values", lambda enc: (original(enc)[0] * 0.5, original(enc)[1])
    )

    with pytest.raises(AdditivityError, match="do not reconcile"):
        explainer.explain(row)


# ----------------------------------------------------------------- ranking ----

def test_attributions_are_ranked_by_absolute_contribution(served):
    _, explainer, frame = served
    explanation = explainer.explain(frame.iloc[[3]])

    magnitudes = [abs(a.shap_value) for a in explanation.attributions]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert [a.abs_rank for a in explanation.attributions] == list(
        range(1, len(explanation.attributions) + 1)
    )


def test_the_same_row_explains_identically_twice(served):
    """Two audit reads of one decision must not disagree, including on rank order."""
    _, explainer, frame = served
    row = frame.iloc[[5]]

    first = explainer.explain(row)
    second = explainer.explain(row)

    assert first.base_value == second.base_value
    assert [(a.feature_name, a.shap_value, a.abs_rank) for a in first.attributions] == [
        (a.feature_name, a.shap_value, a.abs_rank) for a in second.attributions
    ]


def test_residual_carries_exactly_the_untruncated_tail(served):
    _, explainer, frame = served
    explanation = explainer.explain(frame.iloc[[0]])

    tail = sum(a.shap_value for a in explanation.attributions[10:])
    assert explanation.residual(10) == pytest.approx(tail)
    assert explanation.residual(len(explanation.attributions)) == 0.0


# ------------------------------------------------------- source values ----

def test_encoded_columns_report_the_raw_value_a_subject_can_act_on(served):
    """``occupation was Tech-support``, not ``cat__occupation_Tech-support = 1.0``."""
    _, explainer, frame = served
    row = frame.iloc[[0]]
    explanation = explainer.explain(row)
    raw = row.iloc[0].to_dict()

    by_name = {a.feature_name: a for a in explanation.attributions}

    numeric = by_name["num__age"]
    assert numeric.feature_value_str == str(raw["age"])

    categorical = next(a for n, a in by_name.items() if n.startswith("cat__occupation_"))
    assert categorical.feature_value_str == str(raw["occupation"])
    # Every one-hot column of a feature reports that subject's actual value, not
    # the value the column encodes — the audit answer is about the person.
    occupation_columns = [a for n, a in by_name.items() if n.startswith("cat__occupation_")]
    assert {a.feature_value_str for a in occupation_columns} == {str(raw["occupation"])}


# ---------------------------------------------------------------- dispatch ----

def test_an_unrecognized_estimator_is_refused_rather_than_approximated(served):
    """No KernelExplainer fallback. The omission is the design."""
    from sklearn.neural_network import MLPClassifier

    with pytest.raises(UnexplainableModelError, match="no exact SHAP explainer"):
        explainer_for(MLPClassifier())


def test_dispatch_recognizes_the_baseline_as_exactly_explainable(served):
    model, _, _ = served
    _, explainer_type = explainer_for(model)
    assert explainer_type == "linear"
    assert link_of(model, explainer_type) == "logit"


def test_a_forest_is_reconciled_in_probability_space_not_log_odds():
    """A forest's raw output is already a probability; assuming log-odds breaks it."""
    from sklearn.ensemble import RandomForestClassifier

    forest = RandomForestClassifier(n_estimators=3, random_state=0)
    forest.fit(np.zeros((4, 2)), [0, 1, 0, 1])

    _, explainer_type = explainer_for(forest)
    assert explainer_type == "tree"
    assert link_of(forest, explainer_type) == "identity"


# -------------------------------------------------------------- background ----

def test_background_is_deterministic_across_loads(served):
    """Otherwise every restart silently moves the base value the audit reports."""
    _, _, frame = served

    assert build_background(frame, n=50).equals(build_background(frame, n=50))
    assert len(build_background(frame, n=50)) == 50
    assert build_background(frame.iloc[:10], n=50).equals(frame.iloc[:10].reset_index(drop=True))


# -------------------------------------------------------------- integration ----

@pytest.mark.slow
def test_a_provenance_loaded_model_explains_its_own_predictions(catalog, adult_file, gb_root):
    """The whole serving path: Iceberg record -> verified MLflow artifact -> explanation.

    Distinct from the fixtures above, which fit in-process. What this adds is that
    the model whose digest Iceberg vouches for, and the background drawn from the
    snapshot that model was trained on, together still produce attributions that
    reconcile — i.e. the provenance round trip does not perturb the arithmetic.
    """
    ingest_adult(gb_root)
    trained = train_baseline(root=gb_root)

    model, record = load_model_checked(catalog, trained.model_version.model_version_id, gb_root)
    snapshot = get_snapshot(catalog, record["data_snapshot_uuid"])
    frame = F.to_frame(materialize(catalog, snapshot, CREDIT_APPLICATIONS))

    explainer = PredictionExplainer(model, build_background(frame, n=100))
    row = frame.iloc[[0]]
    score = float(model.predict_proba(row)[0, 1])
    explanation = explainer.explain(row, score=score)

    assert explanation.explainer_type == "linear"
    assert reconciles(explanation, score, 10)
