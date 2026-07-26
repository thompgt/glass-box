"""Capability #3: attributing a fairness change to the model or to the data.

The central claim under test is that the two effects sum to the observed change.
Everything else here exists to pin the two degenerate cases — only the config
moved, only the data moved — where the right answer is known in advance and any
attribution to the other cause is unambiguously wrong.

These are slow because they are honest: each decomposition trains real
counterfactual models and registers them, which is the thing that makes the
result checkable by someone else later.
"""

from __future__ import annotations

import pytest

from glassbox.fairness import (
    UncomparableModelsError,
    decompose_regression,
    get_evaluation,
)
from glassbox.fairness.decompose import (
    COUNTERFACTUAL_STATUS,
    METHOD,
    TRAINERS,
    decomposition_id,
)
from glassbox.ingest import ingest_adult
from glassbox.schemas import FAIRNESS_DECOMPOSITIONS
from glassbox.train import train_baseline
from glassbox.train.registry import get_model_version

pytestmark = pytest.mark.slow

METRIC = "demographic_parity_difference"

# The frozen eval holdout here is ~38 rows, so the default min_group_n of 30
# leaves no two groups eligible and nothing to decompose. The cutoff is stated
# rather than defaulted; see test_fairness_evaluate for the guard itself.
SMALL = 5

# A second data batch in which sex determines the label outright. The point is not
# realism; it is that the data change has an unmistakable direction, so an
# attribution that lands on the model instead is visibly wrong rather than
# arguably wrong.
def _skewed_rows(n: int = 400) -> list[str]:
    rows = []
    for i in range(n):
        sex = "Male" if i % 2 else "Female"
        income = ">50K" if sex == "Male" else "<=50K"
        rows.append(
            f"45, Private, {200000 + i}, HS-grad, 9, Married-civ-spouse, "
            f"Craft-repair, Husband, White, {sex}, 0, 0, 40, United-States, {income}"
        )
    return rows


def _append_skewed_data(gb_root) -> None:
    """Add the skewed batch and re-ingest, producing a new training data version.

    The eval holdout is frozen at first ingest, so this moves the training data
    without moving the yardstick — which is the only way the comparison means
    anything.
    """
    path = gb_root / "data" / "raw" / "adult.data"
    path.write_text(
        path.read_text(encoding="utf-8") + "\n".join(_skewed_rows()) + "\n", encoding="utf-8"
    )
    ingest_adult(gb_root)


def decompositions(catalog) -> list[dict]:
    return catalog.load_table(FAIRNESS_DECOMPOSITIONS.identifier).scan().to_arrow().to_pylist()


@pytest.fixture
def data_change(catalog, adult_file, gb_root):
    """Same configuration, different training data."""
    ingest_adult(gb_root)
    a = train_baseline(root=gb_root)
    _append_skewed_data(gb_root)
    b = train_baseline(root=gb_root)
    return a.model_version.model_version_id, b.model_version.model_version_id


@pytest.fixture
def config_change(catalog, adult_file, gb_root):
    """Same training data, different configuration."""
    ingest_adult(gb_root)
    a = train_baseline(root=gb_root)
    b = train_baseline(root=gb_root, hyperparams={"C": 0.005, "max_iter": 1000,
                                                  "solver": "lbfgs", "tol": 1e-6})
    return a.model_version.model_version_id, b.model_version.model_version_id


# ------------------------------------------------------------ the identity ----

def test_the_two_effects_sum_to_the_observed_change(catalog, data_change):
    """A decomposition whose parts do not sum to the whole is not a decomposition."""
    analysis = decompose_regression(catalog, *data_change, min_group_n=SMALL)

    assert analysis.decompositions, "nothing was decomposed"
    for d in analysis.decompositions:
        assert d.model_effect + d.data_effect == pytest.approx(d.total_delta, abs=1e-12)


def test_the_total_is_the_difference_between_the_two_measured_metrics(catalog, data_change):
    """total_delta is not a fifth number; it is M(B) - M(A) and must equal it."""
    analysis = decompose_regression(catalog, *data_change, min_group_n=SMALL)
    d = analysis.for_metric(METRIC)

    assert d.total_delta == pytest.approx(d.measured["B"] - d.measured["A"])
    assert d.measured["A"] == pytest.approx(analysis.evaluations["A"].metric(METRIC))


def test_swapping_baseline_and_candidate_negates_every_effect(catalog, data_change):
    """The attribution is symmetric, which is what averaging both paths buys."""
    a, b = data_change
    forward = decompose_regression(catalog, a, b, min_group_n=SMALL).for_metric(METRIC)
    backward = decompose_regression(catalog, b, a, min_group_n=SMALL).for_metric(METRIC)

    assert backward.total_delta == pytest.approx(-forward.total_delta)
    assert backward.model_effect == pytest.approx(-forward.model_effect)
    assert backward.data_effect == pytest.approx(-forward.data_effect)


# ------------------------------------------------------ the degenerate cases ----

def test_a_pure_data_change_is_attributed_entirely_to_the_data(catalog, data_change):
    analysis = decompose_regression(catalog, *data_change, min_group_n=SMALL)
    d = analysis.for_metric(METRIC)

    assert d.model_effect == pytest.approx(0.0, abs=1e-12)
    assert d.data_effect == pytest.approx(d.total_delta)
    assert d.attributed_to == "data"


def test_a_pure_data_change_trains_no_counterfactuals_at_all(catalog, data_change):
    """A' is the candidate and B' is the baseline. Retraining them would be waste,
    and reusing them is sound only because training is bit-exact."""
    a, b = data_change
    analysis = decompose_regression(catalog, a, b, min_group_n=SMALL)

    assert analysis.counterfactual_a_prime_id == b
    assert analysis.counterfactual_b_prime_id == a
    assert analysis.trained_counterfactuals == 0


def test_a_pure_config_change_is_attributed_entirely_to_the_model(catalog, config_change):
    analysis = decompose_regression(catalog, *config_change, min_group_n=SMALL)
    d = analysis.for_metric(METRIC)

    assert d.data_effect == pytest.approx(0.0, abs=1e-12)
    assert d.model_effect == pytest.approx(d.total_delta)


# ------------------------------------------------- both factors at once ----

def test_a_mixed_change_trains_the_two_corners_that_did_not_exist(catalog, adult_file, gb_root):
    """Config and data both moved, so neither counterfactual is already on record."""
    ingest_adult(gb_root)
    a = train_baseline(root=gb_root).model_version.model_version_id
    _append_skewed_data(gb_root)
    b = train_baseline(
        root=gb_root,
        hyperparams={"C": 0.005, "max_iter": 1000, "solver": "lbfgs", "tol": 1e-6},
    ).model_version.model_version_id

    analysis = decompose_regression(catalog, a, b, min_group_n=SMALL)

    assert analysis.trained_counterfactuals == 2
    assert len(set(analysis.model_version_ids)) == 4
    d = analysis.for_metric(METRIC)
    assert d.model_effect + d.data_effect == pytest.approx(d.total_delta, abs=1e-12)


def test_a_counterfactual_is_registered_and_marked_as_one(catalog, adult_file, gb_root):
    """It is a real model version so the arithmetic can be re-checked, and it is
    flagged so nothing serves it by accident."""
    ingest_adult(gb_root)
    a = train_baseline(root=gb_root).model_version.model_version_id
    _append_skewed_data(gb_root)
    b = train_baseline(
        root=gb_root,
        hyperparams={"C": 0.005, "max_iter": 1000, "solver": "lbfgs", "tol": 1e-6},
    ).model_version.model_version_id

    analysis = decompose_regression(catalog, a, b, min_group_n=SMALL)
    record = get_model_version(catalog, analysis.counterfactual_a_prime_id)

    assert record["status"] == COUNTERFACTUAL_STATUS
    assert record["data_snapshot_uuid"] == get_model_version(catalog, b)["data_snapshot_uuid"]
    assert record["hyperparams_json"] == get_model_version(catalog, a)["hyperparams_json"]


# --------------------------------------------------------------- recording ----

def test_the_decomposition_cites_four_model_versions(catalog, data_change):
    """Cited rather than described, so a reader can reload each and re-derive it."""
    decompose_regression(catalog, *data_change, min_group_n=SMALL)

    row = next(r for r in decompositions(catalog) if r["metric_name"] == METRIC)
    for column in (
        "baseline_model_version_id",
        "candidate_model_version_id",
        "counterfactual_a_prime_id",
        "counterfactual_b_prime_id",
    ):
        assert get_model_version(catalog, row[column]) is not None
    assert row["method"] == METHOD
    assert row["comparable"] is True


def test_the_recorded_effects_are_re_derivable_from_the_evaluation_rows(catalog, data_change):
    """The decomposition duplicates nothing: it is a function of four audit rows."""
    analysis = decompose_regression(catalog, *data_change, min_group_n=SMALL)
    row = next(r for r in decompositions(catalog) if r["metric_name"] == METRIC)

    measured = {}
    for column, role in (
        ("baseline_model_version_id", "A"),
        ("candidate_model_version_id", "B"),
        ("counterfactual_a_prime_id", "A_prime"),
        ("counterfactual_b_prime_id", "B_prime"),
    ):
        recorded = get_evaluation(
            catalog,
            row[column],
            eval_snapshot_uuid=row["eval_snapshot_uuid"],
            threshold=row["decision_threshold"],
            min_group_n=row["min_group_n"],
        )
        measured[role] = recorded[METRIC]

    assert row["total_delta"] == pytest.approx(measured["B"] - measured["A"])
    assert row["model_effect"] == pytest.approx(
        ((measured["B_prime"] - measured["A"]) + (measured["B"] - measured["A_prime"])) / 2
    )
    assert analysis.for_metric(METRIC).measured == pytest.approx(measured)


def test_re_running_the_same_decomposition_does_not_duplicate_rows(catalog, data_change):
    decompose_regression(catalog, *data_change, min_group_n=SMALL)
    before = len(decompositions(catalog))

    decompose_regression(catalog, *data_change, min_group_n=SMALL)

    assert len(decompositions(catalog)) == before


def test_the_decomposition_id_is_derived_not_random(catalog):
    """Two runs must produce the same row identity or idempotence is impossible."""
    args = ("mv-a", "mv-b", "snap", "sex", METRIC, 0.5, 30)

    assert decomposition_id(*args) == decomposition_id(*args)
    assert decomposition_id(*args) != decomposition_id(*args[:-1], 5)
    assert decomposition_id(*args) != decomposition_id("mv-b", "mv-a", *args[2:])


# ---------------------------------------------------------- failing closed ----

def test_comparing_a_version_to_itself_is_refused(catalog, data_change):
    """The result would be a table of zeros claiming an analysis had happened."""
    a, _ = data_change
    with pytest.raises(UncomparableModelsError, match="same model version"):
        decompose_regression(catalog, a, a, min_group_n=SMALL)


def test_two_models_measured_on_different_eval_sets_are_refused(catalog, data_change,
                                                                monkeypatch):
    """Otherwise the eval-set difference lands in total_delta and gets attributed
    to the model or the data — a confident wrong answer."""
    a, b = data_change
    import glassbox.fairness.decompose as module

    real = module.get_model_version

    def shifted(catalog_, mvid):
        record = real(catalog_, mvid)
        if record and mvid == b:
            return {**record, "eval_snapshot_uuid": "some-other-eval-snapshot"}
        return record

    monkeypatch.setattr(module, "get_model_version", shifted)

    with pytest.raises(UncomparableModelsError, match="eval"):
        decompose_regression(catalog, a, b, min_group_n=SMALL)


def test_an_unknown_model_version_is_refused(catalog, data_change):
    a, _ = data_change
    with pytest.raises(UncomparableModelsError, match="no audit.model_versions row"):
        decompose_regression(catalog, a, "not-a-real-model-version", min_group_n=SMALL)


def test_a_recipe_with_no_counterfactual_trainer_is_refused(catalog, data_change, monkeypatch):
    """The decomposition has to retrain each config on the other's data."""
    monkeypatch.setitem(TRAINERS, "baseline-logreg", None)
    monkeypatch.delitem(TRAINERS, "baseline-logreg")

    with pytest.raises(UncomparableModelsError, match="no counterfactual trainer"):
        decompose_regression(catalog, *data_change, min_group_n=SMALL)
