"""Evaluating a registered model against the frozen eval snapshot.

These go through the real path — model loaded from MLflow under its Iceberg
digest, eval rows materialized from a pinned snapshot — because the parts worth
testing here are the couplings, not the arithmetic. The arithmetic is in
test_fairness_metrics.
"""

from __future__ import annotations

import pytest

from glassbox.fairness import (
    FairnessEvaluationConflict,
    SensitiveAttributeError,
    evaluate_fairness,
    get_evaluation,
)
from glassbox.fairness.evaluate import evaluation_id
from glassbox.ingest import ingest_adult
from glassbox.schemas import FAIRNESS_EVALUATIONS
from glassbox.snapshots import SnapshotTombstonedError
from glassbox.train import train_baseline
from glassbox.train.registry import ProvenanceIntegrityError

pytestmark = pytest.mark.slow

# The synthetic eval holdout is ~38 rows, so each sex group sits below the default
# min_group_n of 30 and no across-group difference is defined for it. That is the
# guard behaving correctly — see test_the_default_cutoff_refuses_this_eval_set —
# so tests that need a difference state a cutoff the fixture can actually meet.
SMALL = 5


@pytest.fixture
def trained(catalog, adult_file, gb_root):
    ingest_adult(gb_root)
    return train_baseline(root=gb_root)


def rows(catalog) -> list[dict]:
    return catalog.load_table(FAIRNESS_EVALUATIONS.identifier).scan().to_arrow().to_pylist()


# -------------------------------------------------------------- evaluating ----

def test_every_group_and_every_difference_is_recorded(catalog, trained):
    result = evaluate_fairness(
        catalog, trained.model_version.model_version_id, min_group_n=SMALL
    )

    assert {g.group_value for g in result.groups} == {"Male", "Female"}
    assert result.n > 0
    assert "demographic_parity_difference" in result.differences

    written = rows(catalog)
    groups_written = {r["group_value"] for r in written if r["group_value"]}
    assert groups_written == {"Male", "Female"}
    assert any(r["group_value"] is None for r in written), "no across-group rows"


def test_the_default_cutoff_refuses_this_eval_set(catalog, trained):
    """19 people per group is not enough to quote a disparity from, and the
    default says so rather than quoting one anyway."""
    result = evaluate_fairness(catalog, trained.model_version.model_version_id)

    assert result.groups, "the groups are still measured"
    assert result.differences == {}
    assert all(r["group_value"] is not None for r in rows(catalog))


def test_a_difference_row_carries_no_group_and_a_group_row_carries_no_difference(
    catalog, trained
):
    """The null group_value is what distinguishes the two kinds of claim."""
    evaluate_fairness(catalog, trained.model_version.model_version_id, min_group_n=SMALL)

    for row in rows(catalog):
        if row["group_value"] is None:
            assert row["metric_name"].endswith("_difference")
            assert row["min_group_n"] == SMALL, "the cutoff defines the difference"
        else:
            assert not row["metric_name"].endswith("_difference")
            assert row["min_group_n"] is None, "a group's own rate has no cutoff"


def test_two_cutoffs_are_two_claims_rather_than_a_conflict(catalog, trained):
    """The cutoff is part of what a difference metric means, so changing it
    produces another row rather than contradicting the first."""
    mv = trained.model_version.model_version_id
    evaluate_fairness(catalog, mv, min_group_n=SMALL)
    evaluate_fairness(catalog, mv, min_group_n=SMALL + 1)

    cutoffs = {r["min_group_n"] for r in rows(catalog) if r["group_value"] is None}
    assert cutoffs == {SMALL, SMALL + 1}


def test_the_threshold_is_recorded_because_it_defines_the_metric(catalog, trained):
    """'selection_rate = 0.31' is not checkable without saying 'at or above what'."""
    mv = trained.model_version.model_version_id
    lenient = evaluate_fairness(catalog, mv, threshold=0.2, min_group_n=SMALL)
    strict = evaluate_fairness(catalog, mv, threshold=0.8, min_group_n=SMALL)

    assert {r["decision_threshold"] for r in rows(catalog)} == {0.2, 0.8}

    selected_at = {
        t: sum(g.selection_rate * g.n for g in r.groups)
        for t, r in ((0.2, lenient), (0.8, strict))
    }
    assert selected_at[0.2] > selected_at[0.8], "a higher bar must approve fewer people"


def test_two_thresholds_are_two_evaluations_not_a_conflict(catalog, trained):
    """They are different claims about the same model, and both stay on the record."""
    mv = trained.model_version.model_version_id
    snapshot = trained.model_version.eval_snapshot_uuid
    evaluate_fairness(catalog, mv, threshold=0.5, min_group_n=SMALL)
    evaluate_fairness(catalog, mv, threshold=0.6, min_group_n=SMALL)

    assert {r["decision_threshold"] for r in rows(catalog)} == {0.5, 0.6}

    at_half = get_evaluation(
        catalog, mv, eval_snapshot_uuid=snapshot, threshold=0.5, min_group_n=SMALL
    )
    at_six = get_evaluation(
        catalog, mv, eval_snapshot_uuid=snapshot, threshold=0.6, min_group_n=SMALL
    )
    assert at_half and at_six


def test_the_group_counts_add_up_to_the_eval_set(catalog, trained):
    result = evaluate_fairness(catalog, trained.model_version.model_version_id)

    assert result.n == trained.eval_rows


def test_a_sensitive_attribute_the_eval_set_does_not_have_is_refused(catalog, trained):
    with pytest.raises(SensitiveAttributeError, match="favourite_colour"):
        evaluate_fairness(
            catalog,
            trained.model_version.model_version_id,
            sensitive_attribute="favourite_colour",
        )


def test_race_works_as_well_as_sex(catalog, trained):
    """Nothing in the metric code is specific to the default attribute."""
    result = evaluate_fairness(
        catalog, trained.model_version.model_version_id, sensitive_attribute="race"
    )

    assert len(result.groups) >= 2
    assert all(r["sensitive_attribute"] == "race" for r in rows(catalog))


# ------------------------------------------------------------- idempotence ----

def test_re_evaluating_does_not_duplicate_rows(catalog, trained):
    mv = trained.model_version.model_version_id
    evaluate_fairness(catalog, mv)
    before = len(rows(catalog))

    evaluate_fairness(catalog, mv)

    assert len(rows(catalog)) == before


def test_the_same_evaluation_gets_the_same_row_identity_on_any_machine(catalog, trained):
    """UUID5 over everything that determines the value — not a fresh uuid4."""
    args = ("mv-1", "snap-1", "sex", 0.5, "selection_rate", "Female")

    assert evaluation_id(*args) == evaluation_id(*args)
    assert evaluation_id(*args) != evaluation_id("mv-2", *args[1:])
    assert evaluation_id(*args) != evaluation_id(*args[:3], 0.6, *args[4:])

    # The cutoff separates difference rows, and leaves per-group rows alone.
    difference = ("mv-1", "snap-1", "sex", 0.5, "demographic_parity_difference", None)
    assert evaluation_id(*difference, 30) != evaluation_id(*difference, 5)
    assert evaluation_id(*args, 30) == evaluation_id(*args, 30)


def test_the_same_model_on_the_same_data_giving_a_different_number_is_reported(
    catalog, trained, monkeypatch
):
    """Not overwritten. Either the evaluation is nondeterministic or the frozen
    eval set moved, and both are findings worth more than a tidy table."""
    mv = trained.model_version.model_version_id
    evaluate_fairness(catalog, mv, min_group_n=SMALL)

    import glassbox.fairness.evaluate as module

    real = module.differences

    def drifted(groups, **kwargs):
        return {k: v + 0.1 for k, v in real(groups, **kwargs).items()}

    monkeypatch.setattr(module, "differences", drifted)

    with pytest.raises(FairnessEvaluationConflict, match="already"):
        evaluate_fairness(catalog, mv, min_group_n=SMALL)


# ------------------------------------------------------- failing closed ----

def test_a_model_with_no_provenance_record_cannot_be_evaluated(catalog, trained):
    """A fairness number from an unverified artifact is a number about some model."""
    with pytest.raises(ProvenanceIntegrityError):
        evaluate_fairness(catalog, "not-a-real-model-version")


def test_an_erased_eval_snapshot_refuses_rather_than_reporting_a_stale_number(
    catalog, trained
):
    """The tombstone is the designed answer, not a crash."""
    from pyiceberg.expressions import EqualTo

    from glassbox.schemas import DATA_SNAPSHOTS

    snapshot = trained.model_version.eval_snapshot_uuid
    table = catalog.load_table(DATA_SNAPSHOTS.identifier)
    stored = table.scan(row_filter=EqualTo("data_snapshot_uuid", snapshot)).to_arrow().to_pylist()
    table.delete(EqualTo("data_snapshot_uuid", snapshot))

    from glassbox.writer import append_records

    append_records(catalog, DATA_SNAPSHOTS, [{**stored[0], "status": "tombstoned"}])

    with pytest.raises(SnapshotTombstonedError):
        evaluate_fairness(catalog, trained.model_version.model_version_id)
