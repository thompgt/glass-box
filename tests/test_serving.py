"""Capability #2 end to end — a decision, then the answer to "why?".

These tests take the long way round on purpose. Everything in test_explanations
fits in one process and checks arithmetic; here the model is loaded from MLflow
under its Iceberg digest, the decision goes through the spool onto disk, into
Iceberg, and comes back out through the audit tables only. Nothing in the read
path touches the model.

That separation is the whole claim. If the explanation endpoint could reach the
model, it would be recomputing an attribution rather than reporting the one that
was recorded — and the two differ the moment the model is retrained.
"""

from __future__ import annotations

import datetime as dt

import pytest

from glassbox.explain.dispatch import logit
from glassbox.ingest import ingest_adult
from glassbox.schemas import ATTRIBUTIONS, MODEL_VERSIONS, PREDICTIONS
from glassbox.serve import (
    ExplanationNotFoundError,
    PredictionService,
    UnknownFeatureError,
    explanation_for,
)
from glassbox.train import train_baseline
from glassbox.train.registry import ProvenanceIntegrityError
from glassbox.writer import append_records

pytestmark = pytest.mark.slow

APPLICANT = {
    "age": 39,
    "workclass": "Private",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}


@pytest.fixture
def service(catalog, adult_file, gb_root):
    ingest_adult(gb_root)
    trained = train_baseline(root=gb_root)
    return PredictionService.load(
        catalog,
        trained.model_version.model_version_id,
        gb_root,
        background_rows=100,
    )


# ------------------------------------------------------------- scoring ----

def test_a_prediction_is_recorded_before_it_is_returned(service):
    """The spool holds it the moment predict() returns, before any Iceberg commit."""
    outcome = service.predict(APPLICANT, subject_id="subject-1")

    assert outcome.decision in ("approve", "deny")
    assert 0.0 <= outcome.score <= 1.0
    assert service.spool.pending_count() == 1
    assert service.catalog.load_table(PREDICTIONS.identifier).scan().to_arrow().num_rows == 0


def test_the_decision_follows_the_recorded_threshold(service):
    outcome = service.predict(APPLICANT)

    expected = "approve" if outcome.score >= outcome.threshold else "deny"
    assert outcome.decision == expected
    assert outcome.threshold == service.threshold


def test_an_unknown_feature_is_refused_rather_than_ignored(service):
    """Dropping it silently would record 'considered' for a field never seen."""
    with pytest.raises(UnknownFeatureError, match="favourite_colour"):
        service.predict({**APPLICANT, "favourite_colour": "blue"})


def test_a_missing_feature_is_filled_the_way_ingest_fills_it(service):
    """A request omitting a field must score like a training row that lacked it."""
    sparse = {k: v for k, v in APPLICANT.items() if k != "occupation"}
    row = service.to_row(sparse)

    assert row.loc[0, "occupation"] == "__missing__"
    assert service.predict(sparse).score == pytest.approx(
        float(service.model.predict_proba(row)[0, 1])
    )


def test_two_identical_requests_record_the_same_input_digest(service):
    """Canonicalized, so key order and float formatting do not fork the digest."""
    first = service.predict(APPLICANT)
    reordered = dict(reversed(list(APPLICANT.items())))
    second = service.predict(reordered)

    envelopes = service.spool.pending()
    assert first.prediction_id != second.prediction_id

    service.flush()
    rows = service.catalog.load_table(PREDICTIONS.identifier).scan().to_arrow().to_pylist()
    digests = {r["input_digest"] for r in rows}
    assert len(envelopes) >= 1
    assert len(digests) == 1


# --------------------------------------------------------- explanation ----

def test_a_recorded_decision_explains_itself_from_the_audit_tables_alone(service):
    outcome = service.predict(APPLICANT, subject_id="subject-1")
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id)

    assert report.decision == outcome.decision
    assert report.score == pytest.approx(outcome.score)
    assert report.subject_id == "subject-1"
    assert report.input == APPLICANT
    assert report.reconciles


def test_the_report_carries_every_claim_the_capability_promises(service):
    """Model version, hyperparameters, training row count, and date range."""
    outcome = service.predict(APPLICANT)
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id)
    model_row = (
        service.catalog.load_table(MODEL_VERSIONS.identifier).scan().to_arrow().to_pylist()[0]
    )

    assert report.model_version_id == service.model_version_id
    assert report.recipe == "baseline-logreg"
    assert report.artifact_digest == model_row["artifact_digest"]
    assert report.hyperparams["C"] == 1.0
    assert report.training_row_count > 0

    low, high = report.training_as_of_range
    assert isinstance(low, dt.datetime) and isinstance(high, dt.datetime)
    assert low <= high


def test_a_truncated_report_still_reconciles_to_the_score(service):
    """The residual is what keeps a top-k view honest."""
    outcome = service.predict(APPLICANT)
    service.flush()

    for k in (1, 3, 10):
        report = explanation_for(service.catalog, outcome.prediction_id, top_k=k)
        assert len(report.attributions) == k
        assert report.total_feature_count > k

        total = report.base_value + sum(a.shap_value for a in report.attributions)
        assert total + report.residual == pytest.approx(float(logit(report.score)), abs=1e-9)


def test_the_served_attribution_is_the_one_that_was_recorded(service):
    """Not recomputed on read — the report must match what predict() produced."""
    outcome = service.predict(APPLICANT)
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id, top_k=5)
    recorded = outcome.explanation.top(5)

    assert [a.feature_name for a in report.attributions] == [a.feature_name for a in recorded]
    assert [a.shap_value for a in report.attributions] == pytest.approx(
        [a.shap_value for a in recorded]
    )
    assert report.base_value == pytest.approx(outcome.explanation.base_value)


def test_a_one_hot_column_the_subject_is_not_in_says_so(service):
    """'sex_Female = Male' is not an explanation, it is a false claim.

    Those columns carry real attribution — a zero differs from the training
    population's average and moves the score — so they cannot be dropped without
    breaking additivity. They are stated for what they are instead.
    """
    outcome = service.predict({**APPLICANT, "sex": "Male"})
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id, top_k=200)
    by_name = {a.feature_name: a for a in report.attributions}

    mine = by_name["cat__sex_Male"]
    assert mine.applies
    assert mine.statement == "sex = Male"

    not_mine = by_name["cat__sex_Female"]
    assert not not_mine.applies
    assert not_mine.statement == "sex is not Female"

    numeric = by_name["num__age"]
    assert numeric.applies
    assert numeric.statement.startswith("age = ")


def test_a_feature_name_containing_an_underscore_is_split_correctly(service):
    """``marital_status_Divorced`` must not become feature 'marital'."""
    outcome = service.predict({**APPLICANT, "marital_status": "Divorced"})
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id, top_k=200)
    column = next(
        a for a in report.attributions if a.feature_name == "cat__marital_status_Divorced"
    )

    assert column.source_feature == "marital_status"
    assert column.encoded_category == "Divorced"
    assert column.statement == "marital_status = Divorced"


def test_attributions_report_the_direction_they_pushed_the_decision(service):
    outcome = service.predict(APPLICANT)
    service.flush()

    report = explanation_for(service.catalog, outcome.prediction_id)
    for a in report.attributions:
        expected = "toward approval" if a.shap_value > 0 else "toward denial"
        assert a.direction == expected


def test_the_report_serializes_without_losing_a_claim(service):
    outcome = service.predict(APPLICANT, subject_id="subject-1")
    service.flush()

    payload = explanation_for(service.catalog, outcome.prediction_id).to_dict()

    assert payload["model"]["artifact_digest"]
    assert payload["training_data"]["row_count"] > 0
    assert payload["training_data"]["as_of_from"]
    assert payload["explanation"]["reconciles"] is True
    assert payload["explanation"]["attributions"][0]["abs_rank"] == 1


# ------------------------------------------------------- failing closed ----

def test_an_unknown_prediction_id_is_a_lookup_failure_not_an_integrity_failure(service):
    """404, not 503. Conflating them hides real corruption behind a common typo."""
    with pytest.raises(ExplanationNotFoundError):
        explanation_for(service.catalog, "no-such-prediction")


def test_a_prediction_whose_attributions_are_missing_is_refused(service, catalog):
    """Not reported as an unexplained decision — that state must not read as normal."""
    outcome = service.predict(APPLICANT)
    service.flush()
    catalog.load_table(ATTRIBUTIONS.identifier).delete(
        _equal("prediction_id", outcome.prediction_id)
    )

    with pytest.raises(ProvenanceIntegrityError, match="no audit.attributions rows"):
        explanation_for(catalog, outcome.prediction_id)


def test_a_prediction_citing_an_unknown_model_version_is_refused(service, catalog):
    """The trail has lost the thread; 'model: unknown' would let that persist."""
    outcome = service.predict(APPLICANT)
    service.flush()
    catalog.load_table(MODEL_VERSIONS.identifier).delete(
        _equal("model_version_id", service.model_version_id)
    )

    with pytest.raises(ProvenanceIntegrityError, match="no audit.model_versions row"):
        explanation_for(catalog, outcome.prediction_id)


def test_tampered_attributions_are_caught_on_read(service, catalog):
    """The read-side reconciliation check, doing the job it exists for."""
    outcome = service.predict(APPLICANT)
    service.flush()

    stored = (
        catalog.load_table(ATTRIBUTIONS.identifier)
        .scan(row_filter=_equal("prediction_id", outcome.prediction_id))
        .to_arrow()
        .to_pylist()
    )
    catalog.load_table(ATTRIBUTIONS.identifier).delete(
        _equal("prediction_id", outcome.prediction_id)
    )
    tampered = [{**r, "shap_value": r["shap_value"] + 1.0} for r in stored]
    append_records(catalog, ATTRIBUTIONS, tampered)

    with pytest.raises(ProvenanceIntegrityError, match="do not reconcile"):
        explanation_for(catalog, outcome.prediction_id)


def test_serving_refuses_a_model_version_with_no_provenance_record(catalog, gb_root):
    with pytest.raises(ProvenanceIntegrityError, match="no provenance record"):
        PredictionService.load(catalog, "not-a-real-model-version", gb_root)


def _equal(column: str, value: str):
    from pyiceberg.expressions import EqualTo

    return EqualTo(column, value)
