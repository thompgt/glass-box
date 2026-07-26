"""The contamination report's logic, on membership rows written directly.

Training a model to produce two membership rows would test scikit-learn. What
matters here is what the report does with the rows it finds — particularly the
ones that do not join cleanly, which is the case a naive implementation silently
drops.
"""

from __future__ import annotations

import datetime as dt

import pytest

from glassbox.erasure import contamination_report, membership_of
from glassbox.schemas import (
    ATTRIBUTIONS,
    CREDIT_APPLICATIONS,
    MODEL_VERSIONS,
    PREDICTIONS,
    TRAINING_MEMBERSHIP,
)
from glassbox.writer import append_records

TS = dt.datetime(2026, 3, 14, 15, 9, 26, tzinfo=dt.UTC)


def membership(model_version_id: str, subject_id: str, role: str = "train") -> dict:
    return {
        "model_version_id": model_version_id,
        "subject_id": subject_id,
        "role": role,
        "row_digest": None,
    }


def model_version(model_version_id: str, *, status: str = "active") -> dict:
    return {
        "model_version_id": model_version_id,
        "registered_name": "glassbox-credit",
        "mlflow_run_id": "run-1",
        "mlflow_model_version": None,
        "data_snapshot_uuid": "snap-train",
        "eval_snapshot_uuid": "snap-eval",
        "trained_at": TS,
        "estimator_class": "sklearn.linear_model.LogisticRegression",
        "hyperparams_json": "{}",
        "automl_config_digest": "none",
        "recipe": "baseline-logreg",
        "seed": 42,
        "artifact_digest": "d" * 64,
        "env_digest": "e" * 64,
        "code_git_sha": "abc123",
        "metrics_json": "{}",
        "status": status,
        "retired_at": None,
        "retire_reason": None,
    }


def feature_row(subject_id: str) -> dict:
    return {
        "subject_id": subject_id,
        "as_of_ts": TS,
        "ingest_batch_id": "batch-1",
        "source_row_digest": "f" * 64,
        "split": "train",
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
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
        "label": 0,
    }


def prediction(prediction_id: str, subject_id: str) -> dict:
    return {
        "prediction_id": prediction_id,
        "prediction_ts": TS,
        "model_version_id": "mv-1",
        "subject_id": subject_id,
        "input_json": "{}",
        "input_digest": "d" * 64,
        "score": 0.25,
        "threshold": 0.5,
        "decision": "deny",
        "explainer_status": "complete",
        "latency_ms": 4.0,
        "shap_ms": 1.0,
    }


# ------------------------------------------------------------- the lookup ----

def test_a_subject_is_traced_to_every_model_that_trained_on_them(catalog):
    append_records(catalog, MODEL_VERSIONS, [model_version("mv-1"), model_version("mv-2")])
    append_records(
        catalog,
        TRAINING_MEMBERSHIP,
        [membership("mv-1", "subject-a"), membership("mv-2", "subject-a"),
         membership("mv-1", "subject-b")],
    )

    report = contamination_report(catalog, "subject-a")

    assert report.model_version_ids == ["mv-1", "mv-2"]
    assert report.known


def test_the_role_distinguishes_training_from_evaluation(catalog):
    """Both are contamination, but they are not the same claim about the model."""
    append_records(catalog, MODEL_VERSIONS, [model_version("mv-1")])
    append_records(
        catalog,
        TRAINING_MEMBERSHIP,
        [membership("mv-1", "subject-a", "train"), membership("mv-1", "subject-a", "eval")],
    )

    roles = {c.role for c in contamination_report(catalog, "subject-a").contaminated}

    assert roles == {"train", "eval"}


def test_a_subject_the_trail_has_never_seen_is_reported_as_such(catalog):
    """Not an error. 'You are not in our data' is a legitimate answer to give."""
    report = contamination_report(catalog, "never-heard-of-them")

    assert not report.known
    assert report.contaminated == []
    assert report.live_row_count == 0
    assert report.to_dict()["contaminated_model_count"] == 0


def test_only_the_requested_subject_is_returned(catalog):
    append_records(catalog, MODEL_VERSIONS, [model_version("mv-1")])
    append_records(
        catalog,
        TRAINING_MEMBERSHIP,
        [membership("mv-1", "subject-a"), membership("mv-1", "subject-b")],
    )

    assert [r["subject_id"] for r in membership_of(catalog, "subject-a")] == ["subject-a"]


# ------------------------------------------------- the rows that do not join ----

def test_membership_naming_a_model_that_was_never_committed_is_still_reported(catalog):
    """Training writes membership first precisely so a crash over-reports.

    An inner join here would quietly convert that deliberate over-report back into
    the under-report the write ordering exists to prevent — telling a subject no
    model saw them when one may have.
    """
    append_records(catalog, TRAINING_MEMBERSHIP, [membership("mv-ghost", "subject-a")])

    report = contamination_report(catalog, "subject-a")

    assert report.model_version_ids == ["mv-ghost"]
    assert [c.model_version_id for c in report.orphaned] == ["mv-ghost"]
    assert report.contaminated[0].registered is False
    assert report.contaminated[0].status is None


def test_an_orphan_is_not_counted_as_serving(catalog):
    """It cannot be serving — there is no model version row to serve."""
    append_records(catalog, TRAINING_MEMBERSHIP, [membership("mv-ghost", "subject-a")])

    assert contamination_report(catalog, "subject-a").serving == []


def test_a_retired_model_is_reported_but_not_as_serving(catalog):
    append_records(
        catalog,
        MODEL_VERSIONS,
        [model_version("mv-live"), model_version("mv-old", status="retired")],
    )
    append_records(
        catalog,
        TRAINING_MEMBERSHIP,
        [membership("mv-live", "subject-a"), membership("mv-old", "subject-a")],
    )

    report = contamination_report(catalog, "subject-a")

    assert report.model_version_ids == ["mv-live", "mv-old"]
    assert [c.model_version_id for c in report.serving] == ["mv-live"]


# ------------------------------------------------------- the rest of the trail ----

def test_live_feature_rows_are_counted_per_table(catalog):
    append_records(catalog, CREDIT_APPLICATIONS, [feature_row("subject-a")])

    report = contamination_report(catalog, "subject-a")

    assert report.live_rows[CREDIT_APPLICATIONS.name] == 1
    assert report.live_rows["features.eval_holdout"] == 0
    assert report.live_row_count == 1


def test_decisions_served_about_the_subject_are_listed(catalog):
    append_records(
        catalog,
        PREDICTIONS,
        [prediction("p1", "subject-a"), prediction("p2", "subject-b")],
    )

    report = contamination_report(catalog, "subject-a")

    assert [d["prediction_id"] for d in report.decisions] == ["p1"]
    assert report.known, "a decision alone makes the subject known to the trail"


def test_decisions_can_be_left_out_when_only_models_are_wanted(catalog):
    append_records(catalog, PREDICTIONS, [prediction("p1", "subject-a")])

    report = contamination_report(catalog, "subject-a", include_decisions=False)

    assert report.decisions == []


def test_the_report_serializes_without_losing_a_claim(catalog):
    append_records(catalog, MODEL_VERSIONS, [model_version("mv-1")])
    append_records(catalog, TRAINING_MEMBERSHIP, [membership("mv-1", "subject-a")])
    append_records(catalog, CREDIT_APPLICATIONS, [feature_row("subject-a")])
    append_records(catalog, PREDICTIONS, [prediction("p1", "subject-a")])
    append_records(catalog, ATTRIBUTIONS, [])

    payload = contamination_report(catalog, "subject-a").to_dict()

    assert payload["subject_id"] == "subject-a"
    assert payload["contaminated_model_count"] == 1
    assert payload["serving_model_ids"] == ["mv-1"]
    assert payload["live_row_count"] == 1
    assert payload["decision_count"] == 1
    assert payload["contaminated"][0]["role"] == "train"


@pytest.mark.parametrize("role", ["train", "eval"])
def test_every_role_marks_the_model_as_serving_if_it_is_active(catalog, role):
    """Evaluating on a subject contaminates the model too — it tuned nothing, but
    the subject's data is still part of what justified releasing it."""
    append_records(catalog, MODEL_VERSIONS, [model_version("mv-1")])
    append_records(catalog, TRAINING_MEMBERSHIP, [membership("mv-1", "subject-a", role)])

    assert contamination_report(catalog, "subject-a").serving
