"""Capability #1 — bit-exact reproducibility, and the controls that make it mean something.

A passing reproducibility test is easy to fake: if nothing in the pipeline could
vary, of course two runs agree. The negative controls here exist to prove the
determinism machinery is actually load-bearing — that row order *would* change
the model if it were not pinned, and that a real data change *does* change the
digest.
"""

from __future__ import annotations

import numpy as np
import pytest

from glassbox.ingest import ingest_adult
from glassbox.schemas import CREDIT_APPLICATIONS, TRAINING_MEMBERSHIP
from glassbox.train import features as F
from glassbox.train import train_baseline
from glassbox.train.canonical import UnsupportedModelError, model_digest
from glassbox.train.registry import (
    ProvenanceIntegrityError,
    get_model_version,
    load_model_checked,
)
from glassbox.train.reproduce import EnvironmentDriftError, retrain_from_provenance

pytestmark = pytest.mark.slow


@pytest.fixture
def trained(catalog, adult_file, gb_root):
    ingest_adult(gb_root)
    return train_baseline(root=gb_root)


# ------------------------------------------------------------ the claim ----

def test_retraining_from_provenance_reproduces_the_artifact(trained, gb_root):
    result = retrain_from_provenance(trained.model_version.model_version_id, root=gb_root)

    assert result.recorded_digest == result.reproduced_digest
    assert result.recorded_content_digest == result.reproduced_content_digest
    assert result.predictions_match
    assert result.probe_rows > 0
    assert result.reproduced


def test_training_twice_gives_the_same_artifact(trained, gb_root):
    again = train_baseline(root=gb_root)
    assert again.model_version.artifact_digest == trained.model_version.artifact_digest
    # ...but a genuinely distinct model version, with its own identity.
    assert again.model_version.model_version_id != trained.model_version.model_version_id


# ---------------------------------------------------- negative controls ----

def test_row_order_would_change_the_model_if_it_were_not_pinned(trained, catalog, gb_root):
    """Proves the sort in materialize() is load-bearing rather than decorative.

    If shuffling the training rows produced the same fit, the sort would be
    pointless and this test would be vacuous. It does not, which is exactly why
    an unsorted Iceberg scan is a reproducibility hazard.
    """
    from glassbox.snapshots import get_snapshot, materialize

    snap = get_snapshot(catalog, trained.model_version.data_snapshot_uuid)
    table = materialize(catalog, snap, CREDIT_APPLICATIONS)

    rng = np.random.default_rng(0)
    shuffled = table.take(rng.permutation(table.num_rows))

    ordered_digest = model_digest(_fit_on(table))
    shuffled_digest = model_digest(_fit_on(shuffled))

    assert ordered_digest == trained.model_version.artifact_digest
    assert shuffled_digest != ordered_digest, (
        "shuffling training rows did not change the fit, so this test cannot "
        "demonstrate that the sort matters — investigate before trusting it"
    )


def test_a_data_change_changes_the_digest(trained, catalog, gb_root, adult_file):
    """The digest must actually track the data, not just be stable."""
    extra = gb_root / "data" / "raw" / "more.data"
    extra.write_text(
        "\n".join(
            line.replace("Male", "Female")
            for line in adult_file.read_text(encoding="utf-8").splitlines()[:120]
        )
        + "\n",
        encoding="utf-8",
    )
    from glassbox.ingest import adult
    from glassbox.writer import append_arrow

    append_arrow(catalog, CREDIT_APPLICATIONS, adult.parse_adult(extra, ingest_batch_id="b2"))

    retrained = train_baseline(root=gb_root)
    assert retrained.model_version.data_snapshot_uuid != trained.model_version.data_snapshot_uuid
    assert retrained.model_version.artifact_digest != trained.model_version.artifact_digest


def test_hyperparameter_change_changes_the_digest(trained, gb_root):
    hyperparams = {"C": 0.01, "max_iter": 1000, "solver": "lbfgs"}
    other = train_baseline(root=gb_root, hyperparams=hyperparams)
    assert other.model_version.artifact_digest != trained.model_version.artifact_digest


def test_seed_does_not_affect_a_deterministic_solver(trained, gb_root):
    """Documents a real property rather than asserting a convenient one.

    lbfgs ignores ``random_state`` — scikit-learn only consumes it for sag, saga
    and liblinear. So the baseline is seed-independent, and asserting otherwise
    would be asserting a bug. The seed genuinely drives the fit from Phase 4,
    where it controls the AutoML search.
    """
    other = train_baseline(root=gb_root, seed=trained.model_version.seed + 1)
    assert other.model_version.artifact_digest == trained.model_version.artifact_digest


# ------------------------------------------------- environment guarding ----

def test_reproduction_refuses_under_environment_drift(trained, catalog, gb_root, monkeypatch):
    monkeypatch.setattr("glassbox.train.reproduce.env_digest", lambda: "a-different-environment")

    with pytest.raises(EnvironmentDriftError, match="different experiment"):
        retrain_from_provenance(trained.model_version.model_version_id, root=gb_root)

    # ...and can be overridden explicitly, reporting the result as advisory.
    result = retrain_from_provenance(
        trained.model_version.model_version_id, root=gb_root, strict_env=False
    )
    assert result.digests_match


# ---------------------------------------------------- provenance record ----

def test_model_version_records_every_claim_the_endpoint_must_serve(trained, catalog):
    record = get_model_version(catalog, trained.model_version.model_version_id)

    assert record["status"] == "active"
    assert record["recipe"] == "baseline-logreg"
    assert record["data_snapshot_uuid"] and record["eval_snapshot_uuid"]
    assert record["artifact_digest"] and record["env_digest"]
    assert record["code_git_sha"]
    assert '"C":1.0' in record["hyperparams_json"].replace(" ", "")


def test_training_membership_is_materialized_for_every_subject(trained, catalog):
    membership = catalog.load_table(TRAINING_MEMBERSHIP.identifier).scan().to_arrow()
    mine = membership.filter(
        np_equal(membership["model_version_id"], trained.model_version.model_version_id)
    )
    roles = mine["role"].to_pylist()
    assert roles.count("train") == trained.train_rows
    assert roles.count("eval") == trained.eval_rows

    ids = mine["subject_id"].to_pylist()
    assert len(ids) == len(set(ids)), "a subject must appear once per model version"


def test_load_model_checked_fails_closed_on_digest_mismatch(trained, catalog, gb_root):
    """A prediction that cannot be attributed is worth less than no prediction."""
    model, record = load_model_checked(catalog, trained.model_version.model_version_id, gb_root)
    assert model is not None

    import glassbox.train.registry as reg

    monkey = "deadbeef" * 8
    original = reg.model_digest
    reg.model_digest = lambda _m: monkey
    try:
        with pytest.raises(ProvenanceIntegrityError, match="digest mismatch"):
            load_model_checked(catalog, trained.model_version.model_version_id, gb_root)
    finally:
        reg.model_digest = original


def test_unknown_model_version_is_refused(catalog, gb_root):
    with pytest.raises(ProvenanceIntegrityError, match="no provenance record"):
        load_model_checked(catalog, "not-a-real-model-version", gb_root)


def test_unsupported_estimator_raises_rather_than_guessing():
    from sklearn.neural_network import MLPClassifier

    with pytest.raises(UnsupportedModelError):
        model_digest(MLPClassifier())


# ----------------------------------------------------------------- utils ----

def _fit_on(table):
    from sklearn.linear_model import LogisticRegression

    from glassbox.train.baseline import BASELINE_HYPERPARAMS

    est = LogisticRegression(random_state=42, **BASELINE_HYPERPARAMS)
    model = F.build_pipeline(table, est)
    model.fit(F.to_frame(table), F.target_of(table))
    return model


def np_equal(column, value):
    import pyarrow.compute as pc

    return pc.equal(column, value)
