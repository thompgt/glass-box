"""Phase 1 baseline: logistic regression, trained from a pinned data version.

Deliberately the least interesting model in the project. The walking skeleton
exists to prove the *provenance* path end to end — pinned snapshot in, registered
model version out, every claim recorded — before AutoML is allowed anywhere near
it. A trivial model makes failures in the provenance layer obvious instead of
letting them hide behind model complexity.

Determinism controls applied here, each of which is separately assertable:

* rows arrive already sorted by ``subject_id`` (:func:`glassbox.snapshots.materialize`)
* categories are explicitly sorted (:mod:`glassbox.train.features`)
* ``random_state`` is fixed and the solver is single-threaded
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from ..catalog import glassbox_root, load_catalog
from ..digest import canonical_json, sha256_hex
from ..schemas import CREDIT_APPLICATIONS, EVAL_HOLDOUT
from ..snapshots import capture_snapshot, get_snapshot, materialize
from . import features as F
from .registry import ModelVersion, register

DEFAULT_SEED = 42

# Recorded on every model version so reproduction re-runs this trainer and not
# another one that happens to produce the same estimator class.
RECIPE = "baseline-logreg"

# lbfgs on a single thread. Multi-threaded BLAS reductions sum in a
# nondeterministic order, which perturbs the last bits of the coefficients and
# would make the artifact digest flap for reasons unrelated to the model.
# 'penalty' and 'n_jobs' are deliberately absent: scikit-learn 1.8 deprecated
# both for LogisticRegression (l2 is the default; n_jobs has no effect). Passing
# them would emit FutureWarnings and break on 1.10.
BASELINE_HYPERPARAMS: dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "tol": 1e-6,
}


@dataclass
class TrainingResult:
    model_version: ModelVersion
    model: Any
    metrics: dict[str, float]
    train_rows: int
    eval_rows: int


def _pin_single_threaded() -> None:
    """Force single-threaded BLAS before any array work happens."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"


def config_digest(hyperparams: dict[str, Any], seed: int) -> str:
    """Digest of everything that determines the fit, other than the data.

    For the baseline this is small. From Phase 4 it also covers the AutoML search
    space, iteration budget, and CV specification — so that "same config" is a
    checkable claim rather than an assertion in a commit message.
    """
    return sha256_hex(
        canonical_json({"hyperparams": hyperparams, "seed": seed, "recipe": RECIPE})
    )


def train_baseline(
    *,
    data_snapshot_uuid: str | None = None,
    eval_snapshot_uuid: str | None = None,
    seed: int = DEFAULT_SEED,
    hyperparams: dict[str, Any] | None = None,
    status: str = "active",
    root: Path | None = None,
) -> TrainingResult:
    """Train from a pinned data version and register the result.

    Both snapshots default to the most recent capture, but are recorded
    explicitly on the model version either way — a model that cannot name its
    training data is the failure mode this whole project exists to prevent.
    """
    _pin_single_threaded()
    root = root or glassbox_root()
    catalog = load_catalog(root)
    hyperparams = dict(hyperparams or BASELINE_HYPERPARAMS)

    train_snap = _resolve(catalog, data_snapshot_uuid, CREDIT_APPLICATIONS, split="train")
    eval_snap = _resolve(catalog, eval_snapshot_uuid, EVAL_HOLDOUT)

    fit = fit_baseline(catalog, train_snap, eval_snap, seed=seed, hyperparams=hyperparams)
    model, metrics, train_table, eval_table = (
        fit.model,
        fit.metrics,
        fit.train_table,
        fit.eval_table,
    )

    mv = register(
        catalog,
        model,
        data_snapshot_uuid=train_snap["data_snapshot_uuid"],
        eval_snapshot_uuid=eval_snap["data_snapshot_uuid"],
        hyperparams=hyperparams,
        seed=seed,
        metrics=metrics,
        train_subject_ids=F.subject_ids_of(train_table),
        eval_subject_ids=F.subject_ids_of(eval_table),
        automl_config_digest=config_digest(hyperparams, seed),
        recipe=RECIPE,
        status=status,
        root=root,
    )

    return TrainingResult(
        model_version=mv,
        model=model,
        metrics=metrics,
        train_rows=train_table.num_rows,
        eval_rows=eval_table.num_rows,
    )


@dataclass
class FitResult:
    model: Any
    metrics: dict[str, float]
    train_table: Any
    eval_table: Any


def fit_baseline(
    catalog,
    train_snap: dict,
    eval_snap: dict,
    *,
    seed: int = DEFAULT_SEED,
    hyperparams: dict[str, Any] | None = None,
) -> FitResult:
    """Fit the baseline without registering anything.

    Split out from :func:`train_baseline` so that reproduction can re-run the
    exact same code path without writing a second model version. If reproduction
    had its own copy of the fitting logic, the two could drift and the
    reproducibility check would be testing the copy rather than the original.
    """
    _pin_single_threaded()
    hyperparams = dict(hyperparams or BASELINE_HYPERPARAMS)

    train_table = materialize(catalog, train_snap, CREDIT_APPLICATIONS)
    eval_table = materialize(catalog, eval_snap, EVAL_HOLDOUT)

    X_train, y_train = F.to_frame(train_table), F.target_of(train_table)
    X_eval, y_eval = F.to_frame(eval_table), F.target_of(eval_table)

    np.random.seed(seed)
    estimator = LogisticRegression(random_state=seed, **hyperparams)
    model = F.build_pipeline(train_table, estimator)
    model.fit(X_train, y_train)

    return FitResult(
        model=model,
        metrics=_evaluate(model, X_train, y_train, X_eval, y_eval),
        train_table=train_table,
        eval_table=eval_table,
    )


def _resolve(catalog, snapshot_uuid: str | None, td, *, split: str | None = None) -> dict:
    if snapshot_uuid:
        record = get_snapshot(catalog, snapshot_uuid)
        if record is None:
            raise ValueError(f"no audit.data_snapshots row for {snapshot_uuid}")
        return record
    return capture_snapshot(catalog, td, split=split).to_record()


def _evaluate(model, X_train, y_train, X_eval, y_eval) -> dict[str, float]:
    train_proba = model.predict_proba(X_train)[:, 1]
    eval_proba = model.predict_proba(X_eval)[:, 1]
    return {
        "train_auc": float(roc_auc_score(y_train, train_proba)),
        "eval_auc": float(roc_auc_score(y_eval, eval_proba)),
        "train_accuracy": float(accuracy_score(y_train, train_proba >= 0.5)),
        "eval_accuracy": float(accuracy_score(y_eval, eval_proba >= 0.5)),
        "eval_positive_rate": float(np.mean(eval_proba >= 0.5)),
    }
