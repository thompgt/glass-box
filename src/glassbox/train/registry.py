"""Model registration — where the MLflow/Iceberg boundary is enforced.

The invariant this module implements:

    Iceberg ``audit.*`` is the sole system of record for provenance claims.
    MLflow is a content-addressed blob store. A provenance claim is true iff an
    Iceberg row asserts it **and** the referenced MLflow artifact's canonical
    digest matches the digest recorded in that row.

**Write ordering is strict: MLflow first, Iceberg last.** The Iceberg commit is
the commit point. A crash between the two leaves an orphan MLflow run, which is
garbage and harmless. The reverse ordering would leave an Iceberg row asserting
the existence of an artifact that was never written — an integrity failure that
is indistinguishable from tampering. So the ordering is not a preference.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow

from ..digest import canonical_json, env_digest
from ..schemas import MODEL_VERSIONS, TRAINING_MEMBERSHIP
from ..writer import append_records
from .canonical import canonical_bytes, model_digest
from .tracking import EXPERIMENT, code_git_sha, configure_mlflow

CANONICAL_ARTIFACT = "canonical_model.json"
MODEL_ARTIFACT = "model"


class ProvenanceIntegrityError(RuntimeError):
    """The Iceberg record and the MLflow artifact disagree.

    Raised rather than returning a degraded result: a prediction that cannot be
    attributed is worth less than no prediction, because it looks the same as one
    that can.
    """


@dataclass
class ModelVersion:
    model_version_id: str
    mlflow_run_id: str
    data_snapshot_uuid: str
    eval_snapshot_uuid: str
    estimator_class: str
    hyperparams: dict[str, Any]
    automl_config_digest: str
    recipe: str
    seed: int
    artifact_digest: str
    env_digest: str
    code_git_sha: str
    metrics: dict[str, float] = field(default_factory=dict)
    registered_name: str = "glassbox-credit"
    status: str = "active"
    trained_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def to_record(self) -> dict[str, Any]:
        return {
            "model_version_id": self.model_version_id,
            "registered_name": self.registered_name,
            "mlflow_run_id": self.mlflow_run_id,
            "mlflow_model_version": None,
            "data_snapshot_uuid": self.data_snapshot_uuid,
            "eval_snapshot_uuid": self.eval_snapshot_uuid,
            "trained_at": self.trained_at,
            "estimator_class": self.estimator_class,
            "hyperparams_json": canonical_json(self.hyperparams),
            "automl_config_digest": self.automl_config_digest,
            "recipe": self.recipe,
            "seed": self.seed,
            "artifact_digest": self.artifact_digest,
            "env_digest": self.env_digest,
            "code_git_sha": self.code_git_sha,
            "metrics_json": canonical_json(self.metrics),
            "status": self.status,
            "retired_at": None,
            "retire_reason": None,
        }


def register(
    catalog,
    model,
    *,
    data_snapshot_uuid: str,
    eval_snapshot_uuid: str,
    hyperparams: dict[str, Any],
    seed: int,
    metrics: dict[str, float],
    train_subject_ids: list[str],
    eval_subject_ids: list[str],
    automl_config_digest: str = "none",
    recipe: str = "baseline-logreg",
    status: str = "active",
    registered_name: str = "glassbox-credit",
    root: Path | None = None,
) -> ModelVersion:
    """Log to MLflow, then commit the provenance record to Iceberg."""
    experiment_id = configure_mlflow(root)
    digest = model_digest(model)
    model_version_id = str(uuid.uuid4())

    # --- MLflow first ------------------------------------------------------
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_version_id) as run:
        mlflow.set_tags(
            {
                # Mirrored, not authoritative — the Iceberg row is the truth.
                "glassbox.model_version_id": model_version_id,
                "glassbox.data_snapshot_uuid": data_snapshot_uuid,
                "glassbox.eval_snapshot_uuid": eval_snapshot_uuid,
                "glassbox.artifact_digest": digest,
            }
        )
        mlflow.log_params({k: str(v) for k, v in hyperparams.items()})
        mlflow.log_param("seed", seed)
        mlflow.log_metrics(metrics)

        # The canonical dump is logged as a plain artifact. The MLflow flavor
        # exists only for loading convenience; its bytes are MLflow's business and
        # may change between MLflow versions, which is exactly why the digest is
        # taken over this file and not over whatever the flavor writer emits.
        mlflow.log_text(canonical_bytes(model).decode("utf-8"), CANONICAL_ARTIFACT)
        _log_sklearn_model(model)

        run_id = run.info.run_id

    # --- Iceberg last: this is the commit point ----------------------------
    mv = ModelVersion(
        model_version_id=model_version_id,
        mlflow_run_id=run_id,
        data_snapshot_uuid=data_snapshot_uuid,
        eval_snapshot_uuid=eval_snapshot_uuid,
        estimator_class=_estimator_class_name(model),
        hyperparams=hyperparams,
        automl_config_digest=automl_config_digest,
        recipe=recipe,
        seed=seed,
        artifact_digest=digest,
        env_digest=env_digest(),
        code_git_sha=code_git_sha(),
        metrics=metrics,
        registered_name=registered_name,
        status=status,
    )
    append_records(catalog, MODEL_VERSIONS, [mv.to_record()])
    _write_membership(catalog, model_version_id, train_subject_ids, eval_subject_ids)
    return mv


def _log_sklearn_model(model) -> None:
    """MLflow renamed ``artifact_path`` to ``name`` in 3.x; support both."""
    import mlflow.sklearn

    try:
        mlflow.sklearn.log_model(model, name=MODEL_ARTIFACT)
    except TypeError:
        mlflow.sklearn.log_model(model, artifact_path=MODEL_ARTIFACT)


def _estimator_class_name(model) -> str:
    from .canonical import final_estimator

    est = final_estimator(model)
    return f"{type(est).__module__}.{type(est).__name__}"


def _write_membership(
    catalog, model_version_id: str, train_ids: list[str], eval_ids: list[str]
) -> int:
    """Materialize training-set membership at training time.

    This is the index the erasure contamination report reads. It is written now,
    rather than reconstructed later by scanning historical snapshots, because
    after a snapshot is expired the rows are gone — which is precisely the moment
    the question "which models saw this subject?" has to be answerable.
    """
    records = [
        {"model_version_id": model_version_id, "subject_id": sid, "role": role, "row_digest": None}
        for role, ids in (("train", train_ids), ("eval", eval_ids))
        for sid in ids
    ]
    return append_records(catalog, TRAINING_MEMBERSHIP, records)


# --------------------------------------------------------------- reading ----

def get_model_version(catalog, model_version_id: str) -> dict[str, Any] | None:
    from pyiceberg.expressions import EqualTo

    table = catalog.load_table(MODEL_VERSIONS.identifier)
    arrow = table.scan(row_filter=EqualTo("model_version_id", model_version_id)).to_arrow()
    if arrow.num_rows == 0:
        return None
    return arrow.to_pylist()[0]


def load_model_checked(catalog, model_version_id: str, root: Path | None = None):
    """Load a model and verify it against its Iceberg provenance record.

    Fails closed. If the artifact's canonical digest does not match the digest
    Iceberg recorded, the model is not returned at any severity — the caller
    cannot opt out of the check, because a caller who could would eventually.
    """
    configure_mlflow(root)
    record = get_model_version(catalog, model_version_id)
    if record is None:
        raise ProvenanceIntegrityError(
            f"no audit.model_versions row for {model_version_id}; refusing to serve "
            f"a model with no provenance record"
        )

    import mlflow.sklearn

    uri = f"runs:/{record['mlflow_run_id']}/{MODEL_ARTIFACT}"
    try:
        model = mlflow.sklearn.load_model(uri)
    except Exception as exc:  # noqa: BLE001
        raise ProvenanceIntegrityError(
            f"audit.model_versions references MLflow artifact {uri}, which could not "
            f"be loaded: {exc}"
        ) from exc

    actual = model_digest(model)
    if actual != record["artifact_digest"]:
        raise ProvenanceIntegrityError(
            f"artifact digest mismatch for {model_version_id}: "
            f"Iceberg records {record['artifact_digest']}, MLflow artifact hashes to {actual}"
        )

    return model, record


__all__ = [
    "ModelVersion",
    "ProvenanceIntegrityError",
    "register",
    "get_model_version",
    "load_model_checked",
    "EXPERIMENT",
]
