"""Capability #2 — answering "why was I denied?" from a prediction id alone.

The answer is assembled entirely from ``audit.*``. Nothing here loads a model,
re-scores the row, or recomputes SHAP: the attribution served is the one that was
recorded at the moment the decision was made, which is what makes it a statement
about *that decision* rather than about a model that has since been retrained.

Assembly walks four tables, and each join is a claim that can fail:

    audit.predictions      what was decided, and against which model version
    audit.attributions     the per-feature decomposition, as recorded then
    audit.model_versions   the recipe, hyperparameters, and artifact digest
    audit.data_snapshots   the training data version: row count and date range

A missing row at any step is a :class:`ProvenanceIntegrityError`, not a null
field in the response. A prediction citing a model version that does not exist is
not a slightly incomplete answer, it is an audit trail that has lost the thread —
and reporting it as "model: unknown" would let that state persist unnoticed.

The served attribution list is truncated to the top ``k``, so the residual is
carried explicitly and the reconciliation is checked *here*, against the stored
score, before the report is returned. An explanation that does not add up is
refused rather than displayed.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from typing import Any

from pyiceberg.expressions import EqualTo

from ..explain.dispatch import Explanation, expected_total, link_for_module
from ..schemas import ATTRIBUTIONS, DATA_SNAPSHOTS, MODEL_VERSIONS, PREDICTIONS
from ..train.registry import ProvenanceIntegrityError

DEFAULT_TOP_K = 10
RECONCILIATION_TOL = 1e-6


class ExplanationNotFoundError(LookupError):
    """No prediction with this id was ever recorded.

    Distinct from :class:`ProvenanceIntegrityError`: this is a 404, an id that
    does not exist. An integrity error is a 503 — the id exists and the trail
    behind it is broken, which is a much worse thing and must not be reported as
    "not found".
    """


@dataclass
class AttributionView:
    feature_name: str
    feature_value_str: str | None
    shap_value: float
    abs_rank: int

    @property
    def direction(self) -> str:
        """Which way this feature pushed the decision, in plain words."""
        return "toward approval" if self.shap_value > 0 else "toward denial"


@dataclass
class ExplanationReport:
    prediction_id: str
    prediction_ts: dt.datetime
    decision: str
    score: float
    threshold: float
    subject_id: str | None
    input: dict[str, Any]

    model_version_id: str
    recipe: str
    estimator_class: str
    hyperparams: dict[str, Any]
    artifact_digest: str
    trained_at: dt.datetime

    training_snapshot_uuid: str
    training_row_count: int
    training_as_of_range: tuple[dt.datetime | None, dt.datetime | None]

    base_value: float
    explainer_type: str
    link: str
    attributions: list[AttributionView]
    residual: float
    total_feature_count: int
    reconciles: bool = field(default=False)

    def to_dict(self) -> dict[str, Any]:
        low, high = self.training_as_of_range
        return {
            "prediction_id": self.prediction_id,
            "prediction_ts": self.prediction_ts.isoformat(),
            "decision": self.decision,
            "score": self.score,
            "threshold": self.threshold,
            "subject_id": self.subject_id,
            "input": self.input,
            "model": {
                "model_version_id": self.model_version_id,
                "recipe": self.recipe,
                "estimator_class": self.estimator_class,
                "hyperparams": self.hyperparams,
                "artifact_digest": self.artifact_digest,
                "trained_at": self.trained_at.isoformat(),
            },
            "training_data": {
                "data_snapshot_uuid": self.training_snapshot_uuid,
                "row_count": self.training_row_count,
                "as_of_from": low.isoformat() if low else None,
                "as_of_to": high.isoformat() if high else None,
            },
            "explanation": {
                "base_value": self.base_value,
                "explainer_type": self.explainer_type,
                "link": self.link,
                "attributions": [
                    {
                        "feature_name": a.feature_name,
                        "feature_value": a.feature_value_str,
                        "shap_value": a.shap_value,
                        "abs_rank": a.abs_rank,
                        "direction": a.direction,
                    }
                    for a in self.attributions
                ],
                # Present so a truncated list still sums to the score. Without it
                # a top-10 view of a 33-column encoded matrix silently fails to
                # add up, and the reader has no way to tell.
                "residual": self.residual,
                "total_feature_count": self.total_feature_count,
                "reconciles": self.reconciles,
            },
        }


def explanation_for(
    catalog, prediction_id: str, *, top_k: int = DEFAULT_TOP_K
) -> ExplanationReport:
    """Assemble the full audit answer for one recorded decision."""
    prediction = _one(catalog, PREDICTIONS, "prediction_id", prediction_id)
    if prediction is None:
        raise ExplanationNotFoundError(
            f"no audit.predictions row for {prediction_id!r}"
        )

    model = _one(catalog, MODEL_VERSIONS, "model_version_id", prediction["model_version_id"])
    if model is None:
        raise ProvenanceIntegrityError(
            f"prediction {prediction_id} was served by model version "
            f"{prediction['model_version_id']}, which has no audit.model_versions row"
        )

    snapshot = _one(catalog, DATA_SNAPSHOTS, "data_snapshot_uuid", model["data_snapshot_uuid"])
    if snapshot is None:
        raise ProvenanceIntegrityError(
            f"model version {model['model_version_id']} cites training snapshot "
            f"{model['data_snapshot_uuid']}, which has no audit.data_snapshots row"
        )

    rows = _attributions_for(catalog, prediction_id)
    if not rows:
        raise ProvenanceIntegrityError(
            f"prediction {prediction_id} has no audit.attributions rows; it was "
            f"recorded as {prediction['explainer_status']!r} but cannot be explained"
        )

    report = _assemble(prediction, model, snapshot, rows, top_k)
    _check_reconciliation(report, prediction_id)
    return report


def _assemble(
    prediction: dict[str, Any],
    model: dict[str, Any],
    snapshot: dict[str, Any],
    rows: list[dict[str, Any]],
    top_k: int,
) -> ExplanationReport:
    served = rows[:top_k]
    residual = sum(r["shap_value"] for r in rows[top_k:])

    return ExplanationReport(
        prediction_id=prediction["prediction_id"],
        prediction_ts=prediction["prediction_ts"],
        decision=prediction["decision"],
        score=prediction["score"],
        threshold=prediction["threshold"],
        subject_id=prediction["subject_id"],
        input=json.loads(prediction["input_json"]),
        model_version_id=model["model_version_id"],
        recipe=model["recipe"],
        estimator_class=model["estimator_class"],
        hyperparams=json.loads(model["hyperparams_json"]),
        artifact_digest=model["artifact_digest"],
        trained_at=model["trained_at"],
        training_snapshot_uuid=snapshot["data_snapshot_uuid"],
        training_row_count=snapshot["row_count"],
        training_as_of_range=(snapshot["min_as_of_ts"], snapshot["max_as_of_ts"]),
        base_value=rows[0]["base_value"],
        explainer_type=rows[0]["explainer_type"],
        link=_link_for(rows[0]["explainer_type"], model["estimator_class"]),
        attributions=[
            AttributionView(
                feature_name=r["feature_name"],
                feature_value_str=r["feature_value_str"],
                shap_value=r["shap_value"],
                abs_rank=r["abs_rank"],
            )
            for r in served
        ],
        residual=float(residual),
        total_feature_count=len(rows),
    )


def _check_reconciliation(report: ExplanationReport, prediction_id: str) -> None:
    """Does the stored decomposition still add up to the stored score?

    Checked on read as well as on write, because the two failures are different.
    On write it catches a SHAP output space we did not expect; here it catches an
    audit trail that has been mutated, truncated, or half-committed since — which
    is precisely what a reader of an audit trail needs to be told about rather
    than shown a plausible-looking answer.
    """
    total = report.base_value + sum(a.shap_value for a in report.attributions) + report.residual
    want = expected_total(
        Explanation(
            base_value=report.base_value,
            attributions=[],
            explainer_type=report.explainer_type,
            link=report.link,
        ),
        report.score,
    )

    report.reconciles = abs(total - want) < RECONCILIATION_TOL
    if not report.reconciles:
        raise ProvenanceIntegrityError(
            f"stored attributions for {prediction_id} do not reconcile: base + "
            f"attributions + residual = {total!r}, but the recorded score {report.score!r} "
            f"implies {want!r} in {report.link} space. The audit trail for this "
            f"decision is not internally consistent."
        )


def _link_for(explainer_type: str, estimator_class: str) -> str:
    """Recover the output space from what was recorded at prediction time.

    ``audit.attributions`` stores ``explainer_type`` but not the link, and that is
    deliberate: the link is fully determined by the explainer type plus the
    estimator class, and the estimator class is already on the trail in
    ``audit.model_versions``. A stored link column would be a third copy of a fact
    two columns already fix, free to drift out of agreement with them.

    ``explainer_type`` alone is not enough — ``"tree"`` covers both a scikit-learn
    forest (probability space) and a LightGBM booster (log-odds).
    """
    module = estimator_class.rsplit(".", 1)[0]
    return link_for_module(module, explainer_type)


def _one(catalog, td, column: str, value: str) -> dict[str, Any] | None:
    arrow = catalog.load_table(td.identifier).scan(row_filter=EqualTo(column, value)).to_arrow()
    if arrow.num_rows == 0:
        return None
    return arrow.to_pylist()[0]


def _attributions_for(catalog, prediction_id: str) -> list[dict[str, Any]]:
    """Every stored attribution for one prediction, strongest first.

    Re-sorted on read rather than trusted to come back ordered: the table is
    sorted by ``prediction_id``, which says nothing about the order of rows
    *within* one prediction, and a scan makes no promise across data files.
    """
    arrow = (
        catalog.load_table(ATTRIBUTIONS.identifier)
        .scan(row_filter=EqualTo("prediction_id", prediction_id))
        .to_arrow()
    )
    return sorted(arrow.to_pylist(), key=lambda r: r["abs_rank"])
