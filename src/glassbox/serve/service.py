"""Scoring a request and recording what was decided, as one operation.

The service holds three things that must not drift apart: the model whose digest
Iceberg vouches for, the explainer built from *that model's* training snapshot,
and the spool that records the result. They are loaded together and fail
together — :meth:`PredictionService.load` either returns a service whose model
passed its digest check, or raises. There is no degraded mode where predictions
are served while attribution is unavailable, because a prediction that cannot be
attributed looks exactly like one that can.

Two ordering rules, both the same rule as everywhere else in this system:

* The attribution is computed **before** the response is built, not after. An
  attribution reconstructed later is a statement about a model; the thing a
  right-to-explanation request asks for is a statement about *that decision*.
* The spool append happens **before** the response is returned. A decision that
  reached a caller but not the disk is unrecorded, and unrecorded is the one
  failure this system is built to make impossible.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..digest import canonical_json, sha256_hex
from ..explain import Explanation, PredictionExplainer, build_background
from ..schemas import CREDIT_APPLICATIONS
from ..snapshots import get_snapshot, materialize
from ..train import features as F
from ..train.registry import ProvenanceIntegrityError, load_model_checked
from .spool import Spool, SpoolEnvelope

DEFAULT_THRESHOLD = 0.5
BACKGROUND_ROWS = 200

# Decision vocabulary. "approve"/"deny" rather than 1/0 because the audit row is
# read by people, and a stored 0 means nothing without the convention that
# produced it.
APPROVE = "approve"
DENY = "deny"


class UnknownFeatureError(ValueError):
    """The request carried a field the model has no place for.

    Rejected rather than ignored. Silently dropping an unrecognized field means
    the caller believes it influenced the decision and the audit trail says it
    was considered, when in fact neither is true.
    """


@dataclass
class PredictionOutcome:
    prediction_id: str
    prediction_ts: dt.datetime
    model_version_id: str
    subject_id: str | None
    score: float
    threshold: float
    decision: str
    explanation: Explanation
    latency_ms: float
    shap_ms: float


@dataclass
class PredictionService:
    """A loaded model version, its explainer, and the spool recording its output."""

    catalog: Any
    model: Any
    record: dict[str, Any]
    explainer: PredictionExplainer
    spool: Spool
    threshold: float = DEFAULT_THRESHOLD

    @property
    def model_version_id(self) -> str:
        return self.record["model_version_id"]

    # ------------------------------------------------------------- loading ----

    @classmethod
    def load(
        cls,
        catalog,
        model_version_id: str,
        root: Path,
        *,
        threshold: float = DEFAULT_THRESHOLD,
        spool: Spool | None = None,
        background_rows: int = BACKGROUND_ROWS,
    ) -> PredictionService:
        """Load a model version and everything needed to explain it.

        Raises :class:`ProvenanceIntegrityError` if the artifact does not match
        the digest Iceberg recorded, or if the training snapshot it names is
        gone. Both are refusals to serve, not warnings: the second one matters
        because an explainer built from the wrong background reports attributions
        against a baseline the model never saw.
        """
        model, record = load_model_checked(catalog, model_version_id, root)

        snapshot = get_snapshot(catalog, record["data_snapshot_uuid"])
        if snapshot is None:
            raise ProvenanceIntegrityError(
                f"audit.model_versions cites data snapshot {record['data_snapshot_uuid']} "
                f"for {model_version_id}, but audit.data_snapshots has no such row; "
                f"refusing to explain against a baseline that cannot be reconstructed"
            )

        frame = F.to_frame(materialize(catalog, snapshot, CREDIT_APPLICATIONS))
        explainer = PredictionExplainer(model, build_background(frame, n=background_rows))

        return cls(
            catalog=catalog,
            model=model,
            record=record,
            explainer=explainer,
            spool=spool or Spool(root=root),
            threshold=threshold,
        )

    # ------------------------------------------------------------ scoring ----

    def predict(
        self, payload: dict[str, Any], *, subject_id: str | None = None
    ) -> PredictionOutcome:
        """Score one applicant, explain the score, and durably record both."""
        started = time.perf_counter()
        row = self.to_row(payload)

        score = float(self.model.predict_proba(row)[0, 1])

        shap_started = time.perf_counter()
        explanation = self.explainer.explain(row, score=score)
        shap_ms = (time.perf_counter() - shap_started) * 1000.0

        outcome = PredictionOutcome(
            prediction_id=str(uuid.uuid4()),
            prediction_ts=dt.datetime.now(dt.UTC),
            model_version_id=self.model_version_id,
            subject_id=subject_id,
            score=score,
            threshold=self.threshold,
            decision=APPROVE if score >= self.threshold else DENY,
            explanation=explanation,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            shap_ms=shap_ms,
        )

        self.spool.append(self.to_envelope(outcome, payload))
        return outcome

    def to_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Build the single-row frame the pipeline expects from a request body.

        Missing features are filled the same way ingest fills them, so a request
        omitting a field is scored identically to a training row that lacked it.
        Unknown fields are an error rather than a silent drop.
        """
        unknown = set(payload) - set(F.FEATURE_COLUMNS)
        if unknown:
            raise UnknownFeatureError(
                f"unknown feature(s) {sorted(unknown)}; the model was fit on "
                f"{list(F.FEATURE_COLUMNS)}"
            )

        row: dict[str, Any] = {}
        for column in F.NUMERIC_FEATURES:
            value = payload.get(column)
            row[column] = 0.0 if value is None else float(value)
        for column in F.CATEGORICAL_FEATURES:
            value = payload.get(column)
            # OneHotEncoder was fit with handle_unknown="ignore", so an unseen
            # category encodes to all-zeros rather than raising — the row still
            # scores, and the attribution honestly shows that category
            # contributing nothing.
            row[column] = F.MISSING if value is None else str(value)

        return pd.DataFrame([row])[list(F.FEATURE_COLUMNS)]

    # ------------------------------------------------------------ recording ----

    def to_envelope(self, outcome: PredictionOutcome, payload: dict[str, Any]) -> SpoolEnvelope:
        """The audit rows for one decision: what was decided, and why."""
        input_json = canonical_json(payload)

        prediction = {
            "prediction_id": outcome.prediction_id,
            "prediction_ts": outcome.prediction_ts,
            "model_version_id": outcome.model_version_id,
            "subject_id": outcome.subject_id,
            "input_json": input_json,
            # Over the canonical form, so that two requests differing only in key
            # order or float formatting are recognizably the same input.
            "input_digest": sha256_hex(input_json.encode("utf-8")),
            "score": outcome.score,
            "threshold": outcome.threshold,
            "decision": outcome.decision,
            # Complete at the moment it is written: attributions are computed
            # synchronously, so "pending" would never be true here. The column
            # exists for the async path this design deliberately does not take.
            "explainer_status": "complete",
            "latency_ms": outcome.latency_ms,
            "shap_ms": outcome.shap_ms,
        }

        attributions = [
            {
                "prediction_id": outcome.prediction_id,
                "prediction_ts": outcome.prediction_ts,
                "model_version_id": outcome.model_version_id,
                "feature_name": a.feature_name,
                "feature_value_str": a.feature_value_str,
                "shap_value": a.shap_value,
                "base_value": outcome.explanation.base_value,
                "abs_rank": a.abs_rank,
                "explainer_type": outcome.explanation.explainer_type,
            }
            for a in outcome.explanation.attributions
        ]

        return SpoolEnvelope(prediction=prediction, attributions=attributions)

    def flush(self):
        """Drain the spool into Iceberg. Returns a :class:`FlushResult`."""
        return self.spool.flush(self.catalog)
