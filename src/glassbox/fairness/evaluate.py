"""Evaluate one model version against one frozen eval snapshot, and record it.

The eval set is read from ``features.eval_holdout``, which ingest freezes on
first write and never appends to again. That is what makes cross-version
comparison mean anything: if the eval set could drift between two evaluations,
the difference between their metrics would be partly a statement about the eval
set, and no amount of decomposition downstream could separate the two.

The model is loaded through :func:`load_model_checked`, so a model whose MLflow
artifact no longer matches its Iceberg digest cannot be evaluated at all. A
fairness number computed from an unverified artifact is a number about some
model, and the audit row would claim it was about this one.

Writes are idempotent by construction: ``evaluation_id`` is a UUID5 over
everything that determines the value, so re-evaluating the same model on the same
data under the same threshold produces the same row identities. If it produces
different *values*, that is not smoothed over — it means the evaluation is
nondeterministic or the frozen eval set moved, and both are reported rather than
overwritten.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyiceberg.expressions import And, EqualTo

from ..schemas import EVAL_HOLDOUT, FAIRNESS_EVALUATIONS
from ..snapshots import get_snapshot, materialize
from ..train import features as F
from ..train.registry import load_model_checked
from ..writer import append_records
from .metrics import MIN_GROUP_N, GroupResult, differences, eligible_n, group_metrics

# Fixed namespace, so an evaluation_id is reproducible on any machine.
EVALUATION_NAMESPACE = uuid.UUID("3d5c8a2f-7b41-5e96-8c03-1a9f4d2e7b58")

DEFAULT_SENSITIVE_ATTRIBUTE = "sex"
DEFAULT_THRESHOLD = 0.5


class SensitiveAttributeError(ValueError):
    """The requested attribute is not a column of the eval set."""


class FairnessEvaluationConflict(RuntimeError):
    """The same model on the same frozen data produced a different number."""


def evaluation_id(
    model_version_id: str,
    eval_snapshot_uuid: str,
    sensitive_attribute: str,
    threshold: float,
    metric_name: str,
    group_value: str | None,
    min_group_n: int | None = None,
) -> str:
    """Identity over everything that determines the value, and nothing else.

    ``min_group_n`` participates only for the across-group rows it governs, so a
    per-group rate keeps one identity regardless of what cutoff the caller used.
    """
    key = "|".join(
        [
            model_version_id,
            eval_snapshot_uuid,
            sensitive_attribute,
            # repr, not str: 0.5 and 0.50000000000000001 are different thresholds
            # and must not collapse into one row identity.
            repr(float(threshold)),
            metric_name,
            group_value or "",
            "" if min_group_n is None else str(int(min_group_n)),
        ]
    )
    return str(uuid.uuid5(EVALUATION_NAMESPACE, key))


@dataclass
class FairnessEvaluation:
    """Every fairness number for one (model version, eval snapshot, attribute)."""

    model_version_id: str
    eval_snapshot_uuid: str
    sensitive_attribute: str
    decision_threshold: float
    min_group_n: int
    groups: list[GroupResult]
    differences: dict[str, float]
    evaluated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def n(self) -> int:
        return sum(g.n for g in self.groups)

    @property
    def comparable_n(self) -> int:
        """Subjects behind the difference metrics, after small groups drop out."""
        return eligible_n(self.groups, min_group_n=self.min_group_n)

    def metric(self, name: str) -> float:
        """One difference metric, or raise — an absent metric is not a zero."""
        if name not in self.differences:
            raise KeyError(
                f"{name} is not defined for {self.model_version_id} on "
                f"{self.sensitive_attribute}: fewer than two groups had a defined "
                f"rate and at least {self.min_group_n} subjects. Available: "
                f"{sorted(self.differences)}"
            )
        return self.differences[name]

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for group in self.groups:
            for name, value in sorted(group.as_dict().items()):
                rows.append(self._row(name, value, group.group_value, group.n, None))
        for name, value in sorted(self.differences.items()):
            rows.append(self._row(name, value, None, self.comparable_n, self.min_group_n))
        return rows

    def _row(
        self, metric: str, value: float, group: str | None, n: int, cutoff: int | None
    ) -> dict[str, Any]:
        return {
            "evaluation_id": evaluation_id(
                self.model_version_id,
                self.eval_snapshot_uuid,
                self.sensitive_attribute,
                self.decision_threshold,
                metric,
                group,
                cutoff,
            ),
            "model_version_id": self.model_version_id,
            "eval_snapshot_uuid": self.eval_snapshot_uuid,
            "sensitive_attribute": self.sensitive_attribute,
            "group_value": group,
            "metric_name": metric,
            "metric_value": float(value),
            "n": int(n),
            "evaluated_at": self.evaluated_at,
            "decision_threshold": float(self.decision_threshold),
            "min_group_n": None if cutoff is None else int(cutoff),
        }


def evaluate_fairness(
    catalog,
    model_version_id: str,
    *,
    sensitive_attribute: str = DEFAULT_SENSITIVE_ATTRIBUTE,
    eval_snapshot_uuid: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_n: int = MIN_GROUP_N,
    root: Path | None = None,
    persist: bool = True,
) -> FairnessEvaluation:
    """Score the frozen eval set with one model and record the group metrics.

    ``eval_snapshot_uuid`` defaults to the one the model version recorded at
    training time. Comparisons across versions should pass it explicitly, because
    two models that name different eval snapshots cannot be compared without
    choosing one of them.
    """
    model, record = load_model_checked(catalog, model_version_id, root)
    eval_snapshot_uuid = eval_snapshot_uuid or record["eval_snapshot_uuid"]

    snapshot = get_snapshot(catalog, eval_snapshot_uuid)
    if snapshot is None:
        raise ValueError(f"no audit.data_snapshots row for {eval_snapshot_uuid}")

    # Raises SnapshotTombstonedError if this eval version was destroyed to satisfy
    # an erasure — a designed refusal. See capability #4.
    eval_table = materialize(catalog, snapshot, EVAL_HOLDOUT)
    if eval_table.num_rows == 0:
        raise ValueError(f"eval snapshot {eval_snapshot_uuid} contains no rows")

    if sensitive_attribute not in eval_table.column_names:
        raise SensitiveAttributeError(
            f"{sensitive_attribute!r} is not a column of {EVAL_HOLDOUT.name}; "
            f"available: {sorted(set(eval_table.column_names) - set(F.EXCLUDED))}"
        )

    scores = model.predict_proba(F.to_frame(eval_table))[:, 1]
    decisions = np.asarray(scores >= threshold)

    groups = group_metrics(
        F.target_of(eval_table), decisions, eval_table[sensitive_attribute].to_pylist()
    )
    evaluation = FairnessEvaluation(
        model_version_id=model_version_id,
        eval_snapshot_uuid=eval_snapshot_uuid,
        sensitive_attribute=sensitive_attribute,
        decision_threshold=float(threshold),
        min_group_n=min_group_n,
        groups=groups,
        differences=differences(groups, min_group_n=min_group_n),
    )

    if persist:
        _persist(catalog, evaluation)
    return evaluation


def _persist(catalog, evaluation: FairnessEvaluation) -> int:
    """Write the rows, or verify that the identical rows are already there.

    Deliberately not an upsert. Re-evaluating the same model on the same frozen
    data at the same threshold and getting a different number is a real finding —
    either the evaluation is nondeterministic or the eval set moved despite being
    frozen — and silently replacing the old row would destroy the evidence.
    """
    records = evaluation.to_records()
    existing = _existing_rows(
        catalog,
        evaluation.model_version_id,
        evaluation.eval_snapshot_uuid,
        evaluation.sensitive_attribute,
        evaluation.decision_threshold,
    )
    if not existing:
        return append_records(catalog, FAIRNESS_EVALUATIONS, records)

    previous = {r["evaluation_id"]: r["metric_value"] for r in existing}
    for row in records:
        before = previous.get(row["evaluation_id"])
        if before is not None and before != row["metric_value"]:
            raise FairnessEvaluationConflict(
                f"re-evaluating {evaluation.model_version_id} on "
                f"{evaluation.eval_snapshot_uuid} gave {row['metric_name']}"
                f"{'[' + row['group_value'] + ']' if row['group_value'] else ''} = "
                f"{row['metric_value']!r}, but audit.fairness_evaluations already "
                f"records {before!r} for the same model, data, and threshold"
            )

    fresh = [r for r in records if r["evaluation_id"] not in previous]
    return append_records(catalog, FAIRNESS_EVALUATIONS, fresh)


def _existing_rows(
    catalog,
    model_version_id: str,
    eval_snapshot_uuid: str,
    sensitive_attribute: str,
    threshold: float,
) -> list[dict[str, Any]]:
    table = catalog.load_table(FAIRNESS_EVALUATIONS.identifier)
    predicate = And(
        And(
            EqualTo("model_version_id", model_version_id),
            EqualTo("eval_snapshot_uuid", eval_snapshot_uuid),
        ),
        And(
            EqualTo("sensitive_attribute", sensitive_attribute),
            EqualTo("decision_threshold", float(threshold)),
        ),
    )
    return table.scan(row_filter=predicate).to_arrow().to_pylist()


def get_evaluation(
    catalog,
    model_version_id: str,
    *,
    eval_snapshot_uuid: str,
    sensitive_attribute: str = DEFAULT_SENSITIVE_ATTRIBUTE,
    threshold: float = DEFAULT_THRESHOLD,
    min_group_n: int = MIN_GROUP_N,
) -> dict[str, float]:
    """Recorded difference metrics for one evaluation, read back from Iceberg.

    Reads the aggregate rows only — those with no ``group_value`` — because the
    per-group rows are evidence for these numbers rather than numbers a caller
    compares directly. ``min_group_n`` has to be given because it is part of what
    the number means: the same model on the same data yields a different spread
    under a different cutoff, and returning both would be returning neither.
    """
    rows = _existing_rows(
        catalog, model_version_id, eval_snapshot_uuid, sensitive_attribute, threshold
    )
    return {
        r["metric_name"]: r["metric_value"]
        for r in rows
        if r["group_value"] is None and r["min_group_n"] == min_group_n
    }
