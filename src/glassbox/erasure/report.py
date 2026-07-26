"""Capability #4: which models saw this person?

The answer comes from ``audit.training_membership``, which is materialized at
training time and never reconstructed. That choice is the whole capability. The
obvious alternative — work it out later by time-travelling to each model's
training snapshot and looking for the subject — fails exactly when it is needed,
because satisfying an erasure request destroys those snapshots, and the question
"which models saw this person?" is asked *during* the erasure, not before it.

Two things this report deliberately does not do:

**It does not hide orphaned membership.** A membership row naming a model version
with no ``audit.model_versions`` row is reported, flagged ``registered=False``,
rather than dropped by the join. That state is produced on purpose: training
commits membership first and the model version second, so a crash between them
over-reports contamination. Silently discarding those rows would convert a
deliberate over-report back into the under-report the ordering exists to prevent.

**It does not infer that an erasure is required.** It reports which models trained
on the subject and which of those are still serving. Whether a contaminated model
must be retired is a policy question with a different answer in different
jurisdictions, and a report that answered it would be making the decision rather
than informing it.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from pyiceberg.expressions import EqualTo, In

from ..schemas import (
    CREDIT_APPLICATIONS,
    DATA_SNAPSHOTS,
    EVAL_HOLDOUT,
    MODEL_VERSIONS,
    PREDICTIONS,
    TRAINING_MEMBERSHIP,
)

# Feature tables that hold subject rows and therefore have to be erased from.
SUBJECT_TABLES = (CREDIT_APPLICATIONS, EVAL_HOLDOUT)

ACTIVE_STATUS = "active"


@dataclass(frozen=True)
class Contamination:
    """One model version that trained or evaluated on this subject."""

    model_version_id: str
    role: str  # train | eval
    registered: bool
    status: str | None = None
    recipe: str | None = None
    trained_at: dt.datetime | None = None
    data_snapshot_uuid: str | None = None
    eval_snapshot_uuid: str | None = None
    # Whether the data version this model was trained from is still materializable.
    # False once an erasure has tombstoned it: the model stays fully attributable,
    # but it can no longer be reproduced. Provenance outlives reproducibility.
    reproducible: bool = True

    @property
    def serving(self) -> bool:
        return self.registered and self.status == ACTIVE_STATUS

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version_id": self.model_version_id,
            "role": self.role,
            "registered": self.registered,
            "status": self.status,
            "recipe": self.recipe,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "data_snapshot_uuid": self.data_snapshot_uuid,
            "eval_snapshot_uuid": self.eval_snapshot_uuid,
            "reproducible": self.reproducible,
            "serving": self.serving,
        }


@dataclass
class ContaminationReport:
    """Everything the audit trail knows about one data subject."""

    subject_id: str
    live_rows: dict[str, int]
    contaminated: list[Contamination]
    decisions: list[dict[str, Any]] = field(default_factory=list)
    generated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @property
    def known(self) -> bool:
        """False only if the trail has never heard of this subject at all."""
        return bool(self.contaminated or self.decisions or self.live_row_count)

    @property
    def live_row_count(self) -> int:
        return sum(self.live_rows.values())

    @property
    def model_version_ids(self) -> list[str]:
        return sorted({c.model_version_id for c in self.contaminated})

    @property
    def serving(self) -> list[Contamination]:
        """Contaminated models still marked active — the ones a policy acts on."""
        return [c for c in self.contaminated if c.serving]

    @property
    def orphaned(self) -> list[Contamination]:
        """Membership naming a model version that was never committed.

        Expected, not corrupt: training writes membership first so that a crash
        over-reports rather than under-reports. Reported so the over-report is
        visible as one.
        """
        return [c for c in self.contaminated if not c.registered]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "generated_at": self.generated_at.isoformat(),
            "live_rows": dict(self.live_rows),
            "live_row_count": self.live_row_count,
            "contaminated_model_count": len(self.model_version_ids),
            "contaminated": [c.to_dict() for c in self.contaminated],
            "serving_model_ids": sorted({c.model_version_id for c in self.serving}),
            "orphaned_model_ids": sorted({c.model_version_id for c in self.orphaned}),
            "decision_count": len(self.decisions),
            "decisions": self.decisions,
        }


def contamination_report(
    catalog, subject_id: str, *, include_decisions: bool = True
) -> ContaminationReport:
    """Every model whose training snapshot contained ``subject_id``."""
    membership = membership_of(catalog, subject_id)
    models = _models_by_id(catalog, [m["model_version_id"] for m in membership])
    tombstoned = _tombstoned_snapshots(catalog)

    contaminated = [
        _contamination(row, models.get(row["model_version_id"]), tombstoned)
        for row in sorted(membership, key=lambda r: (r["model_version_id"], r["role"]))
    ]

    return ContaminationReport(
        subject_id=subject_id,
        live_rows=live_rows_for(catalog, subject_id),
        contaminated=contaminated,
        decisions=decisions_about(catalog, subject_id) if include_decisions else [],
    )


def membership_of(catalog, subject_id: str) -> list[dict[str, Any]]:
    """Membership rows for one subject.

    The filter prunes hard. ``training_membership`` is bucketed on ``subject_id``,
    so this reads roughly one sixteenth of the table's data files rather than all
    of them — the same partition choice that bounds the cost of the deletion.
    """
    table = catalog.load_table(TRAINING_MEMBERSHIP.identifier)
    return table.scan(row_filter=EqualTo("subject_id", subject_id)).to_arrow().to_pylist()


def live_rows_for(catalog, subject_id: str) -> dict[str, int]:
    """Rows still present for this subject in each feature table."""
    counts = {}
    for td in SUBJECT_TABLES:
        table = catalog.load_table(td.identifier)
        if table.current_snapshot() is None:
            counts[td.name] = 0
            continue
        scan = table.scan(
            row_filter=EqualTo("subject_id", subject_id), selected_fields=("subject_id",)
        )
        counts[td.name] = scan.to_arrow().num_rows
    return counts


def decisions_about(catalog, subject_id: str) -> list[dict[str, Any]]:
    """Decisions served about this subject.

    Unlike membership this is a full scan: ``audit.predictions`` is partitioned by
    day and sorted by ``prediction_id``, because the query it is tuned for is
    "explain this decision", not "find every decision about this person". An
    erasure request is rare enough that the scan is the right trade.
    """
    table = catalog.load_table(PREDICTIONS.identifier)
    if table.current_snapshot() is None:
        return []
    rows = table.scan(row_filter=EqualTo("subject_id", subject_id)).to_arrow().to_pylist()
    return [
        {
            "prediction_id": r["prediction_id"],
            "prediction_ts": r["prediction_ts"].isoformat() if r["prediction_ts"] else None,
            "model_version_id": r["model_version_id"],
            "decision": r["decision"],
            "score": r["score"],
        }
        for r in sorted(rows, key=lambda r: (r["prediction_ts"], r["prediction_id"]))
    ]


# ---------------------------------------------------------------- internals ----

def _contamination(
    row: dict[str, Any], model: dict[str, Any] | None, tombstoned: set[str]
) -> Contamination:
    if model is None:
        return Contamination(
            model_version_id=row["model_version_id"], role=row["role"], registered=False
        )
    return Contamination(
        model_version_id=row["model_version_id"],
        role=row["role"],
        registered=True,
        status=model["status"],
        recipe=model["recipe"],
        trained_at=model["trained_at"],
        data_snapshot_uuid=model["data_snapshot_uuid"],
        eval_snapshot_uuid=model["eval_snapshot_uuid"],
        reproducible=model["data_snapshot_uuid"] not in tombstoned
        and model["eval_snapshot_uuid"] not in tombstoned,
    )


def _models_by_id(catalog, model_version_ids: list[str]) -> dict[str, dict[str, Any]]:
    if not model_version_ids:
        return {}
    table = catalog.load_table(MODEL_VERSIONS.identifier)
    rows = table.scan(row_filter=In("model_version_id", set(model_version_ids))).to_arrow()
    return {r["model_version_id"]: r for r in rows.to_pylist()}


def _tombstoned_snapshots(catalog) -> set[str]:
    table = catalog.load_table(DATA_SNAPSHOTS.identifier)
    if table.current_snapshot() is None:
        return set()
    rows = table.scan(row_filter=EqualTo("status", "tombstoned")).to_arrow()
    return set(rows["data_snapshot_uuid"].to_pylist())
