"""Carrying out an erasure, and keeping the provenance that outlives it.

Deleting the rows is the easy part and, on its own, a lie. Phase 0 measured two
things that shape everything here:

* A PyIceberg delete is **copy-on-write**: the rows go from the current snapshot,
  but every earlier snapshot still references the original data files, and time
  travel reads them back. The subject is still there.
* ``expire_snapshots`` is **metadata-only** (``files_removed=False``). Dropping the
  snapshots unlinks the *history*, not the Parquet. The subject is still on disk.

So erasure is four steps, not one: delete the live rows, expire every snapshot
that still contains them, unlink the data files no surviving snapshot references,
and — before any of that — record what is about to be destroyed.

**The record is written first.** An ``audit.erasure_requests`` row marked
``in_progress`` that turns out to have deleted nothing is a harmless orphan and is
visible as one. The reverse order destroys a person's data and then dies with no
record of who asked, when, or why — which is not merely an incomplete audit trail
but, for a right-to-erasure log, its own compliance failure. Same reasoning as
:mod:`glassbox.train.registry` and :mod:`glassbox.serve.spool`: write the record
whose orphan is harmless first.

**Tombstones are written before expiry**, for the same reason at the snapshot
level. Once a data version is expired, what it contained is unknowable; the
tombstone preserves its row count, content digest, schema, and the model versions
that depended on it. That is the sense in which provenance survives erasure even
though reproducibility does not — an affected model can still say exactly what it
was trained on, it just can no longer be retrained to prove it.

**Contaminated models are not retired automatically.** Whether a model that
trained on erased data must come out of service is a policy question whose answer
differs by jurisdiction and by how much the individual contributed. Deciding it
here would bury a legal judgement inside a delete function. The models are
reported, and ``retire_contaminated=True`` applies the policy when a caller has
actually made that decision.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pyiceberg.expressions import And, EqualTo, In

from ..digest import canonical_json
from ..schemas import (
    DATA_SNAPSHOTS,
    ERASURE_REQUESTS,
    FAIRNESS_DECOMPOSITIONS,
    MODEL_VERSIONS,
    SNAPSHOT_TOMBSTONES,
    TableDef,
)
from ..writer import append_records
from .report import SUBJECT_TABLES, ContaminationReport, contamination_report

IN_PROGRESS = "in_progress"
COMPLETED = "completed"
FAILED = "failed"

TOMBSTONE_REASON = "erasure"
RETIRE_REASON = "training data erased at subject request"


class ErasureError(RuntimeError):
    """The erasure could not be completed. The request row records how far it got."""


@dataclass
class ErasureResult:
    request_id: str
    subject_id: str
    status: str
    requested_at: dt.datetime
    completed_at: dt.datetime | None
    live_rows_deleted: int
    contaminated_model_ids: list[str]
    snapshots_tombstoned: list[str] = field(default_factory=list)
    iceberg_snapshots_expired: int = 0
    data_files_unlinked: int = 0
    decompositions_invalidated: int = 0
    models_retired: list[str] = field(default_factory=list)
    report: ContaminationReport | None = field(default=None, repr=False)

    def resolution(self) -> str:
        return canonical_json(
            {
                "snapshots_tombstoned": sorted(self.snapshots_tombstoned),
                "iceberg_snapshots_expired": self.iceberg_snapshots_expired,
                "data_files_unlinked": self.data_files_unlinked,
                "decompositions_invalidated": self.decompositions_invalidated,
                "models_retired": sorted(self.models_retired),
            }
        )


def erase_subject(
    catalog,
    subject_id: str,
    *,
    reason: str = TOMBSTONE_REASON,
    retire_contaminated: bool = False,
    unlink_files: bool = True,
    root: Path | None = None,
) -> ErasureResult:
    """Erase one data subject and record what that cost.

    Returns even when the subject is unknown to the trail — an erasure request for
    someone who was never in the data is a legitimate request with the answer
    "nothing to do", and it belongs on the record like any other.
    """
    requested_at = dt.datetime.now(dt.UTC)
    report = contamination_report(catalog, subject_id)
    request_id = str(uuid.uuid4())

    # The record goes down before anything is destroyed. See the module docstring.
    _write_request(
        catalog,
        request_id=request_id,
        subject_id=subject_id,
        requested_at=requested_at,
        status=IN_PROGRESS,
        contaminated_model_ids=report.model_version_ids,
        live_rows_deleted=None,
        completed_at=None,
        resolution_json=None,
    )

    result = ErasureResult(
        request_id=request_id,
        subject_id=subject_id,
        status=IN_PROGRESS,
        requested_at=requested_at,
        completed_at=None,
        live_rows_deleted=0,
        contaminated_model_ids=report.model_version_ids,
        report=report,
    )

    try:
        _execute(catalog, subject_id, reason, retire_contaminated, unlink_files, root, result)
    except Exception as exc:  # noqa: BLE001
        result.status = FAILED
        _finish(catalog, result, resolution=canonical_json({"error": str(exc)}))
        raise ErasureError(
            f"erasure of {subject_id} failed partway; audit.erasure_requests "
            f"{request_id} records what had been done: {exc}"
        ) from exc

    result.status = COMPLETED
    result.completed_at = dt.datetime.now(dt.UTC)
    _finish(catalog, result, resolution=result.resolution())
    return result


def _execute(
    catalog,
    subject_id: str,
    reason: str,
    retire_contaminated: bool,
    unlink_files: bool,
    root: Path | None,
    result: ErasureResult,
) -> None:
    implicated = _snapshots_containing(catalog, subject_id)

    # Tombstone before expiring: after expiry, what the version contained is
    # unknowable, and the tombstone is the only surviving statement of it.
    _write_tombstones(catalog, implicated, reason=reason, request_id=result.request_id)
    _mark_snapshots_tombstoned(catalog, [s["data_snapshot_uuid"] for s in implicated])
    result.snapshots_tombstoned = [s["data_snapshot_uuid"] for s in implicated]

    for td in SUBJECT_TABLES:
        result.live_rows_deleted += _delete_subject_rows(catalog, td, subject_id)

    # Only now can the old snapshots go: the delete above created the current
    # snapshot, and the current snapshot cannot be expired.
    by_table = _iceberg_ids_by_table(implicated)
    for table_name, snapshot_ids in by_table.items():
        result.iceberg_snapshots_expired += _expire(catalog, table_name, snapshot_ids)

    if unlink_files:
        for td in SUBJECT_TABLES:
            result.data_files_unlinked += _unlink_unreferenced_files(catalog, td)

    result.decompositions_invalidated = _invalidate_decompositions(
        catalog, result.snapshots_tombstoned
    )

    if retire_contaminated and result.report is not None:
        result.models_retired = _retire(
            catalog, [c.model_version_id for c in result.report.serving]
        )


# ------------------------------------------------------------- the snapshots ----

def _snapshots_containing(catalog, subject_id: str) -> list[dict[str, Any]]:
    """Every live data version that still holds a row for this subject.

    Checked by scanning each recorded Iceberg snapshot rather than inferred from
    ``training_membership``. Membership answers "which models saw them", which is
    not the same question: a data version can contain a subject without any model
    having been trained from it, and that copy has to go too.
    """
    table = catalog.load_table(DATA_SNAPSHOTS.identifier)
    if table.current_snapshot() is None:
        return []

    live = table.scan(row_filter=EqualTo("status", "live")).to_arrow().to_pylist()
    by_name = {td.name: td for td in SUBJECT_TABLES}

    containing = []
    for record in live:
        td = by_name.get(record["table_identifier"])
        if td is None:
            continue
        if _snapshot_contains(catalog, td, record, subject_id):
            containing.append(record)
    return containing


def _snapshot_contains(catalog, td: TableDef, record: dict[str, Any], subject_id: str) -> bool:
    """Whether this data version actually contains the subject.

    The version's own ``filter_expr`` is applied, not just the subject filter. A
    snapshot pinned to ``split == 'train'`` does not contain an eval-split subject
    even though the underlying Iceberg snapshot has their row, and tombstoning it
    would destroy the reproducibility of every model trained from it to satisfy an
    erasure that never touched its rows.
    """
    table = catalog.load_table(td.identifier)
    predicate = EqualTo("subject_id", subject_id)
    if record.get("filter_expr"):
        # capture_snapshot only ever emits the split form; see snapshots.py.
        split = record["filter_expr"].split("'")[1]
        predicate = And(predicate, EqualTo("split", split))
    try:
        scan = table.scan(
            row_filter=predicate,
            selected_fields=("subject_id",),
            snapshot_id=record["iceberg_snapshot_id"],
        )
        return scan.to_arrow().num_rows > 0
    except ValueError:
        # The snapshot is already gone from this table's history — a previous
        # erasure expired it. Nothing left to erase there.
        return False


def _write_tombstones(
    catalog, snapshots: list[dict[str, Any]], *, reason: str, request_id: str
) -> int:
    if not snapshots:
        return 0
    pinned = _pinned_models(catalog, [s["data_snapshot_uuid"] for s in snapshots])
    now = dt.datetime.now(dt.UTC)
    records = [
        {
            "data_snapshot_uuid": s["data_snapshot_uuid"],
            "iceberg_snapshot_id": s["iceberg_snapshot_id"],
            "table_identifier": s["table_identifier"],
            "expired_at": now,
            "reason": reason,
            "row_count_at_expiry": s["row_count"],
            "content_digest": s["content_digest"],
            "schema_json": _schema_json(catalog, s["table_identifier"]),
            "pinned_model_version_ids": sorted(pinned.get(s["data_snapshot_uuid"], set())),
            "erasure_request_id": request_id,
        }
        for s in snapshots
    ]
    return append_records(catalog, SNAPSHOT_TOMBSTONES, records)


def _pinned_models(catalog, snapshot_uuids: list[str]) -> dict[str, set[str]]:
    """Model versions that named any of these data versions, in either role."""
    if not snapshot_uuids:
        return {}
    table = catalog.load_table(MODEL_VERSIONS.identifier)
    if table.current_snapshot() is None:
        return {}

    wanted = set(snapshot_uuids)
    pinned: dict[str, set[str]] = {}
    for row in table.scan().to_arrow().to_pylist():
        for column in ("data_snapshot_uuid", "eval_snapshot_uuid"):
            if row[column] in wanted:
                pinned.setdefault(row[column], set()).add(row["model_version_id"])
    return pinned


def _schema_json(catalog, table_identifier: str) -> str:
    td = next(t for t in SUBJECT_TABLES if t.name == table_identifier)
    table = catalog.load_table(td.identifier)
    return canonical_json(
        {
            "schema_id": table.schema().schema_id,
            "fields": [
                {"id": f.field_id, "name": f.name, "type": str(f.field_type),
                 "required": f.required}
                for f in table.schema().fields
            ],
        }
    )


def _mark_snapshots_tombstoned(catalog, snapshot_uuids: list[str]) -> int:
    """Flip status to tombstoned, preserving everything else about the row.

    A delete-and-reappend rather than an in-place update, because Iceberg has no
    update. The row keeps its identity; only ``status`` and ``expired_at`` move.
    """
    if not snapshot_uuids:
        return 0
    table = catalog.load_table(DATA_SNAPSHOTS.identifier)
    predicate = In("data_snapshot_uuid", set(snapshot_uuids))
    existing = table.scan(row_filter=predicate).to_arrow().to_pylist()
    if not existing:
        return 0

    now = dt.datetime.now(dt.UTC)
    table.delete(predicate)
    return append_records(
        catalog,
        DATA_SNAPSHOTS,
        [{**row, "status": "tombstoned", "expired_at": now} for row in existing],
    )


def _delete_subject_rows(catalog, td: TableDef, subject_id: str) -> int:
    table = catalog.load_table(td.identifier)
    if table.current_snapshot() is None:
        return 0
    predicate = EqualTo("subject_id", subject_id)
    present = table.scan(row_filter=predicate, selected_fields=("subject_id",)).to_arrow()
    if present.num_rows == 0:
        return 0
    table.delete(predicate)
    return present.num_rows


def _iceberg_ids_by_table(snapshots: list[dict[str, Any]]) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    for s in snapshots:
        grouped.setdefault(s["table_identifier"], set()).add(s["iceberg_snapshot_id"])
    return grouped


def _expire(catalog, table_name: str, snapshot_ids: set[int]) -> int:
    """Drop the named snapshots from the table's history.

    The current snapshot cannot be expired and is skipped rather than treated as
    an error: the delete that just ran made it, and it no longer contains the
    subject, so there is nothing there to remove.
    """
    td = next((t for t in SUBJECT_TABLES if t.name == table_name), None)
    if td is None:
        return 0
    table = catalog.load_table(td.identifier)
    current = table.current_snapshot()
    current_id = current.snapshot_id if current else None

    alive = {s.snapshot_id for s in table.snapshots()}
    targets = sorted((snapshot_ids & alive) - {current_id})
    if not targets:
        return 0

    table.maintenance.expire_snapshots().by_ids(targets).commit()
    return len(targets)


# ------------------------------------------------------------- the bytes ----

def _unlink_unreferenced_files(catalog, td: TableDef) -> int:
    """Delete Parquet files no surviving snapshot references.

    This is the step that makes the erasure real. ``expire_snapshots`` was measured
    in Phase 0 to be metadata-only — it removed the snapshots and left every data
    file in place — so without this the erased rows are gone from the catalogue and
    still sitting in the warehouse.

    Files are matched on basename rather than full path. The paths Iceberg records
    are ``file://`` URIs while the ones on disk are Windows paths, and normalizing
    between the two is a source of bugs that would fail *open*: a mismatch would
    make a live file look unreferenced and delete it. PyIceberg names every data
    file with a UUID, so basenames are unique and comparing them is exact.
    """
    table = catalog.load_table(td.identifier)
    referenced = _referenced_basenames(table)
    data_dir = _table_data_dir(table)
    if data_dir is None or not data_dir.exists():
        return 0

    removed = 0
    for path in data_dir.rglob("*.parquet"):
        if path.name not in referenced:
            path.unlink()
            removed += 1
    return removed


def _referenced_basenames(table) -> set[str]:
    """Basenames of every data file reachable from any surviving snapshot."""
    if table.current_snapshot() is None:
        return set()
    arrow = table.inspect.all_data_files()
    if "file_path" not in arrow.column_names:
        raise ErasureError(
            "inspect.all_data_files() has no file_path column; refusing to guess "
            "which files are unreferenced"
        )
    return {Path(unquote(urlparse(p).path or p)).name for p in arrow["file_path"].to_pylist()}


def _table_data_dir(table) -> Path | None:
    location = table.location()
    parsed = urlparse(location)
    if parsed.scheme in ("", "file"):
        raw = unquote(parsed.path or location)
        # file:///C:/... parses to /C:/..., which is not a path on Windows.
        if len(raw) > 2 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        return Path(raw) / "data"
    return None


# --------------------------------------------------------------- fallout ----

def _invalidate_decompositions(catalog, tombstoned: list[str]) -> int:
    """Mark fairness decompositions whose eval set no longer exists.

    The rows are kept and flagged, never deleted. A comparison that was made and
    later invalidated is itself audit history — it may have justified a release —
    and removing it would rewrite the past to look tidier than it was.
    """
    if not tombstoned:
        return 0
    table = catalog.load_table(FAIRNESS_DECOMPOSITIONS.identifier)
    if table.current_snapshot() is None:
        return 0

    predicate = In("eval_snapshot_uuid", set(tombstoned))
    affected = table.scan(row_filter=predicate).to_arrow().to_pylist()
    stale = [r for r in affected if r["comparable"]]
    if not stale:
        return 0

    table.delete(predicate)
    append_records(
        catalog,
        FAIRNESS_DECOMPOSITIONS,
        [{**r, "comparable": False} for r in affected],
    )
    return len(stale)


def _retire(catalog, model_version_ids: list[str]) -> list[str]:
    if not model_version_ids:
        return []
    table = catalog.load_table(MODEL_VERSIONS.identifier)
    predicate = In("model_version_id", set(model_version_ids))
    existing = table.scan(row_filter=predicate).to_arrow().to_pylist()
    if not existing:
        return []

    now = dt.datetime.now(dt.UTC)
    table.delete(predicate)
    append_records(
        catalog,
        MODEL_VERSIONS,
        [
            {**r, "status": "retired", "retired_at": now, "retire_reason": RETIRE_REASON}
            for r in existing
        ],
    )
    return sorted(r["model_version_id"] for r in existing)


# --------------------------------------------------------------- the row ----

def _write_request(catalog, **row) -> int:
    return append_records(catalog, ERASURE_REQUESTS, [_request_record(**row)])


def _request_record(
    *,
    request_id: str,
    subject_id: str,
    requested_at: dt.datetime,
    status: str,
    contaminated_model_ids: list[str],
    live_rows_deleted: int | None,
    completed_at: dt.datetime | None,
    resolution_json: str | None,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "subject_id": subject_id,
        "requested_at": requested_at,
        "completed_at": completed_at,
        "status": status,
        "live_rows_deleted": live_rows_deleted,
        "contaminated_model_count": len(contaminated_model_ids),
        "contaminated_model_ids": list(contaminated_model_ids),
        "resolution_json": resolution_json,
    }


def _finish(catalog, result: ErasureResult, *, resolution: str) -> None:
    """Replace the in_progress row with the terminal one."""
    table = catalog.load_table(ERASURE_REQUESTS.identifier)
    table.delete(EqualTo("request_id", result.request_id))
    _write_request(
        catalog,
        request_id=result.request_id,
        subject_id=result.subject_id,
        requested_at=result.requested_at,
        status=result.status,
        contaminated_model_ids=result.contaminated_model_ids,
        live_rows_deleted=result.live_rows_deleted,
        completed_at=result.completed_at or dt.datetime.now(dt.UTC),
        resolution_json=resolution,
    )


def get_erasure_request(catalog, request_id: str) -> dict[str, Any] | None:
    table = catalog.load_table(ERASURE_REQUESTS.identifier)
    if table.current_snapshot() is None:
        return None
    rows = table.scan(row_filter=EqualTo("request_id", request_id)).to_arrow().to_pylist()
    return rows[0] if rows else None


def erasure_requests_for(catalog, subject_id: str) -> list[dict[str, Any]]:
    table = catalog.load_table(ERASURE_REQUESTS.identifier)
    if table.current_snapshot() is None:
        return []
    rows = table.scan(row_filter=EqualTo("subject_id", subject_id)).to_arrow().to_pylist()
    return sorted(rows, key=lambda r: r["requested_at"])


__all__ = [
    "COMPLETED",
    "FAILED",
    "IN_PROGRESS",
    "ErasureError",
    "ErasureResult",
    "erase_subject",
    "erasure_requests_for",
    "get_erasure_request",
]
