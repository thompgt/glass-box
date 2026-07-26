"""Capturing and resolving data versions.

An Iceberg snapshot ID *is* the data version. This module wraps it in an audit
record so that a model can reference a data version by a stable identifier that
survives independently of the Iceberg table's own metadata — which matters
because the whole point of the tombstone mechanism is that the audit record
outlives the snapshot.

Capture is idempotent: the ``data_snapshot_uuid`` is a UUID5 derived from
``(table, iceberg_snapshot_id, filter_expr)``, so capturing the same data version
twice returns the same identifier rather than creating a second name for one
thing.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
from pyiceberg.expressions import AlwaysTrue, BooleanExpression, EqualTo

from .digest import digest_arrow_table
from .schemas import DATA_SNAPSHOTS, TableDef
from .writer import append_records

# Fixed namespace so snapshot UUIDs are reproducible across machines and runs.
SNAPSHOT_NAMESPACE = uuid.UUID("6f9b2c1e-6a4d-5c8e-9f10-2b7d4e6a8c30")

# Columns that are ingest bookkeeping rather than content. Excluded from the
# content digest so the same source rows loaded in two batches produce the same
# data version identity.
DIGEST_EXCLUDE = ("ingest_batch_id", "source_row_digest")


@dataclass
class DataSnapshot:
    data_snapshot_uuid: str
    table_identifier: str
    iceberg_snapshot_id: int
    captured_at: dt.datetime
    row_count: int
    content_digest: str
    filter_expr: str | None = None
    min_as_of_ts: dt.datetime | None = None
    max_as_of_ts: dt.datetime | None = None
    iceberg_schema_id: int | None = None
    partition_spec_id: int | None = None
    status: str = "live"
    expired_at: dt.datetime | None = None
    row_digests: list[str] = field(default_factory=list, repr=False)

    def to_record(self) -> dict[str, Any]:
        return {
            "data_snapshot_uuid": self.data_snapshot_uuid,
            "table_identifier": self.table_identifier,
            "iceberg_snapshot_id": self.iceberg_snapshot_id,
            "captured_at": self.captured_at,
            "filter_expr": self.filter_expr,
            "row_count": self.row_count,
            "min_as_of_ts": self.min_as_of_ts,
            "max_as_of_ts": self.max_as_of_ts,
            "content_digest": self.content_digest,
            "iceberg_schema_id": self.iceberg_schema_id,
            "partition_spec_id": self.partition_spec_id,
            "status": self.status,
            "expired_at": self.expired_at,
        }


def snapshot_uuid(table_identifier: str, iceberg_snapshot_id: int, filter_expr: str | None) -> str:
    key = f"{table_identifier}|{iceberg_snapshot_id}|{filter_expr or ''}"
    return str(uuid.uuid5(SNAPSHOT_NAMESPACE, key))


def sorted_scan(
    catalog,
    td: TableDef,
    *,
    iceberg_snapshot_id: int | None = None,
    row_filter: BooleanExpression | None = None,
    sort_key: str = "subject_id",
) -> pa.Table:
    """Scan a table into Arrow with a deterministic row order.

    The sort is not cosmetic. Iceberg gives no ordering guarantee across data
    files, and a model fit on rows in a different order is a different model —
    which would make bit-exact reproducibility flap for reasons that have nothing
    to do with the model. Every training and digest read goes through here.
    """
    table = catalog.load_table(td.identifier)
    scan = table.scan(
        row_filter=row_filter if row_filter is not None else AlwaysTrue(),
        snapshot_id=iceberg_snapshot_id,
    )
    arrow = scan.to_arrow()
    if sort_key in arrow.column_names:
        arrow = arrow.sort_by([(sort_key, "ascending")])
    return arrow.combine_chunks()


def capture_snapshot(
    catalog,
    td: TableDef,
    *,
    split: str | None = None,
    status: str = "live",
) -> DataSnapshot:
    """Record the current state of ``td`` as a named, digested data version.

    ``split`` is the only supported filter for now; it is stored as the
    ``filter_expr`` so the snapshot can be re-materialized exactly.
    """
    table = catalog.load_table(td.identifier)
    current = table.current_snapshot()
    if current is None:
        raise ValueError(f"{td.name} has no snapshot — nothing has been written to it")

    filter_expr = f"split == '{split}'" if split else None
    row_filter = EqualTo("split", split) if split else None

    arrow = sorted_scan(catalog, td, row_filter=row_filter)
    digest, row_digests = digest_arrow_table(arrow, exclude=DIGEST_EXCLUDE)

    min_ts = max_ts = None
    if arrow.num_rows and "as_of_ts" in arrow.column_names:
        min_ts = pc.min(arrow["as_of_ts"]).as_py()
        max_ts = pc.max(arrow["as_of_ts"]).as_py()

    snap = DataSnapshot(
        data_snapshot_uuid=snapshot_uuid(td.name, current.snapshot_id, filter_expr),
        table_identifier=td.name,
        iceberg_snapshot_id=current.snapshot_id,
        captured_at=dt.datetime.now(dt.timezone.utc),
        row_count=arrow.num_rows,
        content_digest=digest,
        filter_expr=filter_expr,
        min_as_of_ts=min_ts,
        max_as_of_ts=max_ts,
        iceberg_schema_id=table.schema().schema_id,
        partition_spec_id=table.spec().spec_id,
        status=status,
        row_digests=row_digests,
    )

    existing = get_snapshot(catalog, snap.data_snapshot_uuid)
    if existing is not None:
        # Idempotent: the same data version keeps its original captured_at.
        # If the digest disagrees, the same (table, snapshot, filter) produced
        # different content, which would mean Iceberg history was mutated.
        if existing["content_digest"] != snap.content_digest:
            raise RuntimeError(
                f"content digest changed for existing snapshot {snap.data_snapshot_uuid}: "
                f"{existing['content_digest']} != {snap.content_digest}"
            )
        snap.captured_at = existing["captured_at"]
        snap.status = existing["status"]
        return snap

    append_records(catalog, DATA_SNAPSHOTS, [snap.to_record()])
    return snap


def get_snapshot(catalog, data_snapshot_uuid: str) -> dict[str, Any] | None:
    table = catalog.load_table(DATA_SNAPSHOTS.identifier)
    arrow = table.scan(row_filter=EqualTo("data_snapshot_uuid", data_snapshot_uuid)).to_arrow()
    if arrow.num_rows == 0:
        return None
    return arrow.to_pylist()[0]


def materialize(catalog, snap: dict[str, Any] | DataSnapshot, td: TableDef) -> pa.Table:
    """Re-read the exact rows a data snapshot refers to, in deterministic order.

    This is the read half of reproducibility: training reads through here, and so
    does the reproduction check, so the two cannot diverge.
    """
    record = snap.to_record() if isinstance(snap, DataSnapshot) else snap
    if record["status"] == "tombstoned":
        raise SnapshotTombstonedError(record["data_snapshot_uuid"])

    filter_expr = record.get("filter_expr")
    row_filter = None
    if filter_expr:
        # Only the split filter form is emitted by capture_snapshot.
        split = filter_expr.split("'")[1]
        row_filter = EqualTo("split", split)

    return sorted_scan(
        catalog,
        td,
        iceberg_snapshot_id=record["iceberg_snapshot_id"],
        row_filter=row_filter,
    )


class SnapshotTombstonedError(RuntimeError):
    """Raised when a data version has been destroyed to satisfy an erasure.

    Deliberately distinct from a generic failure: the caller is meant to be able
    to look up the tombstone and report *what* was destroyed and why, which is the
    whole claim that provenance survives erasure even when reproducibility does not.
    """

    def __init__(self, data_snapshot_uuid: str):
        self.data_snapshot_uuid = data_snapshot_uuid
        super().__init__(
            f"data snapshot {data_snapshot_uuid} was expired to satisfy an erasure "
            f"request; see audit.snapshot_tombstones for what it contained"
        )
