"""Phase 0 — probe PyIceberg's write-side behaviour on the installed version.

Runs a set of capability probes against a throwaway warehouse and writes
``docs/pyiceberg-notes.md``. The point is to find out *before* writing
load-bearing code which of the following actually work here:

* partitioned writes under ``bucket()`` and ``day()`` transforms
* time travel via ``scan(snapshot_id=...)``
* copy-on-write row deletion, and whether history survives it
* schema evolution
* ``expire_snapshots`` — and critically, whether it removes data *files* or only
  metadata references, because the erasure phase depends on the former
* PyArrow type strictness (``timestamp[ns]``, ``large_string``)

Usage:  python scripts/phase0_spike.py
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@dataclass
class Probe:
    name: str
    question: str
    ok: bool = False
    detail: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)


PROBES: list[Probe] = []


def probe(name: str, question: str):
    """Decorator: run the function, capture pass/fail plus any exception."""

    def wrap(fn):
        p = Probe(name=name, question=question)
        try:
            result = fn(p)
            p.ok = True
            if result:
                p.detail = str(result)
        except Exception as exc:  # noqa: BLE001 - the whole point is to record failures
            p.ok = False
            p.error = f"{type(exc).__name__}: {exc}"
            p.notes.append(traceback.format_exc(limit=3).strip().splitlines()[-1])
        PROBES.append(p)
        return fn

    return wrap


def main() -> int:
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.expressions import EqualTo
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import BucketTransform, DayTransform
    from pyiceberg.types import (
        NestedField,
        StringType,
        TimestamptzType,
    )

    import pyiceberg

    tmp = Path(tempfile.mkdtemp(prefix="glassbox-spike-"))
    warehouse = tmp / "warehouse"
    warehouse.mkdir(parents=True)

    catalog = SqlCatalog(
        "spike",
        **{"uri": f"sqlite:///{(tmp / 'catalog.db').as_posix()}", "warehouse": warehouse.as_uri()},
    )
    catalog.create_namespace("spike")

    schema = Schema(
        NestedField(1, "subject_id", StringType(), required=True),
        NestedField(2, "event_ts", TimestamptzType(), required=True),
        NestedField(3, "payload", StringType(), required=False),
    )

    def rows(n: int, offset: int = 0) -> pa.Table:
        base = dt.datetime(2026, 3, 14, 12, 0, tzinfo=dt.timezone.utc)
        return pa.table(
            {
                "subject_id": [f"s{i + offset:04d}" for i in range(n)],
                # microseconds, not nanoseconds — see the type-strictness probe
                "event_ts": pa.array(
                    [base + dt.timedelta(days=i % 3) for i in range(n)],
                    type=pa.timestamp("us", tz="UTC"),
                ),
                "payload": [f"p{i + offset}" for i in range(n)],
            },
            schema=pa.schema(
                [
                    pa.field("subject_id", pa.string(), nullable=False),
                    pa.field("event_ts", pa.timestamp("us", tz="UTC"), nullable=False),
                    pa.field("payload", pa.string(), nullable=True),
                ]
            ),
        )

    bucket_spec = PartitionSpec(
        PartitionField(source_id=1, field_id=1000, transform=BucketTransform(4), name="sub_bucket")
    )
    day_spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=DayTransform(), name="event_day")
    )

    # ---------------------------------------------------------------- probes --

    @probe("bucket-partitioned-write", "Can PyIceberg append to a bucket()-partitioned table?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "bucketed"), schema, partition_spec=bucket_spec)
        t.append(rows(40))
        n = t.scan().to_arrow().num_rows
        files = list((warehouse / "spike.db" / "bucketed" / "data").rglob("*.parquet"))
        p.notes.append(f"{len(files)} data files written across buckets")
        assert n == 40, n
        return f"40 rows, {len(files)} files"

    @probe("day-partitioned-write", "Can PyIceberg append to a day()-partitioned table?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "daily"), schema, partition_spec=day_spec)
        t.append(rows(30))
        n = t.scan().to_arrow().num_rows
        files = list((warehouse / "spike.db" / "daily" / "data").rglob("*.parquet"))
        p.notes.append(f"{len(files)} data files (expect 3 — three distinct days)")
        assert n == 30, n
        return f"30 rows, {len(files)} files"

    @probe("multi-append-snapshots", "Does each append create a distinct snapshot?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "history"), schema)
        t.append(rows(10, 0))
        t.append(rows(10, 100))
        t.append(rows(10, 200))
        snaps = list(t.snapshots())
        assert len(snaps) == 3, len(snaps)
        return f"{len(snaps)} snapshots"

    @probe("time-travel", "Does scan(snapshot_id=...) return the historical state?")
    def _(p: Probe):
        t = catalog.load_table(("spike", "history"))
        snaps = sorted(t.snapshots(), key=lambda s: s.timestamp_ms)
        counts = [t.scan(snapshot_id=s.snapshot_id).to_arrow().num_rows for s in snaps]
        assert counts == [10, 20, 30], counts
        return f"row counts by snapshot: {counts}"

    @probe("scan-order-stability", "Is scan row order stable across repeated scans?")
    def _(p: Probe):
        t = catalog.load_table(("spike", "bucketed"))
        a = t.scan().to_arrow()["subject_id"].to_pylist()
        b = t.scan().to_arrow()["subject_id"].to_pylist()
        stable = a == b
        p.notes.append(
            "Stable here, but this is NOT guaranteed by the spec — training must "
            "still sort explicitly before fitting."
            if stable
            else "UNSTABLE — confirms the explicit sort in training is mandatory."
        )
        return f"stable={stable}"

    @probe("copy-on-write-delete", "Can a single row be deleted, and does history survive?")
    def _(p: Probe):
        t = catalog.load_table(("spike", "bucketed"))
        before = t.current_snapshot().snapshot_id
        t.delete(EqualTo("subject_id", "s0003"))
        live = t.scan().to_arrow().num_rows
        hist = t.scan(snapshot_id=before).to_arrow().num_rows
        assert live == 39, live
        assert hist == 40, hist
        p.notes.append("Delete is copy-on-write: whole data files are rewritten.")
        return f"live={live}, pre-delete snapshot={hist}"

    @probe("delete-rewrite-blast-radius", "How many data files does a 1-row delete rewrite?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "blast"), schema, partition_spec=bucket_spec)
        t.append(rows(400))
        before = {task.file.file_path for task in t.scan().plan_files()}
        t.delete(EqualTo("subject_id", "s0003"))
        after = {task.file.file_path for task in t.scan().plan_files()}
        rewritten = len(before - after)
        p.notes.append(
            f"{rewritten}/{len(before)} files rewritten for a single-row delete "
            f"— this is why features.* is bucketed on subject_id."
        )
        return f"{rewritten} of {len(before)} files rewritten"

    @probe("schema-evolution", "Can a column be added to a populated table?")
    def _(p: Probe):
        from pyiceberg.types import IntegerType

        t = catalog.load_table(("spike", "history"))
        with t.update_schema() as us:
            us.add_column("extra", IntegerType(), required=False)
        t = catalog.load_table(("spike", "history"))
        assert "extra" in t.schema().column_names
        return "added optional column"

    @probe("expire-snapshots", "Is expire_snapshots available, and does it remove data files?")
    def _(p: Probe):
        t = catalog.load_table(("spike", "history"))
        data_dir = warehouse / "spike.db" / "history" / "data"
        files_before = len(list(data_dir.rglob("*.parquet")))
        snaps_before = len(list(t.snapshots()))

        if not hasattr(t, "expire_snapshots"):
            p.notes.append("t.expire_snapshots MISSING on this version.")
            raise AttributeError("Table.expire_snapshots not available")

        newest = max(t.snapshots(), key=lambda s: s.timestamp_ms)
        others = [s for s in t.snapshots() if s.snapshot_id != newest.snapshot_id]
        expirer = t.expire_snapshots()
        for s in others:
            expirer = expirer.expire_snapshot_id(s.snapshot_id)
        expirer.commit()

        t = catalog.load_table(("spike", "history"))
        snaps_after = len(list(t.snapshots()))
        files_after = len(list(data_dir.rglob("*.parquet")))

        removed_metadata = snaps_after < snaps_before
        removed_files = files_after < files_before
        p.notes.append(
            f"snapshots {snaps_before} -> {snaps_after}; "
            f"data files {files_before} -> {files_after}"
        )
        if removed_metadata and not removed_files:
            p.notes.append(
                "METADATA-ONLY: orphaned data files remain on disk. Erasure must "
                "unlink them explicitly (fallback tier 2 in the risk register)."
            )
        return f"metadata_removed={removed_metadata}, files_removed={removed_files}"

    @probe("arrow-ns-timestamp", "Is timestamp[ns] (pandas default) rejected?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "tsprobe"), schema)
        bad = rows(5).set_column(
            1, "event_ts", rows(5)["event_ts"].cast(pa.timestamp("ns", tz="UTC"))
        )
        try:
            t.append(bad)
        except Exception as exc:  # noqa: BLE001
            p.notes.append(f"Rejected as expected: {type(exc).__name__}")
            return "rejected (must coerce to us before every write)"
        p.notes.append("ACCEPTED — coerced silently by this version.")
        return "accepted (silently coerced)"

    @probe("arrow-large-string", "Is large_string (Polars default) rejected?")
    def _(p: Probe):
        t = catalog.create_table(("spike", "lsprobe"), schema)
        bad = rows(5).set_column(
            0, "subject_id", rows(5)["subject_id"].cast(pa.large_string())
        )
        try:
            t.append(bad)
        except Exception as exc:  # noqa: BLE001
            p.notes.append(f"Rejected as expected: {type(exc).__name__}")
            return "rejected (must coerce to string before every write)"
        p.notes.append("ACCEPTED — coerced silently by this version.")
        return "accepted (silently coerced)"

    @probe("upsert", "Is Table.upsert available?")
    def _(p: Probe):
        t = catalog.load_table(("spike", "history"))
        if not hasattr(t, "upsert"):
            raise AttributeError("Table.upsert not available")
        return "available"

    # ---------------------------------------------------------------- report --

    write_report(pyiceberg.__version__, pa.__version__)
    shutil.rmtree(tmp, ignore_errors=True)

    failed = [p for p in PROBES if not p.ok]
    print(f"\n{len(PROBES) - len(failed)}/{len(PROBES)} probes passed")
    for p in failed:
        print(f"  FAIL {p.name}: {p.error}")
    print(f"\nwrote {REPO / 'docs' / 'pyiceberg-notes.md'}")
    # Probe failures are findings, not build failures — the report is the output.
    return 0


def write_report(pyiceberg_version: str, pyarrow_version: str) -> None:
    out = REPO / "docs" / "pyiceberg-notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# PyIceberg write-side behaviour — measured",
        "",
        "Generated by `scripts/phase0_spike.py`. This is Phase 0's deliverable: a",
        "record of what actually works on the pinned versions, so later phases do not",
        "discover a missing write-side capability while depending on it.",
        "",
        f"- `pyiceberg` **{pyiceberg_version}**",
        f"- `pyarrow` **{pyarrow_version}**",
        "",
        "| probe | question | result |",
        "|---|---|---|",
    ]
    for p in PROBES:
        mark = "PASS" if p.ok else "FAIL"
        outcome = p.detail if p.ok else p.error
        lines.append(f"| `{p.name}` | {p.question} | **{mark}** — {outcome} |")

    lines += ["", "## Notes", ""]
    for p in PROBES:
        if not p.notes:
            continue
        lines.append(f"### `{p.name}`")
        lines += [f"- {n}" for n in p.notes]
        lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
